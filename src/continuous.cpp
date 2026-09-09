//===----------------------------------------------------------------------===//
// Phase 10: continuous treatments.
//
// A binary treatment has one number to report. A dose has a whole curve, and two
// different questions:
//
//   do_ape()            what does one more unit of the dose buy, on average?
//   do_dose_response()  what is E[Y(d)] across the range of d?
//
// The first is the average partial effect from a partially linear double-ML
// model: residualise the outcome and the dose on the covariates, then regress
// one residual on the other. That is the same estimator `estimator := 'dml'`
// already uses for a binary treatment - the only thing that changes is that the
// dose model is a regression rather than a logistic.
//
// The second is g-computation over a model quadratic in the dose, averaged over
// the covariate distribution at each grid point. The grid is placed on dose
// quantiles rather than evenly, so it never extrapolates past where data exists.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace duckdb {
namespace duckdo {

namespace {

struct ContinuousGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<ContinuousGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<ContinuousGlobalState>();
	idx_t count = 0;
	while (state.offset < bind.rows.size() && count < STANDARD_VECTOR_SIZE) {
		auto &row = bind.rows[state.offset];
		for (idx_t c = 0; c < row.size(); c++) {
			output.SetValue(c, count, row[c]);
		}
		state.offset++;
		count++;
	}
	output.SetCardinality(count);
}

double RidgeLambdaFor(const CausalFrame &frame) {
	return 1e-6 * static_cast<double>(frame.n) + 1e-8;
}

CausalSpec ContinuousSpec(ClientContext &context, TableFunctionBindInput &input, const char *fn) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	spec.continuous_treatment = true;
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: %s requires outcome := '<column>'", fn);
	}
	if (!spec.model.empty()) {
		throw BinderException("duckdo: %s has no foundation-model backend. Both shipped models take a binary "
		                      "treatment only; drop model := to use the classical estimator",
		                      fn);
	}
	return spec;
}

//! Cross-fitted E[D|X] and E[Y|X]. With a dose both are plain regressions, which
//! is the only structural difference from the binary nuisance fit.
struct ContinuousNuisance {
	vector<double> dose_hat;
	vector<double> outcome_hat;
};

ContinuousNuisance FitContinuousNuisance(const CausalFrame &frame, const CausalSpec &spec) {
	ContinuousNuisance fit;
	fit.dose_hat.assign(frame.n, 0.0);
	fit.outcome_hat.assign(frame.n, 0.0);

	const idx_t folds = std::min<idx_t>(std::max<idx_t>(spec.folds, 2), std::max<idx_t>(2, frame.n / 4));
	const double lambda = RidgeLambdaFor(frame);

	vector<idx_t> order(frame.n);
	for (idx_t i = 0; i < frame.n; i++) {
		order[i] = i;
	}
	std::mt19937_64 rng(static_cast<uint64_t>(spec.seed));
	std::shuffle(order.begin(), order.end(), rng);
	vector<idx_t> assignment(frame.n, 0);
	for (idx_t i = 0; i < frame.n; i++) {
		assignment[order[i]] = i % folds;
	}

	for (idx_t k = 0; k < folds; k++) {
		vector<idx_t> train, test;
		for (idx_t i = 0; i < frame.n; i++) {
			(assignment[i] == k ? test : train).push_back(i);
		}
		if (train.empty() || test.empty()) {
			continue;
		}
		auto dose_model = FitRidge(frame.X, frame.t, train, {}, lambda);
		auto outcome_model = FitRidge(frame.X, frame.y, train, {}, lambda);
		for (auto r : test) {
			fit.dose_hat[r] = dose_model.Eta(frame.X.Row(r), frame.X.cols);
			fit.outcome_hat[r] = outcome_model.Eta(frame.X.Row(r), frame.X.cols);
		}
	}
	return fit;
}

// --- do_ape ------------------------------------------------------------------

unique_ptr<FunctionData> BindApe(ClientContext &context, TableFunctionBindInput &input,
                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = ContinuousSpec(context, input, "do_ape");
	auto frame = BuildFrame(context, spec);
	if (!frame.continuous_treatment) {
		throw BinderException("duckdo: do_ape expects a continuous treatment");
	}
	auto fit = FitContinuousNuisance(frame, spec);

	// Partially linear double ML: theta = E[(D - m)(Y - g)] / E[(D - m)^2].
	double num = 0.0, den = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		const double dose_residual = frame.t[i] - fit.dose_hat[i];
		num += dose_residual * (frame.y[i] - fit.outcome_hat[i]);
		den += dose_residual * dose_residual;
	}
	EffectResult result;
	result.estimator = "dml";
	result.variance_method = "influence function";
	if (!(den > 1e-12)) {
		throw BinderException("duckdo: the covariates explain the treatment almost perfectly, so no variation in "
		                      "'%s' is left to identify a dose-response",
		                      spec.treatment);
	}
	result.estimate = num / den;

	double meat = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		const double dose_residual = frame.t[i] - fit.dose_hat[i];
		const double resid = (frame.y[i] - fit.outcome_hat[i]) - result.estimate * dose_residual;
		const double score = dose_residual * resid;
		meat += score * score;
	}
	const double n = static_cast<double>(frame.n);
	const double bread = den / n;
	result.std_error = std::sqrt(meat / n) / (std::fabs(bread) * std::sqrt(n));
	result.n = frame.n;
	result.Finalize();

	const double dose_sd = StdDev(frame.t);
	const double dose_min = frame.dose_sorted.front();
	const double dose_max = frame.dose_sorted.back();
	vector<string> warnings = frame.warnings;
	warnings.push_back("the average partial effect assumes the dose-response is locally linear; run "
	                   "do_dose_response() to see whether it is");

	names = {"estimand", "estimator", "estimate", "std_error", "ci_low",   "ci_high", "p_value",
	         "n",        "dose_mean", "dose_sd",  "dose_min",  "dose_max", "warnings"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::LIST(LogicalType::VARCHAR)};

	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value("APE"), Value(result.estimator), Value::DOUBLE(result.estimate),
	                      Value::DOUBLE(result.std_error), Value::DOUBLE(result.ci_low), Value::DOUBLE(result.ci_high),
	                      Value::DOUBLE(result.p_value), Value::BIGINT(static_cast<int64_t>(result.n)),
	                      Value::DOUBLE(Mean(frame.t)), Value::DOUBLE(dose_sd), Value::DOUBLE(dose_min),
	                      Value::DOUBLE(dose_max), Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

// --- do_dose_response --------------------------------------------------------

unique_ptr<FunctionData> BindDoseResponse(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = ContinuousSpec(context, input, "do_dose_response");
	auto frame = BuildFrame(context, spec);
	const idx_t grid = std::min<idx_t>(std::max<idx_t>(spec.grid, 3), 200);

	// Standardise the dose so the quadratic term stays conditioned, then build a
	// design of [X, d, d^2]. g-computation averages the fitted surface over the
	// covariate distribution at each grid point, which is what makes the curve a
	// dose-response rather than a conditional mean.
	const double dose_mean = Mean(frame.t);
	const double dose_sd = std::max(StdDev(frame.t), 1e-12);
	const idx_t p = frame.X.cols;
	Matrix design(frame.n, p + 2);
	for (idx_t i = 0; i < frame.n; i++) {
		const double *x = frame.X.Row(i);
		for (idx_t j = 0; j < p; j++) {
			design.At(i, j) = x[j];
		}
		const double d = (frame.t[i] - dose_mean) / dose_sd;
		design.At(i, p) = d;
		design.At(i, p + 1) = d * d;
	}

	vector<idx_t> all(frame.n);
	for (idx_t i = 0; i < frame.n; i++) {
		all[i] = i;
	}
	// Sandwich, because the residual variance of an outcome model is rarely
	// constant across the dose range.
	auto model = FitRidgeWithSandwich(design, frame.y, all, RidgeLambdaFor(frame));

	names = {"grid_point", "dose", "mu", "mu_low", "mu_high", "n_within_decile"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BIGINT};
	auto bind = make_uniq<ResultBindData>();

	const idx_t dim = design.cols + 1;
	vector<double> basis(dim, 0.0);
	for (idx_t g = 0; g < grid; g++) {
		// Place the grid on dose quantiles, so it never runs past the support.
		const double q = static_cast<double>(g) / static_cast<double>(grid - 1);
		idx_t pos = static_cast<idx_t>(q * static_cast<double>(frame.n - 1));
		pos = std::min(pos, frame.n - 1);
		const double dose = frame.dose_sorted[pos];
		const double d = (dose - dose_mean) / dose_sd;

		// The average of the design row over the sample, holding the dose fixed.
		std::fill(basis.begin(), basis.end(), 0.0);
		basis[0] = 1.0;
		for (idx_t i = 0; i < frame.n; i++) {
			const double *x = frame.X.Row(i);
			for (idx_t j = 0; j < p; j++) {
				basis[1 + j] += x[j];
			}
		}
		for (idx_t j = 0; j < p; j++) {
			basis[1 + j] /= static_cast<double>(frame.n);
		}
		basis[1 + p] = d;
		basis[1 + p + 1] = d * d;

		double mu = 0.0;
		for (idx_t a = 0; a < dim; a++) {
			mu += model.model.beta[a] * basis[a];
		}
		// Var(a'beta) = a' V a, exact for a linear model.
		double variance = 0.0;
		for (idx_t a = 0; a < dim; a++) {
			double inner = 0.0;
			for (idx_t b = 0; b < dim; b++) {
				inner += model.cov[a * dim + b] * basis[b];
			}
			variance += basis[a] * inner;
		}
		const double se = variance > 0.0 ? std::sqrt(variance) : 0.0;

		// How much data actually sits near this grid point.
		const double lo =
		    frame.dose_sorted[static_cast<idx_t>(std::max(0.0, (q - 0.05)) * static_cast<double>(frame.n - 1))];
		const double hi =
		    frame.dose_sorted[static_cast<idx_t>(std::min(1.0, (q + 0.05)) * static_cast<double>(frame.n - 1))];
		idx_t nearby = 0;
		for (idx_t i = 0; i < frame.n; i++) {
			if (frame.t[i] >= lo && frame.t[i] <= hi) {
				nearby++;
			}
		}

		bind->rows.push_back({Value::BIGINT(static_cast<int64_t>(g)), Value::DOUBLE(dose), Value::DOUBLE(mu),
		                      Value::DOUBLE(mu - Z95 * se), Value::DOUBLE(mu + Z95 * se),
		                      Value::BIGINT(static_cast<int64_t>(nearby))});
	}
	return std::move(bind);
}

} // namespace

void RegisterContinuousFunctions(ExtensionLoader &loader) {
	struct Entry {
		const char *name;
		table_function_bind_t bind;
	};
	const Entry entries[] = {{"ape", BindApe}, {"dose_response", BindDoseResponse}};
	for (auto &entry : entries) {
		TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, entry.bind, InitGlobal);
		AddCommonNamedParameters(fn);
		fn.named_parameters["grid"] = LogicalType::BIGINT;
		RegisterUnderBothNames(loader, fn, entry.name);
	}
}

} // namespace duckdo
} // namespace duckdb
