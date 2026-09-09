//===----------------------------------------------------------------------===//
// Estimators for the identification strategies do_identify recommends.
//
// do_identify will tell you that the backdoor is closed by an unobserved
// confounder but that a front-door mediator or an instrument is available. Until
// now nothing could act on that, which made the advice hollow. These two close
// the loop:
//
//   do_iv()         two-stage least squares, with the weak-instrument F-statistic
//                   reported rather than left to the reader
//   do_frontdoor()  the front-door product formula for a binary mediator
//
// Neither needs a foundation model, and both belong to the same design rule as
// the diagnostics: the assumption you cannot check from data - exclusion for an
// instrument, no unblocked backdoor into the mediator - is stated in the output
// rather than implied by its absence.
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

struct IdentGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<IdentGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<IdentGlobalState>();
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

//! Design matrix [X, extra], where `extra` are appended columns.
Matrix WithColumns(const Matrix &X, const vector<vector<double>> &extra) {
	Matrix out(X.rows, X.cols + extra.size());
	for (idx_t i = 0; i < X.rows; i++) {
		for (idx_t j = 0; j < X.cols; j++) {
			out.At(i, j) = X.At(i, j);
		}
		for (idx_t k = 0; k < extra.size(); k++) {
			out.At(i, X.cols + k) = extra[k][i];
		}
	}
	return out;
}

// --- do_iv --------------------------------------------------------------------

unique_ptr<FunctionData> BindIv(ClientContext &context, TableFunctionBindInput &input,
                                vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_iv requires outcome := '<column>'");
	}
	if (spec.aux_column.empty()) {
		throw BinderException("duckdo: do_iv requires instrument := '<column>'. Run do_identify() against a "
		                      "registered graph to see which nodes qualify");
	}
	// The treatment enters the regression on its own scale, binary or not.
	spec.continuous_treatment = true;
	auto frame = BuildFrame(context, spec);
	if (!frame.has_aux) {
		throw BinderException("duckdo: the instrument column was not loaded");
	}

	const double lambda = RidgeLambdaFor(frame);
	auto rows = frame.AllRows();

	// First stage: T ~ X + Z. Its explanatory power in Z is the whole question,
	// because a weak instrument makes the second stage worse than useless.
	Matrix first_stage_design = WithColumns(frame.X, {frame.aux});
	auto first_stage = FitRidgeWithSandwich(first_stage_design, frame.t, rows, lambda);
	vector<double> t_hat(frame.n, 0.0);
	for (idx_t i = 0; i < frame.n; i++) {
		t_hat[i] = first_stage.model.Eta(first_stage_design.Row(i), first_stage_design.cols);
	}

	// The instrument's coefficient is the last one; its robust t-statistic
	// squared is the first-stage F for a single instrument.
	const idx_t z_index = first_stage_design.cols; // +1 for the intercept, -1 for 0-based
	const double z_coef = first_stage.model.beta[z_index];
	const idx_t dim = first_stage.dim;
	const double z_var = first_stage.cov.empty() ? 0.0 : first_stage.cov[z_index * dim + z_index];
	const double z_se = z_var > 0.0 ? std::sqrt(z_var) : 0.0;
	const double first_stage_f = z_se > 0.0 ? (z_coef / z_se) * (z_coef / z_se) : 0.0;

	// Second stage: Y ~ X + T_hat.
	Matrix second_stage_design = WithColumns(frame.X, {t_hat});
	auto second_stage = FitRidge(second_stage_design, frame.y, rows, {}, lambda);
	const double estimate = second_stage.beta[second_stage_design.cols];

	// The structural residual uses the ACTUAL treatment, not the fitted one -
	// using t_hat here is the classic 2SLS standard-error mistake.
	Matrix structural = WithColumns(frame.X, {frame.t});
	vector<double> residual(frame.n, 0.0);
	for (idx_t i = 0; i < frame.n; i++) {
		residual[i] = frame.y[i] - second_stage.Eta(structural.Row(i), structural.cols);
	}
	// Sandwich on the projected design.
	auto meat_fit = FitRidgeWithSandwich(second_stage_design, frame.y, rows, lambda);
	double se = 0.0;
	{
		// Recompute the sandwich with the structural residuals.
		const idx_t p = meat_fit.dim;
		vector<double> gram(p * p, 0.0), meat(p * p, 0.0);
		vector<double> row(p);
		for (idx_t i = 0; i < frame.n; i++) {
			row[0] = 1.0;
			const double *src = second_stage_design.Row(i);
			for (idx_t j = 0; j < second_stage_design.cols; j++) {
				row[j + 1] = src[j];
			}
			const double u2 = residual[i] * residual[i];
			for (idx_t a = 0; a < p; a++) {
				for (idx_t b = 0; b < p; b++) {
					gram[a * p + b] += row[a] * row[b];
					meat[a * p + b] += u2 * row[a] * row[b];
				}
			}
		}
		for (idx_t j = 1; j < p; j++) {
			gram[j * p + j] += lambda;
		}
		vector<double> inverse(p * p, 0.0);
		bool ok = true;
		for (idx_t c = 0; c < p && ok; c++) {
			vector<double> rhs(p, 0.0);
			rhs[c] = 1.0;
			vector<double> col, work = gram;
			if (!CholeskySolve(work, p, rhs, col)) {
				ok = false;
				break;
			}
			for (idx_t r = 0; r < p; r++) {
				inverse[r * p + c] = col[r];
			}
		}
		if (ok) {
			const idx_t target = second_stage_design.cols; // treatment coefficient
			double acc = 0.0;
			for (idx_t a = 0; a < p; a++) {
				double inner = 0.0;
				for (idx_t b = 0; b < p; b++) {
					double middle = 0.0;
					for (idx_t k = 0; k < p; k++) {
						middle += meat[b * p + k] * inverse[k * p + target];
					}
					inner += inverse[a * p + b] * middle;
				}
				if (a == target) {
					acc = inner;
				}
			}
			se = acc > 0.0 ? std::sqrt(acc) : 0.0;
		}
	}

	EffectResult result;
	result.estimand = Estimand::ATE;
	result.estimator = "2sls";
	result.variance_method = "sandwich on the structural residual";
	result.estimate = estimate;
	result.std_error = se;
	result.n = frame.n;
	result.Finalize();

	vector<string> warnings = frame.warnings;
	warnings.push_back("the exclusion restriction - that the instrument affects the outcome only through the "
	                   "treatment - is an assumption no data can check");
	if (first_stage_f < 10.0) {
		warnings.push_back(StringUtil::Format(
		    "first-stage F is %.1f, below the conventional threshold of 10: this instrument is weak and the "
		    "estimate is unreliable",
		    first_stage_f));
	}
	if (!frame.continuous_treatment) {
		warnings.push_back("with a binary treatment and a binary instrument this identifies the local average "
		                   "treatment effect among compliers, not the population ATE");
	}

	names = {"estimand", "estimator", "estimate",      "std_error",  "ci_low",          "ci_high",
	         "p_value",  "n",         "first_stage_f", "instrument", "weak_instrument", "warnings"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::BIGINT,  LogicalType::DOUBLE,
	                LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::LIST(LogicalType::VARCHAR)};
	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value("LATE"), Value(result.estimator), Value::DOUBLE(result.estimate),
	                      Value::DOUBLE(result.std_error), Value::DOUBLE(result.ci_low), Value::DOUBLE(result.ci_high),
	                      Value::DOUBLE(result.p_value), Value::BIGINT(static_cast<int64_t>(result.n)),
	                      Value::DOUBLE(first_stage_f), Value(spec.aux_column), Value::BOOLEAN(first_stage_f < 10.0),
	                      Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

// --- do_frontdoor -------------------------------------------------------------

unique_ptr<FunctionData> BindFrontdoor(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_frontdoor requires outcome := '<column>'");
	}
	if (spec.aux_column.empty()) {
		throw BinderException("duckdo: do_frontdoor requires mediator := '<column>'. Run do_identify() against a "
		                      "registered graph to see which node qualifies");
	}
	auto frame = BuildFrame(context, spec);
	if (!frame.has_aux) {
		throw BinderException("duckdo: the mediator column was not loaded");
	}

	// The front-door formula factorises into two pieces that are each estimable
	// without touching the unobserved confounder:
	//
	//   a = E[M | do(T=1)] - E[M | do(T=0)]        the treatment's effect on the mediator
	//   b = sum_t P(t) (E[Y | M=1, t] - E[Y | M=0, t])   the mediator's effect on the outcome
	//   ATE = a * b
	//
	// a is identified because nothing confounds T -> M; b is identified because
	// conditioning on T closes the backdoor from M to Y.
	double m1 = 0.0, m0 = 0.0;
	idx_t n1 = 0, n0 = 0;
	for (idx_t i = 0; i < frame.n; i++) {
		if (frame.t[i] == 1.0) {
			m1 += frame.aux[i];
			n1++;
		} else {
			m0 += frame.aux[i];
			n0++;
		}
	}
	if (n1 == 0 || n0 == 0) {
		throw BinderException("duckdo: both treatment arms must be present for a front-door estimate");
	}
	const double a = m1 / static_cast<double>(n1) - m0 / static_cast<double>(n0);

	// b: within each treatment arm, contrast the outcome across the mediator,
	// then average over the treatment's marginal distribution.
	double b = 0.0;
	bool estimable = true;
	vector<string> warnings = frame.warnings;
	for (double arm : {0.0, 1.0}) {
		double y_m1 = 0.0, y_m0 = 0.0;
		idx_t c1 = 0, c0 = 0;
		for (idx_t i = 0; i < frame.n; i++) {
			if (frame.t[i] != arm) {
				continue;
			}
			if (frame.aux[i] >= 0.5) {
				y_m1 += frame.y[i];
				c1++;
			} else {
				y_m0 += frame.y[i];
				c0++;
			}
		}
		if (c1 == 0 || c0 == 0) {
			estimable = false;
			break;
		}
		const double share =
		    (arm == 1.0 ? static_cast<double>(n1) : static_cast<double>(n0)) / static_cast<double>(frame.n);
		b += share * (y_m1 / static_cast<double>(c1) - y_m0 / static_cast<double>(c0));
	}
	if (!estimable) {
		throw BinderException("duckdo: the mediator '%s' does not take both values within each treatment arm, so "
		                      "the front-door formula is not estimable here. It must be binary and vary within "
		                      "both arms",
		                      spec.aux_column);
	}

	const double estimate = a * b;

	// Delta-method-free interval: bootstrap the whole product.
	std::mt19937_64 rng(static_cast<uint64_t>(spec.seed) ^ 0xFD00FD00ULL);
	std::uniform_int_distribution<idx_t> pick(0, frame.n - 1);
	const idx_t reps = std::max<idx_t>(spec.bootstrap_reps, 100);
	vector<double> draws;
	for (idx_t rep = 0; rep < reps; rep++) {
		double bm1 = 0.0, bm0 = 0.0;
		idx_t bn1 = 0, bn0 = 0;
		double by[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
		idx_t bc[2][2] = {{0, 0}, {0, 0}};
		for (idx_t k = 0; k < frame.n; k++) {
			const idx_t i = pick(rng);
			const int arm = frame.t[i] == 1.0 ? 1 : 0;
			const int med = frame.aux[i] >= 0.5 ? 1 : 0;
			if (arm == 1) {
				bm1 += frame.aux[i];
				bn1++;
			} else {
				bm0 += frame.aux[i];
				bn0++;
			}
			by[arm][med] += frame.y[i];
			bc[arm][med]++;
		}
		if (bn1 == 0 || bn0 == 0 || bc[0][0] == 0 || bc[0][1] == 0 || bc[1][0] == 0 || bc[1][1] == 0) {
			continue;
		}
		const double ba = bm1 / static_cast<double>(bn1) - bm0 / static_cast<double>(bn0);
		double bb = 0.0;
		for (int arm = 0; arm < 2; arm++) {
			const double share = static_cast<double>(arm == 1 ? bn1 : bn0) / static_cast<double>(frame.n);
			bb += share * (by[arm][1] / static_cast<double>(bc[arm][1]) - by[arm][0] / static_cast<double>(bc[arm][0]));
		}
		draws.push_back(ba * bb);
	}
	double se = 0.0;
	if (draws.size() >= 20) {
		double variance = 0.0;
		for (auto d : draws) {
			const double diff = d - estimate;
			variance += diff * diff;
		}
		se = std::sqrt(variance / static_cast<double>(draws.size() - 1));
	}

	warnings.push_back("the front-door criterion assumes the mediator intercepts every directed path from "
	                   "treatment to outcome, and that nothing confounds treatment and mediator - neither is "
	                   "checkable from data. Verify them with do_identify() against a graph");
	warnings.push_back("covariates are not used: this is the unconditional front-door formula");

	names = {"estimand", "estimator",     "estimate",      "std_error", "ci_low",  "ci_high",
	         "n",        "effect_t_on_m", "effect_m_on_y", "mediator",  "warnings"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::VARCHAR,
	                LogicalType::LIST(LogicalType::VARCHAR)};
	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value("ATE"), Value("frontdoor"), Value::DOUBLE(estimate),
	                      se > 0.0 ? Value::DOUBLE(se) : Value(LogicalType::DOUBLE),
	                      se > 0.0 ? Value::DOUBLE(estimate - Z95 * se) : Value(LogicalType::DOUBLE),
	                      se > 0.0 ? Value::DOUBLE(estimate + Z95 * se) : Value(LogicalType::DOUBLE),
	                      Value::BIGINT(static_cast<int64_t>(frame.n)), Value::DOUBLE(a), Value::DOUBLE(b),
	                      Value(spec.aux_column), Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

} // namespace

void RegisterIdentificationFunctions(ExtensionLoader &loader) {
	struct Entry {
		const char *name;
		table_function_bind_t bind;
	};
	const Entry entries[] = {{"iv", BindIv}, {"frontdoor", BindFrontdoor}};
	for (auto &entry : entries) {
		TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, entry.bind, InitGlobal);
		AddCommonNamedParameters(fn);
		fn.named_parameters["instrument"] = LogicalType::VARCHAR;
		fn.named_parameters["mediator"] = LogicalType::VARCHAR;
		RegisterUnderBothNames(loader, fn, entry.name);
	}
}

} // namespace duckdo
} // namespace duckdb
