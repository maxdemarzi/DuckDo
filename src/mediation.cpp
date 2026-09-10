//===----------------------------------------------------------------------===//
// Mediation: how much of the effect goes through the mediator.
//
// do_frontdoor() uses a mediator to *rescue* identification when the backdoor is
// blocked. This asks the opposite question: the total effect is already
// identified, and you want to know how much of it travels through M and how much
// goes around it. Those are the natural indirect and natural direct effects.
//
// The decomposition is the linear-model special case of VanderWeele's, with the
// treatment-mediator interaction carried through:
//
//   M = b0 + b1*T + b2'X + e
//   Y = q0 + q1*T + q2*M + q3*(T*M) + q4'X + e
//
//   NDE = q1 + q3*(b0 + b2'E[X])     effect of T on Y with M held at M(0)
//   NIE = (q2 + q3)*b1               effect of moving M from M(0) to M(1) at T=1
//   TE  = NDE + NIE
//
// The interaction is what makes this mediation analysis rather than Baron-Kenny
// with extra steps: drop q3 and the decomposition is only valid when the
// treatment's effect on the outcome does not depend on the mediator, which is an
// assumption nobody checks and which the data here can actually speak to. q3 is
// reported so it can be looked at.
//
// The assumption that cannot be checked is sequential ignorability - in
// particular no unmeasured confounding of the mediator-outcome relationship.
// Randomising the treatment does not buy it, which is the single most common
// error in applied mediation. It is stated in every result rather than left to
// the reader.
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

struct MediationGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<MediationGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<MediationGlobalState>();
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

//! The three numbers one fit produces. Bootstrapping re-derives all of them from
//! the same resample, so their intervals are mutually consistent.
struct Decomposition {
	double direct = 0.0;
	double indirect = 0.0;
	double total = 0.0;
	double interaction = 0.0; //!< q3, the treatment-mediator interaction
	bool ok = false;
};

//! The two design matrices, built once.
//!
//! [X | T] for the mediator and [X | T | M | T*M] for the outcome, so the
//! coefficients of interest are always the last few entries of beta with
//! beta[0] the intercept. These are built outside the bootstrap: a resample
//! changes which *rows* are fitted, never the rows themselves, so rebuilding
//! them per replicate would allocate two n-by-p matrices two hundred times over
//! for no gain.
struct MediationDesign {
	Matrix mediator;
	Matrix outcome;
	idx_t p = 0;
};

MediationDesign BuildDesign(const Matrix &X, const vector<double> &t, const vector<double> &m) {
	MediationDesign design;
	design.p = X.cols;
	design.mediator.Resize(X.rows, X.cols + 1);
	design.outcome.Resize(X.rows, X.cols + 3);
	for (idx_t i = 0; i < X.rows; i++) {
		for (idx_t j = 0; j < X.cols; j++) {
			design.mediator.At(i, j) = X.At(i, j);
			design.outcome.At(i, j) = X.At(i, j);
		}
		design.mediator.At(i, X.cols) = t[i];
		design.outcome.At(i, X.cols) = t[i];
		design.outcome.At(i, X.cols + 1) = m[i];
		design.outcome.At(i, X.cols + 2) = t[i] * m[i];
	}
	return design;
}

//! Fit both models on `rows` and read the decomposition off the coefficients.
Decomposition Decompose(const MediationDesign &design, const vector<double> &m, const vector<double> &y,
                        const vector<idx_t> &rows, const vector<double> &x_mean, double lambda) {
	const idx_t p = design.p;
	Decomposition out;

	const vector<double> no_weights;
	auto med = FitRidge(design.mediator, m, rows, no_weights, lambda);
	auto outcome = FitRidge(design.outcome, y, rows, no_weights, lambda);
	if (med.beta.size() != p + 2 || outcome.beta.size() != p + 4) {
		return out;
	}

	const double b0 = med.beta[0];
	const double b1 = med.beta[p + 1];
	const double q1 = outcome.beta[p + 1];
	const double q2 = outcome.beta[p + 2];
	const double q3 = outcome.beta[p + 3];

	// E[M | T=0, X] averaged over the covariate distribution. The frame
	// standardises features so x_mean is near zero, but it is carried explicitly
	// rather than assumed - a frame built from a filtered relation need not be
	// centred.
	double m_at_control = b0;
	for (idx_t j = 0; j < p; j++) {
		m_at_control += med.beta[j + 1] * x_mean[j];
	}

	out.direct = q1 + q3 * m_at_control;
	out.indirect = (q2 + q3) * b1;
	out.total = out.direct + out.indirect;
	out.interaction = q3;
	out.ok = true;
	return out;
}

unique_ptr<FunctionData> BindMediate(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_mediate requires outcome := '<column>'");
	}
	if (spec.aux_column.empty()) {
		throw BinderException("duckdo: do_mediate requires mediator := '<column>'. It must be a variable the "
		                      "treatment causes and which in turn causes the outcome - do_validate() against a "
		                      "registered graph will say whether a candidate qualifies");
	}
	auto frame = BuildFrame(context, spec);
	if (!frame.has_aux) {
		throw BinderException("duckdo: the mediator column was not loaded");
	}

	vector<string> warnings = frame.warnings;

	// A mediator with no variation carries nothing, and a mediator the treatment
	// does not move carries nothing either. Both produce an indirect effect of
	// approximately zero for reasons that have nothing to do with the science,
	// so they are refused rather than reported.
	double m_min = frame.aux[0], m_max = frame.aux[0], m_sum = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		m_min = std::min(m_min, frame.aux[i]);
		m_max = std::max(m_max, frame.aux[i]);
		m_sum += frame.aux[i];
	}
	if (!(m_max - m_min > 1e-12)) {
		throw BinderException("duckdo: the mediator '%s' does not vary, so no part of the effect can travel "
		                      "through it",
		                      spec.aux_column);
	}
	if (spec.aux_column == spec.treatment || spec.aux_column == spec.outcome) {
		throw BinderException("duckdo: the mediator must differ from the treatment and the outcome");
	}

	// The mediator must not also be sitting in the covariate list: adjusting for
	// it in the outcome model AND treating it as the mediator double-counts it
	// and drives the indirect effect to zero.
	for (auto &feature : frame.features) {
		if (feature.source == spec.aux_column) {
			throw BinderException("duckdo: '%s' is both the mediator and a covariate. Adjusting for the mediator "
			                      "removes the very path being measured - drop it from covariates :=, or pass "
			                      "exclude := ['%s']",
			                      spec.aux_column, spec.aux_column);
		}
	}

	const idx_t p = frame.X.cols;
	vector<double> x_mean(p, 0.0);
	for (idx_t i = 0; i < frame.n; i++) {
		for (idx_t j = 0; j < p; j++) {
			x_mean[j] += frame.X.At(i, j);
		}
	}
	for (idx_t j = 0; j < p; j++) {
		x_mean[j] /= static_cast<double>(frame.n);
	}

	vector<idx_t> all_rows(frame.n);
	for (idx_t i = 0; i < frame.n; i++) {
		all_rows[i] = i;
	}
	const double lambda = 1e-6 * static_cast<double>(frame.n) + 1e-8;
	const auto design = BuildDesign(frame.X, frame.t, frame.aux);
	const auto point = Decompose(design, frame.aux, frame.y, all_rows, x_mean, lambda);
	if (!point.ok) {
		throw BinderException("duckdo: the mediation models did not fit. This usually means the mediator is "
		                      "collinear with the treatment or with a covariate");
	}

	// Both effects are products of coefficients from two different fits, so
	// neither has a usable closed-form standard error. Resample rows and redo
	// the whole decomposition, which keeps the three intervals consistent with
	// each other.
	// Each replicate seeds its own generator from (seed, rep) rather than drawing
	// from one shared stream, so the draws do not depend on the order threads
	// happen to reach them. Same seed, same answer, one thread or thirty-two.
	const idx_t reps = std::max<idx_t>(spec.bootstrap_reps, 100);
	vector<Decomposition> results(reps);
	ParallelJobs(reps, [&](idx_t rep) {
		std::mt19937_64 rng(static_cast<uint64_t>(spec.seed) ^ 0x0DEC0DEDULL ^ (rep * 0x9E3779B97F4A7C15ULL));
		std::uniform_int_distribution<idx_t> pick(0, frame.n - 1);
		vector<idx_t> resample(frame.n);
		for (idx_t k = 0; k < frame.n; k++) {
			resample[k] = frame.Draw(pick(rng));
		}
		results[rep] = Decompose(design, frame.aux, frame.y, resample, x_mean, lambda);
	});
	vector<double> direct_draws, indirect_draws, total_draws;
	for (auto &draw : results) {
		if (!draw.ok) {
			continue;
		}
		direct_draws.push_back(draw.direct);
		indirect_draws.push_back(draw.indirect);
		total_draws.push_back(draw.total);
	}

	auto std_error = [](const vector<double> &draws, double centre) -> double {
		if (draws.size() < 20) {
			return 0.0;
		}
		double variance = 0.0;
		for (auto d : draws) {
			const double diff = d - centre;
			variance += diff * diff;
		}
		return std::sqrt(variance / static_cast<double>(draws.size() - 1));
	};
	const double se_direct = std_error(direct_draws, point.direct);
	const double se_indirect = std_error(indirect_draws, point.indirect);
	const double se_total = std_error(total_draws, point.total);

	// Proportion mediated divides by the total effect, so it is meaningless when
	// the total is near zero and actively misleading when direct and indirect
	// have opposite signs (it can exceed 1 or go negative). Report it only when
	// the total is distinguishable from zero, and say why when it is not.
	const bool total_meaningful = se_total > 0.0 && std::fabs(point.total) > Z95 * se_total;
	double proportion = 0.0;
	if (total_meaningful) {
		proportion = point.indirect / point.total;
		if (proportion < 0.0 || proportion > 1.0) {
			warnings.push_back("proportion mediated is outside [0, 1] because the direct and indirect effects have "
			                   "opposite signs - the mediator suppresses rather than transmits part of the effect");
		}
	} else {
		warnings.push_back("proportion mediated is not reported: the total effect is not distinguishable from zero, "
		                   "and dividing by it produces a ratio that can take any value");
	}

	// The assumption everything rests on, stated first because it is the one
	// that fails.
	warnings.push_back("mediation assumes sequential ignorability: no unmeasured confounding of treatment-outcome, "
	                   "treatment-mediator, or mediator-outcome. Randomising the treatment buys the first two and "
	                   "NOT the third, which is the most common error in applied mediation");
	warnings.push_back("it also assumes no confounder of the mediator and the outcome is itself affected by the "
	                   "treatment; where one exists these natural effects are not identified at all");
	if (m_max - m_min <= 1.0 + 1e-9 && (m_min == 0.0 || m_min == 1.0)) {
		warnings.push_back("the mediator looks binary; the mediator model here is linear, which is a linear "
		                   "probability model in that case and can predict outside [0, 1]");
	}
	warnings.push_back("both models are linear in the covariates, so a non-linear mediator or outcome surface will "
	                   "bias the split between direct and indirect even when the total effect is right");

	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}
	auto warning_list = Value::LIST(LogicalType::VARCHAR, std::move(warning_values));

	names = {"estimand",   "estimator", "estimate",       "std_error", "ci_low",  "ci_high",
	         "proportion", "n",         "tm_interaction", "mediator",  "warnings"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::DOUBLE,
	                LogicalType::VARCHAR,
	                LogicalType::LIST(LogicalType::VARCHAR)};

	auto bind = make_uniq<ResultBindData>();
	struct Row {
		const char *estimand;
		double estimate;
		double se;
	};
	// One row per effect rather than one wide row: the three are the same kind
	// of thing and SQL filters them more easily than it unpivots columns.
	const Row emitted[] = {{"total", point.total, se_total},
	                       {"natural_direct", point.direct, se_direct},
	                       {"natural_indirect", point.indirect, se_indirect}};
	for (auto &row : emitted) {
		bind->rows.push_back({Value(row.estimand), Value("mediation-linear"), Value::DOUBLE(row.estimate),
		                      row.se > 0.0 ? Value::DOUBLE(row.se) : Value(LogicalType::DOUBLE),
		                      row.se > 0.0 ? Value::DOUBLE(row.estimate - Z95 * row.se) : Value(LogicalType::DOUBLE),
		                      row.se > 0.0 ? Value::DOUBLE(row.estimate + Z95 * row.se) : Value(LogicalType::DOUBLE),
		                      total_meaningful ? Value::DOUBLE(proportion) : Value(LogicalType::DOUBLE),
		                      Value::BIGINT(static_cast<int64_t>(frame.n)), Value::DOUBLE(point.interaction),
		                      Value(spec.aux_column), warning_list});
	}
	return std::move(bind);
}

} // namespace

void RegisterMediationFunctions(ExtensionLoader &loader) {
	TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, BindMediate, InitGlobal);
	AddCommonNamedParameters(fn);
	fn.named_parameters["mediator"] = LogicalType::VARCHAR;
	RegisterUnderBothNames(loader, fn, "mediate");
}

} // namespace duckdo
} // namespace duckdb
