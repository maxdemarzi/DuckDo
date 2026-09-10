//===----------------------------------------------------------------------===//
// Phase 3: diagnostics and refutation.
//
// The point of these functions is that assumptions become rows. Every one
// returns a boolean verdict column so it can be asserted on in a dbt test or a
// CI query, rather than read off a chart by a human who is already convinced.
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

struct DiagGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<DiagGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<DiagGlobalState>();
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

//! Weighted mean and variance of one encoded feature within one arm.
void ArmMoments(const CausalFrame &frame, idx_t feature, double arm, const vector<double> &weights, double &mean,
                double &var) {
	double sw = 0.0, acc = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		if (frame.t[i] != arm) {
			continue;
		}
		const double w = weights.empty() ? 1.0 : weights[i];
		sw += w;
		acc += w * frame.X.At(i, feature);
	}
	mean = sw > 0.0 ? acc / sw : 0.0;
	double vacc = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		if (frame.t[i] != arm) {
			continue;
		}
		const double w = weights.empty() ? 1.0 : weights[i];
		const double d = frame.X.At(i, feature) - mean;
		vacc += w * d * d;
	}
	var = sw > 0.0 ? vacc / sw : 0.0;
}

double StandardizedDifference(const CausalFrame &frame, idx_t feature, const vector<double> &weights) {
	double m1, v1, m0, v0;
	ArmMoments(frame, feature, 1.0, weights, m1, v1);
	ArmMoments(frame, feature, 0.0, weights, m0, v0);
	const double pooled = std::sqrt(std::max((v1 + v0) / 2.0, 1e-12));
	return (m1 - m0) / pooled;
}

//! IPW weights targeting the ATE.
vector<double> AteWeights(const CausalFrame &frame, const vector<double> &e) {
	vector<double> w(frame.n, 1.0);
	for (idx_t i = 0; i < frame.n; i++) {
		w[i] = frame.t[i] == 1.0 ? 1.0 / e[i] : 1.0 / (1.0 - e[i]);
	}
	return w;
}

// --- do_balance -------------------------------------------------------------

unique_ptr<FunctionData> BindBalance(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	auto frame = BuildFrame(context, spec);
	auto e = FitPropensity(frame, spec.folds, spec.seed);
	auto weights = AteWeights(frame, e);

	names = {"covariate", "feature", "smd_raw", "smd_weighted", "variance_ratio", "balanced"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::BOOLEAN};

	auto bind = make_uniq<ResultBindData>();
	for (idx_t j = 0; j < frame.features.size(); j++) {
		const double raw = StandardizedDifference(frame, j, {});
		const double weighted = StandardizedDifference(frame, j, weights);
		double m1, v1, m0, v0;
		ArmMoments(frame, j, 1.0, {}, m1, v1);
		ArmMoments(frame, j, 0.0, {}, m0, v0);
		const double ratio = v0 > 1e-12 ? v1 / v0 : 0.0;
		bind->rows.push_back({Value(frame.features[j].source), Value(frame.features[j].name), Value::DOUBLE(raw),
		                      Value::DOUBLE(weighted), Value::DOUBLE(ratio),
		                      Value::BOOLEAN(std::fabs(weighted) < 0.1)});
	}
	return std::move(bind);
}

// --- do_overlap -------------------------------------------------------------

unique_ptr<FunctionData> BindOverlap(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	auto frame = BuildFrame(context, spec);
	auto e = FitPropensity(frame, spec.folds, spec.seed);

	names = {"bucket", "ps_low", "ps_high", "n_treated", "n_control", "off_support"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BOOLEAN};

	const idx_t buckets = 10;
	vector<idx_t> treated(buckets, 0), control(buckets, 0);
	for (idx_t i = 0; i < frame.n; i++) {
		idx_t b = static_cast<idx_t>(e[i] * static_cast<double>(buckets));
		if (b >= buckets) {
			b = buckets - 1;
		}
		(frame.t[i] == 1.0 ? treated : control)[b]++;
	}

	auto bind = make_uniq<ResultBindData>();
	for (idx_t b = 0; b < buckets; b++) {
		// A bucket with rows from only one arm has no counterfactual support:
		// nothing in the data says what the other arm would have done there.
		const bool off = (treated[b] == 0) != (control[b] == 0);
		bind->rows.push_back({Value::BIGINT(static_cast<int64_t>(b)),
		                      Value::DOUBLE(static_cast<double>(b) / static_cast<double>(buckets)),
		                      Value::DOUBLE(static_cast<double>(b + 1) / static_cast<double>(buckets)),
		                      Value::BIGINT(static_cast<int64_t>(treated[b])),
		                      Value::BIGINT(static_cast<int64_t>(control[b])), Value::BOOLEAN(off)});
	}
	return std::move(bind);
}

// --- do_diagnose ------------------------------------------------------------

void AddCheck(ResultBindData &bind, const string &check, const string &status, const string &detail,
              const string &severity) {
	bind.rows.push_back({Value(check), Value(status), Value(detail), Value(severity)});
}

unique_ptr<FunctionData> BindDiagnose(ClientContext &context, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	auto frame = BuildFrame(context, spec);
	auto e = FitPropensity(frame, spec.folds, spec.seed);
	auto weights = AteWeights(frame, e);

	// `check` is a SQL reserved word, so the column would need quoting on every
	// select. Name it check_name instead.
	names = {"check_name", "status", "detail", "severity"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	auto bind = make_uniq<ResultBindData>();

	// Sample size and treatment prevalence.
	const idx_t n_control = frame.n - frame.n_treated;
	const double prevalence = static_cast<double>(frame.n_treated) / static_cast<double>(frame.n);
	AddCheck(*bind, "sample_size", frame.n >= 200 ? "pass" : "warn",
	         StringUtil::Format("%llu rows: %llu treated, %llu control", static_cast<unsigned long long>(frame.n),
	                            static_cast<unsigned long long>(frame.n_treated),
	                            static_cast<unsigned long long>(n_control)),
	         frame.n >= 200 ? "info" : "medium");
	AddCheck(*bind, "treatment_prevalence", (prevalence > 0.05 && prevalence < 0.95) ? "pass" : "warn",
	         StringUtil::Format("%.1f%% of rows are treated ('%s' vs '%s')", prevalence * 100.0, frame.treated_label,
	                            frame.control_label),
	         (prevalence > 0.05 && prevalence < 0.95) ? "info" : "medium");

	// Positivity: how much probability mass sits at the extremes.
	idx_t extreme = 0;
	double min_e = 1.0, max_e = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		min_e = std::min(min_e, e[i]);
		max_e = std::max(max_e, e[i]);
		if (e[i] < 0.05 || e[i] > 0.95) {
			extreme++;
		}
	}
	const double extreme_share = static_cast<double>(extreme) / static_cast<double>(frame.n);
	AddCheck(*bind, "positivity", extreme_share < 0.05 ? "pass" : (extreme_share < 0.2 ? "warn" : "fail"),
	         StringUtil::Format("propensity spans [%.3f, %.3f]; %.1f%% of rows fall outside [0.05, 0.95]", min_e, max_e,
	                            extreme_share * 100.0),
	         extreme_share < 0.05 ? "info" : (extreme_share < 0.2 ? "medium" : "high"));

	// Covariate balance after weighting.
	double worst = 0.0;
	string worst_name = "(none)";
	for (idx_t j = 0; j < frame.features.size(); j++) {
		const double smd = std::fabs(StandardizedDifference(frame, j, weights));
		if (smd > worst) {
			worst = smd;
			worst_name = frame.features[j].name;
		}
	}
	AddCheck(*bind, "balance", worst < 0.1 ? "pass" : (worst < 0.25 ? "warn" : "fail"),
	         StringUtil::Format("largest weighted SMD is %.3f on '%s'", worst, worst_name),
	         worst < 0.1 ? "info" : (worst < 0.25 ? "medium" : "high"));

	// Outcome variation.
	if (!spec.outcome.empty()) {
		const double sd = StdDev(frame.y);
		AddCheck(*bind, "outcome_variation", sd > 1e-9 ? "pass" : "fail",
		         StringUtil::Format("outcome standard deviation is %.6g", sd), sd > 1e-9 ? "info" : "high");
	}

	// Missingness and dimensionality.
	AddCheck(*bind, "missing_data", frame.n_rows_with_missing == 0 ? "pass" : "warn",
	         StringUtil::Format("%llu rows had at least one missing covariate; they were imputed, not dropped",
	                            static_cast<unsigned long long>(frame.n_rows_with_missing)),
	         frame.n_rows_with_missing == 0 ? "info" : "low");
	const double ratio = static_cast<double>(frame.n) / static_cast<double>(std::max<idx_t>(frame.Cols(), 1));
	AddCheck(*bind, "dimensionality", ratio >= 20.0 ? "pass" : "warn",
	         StringUtil::Format("%llu encoded features for %llu rows (%.1f rows per feature)",
	                            static_cast<unsigned long long>(frame.Cols()), static_cast<unsigned long long>(frame.n),
	                            ratio),
	         ratio >= 20.0 ? "info" : "medium");

	for (auto &w : frame.warnings) {
		AddCheck(*bind, "encoding", "warn", w, "low");
	}
	return std::move(bind);
}

// --- do_refute --------------------------------------------------------------

//! Build a frame restricted to `rows`, preserving encoding and metadata.
CausalFrame SubFrame(const CausalFrame &src, const vector<idx_t> &rows) {
	CausalFrame out = src;
	out.n = rows.size();
	out.X.Resize(rows.size(), src.X.cols);
	out.t.assign(rows.size(), 0.0);
	out.y.assign(rows.size(), 0.0);
	out.source_row.assign(rows.size(), 0);
	if (src.has_id) {
		out.ids.assign(rows.size(), string());
	}
	out.n_treated = 0;
	for (idx_t i = 0; i < rows.size(); i++) {
		const idx_t r = rows[i];
		for (idx_t j = 0; j < src.X.cols; j++) {
			out.X.At(i, j) = src.X.At(r, j);
		}
		out.t[i] = src.t[r];
		out.y[i] = src.y[r];
		out.source_row[i] = src.source_row[r];
		if (src.has_id) {
			out.ids[i] = src.ids[r];
		}
		if (out.t[i] == 1.0) {
			out.n_treated++;
		}
	}
	// A subset is a different set of rows, so the parent's canonical order does
	// not describe it.
	BuildCanonicalOrder(out);
	return out;
}

//! Append one standardised column to a frame's design matrix.
void AppendFeature(CausalFrame &frame, const vector<double> &values, const string &name) {
	const double mean = Mean(values);
	const double sd = std::max(StdDev(values), 1e-12);
	Matrix bigger(frame.n, frame.X.cols + 1);
	for (idx_t i = 0; i < frame.n; i++) {
		for (idx_t j = 0; j < frame.X.cols; j++) {
			bigger.At(i, j) = frame.X.At(i, j);
		}
		bigger.At(i, frame.X.cols) = (values[i] - mean) / sd;
	}
	FeatureInfo info;
	info.name = name;
	info.source = name;
	info.kind = FeatureKind::NUMERIC;
	info.center = mean;
	info.scale = sd;
	frame.features.push_back(info);
	frame.X = std::move(bigger);
	// The rows now carry a column they did not before, so the content-derived
	// order has to be rebuilt before anything draws against it.
	BuildCanonicalOrder(frame);
}

unique_ptr<FunctionData> BindRefute(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_refute requires outcome := '<column>'");
	}
	auto frame = BuildFrame(context, spec);
	const auto original = EstimateEffect(frame, spec, Estimand::ATE);

	const string method = StringUtil::Lower(spec.refute_method);
	std::mt19937_64 rng(static_cast<uint64_t>(spec.seed) ^ 0xD1B54A32D192ED03ULL);

	CausalFrame perturbed = frame;
	string detail;
	bool zero_expected = false;

	if (method == "placebo_treatment") {
		// Permute the arm labels across canonical ranks. Shuffling the vector in
		// place would permute them across storage positions, which makes the
		// refutation depend on how the table happens to be laid out.
		vector<double> labels;
		labels.reserve(frame.n);
		for (idx_t rank = 0; rank < frame.n; rank++) {
			labels.push_back(frame.t[frame.Draw(rank)]);
		}
		std::shuffle(labels.begin(), labels.end(), rng);
		for (idx_t rank = 0; rank < frame.n; rank++) {
			perturbed.t[frame.Draw(rank)] = labels[rank];
		}
		BuildCanonicalOrder(perturbed);
		perturbed.n_treated = 0;
		for (auto v : perturbed.t) {
			if (v == 1.0) {
				perturbed.n_treated++;
			}
		}
		zero_expected = true;
		detail = "treatment permuted at random; a real effect should collapse to zero";
	} else if (method == "random_common_cause") {
		std::normal_distribution<double> normal(0.0, 1.0);
		vector<double> noise(frame.n);
		for (idx_t rank = 0; rank < frame.n; rank++) {
			noise[frame.Draw(rank)] = normal(rng);
		}
		AppendFeature(perturbed, noise, "__duckdo_random_common_cause");
		detail = "an irrelevant covariate was added; the estimate should not move";
	} else if (method == "subset") {
		vector<idx_t> rows;
		std::uniform_real_distribution<double> uniform(0.0, 1.0);
		for (idx_t rank = 0; rank < frame.n; rank++) {
			if (uniform(rng) < spec.fraction) {
				rows.push_back(frame.Draw(rank));
			}
		}
		if (rows.size() < 16) {
			throw BinderException("duckdo: fraction := %g left too few rows to re-estimate on", spec.fraction);
		}
		perturbed = SubFrame(frame, rows);
		detail = StringUtil::Format("re-estimated on a random %.0f%% subset; the estimate should be stable",
		                            spec.fraction * 100.0);
	} else if (method == "bootstrap") {
		std::uniform_int_distribution<idx_t> pick(0, frame.n - 1);
		vector<idx_t> rows(frame.n);
		for (auto &r : rows) {
			r = frame.Draw(pick(rng));
		}
		perturbed = SubFrame(frame, rows);
		detail = "re-estimated on a bootstrap resample; the estimate should be stable";
	} else if (method == "unobserved_confounder") {
		// Simulate a confounder correlated with both arms and the outcome at the
		// requested strength, then see how far the estimate moves.
		std::normal_distribution<double> normal(0.0, 1.0);
		const double y_sd = std::max(StdDev(frame.y), 1e-12);
		const double y_mean = Mean(frame.y);
		vector<double> u(frame.n);
		for (idx_t rank = 0; rank < frame.n; rank++) {
			const idx_t i = frame.Draw(rank);
			u[i] = spec.confounder_strength * (frame.t[i] - 0.5) * 2.0 +
			       spec.confounder_strength * (frame.y[i] - y_mean) / y_sd + normal(rng);
		}
		AppendFeature(perturbed, u, "__duckdo_simulated_confounder");
		detail = StringUtil::Format("a confounder of strength %.2f was simulated on both the treatment and the outcome",
		                            spec.confounder_strength);
	} else {
		throw BinderException("duckdo: unknown refutation method '%s'. Supported: placebo_treatment, "
		                      "random_common_cause, subset, bootstrap, unobserved_confounder",
		                      spec.refute_method);
	}

	const auto refuted = EstimateEffect(perturbed, spec, Estimand::ATE);
	const double difference = refuted.estimate - original.estimate;
	const double tolerance = 2.0 * std::max(original.std_error, 1e-12);
	const bool passed = zero_expected ? std::fabs(refuted.estimate) <= tolerance : std::fabs(difference) <= tolerance;

	names = {"method", "original_estimate", "refuted_estimate", "difference", "tolerance", "passed", "detail"};
	return_types = {LogicalType::VARCHAR, LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::BOOLEAN, LogicalType::VARCHAR};
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value(method), Value::DOUBLE(original.estimate), Value::DOUBLE(refuted.estimate),
	                      Value::DOUBLE(difference), Value::DOUBLE(tolerance), Value::BOOLEAN(passed), Value(detail)});
	return std::move(bind);
}

// --- do_sensitivity ---------------------------------------------------------

unique_ptr<FunctionData> BindSensitivity(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_sensitivity requires outcome := '<column>'");
	}
	auto frame = BuildFrame(context, spec);
	CausalSpec pl_spec = spec;
	pl_spec.estimator = "dml";
	const auto effect = EstimateEffect(frame, pl_spec, Estimand::ATE);

	const double t_stat = effect.std_error > 0.0 ? effect.estimate / effect.std_error : 0.0;
	const double dof = static_cast<double>(frame.n > frame.Cols() + 2 ? frame.n - frame.Cols() - 2 : 1);

	// Cinelli & Hazlett robustness value: the partial R-squared an unobserved
	// confounder would need with BOTH treatment and outcome to explain the
	// estimate away.
	auto robustness = [&](double t_value) {
		const double f = std::fabs(t_value) / std::sqrt(dof);
		if (!(f > 0.0)) {
			return 0.0;
		}
		const double f2 = f * f;
		return 0.5 * (std::sqrt(f2 * f2 + 4.0 * f2) - f2);
	};
	const double rv = robustness(t_stat);
	const double t_ci = std::max(std::fabs(t_stat) - Z95, 0.0);
	const double rv_ci = robustness(t_ci);

	// E-value. For a continuous outcome we go through VanderWeele & Ding's
	// approximate risk ratio for a standardised mean difference.
	const double y_sd = std::max(StdDev(frame.y), 1e-12);
	auto e_value = [&](double estimate) {
		double rr = frame.binary_outcome ? std::exp(estimate) : std::exp(0.91 * (estimate / y_sd));
		if (rr < 1.0 && rr > 0.0) {
			rr = 1.0 / rr;
		}
		if (!(rr > 1.0)) {
			return 1.0;
		}
		return rr + std::sqrt(rr * (rr - 1.0));
	};
	const double ev = e_value(effect.estimate);
	// The CI limit closest to the null is what the E-value for the interval uses.
	double ci_near_null = 0.0;
	if (effect.ci_low > 0.0) {
		ci_near_null = effect.ci_low;
	} else if (effect.ci_high < 0.0) {
		ci_near_null = effect.ci_high;
	}
	const double ev_ci = ci_near_null == 0.0 ? 1.0 : e_value(ci_near_null);

	string interpretation;
	if (rv <= 0.0) {
		interpretation = "the estimate is not distinguishable from zero, so there is nothing for a confounder to "
		                 "explain away";
	} else {
		interpretation = StringUtil::Format(
		    "an unobserved confounder would have to explain %.1f%% of the residual variance of both the treatment and "
		    "the outcome to reduce this estimate to zero, and %.1f%% to push the confidence interval across zero. "
		    "Compare that against the strongest covariate you already measured",
		    rv * 100.0, rv_ci * 100.0);
	}
	if (frame.binary_outcome) {
		interpretation += ". The E-value assumes the estimate is on a risk-difference scale";
	}

	names = {"estimate", "std_error",  "robustness_value", "robustness_value_ci",
	         "e_value",  "e_value_ci", "interpretation"};
	return_types = {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR};
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value::DOUBLE(effect.estimate), Value::DOUBLE(effect.std_error), Value::DOUBLE(rv),
	                      Value::DOUBLE(rv_ci), Value::DOUBLE(ev), Value::DOUBLE(ev_ci), Value(interpretation)});
	return std::move(bind);
}

} // namespace

void RegisterDiagnosticFunctions(ExtensionLoader &loader) {
	struct Entry {
		const char *name;
		table_function_bind_t bind;
	};
	const Entry entries[] = {{"balance", BindBalance},
	                         {"overlap", BindOverlap},
	                         {"diagnose", BindDiagnose},
	                         {"refute", BindRefute},
	                         {"sensitivity", BindSensitivity}};
	for (auto &entry : entries) {
		TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, entry.bind, InitGlobal);
		AddCommonNamedParameters(fn);
		fn.named_parameters["method"] = LogicalType::VARCHAR;
		fn.named_parameters["fraction"] = LogicalType::DOUBLE;
		fn.named_parameters["strength"] = LogicalType::DOUBLE;
		RegisterUnderBothNames(loader, fn, entry.name);
	}
}

} // namespace duckdo
} // namespace duckdb
