//===----------------------------------------------------------------------===//
// Time-varying treatment: marginal structural models by inverse-probability
// weighting.
//
// This exists because of one situation that ordinary adjustment cannot handle
// at all, in either direction. A confounder L_t that is measured over time,
// affects the treatment at t, affects the outcome - and is itself affected by
// the treatment before t.
//
//   * Do not adjust for L_t and it confounds A_t with Y.
//   * Do adjust for L_t and you block part of A_{t-1}'s effect, because that
//     effect travels through L_t. Conditioning on it also opens a collider path
//     when L_t shares an unmeasured cause with Y.
//
// There is no covariate list that fixes this. Robins' answer is to stop
// adjusting and start reweighting: build a pseudo-population in which treatment
// at each period is independent of the history that predicted it, then fit a
// simple model of the outcome on cumulative treatment in that population.
//
//   denominator  P(A_t | A_{t-1}, L_t, V)   what actually drove treatment
//   numerator    P(A_t | A_{t-1}, V)        the part we are content to keep
//   SW_i = prod_t numerator / denominator
//
// The numerator is what makes the weights *stabilised*: without it the weights
// have enormous variance and the estimate is dominated by a handful of units.
// With it, the mean weight sits near 1, and a mean far from 1 is a signal that
// the weight model is misspecified - so it is reported rather than hidden.
//
// The structural model is then E[Y | cumulative treatment] fitted by weighted
// least squares, and its coefficient is the effect of one additional treated
// period.
//
// do_msm_rmst takes the same panel with an event in place of an outcome, and
// compares two regimes - treated in every period, and in none - on expected
// event-free periods within a horizon. It needs no structural model: each unit
// counts toward a regime only while its treatment matches it, and its
// person-periods are weighted by the inverse probability of having stayed on
// it, and of not having dropped out, given the measured history. A weighted
// Kaplan-Meier per regime then gives the curve that regime would have produced.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

namespace duckdb {
namespace duckdo {

namespace {

struct LongGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<LongGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<LongGlobalState>();
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

//! One (unit, period) observation, in the order the periods actually occur.
struct LongRow {
	idx_t unit = 0;
	idx_t period = 0;
	double treatment = 0.0;
	double outcome = 0.0;
	bool outcome_observed = false;
	vector<double> covariates;
};

struct LongPanel {
	vector<string> units;
	vector<string> period_labels;
	vector<string> covariate_names;
	//! Row-major by unit: rows[unit] holds that unit's periods in order.
	vector<vector<LongRow>> by_unit;
	idx_t n_covariates = 0;
};

string RequireParam(const named_parameter_map_t &named, const char *key, const char *fn) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		throw BinderException("duckdo: %s requires %s := '<column>'", fn, key);
	}
	return entry->second.ToString();
}

vector<string> ListParam(const named_parameter_map_t &named, const char *key) {
	vector<string> out;
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		return out;
	}
	for (auto &value : ListValue::GetChildren(entry->second)) {
		if (!value.IsNull()) {
			out.push_back(value.ToString());
		}
	}
	return out;
}

//! Read the long-format table. Unlike the difference-in-differences loader this
//! keeps the covariates at every period, because those are exactly what the
//! weight model needs and what makes the problem hard.
LongPanel LoadLong(ClientContext &context, const string &relation, const string &unit_col, const string &period_col,
                   const string &treatment_col, const string &outcome_col, const vector<string> &covariates,
                   const char *fn) {
	const string rel = RelationSql(relation);

	// Periods are ordered on their own type so that 2 does not sort after 10,
	// then carried as text to key the join.
	auto order_probe =
	    RunQuery(context,
	             "SELECT DISTINCT CAST(" + QuoteIdentifier(period_col) + " AS VARCHAR) AS p FROM " + rel + " WHERE " +
	                 QuoteIdentifier(period_col) + " IS NOT NULL ORDER BY " + QuoteIdentifier(period_col),
	             string(fn) + " ordering the periods of " + relation);

	LongPanel panel;
	std::map<string, idx_t> period_index;
	for (idx_t r = 0; r < order_probe->RowCount(); r++) {
		const string label = order_probe->GetValue(0, r).ToString();
		period_index[label] = panel.period_labels.size();
		panel.period_labels.push_back(label);
	}
	if (panel.period_labels.size() < 2) {
		throw BinderException("duckdo: %s needs at least two periods in '%s'; found %llu. With one period there is no "
		                      "time-varying treatment and do_ate is the right function",
		                      fn, period_col, static_cast<unsigned long long>(panel.period_labels.size()));
	}

	string projection = "CAST(" + QuoteIdentifier(unit_col) + " AS VARCHAR) AS u, CAST(" + QuoteIdentifier(period_col) +
	                    " AS VARCHAR) AS p, CAST(coalesce(CAST(" + QuoteIdentifier(treatment_col) +
	                    " AS BOOLEAN), false) AS DOUBLE) AS a, CAST(" + QuoteIdentifier(outcome_col) +
	                    " AS DOUBLE) AS y";
	for (idx_t c = 0; c < covariates.size(); c++) {
		projection += ", CAST(" + QuoteIdentifier(covariates[c]) + " AS DOUBLE) AS l" + std::to_string(c);
	}
	auto data = RunQuery(context,
	                     "SELECT " + projection + " FROM " + rel + " WHERE " + QuoteIdentifier(unit_col) +
	                         " IS NOT NULL AND " + QuoteIdentifier(period_col) + " IS NOT NULL",
	                     string(fn) + " reading the panel from " + relation);

	panel.covariate_names = covariates;
	panel.n_covariates = covariates.size();

	std::map<string, idx_t> unit_index;
	for (idx_t r = 0; r < data->RowCount(); r++) {
		const string unit = data->GetValue(0, r).ToString();
		auto found = unit_index.find(unit);
		idx_t u;
		if (found == unit_index.end()) {
			u = panel.units.size();
			unit_index[unit] = u;
			panel.units.push_back(unit);
			panel.by_unit.emplace_back();
		} else {
			u = found->second;
		}
		auto period_found = period_index.find(data->GetValue(1, r).ToString());
		if (period_found == period_index.end()) {
			continue;
		}
		LongRow row;
		row.unit = u;
		row.period = period_found->second;
		row.treatment = data->GetValue(2, r).GetValue<double>() >= 0.5 ? 1.0 : 0.0;
		const Value outcome = data->GetValue(3, r);
		row.outcome_observed = !outcome.IsNull();
		row.outcome = row.outcome_observed ? outcome.GetValue<double>() : 0.0;
		row.covariates.resize(panel.n_covariates, 0.0);
		for (idx_t c = 0; c < panel.n_covariates; c++) {
			const Value v = data->GetValue(4 + c, r);
			row.covariates[c] = v.IsNull() ? 0.0 : v.GetValue<double>();
		}
		panel.by_unit[u].push_back(std::move(row));
	}

	for (auto &rows : panel.by_unit) {
		std::sort(rows.begin(), rows.end(), [](const LongRow &a, const LongRow &b) { return a.period < b.period; });
	}
	if (panel.units.size() < 20) {
		throw BinderException("duckdo: %s needs at least twenty units in '%s'; found %llu. Inverse-probability "
		                      "weights are noisy, and on a handful of units the estimate is dominated by whichever "
		                      "one happened to get the largest weight",
		                      fn, unit_col, static_cast<unsigned long long>(panel.units.size()));
	}
	return panel;
}

unique_ptr<FunctionData> BindMsm(ClientContext &context, TableFunctionBindInput &input,
                                 vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("duckdo: do_msm needs a table name or query as its first argument");
	}
	const string relation = input.inputs[0].ToString();
	const string unit_col = RequireParam(input.named_parameters, "unit", "do_msm");
	const string period_col = RequireParam(input.named_parameters, "period", "do_msm");
	const string treatment_col = RequireParam(input.named_parameters, "treatment", "do_msm");
	const string outcome_col = RequireParam(input.named_parameters, "outcome", "do_msm");
	const vector<string> covariates = ListParam(input.named_parameters, "covariates");
	const vector<string> baseline = ListParam(input.named_parameters, "baseline");
	// Weight truncation, as a two-sided quantile. 0 means none, which is the
	// default because truncation trades a positivity problem you can see for a
	// bias you cannot.
	double truncate = 0.0;
	auto truncate_entry = input.named_parameters.find("truncate");
	if (truncate_entry != input.named_parameters.end() && !truncate_entry->second.IsNull()) {
		truncate = truncate_entry->second.GetValue<double>();
		if (truncate < 0.0 || truncate >= 0.5) {
			throw BinderException("duckdo: truncate := must be in [0, 0.5); got %g. It is the quantile trimmed from "
			                      "each tail of the weight distribution, so 0.01 keeps the middle 98%%",
			                      truncate);
		}
	}

	SetNumericThreads(GetSettingIdx(context, "duckdo_threads", 0));
	auto panel = LoadLong(context, relation, unit_col, period_col, treatment_col, outcome_col, covariates, "do_msm");

	vector<string> warnings;
	if (covariates.empty()) {
		warnings.push_back("no covariates := given, so the weights condition only on treatment history. That is the "
		                   "right model only if nothing time-varying drives treatment - otherwise this is an "
		                   "unadjusted estimate wearing a marginal structural model's name");
	}
	// Baseline covariates enter both the numerator and the denominator. Anything
	// in `covariates` that is genuinely time-invariant would cancel too, but
	// naming them separately is what keeps the numerator model honest.
	std::map<string, idx_t> baseline_index;
	for (idx_t c = 0; c < panel.covariate_names.size(); c++) {
		for (auto &b : baseline) {
			if (panel.covariate_names[c] == b) {
				baseline_index[b] = c;
			}
		}
	}
	for (auto &b : baseline) {
		if (baseline_index.find(b) == baseline_index.end()) {
			throw BinderException("duckdo: baseline := ['%s'] must also appear in covariates := , because the "
			                      "numerator and denominator weight models have to be nested for the weights to "
			                      "stabilise",
			                      b);
		}
	}

	// --- 1. pooled treatment models --------------------------------------------
	//
	// One model across all periods with the period index as a feature, rather
	// than one model per period: with a handful of periods the per-period fits
	// are badly under-determined, and the pooled fit is the standard practice.
	//
	// Denominator design: [period, prior treatment, all covariates at t]
	// Numerator design:   [period, prior treatment, baseline covariates only]
	idx_t n_obs = 0;
	for (auto &rows : panel.by_unit) {
		n_obs += rows.size();
	}
	const idx_t den_cols = 2 + panel.n_covariates;
	const idx_t num_cols = 2 + baseline_index.size();
	Matrix den_design(n_obs, den_cols);
	Matrix num_design(n_obs, num_cols);
	vector<double> action(n_obs, 0.0);
	vector<idx_t> obs_unit(n_obs, 0);

	idx_t at = 0;
	for (idx_t u = 0; u < panel.by_unit.size(); u++) {
		double prior = 0.0;
		for (auto &row : panel.by_unit[u]) {
			den_design.At(at, 0) = static_cast<double>(row.period);
			den_design.At(at, 1) = prior;
			for (idx_t c = 0; c < panel.n_covariates; c++) {
				den_design.At(at, 2 + c) = row.covariates[c];
			}
			num_design.At(at, 0) = static_cast<double>(row.period);
			num_design.At(at, 1) = prior;
			idx_t k = 2;
			for (auto &entry : baseline_index) {
				num_design.At(at, k++) = row.covariates[entry.second];
			}
			action[at] = row.treatment;
			obs_unit[at] = u;
			prior = row.treatment;
			at++;
		}
	}

	vector<idx_t> obs_rows(n_obs);
	for (idx_t i = 0; i < n_obs; i++) {
		obs_rows[i] = i;
	}
	const double lambda = 1e-6 * static_cast<double>(n_obs) + 1e-8;
	auto denominator = FitLogistic(den_design, action, obs_rows, {}, lambda, 60);
	auto numerator = FitLogistic(num_design, action, obs_rows, {}, lambda, 60);

	// --- 2. stabilised weights -------------------------------------------------
	const idx_t n_units = panel.units.size();
	vector<double> weight(n_units, 1.0);
	vector<double> cumulative(n_units, 0.0);
	vector<double> final_outcome(n_units, 0.0);
	vector<uint8_t> usable(n_units, 1);
	double min_denominator = 1.0;

	at = 0;
	for (idx_t u = 0; u < panel.by_unit.size(); u++) {
		double product = 1.0;
		double cum = 0.0;
		bool have_outcome = false;
		double last = 0.0;
		for (auto &row : panel.by_unit[u]) {
			const double p_den = denominator.Predict(den_design.Row(at), den_cols);
			const double p_num = numerator.Predict(num_design.Row(at), num_cols);
			// The probability of the action actually taken, not of treatment.
			const double f_den = row.treatment >= 0.5 ? p_den : 1.0 - p_den;
			const double f_num = row.treatment >= 0.5 ? p_num : 1.0 - p_num;
			min_denominator = std::min(min_denominator, f_den);
			// A denominator at zero means this unit's observed history was
			// predicted to be impossible - a positivity violation. Flooring it
			// keeps the arithmetic finite; the floor is reported so the user can
			// see how close to the edge the data ran.
			product *= f_num / std::max(f_den, 1e-6);
			cum += row.treatment;
			if (row.outcome_observed) {
				last = row.outcome;
				have_outcome = true;
			}
			at++;
		}
		weight[u] = product;
		cumulative[u] = cum;
		final_outcome[u] = last;
		usable[u] = have_outcome ? 1 : 0;
	}

	vector<idx_t> unit_rows;
	for (idx_t u = 0; u < n_units; u++) {
		if (usable[u]) {
			unit_rows.push_back(u);
		}
	}

	// Truncation clamps rather than drops: dropping the units with extreme
	// weights would remove exactly the histories the estimate is extrapolating
	// over and quietly change the population being described.
	double truncated_at_low = 0.0, truncated_at_high = 0.0;
	idx_t truncated_count = 0;
	if (truncate > 0.0 && unit_rows.size() > 10) {
		vector<double> sorted;
		sorted.reserve(unit_rows.size());
		for (auto u : unit_rows) {
			sorted.push_back(weight[u]);
		}
		std::sort(sorted.begin(), sorted.end());
		const idx_t last = sorted.size() - 1;
		const idx_t lo_idx = static_cast<idx_t>(truncate * static_cast<double>(last));
		const idx_t hi_idx = static_cast<idx_t>((1.0 - truncate) * static_cast<double>(last));
		truncated_at_low = sorted[lo_idx];
		truncated_at_high = sorted[hi_idx];
		for (auto u : unit_rows) {
			if (weight[u] < truncated_at_low) {
				weight[u] = truncated_at_low;
				truncated_count++;
			} else if (weight[u] > truncated_at_high) {
				weight[u] = truncated_at_high;
				truncated_count++;
			}
		}
	}
	if (unit_rows.size() < 20) {
		throw BinderException("duckdo: only %llu units have an observed outcome, which is too few to fit a marginal "
		                      "structural model",
		                      static_cast<unsigned long long>(unit_rows.size()));
	}

	// Cumulative treatment has to vary, or there is no dose to respond to.
	double cum_min = cumulative[unit_rows[0]], cum_max = cumulative[unit_rows[0]];
	for (auto u : unit_rows) {
		cum_min = std::min(cum_min, cumulative[u]);
		cum_max = std::max(cum_max, cumulative[u]);
	}
	if (!(cum_max - cum_min > 0.5)) {
		throw BinderException("duckdo: every unit received the same total amount of treatment (%g periods), so no "
		                      "dose-response is estimable",
		                      cum_min);
	}

	// --- 3. the structural model ----------------------------------------------
	//
	// E[Y] = b0 + b1 * (periods treated), fitted in the weighted pseudo
	// population. b1 is the effect of one more treated period.
	Matrix msm_design(n_units, 1);
	for (idx_t u = 0; u < n_units; u++) {
		msm_design.At(u, 0) = cumulative[u];
	}
	vector<double> msm_weights;
	msm_weights.reserve(unit_rows.size());
	for (auto u : unit_rows) {
		msm_weights.push_back(weight[u]);
	}
	auto fit = FitRidgeWeightedWithSandwich(msm_design, final_outcome, unit_rows, msm_weights, 1e-8);
	const double estimate = fit.model.beta.size() > 1 ? fit.model.beta[1] : 0.0;
	double se = 0.0;
	if (fit.dim == 2 && fit.cov.size() == 4 && fit.cov[3] > 0.0) {
		se = std::sqrt(fit.cov[3]);
	}

	// --- 4. weight diagnostics -------------------------------------------------
	//
	// The mean stabilised weight should sit near 1. It is the cheapest available
	// check on the weight model, so it goes in the result rather than a log.
	double sum_w = 0.0, sum_w2 = 0.0, max_w = 0.0;
	for (auto u : unit_rows) {
		sum_w += weight[u];
		sum_w2 += weight[u] * weight[u];
		max_w = std::max(max_w, weight[u]);
	}
	const double mean_w = sum_w / static_cast<double>(unit_rows.size());
	const double effective_n = sum_w2 > 0.0 ? (sum_w * sum_w) / sum_w2 : 0.0;

	if (std::fabs(mean_w - 1.0) > 0.1) {
		warnings.push_back("the mean stabilised weight is " + StringUtil::Format("%.3f", mean_w) +
		                   " rather than about 1, which usually means the treatment model is misspecified. Treat "
		                   "the estimate as unreliable until that is fixed");
	}
	if (max_w > 10.0) {
		warnings.push_back("the largest stabilised weight is " + StringUtil::Format("%.1f", max_w) +
		                   "; a few units are carrying the estimate. This is what a near-violation of positivity "
		                   "looks like");
	}
	if (truncated_count > 0) {
		warnings.push_back("truncate := " + StringUtil::Format("%.3f", truncate) + " clamped " +
		                   std::to_string(truncated_count) + " weights into [" +
		                   StringUtil::Format("%.3f", truncated_at_low) + ", " +
		                   StringUtil::Format("%.3f", truncated_at_high) +
		                   "]. That trades variance for bias: the estimate is no longer unbiased even if every "
		                   "model is correct");
	}
	if (max_w > 10.0 && truncate <= 0.0) {
		warnings.push_back("truncate := 0.01 would clamp the tails of the weight distribution, trading some bias "
		                   "for a large reduction in variance");
	}
	if (min_denominator < 0.01) {
		warnings.push_back("some observed treatment histories had probability below 0.01 under the weight model, so "
		                   "the pseudo-population extrapolates rather than reweights");
	}
	if (effective_n < 0.5 * static_cast<double>(unit_rows.size())) {
		warnings.push_back("the weights cut the effective sample size to " + StringUtil::Format("%.0f", effective_n) +
		                   " from " + std::to_string(unit_rows.size()) + " units");
	}
	warnings.push_back("a marginal structural model assumes sequential exchangeability given the measured history: "
	                   "no unmeasured confounder of treatment and outcome at any period. It buys nothing against "
	                   "unmeasured confounding - what it buys is correct handling of measured confounders that the "
	                   "treatment itself affects, which no covariate adjustment can do");
	warnings.push_back("the structural model is linear in cumulative treated periods, so it assumes each additional "
	                   "period is worth the same and that only the total matters, not when it happened");
	warnings.push_back("the interval treats the weights as known rather than estimated, which is conservative for "
	                   "stabilised weights - a unit-level bootstrap would be tighter and slower");

	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}

	names = {"estimand", "estimator", "estimate",    "std_error",  "ci_low",      "ci_high",
	         "n_units",  "n_periods", "mean_weight", "max_weight", "effective_n", "warnings"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::LIST(LogicalType::VARCHAR)};

	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value("effect per treated period"), Value("msm-iptw"), Value::DOUBLE(estimate),
	                      se > 0.0 ? Value::DOUBLE(se) : Value(LogicalType::DOUBLE),
	                      se > 0.0 ? Value::DOUBLE(estimate - Z95 * se) : Value(LogicalType::DOUBLE),
	                      se > 0.0 ? Value::DOUBLE(estimate + Z95 * se) : Value(LogicalType::DOUBLE),
	                      Value::BIGINT(static_cast<int64_t>(unit_rows.size())),
	                      Value::BIGINT(static_cast<int64_t>(panel.period_labels.size())), Value::DOUBLE(mean_w),
	                      Value::DOUBLE(max_w), Value::DOUBLE(effective_n),
	                      Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

//! A count-valued named parameter, or the fallback.
idx_t CountParam(const named_parameter_map_t &named, const char *key, idx_t fallback) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		return fallback;
	}
	const int64_t value = entry->second.GetValue<int64_t>();
	if (value < 0) {
		throw BinderException("duckdo: %s := must not be negative; got %lld", key, static_cast<long long>(value));
	}
	return static_cast<idx_t>(value);
}

//! Per-period weighted hazards for the two regimes, from one sample of units.
struct RegimeHazards {
	//! [regime][period], regime 0 never treated and 1 always treated.
	vector<double> hazard[2];
	//! Unweighted units following the regime and at risk, per period.
	vector<double> followers[2];
	//! Whether the regime had anyone at risk in each period.
	vector<uint8_t> covered[2];
	double max_weight = 0.0;
	idx_t dropouts = 0;
};

//! Clone, censor and weight. Each unit is copied into both regimes and counts
//! toward one only while its treatment matches it. Every person-period it
//! contributes is weighted by the inverse probability of the treatment history
//! that kept it on the regime, and of not having dropped out before it, both from
//! pooled logistic models of the measured history. `units` may repeat, for the
//! bootstrap; each repeat is a separate unit.
RegimeHazards RegimeCurves(const LongPanel &panel, const vector<idx_t> &units, idx_t n_periods) {
	const idx_t cols = 2 + panel.n_covariates;
	idx_t n_obs = 0;
	for (auto u : units) {
		n_obs += panel.by_unit[u].size();
	}
	Matrix treat_design(n_obs, cols), drop_design(n_obs, cols);
	vector<double> action(n_obs, 0.0), dropped(n_obs, 0.0);
	vector<idx_t> all_rows(n_obs), drop_rows;
	std::iota(all_rows.begin(), all_rows.end(), 0);
	RegimeHazards out;
	idx_t at = 0;
	for (auto u : units) {
		const auto &rows = panel.by_unit[u];
		double prior = 0.0;
		for (idx_t k = 0; k < rows.size(); k++) {
			const auto &row = rows[k];
			treat_design.At(at, 0) = static_cast<double>(row.period);
			treat_design.At(at, 1) = prior;
			drop_design.At(at, 0) = static_cast<double>(row.period);
			drop_design.At(at, 1) = row.treatment;
			for (idx_t c = 0; c < panel.n_covariates; c++) {
				treat_design.At(at, 2 + c) = row.covariates[c];
				drop_design.At(at, 2 + c) = row.covariates[c];
			}
			action[at] = row.treatment;
			// Dropout follows a period survived without the event, and cannot
			// follow the last period, where everyone still in is censored by the
			// end of the study rather than by anything a model should explain.
			const bool event = row.outcome >= 0.5;
			if (!event && row.period + 1 < n_periods) {
				drop_rows.push_back(at);
				if (k + 1 == rows.size()) {
					dropped[at] = 1.0;
					out.dropouts++;
				}
			}
			prior = row.treatment;
			at++;
		}
	}
	const double lambda = 1e-6 * static_cast<double>(n_obs) + 1e-8;
	auto treat_model = FitLogistic(treat_design, action, all_rows, {}, lambda, 60);
	LinearModel drop_model;
	if (out.dropouts > 0) {
		drop_model = FitLogistic(drop_design, dropped, drop_rows, {}, lambda, 60);
	}

	vector<double> risk[2], events[2];
	for (int r = 0; r < 2; r++) {
		risk[r].assign(n_periods, 0.0);
		events[r].assign(n_periods, 0.0);
		out.followers[r].assign(n_periods, 0.0);
	}
	at = 0;
	for (auto u : units) {
		const auto &rows = panel.by_unit[u];
		for (int r = 0; r < 2; r++) {
			double treat_weight = 1.0, drop_weight = 1.0;
			for (idx_t k = 0; k < rows.size(); k++) {
				const auto &row = rows[k];
				const idx_t i = at + k;
				if ((row.treatment >= 0.5 ? 1 : 0) != r) {
					break; // left the regime: censored here, and the weights stand in for it
				}
				const double p_treat = treat_model.Predict(treat_design.Row(i), cols);
				treat_weight /= std::max(r == 1 ? p_treat : 1.0 - p_treat, 1e-6);
				const double w = treat_weight * drop_weight;
				risk[r][row.period] += w;
				out.followers[r][row.period] += 1.0;
				out.max_weight = std::max(out.max_weight, w);
				if (row.outcome >= 0.5) {
					events[r][row.period] += w;
					break;
				}
				if (out.dropouts > 0 && row.period + 1 < n_periods) {
					const double p_drop = drop_model.Predict(drop_design.Row(i), cols);
					drop_weight /= std::max(1.0 - p_drop, 1e-6);
				}
			}
		}
		at += rows.size();
	}
	for (int r = 0; r < 2; r++) {
		out.hazard[r].assign(n_periods, 0.0);
		out.covered[r].assign(n_periods, 0);
		for (idx_t t = 0; t < n_periods; t++) {
			if (risk[r][t] > 0.0) {
				out.hazard[r][t] = events[r][t] / risk[r][t];
				out.covered[r][t] = 1;
			}
		}
	}
	return out;
}

//! Expected event-free periods among the first `horizon`, and survival through
//! the last of them, from per-period hazards: sum over t of S(t).
void RegimeRmst(const vector<double> &hazard, idx_t horizon, double &rmst, double &survival) {
	survival = 1.0;
	rmst = 0.0;
	for (idx_t t = 0; t < horizon; t++) {
		survival *= 1.0 - hazard[t];
		rmst += survival;
	}
}

unique_ptr<FunctionData> BindMsmRmst(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, vector<string> &names) {
	const char *fn = "do_msm_rmst";
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("duckdo: do_msm_rmst needs a table name or query as its first argument");
	}
	const auto &named = input.named_parameters;
	const string relation = input.inputs[0].ToString();
	const string unit_col = RequireParam(named, "unit", fn);
	const string period_col = RequireParam(named, "period", fn);
	const string treatment_col = RequireParam(named, "treatment", fn);
	const string event_col = RequireParam(named, "event", fn);
	const vector<string> covariates = ListParam(named, "covariates");
	const idx_t reps = CountParam(named, "bootstrap_reps", GetSettingIdx(context, "duckdo_bootstrap_reps", 200));
	const uint64_t seed = CountParam(named, "seed", GetSettingIdx(context, "duckdo_seed", 42));

	SetNumericThreads(GetSettingIdx(context, "duckdo_threads", 0));
	auto panel = LoadLong(context, relation, unit_col, period_col, treatment_col, event_col, covariates, fn);
	const idx_t n_periods = panel.period_labels.size();
	const idx_t n_units = panel.units.size();

	// Time is counted from the panel's first period, so every unit has to be
	// followed from it, one row per period, and stop at its event.
	idx_t n_events = 0;
	for (idx_t u = 0; u < n_units; u++) {
		const auto &rows = panel.by_unit[u];
		for (idx_t k = 0; k < rows.size(); k++) {
			const auto &row = rows[k];
			if (row.period != k) {
				throw BinderException("duckdo: unit '%s' has no row for period '%s'. do_msm_rmst counts time from the "
				                      "first period, so every unit needs one row per period from the first until its "
				                      "event or its last observation",
				                      panel.units[u], panel.period_labels[k]);
			}
			if (!row.outcome_observed || (row.outcome != 0.0 && row.outcome != 1.0)) {
				throw BinderException("duckdo: event must be 0 or 1 in every row; unit '%s' has %s at period '%s'",
				                      panel.units[u],
				                      row.outcome_observed ? StringUtil::Format("%g", row.outcome) : string("NULL"),
				                      panel.period_labels[row.period]);
			}
			if (row.outcome == 1.0) {
				n_events++;
				if (k + 1 != rows.size()) {
					throw BinderException("duckdo: unit '%s' has rows after its event at period '%s'; a unit leaves "
					                      "the risk set when the event happens",
					                      panel.units[u], panel.period_labels[row.period]);
				}
			}
		}
	}
	if (n_events == 0) {
		throw BinderException(
		    "duckdo: no unit has an event, so every curve stays at 1 and there is nothing to compare");
	}

	// Units in label order, so the result and every bootstrap draw depend on
	// the data rather than on the order the rows were read in.
	vector<idx_t> ordered(n_units);
	std::iota(ordered.begin(), ordered.end(), 0);
	std::sort(ordered.begin(), ordered.end(), [&](idx_t a, idx_t b) { return panel.units[a] < panel.units[b]; });

	auto point = RegimeCurves(panel, ordered, n_periods);

	// The horizon defaults to the last period in which both regimes still had a
	// real number of units following them and at risk. Following a regime is
	// rarer every period, and past that point the curves rest on a handful.
	const idx_t floor_units = std::max<idx_t>(10, n_units / 20);
	idx_t supported = 0;
	while (supported < n_periods && point.followers[0][supported] >= static_cast<double>(floor_units) &&
	       point.followers[1][supported] >= static_cast<double>(floor_units)) {
		supported++;
	}
	vector<string> warnings;
	idx_t horizon = supported;
	auto horizon_entry = named.find("horizon");
	const bool horizon_given = horizon_entry != named.end() && !horizon_entry->second.IsNull();
	if (horizon_given) {
		const int64_t h = horizon_entry->second.GetValue<int64_t>();
		if (h < 1 || h > static_cast<int64_t>(n_periods)) {
			throw BinderException("duckdo: horizon := counts periods from the first, so it must be between 1 and %llu; "
			                      "got %lld",
			                      static_cast<unsigned long long>(n_periods), static_cast<long long>(h));
		}
		horizon = static_cast<idx_t>(h);
		for (idx_t t = 0; t < horizon; t++) {
			if (!point.covered[0][t] || !point.covered[1][t]) {
				throw BinderException("duckdo: by period '%s' no unit is still following the %s regime, so its curve "
				                      "stops there; choose a horizon before it",
				                      panel.period_labels[t], point.covered[1][t] ? "never-treated" : "always-treated");
			}
		}
		if (horizon > supported) {
			warnings.push_back(StringUtil::Format(
			    "horizon := %llu runs past period %llu, the last in which both regimes still had %llu units following "
			    "them, so the far end of each curve rests on very few people",
			    static_cast<unsigned long long>(horizon), static_cast<unsigned long long>(supported),
			    static_cast<unsigned long long>(floor_units)));
		}
	} else if (horizon == 0) {
		throw BinderException("duckdo: fewer than %llu units follow each regime even in the first period, so neither "
		                      "curve can be estimated. Treated in every period and treated in none both need units who "
		                      "actually did that",
		                      static_cast<unsigned long long>(floor_units));
	}

	double rmst_always = 0.0, rmst_never = 0.0, survival_always = 1.0, survival_never = 1.0;
	RegimeRmst(point.hazard[1], horizon, rmst_always, survival_always);
	RegimeRmst(point.hazard[0], horizon, rmst_never, survival_never);
	const double estimate = rmst_always - rmst_never;

	// Resample whole units and redo everything, both weight models included, so
	// the interval accounts for the weights being estimated. Per-replicate
	// seeding keeps the draw independent of which thread runs it.
	vector<double> per_rep(reps, 0.0);
	vector<uint8_t> rep_usable(reps, 0);
	ParallelJobs(reps, [&](idx_t rep) {
		std::mt19937_64 rng(seed ^ 0xC10E5EEDULL ^ (rep * 0x9E3779B97F4A7C15ULL));
		std::uniform_int_distribution<idx_t> pick(0, n_units - 1);
		vector<idx_t> draw(n_units);
		for (auto &u : draw) {
			u = ordered[pick(rng)];
		}
		auto curves = RegimeCurves(panel, draw, n_periods);
		for (idx_t t = 0; t < horizon; t++) {
			if (!curves.covered[0][t] || !curves.covered[1][t]) {
				return;
			}
		}
		double a = 0.0, b = 0.0, sa = 0.0, sb = 0.0;
		RegimeRmst(curves.hazard[1], horizon, a, sa);
		RegimeRmst(curves.hazard[0], horizon, b, sb);
		per_rep[rep] = a - b;
		rep_usable[rep] = 1;
	});
	vector<double> draws;
	for (idx_t rep = 0; rep < reps; rep++) {
		if (rep_usable[rep]) {
			draws.push_back(per_rep[rep]);
		}
	}
	double se = 0.0;
	if (draws.size() >= 20) {
		double variance = 0.0;
		for (auto d : draws) {
			variance += (d - estimate) * (d - estimate);
		}
		se = std::sqrt(variance / static_cast<double>(draws.size() - 1));
	} else if (reps == 0) {
		warnings.push_back("bootstrap_reps := 0, so no interval is reported");
	} else {
		warnings.push_back("fewer than 20 bootstrap draws were usable, so no interval is reported");
	}

	if (covariates.empty()) {
		warnings.push_back("no covariates := given, so the weights condition only on treatment history. That is the "
		                   "right model only if nothing time-varying drives treatment or dropout - otherwise this is "
		                   "an unadjusted comparison of the units who happened to stay on each regime");
	}
	if (!horizon_given) {
		warnings.push_back(StringUtil::Format(
		    "horizon defaulted to %llu periods, the last in which both regimes still had at least %llu units "
		    "following them and at risk. The horizon is part of the estimand - a different one is a different "
		    "quantity",
		    static_cast<unsigned long long>(horizon), static_cast<unsigned long long>(floor_units)));
	}
	if (point.max_weight > 20.0) {
		warnings.push_back(StringUtil::Format(
		    "the largest weight on a person-period is %.1f: a few units who stayed on a regime against the odds "
		    "carry much of its curve. This is what a near-violation of positivity looks like",
		    point.max_weight));
	}
	warnings.push_back("two regimes are compared, treated in every period and treated in none. A unit counts toward "
	                   "a regime only while its treatment matches it, and the weights stand in for the units who left");
	warnings.push_back("the weights assume no unmeasured confounder of treatment and the event at any period, and that "
	                   "dropout depends only on the measured history - on period, current treatment and the "
	                   "covariates, as the dropout model sees it");
	warnings.push_back("this is a difference in expected event-free periods, not a hazard ratio: a hazard ratio "
	                   "compares units still at risk, and treatment changes who is still at risk");

	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}
	names = {"estimand",    "estimator",        "estimate",        "std_error",      "ci_low",  "ci_high",   "horizon",
	         "rmst_always", "rmst_never",       "survival_always", "survival_never", "n_units", "n_periods", "n_events",
	         "n_dropouts",  "followers_always", "followers_never", "max_weight",     "warnings"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::DOUBLE,
	                LogicalType::LIST(LogicalType::VARCHAR)};
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back(
	    {Value("RMST difference, always vs never treated"), Value("iptw-ipcw-regime-kaplan-meier"),
	     Value::DOUBLE(estimate), se > 0.0 ? Value::DOUBLE(se) : Value(LogicalType::DOUBLE),
	     se > 0.0 ? Value::DOUBLE(estimate - Z95 * se) : Value(LogicalType::DOUBLE),
	     se > 0.0 ? Value::DOUBLE(estimate + Z95 * se) : Value(LogicalType::DOUBLE),
	     Value::BIGINT(static_cast<int64_t>(horizon)), Value::DOUBLE(rmst_always), Value::DOUBLE(rmst_never),
	     Value::DOUBLE(survival_always), Value::DOUBLE(survival_never), Value::BIGINT(static_cast<int64_t>(n_units)),
	     Value::BIGINT(static_cast<int64_t>(n_periods)), Value::BIGINT(static_cast<int64_t>(n_events)),
	     Value::BIGINT(static_cast<int64_t>(point.dropouts)),
	     Value::BIGINT(static_cast<int64_t>(point.followers[1][horizon - 1])),
	     Value::BIGINT(static_cast<int64_t>(point.followers[0][horizon - 1])), Value::DOUBLE(point.max_weight),
	     Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

} // namespace

void RegisterLongitudinalFunctions(ExtensionLoader &loader) {
	TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, BindMsm, InitGlobal);
	fn.named_parameters["unit"] = LogicalType::VARCHAR;
	fn.named_parameters["period"] = LogicalType::VARCHAR;
	fn.named_parameters["treatment"] = LogicalType::VARCHAR;
	fn.named_parameters["outcome"] = LogicalType::VARCHAR;
	fn.named_parameters["covariates"] = LogicalType::LIST(LogicalType::VARCHAR);
	fn.named_parameters["baseline"] = LogicalType::LIST(LogicalType::VARCHAR);
	fn.named_parameters["truncate"] = LogicalType::DOUBLE;
	RegisterUnderBothNames(loader, fn, "msm");

	TableFunction regimes("", {LogicalType::VARCHAR}, EmitRows, BindMsmRmst, InitGlobal);
	regimes.named_parameters["unit"] = LogicalType::VARCHAR;
	regimes.named_parameters["period"] = LogicalType::VARCHAR;
	regimes.named_parameters["treatment"] = LogicalType::VARCHAR;
	regimes.named_parameters["event"] = LogicalType::VARCHAR;
	regimes.named_parameters["covariates"] = LogicalType::LIST(LogicalType::VARCHAR);
	regimes.named_parameters["horizon"] = LogicalType::BIGINT;
	regimes.named_parameters["bootstrap_reps"] = LogicalType::BIGINT;
	regimes.named_parameters["seed"] = LogicalType::BIGINT;
	RegisterUnderBothNames(loader, regimes, "msm_rmst");
}

} // namespace duckdo
} // namespace duckdb
