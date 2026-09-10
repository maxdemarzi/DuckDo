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
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"

#include <algorithm>
#include <cmath>
#include <map>

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
}

} // namespace duckdo
} // namespace duckdb
