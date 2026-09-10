//===----------------------------------------------------------------------===//
// Phase 10: panel data and difference-in-differences.
//
// The obvious implementation of DiD is a two-way fixed effects regression, and
// it is wrong whenever units adopt treatment at different times: already-treated
// units end up serving as controls for later adopters, and the coefficient
// becomes a weighted average that can carry negative weights. So DuckDo does not
// ship it.
//
// Instead this computes group-time average treatment effects in the
// Callaway-Sant'Anna sense. For each adoption cohort g and each period t >= g:
//
//   ATT(g, t) = E[Y_t - Y_{g-1} | first treated at g]
//             - E[Y_t - Y_{g-1} | comparison]
//
// with never-treated units as the comparison where any exist, and not-yet-
// treated units otherwise. Those are aggregated by cohort size into an overall
// ATT, and by relative period into an event study - which is the part that
// actually earns its keep, because the pre-treatment periods are a direct read
// on whether parallel trends is plausible. Reporting them is the panel version
// of "assumptions are queryable objects".
//
// Standard errors come from a unit-level cluster bootstrap: units are the
// independent draws here, not unit-periods.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"
#include "duckdo/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>

namespace duckdb {
namespace duckdo {

namespace {

constexpr idx_t NEVER_TREATED = static_cast<idx_t>(-1);

struct PanelGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<PanelGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<PanelGlobalState>();
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

//! A balanced-or-not panel, indexed by unit and period.
struct Panel {
	vector<string> units;
	vector<string> period_labels;
	//! n_units x n_periods outcomes, with a parallel observed mask.
	vector<double> y;
	vector<uint8_t> observed;
	//! Period index at which each unit was first treated, or NEVER_TREATED.
	vector<idx_t> first_treated;
	idx_t n_units = 0;
	idx_t n_periods = 0;
	idx_t n_never = 0;
	//! Unit-level covariates, one standardised row per unit, read at each unit's
	//! first observed period so they are pre-treatment by construction. Present
	//! only when covariates := was given, and then every group-time cell is
	//! estimated doubly robustly rather than by a raw difference of mean changes.
	Matrix X;
	bool has_covariates = false;
	vector<string> warnings;

	double Y(idx_t unit, idx_t period) const {
		return y[unit * n_periods + period];
	}
	bool Seen(idx_t unit, idx_t period) const {
		return observed[unit * n_periods + period] != 0;
	}
};

string RequireColumn(const named_parameter_map_t &named, const char *key, const char *fn) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		throw BinderException("duckdo: %s requires %s := '<column>'", fn, key);
	}
	return entry->second.ToString();
}

Panel LoadPanel(ClientContext &context, const string &relation, const string &unit_col, const string &period_col,
                const string &treatment_col, const string &outcome_col, const char *fn,
                const vector<string> &covariates = {}) {
	const string rel = RelationSql(relation);
	const string sql = "SELECT CAST(" + QuoteIdentifier(unit_col) + " AS VARCHAR) AS u, " + "CAST(" +
	                   QuoteIdentifier(period_col) + " AS VARCHAR) AS p, " + "CAST(" + QuoteIdentifier(outcome_col) +
	                   " AS DOUBLE) AS y, " + "coalesce(CAST(" + QuoteIdentifier(treatment_col) +
	                   " AS BOOLEAN), false) AS d FROM " + rel + " WHERE " + QuoteIdentifier(unit_col) +
	                   " IS NOT NULL AND " + QuoteIdentifier(period_col) + " IS NOT NULL AND " +
	                   QuoteIdentifier(outcome_col) + " IS NOT NULL ORDER BY p, u";
	auto result = RunQuery(context, sql, string(fn) + " reading the panel from " + relation);

	// Periods are ordered by their natural type, which the projection has cast to
	// text; re-derive the order from a numeric read where possible so that 2 does
	// not sort after 10.
	auto order_probe =
	    RunQuery(context,
	             "SELECT DISTINCT CAST(" + QuoteIdentifier(period_col) + " AS VARCHAR) AS p FROM " + rel + " WHERE " +
	                 QuoteIdentifier(period_col) + " IS NOT NULL ORDER BY " + QuoteIdentifier(period_col),
	             string(fn) + " ordering the periods of " + relation);

	Panel panel;
	std::map<string, idx_t> period_index;
	for (idx_t r = 0; r < order_probe->RowCount(); r++) {
		const string label = order_probe->GetValue(0, r).ToString();
		period_index[label] = panel.period_labels.size();
		panel.period_labels.push_back(label);
	}
	panel.n_periods = panel.period_labels.size();
	if (panel.n_periods < 2) {
		throw BinderException("duckdo: %s needs at least two periods in '%s'; found %llu", fn, period_col,
		                      static_cast<unsigned long long>(panel.n_periods));
	}

	std::map<string, idx_t> unit_index;
	vector<idx_t> row_unit(result->RowCount(), 0);
	for (idx_t r = 0; r < result->RowCount(); r++) {
		const string unit = result->GetValue(0, r).ToString();
		auto found = unit_index.find(unit);
		if (found == unit_index.end()) {
			unit_index[unit] = panel.units.size();
			row_unit[r] = panel.units.size();
			panel.units.push_back(unit);
		} else {
			row_unit[r] = found->second;
		}
	}
	panel.n_units = panel.units.size();
	if (panel.n_units < 4) {
		throw BinderException("duckdo: %s needs at least four units in '%s'; found %llu", fn, unit_col,
		                      static_cast<unsigned long long>(panel.n_units));
	}

	panel.y.assign(panel.n_units * panel.n_periods, 0.0);
	panel.observed.assign(panel.n_units * panel.n_periods, 0);
	panel.first_treated.assign(panel.n_units, NEVER_TREATED);

	for (idx_t r = 0; r < result->RowCount(); r++) {
		const idx_t u = row_unit[r];
		auto found = period_index.find(result->GetValue(1, r).ToString());
		if (found == period_index.end()) {
			continue;
		}
		const idx_t p = found->second;
		panel.y[u * panel.n_periods + p] = result->GetValue(2, r).GetValue<double>();
		panel.observed[u * panel.n_periods + p] = 1;
		const Value treated = result->GetValue(3, r);
		if (!treated.IsNull() && treated.GetValue<bool>()) {
			panel.first_treated[u] = std::min(panel.first_treated[u], p);
		}
	}

	for (idx_t u = 0; u < panel.n_units; u++) {
		if (panel.first_treated[u] == NEVER_TREATED) {
			panel.n_never++;
		}
	}
	if (panel.n_never == panel.n_units) {
		throw BinderException("duckdo: no unit in '%s' is ever treated, so there is no effect to estimate", relation);
	}

	if (!covariates.empty()) {
		for (auto &c : covariates) {
			if (StringUtil::CIEquals(c, unit_col) || StringUtil::CIEquals(c, period_col) ||
			    StringUtil::CIEquals(c, treatment_col) || StringUtil::CIEquals(c, outcome_col)) {
				throw BinderException("duckdo: '%s' cannot be a covariate of %s: it is the unit, period, treatment or "
				                      "outcome column",
				                      c, fn);
			}
		}
		string select, casts;
		for (auto &c : covariates) {
			select += ", " + QuoteIdentifier(c);
			casts += ", CAST(" + QuoteIdentifier(c) + " AS DOUBLE)";
		}
		auto typed = RunQuery(context, "SELECT " + select.substr(2) + " FROM " + rel + " LIMIT 0",
		                      string(fn) + " inspecting its covariates");
		for (idx_t j = 0; j < covariates.size(); j++) {
			const auto &type = typed->types[j];
			if (!type.IsNumeric() && type.id() != LogicalTypeId::BOOLEAN) {
				throw BinderException("duckdo: covariate '%s' has type %s; %s takes numeric covariates only. Encode it "
				                      "first, e.g. one indicator column per level",
				                      covariates[j], type.ToString(), fn);
			}
		}
		// Each unit's values at the first period it is observed in: pre-treatment
		// for every cohort, so the treatment cannot have moved them.
		const string u_q = QuoteIdentifier(unit_col), p_q = QuoteIdentifier(period_col);
		auto baseline = RunQuery(context,
		                         "SELECT CAST(" + u_q + " AS VARCHAR)" + casts + " FROM " + rel + " WHERE " + u_q +
		                             " IS NOT NULL AND " + p_q + " IS NOT NULL QUALIFY row_number() OVER (PARTITION BY " +
		                             u_q + " ORDER BY " + p_q + ") = 1",
		                         string(fn) + " reading baseline covariates");
		const idx_t p = covariates.size();
		panel.X.Resize(panel.n_units, p);
		vector<uint8_t> present(panel.n_units * p, 0);
		for (idx_t r = 0; r < baseline->RowCount(); r++) {
			auto found = unit_index.find(baseline->GetValue(0, r).ToString());
			if (found == unit_index.end()) {
				continue;
			}
			for (idx_t j = 0; j < p; j++) {
				const Value v = baseline->GetValue(1 + j, r);
				if (!v.IsNull()) {
					panel.X.At(found->second, j) = v.GetValue<double>();
					present[found->second * p + j] = 1;
				}
			}
		}
		idx_t imputed = 0;
		for (idx_t j = 0; j < p; j++) {
			double sum = 0.0, count = 0.0;
			for (idx_t u = 0; u < panel.n_units; u++) {
				if (present[u * p + j]) {
					sum += panel.X.At(u, j);
					count += 1.0;
				}
			}
			const double mean = count > 0.0 ? sum / count : 0.0;
			double squares = 0.0;
			for (idx_t u = 0; u < panel.n_units; u++) {
				if (!present[u * p + j]) {
					panel.X.At(u, j) = mean;
					imputed++;
				}
				const double d = panel.X.At(u, j) - mean;
				squares += d * d;
			}
			const double sd =
			    std::max(std::sqrt(squares / std::max(1.0, static_cast<double>(panel.n_units) - 1.0)), 1e-12);
			for (idx_t u = 0; u < panel.n_units; u++) {
				panel.X.At(u, j) = (panel.X.At(u, j) - mean) / sd;
			}
		}
		if (imputed > 0) {
			panel.warnings.push_back(StringUtil::Format(
			    "%llu baseline covariate values were NULL and were replaced by the column mean",
			    static_cast<unsigned long long>(imputed)));
		}
		panel.has_covariates = true;
	}
	return panel;
}

//! One group-time cell.
struct GroupTime {
	idx_t cohort = 0;
	idx_t period = 0;
	double att = 0.0;
	idx_t n_treated = 0;
	idx_t n_control = 0;
	bool valid = false;
};

//! One group-time comparison. Without covariates it is the Callaway-Sant'Anna
//! cell - the treated cohort's mean change minus the comparison group's - with
//! the arithmetic unchanged line for line, so an existing do_did returns the
//! same bits it always did.
//!
//! With covariates it is Sant'Anna and Zhao's doubly robust panel estimator:
//!
//!   ATT(g,t) = E[ (w1 - w0) (dY - m0(X)) ]
//!   w1 = D / E[D],   w0 = [p(X)(1-D)/(1-p(X))] / E[p(X)(1-D)/(1-p(X))]
//!
//! m0 regresses the change on X in the comparison group and p models being in
//! the cohort rather than the comparison. It is consistent if either is right.
//! Parametric, as in the paper, not cross-fitted: the interval is the cluster
//! bootstrap, which refits both models on every draw.
struct CellResult {
	double att = 0.0;
	idx_t n_treated = 0;
	idx_t n_control = 0;
	bool valid = false;
};

CellResult CellEffect(const Panel &panel, const vector<idx_t> &treated, const vector<idx_t> &comparison, idx_t t,
                      idx_t base) {
	CellResult cell;
	if (!panel.has_covariates) {
		double treated_change = 0.0;
		idx_t treated_count = 0;
		for (auto u : treated) {
			if (panel.Seen(u, t) && panel.Seen(u, base)) {
				treated_change += panel.Y(u, t) - panel.Y(u, base);
				treated_count++;
			}
		}
		double control_change = 0.0;
		idx_t control_count = 0;
		for (auto u : comparison) {
			if (panel.Seen(u, t) && panel.Seen(u, base)) {
				control_change += panel.Y(u, t) - panel.Y(u, base);
				control_count++;
			}
		}
		if (treated_count == 0 || control_count == 0) {
			return cell;
		}
		cell.att = treated_change / static_cast<double>(treated_count) -
		           control_change / static_cast<double>(control_count);
		cell.n_treated = treated_count;
		cell.n_control = control_count;
		cell.valid = true;
		return cell;
	}

	// A bootstrap draw can repeat a unit; each repeat is another row below, which
	// is exactly the weighting a resample means. A unit is never in both lists:
	// the comparison is never-treated or not yet treated as of t.
	vector<double> change(panel.n_units, 0.0);
	vector<double> in_cohort(panel.n_units, 0.0);
	vector<idx_t> treated_rows, control_rows, all_rows;
	for (auto u : treated) {
		if (panel.Seen(u, t) && panel.Seen(u, base)) {
			change[u] = panel.Y(u, t) - panel.Y(u, base);
			in_cohort[u] = 1.0;
			treated_rows.push_back(u);
			all_rows.push_back(u);
		}
	}
	for (auto u : comparison) {
		if (panel.Seen(u, t) && panel.Seen(u, base)) {
			change[u] = panel.Y(u, t) - panel.Y(u, base);
			control_rows.push_back(u);
			all_rows.push_back(u);
		}
	}
	// Both working models need something to fit. A cohort of one, or a
	// comparison group smaller than the covariate count, is left out rather than
	// guessed at.
	const idx_t cols = panel.X.cols;
	if (treated_rows.size() < 2 || control_rows.size() < cols + 2) {
		return cell;
	}
	const double lambda = 1e-6 * static_cast<double>(all_rows.size()) + 1e-8;
	auto outcome = FitRidge(panel.X, change, control_rows, {}, lambda);
	auto propensity = FitLogistic(panel.X, in_cohort, all_rows, {}, std::max(lambda, 1.0), 30);

	double treated_term = 0.0;
	for (auto u : treated_rows) {
		treated_term += change[u] - outcome.Predict(panel.X.Row(u), cols);
	}
	double control_term = 0.0, control_weight = 0.0;
	for (auto u : control_rows) {
		const double p = std::min(std::max(propensity.Predict(panel.X.Row(u), cols), 1e-3), 1.0 - 1e-3);
		const double w = p / (1.0 - p);
		control_term += w * (change[u] - outcome.Predict(panel.X.Row(u), cols));
		control_weight += w;
	}
	cell.att = treated_term / static_cast<double>(treated_rows.size()) - control_term / control_weight;
	cell.n_treated = treated_rows.size();
	cell.n_control = control_rows.size();
	cell.valid = true;
	return cell;
}

//! Compute every ATT(g, t) over a (possibly resampled) set of units.
vector<GroupTime> GroupTimeEffects(const Panel &panel, const vector<idx_t> &units) {
	vector<GroupTime> cells;
	// Cohorts are the distinct adoption periods, excluding period 0: a unit
	// treated in the very first period has no pre-period to difference against.
	std::map<idx_t, vector<idx_t>> cohorts;
	vector<idx_t> never;
	for (auto u : units) {
		const idx_t g = panel.first_treated[u];
		if (g == NEVER_TREATED) {
			never.push_back(u);
		} else if (g >= 1) {
			cohorts[g].push_back(u);
		}
	}

	for (auto &entry : cohorts) {
		const idx_t g = entry.first;
		const idx_t base = g - 1;
		for (idx_t t = g; t < panel.n_periods; t++) {
			GroupTime cell;
			cell.cohort = g;
			cell.period = t;

			// Comparison: never-treated where they exist, otherwise units not yet
			// treated as of t. Never using already-treated units as controls is the
			// whole point of doing it this way.
			const vector<idx_t> *comparison = &never;
			vector<idx_t> not_yet;
			if (never.empty()) {
				for (auto u : units) {
					const idx_t gu = panel.first_treated[u];
					if (gu == NEVER_TREATED || gu > t) {
						not_yet.push_back(u);
					}
				}
				comparison = &not_yet;
			}

			const auto effect = CellEffect(panel, entry.second, *comparison, t, base);
			cell.att = effect.att;
			cell.n_treated = effect.n_treated;
			cell.n_control = effect.n_control;
			cell.valid = effect.valid;
			cells.push_back(cell);
		}
	}
	return cells;
}

//! Cohort-size weighted average of the valid cells.
double AggregateAtt(const vector<GroupTime> &cells) {
	double weighted = 0.0, weight = 0.0;
	for (auto &cell : cells) {
		if (!cell.valid) {
			continue;
		}
		weighted += cell.att * static_cast<double>(cell.n_treated);
		weight += static_cast<double>(cell.n_treated);
	}
	return weight > 0.0 ? weighted / weight : 0.0;
}

//! Pre- and post-treatment effects indexed by period relative to adoption.
std::map<int64_t, std::pair<double, idx_t>> EventStudy(const Panel &panel, const vector<idx_t> &units) {
	std::map<int64_t, std::pair<double, idx_t>> by_relative;
	std::map<idx_t, vector<idx_t>> cohorts;
	vector<idx_t> never;
	for (auto u : units) {
		const idx_t g = panel.first_treated[u];
		if (g == NEVER_TREATED) {
			never.push_back(u);
		} else if (g >= 1) {
			cohorts[g].push_back(u);
		}
	}

	for (auto &entry : cohorts) {
		const idx_t g = entry.first;
		const idx_t base = g - 1;
		for (idx_t t = 0; t < panel.n_periods; t++) {
			if (t == base) {
				continue; // the reference period, zero by construction
			}
			const vector<idx_t> *comparison = &never;
			vector<idx_t> not_yet;
			if (never.empty()) {
				for (auto u : units) {
					const idx_t gu = panel.first_treated[u];
					if (gu == NEVER_TREATED || gu > std::max(t, g)) {
						not_yet.push_back(u);
					}
				}
				comparison = &not_yet;
			}
			const auto effect = CellEffect(panel, entry.second, *comparison, t, base);
			if (!effect.valid) {
				continue;
			}
			const int64_t relative = static_cast<int64_t>(t) - static_cast<int64_t>(g);
			auto &slot = by_relative[relative];
			slot.first += effect.att * static_cast<double>(effect.n_treated);
			slot.second += effect.n_treated;
		}
	}
	for (auto &entry : by_relative) {
		if (entry.second.second > 0) {
			entry.second.first /= static_cast<double>(entry.second.second);
		}
	}
	return by_relative;
}

vector<idx_t> AllUnits(const Panel &panel) {
	vector<idx_t> units(panel.n_units);
	for (idx_t u = 0; u < panel.n_units; u++) {
		units[u] = u;
	}
	return units;
}

// --- do_did ------------------------------------------------------------------

unique_ptr<FunctionData> BindDid(ClientContext &context, TableFunctionBindInput &input,
                                 vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("duckdo: do_did takes a table name or a query as its first argument");
	}
	const string relation = input.inputs[0].ToString();
	const string unit = RequireColumn(input.named_parameters, "unit", "do_did");
	const string period = RequireColumn(input.named_parameters, "period", "do_did");
	const string treatment = RequireColumn(input.named_parameters, "treatment", "do_did");
	const string outcome = RequireColumn(input.named_parameters, "outcome", "do_did");

	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	auto panel = LoadPanel(context, relation, unit, period, treatment, outcome, "do_did", spec.covariates);
	auto units = AllUnits(panel);
	auto cells = GroupTimeEffects(panel, units);
	const double att = AggregateAtt(cells);

	idx_t valid_cells = 0, cohorts = 0;
	std::map<idx_t, bool> seen_cohorts;
	for (auto &cell : cells) {
		if (cell.valid) {
			valid_cells++;
			seen_cohorts[cell.cohort] = true;
		}
	}
	cohorts = seen_cohorts.size();
	if (valid_cells == 0) {
		throw BinderException("duckdo: no group-time cell in '%s' had both a treated and a comparison unit "
		                      "observed in the base period. Check that units are observed before they are treated",
		                      relation);
	}

	// Cluster bootstrap over units: the independent draws are units, not
	// unit-periods, and treating them otherwise understates the interval badly.
	const idx_t reps = std::max<idx_t>(spec.bootstrap_reps, 50);
	std::mt19937_64 rng(static_cast<uint64_t>(spec.seed) ^ 0xDEADBEEFULL);
	std::uniform_int_distribution<idx_t> pick(0, panel.n_units - 1);
	vector<double> draws;
	draws.reserve(reps);
	for (idx_t b = 0; b < reps; b++) {
		vector<idx_t> resample(panel.n_units);
		for (auto &u : resample) {
			u = pick(rng);
		}
		auto boot_cells = GroupTimeEffects(panel, resample);
		bool any = false;
		for (auto &cell : boot_cells) {
			if (cell.valid) {
				any = true;
				break;
			}
		}
		if (any) {
			draws.push_back(AggregateAtt(boot_cells));
		}
	}
	double se = 0.0;
	if (draws.size() >= 20) {
		double variance = 0.0;
		for (auto d : draws) {
			const double diff = d - att;
			variance += diff * diff;
		}
		se = std::sqrt(variance / static_cast<double>(draws.size() - 1));
	}

	// Pre-trend check: the average pre-treatment effect should be indistinguishable
	// from zero if parallel trends holds. This is reported, not asserted.
	auto event = EventStudy(panel, units);
	double pre_sum = 0.0;
	idx_t pre_weight = 0;
	for (auto &entry : event) {
		if (entry.first < 0) {
			pre_sum += entry.second.first * static_cast<double>(entry.second.second);
			pre_weight += entry.second.second;
		}
	}
	const double pre_trend = pre_weight > 0 ? pre_sum / static_cast<double>(pre_weight) : 0.0;

	vector<string> warnings;
	if (panel.has_covariates) {
		warnings.push_back("group-time ATT aggregated by cohort size (Callaway-Sant'Anna), each cell doubly robust "
		                   "(Sant'Anna-Zhao): parallel trends is assumed only conditional on the covariates, read at "
		                   "each unit's first observed period");
	} else {
		warnings.push_back("group-time ATT aggregated by cohort size (Callaway-Sant'Anna); two-way fixed effects is "
		                   "not used because it misweights under staggered adoption");
	}
	for (auto &w : panel.warnings) {
		warnings.push_back(w);
	}
	if (panel.n_never == 0) {
		warnings.push_back("no never-treated units; not-yet-treated units were used as the comparison");
	}
	if (se > 0.0 && std::fabs(pre_trend) > 2.0 * se) {
		warnings.push_back(StringUtil::Format(
		    "average pre-treatment effect is %.4f, large relative to the standard error - parallel trends looks "
		    "doubtful. Inspect do_event_study()",
		    pre_trend));
	}

	names = {"estimand", "estimator", "estimate",  "std_error",       "ci_low",    "ci_high",
	         "n_units",  "n_periods", "n_cohorts", "n_never_treated", "pre_trend", "warnings"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::DOUBLE,  LogicalType::LIST(LogicalType::VARCHAR)};

	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back(
	    {Value("ATT"), Value(panel.has_covariates ? "callaway-santanna, doubly robust" : "callaway-santanna"),
	     Value::DOUBLE(att),
	     se > 0.0 ? Value::DOUBLE(se) : Value(LogicalType::DOUBLE),
	     se > 0.0 ? Value::DOUBLE(att - Z95 * se) : Value(LogicalType::DOUBLE),
	     se > 0.0 ? Value::DOUBLE(att + Z95 * se) : Value(LogicalType::DOUBLE),
	     Value::BIGINT(static_cast<int64_t>(panel.n_units)), Value::BIGINT(static_cast<int64_t>(panel.n_periods)),
	     Value::BIGINT(static_cast<int64_t>(cohorts)), Value::BIGINT(static_cast<int64_t>(panel.n_never)),
	     Value::DOUBLE(pre_trend), Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

// --- do_event_study ----------------------------------------------------------

unique_ptr<FunctionData> BindEventStudy(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("duckdo: do_event_study takes a table name or a query as its first argument");
	}
	const string relation = input.inputs[0].ToString();
	const string unit = RequireColumn(input.named_parameters, "unit", "do_event_study");
	const string period = RequireColumn(input.named_parameters, "period", "do_event_study");
	const string treatment = RequireColumn(input.named_parameters, "treatment", "do_event_study");
	const string outcome = RequireColumn(input.named_parameters, "outcome", "do_event_study");

	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	auto panel = LoadPanel(context, relation, unit, period, treatment, outcome, "do_event_study", spec.covariates);
	auto units = AllUnits(panel);
	auto point = EventStudy(panel, units);

	// Bootstrap each relative period together, over the same unit resample, so
	// the intervals are mutually consistent.
	const idx_t reps = std::max<idx_t>(spec.bootstrap_reps, 50);
	std::mt19937_64 rng(static_cast<uint64_t>(spec.seed) ^ 0xFEEDFACEULL);
	std::uniform_int_distribution<idx_t> pick(0, panel.n_units - 1);
	std::map<int64_t, vector<double>> draws;
	for (idx_t b = 0; b < reps; b++) {
		vector<idx_t> resample(panel.n_units);
		for (auto &u : resample) {
			u = pick(rng);
		}
		for (auto &entry : EventStudy(panel, resample)) {
			draws[entry.first].push_back(entry.second.first);
		}
	}

	names = {"relative_period", "att", "std_error", "ci_low", "ci_high", "n_treated", "is_pre_treatment"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BOOLEAN};
	auto bind = make_uniq<ResultBindData>();
	for (auto &entry : point) {
		const int64_t relative = entry.first;
		const double att = entry.second.first;
		double se = 0.0;
		auto found = draws.find(relative);
		if (found != draws.end() && found->second.size() >= 20) {
			double variance = 0.0;
			for (auto d : found->second) {
				const double diff = d - att;
				variance += diff * diff;
			}
			se = std::sqrt(variance / static_cast<double>(found->second.size() - 1));
		}
		bind->rows.push_back({Value::BIGINT(relative), Value::DOUBLE(att),
		                      se > 0.0 ? Value::DOUBLE(se) : Value(LogicalType::DOUBLE),
		                      se > 0.0 ? Value::DOUBLE(att - Z95 * se) : Value(LogicalType::DOUBLE),
		                      se > 0.0 ? Value::DOUBLE(att + Z95 * se) : Value(LogicalType::DOUBLE),
		                      Value::BIGINT(static_cast<int64_t>(entry.second.second)), Value::BOOLEAN(relative < 0)});
	}
	return std::move(bind);
}

} // namespace

void RegisterPanelFunctions(ExtensionLoader &loader) {
	struct Entry {
		const char *name;
		table_function_bind_t bind;
	};
	const Entry entries[] = {{"did", BindDid}, {"event_study", BindEventStudy}};
	for (auto &entry : entries) {
		TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, entry.bind, InitGlobal);
		AddCommonNamedParameters(fn);
		fn.named_parameters["unit"] = LogicalType::VARCHAR;
		// Not "time": TIME is a type name, so `time := ...` is a parser error.
		fn.named_parameters["period"] = LogicalType::VARCHAR;
		RegisterUnderBothNames(loader, fn, entry.name);
	}
}

} // namespace duckdo
} // namespace duckdb
