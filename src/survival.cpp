//===----------------------------------------------------------------------===//
// Time-to-event outcomes: the difference in restricted mean survival time.
//
// The obvious thing to report for a survival outcome is a hazard ratio, and
// this deliberately does not. A hazard ratio compares two groups among those
// still at risk at each moment, and treatment changes who is still at risk - so
// after the first events the two risk sets are no longer comparable even in a
// perfect randomised trial. The ratio stays estimable and stops being causal.
// It is not a bug that can be fixed by adjustment; it is what the estimand
// means.
//
// Restricted mean survival time has no such problem. RMST(tau) is the average
// event-free time over the first tau units:
//
//   RMST(tau) = integral from 0 to tau of S(t) dt
//
// so the contrast RMST_1(tau) - RMST_0(tau) is "how much longer, on average,
// does a treated subject stay event-free in the first tau" - a difference in
// expectations of an actual quantity, in the units of the clock, which a
// non-statistician can act on.
//
// Confounding is handled by inverse-probability weighting the two Kaplan-Meier
// curves rather than by modelling the hazard, which keeps the estimand marginal.
//
// With competing risks - death from one cause ends the chance of dying from
// another - do_rmtl reports the restricted mean time lost to one cause:
//
//   RMTL_k(tau) = integral from 0 to tau of F_k(t) dt
//
// where F_k is the cumulative incidence of cause k, estimated by a weighted
// Aalen-Johansen curve. The tempting shortcut, treating the other causes as
// censoring and taking 1 - Kaplan-Meier, estimates the incidence in a world where
// those causes had been abolished, which is a different and larger number.
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

struct SurvivalGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<SurvivalGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<SurvivalGlobalState>();
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

//! One subject, reduced to what a survival curve needs.
struct Subject {
	double duration = 0.0;
	double weight = 1.0;
	bool event = false;
	//! The event code: 0 censored, otherwise which cause. Read by do_rmtl only.
	int64_t cause = 0;
};

//! Area under the weighted Kaplan-Meier curve up to `horizon`.
//!
//! S is a right-continuous step function that starts at 1 and drops only at
//! observed event times, so the integral is a sum of rectangles. Censored
//! subjects never drop the curve; they leave the risk set, which is the whole
//! mechanism by which Kaplan-Meier handles censoring.
double WeightedRmst(vector<Subject> subjects, double horizon) {
	std::sort(subjects.begin(), subjects.end(),
	          [](const Subject &a, const Subject &b) { return a.duration < b.duration; });

	double at_risk = 0.0;
	for (auto &s : subjects) {
		at_risk += s.weight;
	}
	if (!(at_risk > 0.0)) {
		return 0.0;
	}

	double survival = 1.0;
	double area = 0.0;
	double previous_time = 0.0;
	idx_t i = 0;
	while (i < subjects.size()) {
		const double t = subjects[i].duration;
		if (t > horizon) {
			break;
		}
		// Every subject sharing this time leaves the risk set together; those
		// with an event also drop the curve.
		double events_here = 0.0;
		double leaving = 0.0;
		idx_t j = i;
		while (j < subjects.size() && subjects[j].duration == t) {
			leaving += subjects[j].weight;
			if (subjects[j].event) {
				events_here += subjects[j].weight;
			}
			j++;
		}
		if (events_here > 0.0) {
			area += survival * (t - previous_time);
			survival *= 1.0 - events_here / at_risk;
			previous_time = t;
		}
		at_risk -= leaving;
		i = j;
		if (!(at_risk > 1e-12)) {
			break;
		}
	}
	// The curve is flat from the last event to the horizon.
	if (horizon > previous_time) {
		area += survival * (horizon - previous_time);
	}
	return area;
}

//! Area under the weighted Aalen-Johansen cumulative incidence of `cause` up to
//! `horizon`: the restricted mean time lost to that cause.
//!
//! At each event time the incidence rises by the share of those still event-free
//! who had this cause, S(t-) * d_cause / at_risk, and every cause lowers S. A
//! subject who had another cause first is not censored: they can no longer have
//! this one. Treating them as censored - 1 - Kaplan-Meier on this cause alone -
//! would estimate the incidence in a world where the other causes had been
//! abolished.
double WeightedRmtl(vector<Subject> subjects, double horizon, int64_t cause, double &incidence_out) {
	std::sort(subjects.begin(), subjects.end(),
	          [](const Subject &a, const Subject &b) { return a.duration < b.duration; });
	incidence_out = 0.0;
	double at_risk = 0.0;
	for (auto &s : subjects) {
		at_risk += s.weight;
	}
	if (!(at_risk > 0.0)) {
		return 0.0;
	}
	double survival = 1.0;
	double incidence = 0.0;
	double area = 0.0;
	double previous_time = 0.0;
	idx_t i = 0;
	while (i < subjects.size()) {
		const double t = subjects[i].duration;
		if (t > horizon) {
			break;
		}
		double events_here = 0.0;
		double cause_here = 0.0;
		double leaving = 0.0;
		idx_t j = i;
		while (j < subjects.size() && subjects[j].duration == t) {
			leaving += subjects[j].weight;
			if (subjects[j].event) {
				events_here += subjects[j].weight;
				if (subjects[j].cause == cause) {
					cause_here += subjects[j].weight;
				}
			}
			j++;
		}
		if (events_here > 0.0) {
			area += incidence * (t - previous_time);
			incidence += survival * cause_here / at_risk;
			survival *= 1.0 - events_here / at_risk;
			previous_time = t;
		}
		at_risk -= leaving;
		i = j;
		if (!(at_risk > 1e-12)) {
			break;
		}
	}
	// The incidence is flat from the last event to the horizon.
	if (horizon > previous_time) {
		area += incidence * (horizon - previous_time);
	}
	incidence_out = incidence;
	return area;
}

//! do_rmst, and with `lost` do_rmtl: one bind, since the horizon, the weights and
//! the bootstrap are the same and only the curve and the output differ.
unique_ptr<FunctionData> BindSurvival(ClientContext &context, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names, bool lost) {
	const char *fn = lost ? "do_rmtl" : "do_rmst";
	auto named = input.named_parameters;
	auto duration_entry = named.find("duration");
	if (duration_entry == named.end() || duration_entry->second.IsNull()) {
		throw BinderException("duckdo: %s requires duration := '<column>', the observed follow-up time", fn);
	}
	auto event_entry = named.find("event");
	if (event_entry == named.end() || event_entry->second.IsNull()) {
		if (lost) {
			throw BinderException("duckdo: do_rmtl requires event := '<column>', 0 where the subject was censored and "
			                      "the cause's code where an event was observed");
		}
		throw BinderException("duckdo: do_rmst requires event := '<column>', 1 where the event was observed and 0 "
		                      "where the subject was censored");
	}
	int64_t cause = 0;
	if (lost) {
		auto cause_entry = named.find("cause");
		if (cause_entry == named.end() || cause_entry->second.IsNull()) {
			throw BinderException("duckdo: do_rmtl requires cause := <code>, the event code whose time lost to report; "
			                      "every other nonzero code is a competing risk");
		}
		cause = cause_entry->second.GetValue<int64_t>();
		if (cause < 1) {
			throw BinderException("duckdo: cause := must be a positive event code, since 0 means censored; got %lld",
			                      static_cast<long long>(cause));
		}
	}
	// The frame already knows how to load an outcome and one auxiliary numeric
	// column, and it excludes both from the default covariate list. Duration is
	// the outcome; the event indicator rides along as the auxiliary.
	named["outcome"] = duration_entry->second;
	named["mediator"] = event_entry->second;
	auto spec = CausalSpec::Parse(context, input.inputs, named);
	auto frame = BuildFrame(context, spec);
	if (!frame.has_aux) {
		throw BinderException("duckdo: the event column was not loaded");
	}

	vector<string> warnings = frame.warnings;

	double max_duration = 0.0;
	double arm_max[2] = {0.0, 0.0};
	idx_t events[2] = {0, 0};
	idx_t cause_events[2] = {0, 0};
	bool codes_above_one = false;
	idx_t arm_n[2] = {0, 0};
	vector<double> arm_times[2];
	for (idx_t i = 0; i < frame.n; i++) {
		if (!(frame.y[i] >= 0.0) || !std::isfinite(frame.y[i])) {
			throw BinderException("duckdo: duration '%s' must be a non-negative finite time; found %g",
			                      duration_entry->second.ToString(), frame.y[i]);
		}
		const int arm = frame.t[i] >= 0.5 ? 1 : 0;
		arm_n[arm]++;
		arm_max[arm] = std::max(arm_max[arm], frame.y[i]);
		arm_times[arm].push_back(frame.y[i]);
		max_duration = std::max(max_duration, frame.y[i]);
		if (frame.aux[i] >= 0.5) {
			events[arm]++;
		}
		const double code = frame.aux[i];
		codes_above_one = codes_above_one || code >= 1.5;
		if (lost) {
			if (!(code >= 0.0) || std::fabs(code - std::round(code)) > 1e-9) {
				throw BinderException("duckdo: event codes must be non-negative integers, 0 for censored and one code "
				                      "per cause; found %g",
				                      code);
			}
			if (std::llround(code) == cause) {
				cause_events[arm]++;
			}
		}
	}
	if (lost && (cause_events[0] == 0 || cause_events[1] == 0)) {
		throw BinderException("duckdo: arm '%s' has no events of cause %lld, so its cumulative incidence never rises "
		                      "and no time lost is comparable",
		                      cause_events[0] == 0 ? frame.control_label : frame.treated_label,
		                      static_cast<long long>(cause));
	}
	if (!lost && (events[0] == 0 || events[1] == 0)) {
		throw BinderException("duckdo: arm '%s' has no observed events, so its survival curve never falls and no "
		                      "restricted mean is comparable",
		                      events[0] == 0 ? frame.control_label : frame.treated_label);
	}

	// The horizon defaults to the last time BOTH arms still had a real risk set.
	//
	// The obvious default - the smaller of the two arms' last observation - is a
	// trap: on 20,000 subjects it lands at 33.3 where a handful of people remain,
	// and the tail of a Kaplan-Meier curve estimated from a handful of people is
	// almost pure noise that the restricted mean then integrates over. Requiring
	// a floor on the risk set gives a horizon the data can actually support, and
	// it is reported so it is never invisible.
	const idx_t floor_at[2] = {std::max<idx_t>(10, arm_n[0] / 20), std::max<idx_t>(10, arm_n[1] / 20)};
	double supported[2] = {0.0, 0.0};
	for (int arm = 0; arm < 2; arm++) {
		auto &times = arm_times[arm];
		std::sort(times.begin(), times.end());
		// With k subjects required at risk, the last supported time is the
		// (n-k)th smallest duration.
		supported[arm] = times.size() > floor_at[arm] ? times[times.size() - floor_at[arm] - 1] : times.front();
	}
	double horizon = std::min(supported[0], supported[1]);
	const double naive_horizon = std::min(arm_max[0], arm_max[1]);
	auto horizon_entry = named.find("horizon");
	bool horizon_given = false;
	if (horizon_entry != named.end() && !horizon_entry->second.IsNull()) {
		horizon = horizon_entry->second.GetValue<double>();
		horizon_given = true;
		if (!(horizon > 0.0)) {
			throw BinderException("duckdo: horizon := must be positive; got %g", horizon);
		}
		if (horizon > naive_horizon) {
			warnings.push_back("horizon := " + StringUtil::Format("%g", horizon) +
			                   " is beyond the last observation in one arm (" +
			                   StringUtil::Format("%g", naive_horizon) +
			                   "), so the curve is extrapolated flat past that point rather than estimated");
		} else if (horizon > std::min(supported[0], supported[1])) {
			warnings.push_back("horizon := " + StringUtil::Format("%g", horizon) +
			                   " runs past the point where an arm still had " +
			                   std::to_string(std::max(floor_at[0], floor_at[1])) +
			                   " subjects at risk, so the far end of the curve rests on very few people");
		}
	}

	// Propensity, then stabilised weights. Stabilising matters less here than it
	// does for a marginal structural model, but it keeps the weighted risk sets
	// close to the real sample sizes, which is what the curve's variance depends
	// on.
	vector<idx_t> all_rows(frame.n);
	for (idx_t i = 0; i < frame.n; i++) {
		all_rows[i] = i;
	}
	const double lambda = 1e-6 * static_cast<double>(frame.n) + 1e-8;
	const double marginal = static_cast<double>(arm_n[1]) / static_cast<double>(frame.n);

	auto build_weights = [&](const vector<idx_t> &rows, vector<double> &weight_out, double &trimmed_out) {
		auto propensity = FitLogistic(frame.X, frame.t, rows, {}, lambda, 60);
		weight_out.assign(rows.size(), 1.0);
		trimmed_out = 0.0;
		for (idx_t k = 0; k < rows.size(); k++) {
			const idx_t i = rows[k];
			double e = propensity.Predict(frame.X.Row(i), frame.X.cols);
			const double lo = spec.trim, hi = 1.0 - spec.trim;
			if (e < lo || e > hi) {
				trimmed_out += 1.0;
				e = std::min(std::max(e, lo), hi);
			}
			weight_out[k] = frame.t[i] >= 0.5 ? marginal / e : (1.0 - marginal) / (1.0 - e);
		}
	};

	auto rmst_from = [&](const vector<idx_t> &rows, const vector<double> &weights, double h, double &treated,
	                     double &control, double &incidence_treated, double &incidence_control) {
		vector<Subject> arm1, arm0;
		arm1.reserve(rows.size());
		arm0.reserve(rows.size());
		for (idx_t k = 0; k < rows.size(); k++) {
			const idx_t i = rows[k];
			Subject s;
			s.duration = frame.y[i];
			s.event = frame.aux[i] >= 0.5;
			s.cause = std::llround(frame.aux[i]);
			s.weight = weights[k];
			if (frame.t[i] >= 0.5) {
				arm1.push_back(s);
			} else {
				arm0.push_back(s);
			}
		}
		if (lost) {
			treated = WeightedRmtl(std::move(arm1), h, cause, incidence_treated);
			control = WeightedRmtl(std::move(arm0), h, cause, incidence_control);
		} else {
			treated = WeightedRmst(std::move(arm1), h);
			control = WeightedRmst(std::move(arm0), h);
		}
	};

	vector<double> weights;
	double trimmed = 0.0;
	build_weights(all_rows, weights, trimmed);
	double rmst_treated = 0.0, rmst_control = 0.0, incidence_treated = 0.0, incidence_control = 0.0;
	rmst_from(all_rows, weights, horizon, rmst_treated, rmst_control, incidence_treated, incidence_control);
	const double estimate = rmst_treated - rmst_control;

	// The weighted Kaplan-Meier estimator has a closed-form variance only under
	// assumptions the weights break, so the interval comes from resampling
	// subjects and redoing everything including the propensity fit. Anything
	// cheaper would treat the weights as known.
	// Per-replicate seeding, so the answer does not depend on which thread got
	// which replicate. `usable` marks the ones that drew both arms.
	const idx_t reps = std::max<idx_t>(spec.bootstrap_reps, 100);
	vector<double> per_rep(reps, 0.0);
	vector<uint8_t> rep_usable(reps, 0);
	ParallelJobs(reps, [&](idx_t rep) {
		std::mt19937_64 rng(static_cast<uint64_t>(spec.seed) ^ 0x5A1E5A1EULL ^ (rep * 0x9E3779B97F4A7C15ULL));
		std::uniform_int_distribution<idx_t> pick(0, frame.n - 1);
		vector<idx_t> resample;
		if (frame.has_cluster) {
			// Whole clusters: rows of one unit are not independent draws.
			resample = ResampleClusters(frame, rng);
		} else {
			resample.resize(frame.n);
			for (idx_t k = 0; k < frame.n; k++) {
				resample[k] = pick(rng);
			}
		}
		idx_t seen[2] = {0, 0};
		for (auto i : resample) {
			seen[frame.t[i] >= 0.5 ? 1 : 0]++;
		}
		if (seen[0] <= 1 || seen[1] <= 1) {
			return;
		}
		vector<double> boot_weights;
		double boot_trimmed = 0.0;
		build_weights(resample, boot_weights, boot_trimmed);
		double t1 = 0.0, t0 = 0.0, i1 = 0.0, i0 = 0.0;
		rmst_from(resample, boot_weights, horizon, t1, t0, i1, i0);
		per_rep[rep] = t1 - t0;
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
			const double diff = d - estimate;
			variance += diff * diff;
		}
		se = std::sqrt(variance / static_cast<double>(draws.size() - 1));
	}

	const idx_t total_events = events[0] + events[1];
	const double censored_fraction = 1.0 - static_cast<double>(total_events) / static_cast<double>(frame.n);

	if (!horizon_given) {
		warnings.push_back("horizon defaulted to " + StringUtil::Format("%g", horizon) +
		                   ", the last time both arms still had at least 5% of their subjects at risk. The last "
		                   "observation in the shorter arm is " +
		                   StringUtil::Format("%g", naive_horizon) +
		                   ", but the curve out there rests on a handful of people. " +
		                   string(lost ? "The time lost" : "RMST") +
		                   " is part of the estimand, not a display detail - a different horizon is a different "
		                   "quantity");
	}
	if (censored_fraction > 0.5) {
		warnings.push_back(StringUtil::Format("%.0f%%", 100.0 * censored_fraction) +
		                   " of subjects were censored before their event, so most of the curve rests on the "
		                   "assumption that censoring is unrelated to the outcome");
	}
	if (trimmed > 0.0) {
		warnings.push_back(std::to_string(static_cast<idx_t>(trimmed)) + " propensity scores were trimmed to [" +
		                   StringUtil::Format("%g", spec.trim) + ", " + StringUtil::Format("%g", 1.0 - spec.trim) +
		                   "]; those subjects have covariates that "
		                   "almost determine their treatment");
	}
	warnings.push_back("censoring is assumed independent of the event time given the covariates. Nothing in the "
	                   "data can check that, and it fails exactly when subjects leave because they are getting "
	                   "worse");
	if (lost) {
		warnings.push_back(StringUtil::Format(
		    "this is the restricted mean time lost to cause %lld: how much of the first %g units a subject spends, on "
		    "average, after having had it. Subjects who had another cause first are not censored - they can no longer "
		    "have this one - and censoring them instead would estimate a world where the other causes had been "
		    "abolished",
		    static_cast<long long>(cause), horizon));
		warnings.push_back("a treatment can cut the time lost to one cause by leaving people exposed to another; "
		                   "read it beside the other causes' time lost, or beside do_rmst for all of them together");
		warnings.push_back("this is a difference in time lost, not a cause-specific or subdistribution hazard ratio, "
		                   "for the reason do_rmst reports no hazard ratio: those compare subjects still at risk, and "
		                   "treatment changes who is");
	} else {
		if (codes_above_one) {
			warnings.push_back("event has codes above 1, and every nonzero code counts as the event, so this is time "
			                   "free of all of them. For the time lost to one cause, with the others as competing "
			                   "risks, use do_rmtl(..., cause := k)");
		}
		warnings.push_back("this is a difference in restricted mean survival time, not a hazard ratio, and that is "
		                   "deliberate: a hazard ratio compares subjects still at risk, treatment changes who is still "
		                   "at risk, and so the comparison stops being causal after the first events even under "
		                   "randomisation");
	}

	vector<Value> warning_values;
	for (auto &w : warnings) {
		warning_values.push_back(Value(w));
	}

	if (lost) {
		const idx_t lost_events = cause_events[0] + cause_events[1];
		names = {"estimand",          "estimator",         "estimate", "std_error",    "ci_low",
		         "ci_high",           "horizon",           "cause",    "rmtl_treated", "rmtl_control",
		         "incidence_treated", "incidence_control", "n",        "n_events",     "n_competing",
		         "censored_fraction", "warnings"};
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
		                LogicalType::BIGINT,
		                LogicalType::BIGINT,
		                LogicalType::BIGINT,
		                LogicalType::DOUBLE,
		                LogicalType::LIST(LogicalType::VARCHAR)};
		auto bind = make_uniq<ResultBindData>();
		bind->rows.push_back(
		    {Value("RMTL difference"), Value("iptw-aalen-johansen"), Value::DOUBLE(estimate),
		     se > 0.0 ? Value::DOUBLE(se) : Value(LogicalType::DOUBLE),
		     se > 0.0 ? Value::DOUBLE(estimate - Z95 * se) : Value(LogicalType::DOUBLE),
		     se > 0.0 ? Value::DOUBLE(estimate + Z95 * se) : Value(LogicalType::DOUBLE), Value::DOUBLE(horizon),
		     Value::BIGINT(cause), Value::DOUBLE(rmst_treated), Value::DOUBLE(rmst_control),
		     Value::DOUBLE(incidence_treated), Value::DOUBLE(incidence_control),
		     Value::BIGINT(static_cast<int64_t>(frame.n)), Value::BIGINT(static_cast<int64_t>(lost_events)),
		     Value::BIGINT(static_cast<int64_t>(total_events - lost_events)), Value::DOUBLE(censored_fraction),
		     Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
		return std::move(bind);
	}

	names = {"estimand", "estimator",         "estimate",     "std_error",    "ci_low",
	         "ci_high",  "horizon",           "rmst_treated", "rmst_control", "n",
	         "n_events", "censored_fraction", "warnings"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::DOUBLE,
	                LogicalType::LIST(LogicalType::VARCHAR)};

	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value("RMST difference"), Value("iptw-kaplan-meier"), Value::DOUBLE(estimate),
	                      se > 0.0 ? Value::DOUBLE(se) : Value(LogicalType::DOUBLE),
	                      se > 0.0 ? Value::DOUBLE(estimate - Z95 * se) : Value(LogicalType::DOUBLE),
	                      se > 0.0 ? Value::DOUBLE(estimate + Z95 * se) : Value(LogicalType::DOUBLE),
	                      Value::DOUBLE(horizon), Value::DOUBLE(rmst_treated), Value::DOUBLE(rmst_control),
	                      Value::BIGINT(static_cast<int64_t>(frame.n)),
	                      Value::BIGINT(static_cast<int64_t>(total_events)), Value::DOUBLE(censored_fraction),
	                      Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

unique_ptr<FunctionData> BindRmst(ClientContext &context, TableFunctionBindInput &input,
                                  vector<LogicalType> &return_types, vector<string> &names) {
	return BindSurvival(context, input, return_types, names, false);
}

unique_ptr<FunctionData> BindRmtl(ClientContext &context, TableFunctionBindInput &input,
                                  vector<LogicalType> &return_types, vector<string> &names) {
	return BindSurvival(context, input, return_types, names, true);
}

} // namespace

void RegisterSurvivalFunctions(ExtensionLoader &loader) {
	TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, BindRmst, InitGlobal);
	AddCommonNamedParameters(fn);
	fn.named_parameters["cluster"] = LogicalType::VARCHAR;
	fn.named_parameters["duration"] = LogicalType::VARCHAR;
	fn.named_parameters["event"] = LogicalType::VARCHAR;
	fn.named_parameters["horizon"] = LogicalType::DOUBLE;
	RegisterUnderBothNames(loader, fn, "rmst");

	TableFunction lost("", {LogicalType::VARCHAR}, EmitRows, BindRmtl, InitGlobal);
	AddCommonNamedParameters(lost);
	lost.named_parameters["cluster"] = LogicalType::VARCHAR;
	lost.named_parameters["duration"] = LogicalType::VARCHAR;
	lost.named_parameters["event"] = LogicalType::VARCHAR;
	lost.named_parameters["horizon"] = LogicalType::DOUBLE;
	lost.named_parameters["cause"] = LogicalType::BIGINT;
	RegisterUnderBothNames(loader, lost, "rmtl");
}

} // namespace duckdo
} // namespace duckdb
