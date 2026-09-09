//===----------------------------------------------------------------------===//
// Phase 7: the do() surface.
//
// Not "estimate an effect" but "query a world that did not happen". These
// functions all run on the classical backends from phases 2-3; when the
// foundation models land in phase 6 they become an alternative engine behind
// the same SQL, not a new set of functions.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {
namespace duckdo {

namespace {

struct InterventionGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<InterventionGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<InterventionGlobalState>();
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

Value IdValue(const CausalFrame &frame, idx_t i) {
	return frame.has_id ? Value(frame.ids[i]) : Value(LogicalType::VARCHAR);
}

// --- do_counterfactual ------------------------------------------------------

unique_ptr<FunctionData> BindCounterfactual(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_counterfactual requires outcome := '<column>'");
	}
	auto frame = BuildFrame(context, spec);
	auto fit = FitNuisance(frame, spec, spec.seed);
	auto cate = EstimateCate(frame, spec);

	names = {"row_id", "id", "treatment", "observed", "y0", "y1", "effect", "effect_low", "effect_high"};
	return_types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE,  LogicalType::DOUBLE};

	auto bind = make_uniq<ResultBindData>();
	bind->rows.reserve(frame.n);
	for (idx_t i = 0; i < frame.n; i++) {
		bind->rows.push_back({Value::BIGINT(static_cast<int64_t>(frame.source_row[i])), IdValue(frame, i),
		                      Value::DOUBLE(frame.t[i]), Value::DOUBLE(frame.y[i]), Value::DOUBLE(fit.mu0[i]),
		                      Value::DOUBLE(fit.mu1[i]), Value::DOUBLE(cate.cate[i]), Value::DOUBLE(cate.lo[i]),
		                      Value::DOUBLE(cate.hi[i])});
	}
	return std::move(bind);
}

// --- do_predict -------------------------------------------------------------

//! Flatten a STRUCT or MAP literal into (column, value) pairs.
vector<std::pair<string, Value>> ParseIntervention(const Value &value) {
	vector<std::pair<string, Value>> out;
	if (value.IsNull()) {
		return out;
	}
	const auto &type = value.type();
	if (type.id() == LogicalTypeId::STRUCT) {
		const auto &child_types = StructType::GetChildTypes(type);
		const auto &children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < children.size(); i++) {
			out.emplace_back(child_types[i].first, children[i]);
		}
		return out;
	}
	if (type.id() == LogicalTypeId::MAP) {
		for (auto &entry : ListValue::GetChildren(value)) {
			const auto &kv = StructValue::GetChildren(entry);
			if (kv.size() == 2) {
				out.emplace_back(kv[0].ToString(), kv[1]);
			}
		}
		return out;
	}
	throw BinderException("duckdo: intervention := must be a struct or a map, e.g. "
	                      "intervention := {'price': 19.99}. Got %s",
	                      type.ToString());
}

//! Force the named columns to the given values, in the frame's encoded space.
//! An intervention is not a filter: every row is set, which is exactly what
//! distinguishes do(X = x) from conditioning on X = x.
void ApplyIntervention(CausalFrame &frame, const CausalSpec &spec, const vector<std::pair<string, Value>> &pairs) {
	for (auto &entry : pairs) {
		const string &column = entry.first;
		const Value &value = entry.second;

		if (StringUtil::CIEquals(column, spec.treatment)) {
			double arm;
			if (value.type().id() == LogicalTypeId::BOOLEAN) {
				arm = value.GetValue<bool>() ? 1.0 : 0.0;
			} else if (value.type().IsNumeric()) {
				arm = value.GetValue<double>() != 0.0 ? 1.0 : 0.0;
			} else {
				arm = StringUtil::CIEquals(value.ToString(), frame.treated_label) ? 1.0 : 0.0;
			}
			for (idx_t i = 0; i < frame.n; i++) {
				frame.t[i] = arm;
			}
			continue;
		}

		bool matched = false;
		for (idx_t j = 0; j < frame.features.size(); j++) {
			auto &info = frame.features[j];
			if (!StringUtil::CIEquals(info.source, column)) {
				continue;
			}
			matched = true;
			double raw;
			switch (info.kind) {
			case FeatureKind::ONE_HOT:
				raw = StringUtil::CIEquals(value.ToString(), info.level) ? 1.0 : 0.0;
				break;
			case FeatureKind::MISSING_INDICATOR:
				// Intervening means the value is now known.
				raw = 0.0;
				break;
			default:
				if (!value.type().IsNumeric() && value.type().id() != LogicalTypeId::BOOLEAN) {
					throw BinderException("duckdo: intervention on '%s' needs a numeric value, got '%s'", column,
					                      value.ToString());
				}
				raw = value.type().id() == LogicalTypeId::BOOLEAN ? (value.GetValue<bool>() ? 1.0 : 0.0)
				                                                  : value.GetValue<double>();
				break;
			}
			const double encoded = (raw - info.center) / info.scale;
			for (idx_t i = 0; i < frame.n; i++) {
				frame.X.At(i, j) = encoded;
			}
		}
		if (!matched) {
			string available = spec.treatment;
			for (auto &c : frame.covariate_columns) {
				available += ", " + c;
			}
			throw BinderException("duckdo: cannot intervene on '%s' because it is not the treatment and not an "
			                      "encoded covariate. Interveneable columns: %s",
			                      column, available);
		}
	}
}

unique_ptr<FunctionData> BindPredict(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_predict requires outcome := '<column>'");
	}
	auto entry = input.named_parameters.find("intervention");
	if (entry == input.named_parameters.end() || entry->second.IsNull()) {
		throw BinderException("duckdo: do_predict requires intervention := {'column': value}, which is what makes it "
		                      "a do() query rather than a prediction");
	}
	const auto pairs = ParseIntervention(entry->second);
	if (pairs.empty()) {
		throw BinderException("duckdo: intervention := is empty; name at least one column to set");
	}

	auto frame = BuildFrame(context, spec);
	const double lambda = RidgeLambdaFor(frame);

	// Fit the outcome surfaces on the world as it is, BEFORE intervening.
	const auto rows1 = frame.ArmRows(1.0);
	const auto rows0 = frame.ArmRows(0.0);
	LinearModel logistic1, logistic0;
	RidgeFit ridge1, ridge0;
	if (frame.binary_outcome) {
		logistic1 = FitLogistic(frame.X, frame.y, rows1, {}, std::max(lambda, 1.0), 30);
		logistic0 = FitLogistic(frame.X, frame.y, rows0, {}, std::max(lambda, 1.0), 30);
	} else {
		ridge1 = FitRidgeWithCovariance(frame.X, frame.y, rows1, lambda);
		ridge0 = FitRidgeWithCovariance(frame.X, frame.y, rows0, lambda);
	}

	// Now set the world to the counterfactual one.
	CausalFrame intervened = frame;
	ApplyIntervention(intervened, spec, pairs);

	string description;
	for (auto &p : pairs) {
		if (!description.empty()) {
			description += ", ";
		}
		description += p.first + " = " + p.second.ToString();
	}

	names = {"row_id", "id", "intervention", "observed", "predicted", "predicted_low", "predicted_high"};
	return_types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE,  LogicalType::DOUBLE};

	auto bind = make_uniq<ResultBindData>();
	bind->rows.reserve(intervened.n);
	for (idx_t i = 0; i < intervened.n; i++) {
		const double *x = intervened.X.Row(i);
		const bool treated = intervened.t[i] == 1.0;
		double point, se;
		if (frame.binary_outcome) {
			point = (treated ? logistic1 : logistic0).Predict(x, intervened.X.cols);
			se = 0.0;
		} else {
			const RidgeFit &model = treated ? ridge1 : ridge0;
			point = model.model.Eta(x, intervened.X.cols);
			se = model.PredictionStdError(x, intervened.X.cols);
		}
		const Value low = se > 0.0 ? Value::DOUBLE(point - Z95 * se) : Value(LogicalType::DOUBLE);
		const Value high = se > 0.0 ? Value::DOUBLE(point + Z95 * se) : Value(LogicalType::DOUBLE);
		bind->rows.push_back({Value::BIGINT(static_cast<int64_t>(intervened.source_row[i])), IdValue(intervened, i),
		                      Value(description), Value::DOUBLE(frame.y[i]), Value::DOUBLE(point), low, high});
	}
	return std::move(bind);
}

// --- do_policy_value --------------------------------------------------------

//! The targeting rule: either a boolean column the user computed in SQL, or a
//! threshold on the estimated effect.
vector<uint8_t> ResolvePolicy(const CausalFrame &frame, const CausalSpec &spec, const CateResult &cate,
                              string &description) {
	vector<uint8_t> policy(frame.n, 0);
	if (frame.has_policy) {
		description = "column '" + spec.policy_column + "'";
		for (idx_t i = 0; i < frame.n; i++) {
			policy[i] = frame.policy[i];
		}
		return policy;
	}
	description = StringUtil::Format("cate > %g", spec.threshold);
	for (idx_t i = 0; i < frame.n; i++) {
		policy[i] = cate.cate[i] > spec.threshold ? 1 : 0;
	}
	return policy;
}

unique_ptr<FunctionData> BindPolicyValue(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_policy_value requires outcome := '<column>'");
	}
	auto frame = BuildFrame(context, spec);
	auto fit = FitNuisance(frame, spec, spec.seed);
	auto psi = AipwPseudoOutcome(frame, fit);
	auto cate = EstimateCate(frame, spec);

	string description;
	const auto policy = ResolvePolicy(frame, spec, cate, description);

	// The value of a policy, relative to treating nobody, is E[pi(X) * tau(X)].
	// The doubly-robust pseudo-outcome is an unbiased per-row estimate of tau.
	const double n = static_cast<double>(frame.n);
	double targeted = 0.0, value = 0.0, treat_all = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		targeted += policy[i];
		value += policy[i] ? psi[i] : 0.0;
		treat_all += psi[i];
	}
	value /= n;
	treat_all /= n;
	double variance = 0.0;
	for (idx_t i = 0; i < frame.n; i++) {
		const double contribution = (policy[i] ? psi[i] : 0.0) - value;
		variance += contribution * contribution;
	}
	const double se = std::sqrt(variance / (n * (n - 1.0)));

	names = {"policy", "n_targeted", "share_targeted",  "policy_value",     "std_error",
	         "ci_low", "ci_high",    "value_treat_all", "value_treat_none", "lift_over_treat_all"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE};

	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value(description), Value::BIGINT(static_cast<int64_t>(targeted)),
	                      Value::DOUBLE(targeted / n), Value::DOUBLE(value), Value::DOUBLE(se),
	                      Value::DOUBLE(value - Z95 * se), Value::DOUBLE(value + Z95 * se), Value::DOUBLE(treat_all),
	                      Value::DOUBLE(0.0), Value::DOUBLE(value - treat_all)});
	return std::move(bind);
}

// --- do_uplift --------------------------------------------------------------

unique_ptr<FunctionData> BindUplift(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_uplift requires outcome := '<column>'");
	}
	auto frame = BuildFrame(context, spec);
	auto fit = FitNuisance(frame, spec, spec.seed);
	auto psi = AipwPseudoOutcome(frame, fit);
	auto cate = EstimateCate(frame, spec);

	// Rank by estimated effect, best first, and walk down the list.
	vector<idx_t> order(frame.n);
	for (idx_t i = 0; i < frame.n; i++) {
		order[i] = i;
	}
	std::stable_sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return cate.cate[a] > cate.cate[b]; });

	double total = 0.0;
	for (auto v : psi) {
		total += v;
	}
	const double n = static_cast<double>(frame.n);

	names = {"bucket", "fraction_targeted", "n_targeted", "cumulative_gain", "random_gain", "qini"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::BIGINT,
	                LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE};

	auto bind = make_uniq<ResultBindData>();
	const idx_t buckets = 20;
	double running = 0.0;
	idx_t cursor = 0;
	for (idx_t b = 1; b <= buckets; b++) {
		const idx_t upto = static_cast<idx_t>(std::llround(n * static_cast<double>(b) / static_cast<double>(buckets)));
		while (cursor < upto && cursor < frame.n) {
			running += psi[order[cursor]];
			cursor++;
		}
		const double fraction = static_cast<double>(cursor) / n;
		const double gain = running / n;
		const double random = fraction * (total / n);
		bind->rows.push_back({Value::BIGINT(static_cast<int64_t>(b)), Value::DOUBLE(fraction),
		                      Value::BIGINT(static_cast<int64_t>(cursor)), Value::DOUBLE(gain), Value::DOUBLE(random),
		                      Value::DOUBLE(gain - random)});
	}
	return std::move(bind);
}

// --- do_optimal_policy ------------------------------------------------------

struct PolicyLeaf {
	vector<idx_t> rows;
	string rule;
	idx_t depth = 0;
};

//! One greedy split: bucket each feature into deciles, then read off every cut
//! point from prefix sums. O(features * rows) rather than O(features * rows^2).
//! `psi` here is the NET value of treating a row - the estimated effect minus
//! whatever treating costs. That matters: with a cost of zero and an effect that
//! is positive everywhere, no split can ever beat treating everyone, so the tree
//! correctly returns a single leaf and the search looks broken when it is not.
//! Subtracting the cost first is what makes the question "who is worth treating"
//! have an interesting answer.
bool BestSplit(const CausalFrame &frame, const vector<double> &psi, const vector<idx_t> &rows, idx_t &best_feature,
               double &best_threshold, double &best_value) {
	const idx_t kBuckets = 10;
	double baseline = 0.0;
	for (auto r : rows) {
		baseline += psi[r];
	}
	best_value = std::max(0.0, baseline);
	bool found = false;

	for (idx_t j = 0; j < frame.X.cols; j++) {
		vector<double> values;
		values.reserve(rows.size());
		for (auto r : rows) {
			values.push_back(frame.X.At(r, j));
		}
		vector<double> sorted = values;
		std::sort(sorted.begin(), sorted.end());
		vector<double> cuts;
		for (idx_t b = 1; b < kBuckets; b++) {
			const idx_t pos = sorted.size() * b / kBuckets;
			if (pos < sorted.size()) {
				const double cut = sorted[pos];
				if (cuts.empty() || cut > cuts.back()) {
					cuts.push_back(cut);
				}
			}
		}
		if (cuts.empty()) {
			continue;
		}
		vector<double> below(cuts.size(), 0.0);
		vector<idx_t> counts(cuts.size(), 0);
		for (idx_t k = 0; k < rows.size(); k++) {
			const double v = values[k];
			const double p = psi[rows[k]];
			for (idx_t c = 0; c < cuts.size(); c++) {
				if (v <= cuts[c]) {
					below[c] += p;
					counts[c]++;
				}
			}
		}
		for (idx_t c = 0; c < cuts.size(); c++) {
			if (counts[c] < 20 || rows.size() - counts[c] < 20) {
				continue;
			}
			const double value = std::max(0.0, below[c]) + std::max(0.0, baseline - below[c]);
			if (value > best_value + 1e-9) {
				best_value = value;
				best_feature = j;
				best_threshold = cuts[c];
				found = true;
			}
		}
	}
	return found;
}

unique_ptr<FunctionData> BindOptimalPolicy(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_optimal_policy requires outcome := '<column>'");
	}
	auto frame = BuildFrame(context, spec);
	auto fit = FitNuisance(frame, spec, spec.seed);
	auto psi = AipwPseudoOutcome(frame, fit);

	// `threshold` is the cost of treating one row, on the outcome's scale, and
	// the tree searches on the effect net of it. Without this the parameter was
	// accepted, documented and ignored: every leaf came back "treat", because
	// with no cost and a positive effect there is never a reason not to.
	const double cost = spec.threshold;
	vector<double> net = psi;
	for (auto &v : net) {
		v -= cost;
	}

	const idx_t max_depth = std::min<idx_t>(std::max<idx_t>(spec.depth, 1), 3);
	vector<PolicyLeaf> leaves;
	vector<PolicyLeaf> frontier {PolicyLeaf {frame.AllRows(), "all rows", 0}};

	while (!frontier.empty()) {
		PolicyLeaf node = frontier.back();
		frontier.pop_back();
		idx_t feature = 0;
		double threshold = 0.0, value = 0.0;
		if (node.depth < max_depth && node.rows.size() >= 60 &&
		    BestSplit(frame, net, node.rows, feature, threshold, value)) {
			auto &info = frame.features[feature];
			// Report the cut in the column's own units, not standardised space.
			const double original = info.center + info.scale * threshold;
			PolicyLeaf lo, hi;
			lo.depth = hi.depth = node.depth + 1;
			lo.rule = node.rule == "all rows" ? StringUtil::Format("%s <= %.4g", info.name, original)
			                                  : node.rule + StringUtil::Format(" AND %s <= %.4g", info.name, original);
			hi.rule = node.rule == "all rows" ? StringUtil::Format("%s > %.4g", info.name, original)
			                                  : node.rule + StringUtil::Format(" AND %s > %.4g", info.name, original);
			for (auto r : node.rows) {
				(frame.X.At(r, feature) <= threshold ? lo : hi).rows.push_back(r);
			}
			frontier.push_back(std::move(lo));
			frontier.push_back(std::move(hi));
		} else {
			leaves.push_back(std::move(node));
		}
	}

	names = {"leaf", "rule", "n", "mean_effect", "std_error", "cost", "action", "expected_gain"};
	return_types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR, LogicalType::DOUBLE};

	std::stable_sort(leaves.begin(), leaves.end(),
	                 [](const PolicyLeaf &a, const PolicyLeaf &b) { return a.rule < b.rule; });

	auto bind = make_uniq<ResultBindData>();
	for (idx_t l = 0; l < leaves.size(); l++) {
		auto &leaf = leaves[l];
		double sum = 0.0, net_sum = 0.0;
		for (auto r : leaf.rows) {
			sum += psi[r];
			net_sum += net[r];
		}
		const double count = static_cast<double>(leaf.rows.size());
		// mean_effect stays the effect itself, because that is the quantity a
		// reader wants to compare against the cost. The decision is made on the
		// net figure.
		const double mean = count > 0.0 ? sum / count : 0.0;
		double variance = 0.0;
		for (auto r : leaf.rows) {
			const double d = psi[r] - mean;
			variance += d * d;
		}
		const double se = count > 1.0 ? std::sqrt(variance / (count * (count - 1.0))) : 0.0;
		const bool treat = net_sum > 0.0;
		bind->rows.push_back({Value::BIGINT(static_cast<int64_t>(l)), Value(leaf.rule),
		                      Value::BIGINT(static_cast<int64_t>(leaf.rows.size())), Value::DOUBLE(mean),
		                      Value::DOUBLE(se), Value::DOUBLE(cost), Value(treat ? "treat" : "do not treat"),
		                      Value::DOUBLE(treat ? net_sum / static_cast<double>(frame.n) : 0.0)});
	}
	return std::move(bind);
}

} // namespace

void RegisterInterventionFunctions(ExtensionLoader &loader) {
	struct Entry {
		const char *name;
		table_function_bind_t bind;
	};
	const Entry entries[] = {{"counterfactual", BindCounterfactual},
	                         {"predict", BindPredict},
	                         {"policy_value", BindPolicyValue},
	                         {"uplift", BindUplift},
	                         {"optimal_policy", BindOptimalPolicy}};
	for (auto &entry : entries) {
		TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, entry.bind, InitGlobal);
		AddCommonNamedParameters(fn);
		fn.named_parameters["intervention"] = LogicalType::ANY;
		fn.named_parameters["policy"] = LogicalType::VARCHAR;
		fn.named_parameters["threshold"] = LogicalType::DOUBLE;
		fn.named_parameters["depth"] = LogicalType::BIGINT;
		RegisterUnderBothNames(loader, fn, entry.name);
	}
}

} // namespace duckdo
} // namespace duckdb
