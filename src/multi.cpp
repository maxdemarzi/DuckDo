//===----------------------------------------------------------------------===//
// Phase 10: treatments with more than two levels.
//
// A binary estimator can already be pointed at two levels of a multi-valued
// column with treated := / control :=, and the frame drops every other row.
// That answers a narrower question than it appears to: the effect among the
// units that received one of those two levels. When who gets which level
// depends on the covariates - the reason to adjust at all - that subpopulation
// is not the population, and E[Y(B)] - E[Y(A)] over everyone is a different
// number. The test for this function builds a case where the two differ by
// more than half a unit.
//
// do_ate_levels estimates the population contrast of every level against a
// reference, with every row contributing to every contrast. It is the multi-arm
// generalisation of AIPW:
//
//   psi_k(i) = mu_k(x_i) + 1[T_i = k] (Y_i - mu_k(x_i)) / e_k(x_i)
//
// mu_k is an outcome surface fitted on level k alone; e_k = P(T = k | X) comes
// from one-vs-rest logistic fits normalised to sum to one; both are cross-fitted.
// Each contrast is the mean of psi_k - psi_ref, with the standard deviation of
// that score over sqrt(n) as its standard error.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"
#include "duckdo/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace duckdb {
namespace duckdo {

namespace {

struct LevelGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<LevelGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<LevelGlobalState>();
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

//! Folds stratified within every level, over the canonical order. The binary
//! AssignFolds stratifies arms 0 and 1 only, and would leave every row of a
//! third level in fold zero - so that level would never be held out, and its
//! outcome surface would be scored on rows it was fitted to.
vector<idx_t> AssignFoldsByLevel(const CausalFrame &frame, idx_t folds, int64_t seed) {
	vector<idx_t> assignment(frame.n, 0);
	std::mt19937_64 rng(static_cast<uint64_t>(seed));
	for (idx_t level = 0; level < frame.levels.size(); level++) {
		vector<idx_t> rows = frame.ArmRows(static_cast<double>(level));
		std::shuffle(rows.begin(), rows.end(), rng);
		for (idx_t i = 0; i < rows.size(); i++) {
			assignment[rows[i]] = i % folds;
		}
	}
	return assignment;
}

Value WarningValue(const vector<string> &warnings) {
	vector<Value> out;
	for (auto &warning : warnings) {
		out.push_back(Value(warning));
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(out));
}

unique_ptr<FunctionData> BindAteLevels(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: do_ate_levels requires outcome := '<column>'");
	}
	if (!spec.treated_label.empty() || !spec.control_label.empty()) {
		throw BinderException("duckdo: do_ate_levels contrasts every level against one reference; pass reference := "
		                      "'<level>' rather than treated := / control :=");
	}
	// Only an estimator the caller asked for by name is refused. The session
	// default is not the caller's choice for this function.
	auto requested = input.named_parameters.find("estimator");
	if (requested != input.named_parameters.end() && !requested->second.IsNull() &&
	    !StringUtil::CIEquals(requested->second.ToString(), "aipw")) {
		throw BinderException("duckdo: do_ate_levels estimates with aipw, the multi-arm doubly robust estimator; "
		                      "'%s' is not available for more than two levels",
		                      requested->second.ToString());
	}
	spec.multi_treatment = true;
	auto frame = BuildFrame(context, spec);
	const idx_t levels = frame.levels.size();

	idx_t ref = 0;
	if (!spec.reference.empty()) {
		ref = levels;
		for (idx_t k = 0; k < levels; k++) {
			if (frame.levels[k] == spec.reference) {
				ref = k;
			}
		}
		if (ref == levels) {
			string known;
			for (auto &level : frame.levels) {
				known += (known.empty() ? "'" : ", '") + level + "'";
			}
			throw BinderException("duckdo: reference '%s' is not a level of '%s'. Levels: %s", spec.reference,
			                      spec.treatment, known);
		}
	}

	const idx_t folds = std::min<idx_t>(std::max<idx_t>(spec.folds, 2), std::max<idx_t>(2, frame.n / 4));
	// With clusters, a unit's rows share a fold, whatever their levels.
	const auto assignment =
	    frame.has_cluster ? AssignFoldsByCluster(frame, folds, spec.seed) : AssignFoldsByLevel(frame, folds, spec.seed);
	const double lambda = RidgeLambda(frame);

	vector<vector<double>> indicator(levels, vector<double>(frame.n, 0.0));
	for (idx_t i = 0; i < frame.n; i++) {
		indicator[static_cast<idx_t>(frame.t[i])][i] = 1.0;
	}
	vector<vector<double>> e(levels, vector<double>(frame.n, 1.0 / static_cast<double>(levels)));
	vector<vector<double>> mu(levels, vector<double>(frame.n, 0.0));

	for (idx_t f = 0; f < folds; f++) {
		vector<idx_t> train, test;
		vector<vector<idx_t>> train_by_level(levels);
		for (idx_t i = 0; i < frame.n; i++) {
			if (assignment[i] == f) {
				test.push_back(i);
			} else {
				train.push_back(i);
				train_by_level[static_cast<idx_t>(frame.t[i])].push_back(i);
			}
		}
		if (test.empty() || train.empty()) {
			continue;
		}
		for (idx_t k = 0; k < levels; k++) {
			auto model = FitLogistic(frame.X, indicator[k], train, {}, std::max(lambda, 1.0), 30);
			for (auto r : test) {
				e[k][r] = model.Predict(frame.X.Row(r), frame.X.cols);
			}
			if (!train_by_level[k].empty()) {
				FitAndPredictOutcome(frame, train_by_level[k], test, mu[k]);
			}
		}
	}

	// One-vs-rest fits do not sum to one on their own; normalising them is the
	// standard approximation to a multinomial model, and keeps every row's
	// weights a proper distribution over the levels.
	double smallest = 1.0;
	for (idx_t i = 0; i < frame.n; i++) {
		double total = 0.0;
		for (idx_t k = 0; k < levels; k++) {
			e[k][i] = std::min(std::max(e[k][i], 1e-6), 1.0 - 1e-6);
			total += e[k][i];
		}
		for (idx_t k = 0; k < levels; k++) {
			e[k][i] /= total;
			smallest = std::min(smallest, e[k][i]);
		}
	}

	vector<vector<double>> psi(levels, vector<double>(frame.n, 0.0));
	for (idx_t k = 0; k < levels; k++) {
		for (idx_t i = 0; i < frame.n; i++) {
			psi[k][i] = mu[k][i] + indicator[k][i] * (frame.y[i] - mu[k][i]) / e[k][i];
		}
	}

	vector<string> warnings = frame.warnings;
	if (smallest < 0.01) {
		warnings.push_back(StringUtil::Format(
		    "the smallest estimated propensity of any level is %.4f; a level some units almost never receive makes "
		    "its contrast lean on extrapolation - check the overlap per level before trusting it",
		    smallest));
	}

	vector<double> level_sum(levels, 0.0);
	vector<idx_t> level_count(levels, 0);
	for (idx_t i = 0; i < frame.n; i++) {
		const auto k = static_cast<idx_t>(frame.t[i]);
		level_sum[k] += frame.y[i];
		level_count[k]++;
	}

	names = {"level",   "reference", "estimand",         "estimator", "estimate",    "std_error", "ci_low",
	         "ci_high", "p_value",   "naive_difference", "n_level",   "n_reference", "warnings"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::LIST(LogicalType::VARCHAR)};

	auto bind = make_uniq<ResultBindData>();
	const auto n = static_cast<double>(frame.n);
	for (idx_t k = 0; k < levels; k++) {
		if (k == ref) {
			continue;
		}
		double mean = 0.0;
		for (idx_t i = 0; i < frame.n; i++) {
			mean += psi[k][i] - psi[ref][i];
		}
		mean /= n;
		double variance = 0.0;
		for (idx_t i = 0; i < frame.n; i++) {
			const double d = psi[k][i] - psi[ref][i] - mean;
			variance += d * d;
		}
		variance /= std::max(n - 1.0, 1.0);

		EffectResult result;
		result.estimate = mean;
		result.std_error = std::sqrt(variance / n);
		if (frame.has_cluster) {
			vector<double> centred(frame.n, 0.0);
			for (idx_t i = 0; i < frame.n; i++) {
				centred[i] = psi[k][i] - psi[ref][i] - mean;
			}
			result.std_error = ClusterSe(frame, frame.AllRows(), centred);
		}
		result.Finalize();
		const double naive =
		    level_sum[k] / static_cast<double>(level_count[k]) - level_sum[ref] / static_cast<double>(level_count[ref]);
		bind->rows.push_back({Value(frame.levels[k]), Value(frame.levels[ref]), Value("ATE"), Value("aipw"),
		                      Value::DOUBLE(result.estimate), Value::DOUBLE(result.std_error),
		                      Value::DOUBLE(result.ci_low), Value::DOUBLE(result.ci_high),
		                      Value::DOUBLE(result.p_value), Value::DOUBLE(naive),
		                      Value::BIGINT(static_cast<int64_t>(level_count[k])),
		                      Value::BIGINT(static_cast<int64_t>(level_count[ref])), WarningValue(warnings)});
	}
	return std::move(bind);
}

} // namespace

void RegisterMultiLevelFunctions(ExtensionLoader &loader) {
	TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, BindAteLevels, InitGlobal);
	AddCommonNamedParameters(fn);
	fn.named_parameters["cluster"] = LogicalType::VARCHAR;
	fn.named_parameters["reference"] = LogicalType::VARCHAR;
	RegisterUnderBothNames(loader, fn, "ate_levels");
}

} // namespace duckdo
} // namespace duckdb
