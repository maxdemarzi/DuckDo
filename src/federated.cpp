//===----------------------------------------------------------------------===//
// Phase 10: federated estimation - pooling effects estimated at separate sites
// without moving a single row between them.
//
// An AIPW estimate is the mean of per-row influence values psi, and its
// squared standard error is var(psi) / n. So the effect over the union of every
// site's population, with nuisance models fitted at each site, is exactly the
// n-weighted mean of the site estimates, and its variance is exactly
// sum (n_s / N)^2 se_s^2. Each site publishes (n, estimate, std_error) - the
// columns do_ate and do_ate_by already return - and nothing else.
//
// Inverse-variance weighting, the meta-analysis default, answers a different
// question. It weights sites by precision rather than population, so one small,
// quiet site can outvote a large, noisy one, and the answer is the effect in no
// population anyone asked about. It is used here only for the heterogeneity
// test, where precision weighting is what the test needs.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"
#include "duckdo/linalg.hpp"

#include <cmath>

namespace duckdb {
namespace duckdo {

namespace {

constexpr double kZ95 = 1.959963984540054;

struct PoolGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<PoolGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<PoolGlobalState>();
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

double ChiSquareUpper(double q, double df) {
	return UpperGammaQ(df / 2.0, q / 2.0);
}

string NamedColumn(const named_parameter_map_t &named, const char *key) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		return key;
	}
	return entry->second.ToString();
}

Value OptionalDouble(double value) {
	return std::isfinite(value) ? Value::DOUBLE(value) : Value(LogicalType::DOUBLE);
}

unique_ptr<FunctionData> BindPool(ClientContext &context, TableFunctionBindInput &input,
                                  vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("duckdo: do_ate_pool takes a table of per-site results as its first argument, e.g. "
		                      "do_ate_pool('site_results') or do_ate_pool('(SELECT * FROM do_ate_by(...))')");
	}
	const string relation = input.inputs[0].ToString();
	const string rel = RelationSql(relation);
	auto &named = input.named_parameters;

	auto probe = RunQuery(context, "SELECT * FROM " + rel + " LIMIT 0", "do_ate_pool inspecting " + relation);
	auto find = [&](const string &wanted) {
		for (auto &name : probe->names) {
			if (StringUtil::CIEquals(name, wanted)) {
				return name;
			}
		}
		return string();
	};
	vector<string> columns;
	for (auto *key : {"estimate", "std_error", "n"}) {
		const string wanted = NamedColumn(named, key);
		const string found = find(wanted);
		if (found.empty()) {
			throw BinderException("duckdo: do_ate_pool needs a column '%s' in %s; name a different one with %s := "
			                      "'...'",
			                      wanted, relation, key);
		}
		columns.push_back(found);
	}

	vector<string> warnings;
	// The estimand decides the weights. A population ATE weights each site by
	// its size; a population ATT by its treated count; an ATC by its controls.
	string estimand = "ATE";
	const string estimand_col = find("estimand");
	const string complete = QuoteIdentifier(columns[0]) + " IS NOT NULL AND " + QuoteIdentifier(columns[1]) +
	                        " IS NOT NULL AND " + QuoteIdentifier(columns[2]) + " IS NOT NULL";
	if (!estimand_col.empty()) {
		auto kinds = RunQuery(context,
		                      "SELECT DISTINCT upper(CAST(" + QuoteIdentifier(estimand_col) + " AS VARCHAR)) FROM " +
		                          rel + " WHERE " + complete,
		                      "do_ate_pool reading estimands");
		if (kinds->RowCount() > 1) {
			throw BinderException("duckdo: %s mixes estimands. An ATE and an ATT are effects in different "
			                      "populations, and no weighting turns them into one number",
			                      relation);
		}
		if (kinds->RowCount() == 1 && !kinds->GetValue(0, 0).IsNull()) {
			estimand = kinds->GetValue(0, 0).ToString();
		}
	} else {
		warnings.push_back("no estimand column, so every row is taken to be an ATE and weighted by n");
	}
	if (estimand != "ATE" && estimand != "ATT" && estimand != "ATC") {
		throw BinderException("duckdo: do_ate_pool pools ATE, ATT or ATC estimates; %s holds '%s'", relation, estimand);
	}
	string weight_sql = "CAST(" + QuoteIdentifier(columns[2]) + " AS DOUBLE)";
	if (estimand != "ATE") {
		const string treated_col = find("n_treated");
		if (treated_col.empty()) {
			throw BinderException("duckdo: pooling %s estimates weights each site by its %s count, so %s needs an "
			                      "n_treated column",
			                      estimand, estimand == "ATT" ? "treated" : "control", relation);
		}
		weight_sql = estimand == "ATT" ? "CAST(" + QuoteIdentifier(treated_col) + " AS DOUBLE)"
		                               : weight_sql + " - CAST(" + QuoteIdentifier(treated_col) + " AS DOUBLE)";
	}
	const string estimator_col = find("estimator");
	if (!estimator_col.empty()) {
		auto kinds = RunQuery(
		    context, "SELECT count(DISTINCT " + QuoteIdentifier(estimator_col) + ") FROM " + rel + " WHERE " + complete,
		    "do_ate_pool reading estimators");
		if (kinds->GetValue(0, 0).GetValue<int64_t>() > 1) {
			warnings.push_back("the sites used different estimators; pooling is still exact, but a disagreement "
			                   "between sites may be the estimators' rather than the data's");
		}
	}

	auto data =
	    RunQuery(context,
	             "SELECT CAST(" + QuoteIdentifier(columns[0]) + " AS DOUBLE), CAST(" + QuoteIdentifier(columns[1]) +
	                 " AS DOUBLE), CAST(" + QuoteIdentifier(columns[2]) + " AS DOUBLE), " + weight_sql + " FROM " + rel,
	             "do_ate_pool reading " + relation);
	vector<double> est, se, n, w;
	idx_t skipped = 0;
	for (idx_t r = 0; r < data->RowCount(); r++) {
		bool usable = true;
		for (idx_t c = 0; c < 4; c++) {
			usable = usable && !data->GetValue(c, r).IsNull();
		}
		if (!usable) {
			skipped++;
			continue;
		}
		const double e = data->GetValue(0, r).GetValue<double>();
		const double s = data->GetValue(1, r).GetValue<double>();
		const double size = data->GetValue(2, r).GetValue<double>();
		const double weight = data->GetValue(3, r).GetValue<double>();
		if (!std::isfinite(e) || !std::isfinite(s) || s < 0.0 || !(size > 0.0) || !(weight > 0.0)) {
			throw BinderException("duckdo: row %llu of %s has estimate %g, standard error %g, n %g and weight %g. "
			                      "Estimates must be finite, standard errors non-negative and counts positive",
			                      static_cast<unsigned long long>(r + 1), relation, e, s, size, weight);
		}
		est.push_back(e);
		se.push_back(s);
		n.push_back(size);
		w.push_back(weight);
	}
	const idx_t sites = est.size();
	if (sites < 2) {
		throw BinderException("duckdo: do_ate_pool needs at least two sites with an estimate; %s has %llu", relation,
		                      static_cast<unsigned long long>(sites));
	}

	double total_w = 0.0, total_n = 0.0;
	for (idx_t s = 0; s < sites; s++) {
		total_w += w[s];
		total_n += n[s];
	}
	double pooled = 0.0, variance = 0.0;
	for (idx_t s = 0; s < sites; s++) {
		const double share = w[s] / total_w;
		pooled += share * est[s];
		variance += share * share * se[s] * se[s];
	}
	const double std_error = std::sqrt(variance);

	// Cochran's Q, with precision weights: do the sites disagree by more than
	// their own standard errors allow?
	double q = NAN, q_p = NAN, i_squared = NAN;
	bool all_positive = true;
	for (auto s : se) {
		all_positive = all_positive && s > 0.0;
	}
	if (all_positive) {
		double sum_iv = 0.0, fixed = 0.0;
		for (idx_t s = 0; s < sites; s++) {
			sum_iv += 1.0 / (se[s] * se[s]);
			fixed += est[s] / (se[s] * se[s]);
		}
		fixed /= sum_iv;
		q = 0.0;
		for (idx_t s = 0; s < sites; s++) {
			q += (est[s] - fixed) * (est[s] - fixed) / (se[s] * se[s]);
		}
		const double df = static_cast<double>(sites - 1);
		q_p = ChiSquareUpper(q, df);
		i_squared = q > df ? (q - df) / q : 0.0;
	} else {
		warnings.push_back("a site reported a standard error of zero, so the heterogeneity test was not run");
	}

	warnings.insert(warnings.begin(),
	                StringUtil::Format(
	                    "the %s over every site's population together: each site's estimate weighted by its "
	                    "share of %s. Exact for influence-function estimators with models fitted per site. "
	                    "Assumes the sites are independent samples and no unit appears at two of them",
	                    estimand, estimand == "ATE" ? "rows" : (estimand == "ATT" ? "treated rows" : "control rows")));
	if (skipped > 0) {
		warnings.push_back(StringUtil::Format(
		    "%llu rows with a NULL estimate, standard error or count were left out; a do_ate_by group that could "
		    "not be estimated reports NULLs",
		    static_cast<unsigned long long>(skipped)));
	}
	if (std::isfinite(q_p) && q_p < 0.05 && i_squared >= 0.5) {
		warnings.push_back(StringUtil::Format(
		    "the sites disagree (I-squared %.0f%%, p = %.2g): the pooled number averages different effects. It is "
		    "still the effect across all of them, but report the sites as well",
		    100.0 * i_squared, q_p));
	}
	if (std::isfinite(q_p) && sites < 5) {
		warnings.push_back(StringUtil::Format(
		    "with %llu sites the heterogeneity test has little power; a large p-value is weak evidence they agree",
		    static_cast<unsigned long long>(sites)));
	}

	names = {"estimand", "estimate", "std_error", "ci_low",    "ci_high",   "p_value",
	         "n_sites",  "n",        "q",         "q_p_value", "i_squared", "warnings"};
	return_types = {LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::BIGINT,  LogicalType::BIGINT, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::LIST(LogicalType::VARCHAR)};
	vector<Value> warning_values;
	for (auto &warning : warnings) {
		warning_values.push_back(Value(warning));
	}
	auto bind = make_uniq<ResultBindData>();
	const bool has_se = std_error > 0.0;
	bind->rows.push_back({Value(estimand), Value::DOUBLE(pooled), Value::DOUBLE(std_error),
	                      has_se ? Value::DOUBLE(pooled - kZ95 * std_error) : Value(LogicalType::DOUBLE),
	                      has_se ? Value::DOUBLE(pooled + kZ95 * std_error) : Value(LogicalType::DOUBLE),
	                      has_se ? Value::DOUBLE(NormalTwoSidedP(pooled / std_error)) : Value(LogicalType::DOUBLE),
	                      Value::BIGINT(static_cast<int64_t>(sites)),
	                      Value::BIGINT(static_cast<int64_t>(std::llround(total_n))), OptionalDouble(q),
	                      OptionalDouble(q_p), OptionalDouble(i_squared),
	                      Value::LIST(LogicalType::VARCHAR, std::move(warning_values))});
	return std::move(bind);
}

} // namespace

void RegisterFederatedFunctions(ExtensionLoader &loader) {
	TableFunction pool("", {LogicalType::VARCHAR}, EmitRows, BindPool, InitGlobal);
	pool.named_parameters["estimate"] = LogicalType::VARCHAR;
	pool.named_parameters["std_error"] = LogicalType::VARCHAR;
	pool.named_parameters["n"] = LogicalType::VARCHAR;
	RegisterUnderBothNames(loader, pool, "ate_pool");
}

} // namespace duckdo
} // namespace duckdb
