#include "duckdo/functions.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"

namespace duckdb {
namespace duckdo {

unique_ptr<FunctionData> ResultBindData::Copy() const {
	auto copy = make_uniq<ResultBindData>();
	copy->rows = rows;
	return std::move(copy);
}

bool ResultBindData::Equals(const FunctionData &other) const {
	auto &rhs = other.Cast<ResultBindData>();
	return rows == rhs.rows;
}

namespace {

struct ResultGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<ResultGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<ResultGlobalState>();
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

Value WarningList(const vector<string> &warnings) {
	if (warnings.empty()) {
		return Value::LIST(LogicalType::VARCHAR, vector<Value>());
	}
	vector<Value> values;
	values.reserve(warnings.size());
	for (auto &w : warnings) {
		values.push_back(Value(w));
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(values));
}

void RequireOutcome(const CausalSpec &spec, const char *fn) {
	if (spec.outcome.empty()) {
		throw BinderException("duckdo: %s requires outcome := '<column>'", fn);
	}
}

//! Shared bind for do_ate / do_att / do_atc.
unique_ptr<FunctionData> BindEffect(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names, Estimand estimand) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	RequireOutcome(spec, EstimandName(estimand));
	auto frame = BuildFrame(context, spec);
	auto result = EstimateEffect(frame, spec, estimand);

	names = {"estimand", "estimator", "estimate",  "std_error", "ci_low",          "ci_high",
	         "p_value",  "n",         "n_treated", "n_trimmed", "variance_method", "warnings"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::BIGINT,  LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR)};

	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value(EstimandName(result.estimand)), Value(result.estimator), Value::DOUBLE(result.estimate),
	                      Value::DOUBLE(result.std_error), Value::DOUBLE(result.ci_low), Value::DOUBLE(result.ci_high),
	                      Value::DOUBLE(result.p_value), Value::BIGINT(static_cast<int64_t>(result.n)),
	                      Value::BIGINT(static_cast<int64_t>(result.n_treated)),
	                      Value::BIGINT(static_cast<int64_t>(result.n_trimmed)), Value(result.variance_method),
	                      WarningList(result.warnings)});
	return std::move(bind);
}

unique_ptr<FunctionData> BindAte(ClientContext &context, TableFunctionBindInput &input,
                                 vector<LogicalType> &return_types, vector<string> &names) {
	return BindEffect(context, input, return_types, names, Estimand::ATE);
}

unique_ptr<FunctionData> BindAtt(ClientContext &context, TableFunctionBindInput &input,
                                 vector<LogicalType> &return_types, vector<string> &names) {
	return BindEffect(context, input, return_types, names, Estimand::ATT);
}

unique_ptr<FunctionData> BindAtc(ClientContext &context, TableFunctionBindInput &input,
                                 vector<LogicalType> &return_types, vector<string> &names) {
	return BindEffect(context, input, return_types, names, Estimand::ATC);
}

unique_ptr<FunctionData> BindCate(ClientContext &context, TableFunctionBindInput &input,
                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	RequireOutcome(spec, "do_cate");
	auto frame = BuildFrame(context, spec);
	auto cate = EstimateCate(frame, spec);

	names = {"row_id", "id", "treatment", "outcome", "cate", "cate_low", "cate_high", "learner"};
	return_types = {LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::VARCHAR};

	auto bind = make_uniq<ResultBindData>();
	bind->rows.reserve(frame.n);
	for (idx_t i = 0; i < frame.n; i++) {
		bind->rows.push_back({Value::BIGINT(static_cast<int64_t>(frame.source_row[i])),
		                      frame.has_id ? Value(frame.ids[i]) : Value(LogicalType::VARCHAR),
		                      Value::DOUBLE(frame.t[i]), Value::DOUBLE(frame.y[i]), Value::DOUBLE(cate.cate[i]),
		                      Value::DOUBLE(cate.lo[i]), Value::DOUBLE(cate.hi[i]), Value(cate.learner)});
	}
	return std::move(bind);
}

//! Segmented estimation: one independent estimate per group. A group that is
//! too small to estimate yields a NULL row carrying the reason, rather than
//! failing the whole query - one thin segment should not cost you the report.
unique_ptr<FunctionData> BindAteBy(ClientContext &context, TableFunctionBindInput &input,
                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	RequireOutcome(spec, "do_ate_by");

	vector<string> by;
	auto entry = input.named_parameters.find("by");
	if (entry != input.named_parameters.end() && !entry->second.IsNull()) {
		for (auto &child : ListValue::GetChildren(entry->second)) {
			if (!child.IsNull()) {
				by.push_back(child.ToString());
			}
		}
	}
	if (by.empty()) {
		throw BinderException("duckdo: do_ate_by requires by := ['column'] - without it, use do_ate");
	}

	const string rel = RelationSql(spec.relation);
	const idx_t max_groups = GetSettingIdx(context, "duckdo_max_groups", 1000);

	string quoted_by;
	for (idx_t i = 0; i < by.size(); i++) {
		if (i) {
			quoted_by += ", ";
		}
		quoted_by += QuoteIdentifier(by[i]);
	}
	auto groups = RunQuery(context,
	                       "SELECT DISTINCT " + quoted_by + " FROM " + rel + " ORDER BY " + quoted_by + " LIMIT " +
	                           std::to_string(max_groups + 1),
	                       "listing the groups of " + spec.relation);
	if (groups->RowCount() > max_groups) {
		throw BinderException("duckdo: by := produced more than %llu groups. Narrow the grouping, or raise "
		                      "duckdo_max_groups",
		                      static_cast<unsigned long long>(max_groups));
	}

	names.clear();
	return_types.clear();
	for (auto &column : by) {
		names.push_back(column);
		return_types.push_back(LogicalType::VARCHAR);
	}
	for (auto &extra : {"estimand", "estimator"}) {
		names.push_back(extra);
		return_types.push_back(LogicalType::VARCHAR);
	}
	for (auto &extra : {"estimate", "std_error", "ci_low", "ci_high", "p_value"}) {
		names.push_back(extra);
		return_types.push_back(LogicalType::DOUBLE);
	}
	for (auto &extra : {"n", "n_treated"}) {
		names.push_back(extra);
		return_types.push_back(LogicalType::BIGINT);
	}
	names.push_back("warnings");
	return_types.push_back(LogicalType::LIST(LogicalType::VARCHAR));

	auto bind = make_uniq<ResultBindData>();
	for (idx_t g = 0; g < groups->RowCount(); g++) {
		string predicate;
		vector<Value> key;
		for (idx_t c = 0; c < by.size(); c++) {
			const Value cell = groups->GetValue(c, g);
			key.push_back(cell.IsNull() ? Value(LogicalType::VARCHAR) : Value(cell.ToString()));
			if (!predicate.empty()) {
				predicate += " AND ";
			}
			// Compare as text and use IS NOT DISTINCT FROM, so a NULL group is a
			// group like any other rather than a silently dropped one.
			predicate += "CAST(" + QuoteIdentifier(by[c]) + " AS VARCHAR) IS NOT DISTINCT FROM ";
			predicate += cell.IsNull() ? string("CAST(NULL AS VARCHAR)") : QuoteLiteral(cell.ToString());
		}

		CausalSpec group_spec = spec;
		group_spec.relation = "(SELECT * FROM " + rel + " WHERE " + predicate + ")";
		for (auto &column : by) {
			group_spec.exclude.push_back(column);
		}

		vector<Value> row = key;
		try {
			auto frame = BuildFrame(context, group_spec);
			auto result = EstimateEffect(frame, group_spec, Estimand::ATE);
			row.push_back(Value(EstimandName(result.estimand)));
			row.push_back(Value(result.estimator));
			row.push_back(Value::DOUBLE(result.estimate));
			row.push_back(Value::DOUBLE(result.std_error));
			row.push_back(Value::DOUBLE(result.ci_low));
			row.push_back(Value::DOUBLE(result.ci_high));
			row.push_back(Value::DOUBLE(result.p_value));
			row.push_back(Value::BIGINT(static_cast<int64_t>(result.n)));
			row.push_back(Value::BIGINT(static_cast<int64_t>(result.n_treated)));
			row.push_back(WarningList(result.warnings));
		} catch (const std::exception &ex) {
			row.push_back(Value("ATE"));
			row.push_back(Value(group_spec.estimator));
			for (int i = 0; i < 5; i++) {
				row.push_back(Value(LogicalType::DOUBLE));
			}
			row.push_back(Value(LogicalType::BIGINT));
			row.push_back(Value(LogicalType::BIGINT));
			row.push_back(WarningList({string("not estimated: ") + ex.what()}));
		}
		bind->rows.push_back(std::move(row));
	}
	return std::move(bind);
}

} // namespace

void AddCommonNamedParameters(TableFunction &fn) {
	fn.named_parameters["treatment"] = LogicalType::VARCHAR;
	fn.named_parameters["outcome"] = LogicalType::VARCHAR;
	fn.named_parameters["covariates"] = LogicalType::LIST(LogicalType::VARCHAR);
	fn.named_parameters["exclude"] = LogicalType::LIST(LogicalType::VARCHAR);
	fn.named_parameters["estimator"] = LogicalType::VARCHAR;
	fn.named_parameters["model"] = LogicalType::VARCHAR;
	fn.named_parameters["graph"] = LogicalType::VARCHAR;
	fn.named_parameters["id"] = LogicalType::VARCHAR;
	fn.named_parameters["treated"] = LogicalType::VARCHAR;
	fn.named_parameters["control"] = LogicalType::VARCHAR;
	fn.named_parameters["seed"] = LogicalType::BIGINT;
	fn.named_parameters["folds"] = LogicalType::BIGINT;
	fn.named_parameters["bootstrap_reps"] = LogicalType::BIGINT;
	fn.named_parameters["trim"] = LogicalType::DOUBLE;
}

void RegisterUnderBothNames(ExtensionLoader &loader, TableFunction fn, const string &bare_name) {
	fn.name = "do_" + bare_name;
	loader.RegisterFunction(fn);
	fn.name = "duckdo_" + bare_name;
	loader.RegisterFunction(fn);
}

void RegisterEstimationFunctions(ExtensionLoader &loader) {
	struct Entry {
		const char *name;
		table_function_bind_t bind;
	};
	const Entry effects[] = {{"ate", BindAte}, {"att", BindAtt}, {"atc", BindAtc}};
	for (auto &entry : effects) {
		TableFunction fn("", {LogicalType::VARCHAR}, EmitRows, entry.bind, InitGlobal);
		AddCommonNamedParameters(fn);
		RegisterUnderBothNames(loader, fn, entry.name);
	}

	TableFunction cate("", {LogicalType::VARCHAR}, EmitRows, BindCate, InitGlobal);
	AddCommonNamedParameters(cate);
	RegisterUnderBothNames(loader, cate, "cate");

	TableFunction by("", {LogicalType::VARCHAR}, EmitRows, BindAteBy, InitGlobal);
	AddCommonNamedParameters(by);
	by.named_parameters["by"] = LogicalType::LIST(LogicalType::VARCHAR);
	RegisterUnderBothNames(loader, by, "ate_by");
}

} // namespace duckdo
} // namespace duckdb
