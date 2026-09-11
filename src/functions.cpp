#include "duckdo/functions.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdo/estimators.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/runtime.hpp"

#include <atomic>
#include <cmath>

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

//! Resolve model := against the catalog, with an actionable message on a typo.
const ModelInfo &RequireModel(const CausalSpec &spec) {
	auto model = FindModel(spec.model);
	if (!model) {
		throw BinderException("duckdo: unknown model '%s'. Available: %s. Run SELECT * FROM do_list_models() to see "
		                      "which are ready to use",
		                      spec.model, KnownModels());
	}
	return *model;
}

//! A foundation model returns per-row effects; the population effect is their
//! mean. The interval is the dispersion of those effects, which understates the
//! truth because it carries no model uncertainty - so it is labelled as such
//! rather than dressed up as an influence-function interval.
EffectResult CfmEffect(ClientContext &context, const CausalFrame &frame, const CausalSpec &spec, Estimand estimand) {
	const auto &model = RequireModel(spec);
	auto cfm = CfmEstimateCate(context, frame, spec, model);

	vector<idx_t> target;
	for (idx_t i = 0; i < frame.n; i++) {
		if (estimand == Estimand::ATE || (estimand == Estimand::ATT && frame.t[i] == 1.0) ||
		    (estimand == Estimand::ATC && frame.t[i] == 0.0)) {
			target.push_back(i);
		}
	}
	EffectResult result;
	result.estimator = model.id;
	double sum = 0.0;
	for (auto i : target) {
		sum += cfm.cate[i];
	}
	result.estimate = target.empty() ? 0.0 : sum / static_cast<double>(target.size());
	double variance = 0.0;
	for (auto i : target) {
		const double d = cfm.cate[i] - result.estimate;
		variance += d * d;
	}
	double sampling_var = 0.0;
	if (target.size() > 1) {
		variance /= static_cast<double>(target.size() - 1);
		sampling_var = variance / static_cast<double>(target.size());
	}
	if (cfm.draws > 1) {
		// Two independent sources: which rows the model saw, and which rows we
		// happened to estimate over. They add.
		const double between = cfm.ate_between_draw_se;
		result.std_error = std::sqrt(sampling_var + between * between);
		result.variance_method = StringUtil::Format("context ensemble, %llu draws (no parameter uncertainty)",
		                                            static_cast<unsigned long long>(cfm.draws));
	} else {
		result.std_error = std::sqrt(sampling_var);
		result.variance_method = "effect dispersion (no model uncertainty)";
	}
	result.warnings.push_back(StringUtil::Format(
	    "%s ran with a %llu-row context (ladder rung %llu); the interval carries sampling spread only",
	    model.id.c_str(), static_cast<unsigned long long>(cfm.context_used),
	    static_cast<unsigned long long>(cfm.ladder_rung)));
	for (auto &w : cfm.warnings) {
		result.warnings.push_back(w);
	}
	return result;
}

//! Shared bind for do_ate / do_att / do_atc.
unique_ptr<FunctionData> BindEffect(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names, Estimand estimand) {
	auto spec = CausalSpec::Parse(context, input.inputs, input.named_parameters);
	RequireOutcome(spec, EstimandName(estimand));
	auto frame = BuildFrame(context, spec);
	auto result =
	    spec.model.empty() ? EstimateEffect(frame, spec, estimand) : CfmEffect(context, frame, spec, estimand);
	if (!spec.model.empty()) {
		result.estimand = estimand;
		result.n = frame.n;
		result.n_treated = frame.n_treated;
		for (auto &w : frame.warnings) {
			result.warnings.push_back(w);
		}
		result.Finalize();
	}

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
	CateResult cate;
	if (spec.model.empty()) {
		cate = EstimateCate(frame, spec);
	} else {
		const auto &model = RequireModel(spec);
		auto cfm = CfmEstimateCate(context, frame, spec, model);
		cate.cate = std::move(cfm.cate);
		cate.lo.assign(frame.n, 0.0);
		cate.hi.assign(frame.n, 0.0);
		if (cfm.draws > 1 && cfm.cate_se.size() == frame.n) {
			for (idx_t i = 0; i < frame.n; i++) {
				cate.lo[i] = cate.cate[i] - Z95 * cfm.cate_se[i];
				cate.hi[i] = cate.cate[i] + Z95 * cfm.cate_se[i];
			}
			cate.interval_method = StringUtil::Format("context ensemble, %llu draws, pointwise 95%%",
			                                          static_cast<unsigned long long>(cfm.draws));
		} else {
			// One draw funds no interval, and reporting a fabricated one would be
			// worse than reporting none.
			for (idx_t i = 0; i < frame.n; i++) {
				cate.lo[i] = cate.hi[i] = cate.cate[i];
			}
			cate.interval_method = "none";
		}
		cate.learner = model.id;
		for (auto &w : cfm.warnings) {
			cate.warnings.push_back(w);
		}
	}

	names = {"row_id", "id", "treatment", "outcome", "cate", "cate_low", "cate_high", "learner"};
	return_types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::VARCHAR};

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

	// Keys and fallback predicates are read up front, once.
	const idx_t n_groups = groups->RowCount();
	vector<vector<Value>> keys(n_groups);
	vector<string> predicates(n_groups);
	for (idx_t g = 0; g < n_groups; g++) {
		string predicate;
		for (idx_t c = 0; c < by.size(); c++) {
			const Value cell = groups->GetValue(c, g);
			keys[g].push_back(cell.IsNull() ? Value(LogicalType::VARCHAR) : Value(cell.ToString()));
			if (!predicate.empty()) {
				predicate += " AND ";
			}
			// Compare as text and use IS NOT DISTINCT FROM, so a NULL group is a
			// group like any other rather than a silently dropped one.
			predicate += "CAST(" + QuoteIdentifier(by[c]) + " AS VARCHAR) IS NOT DISTINCT FROM ";
			predicate += cell.IsNull() ? string("CAST(NULL AS VARCHAR)") : QuoteLiteral(cell.ToString());
		}
		predicates[g] = predicate;
	}

	// One scan of the relation instead of several per group. Each group used to
	// re-read the whole relation through a text-cast predicate, so the cost grew
	// with the square of the group count: 1000 groups of 200 rows took 9.4 ms a
	// group against 4.2 ms at 100. The relation is now copied once into a private
	// in-memory database, sorted by an integer group id with each group's rows in
	// their original order, so a group is one contiguous range the scan finds from
	// zone maps and every frame sees exactly the rows, in exactly the order, the
	// old predicate gave it. An attached database is visible to the fresh
	// connection each frame is built on; a temporary table would not be. If the
	// copy cannot be made - external access disabled, say - the old path runs.
	struct Scratch {
		ClientContext &context;
		string name;
		bool ok = false;
		explicit Scratch(ClientContext &context_p) : context(context_p) {
		}
		~Scratch() {
			if (ok) {
				try {
					RunQuery(context, "DETACH " + name, "releasing do_ate_by's scratch copy");
				} catch (...) {
				}
			}
		}
	};
	static std::atomic<idx_t> scratch_counter(0);
	Scratch scratch(context);
	scratch.name = "__duckdo_ate_by_" + std::to_string(scratch_counter.fetch_add(1));
	try {
		RunQuery(context, "ATTACH ':memory:' AS " + scratch.name, "attaching do_ate_by's scratch copy");
		scratch.ok = true;
		RunQuery(context,
		         "CREATE TABLE " + scratch.name + ".src AS SELECT * FROM (SELECT *, dense_rank() OVER (ORDER BY " +
		             quoted_by + ") AS __duckdo_group FROM (SELECT *, row_number() OVER () AS __duckdo_row FROM " +
		             rel + ")) ORDER BY __duckdo_group, __duckdo_row",
		         "copying " + spec.relation + " once for do_ate_by");
	} catch (const std::exception &) {
		if (scratch.ok) {
			try {
				RunQuery(context, "DETACH " + scratch.name, "releasing do_ate_by's scratch copy");
			} catch (...) {
			}
		}
		scratch.ok = false;
	}

	vector<vector<Value>> rows(n_groups);
	auto estimate_group = [&](idx_t g) {
		CausalSpec group_spec = spec;
		group_spec.relation = scratch.ok ? "(SELECT * EXCLUDE (__duckdo_group, __duckdo_row) FROM " + scratch.name +
		                                       ".src WHERE __duckdo_group = " + std::to_string(g + 1) + ")"
		                                 : "(SELECT * FROM " + rel + " WHERE " + predicates[g] + ")";
		for (auto &column : by) {
			group_spec.exclude.push_back(column);
		}

		vector<Value> row = keys[g];
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
			row.resize(keys[g].size());
			row.push_back(Value("ATE"));
			row.push_back(Value(group_spec.estimator));
			for (int i = 0; i < 5; i++) {
				row.push_back(Value(LogicalType::DOUBLE));
			}
			row.push_back(Value(LogicalType::BIGINT));
			row.push_back(Value(LogicalType::BIGINT));
			row.push_back(WarningList({string("not estimated: ") + ex.what()}));
		}
		rows[g] = std::move(row);
	};

	// Groups run one after another. Running them in parallel was built and
	// measured, and it bought nothing: 1000 AIPW groups took 0.88 s against 0.93 s
	// on one thread, and 300 t_learner groups 0.40 s against 0.42 s. A group's cost
	// is the queries that build its frame, not the arithmetic. It also had worker
	// threads reading settings through the binding query's client context, which
	// is not made to be shared. The single scan above is where the time went.
	for (idx_t g = 0; g < n_groups; g++) {
		estimate_group(g);
	}

	auto bind = make_uniq<ResultBindData>();
	for (auto &row : rows) {
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
	fn.named_parameters["ensemble"] = LogicalType::BIGINT;
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
		// Only where it is honoured. In the shared list, every function that
		// ignored it would return intervals that look clustered and are not.
		fn.named_parameters["cluster"] = LogicalType::VARCHAR;
		RegisterUnderBothNames(loader, fn, entry.name);
	}

	TableFunction cate("", {LogicalType::VARCHAR}, EmitRows, BindCate, InitGlobal);
	AddCommonNamedParameters(cate);
	cate.named_parameters["cluster"] = LogicalType::VARCHAR;
	RegisterUnderBothNames(loader, cate, "cate");

	TableFunction by("", {LogicalType::VARCHAR}, EmitRows, BindAteBy, InitGlobal);
	AddCommonNamedParameters(by);
	by.named_parameters["by"] = LogicalType::LIST(LogicalType::VARCHAR);
	by.named_parameters["cluster"] = LogicalType::VARCHAR;
	RegisterUnderBothNames(loader, by, "ate_by");
}

} // namespace duckdo
} // namespace duckdb
