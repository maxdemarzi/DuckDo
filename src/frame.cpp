#include "duckdo/frame.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {
namespace duckdo {

const char *EstimandName(Estimand e) {
	switch (e) {
	case Estimand::ATT:
		return "ATT";
	case Estimand::ATC:
		return "ATC";
	default:
		return "ATE";
	}
}

vector<idx_t> CausalFrame::AllRows() const {
	vector<idx_t> rows(n);
	for (idx_t i = 0; i < n; i++) {
		rows[i] = i;
	}
	return rows;
}

vector<idx_t> CausalFrame::ArmRows(double arm) const {
	vector<idx_t> rows;
	for (idx_t i = 0; i < n; i++) {
		if (t[i] == arm) {
			rows.push_back(i);
		}
	}
	return rows;
}

// --- SQL text helpers -------------------------------------------------------

string QuoteIdentifier(const string &id) {
	string out = "\"";
	for (char c : id) {
		if (c == '"') {
			out += '"';
		}
		out += c;
	}
	out += "\"";
	return out;
}

string QuoteLiteral(const string &s) {
	string out = "'";
	for (char c : s) {
		if (c == '\'') {
			out += '\'';
		}
		out += c;
	}
	out += "'";
	return out;
}

static bool IsPlainIdentifier(const string &s) {
	if (s.empty()) {
		return false;
	}
	idx_t start = 0;
	idx_t parts = 0;
	for (idx_t i = 0; i <= s.size(); i++) {
		if (i == s.size() || s[i] == '.') {
			if (i == start) {
				return false;
			}
			const char first = s[start];
			if (!(std::isalpha(static_cast<unsigned char>(first)) || first == '_')) {
				return false;
			}
			for (idx_t j = start; j < i; j++) {
				const char c = s[j];
				if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$')) {
					return false;
				}
			}
			parts++;
			start = i + 1;
		}
	}
	return parts >= 1 && parts <= 3;
}

string RelationSql(const string &relation) {
	string trimmed = relation;
	StringUtil::Trim(trimmed);
	if (trimmed.empty()) {
		throw BinderException("duckdo: the relation argument is empty. Pass a table name, or a query in parentheses, "
		                      "e.g. do_ate('customers', ...) or do_ate('(SELECT * FROM customers WHERE year = 2026)', "
		                      "...)");
	}
	if (IsPlainIdentifier(trimmed)) {
		// Quote each dotted part so reserved words and mixed case survive.
		string out;
		idx_t start = 0;
		for (idx_t i = 0; i <= trimmed.size(); i++) {
			if (i == trimmed.size() || trimmed[i] == '.') {
				if (!out.empty()) {
					out += ".";
				}
				out += QuoteIdentifier(trimmed.substr(start, i - start));
				start = i + 1;
			}
		}
		return out;
	}
	const string upper = StringUtil::Upper(trimmed);
	const bool looks_like_query =
	    StringUtil::StartsWith(upper, "SELECT") || StringUtil::StartsWith(upper, "WITH") || trimmed[0] == '(';
	if (!looks_like_query) {
		throw BinderException("duckdo: '%s' is neither a plain table name nor a query. Pass an identifier such as "
		                      "'customers' or 'main.customers', or a full query such as '(SELECT * FROM customers)'",
		                      relation);
	}
	if (trimmed[0] == '(') {
		return trimmed;
	}
	return "(" + trimmed + ")";
}

unique_ptr<MaterializedQueryResult> RunQuery(ClientContext &context, const string &sql, const string &context_msg) {
	// A fresh connection avoids re-entering the client context that is currently
	// binding this table function. It sees committed data only, which is
	// documented behaviour for now.
	Connection con(*context.db);
	auto result = con.Query(sql);
	if (!result || result->HasError()) {
		const string err = result ? result->GetError() : string("query returned no result");
		throw BinderException("duckdo: failed while %s: %s", context_msg, err);
	}
	return result;
}

idx_t GetSettingIdx(ClientContext &context, const char *name, idx_t fallback) {
	Value v;
	if (!context.TryGetCurrentSetting(name, v) || v.IsNull()) {
		return fallback;
	}
	const auto raw = v.GetValue<int64_t>();
	return raw <= 0 ? fallback : static_cast<idx_t>(raw);
}

double GetSettingDouble(ClientContext &context, const char *name, double fallback) {
	Value v;
	if (!context.TryGetCurrentSetting(name, v) || v.IsNull()) {
		return fallback;
	}
	return v.GetValue<double>();
}

string GetSettingString(ClientContext &context, const char *name, const string &fallback) {
	Value v;
	if (!context.TryGetCurrentSetting(name, v) || v.IsNull()) {
		return fallback;
	}
	const auto s = v.ToString();
	return s.empty() ? fallback : s;
}

// --- spec parsing -----------------------------------------------------------

static string RequireString(const named_parameter_map_t &named, const char *key, const char *fn) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		throw BinderException("duckdo: %s requires the named parameter %s := '<column>'", fn, key);
	}
	return entry->second.ToString();
}

static string OptionalString(const named_parameter_map_t &named, const char *key, const string &fallback) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		return fallback;
	}
	return entry->second.ToString();
}

static vector<string> OptionalStringList(const named_parameter_map_t &named, const char *key) {
	vector<string> out;
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		return out;
	}
	for (auto &child : ListValue::GetChildren(entry->second)) {
		if (!child.IsNull()) {
			out.push_back(child.ToString());
		}
	}
	return out;
}

static double OptionalDouble(const named_parameter_map_t &named, const char *key, double fallback) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		return fallback;
	}
	return entry->second.GetValue<double>();
}

static idx_t OptionalIdx(const named_parameter_map_t &named, const char *key, idx_t fallback) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		return fallback;
	}
	const auto raw = entry->second.GetValue<int64_t>();
	return raw <= 0 ? fallback : static_cast<idx_t>(raw);
}

CausalSpec CausalSpec::Parse(ClientContext &context, const vector<Value> &inputs,
                             const named_parameter_map_t &named) {
	CausalSpec spec;
	if (inputs.empty() || inputs[0].IsNull()) {
		throw BinderException("duckdo: the first argument must be a table name or a query, e.g. do_ate('customers', "
		                      "treatment := 't', outcome := 'y')");
	}
	spec.relation = inputs[0].ToString();
	spec.treatment = RequireString(named, "treatment", "this function");
	spec.outcome = OptionalString(named, "outcome", "");
	spec.covariates = OptionalStringList(named, "covariates");
	spec.exclude = OptionalStringList(named, "exclude");
	spec.estimator = OptionalString(named, "estimator", GetSettingString(context, "duckdo_default_estimator", "aipw"));
	spec.model = OptionalString(named, "model", "");
	spec.graph = OptionalString(named, "graph", "");
	spec.policy = OptionalString(named, "policy", "");
	spec.refute_method = OptionalString(named, "method", "placebo_treatment");
	spec.id_column = OptionalString(named, "id", "");
	spec.policy_column = OptionalString(named, "policy", "");
	spec.grid = OptionalIdx(named, "grid", 20);
	spec.ensemble = OptionalIdx(named, "ensemble", GetSettingIdx(context, "duckdo_ensemble_draws", 1));
	spec.threshold = OptionalDouble(named, "threshold", 0.0);
	spec.depth = OptionalIdx(named, "depth", 2);
	spec.treated_label = OptionalString(named, "treated", "");
	spec.control_label = OptionalString(named, "control", "");
	spec.seed = static_cast<int64_t>(OptionalIdx(named, "seed", GetSettingIdx(context, "duckdo_seed", 42)));
	spec.folds = OptionalIdx(named, "folds", 5);
	spec.bootstrap_reps =
	    OptionalIdx(named, "bootstrap_reps", GetSettingIdx(context, "duckdo_bootstrap_reps", 200));
	spec.trim = OptionalDouble(named, "trim", 0.01);
	spec.fraction = OptionalDouble(named, "fraction", 0.8);
	spec.confounder_strength = OptionalDouble(named, "strength", 0.5);

	if (spec.folds < 2) {
		spec.folds = 2;
	}
	if (spec.trim < 0.0 || spec.trim >= 0.5) {
		throw BinderException("duckdo: trim must be in [0, 0.5); got %f", spec.trim);
	}
	// The model id is checked against the catalog by the caller, which has the
	// registry in scope; here we only reject the obviously empty case.
	return spec;
}

// --- frame construction -----------------------------------------------------

namespace {

struct ColumnPlan {
	string source;
	string sql;
	bool categorical = false;
	//! Sorted distinct levels; levels[0] is the reference level and gets no column.
	vector<string> levels;
};

idx_t FindColumn(const vector<string> &names, const string &needle) {
	for (idx_t i = 0; i < names.size(); i++) {
		if (StringUtil::CIEquals(names[i], needle)) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

string ColumnList(const vector<string> &names) {
	string out;
	for (idx_t i = 0; i < names.size(); i++) {
		if (i) {
			out += ", ";
		}
		out += names[i];
	}
	return out;
}

[[noreturn]] void MissingColumn(const string &role, const string &col, const string &relation,
                                const vector<string> &available) {
	throw BinderException("duckdo: %s column '%s' does not exist in %s. Available columns: %s", role, col, relation,
	                      ColumnList(available));
}

double Median(vector<double> values) {
	if (values.empty()) {
		return 0.0;
	}
	const size_t mid = values.size() / 2;
	std::nth_element(values.begin(), values.begin() + mid, values.end());
	return values[mid];
}

} // namespace

CausalFrame BuildFrame(ClientContext &context, const CausalSpec &spec) {
	// One place to pick up the thread budget for the dense accumulations that
	// dominate every fit. 0 means one per hardware thread.
	SetNumericThreads(GetSettingIdx(context, "duckdo_threads", 0));

	CausalFrame frame;
	const string rel = RelationSql(spec.relation);
	const idx_t max_rows = GetSettingIdx(context, "duckdo_max_rows", 100000);
	const idx_t max_features = GetSettingIdx(context, "duckdo_max_features", 500);
	const idx_t max_levels = GetSettingIdx(context, "duckdo_max_categorical_levels", 32);

	// 1. Probe the schema so every name check happens before we read any data.
	auto schema = RunQuery(context, "SELECT * FROM " + rel + " LIMIT 0", "reading the schema of " + spec.relation);
	const vector<string> names = schema->names;
	const vector<LogicalType> types = schema->types;

	const idx_t t_idx = FindColumn(names, spec.treatment);
	if (t_idx == DConstants::INVALID_INDEX) {
		MissingColumn("treatment", spec.treatment, spec.relation, names);
	}
	idx_t y_idx = DConstants::INVALID_INDEX;
	if (!spec.outcome.empty()) {
		y_idx = FindColumn(names, spec.outcome);
		if (y_idx == DConstants::INVALID_INDEX) {
			MissingColumn("outcome", spec.outcome, spec.relation, names);
		}
	}

	idx_t id_idx = DConstants::INVALID_INDEX;
	if (!spec.id_column.empty()) {
		id_idx = FindColumn(names, spec.id_column);
		if (id_idx == DConstants::INVALID_INDEX) {
			MissingColumn("id", spec.id_column, spec.relation, names);
		}
		frame.has_id = true;
	}

	idx_t policy_idx = DConstants::INVALID_INDEX;
	if (!spec.policy_column.empty()) {
		policy_idx = FindColumn(names, spec.policy_column);
		if (policy_idx == DConstants::INVALID_INDEX) {
			MissingColumn("policy", spec.policy_column, spec.relation, names);
		}
		frame.has_policy = true;
	}

	// 2. Resolve the covariate set.
	vector<string> covariates;
	if (!spec.covariates.empty()) {
		for (auto &c : spec.covariates) {
			const idx_t idx = FindColumn(names, c);
			if (idx == DConstants::INVALID_INDEX) {
				MissingColumn("covariate", c, spec.relation, names);
			}
			if (idx == t_idx || idx == y_idx) {
				throw BinderException("duckdo: '%s' is listed as a covariate but it is also the treatment or the "
				                      "outcome. Remove it from covariates",
				                      c);
			}
			covariates.push_back(names[idx]);
		}
	} else {
		for (idx_t i = 0; i < names.size(); i++) {
			if (i == t_idx || i == y_idx || i == id_idx || i == policy_idx) {
				continue;
			}
			bool excluded = false;
			for (auto &e : spec.exclude) {
				if (StringUtil::CIEquals(names[i], e)) {
					excluded = true;
					break;
				}
			}
			if (!excluded) {
				covariates.push_back(names[i]);
			}
		}
	}

	const string t_quoted = QuoteIdentifier(names[t_idx]);

	// 3. Map the treatment. A continuous dose is passed through on its own
	//    scale; everything else is mapped onto {0, 1}.
	string t_expr;
	if (spec.continuous_treatment) {
		if (!types[t_idx].IsNumeric()) {
			throw BinderException("duckdo: a continuous treatment must be numeric; '%s' has type %s", spec.treatment,
			                      types[t_idx].ToString());
		}
		frame.continuous_treatment = true;
		t_expr = "CAST(" + t_quoted + " AS DOUBLE)";
		frame.control_label = "dose";
		frame.treated_label = "dose";
	} else if (types[t_idx].id() == LogicalTypeId::BOOLEAN) {
		t_expr = "CASE WHEN " + t_quoted + " THEN 1.0 ELSE 0.0 END";
		frame.control_label = "false";
		frame.treated_label = "true";
	} else if (types[t_idx].IsNumeric()) {
		auto probe = RunQuery(context,
		                      "SELECT COUNT(DISTINCT " + t_quoted + "), CAST(MIN(" + t_quoted +
		                          ") AS DOUBLE), CAST(MAX(" + t_quoted + ") AS DOUBLE) FROM " + rel + " WHERE " +
		                          t_quoted + " IS NOT NULL",
		                      "inspecting treatment column " + spec.treatment);
		const auto distinct = probe->GetValue(0, 0).GetValue<int64_t>();
		if (distinct != 2) {
			throw BinderException("duckdo: treatment '%s' has %lld distinct non-NULL values, so this function cannot "
			                      "use it. For a dose, use do_ape() or do_dose_response(); otherwise derive a binary "
			                      "column, or pass treated := / control := to pick two levels",
			                      spec.treatment, static_cast<long long>(distinct));
		}
		const double lo = probe->GetValue(1, 0).GetValue<double>();
		const double hi = probe->GetValue(2, 0).GetValue<double>();
		frame.control_label = StringUtil::Format("%g", lo);
		frame.treated_label = StringUtil::Format("%g", hi);
		t_expr = "CASE WHEN CAST(" + t_quoted + " AS DOUBLE) = " + StringUtil::Format("%.17g", hi) +
		         " THEN 1.0 ELSE 0.0 END";
	} else {
		auto probe = RunQuery(context,
		                      "SELECT DISTINCT CAST(" + t_quoted + " AS VARCHAR) AS v FROM " + rel + " WHERE " +
		                          t_quoted + " IS NOT NULL ORDER BY 1 LIMIT 3",
		                      "inspecting treatment column " + spec.treatment);
		if (probe->RowCount() != 2) {
			throw BinderException("duckdo: treatment '%s' has %lld distinct non-NULL values; DuckDo v0 supports binary "
			                      "treatments only. Pass treated := / control := to pick two levels",
			                      spec.treatment, static_cast<long long>(probe->RowCount()));
		}
		frame.control_label = probe->GetValue(0, 0).ToString();
		frame.treated_label = probe->GetValue(0, 1).ToString();
		t_expr = "CASE WHEN CAST(" + t_quoted + " AS VARCHAR) = " + QuoteLiteral(frame.treated_label) + " THEN 1.0 ELSE 0.0 END";
	}
	if (!spec.treated_label.empty() || !spec.control_label.empty()) {
		if (spec.treated_label.empty() || spec.control_label.empty()) {
			throw BinderException("duckdo: treated := and control := must be given together");
		}
		frame.treated_label = spec.treated_label;
		frame.control_label = spec.control_label;
		t_expr = "CASE WHEN CAST(" + t_quoted + " AS VARCHAR) = " + QuoteLiteral(frame.treated_label) +
		         " THEN 1.0 WHEN CAST(" + t_quoted + " AS VARCHAR) = " + QuoteLiteral(frame.control_label) +
		         " THEN 0.0 ELSE NULL END";
	}

	// 4. Outcome expression, and whether it is binary.
	string y_expr = "CAST(NULL AS DOUBLE)";
	string y_quoted;
	if (y_idx != DConstants::INVALID_INDEX) {
		y_quoted = QuoteIdentifier(names[y_idx]);
		if (types[y_idx].id() == LogicalTypeId::BOOLEAN) {
			y_expr = "CASE WHEN " + y_quoted + " THEN 1.0 ELSE 0.0 END";
			frame.binary_outcome = true;
		} else if (types[y_idx].IsNumeric()) {
			y_expr = "CAST(" + y_quoted + " AS DOUBLE)";
			auto probe = RunQuery(context,
			                      "SELECT COUNT(DISTINCT " + y_quoted + "), CAST(MIN(" + y_quoted +
			                          ") AS DOUBLE), CAST(MAX(" + y_quoted + ") AS DOUBLE) FROM " + rel + " WHERE " +
			                          y_quoted + " IS NOT NULL",
			                      "inspecting outcome column " + spec.outcome);
			const auto distinct = probe->GetValue(0, 0).GetValue<int64_t>();
			if (distinct == 2 && probe->GetValue(1, 0).GetValue<double>() == 0.0 &&
			    probe->GetValue(2, 0).GetValue<double>() == 1.0) {
				frame.binary_outcome = true;
			}
		} else {
			throw BinderException("duckdo: outcome '%s' has type %s. Outcomes must be numeric or boolean", spec.outcome,
			                      types[y_idx].ToString());
		}
	}

	// 5. Plan the covariate encoding, probing levels for categorical columns.
	vector<ColumnPlan> plans;
	for (auto &col : covariates) {
		const idx_t idx = FindColumn(names, col);
		const LogicalType &type = types[idx];
		const string q = QuoteIdentifier(col);
		ColumnPlan plan;
		plan.source = col;
		if (type.id() == LogicalTypeId::BOOLEAN) {
			plan.sql = "CASE WHEN " + q + " IS NULL THEN NULL WHEN " + q + " THEN 1.0 ELSE 0.0 END";
		} else if (type.IsNumeric()) {
			plan.sql = "CAST(" + q + " AS DOUBLE)";
		} else if (type.id() == LogicalTypeId::DATE || type.id() == LogicalTypeId::TIMESTAMP ||
		           type.id() == LogicalTypeId::TIMESTAMP_TZ || type.id() == LogicalTypeId::TIMESTAMP_SEC ||
		           type.id() == LogicalTypeId::TIMESTAMP_MS || type.id() == LogicalTypeId::TIMESTAMP_NS) {
			plan.sql = "CAST(epoch(CAST(" + q + " AS TIMESTAMP)) AS DOUBLE)";
		} else if (type.id() == LogicalTypeId::VARCHAR || type.id() == LogicalTypeId::ENUM) {
			auto probe = RunQuery(context,
			                      "SELECT DISTINCT CAST(" + q + " AS VARCHAR) AS v FROM " + rel + " WHERE " + q +
			                          " IS NOT NULL ORDER BY 1 LIMIT " + std::to_string(max_levels + 1),
			                      "inspecting covariate " + col);
			if (probe->RowCount() > max_levels) {
				frame.warnings.push_back("covariate '" + col + "' has more than " + std::to_string(max_levels) +
				                         " levels and was dropped; raise duckdo_max_categorical_levels or bucket it "
				                         "first");
				continue;
			}
			if (probe->RowCount() < 2) {
				frame.warnings.push_back("covariate '" + col + "' is constant and was dropped");
				continue;
			}
			plan.categorical = true;
			for (idx_t r = 0; r < probe->RowCount(); r++) {
				plan.levels.push_back(probe->GetValue(0, r).ToString());
			}
			plan.sql = "CAST(" + q + " AS VARCHAR)";
		} else {
			frame.warnings.push_back("covariate '" + col + "' has unsupported type " + type.ToString() +
			                         " and was dropped; flatten or cast it first");
			continue;
		}
		plans.push_back(std::move(plan));
	}

	// 6. Project exactly what the encoder needs, with a stable row order.
	// Every numeric expression is cast explicitly: a bare `1.0` literal is
	// DECIMAL in DuckDB, and the scan below reads flat DOUBLE vectors.
	string projection = "CAST(" + t_expr + " AS DOUBLE) AS __duckdo_t, CAST(" + y_expr + " AS DOUBLE) AS __duckdo_y";
	if (frame.has_id) {
		projection += ", CAST(" + QuoteIdentifier(names[id_idx]) + " AS VARCHAR) AS __duckdo_id";
	}
	if (frame.has_policy) {
		projection += ", coalesce(CAST(" + QuoteIdentifier(names[policy_idx]) + " AS BOOLEAN), false) AS __duckdo_policy";
	}
	const idx_t cov_base = 2 + (frame.has_id ? 1 : 0) + (frame.has_policy ? 1 : 0);
	const idx_t policy_col = frame.has_id ? 3 : 2;
	for (idx_t i = 0; i < plans.size(); i++) {
		projection += ", ";
		projection += plans[i].categorical ? plans[i].sql : ("CAST(" + plans[i].sql + " AS DOUBLE)");
		projection += " AS __duckdo_c" + std::to_string(i);
	}
	string where = t_quoted + " IS NOT NULL";
	if (!y_quoted.empty()) {
		where += " AND " + y_quoted + " IS NOT NULL";
	}
	const string data_sql = "SELECT " + projection + " FROM " + rel + " WHERE " + where + " LIMIT " +
	                        std::to_string(max_rows + 1);
	auto data = RunQuery(context, data_sql, "reading data from " + spec.relation);
	if (data->RowCount() > max_rows) {
		throw BinderException("duckdo: %s has more than %llu usable rows. Raise duckdo_max_rows, or pass a sampled "
		                      "query such as '(SELECT * FROM %s USING SAMPLE %llu ROWS)'",
		                      spec.relation, static_cast<unsigned long long>(max_rows), spec.relation,
		                      static_cast<unsigned long long>(max_rows));
	}
	frame.n = data->RowCount();
	if (frame.n < 8) {
		throw BinderException("duckdo: only %llu rows survived after dropping NULL treatment/outcome values. Causal "
		                      "estimation needs meaningfully more data than that",
		                      static_cast<unsigned long long>(frame.n));
	}

	// 7. Scan into raw per-column buffers.
	const idx_t ncols = plans.size();
	vector<vector<double>> raw_num(ncols);
	vector<vector<int32_t>> raw_cat(ncols);
	vector<vector<char>> raw_null(ncols);
	for (idx_t c = 0; c < ncols; c++) {
		raw_null[c].reserve(frame.n);
		if (plans[c].categorical) {
			raw_cat[c].reserve(frame.n);
		} else {
			raw_num[c].reserve(frame.n);
		}
	}
	frame.t.reserve(frame.n);
	frame.y.reserve(frame.n);
	frame.source_row.reserve(frame.n);
	if (frame.has_id) {
		frame.ids.reserve(frame.n);
	}
	if (frame.has_policy) {
		frame.policy.reserve(frame.n);
	}

	ColumnDataScanState scan_state;
	auto &collection = data->Collection();
	collection.InitializeScan(scan_state);
	DataChunk chunk;
	collection.InitializeScanChunk(chunk);
	idx_t emitted = 0;
	while (collection.Scan(scan_state, chunk)) {
		chunk.Flatten();
		const idx_t count = chunk.size();
		const auto *t_data = FlatVector::GetData<double>(chunk.data[0]);
		const auto &t_valid = FlatVector::Validity(chunk.data[0]);
		const auto *y_data = FlatVector::GetData<double>(chunk.data[1]);
		for (idx_t i = 0; i < count; i++) {
			if (!t_valid.RowIsValid(i)) {
				// Rows outside the two treatment levels, when treated:=/control:= narrowed them.
				continue;
			}
			frame.t.push_back(t_data[i]);
			frame.y.push_back(y_data[i]);
			frame.source_row.push_back(emitted + i);
			if (frame.has_id) {
				auto &id_vec = chunk.data[2];
				if (FlatVector::Validity(id_vec).RowIsValid(i)) {
					frame.ids.push_back(FlatVector::GetData<string_t>(id_vec)[i].GetString());
				} else {
					frame.ids.push_back(string());
				}
			}
			if (frame.has_policy) {
				auto &pol_vec = chunk.data[policy_col];
				const bool valid = FlatVector::Validity(pol_vec).RowIsValid(i);
				frame.policy.push_back((valid && FlatVector::GetData<bool>(pol_vec)[i]) ? 1 : 0);
			}
			for (idx_t c = 0; c < ncols; c++) {
				auto &vec = chunk.data[c + cov_base];
				const auto &valid = FlatVector::Validity(vec);
				const bool is_null = !valid.RowIsValid(i);
				raw_null[c].push_back(is_null ? 1 : 0);
				if (plans[c].categorical) {
					int32_t level = -1;
					if (!is_null) {
						const auto *strings = FlatVector::GetData<string_t>(vec);
						const string value = strings[i].GetString();
						for (idx_t l = 0; l < plans[c].levels.size(); l++) {
							if (plans[c].levels[l] == value) {
								level = static_cast<int32_t>(l);
								break;
							}
						}
					}
					raw_cat[c].push_back(level);
				} else {
					const auto *doubles = FlatVector::GetData<double>(vec);
					const double value = is_null ? 0.0 : doubles[i];
					raw_num[c].push_back(std::isfinite(value) ? value : 0.0);
					if (!is_null && !std::isfinite(value)) {
						raw_null[c].back() = 1;
					}
				}
			}
		}
		emitted += count;
	}
	frame.n = frame.t.size();

	// 8. Encode. One feature per numeric column, one per non-reference level of
	//    a categorical column, plus a missingness indicator wherever NULLs
	//    actually occurred - rows are never dropped for missing covariates.
	struct PendingFeature {
		FeatureInfo info;
		vector<double> values;
	};
	vector<PendingFeature> pending;

	for (idx_t c = 0; c < ncols; c++) {
		auto &plan = plans[c];
		bool any_null = false;
		for (auto flag : raw_null[c]) {
			if (flag) {
				any_null = true;
				break;
			}
		}
		if (plan.categorical) {
			for (idx_t l = 1; l < plan.levels.size(); l++) {
				PendingFeature pf;
				pf.info.source = plan.source;
				pf.info.name = plan.source + "=" + plan.levels[l];
				pf.info.kind = FeatureKind::ONE_HOT;
				pf.info.level = plan.levels[l];
				pf.values.resize(frame.n);
				for (idx_t i = 0; i < frame.n; i++) {
					pf.values[i] = (raw_cat[c][i] == static_cast<int32_t>(l)) ? 1.0 : 0.0;
				}
				pending.push_back(std::move(pf));
			}
		} else {
			vector<double> observed;
			observed.reserve(frame.n);
			for (idx_t i = 0; i < frame.n; i++) {
				if (!raw_null[c][i]) {
					observed.push_back(raw_num[c][i]);
				}
			}
			if (observed.empty()) {
				frame.warnings.push_back("covariate '" + plan.source + "' is entirely NULL and was dropped");
				continue;
			}
			const double fill = Median(observed);
			PendingFeature pf;
			pf.info.source = plan.source;
			pf.info.name = plan.source;
			pf.info.kind = FeatureKind::NUMERIC;
			pf.values.resize(frame.n);
			for (idx_t i = 0; i < frame.n; i++) {
				pf.values[i] = raw_null[c][i] ? fill : raw_num[c][i];
			}
			pending.push_back(std::move(pf));
		}
		if (any_null) {
			PendingFeature pf;
			pf.info.source = plan.source;
			pf.info.name = plan.source + "__missing";
			pf.info.kind = FeatureKind::MISSING_INDICATOR;
			pf.values.resize(frame.n);
			for (idx_t i = 0; i < frame.n; i++) {
				pf.values[i] = raw_null[c][i] ? 1.0 : 0.0;
			}
			pending.push_back(std::move(pf));
		}
		if (any_null) {
			for (idx_t i = 0; i < frame.n; i++) {
				if (raw_null[c][i]) {
					frame.n_rows_with_missing++;
					break;
				}
			}
		}
		frame.covariate_columns.push_back(plan.source);
	}

	// Count rows with any missing covariate, rather than columns with any NULL.
	frame.n_rows_with_missing = 0;
	for (idx_t i = 0; i < frame.n; i++) {
		for (idx_t c = 0; c < ncols; c++) {
			if (!raw_null[c].empty() && raw_null[c][i]) {
				frame.n_rows_with_missing++;
				break;
			}
		}
	}

	// Drop constant features and standardise the rest, so IRLS stays conditioned.
	vector<PendingFeature> kept;
	for (auto &pf : pending) {
		const double mean = Mean(pf.values);
		const double sd = StdDev(pf.values);
		if (!(sd > 1e-12)) {
			frame.warnings.push_back("feature '" + pf.info.name + "' is constant and was dropped");
			continue;
		}
		pf.info.center = mean;
		pf.info.scale = sd;
		for (auto &v : pf.values) {
			v = (v - mean) / sd;
		}
		kept.push_back(std::move(pf));
	}
	if (kept.size() > max_features) {
		throw BinderException("duckdo: encoding produced %llu features, above duckdo_max_features (%llu). Pass a "
		                      "shorter covariates := list, or raise the setting",
		                      static_cast<unsigned long long>(kept.size()),
		                      static_cast<unsigned long long>(max_features));
	}

	frame.X.Resize(frame.n, kept.size());
	for (idx_t j = 0; j < kept.size(); j++) {
		frame.features.push_back(kept[j].info);
		for (idx_t i = 0; i < frame.n; i++) {
			frame.X.At(i, j) = kept[j].values[i];
		}
	}

	if (frame.continuous_treatment) {
		frame.dose_sorted = frame.t;
		std::sort(frame.dose_sorted.begin(), frame.dose_sorted.end());
		if (!(StdDev(frame.t) > 1e-12)) {
			throw BinderException("duckdo: the treatment '%s' does not vary, so no dose-response is estimable",
			                      spec.treatment);
		}
		return frame;
	}

	for (idx_t i = 0; i < frame.n; i++) {
		if (frame.t[i] == 1.0) {
			frame.n_treated++;
		}
	}
	if (frame.n_treated == 0 || frame.n_treated == frame.n) {
		throw BinderException("duckdo: every row has the same treatment value, so no effect is estimable. Check that "
		                      "'%s' actually varies in %s",
		                      spec.treatment, spec.relation);
	}
	const idx_t smaller = std::min(frame.n_treated, frame.n - frame.n_treated);
	if (smaller < 5) {
		throw BinderException("duckdo: the smaller treatment arm has only %llu rows. That is not enough to estimate an "
		                      "effect",
		                      static_cast<unsigned long long>(smaller));
	}
	if (smaller < 30) {
		frame.warnings.push_back("the smaller treatment arm has only " + std::to_string(smaller) +
		                         " rows; treat the interval as optimistic");
	}
	return frame;
}

} // namespace duckdo
} // namespace duckdb
