//===----------------------------------------------------------------------===//
// Phase 10: causal discovery - proposing a graph from data, and saying out
// loud how little that proposal is worth on its own.
//
// PC-stable (Spirtes, Glymour and Scheines; Colombo and Maathuis 2014) with
// Fisher-z partial-correlation tests, then v-structures and Meek's rules. The
// result is a CPDAG: some edges oriented, some not, because observational data
// identifies a graph only up to its Markov equivalence class. The "stable"
// variant freezes each variable's neighbours at the start of every level, which
// makes the skeleton independent of column order - the same property the rest
// of DuckDo keeps for row order.
//
// The roadmap put this feature last and set two conditions: loud uncertainty
// and a required review step. The uncertainty is a bootstrap - PC rerun on
// resamples, each edge reported with how often it appeared and how often in
// this direction, including edges the full-data graph left out. The review step
// is enforced by do_graph_create, which refuses do_discover_dot's proposal
// until its marker line is deleted, and refuses its undirected edges until a
// person picks a direction for each.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"
#include "duckdo/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

namespace duckdb {
namespace duckdo {

namespace {

struct DiscoverGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<DiscoverGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<DiscoverGlobalState>();
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

// --- inputs -------------------------------------------------------------------

struct DiscoverSpec {
	string relation;
	vector<string> columns;
	vector<string> exclude;
	double alpha = 0.01;
	idx_t max_conditioning = 3;
	idx_t bootstrap = 50;
	int64_t seed = 42;
};

vector<string> ListParameter(const named_parameter_map_t &named, const char *key) {
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

DiscoverSpec ParseDiscover(ClientContext &context, TableFunctionBindInput &input, const char *fn) {
	DiscoverSpec spec;
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("duckdo: %s takes a table name or a query as its first argument", fn);
	}
	spec.relation = input.inputs[0].ToString();
	auto &named = input.named_parameters;
	spec.columns = ListParameter(named, "columns");
	spec.exclude = ListParameter(named, "exclude");
	auto entry = named.find("alpha");
	if (entry != named.end() && !entry->second.IsNull()) {
		spec.alpha = entry->second.GetValue<double>();
	}
	if (!(spec.alpha > 0.0 && spec.alpha < 1.0)) {
		throw BinderException("duckdo: alpha must be in (0, 1); got %f", spec.alpha);
	}
	entry = named.find("max_conditioning");
	if (entry != named.end() && !entry->second.IsNull()) {
		const auto value = entry->second.GetValue<int64_t>();
		if (value < 0) {
			throw BinderException("duckdo: max_conditioning must be non-negative");
		}
		spec.max_conditioning = static_cast<idx_t>(value);
	}
	entry = named.find("bootstrap");
	if (entry != named.end() && !entry->second.IsNull()) {
		const auto value = entry->second.GetValue<int64_t>();
		if (value < 0) {
			throw BinderException("duckdo: bootstrap must be non-negative");
		}
		spec.bootstrap = static_cast<idx_t>(value);
	}
	spec.seed = static_cast<int64_t>(GetSettingIdx(context, "duckdo_seed", 42));
	entry = named.find("seed");
	if (entry != named.end() && !entry->second.IsNull()) {
		spec.seed = entry->second.GetValue<int64_t>();
	}
	return spec;
}

struct NumericTable {
	vector<string> names;
	idx_t n = 0;
	idx_t p = 0;
	vector<double> data; // n x p, row-major
	idx_t dropped = 0;
};

NumericTable LoadNumeric(ClientContext &context, const DiscoverSpec &spec, const char *fn) {
	const string rel = RelationSql(spec.relation);
	auto probe = RunQuery(context, "SELECT * FROM " + rel + " LIMIT 0", string(fn) + " inspecting " + spec.relation);
	auto usable = [&](idx_t k) {
		const auto &type = probe->types[k];
		return type.IsNumeric() || type.id() == LogicalTypeId::BOOLEAN;
	};
	vector<string> chosen;
	if (!spec.columns.empty()) {
		for (auto &column : spec.columns) {
			idx_t found = DConstants::INVALID_INDEX;
			for (idx_t k = 0; k < probe->names.size(); k++) {
				if (StringUtil::CIEquals(probe->names[k], column)) {
					found = k;
				}
			}
			if (found == DConstants::INVALID_INDEX) {
				throw BinderException("duckdo: column '%s' is not in %s", column, spec.relation);
			}
			if (!usable(found)) {
				throw BinderException("duckdo: column '%s' has type %s; %s works on numeric columns", column,
				                      probe->types[found].ToString(), fn);
			}
			chosen.push_back(probe->names[found]);
		}
	} else {
		for (idx_t k = 0; k < probe->names.size(); k++) {
			bool excluded = false;
			for (auto &e : spec.exclude) {
				excluded = excluded || StringUtil::CIEquals(probe->names[k], e);
			}
			if (!excluded && usable(k)) {
				chosen.push_back(probe->names[k]);
			}
		}
	}
	if (chosen.size() < 2) {
		throw BinderException("duckdo: %s needs at least two numeric columns; found %llu", fn,
		                      static_cast<unsigned long long>(chosen.size()));
	}
	// The number of conditional-independence tests grows combinatorially with
	// the variable count, and so does the chance that some test errs.
	if (chosen.size() > 30) {
		throw BinderException("duckdo: %s over %llu variables would run too many independence tests for the "
		                      "result to mean anything; choose at most 30 with columns := [...]",
		                      fn, static_cast<unsigned long long>(chosen.size()));
	}

	string select;
	for (auto &column : chosen) {
		select += (select.empty() ? "" : ", ") + string("CAST(") + QuoteIdentifier(column) + " AS DOUBLE)";
	}
	auto result = RunQuery(context, "SELECT " + select + " FROM " + rel, string(fn) + " reading " + spec.relation);
	NumericTable table;
	table.names = chosen;
	table.p = chosen.size();
	vector<double> row(table.p);
	for (idx_t r = 0; r < result->RowCount(); r++) {
		bool complete = true;
		for (idx_t j = 0; j < table.p && complete; j++) {
			const Value v = result->GetValue(j, r);
			if (v.IsNull()) {
				complete = false;
			} else {
				row[j] = v.GetValue<double>();
			}
		}
		if (!complete) {
			table.dropped++;
			continue;
		}
		table.data.insert(table.data.end(), row.begin(), row.end());
	}
	table.n = table.data.size() / table.p;
	if (table.n < table.p + 10) {
		throw BinderException("duckdo: %s needs comfortably more complete rows than variables; %llu rows for %llu "
		                      "variables",
		                      fn, static_cast<unsigned long long>(table.n), static_cast<unsigned long long>(table.p));
	}
	for (idx_t j = 0; j < table.p; j++) {
		double lo = table.data[j], hi = lo;
		for (idx_t r = 0; r < table.n; r++) {
			lo = std::min(lo, table.data[r * table.p + j]);
			hi = std::max(hi, table.data[r * table.p + j]);
		}
		if (!(hi > lo)) {
			throw BinderException("duckdo: column '%s' is constant, so it cannot depend on anything", table.names[j]);
		}
	}
	return table;
}

//! Rows in content order, so the bootstrap follows the data rather than where
//! the rows happen to sit - the same guarantee the causal frame gives.
vector<idx_t> ContentOrder(const NumericTable &t) {
	vector<idx_t> order(t.n);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) {
		for (idx_t j = 0; j < t.p; j++) {
			const double x = t.data[a * t.p + j], y = t.data[b * t.p + j];
			if (x != y) {
				return x < y;
			}
		}
		return a < b;
	});
	return order;
}

vector<double> Correlation(const NumericTable &t, const vector<idx_t> &rows) {
	const idx_t p = t.p;
	const double m = static_cast<double>(rows.size());
	vector<double> mean(p, 0.0);
	for (auto r : rows) {
		for (idx_t j = 0; j < p; j++) {
			mean[j] += t.data[r * p + j];
		}
	}
	for (auto &value : mean) {
		value /= m;
	}
	vector<double> cov(p * p, 0.0);
	for (auto r : rows) {
		for (idx_t a = 0; a < p; a++) {
			const double da = t.data[r * p + a] - mean[a];
			for (idx_t b = a; b < p; b++) {
				cov[a * p + b] += da * (t.data[r * p + b] - mean[b]);
			}
		}
	}
	vector<double> C(p * p, 0.0);
	for (idx_t a = 0; a < p; a++) {
		C[a * p + a] = 1.0;
		for (idx_t b = a + 1; b < p; b++) {
			const double denom = std::sqrt(std::max(cov[a * p + a] * cov[b * p + b], 1e-300));
			C[a * p + b] = C[b * p + a] = cov[a * p + b] / denom;
		}
	}
	return C;
}

// --- the tests ---------------------------------------------------------------

//! Partial correlation of i and j given S, from the inverse of the correlation
//! submatrix over {i, j} and S: -P_ij / sqrt(P_ii P_jj).
double PartialCorrelation(const vector<double> &C, idx_t p, idx_t i, idx_t j, const vector<idx_t> &S) {
	if (S.empty()) {
		return C[i * p + j];
	}
	vector<idx_t> index = {i, j};
	index.insert(index.end(), S.begin(), S.end());
	const idx_t k = index.size();
	vector<double> M(k * k);
	for (idx_t a = 0; a < k; a++) {
		for (idx_t b = 0; b < k; b++) {
			M[a * k + b] = C[index[a] * p + index[b]];
		}
	}
	for (int attempt = 0; attempt < 2; attempt++) {
		vector<double> A = M, e(k, 0.0), first, second;
		e[0] = 1.0;
		const bool ok_first = CholeskySolve(A, k, e, first);
		A = M;
		e.assign(k, 0.0);
		e[1] = 1.0;
		const bool ok_second = CholeskySolve(A, k, e, second);
		if (ok_first && ok_second && first[0] > 0.0 && second[1] > 0.0) {
			const double r = -first[1] / std::sqrt(first[0] * second[1]);
			return std::max(-0.9999999, std::min(0.9999999, r));
		}
		// A near-collinear conditioning set: nudge the diagonal and try once more.
		for (idx_t a = 0; a < k; a++) {
			M[a * k + a] += 1e-8;
		}
	}
	return 0.0;
}

//! Fisher's z test of zero partial correlation.
double IndependencePValue(double r, double n, idx_t conditioning) {
	const double dof = n - static_cast<double>(conditioning) - 3.0;
	if (dof <= 0.0) {
		return 1.0;
	}
	const double z = 0.5 * std::log((1.0 + r) / (1.0 - r)) * std::sqrt(dof);
	return NormalTwoSidedP(z);
}

// --- PC-stable -----------------------------------------------------------------

struct Cpdag {
	idx_t p = 0;
	vector<uint8_t> adj;  // symmetric
	vector<uint8_t> head; // head[i*p+j]: an arrowhead at j on the edge i - j, i.e. i -> j
	idx_t conflicts = 0;

	bool Adjacent(idx_t i, idx_t j) const {
		return adj[i * p + j] != 0;
	}
	bool Directed(idx_t i, idx_t j) const {
		return adj[i * p + j] && head[i * p + j] && !head[j * p + i];
	}
	bool Undirected(idx_t i, idx_t j) const {
		return adj[i * p + j] && !head[i * p + j] && !head[j * p + i];
	}
	void Orient(idx_t i, idx_t j) {
		head[i * p + j] = 1;
		head[j * p + i] = 0;
	}
};

Cpdag RunPc(const vector<double> &C, idx_t p, double n, double alpha, idx_t max_conditioning) {
	Cpdag g;
	g.p = p;
	g.adj.assign(p * p, 0);
	g.head.assign(p * p, 0);
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = 0; j < p; j++) {
			g.adj[i * p + j] = i != j ? 1 : 0;
		}
	}
	std::map<std::pair<idx_t, idx_t>, vector<idx_t>> sepset;

	for (idx_t level = 0; level <= max_conditioning; level++) {
		// Stable: neighbours are frozen for the whole level, so which edge is
		// tested first cannot change which conditioning sets are available.
		vector<vector<idx_t>> frozen(p);
		for (idx_t i = 0; i < p; i++) {
			for (idx_t j = 0; j < p; j++) {
				if (g.adj[i * p + j]) {
					frozen[i].push_back(j);
				}
			}
		}
		bool testable = false;
		for (idx_t i = 0; i < p; i++) {
			for (idx_t j = 0; j < p; j++) {
				if (i == j || !g.adj[i * p + j]) {
					continue;
				}
				vector<idx_t> candidates;
				for (auto k : frozen[i]) {
					if (k != j) {
						candidates.push_back(k);
					}
				}
				if (candidates.size() < level) {
					continue;
				}
				testable = true;
				vector<idx_t> pick(level);
				std::iota(pick.begin(), pick.end(), 0);
				while (true) {
					vector<idx_t> S(level);
					for (idx_t a = 0; a < level; a++) {
						S[a] = candidates[pick[a]];
					}
					if (IndependencePValue(PartialCorrelation(C, p, i, j, S), n, level) > alpha) {
						g.adj[i * p + j] = g.adj[j * p + i] = 0;
						sepset[{std::min(i, j), std::max(i, j)}] = S;
						break;
					}
					idx_t a = level;
					while (a > 0 && pick[a - 1] == candidates.size() - level + a - 1) {
						a--;
					}
					if (a == 0) {
						break;
					}
					pick[a - 1]++;
					for (idx_t b = a; b < level; b++) {
						pick[b] = pick[b - 1] + 1;
					}
				}
			}
		}
		if (!testable) {
			break;
		}
	}

	// V-structures: i - k - j with i and j not adjacent and k outside the set
	// that separated them. This is the only place the data orients anything;
	// everything after is propagation.
	for (idx_t k = 0; k < p; k++) {
		for (idx_t i = 0; i < p; i++) {
			for (idx_t j = i + 1; j < p; j++) {
				if (i == k || j == k || !g.adj[i * p + k] || !g.adj[j * p + k] || g.adj[i * p + j]) {
					continue;
				}
				auto found = sepset.find({i, j});
				if (found != sepset.end() &&
				    std::find(found->second.begin(), found->second.end(), k) != found->second.end()) {
					continue;
				}
				if (g.head[k * p + i] || g.head[k * p + j]) {
					g.conflicts++;
					continue;
				}
				g.Orient(i, k);
				g.Orient(j, k);
			}
		}
	}

	// Meek's rules, to a fixed point: orient what any other orientation would
	// either create a new v-structure or a cycle.
	bool changed = true;
	while (changed) {
		changed = false;
		for (idx_t a = 0; a < p; a++) {
			for (idx_t b = 0; b < p; b++) {
				if (a == b || !g.Undirected(a, b)) {
					continue;
				}
				bool orient = false;
				for (idx_t c = 0; c < p && !orient; c++) { // R1: c -> a - b, c and b apart
					orient = c != b && g.Directed(c, a) && !g.adj[c * p + b];
				}
				for (idx_t c = 0; c < p && !orient; c++) { // R2: a -> c -> b
					orient = g.Directed(a, c) && g.Directed(c, b);
				}
				for (idx_t c = 0; c < p && !orient; c++) { // R3: a - c -> b, a - d -> b, c and d apart
					for (idx_t d = c + 1; d < p && !orient; d++) {
						orient = c != b && d != b && g.Undirected(a, c) && g.Undirected(a, d) && g.Directed(c, b) &&
						         g.Directed(d, b) && !g.adj[c * p + d];
					}
				}
				if (orient) {
					g.Orient(a, b);
					changed = true;
				}
			}
		}
	}
	return g;
}

// --- one run: point graph plus bootstrap ------------------------------------------

struct Discovery {
	DiscoverSpec spec;
	NumericTable table;
	Cpdag graph;
	vector<double> adjacent;   // fraction of resamples with i and j adjacent
	vector<double> oriented;   // fraction with i -> j
	vector<double> undirected; // fraction with i - j left unoriented
	vector<string> warnings;
};

Discovery RunDiscovery(ClientContext &context, TableFunctionBindInput &input, const char *fn) {
	Discovery out;
	out.spec = ParseDiscover(context, input, fn);
	out.table = LoadNumeric(context, out.spec, fn);
	const auto &t = out.table;
	const idx_t p = t.p;
	const double n = static_cast<double>(t.n);

	vector<idx_t> all(t.n);
	std::iota(all.begin(), all.end(), 0);
	out.graph = RunPc(Correlation(t, all), p, n, out.spec.alpha, out.spec.max_conditioning);

	out.adjacent.assign(p * p, 0.0);
	out.oriented.assign(p * p, 0.0);
	out.undirected.assign(p * p, 0.0);
	const idx_t reps = out.spec.bootstrap;
	if (reps > 0) {
		const auto order = ContentOrder(t);
		vector<Cpdag> draws(reps);
		// Each replicate is seeded from (seed, rep), so the result does not depend
		// on which thread reaches which replicate.
		ParallelJobs(reps, [&](idx_t rep) {
			std::mt19937_64 rng(static_cast<uint64_t>(out.spec.seed) ^ 0xD15C0DE5ULL ^ (rep * 0x9E3779B97F4A7C15ULL));
			std::uniform_int_distribution<idx_t> pick(0, t.n - 1);
			vector<idx_t> rows(t.n);
			for (auto &r : rows) {
				r = order[pick(rng)];
			}
			draws[rep] = RunPc(Correlation(t, rows), p, n, out.spec.alpha, out.spec.max_conditioning);
		});
		for (auto &g : draws) {
			for (idx_t i = 0; i < p; i++) {
				for (idx_t j = 0; j < p; j++) {
					if (i == j) {
						continue;
					}
					out.adjacent[i * p + j] += g.Adjacent(i, j) ? 1.0 : 0.0;
					out.oriented[i * p + j] += g.Directed(i, j) ? 1.0 : 0.0;
					out.undirected[i * p + j] += g.Undirected(i, j) ? 1.0 : 0.0;
				}
			}
		}
		for (idx_t k = 0; k < p * p; k++) {
			out.adjacent[k] /= static_cast<double>(reps);
			out.oriented[k] /= static_cast<double>(reps);
			out.undirected[k] /= static_cast<double>(reps);
		}
	}

	out.warnings.push_back(StringUtil::Format(
	    "PC-stable with Fisher-z tests at alpha %g on %llu complete rows. It assumes no hidden common cause of any "
	    "two variables, faithfulness, and linear-Gaussian dependence. Real data usually breaks the first, and then "
	    "orientations are wrong even where every edge is right",
	    out.spec.alpha, static_cast<unsigned long long>(t.n)));
	if (t.dropped > 0) {
		out.warnings.push_back(StringUtil::Format("%llu rows with a NULL in a selected column were dropped",
		                                          static_cast<unsigned long long>(t.dropped)));
	}
	if (t.n < 500) {
		out.warnings.push_back(StringUtil::Format(
		    "with %llu rows the tests have little power, so a missing edge is weak evidence of independence",
		    static_cast<unsigned long long>(t.n)));
	}
	if (out.graph.conflicts > 0) {
		out.warnings.push_back(StringUtil::Format(
		    "%llu v-structures conflicted and the first found was kept; orientations around them are unreliable",
		    static_cast<unsigned long long>(out.graph.conflicts)));
	}
	if (reps == 0) {
		out.warnings.push_back("bootstrap := 0, so nothing here says how fragile these edges are");
	}
	out.warnings.push_back("this is a proposal, not a graph: do_graph_create refuses do_discover_dot's output until "
	                       "its review marker is deleted and every undirected edge is given a direction");
	return out;
}

Value WarningValue(const vector<string> &warnings) {
	vector<Value> out;
	for (auto &warning : warnings) {
		out.push_back(Value(warning));
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(out));
}

unique_ptr<FunctionData> BindDiscover(ClientContext &context, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names) {
	auto found = RunDiscovery(context, input, "do_discover");
	const auto &t = found.table;
	const auto &g = found.graph;
	const idx_t p = t.p;
	const bool resampled = found.spec.bootstrap > 0;
	names = {"source", "target", "edge", "in_graph", "stability", "orientation_stability", "warnings"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::BOOLEAN,
	                LogicalType::DOUBLE,
	                LogicalType::DOUBLE,
	                LogicalType::LIST(LogicalType::VARCHAR)};
	auto bind = make_uniq<ResultBindData>();
	const Value warnings = WarningValue(found.warnings);
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = i + 1; j < p; j++) {
			const bool in_graph = g.Adjacent(i, j);
			const double stability = found.adjacent[i * p + j];
			// Report every edge in the graph, and every pair the resamples joined
			// often enough to matter even though the full sample did not.
			if (!in_graph && !(resampled && stability >= 0.25)) {
				continue;
			}
			string source = t.names[i], target = t.names[j], edge = "absent";
			double orientation = 0.0;
			if (in_graph && g.Directed(i, j)) {
				edge = "->";
				orientation = found.oriented[i * p + j];
			} else if (in_graph && g.Directed(j, i)) {
				edge = "->";
				std::swap(source, target);
				orientation = found.oriented[j * p + i];
			} else if (in_graph) {
				edge = "--";
				orientation = found.undirected[i * p + j];
			}
			bind->rows.push_back({Value(source), Value(target), Value(edge), Value::BOOLEAN(in_graph),
			                      resampled ? Value::DOUBLE(stability) : Value(LogicalType::DOUBLE),
			                      resampled && in_graph ? Value::DOUBLE(orientation) : Value(LogicalType::DOUBLE),
			                      warnings});
		}
	}
	return std::move(bind);
}

string QuoteNode(const string &name) {
	string cleaned = name;
	std::replace(cleaned.begin(), cleaned.end(), '"', '\'');
	return "\"" + cleaned + "\"";
}

unique_ptr<FunctionData> BindDiscoverDot(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	auto found = RunDiscovery(context, input, "do_discover_dot");
	const auto &t = found.table;
	const auto &g = found.graph;
	const idx_t p = t.p;
	const bool resampled = found.spec.bootstrap > 0;

	string dot = "digraph discovered {\n";
	dot += "  // do_discover: unreviewed - delete this line only after reviewing every edge below.\n";
	dot += StringUtil::Format("  // PC-stable, alpha %g, %llu bootstrap resamples, %llu complete rows. It assumes no\n",
	                          found.spec.alpha, static_cast<unsigned long long>(found.spec.bootstrap),
	                          static_cast<unsigned long long>(t.n));
	dot += "  // hidden common causes, faithfulness and linear-Gaussian dependence. An edge that is wrong\n";
	dot += "  // here becomes a wrong adjustment set in do_identify and do_validate.\n";
	for (idx_t j = 0; j < p; j++) {
		dot += "  " + QuoteNode(t.names[j]) + ";\n";
	}
	idx_t directed = 0, undirected = 0;
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = i + 1; j < p; j++) {
			if (!g.Adjacent(i, j)) {
				continue;
			}
			if (g.Directed(i, j) || g.Directed(j, i)) {
				const idx_t from = g.Directed(i, j) ? i : j, to = g.Directed(i, j) ? j : i;
				dot += "  " + QuoteNode(t.names[from]) + " -> " + QuoteNode(t.names[to]) + ";";
				if (resampled) {
					dot += StringUtil::Format("  // in %.0f%% of resamples, this direction in %.0f%%",
					                          100.0 * found.adjacent[from * p + to],
					                          100.0 * found.oriented[from * p + to]);
				}
				dot += "\n";
				directed++;
			} else {
				dot += "  " + QuoteNode(t.names[i]) + " -- " + QuoteNode(t.names[j]) + ";  // ";
				if (resampled) {
					dot += StringUtil::Format("in %.0f%% of resamples; ", 100.0 * found.adjacent[i * p + j]);
				}
				dot += "direction not identified from data - write -> one way or the other\n";
				undirected++;
			}
		}
	}
	dot += "}\n";

	names = {"dot", "n_nodes", "n_directed", "n_undirected"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value(dot), Value::BIGINT(static_cast<int64_t>(p)),
	                      Value::BIGINT(static_cast<int64_t>(directed)),
	                      Value::BIGINT(static_cast<int64_t>(undirected))});
	return std::move(bind);
}

void AddDiscoverParameters(TableFunction &fn) {
	fn.named_parameters["columns"] = LogicalType::LIST(LogicalType::VARCHAR);
	fn.named_parameters["exclude"] = LogicalType::LIST(LogicalType::VARCHAR);
	fn.named_parameters["alpha"] = LogicalType::DOUBLE;
	fn.named_parameters["max_conditioning"] = LogicalType::BIGINT;
	fn.named_parameters["bootstrap"] = LogicalType::BIGINT;
	fn.named_parameters["seed"] = LogicalType::BIGINT;
}

} // namespace

void RegisterDiscoveryFunctions(ExtensionLoader &loader) {
	TableFunction discover("", {LogicalType::VARCHAR}, EmitRows, BindDiscover, InitGlobal);
	AddDiscoverParameters(discover);
	RegisterUnderBothNames(loader, discover, "discover");

	TableFunction dot("", {LogicalType::VARCHAR}, EmitRows, BindDiscoverDot, InitGlobal);
	AddDiscoverParameters(dot);
	RegisterUnderBothNames(loader, dot, "discover_dot");
}

} // namespace duckdo
} // namespace duckdb
