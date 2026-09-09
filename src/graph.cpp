//===----------------------------------------------------------------------===//
// Phase 4: causal graphs, identification and covariate validation.
//
// The question this answers is "what should I even adjust for?" - the one users
// get wrong most often. d-separation is the primitive; backdoor, front-door,
// instrument search and covariate grading are all built on top of it.
//===----------------------------------------------------------------------===//
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdo/frame.hpp"
#include "duckdo/functions.hpp"

#include <algorithm>
#include <mutex>
#include <set>
#include <unordered_map>

namespace duckdb {
namespace duckdo {

namespace {

struct Dag {
	vector<string> nodes;
	vector<vector<idx_t>> children;
	vector<vector<idx_t>> parents;
	vector<uint8_t> latent;

	idx_t IndexOf(const string &name) const {
		for (idx_t i = 0; i < nodes.size(); i++) {
			if (StringUtil::CIEquals(nodes[i], name)) {
				return i;
			}
		}
		return DConstants::INVALID_INDEX;
	}

	idx_t Add(const string &name) {
		const idx_t existing = IndexOf(name);
		if (existing != DConstants::INVALID_INDEX) {
			return existing;
		}
		nodes.push_back(name);
		children.emplace_back();
		parents.emplace_back();
		latent.push_back(0);
		return nodes.size() - 1;
	}

	void AddEdge(idx_t from, idx_t to) {
		auto &kids = children[from];
		if (std::find(kids.begin(), kids.end(), to) == kids.end()) {
			kids.push_back(to);
			parents[to].push_back(from);
		}
	}

	idx_t EdgeCount() const {
		idx_t total = 0;
		for (auto &kids : children) {
			total += kids.size();
		}
		return total;
	}
};

// --- DOT parsing ------------------------------------------------------------

//! Tokenise a DOT-ish description into identifiers, arrows and punctuation.
vector<string> Tokenize(const string &text) {
	vector<string> tokens;
	idx_t i = 0;
	while (i < text.size()) {
		const char c = text[i];
		if (std::isspace(static_cast<unsigned char>(c))) {
			i++;
		} else if (c == '-' && i + 1 < text.size() && text[i + 1] == '>') {
			tokens.push_back("->");
			i += 2;
		} else if (c == '{' || c == '}' || c == ';' || c == ',' || c == '[' || c == ']' || c == '=') {
			tokens.push_back(string(1, c));
			i++;
		} else if (c == '"') {
			idx_t j = i + 1;
			string value;
			while (j < text.size() && text[j] != '"') {
				value += text[j++];
			}
			tokens.push_back(value);
			i = j + 1;
		} else if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.') {
			idx_t j = i;
			string value;
			while (j < text.size() &&
			       (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '_' || text[j] == '.')) {
				value += text[j++];
			}
			tokens.push_back(value);
			i = j;
		} else {
			i++;
		}
	}
	return tokens;
}

bool IsKeyword(const string &token) {
	return StringUtil::CIEquals(token, "digraph") || StringUtil::CIEquals(token, "graph") ||
	       StringUtil::CIEquals(token, "strict") || StringUtil::CIEquals(token, "node") ||
	       StringUtil::CIEquals(token, "edge");
}

Dag ParseDot(const string &text) {
	Dag dag;
	auto tokens = Tokenize(text);
	idx_t i = 0;
	// Skip an optional `digraph name {` header.
	while (i < tokens.size() && tokens[i] != "{") {
		i++;
	}
	i = (i < tokens.size()) ? i + 1 : 0;

	string pending;
	bool have_pending = false;
	while (i < tokens.size()) {
		const string &tok = tokens[i];
		if (tok == "}" ) {
			break;
		}
		if (tok == ";" || tok == ",") {
			have_pending = false;
			i++;
			continue;
		}
		if (tok == "[") {
			// Attribute block: only `latent` / `unobserved` carry meaning for us.
			bool mark_latent = false;
			while (i < tokens.size() && tokens[i] != "]") {
				if (StringUtil::CIEquals(tokens[i], "latent") || StringUtil::CIEquals(tokens[i], "unobserved") ||
				    StringUtil::CIEquals(tokens[i], "true")) {
					mark_latent = true;
				}
				i++;
			}
			i++;
			if (mark_latent && have_pending) {
				dag.latent[dag.Add(pending)] = 1;
			}
			continue;
		}
		if (tok == "->") {
			if (!have_pending || i + 1 >= tokens.size()) {
				throw BinderException("duckdo: malformed edge in graph definition near '->'");
			}
			const string target = tokens[i + 1];
			if (IsKeyword(target)) {
				throw BinderException("duckdo: malformed edge in graph definition near '->'");
			}
			const idx_t from = dag.Add(pending);
			const idx_t to = dag.Add(target);
			dag.AddEdge(from, to);
			pending = target;
			have_pending = true;
			i += 2;
			continue;
		}
		if (IsKeyword(tok)) {
			i++;
			continue;
		}
		pending = tok;
		have_pending = true;
		dag.Add(pending);
		i++;
	}
	if (dag.nodes.empty()) {
		throw BinderException("duckdo: the graph definition contains no nodes. Expected something like "
		                      "'digraph { a -> b; b -> c; }'");
	}

	// Reject cycles: a causal DAG that is not acyclic is not a causal DAG.
	vector<idx_t> indegree(dag.nodes.size(), 0);
	for (idx_t n = 0; n < dag.nodes.size(); n++) {
		indegree[n] = dag.parents[n].size();
	}
	vector<idx_t> queue;
	for (idx_t n = 0; n < dag.nodes.size(); n++) {
		if (indegree[n] == 0) {
			queue.push_back(n);
		}
	}
	idx_t visited = 0;
	while (!queue.empty()) {
		const idx_t n = queue.back();
		queue.pop_back();
		visited++;
		for (auto kid : dag.children[n]) {
			if (--indegree[kid] == 0) {
				queue.push_back(kid);
			}
		}
	}
	if (visited != dag.nodes.size()) {
		throw BinderException("duckdo: the graph contains a cycle. Causal graphs must be acyclic");
	}
	return dag;
}

// --- graph algorithms -------------------------------------------------------

std::set<idx_t> Ancestors(const Dag &dag, const std::set<idx_t> &seeds) {
	std::set<idx_t> seen;
	vector<idx_t> stack(seeds.begin(), seeds.end());
	while (!stack.empty()) {
		const idx_t n = stack.back();
		stack.pop_back();
		for (auto p : dag.parents[n]) {
			if (seen.insert(p).second) {
				stack.push_back(p);
			}
		}
	}
	return seen;
}

std::set<idx_t> Descendants(const Dag &dag, idx_t seed) {
	std::set<idx_t> seen;
	vector<idx_t> stack {seed};
	while (!stack.empty()) {
		const idx_t n = stack.back();
		stack.pop_back();
		for (auto kid : dag.children[n]) {
			if (seen.insert(kid).second) {
				stack.push_back(kid);
			}
		}
	}
	return seen;
}

//! d-separation by the ancestral-moral-graph construction: restrict to the
//! ancestral set of X u Y u Z, marry the parents of every common child, drop
//! directions, remove Z, then ask whether X can still reach Y.
bool DSeparated(const Dag &dag, const std::set<idx_t> &x, const std::set<idx_t> &y, const std::set<idx_t> &z) {
	std::set<idx_t> seeds;
	seeds.insert(x.begin(), x.end());
	seeds.insert(y.begin(), y.end());
	seeds.insert(z.begin(), z.end());
	std::set<idx_t> keep = Ancestors(dag, seeds);
	keep.insert(seeds.begin(), seeds.end());

	const idx_t n = dag.nodes.size();
	vector<std::set<idx_t>> adjacency(n);
	for (auto v : keep) {
		// Directed edges become undirected.
		for (auto kid : dag.children[v]) {
			if (keep.count(kid)) {
				adjacency[v].insert(kid);
				adjacency[kid].insert(v);
			}
		}
		// Moralisation: parents of a common child become adjacent.
		vector<idx_t> ps;
		for (auto p : dag.parents[v]) {
			if (keep.count(p)) {
				ps.push_back(p);
			}
		}
		for (idx_t a = 0; a < ps.size(); a++) {
			for (idx_t b = a + 1; b < ps.size(); b++) {
				adjacency[ps[a]].insert(ps[b]);
				adjacency[ps[b]].insert(ps[a]);
			}
		}
	}

	vector<uint8_t> blocked(n, 0);
	for (auto v : z) {
		blocked[v] = 1;
	}
	vector<uint8_t> seen(n, 0);
	vector<idx_t> stack;
	for (auto v : x) {
		if (blocked[v]) {
			continue;
		}
		seen[v] = 1;
		stack.push_back(v);
	}
	while (!stack.empty()) {
		const idx_t v = stack.back();
		stack.pop_back();
		if (y.count(v)) {
			return false;
		}
		for (auto nb : adjacency[v]) {
			if (!seen[nb] && !blocked[nb]) {
				seen[nb] = 1;
				stack.push_back(nb);
			}
		}
	}
	return true;
}

//! A copy of the graph with every edge out of `node` deleted - the graph the
//! backdoor criterion is checked in.
Dag WithoutOutgoing(const Dag &dag, idx_t node) {
	Dag copy = dag;
	for (auto kid : dag.children[node]) {
		auto &ps = copy.parents[kid];
		ps.erase(std::remove(ps.begin(), ps.end(), node), ps.end());
	}
	copy.children[node].clear();
	return copy;
}

bool SatisfiesBackdoor(const Dag &dag, idx_t t, idx_t y, const std::set<idx_t> &z) {
	const auto desc = Descendants(dag, t);
	for (auto v : z) {
		if (v == t || v == y || desc.count(v)) {
			return false;
		}
	}
	const Dag cut = WithoutOutgoing(dag, t);
	return DSeparated(cut, {t}, {y}, z);
}

//! Candidate adjustment variables: observed non-descendants of the treatment
//! that are ancestors of the treatment or the outcome.
std::set<idx_t> BackdoorCandidates(const Dag &dag, idx_t t, idx_t y) {
	const auto anc = Ancestors(dag, {t, y});
	const auto desc = Descendants(dag, t);
	std::set<idx_t> candidates;
	for (auto v : anc) {
		if (v == t || v == y || desc.count(v) || dag.latent[v]) {
			continue;
		}
		candidates.insert(v);
	}
	return candidates;
}

//! Minimal (not necessarily minimum) valid backdoor set, found by greedy removal.
bool FindBackdoorSet(const Dag &dag, idx_t t, idx_t y, std::set<idx_t> &out) {
	auto candidates = BackdoorCandidates(dag, t, y);
	if (!SatisfiesBackdoor(dag, t, y, candidates)) {
		return false;
	}
	// Deterministic order so the answer is reproducible.
	vector<idx_t> ordered(candidates.begin(), candidates.end());
	for (auto v : ordered) {
		auto reduced = candidates;
		reduced.erase(v);
		if (SatisfiesBackdoor(dag, t, y, reduced)) {
			candidates = std::move(reduced);
		}
	}
	out = std::move(candidates);
	return true;
}

//! Does removing `blockers` cut every directed path from t to y?
bool InterceptsAllDirectedPaths(const Dag &dag, idx_t t, idx_t y, const std::set<idx_t> &blockers) {
	vector<uint8_t> seen(dag.nodes.size(), 0);
	vector<idx_t> stack {t};
	seen[t] = 1;
	while (!stack.empty()) {
		const idx_t v = stack.back();
		stack.pop_back();
		for (auto kid : dag.children[v]) {
			if (blockers.count(kid) || seen[kid]) {
				continue;
			}
			if (kid == y) {
				return false;
			}
			seen[kid] = 1;
			stack.push_back(kid);
		}
	}
	return true;
}

bool FindFrontdoorSet(const Dag &dag, idx_t t, idx_t y, std::set<idx_t> &out) {
	for (idx_t m = 0; m < dag.nodes.size(); m++) {
		if (m == t || m == y || dag.latent[m]) {
			continue;
		}
		const std::set<idx_t> mset {m};
		if (!InterceptsAllDirectedPaths(dag, t, y, mset)) {
			continue;
		}
		// No unblocked backdoor path from the treatment to the mediator.
		if (!DSeparated(WithoutOutgoing(dag, t), {t}, {m}, {})) {
			continue;
		}
		// Every backdoor path from the mediator to the outcome is blocked by T.
		if (!DSeparated(WithoutOutgoing(dag, m), {m}, {y}, {t})) {
			continue;
		}
		out = mset;
		return true;
	}
	return false;
}

bool FindInstrument(const Dag &dag, idx_t t, idx_t y, std::set<idx_t> &out) {
	const auto desc_t = Descendants(dag, t);
	for (idx_t z = 0; z < dag.nodes.size(); z++) {
		if (z == t || z == y || dag.latent[z] || desc_t.count(z)) {
			continue;
		}
		// Relevance: the instrument must actually reach the treatment.
		const auto anc_t = Ancestors(dag, {t});
		if (!anc_t.count(z)) {
			continue;
		}
		// Exclusion: no path to the outcome once the treatment's own effect is cut.
		if (!DSeparated(WithoutOutgoing(dag, t), {z}, {y}, {})) {
			continue;
		}
		out = {z};
		return true;
	}
	return false;
}

// --- registry ---------------------------------------------------------------

struct GraphRegistry {
	std::mutex lock;
	std::unordered_map<string, Dag> graphs;

	static GraphRegistry &Get() {
		static GraphRegistry instance;
		return instance;
	}
};

Dag LookupGraph(const string &name) {
	auto &registry = GraphRegistry::Get();
	std::lock_guard<std::mutex> guard(registry.lock);
	auto entry = registry.graphs.find(StringUtil::Lower(name));
	if (entry == registry.graphs.end()) {
		throw BinderException("duckdo: no graph named '%s'. Register one with CALL do_graph_create('%s', 'digraph { a "
		                      "-> b; }') and list them with SELECT * FROM do_graphs()",
		                      name, name);
	}
	return entry->second;
}

idx_t RequireNode(const Dag &dag, const string &name, const char *role) {
	const idx_t idx = dag.IndexOf(name);
	if (idx == DConstants::INVALID_INDEX) {
		string available;
		for (auto &n : dag.nodes) {
			if (!available.empty()) {
				available += ", ";
			}
			available += n;
		}
		throw BinderException("duckdo: %s '%s' is not a node in the graph. Nodes: %s", role, name, available);
	}
	return idx;
}

Value NodeList(const Dag &dag, const std::set<idx_t> &nodes) {
	if (nodes.empty()) {
		return Value::LIST(LogicalType::VARCHAR, vector<Value>());
	}
	vector<Value> values;
	for (auto n : nodes) {
		values.push_back(Value(dag.nodes[n]));
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(values));
}

// --- table functions --------------------------------------------------------

struct GraphGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> InitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<GraphGlobalState>();
}

void EmitRows(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind = data_p.bind_data->Cast<ResultBindData>();
	auto &state = data_p.global_state->Cast<GraphGlobalState>();
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

unique_ptr<FunctionData> BindGraphCreate(ClientContext &, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.size() < 2 || input.inputs[0].IsNull() || input.inputs[1].IsNull()) {
		throw BinderException("duckdo: do_graph_create takes a name and a graph definition, e.g. "
		                      "CALL do_graph_create('sales_dag', 'digraph { season -> discount; }')");
	}
	const string name = input.inputs[0].ToString();
	auto dag = ParseDot(input.inputs[1].ToString());

	idx_t latent_count = 0;
	for (auto flag : dag.latent) {
		latent_count += flag ? 1 : 0;
	}
	const idx_t node_count = dag.nodes.size();
	const idx_t edge_count = dag.EdgeCount();
	{
		auto &registry = GraphRegistry::Get();
		std::lock_guard<std::mutex> guard(registry.lock);
		registry.graphs[StringUtil::Lower(name)] = std::move(dag);
	}

	names = {"name", "n_nodes", "n_edges", "n_latent"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value(name), Value::BIGINT(static_cast<int64_t>(node_count)),
	                      Value::BIGINT(static_cast<int64_t>(edge_count)),
	                      Value::BIGINT(static_cast<int64_t>(latent_count))});
	return std::move(bind);
}

unique_ptr<FunctionData> BindGraphDrop(ClientContext &, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("duckdo: do_graph_drop takes the name of a registered graph");
	}
	const string name = input.inputs[0].ToString();
	bool dropped;
	{
		auto &registry = GraphRegistry::Get();
		std::lock_guard<std::mutex> guard(registry.lock);
		dropped = registry.graphs.erase(StringUtil::Lower(name)) > 0;
	}
	names = {"name", "dropped"};
	return_types = {LogicalType::VARCHAR, LogicalType::BOOLEAN};
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back({Value(name), Value::BOOLEAN(dropped)});
	return std::move(bind);
}

unique_ptr<FunctionData> BindGraphs(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                    vector<string> &names) {
	names = {"name", "n_nodes", "n_edges", "nodes"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT,
	                LogicalType::LIST(LogicalType::VARCHAR)};
	auto bind = make_uniq<ResultBindData>();
	auto &registry = GraphRegistry::Get();
	std::lock_guard<std::mutex> guard(registry.lock);
	vector<string> keys;
	for (auto &entry : registry.graphs) {
		keys.push_back(entry.first);
	}
	std::sort(keys.begin(), keys.end());
	for (auto &key : keys) {
		auto &dag = registry.graphs[key];
		vector<Value> node_values;
		for (auto &n : dag.nodes) {
			node_values.push_back(Value(n));
		}
		bind->rows.push_back({Value(key), Value::BIGINT(static_cast<int64_t>(dag.nodes.size())),
		                      Value::BIGINT(static_cast<int64_t>(dag.EdgeCount())),
		                      node_values.empty() ? Value::LIST(LogicalType::VARCHAR, vector<Value>())
		                                          : Value::LIST(LogicalType::VARCHAR, std::move(node_values))});
	}
	return std::move(bind);
}

string RequireNamed(const named_parameter_map_t &named, const char *key, const char *fn) {
	auto entry = named.find(key);
	if (entry == named.end() || entry->second.IsNull()) {
		throw BinderException("duckdo: %s requires %s := '<name>'", fn, key);
	}
	return entry->second.ToString();
}

unique_ptr<FunctionData> BindIdentify(ClientContext &, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names) {
	const auto dag = LookupGraph(RequireNamed(input.named_parameters, "graph", "do_identify"));
	const idx_t t = RequireNode(dag, RequireNamed(input.named_parameters, "treatment", "do_identify"), "treatment");
	const idx_t y = RequireNode(dag, RequireNamed(input.named_parameters, "outcome", "do_identify"), "outcome");

	names = {"strategy", "identifiable", "adjustment_set", "note"};
	return_types = {LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::LIST(LogicalType::VARCHAR),
	                LogicalType::VARCHAR};
	auto bind = make_uniq<ResultBindData>();

	std::set<idx_t> backdoor;
	if (FindBackdoorSet(dag, t, y, backdoor)) {
		bind->rows.push_back({Value("backdoor"), Value::BOOLEAN(true), NodeList(dag, backdoor),
		                      backdoor.empty() ? Value("no adjustment needed; the treatment has no open backdoor path")
		                                       : Value("adjust for this set to close every backdoor path")});
	} else {
		// Name the culprit: an unobserved common cause of treatment and outcome.
		string culprit;
		const auto anc_t = Ancestors(dag, {t});
		const auto anc_y = Ancestors(dag, {y});
		for (idx_t v = 0; v < dag.nodes.size(); v++) {
			if (dag.latent[v] && anc_t.count(v) && anc_y.count(v)) {
				culprit = dag.nodes[v];
				break;
			}
		}
		bind->rows.push_back(
		    {Value("backdoor"), Value::BOOLEAN(false), Value::LIST(LogicalType::VARCHAR, vector<Value>()),
		     Value(culprit.empty() ? string("no observed set closes every backdoor path")
		                           : "'" + culprit + "' is an unobserved common cause of the treatment and the "
		                                             "outcome, leaving a backdoor path open")});
	}

	std::set<idx_t> frontdoor;
	const bool has_frontdoor = FindFrontdoorSet(dag, t, y, frontdoor);
	bind->rows.push_back({Value("frontdoor"), Value::BOOLEAN(has_frontdoor),
	                      has_frontdoor ? NodeList(dag, frontdoor) : Value::LIST(LogicalType::VARCHAR, vector<Value>()),
	                      Value(has_frontdoor ? "this mediator intercepts every directed path from treatment to outcome"
	                                          : "no mediator satisfies the front-door criterion")});

	std::set<idx_t> instrument;
	const bool has_iv = FindInstrument(dag, t, y, instrument);
	bind->rows.push_back(
	    {Value("iv"), Value::BOOLEAN(has_iv),
	     has_iv ? NodeList(dag, instrument) : Value::LIST(LogicalType::VARCHAR, vector<Value>()),
	     Value(has_iv ? "candidate instrument; the exclusion restriction is an assumption the data cannot check"
	                  : "no node qualifies as an instrument")});
	return std::move(bind);
}

unique_ptr<FunctionData> BindValidate(ClientContext &, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names) {
	const auto dag = LookupGraph(RequireNamed(input.named_parameters, "graph", "do_validate"));
	const idx_t t = RequireNode(dag, RequireNamed(input.named_parameters, "treatment", "do_validate"), "treatment");
	const idx_t y = RequireNode(dag, RequireNamed(input.named_parameters, "outcome", "do_validate"), "outcome");

	vector<string> covariates;
	auto entry = input.named_parameters.find("covariates");
	if (entry != input.named_parameters.end() && !entry->second.IsNull()) {
		for (auto &child : ListValue::GetChildren(entry->second)) {
			if (!child.IsNull()) {
				covariates.push_back(child.ToString());
			}
		}
	}
	if (covariates.empty()) {
		throw BinderException("duckdo: do_validate requires covariates := ['a', 'b'] - the list you were planning to "
		                      "adjust for");
	}

	names = {"covariate", "role", "verdict", "reason"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	auto bind = make_uniq<ResultBindData>();

	const auto desc_t = Descendants(dag, t);
	const auto desc_y = Descendants(dag, y);
	const auto anc_t = Ancestors(dag, {t});
	const auto anc_y = Ancestors(dag, {y});
	// An instrument reaches the outcome ONLY through the treatment, so the test
	// that separates it from a confounder has to be run in the graph with the
	// treatment's own outgoing edges cut.
	const Dag cut_t = WithoutOutgoing(dag, t);
	const auto anc_y_cut = Ancestors(cut_t, {y});

	std::set<idx_t> chosen;
	for (auto &name : covariates) {
		const idx_t idx = dag.IndexOf(name);
		if (idx != DConstants::INVALID_INDEX) {
			chosen.insert(idx);
		}
	}

	for (auto &name : covariates) {
		const idx_t v = dag.IndexOf(name);
		string role, verdict, reason;
		if (v == DConstants::INVALID_INDEX) {
			role = "unknown";
			verdict = "CHECK";
			reason = "not a node in the graph; either add it or drop it from the covariate list";
		} else if (v == t || v == y) {
			role = (v == t) ? "treatment" : "outcome";
			verdict = "DROP";
			reason = "this is the " + role + " itself";
		} else if (dag.latent[v]) {
			role = "latent";
			verdict = "DROP";
			reason = "declared unobserved, so it cannot be adjusted for";
		} else if (desc_y.count(v)) {
			role = "outcome descendant";
			verdict = "DROP";
			reason = "caused by the outcome; adjusting for it biases the estimate";
		} else if (desc_t.count(v)) {
			if (anc_y.count(v)) {
				role = "mediator";
				verdict = "DROP";
				reason = "on a directed path from treatment to outcome; adjusting removes part of the effect you are "
				         "trying to measure";
			} else {
				role = "post-treatment";
				verdict = "DROP";
				reason = "caused by the treatment; adjusting for it can only introduce bias";
			}
		} else if (anc_t.count(v) && anc_y_cut.count(v)) {
			role = "confounder";
			verdict = "keep";
			reason = "a common cause of treatment and outcome; adjusting closes a backdoor path";
		} else if (anc_t.count(v)) {
			role = "instrument";
			verdict = "DROP";
			reason = "reaches the outcome only through the treatment; adjusting inflates variance without reducing "
			         "bias, and can amplify residual confounding";
		} else if (anc_y_cut.count(v)) {
			role = "precision variable";
			verdict = "keep";
			reason = "predicts the outcome only; adjusting tightens the interval without changing the estimand";
		} else {
			// Operational collider test: does adding this variable open a path
			// that was closed without it?
			auto without = chosen;
			without.erase(v);
			auto with = without;
			with.insert(v);
			const Dag cut = WithoutOutgoing(dag, t);
			const bool closed_without = DSeparated(cut, {t}, {y}, without);
			const bool closed_with = DSeparated(cut, {t}, {y}, with);
			if (closed_without && !closed_with) {
				role = "collider";
				verdict = "DROP";
				reason = "conditioning on it opens a path between treatment and outcome that was otherwise blocked";
			} else {
				role = "unrelated";
				verdict = "optional";
				reason = "neither a confounder nor a collider here; harmless but does no work";
			}
		}
		bind->rows.push_back({Value(name), Value(role), Value(verdict), Value(reason)});
	}
	return std::move(bind);
}

// --- do_dseparated (scalar) -------------------------------------------------

void DSeparatedFunction(DataChunk &args, ExpressionState &, Vector &result) {
	auto count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<bool>(result);
	auto &validity = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		const auto graph_value = args.data[0].GetValue(i);
		const auto x_value = args.data[1].GetValue(i);
		const auto y_value = args.data[2].GetValue(i);
		if (graph_value.IsNull() || x_value.IsNull() || y_value.IsNull()) {
			validity.SetInvalid(i);
			continue;
		}
		const auto dag = LookupGraph(graph_value.ToString());
		const idx_t x = RequireNode(dag, x_value.ToString(), "first node");
		const idx_t y = RequireNode(dag, y_value.ToString(), "second node");
		std::set<idx_t> z;
		if (args.ColumnCount() > 3) {
			const auto z_value = args.data[3].GetValue(i);
			if (!z_value.IsNull()) {
				for (auto &child : ListValue::GetChildren(z_value)) {
					if (!child.IsNull()) {
						z.insert(RequireNode(dag, child.ToString(), "conditioning node"));
					}
				}
			}
		}
		out[i] = DSeparated(dag, {x}, {y}, z);
	}
}

} // namespace

void RegisterGraphFunctions(ExtensionLoader &loader) {
	TableFunction create("", {LogicalType::VARCHAR, LogicalType::VARCHAR}, EmitRows, BindGraphCreate, InitGlobal);
	RegisterUnderBothNames(loader, create, "graph_create");

	TableFunction drop("", {LogicalType::VARCHAR}, EmitRows, BindGraphDrop, InitGlobal);
	RegisterUnderBothNames(loader, drop, "graph_drop");

	TableFunction list("", vector<LogicalType>(), EmitRows, BindGraphs, InitGlobal);
	RegisterUnderBothNames(loader, list, "graphs");

	TableFunction identify("", vector<LogicalType>(), EmitRows, BindIdentify, InitGlobal);
	identify.named_parameters["graph"] = LogicalType::VARCHAR;
	identify.named_parameters["treatment"] = LogicalType::VARCHAR;
	identify.named_parameters["outcome"] = LogicalType::VARCHAR;
	RegisterUnderBothNames(loader, identify, "identify");

	TableFunction validate("", vector<LogicalType>(), EmitRows, BindValidate, InitGlobal);
	validate.named_parameters["graph"] = LogicalType::VARCHAR;
	validate.named_parameters["treatment"] = LogicalType::VARCHAR;
	validate.named_parameters["outcome"] = LogicalType::VARCHAR;
	validate.named_parameters["covariates"] = LogicalType::LIST(LogicalType::VARCHAR);
	RegisterUnderBothNames(loader, validate, "validate");

	ScalarFunctionSet dsep_set("do_dseparated");
	dsep_set.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                    LogicalType::BOOLEAN, DSeparatedFunction));
	dsep_set.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR)},
	    LogicalType::BOOLEAN, DSeparatedFunction));
	loader.RegisterFunction(dsep_set);
	ScalarFunctionSet dsep_full("duckdo_dseparated");
	for (auto &fn : dsep_set.functions) {
		dsep_full.AddFunction(fn);
	}
	loader.RegisterFunction(dsep_full);
}

} // namespace duckdo
} // namespace duckdb
