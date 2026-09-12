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
	//! "pc" assumes no hidden common causes; "fci" allows them; "lingam" reads
	//! direction from non-Gaussian disturbances instead of conditional
	//! independence; "both" runs pc and lingam and reports where they differ.
	string algorithm = "pc";
	//! lingam: the smallest standardised coefficient that counts as an edge.
	double min_effect = 0.05;
	bool min_effect_given = false;
	//! Groups of columns in causal order: nothing in a later group may cause
	//! anything in an earlier one. Columns left out are unconstrained.
	vector<vector<string>> tiers;
	//! lingam: a column splitting the rows into datasets that share one causal
	//! order but need not share coefficients.
	string groups;
	//! "pearson" assumes linear-Gaussian dependence; "rank" only a Gaussian copula;
	//! "mixed" a latent Gaussian copula in which two-valued columns are thresholds.
	string test = "pearson";
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
	entry = named.find("algorithm");
	if (entry != named.end() && !entry->second.IsNull()) {
		spec.algorithm = StringUtil::Lower(entry->second.ToString());
	}
	// 'both' is what 'pc+lingam' was called when it was the only union there was.
	if (spec.algorithm == "both") {
		spec.algorithm = "pc+lingam";
	}
	if (spec.algorithm != "pc" && spec.algorithm != "fci" && spec.algorithm != "lingam" && spec.algorithm != "resit" &&
	    spec.algorithm != "rcd" && spec.algorithm != "pc+lingam" && spec.algorithm != "pc+resit") {
		throw BinderException(
		    "duckdo: algorithm must be 'pc', 'fci', 'lingam', 'resit', 'rcd', 'pc+lingam' or 'pc+resit', not '%s'. "
		    "'fci' and 'rcd' allow hidden common causes and name the pairs that have one; 'pc' assumes there are "
		    "none; 'lingam' orients every edge from non-Gaussian disturbances and 'resit' from an added disturbance "
		    "under an effect that may bend, both under assumptions stronger than PC's; 'pc+lingam' and 'pc+resit' "
		    "run PC alongside one of those and say where they differ. 'both' is accepted as the older name for "
		    "'pc+lingam'",
		    spec.algorithm);
	}
	const bool lingam = spec.algorithm == "lingam" || spec.algorithm == "pc+lingam";
	const bool resit = spec.algorithm == "resit" || spec.algorithm == "pc+resit";
	entry = named.find("min_effect");
	if (entry != named.end() && !entry->second.IsNull()) {
		if (!lingam) {
			throw BinderException("duckdo: min_effect is the smallest coefficient LiNGAM counts as an edge, so it "
			                      "applies to algorithm := 'lingam' or 'pc+lingam', not '%s'. RESIT and RCD have no "
			                      "coefficients to threshold; they decide with a test at alpha",
			                      spec.algorithm);
		}
		spec.min_effect = entry->second.GetValue<double>();
		spec.min_effect_given = true;
		if (!(spec.min_effect >= 0.0 && spec.min_effect < 1.0)) {
			throw BinderException("duckdo: min_effect is a standardised coefficient, so it must be in [0, 1); got %f",
			                      spec.min_effect);
		}
	}
	entry = named.find("tiers");
	if (entry != named.end() && !entry->second.IsNull()) {
		for (auto &group : ListValue::GetChildren(entry->second)) {
			vector<string> names;
			if (!group.IsNull()) {
				for (auto &name : ListValue::GetChildren(group)) {
					if (!name.IsNull()) {
						names.push_back(name.ToString());
					}
				}
			}
			if (names.empty()) {
				throw BinderException("duckdo: every group in tiers must name at least one column; tiers reads as "
				                      "[['age', 'sex'], ['discount'], ['revenue']], earliest group first");
			}
			spec.tiers.push_back(names);
		}
		if (spec.tiers.size() < 2) {
			throw BinderException("duckdo: tiers needs at least two groups to say anything; one group forbids nothing");
		}
	}
	entry = named.find("groups");
	if (entry != named.end() && !entry->second.IsNull()) {
		spec.groups = entry->second.ToString();
		if (spec.algorithm != "lingam") {
			throw BinderException(
			    "duckdo: groups splits the rows into datasets that share one causal order but not their "
			    "coefficients, which is a LiNGAM idea (Shimizu 2012), so it applies to algorithm := 'lingam', not "
			    "'%s'. It is refused with 'pc+lingam' on purpose: PC would run on the pooled rows and know nothing "
			    "of the groups, and pooling is exactly what groups is for avoiding - where the coefficients differ "
			    "in sign the pooled correlation goes to nothing and PC drops the edge",
			    spec.algorithm);
		}
	}
	entry = named.find("test");
	if (entry != named.end() && !entry->second.IsNull()) {
		spec.test = StringUtil::Lower(entry->second.ToString());
	}
	if (spec.test != "pearson" && spec.test != "rank" && spec.test != "mixed" && spec.test != "kernel") {
		throw BinderException("duckdo: test must be 'pearson', 'rank', 'mixed' or 'kernel', not '%s'. 'rank' tests on "
		                      "normal scores, which only assumes that monotone transforms of the variables are jointly "
		                      "Gaussian; 'mixed' also reads each two-valued column as the threshold of a latent "
		                      "Gaussian variable; 'kernel' assumes nothing about the shape of the dependence at all",
		                      spec.test);
	}
	if ((lingam || resit) && (spec.test == "rank" || spec.test == "mixed")) {
		throw BinderException("duckdo: algorithm := '%s' cannot run with test := '%s'. LiNGAM reads direction from "
		                      "the shape of each variable's disturbance, and '%s' replaces every column with scores "
		                      "that are Gaussian by construction - exactly the case LiNGAM cannot read",
		                      spec.algorithm, spec.test, spec.test);
	}
	if (spec.algorithm == "lingam" && spec.test != "pearson") {
		throw BinderException("duckdo: algorithm := 'lingam' runs no independence tests, so test := '%s' would do "
		                      "nothing. It is accepted with algorithm := 'pc+lingam', where PC runs the tests and "
		                      "LiNGAM reads the columns as they are",
		                      spec.test);
	}
	if (spec.algorithm == "rcd" && spec.test != "pearson") {
		throw BinderException("duckdo: algorithm := 'rcd' regresses linearly and tests independence with kernels "
		                      "already, so test := '%s' has nothing to change",
		                      spec.test);
	}
	if (spec.algorithm == "resit" && spec.test != "pearson") {
		throw BinderException("duckdo: algorithm := 'resit' is a kernel method already - it regresses on random "
		                      "Fourier features and prunes with the same test as test := 'kernel' - so test := '%s' "
		                      "has nothing to change",
		                      spec.test);
	}
	return spec;
}

struct NumericTable {
	vector<string> names;
	idx_t n = 0;
	idx_t p = 0;
	vector<double> data; // n x p, row-major
	idx_t dropped = 0;
	//! groups := 'site': which group each row belongs to, and what they are called.
	//! Empty when no group column was given.
	vector<idx_t> group;
	vector<string> group_names;

	idx_t Groups() const {
		return group_names.size();
	}
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
	// The group column labels the rows; it is not one of the variables, so it never
	// joins the graph even when it is numeric and would have been picked up.
	string group_column;
	if (!spec.groups.empty()) {
		for (idx_t k = 0; k < probe->names.size(); k++) {
			if (StringUtil::CIEquals(probe->names[k], spec.groups)) {
				group_column = probe->names[k];
			}
		}
		if (group_column.empty()) {
			throw BinderException("duckdo: groups names column '%s', which is not in %s", spec.groups, spec.relation);
		}
		for (idx_t k = 0; k < chosen.size(); k++) {
			if (StringUtil::CIEquals(chosen[k], group_column)) {
				chosen.erase(chosen.begin() + k);
				break;
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
	if (!group_column.empty()) {
		select += ", CAST(" + QuoteIdentifier(group_column) + " AS VARCHAR)";
	}
	auto result = RunQuery(context, "SELECT " + select + " FROM " + rel, string(fn) + " reading " + spec.relation);
	NumericTable table;
	table.names = chosen;
	table.p = chosen.size();
	std::map<string, idx_t> group_index;
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
		string label;
		if (complete && !group_column.empty()) {
			const Value v = result->GetValue(table.p, r);
			if (v.IsNull()) {
				complete = false;
			} else {
				label = v.ToString();
			}
		}
		if (!complete) {
			table.dropped++;
			continue;
		}
		table.data.insert(table.data.end(), row.begin(), row.end());
		if (!group_column.empty()) {
			auto found = group_index.find(label);
			if (found == group_index.end()) {
				found = group_index.insert(std::make_pair(label, table.group_names.size())).first;
				table.group_names.push_back(label);
			}
			table.group.push_back(found->second);
		}
	}
	table.n = table.data.size() / table.p;
	if (table.n < table.p + 10) {
		throw BinderException("duckdo: %s needs comfortably more complete rows than variables; %llu rows for %llu "
		                      "variables",
		                      fn, static_cast<unsigned long long>(table.n), static_cast<unsigned long long>(table.p));
	}
	if (!group_column.empty()) {
		if (table.Groups() < 2) {
			throw BinderException("duckdo: groups := '%s' found %llu group in %s. Sharing a causal order across "
			                      "datasets needs at least two of them",
			                      group_column, static_cast<unsigned long long>(table.Groups()), spec.relation);
		}
		if (table.Groups() > 50) {
			throw BinderException("duckdo: groups := '%s' has %llu distinct values, which reads more like an "
			                      "identifier than a set of datasets; each group is fitted on its own rows",
			                      group_column, static_cast<unsigned long long>(table.Groups()));
		}
		vector<idx_t> counts(table.Groups(), 0);
		for (auto g : table.group) {
			counts[g]++;
		}
		for (idx_t g = 0; g < counts.size(); g++) {
			if (counts[g] < table.p + 10) {
				throw BinderException("duckdo: group '%s' has %llu rows for %llu variables. Every group is fitted on "
				                      "its own rows, so each needs comfortably more rows than variables",
				                      table.group_names[g], static_cast<unsigned long long>(counts[g]),
				                      static_cast<unsigned long long>(table.p));
			}
		}
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

//! Replace each column by its normal scores, Phi^-1(rank / (n + 1)), with tied
//! values sharing their average rank. The correlation of normal scores estimates
//! the latent correlation whenever monotone transforms of the variables are
//! jointly Gaussian - a Gaussian copula - so the same Fisher-z tests then read
//! conditional independence correctly in skewed, heavy-tailed or log-scale data
//! that linear correlation misreads. Returns the columns too coarse for that.
vector<string> RankNormalScores(NumericTable &t) {
	vector<string> coarse;
	vector<idx_t> order(t.n);
	vector<double> column(t.n);
	for (idx_t j = 0; j < t.p; j++) {
		for (idx_t r = 0; r < t.n; r++) {
			column[r] = t.data[r * t.p + j];
		}
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return column[a] < column[b]; });
		idx_t distinct = 0;
		for (idx_t start = 0; start < t.n;) {
			idx_t end = start + 1;
			while (end < t.n && column[order[end]] == column[order[start]]) {
				end++;
			}
			// The average of the 1-based ranks start + 1 .. end.
			const double rank = 0.5 * static_cast<double>(start + end + 1);
			const double score = NormalQuantile(rank / static_cast<double>(t.n + 1));
			for (idx_t k = start; k < end; k++) {
				t.data[order[k] * t.p + j] = score;
			}
			distinct++;
			start = end;
		}
		if (distinct < 10) {
			coarse.push_back(t.names[j]);
		}
	}
	return coarse;
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

//! Centre and scale in place. A column with no spread is left alone; LoadNumeric
//! has already refused a constant one, but a residual can still collapse.
void Standardise(vector<double> &x) {
	const double m = static_cast<double>(x.size());
	double mean = 0.0;
	for (auto value : x) {
		mean += value;
	}
	mean /= m;
	double ss = 0.0;
	for (auto &value : x) {
		value -= mean;
		ss += value * value;
	}
	const double sd = std::sqrt(ss / m);
	if (sd > 1e-12) {
		for (auto &value : x) {
			value /= sd;
		}
	}
}

// --- the kernel test -------------------------------------------------------------
//
// Every test above reads dependence through a correlation, which is why they all
// miss a bend. Take a true chain x -> y -> z with y = x^2: the correlation of x and
// y is 0.009 while the correlation of |x| and y is 0.92, so 'pearson' finds no edge
// at all, and neither does 'rank', because no monotone transform straightens a
// parabola. Worse, Fisher's z then rejects a true conditional independence through
// a bent middle variable 100% of the time.
//
// RCoT (Strobl, Zhang and Visweswaran, Journal of Causal Inference 2019) tests
// x _||_ y | S with no such assumption. It approximates the kernel test KCIT (Zhang,
// Peters, Janzing and Scholkopf 2011) with random Fourier features: by Bochner's
// theorem, sqrt(2/D) cos(w'x + b) with w drawn Gaussian and b uniform has an inner
// product that estimates the Gaussian kernel, so a few hundred numbers per row stand
// in for an n by n kernel matrix. That is what makes it affordable - KCIT is at least
// quadratic in the rows, this is linear.
//
// The features of x and y are residualised on the features of S, and the statistic is
// the squared cross-covariance of what is left. Its null is a weighted sum of
// chi-squares. The obvious approximation, a gamma matched on two moments, is
// anticonservative in the tail, which is the half that decides edges: it rejected true
// independences 3-10% of the time at a nominal 1%. Liu, Tang and Zhang (2009) matches
// three moments instead, by choosing a non-central chi-square with the right skewness,
// and that holds the level at 0.0-2.5% across every null case measured.

//! Rows the kernel test reads. It is linear in them, but it is run once per candidate
//! conditioning set, and a kernel test does not need the rows a correlation does.
constexpr idx_t kKernelRows = 2000;
//! Features standing in for the kernel on a single column, and on a conditioning set.
//! 25 for the set was not enough: what the features could not absorb of S came back as
//! dependence between x and y, and the level went to 10%.
constexpr idx_t kXyFeatures = 5;
constexpr idx_t kSetFeatures = 100;
//! Rows the median-distance bandwidth heuristic looks at.
constexpr idx_t kBandwidthRows = 500;
//! Conditioning sets whose Gram factor is kept. Each is D by D, so 64 of them is a few
//! megabytes against the m by D features they save rebuilding.
constexpr idx_t kGramCacheSets = 64;

//! P(sum_i lambda_i chi2_1 > stat), for lambda the eigenvalues of the symmetric C, by
//! Liu, Tang and Zhang (2009). Its cumulants are traces of powers of C, so this needs
//! no eigenvalues.
double WeightedChiSquareUpper(double stat, const vector<double> &C, idx_t k) {
	vector<double> C2(k * k, 0.0);
	for (idx_t a = 0; a < k; a++) {
		for (idx_t b = 0; b < k; b++) {
			double sum = 0.0;
			for (idx_t c = 0; c < k; c++) {
				sum += C[a * k + c] * C[c * k + b];
			}
			C2[a * k + b] = sum;
		}
	}
	double c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0;
	for (idx_t a = 0; a < k; a++) {
		c1 += C[a * k + a];
		c2 += C2[a * k + a];
		for (idx_t b = 0; b < k; b++) {
			c3 += C2[a * k + b] * C[b * k + a];
			c4 += C2[a * k + b] * C2[b * k + a];
		}
	}
	if (!(c2 > 0.0) || !(c3 > 0.0)) {
		return 1.0;
	}
	const double s1 = c3 / std::pow(c2, 1.5);
	const double s2 = c4 / (c2 * c2);
	double a, delta, dof;
	if (s1 * s1 > s2) {
		a = 1.0 / (s1 - std::sqrt(s1 * s1 - s2));
		delta = s1 * a * a * a - a * a;
		dof = a * a - 2.0 * delta;
	} else {
		a = 1.0 / s1;
		delta = 0.0;
		dof = 1.0 / (s1 * s1);
	}
	const double q = (stat - c1) / std::sqrt(2.0 * c2) * (std::sqrt(2.0) * a) + dof + delta;
	if (!(q > 0.0) || !(dof > 0.0)) {
		return 1.0;
	}
	if (delta <= 0.0) {
		return UpperGammaQ(dof / 2.0, q / 2.0);
	}
	// A non-central chi-square tail is a Poisson mixture of central ones.
	double total = 0.0, weight = std::exp(-0.5 * delta);
	for (idx_t term = 0; term < 400; term++) {
		total += weight * UpperGammaQ(dof / 2.0 + static_cast<double>(term), q / 2.0);
		weight *= 0.5 * delta / static_cast<double>(term + 1);
		if (weight < 1e-15) {
			break;
		}
	}
	return std::min(1.0, std::max(0.0, total));
}

//! One column's standardised values over the sample, and the random Fourier features
//! that stand in for a kernel on it.
struct KernelTest {
	idx_t m = 0;
	idx_t p = 0;
	//! [j] is column j over the sampled rows, standardised.
	vector<vector<double>> value;
	//! [j] is m x kXyFeatures for column j, already centred.
	vector<vector<double>> feature;
	//! kSetFeatures random directions in p dimensions, and their phases. They are drawn
	//! once from duckdo_seed and reused by every resample, so the stability a bootstrap
	//! reports is the sample's, not the features'.
	vector<double> weight;
	vector<double> phase;
	//! The Cholesky factor of the set features' Gram matrix depends only on which columns
	//! are being conditioned on, and building it is most of the cost of a test. PC asks
	//! about the same conditioning set once per pair that could use it, so a handful of
	//! them kept by hand turns that back into once per set. Only the factor is kept, not
	//! the features themselves: D by D against m by D is 5% of the memory for most of the
	//! saving. One test object per bootstrap replicate, so no two threads share this.
	mutable vector<std::pair<vector<idx_t>, vector<double>>> gram_cache;

	double PValue(idx_t i, idx_t j, const vector<idx_t> &S) const;
	void SetFeatures(const vector<idx_t> &S, vector<double> &fz) const;
	bool Residual(const vector<double> &target, const vector<idx_t> &S, const vector<double> &fz,
	              vector<double> &out) const;
	//! Cholesky factor of the set features' Gram matrix, from the cache or freshly built.
	//! Null when the features are degenerate enough that the factorisation fails.
	const vector<double> *GramFactor(const vector<idx_t> &S, const vector<double> &fz) const;
	//! How dependent a residual still is on the set it was regressed off: the squared
	//! cross-covariance of their features, which is an HSIC in the random-feature basis.
	//! Comparable across candidates, which is all a causal-order search needs.
	double ResidualDependence(const vector<double> &residual, const vector<double> &fz) const;
	double LinearResidualVariance(idx_t target, const vector<idx_t> &S) const;
	double IndependenceP(const vector<double> &u, const vector<double> &v) const;
	void VectorFeatures(const vector<double> &u, vector<double> &out) const;
	void LinearResidual(idx_t target, const vector<idx_t> &S, vector<double> &out) const;
};

//! Cholesky factor L with A = L L', lower triangular, in place over the lower half.
bool CholeskyFactorise(vector<double> &A, idx_t n) {
	for (idx_t i = 0; i < n; i++) {
		for (idx_t j = 0; j <= i; j++) {
			double sum = A[i * n + j];
			for (idx_t k = 0; k < j; k++) {
				sum -= A[i * n + k] * A[j * n + k];
			}
			if (i == j) {
				if (!(sum > 0.0)) {
					return false;
				}
				A[i * n + i] = std::sqrt(sum);
			} else {
				A[i * n + j] = sum / A[j * n + j];
			}
		}
	}
	return true;
}

//! Solve A x = b from the factor CholeskyFactorise left behind.
void CholeskySolveFactored(const vector<double> &L, idx_t n, const vector<double> &b, vector<double> &x) {
	x.assign(n, 0.0);
	for (idx_t i = 0; i < n; i++) {
		double sum = b[i];
		for (idx_t k = 0; k < i; k++) {
			sum -= L[i * n + k] * x[k];
		}
		x[i] = sum / L[i * n + i];
	}
	for (idx_t i = n; i-- > 0;) {
		double sum = x[i];
		for (idx_t k = i + 1; k < n; k++) {
			sum -= L[k * n + i] * x[k];
		}
		x[i] = sum / L[i * n + i];
	}
}

bool SmallInverse(vector<double> M, idx_t k, vector<double> &inv);

//! Median pairwise distance over a strided subsample - the usual kernel bandwidth.
double MedianDistance(const KernelTest &k, const vector<idx_t> &columns) {
	const idx_t m = k.m;
	const idx_t take = std::min(m, kBandwidthRows);
	vector<idx_t> pick(take);
	for (idx_t a = 0; a < take; a++) {
		pick[a] = take == m ? a : a * (m - 1) / (take - 1);
	}
	vector<double> distances;
	distances.reserve(take * (take - 1) / 2);
	for (idx_t a = 0; a < take; a++) {
		for (idx_t b = a + 1; b < take; b++) {
			double sum = 0.0;
			for (auto column : columns) {
				const double d = k.value[column][pick[a]] - k.value[column][pick[b]];
				sum += d * d;
			}
			distances.push_back(sum);
		}
	}
	if (distances.empty()) {
		return 1.0;
	}
	const idx_t middle = distances.size() / 2;
	std::nth_element(distances.begin(), distances.begin() + middle, distances.end());
	const double median = std::sqrt(distances[middle]);
	return median > 1e-12 ? median : 1.0;
}

//! Centre each column of an m x d block in place.
void CentreColumns(vector<double> &block, idx_t m, idx_t d) {
	for (idx_t c = 0; c < d; c++) {
		double mean = 0.0;
		for (idx_t r = 0; r < m; r++) {
			mean += block[r * d + c];
		}
		mean /= static_cast<double>(m);
		for (idx_t r = 0; r < m; r++) {
			block[r * d + c] -= mean;
		}
	}
}

const vector<double> *KernelTest::GramFactor(const vector<idx_t> &S, const vector<double> &fz) const {
	const idx_t D = kSetFeatures;
	for (auto &entry : gram_cache) {
		if (entry.first == S) {
			return &entry.second;
		}
	}
	vector<double> built(D * D, 0.0);
	for (idx_t a = 0; a < D; a++) {
		for (idx_t b = a; b < D; b++) {
			double sum = 0.0;
			for (idx_t r = 0; r < m; r++) {
				sum += fz[r * D + a] * fz[r * D + b];
			}
			built[a * D + b] = built[b * D + a] = sum / static_cast<double>(m);
		}
	}
	double trace = 0.0;
	for (idx_t a = 0; a < D; a++) {
		trace += built[a * D + a];
	}
	for (idx_t a = 0; a < D; a++) {
		built[a * D + a] += 1e-8 * trace / static_cast<double>(D);
	}
	if (!CholeskyFactorise(built, D)) {
		return nullptr;
	}
	if (gram_cache.size() >= kGramCacheSets) {
		gram_cache.erase(gram_cache.begin());
	}
	gram_cache.push_back(std::make_pair(S, built));
	return &gram_cache.back().second;
}

//! The random Fourier features of a set of columns, centred: a stand-in for the kernel on
//! their joint values.
void KernelTest::SetFeatures(const vector<idx_t> &S, vector<double> &fz) const {
	const idx_t D = kSetFeatures;
	const double bandwidth = MedianDistance(*this, S);
	fz.assign(m * D, 0.0);
	for (idx_t r = 0; r < m; r++) {
		for (idx_t c = 0; c < D; c++) {
			fz[r * D + c] = phase[c];
		}
	}
	for (auto column : S) {
		const auto &values = value[column];
		for (idx_t c = 0; c < D; c++) {
			const double w = weight[c * p + column] / bandwidth;
			for (idx_t r = 0; r < m; r++) {
				fz[r * D + c] += w * values[r];
			}
		}
	}
	const double scale = std::sqrt(2.0 / static_cast<double>(D));
	for (auto &entry : fz) {
		entry = scale * std::cos(entry);
	}
	CentreColumns(fz, m, D);
}

//! Residual of `target` after regressing it on the features of a set, standardised. This is
//! kernel ridge regression with the kernel replaced by its random-feature approximation, and
//! it is what turns a causal-order search into a nonlinear one.
bool KernelTest::Residual(const vector<double> &target, const vector<idx_t> &S, const vector<double> &fz,
                          vector<double> &out) const {
	const idx_t D = kSetFeatures;
	const vector<double> *factor = GramFactor(S, fz);
	if (!factor) {
		return false;
	}
	vector<double> rhs(D), solution;
	for (idx_t a = 0; a < D; a++) {
		double sum = 0.0;
		for (idx_t r = 0; r < m; r++) {
			sum += fz[r * D + a] * target[r];
		}
		rhs[a] = sum / static_cast<double>(m);
	}
	CholeskySolveFactored(*factor, D, rhs, solution);
	out.assign(m, 0.0);
	for (idx_t r = 0; r < m; r++) {
		double fitted = 0.0;
		for (idx_t a = 0; a < D; a++) {
			fitted += fz[r * D + a] * solution[a];
		}
		out[r] = target[r] - fitted;
	}
	return true;
}

//! Residual variance of the same target under an ordinary linear fit on the same columns.
//! The target is standardised, so this is 1 - R^2, and comparing it with what the kernel
//! regression leaves says whether the relationship bends at all.
//! Independence of two arbitrary vectors over the sampled rows, by the same random-feature
//! statistic the conditional test uses with nothing to condition on. This is the HSIC that
//! a functional causal model needs: "is what is left of the effect independent of the
//! cause" is the whole question those models turn on.
double KernelTest::IndependenceP(const vector<double> &u, const vector<double> &v) const {
	const idx_t d = kXyFeatures;
	vector<double> fu, fv;
	VectorFeatures(u, fu);
	VectorFeatures(v, fv);
	const idx_t k = d * d;
	vector<double> mean(k, 0.0), W(m * k);
	for (idx_t r = 0; r < m; r++) {
		for (idx_t a = 0; a < d; a++) {
			for (idx_t b = 0; b < d; b++) {
				const double product = fu[r * d + a] * fv[r * d + b];
				W[r * k + a * d + b] = product;
				mean[a * d + b] += product;
			}
		}
	}
	for (auto &entry : mean) {
		entry /= static_cast<double>(m);
	}
	double stat = 0.0;
	for (auto entry : mean) {
		stat += entry * entry;
	}
	stat *= static_cast<double>(m);
	vector<double> C(k * k, 0.0);
	for (idx_t a = 0; a < k; a++) {
		for (idx_t b = a; b < k; b++) {
			double sum = 0.0;
			for (idx_t r = 0; r < m; r++) {
				sum += (W[r * k + a] - mean[a]) * (W[r * k + b] - mean[b]);
			}
			C[a * k + b] = C[b * k + a] = sum / static_cast<double>(m);
		}
	}
	return WeightedChiSquareUpper(stat, C, k);
}

//! Random Fourier features of one arbitrary vector, centred, with its own bandwidth.
void KernelTest::VectorFeatures(const vector<double> &u, vector<double> &out) const {
	const idx_t d = kXyFeatures;
	const idx_t take = std::min(m, kBandwidthRows);
	vector<double> spread;
	spread.reserve(take * (take - 1) / 2);
	for (idx_t a = 0; a < take; a++) {
		for (idx_t b = a + 1; b < take; b++) {
			const idx_t ra = take == m ? a : a * (m - 1) / (take - 1);
			const idx_t rb = take == m ? b : b * (m - 1) / (take - 1);
			const double delta = u[ra] - u[rb];
			spread.push_back(delta * delta);
		}
	}
	double bandwidth = 1.0;
	if (!spread.empty()) {
		const idx_t middle = spread.size() / 2;
		std::nth_element(spread.begin(), spread.begin() + middle, spread.end());
		bandwidth = std::sqrt(spread[middle]);
		if (!(bandwidth > 1e-12)) {
			bandwidth = 1.0;
		}
	}
	out.assign(m * d, 0.0);
	const double scale = std::sqrt(2.0 / static_cast<double>(d));
	for (idx_t c = 0; c < d; c++) {
		const double w = weight[c * p] / bandwidth;
		const double b = phase[c];
		for (idx_t r = 0; r < m; r++) {
			out[r * d + c] = scale * std::cos(w * u[r] + b);
		}
	}
	CentreColumns(out, m, d);
}

//! Residual of a column on a set of other columns under an ordinary linear fit. RCD is a
//! linear method, so its regressions are linear; only its independence tests are not.
void KernelTest::LinearResidual(idx_t target, const vector<idx_t> &S, vector<double> &out) const {
	out = value[target];
	const idx_t k = S.size();
	if (k == 0) {
		return;
	}
	vector<double> M(k * k), rhs(k), inverse;
	for (idx_t a = 0; a < k; a++) {
		double cross = 0.0;
		for (idx_t r = 0; r < m; r++) {
			cross += value[S[a]][r] * value[target][r];
		}
		rhs[a] = cross / static_cast<double>(m);
		for (idx_t b = 0; b < k; b++) {
			double sum = 0.0;
			for (idx_t r = 0; r < m; r++) {
				sum += value[S[a]][r] * value[S[b]][r];
			}
			M[a * k + b] = sum / static_cast<double>(m);
		}
	}
	if (!SmallInverse(M, k, inverse)) {
		return;
	}
	for (idx_t a = 0; a < k; a++) {
		double coefficient = 0.0;
		for (idx_t b = 0; b < k; b++) {
			coefficient += inverse[a * k + b] * rhs[b];
		}
		for (idx_t r = 0; r < m; r++) {
			out[r] -= coefficient * value[S[a]][r];
		}
	}
}

double KernelTest::LinearResidualVariance(idx_t target, const vector<idx_t> &S) const {
	const idx_t k = S.size();
	if (k == 0) {
		return 1.0;
	}
	vector<double> M(k * k), rhs(k), inverse;
	for (idx_t a = 0; a < k; a++) {
		double cross = 0.0;
		for (idx_t r = 0; r < m; r++) {
			cross += value[S[a]][r] * value[target][r];
		}
		rhs[a] = cross / static_cast<double>(m);
		for (idx_t b = 0; b < k; b++) {
			double sum = 0.0;
			for (idx_t r = 0; r < m; r++) {
				sum += value[S[a]][r] * value[S[b]][r];
			}
			M[a * k + b] = sum / static_cast<double>(m);
		}
	}
	if (!SmallInverse(M, k, inverse)) {
		return 1.0;
	}
	double explained = 0.0;
	for (idx_t a = 0; a < k; a++) {
		double coefficient = 0.0;
		for (idx_t b = 0; b < k; b++) {
			coefficient += inverse[a * k + b] * rhs[b];
		}
		explained += coefficient * rhs[a];
	}
	return std::max(1.0 - explained, 1e-12);
}

double KernelTest::ResidualDependence(const vector<double> &residual, const vector<double> &fz) const {
	const idx_t d = kXyFeatures, D = kSetFeatures;
	// The residual's own features. Its bandwidth is its median distance, which for a
	// standardised column is close enough to constant that the scores stay comparable.
	vector<double> fr(m * d, 0.0);
	const idx_t take = std::min(m, kBandwidthRows);
	vector<double> spread;
	spread.reserve(take * (take - 1) / 2);
	for (idx_t a = 0; a < take; a++) {
		for (idx_t b = a + 1; b < take; b++) {
			const idx_t ra = take == m ? a : a * (m - 1) / (take - 1);
			const idx_t rb = take == m ? b : b * (m - 1) / (take - 1);
			const double delta = residual[ra] - residual[rb];
			spread.push_back(delta * delta);
		}
	}
	double bandwidth = 1.0;
	if (!spread.empty()) {
		const idx_t middle = spread.size() / 2;
		std::nth_element(spread.begin(), spread.begin() + middle, spread.end());
		bandwidth = std::sqrt(spread[middle]);
		if (!(bandwidth > 1e-12)) {
			bandwidth = 1.0;
		}
	}
	const double scale = std::sqrt(2.0 / static_cast<double>(d));
	for (idx_t c = 0; c < d; c++) {
		const double w = weight[c * p] / bandwidth;
		const double b = phase[c];
		for (idx_t r = 0; r < m; r++) {
			fr[r * d + c] = scale * std::cos(w * residual[r] + b);
		}
	}
	CentreColumns(fr, m, d);
	double total = 0.0;
	for (idx_t a = 0; a < d; a++) {
		for (idx_t b = 0; b < D; b++) {
			double sum = 0.0;
			for (idx_t r = 0; r < m; r++) {
				sum += fr[r * d + a] * fz[r * D + b];
			}
			sum /= static_cast<double>(m);
			total += sum * sum;
		}
	}
	return total;
}

double KernelTest::PValue(idx_t i, idx_t j, const vector<idx_t> &S) const {
	const idx_t d = kXyFeatures;
	vector<double> fx = feature[i], fy = feature[j];
	if (!S.empty()) {
		const idx_t D = kSetFeatures;
		vector<double> fz;
		SetFeatures(S, fz);

		// Regress the x and y features on the set's features and keep the residual.
		// The nudge on the diagonal is only enough to make the solve safe: a real ridge
		// leaves part of S unabsorbed, and that reads as dependence between x and y. At
		// 1e-6 relative the level went from 0.005 to 0.21.
		const vector<double> *factor = GramFactor(S, fz);
		if (!factor) {
			return 1.0;
		}
		vector<double> rhs(D), solution;
		for (int side = 0; side < 2; side++) {
			vector<double> &f = side == 0 ? fx : fy;
			for (idx_t c = 0; c < d; c++) {
				for (idx_t a = 0; a < D; a++) {
					double sum = 0.0;
					for (idx_t r = 0; r < m; r++) {
						sum += fz[r * D + a] * f[r * d + c];
					}
					rhs[a] = sum / static_cast<double>(m);
				}
				CholeskySolveFactored(*factor, D, rhs, solution);
				for (idx_t r = 0; r < m; r++) {
					double fitted = 0.0;
					for (idx_t a = 0; a < D; a++) {
						fitted += fz[r * D + a] * solution[a];
					}
					f[r * d + c] -= fitted;
				}
			}
		}
		CentreColumns(fx, m, d);
		CentreColumns(fy, m, d);
	}

	// The statistic is m times the squared norm of the mean of the outer products, and
	// its null weights are the eigenvalues of their covariance.
	const idx_t k = d * d;
	vector<double> mean(k, 0.0);
	vector<double> W(m * k);
	for (idx_t r = 0; r < m; r++) {
		for (idx_t a = 0; a < d; a++) {
			for (idx_t b = 0; b < d; b++) {
				const double product = fx[r * d + a] * fy[r * d + b];
				W[r * k + a * d + b] = product;
				mean[a * d + b] += product;
			}
		}
	}
	for (auto &entry : mean) {
		entry /= static_cast<double>(m);
	}
	double stat = 0.0;
	for (auto entry : mean) {
		stat += entry * entry;
	}
	stat *= static_cast<double>(m);
	vector<double> C(k * k, 0.0);
	for (idx_t a = 0; a < k; a++) {
		for (idx_t b = a; b < k; b++) {
			double sum = 0.0;
			for (idx_t r = 0; r < m; r++) {
				sum += (W[r * k + a] - mean[a]) * (W[r * k + b] - mean[b]);
			}
			C[a * k + b] = C[b * k + a] = sum / static_cast<double>(m);
		}
	}
	return WeightedChiSquareUpper(stat, C, k);
}

//! Build the kernel test for one sample of rows. The features of each column are
//! computed once here rather than per test, because they do not depend on what the
//! column is being tested against.
KernelTest MakeKernelTest(const NumericTable &t, const vector<idx_t> &rows, int64_t seed) {
	KernelTest k;
	k.p = t.p;
	const idx_t stride = std::max<idx_t>(1, (rows.size() + kKernelRows - 1) / kKernelRows);
	vector<idx_t> sample;
	for (idx_t r = 0; r < rows.size(); r += stride) {
		sample.push_back(rows[r]);
	}
	k.m = sample.size();
	k.value.assign(t.p, vector<double>(k.m));
	for (idx_t j = 0; j < t.p; j++) {
		for (idx_t r = 0; r < k.m; r++) {
			k.value[j][r] = t.data[sample[r] * t.p + j];
		}
		Standardise(k.value[j]);
	}

	std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0x9E3779B97F4A7C15ULL);
	std::normal_distribution<double> gaussian(0.0, 1.0);
	std::uniform_real_distribution<double> uniform(0.0, 2.0 * 3.14159265358979323846);
	k.weight.resize(kSetFeatures * t.p);
	k.phase.resize(kSetFeatures);
	for (idx_t c = 0; c < kSetFeatures; c++) {
		for (idx_t j = 0; j < t.p; j++) {
			k.weight[c * t.p + j] = gaussian(rng);
		}
		k.phase[c] = uniform(rng);
	}

	k.feature.assign(t.p, vector<double>(k.m * kXyFeatures));
	const double scale = std::sqrt(2.0 / static_cast<double>(kXyFeatures));
	vector<idx_t> one(1);
	for (idx_t j = 0; j < t.p; j++) {
		one[0] = j;
		const double bandwidth = MedianDistance(k, one);
		for (idx_t c = 0; c < kXyFeatures; c++) {
			const double w = gaussian(rng) / bandwidth;
			const double b = uniform(rng);
			for (idx_t r = 0; r < k.m; r++) {
				k.feature[j][r * kXyFeatures + c] = scale * std::cos(w * k.value[j][r] + b);
			}
		}
		CentreColumns(k.feature[j], k.m, kXyFeatures);
	}
	return k;
}

// --- the test object: Fisher's z, or the mixed-data Wald test --------------------

//! One conditional-independence test over one sample of rows. Pearson and rank
//! read a correlation matrix with Fisher's z. The mixed test reads latent
//! correlations, and carries each row's influence on every one of them, because
//! its partial correlations do not have Fisher's variance: a binary column's
//! latent correlations are far noisier than a continuous column's, and in
//! simulation a nominal 1% test run with Fisher's z rejected true independences
//! among binary columns 9-29% of the time. The influences give each partial
//! correlation its own variance, and the test holds its level.
struct CiTest {
	vector<double> C;
	idx_t p = 0;
	double n = 0.0;
	bool mixed = false;
	idx_t rows = 0;
	//! [a * p + b] for a < b: each row's influence on C[a][b]. Mixed only.
	vector<vector<float>> psi;
	//! test := 'kernel': the features, held by value because each resample has its own
	//! sample of rows and the bootstrap runs the replicates in parallel.
	bool kernel = false;
	KernelTest features;

	double PValue(idx_t i, idx_t j, const vector<idx_t> &S) const;
};

//! Dense ranks 0..levels-1 of one column over a sample of rows.
void DenseRanks(const NumericTable &t, idx_t j, const vector<idx_t> &rows, vector<idx_t> &rank, idx_t &levels) {
	const idx_t n = rows.size();
	auto value = [&](idx_t k) {
		return t.data[rows[k] * t.p + j];
	};
	vector<idx_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return value(a) < value(b); });
	rank.assign(n, 0);
	levels = 0;
	for (idx_t k = 0; k < n; k++) {
		if (k > 0 && value(order[k]) != value(order[k - 1])) {
			levels++;
		}
		rank[order[k]] = levels;
	}
	levels++;
}

//! h[k] = (1 / (n - 1)) sum_l sign(x_k - x_l) sign(y_k - y_l): each row's share of
//! Kendall's tau, whose mean is tau-a. A Fenwick tree over y's ranks counts, for
//! each row, the rows below and above it in x that sit below or above it in y,
//! in O(n log n).
void KendallKernel(const vector<idx_t> &rx, idx_t mx, const vector<idx_t> &ry, idx_t my, vector<double> &h) {
	const idx_t n = rx.size();
	vector<idx_t> start(mx + 1, 0), by_x(n);
	for (idx_t k = 0; k < n; k++) {
		start[rx[k] + 1]++;
	}
	for (idx_t v = 0; v < mx; v++) {
		start[v + 1] += start[v];
	}
	{
		vector<idx_t> fill(start.begin(), start.end() - 1);
		for (idx_t k = 0; k < n; k++) {
			by_x[fill[rx[k]]++] = k;
		}
	}
	vector<double> tree(my + 1);
	auto add = [&](idx_t rank) {
		for (idx_t i = rank + 1; i <= my; i += i & (~i + 1)) {
			tree[i] += 1.0;
		}
	};
	auto below = [&](idx_t rank) { // rows inserted with a y rank < rank
		double s = 0.0;
		for (idx_t i = rank; i > 0; i -= i & (~i + 1)) {
			s += tree[i];
		}
		return s;
	};
	h.assign(n, 0.0);
	for (int direction = 0; direction < 2; direction++) {
		std::fill(tree.begin(), tree.end(), 0.0);
		double inserted = 0.0;
		for (idx_t step = 0; step < mx; step++) {
			const idx_t v = direction == 0 ? step : mx - 1 - step;
			for (idx_t q = start[v]; q < start[v + 1]; q++) {
				const idx_t k = by_x[q];
				const double less = below(ry[k]), greater = inserted - below(ry[k] + 1);
				// Rows lower in x concord when lower in y; rows higher in x when higher.
				h[k] += direction == 0 ? less - greater : greater - less;
			}
			for (idx_t q = start[v]; q < start[v + 1]; q++) {
				add(ry[by_x[q]]);
				inserted += 1.0;
			}
		}
	}
	for (auto &value : h) {
		value /= static_cast<double>(n - 1);
	}
}

double NormalDensity(double x) {
	return std::exp(-0.5 * x * x) / 2.5066282746310002;
}

double BivariateNormalDensity(double h, double k, double r) {
	const double s = 1.0 - r * r;
	return std::exp(-(h * h - 2.0 * r * h * k + k * k) / (2.0 * s)) / (2.0 * 3.14159265358979323846 * std::sqrt(s));
}

//! d/dh of P(X <= h, Y <= k) with correlation r.
double BivariateCdfDh(double h, double k, double r) {
	return NormalDensity(h) * NormalCdf((k - r * h) / std::sqrt(1.0 - r * r));
}

//! The outermost thresholds of an ordinal column sit at +-kOrdinalBound rather
//! than at infinity; the normal CDF there is 0 or 1 to within 1e-19.
const double kOrdinalBound = 9.0;

//! Normal scores of a column, Phi^-1(rank / (n + 1)), with tied values sharing
//! their average rank.
void ColumnNormalScores(const vector<double> &column, vector<double> &score) {
	const idx_t n = column.size();
	vector<idx_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return column[a] < column[b]; });
	score.assign(n, 0.0);
	for (idx_t start = 0; start < n;) {
		idx_t end = start + 1;
		while (end < n && column[order[end]] == column[order[start]]) {
			end++;
		}
		const double value = NormalQuantile(0.5 * static_cast<double>(start + end + 1) / static_cast<double>(n + 1));
		for (idx_t k = start; k < end; k++) {
			score[order[k]] = value;
		}
		start = end;
	}
}

//! An ordinal column's thresholds - Phi^-1 of its cumulative proportions, padded
//! with -kOrdinalBound and kOrdinalBound - and each inner threshold's row
//! influence, (1[code <= j] - F_j) / phi(threshold j).
void OrdinalThresholds(const vector<idx_t> &code, idx_t levels, vector<double> &cut,
                       vector<vector<double>> &influence) {
	const idx_t n = code.size();
	vector<double> count(levels, 0.0);
	for (auto c : code) {
		count[c] += 1.0;
	}
	cut.assign(levels + 1, 0.0);
	cut[0] = -kOrdinalBound;
	cut[levels] = kOrdinalBound;
	influence.assign(levels - 1, vector<double>(n, 0.0));
	double cumulative = 0.0;
	for (idx_t j = 0; j + 1 < levels; j++) {
		cumulative += count[j];
		const double share = cumulative / static_cast<double>(n);
		cut[j + 1] = NormalQuantile(share);
		const double density = NormalDensity(cut[j + 1]);
		for (idx_t k = 0; k < n; k++) {
			influence[j][k] = ((code[k] <= j ? 1.0 : 0.0) - share) / density;
		}
	}
}

//! The r in [-0.999, 0.999] that maximises a unimodal function, by
//! golden-section search.
template <class F>
double MaximiseCorrelation(F &&f) {
	const double ratio = 0.6180339887498949;
	double lo = -0.999, hi = 0.999;
	double x1 = hi - ratio * (hi - lo), x2 = lo + ratio * (hi - lo);
	double f1 = f(x1), f2 = f(x2);
	for (int it = 0; it < 120; it++) {
		if (f1 < f2) {
			lo = x1;
			x1 = x2;
			f1 = f2;
			x2 = lo + ratio * (hi - lo);
			f2 = f(x2);
		} else {
			hi = x2;
			x2 = x1;
			f2 = f1;
			x1 = hi - ratio * (hi - lo);
			f1 = f(x1);
		}
	}
	return 0.5 * (lo + hi);
}

//! Polychoric correlation of two ordinal columns - a binary column is an
//! ordinal one with two levels - in two steps: thresholds from each column's
//! proportions, then the r that maximises the likelihood of the contingency
//! table, whose cells are rectangles of a bivariate normal. psi receives each
//! row's influence on r: its score, plus its share in every threshold, over
//! the slope of the mean score in r.
double Polychoric(const vector<idx_t> &x, idx_t mx, const vector<idx_t> &y, idx_t my, vector<float> &psi) {
	const idx_t n = x.size();
	vector<double> cx, cy;
	vector<vector<double>> ix, iy;
	OrdinalThresholds(x, mx, cx, ix);
	OrdinalThresholds(y, my, cy, iy);
	vector<double> counts(mx * my, 0.0);
	for (idx_t k = 0; k < n; k++) {
		counts[x[k] * my + y[k]] += 1.0;
	}
	vector<double> grid((mx + 1) * (my + 1)), density((mx + 1) * (my + 1)), cell(mx * my), slope(mx * my);
	// Each cell's probability, and its derivative in r: the bivariate density at
	// its inner corners, since only those move.
	auto fill = [&](double r, const vector<double> &ax, const vector<double> &ay) {
		for (idx_t i = 0; i <= mx; i++) {
			for (idx_t j = 0; j <= my; j++) {
				grid[i * (my + 1) + j] = BivariateNormalCdf(ax[i], ay[j], r);
				density[i * (my + 1) + j] =
				    (i == 0 || i == mx || j == 0 || j == my) ? 0.0 : BivariateNormalDensity(ax[i], ay[j], r);
			}
		}
		for (idx_t a = 0; a < mx; a++) {
			for (idx_t b = 0; b < my; b++) {
				const idx_t hh = (a + 1) * (my + 1) + b + 1, lh = a * (my + 1) + b + 1, hl = (a + 1) * (my + 1) + b,
				            ll = a * (my + 1) + b;
				cell[a * my + b] = std::max(grid[hh] - grid[lh] - grid[hl] + grid[ll], 1e-300);
				slope[a * my + b] = density[hh] - density[lh] - density[hl] + density[ll];
			}
		}
	};
	auto loglik = [&](double r) {
		fill(r, cx, cy);
		double sum = 0.0;
		for (idx_t c = 0; c < mx * my; c++) {
			sum += counts[c] * std::log(cell[c]);
		}
		return sum;
	};
	const double r = MaximiseCorrelation(loglik);
	auto mean_score = [&](double rr, const vector<double> &ax, const vector<double> &ay) {
		fill(rr, ax, ay);
		double sum = 0.0;
		for (idx_t c = 0; c < mx * my; c++) {
			sum += counts[c] * slope[c] / cell[c];
		}
		return sum / static_cast<double>(n);
	};
	const double e = 1e-5;
	const double dr = (mean_score(r + e, cx, cy) - mean_score(r - e, cx, cy)) / (2.0 * e);
	vector<double> total(n);
	fill(r, cx, cy);
	for (idx_t k = 0; k < n; k++) {
		total[k] = slope[x[k] * my + y[k]] / cell[x[k] * my + y[k]];
	}
	for (idx_t j = 0; j + 1 < mx; j++) {
		auto up = cx, down = cx;
		up[j + 1] += e;
		down[j + 1] -= e;
		const double g = (mean_score(r, up, cy) - mean_score(r, down, cy)) / (2.0 * e);
		for (idx_t k = 0; k < n; k++) {
			total[k] += g * ix[j][k];
		}
	}
	for (idx_t j = 0; j + 1 < my; j++) {
		auto up = cy, down = cy;
		up[j + 1] += e;
		down[j + 1] -= e;
		const double g = (mean_score(r, cx, up) - mean_score(r, cx, down)) / (2.0 * e);
		for (idx_t k = 0; k < n; k++) {
			total[k] += g * iy[j][k];
		}
	}
	psi.assign(n, 0.0f);
	for (idx_t k = 0; k < n; k++) {
		psi[k] = static_cast<float>(-total[k] / dr);
	}
	return r;
}

//! Polyserial correlation of an ordinal column with a continuous one, in two
//! steps: thresholds from the ordinal column's proportions, then the r that
//! maximises the likelihood of each row's level given the continuous column's
//! normal score u. The influence treats the normal scores as known.
double Polyserial(const vector<idx_t> &x, idx_t mx, const vector<double> &u, vector<float> &psi) {
	const idx_t n = x.size();
	vector<double> cx;
	vector<vector<double>> ix;
	OrdinalThresholds(x, mx, cx, ix);
	auto row_log = [&](double r, const vector<double> &cut, idx_t k) {
		const double s = std::sqrt(1.0 - r * r);
		return std::log(
		    std::max(NormalCdf((cut[x[k] + 1] - r * u[k]) / s) - NormalCdf((cut[x[k]] - r * u[k]) / s), 1e-300));
	};
	auto loglik = [&](double r) {
		double sum = 0.0;
		for (idx_t k = 0; k < n; k++) {
			sum += row_log(r, cx, k);
		}
		return sum;
	};
	const double r = MaximiseCorrelation(loglik);
	const double e = 1e-5;
	auto row_score = [&](double rr, const vector<double> &cut, idx_t k) {
		return (row_log(rr + e, cut, k) - row_log(rr - e, cut, k)) / (2.0 * e);
	};
	auto mean_score = [&](double rr, const vector<double> &cut) {
		double sum = 0.0;
		for (idx_t k = 0; k < n; k++) {
			sum += row_score(rr, cut, k);
		}
		return sum / static_cast<double>(n);
	};
	const double dr = (mean_score(r + e, cx) - mean_score(r - e, cx)) / (2.0 * e);
	vector<double> total(n);
	for (idx_t k = 0; k < n; k++) {
		total[k] = row_score(r, cx, k);
	}
	for (idx_t j = 0; j + 1 < mx; j++) {
		auto up = cx, down = cx;
		up[j + 1] += e;
		down[j + 1] -= e;
		const double g = (mean_score(r, up) - mean_score(r, down)) / (2.0 * e);
		for (idx_t k = 0; k < n; k++) {
			total[k] += g * ix[j][k];
		}
	}
	psi.assign(n, 0.0f);
	for (idx_t k = 0; k < n; k++) {
		psi[k] = static_cast<float>(-total[k] / dr);
	}
	return r;
}

//! Latent correlations for test := 'mixed' (Fan, Liu, Ning and Zou 2017). Each
//! pair's Kendall's tau is mapped through the bridge for the pair's kinds:
//! sin(pi tau / 2) for two continuous columns, and an inverted bivariate-normal
//! expression when either is binary, with a binary column's threshold estimated
//! from its mean. Each row's influence on each estimate - its U-statistic share
//! of tau, plus its share of each threshold - is kept for the test's variance.
//! A pair with an ordinal column - 3 to 9 values - takes a polychoric estimate,
//! or a polyserial one against a continuous column, with its own influence.
void MixedLatent(const NumericTable &t, const vector<idx_t> &rows, const vector<uint8_t> &binary, CiTest &test,
                 bool *repaired) {
	const idx_t p = t.p, n = rows.size();
	test.mixed = true;
	test.rows = n;
	test.C.assign(p * p, 0.0);
	test.psi.assign(p * p, vector<float>());
	vector<vector<idx_t>> rank(p);
	vector<idx_t> levels(p);
	vector<double> delta(p, 0.0), phat(p, 0.0);
	vector<uint8_t> degenerate(p, 0);
	for (idx_t j = 0; j < p; j++) {
		DenseRanks(t, j, rows, rank[j], levels[j]);
		test.C[j * p + j] = 1.0;
		// A resample can draw a single value; such a column says nothing about any other.
		degenerate[j] = levels[j] < 2 ? 1 : 0;
		if (binary[j] && !degenerate[j]) {
			double high = 0.0;
			for (idx_t k = 0; k < n; k++) {
				high += rank[j][k] == 1 ? 1.0 : 0.0;
			}
			phat[j] = high / static_cast<double>(n);
			delta[j] = NormalQuantile(1.0 - phat[j]);
		}
	}
	const double kPi = 3.14159265358979323846, kRoot2 = std::sqrt(2.0);
	vector<double> h;
	// Normal scores of the continuous columns, for polyserial pairs; built on first use.
	vector<vector<double>> scores(p);
	auto scores_of = [&](idx_t j) -> const vector<double> & {
		if (scores[j].empty()) {
			vector<double> column(n);
			for (idx_t k = 0; k < n; k++) {
				column[k] = t.data[rows[k] * p + j];
			}
			ColumnNormalScores(column, scores[j]);
		}
		return scores[j];
	};
	for (idx_t a = 0; a < p; a++) {
		for (idx_t b = a + 1; b < p; b++) {
			auto &psi = test.psi[a * p + b];
			psi.assign(n, 0.0f);
			if (degenerate[a] || degenerate[b]) {
				continue;
			}
			// A pair with an ordinal column takes a polychoric estimate, or a
			// polyserial one against a continuous column.
			if (binary[a] == 2 || binary[b] == 2) {
				double r;
				if (binary[a] != 0 && binary[b] != 0) {
					r = Polychoric(rank[a], levels[a], rank[b], levels[b], psi);
				} else if (binary[a] == 2) {
					r = Polyserial(rank[a], levels[a], scores_of(b), psi);
				} else {
					r = Polyserial(rank[b], levels[b], scores_of(a), psi);
				}
				test.C[a * p + b] = test.C[b * p + a] = r;
				continue;
			}
			KendallKernel(rank[a], levels[a], rank[b], levels[b], h);
			double tau = 0.0;
			for (auto v : h) {
				tau += v;
			}
			tau /= static_cast<double>(n);
			double r;
			if (!binary[a] && !binary[b]) {
				r = std::sin(kPi * tau / 2.0);
				const double slope = (kPi / 2.0) * std::cos(kPi * tau / 2.0);
				for (idx_t k = 0; k < n; k++) {
					psi[k] = static_cast<float>(slope * 2.0 * (h[k] - tau));
				}
			} else {
				const bool both = binary[a] && binary[b];
				const idx_t u = binary[a] ? a : b; // the binary one, when only one is
				auto bridge = [&](double rr) {
					if (both) {
						return 2.0 *
						       (BivariateNormalCdf(delta[a], delta[b], rr) - NormalCdf(delta[a]) * NormalCdf(delta[b]));
					}
					return 4.0 * BivariateNormalCdf(delta[u], 0.0, rr / kRoot2) - 2.0 * NormalCdf(delta[u]);
				};
				// The bridge rises with r, so bisection finds the r it maps to tau.
				double lo = -0.9999, hi = 0.9999;
				for (int it = 0; it < 100; it++) {
					const double mid = 0.5 * (lo + hi);
					(bridge(mid) < tau ? lo : hi) = mid;
				}
				r = 0.5 * (lo + hi);
				double f_r, f_a = 0.0, f_b = 0.0;
				if (both) {
					f_r = 2.0 * BivariateNormalDensity(delta[a], delta[b], r);
					f_a = 2.0 * (BivariateCdfDh(delta[a], delta[b], r) - NormalDensity(delta[a]) * NormalCdf(delta[b]));
					f_b = 2.0 * (BivariateCdfDh(delta[b], delta[a], r) - NormalDensity(delta[b]) * NormalCdf(delta[a]));
				} else {
					const double q = r / kRoot2;
					f_r = 4.0 * BivariateNormalDensity(delta[u], 0.0, q) / kRoot2;
					(u == a ? f_a : f_b) = 4.0 * BivariateCdfDh(delta[u], 0.0, q) - 2.0 * NormalDensity(delta[u]);
				}
				// bridge(r, thresholds) = tau, so a row moves r by its move of tau, less
				// what it moves the thresholds by, over the bridge's slope in r.
				for (idx_t k = 0; k < n; k++) {
					double value = 2.0 * (h[k] - tau);
					if (binary[a]) {
						value -= f_a * (-((rank[a][k] == 1 ? 1.0 : 0.0) - phat[a]) / NormalDensity(delta[a]));
					}
					if (binary[b]) {
						value -= f_b * (-((rank[b][k] == 1 ? 1.0 : 0.0) - phat[b]) / NormalDensity(delta[b]));
					}
					psi[k] = static_cast<float>(value / f_r);
				}
			}
			test.C[a * p + b] = test.C[b * p + a] = r;
		}
	}
	// Pairwise estimates need not be positive definite together. Floor the
	// eigenvalues and rescale to a unit diagonal; the influences stay as they are.
	vector<double> values, vectors;
	SymmetricEigen(test.C, p, values, vectors);
	const double kFloor = 1e-4;
	if (*std::min_element(values.begin(), values.end()) < kFloor) {
		vector<double> fixed(p * p, 0.0);
		for (idx_t a = 0; a < p; a++) {
			for (idx_t b = 0; b < p; b++) {
				double sum = 0.0;
				for (idx_t m = 0; m < p; m++) {
					sum += vectors[a * p + m] * std::max(values[m], kFloor) * vectors[b * p + m];
				}
				fixed[a * p + b] = sum;
			}
		}
		for (idx_t a = 0; a < p; a++) {
			for (idx_t b = 0; b < p; b++) {
				test.C[a * p + b] = a == b ? 1.0 : fixed[a * p + b] / std::sqrt(fixed[a * p + a] * fixed[b * p + b]);
			}
		}
		if (repaired) {
			*repaired = true;
		}
	}
}

//! Wald test of zero latent partial correlation of i and j given S. The partial
//! correlation is a function of the correlations among {i, j} and S; its
//! gradient, from dP = -P dR P with P the inverse, turns each row's influence on
//! those correlations into its influence on the partial correlation, and the
//! variance is the mean square of that over n.
double MixedPValue(const CiTest &test, idx_t i, idx_t j, const vector<idx_t> &S) {
	vector<idx_t> index = {i, j};
	index.insert(index.end(), S.begin(), S.end());
	const idx_t k = index.size(), p = test.p;
	vector<double> M(k * k), P(k * k);
	for (idx_t a = 0; a < k; a++) {
		for (idx_t b = 0; b < k; b++) {
			M[a * k + b] = test.C[index[a] * p + index[b]];
		}
	}
	bool ok = false;
	for (int attempt = 0; attempt < 2 && !ok; attempt++) {
		ok = true;
		for (idx_t c = 0; c < k && ok; c++) {
			vector<double> A = M, e(k, 0.0), column;
			e[c] = 1.0;
			ok = CholeskySolve(A, k, e, column);
			for (idx_t r = 0; r < k && ok; r++) {
				P[r * k + c] = column[r];
			}
		}
		// A near-collinear conditioning set: nudge the diagonal and try once more.
		for (idx_t a = 0; a < k && !ok; a++) {
			M[a * k + a] += 1e-8;
		}
	}
	const double p00 = P[0], p11 = P[k + 1], p01 = P[1];
	if (!ok || !(p00 > 0.0) || !(p11 > 0.0)) {
		return 1.0;
	}
	const double rho = -p01 / std::sqrt(p00 * p11);
	vector<double> total(test.rows, 0.0);
	for (idx_t a = 0; a < k; a++) {
		for (idx_t b = a + 1; b < k; b++) {
			const double d01 = -(P[a] * P[b * k + 1] + P[b] * P[a * k + 1]);
			const double d00 = -2.0 * P[a] * P[b];
			const double d11 = -2.0 * P[k + a] * P[k + b];
			const double g =
			    -d01 / std::sqrt(p00 * p11) + 0.5 * p01 * std::pow(p00 * p11, -1.5) * (d00 * p11 + p00 * d11);
			const auto &psi = test.psi[std::min(index[a], index[b]) * p + std::max(index[a], index[b])];
			for (idx_t r = 0; r < test.rows; r++) {
				total[r] += g * psi[r];
			}
		}
	}
	double ss = 0.0;
	for (auto v : total) {
		ss += v * v;
	}
	const double se = std::sqrt(ss) / static_cast<double>(test.rows);
	if (!(se > 0.0)) {
		return 1.0;
	}
	return NormalTwoSidedP(rho / se);
}

double CiTest::PValue(idx_t i, idx_t j, const vector<idx_t> &S) const {
	if (kernel) {
		return features.PValue(i, j, S);
	}
	if (mixed) {
		return MixedPValue(*this, i, j, S);
	}
	return IndependencePValue(PartialCorrelation(C, p, i, j, S), n, S.size());
}

// --- background knowledge ---------------------------------------------------
//
// Most of the undirected edges in a real CPDAG are not settled by a cleverer
// test. They are settled by someone saying that age precedes the campaign and
// the campaign precedes revenue. tiers := [['age'], ['discount'], ['revenue']]
// says exactly that: nothing in a later group causes anything in an earlier one.
// A column named in no group is unconstrained, so a tier can be given for the
// two variables a person is sure about without inventing an order for the rest.

constexpr idx_t kNoTier = static_cast<idx_t>(-1);

struct Knowledge {
	idx_t p = 0;
	vector<idx_t> tier;
	bool any = false;

	//! i -> j is ruled out: i sits in a later tier than j.
	bool Forbidden(idx_t i, idx_t j) const {
		return any && tier[i] != kNoTier && tier[j] != kNoTier && tier[i] > tier[j];
	}
	//! The tiers settle this adjacency, in the direction i -> j.
	bool Settles(idx_t i, idx_t j) const {
		return any && tier[i] != kNoTier && tier[j] != kNoTier && tier[i] < tier[j];
	}
};

//! Resolve the tier groups against the columns actually loaded. A name that is
//! not among them, or that appears twice, is a mistake worth stopping for: a
//! silently ignored tier would look like the data had settled an edge the person
//! settled themselves.
Knowledge BuildKnowledge(const NumericTable &t, const DiscoverSpec &spec, const char *fn) {
	Knowledge k;
	k.p = t.p;
	k.tier.assign(t.p, kNoTier);
	if (spec.tiers.empty()) {
		return k;
	}
	k.any = true;
	for (idx_t group = 0; group < spec.tiers.size(); group++) {
		for (auto &name : spec.tiers[group]) {
			idx_t found = DConstants::INVALID_INDEX;
			for (idx_t j = 0; j < t.p; j++) {
				if (StringUtil::CIEquals(t.names[j], name)) {
					found = j;
				}
			}
			if (found == DConstants::INVALID_INDEX) {
				throw BinderException("duckdo: tiers names column '%s', which %s is not reading. Its columns are the "
				                      "numeric ones, or those given in columns := [...]",
				                      name, fn);
			}
			if (k.tier[found] != kNoTier) {
				throw BinderException("duckdo: column '%s' is in two tiers, so its place in the order is not stated",
				                      t.names[found]);
			}
			k.tier[found] = group;
		}
	}
	return k;
}

// --- PC-stable -----------------------------------------------------------------

struct Cpdag {
	idx_t p = 0;
	vector<uint8_t> adj;  // symmetric
	vector<uint8_t> head; // head[i*p+j]: an arrowhead at j on the edge i - j, i.e. i -> j
	idx_t conflicts = 0;
	//! V-structures the tiers refused, which is a person and the data disagreeing.
	idx_t tier_conflicts = 0;

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

//! `sepset_out`, when given, receives the separating set of every pair the
//! skeleton search disconnected - which is what FCI starts from.
Cpdag RunPc(const CiTest &test, double alpha, idx_t max_conditioning, const Knowledge &knowledge,
            std::map<std::pair<idx_t, idx_t>, vector<idx_t>> *sepset_out = nullptr) {
	const idx_t p = test.p;
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
					if (test.PValue(i, j, S) > alpha) {
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

	// Tiers before evidence: they settle every adjacency that crosses them, and
	// they are not something a test can overturn. Doing it here also means Meek's
	// rules below can never reach for a forbidden orientation - every pair they
	// could forbid is already directed, and so no longer undirected.
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = 0; j < p; j++) {
			if (i != j && g.adj[i * p + j] && knowledge.Settles(i, j)) {
				g.Orient(i, j);
			}
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
				if (knowledge.Forbidden(i, k) || knowledge.Forbidden(j, k)) {
					g.tier_conflicts++;
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
	if (sepset_out) {
		*sepset_out = sepset;
	}
	return g;
}

// --- FCI -----------------------------------------------------------------------
//
// PC assumes nothing unmeasured causes two of the measured variables. FCI
// (Spirtes, Glymour and Scheines; Zhang 2008) drops that assumption and pays for
// it in what it can say. Its output is a partial ancestral graph: each end of an
// edge is an arrowhead, a tail, or a circle meaning the data did not decide, and
// a <-> b says neither causes the other and something hidden causes both. It
// starts from PC's skeleton and separating sets, removes the edges a hidden
// common cause can fake (the possible-d-sep stage), and orients with Zhang's rules
// R1-R4 and R8-R10, assuming no selection bias - under which those rules are
// complete. R9 and R10 search paths, with a step budget; a search that runs out
// finds nothing, so the edge keeps its circle and goes to review rather than
// past it.

enum PagMark : uint8_t { kNone = 0, kCircle = 1, kArrow = 2, kTail = 3 };

//! mark[i * p + j] is the mark at j on the edge between i and j.
struct Pag {
	idx_t p = 0;
	vector<uint8_t> mark;

	bool Adjacent(idx_t i, idx_t j) const {
		return mark[i * p + j] != kNone;
	}
	uint8_t At(idx_t i, idx_t j) const {
		return mark[i * p + j];
	}
	void Set(idx_t i, idx_t j, uint8_t value) {
		mark[i * p + j] = value;
	}
	//! i --> j: a tail at i and an arrowhead at j.
	bool Directed(idx_t i, idx_t j) const {
		return At(i, j) == kArrow && At(j, i) == kTail;
	}
};

using Sepsets = std::map<std::pair<idx_t, idx_t>, vector<idx_t>>;

bool InSepset(const Sepsets &sepset, idx_t i, idx_t j, idx_t k) {
	auto found = sepset.find({std::min(i, j), std::max(i, j)});
	return found != sepset.end() && std::find(found->second.begin(), found->second.end(), k) != found->second.end();
}

//! Every edge back to o-o, then the unshielded colliders i *-> k <-* j.
void OrientColliders(Pag &g, const Sepsets &sepset) {
	const idx_t p = g.p;
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = 0; j < p; j++) {
			if (g.Adjacent(i, j)) {
				g.Set(i, j, kCircle);
			}
		}
	}
	for (idx_t k = 0; k < p; k++) {
		for (idx_t i = 0; i < p; i++) {
			for (idx_t j = i + 1; j < p; j++) {
				if (i == k || j == k || !g.Adjacent(i, k) || !g.Adjacent(j, k) || g.Adjacent(i, j) ||
				    InSepset(sepset, i, j, k)) {
					continue;
				}
				g.Set(i, k, kArrow);
				g.Set(j, k, kArrow);
			}
		}
	}
}

//! Possible-D-SEP(x): every vertex reachable from x along a path on which each
//! interior vertex is a collider or forms a triangle with its two neighbours.
//! Whether a vertex qualifies depends on the edge it was entered by, so the
//! search runs over directed edges rather than vertices.
vector<idx_t> PossibleDSep(const Pag &g, idx_t x) {
	const idx_t p = g.p;
	vector<uint8_t> reached(p, 0), entered(p * p, 0);
	vector<std::pair<idx_t, idx_t>> frontier;
	for (idx_t v = 0; v < p; v++) {
		if (v != x && g.Adjacent(x, v)) {
			reached[v] = 1;
			entered[x * p + v] = 1;
			frontier.push_back({x, v});
		}
	}
	while (!frontier.empty()) {
		const auto edge = frontier.back();
		frontier.pop_back();
		const idx_t prev = edge.first, cur = edge.second;
		for (idx_t next = 0; next < p; next++) {
			if (next == prev || next == cur || !g.Adjacent(cur, next) || entered[cur * p + next]) {
				continue;
			}
			const bool collider = g.At(prev, cur) == kArrow && g.At(next, cur) == kArrow;
			if (!collider && !g.Adjacent(prev, next)) {
				continue;
			}
			entered[cur * p + next] = 1;
			if (next != x) {
				reached[next] = 1;
			}
			frontier.push_back({cur, next});
		}
	}
	vector<idx_t> out;
	for (idx_t v = 0; v < p; v++) {
		if (reached[v]) {
			out.push_back(v);
		}
	}
	return out;
}

//! Remove every edge whose ends a subset of either end's Possible-D-SEP
//! separates: the separations a hidden common cause hides from the neighbour
//! sets PC searched. The sets are computed before any removal, as PC-stable
//! freezes neighbours, so the order edges are visited in cannot matter.
void PossibleDSepStage(Pag &g, Sepsets &sepset, const CiTest &test, double alpha, idx_t max_conditioning) {
	const idx_t p = g.p;
	vector<vector<idx_t>> pds(p);
	for (idx_t x = 0; x < p; x++) {
		pds[x] = PossibleDSep(g, x);
	}
	vector<std::pair<idx_t, idx_t>> removed;
	vector<vector<idx_t>> separating;
	for (idx_t x = 0; x < p; x++) {
		for (idx_t y = x + 1; y < p; y++) {
			if (!g.Adjacent(x, y)) {
				continue;
			}
			bool separated = false;
			vector<idx_t> found;
			for (idx_t side = 0; side < 2 && !separated; side++) {
				const idx_t from = side == 0 ? x : y, other = side == 0 ? y : x;
				vector<idx_t> candidates;
				for (auto v : pds[from]) {
					if (v != other) {
						candidates.push_back(v);
					}
				}
				for (idx_t level = 0; level <= max_conditioning && level <= candidates.size() && !separated; level++) {
					vector<idx_t> pick(level);
					std::iota(pick.begin(), pick.end(), 0);
					while (true) {
						vector<idx_t> S(level);
						for (idx_t a = 0; a < level; a++) {
							S[a] = candidates[pick[a]];
						}
						if (test.PValue(x, y, S) > alpha) {
							separated = true;
							found = S;
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
			if (separated) {
				removed.push_back({x, y});
				separating.push_back(found);
			}
		}
	}
	for (idx_t k = 0; k < removed.size(); k++) {
		const idx_t x = removed[k].first, y = removed[k].second;
		g.Set(x, y, kNone);
		g.Set(y, x, kNone);
		sepset[{x, y}] = separating[k];
	}
}

//! Look for a discriminating path <theta, ..., v, b, c> backwards from v, where
//! v is already known to be a collider on it and a parent of c. The next vertex
//! back, d, needs an arrowhead into v; if d is not adjacent to c the path is
//! discriminating, and otherwise d must itself be a collider and a parent of c.
bool ExtendDiscriminating(const Pag &g, idx_t v, idx_t c, vector<uint8_t> &on_path, idx_t &theta) {
	for (idx_t d = 0; d < g.p; d++) {
		if (on_path[d] || !g.Adjacent(d, v) || g.At(d, v) != kArrow) {
			continue;
		}
		if (!g.Adjacent(d, c)) {
			theta = d;
			return true;
		}
		if (g.At(v, d) == kArrow && g.Directed(d, c)) {
			on_path[d] = 1;
			if (ExtendDiscriminating(g, d, c, on_path, theta)) {
				return true;
			}
			on_path[d] = 0;
		}
	}
	return false;
}

//! Whether cur - next can be the next step of an uncovered potentially directed
//! path that has reached cur from prev: no arrowhead at cur and no tail at next,
//! and prev and next not adjacent.
bool PdStep(const Pag &g, idx_t prev, idx_t cur, idx_t next) {
	return g.Adjacent(cur, next) && g.At(next, cur) != kArrow && g.At(cur, next) != kTail && !g.Adjacent(prev, next);
}

//! Is there an uncovered potentially directed path from cur, reached from prev,
//! on to target? Simple paths only, found depth-first. The search gives up after
//! a fixed number of steps, and a search that gave up counts as no path: the
//! rule that asked does not fire, and its edge keeps a circle for review.
bool UncoveredPdPath(const Pag &g, idx_t prev, idx_t cur, idx_t target, vector<uint8_t> &on_path, idx_t &budget) {
	for (idx_t next = 0; next < g.p; next++) {
		if (on_path[next] || !PdStep(g, prev, cur, next)) {
			continue;
		}
		if (next == target) {
			return true;
		}
		if (budget == 0) {
			return false;
		}
		budget--;
		on_path[next] = 1;
		const bool found = UncoveredPdPath(g, cur, next, target, on_path, budget);
		on_path[next] = 0;
		if (found) {
			return true;
		}
	}
	return false;
}

const idx_t kPathSearchBudget = 100000;

//! The neighbours m of a from which an uncovered potentially directed path
//! <a, m, ..., target> leads on to target. m may be target itself.
vector<idx_t> PdPathStarts(const Pag &g, idx_t a, idx_t target) {
	vector<idx_t> starts;
	for (idx_t m = 0; m < g.p; m++) {
		if (m == a || !g.Adjacent(a, m) || g.At(m, a) == kArrow || g.At(a, m) == kTail) {
			continue;
		}
		if (m == target) {
			starts.push_back(m);
			continue;
		}
		vector<uint8_t> on_path(g.p, 0);
		on_path[a] = on_path[m] = 1;
		idx_t budget = kPathSearchBudget;
		if (UncoveredPdPath(g, a, m, target, on_path, budget)) {
			starts.push_back(m);
		}
	}
	return starts;
}

//! R9 and R10 for a o-> c: whether the circle at a can become a tail.
bool TailByR9OrR10(const Pag &g, idx_t a, idx_t c) {
	const idx_t p = g.p;
	// R9: an uncovered potentially directed path <a, b, t, ..., c>, with b and c
	// apart. Together with a o-> c it would close a cycle unless a --> c.
	for (idx_t b = 0; b < p; b++) {
		if (b == a || b == c || !g.Adjacent(a, b) || g.Adjacent(b, c) || g.At(b, a) == kArrow || g.At(a, b) == kTail) {
			continue;
		}
		vector<uint8_t> on_path(p, 0);
		on_path[a] = on_path[b] = 1;
		idx_t budget = kPathSearchBudget;
		if (UncoveredPdPath(g, a, b, c, on_path, budget)) {
			return true;
		}
	}
	// R10: b --> c <-- t, with uncovered potentially directed paths from a to b
	// and from a to t that leave a through two different neighbours, m and w,
	// that are not adjacent.
	vector<idx_t> parents;
	for (idx_t b = 0; b < p; b++) {
		if (b != a && g.Adjacent(b, c) && g.Directed(b, c)) {
			parents.push_back(b);
		}
	}
	for (idx_t x = 0; x < parents.size(); x++) {
		const auto to_b = PdPathStarts(g, a, parents[x]);
		if (to_b.empty()) {
			continue;
		}
		for (idx_t y = x + 1; y < parents.size(); y++) {
			const auto to_t = PdPathStarts(g, a, parents[y]);
			for (auto m : to_b) {
				for (auto w : to_t) {
					if (m != w && !g.Adjacent(m, w)) {
						return true;
					}
				}
			}
		}
	}
	return false;
}

//! Zhang's rules R1-R4 and R8-R10, to a fixed point, assuming no selection bias
//! (so R5-R7 never apply). Every rule turns a circle into something else and
//! nothing turns anything back into a circle, so it ends.
void ApplyRules(Pag &g, const Sepsets &sepset) {
	const idx_t p = g.p;
	bool changed = true;
	while (changed) {
		changed = false;
		for (idx_t a = 0; a < p; a++) {
			for (idx_t b = 0; b < p; b++) {
				if (a == b || !g.Adjacent(a, b)) {
					continue;
				}
				for (idx_t c = 0; c < p; c++) {
					if (c == a || c == b || !g.Adjacent(b, c)) {
						continue;
					}
					// R1: a *-> b o-* c, a and c apart: b --> c.
					if (!g.Adjacent(a, c) && g.At(a, b) == kArrow && g.At(c, b) == kCircle) {
						g.Set(c, b, kTail);
						g.Set(b, c, kArrow);
						changed = true;
					}
					// R2: a --> b *-> c, or a *-> b --> c, with a *-o c: a *-> c.
					if (g.Adjacent(a, c) && g.At(a, c) == kCircle &&
					    ((g.Directed(a, b) && g.At(b, c) == kArrow) || (g.At(a, b) == kArrow && g.Directed(b, c)))) {
						g.Set(a, c, kArrow);
						changed = true;
					}
					// R8: a --> b --> c, or a -o b --> c, with a o-> c: a --> c.
					if (g.Adjacent(a, c) && g.At(a, c) == kArrow && g.At(c, a) == kCircle && g.Directed(b, c) &&
					    (g.Directed(a, b) || (g.At(b, a) == kTail && g.At(a, b) == kCircle))) {
						g.Set(c, a, kTail);
						changed = true;
					}
				}
			}
		}
		// R3: a *-> b <-* c, a *-o t o-* c, a and c apart, t *-o b: t *-> b.
		for (idx_t t = 0; t < p; t++) {
			for (idx_t b = 0; b < p; b++) {
				if (t == b || !g.Adjacent(t, b) || g.At(t, b) != kCircle) {
					continue;
				}
				bool orient = false;
				for (idx_t a = 0; a < p && !orient; a++) {
					for (idx_t c = a + 1; c < p && !orient; c++) {
						orient = a != t && c != t && a != b && c != b && g.Adjacent(a, b) && g.Adjacent(c, b) &&
						         !g.Adjacent(a, c) && g.At(a, b) == kArrow && g.At(c, b) == kArrow &&
						         g.Adjacent(a, t) && g.Adjacent(c, t) && g.At(a, t) == kCircle && g.At(c, t) == kCircle;
					}
				}
				if (orient) {
					g.Set(t, b, kArrow);
					changed = true;
				}
			}
		}
		// R4: a discriminating path <theta, ..., a, b, c> for b, with b o-* c. If
		// b was in the set that separated theta from c, b --> c; otherwise
		// a <-> b <-> c.
		for (idx_t b = 0; b < p; b++) {
			for (idx_t c = 0; c < p; c++) {
				if (b == c || !g.Adjacent(b, c) || g.At(c, b) != kCircle) {
					continue;
				}
				for (idx_t a = 0; a < p; a++) {
					if (a == b || a == c || !g.Adjacent(a, b) || g.At(b, a) != kArrow || !g.Directed(a, c)) {
						continue;
					}
					vector<uint8_t> on_path(p, 0);
					on_path[a] = on_path[b] = on_path[c] = 1;
					idx_t theta = 0;
					if (!ExtendDiscriminating(g, a, c, on_path, theta)) {
						continue;
					}
					if (InSepset(sepset, theta, c, b)) {
						g.Set(c, b, kTail);
						g.Set(b, c, kArrow);
					} else {
						g.Set(a, b, kArrow);
						g.Set(b, a, kArrow);
						g.Set(c, b, kArrow);
						g.Set(b, c, kArrow);
					}
					changed = true;
					break;
				}
			}
		}
		if (changed) {
			continue;
		}
		// R9 and R10 search paths rather than triangles, so they run only once the
		// cheaper rules have nothing left to do.
		for (idx_t a = 0; a < p; a++) {
			for (idx_t c = 0; c < p; c++) {
				if (a != c && g.Adjacent(a, c) && g.At(a, c) == kArrow && g.At(c, a) == kCircle &&
				    TailByR9OrR10(g, a, c)) {
					g.Set(c, a, kTail);
					changed = true;
				}
			}
		}
	}
}

Pag RunFci(const CiTest &test, double alpha, idx_t max_conditioning, const Knowledge &knowledge) {
	const idx_t p = test.p;
	Sepsets sepset;
	const auto skeleton = RunPc(test, alpha, max_conditioning, knowledge, &sepset);
	Pag g;
	g.p = p;
	g.mark.assign(p * p, kNone);
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = 0; j < p; j++) {
			if (i != j && skeleton.Adjacent(i, j)) {
				g.Set(i, j, kCircle);
			}
		}
	}
	OrientColliders(g, sepset);
	PossibleDSepStage(g, sepset, test, alpha, max_conditioning);
	OrientColliders(g, sepset);
	// Tiers, after the collider pass has reset the marks for the last time. A
	// variable cannot cause one in an earlier tier, so the later end of every
	// adjacency that crosses a tier takes an arrowhead: that end is not an
	// ancestor of the other. The earlier end keeps its circle, because a hidden
	// common cause is still allowed - only the reverse cause is not.
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = 0; j < p; j++) {
			if (i != j && g.Adjacent(i, j) && knowledge.Settles(i, j)) {
				g.Set(i, j, kArrow);
			}
		}
	}
	ApplyRules(g, sepset);
	return g;
}

//! The edge read from i's end: "o->" is i o-> j, "<->" is i <-> j.
string PagEdge(const Pag &g, idx_t i, idx_t j) {
	static const char at_i[] = {' ', 'o', '<', '-'};
	static const char at_j[] = {' ', 'o', '>', '-'};
	return string(1, at_i[g.At(j, i)]) + "-" + string(1, at_j[g.At(i, j)]);
}

//! Read an edge with its stronger mark on the right: <-o becomes o->, <-- becomes -->.
void ReadForward(const Pag &g, idx_t &i, idx_t &j) {
	static const int strength[] = {0, 1, 3, 0};
	if (strength[g.At(j, i)] > strength[g.At(i, j)]) {
		std::swap(i, j);
	}
}

// --- LiNGAM ----------------------------------------------------------------------
//
// PC and FCI read conditional independence, and conditional independence cannot
// tell x -> y from y -> x: the two fit a two-variable world equally well. That is
// why do_discover hands back undirected edges for a person to settle. DirectLiNGAM
// (Shimizu, Inazumi, Sogawa, Hyvarinen, Kawahara, Washio, Hoyer and Bollen, JMLR
// 2011) reads a different signal and can settle them. If every effect is linear,
// the graph has no cycles, nothing unmeasured causes two variables, and the
// disturbances are non-Gaussian, then regressing the effect on the cause leaves a
// residual independent of the cause while the reverse regression does not. The
// most exogenous variable is the one no other variable explains in that sense; it
// is peeled off, regressed out of the rest, and the search repeats on the
// residuals, which still satisfy a LiNGAM model.
//
// The price is the assumption list, which is strictly longer than PC's. The last
// item is the one that bites. On Gaussian disturbances the asymmetry is not merely
// weaker, it is absent, and the method returns a confident order that is close to a
// coin flip: in simulation over 40 draws of a six-variable graph, 40 orders exactly
// right with uniform disturbances against 1 with Gaussian ones. So the disturbances
// are tested, and two that cannot be told from Gaussian is a refusal rather than a
// footnote - identifiability allows at most one.
//
// LayeredLiNGAM's peel (Suzuki, ECML-PKDD 2024), which takes every variable tied
// for most exogenous at once, was implemented and measured against this: it cut the
// iteration count by a third and cost recall (0.933 against 1.000) and orders
// (33/40 against 40/40). The speed it buys is for variable counts far above the 30
// do_discover allows, so what runs below is DirectLiNGAM, one variable per pass.

//! Hyvarinen's (1998) negentropy approximation, for a u already standardised.
double EntropyApprox(const vector<double> &u) {
	constexpr double kK1 = 79.047, kK2 = 7.4129, kGamma = 0.37457;
	constexpr double kLn2 = 0.6931471805599453;
	const double m = static_cast<double>(u.size());
	double log_cosh = 0.0, gauss = 0.0;
	for (auto value : u) {
		// log cosh, written so a large value cannot overflow cosh itself.
		const double a = std::fabs(value);
		log_cosh += a + std::log1p(std::exp(-2.0 * a)) - kLn2;
		gauss += value * std::exp(-0.5 * value * value);
	}
	log_cosh /= m;
	gauss /= m;
	const double base = (1.0 + std::log(2.0 * 3.14159265358979323846)) / 2.0;
	return base - kK1 * (log_cosh - kGamma) * (log_cosh - kGamma) - kK2 * gauss * gauss;
}

//! The residual of xi on xj, both standardised, standardised again. With unit
//! variances the slope is just their correlation.
void UniResidual(const vector<double> &xi, const vector<double> &xj, vector<double> &out) {
	const idx_t m = xi.size();
	double beta = 0.0;
	for (idx_t r = 0; r < m; r++) {
		beta += xi[r] * xj[r];
	}
	beta /= static_cast<double>(m);
	out.resize(m);
	for (idx_t r = 0; r < m; r++) {
		out[r] = xi[r] - beta * xj[r];
	}
	Standardise(out);
}

//! Inverse of a small symmetric matrix by Gauss-Jordan with partial pivoting,
//! nudging the diagonal once if the columns are collinear - the same escalation
//! PartialCorrelation makes, for the same reason.
bool SmallInverse(vector<double> M, idx_t k, vector<double> &inv) {
	for (int attempt = 0; attempt < 2; attempt++) {
		vector<double> A = M;
		inv.assign(k * k, 0.0);
		for (idx_t a = 0; a < k; a++) {
			inv[a * k + a] = 1.0;
		}
		bool ok = true;
		for (idx_t col = 0; col < k && ok; col++) {
			idx_t pivot = col;
			for (idx_t row = col + 1; row < k; row++) {
				if (std::fabs(A[row * k + col]) > std::fabs(A[pivot * k + col])) {
					pivot = row;
				}
			}
			if (std::fabs(A[pivot * k + col]) < 1e-12) {
				ok = false;
				break;
			}
			if (pivot != col) {
				for (idx_t c = 0; c < k; c++) {
					std::swap(A[col * k + c], A[pivot * k + c]);
					std::swap(inv[col * k + c], inv[pivot * k + c]);
				}
			}
			const double scale = 1.0 / A[col * k + col];
			for (idx_t c = 0; c < k; c++) {
				A[col * k + c] *= scale;
				inv[col * k + c] *= scale;
			}
			for (idx_t row = 0; row < k; row++) {
				if (row == col) {
					continue;
				}
				const double factor = A[row * k + col];
				if (factor == 0.0) {
					continue;
				}
				for (idx_t c = 0; c < k; c++) {
					A[row * k + c] -= factor * A[col * k + c];
					inv[row * k + c] -= factor * inv[col * k + c];
				}
			}
		}
		if (ok) {
			return true;
		}
		for (idx_t a = 0; a < k; a++) {
			M[a * k + a] += 1e-8;
		}
	}
	return false;
}

//! P-value of the Jarque-Bera statistic against normality. Its null is chi-square
//! on two degrees of freedom, whose survival function is exp(-x / 2) exactly, so
//! this needs no special function at all.
double JarqueBeraP(const vector<double> &u) {
	const double m = static_cast<double>(u.size());
	double mean = 0.0;
	for (auto value : u) {
		mean += value;
	}
	mean /= m;
	double m2 = 0.0, m3 = 0.0, m4 = 0.0;
	for (auto value : u) {
		const double d = value - mean;
		m2 += d * d;
		m3 += d * d * d;
		m4 += d * d * d * d;
	}
	m2 /= m;
	m3 /= m;
	m4 /= m;
	if (m2 < 1e-24) {
		return 1.0;
	}
	const double skew = m3 / std::pow(m2, 1.5);
	const double kurtosis = m4 / (m2 * m2) - 3.0;
	const double jb = m / 6.0 * (skew * skew + kurtosis * kurtosis / 4.0);
	return std::exp(-0.5 * jb);
}

//! At most this many rows carry the ordering statistics. They are comparisons of
//! entropies, which settle long before a coefficient does - 40 of 40 orders exactly
//! right at 1000 rows in simulation - while their cost is quadratic in the
//! variables and linear in the rows.
constexpr idx_t kLingamOrderRows = 4000;

struct Lingam {
	Cpdag graph;
	//! Most exogenous first.
	vector<idx_t> order;
	//! Jarque-Bera p-value of each variable's disturbance. Large means it cannot be
	//! told from Gaussian, which is the case the causal order is not identified in.
	vector<double> gaussian_p;
	//! How many rows the ordering statistics actually saw.
	idx_t order_rows = 0;
	//! Datasets the order was estimated across, from groups := 'site'.
	idx_t groups = 1;
};

//! `C` is the correlation matrix over the same `rows`: with every column
//! standardised the normal equations of every regression are submatrices of it, so
//! the coefficients need no second pass over the data. `disturbances` asks for the
//! normality check, which the bootstrap replicates do not need.
Lingam RunLingam(const NumericTable &t, const vector<idx_t> &rows, const vector<double> &C, const DiscoverSpec &spec,
                 const Knowledge &knowledge, bool disturbances) {
	const idx_t p = t.p;
	Lingam out;
	out.graph.p = p;
	out.graph.adj.assign(p * p, 0);
	out.graph.head.assign(p * p, 0);
	out.gaussian_p.assign(p, 1.0);

	// The ordering compares entropies, so it reads a subsample taken by stride.
	// The rows arrive in content order, so which rows the stride keeps follows the
	// data rather than where the rows happen to sit.
	const idx_t stride = std::max<idx_t>(1, (rows.size() + kLingamOrderRows - 1) / kLingamOrderRows);
	vector<idx_t> sample;
	for (idx_t r = 0; r < rows.size(); r += stride) {
		sample.push_back(rows[r]);
	}
	const idx_t m = sample.size();
	out.order_rows = m;

	// One block of standardised columns per group, or a single block when there are
	// none. MultiGroupDirectLiNGAM (Shimizu 2012) is this and nothing more: the groups
	// are peeled together, which is what constrains them to share a causal order, and
	// their coefficients are fitted separately afterwards because the premise is that
	// those need not match.
	const idx_t groups = std::max<idx_t>(1, t.Groups());
	vector<vector<idx_t>> block(groups);
	for (idx_t r = 0; r < m; r++) {
		block[t.group.empty() ? 0 : t.group[sample[r]]].push_back(r);
	}
	vector<vector<vector<double>>> Z(groups);
	for (idx_t g = 0; g < groups; g++) {
		Z[g].assign(p, vector<double>(block[g].size()));
		for (idx_t j = 0; j < p; j++) {
			for (idx_t r = 0; r < block[g].size(); r++) {
				Z[g][j][r] = t.data[sample[block[g][r]] * p + j];
			}
			Standardise(Z[g][j]);
		}
	}

	vector<idx_t> remaining(p);
	std::iota(remaining.begin(), remaining.end(), 0);
	vector<double> forward, backward, peeled;
	while (remaining.size() > 1) {
		// Tiers narrow what may be peeled next: nothing in a later tier can be
		// exogenous relative to something still sitting in an earlier one.
		vector<idx_t> pool;
		if (knowledge.any) {
			idx_t first = kNoTier;
			for (auto j : remaining) {
				if (knowledge.tier[j] != kNoTier && knowledge.tier[j] < first) {
					first = knowledge.tier[j];
				}
			}
			for (auto j : remaining) {
				if (knowledge.tier[j] == kNoTier || knowledge.tier[j] == first) {
					pool.push_back(j);
				}
			}
		} else {
			pool = remaining;
		}

		idx_t next = pool[0];
		if (pool.size() > 1) {
			// R(a, b) is the likelihood ratio between "b causes a" and "a causes b". It
			// is exactly antisymmetric, so one pass over each pair scores both. Across
			// groups the ratios are added first, each weighted by its own row count,
			// and only the total is penalised. Penalising each group separately and
			// adding those was tried and is worse: under the true order every group's
			// ratio is positive only in expectation, so each one that dips negative on
			// noise charges the true source again, and four groups of 200 rows then
			// recovered the order less often than one group of 200 did.
			vector<double> penalty(pool.size(), 0.0);
			vector<double> entropy(pool.size());
			vector<double> ratio(pool.size() * pool.size(), 0.0);
			for (idx_t g = 0; g < groups; g++) {
				const double weight = static_cast<double>(block[g].size());
				if (weight < 2.0) {
					continue;
				}
				for (idx_t a = 0; a < pool.size(); a++) {
					entropy[a] = EntropyApprox(Z[g][pool[a]]);
				}
				for (idx_t a = 0; a < pool.size(); a++) {
					for (idx_t b = a + 1; b < pool.size(); b++) {
						UniResidual(Z[g][pool[a]], Z[g][pool[b]], forward);
						UniResidual(Z[g][pool[b]], Z[g][pool[a]], backward);
						const double r = (entropy[a] + EntropyApprox(backward)) - (entropy[b] + EntropyApprox(forward));
						ratio[a * pool.size() + b] += weight * r;
					}
				}
			}
			for (idx_t a = 0; a < pool.size(); a++) {
				for (idx_t b = a + 1; b < pool.size(); b++) {
					const double r = ratio[a * pool.size() + b];
					const double against_b = std::min(0.0, r);
					const double against_a = std::min(0.0, -r);
					penalty[b] += against_b * against_b;
					penalty[a] += against_a * against_a;
				}
			}
			idx_t best = 0;
			for (idx_t a = 1; a < pool.size(); a++) {
				if (penalty[a] < penalty[best]) {
					best = a;
				}
			}
			next = pool[best];
		}
		out.order.push_back(next);
		remaining.erase(std::find(remaining.begin(), remaining.end(), next));
		for (idx_t g = 0; g < groups; g++) {
			if (block[g].size() < 2) {
				continue;
			}
			peeled = Z[g][next];
			for (auto i : remaining) {
				UniResidual(Z[g][i], peeled, forward);
				Z[g][i] = forward;
			}
		}
	}
	if (!remaining.empty()) {
		out.order.push_back(remaining[0]);
	}

	// Each variable on its predecessors in that order, over every row. The columns
	// are standardised, so min_effect reads as "a one-standard-deviation move in
	// the cause shifts the effect by this much of its own standard deviation".
	// Each group's own rows, so its coefficients are its own. With no groups this is
	// one block holding everything and the loop below runs once.
	vector<vector<idx_t>> whole(groups);
	for (auto r : rows) {
		whole[t.group.empty() ? 0 : t.group[r]].push_back(r);
	}
	vector<idx_t> votes(p * p, 0);
	vector<vector<vector<double>>> by_group(groups, vector<vector<double>>(p));
	for (idx_t g = 0; g < groups; g++) {
		const vector<double> own = groups == 1 ? C : Correlation(t, whole[g]);
		const double n = static_cast<double>(whole[g].size());
		for (idx_t pos = 1; pos < out.order.size(); pos++) {
			const idx_t j = out.order[pos];
			vector<idx_t> S(out.order.begin(), out.order.begin() + pos);
			const idx_t k = S.size();
			vector<double> M(k * k), rhs(k), inv;
			for (idx_t a = 0; a < k; a++) {
				rhs[a] = own[S[a] * p + j];
				for (idx_t b = 0; b < k; b++) {
					M[a * k + b] = own[S[a] * p + S[b]];
				}
			}
			if (!SmallInverse(M, k, inv)) {
				continue;
			}
			vector<double> coef(k, 0.0);
			double explained = 0.0;
			for (idx_t a = 0; a < k; a++) {
				for (idx_t b = 0; b < k; b++) {
					coef[a] += inv[a * k + b] * rhs[b];
				}
				explained += coef[a] * rhs[a];
			}
			by_group[g][j] = coef;
			// The target has unit variance, so what the predecessors do not explain is
			// what is left of that 1, scaled from a mean square to an unbiased one.
			const double dof = std::max(n - static_cast<double>(k), 1.0);
			const double sigma2 = std::max(1.0 - explained, 1e-12) * n / dof;
			for (idx_t a = 0; a < k; a++) {
				const double variance = std::max(sigma2 * inv[a * k + a] / n, 1e-300);
				const double se = std::sqrt(variance);
				if (std::fabs(coef[a]) >= spec.min_effect && NormalTwoSidedP(coef[a] / se) < spec.alpha) {
					votes[S[a] * p + j]++;
				}
			}
		}
	}
	// An edge needs a majority of the groups. Requiring all of them would contradict
	// the premise that coefficients differ - one group where the effect is near zero
	// would erase it - and requiring one would let the noisiest group write the graph.
	const idx_t needed = (groups + 1) / 2;
	for (idx_t a = 0; a < p; a++) {
		for (idx_t b = 0; b < p; b++) {
			if (a != b && votes[a * p + b] >= needed) {
				out.graph.adj[a * p + b] = out.graph.adj[b * p + a] = 1;
				out.graph.Orient(a, b);
			}
		}
	}
	out.groups = groups;

	if (!disturbances) {
		return out;
	}
	// What is left of each variable once its predecessors are regressed out, inside
	// each group and with that group's own coefficients. Pooling this across groups
	// was tried and is wrong: the pooled residual carries the differences between the
	// groups' coefficients, which is a mixture and looks Gaussian, so the check
	// refused data that every group on its own could read. The order is identified as
	// long as the disturbances are non-Gaussian somewhere, so the smallest p-value
	// across groups is the one that counts.
	//
	// The coefficients used are every predecessor's, not only the ones that cleared
	// min_effect: the disturbance is what the model does not explain, and a
	// coefficient dropped for being small still explained its part.
	vector<double> residual;
	for (idx_t g = 0; g < groups; g++) {
		const idx_t rows_here = block[g].size();
		if (rows_here < 8) {
			continue;
		}
		vector<vector<double>> raw(p, vector<double>(rows_here));
		for (idx_t j = 0; j < p; j++) {
			for (idx_t r = 0; r < rows_here; r++) {
				raw[j][r] = t.data[sample[block[g][r]] * p + j];
			}
			Standardise(raw[j]);
		}
		for (idx_t pos = 0; pos < out.order.size(); pos++) {
			const idx_t j = out.order[pos];
			residual = raw[j];
			const auto &coef = by_group[g][j];
			for (idx_t a = 0; a < coef.size(); a++) {
				const idx_t src = out.order[a];
				for (idx_t r = 0; r < rows_here; r++) {
					residual[r] -= coef[a] * raw[src][r];
				}
			}
			const double here = JarqueBeraP(residual);
			if (g == 0 || here < out.gaussian_p[j]) {
				out.gaussian_p[j] = here;
			}
		}
	}
	return out;
}

// --- RESIT -----------------------------------------------------------------------
//
// LiNGAM buys its orientations with linearity. RESIT (Peters, Mooij, Janzing and
// Scholkopf, JMLR 2014) buys the same thing with a different trade: the effect may
// bend however it likes, so long as the disturbance is added rather than mixed in.
// A continuous additive noise model x_k = f_k(parents) + e_k is identifiable for a
// nonlinear f, and - this is the part that matters next to LiNGAM - for *any*
// disturbance distribution, Gaussian included. So RESIT covers exactly the case
// LiNGAM refuses.
//
// It runs LiNGAM's search from the other end. LiNGAM peels the most exogenous
// variable; RESIT peels a sink, because a sink is the variable whose residual, once
// every other remaining variable is regressed out of it, is most independent of
// them. Regressing here is kernel ridge regression on the same random Fourier
// features test := 'kernel' uses, and "most independent" is the squared cross-
// covariance of their features, an HSIC in that basis. Then the order is pruned:
// a predecessor is a parent only if the pair is still dependent given every other
// predecessor, which is the kernel test again.

struct Resit {
	Cpdag graph;
	//! Most exogenous first, as LiNGAM's is.
	vector<idx_t> order;
	idx_t rows = 0;
	//! Predecessors the pruning stage dropped, which is what separates an order from
	//! a graph: an order alone would make every earlier variable a parent.
	idx_t pruned = 0;
	//! Variables whose fit on their predecessors bends - where the kernel regression
	//! leaves meaningfully less than a straight line does - and how many variables
	//! could have. A continuous additive noise model is identified by the bend or by
	//! a non-Gaussian disturbance; linear effects with Gaussian disturbances is the
	//! one corner neither covers, and the one no method here can read.
	idx_t bending = 0;
	idx_t fitted = 0;
	//! Jarque-Bera p-value of each variable's disturbance.
	vector<double> gaussian_p;
};

Resit RunResit(const KernelTest &k, const DiscoverSpec &spec, const Knowledge &knowledge) {
	const idx_t p = k.p;
	Resit out;
	out.graph.p = p;
	out.graph.adj.assign(p * p, 0);
	out.graph.head.assign(p * p, 0);
	out.rows = k.m;

	vector<idx_t> remaining(p);
	std::iota(remaining.begin(), remaining.end(), 0);
	vector<idx_t> sinks;
	vector<double> fz, residual;
	while (remaining.size() > 1) {
		// Tiers narrow what may be peeled, from the other end than LiNGAM: a sink
		// cannot sit in an earlier tier than something still remaining.
		vector<idx_t> pool;
		if (knowledge.any) {
			idx_t last = 0;
			bool seen = false;
			for (auto j : remaining) {
				if (knowledge.tier[j] != kNoTier && (!seen || knowledge.tier[j] > last)) {
					last = knowledge.tier[j];
					seen = true;
				}
			}
			for (auto j : remaining) {
				if (!seen || knowledge.tier[j] == kNoTier || knowledge.tier[j] == last) {
					pool.push_back(j);
				}
			}
		} else {
			pool = remaining;
		}

		idx_t sink = pool[0];
		if (pool.size() > 1) {
			double best = 0.0;
			bool first = true;
			for (auto candidate : pool) {
				vector<idx_t> others;
				for (auto j : remaining) {
					if (j != candidate) {
						others.push_back(j);
					}
				}
				k.SetFeatures(others, fz);
				if (!k.Residual(k.value[candidate], others, fz, residual)) {
					continue;
				}
				const double score = k.ResidualDependence(residual, fz);
				if (first || score < best) {
					best = score;
					sink = candidate;
					first = false;
				}
			}
		}
		sinks.push_back(sink);
		remaining.erase(std::find(remaining.begin(), remaining.end(), sink));
	}
	if (!remaining.empty()) {
		out.order.push_back(remaining[0]);
	}
	for (idx_t a = sinks.size(); a-- > 0;) {
		out.order.push_back(sinks[a]);
	}

	// Before pruning: is any of this identified? Each variable's disturbance is what
	// the kernel regression on its predecessors leaves, and the fit "bends" when that
	// is meaningfully less than a straight line leaves. A source variable has no fit,
	// so it cannot bend, and its own values stand in for its disturbance.
	out.gaussian_p.assign(p, 1.0);
	for (idx_t pos = 0; pos < out.order.size(); pos++) {
		const idx_t target = out.order[pos];
		const vector<idx_t> predecessors(out.order.begin(), out.order.begin() + pos);
		if (predecessors.empty()) {
			out.gaussian_p[target] = JarqueBeraP(k.value[target]);
			continue;
		}
		k.SetFeatures(predecessors, fz);
		if (!k.Residual(k.value[target], predecessors, fz, residual)) {
			continue;
		}
		out.fitted++;
		double bent = 0.0;
		for (auto entry : residual) {
			bent += entry * entry;
		}
		bent /= static_cast<double>(k.m);
		// Five percent of the target's variance is the line between a fit that curves
		// and one that is a straight line with noise around it.
		if (bent < 0.95 * k.LinearResidualVariance(target, predecessors)) {
			out.bending++;
		}
		out.gaussian_p[target] = JarqueBeraP(residual);
	}

	// Prune. Without this every earlier variable in the order would be a parent of
	// every later one, which is an order dressed up as a graph.
	for (idx_t pos = 1; pos < out.order.size(); pos++) {
		const idx_t target = out.order[pos];
		for (idx_t a = 0; a < pos; a++) {
			const idx_t parent = out.order[a];
			vector<idx_t> rest;
			for (idx_t b = 0; b < pos; b++) {
				if (b != a) {
					rest.push_back(out.order[b]);
				}
			}
			if (k.PValue(parent, target, rest) < spec.alpha) {
				out.graph.adj[parent * p + target] = out.graph.adj[target * p + parent] = 1;
				out.graph.Orient(parent, target);
			} else {
				out.pruned++;
			}
		}
	}
	return out;
}

// --- RCD -------------------------------------------------------------------------
//
// Every method above assumes nothing unmeasured causes two of the measured
// variables, and that is the assumption most likely to be false. PC and FCI differ
// on it, and 'pc+lingam' exposes it indirectly - where a hidden cause is at work the
// two methods disagree, and the disagreement is reported as a conflict without
// saying what is wrong. RCD (Maeda and Shimizu, AISTATS 2020) says what is wrong.
//
// It rests on one clean fact about linear non-Gaussian models. If i causes j and
// nothing hidden links them, regressing j on i leaves a residual independent of i.
// So test both directions of every dependent pair:
//
//   exactly one direction independent  ->  that is the cause, and the pair is clean
//   neither direction independent      ->  no unconfounded model fits: a hidden
//                                          common cause, reported as <->
//   both directions independent        ->  the data admits either, reported o-o
//
// The repetition in the name is the outer loop: once i is known to be an ancestor of
// j, later tests regress it out first, which uncovers relations that were masked.
//
// This needs no refusal of its own, which is the nice property. On Gaussian
// disturbances both directions come back independent and everything is reported
// o-o - the method says it cannot tell, rather than guessing, and LiNGAM's and
// RESIT's refusals exist precisely because they cannot say that.

struct Rcd {
	Pag graph;
	idx_t confounded = 0;
	idx_t undecided = 0;
	idx_t directed = 0;
	idx_t rounds = 0;
	idx_t rows = 0;
};

Rcd RunRcd(const KernelTest &k, const DiscoverSpec &spec, const Knowledge &knowledge) {
	const idx_t p = k.p;
	Rcd out;
	out.rows = k.m;
	out.graph.p = p;
	out.graph.mark.assign(p * p, kNone);

	// Which pairs are related at all. A pair that is independent needs no explanation.
	vector<uint8_t> dependent(p * p, 0);
	const vector<idx_t> nothing;
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = i + 1; j < p; j++) {
			const bool linked = k.PValue(i, j, nothing) < spec.alpha;
			dependent[i * p + j] = dependent[j * p + i] = linked ? 1 : 0;
		}
	}

	// ancestor[i * p + j]: i is a cause of j, with nothing hidden between them.
	vector<uint8_t> ancestor(p * p, 0);
	vector<double> residual_effect, residual_cause;
	bool changed = true;
	while (changed && out.rounds < p + 2) {
		changed = false;
		out.rounds++;
		for (idx_t i = 0; i < p; i++) {
			for (idx_t j = i + 1; j < p; j++) {
				if (!dependent[i * p + j] || ancestor[i * p + j] || ancestor[j * p + i]) {
					continue;
				}
				double best[2] = {0.0, 0.0};
				for (int side = 0; side < 2; side++) {
					const idx_t cause = side == 0 ? i : j;
					const idx_t effect = side == 0 ? j : i;
					if (knowledge.Forbidden(cause, effect)) {
						best[side] = 0.0;
						continue;
					}
					// Regress what is already known out of both, then ask whether what
					// is left of the effect still knows anything about the cause.
					vector<idx_t> above_effect, above_cause;
					for (idx_t a = 0; a < p; a++) {
						if (a != cause && ancestor[a * p + effect]) {
							above_effect.push_back(a);
						}
						if (ancestor[a * p + cause]) {
							above_cause.push_back(a);
						}
					}
					above_effect.push_back(cause);
					k.LinearResidual(effect, above_effect, residual_effect);
					k.LinearResidual(cause, above_cause, residual_cause);
					best[side] = k.IndependenceP(residual_effect, residual_cause);
				}
				const bool forward = best[0] > spec.alpha, backward = best[1] > spec.alpha;
				if (forward && !backward) {
					ancestor[i * p + j] = 1;
					changed = true;
				} else if (backward && !forward) {
					ancestor[j * p + i] = 1;
					changed = true;
				}
			}
		}
	}

	// The ancestor relation includes i -> k -> j as i -> j, because once k is regressed
	// out the coefficient on i is zero either way. Its transitive reduction is the part
	// worth drawing.
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = 0; j < p; j++) {
			if (i == j || !ancestor[i * p + j]) {
				continue;
			}
			bool through = false;
			for (idx_t middle = 0; middle < p && !through; middle++) {
				through = middle != i && middle != j && ancestor[i * p + middle] && ancestor[middle * p + j];
			}
			if (through) {
				continue;
			}
			out.graph.Set(i, j, kArrow);
			out.graph.Set(j, i, kTail);
			out.directed++;
		}
	}

	// Whatever is left dependent and unexplained. Neither direction leaves an
	// independent residual, so no model without a hidden common cause fits the pair.
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = i + 1; j < p; j++) {
			if (!dependent[i * p + j] || out.graph.Adjacent(i, j)) {
				continue;
			}
			if (ancestor[i * p + j] || ancestor[j * p + i]) {
				continue; // an indirect path already drawn through its middle
			}
			double best[2] = {0.0, 0.0};
			for (int side = 0; side < 2; side++) {
				const idx_t cause = side == 0 ? i : j;
				const idx_t effect = side == 0 ? j : i;
				vector<idx_t> above_effect, above_cause;
				for (idx_t a = 0; a < p; a++) {
					if (a != cause && ancestor[a * p + effect]) {
						above_effect.push_back(a);
					}
					if (ancestor[a * p + cause]) {
						above_cause.push_back(a);
					}
				}
				above_effect.push_back(cause);
				k.LinearResidual(effect, above_effect, residual_effect);
				k.LinearResidual(cause, above_cause, residual_cause);
				best[side] = k.IndependenceP(residual_effect, residual_cause);
			}
			if (best[0] > spec.alpha && best[1] > spec.alpha) {
				// Either direction fits. This is what Gaussian disturbances look like.
				out.graph.Set(i, j, kCircle);
				out.graph.Set(j, i, kCircle);
				out.undecided++;
			} else {
				out.graph.Set(i, j, kArrow);
				out.graph.Set(j, i, kArrow);
				out.confounded++;
			}
		}
	}
	return out;
}

// --- one run: point graph plus bootstrap ------------------------------------------

struct Discovery {
	DiscoverSpec spec;
	//! test := 'mixed': which columns are two-valued (1) or ordinal, with 3 to 9 values (2), and whether the full-data
	//! latent correlations needed repair to be positive definite.
	vector<uint8_t> binary;
	bool repaired = false;
	NumericTable table;
	Cpdag graph;
	vector<double> adjacent;   // fraction of resamples with i and j adjacent
	vector<double> oriented;   // fraction with i -> j
	vector<double> undirected; // fraction with i - j left unoriented
	vector<string> warnings;
	//! Set by algorithm := 'fci' and 'rcd', which fill `pag` and `same_marks` instead
	//! of `graph`: both can say that a pair has a hidden common cause, which a CPDAG
	//! has no way to write down.
	bool fci = false;
	bool rcd = false;
	Pag pag;
	vector<double> same_marks; // fraction of resamples giving the pair the same two marks
	//! tiers := [...], resolved against the loaded columns.
	Knowledge knowledge;
	//! Set by algorithm := 'lingam', 'resit' and their unions with PC.
	bool lingam = false;
	bool resit = false;
	vector<idx_t> order;
	vector<double> gaussian_p;
	//! Set by algorithm := 'both': how the two methods landed on each pair.
	bool merged = false;
	vector<uint8_t> agreement;
};

//! How PC and LiNGAM landed on one pair, for algorithm := 'both'.
enum Agreement : uint8_t {
	kAgreeNone = 0,
	kAgreeBoth,            // both found the edge and give it the same direction
	kAgreeLingamDirection, // both found it; PC could not orient it, LiNGAM did
	kAgreeConflict,        // both found it and oriented it opposite ways
	kAgreePcOnly,
	kAgreeLingamOnly
};

const char *AgreementName(uint8_t code) {
	switch (code) {
	case kAgreeBoth:
		return "both";
	case kAgreeLingamDirection:
		return "oriented by lingam";
	case kAgreeConflict:
		return "conflict";
	case kAgreePcOnly:
		return "pc only";
	case kAgreeLingamOnly:
		return "lingam only";
	default:
		return "";
	}
}

//! The union of what PC and LiNGAM found. Where PC oriented an edge its
//! orientation stands, because it rests on the weaker assumptions; where PC left
//! one undirected, LiNGAM's direction fills it in; where the two point opposite
//! ways the edge goes back to undirected, which is the honest reading of two
//! methods contradicting each other, and sends it to review.
Cpdag MergeGraphs(const Cpdag &pc, const Cpdag &li, vector<uint8_t> *agreement) {
	const idx_t p = pc.p;
	Cpdag g;
	g.p = p;
	g.adj.assign(p * p, 0);
	g.head.assign(p * p, 0);
	g.conflicts = pc.conflicts;
	g.tier_conflicts = pc.tier_conflicts;
	if (agreement) {
		agreement->assign(p * p, kAgreeNone);
	}
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = i + 1; j < p; j++) {
			const bool in_pc = pc.Adjacent(i, j), in_li = li.Adjacent(i, j);
			if (!in_pc && !in_li) {
				continue;
			}
			g.adj[i * p + j] = g.adj[j * p + i] = 1;
			uint8_t code = kAgreeNone;
			if (in_pc && in_li) {
				const idx_t from = li.Directed(i, j) ? i : j, to = li.Directed(i, j) ? j : i;
				if (pc.Directed(from, to)) {
					g.Orient(from, to);
					code = kAgreeBoth;
				} else if (pc.Directed(to, from)) {
					code = kAgreeConflict;
				} else {
					g.Orient(from, to);
					code = kAgreeLingamDirection;
				}
			} else if (in_pc) {
				if (pc.Directed(i, j)) {
					g.Orient(i, j);
				} else if (pc.Directed(j, i)) {
					g.Orient(j, i);
				}
				code = kAgreePcOnly;
			} else {
				if (li.Directed(i, j)) {
					g.Orient(i, j);
				} else if (li.Directed(j, i)) {
					g.Orient(j, i);
				}
				code = kAgreeLingamOnly;
			}
			if (agreement) {
				(*agreement)[i * p + j] = (*agreement)[j * p + i] = code;
			}
		}
	}
	return g;
}

//! The test one sample of rows gets: Pearson or rank correlations with Fisher's
//! z, or latent correlations with their row influences for test := 'mixed'.
CiTest MakeTest(const Discovery &out, const vector<idx_t> &rows, bool *repaired = nullptr) {
	const auto &t = out.table;
	CiTest test;
	test.p = t.p;
	test.n = static_cast<double>(t.n);
	if (out.spec.test == "mixed") {
		MixedLatent(t, rows, out.binary, test, repaired);
	} else if (out.spec.test == "kernel") {
		test.kernel = true;
		test.features = MakeKernelTest(t, rows, out.spec.seed);
	} else {
		test.C = Correlation(t, rows);
	}
	return test;
}

//! For test := 'mixed': mark the two-valued columns binary (1) and those with 3
//! to 9 values ordinal (2), and refuse a sample too large to keep every row's
//! influence on every correlation. Nothing is left too coarse to read.
vector<string> ClassifyMixed(Discovery &out, const char *fn) {
	const auto &t = out.table;
	const idx_t pairs = t.p * (t.p - 1) / 2;
	const idx_t kBudget = idx_t(1) << 24;
	if (pairs * t.n > kBudget) {
		throw BinderException("duckdo: %s with test := 'mixed' keeps each row's influence on every correlation, and "
		                      "%llu pairs over %llu rows is more than it will hold; choose fewer columns, or test a "
		                      "sample of the rows (CREATE TABLE s AS SELECT * FROM t USING SAMPLE 20000)",
		                      fn, static_cast<unsigned long long>(pairs), static_cast<unsigned long long>(t.n));
	}
	out.binary.assign(t.p, 0);
	vector<string> coarse;
	for (idx_t j = 0; j < t.p; j++) {
		vector<double> seen;
		for (idx_t r = 0; r < t.n && seen.size() < 10; r++) {
			const double v = t.data[r * t.p + j];
			if (std::find(seen.begin(), seen.end(), v) == seen.end()) {
				seen.push_back(v);
			}
		}
		if (seen.size() == 2) {
			out.binary[j] = 1;
		} else if (seen.size() < 10) {
			out.binary[j] = 2;
		}
	}
	return coarse;
}

//! The FCI path through a discovery: the graph, its bootstrap, and warnings that
//! state what FCI assumes in place of what PC does.
void RunFciDiscovery(Discovery &out) {
	const auto &t = out.table;
	const idx_t p = t.p;
	out.fci = true;
	vector<idx_t> all(t.n);
	std::iota(all.begin(), all.end(), 0);
	out.pag = RunFci(MakeTest(out, all, &out.repaired), out.spec.alpha, out.spec.max_conditioning, out.knowledge);

	out.adjacent.assign(p * p, 0.0);
	out.same_marks.assign(p * p, 0.0);
	const idx_t reps = out.spec.bootstrap;
	if (reps > 0) {
		const auto order = ContentOrder(t);
		vector<Pag> draws(reps);
		// Seeded from (seed, rep), as the PC bootstrap is, so thread scheduling
		// cannot change which resample a replicate draws.
		ParallelJobs(reps, [&](idx_t rep) {
			std::mt19937_64 rng(static_cast<uint64_t>(out.spec.seed) ^ 0xD15C0DE5ULL ^ (rep * 0x9E3779B97F4A7C15ULL));
			std::uniform_int_distribution<idx_t> pick(0, t.n - 1);
			vector<idx_t> rows(t.n);
			for (auto &r : rows) {
				r = order[pick(rng)];
			}
			draws[rep] = RunFci(MakeTest(out, rows), out.spec.alpha, out.spec.max_conditioning, out.knowledge);
		});
		for (auto &g : draws) {
			for (idx_t i = 0; i < p; i++) {
				for (idx_t j = 0; j < p; j++) {
					if (i == j || !g.Adjacent(i, j)) {
						continue;
					}
					out.adjacent[i * p + j] += 1.0;
					if (out.pag.Adjacent(i, j) && PagEdge(g, i, j) == PagEdge(out.pag, i, j)) {
						out.same_marks[i * p + j] += 1.0;
					}
				}
			}
		}
		for (idx_t k = 0; k < p * p; k++) {
			out.adjacent[k] /= static_cast<double>(reps);
			out.same_marks[k] /= static_cast<double>(reps);
		}
	}

	out.warnings.push_back(StringUtil::Format(
	    "FCI with Fisher-z tests at alpha %g on %llu complete rows. It allows hidden common causes, and assumes "
	    "faithfulness, linear-Gaussian dependence and no selection bias. Its orientation rules are R1-R4 and R8-R10, "
	    "complete under those assumptions, so a circle that remains is one the data cannot settle",
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
	if (reps == 0) {
		out.warnings.push_back("bootstrap := 0, so nothing here says how fragile these edges are");
	}
	out.warnings.push_back("this is a proposal, not a graph: do_graph_create refuses do_discover_dot's output until "
	                       "its review marker is deleted and every edge FCI could not settle is given a direction "
	                       "or a hidden common cause");
}

//! The LiNGAM path: the graph, its bootstrap, and the check that decides whether
//! any of it means anything. algorithm := 'both' runs PC alongside it and reports
//! the union, marking every pair with how the two methods landed on it.
//! Turn the resampled graphs into the three fractions do_discover reports.
void Accumulate(Discovery &out, const vector<Cpdag> &draws) {
	const idx_t p = out.table.p;
	out.adjacent.assign(p * p, 0.0);
	out.oriented.assign(p * p, 0.0);
	out.undirected.assign(p * p, 0.0);
	if (draws.empty()) {
		return;
	}
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
		out.adjacent[k] /= static_cast<double>(draws.size());
		out.oriented[k] /= static_cast<double>(draws.size());
		out.undirected[k] /= static_cast<double>(draws.size());
	}
}

//! The RESIT path. algorithm := 'pc+resit' runs PC alongside it and reports the union,
//! the same way 'pc+lingam' does.
void RunResitDiscovery(Discovery &out) {
	const auto &t = out.table;
	const idx_t p = t.p;
	const bool merged = out.spec.algorithm == "pc+resit";
	out.resit = true;
	out.merged = merged;

	const auto order = ContentOrder(t);
	const auto fit = RunResit(MakeKernelTest(t, order, out.spec.seed), out.spec, out.knowledge);
	out.order = fit.order;
	out.gaussian_p = fit.gaussian_p;

	// The corner nothing can read: every fit a straight line, and more than one
	// disturbance that cannot be told from Gaussian. A linear-Gaussian model is the
	// textbook unidentifiable case, and RESIT does not fail quietly there - measured
	// over 15 draws of a four-variable chain it returned 0.47 of 3 true edges and 2.53
	// false ones, with no sign that anything was wrong. LiNGAM refuses the same corner
	// from the other side, so between them the refusal is symmetric.
	vector<idx_t> gaussian;
	for (idx_t j = 0; j < p; j++) {
		if (fit.gaussian_p[j] > 0.05) {
			gaussian.push_back(j);
		}
	}
	if (fit.fitted > 0 && fit.bending == 0 && gaussian.size() >= 2) {
		string names;
		for (auto j : gaussian) {
			names += (names.empty() ? "" : ", ") + t.names[j];
		}
		throw BinderException(
		    "duckdo: do_discover cannot run RESIT here. It reads direction from an effect that bends or a "
		    "disturbance that is not Gaussian, and this data has neither: not one of %llu fitted variables needed "
		    "more than a straight line, and %llu disturbances are indistinguishable from Gaussian (Jarque-Bera "
		    "p > 0.05): %s. Linear effects with Gaussian disturbances is the one case no method here can read, and "
		    "RESIT does not fail quietly in it - in simulation it returned 0.47 of 3 true edges and 2.53 false ones. "
		    "Use algorithm := 'pc', which will leave undecided edges undirected rather than guess",
		    static_cast<unsigned long long>(fit.fitted), static_cast<unsigned long long>(gaussian.size()), names);
	}

	vector<idx_t> all(t.n);
	std::iota(all.begin(), all.end(), 0);
	if (merged) {
		const auto pc = RunPc(MakeTest(out, all), out.spec.alpha, out.spec.max_conditioning, out.knowledge);
		out.graph = MergeGraphs(pc, fit.graph, &out.agreement);
	} else {
		out.graph = fit.graph;
	}

	const idx_t reps = out.spec.bootstrap;
	vector<Cpdag> draws(reps);
	if (reps > 0) {
		ParallelJobs(reps, [&](idx_t rep) {
			std::mt19937_64 rng(static_cast<uint64_t>(out.spec.seed) ^ 0xD15C0DE5ULL ^ (rep * 0x9E3779B97F4A7C15ULL));
			std::uniform_int_distribution<idx_t> pick(0, t.n - 1);
			vector<idx_t> rows(t.n);
			for (auto &r : rows) {
				r = order[pick(rng)];
			}
			const auto kernel = MakeKernelTest(t, rows, out.spec.seed);
			const auto draw = RunResit(kernel, out.spec, out.knowledge);
			if (!merged) {
				draws[rep] = draw.graph;
				return;
			}
			draws[rep] =
			    MergeGraphs(RunPc(MakeTest(out, rows), out.spec.alpha, out.spec.max_conditioning, out.knowledge),
			                draw.graph, nullptr);
		});
	}
	Accumulate(out, draws);

	string causal_order;
	for (auto j : fit.order) {
		causal_order += (causal_order.empty() ? "" : " < ") + t.names[j];
	}
	out.warnings.push_back(StringUtil::Format(
	    "%s on %llu complete rows. It assumes no cycles, no hidden common cause of any two variables, and that each "
	    "variable is some function of its causes plus a disturbance that is added rather than mixed in. The function "
	    "may bend however it likes and the disturbance may be Gaussian, which is where this differs from LiNGAM",
	    merged ? "PC-stable and RESIT together, their edges unioned," : "RESIT (Peters et al. 2014)",
	    static_cast<unsigned long long>(t.n)));
	out.warnings.push_back("the causal order RESIT found, most exogenous first: " + causal_order +
	                       ". It is found by peeling off a sink at a time - the variable whose residual, once the "
	                       "others are regressed out of it, is least dependent on them");
	out.warnings.push_back(StringUtil::Format(
	    "the order was found on %llu of the %llu rows, taken by stride in content order, with kernel ridge "
	    "regression on random Fourier features. %llu predecessor pair(s) were pruned at alpha %g, which is what "
	    "separates a graph from an order",
	    static_cast<unsigned long long>(fit.rows), static_cast<unsigned long long>(t.n),
	    static_cast<unsigned long long>(fit.pruned), out.spec.alpha));
	out.warnings.push_back(StringUtil::Format(
	    "%llu of %llu fitted variables need more than a straight line, and %llu disturbance(s) cannot be told from "
	    "Gaussian. Either one identifies the order; neither, and nothing here could have read this data",
	    static_cast<unsigned long long>(fit.bending), static_cast<unsigned long long>(fit.fitted),
	    static_cast<unsigned long long>(gaussian.size())));
	if (p > 10) {
		out.warnings.push_back(StringUtil::Format(
		    "pruning conditions on every earlier variable in the order, and with %llu variables that is a wide "
		    "conditioning set for %llu random features to represent; read the sparsest edges with that in mind",
		    static_cast<unsigned long long>(p - 1), static_cast<unsigned long long>(kSetFeatures)));
	}
	if (t.dropped > 0) {
		out.warnings.push_back(StringUtil::Format("%llu rows with a NULL in a selected column were dropped",
		                                          static_cast<unsigned long long>(t.dropped)));
	}
	if (t.n < 500) {
		out.warnings.push_back(StringUtil::Format(
		    "with %llu rows a nonparametric regression has little to work with, and the order rests on those "
		    "regressions",
		    static_cast<unsigned long long>(t.n)));
	}
	if (reps == 0) {
		out.warnings.push_back("bootstrap := 0, so nothing here says how fragile these edges are");
	}
	if (merged) {
		idx_t conflicts = 0, from_resit = 0;
		for (idx_t i = 0; i < p; i++) {
			for (idx_t j = i + 1; j < p; j++) {
				conflicts += out.agreement[i * p + j] == kAgreeConflict ? 1 : 0;
				from_resit += out.agreement[i * p + j] == kAgreeLingamDirection ? 1 : 0;
			}
		}
		out.warnings.push_back(StringUtil::Format(
		    "the agreement column says how the two methods landed on each pair. RESIT gave a direction to %llu "
		    "edge(s) PC could not orient; %llu edge(s) they oriented opposite ways, and those are reported "
		    "undirected",
		    static_cast<unsigned long long>(from_resit), static_cast<unsigned long long>(conflicts)));
	}
	out.warnings.push_back("this is a proposal, not a graph: do_graph_create refuses do_discover_dot's output until "
	                       "its review marker is deleted and every undirected edge is given a direction");
}

void RunLingamDiscovery(Discovery &out, const char *fn) {
	const auto &t = out.table;
	const idx_t p = t.p;
	const bool merged = out.spec.algorithm == "pc+lingam";
	out.lingam = true;
	out.merged = merged;

	// LiNGAM reads rows in content order, so the subsample its ordering statistics
	// take follows the data and not where the rows sit in the table.
	const auto order = ContentOrder(t);
	const auto C = Correlation(t, order);
	const auto fit = RunLingam(t, order, C, out.spec, out.knowledge, true);
	out.order = fit.order;
	out.gaussian_p = fit.gaussian_p;

	vector<idx_t> gaussian;
	for (idx_t j = 0; j < p; j++) {
		if (fit.gaussian_p[j] > 0.05) {
			gaussian.push_back(j);
		}
	}
	if (gaussian.size() >= 2) {
		string names;
		for (auto j : gaussian) {
			names += (names.empty() ? "" : ", ") + t.names[j];
		}
		throw BinderException(
		    "duckdo: %s cannot run LiNGAM here. It reads the causal order from the shape of each variable's "
		    "disturbance, and identifiability allows at most one of them to be Gaussian; %llu are indistinguishable "
		    "from Gaussian (Jarque-Bera p > 0.05): %s. On data like this LiNGAM returns a confident order that is "
		    "close to a coin flip - 1 of 40 simulated orders right, against 40 of 40 with non-Gaussian "
		    "disturbances - so this is a refusal rather than a warning. Use algorithm := 'pc', which assumes "
		    "nothing about the shape of the disturbances and leaves undecided edges undirected",
		    fn, static_cast<unsigned long long>(gaussian.size()), names);
	}

	vector<idx_t> all(t.n);
	std::iota(all.begin(), all.end(), 0);
	if (merged) {
		const auto pc = RunPc(MakeTest(out, all), out.spec.alpha, out.spec.max_conditioning, out.knowledge);
		out.graph = MergeGraphs(pc, fit.graph, &out.agreement);
	} else {
		out.graph = fit.graph;
	}

	out.adjacent.assign(p * p, 0.0);
	out.oriented.assign(p * p, 0.0);
	out.undirected.assign(p * p, 0.0);
	const idx_t reps = out.spec.bootstrap;
	if (reps > 0) {
		vector<Cpdag> draws(reps);
		// With groups, the resample is drawn inside each group, so every replicate has
		// the same group sizes the data has. Drawing across them would let a replicate
		// lose a whole dataset, and what it measured would no longer be the same
		// question.
		vector<vector<idx_t>> strata;
		if (t.Groups() > 1) {
			strata.assign(t.Groups(), vector<idx_t>());
			for (auto r : order) {
				strata[t.group[r]].push_back(r);
			}
		}
		ParallelJobs(reps, [&](idx_t rep) {
			std::mt19937_64 rng(static_cast<uint64_t>(out.spec.seed) ^ 0xD15C0DE5ULL ^ (rep * 0x9E3779B97F4A7C15ULL));
			vector<idx_t> rows;
			if (strata.empty()) {
				std::uniform_int_distribution<idx_t> pick(0, t.n - 1);
				rows.assign(t.n, 0);
				for (auto &r : rows) {
					r = order[pick(rng)];
				}
			} else {
				rows.reserve(t.n);
				for (auto &stratum : strata) {
					std::uniform_int_distribution<idx_t> pick(0, stratum.size() - 1);
					for (idx_t a = 0; a < stratum.size(); a++) {
						rows.push_back(stratum[pick(rng)]);
					}
				}
			}
			const auto resampled = Correlation(t, rows);
			const auto draw = RunLingam(t, rows, resampled, out.spec, out.knowledge, false);
			if (!merged) {
				draws[rep] = draw.graph;
				return;
			}
			// The same test PC would have built for itself on these rows. For 'pearson'
			// that is the correlation matrix LiNGAM has already computed.
			CiTest test;
			if (out.spec.test == "kernel") {
				test = MakeTest(out, rows);
			} else {
				test.p = p;
				test.n = static_cast<double>(t.n);
				test.C = resampled;
			}
			draws[rep] =
			    MergeGraphs(RunPc(test, out.spec.alpha, out.spec.max_conditioning, out.knowledge), draw.graph, nullptr);
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

	string causal_order;
	for (auto j : fit.order) {
		causal_order += (causal_order.empty() ? "" : " < ") + t.names[j];
	}
	out.warnings.push_back(StringUtil::Format(
	    "%s on %llu complete rows. It assumes linear effects, no cycles, no hidden common cause of any two "
	    "variables, and non-Gaussian disturbances. That list is strictly longer than PC's, and buys one thing in "
	    "exchange: every edge is oriented, including the ones PC has no way to settle",
	    merged ? "PC-stable and DirectLiNGAM together, their edges unioned," : "DirectLiNGAM (Shimizu et al. 2011)",
	    static_cast<unsigned long long>(t.n)));
	out.warnings.push_back("the causal order LiNGAM found, most exogenous first: " + causal_order +
	                       ". Every edge below runs forward along it, so an order that reads backwards to you is "
	                       "the thing to argue with, not the individual edges");
	if (t.Groups() > 1) {
		string names;
		for (auto &name : t.group_names) {
			names += (names.empty() ? "" : ", ") + name;
		}
		out.warnings.push_back(StringUtil::Format(
		    "that order is one order across %llu datasets, from groups := '%s': %s. They were peeled together, each "
		    "weighted by its own row count, which is what constrains them to share it (Shimizu 2012). Coefficients "
		    "were fitted separately in each, and an edge is reported when at least %llu of them find it - requiring "
		    "all would contradict the premise that the coefficients differ",
		    static_cast<unsigned long long>(t.Groups()), out.spec.groups, names,
		    static_cast<unsigned long long>((t.Groups() + 1) / 2)));
	}
	if (!gaussian.empty()) {
		out.warnings.push_back(StringUtil::Format(
		    "the disturbance of '%s' cannot be told from Gaussian (Jarque-Bera p %.3f). Identifiability allows one, "
		    "so this is not a refusal, but its place in the order rests on less than the others do",
		    t.names[gaussian[0]], fit.gaussian_p[gaussian[0]]));
	}
	out.warnings.push_back(StringUtil::Format(
	    "coefficients below min_effect %g, as a share of the effect's own standard deviation, were not counted as "
	    "edges, nor were those a Wald test could not separate from zero at alpha %g",
	    out.spec.min_effect, out.spec.alpha));
	if (fit.order_rows < t.n) {
		out.warnings.push_back(StringUtil::Format(
		    "the causal order was found on %llu of the %llu rows, taken by stride in content order; the coefficients "
		    "use every row",
		    static_cast<unsigned long long>(fit.order_rows), static_cast<unsigned long long>(t.n)));
	}
	if (t.dropped > 0) {
		out.warnings.push_back(StringUtil::Format("%llu rows with a NULL in a selected column were dropped",
		                                          static_cast<unsigned long long>(t.dropped)));
	}
	if (t.n < 500) {
		out.warnings.push_back(StringUtil::Format(
		    "with %llu rows the entropy comparisons that fix the order are themselves noisy, and in simulation the "
		    "order stopped being reliable well before the disturbance check started objecting",
		    static_cast<unsigned long long>(t.n)));
	}
	if (reps == 0) {
		out.warnings.push_back("bootstrap := 0, so nothing here says how fragile these edges are");
	}
	if (merged) {
		idx_t conflicts = 0, from_lingam = 0;
		for (idx_t i = 0; i < p; i++) {
			for (idx_t j = i + 1; j < p; j++) {
				conflicts += out.agreement[i * p + j] == kAgreeConflict ? 1 : 0;
				from_lingam += out.agreement[i * p + j] == kAgreeLingamDirection ? 1 : 0;
			}
		}
		out.warnings.push_back(StringUtil::Format(
		    "the agreement column says how the two methods landed on each pair. LiNGAM gave a direction to %llu edge"
		    "(s) PC could not orient; %llu edge(s) they oriented opposite ways, and those are reported undirected, "
		    "because two methods contradicting each other is not evidence for either answer",
		    static_cast<unsigned long long>(from_lingam), static_cast<unsigned long long>(conflicts)));
	}
	out.warnings.push_back("this is a proposal, not a graph: do_graph_create refuses do_discover_dot's output until "
	                       "its review marker is deleted and every undirected edge is given a direction");
}

//! The RCD path. Its output is a partial ancestral graph, so it reuses FCI's rows and
//! its proposal, down to turning <-> into a latent node do_identify already understands.
void RunRcdDiscovery(Discovery &out) {
	const auto &t = out.table;
	const idx_t p = t.p;
	out.fci = true;
	out.rcd = true;

	const auto order = ContentOrder(t);
	const auto fit = RunRcd(MakeKernelTest(t, order, out.spec.seed), out.spec, out.knowledge);
	out.pag = fit.graph;

	out.adjacent.assign(p * p, 0.0);
	out.same_marks.assign(p * p, 0.0);
	const idx_t reps = out.spec.bootstrap;
	if (reps > 0) {
		vector<Pag> draws(reps);
		ParallelJobs(reps, [&](idx_t rep) {
			std::mt19937_64 rng(static_cast<uint64_t>(out.spec.seed) ^ 0xD15C0DE5ULL ^ (rep * 0x9E3779B97F4A7C15ULL));
			std::uniform_int_distribution<idx_t> pick(0, t.n - 1);
			vector<idx_t> rows(t.n);
			for (auto &r : rows) {
				r = order[pick(rng)];
			}
			draws[rep] = RunRcd(MakeKernelTest(t, rows, out.spec.seed), out.spec, out.knowledge).graph;
		});
		for (auto &g : draws) {
			for (idx_t i = 0; i < p; i++) {
				for (idx_t j = 0; j < p; j++) {
					if (i == j || !g.Adjacent(i, j)) {
						continue;
					}
					out.adjacent[i * p + j] += 1.0;
					if (out.pag.Adjacent(i, j) && PagEdge(g, i, j) == PagEdge(out.pag, i, j)) {
						out.same_marks[i * p + j] += 1.0;
					}
				}
			}
		}
		for (idx_t k = 0; k < p * p; k++) {
			out.adjacent[k] /= static_cast<double>(reps);
			out.same_marks[k] /= static_cast<double>(reps);
		}
	}

	out.warnings.push_back(StringUtil::Format(
	    "RCD (Maeda and Shimizu 2020) on %llu complete rows, %llu of them read. It assumes linear effects, no cycles "
	    "and non-Gaussian disturbances - but not that the measured variables are unconfounded, which is what "
	    "separates it from LiNGAM. For every related pair it asks whether regressing either one on the other leaves "
	    "an independent residual",
	    static_cast<unsigned long long>(t.n), static_cast<unsigned long long>(fit.rows)));
	out.warnings.push_back(StringUtil::Format(
	    "%llu pair(s) had a direction, %llu had a hidden common cause and are written <->, and %llu admitted either "
	    "direction and are written o-o. Settled in %llu round(s): once a cause is known it is regressed out before "
	    "the later tests, which is what the repetition in the name is for",
	    static_cast<unsigned long long>(fit.directed), static_cast<unsigned long long>(fit.confounded),
	    static_cast<unsigned long long>(fit.undecided), static_cast<unsigned long long>(fit.rounds)));
	out.warnings.push_back("RCD needs no refusal of its own: where the disturbances are Gaussian both directions "
	                       "leave an independent residual and every pair comes back o-o, so it says it cannot tell "
	                       "rather than guessing. A run that is mostly o-o is that, not a finding");
	if (t.dropped > 0) {
		out.warnings.push_back(StringUtil::Format("%llu rows with a NULL in a selected column were dropped",
		                                          static_cast<unsigned long long>(t.dropped)));
	}
	if (reps == 0) {
		out.warnings.push_back("bootstrap := 0, so nothing here says how fragile these edges are");
	}
	out.warnings.push_back("this is a proposal, not a graph: do_graph_create refuses do_discover_dot's output until "
	                       "its review marker is deleted and every edge RCD could not settle is given a direction "
	                       "or a hidden common cause");
}

//! What the tiers asserted, what they settled, and where the data disagreed.
void NoteTiers(Discovery &out) {
	if (!out.knowledge.any) {
		return;
	}
	const idx_t p = out.table.p;
	idx_t settled = 0;
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = 0; j < p; j++) {
			const bool adjacent = out.fci ? out.pag.Adjacent(i, j) : out.graph.Adjacent(i, j);
			settled += adjacent && out.knowledge.Settles(i, j) ? 1 : 0;
		}
	}
	string free_columns;
	for (idx_t j = 0; j < p; j++) {
		if (out.knowledge.tier[j] == kNoTier) {
			free_columns += (free_columns.empty() ? "" : ", ") + out.table.names[j];
		}
	}
	out.warnings.push_back(StringUtil::Format(
	    "tiers settled the direction of %llu edge(s) before any test ran. Those directions are yours, not the "
	    "data's: nothing below can contradict them, and a tier that is wrong makes a wrong graph that looks "
	    "well-supported",
	    static_cast<unsigned long long>(settled)));
	if (!free_columns.empty()) {
		out.warnings.push_back("no tier was given for: " + free_columns +
		                       ". They are unconstrained, so their edges are oriented by the data alone");
	}
	if (!out.fci && out.graph.tier_conflicts > 0) {
		out.warnings.push_back(StringUtil::Format(
		    "%llu v-structure(s) wanted an orientation the tiers forbid, and the tiers won. The data is saying "
		    "something your stated order rules out; one of the two is wrong",
		    static_cast<unsigned long long>(out.graph.tier_conflicts)));
	}
}

//! With test := 'rank' or 'mixed', say what the tests assume in place of
//! linear-Gaussian dependence, and name the columns each reads specially.
void NoteTest(Discovery &out, const vector<string> &coarse) {
	if (out.spec.test == "kernel") {
		for (auto &warning : out.warnings) {
			warning = StringUtil::Replace(warning, "Fisher-z tests",
			                              "kernel tests of conditional independence (RCoT, random Fourier features)");
			warning =
			    StringUtil::Replace(warning, "linear-Gaussian dependence", "no particular shape of dependence at all");
		}
		out.warnings.push_back(StringUtil::Format(
		    "the kernel test read %llu of the %llu rows, taken by stride in content order. It sees a dependence that "
		    "bends, which every other test here misses: on a true chain with a squared middle variable, Fisher's z "
		    "rejects the true conditional independence every time and this holds its level",
		    static_cast<unsigned long long>(std::min(out.table.n, kKernelRows)),
		    static_cast<unsigned long long>(out.table.n)));
		out.warnings.push_back("the random features are drawn once from duckdo_seed and reused by every resample, so "
		                       "what the stabilities measure is the sample rather than the draw");
		return;
	}
	if (out.spec.test == "rank") {
		for (auto &warning : out.warnings) {
			warning = StringUtil::Replace(warning, "Fisher-z tests", "rank-based Fisher-z tests (normal scores)");
			warning = StringUtil::Replace(
			    warning, "linear-Gaussian dependence",
			    "a Gaussian copula (that monotone transforms of the variables are jointly Gaussian)");
		}
		for (auto &name : coarse) {
			out.warnings.push_back(StringUtil::Format(
			    "column '%s' has fewer than 10 distinct values; rank-based tests assume continuous variables, and "
			    "ties this heavy weaken them. test := 'mixed' reads a column with 2 to 9 values as binary or ordinal",
			    name));
		}
		return;
	}
	if (out.spec.test != "mixed") {
		return;
	}
	for (auto &warning : out.warnings) {
		warning = StringUtil::Replace(warning, "Fisher-z tests",
		                              "Wald tests of latent partial correlation (Kendall's tau bridges)");
		warning = StringUtil::Replace(warning, "linear-Gaussian dependence",
		                              "a latent Gaussian copula (each continuous column a monotone transform, and each "
		                              "two-valued column a threshold, of jointly Gaussian variables)");
	}
	string names;
	for (idx_t j = 0; j < out.table.p; j++) {
		if (out.binary[j] == 1) {
			names += (names.empty() ? "" : ", ") + out.table.names[j];
		}
	}
	if (!names.empty()) {
		out.warnings.push_back("read as binary, each the threshold of a latent Gaussian variable: " + names);
	}
	string ordinal;
	for (idx_t j = 0; j < out.table.p; j++) {
		if (out.binary[j] == 2) {
			ordinal += (ordinal.empty() ? "" : ", ") + out.table.names[j];
		}
	}
	if (!ordinal.empty()) {
		out.warnings.push_back(
		    "read as ordinal, each a latent Gaussian variable cut at thresholds, with polychoric and "
		    "polyserial correlations: " +
		    ordinal);
	}
	if (out.repaired) {
		out.warnings.push_back("the latent correlations were not positive definite together, so their eigenvalues "
		                       "were floored at 1e-4 before testing");
	}
}

Discovery RunDiscovery(ClientContext &context, TableFunctionBindInput &input, const char *fn) {
	Discovery out;
	out.spec = ParseDiscover(context, input, fn);
	out.table = LoadNumeric(context, out.spec, fn);
	out.knowledge = BuildKnowledge(out.table, out.spec, fn);
	vector<string> coarse;
	if (out.spec.test == "rank") {
		coarse = RankNormalScores(out.table);
	} else if (out.spec.test == "mixed") {
		coarse = ClassifyMixed(out, fn);
	}
	if (out.spec.algorithm == "fci") {
		RunFciDiscovery(out);
		NoteTest(out, coarse);
		NoteTiers(out);
		return out;
	}
	if (out.spec.algorithm == "lingam" || out.spec.algorithm == "pc+lingam") {
		RunLingamDiscovery(out, fn);
		NoteTest(out, coarse);
		NoteTiers(out);
		return out;
	}
	if (out.spec.algorithm == "rcd") {
		RunRcdDiscovery(out);
		NoteTiers(out);
		return out;
	}
	if (out.spec.algorithm == "resit" || out.spec.algorithm == "pc+resit") {
		RunResitDiscovery(out);
		NoteTest(out, coarse);
		NoteTiers(out);
		return out;
	}
	const auto &t = out.table;
	const idx_t p = t.p;

	vector<idx_t> all(t.n);
	std::iota(all.begin(), all.end(), 0);
	out.graph = RunPc(MakeTest(out, all, &out.repaired), out.spec.alpha, out.spec.max_conditioning, out.knowledge);

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
			draws[rep] = RunPc(MakeTest(out, rows), out.spec.alpha, out.spec.max_conditioning, out.knowledge);
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
	NoteTest(out, coarse);
	NoteTiers(out);
	return out;
}

Value WarningValue(const vector<string> &warnings) {
	vector<Value> out;
	for (auto &warning : warnings) {
		out.push_back(Value(warning));
	}
	return Value::LIST(LogicalType::VARCHAR, std::move(out));
}

//! do_discover's rows for an FCI graph: the same columns as PC, with each edge's
//! two marks - -->, <->, o-> or o-o - in place of -> and --.
unique_ptr<FunctionData> BindFciEdges(const Discovery &found, vector<LogicalType> &return_types,
                                      vector<string> &names) {
	const auto &t = found.table;
	const auto &g = found.pag;
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
			if (!in_graph && !(resampled && stability >= 0.25)) {
				continue;
			}
			idx_t from = i, to = j;
			string edge = "absent";
			if (in_graph) {
				ReadForward(g, from, to);
				edge = PagEdge(g, from, to);
			}
			bind->rows.push_back(
			    {Value(t.names[from]), Value(t.names[to]), Value(edge), Value::BOOLEAN(in_graph),
			     resampled ? Value::DOUBLE(stability) : Value(LogicalType::DOUBLE),
			     resampled && in_graph ? Value::DOUBLE(found.same_marks[i * p + j]) : Value(LogicalType::DOUBLE),
			     warnings});
		}
	}
	return std::move(bind);
}

unique_ptr<FunctionData> BindDiscover(ClientContext &context, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names) {
	auto found = RunDiscovery(context, input, "do_discover");
	if (found.fci) {
		return BindFciEdges(found, return_types, names);
	}
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
	// algorithm := 'both' has one more thing to say about every pair than the
	// single-method runs do, so it says it in a column of its own.
	if (found.merged) {
		names.insert(names.begin() + 6, "agreement");
		return_types.insert(return_types.begin() + 6, LogicalType::VARCHAR);
	}
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
			vector<Value> row = {Value(source),
			                     Value(target),
			                     Value(edge),
			                     Value::BOOLEAN(in_graph),
			                     resampled ? Value::DOUBLE(stability) : Value(LogicalType::DOUBLE),
			                     resampled && in_graph ? Value::DOUBLE(orientation) : Value(LogicalType::DOUBLE)};
			if (found.merged) {
				const char *code = in_graph ? AgreementName(found.agreement[i * p + j]) : "";
				row.push_back(code[0] != '\0' ? Value(code) : Value(LogicalType::VARCHAR));
			}
			row.push_back(warnings);
			bind->rows.push_back(std::move(row));
		}
	}
	return std::move(bind);
}

string QuoteNode(const string &name) {
	string cleaned = name;
	std::replace(cleaned.begin(), cleaned.end(), '"', '\'');
	return "\"" + cleaned + "\"";
}

//! do_discover_dot's proposal for an FCI graph. -->  becomes ->. <-> becomes a
//! node marked [latent] with an arrow into each end, which do_graph_create and
//! do_identify already understand as a hidden common cause. Every edge with a
//! circle is written --, which do_graph_create refuses until someone settles it,
//! with a comment saying which readings the marks allow.
unique_ptr<FunctionData> BindFciDot(const Discovery &found, vector<LogicalType> &return_types, vector<string> &names) {
	const auto &t = found.table;
	const auto &g = found.pag;
	const idx_t p = t.p;
	const bool resampled = found.spec.bootstrap > 0;

	string dot = "digraph discovered {\n";
	dot += "  // do_discover: unreviewed - delete this line only after reviewing every edge below.\n";
	dot += StringUtil::Format("  // FCI, alpha %g, %llu bootstrap resamples, %llu complete rows. It allows hidden\n",
	                          found.spec.alpha, static_cast<unsigned long long>(found.spec.bootstrap),
	                          static_cast<unsigned long long>(t.n));
	dot += "  // common causes, and assumes faithfulness, linear-Gaussian dependence and no selection\n";
	dot += "  // bias. An edge that is wrong here becomes a wrong adjustment set in do_identify.\n";
	for (idx_t j = 0; j < p; j++) {
		dot += "  " + QuoteNode(t.names[j]) + ";\n";
	}
	idx_t directed = 0, undirected = 0, bidirected = 0;
	for (idx_t i = 0; i < p; i++) {
		for (idx_t j = i + 1; j < p; j++) {
			if (!g.Adjacent(i, j)) {
				continue;
			}
			idx_t from = i, to = j;
			ReadForward(g, from, to);
			const string edge = PagEdge(g, from, to);
			const string &a = t.names[from];
			const string &b = t.names[to];
			const string note =
			    resampled ? StringUtil::Format("in %.0f%% of resamples, these marks in %.0f%%",
			                                   100.0 * found.adjacent[i * p + j], 100.0 * found.same_marks[i * p + j])
			              : string();
			if (edge == "-->") {
				dot += "  " + QuoteNode(a) + " -> " + QuoteNode(b) + ";" + (note.empty() ? "" : "  // " + note) + "\n";
				directed++;
			} else if (edge == "<->") {
				const string hidden = QuoteNode("hidden(" + a + ", " + b + ")");
				dot += "  " + hidden + " [latent];  // " + a + " <-> " + b +
				       ": neither causes the other, and something unmeasured causes both" +
				       (note.empty() ? "" : "; " + note) + "\n";
				dot += "  " + hidden + " -> " + QuoteNode(a) + ";\n";
				dot += "  " + hidden + " -> " + QuoteNode(b) + ";\n";
				bidirected++;
			} else {
				const string readings = edge == "o->" ? b + " does not cause " + a + ": write " + a + " -> " + b +
				                                            ", or give them a hidden common cause, or both"
				                                      : "the data did not decide: write " + a + " -> " + b + ", or " +
				                                            b + " -> " + a + ", or give them a hidden common cause";
				dot += "  " + QuoteNode(a) + " -- " + QuoteNode(b) + ";  // " + a + " " + edge + " " + b + ": " +
				       readings + (note.empty() ? "" : "; " + note) + "\n";
				undirected++;
			}
		}
	}
	dot += "}\n";

	if (found.spec.test == "rank") {
		dot = StringUtil::Replace(dot, "linear-Gaussian dependence", "a Gaussian copula (rank-based tests)");
	} else if (found.spec.test == "mixed") {
		dot = StringUtil::Replace(dot, "linear-Gaussian dependence", "a latent Gaussian copula (mixed-data tests)");
	}
	names = {"dot", "n_nodes", "n_directed", "n_undirected", "n_bidirected"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
	                LogicalType::BIGINT};
	auto bind = make_uniq<ResultBindData>();
	bind->rows.push_back(
	    {Value(dot), Value::BIGINT(static_cast<int64_t>(p)), Value::BIGINT(static_cast<int64_t>(directed)),
	     Value::BIGINT(static_cast<int64_t>(undirected)), Value::BIGINT(static_cast<int64_t>(bidirected))});
	return std::move(bind);
}

unique_ptr<FunctionData> BindDiscoverDot(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	auto found = RunDiscovery(context, input, "do_discover_dot");
	if (found.fci) {
		return BindFciDot(found, return_types, names);
	}
	const auto &t = found.table;
	const auto &g = found.graph;
	const idx_t p = t.p;
	const bool resampled = found.spec.bootstrap > 0;

	string dot = "digraph discovered {\n";
	dot += "  // do_discover: unreviewed - delete this line only after reviewing every edge below.\n";
	if (found.resit) {
		dot += StringUtil::Format("  // %s, alpha %g, %llu bootstrap resamples, %llu complete rows.\n",
		                          found.merged ? "PC-stable and RESIT, unioned" : "RESIT", found.spec.alpha,
		                          static_cast<unsigned long long>(found.spec.bootstrap),
		                          static_cast<unsigned long long>(t.n));
		dot += "  // It assumes no hidden common causes, no cycles, and a disturbance added rather than\n";
		dot += "  // mixed in. The effect itself may bend, and the disturbance may be Gaussian.\n";
		string causal_order;
		for (auto j : found.order) {
			causal_order += (causal_order.empty() ? "" : " < ") + t.names[j];
		}
		dot += "  // Causal order, most exogenous first: " + causal_order + "\n";
	} else if (found.lingam) {
		dot +=
		    StringUtil::Format("  // %s, alpha %g, min_effect %g, %llu bootstrap resamples, %llu complete rows.\n",
		                       found.merged ? "PC-stable and DirectLiNGAM, unioned" : "DirectLiNGAM", found.spec.alpha,
		                       found.spec.min_effect, static_cast<unsigned long long>(found.spec.bootstrap),
		                       static_cast<unsigned long long>(t.n));
		dot += "  // It assumes no hidden common causes, linear effects, no cycles, and non-Gaussian\n";
		dot += "  // disturbances. The last is what lets it orient every edge, and it is the one to\n";
		dot += "  // check: the order below is only as good as the shape of the data.\n";
		string causal_order;
		for (auto j : found.order) {
			causal_order += (causal_order.empty() ? "" : " < ") + t.names[j];
		}
		dot += "  // Causal order, most exogenous first: " + causal_order + "\n";
	} else {
		dot += StringUtil::Format(
		    "  // PC-stable, alpha %g, %llu bootstrap resamples, %llu complete rows. It assumes no\n", found.spec.alpha,
		    static_cast<unsigned long long>(found.spec.bootstrap), static_cast<unsigned long long>(t.n));
		dot += "  // hidden common causes, faithfulness and linear-Gaussian dependence. An edge that is wrong\n";
		dot += "  // here becomes a wrong adjustment set in do_identify and do_validate.\n";
	}
	if (found.knowledge.any) {
		dot += "  // tiers were given, so some of the directions below are yours rather than the data's.\n";
	}
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

	if (found.spec.test == "rank") {
		dot = StringUtil::Replace(dot, "linear-Gaussian dependence", "a Gaussian copula (rank-based tests)");
	} else if (found.spec.test == "mixed") {
		dot = StringUtil::Replace(dot, "linear-Gaussian dependence", "a latent Gaussian copula (mixed-data tests)");
	}
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
	fn.named_parameters["algorithm"] = LogicalType::VARCHAR;
	fn.named_parameters["test"] = LogicalType::VARCHAR;
	fn.named_parameters["min_effect"] = LogicalType::DOUBLE;
	fn.named_parameters["tiers"] = LogicalType::LIST(LogicalType::LIST(LogicalType::VARCHAR));
	fn.named_parameters["groups"] = LogicalType::VARCHAR;
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
