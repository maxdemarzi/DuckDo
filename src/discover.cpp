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
	if (spec.algorithm != "pc" && spec.algorithm != "fci" && spec.algorithm != "lingam" && spec.algorithm != "both") {
		throw BinderException(
		    "duckdo: algorithm must be 'pc', 'fci', 'lingam' or 'both', not '%s'. 'fci' allows hidden common causes; "
		    "'pc' assumes there are none; 'lingam' orients every edge from non-Gaussian disturbances, under "
		    "assumptions stronger than PC's; 'both' runs 'pc' and 'lingam' and says where they differ",
		    spec.algorithm);
	}
	const bool lingam = spec.algorithm == "lingam" || spec.algorithm == "both";
	entry = named.find("min_effect");
	if (entry != named.end() && !entry->second.IsNull()) {
		if (!lingam) {
			throw BinderException("duckdo: min_effect is the smallest coefficient LiNGAM counts as an edge, so it "
			                      "applies to algorithm := 'lingam' or 'both', not '%s'",
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
	entry = named.find("test");
	if (entry != named.end() && !entry->second.IsNull()) {
		spec.test = StringUtil::Lower(entry->second.ToString());
	}
	if (spec.test != "pearson" && spec.test != "rank" && spec.test != "mixed") {
		throw BinderException("duckdo: test must be 'pearson', 'rank' or 'mixed', not '%s'. 'rank' tests on normal "
		                      "scores, which only assumes that monotone transforms of the variables are jointly "
		                      "Gaussian; 'mixed' also reads each two-valued column as the threshold of a latent "
		                      "Gaussian variable",
		                      spec.test);
	}
	if (lingam && spec.test != "pearson") {
		throw BinderException("duckdo: algorithm := '%s' cannot run with test := '%s'. LiNGAM reads direction from "
		                      "the shape of each variable's disturbance, and '%s' replaces every column with scores "
		                      "that are Gaussian by construction - exactly the case LiNGAM cannot read",
		                      spec.algorithm, spec.test, spec.test);
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

	vector<vector<double>> Z(p, vector<double>(m));
	for (idx_t j = 0; j < p; j++) {
		for (idx_t r = 0; r < m; r++) {
			Z[j][r] = t.data[sample[r] * p + j];
		}
		Standardise(Z[j]);
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
			vector<double> entropy(pool.size());
			for (idx_t a = 0; a < pool.size(); a++) {
				entropy[a] = EntropyApprox(Z[pool[a]]);
			}
			// R(a, b) is the likelihood ratio between "b causes a" and "a causes b".
			// It is exactly antisymmetric, so one pass over each pair scores both.
			vector<double> penalty(pool.size(), 0.0);
			for (idx_t a = 0; a < pool.size(); a++) {
				for (idx_t b = a + 1; b < pool.size(); b++) {
					UniResidual(Z[pool[a]], Z[pool[b]], forward);
					UniResidual(Z[pool[b]], Z[pool[a]], backward);
					const double r = (entropy[a] + EntropyApprox(backward)) - (entropy[b] + EntropyApprox(forward));
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
		peeled = Z[next];
		for (auto i : remaining) {
			UniResidual(Z[i], peeled, forward);
			Z[i] = forward;
		}
	}
	if (!remaining.empty()) {
		out.order.push_back(remaining[0]);
	}

	// Each variable on its predecessors in that order, over every row. The columns
	// are standardised, so min_effect reads as "a one-standard-deviation move in
	// the cause shifts the effect by this much of its own standard deviation".
	const double n = static_cast<double>(rows.size());
	vector<vector<double>> coefficients(p);
	for (idx_t pos = 1; pos < out.order.size(); pos++) {
		const idx_t j = out.order[pos];
		vector<idx_t> S(out.order.begin(), out.order.begin() + pos);
		const idx_t k = S.size();
		vector<double> M(k * k), rhs(k), inv;
		for (idx_t a = 0; a < k; a++) {
			rhs[a] = C[S[a] * p + j];
			for (idx_t b = 0; b < k; b++) {
				M[a * k + b] = C[S[a] * p + S[b]];
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
		coefficients[j] = coef;
		// The target has unit variance, so what the predecessors do not explain is
		// what is left of that 1, scaled from a mean square to an unbiased one.
		const double dof = std::max(n - static_cast<double>(k), 1.0);
		const double sigma2 = std::max(1.0 - explained, 1e-12) * n / dof;
		for (idx_t a = 0; a < k; a++) {
			const double variance = std::max(sigma2 * inv[a * k + a] / n, 1e-300);
			const double se = std::sqrt(variance);
			if (std::fabs(coef[a]) >= spec.min_effect && NormalTwoSidedP(coef[a] / se) < spec.alpha) {
				out.graph.adj[S[a] * p + j] = out.graph.adj[j * p + S[a]] = 1;
				out.graph.Orient(S[a], j);
			}
		}
	}

	if (!disturbances) {
		return out;
	}
	// What is left of each variable once its predecessors are regressed out. The
	// coefficients here are every predecessor's, not only the ones that cleared
	// min_effect: the disturbance is what the model does not explain, and a
	// coefficient dropped for being small still explained its part.
	vector<vector<double>> raw(p, vector<double>(m));
	for (idx_t j = 0; j < p; j++) {
		for (idx_t r = 0; r < m; r++) {
			raw[j][r] = t.data[sample[r] * p + j];
		}
		Standardise(raw[j]);
	}
	vector<double> residual(m);
	for (idx_t pos = 0; pos < out.order.size(); pos++) {
		const idx_t j = out.order[pos];
		residual = raw[j];
		const auto &coef = coefficients[j];
		for (idx_t a = 0; a < coef.size(); a++) {
			const idx_t src = out.order[a];
			for (idx_t r = 0; r < m; r++) {
				residual[r] -= coef[a] * raw[src][r];
			}
		}
		out.gaussian_p[j] = JarqueBeraP(residual);
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
	//! Set by algorithm := 'fci', which fills `pag` and `same_marks` instead.
	bool fci = false;
	Pag pag;
	vector<double> same_marks; // fraction of resamples giving the pair the same two marks
	//! tiers := [...], resolved against the loaded columns.
	Knowledge knowledge;
	//! Set by algorithm := 'lingam' and 'both'.
	bool lingam = false;
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
void RunLingamDiscovery(Discovery &out, const char *fn) {
	const auto &t = out.table;
	const idx_t p = t.p;
	const bool merged = out.spec.algorithm == "both";
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
		ParallelJobs(reps, [&](idx_t rep) {
			std::mt19937_64 rng(static_cast<uint64_t>(out.spec.seed) ^ 0xD15C0DE5ULL ^ (rep * 0x9E3779B97F4A7C15ULL));
			std::uniform_int_distribution<idx_t> pick(0, t.n - 1);
			vector<idx_t> rows(t.n);
			for (auto &r : rows) {
				r = order[pick(rng)];
			}
			const auto resampled = Correlation(t, rows);
			const auto draw = RunLingam(t, rows, resampled, out.spec, out.knowledge, false);
			if (!merged) {
				draws[rep] = draw.graph;
				return;
			}
			// The same correlations PC would have built for itself on these rows.
			CiTest test;
			test.p = p;
			test.n = static_cast<double>(t.n);
			test.C = resampled;
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
	if (out.spec.algorithm == "lingam" || out.spec.algorithm == "both") {
		RunLingamDiscovery(out, fn);
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
	if (found.lingam) {
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
