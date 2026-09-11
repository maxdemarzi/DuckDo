//===----------------------------------------------------------------------===//
//                         DuckDo - causal inference in DuckDB
//
// duckdo/frame.hpp
//
// The causal frame: one binder and one encoder that every estimator reuses.
// A CausalFrame is an encoded float matrix plus a treatment vector, an outcome
// vector, column metadata and provenance back to the source relation. Getting
// this right is what makes the estimators cheap.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"
#include "duckdo/linalg.hpp"

#include <random>
#include <string>

namespace duckdb {
namespace duckdo {

using std::string;

//! Everything a causal query says about itself, parsed from the positional and
//! named parameters of a table function.
struct CausalSpec {
	string relation;
	string treatment;
	string outcome;
	//! Empty AND `covariates_given` false means "every column except the
	//! treatment, the outcome and `exclude`". Written explicitly as
	//! `covariates := []` it means no covariates at all - the two have to be
	//! distinguishable, because silently adjusting for every column when
	//! somebody asked for none is a wrong answer rather than a surprising one.
	vector<string> covariates;
	bool covariates_given = false;
	vector<string> exclude;
	string estimator;
	string model;
	string graph;
	string policy;
	string refute_method;
	//! Optional column carried through to per-row output so results can be joined.
	string id_column;
	//! Optional column naming the independent unit when a row is not one - an
	//! order joined to its customer, a visit joined to its patient. Folds,
	//! standard errors and bootstrap resamples then treat the cluster as the unit.
	string cluster_column;
	//! Optional boolean column naming a targeting rule, for do_policy_value.
	string policy_column;
	//! Optional numeric column carried through unencoded: the instrument for
	//! do_iv, the mediator for do_frontdoor.
	string aux_column;
	//! Treat a row when its estimated effect exceeds this, for do_policy_value.
	double threshold = 0.0;
	//! Maximum depth of the tree do_optimal_policy searches.
	idx_t depth = 2;
	//! Set by the continuous-treatment entry points. A binary function seeing a
	//! continuous treatment must still refuse, so this is opt-in per function
	//! rather than inferred from the data.
	bool continuous_treatment = false;
	//! Set by do_ate_levels. The treatment is kept as a level index rather than
	//! mapped onto {0, 1}, and `reference` names the level every other one is
	//! contrasted against. Opt-in per function for the same reason as
	//! continuous_treatment: a binary function handed three levels must refuse.
	bool multi_treatment = false;
	string reference;
	//! Grid points for a dose-response curve.
	idx_t grid = 20;
	//! Independent context draws for a foundation model. One means a single
	//! draw and no interval; more than one funds a real one.
	idx_t ensemble = 1;
	//! Which draw of that ensemble is being run. Draw 0 is the single-pass
	//! behaviour and must stay deterministic; later draws vary the context and,
	//! where the model's covariate budget binds, the covariate subset.
	idx_t draw_index = 0;
	//! Explicit two-level mapping when the treatment is not already 0/1.
	string treated_label;
	string control_label;
	int64_t seed = 42;
	idx_t folds = 5;
	idx_t bootstrap_reps = 200;
	double trim = 0.01;
	double fraction = 0.8;
	double confounder_strength = 0.5;

	//! Parse from a table function bind input. Throws BinderException with an
	//! actionable message on anything malformed.
	static CausalSpec Parse(ClientContext &context, const vector<Value> &inputs, const named_parameter_map_t &named);
};

enum class FeatureKind : uint8_t { NUMERIC, ONE_HOT, MISSING_INDICATOR };

//! One column of the encoded matrix, and how it got there.
struct FeatureInfo {
	string name;
	string source;
	FeatureKind kind = FeatureKind::NUMERIC;
	string level;
	double center = 0.0;
	double scale = 1.0;
};

//! The estimand a result refers to. Everything DuckDo returns names one.
enum class Estimand : uint8_t { ATE, ATT, ATC };

const char *EstimandName(Estimand e);

struct CausalFrame {
	idx_t n = 0;
	//! n x p encoded covariates, standardised.
	Matrix X;
	//! Treatment. 0.0 or 1.0 for a binary treatment; the dose on its own scale
	//! when continuous_treatment is set; a level index when multi_treatment is.
	vector<double> t;
	//! True when `t` holds a dose rather than an arm indicator.
	bool continuous_treatment = false;
	//! True when `t` holds a level index, 0 .. levels.size() - 1.
	bool multi_treatment = false;
	//! Treatment levels in the column's natural order; `t` indexes into this.
	vector<string> levels;
	//! Dose quantiles, ascending, for choosing a well-supported grid.
	vector<double> dose_sorted;
	//! Outcome on its original scale.
	vector<double> y;
	//! Row index in the source relation, so per-row output can be aligned.
	vector<idx_t> source_row;
	//! Values of the `id :=` column, rendered as text, when one was given.
	vector<string> ids;
	bool has_id = false;
	//! The `cluster :=` column: raw labels as read, then a dense index per row,
	//! numbered in canonical order so the numbering follows the data rather
	//! than storage. Filled by BuildCanonicalOrder, on the final rows.
	vector<string> cluster_labels;
	vector<idx_t> cluster;
	idx_t n_clusters = 0;
	bool has_cluster = false;
	string cluster_name;
	//! Rows of each cluster, in canonical order, for resampling whole clusters.
	vector<vector<idx_t>> cluster_members;
	//! Values of the `policy :=` boolean column, when one was given.
	vector<uint8_t> policy;
	bool has_policy = false;
	//! Values of the auxiliary numeric column (instrument or mediator).
	vector<double> aux;
	bool has_aux = false;
	vector<FeatureInfo> features;
	//! Source covariate columns that survived encoding.
	vector<string> covariate_columns;
	idx_t n_treated = 0;
	idx_t n_rows_with_missing = 0;
	bool binary_outcome = false;
	string treated_label = "1";
	string control_label = "0";
	//! Non-fatal problems worth putting in front of the user.
	vector<string> warnings;

	//! A permutation of 0..n-1 ordered by what each row *contains* rather than
	//! where it happens to sit in the table.
	//!
	//! Every seeded draw in DuckDo - fold assignment, bootstrap resampling,
	//! foundation-model context selection, the placebo shuffle - used to index
	//! storage positions. That made the estimate depend on row order: the same
	//! rows written in a different order landed in different folds and returned
	//! a different, equally valid answer, worth about 2% of a standard error for
	//! `aipw` and 6% for `ipw`. A table reordered by an unrelated ETL change
	//! would move a published number, which is not a property a database
	//! extension should have.
	//!
	//! Draws index this instead, so they follow the data. Built once in
	//! BuildFrame rather than lazily, because bootstrap replicates read the
	//! frame from several threads at once and a lazy cache would be a race.
	vector<idx_t> canonical;

	idx_t Cols() const {
		return X.cols;
	}
	//! Row indices of every row, of the treated arm, and of the control arm,
	//! all in canonical order.
	vector<idx_t> AllRows() const;
	vector<idx_t> ArmRows(double arm) const;
	//! Map a canonical rank to a storage row. Seeded draws pick a rank.
	inline idx_t Draw(idx_t rank) const {
		return canonical[rank];
	}
};

//! Order rows by content, lexicographically: outcome, treatment, auxiliary
//! column, then the encoded covariates, with the storage index breaking ties only
//! between rows identical in every value an estimator can see - where which one
//! goes first cannot change any result. Not a hash: `X` is standardised with a
//! mean summed in storage order, and a hash turns that mean's last-bit drift
//! under reordering into a different sort key, where a comparison does not.
void BuildCanonicalOrder(CausalFrame &frame);

//! A bootstrap resample of whole clusters: as many clusters as the frame has,
//! drawn with replacement, each contributing all of its rows. Rows of one unit
//! are not independent draws, so resampling them one at a time would give the
//! same too-narrow interval clustering exists to correct. Only for a frame with
//! cluster := - an unclustered caller keeps its own row draws, so its results
//! do not move.
vector<idx_t> ResampleClusters(const CausalFrame &frame, std::mt19937_64 &rng);

//! Fold ids by cluster: every row of a cluster in one fold, clusters shuffled
//! and dealt round-robin. For estimators whose own folds are not stratified by
//! a binary arm - a continuous dose, several levels.
vector<idx_t> AssignFoldsByCluster(const CausalFrame &frame, idx_t folds, int64_t seed);

//! Materialise a causal frame from the source relation. Runs the schema probe,
//! the level probes and the projection query, then encodes.
CausalFrame BuildFrame(ClientContext &context, const CausalSpec &spec);

// --- shared plumbing, also used by the diagnostics and graph functions ------

//! Run a query on a fresh connection and return the materialised result.
//! Throws with `context_msg` prefixed on failure.
unique_ptr<MaterializedQueryResult> RunQuery(ClientContext &context, const string &sql, const string &context_msg);

//! Render the user's relation argument as a FROM-clause fragment. A bare
//! identifier (optionally qualified) is passed through quoted; anything else
//! must be a SELECT/WITH query and is wrapped in parentheses.
string RelationSql(const string &relation);
//! Quote a SQL identifier, doubling any embedded quote.
string QuoteIdentifier(const string &identifier);
//! Quote a SQL string literal, doubling any embedded apostrophe.
string QuoteLiteral(const string &text);

idx_t GetSettingIdx(ClientContext &context, const char *name, idx_t fallback);
double GetSettingDouble(ClientContext &context, const char *name, double fallback);
string GetSettingString(ClientContext &context, const char *name, const string &fallback);

} // namespace duckdo
} // namespace duckdb
