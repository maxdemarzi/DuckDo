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
	//! Empty means "every column except the treatment, the outcome and `exclude`".
	vector<string> covariates;
	vector<string> exclude;
	string estimator;
	string model;
	string graph;
	string policy;
	string refute_method;
	//! Optional column carried through to per-row output so results can be joined.
	string id_column;
	//! Optional boolean column naming a targeting rule, for do_policy_value.
	string policy_column;
	//! Treat a row when its estimated effect exceeds this, for do_policy_value.
	double threshold = 0.0;
	//! Maximum depth of the tree do_optimal_policy searches.
	idx_t depth = 2;
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
	//! Treatment indicator, 0.0 or 1.0.
	vector<double> t;
	//! Outcome on its original scale.
	vector<double> y;
	//! Row index in the source relation, so per-row output can be aligned.
	vector<idx_t> source_row;
	//! Values of the `id :=` column, rendered as text, when one was given.
	vector<string> ids;
	bool has_id = false;
	//! Values of the `policy :=` boolean column, when one was given.
	vector<uint8_t> policy;
	bool has_policy = false;
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

	idx_t Cols() const {
		return X.cols;
	}
	//! Row indices of every row, of the treated arm, and of the control arm.
	vector<idx_t> AllRows() const;
	vector<idx_t> ArmRows(double arm) const;
};

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
