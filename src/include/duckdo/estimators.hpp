//===----------------------------------------------------------------------===//
//                         DuckDo - causal inference in DuckDB
//
// duckdo/estimators.hpp
//
// Classical causal estimators. Every one of them consumes a CausalFrame and
// produces an EffectResult that names its own estimand, estimator and variance
// method - no result is ever a bare number.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdo/frame.hpp"

namespace duckdb {
namespace duckdo {

//! The uniform result contract shared by every estimator.
struct EffectResult {
	Estimand estimand = Estimand::ATE;
	string estimator;
	double estimate = 0.0;
	double std_error = 0.0;
	double ci_low = 0.0;
	double ci_high = 0.0;
	double p_value = 1.0;
	idx_t n = 0;
	idx_t n_treated = 0;
	idx_t n_trimmed = 0;
	string variance_method;
	vector<string> warnings;

	//! Fill ci_low/ci_high/p_value from estimate and std_error.
	void Finalize();
};

//! Cross-fitted nuisance predictions: propensity and both potential-outcome
//! surfaces, each predicted out-of-fold so the estimator stays honest.
struct NuisanceFit {
	vector<double> e;
	vector<double> mu0;
	vector<double> mu1;
	//! Rows surviving propensity trimming.
	vector<idx_t> rows;
	idx_t n_trimmed = 0;
};

//! Fit propensity and outcome models with K-fold cross-fitting.
NuisanceFit FitNuisance(const CausalFrame &frame, const CausalSpec &spec, int64_t seed);

//! Cross-fitted propensity scores only. Used by the diagnostics, which do not
//! need an outcome column at all.
vector<double> FitPropensity(const CausalFrame &frame, idx_t folds, int64_t seed);

//! The ridge penalty every outcome fit uses, proportional to n.
//! Cluster-robust standard error of a mean over `rows`, from each row's centred
//! contribution: contributions are summed within each cluster before squaring,
//! with a G / (G - 1) correction. With every row its own cluster this is the
//! ordinary influence-function standard error.
double ClusterSe(const CausalFrame &frame, const vector<idx_t> &rows, const vector<double> &centred);

double RidgeLambda(const CausalFrame &frame);

//! Fit an outcome model on `train` and write its prediction for every row in
//! `targets` into `out`: logistic for a binary outcome, ridge otherwise.
void FitAndPredictOutcome(const CausalFrame &frame, const vector<idx_t> &train, const vector<idx_t> &targets,
                          vector<double> &out);

//! Estimate a population effect with the estimator named in the spec.
EffectResult EstimateEffect(const CausalFrame &frame, const CausalSpec &spec, Estimand estimand);

//! Per-row conditional effects, with pointwise intervals.
struct CateResult {
	vector<double> cate;
	vector<double> lo;
	vector<double> hi;
	string learner;
	string interval_method;
	vector<string> warnings;
};

CateResult EstimateCate(const CausalFrame &frame, const CausalSpec &spec);

//! The AIPW pseudo-outcome, whose mean is the doubly-robust ATE and whose
//! regression on X is the DR-learner. Shared by estimation and diagnostics.
vector<double> AipwPseudoOutcome(const CausalFrame &frame, const NuisanceFit &fit);

//! True when `name` is an estimator DuckDo knows about.
bool IsKnownEstimator(const string &name);
//! Comma-separated list of supported estimators, for error messages.
string KnownEstimators();

} // namespace duckdo
} // namespace duckdb
