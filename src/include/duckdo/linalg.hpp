//===----------------------------------------------------------------------===//
//                         DuckDo - causal inference in DuckDB
//
// duckdo/linalg.hpp
//
// Small dense linear algebra and the two base learners every estimator is
// built on: ridge-penalised least squares and logistic regression by IRLS.
// Deliberately minimal - no external dependency, no BLAS. The matrices we
// solve are (p+1)x(p+1) normal equations, which stay small because
// duckdo_max_features caps p.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"

#include <functional>

namespace duckdb {
namespace duckdo {

//! Dense row-major matrix of doubles.
struct Matrix {
	idx_t rows = 0;
	idx_t cols = 0;
	vector<double> data;

	Matrix() = default;
	Matrix(idx_t rows_p, idx_t cols_p) : rows(rows_p), cols(cols_p), data(rows_p * cols_p, 0.0) {
	}

	inline double &At(idx_t r, idx_t c) {
		return data[r * cols + c];
	}
	inline double At(idx_t r, idx_t c) const {
		return data[r * cols + c];
	}
	inline const double *Row(idx_t r) const {
		return data.data() + r * cols;
	}
	void Resize(idx_t r, idx_t c) {
		rows = r;
		cols = c;
		data.assign(r * c, 0.0);
	}
};

//! A fitted generalised linear model. beta[0] is the intercept; beta[1..p] are
//! the coefficients on the matrix columns in order.
struct LinearModel {
	vector<double> beta;
	bool logistic = false;

	//! Linear predictor for a single row of length cols.
	double Eta(const double *x, idx_t cols) const;
	//! Response-scale prediction: identity for linear, sigmoid for logistic.
	double Predict(const double *x, idx_t cols) const;
};

//! In-place Cholesky solve of A*out = rhs for symmetric positive definite A
//! (n x n, row-major). Returns false when A is not positive definite.
bool CholeskySolve(vector<double> &A, idx_t n, const vector<double> &rhs, vector<double> &out);

//! Ridge-penalised least squares over the subset `rows` of X. The intercept is
//! added internally and is never penalised. `weights` may be empty for unit
//! weights. The ridge is escalated automatically if the normal equations are
//! singular, so this never fails on collinear covariates.
LinearModel FitRidge(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows, const vector<double> &weights,
                     double lambda);

//! A ridge fit that also carries the covariance of its coefficients, so a
//! prediction can report a real pointwise interval rather than a flat band.
struct RidgeFit {
	LinearModel model;
	//! (p+1) x (p+1) coefficient covariance, row-major, already scaled by sigma^2.
	vector<double> cov;
	idx_t dim = 0;

	//! Standard error of the linear prediction at a single row.
	double PredictionStdError(const double *x, idx_t cols) const;
};

RidgeFit FitRidgeWithCovariance(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows, double lambda);

//! The same fit with a heteroskedasticity-robust (sandwich) covariance,
//! A^-1 B A^-1 with HC1 correction. Use this whenever the residual variance
//! varies across rows - which it does badly for a doubly-robust pseudo-outcome,
//! whose variance scales with 1/e(x) and 1/(1-e(x)).
RidgeFit FitRidgeWithSandwich(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows, double lambda);

//! The sandwich fit with per-row case weights - the estimating equation is
//! sum_i w_i x_i (y_i - x_i'b) = 0, so the bread is sum w x x' and the meat is
//! sum w^2 e^2 x x'. This is what a marginal structural model needs: the
//! inverse-probability weights change both halves, and using the unweighted
//! sandwich on weighted data understates the variance.
RidgeFit FitRidgeWeightedWithSandwich(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows,
                                      const vector<double> &weights, double lambda);

//! Logistic regression by iteratively reweighted least squares, with the same
//! ridge and intercept conventions as FitRidge.
LinearModel FitLogistic(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows,
                        const vector<double> &weights, double lambda, idx_t max_iter);

//! Threads used by the dense accumulations that dominate every fit. Set once
//! per query from duckdo_threads; 0 means "one per hardware thread". The work
//! is a sum over rows into a (p+1)^2 matrix, so it splits cleanly by row block.
void SetNumericThreads(idx_t threads);
idx_t NumericThreads();

//! Run `count` independent jobs across the same thread budget, waiting for all
//! of them. `job(i)` must write only to slot i of whatever the caller is
//! filling, so results land in index order regardless of scheduling - a
//! bootstrap whose answer depended on which thread finished first would not be
//! reproducible.
//!
//! Row-block threading inside each job is switched off for the duration:
//! bootstrap replicates already saturate the machine, and nesting the two
//! oversubscribes it badly. It is restored before returning.
void ParallelJobs(idx_t count, const std::function<void(idx_t)> &job);

double Sigmoid(double x);
double Mean(const vector<double> &v);
double Variance(const vector<double> &v);
double StdDev(const vector<double> &v);
//! Two-sided normal p-value for |z|.
double NormalTwoSidedP(double z);
//! 97.5th percentile of the standard normal, i.e. the 95% CI multiplier.
static constexpr double Z95 = 1.959963984540054;

} // namespace duckdo
} // namespace duckdb
