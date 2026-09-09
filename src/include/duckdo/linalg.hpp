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

//! Logistic regression by iteratively reweighted least squares, with the same
//! ridge and intercept conventions as FitRidge.
LinearModel FitLogistic(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows,
                        const vector<double> &weights, double lambda, idx_t max_iter);

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
