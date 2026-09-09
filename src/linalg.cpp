#include "duckdo/linalg.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {
namespace duckdo {

double Sigmoid(double x) {
	if (x >= 0) {
		return 1.0 / (1.0 + std::exp(-x));
	}
	const double e = std::exp(x);
	return e / (1.0 + e);
}

double LinearModel::Eta(const double *x, idx_t cols) const {
	double acc = beta[0];
	for (idx_t j = 0; j < cols; j++) {
		acc += beta[j + 1] * x[j];
	}
	return acc;
}

double LinearModel::Predict(const double *x, idx_t cols) const {
	const double eta = Eta(x, cols);
	return logistic ? Sigmoid(eta) : eta;
}

bool CholeskySolve(vector<double> &A, idx_t n, const vector<double> &rhs, vector<double> &out) {
	// Cholesky factorisation A = L*L', overwriting the lower triangle of A.
	for (idx_t i = 0; i < n; i++) {
		for (idx_t j = 0; j <= i; j++) {
			double sum = A[i * n + j];
			for (idx_t k = 0; k < j; k++) {
				sum -= A[i * n + k] * A[j * n + k];
			}
			if (i == j) {
				if (!(sum > 1e-12)) {
					return false;
				}
				A[i * n + j] = std::sqrt(sum);
			} else {
				A[i * n + j] = sum / A[j * n + j];
			}
		}
	}
	// Forward substitution L*z = rhs.
	out.assign(n, 0.0);
	for (idx_t i = 0; i < n; i++) {
		double sum = rhs[i];
		for (idx_t k = 0; k < i; k++) {
			sum -= A[i * n + k] * out[k];
		}
		out[i] = sum / A[i * n + i];
	}
	// Back substitution L'*b = z.
	for (idx_t ii = n; ii-- > 0;) {
		double sum = out[ii];
		for (idx_t k = ii + 1; k < n; k++) {
			sum -= A[k * n + ii] * out[k];
		}
		out[ii] = sum / A[ii * n + ii];
	}
	return true;
}

//! Accumulate the weighted normal equations X'WX (with intercept) and X'Wz.
static void BuildNormalEquations(const Matrix &X, const vector<double> &z, const vector<idx_t> &rows,
                                 const vector<double> &w, vector<double> &xtx, vector<double> &xtz) {
	const idx_t p = X.cols + 1;
	xtx.assign(p * p, 0.0);
	xtz.assign(p, 0.0);
	vector<double> row(p);
	for (idx_t idx = 0; idx < rows.size(); idx++) {
		const idx_t r = rows[idx];
		const double wi = w.empty() ? 1.0 : w[idx];
		if (wi == 0.0) {
			continue;
		}
		row[0] = 1.0;
		const double *src = X.Row(r);
		for (idx_t j = 0; j < X.cols; j++) {
			row[j + 1] = src[j];
		}
		const double zi = z[r];
		for (idx_t a = 0; a < p; a++) {
			const double wa = wi * row[a];
			xtz[a] += wa * zi;
			for (idx_t b = 0; b <= a; b++) {
				xtx[a * p + b] += wa * row[b];
			}
		}
	}
	// Mirror the lower triangle into the upper one.
	for (idx_t a = 0; a < p; a++) {
		for (idx_t b = 0; b < a; b++) {
			xtx[b * p + a] = xtx[a * p + b];
		}
	}
}

//! Solve the normal equations, escalating the ridge until the system is
//! positive definite. Collinear covariates are common in real tables, so
//! failing here is not an option - we regularise until it solves.
static vector<double> SolveWithEscalation(vector<double> xtx, const vector<double> &xtz, idx_t p, double lambda) {
	vector<double> out;
	double ridge = lambda;
	for (int attempt = 0; attempt < 24; attempt++) {
		vector<double> a = xtx;
		// The intercept (index 0) is never penalised.
		for (idx_t j = 1; j < p; j++) {
			a[j * p + j] += ridge;
		}
		if (CholeskySolve(a, p, xtz, out)) {
			return out;
		}
		ridge = (ridge <= 0.0) ? 1e-8 : ridge * 10.0;
	}
	return vector<double>(p, 0.0);
}

LinearModel FitRidge(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows, const vector<double> &weights,
                     double lambda) {
	LinearModel model;
	model.logistic = false;
	const idx_t p = X.cols + 1;
	if (rows.empty()) {
		model.beta.assign(p, 0.0);
		return model;
	}
	vector<double> xtx, xty;
	BuildNormalEquations(X, y, rows, weights, xtx, xty);
	model.beta = SolveWithEscalation(std::move(xtx), xty, p, lambda);
	return model;
}

LinearModel FitLogistic(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows,
                        const vector<double> &weights, double lambda, idx_t max_iter) {
	LinearModel model;
	model.logistic = true;
	const idx_t p = X.cols + 1;
	model.beta.assign(p, 0.0);
	if (rows.empty()) {
		return model;
	}

	// Start the intercept at the empirical log-odds so the first IRLS step is
	// already in the right neighbourhood.
	double pos = 0.0, tot = 0.0;
	for (idx_t idx = 0; idx < rows.size(); idx++) {
		const double wi = weights.empty() ? 1.0 : weights[idx];
		pos += wi * y[rows[idx]];
		tot += wi;
	}
	double base = (tot > 0.0) ? (pos + 0.5) / (tot + 1.0) : 0.5;
	base = std::min(std::max(base, 1e-6), 1.0 - 1e-6);
	model.beta[0] = std::log(base / (1.0 - base));

	vector<double> working(X.rows, 0.0);
	vector<double> irls_w(rows.size(), 0.0);
	vector<double> prev;

	for (idx_t iter = 0; iter < max_iter; iter++) {
		prev = model.beta;
		for (idx_t idx = 0; idx < rows.size(); idx++) {
			const idx_t r = rows[idx];
			const double eta = model.Eta(X.Row(r), X.cols);
			const double mu = Sigmoid(eta);
			// Clamp the IRLS weight away from zero; without this, rows the model
			// is already certain about make the normal equations singular.
			const double v = std::max(mu * (1.0 - mu), 1e-6);
			irls_w[idx] = v * (weights.empty() ? 1.0 : weights[idx]);
			working[r] = eta + (y[r] - mu) / v;
		}
		vector<double> xtx, xtz;
		BuildNormalEquations(X, working, rows, irls_w, xtx, xtz);
		model.beta = SolveWithEscalation(std::move(xtx), xtz, p, lambda);

		double delta = 0.0;
		for (idx_t j = 0; j < p; j++) {
			delta = std::max(delta, std::fabs(model.beta[j] - prev[j]));
		}
		if (delta < 1e-8) {
			break;
		}
	}
	return model;
}

RidgeFit FitRidgeWithCovariance(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows, double lambda) {
	RidgeFit out;
	const idx_t p = X.cols + 1;
	out.dim = p;
	out.model = FitRidge(X, y, rows, {}, lambda);
	out.cov.assign(p * p, 0.0);
	if (rows.empty()) {
		return out;
	}

	double rss = 0.0;
	for (auto r : rows) {
		const double resid = y[r] - out.model.Eta(X.Row(r), X.cols);
		rss += resid * resid;
	}
	const double dof = static_cast<double>(rows.size() > p ? rows.size() - p : 1);
	const double sigma2 = rss / dof;

	vector<double> xtx(p * p, 0.0);
	vector<double> row(p);
	for (auto r : rows) {
		row[0] = 1.0;
		const double *src = X.Row(r);
		for (idx_t j = 0; j < X.cols; j++) {
			row[j + 1] = src[j];
		}
		for (idx_t a = 0; a < p; a++) {
			for (idx_t b = 0; b < p; b++) {
				xtx[a * p + b] += row[a] * row[b];
			}
		}
	}
	for (idx_t j = 1; j < p; j++) {
		xtx[j * p + j] += lambda;
	}
	// Invert column by column; a failure leaves the covariance zeroed, which
	// reports a zero-width interval rather than a wrong one.
	for (idx_t c = 0; c < p; c++) {
		vector<double> rhs(p, 0.0);
		rhs[c] = 1.0;
		vector<double> col;
		vector<double> a = xtx;
		if (!CholeskySolve(a, p, rhs, col)) {
			out.cov.assign(p * p, 0.0);
			break;
		}
		for (idx_t r = 0; r < p; r++) {
			out.cov[r * p + c] = col[r] * sigma2;
		}
	}
	return out;
}

double RidgeFit::PredictionStdError(const double *x, idx_t cols) const {
	vector<double> row(dim);
	row[0] = 1.0;
	for (idx_t j = 0; j < cols; j++) {
		row[j + 1] = x[j];
	}
	double acc = 0.0;
	for (idx_t a = 0; a < dim; a++) {
		double inner = 0.0;
		for (idx_t b = 0; b < dim; b++) {
			inner += cov[a * dim + b] * row[b];
		}
		acc += row[a] * inner;
	}
	return acc > 0.0 ? std::sqrt(acc) : 0.0;
}

double Mean(const vector<double> &v) {
	if (v.empty()) {
		return 0.0;
	}
	double acc = 0.0;
	for (auto x : v) {
		acc += x;
	}
	return acc / static_cast<double>(v.size());
}

double Variance(const vector<double> &v) {
	if (v.size() < 2) {
		return 0.0;
	}
	const double m = Mean(v);
	double acc = 0.0;
	for (auto x : v) {
		const double d = x - m;
		acc += d * d;
	}
	return acc / static_cast<double>(v.size() - 1);
}

double StdDev(const vector<double> &v) {
	return std::sqrt(Variance(v));
}

double NormalTwoSidedP(double z) {
	return std::erfc(std::fabs(z) / std::sqrt(2.0));
}

} // namespace duckdo
} // namespace duckdb
