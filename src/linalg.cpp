#include "duckdo/linalg.hpp"

#include <atomic>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

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

static std::atomic<idx_t> numeric_threads(0);
//! True on a thread while it runs a ParallelJobs job, whose fits then run on that
//! one thread. Nested parallelism used to be switched off by saving the shared
//! thread budget, setting it to 1 and restoring it afterwards - a plain global,
//! so two connections estimating at once could interleave the save and the
//! restore, and leave every later query in the process on a single thread with
//! nothing to say so. A per-thread flag has no such interleaving.
static thread_local bool inside_job = false;

namespace {
struct InsideJob {
	bool was;
	InsideJob() : was(inside_job) {
		inside_job = true;
	}
	~InsideJob() {
		inside_job = was;
	}
};
} // namespace

void SetNumericThreads(idx_t threads) {
	numeric_threads.store(threads);
}

void ParallelJobs(idx_t count, const std::function<void(idx_t)> &job) {
	if (count == 0) {
		return;
	}
	idx_t workers = NumericThreads();
	if (workers > count) {
		workers = count;
	}
	if (workers <= 1) {
		for (idx_t i = 0; i < count; i++) {
			job(i);
		}
		return;
	}

	// Each job is already a full pass over its data, so the jobs get the threads
	// and each fit inside one runs on the thread that took it.
	std::atomic<idx_t> next(0);
	vector<std::thread> pool;
	pool.reserve(workers - 1);
	auto pump = [&]() {
		InsideJob scope;
		for (;;) {
			const idx_t i = next.fetch_add(1);
			if (i >= count) {
				return;
			}
			job(i);
		}
	};
	for (idx_t w = 0; w + 1 < workers; w++) {
		pool.emplace_back(pump);
	}
	pump();
	for (auto &worker : pool) {
		worker.join();
	}
}

idx_t NumericThreads() {
	if (inside_job) {
		return 1;
	}
	const idx_t configured = numeric_threads.load();
	if (configured > 0) {
		return configured;
	}
	const unsigned hardware = std::thread::hardware_concurrency();
	return hardware > 0 ? static_cast<idx_t>(hardware) : 1;
}

//! Accumulate the weighted normal equations X'WX (with intercept) and X'Wz over
//! one block of rows. Only the lower triangle is filled; the caller mirrors it.
static void AccumulateBlock(const Matrix &X, const vector<double> &z, const vector<idx_t> &rows,
                            const vector<double> &w, idx_t begin, idx_t end, double *xtx, double *xtz) {
	const idx_t p = X.cols + 1;
	vector<double> row(p);
	for (idx_t idx = begin; idx < end; idx++) {
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
}

//! Accumulate the weighted normal equations, split across threads by row block.
//! This is the dominant cost of every fit - O(n * p^2), run once per IRLS
//! iteration per fold - and each thread only needs its own (p+1)^2 scratch
//! matrix, which is kilobytes. The reduction order is fixed by block index, so
//! results stay bit-identical run to run.
static void BuildNormalEquations(const Matrix &X, const vector<double> &z, const vector<idx_t> &rows,
                                 const vector<double> &w, vector<double> &xtx, vector<double> &xtz) {
	const idx_t p = X.cols + 1;
	xtx.assign(p * p, 0.0);
	xtz.assign(p, 0.0);

	// How the rows are split is a function of the DATA and nothing else - never
	// of the thread budget. Floating-point addition is not associative, so a
	// split that follows the core count makes the answer follow the core count
	// too: before this, the same query on the same seed returned 3.0031935719077567
	// on one thread and 3.0031935719077549 on sixteen. Two ULPs is harmless
	// arithmetically and corrosive to a promise of reproducibility, and the cost
	// of keeping the promise is a few extra block boundaries.
	//
	// Below kMinRowsPerBlock the per-block overhead costs more than the work
	// saved; above kMaxBlocks the scratch and the reduction start to matter.
	const idx_t kMinRowsPerBlock = 4096;
	const idx_t kMaxBlocks = 64;
	idx_t blocks = std::max<idx_t>(rows.size() / kMinRowsPerBlock, 1);
	blocks = std::min<idx_t>(blocks, kMaxBlocks);

	if (blocks <= 1) {
		AccumulateBlock(X, z, rows, w, 0, rows.size(), xtx.data(), xtz.data());
	} else {
		vector<vector<double>> partial_xtx(blocks, vector<double>(p * p, 0.0));
		vector<vector<double>> partial_xtz(blocks, vector<double>(p, 0.0));
		const idx_t span = (rows.size() + blocks - 1) / blocks;
		auto run_block = [&](idx_t b) {
			const idx_t begin = std::min<idx_t>(b * span, rows.size());
			const idx_t end = std::min<idx_t>(begin + span, rows.size());
			AccumulateBlock(X, z, rows, w, begin, end, partial_xtx[b].data(), partial_xtz[b].data());
		};

		// The blocks are fixed; only how many threads chew through them varies.
		idx_t workers = std::min<idx_t>(NumericThreads(), blocks);
		if (workers <= 1) {
			for (idx_t b = 0; b < blocks; b++) {
				run_block(b);
			}
		} else {
			std::atomic<idx_t> next(0);
			vector<std::thread> pool;
			pool.reserve(workers - 1);
			auto pump = [&]() {
				for (;;) {
					const idx_t b = next.fetch_add(1);
					if (b >= blocks) {
						return;
					}
					run_block(b);
				}
			};
			for (idx_t t = 0; t + 1 < workers; t++) {
				pool.emplace_back(pump);
			}
			pump();
			for (auto &worker : pool) {
				worker.join();
			}
		}

		// Reduced in block order, which is the same order on every machine.
		for (idx_t b = 0; b < blocks; b++) {
			for (idx_t i = 0; i < p * p; i++) {
				xtx[i] += partial_xtx[b][i];
			}
			for (idx_t i = 0; i < p; i++) {
				xtz[i] += partial_xtz[b][i];
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

//! Accumulate the penalised Gram matrix A = X'X + lambda*I (intercept
//! unpenalised) and, optionally, the meat B = sum r_i^2 x_i x_i'.
static void BuildGram(const Matrix &X, const vector<idx_t> &rows, double lambda, const vector<double> *residuals,
                      const vector<double> *case_weights, vector<double> &gram, vector<double> &meat) {
	const idx_t p = X.cols + 1;
	gram.assign(p * p, 0.0);
	if (residuals) {
		meat.assign(p * p, 0.0);
	}
	vector<double> row(p);
	for (idx_t idx = 0; idx < rows.size(); idx++) {
		const idx_t r = rows[idx];
		row[0] = 1.0;
		const double *src = X.Row(r);
		for (idx_t j = 0; j < X.cols; j++) {
			row[j + 1] = src[j];
		}
		// Bread accumulates w*x*x'; meat accumulates w^2*e^2*x*x'. With unit
		// weights both reduce to the ordinary HC1 sandwich, so the OLS callers
		// are unchanged.
		const double w = case_weights ? (*case_weights)[idx] : 1.0;
		const double resid_sq = residuals ? (*residuals)[idx] * (*residuals)[idx] : 0.0;
		const double meat_scale = w * w * resid_sq;
		for (idx_t a = 0; a < p; a++) {
			for (idx_t b = 0; b < p; b++) {
				gram[a * p + b] += w * row[a] * row[b];
				if (residuals) {
					meat[a * p + b] += meat_scale * row[a] * row[b];
				}
			}
		}
	}
	for (idx_t j = 1; j < p; j++) {
		gram[j * p + j] += lambda;
	}
}

//! Invert a symmetric positive definite matrix column by column. Returns false
//! if it is singular, in which case the caller reports no interval rather than
//! a wrong one.
static bool InvertSpd(const vector<double> &matrix, idx_t p, vector<double> &inverse) {
	inverse.assign(p * p, 0.0);
	for (idx_t c = 0; c < p; c++) {
		vector<double> rhs(p, 0.0);
		rhs[c] = 1.0;
		vector<double> column;
		vector<double> work = matrix;
		if (!CholeskySolve(work, p, rhs, column)) {
			inverse.assign(p * p, 0.0);
			return false;
		}
		for (idx_t r = 0; r < p; r++) {
			inverse[r * p + c] = column[r];
		}
	}
	return true;
}

RidgeFit FitRidgeWithSandwich(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows, double lambda,
                              const vector<idx_t> *cluster) {
	return FitRidgeWeightedWithSandwich(X, y, rows, {}, lambda, cluster);
}

RidgeFit FitRidgeWeightedWithSandwich(const Matrix &X, const vector<double> &y, const vector<idx_t> &rows,
                                      const vector<double> &weights, double lambda, const vector<idx_t> *cluster) {
	RidgeFit out;
	const idx_t p = X.cols + 1;
	out.dim = p;
	out.model = FitRidge(X, y, rows, weights, lambda);
	out.cov.assign(p * p, 0.0);
	if (rows.size() <= p) {
		return out;
	}

	vector<double> residuals(rows.size(), 0.0);
	for (idx_t idx = 0; idx < rows.size(); idx++) {
		const idx_t r = rows[idx];
		residuals[idx] = y[r] - out.model.Eta(X.Row(r), X.cols);
	}

	vector<double> gram, meat;
	BuildGram(X, rows, lambda, &residuals, weights.empty() ? nullptr : &weights, gram, meat);
	vector<double> inverse;
	if (!InvertSpd(gram, p, inverse)) {
		return out;
	}

	// V = A^-1 B A^-1, with the HC1 small-sample correction.
	double correction = static_cast<double>(rows.size()) / static_cast<double>(rows.size() - p);
	if (cluster) {
		// Clustered: B is the sum over clusters of (sum of w e x)(sum of w e x)',
		// and the correction is CR1. Without this, k rows of one unit count as k
		// independent observations and the interval narrows by about sqrt(k).
		idx_t max_id = 0;
		for (auto r : rows) {
			max_id = std::max(max_id, (*cluster)[r]);
		}
		vector<double> sums((max_id + 1) * p, 0.0);
		vector<uint8_t> seen(max_id + 1, 0);
		for (idx_t idx = 0; idx < rows.size(); idx++) {
			const idx_t r = rows[idx];
			const idx_t g = (*cluster)[r];
			const double s = (weights.empty() ? 1.0 : weights[idx]) * residuals[idx];
			double *acc = &sums[g * p];
			acc[0] += s;
			const double *src = X.Row(r);
			for (idx_t j = 0; j < X.cols; j++) {
				acc[j + 1] += s * src[j];
			}
			seen[g] = 1;
		}
		std::fill(meat.begin(), meat.end(), 0.0);
		double groups = 0.0;
		for (idx_t g = 0; g <= max_id; g++) {
			if (!seen[g]) {
				continue;
			}
			groups += 1.0;
			const double *acc = &sums[g * p];
			for (idx_t a = 0; a < p; a++) {
				for (idx_t b = 0; b < p; b++) {
					meat[a * p + b] += acc[a] * acc[b];
				}
			}
		}
		if (groups < 2.0) {
			return out;
		}
		correction =
		    groups / (groups - 1.0) * static_cast<double>(rows.size() - 1) / static_cast<double>(rows.size() - p);
	}
	vector<double> temp(p * p, 0.0);
	for (idx_t a = 0; a < p; a++) {
		for (idx_t b = 0; b < p; b++) {
			double acc = 0.0;
			for (idx_t k = 0; k < p; k++) {
				acc += inverse[a * p + k] * meat[k * p + b];
			}
			temp[a * p + b] = acc;
		}
	}
	for (idx_t a = 0; a < p; a++) {
		for (idx_t b = 0; b < p; b++) {
			double acc = 0.0;
			for (idx_t k = 0; k < p; k++) {
				acc += temp[a * p + k] * inverse[k * p + b];
			}
			out.cov[a * p + b] = correction * acc;
		}
	}
	return out;
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

double NormalCdf(double x) {
	return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

//! Upper regularised incomplete gamma Q(a, x): the series below a + 1, Lentz's
//! continued fraction above, each where it converges fast.
double UpperGammaQ(double a, double x) {
	if (!(x > 0.0)) {
		return 1.0;
	}
	const double log_prefix = a * std::log(x) - x - std::lgamma(a);
	if (x < a + 1.0) {
		double ap = a, term = 1.0 / a, sum = term;
		for (int i = 0; i < 10000; i++) {
			ap += 1.0;
			term *= x / ap;
			sum += term;
			if (std::fabs(term) < std::fabs(sum) * 1e-16) {
				break;
			}
		}
		return std::max(0.0, 1.0 - sum * std::exp(log_prefix));
	}
	const double tiny = 1e-300;
	double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
	for (int i = 1; i < 10000; i++) {
		const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
		b += 2.0;
		d = an * d + b;
		if (std::fabs(d) < tiny) {
			d = tiny;
		}
		c = b + an / c;
		if (std::fabs(c) < tiny) {
			c = tiny;
		}
		d = 1.0 / d;
		const double delta = d * c;
		h *= delta;
		if (std::fabs(delta - 1.0) < 1e-16) {
			break;
		}
	}
	return std::exp(log_prefix) * h;
}

//! P(X <= h, Y <= k) for a standard bivariate normal with correlation r:
//! Genz's algorithm (Drezner-Wesolowsky with Gauss-Legendre quadrature, and a
//! separate expansion for |r| >= 0.925), accurate to about 1e-15.
double BivariateNormalCdf(double h_in, double k_in, double r) {
	static const double x6[] = {-0.9324695142031522, -0.6612093864662647, -0.2386191860831970};
	static const double w6[] = {0.1713244923791705, 0.3607615730481384, 0.4679139345726904};
	static const double x12[] = {-0.9815606342467191, -0.9041172563704750, -0.7699026741943050,
	                             -0.5873179542866171, -0.3678314989981802, -0.1252334085114692};
	static const double w12[] = {0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
	                             0.2031674267230659,  0.2334925365383547, 0.2491470458134029};
	static const double x20[] = {-0.9931285991850949, -0.9639719272779138, -0.9122344282513259, -0.8391169718222188,
	                             -0.7463319064601508, -0.6360536807265150, -0.5108670019508271, -0.3737060887154196,
	                             -0.2277858511416451, -0.07652652113349733};
	static const double w20[] = {0.01761400713915212, 0.04060142980038694, 0.06267204833410906, 0.08327674157670475,
	                             0.1019301198172404,  0.1181945319615184,  0.1316886384491766,  0.1420961093183821,
	                             0.1491729864726037,  0.1527533871307259};
	const double kPi = 3.14159265358979323846;
	const double *x, *w;
	int lg;
	if (std::fabs(r) < 0.3) {
		x = x6, w = w6, lg = 3;
	} else if (std::fabs(r) < 0.75) {
		x = x12, w = w12, lg = 6;
	} else {
		x = x20, w = w20, lg = 10;
	}
	// Genz works with the upper orthant P(X > h, Y > k); the lower one is that at (-h, -k).
	double h = -h_in, k = -k_in, hk = h * k, bvn = 0.0;
	if (std::fabs(r) < 0.925) {
		const double hs = (h * h + k * k) / 2.0, asr = std::asin(r);
		for (int i = 0; i < lg; i++) {
			for (int side = -1; side <= 1; side += 2) {
				const double sn = std::sin(asr * (1.0 + side * x[i]) / 2.0);
				bvn += w[i] * std::exp((sn * hk - hs) / (1.0 - sn * sn));
			}
		}
		bvn = bvn * asr / (4.0 * kPi) + NormalCdf(-h) * NormalCdf(-k);
	} else {
		if (r < 0.0) {
			k = -k;
			hk = -hk;
		}
		if (std::fabs(r) < 1.0) {
			const double as = (1.0 - r) * (1.0 + r), bs = (h - k) * (h - k);
			double a = std::sqrt(as);
			const double c = (4.0 - hk) / 8.0, d = (12.0 - hk) / 16.0;
			bvn = a * std::exp(-(bs / as + hk) / 2.0) *
			      (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 + c * d * as * as / 5.0);
			if (hk > -160.0) {
				const double b = std::sqrt(bs);
				bvn -= std::exp(-hk / 2.0) * std::sqrt(2.0 * kPi) * NormalCdf(-b / a) * b *
				       (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
			}
			a /= 2.0;
			for (int i = 0; i < lg; i++) {
				for (int side = -1; side <= 1; side += 2) {
					const double xs = (a * (side * x[i] + 1.0)) * (a * (side * x[i] + 1.0));
					const double rs = std::sqrt(1.0 - xs);
					const double asr = -(bs / xs + hk) / 2.0;
					if (asr > -100.0) {
						bvn += a * w[i] * std::exp(asr) *
						       (std::exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs - (1.0 + c * xs * (1.0 + d * xs)));
					}
				}
			}
			bvn = -bvn / (2.0 * kPi);
		}
		if (r > 0.0) {
			bvn += NormalCdf(-std::max(h, k));
		} else {
			bvn = -bvn;
			if (k > h) {
				bvn += NormalCdf(k) - NormalCdf(h);
			}
		}
	}
	return std::max(0.0, std::min(1.0, bvn));
}

//! Eigen-decomposition of a symmetric n x n matrix (row-major) by cyclic Jacobi
//! rotations: A = V diag(values) V'. Columns of V are the eigenvectors.
void SymmetricEigen(vector<double> A, idx_t n, vector<double> &values, vector<double> &V) {
	V.assign(n * n, 0.0);
	for (idx_t i = 0; i < n; i++) {
		V[i * n + i] = 1.0;
	}
	for (int sweep = 0; sweep < 100; sweep++) {
		double off = 0.0;
		for (idx_t i = 0; i < n; i++) {
			for (idx_t j = i + 1; j < n; j++) {
				off += A[i * n + j] * A[i * n + j];
			}
		}
		if (off < 1e-30) {
			break;
		}
		for (idx_t p = 0; p < n; p++) {
			for (idx_t q = p + 1; q < n; q++) {
				const double apq = A[p * n + q];
				if (std::fabs(apq) < 1e-300) {
					continue;
				}
				const double theta = (A[q * n + q] - A[p * n + p]) / (2.0 * apq);
				const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
				const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
				for (idx_t k = 0; k < n; k++) {
					const double akp = A[k * n + p], akq = A[k * n + q];
					A[k * n + p] = c * akp - s * akq;
					A[k * n + q] = s * akp + c * akq;
				}
				for (idx_t k = 0; k < n; k++) {
					const double apk = A[p * n + k], aqk = A[q * n + k];
					A[p * n + k] = c * apk - s * aqk;
					A[q * n + k] = s * apk + c * aqk;
				}
				for (idx_t k = 0; k < n; k++) {
					const double vkp = V[k * n + p], vkq = V[k * n + q];
					V[k * n + p] = c * vkp - s * vkq;
					V[k * n + q] = s * vkp + c * vkq;
				}
			}
		}
	}
	values.resize(n);
	for (idx_t i = 0; i < n; i++) {
		values[i] = A[i * n + i];
	}
}

double NormalQuantile(double p) {
	if (!(p > 0.0 && p < 1.0)) {
		return p <= 0.0 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
	}
	// Acklam's rational approximation, good to about 1e-9, then one Halley step
	// against the exact CDF, which takes it to full double precision.
	static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,  -2.759285104469687e+02,
	                           1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
	static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
	                           6.680131188771972e+01, -1.328068155288572e+01};
	static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
	                           -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
	static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
	                           3.754408661907416e+00};
	const double low = 0.02425;
	double x;
	if (p < low) {
		const double q = std::sqrt(-2.0 * std::log(p));
		x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
		    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
	} else if (p <= 1.0 - low) {
		const double q = p - 0.5, r = q * q;
		x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
		    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
	} else {
		const double q = std::sqrt(-2.0 * std::log(1.0 - p));
		x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
		    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
	}
	const double kSqrt2Pi = 2.5066282746310002;
	const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
	const double u = e * kSqrt2Pi * std::exp(0.5 * x * x);
	return x - u / (1.0 + 0.5 * x * u);
}

double NormalTwoSidedP(double z) {
	return std::erfc(std::fabs(z) / std::sqrt(2.0));
}

} // namespace duckdo
} // namespace duckdb
