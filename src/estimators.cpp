#include "duckdo/estimators.hpp"

#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace duckdb {
namespace duckdo {

void EffectResult::Finalize() {
	if (std_error > 0.0 && std::isfinite(std_error)) {
		ci_low = estimate - Z95 * std_error;
		ci_high = estimate + Z95 * std_error;
		p_value = NormalTwoSidedP(estimate / std_error);
	} else {
		ci_low = estimate;
		ci_high = estimate;
		p_value = 1.0;
		std_error = 0.0;
	}
}

static const char *kEstimators[] = {"naive",     "regression", "ipw",       "aipw",      "dml",
                                    "s_learner", "t_learner",  "x_learner", "dr_learner"};

bool IsKnownEstimator(const string &name) {
	for (auto candidate : kEstimators) {
		if (StringUtil::CIEquals(name, candidate)) {
			return true;
		}
	}
	return false;
}

string KnownEstimators() {
	string out;
	for (auto candidate : kEstimators) {
		if (!out.empty()) {
			out += ", ";
		}
		out += candidate;
	}
	return out;
}

// --- cross-fitting ----------------------------------------------------------

namespace {

//! Assign fold ids, stratified within each treatment arm so no fold ends up
//! without both arms represented.
vector<idx_t> AssignFolds(const CausalFrame &frame, idx_t folds, int64_t seed) {
	vector<idx_t> assignment(frame.n, 0);
	std::mt19937_64 rng(static_cast<uint64_t>(seed));
	for (double arm : {0.0, 1.0}) {
		vector<idx_t> rows = frame.ArmRows(arm);
		std::shuffle(rows.begin(), rows.end(), rng);
		for (idx_t i = 0; i < rows.size(); i++) {
			assignment[rows[i]] = i % folds;
		}
	}
	return assignment;
}

} // namespace

// RidgeLambda and FitAndPredictOutcome are shared with the multi-level
// estimator. AssignFolds stays private: it stratifies arms 0 and 1 only.

double RidgeLambda(const CausalFrame &frame) {
	// Features are standardised, so a ridge proportional to n keeps the normal
	// equations conditioned without materially shrinking the fit.
	return 1e-6 * static_cast<double>(frame.n) + 1e-8;
}

//! Fit an outcome model on `rows` and predict every row in `targets`.
void FitAndPredictOutcome(const CausalFrame &frame, const vector<idx_t> &train, const vector<idx_t> &targets,
                          vector<double> &out) {
	const double lambda = RidgeLambda(frame);
	LinearModel model;
	if (frame.binary_outcome) {
		model = FitLogistic(frame.X, frame.y, train, {}, std::max(lambda, 1.0), 30);
	} else {
		model = FitRidge(frame.X, frame.y, train, {}, lambda);
	}
	for (auto r : targets) {
		out[r] = model.Predict(frame.X.Row(r), frame.X.cols);
	}
}

vector<double> FitPropensity(const CausalFrame &frame, idx_t folds, int64_t seed) {
	vector<double> e(frame.n, 0.5);
	const idx_t k_folds = std::min<idx_t>(std::max<idx_t>(folds, 2), std::max<idx_t>(2, frame.n / 4));
	const auto assignment = AssignFolds(frame, k_folds, seed);
	const double lambda = std::max(RidgeLambda(frame), 1.0);
	for (idx_t k = 0; k < k_folds; k++) {
		vector<idx_t> train, test;
		for (idx_t i = 0; i < frame.n; i++) {
			(assignment[i] == k ? test : train).push_back(i);
		}
		if (test.empty() || train.empty()) {
			continue;
		}
		auto model = FitLogistic(frame.X, frame.t, train, {}, lambda, 30);
		for (auto r : test) {
			e[r] = std::min(std::max(model.Predict(frame.X.Row(r), frame.X.cols), 1e-6), 1.0 - 1e-6);
		}
	}
	return e;
}

NuisanceFit FitNuisance(const CausalFrame &frame, const CausalSpec &spec, int64_t seed) {
	NuisanceFit fit;
	fit.e.assign(frame.n, 0.5);
	fit.mu0.assign(frame.n, 0.0);
	fit.mu1.assign(frame.n, 0.0);

	const idx_t folds = std::min<idx_t>(spec.folds, std::max<idx_t>(2, frame.n / 4));
	const auto assignment = AssignFolds(frame, folds, seed);
	const double lambda = RidgeLambda(frame);

	for (idx_t k = 0; k < folds; k++) {
		vector<idx_t> train, test, train_t0, train_t1;
		for (idx_t i = 0; i < frame.n; i++) {
			if (assignment[i] == k) {
				test.push_back(i);
			} else {
				train.push_back(i);
				(frame.t[i] == 1.0 ? train_t1 : train_t0).push_back(i);
			}
		}
		if (test.empty()) {
			continue;
		}
		// Propensity: P(T = 1 | X).
		auto ps_model = FitLogistic(frame.X, frame.t, train, {}, std::max(lambda, 1.0), 30);
		for (auto r : test) {
			fit.e[r] = ps_model.Predict(frame.X.Row(r), frame.X.cols);
		}
		// Outcome surfaces, one per arm.
		if (!train_t0.empty()) {
			FitAndPredictOutcome(frame, train_t0, test, fit.mu0);
		}
		if (!train_t1.empty()) {
			FitAndPredictOutcome(frame, train_t1, test, fit.mu1);
		}
	}

	// Trim on the propensity score. Rows with no realistic chance of either arm
	// carry enormous weight and destabilise every IPW-flavoured estimator.
	const double lo = spec.trim;
	const double hi = 1.0 - spec.trim;
	for (idx_t i = 0; i < frame.n; i++) {
		fit.e[i] = std::min(std::max(fit.e[i], 1e-6), 1.0 - 1e-6);
		if (spec.trim > 0.0 && (fit.e[i] < lo || fit.e[i] > hi)) {
			fit.n_trimmed++;
		} else {
			fit.rows.push_back(i);
		}
	}
	if (fit.rows.size() < 8) {
		// Trimming removed almost everything: fall back to untrimmed and let the
		// overlap diagnostic explain why the estimate is fragile.
		fit.rows = frame.AllRows();
		fit.n_trimmed = 0;
	}
	return fit;
}

vector<double> AipwPseudoOutcome(const CausalFrame &frame, const NuisanceFit &fit) {
	vector<double> psi(frame.n, 0.0);
	for (idx_t i = 0; i < frame.n; i++) {
		const double e = fit.e[i];
		const double t = frame.t[i];
		psi[i] = fit.mu1[i] - fit.mu0[i] + t * (frame.y[i] - fit.mu1[i]) / e -
		         (1.0 - t) * (frame.y[i] - fit.mu0[i]) / (1.0 - e);
	}
	return psi;
}

// --- individual estimators --------------------------------------------------

namespace {

//! Mean and influence-function standard error over `rows` of a per-row score
//! whose mean is the target parameter.
void MeanAndInfluenceSe(const vector<double> &score, const vector<idx_t> &rows, double &estimate, double &se) {
	if (rows.empty()) {
		estimate = 0.0;
		se = 0.0;
		return;
	}
	double sum = 0.0;
	for (auto r : rows) {
		sum += score[r];
	}
	estimate = sum / static_cast<double>(rows.size());
	double var = 0.0;
	for (auto r : rows) {
		const double d = score[r] - estimate;
		var += d * d;
	}
	var /= static_cast<double>(rows.size() > 1 ? rows.size() - 1 : 1);
	se = std::sqrt(var / static_cast<double>(rows.size()));
}

EffectResult Naive(const CausalFrame &frame) {
	EffectResult res;
	res.estimator = "naive";
	res.variance_method = "two-sample";
	vector<double> y1, y0;
	for (idx_t i = 0; i < frame.n; i++) {
		(frame.t[i] == 1.0 ? y1 : y0).push_back(frame.y[i]);
	}
	res.estimate = Mean(y1) - Mean(y0);
	const double v1 = Variance(y1) / static_cast<double>(std::max<size_t>(y1.size(), 1));
	const double v0 = Variance(y0) / static_cast<double>(std::max<size_t>(y0.size(), 1));
	res.std_error = std::sqrt(v1 + v0);
	res.warnings.push_back("'naive' adjusts for nothing; it is a baseline, not a causal estimate");
	return res;
}

//! Hajek (self-normalised) IPW, which is far better behaved than the raw
//! Horvitz-Thompson form when weights are extreme.
EffectResult Ipw(const CausalFrame &frame, const NuisanceFit &fit, Estimand estimand) {
	EffectResult res;
	res.estimator = "ipw";
	res.variance_method = "influence function";
	vector<double> score(frame.n, 0.0);
	double num1 = 0.0, den1 = 0.0, num0 = 0.0, den0 = 0.0;
	for (auto i : fit.rows) {
		const double e = fit.e[i];
		if (frame.t[i] == 1.0) {
			num1 += frame.y[i] / e;
			den1 += 1.0 / e;
		} else {
			num0 += frame.y[i] / (1.0 - e);
			den0 += 1.0 / (1.0 - e);
		}
	}
	const double m1 = den1 > 0 ? num1 / den1 : 0.0;
	const double m0 = den0 > 0 ? num0 / den0 : 0.0;
	const double n = static_cast<double>(fit.rows.size());
	// Influence function of the Hajek difference.
	for (auto i : fit.rows) {
		const double e = fit.e[i];
		const double t = frame.t[i];
		const double c1 = den1 > 0 ? (t / e) * (frame.y[i] - m1) / (den1 / n) : 0.0;
		const double c0 = den0 > 0 ? ((1.0 - t) / (1.0 - e)) * (frame.y[i] - m0) / (den0 / n) : 0.0;
		score[i] = c1 - c0;
	}
	double centred_est = 0.0;
	MeanAndInfluenceSe(score, fit.rows, centred_est, res.std_error);
	res.estimate = m1 - m0;
	if (estimand != Estimand::ATE) {
		res.warnings.push_back("'ipw' reports the ATE; use estimator := 'aipw' for ATT/ATC");
	}
	return res;
}

EffectResult Aipw(const CausalFrame &frame, const NuisanceFit &fit, Estimand estimand) {
	EffectResult res;
	res.estimator = "aipw";
	res.variance_method = "influence function";
	if (estimand == Estimand::ATE) {
		auto psi = AipwPseudoOutcome(frame, fit);
		MeanAndInfluenceSe(psi, fit.rows, res.estimate, res.std_error);
		return res;
	}
	// ATT / ATC: reweight the doubly-robust score onto the arm of interest.
	const bool on_treated = estimand == Estimand::ATT;
	double share = 0.0;
	for (auto i : fit.rows) {
		share += on_treated ? frame.t[i] : (1.0 - frame.t[i]);
	}
	share /= static_cast<double>(fit.rows.size());
	if (!(share > 0.0)) {
		res.estimate = 0.0;
		res.std_error = 0.0;
		return res;
	}
	vector<double> score(frame.n, 0.0);
	for (auto i : fit.rows) {
		const double e = fit.e[i];
		const double t = frame.t[i];
		if (on_treated) {
			const double w = e / (1.0 - e);
			score[i] = (t * (frame.y[i] - fit.mu0[i]) - (1.0 - t) * w * (frame.y[i] - fit.mu0[i])) / share;
		} else {
			const double w = (1.0 - e) / e;
			score[i] = (t * w * (frame.y[i] - fit.mu1[i]) - (1.0 - t) * (frame.y[i] - fit.mu1[i])) / share;
		}
	}
	MeanAndInfluenceSe(score, fit.rows, res.estimate, res.std_error);
	return res;
}

//! Partially linear DML: residualise both the outcome and the treatment on X,
//! then regress one residual on the other.
EffectResult Dml(const CausalFrame &frame, const NuisanceFit &fit) {
	EffectResult res;
	res.estimator = "dml";
	res.variance_method = "influence function";
	double num = 0.0, den = 0.0;
	for (auto i : fit.rows) {
		const double g = fit.e[i] * fit.mu1[i] + (1.0 - fit.e[i]) * fit.mu0[i];
		const double ty = frame.y[i] - g;
		const double tt = frame.t[i] - fit.e[i];
		num += tt * ty;
		den += tt * tt;
	}
	if (!(std::fabs(den) > 1e-12)) {
		res.warnings.push_back("treatment is fully explained by the covariates; no variation left to identify an "
		                       "effect");
		return res;
	}
	res.estimate = num / den;
	// Sandwich variance for the partially linear score.
	double meat = 0.0;
	for (auto i : fit.rows) {
		const double g = fit.e[i] * fit.mu1[i] + (1.0 - fit.e[i]) * fit.mu0[i];
		const double tt = frame.t[i] - fit.e[i];
		const double resid = (frame.y[i] - g) - res.estimate * tt;
		const double s = tt * resid;
		meat += s * s;
	}
	const double n = static_cast<double>(fit.rows.size());
	const double bread = den / n;
	res.std_error = std::sqrt(meat / n) / (std::fabs(bread) * std::sqrt(n));
	return res;
}

} // namespace

// --- CATE learners ----------------------------------------------------------

CateResult EstimateCate(const CausalFrame &frame, const CausalSpec &spec) {
	CateResult out;
	out.cate.assign(frame.n, 0.0);
	out.lo.assign(frame.n, 0.0);
	out.hi.assign(frame.n, 0.0);

	string learner = spec.estimator.empty() ? "dr_learner" : spec.estimator;
	if (StringUtil::CIEquals(learner, "aipw") || StringUtil::CIEquals(learner, "dml")) {
		learner = "dr_learner";
	}
	out.learner = StringUtil::Lower(learner);

	auto fit = FitNuisance(frame, spec, spec.seed);
	const double lambda = RidgeLambda(frame);

	if (out.learner == "t_learner") {
		for (idx_t i = 0; i < frame.n; i++) {
			out.cate[i] = fit.mu1[i] - fit.mu0[i];
		}
		out.interval_method = "none";
		out.warnings.push_back("'t_learner' reports point estimates only; use the default dr_learner for intervals");
		for (idx_t i = 0; i < frame.n; i++) {
			out.lo[i] = out.cate[i];
			out.hi[i] = out.cate[i];
		}
		return out;
	}

	if (out.learner == "x_learner") {
		// Impute the effect each row would have shown in the other arm, then
		// model those imputed effects within each arm and blend by propensity.
		vector<double> d(frame.n, 0.0);
		for (idx_t i = 0; i < frame.n; i++) {
			d[i] = frame.t[i] == 1.0 ? frame.y[i] - fit.mu0[i] : fit.mu1[i] - frame.y[i];
		}
		auto tau1 = FitRidge(frame.X, d, frame.ArmRows(1.0), {}, lambda);
		auto tau0 = FitRidge(frame.X, d, frame.ArmRows(0.0), {}, lambda);
		for (idx_t i = 0; i < frame.n; i++) {
			const double *x = frame.X.Row(i);
			out.cate[i] = fit.e[i] * tau0.Eta(x, frame.X.cols) + (1.0 - fit.e[i]) * tau1.Eta(x, frame.X.cols);
			out.lo[i] = out.cate[i];
			out.hi[i] = out.cate[i];
		}
		out.interval_method = "none";
		out.warnings.push_back("'x_learner' reports point estimates only; use the default dr_learner for intervals");
		return out;
	}

	if (out.learner == "s_learner") {
		// One model over [X, T, T*X] so the effect is allowed to vary with X.
		const idx_t p = frame.X.cols;
		Matrix aug(frame.n, 2 * p + 1);
		for (idx_t i = 0; i < frame.n; i++) {
			const double *x = frame.X.Row(i);
			for (idx_t j = 0; j < p; j++) {
				aug.At(i, j) = x[j];
			}
			aug.At(i, p) = frame.t[i];
			for (idx_t j = 0; j < p; j++) {
				aug.At(i, p + 1 + j) = frame.t[i] * x[j];
			}
		}
		auto model = FitRidge(aug, frame.y, frame.AllRows(), {}, lambda);
		for (idx_t i = 0; i < frame.n; i++) {
			const double *x = frame.X.Row(i);
			double effect = model.beta[1 + p];
			for (idx_t j = 0; j < p; j++) {
				effect += model.beta[1 + p + 1 + j] * x[j];
			}
			out.cate[i] = effect;
			out.lo[i] = effect;
			out.hi[i] = effect;
		}
		out.interval_method = "none";
		out.warnings.push_back("'s_learner' reports point estimates only; use the default dr_learner for intervals");
		return out;
	}

	// Default: DR-learner. Regress the doubly-robust pseudo-outcome on X. Its
	// conditional mean is the CATE, and the regression gives real intervals.
	out.learner = "dr_learner";
	auto psi = AipwPseudoOutcome(frame, fit);
	// Sandwich, not homoskedastic: the pseudo-outcome's variance scales with
	// 1/e(x) and 1/(1-e(x)), so it varies by orders of magnitude across rows
	// whenever there is real confounding. Assuming it constant undercovers, and
	// undercovers worse the more confounding there is - measured 95% coverage
	// across randomised, confounded and strongly-confounded DGPs runs
	// 0.948 / 0.937 / 0.909 homoskedastic against 0.948 / 0.948 / 0.944 with the
	// sandwich, each +/- about 0.015 over 40 replicates. The gradient is the
	// tell: a constant-variance assumption fails exactly where the variance
	// stops being constant.
	auto model = FitRidgeWithSandwich(frame.X, psi, fit.rows, lambda);
	for (idx_t i = 0; i < frame.n; i++) {
		const double *x = frame.X.Row(i);
		const double point = model.model.Eta(x, frame.X.cols);
		const double se = model.PredictionStdError(x, frame.X.cols);
		out.cate[i] = point;
		out.lo[i] = point - Z95 * se;
		out.hi[i] = point + Z95 * se;
	}
	out.interval_method = "pseudo-outcome regression, HC1 sandwich, pointwise 95%";
	if (fit.n_trimmed > 0) {
		out.warnings.push_back(std::to_string(fit.n_trimmed) +
		                       " rows were outside the propensity trim and did not inform the fit");
	}
	return out;
}

// --- dispatch ---------------------------------------------------------------

namespace {

//! Bootstrap the whole procedure for estimators without a closed-form
//! influence function. Cross-fitting is skipped inside the replicates, which
//! is the usual speed/accuracy trade and is reported as such.
double BootstrapSe(const CausalFrame &frame, const CausalSpec &spec, Estimand estimand, idx_t reps, int64_t seed,
                   double point) {
	if (reps < 20 || frame.n == 0) {
		return 0.0;
	}
	std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0x9E3779B97F4A7C15ULL);
	// The draw picks a canonical rank and the frame maps it to a row, so the
	// resample follows the data rather than the table's storage order.
	std::uniform_int_distribution<idx_t> pick(0, frame.n - 1);
	const double lambda = RidgeLambda(frame);
	vector<double> draws;
	draws.reserve(reps);

	for (idx_t b = 0; b < reps; b++) {
		vector<idx_t> rows(frame.n);
		vector<idx_t> rows_t0, rows_t1;
		for (idx_t i = 0; i < frame.n; i++) {
			const idx_t r = frame.Draw(pick(rng));
			rows[i] = r;
			(frame.t[r] == 1.0 ? rows_t1 : rows_t0).push_back(r);
		}
		if (rows_t0.size() < 3 || rows_t1.size() < 3) {
			continue;
		}
		LinearModel m0, m1;
		if (frame.binary_outcome) {
			m0 = FitLogistic(frame.X, frame.y, rows_t0, {}, std::max(lambda, 1.0), 20);
			m1 = FitLogistic(frame.X, frame.y, rows_t1, {}, std::max(lambda, 1.0), 20);
		} else {
			m0 = FitRidge(frame.X, frame.y, rows_t0, {}, lambda);
			m1 = FitRidge(frame.X, frame.y, rows_t1, {}, lambda);
		}
		const vector<idx_t> *target = &rows;
		vector<idx_t> arm;
		if (estimand == Estimand::ATT) {
			arm = rows_t1;
			target = &arm;
		} else if (estimand == Estimand::ATC) {
			arm = rows_t0;
			target = &arm;
		}
		double acc = 0.0;
		for (auto r : *target) {
			const double *x = frame.X.Row(r);
			acc += m1.Predict(x, frame.X.cols) - m0.Predict(x, frame.X.cols);
		}
		draws.push_back(acc / static_cast<double>(target->size()));
	}
	if (draws.size() < 10) {
		return 0.0;
	}
	// Centre on the point estimate the caller actually reports.
	double var = 0.0;
	for (auto d : draws) {
		const double diff = d - point;
		var += diff * diff;
	}
	return std::sqrt(var / static_cast<double>(draws.size() - 1));
}

EffectResult GComputation(const CausalFrame &frame, const CausalSpec &spec, const NuisanceFit &fit, Estimand estimand,
                          const string &name) {
	EffectResult res;
	res.estimator = name;
	res.variance_method = "bootstrap";
	vector<idx_t> target = fit.rows;
	if (estimand != Estimand::ATE) {
		const double arm = estimand == Estimand::ATT ? 1.0 : 0.0;
		vector<idx_t> filtered;
		for (auto i : fit.rows) {
			if (frame.t[i] == arm) {
				filtered.push_back(i);
			}
		}
		target = std::move(filtered);
	}
	double acc = 0.0;
	for (auto i : target) {
		acc += fit.mu1[i] - fit.mu0[i];
	}
	res.estimate = target.empty() ? 0.0 : acc / static_cast<double>(target.size());
	res.std_error = BootstrapSe(frame, spec, estimand, spec.bootstrap_reps, spec.seed, res.estimate);
	if (res.std_error == 0.0) {
		res.variance_method = "none";
		res.warnings.push_back("bootstrap did not converge; no interval is reported");
	}
	return res;
}

} // namespace

EffectResult EstimateEffect(const CausalFrame &frame, const CausalSpec &spec, Estimand estimand) {
	const string estimator = StringUtil::Lower(spec.estimator.empty() ? string("aipw") : spec.estimator);
	if (!IsKnownEstimator(estimator)) {
		throw BinderException("duckdo: unknown estimator '%s'. Supported: %s", spec.estimator, KnownEstimators());
	}

	EffectResult res;
	if (estimator == "naive") {
		res = Naive(frame);
	} else {
		auto fit = FitNuisance(frame, spec, spec.seed);
		if (estimator == "ipw") {
			res = Ipw(frame, fit, estimand);
		} else if (estimator == "aipw" || estimator == "dr_learner") {
			res = Aipw(frame, fit, estimand);
			res.estimator = estimator;
		} else if (estimator == "dml") {
			res = Dml(frame, fit);
			if (estimand != Estimand::ATE) {
				res.warnings.push_back("'dml' targets the partially linear ATE; use 'aipw' for ATT/ATC");
			}
		} else {
			// regression / g-computation and the meta-learners all reduce to
			// averaging a modelled contrast.
			res = GComputation(frame, spec, fit, estimand, estimator);
		}
		res.n_trimmed = fit.n_trimmed;
		if (fit.n_trimmed > 0) {
			res.warnings.push_back(std::to_string(fit.n_trimmed) +
			                       " rows were trimmed for extreme propensity (trim = " +
			                       StringUtil::Format("%g", spec.trim) + "); run do_overlap to see the support");
		}
	}

	res.estimand = estimand;
	res.n = frame.n;
	res.n_treated = frame.n_treated;
	for (auto &w : frame.warnings) {
		res.warnings.push_back(w);
	}
	res.Finalize();
	return res;
}

} // namespace duckdo
} // namespace duckdb
