"""Grade do_synth's weight solver against scipy on the same panel.

Synthetic-control weights solve a small quadratic program: least squares on the
treated unit's pre-treatment path, with the weights non-negative and summing to
one. DuckDo solves it with accelerated projected gradient onto the simplex; this
checks the answer against scipy's SLSQP on identical data.

The panel is chosen so the comparison means something. With more donors than
pre-treatment periods the optimum can be a whole face of the simplex rather than
a point, and two correct solvers can return different weights - and so different
post-treatment paths - from the same pre-treatment fit. Fifteen donors against
twenty-five pre-periods makes the problem strictly convex: one optimum, so the
weights, the synthetic path, the effect and the placebo p-value all have to
agree, not just the objective.

Dev-only. Not shipped with the extension.

    python scripts/synth_check.py [--duckdb build/release/duckdb.exe]
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import tempfile

import numpy as np


def simulate(seed=5, donors=15, periods=40, adoption=25, effect=3.0):
    """Two latent factors, and a treated unit whose loadings are a convex
    combination of three donors' - so an exact synthetic control exists."""
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    f1 = 0.1 * t + 0.3 * rng.normal(scale=0.3, size=periods).cumsum()
    f2 = np.sin(t / 4.0)
    loadings = rng.uniform(0.0, 2.0, size=(donors, 2))
    pick = rng.choice(donors, size=3, replace=False)
    treated = np.array([0.5, 0.3, 0.2]) @ loadings[pick]
    units = np.vstack([treated, loadings])  # unit 0 is the treated unit
    Y = 5.0 + 0.2 * t[None, :] + units[:, [0]] * f1[None, :] + units[:, [1]] * f2[None, :]
    Y = Y + rng.normal(scale=0.05, size=Y.shape)
    Y[0, adoption:] += effect
    return Y, adoption, effect


def write_csv(path, Y, adoption):
    with io.open(path, "w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit", "period", "treated", "y"])
        for u in range(Y.shape[0]):
            for p in range(Y.shape[1]):
                writer.writerow(["u%02d" % u, p + 1, "true" if (u == 0 and p >= adoption) else "false",
                                 "%.12f" % Y[u, p]])


def duckdo(duckdb, sql):
    proc = subprocess.run([duckdb, "-csv", "-noheader", "-c", sql], capture_output=True, encoding="utf8",
                          errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-400:])
    return [line.split(",") for line in proc.stdout.strip().splitlines()]


def solve(a, B):
    """scipy's answer to the same program, solved as tightly as SLSQP allows."""
    from scipy.optimize import minimize
    J = B.shape[1]
    result = minimize(lambda w: float(np.sum((a - B @ w) ** 2)), np.full(J, 1.0 / J),
                      jac=lambda w: -2.0 * B.T @ (a - B @ w), method="SLSQP",
                      bounds=[(0.0, 1.0)] * J,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                                    "jac": lambda w: np.ones(J)}],
                      options={"ftol": 1e-15, "maxiter": 5000})
    return result.x


def fit(Y, treated, donors, adoption):
    a, B = Y[treated, :adoption], Y[donors, :adoption].T
    w = solve(a, B)
    synthetic = Y[donors, :].T @ w
    gap = Y[treated] - synthetic
    pre = float(np.sqrt(np.mean(gap[:adoption] ** 2)))
    post = float(np.sqrt(np.mean(gap[adoption:] ** 2)))
    return w, synthetic, float(np.mean(gap[adoption:])), post / max(pre, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    args = parser.parse_args()
    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    Y, adoption, effect = simulate()
    handle, path = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    write_csv(path, Y, adoption)
    relation = "'(SELECT * FROM read_csv_auto(''%s''))'" % path.replace("\\", "/")
    named = "unit := 'unit', period := 'period', treatment := 'treated', outcome := 'y'"
    try:
        estimate_d, p_value_d = map(float, duckdo(args.duckdb, "SELECT estimate, p_value FROM do_synth(%s, %s);"
                                                  % (relation, named))[0])
        weights_d = np.zeros(Y.shape[0] - 1)
        for key, value in duckdo(args.duckdb, "SELECT e.key, e.value FROM (SELECT unnest(map_entries(weights)) AS e "
                                              "FROM do_synth(%s, %s));" % (relation, named)):
            weights_d[int(key[1:]) - 1] = float(value)
        synthetic_d = np.array([float(r[1]) for r in duckdo(
            args.duckdb, "SELECT period, synthetic FROM do_synth_path(%s, %s) ORDER BY CAST(period AS INT);"
            % (relation, named))])
    finally:
        os.unlink(path)

    donors = list(range(1, Y.shape[0]))
    weights_s, synthetic_s, estimate_s, ratio_s = fit(Y, 0, donors, adoption)
    # Placebos exactly as DuckDo runs them: each donor treated at the same
    # period, matched from the other donors, its ratio ranked against the real one.
    at_least = sum(1 for d in donors
                   if fit(Y, d, [o for o in donors if o != d], adoption)[3] >= ratio_s)
    p_value_s = (1.0 + at_least) / (1.0 + len(donors))

    a = Y[0, :adoption]
    objective_d = float(np.sum((a - synthetic_d[:adoption]) ** 2))
    objective_s = float(np.sum((a - synthetic_s[:adoption]) ** 2))

    print("%-34s %14s %14s %12s" % ("", "duckdo", "scipy", "difference"))
    print("%-34s %14.8f %14.8f %12.2e" % ("pre-treatment squared error", objective_d, objective_s,
                                          objective_d - objective_s))
    print("%-34s %14.8f %14.8f %12.2e" % ("effect (true %.1f)" % effect, estimate_d, estimate_s,
                                          abs(estimate_d - estimate_s)))
    print("%-34s %14.6f %14.6f %12.2e" % ("largest weight gap", 0, 0, float(np.max(np.abs(weights_d - weights_s)))))
    print("%-34s %14s %14s %12.2e" % ("largest synthetic-path gap", "", "",
                                      float(np.max(np.abs(synthetic_d - synthetic_s)))))
    print("%-34s %14.6f %14.6f" % ("placebo p-value", p_value_d, p_value_s))

    failures = []
    # DuckDo must be at least as good a minimiser as scipy, with room for
    # rounding: it may find a lower objective, never a meaningfully higher one.
    if objective_d > objective_s * (1.0 + 1e-6) + 1e-9:
        failures.append("objective worse than scipy's")
    if abs(estimate_d - estimate_s) > 1e-3:
        failures.append("effect differs")
    if float(np.max(np.abs(weights_d - weights_s))) > 1e-3:
        failures.append("weights differ")
    if abs(p_value_d - p_value_s) > 1e-12:
        failures.append("placebo p-value differs")
    print()
    if failures:
        print("CHECK FAILED: " + "; ".join(failures))
        return 1
    print("CHECK PASSED: DuckDo's simplex solver matches scipy's SLSQP on a strictly convex synthetic-control fit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
