"""Measure empirical coverage of do_cate's pointwise intervals.

A 95% interval that covers 80% of the time is worse than no interval, because it
invites confidence it has not earned. This script generates DGPs with a known
per-row effect, runs do_cate, and reports the fraction of rows whose interval
actually contains the truth.

Dev-only. Not shipped with the extension.

    python scripts/coverage_check.py [--duckdb build/release/duckdb.exe] [--reps 12]
"""

import argparse
import math
import os
import subprocess
import sys

# Three DGPs. The confounded one is where a homoskedastic interval is most
# likely to be too narrow, because extreme propensity scores make the
# doubly-robust pseudo-outcome's variance vary by orders of magnitude.
SCENARIOS = {
    "randomised": dict(ps="0.5", tau="3.0 + 2.0*x1"),
    "confounded": dict(ps="1.0/(1.0+exp(-(0.9*x1 - 0.6*x2)))", tau="3.0 + 2.0*x1"),
    "strong-confounding": dict(ps="1.0/(1.0+exp(-(1.8*x1 - 1.2*x2)))", tau="3.0 + 2.0*x1"),
}

SQL = """
SELECT setseed({seed});
CREATE OR REPLACE TABLE d AS
WITH u AS (
  SELECT i,
         sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS x1,
         sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS x2,
         sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS x3,
         sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS noise,
         random() AS draw
  FROM range({n}) t(i)
), p AS (SELECT *, {ps} AS ps FROM u)
SELECT i AS id, x1, x2, x3,
       (draw < ps)::INTEGER AS t,
       2.0 + 1.5*x1 + 0.7*x2 - x3 + ({tau})*((draw < ps)::INTEGER) + noise AS y,
       ({tau}) AS true_cate
FROM p;

SELECT avg((c.cate_low <= d.true_cate AND d.true_cate <= c.cate_high)::INT) AS coverage,
       avg(c.cate_high - c.cate_low) AS mean_width,
       corr(c.cate, d.true_cate) AS corr
FROM do_cate('d', treatment := 't', outcome := 'y', exclude := ['id', 'true_cate'],
             id := 'id') c
JOIN d ON d.id = c.id::BIGINT;
"""


def run(duckdb_exe, scenario, seed, n):
    spec = SCENARIOS[scenario]
    sql = SQL.format(seed=seed, n=n, ps=spec["ps"], tau=spec["tau"])
    proc = subprocess.run([duckdb_exe, "-csv", "-noheader", "-c", sql],
                          capture_output=True, encoding="utf8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError("duckdb failed: %s%s" % (proc.stdout, proc.stderr))
    rows = [line for line in proc.stdout.strip().splitlines() if line.count(",") == 2]
    if not rows:
        raise RuntimeError("no result rows: %s%s" % (proc.stdout, proc.stderr))
    coverage, width, corr = rows[-1].split(",")
    return float(coverage), float(width), float(corr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    # Coverage per replicate is extremely variable: one replicate's rows all
    # share the same fitted nuisance models, so a single bad draw drags its whole
    # coverage down. At 12 replicates the reported mean moved by 0.12 between
    # runs at different rep counts, which is enough to invent a finding that is
    # not there. 40 is the point at which the standard error is small enough for
    # the number to mean something.
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--n", type=int, default=4000)
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    print("%-20s %10s %8s %14s %10s %10s"
          % ("scenario", "coverage", "+/- se", "range", "width", "corr"))
    print("-" * 78)
    worst = 1.0
    worst_scenario = ""
    for scenario in SCENARIOS:
        coverages, widths, corrs = [], [], []
        for rep in range(args.reps):
            # setseed wants [-1, 1].
            seed = -0.9 + 1.8 * rep / max(args.reps - 1, 1)
            coverage, width, corr = run(args.duckdb, scenario, round(seed, 4), args.n)
            coverages.append(coverage)
            widths.append(width)
            corrs.append(corr)
        mean_cov = sum(coverages) / len(coverages)
        # Standard error of the mean across replicates. Replicates are the
        # independent unit here - rows within one share its fitted models - so
        # this is what says whether a gap from nominal is real.
        if len(coverages) > 1:
            var = sum((c - mean_cov) ** 2 for c in coverages) / (len(coverages) - 1)
            se = math.sqrt(var / len(coverages))
        else:
            se = float("nan")
        if mean_cov < worst:
            worst, worst_scenario = mean_cov, scenario
        print("%-20s %10.3f %8.3f %14s %10.3f %10.4f"
              % (scenario, mean_cov, se, "%.2f-%.2f" % (min(coverages), max(coverages)),
                 sum(widths) / len(widths), sum(corrs) / len(corrs)))

    print()
    print("nominal 0.95; worst scenario is %s at %.3f, over %d reps of %d rows"
          % (worst_scenario, worst, args.reps, args.n))
    print("Report coverage with its standard error. A single replicate covers between")
    print("half and nearly all of its rows depending on where its nuisance fits land,")
    print("so a mean quoted without one is not a measurement.")
    # Coverage this far below nominal means the interval is misleading, not merely
    # imprecise, so treat it as a failure.
    return 0 if worst >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
