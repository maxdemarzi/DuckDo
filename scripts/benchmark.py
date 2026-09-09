"""Track DuckDo's estimation cost across table sizes and estimators.

Phase 8 of the roadmap asks for a fixed benchmark set so regressions are visible
rather than discovered. This prints a table and, with --json, writes one that can
be diffed between commits.

Dev-only. Not shipped with the extension.

    python scripts/benchmark.py [--duckdb build/release/duckdb.exe] [--json bench.json]
"""

import argparse
import json
import os
import subprocess
import sys
import time

# (rows, covariates, estimator). The 1M x 50 row is the roadmap's stated gate.
CASES = [
    (100_000, 5, "aipw"),
    (100_000, 50, "aipw"),
    (1_000_000, 5, "aipw"),
    (1_000_000, 50, "aipw"),
    (1_000_000, 50, "dml"),
    (100_000, 5, "regression"),
    (100_000, 5, "ipw"),
]

GATE_SECONDS = 30.0


def build_sql(rows, covariates, estimator):
    noise = "sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random())"
    cols = ", ".join("%s AS x%d" % (noise, i) for i in range(1, covariates + 1))
    sel = ", ".join("x%d" % i for i in range(1, covariates + 1))
    return """
SET duckdo_max_rows=2000000;
SELECT setseed(0.1);
CREATE OR REPLACE TABLE bench AS
WITH u AS (SELECT i, {cols}, {noise} AS noise, random() AS draw FROM range({rows}) t(i)),
     p AS (SELECT *, 1.0/(1.0+exp(-(0.9*x1-0.6*x2))) AS ps FROM u)
SELECT i AS id, {sel}, (draw<ps)::INTEGER AS t,
       2.0+1.5*x1+0.7*x2+3.0*((draw<ps)::INTEGER)+noise AS y FROM p;
SELECT estimate FROM do_ate('bench', treatment:='t', outcome:='y',
                            exclude:=['id'], estimator:='{estimator}');
""".format(cols=cols, sel=sel, noise=noise, rows=rows, estimator=estimator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    print("%10s %6s %-12s %10s %12s" % ("rows", "covs", "estimator", "seconds", "estimate"))
    print("-" * 56)
    results = []
    over_gate = []
    for rows, covariates, estimator in CASES:
        sql = build_sql(rows, covariates, estimator)
        started = time.time()
        proc = subprocess.run([args.duckdb, "-csv", "-noheader", "-c", sql],
                              capture_output=True, text=True)
        elapsed = time.time() - started
        if proc.returncode != 0:
            print("FAILED %s" % (proc.stderr.strip()[:200]))
            return 1
        # The generation query is included in the timing, so this is an upper
        # bound on the estimator's own cost - which is the honest direction.
        estimate = float(proc.stdout.strip().splitlines()[-1])
        print("%10d %6d %-12s %10.2f %12.4f" % (rows, covariates, estimator, elapsed, estimate))
        results.append(dict(rows=rows, covariates=covariates, estimator=estimator,
                            seconds=round(elapsed, 3), estimate=estimate))
        if rows >= 1_000_000 and covariates >= 50 and elapsed > GATE_SECONDS:
            over_gate.append((rows, covariates, estimator, elapsed))

    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle, indent=2)
        print("\nwrote %s" % args.json)

    print()
    if over_gate:
        print("GATE FAILED: %d case(s) over %.0fs" % (len(over_gate), GATE_SECONDS))
        for rows, covariates, estimator, elapsed in over_gate:
            print("  %d x %d %s took %.1fs" % (rows, covariates, estimator, elapsed))
        return 1
    print("GATE PASSED: 1M rows x 50 covariates stays under %.0fs (timings include "
          "generating the table)" % GATE_SECONDS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
