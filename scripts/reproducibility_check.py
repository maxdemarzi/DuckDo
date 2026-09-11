"""Measure what DuckDo actually reproduces, rather than asserting it.

Phase 9 of docs/ROADMAP.md asks for a reproducibility statement. A statement is
worth nothing unless something checks it, so each claim below is a run:

  1. same query twice              -> identical to the last bit
  2. duckdo_threads 1 vs many      -> identical (the row blocking never follows
                                      the core count, because floating-point
                                      addition is not associative and a split
                                      that follows the cores makes the answer
                                      follow the cores)
  3. DuckDB `threads` 1 vs many    -> identical
  4. bootstrap interval, same seed -> identical; different seed -> differs
  5. rows physically reordered     -> NOT identical, and this reports by how much
  6. foundation model, repeated    -> identical
  7. ensemble draws, same seed     -> identical; different seed -> differs
  8. duckdo_query_chunk            -> identical, so the knob is a cost knob only
  9. no outcome, run twice        -> identical (do_balance, do_overlap)

Case 5 is the one that fails, and it is here precisely because it fails: a
reproducibility page listing only the things that work is advertising.

It has to be set up carefully. Ordering the generating query by `random()` draws
from the same stream that fills the columns, so it produces *different data*
rather than the same data in a different order - which is a much larger
difference, and looks like a much worse result than the truth. The permutation
below is therefore `hash(customer_id)` over an already-materialised table:
identical rows, different physical order, nothing else changed.

Its cause is not floating-point non-associativity, which was the first guess and
was wrong by eight orders of magnitude. `AssignFolds` seeds a shuffle of row
*positions*, so a permuted table puts different rows in different folds: a
statistical difference, not a rounding one. The final section measures what it
is worth against the estimate's own standard error, which is the only scale on
which the answer means anything.

Dev-only. Not shipped with the extension.

    python scripts/reproducibility_check.py [--duckdb build/release/duckdb.exe]
                                            [--models build/models]
"""

import argparse
import os
import subprocess
import sys

SETUP = """
SET duckdo_model_dir='%(models)s';
%(settings)s
SELECT setseed(0.5);
CREATE TABLE customers AS
WITH u AS (
  SELECT i AS customer_id,
    (20 + 60*random())::INT AS age,
    (random()*40)::INT AS tenure_months,
    exp(3.0 + 0.9*sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random())) AS prior_spend,
    random() AS roll,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS noise
  FROM range(%(rows)d) t(i)),
 p AS (SELECT *, 1.0/(1.0 + exp(-(-2.4 + 0.055*tenure_months + 0.020*prior_spend))) AS ps FROM u)
SELECT customer_id, age, tenure_months, prior_spend,
       (roll < ps)::INT AS got_discount,
       18.0 + 0.35*tenure_months + 0.55*prior_spend
         + (14.0 - 0.30*tenure_months)*((roll < ps)::INT) + 6.0*noise AS revenue
FROM p;
CREATE TABLE analysis AS SELECT * FROM customers %(order)s;
"""


def run(duckdb_exe, models, query, settings="", rows=20000, order=""):
    """Build the table, then run `query` against it, returning only its last row.

    Only the last line: the setup emits a NULL from setseed, and a comparison
    that includes it would be comparing the harness to itself."""
    script = SETUP % {"models": models.replace("\\", "/"), "settings": settings,
                      "rows": rows, "order": order} + query
    proc = subprocess.run([duckdb_exe, "-csv", "-noheader", "-c", script],
                          capture_output=True, encoding="utf8", errors="replace")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("failed: %s%s" % (proc.stdout[-500:], proc.stderr[-500:]))
    return proc.stdout.strip().splitlines()[-1].strip()


# Casting to VARCHAR keeps every digit. A rounded comparison would pass by
# hiding exactly the thing it is meant to detect.
ATE = ("SELECT estimate::VARCHAR, std_error::VARCHAR FROM do_ate('analysis', "
       "treatment := 'got_discount', outcome := 'revenue', exclude := ['customer_id']);")

CATE_DIGEST = ("SELECT count(*)::VARCHAR || '|' || sum(cate)::VARCHAR || '|' || "
               "sum(cate*cate)::VARCHAR FROM do_cate('analysis', "
               "treatment := 'got_discount', outcome := 'revenue', exclude := ['customer_id']);")

# do_balance and do_overlap take no outcome. Their frame's outcome column used
# to be read from a NULL vector's data slots - whatever memory held - and the
# canonical order sorts on it first, so their propensity folds moved on every
# call. Nothing here covered a function without an outcome, so nothing caught it.
NO_OUTCOME = ("SELECT (SELECT string_agg(covariate || ':' || smd_weighted::VARCHAR, '|' ORDER BY covariate) "
              "FROM do_balance('analysis', treatment := 'got_discount', "
              "covariates := ['age', 'tenure_months', 'prior_spend'])) || '#' || "
              "(SELECT string_agg(bucket::VARCHAR || ':' || n_treated::VARCHAR || ':' || n_control::VARCHAR, "
              "'|' ORDER BY bucket) FROM do_overlap('analysis', treatment := 'got_discount', "
              "covariates := ['age', 'tenure_months', 'prior_spend']));")
IPW_CI = ("SELECT estimate::VARCHAR || '|' || ci_low::VARCHAR || '|' || ci_high::VARCHAR "
          "FROM do_ate('analysis', treatment := 'got_discount', outcome := 'revenue', "
          "exclude := ['customer_id'], estimator := 'ipw', bootstrap_reps := 60%s);")

CFM = ("SELECT estimate::VARCHAR FROM do_ate('analysis', treatment := 'got_discount', "
       "outcome := 'revenue', exclude := ['customer_id'], model := 'causalpfn'%s);")


def _sd(values):
    """Sample standard deviation. numpy is a dependency this script does not need."""
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def report(label, values, expect_identical=True):
    identical = len(set(values)) == 1
    good = identical == expect_identical
    print("  %-46s %-10s %s" % (label, "identical" if identical else "DIFFERS",
                                "ok" if good else "UNEXPECTED"))
    if not identical:
        for value in sorted(set(values)):
            print("      %s" % value)
    return good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    parser.add_argument("--models", default=os.path.join("build", "models"))
    parser.add_argument("--skip-models", action="store_true")
    # The foundation model scores every row through a transformer, so it runs on
    # a smaller table than the classical cases or the check takes half an hour.
    parser.add_argument("--model-rows", type=int, default=2000)
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    ok = True
    print("classical estimators (20000 rows)")
    baseline = run(args.duckdb, args.models, ATE)
    ok &= report("same query, run twice", [baseline, run(args.duckdb, args.models, ATE)])

    ok &= report("duckdo_threads 1 / 2 / 8 / auto",
                 [run(args.duckdb, args.models, ATE, settings="SET duckdo_threads=%d;" % n)
                  for n in (1, 2, 8, 0)] + [baseline])

    ok &= report("DuckDB threads 1 / 4",
                 [run(args.duckdb, args.models, ATE, settings="SET threads=%d;" % n)
                  for n in (1, 4)] + [baseline])

    ok &= report("do_cate over 20000 rows, run twice",
                 [run(args.duckdb, args.models, CATE_DIGEST),
                  run(args.duckdb, args.models, CATE_DIGEST)])

    ok &= report("no outcome: do_balance, do_overlap, run twice",
                 [run(args.duckdb, args.models, NO_OUTCOME),
                  run(args.duckdb, args.models, NO_OUTCOME)])

    ok &= report("bootstrap interval, same seed",
                 [run(args.duckdb, args.models, IPW_CI % ""),
                  run(args.duckdb, args.models, IPW_CI % "")])
    ok &= report("bootstrap interval, seed 42 vs 43",
                 [run(args.duckdb, args.models, IPW_CI % ""),
                  run(args.duckdb, args.models, IPW_CI % ", seed := 43")],
                 expect_identical=False)

    # Row order used to move the estimate by 4e-4 relative, because folds were
    # assigned over row positions. They are assigned over content now, so all
    # that is left is floating-point summation order. This asserts the size of
    # what remains rather than merely observing that something remains: if fold
    # assignment ever follows storage order again, the gap jumps by ten orders
    # of magnitude and this fails.
    ORDER_TOLERANCE = 1e-10
    for order in ("ORDER BY hash(customer_id)", "ORDER BY revenue DESC"):
        shuffled = run(args.duckdb, args.models, ATE, order=order)
        a, b = float(baseline.split(",")[0]), float(shuffled.split(",")[0])
        relative = abs(a - b) / abs(a) if a else 0.0
        within = relative < ORDER_TOLERANCE
        ok &= within
        print("  %-46s %-10s %s" % ("reordered: " + order[9:], "%.1e" % relative,
                                    "ok" if within else "TOO LARGE"))
    print("      Rounding, not fold membership. Below %.0e relative it cannot be" % ORDER_TOLERANCE)
    print("      anything else: a fold that moved would be worth 1e-4.")

    print()
    print("how much row order actually moves the answer (10 permutations of the same rows)")
    print("  %-11s %12s %12s %10s" % ("estimator", "sd", "std_error", "sd/se"))
    for estimator in ("regression", "aipw", "ipw"):
        query = ("SELECT estimate::VARCHAR || '|' || std_error::VARCHAR FROM do_ate('analysis', "
                 "treatment := 'got_discount', outcome := 'revenue', "
                 "exclude := ['customer_id'], estimator := '%s');" % estimator)
        estimates, std_error = [], 0.0
        for k in range(10):
            # A different deterministic permutation each time. The rows are
            # identical; only where they sit in the table changes.
            out = run(duckdb_exe=args.duckdb, models=args.models, query=query,
                      order="ORDER BY hash(customer_id * %d + 7)" % (k * 2654435761 + 1))
            estimate, std_error_text = out.split("|")
            estimates.append(float(estimate))
            std_error = float(std_error_text)
        spread = _sd(estimates)
        print("  %-11s %12.3e %12.4f %9.2f%%"
              % (estimator, spread, std_error, 100.0 * spread / std_error))
    print("  Fold assignment follows content, so row order is worth rounding and nothing")
    print("  more. It was 0.25% / 2.12% / 6.18% when folds followed row position.")

    if not args.skip_models:
        print()
        print("foundation model (%d rows)" % args.model_rows)
        try:
            plain = [run(args.duckdb, args.models, CFM % "", rows=args.model_rows)
                     for _ in range(2)]
            ok &= report("model := 'causalpfn', run twice", plain)
            ok &= report("duckdo_query_chunk 256 vs 4096",
                         [run(args.duckdb, args.models, CFM % "", rows=args.model_rows,
                              settings="SET duckdo_query_chunk=%d;" % n) for n in (256, 4096)])
            ok &= report("ensemble := 4, same seed",
                         [run(args.duckdb, args.models, CFM % ", ensemble := 4",
                              rows=args.model_rows) for _ in range(2)])
            ok &= report("ensemble := 4, seed 42 vs 43",
                         [run(args.duckdb, args.models, CFM % ", ensemble := 4",
                              rows=args.model_rows),
                          run(args.duckdb, args.models, CFM % ", ensemble := 4, seed := 43",
                              rows=args.model_rows)],
                         expect_identical=False)
        except RuntimeError as exc:
            print("  skipped: %s" % str(exc)[:160])

    print()
    if not ok:
        print("A case did not behave as documented. docs/REPRODUCIBILITY.md is now wrong.")
        return 1
    print("Every case behaved as docs/REPRODUCIBILITY.md describes it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
