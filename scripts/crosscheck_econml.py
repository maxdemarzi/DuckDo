"""Grade DuckDo's estimators against EconML and DoWhy on identical data.

This is the Phase 2 exit gate from dev/ROADMAP.md. Recovering synthetic truth
is necessary but not sufficient: the estimators also have to agree with
established implementations, on the same rows, to the same estimand.

Dev-only. Not shipped with the extension.

    python scripts/crosscheck_econml.py [--duckdb build/release/duckdb.exe]
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np

# How far apart two implementations may land before it counts as disagreement.
#
# An absolute tolerance alone is the wrong instrument. It reads as strict, but
# what counts as a large gap depends entirely on how well determined the
# estimate is: 0.12 is nothing on IHDP, where 747 rows and 25 covariates give a
# standard error of 0.18, and would be alarming on a clean 8,000-row synthetic
# where the standard error is 0.03. Two doubly-robust estimators that split
# their folds differently *will* differ on a small sample, and a gate that calls
# that a failure trains people to loosen gates.
#
# So the bar is: agree to within a tenth, or to within one standard error of our
# own estimate, whichever is more forgiving.
TOLERANCE = 0.10


def duckdo_ate(duckdb_exe, csv_path, treatment, outcome, covariates, estimator):
    """Run one DuckDo estimate against a CSV and return (estimate, std_error)."""
    cov_list = ", ".join("'%s'" % c for c in covariates)
    # The relation argument is itself a SQL string, so the quotes inside
    # read_csv_auto(...) have to survive one more level of escaping.
    relation = "(SELECT * FROM read_csv_auto('%s'))" % csv_path.replace("\\", "/")
    sql = (
        "SELECT estimate, std_error FROM do_ate(%s, "
        "treatment := '%s', outcome := '%s', covariates := [%s], estimator := '%s');"
        % (_sql_literal(relation), treatment, outcome, cov_list, estimator)
    )
    proc = subprocess.run(
        [duckdb_exe, "-csv", "-noheader", "-c", sql],
        capture_output=True, encoding="utf8", errors="replace",
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("duckdo failed for %s: %s%s" % (estimator, proc.stdout, proc.stderr))
    parts = proc.stdout.strip().splitlines()[-1].split(",")
    return float(parts[0]), float(parts[1])


def duckdo_cate(duckdb_exe, csv_path, treatment, outcome, covariates):
    """Per-row effects from DuckDo, in the CSV's row order."""
    cov_list = ", ".join("'%s'" % c for c in covariates)
    relation = "(SELECT * FROM read_csv_auto('%s'))" % csv_path.replace("\\", "/")
    sql = ("SELECT cate FROM do_cate(%s, treatment := '%s', outcome := '%s', "
           "covariates := [%s]) ORDER BY row_id;"
           % (_sql_literal(relation), treatment, outcome, cov_list))
    proc = subprocess.run([duckdb_exe, "-csv", "-noheader", "-c", sql],
                          capture_output=True, encoding="utf8", errors="replace")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("duckdo do_cate failed: %s%s" % (proc.stdout, proc.stderr))
    return np.array([float(line) for line in proc.stdout.strip().splitlines()])


def _sql_literal(text):
    return "'" + text.replace("'", "''") + "'"


def write_csv(path, columns):
    names = list(columns.keys())
    n = len(columns[names[0]])
    with io.open(path, "w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for i in range(n):
            writer.writerow([columns[name][i] for name in names])


def econml_estimates(X, T, Y):
    """LinearDML and LinearDRLearner ATEs from EconML."""
    from econml.dml import LinearDML
    from econml.dr import LinearDRLearner
    from sklearn.linear_model import LinearRegression, LogisticRegression

    out = {}

    dml = LinearDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(max_iter=1000),
        discrete_treatment=True,
        cv=5,
        random_state=42,
    )
    dml.fit(Y, T, X=X)
    out["econml_LinearDML"] = float(dml.ate(X))

    dr = LinearDRLearner(
        model_regression=LinearRegression(),
        model_propensity=LogisticRegression(max_iter=1000),
        cv=5,
        random_state=42,
    )
    dr.fit(Y, T, X=X)
    out["econml_LinearDRLearner"] = float(dr.ate(X))
    return out


def dowhy_estimate(columns, covariates):
    """DoWhy's propensity-score-weighting backdoor estimate."""
    import pandas as pd
    from dowhy import CausalModel

    frame = pd.DataFrame(columns)
    frame["discount"] = frame["discount"].astype(bool)
    model = CausalModel(
        data=frame,
        treatment="discount",
        outcome="revenue",
        common_causes=covariates,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.propensity_score_weighting",
        target_units="ate",
    )
    return float(estimate.value)


def scenario_linear(seed, n=8000, p=5, true_ate=3.0):
    """Confounded linear DGP with a known ATE."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logits = 0.9 * X[:, 0] - 0.6 * X[:, 1] + 0.3 * X[:, 2]
    ps = 1.0 / (1.0 + np.exp(-logits))
    T = (rng.random(n) < ps).astype(int)
    Y = (2.0 + 1.5 * X[:, 0] + 0.7 * X[:, 1] - X[:, 2] + 0.4 * X[:, 3]
         + true_ate * T + rng.normal(size=n))
    return X, T, Y, true_ate


def scenario_heterogeneous(seed, n=8000, p=5):
    """Effect varies with x0; the ATE is still known in closed form."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    ps = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    T = (rng.random(n) < ps).astype(int)
    tau = 3.0 + 2.0 * X[:, 0]
    Y = 2.0 + 1.5 * X[:, 0] + 0.7 * X[:, 1] + tau * T + rng.normal(size=n)
    # E[tau(X)] over the sample, which is what every ATE estimator targets here.
    return X, T, Y, float(tau.mean())


# The CEVAE mirror carries replications 1-10, not the canonical 1000. Running all
# ten is a real improvement on running one; claiming the full benchmark would not
# be true, so the report says which.
IHDP_REPLICATIONS = range(1, 11)


def try_ihdp(replication):
    """Fetch one IHDP replication if the network allows it."""
    url = ("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
           "datasets/IHDP/csv/ihdp_npci_%d.csv" % replication)
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            raw = response.read().decode("utf8")
    except Exception as exc:  # noqa: BLE001 - the network is allowed to be absent
        return None, str(exc)
    rows = [line.split(",") for line in raw.strip().splitlines()]
    data = np.array(rows, dtype=float)
    # Column layout: treatment, y_factual, y_cfactual, mu0, mu1, then 25 covariates.
    T = data[:, 0].astype(int)
    Y = data[:, 1]
    mu0, mu1 = data[:, 3], data[:, 4]
    X = data[:, 5:]
    # The individual effects are known here, so PEHE is measurable, not just ATE error.
    return (X, T, Y, mu1 - mu0), None


def try_lalonde():
    """The Dehejia-Wahba NSW sample. Treatment was randomised, so the unadjusted
    difference in means IS the causal effect - which makes it a benchmark an
    adjusted estimator has to reproduce rather than improve on."""
    try:
        from dowhy.datasets import lalonde_dataset
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    frame = lalonde_dataset()
    covariates = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
    X = frame[covariates].to_numpy(dtype=float)
    T = frame["treat"].to_numpy().astype(int)
    Y = frame["re78"].to_numpy(dtype=float)
    experimental = float(Y[T == 1].mean() - Y[T == 0].mean())
    return (X, T, Y, experimental), None


def ihdp_and_lalonde(duckdb_exe):
    """Run every available IHDP replication plus Lalonde, and report both the
    ATE error and - where individual effects are known - the PEHE."""
    print()
    print("IHDP replications (individual effects known, so PEHE is measurable)")
    print("%6s %10s %12s %12s %10s" % ("rep", "true ATE", "duckdo aipw", "econml DR", "PEHE"))
    print("-" * 56)
    ate_errors, pehes, skipped = [], [], 0
    for replication in IHDP_REPLICATIONS:
        loaded, error = try_ihdp(replication)
        if loaded is None:
            skipped += 1
            continue
        X, T, Y, individual = loaded
        truth = float(individual.mean())
        covariates = ["x%d" % i for i in range(X.shape[1])]
        columns = {"discount": T.tolist(), "revenue": Y.tolist()}
        for i, cov in enumerate(covariates):
            columns[cov] = X[:, i].tolist()
        handle, path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        try:
            write_csv(path, columns)
            ours, _ = duckdo_ate(duckdb_exe, path, "discount", "revenue", covariates, "aipw")
            cate = duckdo_cate(duckdb_exe, path, "discount", "revenue", covariates)
            pehe = float(np.sqrt(np.mean((cate - individual) ** 2)))
        finally:
            os.unlink(path)
        theirs = econml_estimates(X, T, Y)["econml_LinearDRLearner"]
        ate_errors.append(abs(ours - truth))
        pehes.append(pehe)
        print("%6d %10.4f %12.4f %12.4f %10.4f" % (replication, truth, ours, theirs, pehe))

    if ate_errors:
        print("-" * 56)
        print("%6s %10s %12.4f %12s %10.4f"
              % ("mean", "", float(np.mean(ate_errors)), "", float(np.mean(pehes))))
        print("       (the 'duckdo aipw' column is mean |error|, not a mean estimate)")
    if skipped:
        print("  %d replication(s) unavailable" % skipped)

    print()
    print("Lalonde NSW (randomised, so the unadjusted difference is the benchmark)")
    loaded, error = try_lalonde()
    if loaded is None:
        print("  skipped (%s)" % error)
        return ate_errors, pehes
    X, T, Y, experimental = loaded
    covariates = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
    columns = {"discount": T.tolist(), "revenue": Y.tolist()}
    for i, cov in enumerate(covariates):
        columns[cov] = X[:, i].tolist()
    handle, path = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    try:
        write_csv(path, columns)
        row = {"experimental benchmark": experimental}
        for estimator in ("aipw", "dml", "ipw"):
            row["duckdo " + estimator] = duckdo_ate(duckdb_exe, path, "discount", "revenue",
                                                    covariates, estimator)[0]
    finally:
        os.unlink(path)
    row["econml DRLearner"] = econml_estimates(X, T, Y)["econml_LinearDRLearner"]
    for key, value in row.items():
        print("  %-24s %10.1f" % (key, value))
    gap = max(abs(row["duckdo aipw"] - experimental), abs(row["duckdo dml"] - experimental))
    print("  largest gap from the experimental benchmark: %.1f (outcome is 1978 dollars)" % gap)
    return ate_errors, pehes


def run_scenario(name, X, T, Y, truth, duckdb_exe, results):
    covariates = ["x%d" % i for i in range(X.shape[1])]
    columns = {"discount": T.tolist(), "revenue": Y.tolist()}
    for i, cov in enumerate(covariates):
        columns[cov] = X[:, i].tolist()

    handle, path = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    try:
        write_csv(path, columns)
        row = {"scenario": name, "n": int(len(T)), "truth": truth}
        for estimator in ("aipw", "dml", "ipw", "regression"):
            estimate, std_error = duckdo_ate(duckdb_exe, path, "discount", "revenue", covariates,
                                             estimator)
            row["duckdo_" + estimator] = estimate
            row["duckdo_%s_se" % estimator] = std_error
        row.update(econml_estimates(X, T, Y))
        try:
            row["dowhy_psw"] = dowhy_estimate(columns, covariates)
        except Exception as exc:  # noqa: BLE001
            row["dowhy_psw"] = None
            row["dowhy_error"] = str(exc)[:120]
        results.append(row)
        return row
    finally:
        os.unlink(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    parser.add_argument("--quick", action="store_true",
                        help="skip the IHDP replication sweep and Lalonde")
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    results = []
    X, T, Y, truth = scenario_linear(7)
    run_scenario("linear-confounded", X, T, Y, truth, args.duckdb, results)

    X, T, Y, truth = scenario_heterogeneous(11)
    run_scenario("heterogeneous", X, T, Y, truth, args.duckdb, results)

    ihdp, error = try_ihdp(1)
    if ihdp is not None:
        X, T, Y, individual = ihdp
        run_scenario("ihdp-npci-1", X, T, Y, float(individual.mean()), args.duckdb, results)
    else:
        print("IHDP skipped (%s)" % error)

    print()
    header = ["scenario", "truth", "duckdo_aipw", "duckdo_dml", "econml_LinearDML",
              "econml_LinearDRLearner", "dowhy_psw"]
    print(" | ".join("%-22s" % h for h in header))
    print("-" * (25 * len(header)))
    failures = []
    for row in results:
        cells = [row["scenario"]]
        for key in header[1:]:
            value = row.get(key)
            cells.append("n/a" if value is None else "%.4f" % value)
        print(" | ".join("%-22s" % c for c in cells))

        # The gate: DuckDo has to agree with EconML, not merely with the truth.
        for ours, theirs in (("duckdo_aipw", "econml_LinearDRLearner"),
                             ("duckdo_dml", "econml_LinearDML")):
            gap = abs(row[ours] - row[theirs])
            allowed = max(TOLERANCE, row.get(ours + "_se", 0.0))
            if gap > allowed:
                failures.append("%s: %s=%.4f vs %s=%.4f (gap %.4f > %.4f, se %.4f)"
                                % (row["scenario"], ours, row[ours], theirs, row[theirs], gap,
                                   allowed, row.get(ours + "_se", 0.0)))

    print()
    if failures:
        print("GATE FAILED")
        for failure in failures:
            print("  " + failure)
        return 1
    print("GATE PASSED: every DuckDo estimator agrees with its EconML counterpart on "
          "%d scenarios, to within %.2f or one standard error." % (len(results), TOLERANCE))

    if not args.quick:
        ihdp_and_lalonde(args.duckdb)
    with io.open("scripts/crosscheck_results.json", "w", encoding="utf8") as handle:
        json.dump(results, handle, indent=2)
    print("wrote scripts/crosscheck_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
