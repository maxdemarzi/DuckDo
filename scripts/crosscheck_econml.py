"""Grade DuckDo's estimators against EconML and DoWhy on identical data.

This is the Phase 2 exit gate from docs/ROADMAP.md. Recovering synthetic truth
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

TOLERANCE = 0.10  # absolute difference in the ATE we are willing to accept


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
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("duckdo failed for %s: %s%s" % (estimator, proc.stdout, proc.stderr))
    parts = proc.stdout.strip().splitlines()[-1].split(",")
    return float(parts[0]), float(parts[1])


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


def try_ihdp():
    """Fetch the standard IHDP replication if the network allows it."""
    url = ("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
           "datasets/IHDP/csv/ihdp_npci_1.csv")
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
    return (X, T, Y, float((mu1 - mu0).mean())), None


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
            estimate, _ = duckdo_ate(duckdb_exe, path, "discount", "revenue", covariates, estimator)
            row["duckdo_" + estimator] = estimate
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
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    results = []
    X, T, Y, truth = scenario_linear(7)
    run_scenario("linear-confounded", X, T, Y, truth, args.duckdb, results)

    X, T, Y, truth = scenario_heterogeneous(11)
    run_scenario("heterogeneous", X, T, Y, truth, args.duckdb, results)

    ihdp, error = try_ihdp()
    if ihdp is not None:
        X, T, Y, truth = ihdp
        run_scenario("ihdp-npci-1", X, T, Y, truth, args.duckdb, results)
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
            if gap > TOLERANCE:
                failures.append("%s: %s=%.4f vs %s=%.4f (gap %.4f > %.2f)"
                                % (row["scenario"], ours, row[ours], theirs, row[theirs], gap, TOLERANCE))

    print()
    if failures:
        print("GATE FAILED")
        for failure in failures:
            print("  " + failure)
        return 1
    print("GATE PASSED: every DuckDo estimator agrees with its EconML counterpart "
          "to within %.2f on %d scenarios." % (TOLERANCE, len(results)))
    with io.open("scripts/crosscheck_results.json", "w", encoding="utf8") as handle:
        json.dump(results, handle, indent=2)
    print("wrote scripts/crosscheck_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
