"""Grade DuckDo against the Python causal stack on identical rows, and time it.

`scripts/crosscheck_econml.py` answers "does DuckDo agree with EconML?" - a
correctness gate. This answers the different question the docs have to answer
honestly: on the standard benchmarks, *who is closer to the truth, and what does
each one cost*. Classical DuckDo, DuckDo running CausalPFN through ONNX, EconML,
DoWhy, and the CausalPFN package itself, all on the same rows.

Two metrics, because they disagree and the disagreement is the point:

  |ATE error|  distance from the known average effect. What a report quotes.
  PEHE         root mean squared error of the *per-row* effect, where the DGP
               exposes individual truths. What a targeting rule actually rides
               on. An estimator can win the first and lose the second badly.

Every method runs in its own process. That is not tidiness: torch and
scikit-learn each bring their own OpenMP runtime, and loading both into one
interpreter segfaults this machine part-way through the sweep. Isolation also
stops one library's thread pool from warming or starving the next one's timing.

Timings are wall-clock around the fit and the estimate only, measured inside the
process that does the work. Model construction and weight loading are excluded,
as is interpreter start - the framing that flatters Python, so it is the one
used. DuckDo's figure is the whole subprocess, including start-up and CSV
parsing, so its side of the comparison is if anything unfair to DuckDo.

Dev-only. Not shipped with the extension. Writes scripts/bench_results.json.

    python scripts/bench_headtohead.py [--duckdb build/release/duckdb.exe]
                                       [--models build/models] [--quick]
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

CACHE = os.path.join(tempfile.gettempdir(), "duckdo_bench_cache")


# ---------------------------------------------------------------- DuckDo side

def _sql_literal(text):
    return "'" + text.replace("'", "''") + "'"


def duckdo_run(duckdb_exe, models, sql):
    """Run one statement, with the model directory set, and return stdout rows."""
    prelude = "SET duckdo_model_dir='%s'; " % models.replace("\\", "/") if models else ""
    proc = subprocess.run([duckdb_exe, "-csv", "-noheader", "-c", prelude + sql],
                          capture_output=True, encoding="utf8", errors="replace")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("duckdo failed: %s%s" % (proc.stdout[-400:], proc.stderr[-400:]))
    return proc.stdout.strip().splitlines()


def duckdo_relation(csv_path):
    return _sql_literal("(SELECT * FROM read_csv_auto('%s'))" % csv_path.replace("\\", "/"))


def duckdo_ate(duckdb_exe, models, csv_path, covariates, spec):
    """spec is either estimator := 'aipw' or model := 'causalpfn'."""
    cov_list = ", ".join("'%s'" % c for c in covariates)
    sql = ("SELECT estimate FROM do_ate(%s, treatment := 't', outcome := 'y', "
           "covariates := [%s], %s);" % (duckdo_relation(csv_path), cov_list, spec))
    return float(duckdo_run(duckdb_exe, models, sql)[-1])


def duckdo_cate(duckdb_exe, models, csv_path, covariates, spec):
    cov_list = ", ".join("'%s'" % c for c in covariates)
    sql = ("SELECT cate FROM do_cate(%s, treatment := 't', outcome := 'y', "
           "covariates := [%s], %s) ORDER BY row_id;"
           % (duckdo_relation(csv_path), cov_list, spec))
    return np.array([float(line) for line in duckdo_run(duckdb_exe, models, sql)])


# ---------------------------------------------------------- Python competitors
#
# Each returns (ate, per-row effects or None). Setup that is not part of the
# estimate - loading 75 MB of transformer weights, say - belongs in `build_call`
# below, so it stays outside the timed region.

def econml_dr(X, T, Y):
    from econml.dr import LinearDRLearner
    from sklearn.linear_model import LinearRegression, LogisticRegression
    model = LinearDRLearner(model_regression=LinearRegression(),
                            model_propensity=LogisticRegression(max_iter=1000),
                            cv=5, random_state=42)
    model.fit(Y, T, X=X)
    return float(model.ate(X)), model.effect(X).reshape(-1)


def econml_dml(X, T, Y):
    from econml.dml import LinearDML
    from sklearn.linear_model import LinearRegression, LogisticRegression
    model = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(max_iter=1000),
                      discrete_treatment=True, cv=5, random_state=42)
    model.fit(Y, T, X=X)
    return float(model.ate(X)), model.effect(X).reshape(-1)


def econml_forest(X, T, Y):
    """A nonlinear base learner, which is where EconML earns its keep on PEHE."""
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42),
        discrete_treatment=True, cv=5, n_estimators=500, random_state=42)
    model.fit(Y, T, X=X)
    return float(model.ate(X)), model.effect(X).reshape(-1)


def dowhy_psw(X, T, Y):
    import pandas as pd
    from dowhy import CausalModel
    covariates = ["x%d" % i for i in range(X.shape[1])]
    frame = pd.DataFrame({c: X[:, i] for i, c in enumerate(covariates)})
    frame["t"] = T.astype(bool)
    frame["y"] = Y
    model = CausalModel(data=frame, treatment="t", outcome="y", common_causes=covariates)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting",
                                     target_units="ate")
    return float(estimate.value), None


def build_call(key, X, T, Y):
    """A zero-argument callable with all setup already done, so the clock covers
    the estimate rather than the import and the weight load."""
    if key == "econml_dr":
        return lambda: econml_dr(X, T, Y)
    if key == "econml_dml":
        return lambda: econml_dml(X, T, Y)
    if key == "econml_forest":
        return lambda: econml_forest(X, T, Y)
    if key == "dowhy_psw":
        return lambda: dowhy_psw(X, T, Y)
    if key in ("cpfn_cpu", "cpfn_gpu"):
        from causalpfn import CATEEstimator
        estimator = CATEEstimator(device="cuda" if key == "cpfn_gpu" else "cpu")
        features = X.astype(np.float32)

        def call():
            estimator.fit(features, T.astype(np.int64), Y.astype(np.float32))
            cate = np.asarray(estimator.estimate_cate(features)).reshape(-1)
            return float(cate.mean()), cate
        return call
    raise KeyError(key)


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch is an answer, not a crash
        return False


# --------------------------------------------------------------------- worker
#
# One method, one interpreter. The parent process never imports torch, sklearn
# or pandas, which is what stops the two OpenMP runtimes from meeting.

def worker(key, data_path, result_path):
    payload = np.load(data_path)
    X, T, Y = payload["X"], payload["T"], payload["Y"]
    call = build_call(key, X, T, Y)
    start = time.perf_counter()
    estimate, cate = call()
    seconds = time.perf_counter() - start
    result = {"estimate": float(estimate), "seconds": seconds}
    if cate is not None:
        np.save(result_path + ".npy", np.asarray(cate, dtype=float).reshape(-1))
        result["cate"] = True
    with io.open(result_path, "w", encoding="utf8") as handle:
        json.dump(result, handle)
    return 0


def run_python_method(key, data_path):
    handle, result_path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        proc = subprocess.run(
            [sys.executable, "-W", "ignore", os.path.abspath(__file__),
             "--worker", key, "--data", data_path, "--result", result_path],
            capture_output=True, encoding="utf8", errors="replace")
        if proc.returncode != 0 or not os.path.getsize(result_path):
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError((tail[-1] if tail else "exit %d" % proc.returncode)[:200])
        with io.open(result_path, encoding="utf8") as source:
            result = json.load(source)
        cate = np.load(result_path + ".npy") if result.get("cate") else None
        return (result["estimate"], cate), result["seconds"]
    finally:
        for path in (result_path, result_path + ".npy"):
            if os.path.exists(path):
                os.unlink(path)


# ------------------------------------------------------------------- datasets

def write_csv(path, X, T, Y, covariates):
    with io.open(path, "w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "y"] + covariates)
        for i in range(len(T)):
            writer.writerow([int(T[i]), float(Y[i])] + [float(v) for v in X[i]])


def ihdp(replication):
    """One IHDP replication from the CEVAE mirror, cached on disk.

    Individual potential outcomes are known here, so PEHE is measurable rather
    than assumed. The mirror carries replications 1-10, not the canonical 1000."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "ihdp_npci_%d.csv" % replication)
    if not os.path.exists(path):
        url = ("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
               "datasets/IHDP/csv/ihdp_npci_%d.csv" % replication)
        with urllib.request.urlopen(url, timeout=30) as response:
            with io.open(path, "wb") as out:
                out.write(response.read())
    data = np.loadtxt(path, delimiter=",")
    T = data[:, 0].astype(int)
    Y = data[:, 1]
    individual = data[:, 4] - data[:, 3]  # mu1 - mu0
    return data[:, 5:], T, Y, float(individual.mean()), individual


def lalonde():
    """Dehejia-Wahba NSW. Treatment was randomised, so the unadjusted difference
    IS the effect - a benchmark to reproduce, not to beat.

    Loaded through a subprocess for the same reason the estimators are: this
    import pulls in pandas, and the parent stays free of it."""
    handle, path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    names = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
    script = (
        "import json, io\n"
        "from dowhy.datasets import lalonde_dataset\n"
        "f = lalonde_dataset()\n"
        "names = %r\n"
        "io.open(%r, 'w').write(json.dumps({'X': f[names].values.tolist(),"
        " 'T': f['treat'].astype(int).tolist(), 'Y': f['re78'].astype(float).tolist()}))\n"
        % (names, path))
    try:
        proc = subprocess.run([sys.executable, "-W", "ignore", "-c", script],
                              capture_output=True, encoding="utf8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[-200:])
        with io.open(path, encoding="utf8") as source:
            payload = json.load(source)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    X = np.array(payload["X"], dtype=float)
    T = np.array(payload["T"], dtype=int)
    Y = np.array(payload["Y"], dtype=float)
    return X, T, Y, float(Y[T == 1].mean() - Y[T == 0].mean()), None


def heterogeneous(seed=11, n=8000, p=5):
    """Effect linear in x0, outcome surface linear. Individual truths known."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    ps = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    T = (rng.random(n) < ps).astype(int)
    tau = 3.0 + 2.0 * X[:, 0]
    Y = 2.0 + 1.5 * X[:, 0] + 0.7 * X[:, 1] + tau * T + rng.normal(size=n)
    return X, T, Y, float(tau.mean()), tau


def nonlinear(seed=13, n=8000, p=5):
    """The effect is a step function of x0 and the outcome surface is curved.

    Included because it is the case a linear base learner cannot fit, and a
    benchmark set that omitted it would flatter DuckDo's classical path."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    ps = 1.0 / (1.0 + np.exp(-(0.7 * X[:, 0] - 0.4 * X[:, 2])))
    T = (rng.random(n) < ps).astype(int)
    tau = 1.0 + 3.0 * (X[:, 0] > 0.0) + 1.5 * np.sin(X[:, 1])
    base = 2.0 + X[:, 0] ** 2 - 1.2 * np.abs(X[:, 1]) + 0.8 * X[:, 2] * X[:, 3]
    Y = base + tau * T + rng.normal(size=n)
    return X, T, Y, float(tau.mean()), tau


# --------------------------------------------------------------------- runner

def evaluate(name, X, T, Y, truth, individual, args, results):
    covariates = ["x%d" % i for i in range(X.shape[1])]
    handle, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    write_csv(csv_path, X, T, Y, covariates)
    handle, data_path = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    np.savez(data_path, X=X, T=T, Y=Y)

    def duckdo_pair(spec):
        """The ATE call is what a user runs, so it is what is timed. The per-row
        call is made only where PEHE is measurable, and is not on the clock."""
        start = time.perf_counter()
        estimate = duckdo_ate(args.duckdb, args.models, csv_path, covariates, spec)
        seconds = time.perf_counter() - start
        cate = None
        if individual is not None:
            cate = duckdo_cate(args.duckdb, args.models, csv_path, covariates, spec)
        return (estimate, cate), seconds

    methods = [
        ("DuckDo aipw", lambda: duckdo_pair("estimator := 'aipw'")),
        ("DuckDo dml", lambda: duckdo_pair("estimator := 'dml'")),
        ("DuckDo causalpfn", lambda: duckdo_pair("model := 'causalpfn'")),
        ("EconML LinearDRLearner", lambda: run_python_method("econml_dr", data_path)),
        ("EconML LinearDML", lambda: run_python_method("econml_dml", data_path)),
        ("EconML CausalForestDML", lambda: run_python_method("econml_forest", data_path)),
        ("DoWhy PSW", lambda: run_python_method("dowhy_psw", data_path)),
        ("CausalPFN (python, cpu)", lambda: run_python_method("cpfn_cpu", data_path)),
    ]
    if args.cuda:
        methods.append(("CausalPFN (python, gpu)",
                        lambda: run_python_method("cpfn_gpu", data_path)))

    try:
        for label, call in methods:
            try:
                (estimate, cate), seconds = call()
            except Exception as exc:  # noqa: BLE001 - a missing dependency is data, not a crash
                results.append({"dataset": name, "method": label, "error": str(exc)[:200]})
                print("  %-24s FAILED  %s" % (label, str(exc)[:90]))
                continue
            row = {"dataset": name, "method": label, "estimate": estimate, "truth": truth,
                   "ate_error": abs(estimate - truth), "seconds": seconds, "n": int(len(T)),
                   "p": int(X.shape[1])}
            if individual is not None and cate is not None:
                row["pehe"] = float(np.sqrt(np.mean(
                    (np.asarray(cate).reshape(-1) - individual) ** 2)))
            results.append(row)
            print("  %-24s est %9.4f  |err| %8.4f  %s%7.2fs"
                  % (label, estimate, row["ate_error"],
                     ("PEHE %6.3f  " % row["pehe"]) if "pehe" in row else " " * 13,
                     seconds))
    finally:
        for path in (csv_path, data_path):
            if os.path.exists(path):
                os.unlink(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    parser.add_argument("--models", default=os.path.join("build", "models"))
    parser.add_argument("--quick", action="store_true", help="one IHDP replication, not ten")
    parser.add_argument("--json", default=os.path.join("scripts", "bench_results.json"))
    parser.add_argument("--worker", help="internal: run one method in this process")
    parser.add_argument("--data")
    parser.add_argument("--result")
    args = parser.parse_args()

    if args.worker:
        return worker(args.worker, args.data, args.result)

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2
    args.cuda = cuda_available()
    print("CUDA %s" % ("available" if args.cuda else "not available"))

    results = []

    print()
    print("heterogeneous (n=8000, effect linear in x0, individual truths known)")
    X, T, Y, truth, individual = heterogeneous()
    evaluate("heterogeneous", X, T, Y, truth, individual, args, results)

    print()
    print("nonlinear (n=8000, step effect and a curved outcome surface)")
    X, T, Y, truth, individual = nonlinear()
    evaluate("nonlinear", X, T, Y, truth, individual, args, results)

    for replication in ([1] if args.quick else range(1, 11)):
        print()
        print("ihdp-%d (n=747, 25 covariates, individual truths known)" % replication)
        X, T, Y, truth, individual = ihdp(replication)
        evaluate("ihdp-%d" % replication, X, T, Y, truth, individual, args, results)

    print()
    print("lalonde NSW (randomised: the benchmark is the unadjusted difference)")
    X, T, Y, truth, individual = lalonde()
    evaluate("lalonde", X, T, Y, truth, individual, args, results)

    with io.open(args.json, "w", encoding="utf8") as handle:
        json.dump(results, handle, indent=2)
    print()
    print("wrote %s (%d rows)" % (args.json, len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
