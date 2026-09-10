"""Should DuckDo export CausalFM? The evidence, as a script.

CausalFM (Ma, Frauen, Javurek, Feuerriegel; ICLR 2026; Apache-2.0) ships three
pretrained checkpoints in https://github.com/yccm/CausalFM-toolkit - standard
CATE, front-door, and binary IV. The roadmap wanted it for the second and third:
model-based front-door and IV estimation where DuckDo only has the classical
do_frontdoor and do_iv. This script is why DuckDo does not ship it.

It checks four things, in the order a skeptic would:

  1. Does it load and reproduce the authors' own published numbers, on their own
     test sets, with their own evaluation protocol? (If not, the harness is wrong
     and nothing below means anything.)
  2. Is each checkpoint's output actually a function of its input? Compared
     against the PEHE of predicting a constant - the baseline any evaluation of a
     CATE model should report and this one did not.
  3. How does DuckDo's classical estimator do on the same rows?
  4. What happens a little outside the training distribution: the same settings,
     a different data-generating process, a known answer.

Dev-only. Not shipped with the extension. Needs torch and a clone of the toolkit:

    git clone https://github.com/yccm/CausalFM-toolkit   # checked at bb74ef7
    python scripts/causalfm_check.py --toolkit path/to/CausalFM-toolkit
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import types

import numpy as np

REPORTED = {"front-door": (0.8466, 0.3429), "IV binary": (0.4223, 0.1625)}


def import_models(toolkit):
    """Load the three network classes without running the toolkit's package init.

    The bundled TabPFN fork's __init__ imports its sklearn compatibility layer,
    which needs a private scikit-learn function (_is_pandas_df) that 1.9 removed.
    The networks themselves never touch it - their import closure is encoders,
    layer, memory, mlp and multi_head_attention - so tabpfn and tabpfn.model are
    registered as bare packages and only the model modules are loaded."""
    root = os.path.join(os.path.abspath(toolkit), "src", "tabpfn")
    for name, path in (("tabpfn", root), ("tabpfn.model", os.path.join(root, "model"))):
        package = types.ModuleType(name)
        package.__path__ = [path]
        sys.modules[name] = package
    from tabpfn.model.causalFM import PerFeatureTransformerCATE as Standard
    from tabpfn.model.causalFM4FD import PerFeatureTransformerCATE as FrontDoor
    from tabpfn.model.causalFM4IV import PerFeatureTransformerCATE as IV
    return Standard, FrontDoor, IV


def load(torch, cls, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = cls()
    net.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    net.eval()
    meta = {k: checkpoint[k] for k in ("epoch", "val_loss") if isinstance(checkpoint, dict) and k in checkpoint}
    return net, meta


def tensor(torch, values):
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def read_test_csv(path):
    rows = list(csv.reader(open(path)))
    head, body = rows[0], np.array(rows[1:], dtype=np.float32)
    column = {h: body[:, j] for j, h in enumerate(head)}
    # The toolkit's loader feeds only the x* columns; the u* columns in these
    # files are the unobserved confounders, kept for reference.
    return column, np.column_stack([column[h] for h in head if h.startswith("x")])


def duckdo(duckdb, fn, columns, extra, select="estimate"):
    """One DuckDo call on an in-memory table written to a temporary CSV."""
    handle, path = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    names = list(columns)
    with open(path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(names)
        for i in range(len(columns[names[0]])):
            writer.writerow([float(columns[k][i]) for k in names])
    covariates = ", ".join("'%s'" % k for k in names if k.startswith("x"))
    sql = ("SELECT %s FROM %s('(SELECT * FROM read_csv_auto(''%s''))', treatment := 'a', outcome := 'y', "
           "%s, covariates := [%s]);" % (select, fn, path.replace("\\", "/"), extra, covariates))
    proc = subprocess.run([duckdb, "-csv", "-noheader", "-c", sql], capture_output=True, text=True)
    os.unlink(path)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.strip().splitlines()[-1].split(",")


def rmse(errors):
    return float(np.sqrt(np.mean(np.asarray(errors) ** 2)))


def home_ground(torch, duckdb, nets, toolkit):
    """Sections 1-3: the authors' test sets, the authors' 80/20 protocol."""
    settings = [
        ("front-door", nets["fd"], "DATA_FD/frontdoor_TEST", "frontdoor_test_dataset_", "mediator",
         "do_frontdoor", "mediator := 'w'"),
        ("IV binary", nets["iv"], "DATA_IV/iv_binary_TEST", "iv_binary_test_dataset_", "z",
         "do_iv", "instrument := 'w'"),
    ]
    verdicts = {}
    for label, (net, meta), folder, prefix, aux, fn, extra in settings:
        print("== %s, on the authors' own ten test sets   (checkpoint %s)" % (label, meta))
        print("  %2s %9s %9s %14s %10s %13s   %s"
              % ("ds", "cate sd", "corr", "PEHE model", "PEHE zero", "truth mean", "%s (F, weak)" % fn))
        model_pehe, zero_pehe, spreads = [], [], []
        for i in range(1, 11):
            col, X = read_test_csv(os.path.join(toolkit, folder, "%s%d.csv" % (prefix, i)))
            A, Y, W, ite = col["treatment"], col["outcome"], col[aux], col["ite"]
            cut = int(len(Y) * 0.8)
            with torch.no_grad():
                call = net.estimate_cate_fd if aux == "mediator" else net.estimate_cate_iv
                cate = call(tensor(torch, X[:cut]), tensor(torch, A[:cut])[:, None],
                            tensor(torch, Y[:cut])[:, None], tensor(torch, W[:cut])[:, None],
                            tensor(torch, X[cut:]))["cate"].numpy().reshape(-1)
            truth = ite[cut:]
            model_pehe.append(rmse(cate - truth))
            zero_pehe.append(rmse(truth))
            spreads.append(float(cate.std()))
            corr = np.corrcoef(cate, truth)[0, 1] if cate.std() > 1e-6 else float("nan")
            columns = {"a": A, "y": Y, "w": W, **{"x%d" % j: X[:, j] for j in range(X.shape[1])}}
            if fn == "do_iv":
                got = duckdo(duckdb, fn, columns, extra,
                             "round(estimate,3), round(first_stage_f,1), weak_instrument")
                shown = "%s (F %s, %s)" % tuple(got) if got else "failed"
            else:
                got = duckdo(duckdb, fn, columns, extra, "round(estimate,3)")
                shown = got[0] if got else "failed"
            print("  %2d %9.2e %9.4f %14.4f %10.4f %13.4f   %s"
                  % (i, cate.std(), corr, model_pehe[-1], zero_pehe[-1], ite.mean(), shown))
        reported = REPORTED[label]
        print("  avg PEHE %.4f +/- %.4f (authors report %.4f +/- %.4f) | predicting zero: %.4f\n"
              % (np.mean(model_pehe), np.std(model_pehe), reported[0], reported[1], np.mean(zero_pehe)))
        verdicts[label] = {"reproduced": abs(np.mean(model_pehe) - reported[0]) < 1e-3,
                           "constant": max(spreads) < 1e-6,
                           "beats_zero_by": 1.0 - np.mean(model_pehe) / np.mean(zero_pehe)}
    return verdicts


def standardise(v):
    return (v - v.mean(0)) / (v.std(0) + 1e-8)


def away_from_home(torch, duckdb, nets):
    """Section 4: the same three settings, a different DGP, a known answer.

    Outcomes and covariates are standardised as the authors' test data is; the
    mediator and instrument are left raw, as theirs are. The effect is constant
    so the ATE, the CATE and the LATE coincide and every estimator is aiming at
    the same number."""
    print("== away from home: a different DGP, a known constant effect")
    for p in (5, 10):
        rng = np.random.default_rng(7 + p)
        n = 1000
        X = rng.normal(size=(n, p))
        U = rng.normal(size=n)

        A = (0.8 * U + 0.5 * X[:, 0] + rng.normal(size=n) > 0).astype(float)
        M = 1.2 * A + 0.3 * X[:, 1] + 0.5 * rng.normal(size=n)
        Y = 1.5 * M + 1.0 * U + 0.7 * X[:, 0] + rng.normal(size=n)
        with torch.no_grad():
            cate = nets["fd"][0].estimate_cate_fd(
                tensor(torch, standardise(X)), tensor(torch, A)[:, None], tensor(torch, standardise(Y))[:, None],
                tensor(torch, M)[:, None], tensor(torch, standardise(X)))["cate"].numpy()
        got = duckdo(duckdb, "do_frontdoor", {"a": A, "w": M, "y": Y, **{"x%d" % j: X[:, j] for j in range(p)}},
                     "mediator := 'w'", "round(estimate,4)")
        print("  front-door p=%-2d truth 1.8000  naive %.4f  CausalFM %.4f  do_frontdoor %s"
              % (p, Y[A == 1].mean() - Y[A == 0].mean(), float(cate.mean() * Y.std()), got[0] if got else "failed"))

        Z = (rng.random(n) < 0.5).astype(float)
        A = (1.2 * Z + 0.8 * U + 0.3 * X[:, 0] + rng.normal(size=n) > 0.6).astype(float)
        Y = 2.0 * A + 1.0 * U + 0.5 * X[:, 0] + rng.normal(size=n)
        with torch.no_grad():
            cate = nets["iv"][0].estimate_cate_iv(
                tensor(torch, standardise(X)), tensor(torch, A)[:, None], tensor(torch, standardise(Y))[:, None],
                tensor(torch, Z)[:, None], tensor(torch, standardise(X)))["cate"].numpy()
        got = duckdo(duckdb, "do_iv", {"a": A, "w": Z, "y": Y, **{"x%d" % j: X[:, j] for j in range(p)}},
                     "instrument := 'w'", "round(estimate,4)")
        print("  IV         p=%-2d truth 2.0000  naive %.4f  CausalFM %.4f  do_iv        %s"
              % (p, Y[A == 1].mean() - Y[A == 0].mean(), float(cate.mean() * Y.std()), got[0] if got else "failed"))

    # The standard checkpoint, on the heterogeneous DGP the head-to-head uses,
    # where CausalPFN and AIPW both reach a PEHE near 0.1.
    for p in (5, 10, 25):
        rng = np.random.default_rng(11 + 1000 + p)
        X = rng.normal(size=(1000, p)).astype(np.float32)
        ps = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
        T = (rng.random(1000) < ps).astype(np.float32)
        tau = 3.0 + 2.0 * X[:, 0]
        Y = (2.0 + 1.5 * X[:, 0] + 0.7 * X[:, 1] + tau * T + rng.normal(size=1000)).astype(np.float32)
        with torch.no_grad():
            cate = nets["std"][0].estimate_cate(
                tensor(torch, standardise(X)), tensor(torch, T)[:, None], tensor(torch, standardise(Y))[:, None],
                tensor(torch, standardise(X)))["cate"].numpy().reshape(-1) * Y.std()
        print("  standard   p=%-2d truth %.4f  CausalFM %.4f  PEHE %.3f  corr %.3f"
              % (p, tau.mean(), cate.mean(), rmse(cate - tau), np.corrcoef(cate, tau)[0, 1]))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit", required=True, help="a clone of yccm/CausalFM-toolkit")
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    args = parser.parse_args()

    import torch

    toolkit = os.path.abspath(args.toolkit)
    duckdb = os.path.abspath(args.duckdb)
    Standard, FrontDoor, IV = import_models(toolkit)
    ckpt = lambda sub: os.path.join(toolkit, "checkpoints", sub, "best_model.pth")
    nets = {"std": load(torch, Standard, ckpt("checkpoints_standard")),
            "fd": load(torch, FrontDoor, ckpt("checkpoints_FD")),
            "iv": load(torch, IV, ckpt("checkpoints_IV_binary"))}

    verdicts = home_ground(torch, duckdb, nets, toolkit)
    away_from_home(torch, duckdb, nets)

    print("== verdict")
    for label, v in verdicts.items():
        print("  %-10s reproduced the published PEHE: %s | output constant: %s | better than predicting zero by %.1f%%"
              % (label, v["reproduced"], v["constant"], 100.0 * v["beats_zero_by"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
