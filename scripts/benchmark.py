"""Track DuckDo's estimation cost across table sizes and estimators.

Phase 8 of the roadmap asks for a fixed benchmark set so regressions are visible
rather than discovered. This prints a table and, with --json, writes one that can
be diffed between commits.

Each shape is generated once into a Parquet file, outside the measurement. Every
case then runs in its own process reading that file, so the seconds and the peak
memory describe the estimator rather than the table generator - an in-memory
source table of the same shape would otherwise account for a third of the peak
and none of the work.

Dev-only. Not shipped with the extension.

    python scripts/benchmark.py [--duckdb build/release/duckdb.exe] [--json bench.json]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
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
# The encoded matrix for 1M x 50 is 400 MB and the encoder needs about twice that
# while it is being built (see Phase 8 of the roadmap). This gate sits above the
# measured 897 MB with room for noise, and low enough that reintroducing a stray
# full copy of the frame would trip it.
GATE_PEAK_MB = 1200.0


def peak_rss_mb(process):
    """Peak resident memory of a finished child, or None where we cannot get it.

    Windows keeps the counter readable through the process handle after exit;
    Linux exposes VmHWM only while the process is alive, so it has to be sampled.
    No accurate per-child figure is available on macOS without psutil, and a wrong
    number is worse than an absent one, so that returns None.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = int(process._handle)  # open until the Popen object is collected
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
                wintypes.HANDLE(handle), ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    return None


def sample_peak_linux(pid, stop, out):
    """Poll VmHWM until the process exits. Linux only."""
    path = "/proc/%d/status" % pid
    while not stop.is_set():
        try:
            with open(path, encoding="utf8") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        out[0] = max(out[0], float(line.split()[1]) / 1024.0)
                        break
        except OSError:
            return
        stop.wait(0.05)


def run_case(duckdb, sql):
    """Run one case, returning (stdout, stderr, returncode, seconds, peak_mb)."""
    started = time.time()
    process = subprocess.Popen([duckdb, "-csv", "-noheader", "-c", sql],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8", errors="replace")
    sampled = [0.0]
    stop = threading.Event()
    sampler = None
    if sys.platform.startswith("linux"):
        sampler = threading.Thread(target=sample_peak_linux, args=(process.pid, stop, sampled))
        sampler.daemon = True
        sampler.start()
    stdout, stderr = process.communicate()
    elapsed = time.time() - started
    stop.set()
    if sampler is not None:
        sampler.join(timeout=1.0)
        peak = sampled[0] or None
    else:
        peak = peak_rss_mb(process)
    return stdout, stderr, process.returncode, elapsed, peak


def generate_sql(rows, covariates, path):
    """Write one shape to Parquet. Not measured - this is setup, not the subject."""
    noise = "sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random())"
    cols = ", ".join("%s AS x%d" % (noise, i) for i in range(1, covariates + 1))
    sel = ", ".join("x%d" % i for i in range(1, covariates + 1))
    return """
SELECT setseed(0.1);
COPY (
WITH u AS (SELECT i, {cols}, {noise} AS noise, random() AS draw FROM range({rows}) t(i)),
     p AS (SELECT *, 1.0/(1.0+exp(-(0.9*x1-0.6*x2))) AS ps FROM u)
SELECT i AS id, {sel}, (draw<ps)::INTEGER AS t,
       2.0+1.5*x1+0.7*x2+3.0*((draw<ps)::INTEGER)+noise AS y FROM p
) TO '{path}' (FORMAT parquet);
""".format(cols=cols, sel=sel, noise=noise, rows=rows, path=path.replace("\\", "/"))


def build_sql(path, estimator):
    return """
SET duckdo_max_rows=2000000;
SELECT estimate FROM do_ate('(SELECT * FROM read_parquet(''{path}''))',
                            treatment:='t', outcome:='y', exclude:=['id'],
                            estimator:='{estimator}');
""".format(path=path.replace("\\", "/"), estimator=estimator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    workdir = tempfile.mkdtemp(prefix="duckdo-bench-")
    shapes = {}
    try:
        for rows, covariates, _ in CASES:
            if (rows, covariates) in shapes:
                continue
            path = os.path.join(workdir, "bench_%d_%d.parquet" % (rows, covariates))
            print("generating %d x %d ..." % (rows, covariates))
            made = subprocess.run([args.duckdb, "-c", generate_sql(rows, covariates, path)],
                                  capture_output=True, encoding="utf8", errors="replace")
            if made.returncode != 0:
                print("FAILED to generate: %s" % made.stderr.strip()[:200])
                return 1
            shapes[(rows, covariates)] = path
        return run_cases(args, shapes)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_cases(args, shapes):

    print("%10s %6s %-12s %10s %10s %12s"
          % ("rows", "covs", "estimator", "seconds", "peak MB", "estimate"))
    print("-" * 66)
    results = []
    over_gate = []
    over_memory = []
    for rows, covariates, estimator in CASES:
        sql = build_sql(shapes[(rows, covariates)], estimator)
        stdout, stderr, code, elapsed, peak = run_case(args.duckdb, sql)
        if code != 0:
            print("FAILED %s" % (stderr.strip()[:200]))
            return 1
        estimate = float(stdout.strip().splitlines()[-1])
        print("%10d %6d %-12s %10.2f %10s %12.4f"
              % (rows, covariates, estimator, elapsed,
                 "-" if peak is None else "%.0f" % peak, estimate))
        results.append(dict(rows=rows, covariates=covariates, estimator=estimator,
                            seconds=round(elapsed, 3), estimate=estimate,
                            peak_mb=None if peak is None else round(peak, 1)))
        if rows >= 1_000_000 and covariates >= 50:
            if elapsed > GATE_SECONDS:
                over_gate.append((rows, covariates, estimator, elapsed))
            if peak is not None and peak > GATE_PEAK_MB:
                over_memory.append((rows, covariates, estimator, peak))

    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle, indent=2)
        print("\nwrote %s" % args.json)

    print()
    failed = False
    if over_gate:
        print("GATE FAILED: %d case(s) over %.0fs" % (len(over_gate), GATE_SECONDS))
        for rows, covariates, estimator, elapsed in over_gate:
            print("  %d x %d %s took %.1fs" % (rows, covariates, estimator, elapsed))
        failed = True
    if over_memory:
        print("GATE FAILED: %d case(s) over %.0f MB peak" % (len(over_memory), GATE_PEAK_MB))
        for rows, covariates, estimator, peak in over_memory:
            print("  %d x %d %s peaked at %.0f MB" % (rows, covariates, estimator, peak))
        failed = True
    if failed:
        return 1
    measured = [r["peak_mb"] for r in results if r["peak_mb"] is not None]
    print("GATE PASSED: 1M rows x 50 covariates stays under %.0fs" % GATE_SECONDS)
    if measured:
        print("GATE PASSED: peak resident memory stays under %.0f MB (highest was %.0f MB)"
              % (GATE_PEAK_MB, max(measured)))
    else:
        print("peak memory not measured on this platform")
    return 0


if __name__ == "__main__":
    sys.exit(main())
