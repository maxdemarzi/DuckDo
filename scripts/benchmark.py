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
# while it is being built (see Phase 8 of the roadmap). The same case peaks at
# 897 MB on Windows and 1386 MB on the Linux CI runner, where glibc keeps more
# memory per thread and the peak is sampled from /proc rather than read from the
# process. Each gate sits above its own platform's measurement with room for
# noise, and low enough that a stray full copy of the frame (400 MB) trips it.
GATE_PEAK_MB = 1700.0 if sys.platform.startswith("linux") else 1200.0


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


def run_case(duckdb, sql, script=None):
    """Run one case, returning (stdout, stderr, returncode, seconds, peak_mb).

    A long workload goes through a script file: Windows caps a command line at
    32,767 characters, and the leak check's is longer than that.
    """
    command = [duckdb, "-csv", "-noheader", "-c", sql]
    if script is not None:
        with open(script, "w", encoding="utf8") as handle:
            handle.write(sql)
        command = [duckdb, "-csv", "-noheader", "-c", ".read " + script.replace("\\", "/")]
    started = time.time()
    process = subprocess.Popen(command,
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


# The leak check runs one mixed workload at three lengths, each in a fresh
# process, and compares how fast the peak grows early and late. A leak costs the
# same bytes on every call, so its growth per call stays flat. Allocators and
# caches warming up cost less and less, so theirs falls away. A fixed ceiling on
# growth cannot tell those apart - it passes a slow leak and fails a big warm-up -
# so the gate is on the slope: the late per-call growth has to be well under the
# early one, or negligible outright.
#
# The windows start at 300 calls because the warm-up is long. DuckDB's own
# allocators are still settling after a thousand calls: do_ate_by alone grew
# 135 KB a call over its first 240, 19.7 KB a call from 300 to 1,000, and 4.9 KB
# from 1,000 to 3,000, levelling off near 110 MB. An earlier version compared 50
# to 300 against 300 to 1,000 calls, measured nothing but the warm-up, and failed
# on a workload that plateaus. The workload covers every estimator family,
# do_ate_by (which attaches and detaches a scratch database on every call),
# do_cate, and the bootstrap. AddressSanitizer in CI is the precise check; this
# is the one that runs anywhere.
LEAK_CALLS = (300, 1000, 3000)
LEAK_SHAPE = (20_000, 10)
LEAK_FLAT_KB = 5.0
LEAK_SLOWDOWN = 0.5


def leak_sql(path, calls):
    src = path.replace("\\", "/")
    rel = "(SELECT *, id %% 20 AS grp FROM read_parquet(''%s''))" % src
    common = "treatment:='t', outcome:='y'"
    work = [
        "SELECT count(*) FROM do_ate('%s', %s, exclude:=['id','grp'], estimator:='aipw');" % (rel, common),
        "SELECT count(*) FROM do_ate('%s', %s, exclude:=['id','grp'], estimator:='dml');" % (rel, common),
        "SELECT count(*) FROM do_ate('%s', %s, exclude:=['id','grp'], estimator:='t_learner', bootstrap_reps:=50);"
        % (rel, common),
        "SELECT count(*) FROM do_ate_by('%s', %s, exclude:=['id'], by:=['grp']);" % (rel, common),
        "SELECT count(*) FROM do_cate('%s', %s, exclude:=['id','grp']);" % (rel, common),
    ]
    return "\n".join(work[i % len(work)] for i in range(calls)) + "\n"


def run_leak(args, workdir):
    rows, covariates = LEAK_SHAPE
    path = os.path.join(workdir, "leak_%d_%d.parquet" % (rows, covariates))
    print("generating %d x %d ..." % (rows, covariates))
    made = subprocess.run([args.duckdb, "-c", generate_sql(rows, covariates, path)],
                          capture_output=True, encoding="utf8", errors="replace")
    if made.returncode != 0:
        print("FAILED to generate: %s" % made.stderr.strip()[:200])
        return 1
    peaks = {}
    for calls in LEAK_CALLS:
        script = os.path.join(workdir, "leak_%d.sql" % calls)
        _, stderr, code, elapsed, peak = run_case(args.duckdb, leak_sql(path, calls), script=script)
        if code != 0:
            print("FAILED after %d calls: %s" % (calls, stderr.strip()[:300]))
            return 1
        if peak is None:
            print("peak memory is not measured on this platform, and the leak check needs it")
            return 2
        peaks[calls] = peak
        print("%4d calls: %7.1f s, peak %7.1f MB" % (calls, elapsed, peak))
    first, middle, last = LEAK_CALLS
    early = (peaks[middle] - peaks[first]) * 1024.0 / (middle - first)
    late = (peaks[last] - peaks[middle]) * 1024.0 / (last - middle)
    print("growth per call: %+.1f KB from %d to %d calls, %+.1f KB from %d to %d"
          % (early, first, middle, late, middle, last))
    if late <= LEAK_FLAT_KB or late <= LEAK_SLOWDOWN * early:
        print("LEAK GATE PASSED: late growth %+.1f KB/call is %s"
              % (late, "negligible" if late <= LEAK_FLAT_KB else "under half the early %+.1f KB/call" % early))
        return 0
    print("LEAK GATE FAILED: growth per call did not fall away (%+.1f KB early, %+.1f KB late) - "
          "that is the shape of a leak" % (early, late))
    return 1


def main():
    global GATE_SECONDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    parser.add_argument("--json", default=None)
    parser.add_argument("--leak", action="store_true",
                        help="run the repeated-invocation leak check instead of the timing suite")
    parser.add_argument("--time-gate", type=float, default=GATE_SECONDS,
                        help="seconds allowed per case. The 30 s default is the roadmap's laptop gate; a shared CI "
                             "runner with fewer cores needs more, while the memory gate holds everywhere")
    args = parser.parse_args()
    GATE_SECONDS = args.time_gate

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    workdir = tempfile.mkdtemp(prefix="duckdo-bench-")
    shapes = {}
    try:
        if args.leak:
            return run_leak(args, workdir)
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
