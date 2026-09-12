# Reproducibility, and no telemetry

What DuckDo reproduces exactly, what it does not, and how to pin the parts that
move. Every claim on this page is a case in `scripts/reproducibility_check.py`,
which fails if the behaviour stops matching the text.

**The short version.** Same table, same settings → **identical to the last bit**,
on any number of threads. Reorder the rows and the answer moves by about 2e-15
relative, which is rounding. And DuckDo sends nothing anywhere, ever.

---

## No telemetry

**DuckDo makes no network request you did not ask for by name.**

Not a usage report, not a crash report, not a version check, not an analytics ping.
The extension contains no HTTP client and no socket code. The source contains three
URLs. Two are attribution strings naming where each model came from; `do_list_models()`
prints them and nothing fetches them. The third is the pinned address of the hosted
CausalPFN release, fetched only when you call `do_download('causalpfn')` with no
source. Model weights are always read from a local directory, whether you downloaded
them that way or exported them yourself with `scripts/export/`.

One function can reach the network: `do_download`, and only when you call it. Pass
a `source` URL you typed, or give no source for `causalpfn` to fetch DuckDo's
pinned release. Even then DuckDo does not open the connection. It
reads through DuckDB's own file system, which passes a URL to DuckDB's `httpfs`
extension, so the request is DuckDB's, made by an extension you can see in
`duckdb_extensions()`. If extension autoloading is enabled, as it is in DuckDB's
default builds, DuckDB may *install* `httpfs` the first time a URL is opened. That
is DuckDB's behaviour and a network fetch of its own, and it is worth knowing about.
The one default source, CausalPFN's, is pinned to a single commit, and every file is
checked against a SHA-256 compiled into the extension before it is installed. So
the bytes cannot change under you, and a file that fails the check is never
installed. Pointed at a directory, `do_download` touches no network at all.

You do not have to take that on trust — it is one grep:

```sh
grep -rniE "http://|https://|curl|socket|inet_|getaddrinfo|telemetry" src/
```

This is not a privacy posture, it is a requirement. Causal analysis runs on the
data that is worth analysing: patient records, payroll, transactions, churn. A
column name from that table is itself sensitive, and column names are exactly what
a "harmless" usage ping tends to carry.

The estimators do not need the network. `INSTALL duckdo; LOAD duckdo;` and every
classical estimator works fully offline, with no model download and no ONNX
Runtime — that is the default build.

---

## What is exactly reproducible

Verified on 20,000 rows, comparing full-precision output rather than rounded:

| | |
|---|---|
| the same query, run twice | **identical** |
| `duckdo_threads` = 1, 2, 8, auto | **identical** |
| DuckDB `threads` = 1, 4 | **identical** |
| `do_cate` over 20,000 rows, run twice | **identical** |
| `do_balance` and `do_overlap`, which take no outcome, run twice | **identical**, since a fix: see below |
| a bootstrap interval, same seed | **identical** |
| `model := 'causalpfn'`, run twice | **identical** |
| `duckdo_query_chunk` = 256 vs 4096 | **identical** |
| `ensemble := 4`, same seed | **identical** |
| the same rows in a different physical order | to 2e-15 relative |

**Thread-independence is not free and did not come for free.** Floating-point
addition is not associative, so `(a+b)+c` and `a+(b+c)` differ in the last bits. If
the dense accumulations were split into one block per core, the *number of blocks*
would follow the core count, and so would the answer — the same query would return
subtly different results on a laptop and on a build server, which is the kind of
difference that costs a day to track down. DuckDo therefore blocks rows in fixed
sizes that depend only on the row count:

```cpp
const idx_t kMinRowsPerBlock = 4096;
const idx_t kMaxBlocks = 64;
```

Bootstrap replicates are parallel across replicates, each seeded from `(seed, rep)`
rather than drawing from one shared stream, so replicate 7 is replicate 7 no matter
which thread reaches it or when.

`duckdo_query_chunk` behaves the same way for the foundation models: it changes how
many rows are scored per forward pass, and **not** the answer. The same holds
across the move from one graph to two. The split encode/decode export returns
2.9705805124938487 on the benchmark table where the single graph did, digit for
digit, and the ensemble draws are unchanged too — see
[BENCHMARKS.md](BENCHMARKS.md#encoding-the-context-once).

---

## What moves, and by how much

### Row order, which used to matter and no longer does

Permute the rows of a table — same rows, nothing else changed — and the estimate
moves by:

```
estimator             sd    std_error      sd/se
regression     1.620e-13       0.1092      0.00%
aipw           3.163e-14       0.1031      0.00%
ipw            2.043e-13       0.3459      0.00%
```

Ten deterministic permutations of one 20,000-row table; `sd` is the standard
deviation of the estimate across those permutations. That is rounding, and the
check asserts it stays below 1e-10 relative rather than merely observing it.

**It used to be 0.25% / 2.12% / 6.18% of a standard error**, and the reason is
worth keeping. Fold assignment seeded a shuffle of row *positions*:

```cpp
vector<idx_t> rows = frame.ArmRows(arm);
std::shuffle(rows.begin(), rows.end(), rng);   // seeded, but over row *positions*
```

Move a row to a different position and it landed in a different fold, so a
cross-fitted estimator held out a different subset and returned a different —
equally valid — estimate. Sampling variation rather than error, but a table
rebuilt by an unrelated ETL change would move a published number.

The frame now carries a **canonical order**: a permutation of the rows derived
from what they contain. Every seeded draw indexes that instead of storage — fold
assignment, bootstrap resampling, the foundation model's context sample, and all
five refutation methods.

Hashing each row was the obvious way to build that order, and it is wrong in an
instructive way. `X` is standardised, so every value carries a mean that was
itself summed in storage order; permuting the table moves that mean by an ulp,
and a hash turns an ulp into a completely different sort key. The hash-ordered
sort came out *more* order-sensitive than the thing it replaced. Comparison is
stable where hashing is not, because standardising is monotone within a column
and an ulp of drift cannot reorder two distinct values.

It costs 0.8 s on the 1M × 50 frame, and the estimates are unchanged in accuracy:
across ten IHDP replications the paired change in absolute error is 0.027 ± 0.039,
an interval comfortably containing zero.

### No outcome, which used to matter and no longer does

`do_balance` and `do_overlap` take no outcome, and until a fix, neither repeated itself. Called
twice on the same table, `do_balance` gave a weighted standardised difference of -0.00581 once and
-0.00353 the next time. `do_overlap` moved rows between propensity buckets.

The frame's outcome column, with no outcome to read, is `CAST(NULL AS DOUBLE)`. It was copied from
the vector's data slots without checking them, so its "values" were whatever memory held. The
canonical row order sorts on the outcome first, so the order, the propensity folds and the weights
changed from call to call. `do_diagnose` without an outcome had the same flaw, hidden by its
three-decimal rounding. A missing outcome now reads as 0.0, and every function that has an outcome
is bit-identical to before. None of the checks called a function without an outcome, which is why
none caught it. `scripts/reproducibility_check.py` now does, and the test suite repeats all three
functions and requires every run to match the first.

### Deliberately different

| | |
|---|---|
| a bootstrap interval, `seed := 42` vs `43` | differs — 8.2736 vs 8.3157 |
| `ensemble := 4`, `seed := 42` vs `43` | differs — 8.7702 vs 8.7813 |

Both are resampling, so a different seed is a different sample. The default seed is
42 and every function takes `seed :=`.

---

## Pinning a result you will need to defend

1. **The extension version.** `SELECT * FROM duckdb_extensions() WHERE extension_name = 'duckdo';`
2. ~~**The row order.**~~ No longer necessary — see above. Left in the list because
   it was necessary until recently, and a pinning recipe that quietly drops a step
   is worse than one that says why the step is gone.
3. **The seed.** Pass `seed :=` explicitly rather than relying on the default,
   which is a value that could change between versions.
4. **The covariate list.** `exclude :=` is relative to whatever columns the table
   happens to have, so a column added upstream silently joins the adjustment set.
   `covariates := [...]` is explicit and does not drift. To record what was
   actually adjusted for, `do_frame_summary` returns one row per encoded feature —
   name, source column, kind, and the centre and scale it was standardised by.
5. **The model, if you used one.** `do_list_models()` reports the licence and the
   covariate budget; the manifest beside the weights carries the export's opset and
   its parity check against PyTorch.

Point 4 is the one that actually bites. A rerun that differs is nearly always a
different adjustment set, not a different estimator.

---

## Across machines

Not verified, and not claimed. DuckDo is built with different compilers on each
platform, and a compiler is free to contract `a*b+c` into a fused multiply-add,
which changes the last bits. Everything on this page was measured on one Windows
build; the guarantees are within a platform, not across platforms.

That said, the gap between "the last bits differ" and "the answer differs" is worth
keeping narrow, and one thing on the wrong side of it has been fixed. The random
Fourier features behind `test := 'kernel'` and `algorithm := 'resit'` were drawn with
`std::normal_distribution` and `std::uniform_real_distribution`, which are
implementation-defined: the same engine and the same seed give a different sequence
under libstdc++ than under MSVC. Those features decide the graph rather than merely
perturbing it, and CI caught the consequence as one platform finding an edge the other
did not. They are now drawn from the engine's bits and put through the inverse normal
CDF, both fixed by arithmetic, so the same rows and the same `duckdo_seed` give the
same features everywhere.

Resampling is still drawn with `std::uniform_int_distribution`, which is
implementation-defined in the same way, so **`stability` and `orientation_stability`
can differ across platforms** while the graph itself does not. The discovery tests
assert graphs exactly and stabilities only as bounds, for that reason.

The foundation-model path has one further caveat, already measured during the
export: ONNX Runtime decomposes attention where PyTorch fuses it, and over twelve
layers that is worth about 1e-3 of relative agreement at the logit level. It comes
to **0.00087 on the ATE** and a CATE correlation of 0.9994, which is why the export
script gates on the estimand rather than on the tensors.

**The GPU is a different machine for this purpose.** With `duckdo_device = 'cuda'`, a
foundation-model result repeats to the last bit on the same GPU. That holds because
DuckDo runs CUDA with deterministic kernels. Without them, ten repeats in one process
gave four different estimates. It does not match the CPU to the last bit: at fp32
the CausalPFN estimate differs from the CPU's by about 2e-7, and at
`duckdo_gpu_precision = 'tf32'` by about 2e-5. That is why the
CPU is the default. A result should not move because a GPU happens to be present,
and every result that ran on CUDA says so, with its precision, in `warnings`.

---

## Running the checks

```sh
python scripts/reproducibility_check.py
```

Exits non-zero if any case stops behaving as this page describes — including the
cases that are *supposed* to differ, so a change that accidentally made bootstrap
seeds inert would fail here too. `--skip-models` drops the foundation-model
section, which needs exported weights.
