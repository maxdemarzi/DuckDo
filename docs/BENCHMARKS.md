# Benchmarks

DuckDo against the Python causal stack — EconML, DoWhy, and the CausalPFN package
itself — on identical rows, with the losses shown.

Every number here is output from `python scripts/bench_headtohead.py`, which runs
each method in its own process against the same data and writes
`scripts/bench_results.json`. Nothing is quoted from a paper.

**The short version.** On real benchmark data DuckDo's foundation-model path is the
most accurate thing measured here, on both metrics, by a wide margin. DuckDo's
classical estimators are competitive on the average effect and **lose badly on
per-row effects** whenever the truth is not linear. And DuckDo's CFM path is
**8× slower than the same model in Python** on eight thousand rows, for a reason
that is understood and fixed by one setting.

---

## What is being measured

Two numbers, because they disagree and the disagreement is the point.

| | what it is | what it decides |
|---|---|---|
| **\|ATE error\|** | distance from the known average effect | the number a report quotes |
| **PEHE** | RMSE of the *per-row* effect | whether a targeting rule works |

An estimator can win the first and lose the second badly — see IHDP below, where
`aipw` lands within 0.16 of the average effect while getting individual effects
wrong by 2.3.

**Timing.** Python methods are timed inside the worker process, around the fit
and the estimate only: interpreter start, imports and the 75 MB weight load are
all excluded. DuckDo is timed as the whole subprocess, including start-up and
parsing a CSV off disk. That framing favours Python, which is why it is the one
used.

Hardware: one Windows laptop, 16 logical cores, RTX 3060. DuckDo's ONNX session
is **CPU-only**; the GPU row exists to show what DuckDo is not doing.

---

## IHDP, ten replications

The standard semi-synthetic benchmark: 747 rows, 25 covariates, real covariates
with a simulated outcome, so individual effects are known and PEHE is measurable.
The CEVAE mirror carries replications 1–10, not the canonical 1000 — so this is
ten replications, and says so.

| method | mean \|error\| | median \|error\| | **mean PEHE** | mean s |
|---|---|---|---|---|
| **DuckDo `causalpfn`** | **0.078** | **0.070** | **0.416** | 2.16 |
| CausalPFN (python) | 0.082 | 0.075 | 0.423 | 1.76 |
| DoWhy PSW | 0.125 | 0.096 | — | 17.25 |
| EconML `LinearDML` | 0.147 | 0.116 | 2.161 | 3.23 |
| DuckDo `aipw` | 0.164 | 0.152 | 2.252 | **0.19** |
| EconML `LinearDRLearner` | 0.259 | 0.166 | 3.131 | 3.20 |
| EconML `CausalForestDML` | 0.390 | 0.066 | 2.749 | 5.34 |
| DuckDo `dml` | 0.857 | 0.196 | 2.252 | 0.18 |

Four things worth reading carefully.

**The foundation model wins, and it is not close.** Mean PEHE **0.416** against
2.16 for the best classical method here — a factor of five. This is the case CFMs
were built for: a real covariate distribution whose structure you do not know in
advance. It is also a direct reversal of what the same comparison shows on a
linear DGP, where AIPW is correctly specified and the model has nothing to add.

**DuckDo's ONNX export matches the reference implementation.** 0.078 against 0.082,
PEHE 0.416 against 0.423, on ten independent replications. This is the strongest
end-to-end evidence that the export is faithful — much stronger than the
logit-level parity check in the export script, which only proves the tensors agree
on one synthetic batch.

The residual gap is small but it is not zero, and it is not context selection:
747 rows is below the 4,096 context limit, so both sides feed the model every row.
Two things do differ. DuckDo passes its own standardised encoding of the
covariates where the package passes them raw, and ONNX Runtime decomposes the
attention that PyTorch fuses — worth about 1e-3 of relative agreement at the logit
level, 0.00087 on the ATE, per the export's own parity check.

**`dml`'s mean is a single replication.** 0.857 mean against a 0.196 median: on
replication 9, whose true effect is 10.47 where every other replication sits near
4, `dml` returns 4.09 and misses by **6.4**. Partialling out a treatment this
strongly predicted, with a linear first stage, on 747 rows, is where that estimator
breaks. `aipw` on the same replication returns 9.95. **Use the median column**;
the mean is one bad case wearing a suit.

**Ten replications cannot separate the middle of this table.** `aipw` at 0.164 and
`LinearDML` at 0.147 are not distinguishable, and neither figure is stable: changing
only *which rows land in which cross-fitting fold* — a change that provably does not
affect accuracy — moved `aipw`'s mean from 0.137 to 0.164, a paired difference of
0.027 ± 0.039. Read the order of magnitude, not the ranking.

**`CausalForestDML` has the best median of any classical method** (0.066) and one
of the worst means (0.390), for the same reason and on the same replication.

---

## Where the classical path loses

Two synthetic datasets of 8,000 rows, identical except in shape. In the first the
effect is linear in a covariate; in the second it is a step function and the
outcome surface is curved.

| | \|error\| linear | PEHE linear | \|error\| **nonlinear** | **PEHE nonlinear** |
|---|---|---|---|---|
| DuckDo `aipw` | **0.008** | 0.073 | 0.024 | 0.990 |
| DuckDo `dml` | 0.006 | 0.073 | 0.057 | 0.990 |
| DuckDo `causalpfn` | 0.023 | **0.061** | 0.076 | 0.420 |
| EconML `LinearDRLearner` | 0.022 | 0.067 | 0.039 | 0.987 |
| EconML `LinearDML` | 0.006 | 0.075 | 0.047 | 1.000 |
| EconML `CausalForestDML` | 0.023 | 0.263 | **0.011** | **0.213** |
| DoWhy PSW | 0.027 | — | 0.038 | — |
| CausalPFN (python) | 0.007 | 0.107 | 0.068 | 0.282 |

**The average effect barely moves.** Every method lands within 0.08 of the truth on
both datasets. If all you need is one number, the choice of estimator is close to
irrelevant — which is the same conclusion `ASSUMPTIONS.md` reaches from the other
direction, that getting the covariate list right matters more than the estimator.

**Per-row effects move by a factor of five.** On the nonlinear dataset DuckDo's
`aipw` posts a PEHE of **0.990** against `CausalForestDML`'s **0.213**. DuckDo's
base learners are regularised GLMs; a step function is exactly what they cannot
fit, and a random forest fits it easily. **If you are building a targeting rule on
data with sharp thresholds, EconML's forest learners will beat DuckDo's classical
path, and DuckDo says so rather than omitting the dataset.**

That is also the reversal of the linear column, where the forest is the *worst*
method on PEHE (0.263 against 0.073) because it fits noise a straight line ignores.
Neither result is a ranking of libraries. Both are a ranking of base learners
against the shape of one dataset.

---

## Lalonde NSW

Treatment was randomised, so the unadjusted difference **is** the causal effect.
Nothing here should beat it; the question is who reproduces it.

| method | estimate | error (1978 dollars) |
|---|---|---|
| experimental benchmark | **1794.3** | — |
| EconML `CausalForestDML` | 1798.2 | 3.9 |
| DuckDo `dml` | 1709.8 | 84.5 |
| EconML `LinearDRLearner` | 1688.6 | 105.8 |
| DuckDo `aipw` | 1680.6 | 113.7 |
| DuckDo `causalpfn` | 1669.9 | 124.5 |
| EconML `LinearDML` | 1668.0 | 126.4 |
| CausalPFN (python) | 1653.9 | 140.4 |
| DoWhy PSW | 1559.8 | 234.5 |

Every method lands within 13% on a noisy 445-row sample (185 treated), and the ordering
across the middle of that table is not a meaningful ranking. `CausalForestDML`'s
3.9 dollars is a good result and partly luck — its 80-dollar advantage over
`dml` is a third of what separates the best and worst methods on a sample this
small.

Note that DuckDo's CFM path (124.5) and the reference package (140.4) bracket each
other here too.

---

## Cost

Seconds for one ATE, same machine.

| | n=747 | n=8,000 |
|---|---|---|
| DuckDo `aipw` / `dml` | **0.19** | **0.22** |
| DuckDo `causalpfn` | 2.16 | 163.8 |
| DuckDo `causalpfn`, `duckdo_query_chunk=8192` | — | **23.3** |
| CausalPFN (python, cpu) | 1.76 | 20.3 |
| CausalPFN (python, gpu) | 1.70 | 11.3 |
| EconML `LinearDML` | 3.23 | 3.17 |
| EconML `CausalForestDML` | 5.34 | 15.5 |
| DoWhy PSW | 17.25 | 2.84 |

**DuckDo's classical path is an order of magnitude faster than anything else here**,
and its 0.19 s includes launching a process and parsing a CSV. EconML's floor of
about 3 s is scikit-learn's cross-fitting machinery, and it does not shrink for
small data — which is why DuckDo beats it by 17× on 747 rows and 14× on 8,000.

### The CFM path is 8× slower by default

164 seconds against the reference package's 20, on the same weights and the same
CPU. That is not the model; it is `duckdo_query_chunk`, which defaults to 512.
DuckDo scores query rows in chunks, and **re-encodes the entire 4,096-row context
on every chunk** — so 8,000 rows at 512 a chunk pays for the context sixteen times.

| `duckdo_query_chunk` | seconds | peak MB | estimate |
|---|---|---|---|
| 512 (default) | 125.8 | 2,062 | 2.9705805124938487 |
| 2048 | 42.8 | 3,335 | 2.9705805124938487 |
| 4096 | 28.4 | 3,594 | 2.9705805124938487 |
| 8192 | 23.3 | 5,348 | 2.9705805124938487 |

**The estimate is bit-identical at every chunk size** — all eighteen digits — so
this is purely a time-versus-memory dial with no accuracy cost. At 8192 DuckDo
(23.3 s) is within striking distance of the Python package on CPU (20.3 s), which
is the real comparison: the two are about the same speed, and the default was
giving away a factor of five.

The default stays at 512 because 5.3 GB of peak memory is not a reasonable thing to
impose on an unknown machine, and this is exactly the number a laptop can fail on.
**If you are running a foundation model over more than a few thousand rows and have
the memory, raise it.** The proper fix — encoding the context once and reusing it
across chunks — needs a re-export of the graph with the context and query stages
split, and is the "CFM context caching" item in Phase 8 of the roadmap.

DuckDo will not match the GPU row. Its ONNX session is CPU-only.

---

## Reproducing this

```sh
pip install econml dowhy causalpfn
python scripts/export/export_causalpfn.py --out build/models
python scripts/bench_headtohead.py
```

IHDP is fetched from the CEVAE mirror on first run and cached. `--quick` runs one
replication instead of ten. The script is a dev tool and is not shipped with the
extension.

Two related harnesses: `scripts/crosscheck_econml.py` gates DuckDo against EconML
on agreement rather than accuracy, and `scripts/benchmark.py` tracks DuckDo's own
cost and memory across table sizes.

---

## What this does not measure

- **Only ten IHDP replications**, not the canonical 1000, because that is what the
  public mirror carries. Means over ten replications are noisy — see `dml`.
- **No ACIC**, which is the benchmark that would test many more DGP shapes.
- **One machine**, one run per cell. The timings are not averaged over repeats and
  should be read as order-of-magnitude.
- **No standard errors on the errors.** Whether 0.078 beats 0.082 is not something
  ten replications can settle; whether 0.416 beats 2.226 is not in doubt.
- **Accuracy against a known truth is not accuracy on your data.** Every dataset
  here satisfies the assumptions by construction. The thing that actually breaks
  causal estimates is an unmeasured confounder, and no benchmark of this kind can
  measure that. See [ASSUMPTIONS.md](ASSUMPTIONS.md).
