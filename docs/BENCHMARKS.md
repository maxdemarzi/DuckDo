# Benchmarks

DuckDo against the Python causal stack — EconML, DoWhy, and the CausalPFN package
itself — on identical rows, with the losses shown.

Every number here is output from `python scripts/bench_headtohead.py`, which runs
each method in its own process against the same data and writes
`scripts/bench_results.json`. Nothing is quoted from a paper.

**The short version.** On real benchmark data DuckDo's foundation-model path is the
most accurate thing measured here, on both metrics, by a wide margin. DuckDo's
classical estimators are competitive on the average effect and **lose badly on
per-row effects** whenever the truth is not linear. And DuckDo's CFM path now runs
within about 1.4× of the same model in Python on CPU, after a fix that removed a
fourfold slowdown in how it scored the context.

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
| **DuckDo `causalpfn`** | **0.078** | **0.070** | **0.416** | 3.13 |
| CausalPFN (python) | 0.082 | 0.075 | 0.423 | 2.31 |
| DoWhy PSW | 0.125 | 0.096 | — | 18.80 |
| EconML `LinearDML` | 0.147 | 0.116 | 2.161 | 3.88 |
| DuckDo `aipw` | 0.164 | 0.152 | 2.252 | **0.21** |
| EconML `LinearDRLearner` | 0.259 | 0.166 | 3.131 | 3.90 |
| EconML `CausalForestDML` | 0.390 | 0.066 | 2.749 | 6.14 |
| DuckDo `dml` | 0.857 | 0.196 | 2.252 | 0.20 |

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

## ACIC 2016, ten simulations

4,802 rows and 58 real covariates (79 once three categoricals are one-hot encoded), with ten
simulated treatment and outcome settings over the same covariates, from the 2016 Atlantic Causal
Inference Conference competition. The data comes from the CSVs causallib ships (Apache-2.0). The
harness pulls them out of its wheel without installing it, because causallib pins its own
scikit-learn and would change the environment the other methods run in. The categoricals are
encoded once, so every method sees the same numeric matrix.

| method | mean \|error\| | median \|error\| | mean PEHE | median PEHE |
|---|---|---|---|---|
| EconML `CausalForestDML` | **0.114** | 0.097 | 1.553 | 1.461 |
| DuckDo `causalpfn` | 0.136 | 0.126 | 1.453 | 1.140 |
| CausalPFN (python) | 0.137 | **0.091** | **1.379** | **1.119** |
| EconML `LinearDML` | 0.348 | 0.168 | 2.605 | 2.508 |
| DuckDo `aipw` | 0.433 | 0.353 | 3.086 | 3.269 |
| EconML `LinearDRLearner` | 0.506 | 0.446 | 2.903 | 2.934 |
| DuckDo `dml` | 0.530 | 0.597 | 3.086 | 3.269 |
| DoWhy PSW | 0.537 | 0.512 | — | — |

**This is the hardest benchmark here, and it is where DuckDo's classical path loses most.** `aipw`
misses the average by 0.43 and posts a PEHE of 3.09. EconML's `LinearDML`, also linear, does better
on both (0.35, 2.61). With 79 covariates and response surfaces that are not linear, a regularised
GLM base learner is the wrong tool, and the table says so.

**The flexible methods cluster at the top**, and which one leads depends on the metric.
`CausalForestDML` has the best mean error on the average effect (0.114). The two CausalPFN runs have
the best per-row error, with a PEHE around 1.4 against the forest's 1.55. On IHDP the foundation
model led on both metrics by a wide margin; here it and the forest are close.

**The ONNX export tracks the reference, but less tightly than on IHDP**: 0.136 against 0.137 on the
average, and 1.453 against 1.379 on PEHE. The wider gap has nothing to do with the graph. 4,802
rows is more than the 4,096-row context, so each side picks a subset of rows to condition on, and
they pick different ones. On IHDP's 747 rows both see every row, and the two agree to within 0.007.

Seconds are not reported for this benchmark. The machine was running other work during the sweep,
so the timings would mislead; accuracy is unaffected.

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
| DuckDo `aipw` / `dml` | **0.21** | **0.23** |
| DuckDo `causalpfn` | 3.13 | 27.2 |
| DuckDo `causalpfn`, `duckdo_query_chunk=2048` | — | 23.0 |
| CausalPFN (python, cpu) | 2.31 | 22.5 |
| CausalPFN (python, gpu) | 1.89 | 12.7 |
| EconML `LinearDML` | 3.88 | 3.55 |
| EconML `CausalForestDML` | 6.14 | 15.9 |
| DoWhy PSW | 18.80 | 3.12 |

**DuckDo's classical path is an order of magnitude faster than anything else here**,
and its 0.21 s includes launching a process and parsing a CSV. EconML's floor of
about 3 s is scikit-learn's cross-fitting machinery, and it does not shrink for
small data — which is why DuckDo beats it by 19× on 747 rows and 15× on 8,000.

Every cell above is one run. On this machine a single run of the same
configuration has varied by up to a quarter, so read the table as orders of
magnitude; the chunk table below is the median of three.

### Encoding the context once

Until this version DuckDo's CFM path was 8× slower than the reference package on
8,000 rows, on the same weights and the same CPU: 164 s against 20. DuckDo scores
query rows in chunks, and the exported graph **re-encoded the entire 4,096-row
context on every chunk**. Every CausalPFN layer takes its keys and values from the
context alone, so that work came out the same every time. At 512-row chunks it was
eight times the work spent on the queries themselves.

CausalPFN is now exported as two graphs. One encodes the context into a per-layer
key/value cache, and the other scores query chunks against it.

| `duckdo_query_chunk` | one graph | two graphs | peak MB, two graphs | estimate |
|---|---|---|---|---|
| 512 (default) | 125.8 s | **31.3 s** | 1,938 | 2.9705805124938487 |
| 2048 | 42.8 s | 23.0 s | 2,728 | 2.9705805124938487 |
| 8192 | 23.3 s | 22.2 s | 5,030 | 2.9705805124938487 |

**The estimate is bit-identical in every cell**, across both layouts and every chunk
size, all eighteen digits.

Chunk size used to be a real trade, costing 5.3 GB to reach parity, and mostly no
longer is. The default runs four times faster than it did, at slightly less memory
than it used. Raising it to 2048 buys the rest for another 0.8 GB and reaches the
package's CPU speed; past that there is nothing left to buy.

Two costs. The weights grow from 75 MB to 120 MB, because both graphs carry the
layers. And the comparison is lopsided: the two-graph column is the median of three
runs, while the one-graph column is single runs of a layout that no longer exists.
A single run first put the two graphs *slower* at 8192, 29.2 s against 23.3 s. With
three runs that became 22.2 s. Read the one-graph column as order of magnitude.

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
- **ACIC is ten simulations**: the ones causallib ships, one replication each, not the full
  competition set.
- **One machine**, one run per cell. The timings are not averaged over repeats and
  should be read as order-of-magnitude.
- **No standard errors on the errors.** Whether 0.078 beats 0.082 is not something
  ten replications can settle; whether 0.416 beats 2.226 is not in doubt.
- **Accuracy against a known truth is not accuracy on your data.** Every dataset
  here satisfies the assumptions by construction. The thing that actually breaks
  causal estimates is an unmeasured confounder, and no benchmark of this kind can
  measure that. See [ASSUMPTIONS.md](ASSUMPTIONS.md).
