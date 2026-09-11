# DuckDo

**Causal inference inside DuckDB.** `SELECT` tells you what happened. `do_ate()` tells you what your
last change was worth.

DuckDo makes DuckDB an in-process causal engine: treatment effects, assumption diagnostics and
graph-based identification expressed as ordinary SQL over ordinary tables. No Python, no training
loop, no data leaving the process, and **no network request you did not ask for by name**: no
usage ping, no version check. The one function that can reach the network is `do_download`, and it
does so only when you call it: with a URL, or with no source for CausalPFN, the one model DuckDo
hosts, fetched from a pinned revision and checked against compiled-in checksums. Causal analysis runs
on the data that is worth analysing.

```sql
SELECT estimand, estimator, round(estimate, 2) AS estimate, round(ci_low, 2), round(ci_high, 2)
FROM do_ate('customers',
     treatment  := 'received_discount',
     outcome    := 'revenue',
     covariates := ['age', 'income', 'tenure']);
-- ATE | aipw | 12.84 | 8.59 | 17.09
```

New here? **[docs/TUTORIAL.md](docs/TUTORIAL.md)** walks one real question end to end in about
five minutes, and **[docs/EXAMPLES.md](docs/EXAMPLES.md)** works three more — an A/B test nobody
complied with, a pricing question where the naive answer has the wrong *sign*, and a retention
call worth targeting. **[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)** is what every estimate rests
on, written for analysts. Full signatures: **[docs/FUNCTIONS.md](docs/FUNCTIONS.md)**.

Weighing it against something else? **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** grades DuckDo
against EconML, DoWhy and the CausalPFN package on identical rows, with the losses shown, and
**[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** says what is reproducible to the bit, what
is not, and how to pin it.

## ELI5

You email a discount to some customers. The ones who got it spend more. Did the discount cause
that, or did you send it to the people who were going to spend more anyway?

`SELECT` cannot separate those. It reports what happened, and comparing the two groups only
answers the question when the groups are alike in every other way. Send the coupon to your most
loyal customers and the coupon takes credit for their loyalty.

```sql
-- What the groups did. The discount and everything else that differs between them.
SELECT received_discount, avg(revenue) FROM customers GROUP BY received_discount;

-- What the discount was worth: like compared with like, with an interval around it.
SELECT estimate, ci_low, ci_high
FROM do_ate('customers', treatment := 'received_discount', outcome := 'revenue',
            covariates := ['age', 'income', 'tenure']);
```

`do_ate` asks what would have happened to the same people under the other choice. It weighs each
customer against others who look like them — same age, income and tenure, one who got the discount
and one who did not — and reports the difference, how uncertain it is, and anything that looked
wrong along the way.

The catch is the covariate list. This works only if you measured the things that drive both who
got the discount and what they spent; anything you left out lands in the answer as bias, and no
amount of data fixes it. [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) is that list, in plain words,
and `do_diagnose` checks the parts data can check.

## What works today

Everything below runs with no downloads, no ONNX and no network.

### Estimation

| Function | Returns |
|---|---|
| `do_ate` / `do_att` / `do_atc` | one row: `estimand, estimator, estimate, std_error, ci_low, ci_high, p_value, n, n_treated, n_trimmed, variance_method, warnings` |
| `do_cate` | one row per input row: `row_id, id, treatment, outcome, cate, cate_low, cate_high, learner` |
| `do_ate_by` | one independent estimate per segment; a group too thin to estimate returns NULLs and the reason, rather than failing the query |

Estimators via `estimator :=` - `aipw` (doubly robust, the default), `dml` (cross-fitted partially
linear), `ipw` (Hajek), `regression` (g-computation), `naive`, and the `s_learner` / `t_learner` /
`x_learner` / `dr_learner` meta-learners. Propensity and outcome models are cross-fitted 5-fold with
a seeded, stratified split, so results are reproducible run to run.

### Diagnostics - assumptions as rows

Every diagnostic returns a boolean verdict column, so it drops straight into a dbt test or a CI query.

```sql
SELECT * FROM do_balance('customers', treatment := 'discount');       -- SMD raw vs weighted
SELECT * FROM do_overlap('customers', treatment := 'discount');       -- propensity support
SELECT * FROM do_diagnose('customers', treatment := ..., outcome := ...);
SELECT * FROM do_refute(..., method := 'placebo_treatment');          -- also random_common_cause,
                                                                      -- subset, bootstrap,
                                                                      -- unobserved_confounder
SELECT * FROM do_sensitivity(...);   -- E-value and Cinelli-Hazlett robustness value
```

### Graphs - what should I even adjust for?

```sql
CALL do_graph_create('sales_dag', 'digraph {
  intent [latent];
  season -> discount;  season -> revenue;
  discount -> clicks;  clicks -> revenue;
  intent -> discount;  intent -> revenue;
  coupon_mail -> discount;
}');

SELECT strategy, identifiable, adjustment_set FROM do_identify(
       graph := 'sales_dag', treatment := 'discount', outcome := 'revenue');
-- backdoor  | false | []              -- 'intent' is an unobserved common cause
-- frontdoor | true  | [clicks]
-- iv        | true  | [coupon_mail]

SELECT covariate, role, verdict FROM do_validate(
       graph := 'sales_dag', treatment := 'discount', outcome := 'revenue',
       covariates := ['season', 'clicks', 'coupon_mail']);
-- season      | confounder | keep
-- clicks      | mediator   | DROP    -- adjusting removes part of the effect
-- coupon_mail | instrument | DROP    -- reaches the outcome only through the treatment

SELECT do_dseparated('sales_dag', 'season', 'loyalty', ['revenue']);  -- false: collider opened
```

Also `do_graphs()`, `do_graph_drop()`. Graphs live in a `duckdo_graphs` table, so they survive a
restart and can be inspected, backed up or version-controlled like any other data.

### When the backdoor is closed, take another door

`do_identify` will tell you the backdoor is blocked by an unobserved confounder but that a front-door
mediator or an instrument is available. Both are estimable:

```sql
-- Two-stage least squares, with the weak-instrument check in the output rather
-- than left to the reader.
SELECT estimate, ci_low, ci_high, first_stage_f, weak_instrument
FROM do_iv('trial', treatment := 'took_drug', outcome := 'health',
           instrument := 'assigned_to_drug', covariates := ['age']);
-- 2.044 | 2.012 | 2.075 | 5875.2 | false

-- The front-door product formula, with both halves reported separately because
-- that is where it goes wrong.
SELECT estimate, effect_t_on_m, effect_m_on_y
FROM do_frontdoor('sales', treatment := 'discount', outcome := 'revenue', mediator := 'clicks');
-- 1.997 | 0.501 | 3.984
```

On a DGP with an unobserved confounder and a true effect of 2.0, ordinary regression gives 2.582 and
`do_iv` gives **2.044**; where the treatment acts only through a mediator, the naive difference gives
4.903 and `do_frontdoor` gives **1.997**. Point a useless variable at `instrument :=` and the
first-stage F comes back at 0.08 with `weak_instrument = true`.

### How much of the effect goes through the mediator

```sql
SELECT estimand, estimate, ci_low, ci_high, proportion
FROM do_mediate('signups', treatment := 'onboarding', outcome := 'retention',
                mediator := 'activated', covariates := ['plan']);
-- total            | 3.131 | 3.096 | 3.166 | 0.515
-- natural_direct   | 1.518 | 1.469 | 1.568 | 0.515
-- natural_indirect | 1.613 | 1.568 | 1.659 | 0.515
```

Natural direct and indirect effects, carrying the treatment-mediator interaction — which is what
separates this from Baron-Kenny. On a DGP whose true split is 1.80 / 2.80, fitting without that
interaction gives 2.41 / 2.22 while the total stays correct at 4.63. A wrong split under a right
total is the failure that survives a sanity check, so the interaction is reported as
`tm_interaction` rather than absorbed.

`proportion` comes back `NULL`, with a warning, when the total effect straddles zero — a proportion
mediated is a ratio, and that denominator makes it meaningless. Every row carries the assumption
the whole thing rests on: **randomising the treatment does not buy sequential ignorability**. It
leaves mediator-outcome confounding completely untouched.

### When the confounder moves with the treatment

```sql
SELECT estimate, ci_low, ci_high, mean_weight, effective_n
FROM do_msm('patient_months', unit := 'patient_id', period := 'month',
            treatment := 'on_drug', outcome := 'bp', covariates := ['creatinine']);
-- 2.130 | 1.962 | 2.298 | 0.989 | 6245
```

A confounder that affects treatment, affects the outcome, *and* is affected by earlier treatment
has no correct adjustment set: leave it out and it confounds, put it in and it blocks part of the
effect. On a DGP built that way with a true effect of 2.0 per treated period, no adjustment gives
3.92, adjusting for every measured confounder gives 3.47 — faithfully recovering a coefficient that
is not the causal effect — and `do_msm` gives **2.13**.

`mean_weight` should sit near 1, and the result says so when it does not. `truncate :=` is off by
default because on this data it raises effective *n* from 6,245 to 12,101 and halves the interval
while moving the estimate from one that covers the truth to one that excludes it.

### Time-to-event outcomes

```sql
SELECT estimate, ci_low, ci_high, horizon, rmst_treated, rmst_control
FROM do_rmst('trial', treatment := 'arm', duration := 'days_followed', event := 'relapsed',
             covariates := ['age', 'stage']);
-- 0.904 | 0.851 | 0.957 | 5.0 | 3.218 | 2.315
```

The difference in **restricted mean survival time** — how much longer a treated subject stays
event-free within the horizon. There is deliberately no hazard ratio: a hazard ratio compares
subjects still at risk, treatment changes who is still at risk, and the comparison stops being
causal after the first events even under randomisation.

On an exponential DGP whose closed-form truth is 0.9239 this returns 0.904 with an interval
covering it; unadjusted it returns 0.443, so confounding hides more than half the benefit. The
horizon is part of the estimand and defaults to the last time both arms still had 5% of subjects
at risk, because the obvious default lands where a handful of people remain.

### Causal foundation models

```sql
SELECT model, setting, license, max_covariates, available FROM do_list_models();
-- causalpfn | backdoor (ignorability) | CausalPFN License 1.0 | 99 | true
-- do_pfn    | non-identifiable prior  | none stated upstream  |  5 | true

-- Same SQL, different engine. The result names the model, not a classical estimator.
SELECT estimator, estimate, variance_method
FROM do_ate('customers', treatment := 'discount', outcome := 'revenue', model := 'causalpfn');
-- causalpfn | 3.061 | effect dispersion (no model uncertainty)
```

| | [CausalPFN](https://github.com/vdblm/CausalPFN) | [Do-PFN](https://github.com/jr2021/Do-PFN) |
|---|---|---|
| Parameters | 18.8M | 7.3M |
| Identification setting | backdoor (ignorability) | non-identifiable prior |
| Licence | CausalPFN License 1.0 (Apache-2.0's terms) | **none stated upstream**: all rights reserved |
| Covariates accepted | 99 | 5 |
| Context length | dynamic to 4096 | fixed ladder: 128/512/1024/2048 |
| Shrinks the ATE? | no | **yes, materially** |

**Start with CausalPFN.** It targets the backdoor setting directly rather than hedging under a
non-identifiable prior, and it does not exhibit the shrinkage Do-PFN does.

**Ask for an interval.** A single forward pass cannot see how much its answer depends on which rows
landed in the model's context, so it reports an interval that is far too narrow. `ensemble := k`
re-draws the context k times and folds the spread in:

```sql
SELECT estimate, ci_low, ci_high, variance_method
FROM do_ate('customers', treatment := 'discount', outcome := 'revenue',
            model := 'causalpfn', ensemble := 8);
-- 3.0637 | 3.007 | 3.121 | context ensemble, 8 draws (no parameter uncertainty)
```

On a DGP with a true effect of 3.0, the single-draw interval is `[3.035, 3.048]` — it **excludes the
truth**. The 8-draw interval `[3.007, 3.121]` contains it. Each draw is a full forward pass, so this
costs k times as much.

**On an NVIDIA GPU, CausalPFN is about 8x faster.** Build with `-DDUCKDO_ORT_FLAVOUR=cuda12` and
`SET duckdo_device = 'cuda'`. On an RTX 3060, 8,000 rows take 5.1 s against 40.0 s on the CPU
(medians of three runs), and the estimate matches the CPU's to about 2e-7. The CPU stays the default, so no result moves
because a GPU happens to be present. Do-PFN runs on the CPU only: on CUDA its graph crashed the
process, so it refuses.

The extension itself never ships weights. Getting a model running takes two steps:

```sh
# 1. build with ONNX Runtime - the pinned release is fetched and staged for you
cmake -DDUCKDO_WITH_ONNX=ON ...
```

```sql
-- 2. fetch CausalPFN's hosted export: pinned to one revision, every file checked
--    against a SHA-256 compiled into the extension before it is installed
SELECT file, status FROM do_download('causalpfn');
```

The export is published at [maxdemarzi/duckdo-causalpfn](https://huggingface.co/maxdemarzi/duckdo-causalpfn),
with the upstream licence, a notice of every changed file, and the authors' citation. To build it
yourself instead, run `pip install causalpfn` and then
`python scripts/export/export_causalpfn.py --out ~/.cache/duckdo`.

Do-PFN's upstream repository states no licence, so its weights are all rights reserved and DuckDo
does not host them. If you have the right to use them, export them for yourself:

```sh
git clone https://github.com/jr2021/Do-PFN
python scripts/export/export_dopfn.py --repo ./Do-PFN --out ~/.cache/duckdo
```

Files go to `duckdo_model_dir`, which defaults to `~/.cache/duckdo`.

Both exports refuse to finish unless the graph reproduces PyTorch. Do-PFN is gated on logits, to
**1e-4** (measured 4.3e-06 to 8.1e-06). CausalPFN is gated on the *estimand* instead, because
PyTorch runs a fused attention kernel where ONNX Runtime decomposes it: over 12 layers and a
1024-bin softmax that costs ~1e-2 of logit agreement in fp32, while the ATE still agrees to
**0.00087** with a CATE correlation of **0.9994**. Gating on the number a user actually reads is
the honest test.

### Continuous treatments - a dose has a curve, not a number

```sql
-- What does one more unit of the dose buy, on average?
SELECT estimand, estimate, ci_low, ci_high, dose_min, dose_max
FROM do_ape('pricing', treatment := 'price', outcome := 'revenue');
-- APE | 2.0019 | 1.9921 | 2.0118 | -5.91 | 5.40

-- And the whole response, so you can see whether that average means anything.
SELECT dose, mu, mu_low, mu_high, n_within_decile
FROM do_dose_response('pricing', treatment := 'price', outcome := 'revenue', grid := 20);
```

`do_ape` is the average partial effect from a partially linear double-ML model — residualise both
the outcome and the dose on the covariates, then regress one residual on the other.
`do_dose_response` is g-computation over a model quadratic in the dose, averaged across the
covariate distribution at each grid point. The grid sits on **dose quantiles**, so it never
extrapolates past where data exists, and `n_within_decile` says how much support each point has.

The binary estimators still refuse a dose outright, and point here.

### More than two arms

```sql
SELECT level, reference, estimate, ci_low, ci_high, naive_difference
FROM do_ate_levels('arms', treatment := 'arm', outcome := 'y', exclude := ['id']);
-- B | A |  1.472 |  1.375 |  1.568 |  4.185
-- C | A | -0.992 | -1.085 | -0.899 | -2.717
```

`do_ate_levels` returns one row per level, each against a reference (`reference :=`, the first level
by default), and every row contributes to every contrast. Under the hood it is a multi-arm AIPW with
cross-fitted one-vs-rest propensities and an outcome surface per level. The truths here are 1.5 and
-1.0; the naive differences are off by nearly a factor of three.

`do_ate(..., treated := 'B', control := 'A')` also works on a three-level column, but it answers a
different question: the effect among the units that received A or B. When who gets which arm
depends on the covariates, which is the reason to adjust at all, that subpopulation is not the
population. On the table above the pairwise route returns **2.467**, right on that subpopulation's
truth of 2.464 and a full unit from the population's 1.5. Neither is wrong; they are different
estimands, and the pairwise one is rarely the one a decision needs.

### Panel data - difference-in-differences that does not misweight

```sql
SELECT estimand, estimator, estimate, ci_low, ci_high, n_cohorts, pre_trend
FROM do_did('rollout', unit := 'store_id', period := 'month',
            treatment := 'has_feature', outcome := 'revenue');
-- ATT | callaway-santanna | 4.878 | 4.671 | 5.084 | 2 | 0.126

-- The pre-treatment periods are the parallel-trends check. Look at them.
SELECT relative_period, att, ci_low, ci_high, is_pre_treatment
FROM do_event_study('rollout', unit := 'store_id', period := 'month',
                    treatment := 'has_feature', outcome := 'revenue');
```

**DuckDo does not ship two-way fixed effects.** Under staggered adoption TWFE lets already-treated
units serve as controls for later adopters, and the coefficient becomes a weighted average that can
carry negative weights. `do_did` computes group-time effects in the Callaway–Sant'Anna sense
instead — never-treated units as the comparison where any exist, not-yet-treated otherwise —
aggregated by cohort size. Standard errors come from a **unit-level** cluster bootstrap, because
units are the independent draws, not unit-periods.

`do_event_study` is the part that earns its keep: pre-treatment periods are a direct read on whether
parallel trends is plausible, which is the panel version of *assumptions are queryable objects*.
`do_did` also reports the average pre-trend and warns when it is large relative to its standard
error.

On a staggered panel with a true ATT of exactly 5.0, comparing treated to untreated rows gives
**6.560**; `do_did` gives **4.878** with an interval covering the truth, and the event study shows
pre-periods at −0.02 and 0.20 against post-periods of 4.93 to 5.03.

**When parallel trends holds only given covariates, pass them.** `do_did(..., covariates := ['x'])`
estimates each cell doubly robustly (Sant'Anna–Zhao). It matches the comparison group's trend to the
treated cohort's covariates through an outcome model, a propensity model, or both. The test case is
a panel where units with a high `x` trend up faster *and* are likelier to be treated, with a true
effect of 2.0:

| | estimate | 95% interval | pre-trend |
|---|---|---|---|
| plain `do_did` | 3.877 | [3.638, 4.115] | −0.982 |
| `covariates := ['x']` | **1.841** | [1.600, 2.082] | −0.049 |

The event study shows the difference directly. Unconditionally the pre-periods drift, down to −1.34,
and the post-periods climb from 2.63 to 5.12 as the x-driven trend compounds. Conditioning on x, the
pre-periods sit at −0.08 and −0.02 and the post-periods stay between 1.74 and 1.92. The pre-trend is
the diagnostic that catches this, which is why `do_event_study` takes the same `covariates :=`.

**When no comparison group trends like the treated unit, build one.** `do_synth` fits a synthetic
control (Abadie, Diamond and Hainmueller): non-negative donor weights, summing to one, that reproduce
the treated unit's pre-treatment path. The effect is then read off the gap after treatment.
Inference is by in-space placebos: each donor is treated as if it had been, and the real unit's
post/pre fit ratio is ranked among theirs.

```sql
SELECT estimate, pre_rmspe, rmspe_ratio, p_value, weights
FROM do_synth('regions', unit := 'region', period := 'quarter',
              treatment := 'policy', outcome := 'sales');
-- 5.053 | 0.154 | 32.9 | 0.0476 | {unit_20=0.409, unit_17=0.29, unit_19=0.214, ...}

SELECT period, actual, synthetic, gap, is_pre_treatment
FROM do_synth_path('regions', unit := 'region', period := 'quarter',
                   treatment := 'policy', outcome := 'sales');
```

Take a one-factor panel where the treated unit trends faster than the average donor, with a true
effect of 5.0. `do_did` gives 6.277. `do_synth` gives 5.053, with a pre-period RMSPE of 0.154, and
the real unit ranks first of 21 placebo runs, the smallest p-value that 20 donors allow. The weight
solver is certified against scipy's SLSQP: on a strictly convex fit they agree to 1.6e-8 in the
weights and 6e-14 in the objective (`scripts/synth_check.py`).

**When you have no graph, discovery can propose one, but it can't register one.** `do_discover`
runs PC-stable and reports every edge with its bootstrap stability. `do_discover_dot` writes the
proposal as DOT, and `do_graph_create` refuses that DOT twice: once until you delete its
`do_discover: unreviewed` line, and once for every edge the data could not orient (written `a -- b`)
until you pick a direction.

```sql
SELECT source, edge, target, stability, orientation_stability
FROM do_discover('measurements');
-- a -> c | b -> c | c -> d | d -> f   each in 100% of resamples
-- g -- h                              no direction the data can supply

SELECT dot FROM do_discover_dot('measurements');   -- review, edit, then do_graph_create
```

On a seven-variable linear-Gaussian world, the full-data graph above is exactly right. Everything it
orients traces back to one collider, a → c ← b, and Meek's rules carry that down the chain. Drop `b`
and a → c → d → f comes back entirely undirected, which is correct: a chain with no collider
implies the same distribution in either direction. Also read `orientation_stability`. Here it is
0.88 even though the graph is right, because resamples reject independence more often than `alpha`
and sometimes join a and b. At `alpha := 0.001` it is 0.98. Discovery assumes no hidden common
causes, and real data usually has them.

**The `--` edges have two honest ways out, and a guess is not one of them.** `tiers :=
[['age', 'sex'], ['discount'], ['revenue']]` says nothing in a later group causes anything in an
earlier one, which settles every edge that crosses a tier before any test runs — that is how these
edges get settled in practice, by someone who knows the order rather than by a better test.
`algorithm := 'lingam'` settles them from the data instead: where PC reads conditional
independence, which cannot tell `x -> y` from `y -> x`, DirectLiNGAM reads the shape of each
variable's disturbance, which can. It costs a longer assumption list, and the last item is enforced
rather than mentioned — two disturbances that cannot be told from Gaussian is a refusal, because
identifiability allows one. `algorithm := 'both'` runs the two and adds an `agreement` column
saying which method gave each direction, leaving the pairs they contradict each other on
undirected and in the review pile.

```sql
SELECT source, edge, target, agreement
FROM do_discover('measurements', algorithm := 'both');
-- a -> c   both                 both methods, same direction
-- g -> h   oriented by lingam   the edge PC had to leave for a person
```

**When rows are not units, say what the unit is.** Joining customers to their orders turns each
customer into several rows, and every estimator treats rows as independent. The estimate survives
this, but the interval does not:

| 4,000 customers | estimate | std. error |
|---|---|---|
| one row each | 1.978 | 0.0352 |
| joined to 3 orders each | 1.979 | 0.0203 |
| joined, `cluster := 'customer_id'` | 1.978 | 0.0352 |

```sql
SELECT estimate, std_error, variance_method
FROM do_ate('(SELECT * FROM customers JOIN orders USING (customer_id))',
            treatment := 'discount', outcome := 'revenue', cluster := 'customer_id');
```

The join gives an interval 42% too narrow, with no warning, because a join leaves no trace in the
rows. `cluster :=` makes the customer the unit for cross-fitting folds, standard errors and
bootstrap resamples, and gives back the customers' own answer. A repeated `id :=` is the one sign
DuckDo can see without being told, so any function given one warns when its values repeat.

**When the data can't leave the site, pool the answers instead.** Each site runs `do_ate` on its
own rows and publishes the one row it gets back. `do_ate_pool` combines those rows into the effect
over every site's population together. The pooling is exact: an AIPW estimate is a mean of
per-row influence values, so the size-weighted mean of site estimates is the pooled mean.

```sql
SELECT estimate, ci_low, ci_high, i_squared FROM do_ate_pool('site_results');
```

Take three sites of 2,000, 1,000 and 3,000 units, with effects of 1, 2 and 3, so the effect over
all of them is 2.167:

| method | estimate | 95% interval |
|---|---|---|
| `do_ate_pool`, no rows shared | 2.138 | [2.084, 2.192] |
| inverse-variance weighting | 1.991 | [1.978, 2.004] |
| `do_ate` with every row in one place | 2.134 | [2.070, 2.198] |

Inverse-variance weighting is the meta-analysis default, and here it is precise and wrong. The
middle site's outcome is nearly noiseless, so its tiny standard error outvotes the other two.
`do_ate_pool` uses precision weights only for its heterogeneity test. Here I² is 0.998, so the
result warns that the pooled number averages three different effects.

### Interventions - querying a world that did not happen

```sql
-- Force every row into a counterfactual state. This is do(X = x), not a filter
-- on X = x: no rows are dropped, the world is changed.
SELECT avg(predicted) FROM do_predict('customers',
       treatment := 'discount', outcome := 'revenue',
       intervention := {'price': 19.99});

SELECT * FROM do_counterfactual(...);   -- per-row y0, y1, effect and interval
SELECT * FROM do_uplift(...);           -- Qini curve rows, ready to plot
SELECT * FROM do_optimal_policy(..., depth := 2);  -- a shallow, deployable rule

-- Is my targeting rule worth anything? A negative lift means it is worse than
-- treating everyone.
SELECT policy, policy_value, lift_over_treat_all
FROM do_policy_value('(SELECT *, tenure > 12 AS my_rule FROM customers)',
     treatment := 'discount', outcome := 'revenue', policy := 'my_rule');
```

Every function is registered twice: the short `do_*` name and the unambiguous `duckdo_*` full name.

## Does it actually work?

On a confounded synthetic DGP with a true ATE of exactly 3.0 (20,000 rows, `x1` driving both
treatment assignment and the outcome):

| estimator | estimate | std_error | 95% interval |
|---|---|---|---|
| `naive` | 3.718 | 0.030 | [3.659, 3.777] |
| `aipw` | **2.996** | 0.016 | [2.964, 3.028] |
| `dml` | **3.002** | 0.016 | [2.971, 3.033] |
| `ipw` | **2.999** | 0.037 | [2.926, 3.072] |
| `regression` | **3.002** | 0.016 | [2.971, 3.034] |

The naive contrast is biased by +0.72, which is precisely the confounding the adjusted estimators
remove. With a heterogeneous effect of `3 + 2*x1`, `do_cate` correlates 0.9995 with the truth and
`do_att` (3.72) > ATE (3.00) > `do_atc` (2.29), as it must when the treated have higher `x1`.

**Recovering synthetic truth is not enough**, so the estimators are also graded against established
implementations on identical rows:

| scenario | truth | duckdo `aipw` | econml `LinearDRLearner` | dowhy PSW |
|---|---|---|---|---|
| linear confounded | 3.0000 | 3.0179 | 3.0180 | 3.0337 |
| heterogeneous | 2.9936 | 2.9858 | 2.9716 | 2.9662 |
| IHDP npci-1 | 4.0161 | 3.8310 | 3.9555 | 4.0287 |

The gate is agreement to within a tenth **or one standard error**, whichever is more forgiving.
An absolute tolerance alone reads as strict and is not: 0.12 is nothing on IHDP, where 747 rows
give a standard error of 0.18, and would be alarming on a clean 8,000-row synthetic where it is
0.03. Two doubly-robust estimators that split their folds differently *will* differ on a small
sample, and a gate that calls that a failure only teaches people to loosen gates.

Across **all ten IHDP replications** the mean absolute ATE error is **0.164**, and `do_cate`'s mean
PEHE is 2.25 — dominated by replication 9, whose true effect is 10.5 where the others sit near 4.

On **ten ACIC 2016 simulations** (4,802 rows, 79 encoded covariates, nonlinear surfaces) the
linear estimators trail. `aipw` misses the average by 0.43, behind EconML's `LinearDML` at 0.35,
while `model := 'causalpfn'` lands at 0.136 beside EconML's causal forest at 0.114. The full
tables, including where each method loses, are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

And on **Lalonde NSW**, where treatment was randomised so the unadjusted difference *is* the causal
effect, the adjusted estimators have a benchmark to reproduce rather than improve on:

| | 1978 dollars |
|---|---|
| experimental benchmark | **1794.3** |
| duckdo `dml` | 1709.8 |
| econml `DRLearner` | 1688.6 |
| duckdo `aipw` | 1680.6 |
| duckdo `ipw` | 1665.7 |

All four land within 7% of the benchmark on a noisy 445-row sample (185 treated).

Reproduce the first table with the estimator tests in the repository, and the cross-check with
`python scripts/crosscheck_econml.py`, which needs `econml` and `dowhy` installed.

### And does the foundation model beat them?

**On a linear DGP, no. On real benchmark data, decisively — and mostly on the metric that
decides whether a targeting rule works.**

Three engines, identical SQL, 1500 rows of a *linear* DGP, true ATE **2.9806**:

| engine | estimate | error | CATE correlation |
|---|---|---|---|
| `estimator := 'aipw'` | 3.0628 | +0.082 | 0.9995 |
| `model := 'causalpfn'` | **3.0610** | **+0.080** | **0.9995** |
| `model := 'do_pfn'` | 2.6307 | −0.350 | 0.984 |

**CausalPFN matches AIPW** where AIPW is correctly specified, which is the most it could do.
Do-PFN **shrinks the population effect** — the documented weakness of that model class,
reproduced here rather than hidden — while still ranking individuals well.

Change the data to IHDP, ten replications of a real covariate distribution, and the gap opens:

| | mean \|ATE error\| | **mean PEHE** |
|---|---|---|
| `model := 'causalpfn'` | **0.078** | **0.416** |
| CausalPFN, the Python package | 0.082 | 0.423 |
| `estimator := 'aipw'` | 0.137 | 2.226 |
| econml `LinearDML` | 0.147 | 2.161 |
| econml `CausalForestDML` | 0.390 | 2.749 |

**PEHE 0.42 against 2.16** — a factor of five on per-row effects, which is what a targeting rule
rides on. That is the case foundation models were built for: a covariate structure nobody
specified in advance. It is also the reason the linear result above is not the headline.

DuckDo's ONNX export lands within 0.007 of the reference package on both metrics across all ten
replications (0.0041 on mean |error|, 0.0066 on mean PEHE), which is much stronger evidence that the export is faithful than the tensor-level
parity check in the export script.

The losses, the costs, and the datasets where EconML's forest learners beat DuckDo's classical
path outright are in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

## Settings

`duckdo_default_estimator`, `duckdo_max_rows` (1M), `duckdo_max_features` (500),
`duckdo_max_memory` (half of DuckDB's `memory_limit`), `duckdo_max_categorical_levels` (32),
`duckdo_max_groups` (1000), `duckdo_seed` (42), `duckdo_bootstrap_reps` (200), `duckdo_threads`
(0 = one per hardware thread). Every guardrail names the setting to raise when it trips, and
the memory one also names the row count that would fit at the frame's width.

### Cost

The dense accumulations that dominate every fit are split across row blocks. **How** they are split
depends on the data and never on the thread budget, so results are bit-identical whatever
`duckdo_threads` is set to — floating-point addition is not associative, and a split that follows
the core count makes the answer follow the core count too. Bootstrap replicates are parallel across
replicates, with each replicate seeded from `(seed, rep)` rather than a shared stream, so the same
seed gives the same interval on one core or thirty-two. Each shape is generated to Parquet once, outside the
measurement, so both columns describe the estimator rather than the table generator
(`scripts/benchmark.py`):

| rows | covariates | estimator | seconds | peak MB |
|---|---|---|---|---|
| 100k | 5 | `aipw` | 0.2 | 35 |
| 100k | 50 | `aipw` | 1.6 | 108 |
| 1M | 5 | `aipw` | 1.7 | 143 |
| 1M | 50 | `aipw` | 14.2 | 882 |
| 1M | 50 | `dml` | 14.2 | 882 |

The 1M × 50 case was 101 s single-threaded before the accumulation was parallelised.

Memory is the other half of the cost, and it was worse than the timings. The encoded matrix for
1M × 50 is 400 MB, and every estimator indexes it by row, so it stays resident for the whole
query. The peak while building it was **1,645 MB** — four times the matrix — because the
materialised scan result, the raw per-column buffers and the staged features all stayed live long
after they were dead. Releasing each at the point it goes dead brings the peak to **882 MB**, with
every estimate identical to the last digit and no change in runtime. The benchmark now gates on
this, so a stray full copy of the frame shows up as a failure rather than as a slow laptop.

## Known limitations

Stated plainly, because a causal tool that hides its limits is worse than none.

- **The effect estimators take a binary treatment. A dose goes through `do_ape` and
  `do_dose_response`, and three or more arms through `do_ate_levels`.** The binary path refuses
  anything else rather than silently binarising it, and says where to go. `do_ate_levels` offers
  AIPW only. The dose-response model is quadratic in the dose, so a sharply non-monotone response
  will be smoothed.
- **`do_cate` intervals are calibrated, and an earlier claim here that they were not was noise.**
  Measured 95% coverage is 0.948 ± 0.014, 0.948 ± 0.016 and 0.944 ± 0.014 across randomised,
  confounded and strongly-confounded DGPs — 40 replicates of 4,000 rows, standard error across
  replicates (`scripts/coverage_check.py`). None of the three is distinguishable from nominal.
  The interval uses an HC1 sandwich; without it the same measurement gives 0.948 / 0.937 / 0.909,
  degrading exactly as confounding strengthens, which is the signature of the heteroskedasticity
  it corrects. A single replicate covers anywhere between 0.56 and 1.00 of its rows, so coverage
  quoted without a standard error is not a measurement — this README previously quoted one.
- **Base learners are regularised GLMs, and this is measurable.** On a DGP whose effect is a step
  function, `do_cate` posts a PEHE of **0.992** where EconML's `CausalForestDML` posts **0.213** and
  `model := 'causalpfn'` posts 0.451. The average effect is barely touched (0.020 against 0.011),
  so this costs you targeting rules, not headline numbers. Gradient-boosted base learners are
  planned; until then, reach for the foundation model when the structure is unknown.
  Full table in [docs/BENCHMARKS.md](docs/BENCHMARKS.md#where-the-classical-path-loses).
- **Data is read on a separate connection**, so uncommitted changes in your current transaction are
  not visible to an estimation call.
- **Foundation model intervals need `ensemble :=`, and still miss parameter uncertainty.** A single
  forward pass reports only the dispersion of its own point estimates, which is far too narrow —
  on the DGP above the single-draw interval is [3.035, 3.048] and **excludes the true 3.0**, while
  `ensemble := 8` gives [3.007, 3.121] and covers it. Even then the interval covers context
  selection and sampling, not the model's own weights, and it costs one forward pass per draw.
- **Do-PFN's five-covariate budget is disqualifying on a real table.** On twelve contributing
  covariates with a true effect of 3.0 it returns 5.24, while CausalPFN returns 2.984 — and a
  correctly specified AIPW restricted to the same five covariates returns 4.07, so the budget
  rather than the model does most of the damage. `ensemble := k` spreads the choice of covariates
  across draws, which widens the interval from ±0.006 to ±0.31, but dropped confounders are bias
  and no resampling interval covers bias. Use CausalPFN.
- **CausalFM is not exported, deliberately.** Its published front-door checkpoint returns the
  same number, -0.032, for every row of every dataset, and its reported PEHE is exactly what that
  constant scores (predicting zero does slightly better). Its IV checkpoint responds to its input
  but beats predicting zero by only 7%, and where an instrument is strong enough to identify the
  effect it returns roughly the confounded naive difference: 3.00 against a truth of 2.0, where
  `do_iv` returns 2.04. The evidence reproduces with `scripts/causalfm_check.py`.
- **The foundation-model path is close to the CausalPFN package's CPU speed, not level with it.**
  On 8,000 rows the default takes 31.3 s (median of three runs) against the package's 22.5 s, and
  `duckdo_query_chunk = 2048` closes the gap: 23.0 s at a 2.7 GB peak. It used to take 126 s,
  because the exported graph re-encoded the whole 4,096-row context for every 512-row query
  chunk. CausalPFN is now exported as two graphs: one encodes the context once, the other scores
  chunks against the cached keys and values. The estimate is **bit-identical** to the old single
  graph. Re-export with `scripts/export/export_causalpfn.py` to get the two-graph layout. The
  weights grow from 75 MB to 120 MB, because both graphs carry the layers.
- **ONNX Runtime links dynamically.** `-DDUCKDO_WITH_ONNX=ON` fetches the pinned release and stages
  its libraries next to the binaries, so this is handled rather than manual — but the community
  build still ships without it, because a shared dependency is exactly what the dependency-free
  default is protecting. There is no static build to link: none is published upstream.

## Running the extension

Build it first ([CONTRIBUTING.md](CONTRIBUTING.md)), start the shell with
`./build/release/duckdb`, and then:

```sql
-- A tiny randomised experiment: treatment is assigned by coin flip, so the
-- naive difference is already unbiased and every estimator should agree.
CREATE TABLE trial AS
SELECT i AS user_id,
       (i % 2) AS nudged,
       (i % 7) * 1.0 AS tenure,
       10.0 + 0.5 * (i % 7) + 2.0 * (i % 2) + ((i * 37) % 11) * 0.1 AS minutes
FROM range(2000) t(i);

SELECT estimand, estimator, round(estimate, 3) AS estimate,
       round(ci_low, 3) AS ci_low, round(ci_high, 3) AS ci_high, n
FROM do_ate('trial', treatment := 'nudged', outcome := 'minutes',
            covariates := ['tenure']);
```

Then check the assumptions behind that number:

```sql
SELECT check_name, status, severity, detail
FROM do_diagnose('trial', treatment := 'nudged', outcome := 'minutes',
                 covariates := ['tenure']);
```

## Building it, and working on it

Building from source, running the test suite, the development harnesses and the release
checklist are in [CONTRIBUTING.md](CONTRIBUTING.md).
