# DuckDo function reference

Every function is registered twice: the short `do_*` name shown here, and an
identical `duckdo_*` full name for scripts where `do_` might be ambiguous.

All estimation functions take the relation as their first positional argument —
either a plain table name (`'customers'`, `'main.customers'`) or a query in
parentheses (`'(SELECT * FROM customers WHERE year = 2026)'`).

---

## Shared named parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `treatment` | VARCHAR | required | Binary treatment column |
| `outcome` | VARCHAR | required¹ | Numeric or boolean outcome column |
| `covariates` | VARCHAR[] | all others | Adjustment set. Omit to use every column except the treatment, outcome, `id`, `policy` and `exclude` |
| `exclude` | VARCHAR[] | `[]` | Columns to drop when `covariates` is not given |
| `estimator` | VARCHAR | `aipw` | See the estimator table below |
| `id` | VARCHAR | — | Column carried through to per-row output so results can be joined back |
| `treated` / `control` | VARCHAR | — | Explicit level mapping when the treatment is not already 0/1. Must be given together |
| `seed` | BIGINT | 42 | Seeds fold assignment, bootstrap and every other draw |
| `folds` | BIGINT | 5 | Cross-fitting folds |
| `bootstrap_reps` | BIGINT | 200 | Replicates for estimators without a closed-form influence function |
| `trim` | DOUBLE | 0.01 | Drop rows whose propensity falls outside `[trim, 1-trim]` |

¹ `do_balance` and `do_overlap` do not need an outcome.

### Estimators

| Name | Method | Variance |
|---|---|---|
| `aipw` | Augmented IPW, doubly robust. **Default.** | influence function |
| `dml` | Cross-fitted partially linear double ML | influence function |
| `ipw` | Hajek (self-normalised) inverse propensity weighting | influence function |
| `regression` | g-computation over per-arm outcome models | bootstrap |
| `naive` | Raw difference in means. A baseline, not a causal estimate | two-sample |
| `s_learner` | One model over `[X, T, T·X]` | — |
| `t_learner` | Separate model per arm | — |
| `x_learner` | Two-stage imputed-effect learner | — |
| `dr_learner` | Doubly-robust pseudo-outcome regression | influence function |

---

## Estimation

### `do_ate` / `do_att` / `do_atc`

One row. `do_att` targets the treated, `do_atc` the untreated.

```sql
SELECT * FROM do_ate('customers', treatment := 'discount', outcome := 'revenue',
                     covariates := ['age', 'income']);
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, p_value, n,
n_treated, n_trimmed, variance_method, warnings`.

### `do_cate`

One row per input row, with a pointwise 95% interval.

```sql
SELECT id, cate, cate_low, cate_high
FROM do_cate('customers', treatment := 'discount', outcome := 'revenue',
             id := 'customer_id');
```

Returns `row_id, id, treatment, outcome, cate, cate_low, cate_high, learner`.
Intervals are produced only by the default `dr_learner`; the S/T/X learners
report point estimates with `cate_low = cate_high` and say so in a warning.

### `do_ate_by`

One independent estimate per group. Extra parameter: `by` (VARCHAR[], required).

```sql
SELECT * FROM do_ate_by('customers', treatment := 'discount', outcome := 'revenue',
                        by := ['region', 'plan']);
```

The grouping columns come back first, as VARCHAR, followed by the `do_ate`
columns. A group too small to estimate yields a row with NULL estimates and the
reason in `warnings`, rather than failing the whole query. Capped by
`duckdo_max_groups`.

### `do_ape`

The average partial effect of a **continuous** treatment: what one more unit of the dose buys, on
average. A partially linear double-ML estimator — residualise the outcome and the dose on the
covariates, then regress one residual on the other.

```sql
SELECT * FROM do_ape('pricing', treatment := 'price', outcome := 'revenue');
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, p_value, n, dose_mean, dose_sd,
dose_min, dose_max, warnings`. The dose range is reported because an average partial effect over a
range you have not looked at is easy to misread — which is what `do_dose_response` is for.

Works on a binary treatment too, where the average partial effect coincides with the ATE.

### `do_dose_response`

The curve behind that average. Extra parameter: `grid` (BIGINT, default 20, capped at 200).

```sql
SELECT * FROM do_dose_response('pricing', treatment := 'price', outcome := 'revenue', grid := 20);
```

Returns `grid_point, dose, mu, mu_low, mu_high, n_within_decile`. `mu` is E[Y(d)] by g-computation
over a model quadratic in the dose, averaged across the covariate distribution at each point;
intervals come from the model's HC1 sandwich covariance, exact for a linear model.

Grid points sit on **dose quantiles**, not on an even spacing, so the curve never extrapolates past
the observed support. `n_within_decile` reports how many rows sit near each point — treat a grid
point with thin support as decoration.

The binary estimators (`do_ate` and friends) refuse a continuous treatment and point here.

---

## Panel data

Both functions take `unit`, `period`, `treatment` and `outcome` as named parameters. The treatment
is a per-row boolean; each unit's adoption period is derived as the first period where it is true,
and a unit that is never true is a never-treated control.

The parameter is `period`, not `time`: `TIME` is a type name, so `time := ...` is a parser error.

### `do_did`

```sql
SELECT * FROM do_did('rollout', unit := 'store_id', period := 'month',
                     treatment := 'has_feature', outcome := 'revenue');
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, n_units, n_periods, n_cohorts,
n_never_treated, pre_trend, warnings`.

Group-time average treatment effects in the Callaway–Sant'Anna sense, aggregated by cohort size.
**Two-way fixed effects is deliberately not implemented**: under staggered adoption it uses
already-treated units as controls for later adopters and can put negative weights on some cells.
The comparison group here is never-treated units where any exist, and not-yet-treated units
otherwise; `warnings` says which was used. Standard errors come from a unit-level cluster bootstrap
(`bootstrap_reps`, minimum 50).

`pre_trend` is the cohort-weighted average effect over pre-treatment periods. Under parallel trends
it should be near zero, and a value large relative to the standard error raises a warning.

### `do_event_study`

```sql
SELECT * FROM do_event_study('rollout', unit := 'store_id', period := 'month',
                             treatment := 'has_feature', outcome := 'revenue');
```

Returns `relative_period, att, std_error, ci_low, ci_high, n_treated, is_pre_treatment` — one row
per period relative to adoption, pooled across cohorts.

Relative period −1 is the reference and is **omitted** rather than reported as a zero nobody
estimated. Rows with `is_pre_treatment` are the parallel-trends evidence; if they are not flat, the
post-treatment numbers do not mean what they appear to.

---

## Diagnostics

### `do_balance`

Per encoded feature: `covariate, feature, smd_raw, smd_weighted, variance_ratio,
balanced`. `balanced` is `|smd_weighted| < 0.1`.

### `do_overlap`

Ten propensity buckets: `bucket, ps_low, ps_high, n_treated, n_control,
off_support`. `off_support` marks a bucket containing only one arm — there is no
counterfactual evidence there.

### `do_diagnose`

The whole battery: `check_name, status, detail, severity`. Checks are
`sample_size`, `treatment_prevalence`, `positivity`, `balance`,
`outcome_variation`, `missing_data`, `dimensionality`, plus one `encoding` row
per warning. `status` is `pass` / `warn` / `fail`.

### `do_refute`

Extra parameters: `method` (VARCHAR, default `placebo_treatment`), `fraction`
(DOUBLE, default 0.8, for `subset`), `strength` (DOUBLE, default 0.5, for
`unobserved_confounder`).

| method | What it does | Expectation |
|---|---|---|
| `placebo_treatment` | Permutes the treatment | Effect collapses to zero |
| `random_common_cause` | Adds an irrelevant covariate | Estimate does not move |
| `subset` | Re-estimates on a random subset | Estimate is stable |
| `bootstrap` | Re-estimates on a resample | Estimate is stable |
| `unobserved_confounder` | Simulates a confounder of the given strength | Reports the resulting shift |

Returns `method, original_estimate, refuted_estimate, difference, tolerance,
passed, detail`.

### `do_sensitivity`

How strong hidden confounding would have to be to overturn the result.

Returns `estimate, std_error, robustness_value, robustness_value_ci, e_value,
e_value_ci, interpretation`. The robustness value is the Cinelli–Hazlett partial
R² an unobserved confounder would need with *both* treatment and outcome; the
E-value is the VanderWeele–Ding risk-ratio equivalent.

---

## Graphs

### `do_graph_create(name, definition)` / `do_graph_drop(name)` / `do_graphs()`

```sql
CALL do_graph_create('sales_dag', 'digraph {
  intent [latent];
  season -> discount; season -> revenue;
  discount -> clicks; clicks -> revenue;
  intent -> discount; intent -> revenue;
}');
```

A DOT subset: `a -> b` edges, chains (`a -> b -> c`), and a `[latent]` (or
`[unobserved]`) attribute marking a node as unmeasured. Cycles are rejected —
and an invalid definition never reaches the store.

Graphs are kept in an ordinary table named `duckdo_graphs`, created on first use,
so they survive a restart of a persistent database and can be inspected, backed
up or version-controlled like any other data:

```sql
SELECT name, definition FROM duckdo_graphs;
```

Re-registering a name replaces it. The in-process map is only a parse cache in
front of that table.

### `do_identify`

Named parameters only: `graph`, `treatment`, `outcome`.

Returns one row per strategy — `backdoor`, `frontdoor`, `iv` — with `strategy,
identifiable, adjustment_set, note`. When the backdoor criterion fails, the note
names the unobserved common cause responsible.

### `do_validate`

Named parameters: `graph`, `treatment`, `outcome`, `covariates` (required).

Grades each covariate: `covariate, role, verdict, reason`.

| role | verdict | why |
|---|---|---|
| `confounder` | keep | closes a backdoor path |
| `precision variable` | keep | predicts the outcome only; tightens the interval |
| `mediator` | DROP | on a directed path; adjusting removes part of the effect |
| `post-treatment` | DROP | caused by the treatment |
| `outcome descendant` | DROP | caused by the outcome |
| `instrument` | DROP | reaches the outcome only through the treatment |
| `collider` | DROP | conditioning opens a path that was blocked |
| `latent` | DROP | declared unobserved |

### `do_dseparated(graph, x, y [, z])`

Scalar, returns BOOLEAN. `z` is an optional VARCHAR[] conditioning set.

### `do_iv`

Two-stage least squares. Extra parameter: `instrument` (VARCHAR, required).

```sql
SELECT * FROM do_iv('trial', treatment := 'took_drug', outcome := 'health',
                    instrument := 'assigned_to_drug', covariates := ['age']);
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, p_value, n, first_stage_f,
instrument, weak_instrument, warnings`.

`first_stage_f` is the instrument's robust F in the first stage; `weak_instrument` is true below the
conventional threshold of 10, and a warning says so. The exclusion restriction — that the instrument
touches the outcome only through the treatment — is an assumption no data can check, and the output
says that too. With a binary treatment this identifies the local average treatment effect among
compliers, not the population ATE; the estimand column reads `LATE` accordingly.

Standard errors use the **structural** residual, computed from the actual treatment rather than the
fitted one. Using the fitted treatment there is the classic 2SLS standard-error mistake.

### `do_frontdoor`

The front-door product formula, for a **binary** mediator. Extra parameter: `mediator` (VARCHAR,
required).

```sql
SELECT * FROM do_frontdoor('sales', treatment := 'discount', outcome := 'revenue',
                           mediator := 'clicks');
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, n, effect_t_on_m,
effect_m_on_y, mediator, warnings`.

The estimate is the product of the treatment's effect on the mediator and the mediator's effect on
the outcome, and both halves are reported because that is where a front-door estimate goes wrong.
Intervals come from a bootstrap of the whole product rather than a delta-method approximation.

Covariates are not used: this is the unconditional formula. The mediator must vary within both
treatment arms, and a mediator that does not gets an explicit error rather than a silent NaN.

### `do_mediate`

How much of an *already identified* effect travels through the mediator. `do_frontdoor` uses a
mediator to rescue identification; this one assumes identification and splits the effect up. Extra
parameter: `mediator` (VARCHAR, required).

```sql
SELECT estimand, estimate, ci_low, ci_high, proportion
FROM do_mediate('signups', treatment := 'onboarding', outcome := 'retention',
                mediator := 'activated', covariates := ['plan', 'region']);
-- total            | 3.131 | 3.096 | 3.166 | 0.515
-- natural_direct   | 1.518 | 1.469 | 1.568 | 0.515
-- natural_indirect | 1.613 | 1.568 | 1.659 | 0.515
```

Three rows, one per effect. Columns: `estimand, estimator, estimate, std_error, ci_low, ci_high,
proportion, n, tm_interaction, mediator, warnings`. `proportion` is the proportion mediated and is
the same on every row, so it can be read off whichever one you filter to.

The decomposition is the linear case of VanderWeele's, fitting

```
M = b0 + b1*T + b2'X + e
Y = q0 + q1*T + q2*M + q3*(T*M) + q4'X + e
```

and reading off `NDE = q1 + q3*(b0 + b2'E[X])` and `NIE = (q2 + q3)*b1`. Intervals come from a
row-level bootstrap that redoes both fits, so the three are mutually consistent and the total
always equals direct plus indirect exactly.

**The `T*M` interaction is what makes this mediation analysis rather than Baron-Kenny with extra
steps.** Drop `q3` and the split is only valid when the treatment's effect on the outcome does not
depend on the mediator. On a DGP with a true `q3` of 0.6, fitting without the interaction returns a
direct effect of 2.41 against a truth of 1.80 and an indirect effect of 2.22 against a truth of
2.80 — a third wrong in opposite directions — while the *total* stays correct at 4.63 either way.
A wrong split with a right total is exactly the failure that survives a sanity check, so `q3` is
carried and reported as `tm_interaction`.

`proportion` is `NULL` when the total effect is not distinguishable from zero, with a warning
saying so: it is a ratio, and dividing by a total that straddles zero produces a number that can
take any value. When direct and indirect have opposite signs the proportion falls outside `[0, 1]`
and a warning names that too.

Passing the mediator in `covariates :=` is an error, not a silent zero — adjusting for the mediator
in the outcome model removes the very path being measured.

Cost is bootstrap-bound: `bootstrap_reps` replicates times two ridge fits. 200k rows by 5
covariates takes 3.7 s; 1M by 50 takes 194 s. Lower `bootstrap_reps :=` to trade interval
precision for time — the point estimates do not depend on it.

Every result carries the assumption this rests on: **sequential ignorability**. Randomising the
treatment does not buy it. It removes confounding of treatment-outcome and treatment-mediator, and
leaves mediator-outcome confounding entirely untouched — which is the most common error in applied
mediation, so it is in the output rather than in a footnote.

---

## Interventions

### `do_counterfactual`

Per row: `row_id, id, treatment, observed, y0, y1, effect, effect_low,
effect_high`.

### `do_predict`

Extra parameter: `intervention` (STRUCT or MAP, required).

```sql
SELECT * FROM do_predict('customers', treatment := 'discount', outcome := 'revenue',
                         intervention := {'price': 19.99});
```

Forces the named columns to the given values **for every row** — that is what
makes it `do(X = x)` rather than a filter on `X = x` — then predicts the outcome.
Intervening on the treatment column is allowed and gives `E[Y | do(T = t), X]`.

Returns `row_id, id, intervention, observed, predicted, predicted_low,
predicted_high`. Intervals are NULL for binary outcomes.

### `do_policy_value`

Extra parameters: `policy` (VARCHAR — names a boolean column in the relation) or
`threshold` (DOUBLE, default 0.0 — treat when the estimated effect exceeds it).

```sql
SELECT * FROM do_policy_value('(SELECT *, tenure > 12 AS my_rule FROM customers)',
       treatment := 'discount', outcome := 'revenue', policy := 'my_rule');
```

Returns `policy, n_targeted, share_targeted, policy_value, std_error, ci_low,
ci_high, value_treat_all, value_treat_none, lift_over_treat_all`. The value is
measured against treating nobody; a negative `lift_over_treat_all` means the rule
is worse than treating everyone.

### `do_uplift`

Twenty rows tracing the Qini curve: `bucket, fraction_targeted, n_targeted,
cumulative_gain, random_gain, qini`. Rows are ranked by estimated effect,
best first.

### `do_optimal_policy`

Extra parameter: `depth` (BIGINT, default 2, capped at 3).

A shallow, deployable targeting rule found by greedily maximising the
doubly-robust value. Returns one row per leaf: `leaf, rule, n, mean_effect,
std_error, action, expected_gain`. Thresholds in `rule` are reported in the
column's own units, not the standardised space the model works in.

---

## Causal foundation models

Opt-in at build time (`-DDUCKDO_WITH_ONNX=ON`); without it every entry point below still exists
and says specifically that the build lacks ONNX Runtime, rather than falling back to a classical
estimator while still claiming to be the model.

`-DDUCKDO_WITH_ONNX=ON` fetches the pinned ONNX Runtime release for the target platform and, on
Windows, stages its DLLs next to the shell and test binaries. Three knobs adjust that:
`-DDUCKDO_ONNXRUNTIME_ROOT=<dir>` uses a release you have already unpacked, `-DDUCKDO_ORT_URL=<url>`
points at a mirror for networks where GitHub releases are unreachable, and `-DDUCKDO_ORT_VERSION`
moves the pin.

### `do_list_models()` / `do_models()`

`model, setting, license, commercial, attribution_required, max_features, context_ladder,
available, detail`. `detail` says why a model is unavailable and what to do about it.

| model | setting | licence | covariates | context |
|---|---|---|---|---|
| `causalpfn` | backdoor (ignorability) | Apache-2.0 | 99 | dynamic, to 4096 |
| `do_pfn` | non-identifiable prior | CC BY 4.0 (attribution required) | 5 | fixed ladder: 128 / 512 / 1024 / 2048 |

**Start with `causalpfn`.** It targets the backdoor setting directly, accepts twenty times the
covariates, has no fixed context ladder, carries no licence obligation, and does not shrink the
population effect the way `do_pfn` does.

### `do_devices()`

`device, available`. Reports `cpu` when the build has ONNX Runtime; CUDA/ROCm/MLX are listed as
unavailable placeholders.

### Using a model

Pass `model :=` to `do_ate`, `do_att`, `do_atc` or `do_cate`. It replaces `estimator :=` — the SQL
and the returned columns are otherwise identical.

```sql
SET duckdo_model_dir = '~/.cache/duckdo';
SELECT estimator, estimate, variance_method
FROM do_ate('customers', treatment := 'discount', outcome := 'revenue', model := 'do_pfn');
```

What the runtime does, and reports in `warnings`:

- **Covariate budget.** Where the model's budget binds (5 for `do_pfn`, 99 for `causalpfn`), the
  covariates most correlated with the outcome are kept and the rest named as dropped.
- **Context selection.** `causalpfn` takes up to 4096 rows directly; `do_pfn` uses the largest
  ladder rung below the row count. Either way the context is sampled seeded and stratified, so the
  treatment proportions are preserved and a repeat run feeds the model identical rows.
- **Outcome scaling.** `do_pfn`'s bar distribution lives in standardised outcome space, so the
  context outcome is standardised and the effect scaled back. `causalpfn` standardises per arm
  internally and returns the outcome on its own scale.
- **The intervention.** Query rows are run twice, forced to treated and to control. That is
  `do(T = t)` for every row, not a filter on rows where `T` happened to equal `t`.

### Intervals: `ensemble :=`

With a single forward pass, `do_ate` reports `variance_method = 'effect dispersion (no model
uncertainty)'` and `do_cate` returns `cate_low = cate_high = cate`. Neither is a calibrated
interval, and both say so — because a single pass cannot see how much the answer depends on which
rows landed in the model's context.

`ensemble := k` (or `SET duckdo_ensemble_draws = k`) takes k bootstrapped context draws and folds
the spread across them into the interval. The population standard error becomes
`sqrt(sampling_var + between_draw_var)`, and `do_cate` gets a real per-row interval.

Measured on a DGP with a true effect of 3.0: the single-draw interval is `[3.035, 3.048]`, which
excludes the truth; `ensemble := 8` gives `[3.007, 3.121]`, which contains it. Per-row intervals
covered the truth for 88% of rows — still short of nominal, because the model's own weights are
fixed and their uncertainty is not measured.

Each draw is a full forward pass, so the cost is linear in `k`. If every draw comes back identical —
which happens when the table already fits inside the model's context window and there is nothing to
resample — the ensemble is dropped and a warning says so, rather than reporting an interval that
measured nothing.

Measured on a heterogeneous DGP with true ATE 2.9806: `aipw` 3.0628, `causalpfn` 3.0610,
`do_pfn` 2.6307. Do-PFN's shrinkage is a documented property of its model class, not a bug in the
plumbing — but it is why `causalpfn` is the one to reach for.

### Exporting the graphs

The extension ships no weights. Produce them yourself:

```sh
pip install causalpfn
python scripts/export/export_causalpfn.py --out ~/.cache/duckdo

git clone https://github.com/jr2021/Do-PFN            # optional, second model
python scripts/export/export_dopfn.py --repo ./Do-PFN --out ~/.cache/duckdo
```

Do-PFN writes four weight-free graphs plus one shared weight blob, and fails unless every graph
reproduces PyTorch to within 1e-4 at two different query lengths. CausalPFN writes one graph with
both axes dynamic, and is gated on the estimand instead — the ATE must agree with PyTorch to 0.01
with a CATE correlation above 0.999 — because attention-kernel differences cost ~1e-2 of
logit-level agreement without moving the number anyone reads.

---

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `duckdo_default_estimator` | `aipw` | Used when `estimator :=` is omitted |
| `duckdo_max_rows` | 1000000 | Refuse frames with more rows than this |
| `duckdo_max_features` | 500 | Refuse encodings wider than this |
| `duckdo_max_memory` | half of `memory_limit` | Ceiling on the encoded matrix, as a memory string such as `'4GB'`. `'-1'` removes it |
| `duckdo_max_categorical_levels` | 32 | Drop categoricals with more levels, with a warning |
| `duckdo_max_groups` | 1000 | Cap on `do_ate_by` groups |
| `duckdo_seed` | 42 | Global seed |
| `duckdo_bootstrap_reps` | 200 | Bootstrap replicates |
| `duckdo_model_dir` | `~/.cache/duckdo` | Where exported model graphs and weights live |
| `duckdo_threads` | 0 | Threads for model inference and the dense accumulations inside every estimator; 0 means one per hardware thread |
| `duckdo_query_chunk` | 512 | Rows scored per model forward pass |
| `duckdo_ensemble_draws` | 1 | Context draws a foundation model takes; more than one funds an interval |

Every guardrail names the setting to raise when it trips, rather than silently
truncating. `duckdo_max_memory` also names the row count that *would* fit at the
frame's width, so the fix can be pasted rather than derived:

```
duckdo: encoding customers would need 389.0 MiB for the 1000000 x 51 feature
matrix, above duckdo_max_memory (95.3 MiB). Raise duckdo_max_memory, pass a
shorter covariates := list, or sample the input - about 245098 rows fit at this
width, e.g. '(SELECT * FROM customers USING SAMPLE 245098 ROWS)'
```

The matrix is n x p doubles and every estimator indexes it by row, so it is
resident for the whole query — 1M rows by 50 covariates is 400 MB, and the
measured peak while building it is 882 MB. That is why the default budget is
half of DuckDB's `memory_limit` rather than all of it.
