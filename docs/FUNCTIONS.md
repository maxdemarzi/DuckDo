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
| `covariates` | VARCHAR[] | all others | Adjustment set. Omit to use every column except the treatment, outcome, `id`, `cluster`, `policy` and `exclude` |
| `exclude` | VARCHAR[] | `[]` | Columns to drop when `covariates` is not given |
| `estimator` | VARCHAR | `aipw` | See the estimator table below |
| `id` | VARCHAR | — | Column carried through to per-row output so results can be joined back. When its values repeat, a warning says so: repeated ids usually mean rows are not independent units |
| `cluster` | VARCHAR | — | `do_ate`, `do_att`, `do_atc` and `do_ate_by` only. The column naming the independent unit when a row is not one, as after joining customers to their orders. See [Clustered rows](#clustered-rows-effects-across-a-one-to-many-join) |
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

### Clustered rows: effects across a one-to-many join

```sql
SELECT estimate, std_error, variance_method
FROM do_ate('(SELECT * FROM customers JOIN orders USING (customer_id))',
            treatment := 'discount', outcome := 'revenue', covariates := ['tenure'],
            cluster := 'customer_id');
```

Every estimator treats rows as independent units. A one-to-many join breaks that without
saying so. Join 4,000 customers to three orders each and you get 12,000 rows, but still only
4,000 independent units. Treated as independent, those rows give the same estimate with a
standard error of 0.0203, against 0.0352 for the customers themselves. The interval is 42% too
narrow, and nothing in the result says so. With `cluster := 'customer_id'`, the joined rows give
back 1.978 and 0.0352, the customers' own answer to every printed digit.

`cluster :=` names the unit, and three things then follow it:

- **Folds.** All of a cluster's rows go to one fold, stratified by the cluster's majority arm.
  Split across folds, a unit's rows would let the nuisance models see the rows they are scored
  on.
- **Standard errors.** Each row's influence contribution is summed within its cluster before
  squaring, with a G/(G−1) correction. This covers the influence function (`aipw`, `ipw`,
  `dr_learner`), the DML sandwich and `naive`'s two-sample formula. With every row its own
  cluster, it reduces to the ordinary standard error.
- **The bootstrap.** Estimators with bootstrap intervals resample whole clusters.

`variance_method` names the column and the cluster count. Clusters are numbered in content
order, so the result still does not depend on row order. NULL cluster values are refused, and
so are fewer than 10 clusters. Below 50 clusters a warning says the interval is optimistic.

`cluster :=` is accepted only by `do_ate`, `do_att`, `do_atc` and `do_ate_by`. Every other
function rejects it rather than ignoring it, because an interval that looks clustered and is not
is worse than none.

DuckDo cannot detect the problem on its own: a join leaves no trace in the rows. The one sign it
can see is a repeated `id :=`, and any function given one warns when its values repeat.

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

### `do_ate_levels`

Average treatment effects for a treatment with **more than two levels**, each contrasted against a
reference level.

```sql
SELECT level, reference, estimate, ci_low, ci_high, naive_difference
FROM do_ate_levels('trial', treatment := 'arm', outcome := 'y', reference := 'placebo');
```

| Parameter | Default | |
|---|---|---|
| `reference` | the first level | the level every other level is contrasted against |

The common parameters apply as well. Levels sort in their natural order, so 2 comes before 10, and
there can be up to 20. Past that, the column is a dose (use `do_ape`) or needs bucketing. Every level
needs at least 5 rows, and fewer than 30 draws a warning.

Returns one row per non-reference level: `level, reference, estimand, estimator, estimate,
std_error, ci_low, ci_high, p_value, naive_difference, n_level, n_reference, warnings`.

The estimator is a multi-arm AIPW. It fits an outcome surface for each level and one-vs-rest
propensities normalised to sum to one, both cross-fitted with folds stratified within every level.
Each contrast is the mean of ψ_k − ψ_ref, with an influence-function interval. Only `aipw` is
available; naming any other estimator is refused.

**Why not `do_ate(..., treated := 'B', control := 'A')`?** That works too, but it estimates the
effect among the units that received A or B. Take a three-arm DGP where B goes to high-x units and
its effect grows with x. There the pairwise route returns 2.467, on the subpopulation's truth of
2.464. The population contrast is 1.5, and `do_ate_levels` returns 1.472 [1.375, 1.568].

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

**`covariates := [...]` makes every cell doubly robust** (Sant'Anna and Zhao, 2020). Plain DiD
assumes the treated cohort and its comparison group would have trended in parallel. With covariates,
that is assumed only among units that are alike on them. Each group-time cell then combines two
working models: a regression of the outcome's change on the covariates in the comparison group, and
a propensity model for being in the cohort rather than the comparison. The cell is consistent if
either model is right.

- Covariates are read at each unit's first observed period, so they are pre-treatment by
  construction.
- They must be numeric or boolean. NULLs are replaced by the column mean, with a warning.
- The unit, period, treatment and outcome columns are refused as covariates.
- The `estimator` column reads `callaway-santanna, doubly robust`, and the bootstrap refits both
  models on every draw.

`do_event_study` takes the same `covariates :=` and estimates every relative period the same way.
That turns its pre-periods into a check of *conditional* parallel trends. Take a panel where units
with high `x` trend faster and are also likelier to be treated, with a true effect of 2.0: plain
DiD gives 3.877, and `covariates := ['x']` gives 1.841 [1.600, 2.082].

Without covariates the output is unchanged to the last bit. The parameter used to be accepted and
silently ignored; it now does what it says.

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

### `do_synth`

```sql
SELECT unit, estimate, pre_rmspe, rmspe_ratio, p_value, weights
FROM do_synth('regions', unit := 'region', period := 'quarter',
              treatment := 'policy', outcome := 'sales');
```

Returns one row per treated unit: `unit, adoption_period, estimand, estimator, estimate, pre_rmspe,
post_rmspe, rmspe_ratio, p_value, n_donors, n_pre_periods, n_post_periods, weights, warnings`.

For each treated unit this fits a synthetic control (Abadie, Diamond and Hainmueller 2010):
non-negative weights over the never-treated units, summing to one, that best reproduce the treated
unit's outcome before treatment. `estimate` is the mean gap between the unit and its synthetic
control after treatment. `weights` is a `MAP` from donor to weight, largest first, holding every
donor above 1e-4, so `weights['store_07']` works.

- **Donors** are never-treated units observed in every period. Units missing any period are left
  out, with a warning, and at least two donors are needed.
- **Treated units** need at least two pre-treatment periods and every period observed. Units that
  fail this are skipped, with a warning. Each treated unit is fitted on its own.
- **Inference is by in-space placebos.** Each donor is treated as if it had been, at the same
  period, and matched from the other donors. `p_value` is the share of runs, counting the real
  unit's, whose post/pre RMSPE ratio is at least the real unit's. With J donors the smallest
  attainable value is 1/(J+1), and a warning states it. Past 100 donors, a seeded subset of 100 is
  used.
- **There is no intercept.** A treated unit whose outcome lies outside every donor's cannot be
  matched by any convex combination, and a warning counts the pre-periods where that happens. Read
  `pre_rmspe` before `estimate`: a synthetic control that did not track the unit beforehand says
  nothing about afterwards.
- **There are no covariates.** Matching on every pre-treatment outcome already absorbs what
  covariates predict about them. `covariates :=` is refused rather than ignored. For conditional
  parallel trends, use `do_did(..., covariates := [...])`.

The weights solve a simplex-constrained least-squares problem: accelerated projected gradient, then
an exact solve on the support. `scripts/synth_check.py` grades the solver against scipy's SLSQP,
and the two agree to 1.6e-8 in the weights.

### `do_synth_path`

```sql
SELECT period, actual, synthetic, gap, is_pre_treatment
FROM do_synth_path('regions', unit := 'region', period := 'quarter',
                   treatment := 'policy', outcome := 'sales');
```

Returns `unit, period, relative_period, actual, synthetic, gap, is_pre_treatment`: one row per
treated unit and period, from the same fit as `do_synth`. This is the plot to look at. The gap
should hover near zero before treatment and open afterwards.

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

Comments (`//`, `#` and `/* ... */`) are skipped. An undirected edge `a -- b` is an
error, not a silently dropped edge: DuckDo graphs are directed, so pick a direction.

### `do_discover` / `do_discover_dot`

```sql
SELECT source, edge, target, in_graph, stability, orientation_stability
FROM do_discover('measurements', alpha := 0.01, bootstrap := 50);

SELECT dot FROM do_discover_dot('measurements');
```

These functions propose a graph from data. Treat the output as a draft to review, not an answer.

`do_discover` returns `source, target, edge, in_graph, stability, orientation_stability, warnings`,
with one row per pair of variables that is joined in the full-data graph or in at least 25% of
bootstrap resamples. `edge` is `->` when the data orients it, `--` when it cannot, and `absent` for a
pair that only the resamples joined. `stability` is the share of resamples containing the edge.
`orientation_stability` is the share that gave it this same orientation, or left it unoriented when
`edge` is `--`.

The algorithm is PC-stable (Colombo and Maathuis 2014), with Fisher-z tests of partial correlation.
It orients v-structures and then applies Meek's rules. The result is a CPDAG. Observational data
identifies a graph only up to its Markov equivalence class, so some edges have no direction the data
can supply: `x -> y` and `y -> x` fit a two-variable world equally well. The "stable" variant makes
the skeleton independent of column order. The bootstrap draws rows in content order, so it does not
depend on row order either.

- **Columns.** The default is every numeric or boolean column. Use `columns := [...]` to choose, or
  `exclude := [...]` to leave some out. At most 30 variables are allowed, because the number of
  tests, and the chance that one of them errs, grows combinatorially. Rows with a NULL in a
  selected column are dropped, with a warning.
- **`alpha`** (default 0.01) is the level of each independence test. **`max_conditioning`**
  (default 3) caps the size of the conditioning sets. **`bootstrap`** (default 50) sets the number
  of resamples, and `0` turns them off, with a warning. **`seed`** follows `duckdo_seed`.
- **Assumptions**, restated in `warnings`: no hidden common cause of any two variables,
  faithfulness, and linear-Gaussian dependence. Real data usually breaks the first. When it does,
  the orientations can be wrong even where every edge is right.
- **The stabilities lean pessimistic.** A resample carries the sample's own error on top of its
  own, so its independence tests reject more often than `alpha`, and resamples join pairs that
  are truly independent. All orientations flow from colliders, so one spurious edge that erases a
  collider unorients everything downstream of it. On a world where the full-data graph is exactly
  right, orientation stability comes out at 0.88 at `alpha := 0.01` and 0.98 at `0.001`.

`do_discover_dot` returns `dot, n_nodes, n_directed, n_undirected`. The proposal is DOT with each
edge's bootstrap stability as a comment, and **`do_graph_create` refuses it twice over**:

1. The DOT contains a line marked `do_discover: unreviewed`. Until that line is deleted,
   `do_graph_create` refuses it.
2. Every edge the data could not orient is written `a -- b`, and `do_graph_create` refuses those
   until someone writes `a -> b` or `b -> a`.

That makes the review step required rather than suggested. A graph registered with
`do_graph_create` is what `do_identify` and `do_validate` treat as true, so an edge that was never
reviewed would become a wrong adjustment set.

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

Cost is bootstrap-bound: `bootstrap_reps` replicates times two ridge fits, run in parallel across
replicates. 200k rows by 5 covariates takes 1.6 s; 1M by 50 takes 141 s, where the limit is memory
bandwidth rather than cores — each replicate reads the whole design matrix in random row order.
Lower `bootstrap_reps :=` to trade interval precision for time; the point estimates do not depend
on it.

Every result carries the assumption this rests on: **sequential ignorability**. Randomising the
treatment does not buy it. It removes confounding of treatment-outcome and treatment-mediator, and
leaves mediator-outcome confounding entirely untouched — which is the most common error in applied
mediation, so it is in the output rather than in a footnote.

### `do_msm`

Treatment that varies over time, when a confounder varies with it — and is itself affected by it.
Named parameters: `unit`, `period`, `treatment`, `outcome` (all required), `covariates` (the
time-varying confounders), `baseline` (a subset of `covariates` that is time-invariant), and
`truncate`.

```sql
SELECT estimate, ci_low, ci_high, mean_weight, max_weight, effective_n
FROM do_msm('patient_months', unit := 'patient_id', period := 'month',
            treatment := 'on_drug', outcome := 'bp', covariates := ['creatinine']);
-- 2.130 | 1.962 | 2.298 | 0.989 | 81.33 | 6245
```

Returns one row: `estimand, estimator, estimate, std_error, ci_low, ci_high, n_units, n_periods,
mean_weight, max_weight, effective_n, warnings`. The estimate is the effect of **one additional
treated period**.

**Why this needs its own function.** Take a confounder `L` measured each period, which affects
treatment, affects the outcome, and is itself affected by *earlier* treatment. Leave `L` out of the
adjustment set and it confounds. Put it in and you block the part of the earlier treatment's effect
that travels through it. No covariate list is correct. On a DGP built exactly that way, whose true
effect per treated period is 2.0:

| approach | estimate |
|---|---|
| no adjustment | 3.92 |
| adjust for every measured confounder | 3.47 |
| **`do_msm`** | **2.13** |

The middle row is the trap: it faithfully recovers the outcome model's coefficient on treatment
(3.5) and that coefficient is not the causal effect. `do_msm` stops adjusting and reweights
instead, building a pseudo-population in which treatment at each period is independent of the
history that predicted it, then fits the outcome on cumulative treated periods in that population.

Weights are **stabilised** — `P(A_t | A_{t-1}, V) / P(A_t | A_{t-1}, L_t, V)`, with both models
pooled across periods and the period index as a feature. `mean_weight` should sit near 1; a mean
away from 1 means the treatment model is misspecified, and the result says so rather than leaving
you to notice.

**`truncate` looks like it helps and does not.** On the same data, `truncate := 0.01` raises
`effective_n` from 6,245 to 12,101 and cuts the interval by more than half — while moving the
estimate from 2.13, whose interval covers the true 2.0, to 2.51, whose interval excludes it. Every
number that looks like a quality signal improves while the answer gets worse. It is off by default,
and using it adds a warning saying what it traded.

A marginal structural model buys **nothing** against unmeasured confounding. What it buys is
correct handling of measured confounders that the treatment itself affects — which no amount of
covariate adjustment can do. The structural model is linear in cumulative treated periods, so it
assumes every period is worth the same and that only the total matters, not when it happened.

### `do_rmst`

Time-to-event outcomes. Named parameters: `duration` (observed follow-up time) and `event`
(1 where the event happened, 0 where the subject was censored), both required, plus `horizon`.

```sql
SELECT estimate, ci_low, ci_high, horizon, rmst_treated, rmst_control, censored_fraction
FROM do_rmst('trial', treatment := 'arm', duration := 'days_followed', event := 'relapsed',
             covariates := ['age', 'stage']);
-- 0.904 | 0.851 | 0.957 | 5.0 | 3.218 | 2.315 | 0.328
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, horizon, rmst_treated,
rmst_control, n, n_events, censored_fraction, warnings`. The estimate is the difference in
**restricted mean survival time**: how much longer, on average, a treated subject stays event-free
within the first `horizon` units of the clock.

**There is deliberately no hazard ratio.** A hazard ratio compares subjects still at risk at each
moment, and treatment changes who is still at risk — so after the first events the two risk sets
are no longer comparable, even under perfect randomisation. That is not a bug adjustment can fix;
it is what the estimand means. RMST has no such problem, and it comes out in units a
non-statistician can act on.

Confounding is handled by inverse-probability weighting the two Kaplan–Meier curves rather than by
modelling the hazard, which keeps the estimand marginal. On an exponential DGP whose closed-form
truth is 0.9239, `do_rmst` returns 0.904 with an interval covering it, while the same call with
`covariates := []` returns 0.443 — the confounded value, which here is less than half the real
benefit.

**`horizon` is part of the estimand, not a display option.** A different horizon is a different
quantity. The default is the last time *both* arms still had at least 5% of their subjects at
risk. The obvious alternative — the last time both arms were observed at all — is a trap: on
20,000 subjects it lands where a handful of people remain, and the restricted mean then integrates
over a stretch of curve that is almost pure noise. A `horizon` past the end of the data is allowed
and warns that it is extrapolating flat.

Censoring is assumed independent of the event time given the covariates. Nothing in the data can
check that, and it fails exactly when subjects leave because they are getting worse.

### `do_frame_summary`

What the encoder did, one row per encoded feature. Takes the same arguments as `do_ate`.

```sql
SELECT feature, source, kind, level, center, scale, n_rows_with_missing
FROM do_frame_summary('customers', treatment := 'discount', outcome := 'revenue');
-- tenure          | tenure | numeric            | NULL  | 2.995 | 1.794 | 80
-- tenure__missing | tenure | missing_indicator  | NULL  | 0.200 | 0.401 | 80
-- colour=green    | colour | one_hot            | green | 0.333 | 0.472 | 80
-- colour=red      | colour | one_hot            | red   | 0.335 | 0.473 | 80
```

Every estimator runs on this frame, so which columns were adjusted for, which categoricals became
one-hot columns and which reference level was dropped, where NULLs were filled and how many rows
that touched, and what standardisation the reported coefficients are on, should all be answerable
without reading the source. `n`, `n_features`, `n_treated`, `n_rows_with_missing`,
`treated_label`, `control_label` and `warnings` repeat on every row so one call answers both
"what is in here" and "what became of column X".

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

Extra parameters: `depth` (BIGINT, default 2, capped at 3) and `threshold` (DOUBLE,
default 0) — **the cost of treating one row**, on the outcome's scale.

A shallow, deployable targeting rule found by greedily maximising the doubly-robust
value net of that cost. Returns one row per leaf: `leaf, rule, n, mean_effect,
std_error, cost, action, expected_gain`. Thresholds in `rule` are reported in the
column's own units, not the standardised space the model works in, and `mean_effect`
is the effect itself so it can be compared against `cost` directly.

```sql
SELECT rule, n, round(mean_effect,2) AS mean_effect, cost, action
FROM do_optimal_policy('customers', treatment := 'got_discount', outcome := 'revenue',
                       exclude := ['customer_id'], depth := 1, threshold := 8.0);
-- tenure_months <= 20 | 20552 | 10.99 | 8.0 | treat
-- tenure_months > 20  | 19448 |  4.93 | 8.0 | do not treat
```

**Leave `threshold` at zero and the answer is usually "treat everyone"** — and that is
correct, because a free intervention with a positive effect should go to everybody. The
question only becomes interesting once treating costs something. On the DGP above the
true break-even is at twenty months' tenure, which the tree recovers from the data.

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

When a model's covariate budget binds, `ensemble := k` also varies **which** covariates each draw
uses — sampled without replacement, weighted by outcome correlation — so the interval covers the
choice of covariates rather than treating it as free. Where the budget does not bind this is a
no-op, since every draw gets every covariate.

That widening is honest but it is not a repair. On twelve contributing covariates with a true
effect of 3.0, against Do-PFN's budget of five, the interval goes from ±0.006 to ±0.31 and still
does not cover the truth — because a correctly specified AIPW restricted to those same five
covariates returns 4.07. **Dropped confounders are bias, and no resampling interval covers bias.**
When more covariates are dropped than kept, the result says so and points at a model whose budget
fits; CausalPFN returns 2.984 on the same data.

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

### `do_download(model, source := ..., overwrite := false)`

Copies every artifact the catalog names for `model` (graphs, weights and manifest) from `source`
into `duckdo_model_dir`, and returns one row per file: `model, file, bytes, sha256, status, source`.

```sql
SELECT file, bytes, status FROM do_download('causalpfn', source := '/mnt/shared/duckdo-models');
```

- `source` is a directory or a URL, and there is **no default**. No hosted copy of the weights
  exists, and a URL you did not type is a network request you did not make. A URL is read through
  DuckDB's file system, so it needs DuckDB's `httpfs` extension; DuckDo itself opens no connection.
- Each file is written to a `.part` file, hashed on the way, and renamed only once complete, so an
  interrupted copy never looks like a model to `do_list_models()`.
- Files already present are kept, with their hash reported, unless you pass `overwrite := true`.
- `sha256` is the digest of the bytes that landed. Compare it against a checksum you trust from
  wherever the files came from. DuckDo does not pin checksums, because there is no canonical copy to
  pin them to yet.

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
| `duckdo_query_chunk` | 512 | Rows scored per model forward pass. The estimate is bit-identical at every setting. The context is encoded once and cached, so this is now a small dial: 8,000 rows take 31.3 s at 512 (1.9 GB peak), 23.0 s at 2048 (2.7 GB) and 22.2 s at 8192 (5.0 GB), medians of three runs. 2048 is worth it if the memory is spare; past that there is nothing left to buy |
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
