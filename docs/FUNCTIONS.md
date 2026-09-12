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
| `cluster` | VARCHAR | — | The column naming the independent unit when a row is not one, as after joining customers to their orders. Accepted by every estimator that reports an interval, and by the diagnostics. `do_msm` and the panel functions take `unit :=` instead. See [Clustered rows](#clustered-rows-effects-across-a-one-to-many-join) |
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

Groups are estimated from one scan. The relation is copied once into a private
in-memory database, sorted by group, and each group reads only its own rows.
1,000 groups of 200 rows take 0.88 s, against 9.4 s when each group re-read the
whole relation, and the estimates are bit-identical to that older path. Groups run
one after another, and each fit uses every thread for its own work. The copy lasts
only for the call, but it is held in memory, so a very large relation needs room
for a second copy while it runs.

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

`cluster :=` works the same way in every estimator that reports an interval. Each one clusters
through whichever variance method it uses:

| how the interval is made | functions |
|---|---|
| influence function, summed within clusters | `do_ate`, `do_att`, `do_atc`, `do_ate_by`, `do_ate_levels`, `do_ape`, `do_policy_value`, `do_optimal_policy` |
| cluster-robust (CR1) sandwich | `do_cate`, `do_dose_response`, `do_iv` (both stages), `do_predict`, `do_counterfactual` |
| bootstrap of whole clusters | the bootstrap estimators behind `do_ate`, `do_frontdoor`, `do_mediate`, `do_rmst`, `do_rmtl` |
| resampling whole clusters | `do_refute`'s placebo, subset and bootstrap refuters |
| through the clustered estimate | `do_sensitivity`, and `do_refute`'s tolerance |
| folds and counts only, no interval reported | `do_uplift`, `do_balance`, `do_overlap`, `do_diagnose` |

Measured on 3,000 rows joined to three copies each: unclustered, every interval narrows to about
0.58 of the rows' own, which is 1/sqrt(3). Clustered on the row id, the closed-form intervals come
back to 1.00 of it (0.99 for `do_predict`), and the bootstrap ones to between 1.00 and 1.05.
Without `cluster :=`, every result is bit-identical to what it was before clustering existed.

The diagnostics accept it too. `do_refute`'s placebo permutes treatment across clusters, its
subset refuter keeps or drops whole clusters, and its bootstrap draws whole clusters, counting a
cluster drawn twice as two. The standard errors its tolerance is built from are the clustered
ones. `do_sensitivity` builds its robustness value and E-value from the clustered
interval, and `do_diagnose` counts clusters, not rows, as the sample. `do_frame_summary` reports
no uncertainty and refuses the parameter. `do_msm` and the panel functions take the unit as
`unit :=` already.

DuckDo cannot detect the problem on its own: a join leaves no trace in the rows. The one sign it
can see is a repeated `id :=`, and any function given one warns when its values repeat.

### `do_ate_pool`

```sql
-- Each site runs this on its own data and publishes the one row it returns:
SELECT estimand, estimate, std_error, n, n_treated
FROM do_ate('local_patients', treatment := 'drug', outcome := 'recovery');

-- Anyone holding the published rows pools them:
SELECT estimate, ci_low, ci_high, q_p_value, i_squared
FROM do_ate_pool('site_results');
```

This combines effects estimated at separate sites, without moving any rows between them. It
returns `estimand, estimate, std_error, ci_low, ci_high, p_value, n_sites, n, q, q_p_value,
i_squared, warnings`. The input is a table with one row per site. The columns `do_ate` and
`do_ate_by` return already fit it, so `do_ate_pool('(SELECT * FROM do_ate_by(...))')` works as it
stands. Use `estimate :=`, `std_error :=` and `n :=` to map other column names.

The result is the effect over every site's population taken together. An AIPW estimate is the
mean of per-row influence values, so the pooled estimate is exactly the size-weighted mean of the
site estimates, and its variance is exactly Σ(wₛ/W)²·seₛ². The weight is the site's `n` for an
ATE, its `n_treated` for an ATT, and its control count for an ATC. A table that mixes estimands
is refused. A table with no `estimand` column is treated as ATEs, with a warning.

- **Inverse-variance weighting is used only for the heterogeneity test.** It is the
  meta-analysis default, but it weights sites by precision rather than by population, so a small
  site with a quiet outcome can outvote the rest. That gives the effect in no population anyone
  asked about.
- **Heterogeneity** is reported as Cochran's Q, its chi-square p-value and I². When the sites
  clearly disagree, a warning says the pooled number averages different effects. It is still the
  effect across all the sites, but report the sites as well. Below five sites, a warning says
  the test has little power.
- **Rows with a NULL estimate** are left out and counted in `warnings`. `do_ate_by` reports those
  for a group it could not estimate.
- **Assumptions:** the sites are independent samples, and no unit appears at two of them. Models
  are fitted per site, so each site's estimate has to stand on its own data.

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

Without covariates the output is unchanged to the last bit.

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

Whether weighting actually made the arms comparable. Takes `treatment` and `covariates`, and no
outcome: balance is a property of the treatment model alone.

```sql
SELECT * FROM do_balance('customers', treatment := 'discount',
                         covariates := ['age', 'income', 'tenure']);
```

Per encoded feature: `covariate, feature, smd_raw, smd_weighted, variance_ratio,
balanced`. `smd_raw` is the standardised mean difference between the arms as the rows stand;
`smd_weighted` is the same after inverse-probability weighting, and it is the one that matters,
because the weighted population is the one the estimate describes. `balanced` is
`|smd_weighted| < 0.1`, the usual rule of thumb. A feature that starts unbalanced and stays
unbalanced is one the propensity model could not fix: the arms barely overlap on it, and
reweighting cannot invent rows that are not there.

### `do_overlap`

Where the two arms share support, which is what positivity means in practice. Takes `treatment`
and `covariates`, and no outcome.

```sql
SELECT * FROM do_overlap('customers', treatment := 'discount', covariates := ['age', 'income']);
```

Ten propensity buckets: `bucket, ps_low, ps_high, n_treated, n_control,
off_support`. `off_support` marks a bucket holding only one arm — there is no counterfactual
evidence there, so whatever the estimator reports for those rows comes from the model's shape
rather than from a comparison. A few off-support rows are normal. A bucket full of them means the
question is answerable only for part of your population, and `trim :=` decides what becomes of the
rest.

### `do_diagnose`

Every check at once, as rows, so a dbt test or a CI query can assert on them. Takes the same
arguments as `do_ate`. `outcome` is optional, and the checks that need one are skipped without it.

```sql
SELECT check_name, status, severity, detail
FROM do_diagnose('customers', treatment := 'discount', outcome := 'revenue',
                 covariates := ['age', 'income', 'tenure']);
```

The whole battery: `check_name, status, detail, severity`. Checks are
`sample_size`, `treatment_prevalence`, `positivity`, `balance`,
`outcome_variation`, `missing_data`, `dimensionality`, plus one `encoding` row
per warning. `status` is `pass` / `warn` / `fail`, and `detail` carries the number behind the
verdict and the setting that governs it. A `fail` is not a refusal to estimate; it is the reason
to distrust the estimate you will get.

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

`passed` compares against two standard errors, and which standard error depends on what the
method expects. The placebo expects zero, so it is judged by its own: `|refuted_estimate|` against
twice the standard error of the estimate on permuted data, which is the noisier of the two (0.051
against 0.040 on the example above). Judging it by the original's turned a nominal 5% false alarm
into 12% over 40 seeds. The other methods expect the estimate not to move, so they are judged by
the original's: `|difference|` against twice its standard error. Both rules are tests at about the
5% level, so an occasional `false` on sound data is the rule working, not a defect — read the
estimate and the tolerance beside it.

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

Each returns one row. `do_graph_create`: `name, n_nodes, n_edges, n_latent` — worth reading back,
since a typo in a node name makes a new node rather than an error. `do_graph_drop`:
`name, dropped`, where `dropped` is false when no graph of that name was registered. `do_graphs`:
one row per registered graph, `name, n_nodes, n_edges, nodes`, with `nodes` a `VARCHAR[]`.

### `do_discover` / `do_discover_dot`

```sql
SELECT source, edge, target, in_graph, stability, orientation_stability
FROM do_discover('measurements', alpha := 0.01, bootstrap := 50);

SELECT source, edge, target, agreement
FROM do_discover('measurements', algorithm := 'both',
                 tiers := [['age', 'sex'], ['discount'], ['revenue']]);

SELECT dot FROM do_discover_dot('measurements');
```

These functions propose a graph from data. Treat the output as a draft to review, not an answer.

`do_discover` returns `source, target, edge, in_graph, stability, orientation_stability, warnings`,
with one row per pair of variables that is joined in the full-data graph or in at least 25% of
bootstrap resamples. `edge` is `->` when the data orients it, `--` when it cannot, and `absent` for a
pair that only the resamples joined. `stability` is the share of resamples containing the edge.
`orientation_stability` is the share that gave it this same orientation, or left it unoriented when
`edge` is `--`.

With `algorithm := 'both'` there is one more column, `agreement`, described below.

The default algorithm is PC-stable (Colombo and Maathuis 2014), with Fisher-z tests of partial
correlation.
It orients v-structures and then applies Meek's rules. The result is a CPDAG. Observational data
identifies a graph only up to its Markov equivalence class, so some edges have no direction the data
can supply: `x -> y` and `y -> x` fit a two-variable world equally well. The "stable" variant makes
the skeleton independent of column order. The bootstrap draws rows in content order, so it does not
depend on row order either.

**`algorithm := 'fci'`** drops the assumption PC leans on most: that nothing unmeasured causes two
of the measured variables. FCI (Spirtes, Glymour and Scheines; Zhang 2008) returns a partial
ancestral graph, where each end of an edge is an arrowhead, a tail, or a circle meaning the data did
not decide. `edge` then reads:

- `-->`: a cause;
- `<->`: neither causes the other, and something hidden causes both;
- `o->`: the target does not cause the source; the source causes the target, a hidden cause links
  them, or both;
- `o-o`: undecided.

It starts from PC's skeleton, removes the edges a hidden common cause can fake with a
possible-d-sep search, and orients with Zhang's rules R1-R4 and R8-R10, assuming no selection bias
(R5-R7 exist only for selection bias). Under that assumption the orientation is complete: a circle
that remains is one the data cannot settle. R9 and R10 follow uncovered potentially directed paths
rather than triangles. Each path search stops after a fixed number of steps, and a search that
stops finds nothing, so on a very large, dense graph an edge can keep a circle a complete search
would have removed. It then goes to review rather than past it. `orientation_stability` is the
share of resamples that give the pair the same two marks. Tails found by R9 and R10 depend on a
longer stretch of the graph being right, so they tend to be less stable than the colliders they
start from: on the five-variable world above, 0.72 against 0.8 before those rules existed.

Take a -> b <- u -> c <- d, with u unmeasured. PC finds a collider at b and another at c, claiming
the b-c edge in opposite directions, and warns that they conflict. FCI returns `a o-> b`,
`b <-> c` and `d o-> c`, naming the hidden cause. On a world with no hidden cause, it invents no
`<->`.

**`algorithm := 'lingam'`** orients the edges PC has no way to settle. Conditional independence
cannot tell `x -> y` from `y -> x`, because the two fit a two-variable world equally well, which is
why the `--` edges exist at all. DirectLiNGAM (Shimizu, Inazumi, Sogawa, Hyvärinen, Kawahara,
Washio, Hoyer and Bollen 2011) reads a different signal. If every effect is linear, the graph has
no cycles, nothing unmeasured causes two variables, and the disturbances are non-Gaussian, then
regressing the effect on the cause leaves a residual independent of the cause while the reverse
regression does not. The most exogenous variable is the one no other variable explains in that
sense; it is peeled off, regressed out of the rest, and the search repeats on the residuals. What
comes back is a causal order — reported in `warnings` — and then one regression of each variable on
its predecessors. Every edge is directed, so `do_discover_dot`'s output needs only its review
marker deleted, not a direction invented for each undecided pair.

That is bought with an assumption list strictly longer than PC's, and the last item is the one that
bites. On Gaussian disturbances the asymmetry is not weaker but absent, and the method returns a
confident order that is close to a coin flip: over 40 draws of a six-variable graph, 40 orders
exactly right with uniform disturbances against 1 with Gaussian ones. So each variable's
disturbance is tested with Jarque–Bera, and **two that cannot be told from Gaussian is a refusal**,
because identifiability allows at most one. The refusal tracks the failure rather than guessing at
it: at 1000 rows and above it fired on 0 of 40 draws for uniform, heavy-tailed, mildly skewed and
near-Gaussian-mixture disturbances, and on 40 of 40 for Gaussian ones. At 200 rows it fired far
more often — 18 to 39 of 40 for those three mildly non-Gaussian cases — but the order was
unreliable there too, exactly right only 15 to 25 times of 40, so it is refusing cases it should.
Strongly non-Gaussian uniform disturbances at the same 200 rows were refused once in 40, with the
order right 39 times. `test := 'rank'` and `test := 'mixed'` are rejected at
bind time with this algorithm, since normal scores are Gaussian by construction and erase the only
signal it reads.

**`algorithm := 'resit'`** buys the same orientations with a different trade: the effect may bend
however it likes, so long as the disturbance is *added* rather than mixed in. A continuous
additive noise model `x = f(parents) + e` is identifiable for a nonlinear `f` and — the part that
matters next to LiNGAM — for **any** disturbance distribution, Gaussian included. So RESIT covers
exactly the case LiNGAM refuses.

RESIT (Peters, Mooij, Janzing and Schölkopf, JMLR 2014) runs LiNGAM's search from the other end.
LiNGAM peels the most exogenous variable; RESIT peels a *sink* — the variable whose residual, once
every other remaining variable is regressed out of it, is least dependent on them. The regression
is kernel ridge regression on the same random Fourier features `test := 'kernel'` uses, and
"least dependent" is the squared cross-covariance of their features. Then the order is pruned:
a predecessor is a parent only if the pair is still dependent given every other predecessor, which
is the kernel test again. `min_effect` does not apply — there are no coefficients to threshold —
and `test :=` is refused, because RESIT is a kernel method already.

Measured on a four-variable world (`a -> b -> c` with `a -> d`) over 15 draws, as oriented true
edges out of 3 and false edges per run:

| world | LiNGAM | RESIT |
|---|---|---|
| straight links, non-Gaussian | 1.00 true, 0.00–0.07 false | 1.00 true, 0.00–0.07 false |
| straight links, Gaussian | refused | refused |
| bent links, non-Gaussian | 0.58–0.64 true, 1.07–1.27 false | **1.00 true, 0.00–0.20 false** |
| bent links, Gaussian | 0.00–0.02 true, 2.40–2.47 false | **1.00 true, 0.00–0.07 false** |

RESIT is at least as good as LiNGAM everywhere both run, and the only reason to prefer LiNGAM is
speed: about 0.01 s against 0.76 s on those runs. **Read the bent-Gaussian row carefully**, because
it is a caveat on LiNGAM rather than a point for RESIT: LiNGAM does not refuse there, even though
its disturbances are Gaussian. Its check tests what is left after a *linear* fit, so a bend it
cannot model lands in the residual and reads as the non-Gaussianity it is looking for. It then
reports a confident causal order that is wrong.

RESIT has its own refusal, and it is the same corner from the other side: **when no fit needs more
than a straight line and two or more disturbances cannot be told from Gaussian**, it stops. Linear
effects with Gaussian disturbances is the textbook unidentifiable case, and RESIT does not fail
quietly in it — in the simulation above it returned 0.47 of 3 true edges and 2.53 false ones with
no sign anything was wrong. A fit counts as bending when the kernel regression leaves at least 5%
less of the target's variance than a straight line does, and `warnings` reports how many did.

**`algorithm := 'rcd'`** drops the assumption every functional method above leans on: that
nothing unmeasured causes two of the measured variables. It rests on one clean fact — if i causes
j and nothing hidden links them, regressing j on i leaves a residual independent of i — so it tests
both directions of every dependent pair and reads the answer off which of them hold:

| both directions tested | reported as | meaning |
|---|---|---|
| exactly one leaves an independent residual | `-->` | that is the cause, and the pair is clean |
| neither does | `<->` | no unconfounded model fits: a hidden common cause |
| both do | `o-o` | the data admits either direction |

The repetition in RCD's name (Maeda and Shimizu, AISTATS 2020) is the outer loop: once i is known
to cause j, later tests regress it out first, which uncovers relations that were masked. The output
is a partial ancestral graph, so it reuses FCI's rows and FCI's proposal — a `<->` becomes a node
marked `[latent]` that `do_graph_create` and `do_identify` already treat as a hidden common cause.

Take a and c sharing a hidden cause, with b also causing c. LiNGAM has to explain the pair with a
direct edge and gets the rest of the graph wrong doing it, returning `b -> a`, `b -> c`, `c -> a`.
FCI finds the skeleton but will not commit: `a o-> c` still allows a to cause c. RCD returns
`a <-> c` and `b --> c`.

**It needs no refusal of its own**, and that is the nicest thing about it. Where the disturbances
are Gaussian, both directions leave an independent residual and every pair comes back `o-o` — it
says it cannot tell, in its own output vocabulary, which is exactly what LiNGAM and RESIT have no
way to express and why they have to refuse instead. A run that is mostly `o-o` is that, not a
finding.

Two measured limits. **It does not cry wolf**: on a four-variable world with no hidden cause it
invented 0.00 `<->` per run over 40 runs, and returned the exact true graph 19/20 at 2,000 rows and
20/20 at 6,000 — fully oriented, which PC on the same world cannot manage. But **detection is a
band, not a threshold**. With the hidden cause contributing this share of the variance of each of
the two variables it links, over 20 runs it was named:

| share | 2,000 rows | 6,000 rows |
|---|---|---|
| 30% | 0/20 | 1/20 |
| 50% | 3/20 | 6/20 |
| 70% | 12/20 | 15/20 |
| 90% | 0/20 | 0/20 |

The fall-off at 90% is not a bug and is worth understanding: when the hidden cause is almost all of
what a variable is, that variable is a stand-in for it, and "a causes c" is very nearly the true
model. RCD reports `a --> c`, which is close to right for the wrong reason. Throughout the sweep the
rest of the graph was unaffected — `b --> c` was still found 18-20 times out of 20 — which is the
practical point: a confounded pair is quarantined rather than poisoning everything around it.

One interaction with `tiers` to know about. A tier that contradicts the data does not quietly win
here. Told that c cannot cause d when it does, RCD finds that neither allowed direction leaves an
independent residual and reports `c <-> d`. That is the honest answer to its own question — no
unconfounded model fits what you have told me — and a `<->` that appears only after a tier was
added is a sign the tier is wrong.

**`algorithm := 'pc+lingam'`** (also spelled `'both'`, its name before RESIT existed) and
**`algorithm := 'pc+resit'`** run PC-stable over the same rows and report the union, with an extra
`agreement` column saying how the two landed on each pair:

| `agreement` | meaning |
|---|---|
| `both` | both found the edge and gave it the same direction |
| `oriented by lingam` | both found it; PC could not orient it, the functional method did |
| `conflict` | both found it and oriented it opposite ways, so `edge` is `--` |
| `pc only` | only PC's skeleton had it |
| `lingam only` | only the functional method had it |

Where PC oriented an edge its orientation stands, because it rests on the weaker assumptions.
Where PC left one undirected, the functional method's direction fills it in. Where the two point opposite ways the
edge goes back to `--` and to review, because two methods contradicting each other is not evidence
for either answer. A `conflict` is worth reading closely rather than resolving: both methods assume
nothing unmeasured causes two variables, so a hidden common cause breaks both, and it breaks them
differently. On a -> c <- b with a and c also sharing a hidden cause, PC reads the v-structure off
the skeleton while LiNGAM puts the confounded pair in an order of its own, and `both` returns
`a -- c  conflict` alongside an edge `b -> a` that LiNGAM invented and PC never saw.

- **Columns.** The default is every numeric or boolean column. Use `columns := [...]` to choose, or
  `exclude := [...]` to leave some out. At most 30 variables are allowed, because the number of
  tests, and the chance that one of them errs, grows combinatorially. Rows with a NULL in a
  selected column are dropped, with a warning.
- **`alpha`** (default 0.01) is the level of each independence test. **`max_conditioning`**
  (default 3) caps the size of the conditioning sets. **`bootstrap`** (default 50) sets the number
  of resamples, and `0` turns them off, with a warning. **`seed`** follows `duckdo_seed`.
- **`tiers`** states an order you already know, and settles the edges that cross it without any
  appeal to the data: `tiers := [['age', 'sex'], ['discount'], ['revenue']]` says nothing in a
  later group causes anything in an earlier one. That is how most undirected edges get settled in
  practice — not by a cleverer test, but by someone saying demographics precede the campaign. A
  column named in no group is unconstrained, so a tier can be given for the pair you are sure
  about without inventing an order for the rest, and `warnings` names the columns left free. The
  tiers are applied before any test runs and nothing can overturn them, so a tier that is wrong
  makes a wrong graph that looks well-supported; when a v-structure wanted an orientation the
  tiers forbid, the tiers win and `warnings` says how often. With `algorithm := 'fci'` a tier
  becomes an arrowhead at the later end and nothing more: `g o-> h` says h does not cause g while
  still allowing a hidden common cause of the two, because a tier is not a claim that there is
  none. With `algorithm := 'lingam'` it restricts which variable may be peeled next.
- **`groups`** (`algorithm := 'lingam'` only) names a column splitting the rows into datasets
  that share one causal order but need not share coefficients — sites, regions, cohorts,
  experiment arms. MultiGroupDirectLiNGAM (Shimizu 2012) peels them together, each weighted by its
  own row count, which is what constrains them to one order; coefficients are then fitted inside
  each group, and an edge is reported when at least half of them find it. Requiring all would
  contradict the premise that the coefficients differ; requiring one would let the noisiest group
  write the graph.

  The point is that this is not the same as pooling the rows. Take a chain x → y → z observed at
  four sites with the sign of every effect flipped at two of them: pooled, the correlation of x
  and y is under 0.1 and LiNGAM recovers neither edge; told which rows belong together it recovers
  both. Over 40 draws that is 40/40 against **3/40**. On a denser five-variable world with
  independently drawn per-group coefficients, the joint order was exactly right 25/25 at 120, 200
  and 400 rows per group, where ignoring the labels managed 13/25, 11/25 and 18/25.

  The group column is a label rather than a variable, so it never joins the graph even when it is
  numeric. Resampling draws inside each group, so a replicate cannot lose a dataset. Each group
  needs comfortably more rows than variables and is named if it does not have them, at most 50
  groups are allowed, and the disturbance check runs inside each group rather than pooled —
  a pooled residual carries the differences between the groups' coefficients, which looks Gaussian
  and refused data every group could read on its own. It is refused with `'pc+lingam'` on purpose:
  PC would run on the pooled rows and know nothing of the groups, which is the thing `groups` is
  for avoiding.
- **`min_effect`** (default 0.05, `algorithm := 'lingam'` or `'pc+lingam'` only) is the smallest
  coefficient LiNGAM counts as an edge, as a share of the effect's own standard deviation: a
  one-standard-deviation move in the cause has to shift the effect by at least this much. A
  coefficient also has to clear a Wald test at `alpha`, which matters at small samples the way
  `min_effect` matters at large ones, where anything is significant. In simulation over 40
  six-variable graphs the edge set was fully recovered at every setting up to 0.10, with the false
  positive rate falling from 0.007 at 0 to 0.000 at 0.10; 0.05 is the default because the
  simulation's weakest true effect was 0.4 and a real one can be smaller. Passing it to `'pc'`,
  `'fci'` or `'resit'` is an error rather than a silent no-op.
- **`test`** (default `'pearson'`) chooses the independence test. `'rank'` replaces each column
  by its normal scores, the inverse normal CDF of rank / (n + 1) with ties sharing their average
  rank, and runs the same Fisher-z tests on those (the nonparanormal of Liu, Lafferty and
  Wasserman 2009). It then assumes only a Gaussian copula: that *some* monotone transform of each
  variable makes them jointly Gaussian. Log-scale, skewed and heavy-tailed columns qualify, and it
  never needs to know which transform. Take a linear-Gaussian chain a -> c <- b, c -> d -> f,
  observed as `exp(1.5*a)`, `b^3`, `exp(c)`, `d + d^3` and `sinh(2*f)`. Pearson returns six
  edges, two of them spurious and one reversed; `'rank'` returns the true four. Transform the
  columns again, monotonically, and `'rank'` returns the same output bit for bit. Where the data
  really is Gaussian, both find the same graph. Ranks mean little in a column with many ties, so
  any column with fewer than 10 distinct values is named in `warnings`.
- **`test := 'mixed'`** is for binary and mixed data. It reads each two-valued column as the
  threshold of a latent Gaussian variable, and every other column as a monotone transform of one:
  the latent Gaussian copula for mixed data of Fan, Liu, Ning and Zou (2017). Each pair's latent
  correlation comes from Kendall's tau through a bridge function for the pair's kinds. Fisher's z
  would be wrong on those: a binary column's latent correlations are far noisier than a continuous
  column's. In simulation, a nominal 1% Fisher test rejected true independences between two
  binary columns 9-29% of the time, and 5-8% where one side was continuous. So each partial
  correlation gets its own variance, from every row's influence on the correlations it is built
  from, and a Wald test. In the same simulation that held the level: 0-2% at 1% and 4-7% at 5%,
  across marginal and conditional tests of binary and continuous columns. Take the chain
  a -> c <- b, c -> d -> f with a, c and f observed only as yes/no, b cubed and d on a log scale.
  Pearson returns nine edges, five of them spurious and every one of those at stability 1.0,
  because conditioning on a yes/no c is not conditioning on the c the others respond to. `'rank'`
  returns six. `'mixed'` returns the true four. A column with 3 to 9 distinct values is read as
  **ordinal**: a latent Gaussian variable cut at thresholds, with a polychoric correlation against
  another discrete column and a polyserial one against a continuous column, each with its own row
  influences for the Wald test. Reading such a column as continuous is what failed before. When two
  variables were independent given an ordinal one, the test rejected that 94–98% of the time at a
  nominal 1% in simulation, because a coarse reading of the middle variable cannot screen off its
  neighbours. Read as ordinal, the same tests rejected 0.5–1.0% at 1% and 3.5–5.5% at 5%, over 600
  draws of each case. On the same chain with c cut into 4 levels and d into 5, the mixed test now
  returns the true four edges, where reading them as continuous added a->d, b->d and c->f.
  `warnings` names the columns read as binary and as ordinal. The
  row influences cost memory, one number per pair of columns per row, so more than 2^24 of them is
  refused with a suggestion to test a sample. Both algorithms take every test.
- **`test := 'kernel'`** assumes nothing about the shape of the dependence, and is the one to
  reach for when an effect might bend. Every other test here reads dependence through a
  correlation, and `'rank'` widens that only to *monotone* transforms — no monotone transform
  straightens a parabola. Take a true chain x -> y -> z with y = x², measured on 8,000 rows: the
  correlation of x and y is 0.009 while the correlation of |x| and y is 0.920, and `'pearson'`,
  `'rank'` and `'mixed'` all return `y -- z` alone, dropping the strongest dependence in the data.
  The reverse error is worse, because it invents rather than omits: where x and y are both
  parabolas in z and genuinely independent given it, Fisher's z rejects that true conditional
  independence **100% of the time**, so the spurious edge arrives at full stability.

  The method is RCoT (Strobl, Zhang and Visweswaran 2019), which approximates the kernel test
  KCIT (Zhang, Peters, Janzing and Schölkopf 2011) with random Fourier features: a few hundred
  numbers per row stand in for an n × n kernel matrix, which is what makes it linear in the rows
  rather than quadratic. The features of x and y are residualised on the features of the
  conditioning set, and the statistic is the squared cross-covariance of what is left. Its null is
  a weighted sum of chi-squares, approximated by the three-moment method of Liu, Tang and Zhang
  (2009); the obvious two-moment gamma is anticonservative in exactly the tail that decides edges,
  rejecting true independences 3–10% of the time at a nominal 1%.

  Measured end to end, as how often PC joins a pair it should not: 0.00–0.04 across four null
  cases at 1,000 to 8,000 rows, against 1.00 for both `'pearson'` and `'rank'` on the bent one.
  Power on a bent dependence is 1.00 where they manage 0.04–0.06. On data that really is linear it
  returns the same graph as `'pearson'`, so choosing it is not a trade.

  Two costs, both real. It is **about 100× slower**: on seven variables and 20,000 rows, 1.3 s
  against 0.012 s without resampling, and 13.6 s against 0.012 s with the default 50 resamples.
  Lower `bootstrap` if that matters. And its power falls away when the conditioning set nearly
  determines one of the pair, because the bend then has to be found in a small remainder: on the
  chain above, recovery of x -- y over 40 draws was 40/40 while z took 81% or less of y, and
  38–40/40 at 90%. It reads at most 2,000 rows, taken by stride in content order, and the random
  features are drawn once from `duckdo_seed` and reused by every resample, so the stabilities
  measure the sample rather than the draw.
- **Assumptions**, restated in `warnings`: no hidden common cause of any two variables (PC only),
  faithfulness, and linear-Gaussian dependence, or a Gaussian copula with `test := 'rank'`, or a
  latent Gaussian copula with `test := 'mixed'`, or nothing about the shape of the dependence at
  all with `test := 'kernel'`. Real
  data usually breaks the first. When it does, the orientations can be wrong even where every edge
  is right.
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

With `algorithm := 'fci'`, `do_discover_dot` also returns `n_bidirected`, and writes each kind of
edge differently:

- `-->` becomes `->`.
- `<->` becomes a node marked `[latent]` with an arrow into each end. `do_graph_create` and
  `do_identify` already treat that as a hidden common cause, so confounding found by discovery
  reaches identification.
- Every edge with a circle is written `--`, with a comment naming the readings its marks allow.
  `do_graph_create` refuses it until a person settles it: a direction, a hidden common cause, or
  both.

### `do_identify`

What the graph says you may estimate, and with which columns. Named parameters only: `graph`,
`treatment`, `outcome`. It reads the registered graph and no data.

```sql
SELECT strategy, identifiable, adjustment_set, note
FROM do_identify(graph := 'sales_dag', treatment := 'discount', outcome := 'revenue');
```

Returns one row per strategy — `backdoor`, `frontdoor`, `iv` — with `strategy,
identifiable, adjustment_set, note`. When the backdoor criterion fails, the note names the
unobserved common cause responsible, which is the part worth reading: it says which variable you
would have to measure for the ordinary estimators to be legitimate here. `adjustment_set` is what
to pass as `covariates :=`; for the other two strategies it names the mediator or the instrument
to hand to `do_frontdoor` or `do_iv`.

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

Scalar, returns BOOLEAN. `z` is an optional VARCHAR[] conditioning set. It reads a registered
graph and nothing else: no table, no rows.

**Does the graph imply that `x` and `y` are independent once `z` is held fixed?** True when every
path between them is blocked, false when at least one stays open. This is the primitive under
`do_identify` and `do_validate` — an adjustment set is valid exactly when it blocks every backdoor
path from the treatment to the outcome — exposed so you can put your own questions to the graph.

```sql
CALL do_graph_create('sales_dag', 'digraph {
  intent [latent];
  season -> discount;  season -> revenue;
  discount -> clicks;  clicks -> revenue;
  intent -> discount;  intent -> revenue;
  coupon_mail -> discount;
}');

SELECT do_dseparated('sales_dag', 'season', 'coupon_mail');                -- true
SELECT do_dseparated('sales_dag', 'season', 'coupon_mail', ['discount']);  -- false
```

Those two lines are the whole point. Nothing links the season to the coupon mailing, so the graph
says they are independent — and then conditioning on `discount` makes them dependent. Both cause
it, and among customers who got the discount, learning that it was peak season explains the
discount away, so the mailing becomes less likely. **Adding a variable to an adjustment set can
create the association you were adjusting to remove.**

Which is why a path is not the same as a dependence:

| the path through `m` | carries dependence when |
|---|---|
| chain, `x -> m -> y` | `m` is **not** in `z` |
| fork, `x <- m -> y` | `m` is **not** in `z` |
| collider, `x -> m <- y` | `m` **or one of its descendants** is in `z` |

Blocking one path is not enough; every path has to be blocked. Conditioning on `discount` closes
the chain from the mailing to revenue and opens two collider paths at once, so the mailing and
revenue stay dependent until the rest are closed too:

```sql
SELECT do_dseparated('sales_dag', 'coupon_mail', 'revenue');                     -- false: open chain
SELECT do_dseparated('sales_dag', 'coupon_mail', 'revenue', ['discount']);       -- false: collider opened
SELECT do_dseparated('sales_dag', 'coupon_mail', 'revenue',
                     ['discount', 'intent', 'season']);                          -- true
```

**Why `d`-separated and not `separated`.** d-separation is the standard name for this criterion
(Pearl), and the `d` is for *directional*: separation that reads the direction of the arrows.
Ordinary graph separation asks whether a path exists between two nodes, which would make the
season and the coupon mailing "connected" through `discount` and stop there. The question worth
asking is whether a path carries dependence, and the answer turns on direction — the collider row
in the table above is the entire difference. Keeping the term means what DuckDo returns is what
the textbooks, and every other causal library, call d-separation.

Every d-separation is also a prediction you can test: if the graph says `x` and `y` are
independent given `z` and your rows disagree, the graph is wrong. That is the check `do_discover`
automates in the other direction, reading a graph off the data instead of grading one.

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
time-varying confounders), `baseline` (a subset of `covariates` that is time-invariant),
`truncate`, and `model` (`'cumulative'`, the default, `'by_period'` or `'saturated'`).

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

**`model := 'by_period'`** drops that assumption. It fits one effect for each period in the same
weighted pseudo-population, and returns a row for each, plus a row for always treated against
never: their sum, with its variance from the fit's whole covariance. On the DGP above both periods
are worth 2.0, and it returns 2.073 [1.904, 2.243] and 2.193 [1.993, 2.392], and 4.266
[3.927, 4.605] against 4.0. Make the second period worth 1.0 instead and it returns 2.073 and
1.193, and 3.266 against 3.0. The cumulative model, asked the same question, answers 1.655
[1.483, 1.826] per treated period, an average of the two whose interval covers neither. It still
assumes no interaction: treatment in one period does not change what treatment in another does.
It needs every unit observed once in every period.

**`model := 'saturated'`** drops that last assumption too. It gives every treatment history its
own mean, and reports each against never treated, for up to 6 periods (64 histories). Where the
periods simply add, it returns 2.154 for period 1 only, 2.296 for period 2 only and 4.221 for
both, against 2.0, 2.0 and 4.0. Add an interaction worth 1.5 when both periods are treated, and it
returns 5.721 for both against 5.5, and still 2.154 for period 1 alone. `by_period` then reports
2.603 [2.445, 2.760] as the effect of period 1: an interval that covers no regime anyone could
follow. The price is that each history's mean rests only on the units who followed it. Histories
that no unit followed, or fewer than ten did, are named in `warnings`.

**The interval is too narrow when the weights are heavy.** It treats the weights as known. Over
300 simulated panels like the one above, where the largest stabilised weight is typically around
100, nominal 95% intervals covered 85–93% of the time, in every model. A bootstrap that resamples
whole units and refits both weight models did no better. With weights under about 10, the same
intervals covered 97–99%. The estimates were unbiased in both cases. `max_weight` and
`effective_n` are there to be read before the interval is.

### `do_msm_rmst`

Survival when the treatment changes over follow-up and a confounder changes with it. It reads
the panel `do_msm` reads, with `event` in place of `outcome`: one row per unit per period, from
the first period until the unit's event or its last observation. Named parameters: `unit`,
`period`, `treatment`, `event` (all required), `covariates`, `horizon` (in periods, counted from
the first), `bootstrap_reps` and `seed`.

```sql
SELECT estimate, ci_low, ci_high, horizon, rmst_always, rmst_never
FROM do_msm_rmst('patient_months', unit := 'patient_id', period := 'month',
                 treatment := 'on_drug', event := 'died', covariates := ['creatinine']);
-- 0.456 | 0.410 | 0.502 | 4 | 3.689 | 3.233
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, horizon, rmst_always,
rmst_never, survival_always, survival_never, n_units, n_periods, n_events, n_dropouts,
followers_always, followers_never, max_weight, warnings`. The estimate compares two regimes,
treated in every period and treated in none, on **expected event-free periods within `horizon`**.
`survival_*` is each regime's survival through the last of those periods.

It needs no structural model. Each unit is copied into both regimes and counts toward one only
while its treatment matches it: clone, censor, weight. Every person-period it contributes is
weighted by the inverse probability of the treatment history that kept it on the regime, and of
not having dropped out before then. Both come from pooled logistic models of the measured
history, with the period as a linear term. A weighted Kaplan–Meier curve per regime then gives the
survival that regime would have produced.

On a ten-period world where a confounder raises both the event and further treatment, and is
lowered by earlier treatment, the truths come from simulating a million units under each regime:

| horizon | `do_msm_rmst` returns | the truth |
|---|---|---|
| 4 (the default here) | 0.456, interval 0.410 to 0.502 | 0.439 |
| 10 | 2.106, interval 1.836 to 2.375 | 2.298 |
| 10, without `covariates` | 1.267 | 2.298 |

Comparing the units who happened to stay on each regime, without weights, gives about 1.4. On
this world the dropout model changes little (in simulation, 2.35 without it against 2.31 with it),
though dropout depends on the confounder. On a million units per sample the estimator lands
within about 1% of the truth, and the true propensities give the same answer as the fitted ones.

**Staying on a regime gets rarer every period**, and the weights say so. `max_weight` here is
about 3,800, on a person-period of someone who stayed untreated when the model expected
treatment, and the result warns about it. The default horizon is the last period in which both
regimes still had 5% of the units following them. Here that is 4 of 10.

The interval resamples whole units and refits both weight models, so it accounts for the weights
being estimated. `do_msm`'s interval does not. Over 86 simulated panels of 5,000 units from the world above, the 95% interval covered the truth 93% of the time at horizons 4 and 10: close to nominal, and slightly narrow. Its standard error averaged 0.046 against a true spread of 0.048 at horizon 4, and 0.30 against 0.34 at horizon 10. 14 more panels had a period in which no unit was still following one of the regimes, and `do_msm_rmst` refuses a horizon that runs into such a period. The assumptions are those of any inverse-probability
method: no unmeasured confounder of treatment and the event at any period, and dropout that
depends only on the measured history.

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

### `do_rmtl`

Competing risks: an event from one cause ends the chance of an event from another. The same
parameters as `do_rmst`, plus `cause`. `event` holds a code per subject: 0 where the subject was
censored, and the cause's code where an event was observed.

```sql
SELECT estimate, ci_low, ci_high, rmtl_treated, rmtl_control
FROM do_rmtl('patients', treatment := 'drug', duration := 'years_followed', event := 'status',
             cause := 1, covariates := ['age', 'stage'], horizon := 5.0);
-- -0.769 | -0.825 | -0.712 | 1.230 | 1.999
```

Returns `estimand, estimator, estimate, std_error, ci_low, ci_high, horizon, cause, rmtl_treated,
rmtl_control, incidence_treated, incidence_control, n, n_events, n_competing, censored_fraction,
warnings`. The estimate is the difference in **restricted mean time lost** to `cause` within
`horizon`: the area under that cause's cumulative incidence curve. A negative estimate means
treated subjects spend less of the horizon having had it. `incidence_treated` and
`incidence_control` are each arm's cumulative incidence at the horizon.

**An event from another cause is not censoring.** A subject who died of something else can no
longer die of this, and the curve, a weighted Aalen–Johansen estimate, counts them that way.
Treating them as censored and taking one minus Kaplan–Meier is the common mistake. It estimates
the incidence in a world where the other causes had been abolished, a different and larger
number.

The test world has two causes with constant cause-specific hazards, and a confounder driving both
treatment and the hazards, so every truth is closed-form. At a horizon of 5:

| | `do_rmtl` returns | the truth |
|---|---|---|
| time lost to cause 1, adjusted | −0.769, interval −0.825 to −0.712 | −0.767 |
| the same, unadjusted | −0.521 | −0.514, the confounded value |
| cause 2's events censored instead | −0.665 | −0.662, a world without cause 2 |

**Read one cause beside the others.** In that world treatment barely changes how long people
stay event-free: `do_rmst` returns −0.018. It moves them from cause 1 to cause 2, whose time lost
rises by 0.787. The numbers agree exactly, because the horizon minus RMST is the sum of every
cause's time lost. A treatment can cut the time lost to one cause just by leaving people exposed
to another, and the warnings say so.

The horizon rule, the weighting, the bootstrap and `cluster` work as in `do_rmst`, and so does
the refusal to report a hazard ratio, cause-specific or subdistribution. Event codes must be
non-negative integers, and both arms need at least one event of `cause`. On the same data,
`do_rmst` counts every nonzero code as the event, and warns that it is doing so when it sees codes
above 1.

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

Both arms for every row: what the model predicts with the treatment off and on. Takes the same
arguments as `do_ate`, plus `id :=` to carry your own key into the output.

```sql
SELECT * FROM do_counterfactual('customers', treatment := 'discount', outcome := 'revenue',
                                covariates := ['age', 'tenure'], id := 'customer_id');
```

Per row: `row_id, id, treatment, observed, y0, y1, effect, effect_low,
effect_high`. `observed` is the outcome the row actually has, while `y0` and `y1` are the
predictions under each arm — so one of them is a counterfactual and the other a fitted value.
`effect` is `y1 - y0`, carrying the same pointwise interval `do_cate` reports. The average of
`effect` is the ATE; a single row's effect is the model's structure applied to that row, and is
only as good as that model.

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

Whether targeting by estimated effect beats treating at random, as a curve you can plot. Takes the
same arguments as `do_ate`.

```sql
SELECT * FROM do_uplift('customers', treatment := 'discount', outcome := 'revenue',
                        covariates := ['age', 'tenure']);
```

Twenty rows tracing the Qini curve: `bucket, fraction_targeted, n_targeted,
cumulative_gain, random_gain, qini`. Rows are ranked by estimated effect, best first, so
`fraction_targeted` runs from 5% to 100%. `cumulative_gain` is the effect accumulated by treating
that share in that order, `random_gain` the straight line from treating the same number at random,
and `qini` the gap between them. A curve that hugs the line says the ranking carries no
information, which is worth knowing before building a targeting rule on it. `do_optimal_policy`
turns a ranking that does carry information into a rule you can deploy.

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

`model, setting, license, commercial, attribution_required, max_covariates, context_ladder,
available, detail`. `detail` says why a model is unavailable and what to do about it.

| model | setting | licence | covariates | context |
|---|---|---|---|---|
| `causalpfn` | backdoor (ignorability) | CausalPFN License 1.0 (Apache-2.0's terms) | 99 | dynamic, to 4096 |
| `do_pfn` | non-identifiable prior | none stated upstream: all rights reserved | 5 | fixed ladder: 128 / 512 / 1024 / 2048 |

**Start with `causalpfn`.** It targets the backdoor setting directly, accepts twenty times the
covariates, has no fixed context ladder, puts no licence obligation on your results, and does not
shrink the population effect the way `do_pfn` does.

The licences are reported as they stand in each upstream repository, not as they are usually
described. CausalPFN's `LICENSE` is titled "CausalPFN License, Version 1.0". Its terms are
Apache-2.0's, but that is not its name. Do-PFN's repository has no licence file, and its README
states no terms. Without a licence, every right is reserved, so `do_pfn` is marked non-commercial,
every result that uses it says so, and DuckDo never redistributes it. Whether you may use it is
between you and its authors.

### `do_download(model, source := ..., overwrite := false, verify := ...)`

Copies every artifact the catalog names for `model` (graphs, weights and manifest) from `source`
into `duckdo_model_dir`, and returns one row per file: `model, file, bytes, sha256, status, source`.

```sql
SELECT file, bytes, status FROM do_download('causalpfn');                  -- the hosted release
SELECT file, bytes, status FROM do_download('causalpfn', source := '/mnt/shared/duckdo-models');
```

- **With no `source`, `causalpfn` comes from DuckDo's hosted release,**
  [maxdemarzi/duckdo-causalpfn](https://huggingface.co/maxdemarzi/duckdo-causalpfn) on Hugging
  Face. The files are published there with the upstream licence, a `NOTICE` naming every changed
  file, and the authors' citation. The URL is pinned to one commit, so the bytes behind it cannot
  change under a released DuckDo, and every file is checked against a SHA-256 compiled into the
  extension before it is installed.
- **`do_pfn` has no hosted copy.** Its upstream states no licence, so its weights cannot be
  redistributed. Export them yourself and pass `source :=`.
- **`source`** is a directory or a URL. A URL is read through DuckDB's file system, so it needs
  DuckDB's `httpfs` extension. DuckDo itself opens no connection.
- **`verify`** checks each file against the pinned digests before installing it. It defaults to
  true exactly when no `source` is given. An export you ran yourself has different bytes and
  nothing to be checked against. A file that fails is never installed, and a file already present
  that differs from the release is refused.
- Each file is written to a `.part` file, hashed on the way, and renamed only once complete, so an
  interrupted copy never looks like a model to `do_list_models()`.
- Files already present are kept, with their hash reported, unless you pass `overwrite := true`.
- `sha256` is the digest of the bytes that landed.

### `do_devices()`

`device, available, detail`. `detail` says why a device is or is not available, rather than
leaving a bare `false`.

For CUDA, `do_devices()` attaches the provider for real rather than trusting ONNX Runtime's list.
The provider library, CUDA and cuDNN only load when a session asks for them, so asking is the
test. When it fails, `detail` names what is missing: a DLL, or a version mismatch.

**Running CausalPFN on an NVIDIA GPU.** Build with `-DDUCKDO_ORT_FLAVOUR=cuda12`, put CUDA 12.8 and
cuDNN 9 on the library path, and:

```sql
SET duckdo_device = 'cuda';
SELECT estimate, ci_low, ci_high
FROM do_ate('customers', treatment := 'discount', outcome := 'revenue', model := 'causalpfn');
```

On an RTX 3060, 8,000 rows took 5.1 s against 40.0 s on the CPU, as medians of three runs. fp32 on
the GPU differs from the CPU by about 2e-7, and repeats exactly because CUDA sessions run
deterministic kernels. `duckdo_gpu_precision = 'tf32'` saves little (4.7 s) for about 140 times the
error. The result carries a warning naming the device and precision it ran
at.

- **The CUDA 12 package needs CUDA 12.8 exactly or later.** ONNX Runtime 1.29 is built against
  it. Older 12.x libraries, such as the ones PyTorch bundles, load and then fail with error 127.
  NVIDIA's `nvidia-cuda-runtime-cu12` and `nvidia-cublas-cu12` pip packages at 12.8 are enough
  without the full toolkit.
- **`do_pfn` runs on the CPU only.** On CUDA its graph crashed the process, so it refuses
  `duckdo_device = 'cuda'` before touching the GPU.
- **ROCm and Apple GPUs are not built.** `do_devices()` lists them with the reason.

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
| `duckdo_device` | `cpu` | Where causal foundation models run: `cpu` or `cuda`. `cuda` needs a build with `-DDUCKDO_ORT_FLAVOUR=cuda12`, and CUDA 12 with cuDNN 9 on the library path. Its results agree with the CPU's closely but not bit for bit, which is why the CPU is the default. `do_devices()` says whether CUDA loads here |
| `duckdo_gpu_precision` | `fp32` | `fp32` or `tf32` on `cuda`. ONNX Runtime defaults to tf32 and DuckDo does not: tf32 rounds matrix products to 10 mantissa bits, which can flip the sign of an effect near zero. `bf16` and `fp16` are refused, because the graphs are exported in fp32 |
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
