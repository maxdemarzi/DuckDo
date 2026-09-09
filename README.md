# DuckDo

**Causal inference inside DuckDB.** `SELECT` tells you what happened. `do_ate()` tells you what your
last change was worth.

DuckDo makes DuckDB an in-process causal engine: treatment effects, assumption diagnostics and
graph-based identification expressed as ordinary SQL over ordinary tables. No Python, no training
loop, no data leaving the process.

```sql
SELECT estimand, estimator, round(estimate, 2) AS estimate, round(ci_low, 2), round(ci_high, 2)
FROM do_ate('customers',
     treatment  := 'received_discount',
     outcome    := 'revenue',
     covariates := ['age', 'income', 'tenure']);
-- ATE | aipw | 12.84 | 8.59 | 17.09
```

> **Status: every roadmap phase through 9 is implemented.** Estimators, diagnostics, graph
> identification, the `do()` surface and segmented estimation are cross-checked against EconML and
> DoWhy. **Two causal foundation models run inside DuckDB** on ONNX Runtime — CausalPFN and Do-PFN.
> That path is opt-in at build time (`-DDUCKDO_ONNXRUNTIME_ROOT`) so the default build keeps zero
> dependencies and needs no downloads.

Full signatures: **[docs/FUNCTIONS.md](docs/FUNCTIONS.md)**.

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

Also `do_graphs()`, `do_graph_drop()`.

### Causal foundation models

```sql
SELECT model, setting, license, max_covariates, available FROM do_list_models();
-- causalpfn | backdoor (ignorability) | Apache-2.0 | 99 | true
-- do_pfn    | non-identifiable prior  | CC BY 4.0  |  5 | true

-- Same SQL, different engine. The result names the model, not a classical estimator.
SELECT estimator, estimate, variance_method
FROM do_ate('customers', treatment := 'discount', outcome := 'revenue', model := 'causalpfn');
-- causalpfn | 3.061 | effect dispersion (no model uncertainty)
```

| | [CausalPFN](https://github.com/vdblm/CausalPFN) | [Do-PFN](https://github.com/jr2021/Do-PFN) |
|---|---|---|
| Parameters | 18.8M | 7.3M |
| Identification setting | backdoor (ignorability) | non-identifiable prior |
| Licence | Apache-2.0, no obligation | CC BY 4.0, **attribution required** |
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

Getting either running takes three steps, and the extension never ships or redistributes weights:

```sh
# 1. build with ONNX Runtime (download a release from onnxruntime.ai)
cmake -DDUCKDO_ONNXRUNTIME_ROOT=/path/to/onnxruntime-1.29.0 ...

# 2. export the graphs yourself, from the upstream checkpoints
pip install causalpfn
python scripts/export/export_causalpfn.py --out ~/.cache/duckdo

git clone https://github.com/jr2021/Do-PFN            # optional, second model
python scripts/export/export_dopfn.py --repo ./Do-PFN --out ~/.cache/duckdo

# 3. point DuckDo at them
SET duckdo_model_dir = '~/.cache/duckdo';
```

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
implementations on identical rows, including the standard IHDP replication:

| scenario | truth | duckdo `aipw` | econml `LinearDRLearner` | dowhy PSW |
|---|---|---|---|---|
| linear confounded | 3.0000 | 3.0193 | 3.0180 | 3.0337 |
| heterogeneous | 2.9936 | 2.9847 | 2.9716 | 2.9662 |
| IHDP npci-1 | 4.0161 | 3.8766 | 3.9555 | 4.0287 |

Reproduce with `test/sql/estimators.test` and `python scripts/crosscheck_econml.py`
(the latter needs `econml` and `dowhy`; it is a dev tool, not shipped).

### And does the foundation model beat them?

Not on these data, and the honest answer is worth more than a flattering one:

Three engines, identical SQL, 1500 rows, true ATE **2.9806**:

| engine | estimate | error | CATE correlation |
|---|---|---|---|
| `estimator := 'aipw'` | 3.0628 | +0.082 | 0.9995 |
| `model := 'causalpfn'` | **3.0610** | **+0.080** | **0.9995** |
| `model := 'do_pfn'` | 2.6307 | −0.350 | 0.984 |

**CausalPFN matches AIPW.** Do-PFN **shrinks the population effect** — the documented weakness of
that model class, reproduced here rather than hidden — while still ranking individuals well. So
Do-PFN remains usable for *who to treat* and is the wrong tool for *what the programme is worth*;
CausalPFN is fine for both, which is why it is the one to start with.

Note that none of these beat AIPW on a linear DGP, where AIPW is correctly specified. The
foundation models earn their keep on data whose structure you do not already know. Read
`do_sensitivity` before believing any of them.

Inference cost: ~16 s for 1500 rows across both models, single CPU.

## Settings

`duckdo_default_estimator`, `duckdo_max_rows` (1M), `duckdo_max_features` (500),
`duckdo_max_categorical_levels` (32), `duckdo_max_groups` (1000), `duckdo_seed` (42),
`duckdo_bootstrap_reps` (200), `duckdo_threads` (0 = one per hardware thread). Every guardrail
names the setting to raise when it trips.

### Cost

The dense accumulations that dominate every fit are split across row blocks, with a fixed reduction
order so results stay bit-identical run to run. Timings include generating the table
(`scripts/benchmark.py`):

| rows | covariates | estimator | seconds |
|---|---|---|---|
| 100k | 5 | `aipw` | 0.6 |
| 100k | 50 | `aipw` | 2.1 |
| 1M | 5 | `aipw` | 2.3 |
| 1M | 50 | `aipw` | 18.9 |
| 1M | 50 | `dml` | 18.7 |

The 1M × 50 case was 101 s single-threaded before the accumulation was parallelised.

## Known limitations

Stated plainly, because a causal tool that hides its limits is worse than none.

- **Binary treatments for the effect estimators; continuous ones only through `do_ape` and
  `do_dose_response`.** The binary path refuses a dose rather than silently binarising it, and says
  where to go. Multi-valued categorical treatments are still unsupported, and the dose-response
  model is quadratic in the dose, so a sharply non-monotone response will be smoothed.
- **`do_cate` intervals run a touch narrow under heavy confounding.** Measured 95% coverage is
  0.973 / 0.935 / 0.945 across randomised, confounded and strongly-confounded DGPs
  (`scripts/coverage_check.py`). The interval uses an HC1 sandwich, because the doubly-robust
  pseudo-outcome's variance scales with `1/e(x)` — assuming it constant gave 0.896. The residual
  gap is nuisance-estimation uncertainty, second-order under cross-fitting.
- **Base learners are regularised GLMs.** Strongly non-linear confounding will not be fully removed.
  Gradient-boosted base learners are a Phase 2 follow-up.
- **Graphs live in process memory**, not the DuckDB catalog, so they do not survive a restart.
- **Data is read on a separate connection**, so uncommitted changes in your current transaction are
  not visible to an estimation call.
- **Foundation model intervals need `ensemble :=`, and still miss parameter uncertainty.** A single
  forward pass reports only the dispersion of its own point estimates, which is far too narrow —
  on the DGP above the single-draw interval is [3.035, 3.048] and **excludes the true 3.0**, while
  `ensemble := 8` gives [3.007, 3.121] and covers it. Even then the interval covers context
  selection and sampling, not the model's own weights, and it costs one forward pass per draw.
- **Do-PFN accepts only five covariates and a fixed context ladder, and shrinks population effects.**
  Where a covariate budget binds, DuckDo keeps the most outcome-correlated and names the rest in
  `warnings`. `model := 'causalfm'` is not exported.
- **ONNX Runtime links dynamically.** The community build ships without it; a build that enables it
  needs `onnxruntime.dll`/`.so` alongside the binary. Static linking is not done.

## Testing

```sh
./build/release/test/unittest "test/*"
```

```sh
# with foundation models as well
DUCKDO_MODEL_DIR=$(pwd)/build/models ./build/release/test/unittest "test/*"
```

116 assertions in the dependency-free build, 135 with the foundation-model path enabled, across
estimator recovery, diagnostics, error paths, graph identification, the `do()` surface and
end-to-end inference for both models.

Two further dev-only harnesses, neither shipped:

```sh
python scripts/crosscheck_econml.py    # grades the estimators against EconML and DoWhy on IHDP
python scripts/coverage_check.py       # measures do_cate's empirical interval coverage
```

## Submitting to community extensions

[description.yml](description.yml) is ready to copy into a fork of
`duckdb/community-extensions`; update `version` and `ref` to the release commit first. Its
`hello_world` deliberately needs no download - a first impression that requires a 300 MB fetch is a
first impression most people never have.

## Building
### Managing dependencies
DuckDo currently has **no external dependencies** - the numerics are hand-rolled in `src/linalg.cpp`,
so `vcpkg.json` lists nothing and you can skip vcpkg entirely. That changes in Phase 5, when ONNX
Runtime arrives for the foundation-model path.

### Build steps
Now to build the extension, run:
```sh
make
```
The main binaries that will be built are:
```sh
./build/release/duckdb
./build/release/test/unittest
./build/release/extension/duckdo/duckdo.duckdb_extension
```
- `duckdb` is the binary for the duckdb shell with the extension code automatically loaded.
- `unittest` is the test runner of duckdb. Again, the extension is already linked into the binary.
- `duckdo.duckdb_extension` is the loadable binary as it would be distributed.

## Running the extension

Start the shell with `./build/release/duckdb`, then:

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

## Running the tests
Different tests can be created for DuckDB extensions. The primary way of testing DuckDB extensions should be the SQL tests in `./test/sql`. These SQL tests can be run using:
```sh
make test
```

### Installing the deployed binaries
To install your extension binaries from S3, you will need to do two things. Firstly, DuckDB should be launched with the
`allow_unsigned_extensions` option set to true. How to set this will depend on the client you're using. Some examples:

CLI:
```shell
duckdb -unsigned
```

Python:
```python
con = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
```

NodeJS:
```js
db = new duckdb.Database(':memory:', {"allow_unsigned_extensions": "true"});
```

Secondly, you will need to set the repository endpoint in DuckDB to the HTTP url of your bucket + version of the extension
you want to install. To do this run the following SQL query in DuckDB:
```sql
SET custom_extension_repository='bucket.s3.eu-west-1.amazonaws.com/<your_extension_name>/latest';
```
Note that the `/latest` path will allow you to install the latest extension version available for your current version of
DuckDB. To specify a specific version, you can pass the version instead.

After running these steps, you can install and load your extension using the regular INSTALL/LOAD commands in DuckDB:
```sql
INSTALL duckdo;
LOAD duckdo;
```

## Setting up CLion

### Opening project
Configuring CLion with this extension requires a little work. Firstly, make sure that the DuckDB submodule is available.
Then make sure to open `./duckdb/CMakeLists.txt` (so not the top level `CMakeLists.txt` file from this repo) as a project in CLion.
Now to fix your project path go to `tools->CMake->Change Project Root`([docs](https://www.jetbrains.com/help/clion/change-project-root-directory.html)) to set the project root to the root dir of this repo.

### Debugging
To set up debugging in CLion, there are two simple steps required. Firstly, in `CLion -> Settings / Preferences -> Build, Execution, Deploy -> CMake` you will need to add the desired builds (e.g. Debug, Release, RelDebug, etc). There's different ways to configure this, but the easiest is to leave all empty, except the `build path`, which needs to be set to `../build/{build type}`, and CMake Options to which the following flag should be added, with the path to the extension CMakeList:

```
-DDUCKDB_EXTENSION_CONFIGS=<path_to_the_exentension_CMakeLists.txt>
```

The second step is to configure the unittest runner as a run/debug configuration. To do this, go to `Run -> Edit Configurations` and click `+ -> Cmake Application`. The target and executable should be `unittest`. This will run all the DuckDB tests. To specify only running the extension specific tests, add `--test-dir ../../.. [sql]` to the `Program Arguments`. Note that it is recommended to use the `unittest` executable for testing/development within CLion. The actual DuckDB CLI currently does not reliably work as a run target in CLion.
