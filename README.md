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

> **Status: 0.4 (Phases 0-4 of the [roadmap](docs/ROADMAP.md)).** Classical estimators, diagnostics
> and graph identification all work and are tested. **Causal foundation models (CausalPFN, Do-PFN,
> CausalFM) are not wired up yet** - that is Phases 5-6. Passing `model := 'causalpfn'` today returns
> an error saying so.

## What works today

Everything below runs with no downloads, no ONNX and no network.

### Estimation

| Function | Returns |
|---|---|
| `do_ate` / `do_att` / `do_atc` | one row: `estimand, estimator, estimate, std_error, ci_low, ci_high, p_value, n, n_treated, n_trimmed, variance_method, warnings` |
| `do_cate` | one row per input row: `row_id, id, treatment, outcome, cate, cate_low, cate_high, learner` |

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

Reproduce with `test/sql/estimators.test`.

## Settings

`duckdo_default_estimator`, `duckdo_max_rows` (100k), `duckdo_max_features` (500),
`duckdo_max_categorical_levels` (32), `duckdo_seed` (42), `duckdo_bootstrap_reps` (200).

## Known limitations

Stated plainly, because a causal tool that hides its limits is worse than none.

- **Binary treatments only.** Continuous and multi-valued treatments are refused with an explicit
  error rather than silently binarised. Planned for Phase 10.
- **`do_cate` intervals are slightly narrow.** Measured 95% coverage is about 0.90 on synthetic data,
  because the pseudo-outcome regression does not propagate uncertainty from the nuisance models.
- **Base learners are regularised GLMs.** Strongly non-linear confounding will not be fully removed.
  Gradient-boosted base learners are a Phase 2 follow-up.
- **Graphs live in process memory**, not the DuckDB catalog, so they do not survive a restart.
- **Data is read on a separate connection**, so uncommitted changes in your current transaction are
  not visible to an estimation call.
- **No foundation models yet.** See Phases 5-6 of the [roadmap](docs/ROADMAP.md).

## Testing

```sh
./build/release/test/unittest "test/*"
```

67 assertions across estimator recovery, diagnostics, error paths and graph identification.

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
