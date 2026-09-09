# DuckDo — Causal Foundation Models in DuckDB

**An implementation roadmap, v1 (2026-09-08)**

> **Progress: every phase through 9 is implemented, built and tested.** 116 assertions in the
> dependency-free build and 135 with the foundation-model path enabled, against DuckDB v1.5.4, plus
> an EconML/DoWhy cross-check on IHDP and PyTorch parity gates on the exported ONNX graphs.
> **CausalPFN and Do-PFN both run end to end inside DuckDB.** Phase 10 remains future work; see
> section 9 for what is still open.

---

## 1. Thesis

SQL answers *what happened*. ML extensions like [`anofox_tabfm`](https://duckdb.org/community_extensions/extensions/anofox_tabfm) and `infera` answer *what will probably happen*. Nothing in the DuckDB ecosystem answers **what would happen if I did X** — the interventional question, `P(Y | do(X = x))`.

DuckDo closes that gap. It makes DuckDB an in-process causal inference engine: treatment effects, interventional predictions, counterfactuals, and policy evaluation expressed as ordinary SQL over ordinary tables, with no Python, no training loop, and no data leaving the process.

The timing is favourable. A class of **causal foundation models** (CFMs) — transformers pretrained on synthetic structural causal models that estimate treatment effects on unseen datasets by in-context learning — matured through 2026 and now has three credible open implementations:

| Model | Setting | Params | Weights | License | Notes |
|---|---|---|---|---|---|
| **CausalPFN** ([repo](https://github.com/vdblm/CausalPFN), [HF](https://huggingface.co/vdblm/causalpfn)) | Backdoor (ignorability) | — | Hugging Face | Apache-2.0 | CATE + ATE with calibrated intervals. Best default. |
| **Do-PFN** ([repo](https://github.com/jr2021/Do-PFN)) | Non-identifiable prior | 7.3M | GitHub | CC BY 4.0 | Full interventional distribution; expresses uncertainty under hidden confounding. Tiny. |
| **CausalFM** ([repo](https://github.com/yccm/CausalFM)) | Backdoor / front-door / IV | — | GitHub | verify at integration | Separate checkpoint per identification setting. |

The reference architecture for running such models inside DuckDB already exists and is proven: `anofox_tabfm` statically links ONNX Runtime, compiles **weight-free** ONNX graphs into the extension binary, and downloads weights from Hugging Face into a user-owned cache on first use. DuckDo should borrow that shape rather than reinvent it.

### Positioning in one line

> `SELECT` tells you what happened. `do_ate()` tells you what your last change was worth. `do_predict(intervention := ...)` tells you what the next one will be worth.

### What DuckDo is *not*

- Not a causal **discovery** engine (learning the DAG from data). That is post-1.0, and deliberately so — the field is not reliable enough to ship as a default.
- Not a replacement for a statistician. Every estimand carries assumptions. DuckDo's job is to make those assumptions **checkable in SQL**, not to hide them.
- Not a general ML inference extension. If you want zero-shot classification, use `anofox_tabfm`. DuckDo should coexist with it cleanly.

---

## 2. Design principles

These are the tie-breakers for every decision below.

1. **Useful before it is clever.** Phase 2 ships classical estimators that require zero downloads and no ONNX. A user gets real value from `INSTALL duckdo` before a single foundation model is involved. The CFM path is an upgrade, not a prerequisite.
2. **The estimand is explicit.** Every result carries the estimand (`ATE`/`ATT`/`CATE`), the estimator used, the identification strategy assumed, and the sample size. No result is a bare float.
3. **Assumptions are first-class queryable objects.** `do_balance`, `do_overlap`, `do_refute`, `do_sensitivity` are not an afterthought section of the docs; they are functions that return rows, so they can be asserted on in dbt tests and CI.
4. **Never redistribute weights.** The repo contains zero weight bytes. Graphs are compiled in; weights are downloaded by the user under the model's own license, into a user-owned cache. This is both a licensing requirement and the only way to stay inside community-extension size budgets.
5. **Degrade loudly, not silently.** Too few rows, no overlap, a context limit exceeded, a covariate that is actually a collider — each raises a DuckDB exception or a warning column naming the remediation. Never a quietly wrong number.
6. **Two names for everything.** Full names `duckdo_*` for unambiguous scripting, short `do_*` aliases for ergonomics — mirroring the `anofox_tabfm_*` / `tabfm_*` convention.
7. **Determinism by default.** A fixed seed, `fp32`, and a stable row ordering into the model context. Reproducibility beats the last 3% of throughput.

---

## 3. Target SQL surface

The full surface, stated up front so later phases have a fixed target. Phase annotations in brackets.

### Estimation

```sql
-- Population effect. [P2 classical, P6 CFM]
SELECT * FROM do_ate('customers',
    treatment  := 'received_discount',
    outcome    := 'revenue',
    covariates := ['age', 'income', 'tenure'],
    estimator  := 'aipw');          -- or model := 'causalpfn'

-- estimand | estimator | estimate | std_error | ci_low | ci_high | n | n_treated | warnings
-- ATE      | aipw      |    12.84 |      2.17 |   8.59 |   17.09 | … |         … | []

-- Per-row heterogeneous effect. [P2 meta-learners, P6 CFM]
SELECT customer_id, cate, cate_low, cate_high
FROM do_cate('customers',
    treatment := 'received_discount',
    outcome   := 'revenue',
    model     := 'causalpfn');

-- Effect on the treated / on the controls. [P2]
SELECT * FROM do_att(...);
SELECT * FROM do_atc(...);

-- Segmented estimation, one estimate per group. [P8]
SELECT * FROM do_ate_by('customers', ..., by := ['region', 'plan']);
```

### Diagnostics and assumption checking

```sql
-- Standardized mean differences, before and after weighting. [P3]
SELECT * FROM do_balance('customers', treatment := 'discount', covariates := [...]);
-- covariate | smd_raw | smd_adjusted | variance_ratio | balanced

-- Positivity / common support. [P3]
SELECT * FROM do_overlap('customers', treatment := 'discount', covariates := [...]);
-- bucket | n_treated | n_control | min_ps | max_ps | off_support_rows

-- One-call assumption report. [P3]
SELECT * FROM do_diagnose('customers', treatment := ..., outcome := ..., covariates := ...);
-- check | status | detail | severity

-- Refutation: does the estimate survive attack? [P3]
SELECT * FROM do_refute('customers', ..., method := 'placebo_treatment');
SELECT * FROM do_refute('customers', ..., method := 'random_common_cause');
SELECT * FROM do_refute('customers', ..., method := 'subset', fraction := 0.8);
SELECT * FROM do_refute('customers', ..., method := 'unobserved_confounder');

-- How strong would a hidden confounder need to be to nullify this? [P3]
SELECT * FROM do_sensitivity('customers', ...);
-- e_value | robustness_value | partial_r2_to_nullify | interpretation
```

### Graphs and identification

```sql
-- Register a DAG once, reuse it everywhere. [P4]
CALL do_graph_create('sales_dag', '
  digraph {
    season -> discount; season -> revenue;
    discount -> clicks; clicks -> revenue;
    intent -> discount; intent -> revenue;
  }');

-- What must I adjust for, and is that even possible? [P4]
SELECT * FROM do_identify(graph := 'sales_dag', treatment := 'discount', outcome := 'revenue');
-- strategy  | adjustment_set | identifiable | note
-- backdoor  | [season]       | false        | intent is an unblocked backdoor
-- frontdoor | [clicks]       | true         |
-- iv        | [coupon_mail]  | true         |

-- Is the covariate list I typed actually safe? [P4]
SELECT * FROM do_validate(graph := 'sales_dag', treatment := 'discount',
                          outcome := 'revenue', covariates := ['season', 'clicks']);
-- covariate | role       | verdict | reason
-- season    | confounder | keep    | closes a backdoor path
-- clicks    | mediator   | DROP    | descendant of treatment; adjusting removes part of the effect

SELECT do_dseparated('sales_dag', 'discount', 'revenue', ['season']);   -- boolean
```

### Interventions and counterfactuals

```sql
-- What happens if I set price to 19.99 for everyone? [P7]
SELECT * FROM do_predict('customers',
    intervention := {'price': 19.99},
    outcome      := 'conversion',
    model        := 'do_pfn');

-- Counterfactual: what would THIS row have done under the other arm? [P7]
SELECT customer_id, revenue AS actual, y0, y1
FROM do_counterfactual('customers', treatment := 'discount', outcome := 'revenue');

-- Is my targeting rule worth anything? [P7]
SELECT * FROM do_policy_value('customers', ..., policy := 'cate > 5');
SELECT * FROM do_uplift('customers', ...);   -- Qini / AUUC curve rows
```

### Model lifecycle

```sql
CALL do_download(model := 'causalpfn');       -- [P5]
CALL do_load(model := 'causalpfn');
CALL do_unload();
CALL do_remove(model := 'do_pfn');
SELECT * FROM do_list_models();   -- model | setting | license | commercial | cached | bytes
SELECT * FROM do_devices();       -- cpu / cuda / rocm / mlx availability
CALL do_register_model(id := 'mine', graph := 'model.onnx', weights := 'model.safetensors',
                       setting := 'backdoor', license := 'apache-2.0');
```

### Settings

`duckdo_default_model`, `duckdo_default_estimator`, `duckdo_cache_dir`, `duckdo_max_rows`, `duckdo_max_features`, `duckdo_max_memory`, `duckdo_threads`, `duckdo_device`, `duckdo_gpu_precision`, `duckdo_seed`, `duckdo_bootstrap_reps`, `duckdo_accept_model_license`, `duckdo_context_strategy`, `duckdo_trace_level`.

---

## 4. Architecture

```
src/
  duckdo_extension.cpp            entry point, registration, settings
  include/duckdo/
    common/
      settings.hpp                DBConfig-backed options, one place
      errors.hpp                  every exception names its remediation
      linalg.hpp                  small dense ops, Cholesky, IRLS
      rng.hpp                     seeded, reproducible
    frame/
      spec.hpp                    parsed treatment/outcome/covariate/graph spec
      binder.hpp                  named-param bind shared by all table functions
      encoder.hpp                 typed columns -> float32 matrix; categoricals, NULLs
      frame.hpp                   materialized estimation frame + row provenance
    estimator/
      base.hpp                    Estimator interface: fit(frame) -> EffectResult
      naive.hpp  ipw.hpp  aipw.hpp  regression.hpp
      metalearner.hpp             S-, T-, X-learner over a pluggable base learner
      dml.hpp                     cross-fitted double machine learning
      variance.hpp                influence functions, bootstrap, cluster-robust
    diag/
      balance.hpp  overlap.hpp  refute.hpp  sensitivity.hpp
    graph/
      dag.hpp                     adjacency, topological ops, DOT/JSON parse
      identify.hpp                backdoor, front-door, IV search
      validate.hpp                collider / mediator / M-bias detection
    runtime/
      registry.hpp                built-in + user-registered model catalog
      cache.hpp                   ~/.cache/duckdo, integrity, resume
      download.hpp                httpfs + DuckDB secrets for gated repos
      session.hpp                 ORT session pool keyed by (model, device, precision)
      tensor.hpp                  frame -> ORT tensors, padding, dynamic axes
    cfm/
      cfm_estimator.hpp           shared context/query split and chunking
      causalpfn.hpp  dopfn.hpp  causalfm.hpp
      calibration.hpp             interval calibration, shrinkage correction
```

**The load-bearing abstraction is `frame/`.** Every estimator — classical or foundation-model — consumes the same `CausalFrame`: an encoded float32 matrix plus treatment vector, outcome vector, column metadata, and row provenance back to the source relation. Getting this right in Phase 1 is what makes Phases 2 and 6 cheap. Getting it wrong is what makes both expensive, twice.

---

## 5. Phase overview

| Phase | Name | Version | Works with no download? | Exit gate |
|---|---|---|---|---|
| 0 | Foundation and spikes | 0.0.x | — | **DONE** — builds on MSVC 19.44 / DuckDB v1.5.4; `do_` prefix confirmed usable |
| 1 | The causal frame | 0.1.0 | yes | **DONE** — binder, encoder, guardrails, provenance, `id :=` join key |
| 2 | Classical estimators | 0.2.0 | yes | **DONE** — recovers a known ATE to 0.004, and agrees with EconML to within 0.08 on IHDP (`scripts/crosscheck_econml.py`) |
| 3 | Diagnostics and refutation | 0.3.0 | yes | **DONE** — balance, overlap, diagnose, 5 refuters, E-value + robustness value |
| 4 | Graphs and identification | 0.4.0 | yes | **DONE** — d-separation, backdoor/front-door/IV, covariate grading |
| 5 | Inference runtime | 0.5.0 | yes (models opt-in) | **DONE** — ONNX Runtime linked behind a build flag, model catalog, session cache, and a parity gate the export refuses to pass below 1e-4 (measured 4.3e-06 to 8.1e-06) |
| 6 | CFM estimators | 0.6.0 | opt-in | **DONE for CausalPFN and Do-PFN** — CausalPFN matches AIPW (3.061 vs 3.063, truth 2.981) with CATE correlation 0.9995 and no shrinkage; Do-PFN shrinks to 2.631, reproduced and reported. CausalFM not exported |
| 7 | The `do()` surface | 0.7.0 | **yes** | **DONE on classical backends** — `do_predict`, `do_counterfactual`, `do_policy_value`, `do_uplift`, `do_optimal_policy`. Gains a CFM engine in phase 6 |
| 8 | Scale and performance | 0.8.0 | — | **DONE bar spill** — `do_ate_by`, parallel dense accumulation (1M × 50 in 18.9 s against a 30 s gate, down from 101 s), `duckdo_max_rows` default raised to 1M, `scripts/benchmark.py`. Memory spill outstanding |
| 9 | Ship | 1.0.0 | — | **PARTIAL** — `description.yml` and `docs/FUNCTIONS.md` are written; the submission PR and the wider docs site are outstanding |
| 10 | Frontier | post-1.0 | — | **IN PROGRESS** — continuous treatments (`do_ape`, `do_dose_response`) and panel/DiD (`do_did`, `do_event_study`) landed; longitudinal, survival, mediation and discovery outstanding |

Phases 2, 3, and 4 are independently valuable and can proceed in parallel once Phase 1 lands. Phases 5 and 6 are strictly sequential.

**What actually happened, versus the plan:**

- **Phase 7 did not need the foundation models at all.** Interventional prediction, counterfactuals,
  policy value, uplift and policy trees all run on the phase-2 nuisance models. Phase 6 added a
  second engine behind the same SQL rather than a new set of functions — a better outcome than the
  plan assumed, and it meant phase 7 could ship before phases 5-6.
- **The ONNX export was the risk the plan said it was, and it bit in a specific way.** Do-PFN's
  context/query split is a Python int the torchscript tracer bakes in as a constant, so a graph
  traced at one context length silently returns the wrong rows at another — wrong numbers, not a
  shape error. The dynamo exporter would carry it symbolically but cannot get past the model's
  data-dependent `ModuleList` indexing. The resolution was a *ladder* of graphs at fixed context
  lengths with the query axis dynamic, which costs nothing real because the model caps at 2200 rows
  and DuckDo has to subsample context anyway.
- **Starting with Do-PFN was the right call.** At 7.3M parameters it exported in one afternoon of
  iteration; the two obstacles (an unsupported `aten::nansum`, a data-dependent debug assert) were
  both small and both fixable without changing model semantics.
- **The two foundation models behaved very differently, and that is the finding.** Do-PFN shrinks
  the population effect badly (2.631 against a truth of 2.981); CausalPFN does not (3.061, beside
  AIPW's 3.063). The plan treated "CFM" as one thing; it is not. Both results are in the README as
  a table rather than buried, which is what design principle 5 requires.
- **CausalPFN's export was easier in the way that mattered.** Its context length comes from a
  tensor shape rather than a Python int, so both axes stayed dynamic and no ladder was needed. It
  needed its own accommodation instead — an in-place `&=` with no opset-17 equivalent — and its
  logit-level parity is ~1e-2 because PyTorch fuses attention where ONNX Runtime decomposes it.
  Gating on the estimand (ATE agrees to 0.00087) rather than the logits is the defensible test.
- **`do_cate`'s interval was measurably wrong and is now measurably better.** Coverage was 0.896
  against a nominal 0.95 under strong confounding, because the doubly-robust pseudo-outcome's
  variance scales with `1/e(x)` and the interval assumed it constant. An HC1 sandwich took the
  worst case to 0.945. `scripts/coverage_check.py` keeps it honest.

---

## Phase 0 — Foundation and spikes

**Goal:** remove every unknown that could force a redesign later, and get a boring, green CI.

### Work

1. **Submodules and first build.** `git submodule update --init --recursive`, then `make` on Linux, macOS (arm64 + x64), and Windows. Confirm the DuckDB v1.5.4 / `v1.5-variegata` CI toolchain pinned in `.github/workflows/MainDistributionPipeline.yml` actually builds this repo.
2. **Drop the template's OpenSSL dependency.** Remove `find_package(OpenSSL)` and the `target_link_libraries` lines from `CMakeLists.txt`, drop it from `vcpkg.json`, and delete `duckdo_openssl_version` from the source and the test. It is scaffolding, and it constrains the vcpkg manifest we will need for real dependencies.
3. **Spike: is `do_` a usable prefix? RESOLVED — yes.** Verified against a DuckDB v1.5.4 build on 2026-09-08: `SELECT do_ate('x')`, `SELECT * FROM do_ate('x', treatment := 'a')` and `CALL do_download(model := 'causalpfn')` all reach name resolution and fail with a *Catalog* error, meaning the parser accepted them. Only the bare word is reserved — `SELECT do(1)` is a parser error. **`do_*` is therefore the primary surface**, with `duckdo_*` full names registered alongside.
4. **Spike: relation input shape.** `anofox_tabfm` takes a table name or subquery *as a string*. DuckDB also supports table in-out functions that accept a real relation. Prototype both; prefer the native relation if the binder can see column types at bind time, because it gives us error messages at plan time instead of run time. Fall back to the string form if not.
5. **Spike: named parameters and list/struct parameters.** Confirm `covariates := ['a','b']` (LIST) and `intervention := {'price': 19.99}` (STRUCT) bind cleanly in a table function, including the type of an empty list.
6. **Choose a linear algebra dependency.** Eigen (header-only, MPL2) via vcpkg is the default recommendation — no runtime, easy static link, and enough for GLMs and cross-fitting. Confirm it builds for every target platform before committing.
7. **Test harness conventions.** Establish `test/sql/` layout: `unit/` for function-level, `estimator/` for numerical, `golden/` for reference-value comparisons. Add a Python-side generator (dev-only, not shipped) that writes deterministic synthetic datasets with **known** ground-truth effects into Parquet fixtures — the entire numerical test strategy rests on this.
8. **Settings scaffolding.** Register every `duckdo_*` setting up front as no-ops with correct types and defaults, so later phases only wire behaviour.
9. **Error convention.** A single `DuckDoException` helper enforcing that every message states what failed *and* what to do about it.

### Exit gate

- `make test` green on Linux x64, macOS arm64, Windows x64.
- The `do_` naming question is answered and written into this document. ✅ `do_*` confirmed usable.
- Zero OpenSSL references remain.
- A synthetic fixture with a known ATE exists and is loadable from a test.

### Risks

| Risk | Mitigation |
|---|---|
| `do_` collides with grammar | Spike it first; `duckdo_*` primary + `causal_*` alias is a fine fallback |
| Eigen fails on a target platform | Hand-rolled Cholesky/IRLS is ~300 lines; keep the interface in `linalg.hpp` so the backend is swappable |

---

## Phase 1 — The causal frame

**Goal:** one binder and one encoder that every estimator in every later phase reuses. This phase ships no user-visible estimation, only the machinery — but it is the phase most worth doing slowly.

### Work

1. **`CausalSpec` parsing.** From named parameters: relation, `treatment`, `outcome`, `covariates` (explicit list, or default = all columns except treatment/outcome/`exclude`), `exclude`, `by`, `weights`, `cluster`, `graph`. Validate names against the bound relation's schema at bind time and error with a did-you-mean suggestion on a typo.
2. **Treatment typing.** v1: binary only — BOOLEAN, or a two-valued INTEGER/VARCHAR with an explicit `control :=` / `treated :=` mapping when the values are not 0/1. Reject multi-valued and continuous treatments with a message pointing at Phase 10, rather than silently binarizing.
3. **Outcome typing.** Continuous (any numeric) and binary. Record which, because it selects the link function in the outcome model and changes how `do_ate` reports (difference in means vs. risk difference).
4. **The encoder.** Typed DuckDB columns to a dense `float32` matrix:
   - Numerics: cast, optional standardization (record the mean/sd so results can be reported on the original scale).
   - `BOOLEAN`: 0/1.
   - `VARCHAR`/`ENUM`: one-hot below a cardinality threshold, otherwise target/ordinal encoding computed **within cross-fitting folds** to avoid leakage. High-cardinality columns get a warning.
   - `DATE`/`TIMESTAMP`: epoch numeric plus optional cyclical parts; never one-hot.
   - `LIST`/`STRUCT`/`MAP`: rejected with a message suggesting the user flatten them.
   - `NULL`: an explicit missingness indicator column per covariate plus median/mode fill. Never drop rows silently; report `n_rows_with_missing` in the result.
5. **Row provenance.** Every frame row keeps its source row index so `do_cate` can return per-row output aligned to the input relation, including after filtering and sampling.
6. **Guardrails.** `duckdo_max_rows` (default 1M), `duckdo_max_features` (default 500),
   `duckdo_max_memory` (default: half of DuckDB's own `memory_limit`, so raising one raises the
   other; `'-1'` removes the ceiling). Exceeding a limit is an exception naming the setting to
   raise, not a silent truncation — and the memory message carries the row count that *would*
   fit at the frame's width, so the fix can be pasted rather than derived.
7. **Deterministic ordering.** Frames are materialized in a stable order regardless of DuckDB's parallel scan order, so results are reproducible run to run.

### Exit gate

- Property tests: every supported DuckDB type encodes and the frame shape is exactly as specified.
- A frame built from a table, a subquery, and a view produce identical matrices.
- Encoding a 100k x 200 frame stays inside a stated memory budget.
- Fuzz the binder with malformed parameters; every failure path produces an actionable message.

### Risks

| Risk | Mitigation |
|---|---|
| Encoder choices silently bias estimates | Every encoding decision is reported in a `do_frame_summary()` debug function; leakage-prone encodings are fold-local by construction |
| Over-engineering ahead of real estimator needs | Build only what Phase 2 consumes; leave hooks, not implementations |

---

## Phase 2 — Classical estimators

**Goal:** a genuinely useful v0.2 that needs no model weights, no network, and no ONNX. This is the phase that earns the extension its first users, and it doubles as the correctness baseline that Phase 6 is graded against.

### Work

1. **Base learners.** Two, both in-house:
   - Regularized GLM (linear + logistic via IRLS, ridge penalty, cross-fitted). Sufficient for propensity scores and outcome regressions, cheap, stable, and fully deterministic.
   - Gradient-boosted trees, shallow, histogram-based. Optional and behind a flag; only if profiling shows the GLM is the accuracy bottleneck on the benchmark suite. **Do not start here** — it is a large amount of code for a second-order gain at this stage.
2. **Estimators**, in this order:
   - `naive` — raw difference in means. Included precisely so users can see how far it is from the adjusted estimate.
   - `regression` — outcome regression / g-computation.
   - `ipw` — inverse propensity weighting, with stabilized weights and configurable trimming.
   - `aipw` — augmented IPW, the doubly-robust default. **This is `duckdo_default_estimator`.**
   - `dml` — cross-fitted double ML (partially linear and interactive regression models).
   - `s_learner`, `t_learner`, `x_learner` — meta-learners producing per-row CATE.
3. **Estimands.** ATE, ATT, ATC — each a distinct weighting of the same fitted nuisance models, not a separate implementation.
4. **Variance.** Influence-function standard errors as the default (fast, correct for AIPW/DML), bootstrap as an option (`duckdo_bootstrap_reps`), cluster-robust when `cluster :=` is given. Report the method in the output so nobody has to guess.
5. **Result contract.** Every estimator returns the same row shape: `estimand, estimator, estimate, std_error, ci_low, ci_high, p_value, n, n_treated, n_trimmed, warnings`.
6. **Validation suite.** This is the heart of the phase:
   - Synthetic DGPs with analytically known ATE, sweeping confounding strength, overlap, nonlinearity, and treatment prevalence.
   - **IHDP** (1000 replications) — report PEHE and eps-ATE.
   - **Jobs / Lalonde** — report ATT against the experimental benchmark.
   - **ACIC 2016/2018** — a subset.
   - **Twins** — binary outcome.
   - Cross-check every estimate against EconML and DoWhy in a dev-only Python harness; failures outside tolerance block the release.

### Exit gate

- AIPW and DML match EconML within Monte Carlo error on the full suite.
- IPW with poor overlap emits a warning rather than a confidently wrong number.
- End-to-end `do_ate` on a 100k-row table finishes in under a second on one core.

### Risks

| Risk | Mitigation |
|---|---|
| Reimplementing statistics incorrectly | Every estimator is graded against an established library before it ships; no estimator merges without a golden test |
| Users treat `naive` output as causal | Result rows carry the estimator name and a warning for `naive`; docs lead with AIPW |

---

## Phase 3 — Diagnostics and refutation

**Goal:** make the assumptions checkable. This is DuckDo's strongest differentiator over "ML in SQL" extensions, and it is entirely classical code — no models required.

### Work

1. **`do_balance`** — standardized mean differences per covariate, raw and weighted; variance ratios; a `balanced` verdict against the conventional 0.1 threshold.
2. **`do_overlap`** — propensity distribution by arm, common-support bounds, count of off-support rows, and a bucketed histogram suitable for direct plotting.
3. **`do_diagnose`** — one call that runs the battery and returns `check | status | detail | severity`: positivity, balance, treatment prevalence, sample size adequacy, outcome variance, covariate collinearity, and (if a graph is registered) the Phase 4 covariate-role check.
4. **`do_refute`** — the DoWhy-style attacks, each returning the original estimate, the refuted estimate, and a pass/fail:
   - `placebo_treatment` — replace treatment with noise; the effect should collapse to zero.
   - `random_common_cause` — add an irrelevant confounder; the estimate should not move.
   - `subset` — re-estimate on a random subset; the estimate should be stable.
   - `unobserved_confounder` — simulate a confounder of specified strength and report the resulting bias.
   - `bootstrap` — resampling stability.
5. **`do_sensitivity`** — how strong must hidden confounding be to overturn the conclusion:
   - **E-value** (VanderWeele–Ding) for the point estimate and the CI bound.
   - **Robustness value** and partial R-squared bounds (Cinelli–Hazlett).
   - A plain-language `interpretation` column, because the whole point is that a non-statistician reads it.
6. **Assertion-friendly output.** Every diagnostic returns rows with a boolean verdict column so it drops straight into a dbt test or a CI query.

### Exit gate

- On synthetic data with planted unmeasured confounding, `do_refute` and `do_sensitivity` flag it at the expected strength.
- On a genuinely randomized dataset, every check passes cleanly with no false alarms.
- `do_diagnose` on a realistic 100k-row table completes in a few seconds.

---

## Phase 4 — Graphs and identification

**Goal:** answer "what should I even adjust for?" — the question users get wrong most often, and the one no ML-in-SQL extension addresses at all.

### Work

1. **DAG representation and parsing.** Accept a DOT subset and a JSON edge list. Store graphs in a catalog-backed registry (`do_graph_create` / `do_graph_drop` / `do_graphs()`), persisted with the database where DuckDB's storage allows, and reject cycles at creation.
2. **d-separation.** The core primitive: `do_dseparated(graph, x, y, z)`. Everything else is built on it, so it gets exhaustive testing against a corpus of published DAGs with known separations.
3. **Backdoor identification.** Enumerate valid adjustment sets; return the minimal one and, optionally, the optimal one by estimated variance. Report `identifiable = false` with the specific unblocked path when no valid set exists over the observed variables.
4. **Front-door identification.** Detect a mediator set satisfying the front-door criterion.
5. **Instrumental variables.** Detect candidate instruments; report exclusion-restriction requirements as assumptions the user must accept, since they cannot be verified from data.
6. **`do_validate`** — grade a user's covariate list against the graph, per covariate: confounder (keep), mediator (drop — adjusting removes part of the effect), collider (drop — adjusting induces bias), instrument (drop from the outcome model — inflates variance), descendant of the outcome, or M-bias structure. This function will prevent more wrong answers than any estimator in this roadmap.
7. **Integration.** When `graph :=` is supplied to `do_ate`/`do_cate`, covariates default to the identified adjustment set and a covariate that fails validation raises an error rather than a warning.

### Exit gate

- d-separation is exhaustively correct on a corpus of DAGs up to ~10 nodes, checked against a reference implementation.
- `do_validate` correctly labels collider, mediator, and M-bias structures on hand-built test graphs.
- Adjusting for a mediator, when a graph is present, is a hard error.

### Risks

| Risk | Mitigation |
|---|---|
| Adjustment-set enumeration blows up combinatorially | Cap graph size; return the minimal set by default and enumerate lazily |
| Users have no DAG and skip the phase entirely | Graphs stay optional; `do_validate` also runs in a degraded heuristic mode that flags obvious post-treatment variables by name convention and correlation structure |

---

## Phase 5 — Inference runtime

**Goal:** run ONNX models inside DuckDB, with weights the user owns. Deliberately mirrors the `anofox_tabfm` architecture, which is proven against exactly these constraints.

### Work

1. **ONNX Runtime, statically linked**, CPU execution provider, fetched as a prebuilt archive with a build-from-source vcpkg feature as a fallback. CPU is the only flavour submitted to community extensions; CUDA/ROCm/MLX flavours live in-tree for self-builds and load their execution providers dynamically.
2. **Model export.** Convert the reference PyTorch checkpoints to ONNX with dynamic axes on both the context (n rows) and feature dimensions:
   - **CausalPFN** — from the Hugging Face checkpoint. Its Python package pulls `faiss-cpu`, which strongly suggests retrieval-based context selection for datasets larger than the context window; that selection logic must be reimplemented in C++, not exported into the graph.
   - **Do-PFN** — 7.3M parameters, small enough to be an easy first target. Start here to de-risk the export pipeline.
   - **CausalFM** — one checkpoint per identification setting, so the registry must key on `(model, setting)`.
   - Export scripts live in `scripts/export/`, are pinned to exact upstream commits, and emit a manifest recording checkpoint hash, opset, and dynamic-axis names.
3. **Weight-free graphs compiled into the binary**; weights downloaded on demand.
4. **Download and cache.** `do_download` over `httpfs`, into `~/.cache/duckdo` (override via `duckdo_cache_dir`), with checksum verification, resume, and a clear message if the cache is not writable. Gated repositories use standard DuckDB secrets:
   ```sql
   CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_xxx', SCOPE 'https://huggingface.co');
   ```
5. **License gating.** `do_list_models()` reports each model's license and a `commercial` boolean. Non-commercial weights require `SET duckdo_accept_model_license = true` before download. Note that **Do-PFN is CC BY 4.0**, which carries an attribution obligation that must be surfaced in the docs and in `do_list_models()`.
6. **Session pool.** ORT sessions keyed by `(model, device, precision)`, bounded by `duckdo_max_sessions`, LRU eviction, thread-count from `duckdo_threads`.
7. **Tensor marshalling.** `CausalFrame` to ORT tensors: zero-padding to the model's expected shape, correct dtype, and a documented column order — the single most likely source of silently wrong output, so it gets its own golden tests.
8. **Numerical parity harness.** For each exported model, a fixture of inputs and PyTorch reference outputs; the C++ path must match to 1e-4 in `fp32`. This runs in CI.

### Exit gate

- `CALL do_download(model := 'do_pfn'); CALL do_load(...)` works cold on all three desktop platforms.
- Parity harness green for every built-in model.
- Extension binary stays within the community-extension size budget with weights excluded.
- No weight bytes anywhere in the repository or the binary.

### Risks

| Risk | Mitigation |
|---|---|
| A checkpoint will not export cleanly to ONNX | Start with Do-PFN (7.3M params, simplest); if a model resists, ship the ones that work and document the gap rather than blocking the phase |
| Upstream checkpoints change without notice | Pin commit hashes; verify checksums; fail loudly on mismatch |
| Static ONNX Runtime breaks a platform | Follow `anofox_tabfm`'s `excluded_platforms` precedent: wasm, mingw, and musl are out of scope |

---

## Phase 6 — CFM estimators

**Goal:** the headline feature — `model := 'causalpfn'` and get a state-of-the-art causal estimate in one SQL statement.

### Work

1. **Context/query split.** CFMs are in-context learners: labelled rows form the context, target rows are queried against it. Define the split precisely and document it, because it determines what the numbers mean.
2. **Context selection beyond the window.** Real tables exceed any model's context. Strategies behind `duckdo_context_strategy`:
   - `subsample` — stratified by treatment arm, seeded, with the resulting variance inflation reported.
   - `retrieval` — nearest-neighbour context per query row (the approach CausalPFN's `faiss` dependency implies).
   - `ensemble` — multiple context draws, averaged, with between-draw variance folded into the reported interval.
   Whichever runs, the result row says so and reports the effective sample size.
3. **CATE and ATE paths.** Per-row CATE from the model directly; ATE by marginalizing, with the **shrinkage bias** the CFM overview paper documents for population-level estimates corrected or, at minimum, reported. Do not quietly ship a known-biased ATE.
4. **Interval calibration.** Use the models' native calibrated uncertainty where available; validate empirical coverage on held-out synthetic DGPs and correct if coverage is off. A 95% interval that covers 80% of the time is worse than no interval.
5. **Model routing.** `model := 'causalpfn' | 'do_pfn' | 'causalfm'`, with the identification setting checked against the registered graph when one exists — asking for a backdoor model on a graph with an unblocked backdoor path is an error worth raising.
6. **Classical fallback.** If weights are not downloaded, `do_ate` falls back to AIPW with an explicit note in the result rather than failing — but never the reverse, and never silently.
7. **Head-to-head validation.** Same benchmark suite as Phase 2 (IHDP, Lalonde, ACIC, RealCause-Lalonde). Publish DuckDo's CFM numbers next to the published Python numbers and next to DuckDo's own classical estimators. If the CFM does not beat AIPW on a benchmark, say so in the docs.

### Exit gate

- PEHE within 5% of the published Python reference on IHDP.
- Interval coverage within a few points of nominal on synthetic DGPs.
- A cold-start user path — install, download, estimate — works in under five minutes.

### Risks

| Risk | Mitigation |
|---|---|
| CFMs underperform classical estimators on real data | Publish the comparison honestly; the extension is valuable either way because it ships both |
| Shrinkage bias on ATE | Documented in the source literature; correct it, and report the correction |
| Context subsampling silently changes the estimand | Effective sample size and strategy are always in the output row |

---

## Phase 7 — The `do()` surface

**Goal:** the conceptual payoff. Not just "estimate an effect" but "query a hypothetical world."

### Work

1. **`do_predict(intervention := {...})`** — set one or more variables and predict the outcome distribution. Do-PFN's interventional distribution is the natural backend. Multi-variable interventions and interventions on non-treatment columns are what make this more than a rename of `do_cate`.
2. **`do_counterfactual`** — per-row potential outcomes `y0`/`y1` alongside the observed outcome, with intervals.
3. **`do_policy_value`** — evaluate a targeting rule given as a SQL expression over the row (`policy := 'cate > 5'`); return expected value, lift over treat-all and treat-none, and standard errors.
4. **`do_uplift`** — Qini and AUUC curve rows, ready to plot, plus the summary statistic.
5. **`do_optimal_policy`** — a shallow interpretable policy tree over covariates maximizing estimated value, returned as rows describing the splits. Deliberately shallow: a rule an operator will actually deploy.
6. **Composability.** All of these are table functions over relations, so they join, filter, aggregate, and land in a view or a dbt model like anything else. A worked example of that composition belongs in the README.

### Exit gate

- `do_policy_value` on a randomized holdout recovers the true policy value within its stated interval.
- Uplift curves match a reference implementation on a shared fixture.

---

## Phase 8 — Scale and performance

**Goal:** stop being a toy on real tables.

### Work

1. **Grouped estimation** — `do_ate_by(..., by := ['region'])` runs an independent estimation per group with shared nuisance-model infrastructure, parallelized across groups.
2. **Parallelism** — cross-fitting folds, bootstrap replicates, and CFM ensemble draws are all embarrassingly parallel; use DuckDB's task scheduler rather than raw threads so the extension respects the engine's thread budget.
3. **Context caching** — for CFMs, encode a fixed context once and reuse it across query chunks (`anofox_tabfm` does exactly this behind a setting).
4. ~~**Streaming and chunking**~~ — **partly done, and the other part was rejected.**
   `duckdo_max_memory` is now a real ceiling: the encoded matrix's size is computed before
   it is allocated, and exceeding the budget raises an error naming the setting and the row
   count that would fit. The peak was also measured and cut. What was *not* built is spilling.
   Every estimator indexes the matrix by row, cross-fitting reads it five times over and the
   bootstrap reads it in random row order two hundred times more; a spilled matrix would turn
   a fast refusal into an unbounded thrash. Refusing with an actionable message is the better
   failure, and it is also what section 6's guardrail rule already specified.
5. **GPU flavours** — CUDA and ROCm builds in-tree, with `do_devices()` reporting availability and `duckdo_gpu_precision` controlling `fp32`/`tf32`/`bf16`. Document that reduced precision can flip signs near zero effect.
6. **Benchmark suite in CI** — a fixed set of table sizes and estimators, tracked over time so
   regressions are visible. `scripts/benchmark.py` now gates on peak resident memory as well as
   seconds, and generates each shape outside the measurement so both numbers describe the
   estimator rather than the generator.

### Measured

Peak resident memory, from `scripts/benchmark.py`, which reads a pre-generated Parquet file so
the figure describes the estimator and not the table generator. The encoded matrix for the last
row is 400 MB.

| rows x covariates | before | after |
|---|---|---|
| 100k x 5 | 42 MB | 35 MB |
| 100k x 50 | 185 MB | 108 MB |
| 1M x 5 | 220 MB | 143 MB |
| 1M x 50 | 1,645 MB | 882 MB |

Every estimate is identical to the last digit before and after, and the timings did not move.
The 4.1x was three avoidable copies stacked on the matrix: the materialised scan result stayed
live until the function returned, the raw per-column buffers stayed live until every column had
been encoded, and the staged features stayed live until every one had been copied into the
matrix. Releasing each at the point it goes dead removed all three. The benchmark gates on the
peak now, so a reintroduced copy fails rather than being noticed.

The remaining 2.2x is a floor, not an oversight. Encoding needs statistics over a whole column
before it can write a single value of it — the median that fills NULLs, the mean and standard
deviation that standardise — so the raw column and the encoded one are unavoidably resident
together. Reordering the passes moves the peak around without lowering it: staging the encoded
features costs the same as keeping the raw buffers alive to write them. The only way past it is
to scan the source twice, once for statistics and once to fill the matrix, and that is unsafe
here — `USING SAMPLE` and any other non-deterministic source would hand the second pass a
different table than the first. The default budget is a *fraction* of DuckDB's limit precisely
to leave room for the 2.2x.

### Exit gate

- 1M rows, 50 covariates, AIPW, under 30 seconds on a laptop core count.
- `do_ate_by` over 1000 groups scales roughly linearly.
- No memory growth across repeated invocations (leak check under valgrind/ASan).

---

## Phase 9 — Ship

**Goal:** `INSTALL duckdo FROM community; LOAD duckdo;` works for a stranger.

### Work

1. **Community extension submission.** A `description.yml` PR to `duckdb/community-extensions`, following the established shape:
   ```yaml
   extension:
     name: duckdo
     description: Causal inference inside DuckDB — treatment effects, interventions
       and counterfactuals in SQL, with classical estimators and causal foundation
       models (CausalPFN, Do-PFN, CausalFM) on ONNX Runtime
     version: '2026.MM.DD'
     language: C++
     build: cmake
     license: MIT
     maintainers: [maxdemarzi]
     excluded_platforms: "wasm_mvp;wasm_eh;wasm_threads;windows_amd64_mingw;linux_amd64_musl"
   repo:
     github: maxdemarzi/DuckDo
     ref: <commit sha>
   docs:
     hello_world: |
       -- a self-contained example that needs no download
     extended_description: |
       -- the surface, the models, the licenses
   ```
   The `hello_world` must run with **no model download** — a classical AIPW estimate on inline `VALUES`. A first impression that requires a 300MB fetch is a first impression most people never have.
2. **Documentation.** Function reference; a "your first causal question" tutorial; an assumptions guide written for analysts rather than statisticians; a model catalog with licenses and the Do-PFN attribution notice; a page of worked examples (A/B test with non-compliance, observational pricing, churn intervention).
3. **Honest benchmark page.** DuckDo classical vs DuckDo CFM vs Python EconML/DoWhy/CausalPFN, on the standard benchmarks, with the losses shown.
4. **Reproducibility statement.** What is deterministic, what is not, and how to pin it.
5. **No telemetry.** Causal analysis runs on sensitive data. Shipping zero telemetry is both the right default and a genuine differentiator worth stating explicitly.

### Exit gate

- Extension accepted and installable from the community repository on every non-excluded platform.
- A user who has never seen DuckDo gets a correct ATE within five minutes of the docs landing page.

---

## Phase 10 — Frontier (post-1.0)

Roughly in order of value per unit of effort:

1. ~~**Continuous treatments** — dose-response curves.~~ **DONE**: `do_ape` recovers a known average partial effect of 2.0 to 2.0019 where a naive slope gives 2.75, and `do_dose_response` traces the curve on a quantile grid. Multi-valued *categorical* treatments are still open.
2. ~~**Panel data and difference-in-differences**~~ **DONE** for the core: `do_did` computes group-time effects in the Callaway–Sant'Anna sense and `do_event_study` exposes the pre-trend check. Two-way fixed effects was deliberately *not* implemented — it misweights under staggered adoption. Synthetic control and covariate-conditional (doubly-robust) DiD are still open.
3. **Longitudinal / time-varying treatment** — g-methods, marginal structural models; the Causal Longitudinal PFN line of work as a model backend.
4. **Survival outcomes** — time-to-event treatment effects.
5. ~~**Mediation analysis** — natural direct and indirect effects.~~ **DONE**: `do_mediate`
   returns the natural direct and indirect effects and the proportion mediated, with a row-level
   bootstrap so the three intervals are mutually consistent and the total is exactly direct plus
   indirect. It recovers a known 1.5 / 1.6 split as 1.518 / 1.613, and its total agrees with
   `do_ate` on the same data to four decimal places through an entirely separate code path.

   The treatment-mediator interaction is carried, and that is the substantive choice. On a DGP
   with a true `q3` of 0.6, omitting it returns 2.41 / 2.22 against a truth of 1.80 / 2.80 while
   the *total* stays correct at 4.63 — a wrong split under a right total, which is the failure
   mode that survives a sanity check. `proportion` is refused rather than reported when the total
   straddles zero. Sequential ignorability is stated in every result: randomisation buys the
   treatment-outcome and treatment-mediator arms of it and leaves mediator-outcome confounding
   untouched, which is the most common error in applied mediation. Graph-based verification that a
   named mediator actually *is* one is still `do_validate`'s job rather than built in here.
6. **Causal discovery** — `do_discover()` proposing a DAG from data. Deliberately last: it is the feature users most want and the one most likely to produce confident nonsense. If it ships, it ships with loud uncertainty and a required review step.
7. **Federated / multi-table estimation** — effects across joins without materializing the join.

---

## 6. Cross-cutting concerns

### Testing strategy

Four tiers, all in CI:

1. **SQL tests** (`test/sql/`) — the DuckDB-native harness; every function, every error path.
2. **Numerical golden tests** — fixed fixtures with reference values from EconML/DoWhy/PyTorch, tolerance-checked. Regenerating a golden value requires an explicit script run and shows up in the diff.
3. **Property tests** — encoder round-trips, d-separation invariants, estimator equivariance under row permutation and column reordering.
4. **Benchmark validation** — IHDP/Lalonde/ACIC/Twins, run on a schedule rather than per-commit, tracked over time.

Synthetic data with known ground truth is the backbone. A generator that emits DGPs with a specified confounding strength, overlap, and effect heterogeneity — and reports the true ATE and CATE — makes almost every other test cheap to write.

### Licensing

- Extension code: MIT (already in `LICENSE`).
- **Zero model weights in the repository.** Only weight-free graphs and random-init test fixtures.
- Per-model license reported in `do_list_models()` with a `commercial` boolean, gated behind `duckdo_accept_model_license` for anything non-permissive.
- **Do-PFN is CC BY 4.0** — attribution is required downstream, so it must be surfaced in the docs and the model catalog, not buried.
- CausalFM's license must be verified at integration time; the table in section 1 marks it as unconfirmed.

### Security and privacy

- No telemetry, stated in the README as a feature.
- No data leaves the process. Weight downloads are the only network activity, they are user-initiated via `do_download`, and they never carry query data.
- Model files are checksum-verified before load.
- User-supplied ONNX via `do_register_model` is a code-execution surface: document it as trusted-input-only, and keep it off by default in any hosted context.

### Naming and compatibility

- Full `duckdo_*` names, short `do_*` aliases — pending the Phase 0 grammar spike.
- Result schemas are part of the public API from 0.5.0 onward; adding columns is a minor version, changing or removing them is a major.
- Settings names are stable from 0.5.0.

---

## 7. Risk register

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | `do_` prefix collides with SQL grammar | High | Medium | Phase 0 spike; `duckdo_*` primary fallback |
| 2 | CFM checkpoints resist ONNX export | High | Medium | Start with Do-PFN (7.3M params); ship what exports; classical path is independent |
| 3 | CFMs do not beat classical estimators on real data | Medium | Medium | Phases 2–4 are valuable standalone; publish the comparison honestly |
| 4 | ~~ONNX Runtime static link breaks a platform~~ **retired** | Medium | Medium | There is no static link. The official prebuilt is fetched and linked shared, following `anofox_tabfm`. The live risk is now the Windows System32 `onnxruntime.dll` shadowing ours, mitigated by staging (next action 3) |
| 5 | Statistical errors in hand-rolled estimators | High | Medium | Grade every estimator against EconML/DoWhy before merge |
| 6 | Users misinterpret correlational output as causal | High | High | Estimand + estimator + warnings in every result row; diagnostics are first-class; docs lead with assumptions |
| 7 | Community-extension size limit exceeded | Medium | Low | Weight-free graphs; CPU-only flavour for the community build |
| 8 | Scope sprawl stalls the project before 1.0 | High | High | Phases 2–4 are shippable without any of Phases 5–7; ship 0.2.0 early and publicly |
| 9 | Upstream model licenses change | Low | Low | Weights are never redistributed; the license is reported at download time from the source |

---

## 8. Open questions

1. **Relation input**: native table in-out function, or `anofox_tabfm`-style string relation names? Phase 0 spike decides; the native form gives better bind-time errors if it can see column types.
2. **Do we ship gradient-boosted trees as a base learner**, or stay with regularized GLMs through 1.0? Decide from Phase 2 benchmark results, not in advance.
3. ~~**How are graphs persisted?**~~ **ANSWERED: a plain table.** `duckdo_graphs` is created on first use, survives restarts, and is inspectable and backup-able like any other data. Catalog integration would have been tidier in principle and would have coupled DuckDo to internals that move between DuckDB versions.
4. **Should `do_ate` auto-select an estimator** based on diagnostics, or always require an explicit one? Auto-selection is friendlier and less auditable. Current lean: explicit default (`aipw`), with `estimator := 'auto'` available and loud about what it chose.
5. **Sampling weights and survey designs** — in scope for 1.0, or Phase 10?
6. **Multiple testing** across `do_ate_by` groups — report adjusted p-values by default, or leave it to the user? Current lean: report both, defaulting to flagging when the group count is large.

---

## 9. Immediate next actions

Phases 0–4, 7, 8 (partially) and 9 (partially) are done. What is next, in order:

1. ~~**Give the CFM path an interval.**~~ **DONE** via `ensemble := k`, the context-ensemble
   strategy this document already specified in Phase 6. On a DGP with a true effect of 3.0 the
   single-draw interval `[3.035, 3.048]` excludes the truth and the 8-draw interval
   `[3.007, 3.121]` contains it. What remains is the model's *own* parameter uncertainty, which the
   ensemble does not touch — CausalPFN's package ships a calibration routine DuckDo has not
   reimplemented.
2. **Export CausalFM.** Two of the three models from section 1 now run. CausalFM would add
   *model-based* front-door and IV estimation; the classical versions already exist as `do_iv` and
   `do_frontdoor`, so this is no longer a coherence gap, just an additional engine.
3. ~~**Statically link ONNX Runtime**~~ **DONE differently — the premise was wrong.** No official
   static build of ONNX Runtime is published, building one per platform is a multi-hour job, and
   the extension this roadmap named as its precedent (`anofox_tabfm`, risk 4) does not static-link
   either: it fetches the official prebuilt and links it shared. So does DuckDo now.

   The actual blocker was never linkage. It was that the model path only built if you unpacked an
   archive by hand and passed `-DDUCKDO_ONNXRUNTIME_ROOT`, and on Windows only ran if you then
   copied `onnxruntime.dll` next to the binaries yourself — an undocumented manual step that had
   in fact been done by hand in this repo. `-DDUCKDO_WITH_ONNX=ON` now fetches the pinned release
   for the target platform, links it, and stages its DLLs. `DUCKDO_ONNXRUNTIME_ROOT` still works
   as an override, `DUCKDO_ORT_URL` points at a mirror where GitHub releases are unreachable, and
   the default build is still dependency-free.

   The DLL staging is not cosmetic. Windows ships its own `C:\Windows\System32\onnxruntime.dll`
   with Windows ML — it is present on this machine — and it is old enough to return a null
   `OrtApi` for the `ORT_API_VERSION` DuckDo compiles against. It does not fail to load; it
   crashes on the first inference. The loader searches the host executable's directory first, so
   staging our copy there is what makes ours win.
4. **Consider retiring or demoting Do-PFN.** CausalPFN is better on every axis measured here —
   licence, covariate budget, dynamic context, and no shrinkage. Do-PFN remains interesting only
   for its explicit treatment of unobserved confounding, which DuckDo does not currently exploit.
5. **Ensemble over covariate subsets** where a model's budget binds, instead of keeping only the
   most outcome-correlated.
6. ~~**Spill past `duckdo_max_memory`.**~~ **DONE, by enforcing rather than spilling.** The
   setting is now a real ceiling with an actionable message, and the measured peak for the 1M x 50
   frame fell from 1,645 MB to 882 MB against a 400 MB matrix. Spilling was considered and
   rejected: see Phase 8, which also shows why the residual 2.2x is a floor rather than an
   oversight.
7. ~~**Persist graphs**~~ **DONE**: graphs live in a `duckdo_graphs` table, so they survive a restart and are inspectable as ordinary data. Open question 3 is answered — a plain table beat catalog integration, which would have coupled DuckDo to internals that move between DuckDB versions.
8. **Broaden the cross-check further.** It now covers all ten IHDP replications available from the
   CEVAE mirror (mean |ATE error| 0.137, mean PEHE 2.23) and Lalonde NSW against its experimental
   benchmark (1794.3, with `dml` at 1759.3). The canonical 1000-replication IHDP set is not at that
   source; an ACIC subset is still open.
9. **Host the exported graphs** so `do_download` can fetch them, rather than requiring every user to
   run the export script.
10. **Submit** the `description.yml` PR to `duckdb/community-extensions`.
