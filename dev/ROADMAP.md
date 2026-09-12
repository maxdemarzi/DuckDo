# DuckDo — Causal Foundation Models in DuckDB

**An implementation roadmap, v1 (2026-09-08)**

> **Progress: every phase is implemented, built and tested, Phase 10 included.** 810 assertions
> in 29 test files pass against DuckDB v1.5.4. Three more files run when `DUCKDO_MODEL_DIR` points
> at exported model weights. There is also an EconML/DoWhy cross-check on IHDP, and PyTorch parity
> gates on the exported ONNX graphs. **CausalPFN and Do-PFN both run end to end inside DuckDB.**
> Section 9 lists what each Phase 10 item still leaves open.

---

## 1. Thesis

SQL answers *what happened*. ML extensions like [`anofox_tabfm`](https://duckdb.org/community_extensions/extensions/anofox_tabfm) and `infera` answer *what will probably happen*. Nothing in the DuckDB ecosystem answers **what would happen if I did X** — the interventional question, `P(Y | do(X = x))`.

DuckDo closes that gap. It makes DuckDB an in-process causal inference engine: treatment effects, interventional predictions, counterfactuals, and policy evaluation expressed as ordinary SQL over ordinary tables, with no Python, no training loop, and no data leaving the process.

The timing is favourable. A class of **causal foundation models** (CFMs) — transformers pretrained on synthetic structural causal models that estimate treatment effects on unseen datasets by in-context learning — matured through 2026 and now has three credible open implementations:

| Model | Setting | Params | Weights | License | Notes |
|---|---|---|---|---|---|
| **CausalPFN** ([repo](https://github.com/vdblm/CausalPFN), [HF](https://huggingface.co/vdblm/causalpfn)) | Backdoor (ignorability) | — | Hugging Face | CausalPFN License 1.0 (Apache-2.0's terms under its own name) | CATE + ATE with calibrated intervals. Best default. |
| **Do-PFN** ([repo](https://github.com/jr2021/Do-PFN)) | Non-identifiable prior | 7.3M | GitHub | **none stated**: no LICENSE file, so all rights reserved (this table said CC BY 4.0, which nothing upstream grants) | Full interventional distribution; expresses uncertainty under hidden confounding. Tiny. |
| **CausalFM** ([repo](https://github.com/yccm/CausalFM)) | Backdoor / front-door / IV | — | GitHub | Apache-2.0 (verified) | **Evaluated and not exported**: see section 9, item 2. |

The reference architecture for running such models inside DuckDB already exists and is proven: `anofox_tabfm` statically links ONNX Runtime, compiles **weight-free** ONNX graphs into the extension binary, and downloads weights from Hugging Face into a user-owned cache on first use. DuckDo should borrow that shape rather than reinvent it.

### Positioning in one line

> `SELECT` tells you what happened. `do_ate()` tells you what your last change was worth. `do_predict(intervention := ...)` tells you what the next one will be worth.

### What DuckDo is *not*

- Not a causal **discovery** engine you should trust unreviewed. `do_discover` exists, post-1.0, as a *proposer*: it reports how fragile every edge is, and `do_graph_create` refuses its output until a person has reviewed it and oriented what the data could not. Discovery is never a default and never feeds an estimate directly.
- Not a replacement for a statistician. Every estimand carries assumptions. DuckDo's job is to make those assumptions **checkable in SQL**, not to hide them.
- Not a general ML inference extension. If you want zero-shot classification, use `anofox_tabfm`. DuckDo should coexist with it cleanly.

---

## 2. Design principles

These are the tie-breakers for every decision below.

1. **Useful before it is clever.** Phase 2 ships classical estimators that require zero downloads and no ONNX. A user gets real value from `INSTALL duckdo` before a single foundation model is involved. The CFM path is an upgrade, not a prerequisite.
2. **The estimand is explicit.** Every result carries the estimand (`ATE`/`ATT`/`CATE`), the estimator used, the identification strategy assumed, and the sample size. No result is a bare float.
3. **Assumptions are first-class queryable objects.** `do_balance`, `do_overlap`, `do_refute`, `do_sensitivity` are not an afterthought section of the docs; they are functions that return rows, so they can be asserted on in dbt tests and CI.
4. **Weights never live in the extension.** The repo and the binary contain zero weight bytes. Weights are downloaded by the user, into a user-owned cache, under the model's own licence. DuckDo hosts an exported copy only where that licence permits redistribution. CausalPFN's does, and its export is published with the licence text and a notice of every changed file, pinned to one revision. Do-PFN's upstream states no licence, so it is never redistributed. Keeping weights out of the binary is both a licensing requirement and the only way to stay inside community-extension size budgets.
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
| 5 | Inference runtime | 0.5.0 | yes (models opt-in) | **DONE** — ONNX Runtime linked behind a build flag, model catalog, session cache, and a parity gate the export refuses to pass. For Do-PFN, logits must agree within 1e-4 (measured 4.3e-06 to 8.1e-06). For the split CausalPFN graphs, the ATE must agree within 0.01 and the CATE correlation must be at least 0.999 (measured 0.0009 and 0.9994), because fused versus decomposed attention kernels leave a raw-output gap of 0.066 |
| 6 | CFM estimators | 0.6.0 | opt-in | **DONE for CausalPFN and Do-PFN** — CausalPFN matches AIPW (3.061 vs 3.063, truth 2.981) with CATE correlation 0.9995 and no shrinkage; Do-PFN shrinks to 2.631, reproduced and reported. CausalFM evaluated and declined (section 9, item 2) |
| 7 | The `do()` surface | 0.7.0 | **yes** | **DONE on classical backends** — `do_predict`, `do_counterfactual`, `do_policy_value`, `do_uplift`, `do_optimal_policy`. Gains a CFM engine in phase 6 |
| 8 | Scale and performance | 0.8.0 | — | **DONE**, with spilling rejected by design. 1M × 50 AIPW runs in 14.2 s against a 30 s gate, down from 101 s. `do_ate_by` is built from one scan: 1,000 groups in 0.88 s, down from 9.44 s, and bit-identical. CausalPFN caches its context and runs on CUDA. Nested parallelism is switched off per thread. A leak check shows memory growth levelling off. CI benchmark and sanitizer jobs are written but not yet run |
| 9 | Ship | 1.0.0 | — | **READY, not submitted** — `description.yml`, the function reference, tutorial, worked examples, assumptions guide, benchmarks and reproducibility statement are written and checked by `scripts/check_docs.py`, including the `hello_world` a stranger runs first. A documentation site builds with `mkdocs build --strict`, ready to publish once GitHub Pages is switched on. The submission PR is held by the maintainer's choice |
| 10 | Frontier | post-1.0 | — | **DONE** — continuous and multi-valued treatments, panel/DiD (plain and doubly robust), synthetic control, longitudinal MSMs, survival (RMST, time lost under competing risks, and survival under time-varying treatment), mediation, causal discovery, cluster-robust estimation across one-to-many joins, and federated pooling across sites (`do_ate_pool`) landed. All seven items are done; what each leaves open is listed under Phase 10 below |

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
- **`do_cate`'s interval was measurably wrong and is now measurably better — but the first
  measurement of it was noise.** The doubly-robust pseudo-outcome's variance scales with `1/e(x)`,
  so assuming it constant undercovers. Properly measured at 40 replicates, coverage runs
  0.948 / 0.937 / 0.909 (± ~0.015) across randomised, confounded and strongly-confounded DGPs
  without the sandwich, and 0.948 / 0.948 / 0.944 with it. The *gradient* is the evidence: a
  constant-variance assumption fails exactly where the variance stops being constant.

  The original figures here — 0.896 before, 0.945 after — came from a 12-replicate run, and a
  replicate's coverage ranges from 0.56 to 1.00 depending on where its nuisance fits land. At that
  rep count the harness moved by 0.12 between runs, which is enough to invent a finding. The
  conclusion held; the numbers did not. `scripts/coverage_check.py` now defaults to 40 replicates
  and prints a standard error, a range, and a line saying that a mean without one is not a
  measurement.

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
   - **CausalFM** — one checkpoint per identification setting, so the registry must key on `(model, setting)`. *Evaluated and not exported; see section 9, item 2.*
   - Export scripts live in `scripts/export/`, are pinned to exact upstream commits, and emit a manifest recording checkpoint hash, opset, and dynamic-axis names.
3. **Weight-free graphs compiled into the binary**; weights downloaded on demand.
4. **Download and cache.** `do_download` over `httpfs`, into `~/.cache/duckdo` (override via `duckdo_cache_dir`), with checksum verification, resume, and a clear message if the cache is not writable. Gated repositories use standard DuckDB secrets:
   ```sql
   CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_xxx', SCOPE 'https://huggingface.co');
   ```
5. **License gating.** `do_list_models()` reports each model's license and a `commercial` boolean. Non-commercial weights require `SET duckdo_accept_model_license = true` before download. Do-PFN's upstream states **no licence at all**. The catalog reports that as it is, marks the model non-commercial, and attaches a warning to every result that uses it. This line once said CC BY 4.0, which nothing upstream grants.
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
5. **Model routing.** `model := 'causalpfn' | 'do_pfn'` (`'causalfm'` was planned and declined, see section 9), with the identification setting checked against the registered graph when one exists — asking for a backdoor model on a graph with an unblocked backdoor path is an error worth raising.
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

1. ~~**Grouped estimation**~~ **DONE.** `do_ate_by` existed, but it did not scale. Each group
   re-read the whole relation, several times over, through a text-cast predicate, so the cost
   grew with the square of the group count:

   | groups x 200 rows | before | after |
   |---|---|---|
   | 100 | 0.42 s (4.2 ms/group) | 0.16 s (1.6 ms/group) |
   | 300 | 1.75 s (5.8 ms/group) | 0.39 s (1.3 ms/group) |
   | 1,000 | 9.44 s (9.4 ms/group) | 0.88 s (0.9 ms/group) |

   The relation is now copied once into a private in-memory database, sorted by an integer group
   id with each group's rows in their original order. Each group then reads one contiguous range.
   A temporary table would not do, because every frame is built on a fresh connection, and
   temporary tables are per-connection. An attached database is visible to all of them. 1,000
   groups of 2,000 rows each, 2M rows in all, take 1.36 s. All 43 values of a full-precision
   baseline are bit-identical to the old path: NULL groups, two grouping columns, a group too small
   to estimate, clustering, and default covariates.

   Running groups in parallel was built, measured, and removed. 1,000 AIPW groups took 0.88 s
   against 0.93 s on one thread, and 300 `t_learner` groups 0.40 s against 0.42 s. A group's cost
   is the queries that build its frame, not its arithmetic. The worker threads also read settings
   through the binding query's client context, which is not made to be shared. Five percent was
   not worth that.
2. ~~**Parallelism**~~ **DONE for bootstrap replicates**, which were the remaining serial cost:
   `do_mediate`, `do_rmst` and `do_frontdoor` now run their replicates in parallel with the
   row-block threading switched off inside each, since nesting the two oversubscribes badly. Each
   replicate seeds from `(seed, rep)` rather than a shared stream, so the answer cannot depend on
   which thread reached which replicate. `do_mediate` on 200k x 5 went 3.7 s to 1.6 s; on 1M x 50
   it went 194 s to 141 s, where the limit is memory bandwidth rather than cores.

   The same change closed a hole in design principle 7. The row-block count used to follow the
   thread budget, and floating-point addition is not associative, so the same query on the same
   seed returned `3.0031935719077567` on one thread and `3.0031935719077549` on sixteen. The block
   count is now a function of the data alone and the blocks are reduced in index order, so results
   are bit-identical at any thread count — and the work-stealing dispatch that replaced the even
   split is also *faster*: 1M x 50 AIPW fell from 16.7 s to 14.2 s. Cross-fitting folds and CFM
   ensemble draws are still serial, and DuckDB's own task scheduler is still not used.

   Nested parallelism is now switched off per thread. It used to be switched off by saving the
   shared thread budget, setting it to 1 and restoring it afterwards. That was a plain global, so
   two connections estimating at once could interleave the save and the restore and leave every
   later query in the process on one thread, with nothing to say so. A thread-local flag has no
   such interleaving, and the full-precision baselines are bit-identical across the change.

   Cross-fitting folds stay serial, and that is a decision now rather than a gap. A large fit
   already spreads its row blocks across every thread, so running five folds side by side would
   cap it at five threads. A small fit has too little work for the folds to be worth
   parallelising. Groups in `do_ate_by` are where small fits come in numbers, and those now run in
   parallel (item 1).
3. ~~**Context caching** — for CFMs, encode a fixed context once and reuse it across query chunks.~~
   **DONE**, and not behind a setting: it is the only path. CausalPFN is exported as two graphs.
   The encoder turns the context into a key/value cache once. The decoder scores every query
   chunk, and both arms, against that cache. On 8,000 rows the default went from 126 s to 31.3 s
   with a bit-identical estimate (section 9, item 10).
4. ~~**Streaming and chunking**~~ — **partly done, and the other part was rejected.**
   `duckdo_max_memory` is now a real ceiling: the encoded matrix's size is computed before
   it is allocated, and exceeding the budget raises an error naming the setting and the row
   count that would fit. The peak was also measured and cut. What was *not* built is spilling.
   Every estimator indexes the matrix by row, cross-fitting reads it five times over and the
   bootstrap reads it in random row order two hundred times more; a spilled matrix would turn
   a fast refusal into an unbounded thrash. Refusing with an actionable message is the better
   failure, and it is also what section 6's guardrail rule already specified.
5. ~~**GPU flavours**~~ **DONE for CUDA**; ROCm and bf16 are not built, for reasons below.
   `-DDUCKDO_ORT_FLAVOUR=cuda12` fetches ONNX Runtime's CUDA 12 package in place of the CPU one.
   `duckdo_device` ('cpu' or 'cuda') and `duckdo_gpu_precision` ('fp32' or 'tf32') choose where
   and how. `do_devices()` attaches the CUDA provider for real rather than trusting its listing,
   because the provider library, CUDA and cuDNN load only when a session asks for them. It then
   says exactly what is missing when that fails: a named DLL (error 126), or a version mismatch
   (error 127).

   CausalPFN on an RTX 3060, 8,000 rows by 5 covariates, each run in a fresh process, as medians of
   three:

   | device | time | the three runs | estimate |
   |---|---|---|---|
   | CPU | 40.0 s | 36.6, 40.0, 49.8 | 3.008895 |
   | CUDA fp32 | **5.1 s** | 5.1, 5.1, 5.0 | 3.008895 |
   | CUDA tf32 | 4.7 s | 4.8, 4.6, 4.7 | 3.008894 |

   That is about 7.8 times faster. The CPU runs spread over 13 seconds with identical code, which is
   why single runs are not quoted. fp32 on the GPU differs from the CPU by 1.75e-7 on a 2,000-row
   table and repeats bit for bit. tf32 differs by 2.44e-5, about 140 times as much, to save 0.4 s.
   ONNX Runtime turns tf32 on by default, so 'fp32' switches it off explicitly; otherwise "fp32"
   would quietly mean tf32. The CPU stays the default: a result should not move because a GPU happens
   to be present.

   Five things were found on the way:
   - **CUDA was not deterministic until told to be.** Ten repeats of one query in one process gave
     four different estimates, spread over 2e-8, while the first run in every fresh process
     matched. An earlier check that two runs agreed had matched by luck; `gpu.test` caught it. CUDA
     sessions now run with deterministic compute and a fixed cuDNN algorithm rather than one
     benchmarked on the first runs. After that, ten repeats and three fresh processes all give the
     same estimate at both precisions, for about 0.3 s.
   - **ONNX Runtime 1.29's Windows CUDA package is built against CUDA 12.8.** PyTorch's bundled
     12.6 libraries load and then fail with error 127, because `cublasLt` 12.6 lacks functions
     the provider imports. A diagnosis comparing every import against every export confirmed
     that 12.8's libraries have them all.
   - **Do-PFN crashes the process on CUDA.** ONNX Runtime's CUDA kernel warns that the ScatterND
     its graph uses "only guarantees to be correct if indices are not duplicated", and the run
     then died with a segmentation fault. So `do_pfn` refuses `duckdo_device = 'cuda'` before
     touching the GPU.
   - **A CUDA build on the CPU changes nothing.** The CUDA package's CPU path returns
     3.044465189457, the same as the CPU package, and both suites pass without any CUDA library on
     the path.
   - **bf16 and fp16 are refused.** They need graphs exported at that precision, and DuckDo
     exports fp32 graphs.

   ROCm is not built. ONNX Runtime publishes no prebuilt ROCm package for Windows, and there is no
   AMD GPU here to test one on. Shipping an untested device path would be worse than naming the
   gap, so `do_devices()` reports it as not built, with the reason.

   `test/sql/gpu.test` covers the GPU path, and runs only with `DUCKDO_CUDA_TESTS` set.
6. ~~**Benchmark suite in CI**~~ **WRITTEN, not yet run.** `scripts/benchmark.py` gates on peak resident
   memory as well as seconds, and generates each shape outside the measurement, so both numbers
   describe the estimator rather than the generator. `.github/workflows/Checks.yml` runs it weekly
   and on pushes that touch the source, uploads the JSON so the numbers can be tracked across
   commits, and runs the leak check and the full test suite under AddressSanitizer and UBSan. The
   time gate is loosened to 120 s there, because the roadmap's 30 s is for a laptop core count and
   a shared runner has four cores. The memory gates hold on any machine. None of this has run yet:
   GitHub Actions runs only once the branch is pushed, and it has not been.

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

- ~~1M rows, 50 covariates, AIPW, under 30 seconds on a laptop core count.~~ **Met**: 14.2 s.
- ~~`do_ate_by` over 1000 groups scales roughly linearly.~~ **Met, and better than linear**: the
  cost per group falls from 1.6 ms at 100 groups to 0.9 ms at 1,000 (item 1).
- ~~No memory growth across repeated invocations.~~ **Met as far as this machine can measure.**
  `scripts/benchmark.py --leak` runs a mixed workload of every estimator family, `do_ate_by`,
  `do_cate` and the bootstrap, at 300, 1,000 and 3,000 calls. The peak grew 25.4 KB per call from
  300 to 1,000 calls and 7.4 KB per call from 1,000 to 3,000. That is warm-up levelling off; a leak
  holds flat. `do_ate_by` alone reached a plateau near 110 MB by 3,000 calls. Most of the early
  growth is DuckDB's own: a plain SQL loop that attaches and detaches a database grows the same way
  and levels off the same way. The precise check, the full suite under AddressSanitizer with
  LeakSanitizer, is written into CI and has not run yet.

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
       models (CausalPFN, Do-PFN) on ONNX Runtime
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
2. **Documentation.** ~~Function reference~~ **DONE** (`docs/FUNCTIONS.md`); ~~a "your first
   causal question" tutorial~~ **DONE** (`docs/TUTORIAL.md`, built entirely from real output —
   naive $18.67 against a true $8.00, then diagnostics, refutation, heterogeneity and a shippable
   targeting rule); ~~an assumptions guide written for analysts rather than statisticians~~
   **DONE** (`docs/ASSUMPTIONS.md`, organised by whether the data can settle each assumption at
   all); ~~a model catalog with licenses and the Do-PFN attribution notice~~ **DONE** (in the
   function reference and `do_list_models()`); ~~a page of worked examples (A/B test with
   non-compliance, observational pricing, churn intervention)~~ **DONE** (`docs/EXAMPLES.md`) —
   all three, and the naive answer is wrong in a different direction in each: too big (0.1749
   against 0.12), the wrong sign (+1.49 against -2.5), and too small (-0.0248 against -0.0896).

   `scripts/check_docs.py` keeps them true. The tutorial and the worked examples are executed
   block by block in document order; the README, assumptions guide and function reference are
   parsed rather than executed, because they use illustrative fragments against tables that do
   not exist. The two tiers are named in the output so nobody mistakes the weaker one for the
   stronger. A deliberately broken query was planted to confirm the checker fails when it
   should.

   The checker now covers the first thing a stranger runs as well. The `hello_world` block in
   `description.yml` is what the community page shows verbatim. It is executed on a fresh database
   with no download, and the interval it prints has to contain the true effect its own comment
   promises. Today it prints [1.971, 2.027] around a true 2.0. That is the "correct ATE within five
   minutes" exit gate, checked by a script instead of asserted.

   A documentation site is built from the same files, unchanged. `mkdocs.yml` and
   `scripts/mkdocs_hooks.py` make the README the home page and point links that leave `docs/` at
   the file on GitHub. The build runs with `--strict`, so a broken link fails it.
   `.github/workflows/Docs.yml` publishes it to GitHub Pages on a push to `main`. That needs Pages
   switched on once in the repository settings, which is the maintainer's call, and nothing is
   published until then.
3. ~~**Honest benchmark page.**~~ **DONE** (`docs/BENCHMARKS.md`, from `scripts/bench_headtohead.py`,
   every method in its own process against identical rows). The headline reverses what this
   repository previously claimed: on IHDP, `model := 'causalpfn'` posts a mean PEHE of **0.416**
   against 2.161 for the best classical method measured, and lands within 0.007 of the CausalPFN
   package on both metrics across ten replications (0.0041 and 0.0066) — the strongest evidence yet that the ONNX
   export is faithful. The README's "the foundation model does not beat AIPW" was true only of the
   linear DGP it was measured on, and has been corrected.

   The losses are on the page rather than in a footnote. On a step-function DGP EconML's
   `CausalForestDML` beats DuckDo's classical `do_cate` on PEHE by **0.213 against 0.992**, because
   regularised GLMs cannot fit a cliff. `dml`'s mean IHDP error of 0.786 against a median of 0.160
   is one replication where it misses by 6.4. And the CFM path was **8× slower than the same model
   in Python**: the single exported graph re-encoded the whole 4,096-row context for every 512-row
   query chunk. That is fixed by item 10 below. The default path went from 126 s to 31.3 s on 8,000
   rows (median of three runs), against the package's 22.5 s. At chunk 2048 it takes 23.0 s, level with
   the package, and the estimate is bit-identical.
4. ~~**Reproducibility statement.**~~ **DONE** (`docs/REPRODUCIBILITY.md`, each claim a case in
   `scripts/reproducibility_check.py`, which fails if the behaviour stops matching the prose).
   Identical to the last bit across repeat runs, `duckdo_threads` 1/2/8/auto, DuckDB `threads` 1/4,
   `duckdo_query_chunk` 256 vs 4096, and repeated foundation-model runs.

   The case that failed was row order, and finding out *why* mattered: the first explanation was
   floating-point non-associativity, and it was wrong by eight orders of magnitude. `AssignFolds`
   seeded a shuffle of row **positions**, so a permuted table put different rows in different folds
   — worth 0.25% of one standard error for `regression`, 2.1% for `aipw` and 6.2% for `ipw`. That is
   now **fixed**, see item 9 below; the residual is 2e-15 relative and the check asserts that bound.
   The first harness for this was itself wrong: ordering the generating query by `random()` draws
   from the same stream that fills the columns, so it compared *different data* and made the result
   look far worse than it is.

   A later pass found two functions the check had never covered, and both failed it. `do_balance`
   and `do_overlap` take no outcome, and the frame read that missing outcome from a NULL vector's
   data slots, whatever memory held. The canonical order sorts on the outcome, so their propensity
   folds moved on every call. It is fixed; the other functions are bit-identical across the fix,
   and the check now runs a no-outcome case.
5. ~~**No telemetry.**~~ **DONE and verified rather than asserted.** The extension contains no HTTP
   client and no socket code. `src/` holds three URLs. Two are attribution strings naming where each
   model came from, which `do_list_models()` prints and nothing fetches. The third is the pinned
   address of the hosted CausalPFN release, which is fetched only when someone calls
   `do_download('causalpfn')` with no source. `docs/REPRODUCIBILITY.md` ships the one-line grep that
   checks this. `do_download` reads through DuckDB's file system, so DuckDo itself still opens no
   connection. It verifies every file against a compiled-in SHA-256 before installing it. A build
   without ONNX Runtime, the one the community repository ships, refuses the default source
   outright. That build cannot run a model, so its only network request would be a user-typed URL.

### Exit gate

- Extension accepted and installable from the community repository on every non-excluded platform.
- A user who has never seen DuckDo gets a correct ATE within five minutes of the docs landing page.

---

## Phase 10 — Frontier (post-1.0)

Roughly in order of value per unit of effort:

1. ~~**Continuous treatments** — dose-response curves.~~ **DONE**: `do_ape` recovers a known average partial effect of 2.0 to 2.0019 where a naive slope gives 2.75, and `do_dose_response` traces the curve on a quantile grid. Multi-valued *categorical* treatments are **DONE** too: `do_ate_levels` contrasts every level against a reference with a multi-arm AIPW over the whole population, with folds stratified within every level. The binary `AssignFolds` would have left every row of a third level in fold zero, so that level would never be held out. On a three-arm DGP it returns 1.472 [1.375, 1.568] against a truth of 1.5, and -0.992 against -1.0, where the naive differences are 4.185 and -2.717. Building it exposed a bug. On a column with three values, `treated := / control :=` was refused by the two-level probe before the labels were ever consulted, and the refusal's own message told the caller to pass exactly those labels. So the pairwise route had never worked on the columns it existed for. It works now, and on the same table it returns 2.467: the truth among units that received A or B, a full unit from the population's 1.5.
2. ~~**Panel data and difference-in-differences**~~ **DONE** for the core: `do_did` computes group-time effects in the Callaway–Sant'Anna sense and `do_event_study` exposes the pre-trend check. Two-way fixed effects was deliberately *not* implemented — it misweights under staggered adoption. Covariate-conditional DiD is **DONE**: `covariates :=` makes every group-time cell doubly robust (Sant'Anna–Zhao). The covariates are read at each unit's first observed period. The cells come from a shared routine whose no-covariate branch keeps the old arithmetic line for line, and `do_did` and `do_event_study` without covariates return the same bits as before, confirmed by recording them at full precision before the change and diffing after. On a panel where high-x units trend faster and are likelier to be treated, with a true effect of 2.0, plain DiD gives 3.877 with a pre-trend of -0.982. The doubly robust version gives 1.841 [1.600, 2.082] with a pre-trend of -0.049. Building it exposed a silent-ignore bug: both panel functions had always accepted `covariates :=` through the shared parameter list and never read it. Synthetic control is **DONE** too. `do_synth` returns one row per treated unit, with its donor weights as a `MAP`, pre- and post-period RMSPE, and an in-space placebo p-value. `do_synth_path` gives the actual-versus-synthetic path to plot. The weights solve a simplex-constrained least-squares problem: accelerated projected gradient finds roughly the right support, and an exact KKT solve on that support finishes the job. The last step exists because the cross-check against scipy's SLSQP (`scripts/synth_check.py`) failed without it. The objective came out 1e-6 relative above scipy's, with weights up to 7e-4 apart; statistically nothing, but visible in the printed weights. With the polish they agree to 1.6e-8 in the weights and 6e-14 in the objective. The check uses fewer donors than pre-periods on purpose, which makes the problem strictly convex, so any disagreement means a bug rather than a tie between optima. On a one-factor panel where the treated unit trends faster than the average donor, `do_did` gives 6.277 against a true 5.0, while `do_synth` gives 5.053 and ranks first of 21 placebo runs. `do_synth` refuses `covariates :=` rather than ignoring it. It is classic Abadie with no intercept, so a treated unit outside the donors' range cannot be matched; the output counts the pre-periods where that happens.
3. ~~**Longitudinal / time-varying treatment**~~ **DONE for the core**: `do_msm` fits a marginal
   structural model by stabilised inverse-probability-of-treatment weighting. On a DGP where a
   time-varying confounder is itself affected by prior treatment — the case with no correct
   adjustment set — the truth is 2.0 per treated period, no adjustment gives 3.92, adjusting for
   every measured confounder gives 3.47, and `do_msm` gives 2.13 with an interval covering 2.0.

   Weight diagnostics are in the result rather than a log: `mean_weight` near 1 is the cheapest
   check on the treatment model, and `effective_n` shows what the weights cost. `truncate` is
   available and **off by default**, because on that same data it nearly doubles effective *n* and
   halves the interval while moving the estimate off the truth — every quality signal improving as
   the answer gets worse. A structural model beyond "cumulative periods treated" is **DONE**: `model := 'by_period'` fits one effect per period, and always against never as their sum. On a world where the first period is worth 2.0 and the second 1.0, it returns 2.073 and 1.193, and 3.266 against 3.0. The cumulative model's 1.655 per treated period has an interval that covers neither period's effect. The default output is byte-identical to before on four fingerprinted calls. `model := 'saturated'` is **DONE** too. It gives every treatment history its own mean, for up to six periods. Add an interaction worth 1.5 when both periods are treated, and it returns 5.721 for both against a truth of 5.5, while by_period's effect of period 1, 2.603 [2.445, 2.760], describes no regime anyone could follow. Building it corrected a claim `do_msm` had made since it shipped. Its warning said that treating the weights as known is conservative, and that a unit bootstrap would be tighter. In 300 simulated panels with weights near 100, coverage was 85-93% for every model, and the bootstrap did no better. With weights under 10 it was 97-99%. The warning now says so, and the estimates were unbiased throughout. Still open: causal longitudinal PFNs as a model backend.
4. ~~**Survival outcomes** — time-to-event treatment effects.~~ **DONE**: `do_rmst` returns the
   difference in restricted mean survival time, from inverse-probability-weighted Kaplan-Meier
   curves. On an exponential DGP with a closed-form truth of 0.9239 it returns 0.904 with an
   interval covering it, against 0.443 unadjusted — confounding hides more than half the benefit
   there.

   It deliberately reports no hazard ratio. A hazard ratio conditions on being still at risk, and
   treatment changes who is still at risk, so the comparison stops being causal after the first
   events even under randomisation. The horizon is treated as part of the estimand and defaults to
   the last time both arms still had 5% of their subjects at risk — the obvious default, the last
   time both arms were observed at all, lands where a handful of people remain and makes the
   restricted mean integrate over noise.

   Competing risks are **DONE**, as `do_rmtl`: the restricted mean time lost to one cause, from inverse-probability-weighted Aalen-Johansen curves, with the other causes treated as competing rather than as censoring. On a world with closed-form truths it returns -0.769 against -0.767, where censoring the competing cause gives -0.665 (the truth of a world without it) and no adjustment gives -0.521. The unweighted curves match an independent Python Aalen-Johansen to every printed digit, and `do_rmst`'s output is byte-identical to before. Time-varying treatment within the survival setting is **DONE** as `do_msm_rmst`. It compares always treated with never treated on expected event-free periods, by clone-censor-weight: inverse-probability weights for staying on each regime and for not dropping out, then weighted Kaplan-Meier. On a ten-period world with treatment-confounder feedback it returns 2.106 [1.836, 2.375] against a simulated truth of 2.298 within ten periods, and 0.456 [0.410, 0.502] against 0.439 within four. Without the confounder it returns 1.267. The test builds that world on one thread, because a recursive query's random draws otherwise follow thread scheduling, and the numbers here reproduce exactly. Checked against an independent Python version on the same panel, it agrees to 1e-14. Its unit-bootstrap interval covered the truth 93% of the time over 86 simulated panels, at horizons 4 and 10. `do_msm` is untouched: the change adds code and removes none.
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
6. ~~**Causal discovery** — `do_discover()` proposing a DAG from data.~~ **DONE**, under the two conditions this item set. It was deliberately last: it is the feature users most want and the one most likely to produce confident nonsense.

   `do_discover` runs PC-stable, with Fisher-z tests, v-structures and Meek's rules. It reports every edge with its bootstrap stability, including edges the full-data graph left out, so the uncertainty shows in both directions.

   The review step is enforced, not suggested. `do_discover_dot` writes a proposal that `do_graph_create` refuses until its `do_discover: unreviewed` line is deleted, and refuses again for each edge the data could not orient, written `a -- b`, until a person picks a direction.

   On a seven-variable linear-Gaussian world with a single collider, the full-data graph is exactly right, and the one pair no data can orient is left undirected. Orientation stability comes out at 0.88, not 1.0. A resample carries the sample's error on top of its own, so its tests reject independence more often than `alpha`, and sometimes a and b stay joined, which erases the collider every orientation depends on. At `alpha := 0.001` it is 0.98. The bootstrap therefore understates reliability, which is the safe direction for this feature, and the docs say so.

   Building it exposed two silent bugs in the DOT parser, both older than discovery:
   - Comments were tokenised, so every word of one became a node: `// the treatment` registered nodes `the` and `treatment`.
   - `a -- b` was dropped without a word: two nodes and no edge.

   Comments are now skipped, and an undirected edge is an error that names both ends. That same error is how the proposal's unoriented edges are made impossible to skip.

   FCI is **DONE**, as `algorithm := 'fci'`. It starts from PC's skeleton and separating sets, runs the possible-d-sep stage to remove edges a hidden common cause can fake, and orients with rules R1-R4 and R8-R10, assuming no selection bias, which makes its orientations complete under that assumption. Take a -> b <- u -> c <- d, with u unmeasured. PC's two colliders claim the b-c edge in opposite directions, and it warns of the conflict. FCI returns a o-> b <-> c <-o d, naming the hidden cause. On a world with no hidden cause it invents none. In the proposal, `<->` becomes a latent node with an arrow into each end, which `do_graph_create` and `do_identify` already understand, so hidden confounding found by discovery reaches identification.

   Tests for non-Gaussian data are **DONE**, as `test := 'rank'`: Fisher-z on normal scores, the nonparanormal of Liu, Lafferty and Wasserman (2009), which assumes only that monotone transforms of the variables are jointly Gaussian. On the same chain observed through exp, cube and sinh, Pearson returns six edges, two spurious and one reversed; ranks return the true four, and a further monotone transform changes nothing, to the bit. The default stays Pearson, and its output is byte-identical to before.

   FCI's rules R9 and R10 are **DONE**. They follow uncovered potentially directed paths, and each fired on a world built for it: a -> b -> m -> c <- a gives a --> c by R9 where the other rules leave a o-> c, and on a five-variable world found by searching random DAGs for one that needs R10, R9 settles two parents of c and R10 then settles the third. A Python copy of the rule stage run on oracle PAGs predicted both outputs, and the binary built before these rules gives exactly what that copy gives without them. Every earlier discovery output, PC and FCI, is byte-identical.

   Tests for binary and mixed data are **DONE**, as `test := 'mixed'`: the latent Gaussian copula for mixed data (Fan, Liu, Ning and Zou 2017), with latent correlations from Kendall's tau. The obvious version, Fisher's z on those correlations, is wrong: in simulation a nominal 1% test rejected true independences between binary columns 9-29% of the time. So each partial correlation gets its own variance from every row's influence on the correlations under it, and a Wald test, which held 0-2% at 1% across the same cases. The C++ estimator matches an independent Python version to 1e-14 on the correlations and 3e-11 on every p-value. On a chain observed partly as yes/no columns, Pearson returns nine edges, five spurious at stability 1.0; the mixed test returns the true four. Pearson and rank output is byte-identical to before.

   Ordinal columns are **DONE**. A column with 3 to 9 levels is read as a latent Gaussian variable cut at thresholds, with two-step polychoric and polyserial correlations and their influence functions, including the part the estimated thresholds contribute. Reading such a column as continuous, as the mixed test did before, rejected a true independence given an ordinal variable 94–98% of the time at a nominal 1%. Read as ordinal, every case held 0.5–1.0% at 1% over 600 draws. The C++ matches an independent Python version to 3e-8 on every latent correlation. Output with no ordinal column is byte-identical to before.

   A test that does not read dependence through a correlation is **DONE**, as `test := 'kernel'`. Every other test here, `'rank'` included, misses a bend: no monotone transform straightens a parabola. On a true chain x -> y -> z with y = x^2 over 8,000 rows, corr(x, y) is 0.009 while corr(|x|, y) is 0.920, and `'pearson'`, `'rank'` and `'mixed'` all return `y -- z` alone. The reverse error is worse because it invents rather than omits: where two variables are parabolas in a third and independent given it, Fisher's z rejects that true conditional independence 100% of the time, at full stability.

   The method is RCoT (Strobl, Zhang and Visweswaran 2019), approximating KCIT with random Fourier features so the cost is linear in the rows rather than quadratic. Getting it right was the null, not the statistic. A two-moment gamma is the obvious approximation to a weighted sum of chi-squares and it is anticonservative in exactly the tail that decides edges, rejecting true independences 3-10% of the time at a nominal 1%; the three-moment method of Liu, Tang and Zhang (2009) holds it. 25 features for the conditioning set were not enough either - what they could not absorb came back as dependence between x and y, at 10% - and neither was a real ridge on the Gram matrix, which under-residualises: at 1e-6 relative the level went from 0.005 to 0.21. With 100 features and a 1e-8 nudge, the measured level end to end, as how often PC joins a truly independent pair, is 0.00-0.04 across four null cases at 1,000 to 8,000 rows against 1.00 for `'pearson'` and `'rank'` on the bent one, and the power on a bent dependence is 1.00 against their 0.04-0.06. On linear data it returns the same graph as `'pearson'`.

   Two costs, both measured and both documented rather than discovered by a reader. It is about 100x slower: seven variables and 20,000 rows take 1.3 s without resampling and 13.6 s with the default 50, against 0.012 s. Caching the Gram factor per conditioning set, which PC asks about once per pair that could use it, took that from 29.6 s. And its power falls away when the conditioning set nearly determines one of the pair, since the bend is then in a small remainder: recovery of the bent edge over 40 draws was 40/40 while the third variable took 81% or less of the middle one, and 38-40/40 at 90%.

   Orienting the edges PC cannot is **DONE**, in two ways, because the `--` edges were the part of discovery that pushed the most work back onto the person reading it.

   `tiers := [['age', 'sex'], ['discount'], ['revenue']]` states an order and settles every edge that crosses it before a test runs. This is how these edges actually get settled in practice. It reaches all three algorithms: in PC it pre-orients the crossing adjacencies, which also means Meek's rules can never reach for an orientation it forbids; in FCI it is an arrowhead at the later end and nothing more, since `g o-> h` still allows a hidden common cause; in LiNGAM it restricts which variable may be peeled next. A v-structure the tiers refuse is counted and reported, because that is the data contradicting a person, not a detail.

   `algorithm := 'lingam'` is DirectLiNGAM (Shimizu et al. 2011): the direction comes from the shape of each variable's disturbance rather than from conditional independence, so every edge is oriented and the proposal needs only its review marker deleted. The C++ matches an independent Python version exactly — same causal order and same edge set on 12 random problems of 4 to 7 variables and 800 to 4,000 rows.

   The assumption is enforced, not mentioned. On Gaussian disturbances the asymmetry LiNGAM reads does not weaken, it vanishes, and the method still returns a confident order: 1 of 40 simulated orders right, against 40 of 40 with uniform disturbances. So each disturbance gets a Jarque-Bera test and two that cannot be told from Gaussian is a refusal, identifiability allowing one. The test tracks the failure rather than approximating it: at 1,000 rows and up it fired on 0 of 40 draws for uniform, t(8), mildly skewed and near-Gaussian-mixture disturbances and 40 of 40 for Gaussian; at 200 rows it fired on 18-39 of 40 for the three mildly non-Gaussian ones, where the order was itself exactly right only 15-25 times of 40. Strongly non-Gaussian uniform disturbances at 200 rows were refused once in 40, with the order right 39 times. `test := 'rank'` and `'mixed'` are refused with it, since normal scores are Gaussian by construction.

   LayeredLiNGAM's peel (Suzuki, ECML-PKDD 2024) was built and measured against plain DirectLiNGAM rather than assumed: it cut the iteration count by a third and cost recall (0.933 against 1.000) and orders (33/40 against 40/40). Its speedup is for variable counts far above the 30 `do_discover` allows, so it is not what ships. `min_effect` defaults to 0.05 as a standardised coefficient; over 40 six-variable graphs the edge set was fully recovered at every setting to 0.10, with the false positive rate falling from 0.007 to 0.000.

   `algorithm := 'both'` runs PC and LiNGAM over the same rows and unions them, with an `agreement` column per pair: `both`, `oriented by lingam`, `conflict`, `pc only`, `lingam only`. PC's orientation wins where it has one, because it rests on weaker assumptions; LiNGAM fills in what PC left undirected; a contradiction goes back to `--` and to review. The conflicts are the useful part — both methods assume nothing unmeasured causes two variables, and a hidden common cause breaks them differently, so the disagreement is where that assumption shows.
   Nonlinear orientation is **DONE**, as `algorithm := 'resit'` (Peters, Mooij, Janzing and Scholkopf, JMLR 2014). It runs LiNGAM's search from the other end - peeling a sink, the variable whose residual once the others are regressed out of it is least dependent on them - with kernel ridge regression on the random Fourier features `test := 'kernel'` already builds, then prunes the order into a graph with the same kernel test. A continuous additive noise model is identified by a nonlinear effect *or* a non-Gaussian disturbance, so RESIT covers exactly the case LiNGAM refuses.

   Measured on `a -> b -> c` with `a -> d`, 15 draws, as oriented true edges of 3 and false edges per run: straight links and non-Gaussian disturbances, LiNGAM 1.00/0.00-0.07 and RESIT 1.00/0.00-0.07; bent links and non-Gaussian, LiNGAM 0.58-0.64/1.07-1.27 and RESIT 1.00/0.00-0.20; bent links and Gaussian, LiNGAM 0.00-0.02/2.40-2.47 and RESIT 1.00/0.00-0.07. RESIT is at least as good as LiNGAM wherever both run; the only reason to keep LiNGAM is speed, 0.01 s against 0.76 s.

   That third row exposed a real blind spot in LiNGAM's own refusal, which is now written down rather than left for a user to hit: it tests what is left after a *linear* fit, so a bend it cannot model lands in the residual and reads as the non-Gaussianity it is looking for. On bent Gaussian data LiNGAM does not refuse - it returns a confident order and gets none of three true edges.

   RESIT has a refusal of its own, the same corner from the other side: no fit needing more than a straight line, and two or more disturbances indistinguishable from Gaussian. That is the textbook unidentifiable case, and RESIT does not fail quietly in it - 0.47 of 3 true and 2.53 false, with no sign anything was wrong. A fit counts as bending when the kernel regression leaves at least 5% less of the target's variance than a straight line does.

   `algorithm := 'both'` became `'pc+lingam'`, with `'both'` kept as its older name, and `'pc+resit'` is the matching union.

7. **Federated / multi-table estimation** — effects across joins without materializing the join. **DONE**, in two parts: correct inference across one-to-many joins, and pooling across sites that cannot share rows.

   Taken literally, "without materializing the join" is not a goal DuckDo should have. Cross-fitted nuisance models need row-level data, so the frame is materialised by design, and a join is already a valid first argument. What actually went wrong across joins was correctness. A one-to-many join, such as customers joined to their orders, turns one unit into several rows, and every estimator treated them as independent. On 4,000 units joined to three orders each, the estimate was unchanged, but the standard error fell from 0.0352 to 0.0203: an interval 42% too narrow, with no warning. `id :=` did not help, because it was never checked for repeats.

   `cluster :=` on `do_ate`, `do_att`, `do_atc` and `do_ate_by` makes the named column the unit, in three places:
   - Folds are assigned by cluster, stratified by majority arm. Before, a unit's rows landed on both sides of a split, and the nuisance models saw the rows they were scored on.
   - Influence-function, DML-sandwich and two-sample standard errors are summed within clusters.
   - Bootstrap estimators resample whole clusters.

   The joined rows then give back 1.978 and 0.0352, the units' own answer to every printed digit. Singleton clusters reproduce the ordinary standard error, as the G/(G−1) algebra says they should.

   The parameter is registered only on the functions that honour it, so everywhere else it is rejected rather than ignored. Unclustered results are bitwise identical to before, across all nine estimators, ATT, ATC and `do_ate_by`, confirmed by diffing 17 values recorded at full precision. A repeated `id :=` now warns everywhere, since it is the one sign of the problem DuckDo can see without being told.

   Federated estimation is `do_ate_pool`. Each site runs `do_ate`, or a group runs `do_ate_by`, and publishes one row: n, estimate and standard error. The pooled estimate is the size-weighted mean of those rows, and its variance is Σ(wₛ/W)²·seₛ². Both are exact for influence-function estimators with models fitted per site, since a site's estimate is the mean of its ψ and its squared standard error is var(ψ)/n. For ATT the weight is the treated count, and for ATC the control count; mixing estimands is refused.

   Inverse-variance weighting, the meta-analysis default, is used only for the heterogeneity test (Cochran's Q, p-value, I²). It weights by precision rather than population. Take three sites of 2,000, 1,000 and 3,000 units with effects of 1, 2 and 3, where the population effect is 2.167. Inverse-variance weighting gives 1.991 with an interval of roughly [1.978, 2.004], precise and wrong, because the middle site's near-noiseless outcome outvotes the others. `do_ate_pool` gives 2.138 [2.084, 2.192]. Centralising every row gives 2.134 [2.070, 2.198].

   The chi-square tail is a hand-written regularised incomplete gamma, checked against scipy to 1e-12 in both branches (series and continued fraction). The pooled estimates and standard errors are checked against scipy to the same tolerance.

   Clustering now reaches every estimator that reports an interval, through three shared pieces: the influence-function sum, a cluster-robust (CR1) sandwich in `linalg`, and a resampler that draws whole clusters. Every unclustered result is bit-identical to before, checked by fingerprinting the full output of 14 functions. On 3,000 rows joined to three copies each, every clustered interval comes back to the rows' own: 1.00 for the closed-form ones, 0.99 for `do_predict`, and 1.00 to 1.05 for the bootstraps. Unclustered they sit at 0.56 to 0.60, around 1/sqrt(3). `test/sql/cluster_estimators.test` holds each one to that.

   The diagnostics cluster too. `do_refute`'s placebo, subset and bootstrap refuters randomise whole clusters, and its tolerance is the clustered one. `do_sensitivity` works from the clustered interval, and `do_diagnose` counts clusters as the sample. Enabling them exposed a latent crash. A refuter that reorders a frame renumbered its clusters from labels already released after the first numbering, and the subset frame copied its parent's cluster ids unchanged. The numbering is now safe to rerun, and a subset renumbers its own clusters. Nothing had reached either path, because the diagnostics refused `cluster :=`.

   Still open: a secure-aggregation protocol, since `do_ate_pool` trusts the rows it is given.

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
- ~~Do-PFN is CC BY 4.0~~ **Corrected.** Do-PFN's repository has no LICENSE file, and its README states no terms. The CC BY 4.0 this plan assumed is granted nowhere upstream. Without a licence its weights are all rights reserved: the catalog says so, the model is marked non-commercial, every result carries a warning, and it is never redistributed. CausalPFN's licence is not "Apache-2.0" by name either. It is the "CausalPFN License, Version 1.0", Apache-2.0's terms under its own title, and the catalog now reports it by that name.
- ~~CausalFM's license must be verified at integration time.~~ Verified: Apache-2.0, for both the original repository and the toolkit that ships the weights. Moot, since the model was not exported.

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
| 9 | Upstream model licenses change | Low | Low | Only CausalPFN is redistributed, pinned to one revision with its licence alongside; the catalog reports each licence as verified in the upstream repository |

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
2. ~~**Export CausalFM.**~~ **Evaluated and declined.** The evidence is in `scripts/causalfm_check.py`.
   The weights exist: `yccm/CausalFM-toolkit` ships four checkpoints under Apache-2.0. They load, and
   they reproduce the authors' published PEHE exactly, 0.8466 for front-door and 0.4223 for binary IV,
   on the authors' own test sets with the authors' own protocol. That evaluation never reported the
   baseline of predicting a constant. Measured against that baseline:

   | checkpoint | PEHE | predicting zero | what the output does |
   |---|---|---|---|
   | front-door | 0.8466 | **0.8388** | varies by 4e-9 per row: -0.032 for every row of every dataset |
   | binary IV | 0.4223 | 0.4527 | responds, 7% better than zero, on test sets where 8 of 10 instruments are weak (first-stage F 0.0-7.7) |
   | standard | - | - | PEHE 0.85-1.85 against 0.07-0.1 for CausalPFN and AIPW on the same DGP; shrinks 24% at 25 covariates |

   **The published front-door checkpoint is a constant function.** Its reported number is exactly
   the score of a constant predictor. The IV checkpoint does respond to its input, but on a
   strong-instrument DGP where the effect is identified it returns 2.997 against a truth of 2.0,
   with the confounded naive difference at 2.937, while `do_iv` returns 2.039. Exporting either would
   put a wrong answer behind DuckDo's SQL with DuckDo's name on it. The classical `do_frontdoor`
   and `do_iv` remain the only engines for those settings. They are not immune either: on the
   authors' weak-instrument data `do_iv` swings from -2.2 to 17.5. But it flags every one of those
   cases `weak_instrument = true` instead of returning a confident number.

   Two notes for anyone revisiting this. First, the toolkit's bundled TabPFN fork does not import
   under scikit-learn 1.9, because its compat layer needs the removed `_is_pandas_df`. The check
   script loads the networks by registering the package without running its `__init__`, which the
   networks never needed. Second, the models require standardised X and Y. Given raw inputs, the
   standard checkpoint's CATE correlates *negatively* with the truth.
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
   The covariate-subset work above sharpened this: on twelve covariates Do-PFN returns 5.24 against
   a truth of 3.0 purely because five is not enough, while CausalPFN returns 2.984. The five-column
   budget is not a detail, it is disqualifying on any real table, and the docs should stop treating
   the two models as peers.
5. ~~**Ensemble over covariate subsets**~~ **DONE, and it measured its own limits.** Where a
   model's covariate budget binds, each ensemble draw now samples a different subset weighted by
   outcome correlation, instead of every draw using the same top-k. The interval therefore covers
   *which covariates were chosen*, which a single top-k pass treats as free.

   On a DGP with twelve contributing covariates and a true effect of 3.0, against Do-PFN's budget
   of five, the interval widens from ±0.006 to ±0.31 — and still does not cover the truth:

   | | estimate |
   |---|---|
   | AIPW, all twelve | 2.939 |
   | CausalPFN, budget 99 | 2.984 |
   | AIPW restricted to the five Do-PFN keeps | 4.071 |
   | Do-PFN, single draw | 4.970 |
   | Do-PFN, ensemble of 8 | 5.242 ± 0.31 |

   A correctly specified estimator restricted to the same five covariates returns 4.07, so the
   budget rather than the model accounts for most of the damage. **Dropping confounders is bias,
   and no interval built by resampling can cover bias.** The result now says that explicitly when
   more covariates are dropped than kept, because that is exactly when a healthier-looking interval
   is most likely to be believed.
6. ~~**Spill past `duckdo_max_memory`.**~~ **DONE, by enforcing rather than spilling.** The
   setting is now a real ceiling with an actionable message, and the measured peak for the 1M x 50
   frame fell from 1,645 MB to 882 MB against a 400 MB matrix. Spilling was considered and
   rejected: see Phase 8, which also shows why the residual 2.2x is a floor rather than an
   oversight.
7. ~~**Persist graphs**~~ **DONE**: graphs live in a `duckdo_graphs` table, so they survive a restart and are inspectable as ordinary data. Open question 3 is answered — a plain table beat catalog integration, which would have coupled DuckDo to internals that move between DuckDB versions.
8. ~~**Broaden the cross-check further.**~~ **DONE for ACIC.** `scripts/bench_headtohead.py` now runs
   ten ACIC 2016 simulations beside the ten IHDP replications and Lalonde. They are the simulations
   causallib ships as CSVs, extracted from its wheel without installing it, since causallib pins
   its own scikit-learn. ACIC is where DuckDo's classical path loses most. `aipw` misses the
   average by 0.43 with a PEHE of 3.09, behind EconML's `LinearDML`, which is also linear (0.35,
   2.61). The flexible methods lead: `CausalForestDML` on the average effect (0.114), and the two
   CausalPFN runs on per-row effects (PEHE about 1.4). DuckDo's CausalPFN tracks the reference less
   tightly than on IHDP, 1.453 against 1.379 on PEHE, because 4,802 rows exceed the 4,096-row
   context and the two pick different subsets to condition on. Since the fold change, IHDP reads
   0.164 mean |ATE error| and 2.25 mean PEHE for `aipw`, and Lalonde reads 1709.8 for `dml`
   against the 1794.3 benchmark. The canonical 1000-replication IHDP set and the full ACIC
   competition set are still not run.
9. ~~**Assign folds by row content, not row position.**~~ **DONE.** `CausalFrame` carries a
   canonical order derived from what each row contains, and every seeded draw indexes that instead
   of storage: fold assignment, bootstrap resampling in `do_ate`, `do_iv` and `do_mediate`, the
   foundation model's context sample, and all five refutation methods. The spread across ten
   permutations falls from 0.25% / 2.12% / 6.18% of a standard error to 0.00%, leaving 2e-15
   relative, which is summation order.

   **Hashing the rows was the obvious approach and it is wrong**, which is the part worth keeping.
   `X` is standardised, so every value carries a mean summed in storage order; permuting the table
   moves it by an ulp, and a hash turns an ulp into a completely different sort key — the
   hash-ordered sort was *more* order-sensitive than what it replaced. Comparison is stable where
   hashing is not, because standardising is monotone within a column. Cost is 0.8 s on the 1M × 50
   frame, most of which came back by sorting (outcome, row) pairs rather than bare indices, since
   sorting indices chases `y` at a random offset on every comparison.

   Accuracy is unchanged: across ten IHDP replications the paired difference in absolute error is
   0.027 ± 0.039. The published numbers moved anyway, and regenerating them surfaced two
   documentation defects — `check_docs.py` verified that documented SQL *ran* but never that it
   still produced the documented numbers, and both executed docs were quoting stale output; and the
   tutorial mis-taught `do_refute`, reading `passed = false` as the desired outcome on a placebo
   test when it means the refutation did not behave as it should.
10. ~~**Cache the foundation model's context encoding.**~~ **DONE.** Every CausalPFN layer takes its
    keys and values from the context prefix alone (`k, v = kv_proj(h[:, :context_length])`), so the
    context's representation cannot depend on the query. The export now writes two graphs:
    `causalpfn_encode.onnx` produces a per-layer key/value cache (20 layers × context × 384, twice)
    plus the context-derived input statistics and outcome scaling, and `causalpfn_decode.onnx`
    scores query chunks against it. The runtime encodes once per ensemble draw and decodes each
    chunk once per arm against the same cache, borrowing the cache rather than copying it.

    The split was prototyped in PyTorch before anything was exported. It reproduced `predict_cepo`
    to 1e-7 across four context/query shapes. Through ONNX its estimate is **bit-identical** to the
    single graph's, 2.9705805124938487 at every chunk size, and so are the ensemble draws.

    | `duckdo_query_chunk` | single graph (one run) | two graphs (median of three) |
    |---|---|---|
    | 512 (default) | 125.8 s, 2,062 MB | **31.3 s, 1,938 MB** |
    | 2048 | 42.8 s, 3,335 MB | 23.0 s, 2,728 MB |
    | 8192 | 23.3 s, 5,347 MB | 22.2 s, 5,030 MB |

    The default is four times faster and uses slightly less memory. Chunk 2048 reaches the
    package's CPU speed. The weights grow from 75 MB to 120 MB, because both graphs carry the
    layers.

    One measurement nearly became a finding. A single run put the two graphs *slower* than the
    one at chunk 8192, 29.2 s against 23.3 s, and an explanation was already drafted: the decode
    graph takes k and v as inputs, so ONNX Runtime's fused attention kernel would not match it.
    Three runs put that cell at 22.2 s. The slowdown was noise, and so was the explanation. A
    single run of the same configuration on this machine varies by up to a quarter, which is
    why every figure quoted here is now a median.
11. ~~**Write `do_download`**~~ **DONE**, and the CausalPFN export is hosted, so
    `do_download('causalpfn')` needs no source. `do_download(model, source := ...)` copies a model's artifacts from a
    directory or URL into the model directory through DuckDB's virtual file system. A URL
    therefore goes through DuckDB's `httpfs`, and DuckDo still contains no HTTP client. Each file
    lands in a `.part` file that is renamed only on success. Existing files are kept unless
    `overwrite := true`. Every row carries the SHA-256 of the bytes that landed, and the tests
    check it against DuckDB's own `sha256()`.

    Hosting was the maintainer's call, and it is made. The CausalPFN export is published at
    [maxdemarzi/duckdo-causalpfn](https://huggingface.co/maxdemarzi/duckdo-causalpfn), with the
    upstream licence, a `NOTICE` naming every changed file, the parity numbers and the authors'
    citation. The extension pins that repository to one commit, compiles in the SHA-256 of every
    file, and checks each one before installing it. A file that fails is never installed, even
    briefly.

    Checking the licences before publishing corrected two claims this plan had made. CausalPFN's
    licence is the "CausalPFN License, Version 1.0", Apache-2.0's terms under its own name, not
    Apache-2.0. Do-PFN's repository states no licence at all, not CC BY 4.0, so its weights are all
    rights reserved and are not hosted.
12. **Submit** the `description.yml` PR to `duckdb/community-extensions`.
