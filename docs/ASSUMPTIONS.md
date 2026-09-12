# What has to be true

Every causal estimate rests on assumptions. Some you can check against the data.
Some you cannot check with any amount of data, ever, and those are the ones that
usually break.

This page is the map. For each assumption: what it means in plain terms, what
breaks it, which DuckDo function speaks to it, and whether the data can settle it.

If you read one section, read [**No unmeasured confounding**](#1-no-unmeasured-confounding)
— it is the one that is nearly always the reason a causal estimate is wrong.

---

## 1. No unmeasured confounding

> Among people who look identical on the covariates you adjusted for, whether they
> got the treatment is as good as random.

**What breaks it.** Anything that influenced both who got treated and what happened
to them, that you did not measure. A salesperson's judgment about which accounts to
call. A doctor's sense of which patients can tolerate a drug. Whether a customer had
already decided to churn before you emailed them.

**Can the data check it?** **No.** This is the central difficulty of causal inference
and no method removes it. An estimator cannot adjust for a column you do not have.

**What DuckDo gives you instead.** A way to say how strong the missing confounder
would have to be:

```sql
SELECT robustness_value, e_value, interpretation
FROM do_sensitivity('customers', treatment := 'got_discount', outcome := 'revenue');
```

A robustness value of 0.40 means an unmeasured confounder would have to explain 40% of
the residual variance in *both* the treatment and the outcome to move your estimate to
zero. Compare that against the covariates you did measure. If nothing you measured gets
near 40%, a confounder that strong is a big claim — but it is a judgement, not a test.

**What to do.** Write down the causal story before you run anything, ideally as a
graph, and let `do_validate` grade your covariate list against it. Getting the *list*
right matters more than which estimator you pick.

---

## 2. Positivity (overlap)

> Everybody had some chance of getting the treatment, and some chance of not.

**What breaks it.** A rule. If every customer over $10,000 lifetime value automatically
got the discount, there is no comparable untreated customer above $10,000 and nothing
can tell you what would have happened to them. Adjustment silently extrapolates.

**Can the data check it?** **Yes**, and you should.

```sql
SELECT * FROM do_overlap('customers', treatment := 'got_discount', outcome := 'revenue');
```

`do_diagnose` includes it, reporting the propensity range and what fraction of rows sit
outside `[0.05, 0.95]`. A propensity of 0.999 means "this row was always going to be
treated" — its untreated counterfactual is a guess.

**What to do.** Restrict to the region of overlap and say so — the estimate then applies
to that region, not the whole population. Or use `trim :=` and read `n_trimmed` in the
result. An estimate over a population with no comparison group is not conservative, it
is fictional.

---

## 3. No interference

> One unit's treatment does not change another unit's outcome.

**What breaks it.** Marketplaces, social networks, anything with shared capacity. If you
discount for half your users and they buy up limited stock, the other half's outcome
changed because of a treatment they never got. Vaccines, referral programs, ranking
changes — all interfere by design.

**Can the data check it?** **Not from a single table.** DuckDo assumes no interference
everywhere and cannot detect its violation.

**What to do.** If units interact, the unit of analysis is usually wrong. Aggregate to
whatever the interference stops at — market, region, cohort — and treat that as the
unit, accepting the much smaller sample. A clean answer about 40 markets beats a
confident wrong answer about 4 million users.

---

## 4. Consistency, and one well-defined treatment

> "Treated" means the same thing for everyone it applies to.

**What breaks it.** A `discount` column that mixes 10% off, free shipping and a $5
credit. The estimate is then an average over an unstated mix of interventions, and it
will not reproduce when that mix changes.

**Can the data check it?** **No**, but you can look. If your treatment column came from
several campaigns joined together, split it and estimate them separately.

---

## Assumptions that belong to particular functions

### `do_iv` — the exclusion restriction

The instrument affects the outcome **only** through the treatment. No data can check
this; it is an argument about how the world works. What DuckDo does check is
*relevance* — whether the instrument moves the treatment at all — and reports the
first-stage F. Below about 10 the instrument is weak, and a weak instrument makes 2SLS
worse than doing nothing: `weak_instrument` comes back true.

### `do_frontdoor` — the mediator intercepts everything

Every directed path from treatment to outcome runs through the mediator, and nothing
confounds treatment and mediator. Both halves of the product are reported separately
because that is where a front-door estimate goes wrong.

### `do_mediate` — sequential ignorability

The one people get wrong. **Randomising the treatment does not buy this.** It removes
confounding between treatment and outcome, and between treatment and mediator, and
leaves confounding between the *mediator* and the outcome completely untouched. In a
perfect randomised trial, the direct/indirect split can still be badly wrong. Every
`do_mediate` result says so.

### `do_discover` with `algorithm := 'lingam'` — non-Gaussian disturbances

PC and FCI say nothing about the shape of the data. LiNGAM's whole ability to orient
`x -- y` comes from that shape: if the disturbances are non-Gaussian, regressing the
effect on the cause leaves a residual independent of the cause while the reverse
regression does not. If they are Gaussian, that asymmetry does not weaken, it vanishes,
and the method still returns an order — a confident one, and close to a coin flip. This
is the one assumption in DuckDo enforced rather than reported: each disturbance is tested
with Jarque–Bera, and two that cannot be told from Gaussian makes `do_discover` refuse,
because identifiability allows at most one. It also assumes linear effects, no cycles,
and no hidden common cause of any two variables — the last shared with PC, and the
likeliest of the four to be false in real data. `algorithm := 'pc+lingam'` is the way to
see that: where a hidden cause is at work, PC and LiNGAM tend to disagree, and the
disagreement is reported rather than resolved.

The refusal has one blind spot worth knowing. It tests what is left after a *linear*
fit, so if the true effect bends, what LiNGAM cannot model lands in the residual and
reads as the non-Gaussianity it is looking for. It then runs on data it should not, and
reports a confident order that is wrong. `algorithm := 'resit'`, below, is the method
for that case.

### `do_discover` with `algorithm := 'resit'` — an added disturbance

RESIT lets the effect bend, which LiNGAM does not, and lets the disturbance be
Gaussian, which LiNGAM cannot. What it will not let you have is a disturbance that is
*mixed into* the effect rather than added to it: `x = f(parents) + e`, not
`x = f(parents, e)`. It shares LiNGAM's other two assumptions, no cycles and no hidden
common cause of any two variables, and the second is again the likeliest to be false.

It refuses one corner outright, and it is the same one LiNGAM refuses from the other
side: when no fit needs more than a straight line and two or more disturbances cannot
be told from Gaussian, nothing here can read the data, so `do_discover` stops rather
than returning an order. Note the asymmetry this exposes in LiNGAM's own check: LiNGAM
tests what is left after a *linear* fit, so a bend it cannot model shows up as
non-Gaussianity and lets it run on data it should not. On a bent world with Gaussian
disturbances LiNGAM recovered none of three true edges and invented 2.4; RESIT
recovered all three.

### `do_discover` with `algorithm := 'rcd'` — the one that drops causal sufficiency

Every other functional method here assumes nothing unmeasured causes two of your
variables, which is the assumption most likely to be false. RCD does not. It asks, of
each related pair, whether regressing either one on the other leaves an independent
residual; when neither does, no model without a hidden common cause fits, and it says
so with `<->`. It still assumes linear effects, no cycles, and non-Gaussian
disturbances.

It is the only discovery method here that needs no refusal, because it can express its
own failure: on Gaussian disturbances every pair comes back `o-o`, undecided. Read a
mostly-`o-o` result as "this data cannot answer the question", not as a finding.

Two limits, both measured. It does not invent hidden causes — 0.00 per run over 40 runs
on a world with none. But it detects one best when it is moderate: naming it 12-15 times
in 20 when the hidden cause is 70% of each variable it links, 3-6 in 20 at 50%, and 0 in
20 at both 30% and 90%. The 90% case is the interesting one — when the hidden cause is
nearly all of what a variable is, that variable stands in for it and "a causes c" is
very nearly true.

### `do_msm` — sequential exchangeability

No unmeasured confounder of treatment and outcome at *any* period. This buys nothing
against unmeasured confounding; what it buys is correct handling of measured
confounders that the treatment itself affects — which no covariate adjustment can do.
Check `mean_weight`: it should sit near 1, and a mean far from 1 means the treatment
model is misspecified.

`do_msm_rmst` needs the same, and one thing more: dropout that depends only on the
measured history, which its dropout model conditions on. It also needs units who
actually stayed on each regime. Check `max_weight` and the followers counts: a handful
of units carrying a regime's curve is what a positivity problem looks like there.

### `do_rmst` and `do_rmtl` — independent censoring

Subjects who leave the study are not leaving *because* of where their outcome was
heading. Nothing in the data can check it, and it fails exactly when patients drop out
because they are getting worse. The horizon is part of the estimand: a different
horizon is a different quantity, not a different view of the same one.

`do_rmtl` adds one thing that is not an assumption but is easy to get wrong: an event
from another cause is not censoring. A subject who died of something else can no longer
die of this, and treating them as censored estimates a world where the other causes had
been abolished.

### `do_did` — parallel trends

Without the treatment, treated and untreated groups would have moved the same way.
`do_event_study` shows the pre-treatment periods, which is the closest available check:
if the groups were already diverging before treatment, the assumption is in trouble.
Passing it for pre-periods does not guarantee it for post-periods.

---

## What "the estimate is fine" would actually require

| | can the data check it? | DuckDo function |
|---|---|---|
| No unmeasured confounding | **no** | `do_sensitivity`, `do_validate` |
| Positivity / overlap | yes | `do_overlap`, `do_diagnose` |
| Covariate balance after adjustment | yes | `do_balance` |
| No interference | no | — |
| Well-defined treatment | no | — |
| Correct covariate list (no mediators, no colliders) | with a graph | `do_validate` |
| Model specification | partly | `do_refute`, comparing estimators |

Two entries deserve emphasis.

**Adjusting for the wrong thing is worse than adjusting for nothing.** A *mediator* sits
on the causal path and adjusting for it removes part of the effect you are trying to
measure. A *collider* is caused by both treatment and outcome, and adjusting for it
manufactures an association that does not exist. Both feel like diligence — more
controls, more careful — and both make the answer worse. `do_validate` against a graph
is the only thing here that catches them.

**Estimator choice matters less than you would think.** If the covariate list is right
and overlap holds, `aipw`, `dml` and `regression` will land in the same place; if the
list is wrong they will agree with each other and all be wrong together. Agreement
across estimators is not evidence of correctness.

---

## A workable order

1. Write the causal story down. `CALL do_graph_create(...)`.
2. Grade your covariate list against it. `do_validate`.
3. Estimate. `do_ate`.
4. Check what can be checked. `do_diagnose`.
5. Try to break it. `do_refute`.
6. Quantify what cannot be checked. `do_sensitivity`.
7. Report the assumption you are resting on, alongside the number.

Step 7 is not a formality. An estimate reported without its assumptions will be quoted
without them too, and by then nobody remembers what it was resting on.

See **[TUTORIAL.md](TUTORIAL.md)** for that sequence run end to end on real output.
