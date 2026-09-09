# Worked examples

Three problems that come up constantly and that a naive query gets wrong — each in
a different direction. Every figure below is real output from the SQL shown, on
synthetic data where the true answer is known.

| | naive answer | DuckDo | truth |
|---|---|---|---|
| [A/B test with non-compliance](#1-an-ab-test-nobody-complied-with) | 0.1749 | **0.1174** | 0.12 |
| [Observational pricing](#2-pricing-where-the-sign-is-wrong) | **+1.49** | **−2.5025** | −2.5 |
| [Churn intervention](#3-a-retention-call-worth-targeting) | −0.0248 | **−0.0908** | −0.0896 |

The middle row is the one to look at twice. The naive answer is not merely too big;
it has the **wrong sign**. The raw data says raising prices increases sales.

---

## 1. An A/B test nobody complied with

You randomised a new onboarding flow. Assignment was clean. Uptake was not — busy
users skipped it, and busy users convert less anyway. So the obvious analysis,
comparing users who *completed* onboarding against those who did not, is confounded
even though the experiment itself was perfect.

```sql
SELECT setseed(0.11);
CREATE TABLE trial AS
WITH u AS (
  SELECT i AS user_id,
    (random() < 0.5)::INT AS assigned,                                -- randomised
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS busyness,     -- never observed
    random() AS r_take, random() AS r_conv
  FROM range(30000) t(i)),
 s AS (SELECT *, (r_take < 1.0/(1.0+exp(-(-1.6 + 2.6*assigned - 0.9*busyness))))::INT AS completed FROM u)
SELECT user_id, assigned, completed,
       (r_conv < 0.30 + 0.12*completed - 0.10*busyness)::INT AS converted
FROM s;
```

The true effect of *completing* onboarding is **0.12**. Note that `busyness` is not a
column — it is exactly the confounder you would not have.

```
uptake in control                       0.201
uptake in treatment                     0.706
per-protocol (completed vs not)         0.1749
intention-to-treat (assigned vs not)    0.0592
```

Two wrong answers and no obvious way to choose between them. **Per-protocol** (0.1749)
compares compliers to non-compliers and inherits every difference between those groups.
**Intention-to-treat** (0.0592) is unbiased but answers a different question: it is the
effect of *offering* the flow, diluted by everyone who ignored it. If you are deciding
whether to build the feature, ITT is right. If you are asking whether the flow works,
neither is.

Random assignment is a valid **instrument**: it shifted uptake, and it cannot plausibly
affect conversion any other way.

```sql
SELECT round(estimate,4) AS late, round(ci_low,4) AS lo, round(ci_high,4) AS hi,
       round(first_stage_f,1) AS first_stage_f, weak_instrument
FROM do_iv('trial', treatment := 'completed', outcome := 'converted',
           instrument := 'assigned', covariates := []);
```

```
late    lo      hi      first_stage_f  weak_instrument
──────  ──────  ──────  ─────────────  ───────────────
0.1174  0.0962  0.1386        10338.9  false
```

**0.1174**, and the interval covers 0.12.

Two things to check whenever you do this. The **first-stage F** at 10,339 says the
instrument moves the treatment hard; below about 10 it would be weak, and a weak
instrument is worse than no instrument. And the arithmetic ties out: LATE × the shift
in uptake = 0.1174 × 0.505 = 0.0593, which is the ITT. That identity is a free check
that nothing is wired up backwards.

The estimand is **LATE** — the effect among *compliers*, people who did the flow
because they were offered it. That is not the whole population and DuckDo labels it
accordingly. The assumption doing the work is the exclusion restriction: assignment
affects conversion only through completion. No data can check that.

---

## 2. Pricing, where the sign is wrong

Price was never randomised. The team charged more for premium SKUs in high-demand
categories, which also sell more. So the raw relationship between price and volume
comes out **positive**.

```sql
SELECT setseed(0.23);
CREATE TABLE sales AS
WITH u AS (
  SELECT i AS sku_week,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS demand_index,
    (random()*3)::INT AS category,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS e1,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS e2
  FROM range(25000) t(i)),
 p AS (SELECT *, 20.0 + 4.0*demand_index + 1.5*category + e1 AS price FROM u)
SELECT sku_week, category, round(demand_index,3) AS demand_index, round(price,2) AS price,
       round(120.0 - 2.5*price + 18.0*demand_index + 3.0*category + 4.0*e2, 2) AS units
FROM p;
```

True effect of $1 of price on weekly units: **−2.5**.

```sql
SELECT round(regr_slope(units, price),3) AS naive_slope FROM sales;   -- +1.49
```

**+1.49.** A model fitted on this would recommend raising prices to sell more. This is
the failure mode that matters most in practice, because it does not look like an error
— it looks like a finding.

Price is continuous, so the binary estimators refuse it and point here instead:

```sql
SELECT round(estimate,4) AS ape, round(ci_low,3) AS lo, round(ci_high,3) AS hi
FROM do_ape('sales', treatment := 'price', outcome := 'units', exclude := ['sku_week']);
```

```
ape      lo      hi
───────  ──────  ──────
-2.5025  -2.552  -2.453
```

**−2.5025** against a truth of −2.5. Right magnitude, right sign.

For a price *decision* you want the curve, not the slope:

```sql
SELECT round(dose,2) AS price, round(mu,1) AS expected_units,
       round(mu_low,1) AS lo, round(mu_high,1) AS hi, n_within_decile
FROM do_dose_response('sales', treatment := 'price', outcome := 'units',
                      exclude := ['sku_week'], grid := 5);
```

```
price   expected_units  lo     hi     n_within_decile
──────  ──────────────  ─────  ─────  ───────────────
  3.51           115.3  114.2  116.4             1250
 19.23            76.3   76.1   76.5             2515
 22.25            68.8   68.7   68.8             2532
 25.21            61.3   61.2   61.5             2521
 41.01            21.5   20.4   22.6             1253
```

The grid is placed on observed quantiles, so it never extrapolates past prices you have
actually charged. `n_within_decile` is how much data sits near each point — 1,250 at the
ends against 2,500 in the middle — which is why the intervals fan out at the extremes.
**Trust the middle of this curve more than the ends.**

---

## 3. A retention call worth targeting

Support calls customers who complain. Complainers churn more anyway, so the call looks
useless. It is not — it works on customers who have not yet disengaged, and does nothing
for those who have.

```sql
SELECT setseed(0.41);
CREATE TABLE accounts AS
WITH u AS (
  SELECT i AS account_id, random() AS engagement, (random()*5)::INT AS support_tickets,
         random() AS r_call, random() AS r_churn
  FROM range(30000) t(i)),
 c AS (SELECT *, (r_call < 1.0/(1.0+exp(-(-1.5 + 0.55*support_tickets))))::INT AS got_call FROM u)
SELECT account_id, round(engagement,4) AS engagement, support_tickets, got_call,
       (r_churn < 0.15 + 0.06*support_tickets - 0.18*got_call*(engagement < 0.5)::INT)::INT AS churned
FROM c;
```

The call cuts churn by 18 points for the half of the base below 0.5 engagement, and does
nothing above it. Averaged over everyone: **−0.0896**.

```
naive churn difference   -0.0248
do_ate                   -0.0908   [-0.1011, -0.0805]
```

The naive comparison makes the call look **3.6× weaker** than it is.

Where does it work?

```sql
SELECT CASE WHEN a.engagement < 0.5 THEN 'low engagement' ELSE 'high engagement' END AS segment,
       round(avg(c.cate),4) AS estimated, round(avg(-0.18*(a.engagement<0.5)::INT),4) AS truth
FROM do_cate('accounts', treatment := 'got_call', outcome := 'churned',
             exclude := ['account_id'], id := 'account_id') c
JOIN accounts a ON a.account_id = CAST(c.id AS BIGINT)
GROUP BY 1;
```

```
segment           estimated   truth
────────────────  ──────────  ──────
low engagement       -0.1577   -0.18
high engagement      -0.0246    0.00
```

Directionally right, and **visibly shrunk**. The truth is a step function and the CATE
model is linear in the covariates, so it smooths the cliff. This is a real limitation,
not a rounding artefact — see the base-learner note in
[ASSUMPTIONS.md](ASSUMPTIONS.md).

It matters less than it looks, because a targeting rule only needs the *ordering*:

```sql
-- churn is bad-when-high, so flip it: do_optimal_policy treats where the effect
-- exceeds the cost, and that only makes sense on an outcome you want more of.
CREATE TABLE retention AS
SELECT account_id, engagement, support_tickets, got_call, 1 - churned AS retained FROM accounts;

SELECT rule, n, round(mean_effect,4) AS mean_effect, cost, action
FROM do_optimal_policy('retention', treatment := 'got_call', outcome := 'retained',
                       exclude := ['account_id'], depth := 1, threshold := 0.05);
```

```
rule                    n      mean_effect  cost  action
──────────────────────  ─────  ───────────  ────  ────────────
engagement <= 0.5021    15003       0.1764  0.05  treat
engagement > 0.5021     14997       0.0053  0.05  do not treat
```

**The true cutoff is 0.5 and the tree found 0.5021** — from data whose per-row estimates
were shrunk. `threshold := 0.05` is what a call costs, in the units of the outcome.

How much does targeting buy?

```sql
SELECT round(fraction_targeted,2) AS share, n_targeted,
       round(cumulative_gain,4) AS targeted_gain, round(random_gain,4) AS random_gain,
       round(qini,4) AS qini
FROM do_uplift('retention', treatment := 'got_call', outcome := 'retained',
               exclude := ['account_id'])
WHERE bucket IN (4, 8, 10, 12, 16, 20) ORDER BY bucket;
```

```
share  n_targeted  targeted_gain  random_gain  qini
─────  ──────────  ─────────────  ───────────  ──────
 0.20        6000         0.0375       0.0182  0.0193
 0.40       12000         0.0710       0.0363  0.0346
 0.50       15000         0.0853       0.0454  0.0398
 0.60       18000         0.0897       0.0545  0.0352
 0.80       24000         0.0926       0.0727  0.0199
 1.00       30000         0.0908       0.0908  0.0000
```

Calling the right **half** captures 0.0853 of the 0.0908 available from calling
everyone — 94% of the benefit for half the cost. The Qini column peaks exactly at 0.5,
which is the truth: precisely half the base benefits. Past that, every extra call is
spent on someone the intervention does not help.

---

## What the three have in common

The naive answer was wrong in a different direction each time — too big, wrong sign, too
small — so there is no rule of thumb that rescues you. What they share is that the
correct analysis needed a **different estimand**, not a better model: LATE instead of a
difference in means, an average partial effect instead of a slope, a conditional effect
instead of an average.

Getting the estimand right is most of the work. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for
what each one costs you in assumptions, and [TUTORIAL.md](TUTORIAL.md) for the full
sequence on one problem.
