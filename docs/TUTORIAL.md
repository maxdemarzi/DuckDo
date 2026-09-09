# Your first causal question

You ran a discount campaign. Revenue is up. **Was it the discount?**

This walks that question end to end in SQL. Every number below is real output from
the queries shown — you can paste the whole thing into a DuckDB shell and get the
same figures. It takes about five minutes.

The example is synthetic on purpose: the true effect is known, so you can see how
far each answer lands from it. In your data you will not have that luxury, which is
why the middle section — checking the assumptions — is the part that matters.

---

## The data

40,000 customers. The discount was **targeted, not randomised**: the team sent it to
customers who were already engaged. The true effect is `14.00 − 0.30 × tenure_months`,
so the discount is worth $14 to a brand-new customer and $2 to a four-year one.
Averaged across the customer base, exactly **$8.00**.

```sql
LOAD duckdo;
SELECT setseed(0.5);

CREATE TABLE customers AS
WITH u AS (
  SELECT i AS customer_id,
    (20 + 60*random())::INT AS age,
    (random()*40)::INT AS tenure_months,
    exp(3.0 + 0.9*sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random())) AS prior_spend,
    random() AS roll,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS noise
  FROM range(40000) t(i)),
 p AS (SELECT *, 1.0/(1.0 + exp(-(-2.4 + 0.055*tenure_months + 0.020*prior_spend))) AS ps FROM u)
SELECT customer_id, age, tenure_months, round(prior_spend,2) AS prior_spend,
       (roll < ps)::INT AS got_discount,
       round(18.0 + 0.35*tenure_months + 0.55*prior_spend
             + (14.0 - 0.30*tenure_months)*((roll < ps)::INT) + 6.0*noise, 2) AS revenue
FROM p;
```

## Step 1 — the answer you already have

```sql
SELECT round(avg(CASE WHEN got_discount=1 THEN revenue END)
           - avg(CASE WHEN got_discount=0 THEN revenue END), 2) AS naive_difference
FROM customers;
```

```
naive_difference
────────────────
          18.67
```

**$18.67.** The true answer is $8.00. This is not a small error or a noisy one — it is
more than double, and no amount of extra data will shrink it, because it is not a
sampling problem. Customers who got the discount were already spending more. The
comparison is measuring who was picked as much as what the discount did.

## Step 2 — the answer

```sql
SELECT estimand, estimator, round(estimate,2) AS estimate, round(std_error,3) AS std_error,
       round(ci_low,2) AS ci_low, round(ci_high,2) AS ci_high, variance_method, n, n_treated
FROM do_ate('customers', treatment := 'got_discount', outcome := 'revenue',
            exclude := ['customer_id']);
```

```
estimand  estimator  estimate  std_error  ci_low  ci_high  variance_method      n      n_treated
────────  ─────────  ────────  ─────────  ──────  ───────  ──────────────────  ─────  ─────────
ATE       aipw           8.05      0.071    7.90     8.19  influence function  40000      13961
```

**$8.05**, against a truth of $8.00, with the interval covering it.

Note what the row carries besides the number: which **estimand** it is, which
**estimator** produced it, and how the **interval** was made. No result in DuckDo is a
bare number, because a number without those three things cannot be checked.

`exclude := ['customer_id']` keeps the row identifier out of the covariates. Everything
else — `age`, `tenure_months`, `prior_spend` — is adjusted for by default.

> **`covariates := []` means no covariates.** Omitting the argument means *all* of them.
> The two are different on purpose.

## Step 3 — is that number allowed to be causal?

This is the step that separates causal inference from prediction, and it is one call.

```sql
SELECT check_name, status, severity, detail
FROM do_diagnose('customers', treatment := 'got_discount', outcome := 'revenue',
                 exclude := ['customer_id']);
```

```
check_name             status  severity  detail
─────────────────────  ──────  ────────  ───────────────────────────────────────────────────────
sample_size            pass    info      40000 rows: 13961 treated, 26039 control
treatment_prevalence   pass    info      34.9% of rows are treated ('1' vs '0')
positivity             pass    info      propensity spans [0.085, 1.000]; 0.5% of rows fall
                                         outside [0.05, 0.95]
balance                pass    info      largest weighted SMD is 0.023 on 'prior_spend'
outcome_variation      pass    info      outcome standard deviation is 21.1576
missing_data           pass    info      0 rows had at least one missing covariate; they were
                                         imputed, not dropped
dimensionality         pass    info      3 encoded features for 40000 rows (13333.3 per feature)
```

Every row has a `status`, so this drops straight into a dbt test or a CI query.

The one worth understanding is **balance** — whether the adjustment actually removed
the difference between the two groups:

```sql
SELECT covariate, round(smd_raw,3) AS smd_before, round(smd_weighted,3) AS smd_after, balanced
FROM do_balance('customers', treatment := 'got_discount', outcome := 'revenue',
                exclude := ['customer_id']);
```

```
covariate       smd_before  smd_after  balanced
──────────────  ──────────  ─────────  ────────
age                  0.009     -0.003  true
tenure_months        0.590     -0.000  true
prior_spend          0.477      0.023  true
```

Before adjustment the treated group differed from the control group by 0.59 standard
deviations in tenure. After weighting, essentially zero. That gap is the $10.62 of
the naive answer that was never the discount.

## Step 4 — try to break it

A number that survives no attempt to break it has not been tested.

```sql
SELECT method, round(original_estimate,2) AS original, round(refuted_estimate,2) AS refuted,
       passed, detail
FROM do_refute('customers', treatment := 'got_discount', outcome := 'revenue',
               exclude := ['customer_id'], method := 'placebo_treatment')
UNION ALL
SELECT method, round(original_estimate,2), round(refuted_estimate,2), passed, detail
FROM do_refute('customers', treatment := 'got_discount', outcome := 'revenue',
               exclude := ['customer_id'], method := 'random_common_cause');
```

```
method                original  refuted  passed  detail
────────────────────  ────────  ───────  ──────  ────────────────────────────────────────────
placebo_treatment         8.05     0.16  false   treatment permuted at random; a real effect
                                                 should collapse to zero
random_common_cause       8.05     8.05  true    an irrelevant covariate was added; the
                                                 estimate should not move
```

Read `passed` as "the refutation succeeded in breaking the estimate". The placebo test
shuffles the treatment column, so any effect it finds is noise: 8.05 collapses to 0.16.
`passed = false` there is the outcome you want.

Then the question nothing in the data can answer — **what if we missed a confounder?**

```sql
SELECT round(robustness_value,3) AS rv, round(e_value,3) AS e_value, interpretation
FROM do_sensitivity('customers', treatment := 'got_discount', outcome := 'revenue',
                    exclude := ['customer_id']);
```

```
rv     e_value  interpretation
─────  ───────  ──────────────────────────────────────────────────────────────────────
0.398    2.092  an unobserved confounder would have to explain 39.8% of the residual
                variance of both the treatment and the outcome to reduce this estimate
                to zero, and 39.2% to push the confidence interval across zero. Compare
                that against the strongest covariate you already measured
```

That last sentence is the whole method. 39.8% is a lot — more than any covariate here
explains — so an unmeasured confounder would have to be stronger than anything you
already know about. That is not proof. It is the shape of the argument you would have
to make against the result.

## Step 5 — the average is hiding the useful part

$8.05 is one number for 40,000 different people.

```sql
SELECT CASE WHEN m.tenure_months < 10 THEN 'a. 0-9 months'
            WHEN m.tenure_months < 20 THEN 'b. 10-19'
            WHEN m.tenure_months < 30 THEN 'c. 20-29'
            ELSE 'd. 30-40' END AS tenure_band,
       round(avg(c.cate),2) AS estimated_effect,
       round(avg(14.0 - 0.30*m.tenure_months),2) AS true_effect,
       round(avg(m.got_discount),3) AS share_targeted
FROM do_cate('customers', treatment := 'got_discount', outcome := 'revenue',
             exclude := ['customer_id'], id := 'customer_id') c
JOIN customers m ON m.customer_id = CAST(c.id AS BIGINT)
GROUP BY 1 ORDER BY 1;
```

```
tenure_band     estimated_effect  true_effect  share_targeted
──────────────  ────────────────  ───────────  ──────────────
a. 0-9 months              12.62        12.58           0.187
b. 10-19                    9.68         9.65           0.280
c. 20-29                    6.68         6.65           0.391
d. 30-40                    3.61         3.58           0.523
```

Two things. The estimates track the truth to within four cents in every band. And the
targeting ran **exactly backwards**: 18.7% of the customers it helps most got it,
against 52.3% of the ones it helps least.

You can see the same thing without any grouping, because DuckDo reports the treated
and untreated populations separately:

```sql
SELECT 'do_att' AS f, round(estimate,2) AS estimate FROM do_att(...)   -- 6.77
UNION ALL SELECT 'do_atc', round(estimate,2) FROM do_atc(...);         -- 8.73
```

`do_att` is what the campaign achieved on the people it reached (**6.77**). `do_atc` is
what it would have achieved on the people it skipped (**8.73**). When ATT is below ATC,
the targeting is upside down.

## Step 6 — what to do next time

The discount costs money. Say $8 per customer. Who is worth sending it to?

```sql
SELECT leaf, rule, n, round(mean_effect,2) AS mean_effect, cost, action,
       round(expected_gain,3) AS expected_gain
FROM do_optimal_policy('customers', treatment := 'got_discount', outcome := 'revenue',
                       exclude := ['customer_id'], depth := 1, threshold := 8.0);
```

```
leaf  rule                  n      mean_effect  cost  action        expected_gain
────  ────────────────────  ─────  ───────────  ────  ────────────  ─────────────
   0  tenure_months <= 20   20552        10.99   8.0  treat                 1.534
   1  tenure_months > 20    19448         4.93   8.0  do not treat          0.000
```

A rule you can ship, in the column's own units. The true break-even is
`14.00 − 0.30 × tenure = 8`, i.e. tenure = 20 — which is what it found from the data
alone.

`threshold` is the cost of treating one customer. Leave it at zero and the honest
answer is "treat everyone", because a free intervention with a positive effect should
go to everybody.

---

## What you have actually established

An effect of **$8.05** [7.90, 8.19], from a doubly-robust estimator, on 40,000 rows
whose covariate balance is confirmed, which collapses under a placebo test and does
not move under an irrelevant covariate, and which would need an unmeasured confounder
explaining 40% of the residual variance in both treatment and outcome to be explained
away. Concentrated in customers under twenty months' tenure.

**What you have not established** is that no such confounder exists. Nothing in this
document, or in any dataset, can do that. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for what
each estimate is resting on and which of those things you can check.

## Where to go next

- **[ASSUMPTIONS.md](ASSUMPTIONS.md)** — what has to be true, in plain language.
- **[FUNCTIONS.md](FUNCTIONS.md)** — all 35 functions with their full signatures.
- **A DAG, if you have one.** `do_validate` grades your covariate list against a graph
  and will tell you a variable you are adjusting for is a mediator or a collider — the
  two mistakes that make an answer worse the more carefully you make them.
