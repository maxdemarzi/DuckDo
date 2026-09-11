"""Check docs/FUNCTIONS.md against the extension it documents.

check_docs.py proves the SQL in the documentation still parses and that quoted
output is still produced. Neither catches the failure this script is for: a
function that gained a column, lost a parameter, or was never written up at all.
The reference is checked against the running extension rather than the source,
so what it is graded on is what a user actually gets.

Three checks, each of which has caught something:

  COVERAGE    Every registered do_* function has an entry, and every entry names
              a function that exists.

  COLUMNS     Every column a function returns is named somewhere in its entry.
              This found do_list_models documenting `max_features` for a column
              called `max_covariates`, and the graph functions documenting no
              return columns at all.

              An entry may defer to another - do_ate_by returns "the `do_ate`
              columns" - and a reference like that counts as documentation for
              the columns the other entry names.

  PARAMETERS  Every named parameter is documented somewhere in the reference.
              Shared parameters are registered on every function by one helper,
              so requiring each entry to repeat them would be noise; the check is
              that the reference mentions them at all. Positional placeholders
              (col0, col1, ...) are not parameters and are skipped.

Two functions are not run: do_download reaches the network, and do_dseparated is
a scalar function whose signature the entry states in its heading.

Dev-only. Not shipped with the extension.

    python scripts/check_functions.py [--duckdb build/release/duckdb.exe]
"""

import argparse
import collections
import os
import re
import subprocess
import sys

FIXTURES = """
SET threads = 1;
SELECT setseed(0.4);
CREATE TABLE demo AS
WITH u AS (
  SELECT i,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS x1,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS x2,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS e1,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS e2,
    sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS uc,
    random() AS r1, random() AS r2, random() AS r3, random() AS r4, random() AS r5, random() AS r6
  FROM range(1500) t(i)),
 a AS (SELECT *, (r1 < 1/(1+exp(-0.8*x1)))::INT AS t, (r2 < 0.5)::INT AS z,
       1.0 + 0.6*x1 + e2 AS dose,
       CASE WHEN r5 < 0.33 THEN 'A' WHEN r5 < 0.66 THEN 'B' ELSE 'C' END AS arm FROM u)
SELECT i, x1, x2, t, z, dose, arm,
  2.0*t + x1 + e1 AS y,
  ((0.3 + 0.8*z + 0.5*uc) > r3*1.8)::INT AS d,
  1.5*((0.3 + 0.8*z + 0.5*uc) > r3*1.8)::INT + uc + x1 + e2 AS y_iv,
  (r4 < 1/(1+exp(-(1.2*t - 0.6))))::INT AS m,
  1.8*(r4 < 1/(1+exp(-(1.2*t - 0.6))))::INT + 0.5*e1 AS y_fd,
  0.9*t + 0.5*x1 + e2 AS med,
  1.0*t + 1.2*(0.9*t + 0.5*x1 + e2) + x1 + e1 AS y_med,
  1.5*dose + x1 + e1 AS y_dose,
  CASE arm WHEN 'A' THEN 0.0 WHEN 'B' THEN 1.5 ELSE -1.0 END + x1 + e1 AS y_arm,
  least(-ln(r3 + 1e-12) / (0.3*exp(0.5*x1 - 0.5*t)), -ln(r6 + 1e-12) / 0.1) AS followup,
  (-ln(r3 + 1e-12) / (0.3*exp(0.5*x1 - 0.5*t)) <= -ln(r6 + 1e-12) / 0.1)::INT AS observed,
  CASE WHEN -ln(r6 + 1e-12) / 0.1 < -ln(r3 + 1e-12) / (0.3*exp(0.5*x1 - 0.5*t)) THEN 0
       WHEN r5 < 0.6 THEN 1 ELSE 2 END AS status
FROM a;

CREATE TABLE panel AS
SELECT u AS unit_id, p AS period,
       (u % 2 = 0 AND p >= 5)::INT AS treated,
       10.0 + 0.5*p + 2.0*(u % 2 = 0 AND p >= 5)::INT + (u % 7) * 0.3 + random() AS revenue,
       (u % 7) * 0.1 AS x
FROM range(60) a(u), range(1, 9) b(p);

CREATE TABLE months AS
WITH RECURSIVE sim(id, period, l, a, died, dropped) AS (
  SELECT id, 1, l, a,
         (random() < 1/(1+exp(-(-2.6 + 0.8*l - 0.7*a))))::INT,
         (random() < 1/(1+exp(-(-3.2 + 0.7*l))))::INT
  FROM (SELECT id, l, (random() < 1/(1+exp(-(-0.3 + 0.6*l))))::INT AS a
        FROM (SELECT i AS id, sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS l FROM range(3000) r(i)))
  UNION ALL
  SELECT id, period, l, a,
         (random() < 1/(1+exp(-(-2.6 + 0.8*l - 0.7*a))))::INT,
         (random() < 1/(1+exp(-(-3.2 + 0.7*l))))::INT
  FROM (SELECT id, period, l, (random() < 1/(1+exp(-(-0.3 + 0.6*l + 1.0*a_prev))))::INT AS a
        FROM (SELECT id, period + 1 AS period,
                     0.7*l - 0.6*a + 0.5*sqrt(-2*ln(random()+1e-12))*cos(2*pi()*random()) AS l, a AS a_prev
              FROM sim WHERE died = 0 AND dropped = 0 AND period < 6))
)
SELECT id, period, l, a, died, 1.0*period + l AS bp FROM sim;

CREATE TABLE site_results AS
SELECT * FROM do_ate('(SELECT * FROM demo WHERE i % 3 = 0)', treatment := 't', outcome := 'y', covariates := ['x1'])
UNION ALL
SELECT * FROM do_ate('(SELECT * FROM demo WHERE i % 3 = 1)', treatment := 't', outcome := 'y', covariates := ['x1'])
UNION ALL
SELECT * FROM do_ate('(SELECT * FROM demo WHERE i % 3 = 2)', treatment := 't', outcome := 'y', covariates := ['x1']);

CALL do_graph_create('check_dag', 'digraph {
  intent [latent];
  season -> discount;  season -> revenue;
  discount -> clicks;  clicks -> revenue;
  intent -> discount;  intent -> revenue;
  coupon_mail -> discount;
}');
"""

BASE = "'demo', treatment := 't', outcome := 'y', covariates := ['x1', 'x2']"
CALLS = {
    "do_ate": "do_ate(%s)" % BASE,
    "do_att": "do_att(%s)" % BASE,
    "do_atc": "do_atc(%s)" % BASE,
    "do_cate": "do_cate(%s)" % BASE,
    "do_ate_by": "do_ate_by(%s, by := ['arm'])" % BASE,
    "do_ate_pool": "do_ate_pool('site_results')",
    "do_ape": "do_ape('demo', treatment := 'dose', outcome := 'y_dose', covariates := ['x1'])",
    "do_dose_response": "do_dose_response('demo', treatment := 'dose', outcome := 'y_dose', "
                        "covariates := ['x1'], grid := 5)",
    "do_ate_levels": "do_ate_levels('demo', treatment := 'arm', outcome := 'y_arm', covariates := ['x1'])",
    "do_did": "do_did('panel', unit := 'unit_id', period := 'period', treatment := 'treated', outcome := 'revenue')",
    "do_event_study": "do_event_study('panel', unit := 'unit_id', period := 'period', treatment := 'treated', "
                      "outcome := 'revenue')",
    "do_synth": "do_synth('panel', unit := 'unit_id', period := 'period', treatment := 'treated', "
                "outcome := 'revenue')",
    "do_synth_path": "do_synth_path('panel', unit := 'unit_id', period := 'period', treatment := 'treated', "
                     "outcome := 'revenue')",
    "do_balance": "do_balance('demo', treatment := 't', covariates := ['x1', 'x2'])",
    "do_overlap": "do_overlap('demo', treatment := 't', covariates := ['x1', 'x2'])",
    "do_diagnose": "do_diagnose(%s)" % BASE,
    "do_refute": "do_refute(%s, method := 'placebo_treatment')" % BASE,
    "do_sensitivity": "do_sensitivity(%s)" % BASE,
    "do_graph_create": "do_graph_create('check_tmp', 'digraph { a -> b; }')",
    "do_graphs": "do_graphs()",
    "do_graph_drop": "do_graph_drop('check_tmp')",
    "do_identify": "do_identify(graph := 'check_dag', treatment := 'discount', outcome := 'revenue')",
    "do_validate": "do_validate(graph := 'check_dag', treatment := 'discount', outcome := 'revenue', "
                   "covariates := ['season'])",
    "do_discover": "do_discover('demo', columns := ['x1', 'x2', 'y', 'dose'], bootstrap := 5)",
    "do_discover_dot": "do_discover_dot('demo', columns := ['x1', 'x2', 'y', 'dose'], bootstrap := 5)",
    "do_iv": "do_iv('demo', treatment := 'd', outcome := 'y_iv', instrument := 'z', covariates := ['x1'])",
    "do_frontdoor": "do_frontdoor('demo', treatment := 't', outcome := 'y_fd', mediator := 'm', covariates := ['x1'])",
    "do_mediate": "do_mediate('demo', treatment := 't', outcome := 'y_med', mediator := 'med', covariates := ['x1'])",
    "do_msm": "do_msm('months', unit := 'id', period := 'period', treatment := 'a', outcome := 'bp', "
              "covariates := ['l'])",
    "do_msm_rmst": "do_msm_rmst('months', unit := 'id', period := 'period', treatment := 'a', event := 'died', "
                   "covariates := ['l'], bootstrap_reps := 0)",
    "do_rmst": "do_rmst('demo', treatment := 't', duration := 'followup', event := 'observed', "
               "covariates := ['x1'], horizon := 5.0)",
    "do_rmtl": "do_rmtl('demo', treatment := 't', duration := 'followup', event := 'status', cause := 1, "
               "covariates := ['x1'], horizon := 5.0)",
    "do_frame_summary": "do_frame_summary(%s)" % BASE,
    "do_counterfactual": "do_counterfactual(%s)" % BASE,
    "do_predict": "do_predict(%s, intervention := {'x1': 1.0})" % BASE,
    "do_policy_value": "do_policy_value('(SELECT *, x1 > 0 AS rule FROM demo)', treatment := 't', outcome := 'y', "
                       "covariates := ['x1', 'x2'], policy := 'rule')",
    "do_uplift": "do_uplift(%s)" % BASE,
    "do_optimal_policy": "do_optimal_policy(%s, depth := 1)" % BASE,
    "do_list_models": "do_list_models()",
    "do_models": "do_models()",
    "do_devices": "do_devices()",
}
# Columns named by the caller rather than the function: do_ate_by puts its grouping
# columns first, and here those are the fixture's own.
CALLER_NAMED = {"do_ate_by": {"arm"}}

NOT_RUN = {
    "do_download": "reaches the network",
    "do_dseparated": "scalar function; its signature is in the entry heading",
}
REFERENCE = re.compile(r"`(do_[a-z_]+)`")


def run(duckdb, sql):
    process = subprocess.run([duckdb, "-batch", "-list", "-noheader"], input=sql, capture_output=True, text=True)
    return process.stdout, process.stderr


def main():
    parser = argparse.ArgumentParser()
    default = "build/release/duckdb.exe" if os.name == "nt" else "build/release/duckdb"
    parser.add_argument("--duckdb", default=default)
    parser.add_argument("--docs", default="docs/FUNCTIONS.md")
    args = parser.parse_args()
    duckdb = os.path.abspath(args.duckdb)
    if not os.path.exists(duckdb):
        print("no duckdb binary at %s; build it first, or pass --duckdb" % duckdb)
        return 1

    statements = [FIXTURES]
    for name, call in CALLS.items():
        statements.append("SELECT 'COL|%s|' || column_name FROM (DESCRIBE SELECT * FROM %s);" % (name, call))
    statements.append("SELECT 'PARAM|' || function_name || '|' || p FROM "
                      "(SELECT function_name, unnest(parameters) AS p FROM duckdb_functions() "
                      "WHERE function_name LIKE 'do\\_%' ESCAPE '\\');")
    statements.append("SELECT DISTINCT 'FN|' || function_name FROM duckdb_functions() "
                      "WHERE function_name LIKE 'do\\_%' ESCAPE '\\';")
    out, err = run(duckdb, "\n".join(statements))

    columns = collections.defaultdict(list)
    parameters = collections.defaultdict(set)
    registered = set()
    for line in out.splitlines():
        bits = line.split("|")
        if bits[0] == "COL":
            columns[bits[1]].append(bits[2])
        elif bits[0] == "PARAM" and not re.fullmatch(r"col\d+", bits[2]):
            parameters[bits[1]].add(bits[2])
        elif bits[0] == "FN":
            registered.add(bits[1])

    doc = open(args.docs, encoding="utf8").read()
    parts = re.split(r"^### (.+)$", doc, flags=re.M)
    entries = {}
    for i in range(1, len(parts), 2):
        body = parts[i] + parts[i + 1]
        for fn in re.findall(r"do_[a-z_]+", parts[i]):
            entries[fn] = entries.get(fn, "") + body

    failures = []
    if err.strip():
        failures.append("the fixture SQL or a call did not run:\n    " + err.strip().splitlines()[0])

    missing_entry = sorted(f for f in registered if f not in entries)
    stale_entry = sorted(f for f in entries if f not in registered)
    if missing_entry:
        failures.append("registered but undocumented: " + ", ".join(missing_entry))
    if stale_entry:
        failures.append("documented but not registered: " + ", ".join(stale_entry))

    for fn in sorted(columns):
        body = entries.get(fn, "")
        # An entry may defer to another: "the `do_ate` columns".
        for other in set(REFERENCE.findall(body)) - {fn}:
            body += entries.get(other, "")
        skip = CALLER_NAMED.get(fn, set())
        unnamed = [c for c in columns[fn] if c not in skip and not re.search(r"`[^`]*\b" + re.escape(c) + r"\b[^`]*`", body)]
        if unnamed:
            failures.append("%s returns %s, which its entry never names" % (fn, ", ".join(unnamed)))

    for name in sorted({p for names in parameters.values() for p in names}):
        if ("`%s`" % name) not in doc and ("%s :=" % name) not in doc:
            holders = sorted(f for f, names in parameters.items() if name in names)
            failures.append("parameter %s is registered on %s and documented nowhere"
                            % (name, ", ".join(holders[:4]) + (" ..." if len(holders) > 4 else "")))

    print("%d functions registered, %d documented, %d exercised" % (len(registered), len(entries), len(columns)))
    for fn, why in sorted(NOT_RUN.items()):
        print("  %s not run: %s" % (fn, why))
    print()
    if failures:
        for line in failures:
            print("  " + line)
        print("\n%d problem(s) in %s" % (len(failures), args.docs))
        return 1
    print("The function reference matches the extension: every function documented, every column\n"
          "named in its entry, every parameter documented somewhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
