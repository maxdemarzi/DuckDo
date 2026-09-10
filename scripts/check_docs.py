"""Check the SQL in the documentation, at the strength each document claims.

Two tiers, because the documents make two different promises:

  EXECUTED   The tutorial and the worked examples quote real output. Every block
             is executed in document order against a fresh database, and every
             number in the block quoted underneath it has to be a number the
             query still produces.

             Running the SQL was the original check and it was not enough. It
             proves the query still parses and binds; it says nothing about the
             figures, and the figures are what the prose reasons about. Both
             documents were found quoting output their own queries no longer
             produced - a LATE of 0.1174 where the query returned 0.1236 - while
             passing a checker that only ran them. A document that argues from
             stale numbers is worse than one with no numbers at all.

  PARSED     The README and the assumptions guide use illustrative fragments
             against tables that do not exist ('customers', 'sales'). Executing
             them would mean inventing fixtures and distorting the prose. They
             are parsed instead, which catches typos, unbalanced parentheses and
             malformed named-parameter syntax - but NOT a stale column name.
             That gap is real; it is the price of a landing page that reads well.

And one more, stricter than both:

  HELLO      The hello_world block in description.yml is what the community
             extension page shows a stranger, verbatim. It is executed on a
             fresh database, and the interval it prints has to contain the true
             effect its own comment promises. Running without error is not
             enough for the one query most people will ever copy.

Blocks containing `...` are deliberately abbreviated, and are counted and
reported rather than silently ignored.

Dev-only. Not shipped with the extension.

    python scripts/check_docs.py [--duckdb build/release/duckdb.exe]
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap

EXECUTED = ["docs/TUTORIAL.md", "docs/EXAMPLES.md"]
PARSED = ["README.md", "docs/ASSUMPTIONS.md", "docs/FUNCTIONS.md",
          "docs/REPRODUCIBILITY.md", "docs/BENCHMARKS.md"]
BLOCK = re.compile(r"```sql\n(.*?)```", re.DOTALL)
# A ```sql block followed by a plain fenced block: the query and the output it
# is documented as producing.
# Neither capture may contain a fence. Without the lookaheads the lazy `.*?`
# happily runs past its own block's closing fence and through the prose and
# the next query, so only the last pair in a run ever matched.
PAIR = re.compile(r"```sql\n((?:(?!```).)*?)```\n[ \t]*\n```\n((?:(?!```).)*?)```", re.DOTALL)
NUMBER = re.compile(r"-?\d+\.\d+|-?\d+")


def blocks(path):
    with open(path, encoding="utf8") as handle:
        found = BLOCK.findall(handle.read())
    runnable = [b for b in found if "..." not in b]
    return runnable, len(found) - len(runnable)


def quoted_outputs(path):
    """Map a runnable block's index to the output the document claims it prints.

    Keyed by the SQL text so it lines up with the executed list, which drops the
    abbreviated blocks and would otherwise shift every index after the first."""
    with open(path, encoding="utf8") as handle:
        text = handle.read()
    by_sql = {sql: output for sql, output in PAIR.findall(text)}
    runnable, _ = blocks(path)
    return {i: by_sql[b] for i, b in enumerate(runnable) if b in by_sql}


def numbers(text):
    """Every numeric token, as floats. Comparing text would trip over 0.145 and
    0.1450, which are the same number differently rendered."""
    out = []
    for token in NUMBER.findall(text):
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


MARK = "@@DUCKDO_BLOCK_%d@@"


def run_executed(duckdb, doc):
    """Execute every block, in order, as one script - later blocks select from
    tables earlier ones create - then check that what came back is what the
    document says came back.

    Running the SQL only proves it still parses and binds. It says nothing about
    whether the numbers underneath it are still the numbers it produces, and a
    document whose prose reasons about stale figures is worse than one with no
    figures at all. A sentinel between blocks makes the single transcript
    splittable, so each block's own output can be compared against its own
    quoted block."""
    runnable, skipped = blocks(doc)
    expected = quoted_outputs(doc)

    script = []
    for i, block in enumerate(runnable):
        script.append("SELECT '" + (MARK % i) + "' AS mark;")
        script.append(block)
    script.append("SELECT '" + (MARK % len(runnable)) + "' AS mark;")

    handle, path = tempfile.mkstemp(suffix=".sql")
    os.close(handle)
    try:
        with open(path, "w", encoding="utf8", newline="\n") as out:
            out.write("\n".join(script))
        proc = subprocess.run([duckdb, "-c", ".read " + path.replace("\\", "/")],
                              capture_output=True, encoding="utf8", errors="replace")
    finally:
        os.unlink(path)

    errors = [line.strip() for line in proc.stdout.splitlines() + proc.stderr.splitlines()
              if "Error:" in line]
    if errors:
        return len(runnable), skipped, errors, 0

    pieces = re.split(r"@@DUCKDO_BLOCK_(\d+)@@", proc.stdout)
    actual = {}
    for k in range(1, len(pieces) - 1, 2):
        actual[int(pieces[k])] = pieces[k + 1]

    checked = 0
    for index, quoted in sorted(expected.items()):
        produced = numbers(actual.get(index, ""))
        missing = []
        for value in numbers(quoted):
            # The quoted block is a hand-aligned rendering, so match on value
            # rather than on text, and allow the last displayed digit to differ.
            if not any(abs(value - other) <= 5e-4 * max(1.0, abs(value)) for other in produced):
                missing.append(value)
        checked += 1
        if missing:
            shown = ", ".join(("%g" % v) for v in missing[:6])
            errors.append("block %d quotes %d value(s) it no longer produces: %s"
                          % (index, len(missing), shown))
    return len(runnable), skipped, errors, checked


def split_sql(text):
    """Split on semicolons that are not inside a string literal.

    A DOT graph passed to do_graph_create is a single-quoted literal full of
    semicolons, so a naive split shreds it into nonsense and then reports the
    nonsense as a documentation error."""
    parts, current, in_string = [], [], False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "'":
            # '' inside a literal is an escaped quote, not the end of one.
            if in_string and i + 1 < len(text) and text[i + 1] == "'":
                current.append("''")
                i += 2
                continue
            in_string = not in_string
            current.append(c)
        elif c == ";" and not in_string:
            parts.append("".join(current))
            current = []
        else:
            current.append(c)
        i += 1
    parts.append("".join(current))
    return parts


def statements(block):
    """Split a block into statements, ignoring dot-commands and comment-only text."""
    out = []
    for raw in split_sql(block):
        cleaned = "\n".join(line for line in raw.splitlines()
                            if not line.strip().startswith("--") and not line.strip().startswith("."))
        if cleaned.strip():
            out.append(cleaned.strip())
    return out


def run_parsed(duckdb, doc):
    """Parse every statement without binding it, so illustrative fragments against
    tables that do not exist are still checked for syntax."""
    runnable, skipped = blocks(doc)
    checked, errors = 0, []
    for block in runnable:
        for statement in statements(block):
            literal = statement.replace("'", "''")
            proc = subprocess.run(
                [duckdb, "-csv", "-noheader", "-c",
                 "SELECT json_serialize_sql('%s')" % literal],
                capture_output=True, encoding="utf8", errors="replace")
            checked += 1
            text = proc.stdout.strip()
            if proc.returncode != 0:
                errors.append("%s -> %s" % (statement.splitlines()[0][:60], proc.stderr.strip()[:120]))
                continue
            try:
                parsed = json.loads(text.strip('"').replace('""', '"'))
            except Exception:  # noqa: BLE001 - a shape we do not recognise is not a parse failure
                continue
            # A non-SELECT statement reaches "not implemented" only by parsing
            # successfully first - json_serialize_sql reports error_type
            # "parser" for anything malformed, whatever the statement kind. So
            # only a parser error is a documentation failure.
            if isinstance(parsed, dict) and parsed.get("error") and parsed.get("error_type") == "parser":
                errors.append("%s -> %s" % (statement.splitlines()[0][:60],
                                            parsed.get("error_message", "")[:120]))
    return checked, skipped, errors


def hello_world(path="description.yml"):
    """The hello_world block from the community-extension descriptor, dedented."""
    with open(path, encoding="utf8") as handle:
        lines = handle.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "hello_world: |")
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return textwrap.dedent("\n".join(body))


def run_hello_world(duckdb, path="description.yml", truth=2.0):
    """The first thing a stranger runs. The community page shows it verbatim, so
    it has to run on a fresh database with no download, and give the answer its
    own comment promises: an interval around a true effect of exactly 2.0."""
    handle, script = tempfile.mkstemp(suffix=".sql")
    os.close(handle)
    try:
        with open(script, "w", encoding="utf8", newline="\n") as out:
            out.write(hello_world(path))
        proc = subprocess.run([duckdb, "-csv", "-c", ".read " + script.replace("\\", "/")],
                              capture_output=True, encoding="utf8", errors="replace")
    finally:
        os.unlink(script)
    errors = [line.strip() for line in proc.stdout.splitlines() + proc.stderr.splitlines() if "Error:" in line]
    if errors:
        return errors, "did not run"
    rows = list(csv.reader(io.StringIO(proc.stdout)))
    for i, row in enumerate(rows[:-1]):
        if "ci_low" in row and "ci_high" in row:
            low = float(rows[i + 1][row.index("ci_low")])
            high = float(rows[i + 1][row.index("ci_high")])
            if low <= truth <= high:
                return [], "ran; [%g, %g] covers the true %g" % (low, high, truth)
            return ["the interval [%g, %g] misses the true effect %g" % (low, high, truth)], "wrong answer"
    return ["printed no interval to check"], "no interval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=os.path.join("build", "release", "duckdb.exe"))
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print("duckdb binary not found at %s" % args.duckdb)
        return 2

    failures = 0
    print("%-24s %-10s %-8s %s" % ("document", "tier", "status", "detail"))
    print("-" * 72)
    for doc, tier in [(d, "EXECUTED") for d in EXECUTED] + [(d, "PARSED") for d in PARSED]:
        if not os.path.exists(doc):
            continue
        if tier == "EXECUTED":
            count, skipped, errors, checked = run_executed(args.duckdb, doc)
            detail = "%d blocks, %d abbreviated, %d outputs verified" % (count, skipped, checked)
        else:
            count, skipped, errors = run_parsed(args.duckdb, doc)
            detail = "%d statements, %d abbreviated" % (count, skipped)
        status = "ok" if not errors else "FAILED"
        failures += bool(errors)
        print("%-24s %-10s %-8s %s" % (doc, tier, status, detail))
        for line in errors[:5]:
            print("    %s" % line)

    if os.path.exists("description.yml"):
        errors, detail = run_hello_world(args.duckdb)
        failures += bool(errors)
        print("%-24s %-10s %-8s %s" % ("description.yml", "EXECUTED", "ok" if not errors else "FAILED",
                                       "hello_world " + detail))
        for line in errors[:5]:
            print("    %s" % line)

    print()
    if failures:
        print("%d document(s) contain SQL that does not check out" % failures)
        return 1
    print("Documentation SQL checks out. EXECUTED docs ran and match their quoted\n"
          "output; PARSED docs parsed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
