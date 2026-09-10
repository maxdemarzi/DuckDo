"""Check the SQL in the documentation, at the strength each document claims.

Two tiers, because the documents make two different promises:

  EXECUTED   The tutorial and the worked examples quote real output. Quoting real
             output is worth nothing if the queries above it no longer run, so
             every block is executed in document order against a fresh database.
             A failure here means the docs are lying.

  PARSED     The README and the assumptions guide use illustrative fragments
             against tables that do not exist ('customers', 'sales'). Executing
             them would mean inventing fixtures and distorting the prose. They
             are parsed instead, which catches typos, unbalanced parentheses and
             malformed named-parameter syntax - but NOT a stale column name.
             That gap is real; it is the price of a landing page that reads well.

Blocks containing `...` are deliberately abbreviated, and are counted and
reported rather than silently ignored.

Dev-only. Not shipped with the extension.

    python scripts/check_docs.py [--duckdb build/release/duckdb.exe]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

EXECUTED = ["docs/TUTORIAL.md", "docs/EXAMPLES.md"]
PARSED = ["README.md", "docs/ASSUMPTIONS.md", "docs/FUNCTIONS.md",
          "docs/REPRODUCIBILITY.md", "docs/BENCHMARKS.md"]
BLOCK = re.compile(r"```sql\n(.*?)```", re.DOTALL)


def blocks(path):
    with open(path, encoding="utf8") as handle:
        found = BLOCK.findall(handle.read())
    runnable = [b for b in found if "..." not in b]
    return runnable, len(found) - len(runnable)


def run_executed(duckdb, doc):
    """Execute every block, in order, as one script - later blocks select from
    tables earlier ones create."""
    runnable, skipped = blocks(doc)
    handle, path = tempfile.mkstemp(suffix=".sql")
    os.close(handle)
    try:
        with open(path, "w", encoding="utf8", newline="\n") as out:
            out.write("\n".join(runnable))
        proc = subprocess.run([duckdb, "-c", ".read " + path.replace("\\", "/")],
                              capture_output=True, encoding="utf8", errors="replace")
    finally:
        os.unlink(path)
    errors = [line.strip() for line in proc.stdout.splitlines() + proc.stderr.splitlines()
              if "Error:" in line]
    return len(runnable), skipped, errors


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
        count, skipped, errors = (run_executed if tier == "EXECUTED" else run_parsed)(args.duckdb, doc)
        status = "ok" if not errors else "FAILED"
        failures += bool(errors)
        unit = "blocks" if tier == "EXECUTED" else "statements"
        print("%-24s %-10s %-8s %d %s, %d abbreviated"
              % (doc, tier, status, count, unit, skipped))
        for line in errors[:5]:
            print("    %s" % line)

    print()
    if failures:
        print("%d document(s) contain SQL that does not check out" % failures)
        return 1
    print("Documentation SQL checks out. EXECUTED docs ran; PARSED docs parsed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
