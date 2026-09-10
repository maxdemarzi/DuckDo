"""MkDocs hooks for the DuckDo documentation site. Dev and CI only; not shipped.

The Markdown in docs/ and the README is written to read correctly on GitHub, and
the site takes it as it is. It fixes only what a site needs and GitHub does not:
the README becomes the home page, and a link that points outside docs/ - a
script, a source file, the licence - goes to that file on GitHub rather than to
a page the site does not have. mkdocs build --strict then fails on any link that
is still broken, so the site cannot drift from the files it is built from.
"""

import os
import re

from mkdocs.structure.files import File

REPO = "https://github.com/maxdemarzi/DuckDo/blob/main/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\]\(([^)\s]+)\)")


def on_files(files, config):
    with open(os.path.join(ROOT, "README.md"), encoding="utf8") as handle:
        files.append(File.generated(config, "index.md", content=handle.read()))
    return files


def _rewrite(target, from_readme):
    if re.match(r"^(https?:|mailto:|#)", target):
        return target
    path, _, anchor = target.partition("#")
    # Where the link points, as a path from the repository root.
    repo_path = path if from_readme else os.path.normpath(os.path.join("docs", path)).replace("\\", "/")
    if repo_path == "README.md":
        new = "index.md"
    elif repo_path.startswith("docs/") and repo_path.endswith(".md"):
        new = repo_path[len("docs/"):]
    else:
        new = REPO + repo_path
    return new + ("#" + anchor if anchor else "")


def on_page_markdown(markdown, page, config, files):
    from_readme = page.file.src_uri == "index.md"
    return LINK.sub(lambda m: "](" + _rewrite(m.group(1), from_readme) + ")", markdown)
