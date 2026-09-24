"""Round 8 of the student study, guides (R8-09): the four guide tutorials.

tutorials/run, evaluate, plot and discover join no two facts with ';' in
their prose. Code blocks, code comments and inline code are not checked.

Needs the docs source (SCMULTIBENCH_DOCS=<docs dir>); skipped without it.
"""
import re

import pytest

from tests.test_docs_r4 import _read, needs_docs


def _prose_lines(page):
    """The page's lines outside fenced code, without inline code or link targets."""
    out, fenced = [], False
    for line in _read(page).splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(re.sub(r"\]\([^)]*\)", "]", re.sub(r"`[^`]*`", "", line)))
    return out


@needs_docs
@pytest.mark.parametrize("page", ["run", "evaluate", "plot", "discover"])
def test_guide_prose_has_no_semicolon_joins(page):
    joined = [l.strip() for l in _prose_lines(f"tutorials/{page}.md") if ";" in l]
    assert not joined, f"tutorials/{page}.md joins facts with ';': {joined}"
