"""Study round 5, work package wf: R5-01, R5-02, R5-03, R5-12 and R5-13.

R5-01: rescore(labels=) matched a barcode-indexed Series by position, and a
batch Series given with it was forced to the embedding rows.

R5-02: a batch or labels CSV written with its index (barcodes in the first
column) was matched by position wherever the target carries cell ids.

R5-03: summary.csv had no reason for a SKIPPED row; the run-all stderr line
called a skip a failure; the exit-code texts did not say that only a skip of
a method named in --methods sets exit 3.

R5-12: reference Notes without dash tails.

R5-13: no docstring list item has a continuation line that starts with
'- ' (Markdown renders it as a nested bullet).
"""
import ast
from pathlib import Path

import multibench

PKG = Path(multibench.__file__).parent


# ======================================================================= R5-13
def _dash_continuations(doc: str) -> list[str]:
    """Lines of ``doc`` that continue a list item but start with ``- ``.

    A continuation sits two spaces right of its item's ``- ``; starting it
    with ``- `` makes it a nested bullet. A nested list introduced by a line
    ending in ``:`` is left alone.
    """
    lines, found, item = doc.splitlines(), [], None
    for i, line in enumerate(lines):
        text = line.lstrip()
        indent = len(line) - len(text)
        if not text:
            item = None
            continue
        if (text.startswith("- ") and item is not None and indent == item + 2
                and not lines[i - 1].rstrip().endswith(":")):
            found.append(line)
        if text.startswith(("- ", "* ")):
            item = indent
        elif item is not None and indent <= item:
            item = None
    return found


def test_no_list_item_continues_on_a_line_starting_with_a_dash():
    bad = []
    for path in sorted(PKG.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                doc = ast.get_docstring(node, clean=True) or ""
                bad += [f"{path.relative_to(PKG)} {getattr(node, 'name', '<module>')}: "
                        f"{line.strip()}" for line in _dash_continuations(doc)]
    assert not bad, "\n".join(bad)


def test_the_check_sees_a_dash_continuation():
    doc = "- An item that runs on\n  - ``ValueError``, before any method runs.\n"
    assert _dash_continuations(doc) == ["  - ``ValueError``, before any method runs."]
    nested = "- Three kinds:\n  - one\n  - two\n"
    assert _dash_continuations(nested) == []
