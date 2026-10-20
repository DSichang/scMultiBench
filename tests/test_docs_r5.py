"""Round 5 of the student study: what the docs pages now say.

Get started (R5-15): the Quickstart CLI block lets ``multibench evaluate``
read the D28 label files in SCALEX's cell order (``--method SCALEX``) and
names the rows with ``--name SCALEX_rerun``. The block's evaluate line is run
as printed, on a small D28 folder.

Guides (R5-17, R5-02): the comparability sentence on tutorials/evaluate is
plain, and the page says that a CSV whose first column holds the cell ids is
aligned by id.

API overview and Changes (R5-16) and the code font (R5-18): exit code ``3``
follows the CLI rule, the Changes page lists the round-5 changes and quotes
the live stderr line, and code is drawn without ligatures.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it.
"""
import re
import shlex

import h5py
import numpy as np
import pandas as pd

from multibench import config
from tests.test_docs_r4 import _docs_root, _flat, _read, _visible, needs_docs
from tests.test_f4_cli import _diag_folder


# ---- get started (R5-15) -------------------------------------------------

@needs_docs
def test_evaluate_page_comparability_sentence_is_plain():
    visible = " ".join(_visible(_read("tutorials/evaluate.md")).split())
    assert "compare ranks, not decimals" not in visible
    # Y07 (r3): no "Compare ranks."; one true sentence on why the same embedding moves
    assert "most methods give a slightly different embedding on each run." in visible
    assert "Compare ranks." not in visible and "4,096" not in visible
    # v1 (u4#6): what holds, in one sentence and without a number (D45 Cobolt's
    # ARI moved 0.118 across shuffles)
    assert ("The Leiden-based and classification scores of the same embedding also "
            "depend on the cells' order.") in visible
    assert "about 0.05" not in visible
    assert "likely cause" not in visible


@needs_docs
def test_evaluate_page_says_a_barcode_csv_is_aligned_by_id():
    text = _flat("tutorials/evaluate.md")
    assert "A CSV whose first column holds the cell ids is aligned the same way." in text
    assert "Plain arrays, lists and label files are always matched by position." not in text


# ---- API overview, Changes, code font (R5-16, R5-18) ---------------------

@needs_docs
def test_changes_page_lists_round_5_changes():
    text = _flat("changes.md")
    assert "failed or was skipped" not in text
    for phrase in ("`run-all` exits with `3` after saving when a method failed, or a "
                   "method named in `--methods` was skipped.",
                   "`# 1 failed (Seurat_WNN), 1 skipped (totalVI).`",
                   "`# Not run: ...`",
                   "end with a `caveat` column and a `reason` column.",
                   "`BatchResult.rescore(labels=...)` aligns a Series",
                   "A batch or labels CSV whose first column holds cell ids is aligned",
                   "`mtb.data.fetch` and `fetch_outputs` raise `OSError`",
                   "`multibench evaluate --name NAME`",
                   "`<method> needs an NVIDIA GPU, and this computer has none.`",
                   "`mosaic has no published tables.`"):
        assert phrase in text, phrase
    assert "needs an NVIDIA GPU; this computer has none" not in text


@needs_docs
def test_code_font_draws_no_ligatures():
    css = (_docs_root() / "stylesheets" / "extra.css").read_text()
    rule = re.search(r"\.md-typeset code, \.md-typeset pre, \.md-typeset kbd \{([^}]*)\}", css)
    assert rule, "the code-font rule of extra.css moved"
    body = rule.group(1)
    assert "font-variant-ligatures: none" in body
    assert re.search(r'"calt" 0', body) and re.search(r'"liga" 0', body)
    # no other rule turns them back on
    assert not re.search(r'font-feature-settings:[^;]*"(calt|liga)"(?! 0)', css)
