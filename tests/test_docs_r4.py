"""Round 4 of the student study: what the docs pages now say (get-started
R4-15 to R4-17, run guide R4-18, API overview and Changes R4-19), checked
against the live package where the page quotes a number or runs code.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it. The round-3 assertions that R4-16 replaced were re-pointed
in place (tests/test_docs_r3.py, tests/test_review_f3_int.py,
tests/test_docs_consistency.py).
"""
import os
from pathlib import Path

import pytest

import multibench as mtb


def _docs_root():
    root = os.environ.get("SCMULTIBENCH_DOCS")
    return Path(root) if root and Path(root).is_dir() else None


needs_docs = pytest.mark.skipif(_docs_root() is None, reason="SCMULTIBENCH_DOCS not set")


def _read(page):
    return (_docs_root() / page).read_text()


def _flat(page):
    return " ".join(_read(page).split())


def _visible(text):
    """The page without its collapsed ``??? `` Details blocks."""
    out, skip = [], False
    for line in text.splitlines():
        if line.startswith("??? "):
            skip = True
            continue
        if skip and line.strip() and not line.startswith("    "):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)


# ---- R4-18: the ATAC override; one Multiome folder serves every method ------
@needs_docs
def test_run_guide_skips_an_atac_mismatch_and_needs_one_multiome_folder():
    """The layer-1 sentence stays true after R4-01 (a named method is skipped
    too). The override allow_atac_mismatch= is deprecated in 0.4.0 and left
    off the page. The recipe that drew the results of a peak folder and a
    gene-activity folder as one dataset is gone: Matilda, scMDC and UnitedNet
    read peaks, so one Multiome folder serves every vertical RNA + ATAC
    method."""
    run = _read("tutorials/run.md")
    assert "also when `methods=` names it" in _flat("tutorials/run.md")
    assert "also when `methods=` names it" in " ".join(_visible(run).split())
    assert "allow_atac_mismatch" not in run and "--allow-atac-mismatch" not in run
    assert mtb.find_methods("vertical", modalities=["atac_gas"]) == []
    assert "Vertical, mosaic and cross methods read ATAC as peaks." in _flat("tutorials/run.md")
    assert "gene-activity folder" not in run and "second folder" not in _flat("tutorials/run.md")
    assert '.assign(dataset="MYMULTIOME")' not in run


# ---- R4-19: Changes page (the API-page exit-code sentence: tests/test_docs_r5.py) --
