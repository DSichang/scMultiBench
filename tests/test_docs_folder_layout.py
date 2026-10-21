"""The run guide's "Dataset folder layout" section against the registry.

The section lists the files of a dataset folder, the ATAC files of each
diagonal method and the batch patterns of mosaic and cross. Messages point to
it (``resolve.LAYOUT_URL``). Its method lists are copies of the registry, so
these tests fail when a method is added or changes what it reads.

Needs the docs source (SCMULTIBENCH_DOCS=<docs dir>); skipped without it.
"""
import os
from pathlib import Path

import pytest

import multibench as mtb
from multibench import workflow as W
from multibench._text import _and_list
from multibench.engine import resolve
from multibench.engine.schema import _batch_of, base_modality

DOCS = os.environ.get("SCMULTIBENCH_DOCS")
pytestmark = pytest.mark.skipif(not (DOCS and Path(DOCS).is_dir()),
                                reason="SCMULTIBENCH_DOCS not set")
HEADING = "### Dataset folder layout { #folder-layout }"


def _section() -> str:
    text = (Path(DOCS) / "tutorials" / "run.md").read_text()
    assert HEADING in text
    return text.split(HEADING, 1)[1].split("\n### ", 1)[0]


def _rows(first: str) -> list[list[str]]:
    """The cells of the table rows whose first cell is ``first``."""
    out = []
    for line in _section().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == first:
            out.append(cells)
    return out


def test_messages_point_to_the_section():
    assert resolve.LAYOUT_URL.endswith("/tutorials/run/#folder-layout")


@pytest.mark.parametrize("category", ["mosaic", "cross"])
def test_the_batch_pattern_table_is_the_registrys(category):
    want = []
    for roles, ids in W.batch_patterns(category):
        per: dict = {}
        for r in roles:
            per.setdefault(_batch_of(r), []).append(base_modality(r))
        want.append(([("+".join(ms)) for _, ms in sorted(per.items())], ", ".join(ids)))
    got = [([c for c in row[1:-1] if c], row[-1]) for row in _rows(category)]
    assert got == want


def test_the_diagonal_atac_table_is_the_registrys():
    peak = mtb.find_methods("diagonal", modalities=["atac_peak"])
    gas = mtb.find_methods("diagonal", modalities=["atac_gas"])
    both = list(dict.fromkeys(s.id for s, _v, _c, mods in W._variant_rows("diagonal")
                              if {"atac_peak", "atac_gas"} <= set(mods)))
    table = {row[0]: row[1] for row in _rows("both") + _rows("`atac_peak.h5`")
             + _rows("`atac_gas.h5`")}
    assert table == {"both": ", ".join(both),
                     "`atac_peak.h5`": ", ".join(m for m in peak if m not in both),
                     "`atac_gas.h5`": ", ".join(m for m in gas if m not in both)}


def test_the_several_batch_diagonal_methods_are_the_registrys():
    several = resolve._several_batch_diagonal_methods()
    assert (f"For several RNA and several ATAC batches in diagonal, {_and_list(several)} "
            f"read exactly 3 of each") in " ".join(_section().split())
