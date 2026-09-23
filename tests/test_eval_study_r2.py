"""Scoring record and metric definitions after the second student study.

M34: evaluate states the neighbourhood each metric uses (cLISI and iLISI use
scib's k0 = 90, not the 15-neighbour graph), states the scib requirement
instead of 'scib 1.x', and records the installed scib in attrs. The catalog
describes cLISI without 'how few', which read as 'higher = fewer'.
M35: a wide CSV read back carries no scoring record; to_long says so in a
warning and writes scored_with = 'unknown' instead of dropping the column.
"""
import inspect
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.eval import pipeline

pytest.importorskip("scib")


def _blobs(n_per=40, n_labels=3, dims=5, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 8, size=(n_labels, dims))
    emb = np.vstack([centres[i] + rng.normal(0, 1, size=(n_per, dims))
                     for i in range(n_labels)])
    lab = np.array([f"t{i}" for i in range(n_labels) for _ in range(n_per)])
    return emb, lab


def _user_warnings(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


# --- M34 ---------------------------------------------------------------------

def test_evaluate_records_the_installed_scib():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    assert wide.attrs["scib_version"] == version("scib")
    long = mtb.to_long(wide, method="M")
    assert long.attrs["scib_version"] == version("scib")
    # scored_with keeps its three parts
    assert long["scored_with"].iloc[0] == f"none/none/{mtb.__version__}"


def test_evaluate_notes_name_each_neighbourhood():
    doc = " ".join(inspect.getdoc(pipeline.evaluate).split())
    assert "scib 1.x" not in doc
    assert "``scib >= 1.1``" in doc and 'attrs["scib_version"]' in doc
    assert "cLISI and iLISI use k0 = 90 neighbours (scib default; perplexity k0/3)" in doc
    assert ("the Leiden sweep (ARI, NMI, iF1) and GC use scanpy's default neighbour "
            "graph (15 neighbours)") in doc
    assert "ARI can be slightly below 0, which means a random clustering" in doc
    # the table is a fenced block: a trailing '::' rendered as text on the site
    assert not [ln for ln in inspect.getdoc(pipeline.evaluate).splitlines()
                if ln.rstrip().endswith("::")]


def test_stated_scib_facts_match_the_requirement_and_scib_defaults():
    """The Notes restate two facts owned elsewhere; fail when either moves."""
    import re
    from pathlib import Path
    pyproject = (Path(pipeline.__file__).parents[2] / "pyproject.toml").read_text()
    spec = re.search(r'"scib>=([0-9.]+)"', pyproject).group(1)
    doc = " ".join(inspect.getdoc(pipeline.evaluate).split())
    assert f"``scib >= {spec}``" in doc
    import scib.metrics as me
    for fn in (me.clisi_graph, me.ilisi_graph):
        assert inspect.signature(fn).parameters["k0"].default == 90, fn


def test_catalog_describes_clisi_in_plain_direction():
    d = mtb.catalog.metrics()
    text = d.loc[d["metric"] == "cLISI", "description"].iloc[0]
    assert "how few" not in text
    assert "how much each cell's neighbours share its cell type" in text
    assert "higher = better" in text


# --- M35 ---------------------------------------------------------------------

def test_wide_csv_read_back_is_marked_unknown(tmp_path):
    emb, lab = _blobs()
    mtb.evaluate(emb, labels=lab, metrics=["ASW"]).to_csv(tmp_path / "metric.csv")
    back = pd.read_csv(tmp_path / "metric.csv", index_col=0)
    long, msgs = _user_warnings(mtb.to_long, back, method="X", dataset="D11",
                                category="vertical")
    assert long.columns.tolist() == pipeline.LONG_COLUMNS + ["scored_with"]
    assert set(long["scored_with"]) == {"unknown"}
    assert msgs == [pipeline.NO_SCORING_RECORD]
    assert msgs[0].startswith("these scores carry no record of how they were scored")
    assert "mtb.to_long(...)" in msgs[0]


def test_evaluate_output_converts_without_a_warning():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    long, msgs = _user_warnings(mtb.to_long, wide, method="X")
    assert msgs == []
    assert "unknown" not in set(long["scored_with"])
