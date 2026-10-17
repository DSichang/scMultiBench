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
    wide = mtb.evaluate(emb, labels=lab, metrics=["cLISI"])
    assert wide.attrs["scib_version"] == version("scib")
    long = mtb.to_long(wide, method="M")
    assert long.attrs["scib_version"] == version("scib")
    # scored_with keeps its three parts
    assert long["scored_with"].iloc[0] == f"none/none/{mtb.__version__}"


def test_evaluate_notes_name_each_neighbourhood():
    doc = " ".join(inspect.getdoc(pipeline.evaluate).split())
    assert "scib 1.x" not in doc
    # 0.4: the Notes that described each neighbourhood are cut from the
    # docstring; the retired wordings stay out of it
    assert "nearest by path length on it" not in doc
    assert "k0 = 90 neighbours" not in doc
    # the table is a fenced block: a trailing '::' rendered as text on the site
    assert not [ln for ln in inspect.getdoc(pipeline.evaluate).splitlines()
                if ln.rstrip().endswith("::")]


def test_stated_scib_facts_match_the_requirement_and_scib_defaults():
    """The scib defaults the LISI scores rely on; fail when one moves."""
    import scib.metrics as me
    for fn in (me.clisi_graph, me.ilisi_graph):
        assert inspect.signature(fn).parameters["k0"].default == 90, fn
    # the graph LISI walks on: scib recomputes a 15-neighbour graph for
    # type_="embed", perplexity defaults to k0 / 3, and a cell without k0
    # reachable cells gets a Simpson index of 1 (one label)
    from scib.metrics import lisi
    knn = inspect.getsource(lisi.recompute_knn)
    assert 'if type_ == "embed":' in knn and "n_neighbors=15" in knn
    graph_py = inspect.getsource(lisi.lisi_graph_py)
    assert "np.floor(n_neighbors / 3)" in graph_py
    assert "simpson[i] = 1" in inspect.getsource(lisi.compute_simpson_index_graph)
    # the shortest-path search reads the connectivity weights as edge lengths
    assert 'connectivities = adata.obsp["connectivities"]' in graph_py
    assert "mmwrite(mtx_file_path, connectivities" in graph_py


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
    assert msgs[0].startswith("These scores carry no record of how they were scored.")
    assert "mtb.to_long(...)" in msgs[0]


def test_evaluate_output_converts_without_a_warning():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    long, msgs = _user_warnings(mtb.to_long, wide, method="X")
    assert msgs == []
    assert "unknown" not in set(long["scored_with"])
