"""evaluate warns when the labels look like a number measured for each cell.

A count column such as ``nCount_RNA`` passed where the cell types belong was
scored without a word. The warning fires for at least 20 cells with more
distinct labels than half of them (``multibench.eval.io._labels_look_measured``).

It does not fire on whole-number cluster ids, and on no label file of a
dataset the package ships.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench.eval import io
from multibench.eval import scib as escib

ad = pytest.importorskip("anndata")

N = 60
TYPES = np.array(["T", "B", "NK"] * (N // 3))
EMB = np.random.default_rng(0).normal(size=(N, 5)) + pd.Categorical(TYPES).codes[:, None] * 4.0
COUNTS = np.arange(1000, 1000 + N)                    # one value per cell
CHECK = "Check that they are the cell types."


@pytest.fixture(autouse=True)
def no_scoring(monkeypatch):
    monkeypatch.setattr(escib, "compute", lambda *a, **k: pd.DataFrame(
        {"Value": [1.0]}, index=pd.Index(["ARI"], name="metric")))


def _warned(*args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mtb.evaluate(*args, metrics=["ARI"], **kw)
    return [str(w.message) for w in rec]


def _adata():
    a = ad.AnnData(np.ones((N, 3)))
    a.obs_names = [f"cell{i}-1" for i in range(N)]
    a.obsm["X_emb"] = EMB
    a.obs["celltype"] = TYPES
    a.obs["nCount_RNA"] = COUNTS
    a.obs["leiden"] = pd.Categorical(np.tile(np.arange(6), N // 6).astype(str))
    a.obs["cluster"] = np.tile(np.arange(6), N // 6)
    return a


# ------------------------------------------------------------------ the rule
def test_the_rule():
    look = io._labels_look_measured
    assert look(COUNTS) == (60, 60)
    assert look(np.arange(31).tolist() + [0] * 29) == (31, 60)
    assert look(np.arange(30).tolist() + [0] * 30) is None           # exactly half
    assert look(np.arange(19)) is None                               # fewer than 20 cells
    assert look(np.array([0.5, 1.5] * 30)) is None                   # few values
    assert look(np.array([0.0, 1.0, 2.0] * 20)) is None              # whole numbers as floats
    assert look(np.array([0.0, 1.0, np.nan] * 20)) is None
    assert look(np.tile(np.arange(12), 5)) is None                   # cluster ids 0..11
    assert look(np.array(["0.5", "1.5"] * 30)) is None               # text, not numbers
    assert look(TYPES) is None and look(pd.Categorical(TYPES)) is None
    assert look(np.array([])) is None


# -------------------------------------------------------------------- Python
COUNT = f"The labels have 60 different values for 60 cells. {CHECK}"


def test_an_obs_column_of_counts():
    assert _warned(_adata(), labels="nCount_RNA") == [COUNT]
    assert _warned(_adata(), labels="obs:nCount_RNA") == [COUNT]


def test_a_series_an_array_and_a_file(tmp_path):
    a = _adata()
    assert _warned(EMB, labels=a.obs["nCount_RNA"].to_numpy()) == [COUNT]
    assert _warned(a, labels=a.obs["nCount_RNA"]) == [COUNT]
    one = tmp_path / "counts.csv"
    pd.DataFrame({"x": COUNTS}).to_csv(one, index=False)
    assert _warned(EMB, labels=str(one)) == [COUNT]
    two = tmp_path / "more.csv"
    pd.DataFrame({"x": COUNTS[:30] + 0.5}).to_csv(two, index=False)
    pd.DataFrame({"x": COUNTS[:30]}).to_csv(one, index=False)
    assert _warned(EMB, labels=[str(one), str(two)]) == [COUNT]


def test_cell_types_and_cluster_ids_do_not_warn(tmp_path):
    a = _adata()
    for labels in ("celltype", "leiden", "cluster"):
        assert _warned(a, labels=labels) == []
    path = tmp_path / "clusters.csv"
    pd.DataFrame({"x": np.tile(np.arange(12), 5)}).to_csv(path, index=False)
    assert _warned(EMB, labels=str(path)) == []
    assert _warned(EMB, labels=list(TYPES)) == []
    assert _warned(EMB[:19], labels=[f"type{i}" for i in range(19)]) == []


# ------------------------------------------------------------ shipped datasets
def test_no_label_file_of_a_local_dataset_looks_measured():
    """Every label file of the datasets under the data path (the demo and
    the mini datasets of the tutorials, when they were fetched)."""
    root = config.data_path
    files = sorted(root.glob("*/*cty*.csv")) if root.is_dir() else []
    if not files:
        pytest.skip(f"no dataset under {root}")
    for f in files:
        values = io.read_labels(f)
        assert io._labels_look_measured(values[~io.missing_mask(values)]) is None, f
