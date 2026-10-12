"""A batch Series that numbers its rows is matched by position, and says so.

``batch=pd.Series(values)`` has the index 0..n-1. When the cells have
barcodes, such a Series cannot be matched by barcode: it is taken row by
row, and a Series that was filtered or sorted before goes to other cells.
``export_dataset``, ``evaluate`` and ``run_all`` now warn. A column of
``adata.obs``, which is indexed by barcode, a list and an array give no
message.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import ingest
from multibench.eval import scib as escib

from .test_workflow_batch import N as CITE_N, cite  # noqa: F401 - the run_all fixture

ad = pytest.importorskip("anndata")

N = 60
DONORS = np.array(["d1", "d2", "d3"] * (N // 3))


#: the same text from export_dataset, evaluate and run_all: it names neither
#: the number of rows nor the order the values must follow
TEXT = ("batch is a Series without barcodes and is matched to the cells by position. To "
        "match by barcode, pass a Series indexed by barcode.")


def _adata(names=True):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(2.0, (N, 8)).astype(float))
    if names:
        a.obs_names = [f"cell{i}-1" for i in range(N)]
    a.var_names = [f"GENE{i}" for i in range(8)]
    a.obs["celltype"] = ["T", "B"] * (N // 2)
    a.obs["donor"] = DONORS
    a.obsm["X_emb"] = rng.normal(size=(N, 4))
    return a


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in rec if "Series" in str(w.message)]


# ------------------------------------------------------------- export_dataset
def test_export_dataset_warns_and_takes_the_rows_in_order(tmp_path):
    a = _adata()
    kw = dict(rna="X", labels="obs:celltype", category="cross")
    out, said = _caught(lambda: ingest.export_dataset(
        a, tmp_path / "ROWS", batch=pd.Series(DONORS), **kw))
    assert said == [TEXT]
    assert "adata" not in TEXT and "donor" not in TEXT   # no name of the caller is guessed
    # by position: batch 1 holds every third cell, starting with the first
    assert list(ingest.read_canonical(out / "rna1.h5").obs_names) == list(a.obs_names[::3])
    # as it says: the same values indexed by barcode, in another row order
    turned = pd.Series(DONORS, index=a.obs_names).iloc[::-1]
    by_barcode, said = _caught(lambda: ingest.export_dataset(
        a, tmp_path / "BARCODES", batch=turned, **kw))
    assert said == []
    assert list(ingest.read_canonical(by_barcode / "rna1.h5").obs_names) \
        == list(a.obs_names[::3])


def test_export_dataset_is_silent_for_the_usual_forms(tmp_path):
    a = _adata()
    kw = dict(rna="X", labels="obs:celltype", category="cross")
    for i, batch in enumerate((a.obs["donor"], "obs:donor", list(DONORS), DONORS,
                               a.obs["donor"].iloc[::-1])):
        _, said = _caught(lambda: ingest.export_dataset(a, tmp_path / f"OK{i}", batch=batch,
                                                        **kw))
        assert said == []
    # cells without barcodes are named 0, 1, ...: the row numbers are their names
    plain = _adata(names=False)
    _, said = _caught(lambda: ingest.export_dataset(plain, tmp_path / "PLAIN",
                                                    batch=pd.Series(DONORS), **kw))
    assert said == []


# ------------------------------------------------------------------- evaluate
@pytest.fixture
def no_scoring(monkeypatch):
    monkeypatch.setattr(escib, "compute", lambda *a, **k: pd.DataFrame(
        {"Value": [1.0]}, index=pd.Index(["ARI"], name="metric")))


def test_evaluate_warns_when_the_output_has_barcodes(tmp_path, no_scoring):
    a = _adata()
    frame = pd.DataFrame(a.obsm["X_emb"], index=a.obs_names)
    frame.to_csv(tmp_path / "emb.csv")
    labels = a.obs["celltype"].to_numpy()
    for output in (a, frame, str(tmp_path / "emb.csv")):
        _, said = _caught(lambda: mtb.evaluate(output, labels=labels,
                                               batch=pd.Series(DONORS), metrics="all"))
        assert said == [TEXT]


def test_evaluate_is_silent_for_the_usual_forms(no_scoring):
    a = _adata()
    for batch in ("donor", "obs:donor", a.obs["donor"], a.obs["donor"].iloc[::-1],
                  DONORS, list(DONORS)):
        _, said = _caught(lambda: mtb.evaluate(a, labels="celltype", batch=batch,
                                               metrics="all"))
        assert said == []
    # an array has no cell ids, and an AnnData without barcodes names its cells 0, 1, ...
    for output in (a.obsm["X_emb"], _adata(names=False)):
        _, said = _caught(lambda: mtb.evaluate(
            output, labels=a.obs["celltype"].to_numpy(), batch=pd.Series(DONORS),
            metrics="all"))
        assert said == []


# -------------------------------------------------------------------- run_all
def test_run_all_warns_and_is_silent_for_a_series_with_barcodes(cite, tmp_path):  # noqa: F811
    data, _ = cite
    batch = np.tile(["d1", "d2"], CITE_N // 2)
    kw = dict(methods=["Matilda"], modalities=["rna", "adt"], data_path=data,
              verbose=False)
    res, said = _caught(lambda: mtb.run_all("MYCITE", "vertical", batch=pd.Series(batch),
                                            out_dir=tmp_path / "rows", **kw))
    assert said == [TEXT]
    assert res.summary.loc[0, "n_batches"] == 2
    cells = [f"c{i}" for i in range(CITE_N)]
    for i, value in enumerate((pd.Series(batch, index=cells), batch, list(batch))):
        _, said = _caught(lambda: mtb.run_all("MYCITE", "vertical", batch=value,
                                              out_dir=tmp_path / f"ok{i}", **kw))
        assert said == []
