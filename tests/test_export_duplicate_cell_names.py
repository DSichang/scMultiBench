"""export_dataset refuses labels it cannot match to the cells.

Labels taken from another modality's obs are matched to the exported cells
by name. With a cell name given twice and the modality in another order, no
name match exists: 0.3.8 wrote the labels by position, every row to another
cell, without a word. That is an error now, raised before anything is
written. Unique names are matched as before, and names in the same order as
the exported cells are taken by position as before (the rows then agree).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench.engine import ingest

ad = pytest.importorskip("anndata")
md = pytest.importorskip("mudata")

NAMES = ["c0", "c1", "c1", "c2", "c3", "c4"]
UNIQUE = ["c0", "c1", "c2", "c3", "c4", "c5"]
#: the 0.4.0 refusal of repeated barcodes (X09)
REPEAT = "1 barcodes repeat, e.g. c1. Make them unique (prefix the sample)."


def _mdata(names, atac_reversed):
    """rna row i is cell i, labelled a<i> in the atac obs."""
    def one(order):
        a = ad.AnnData(np.arange(18, dtype=float).reshape(6, 3)[order])
        a.obs_names = [names[i] for i in order]
        a.var_names = ["G0", "G1", "G2"]
        a.obs["ct"] = [f"a{i}" for i in order]
        return a
    order = list(range(6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # mudata, anndata: names repeat
        return md.MuData({"rna": one(order),
                          "atac": one(order[::-1] if atac_reversed else order)})


def _labels(out):
    return pd.read_csv(out / "cty.csv")["x"].tolist()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("what, category", [("labels", "vertical"), ("batch", "mosaic")])
def test_repeated_names_in_another_order_are_an_error(tmp_path, what, category):
    out = tmp_path / "OUT"
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(_mdata(NAMES, True), out, rna="mod:rna",
                              category=category, **{what: "mod:atac.obs:ct"})
    # repeated barcodes are refused before anything else since 0.4.0 (X09)
    assert str(e.value) == REPEAT
    assert not (out / "rna.h5").exists() and not (out / "cty.csv").exists()


def test_repeated_names_in_the_same_order_are_refused_too(tmp_path):
    # 0.3.9 took them by position; 0.4.0 refuses repeated barcodes (X09)
    with pytest.raises(ValueError, match=r"barcodes repeat"):
        ingest.export_dataset(_mdata(NAMES, False), tmp_path / "OUT", rna="mod:rna",
                              labels="mod:atac.obs:ct", category="vertical")


def test_unique_names_in_another_order_are_matched_by_name(tmp_path):
    out = ingest.export_dataset(_mdata(UNIQUE, True), tmp_path / "OUT", rna="mod:rna",
                                labels="mod:atac.obs:ct", category="vertical")
    assert _labels(out) == [f"a{i}" for i in range(6)]


@pytest.mark.filterwarnings("ignore")
def test_the_advice_works(tmp_path):
    """One unique name per cell, the same in every modality: the labels match."""
    m = _mdata(NAMES, True)
    for mod in ("rna", "atac"):                     # ct a<i> tells which cell a row is
        m.mod[mod].obs_names = [f"cell{x[1:]}" for x in m.mod[mod].obs["ct"]]
    m.update()
    out = ingest.export_dataset(m, tmp_path / "OUT", rna="mod:rna",
                                labels="mod:atac.obs:ct", category="vertical")
    assert _labels(out) == [f"a{i}" for i in range(6)]


