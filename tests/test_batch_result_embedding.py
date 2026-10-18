"""BatchResult.embedding(method): the module-2 deliverable as an AnnData (0.4.0, O7)."""
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine.ingest import read_canonical


def _result(tmp_path, status="CHAIN_OK", method="Matilda"):
    """A run_all result of one D11 method whose output folder holds an embedding."""
    n = len(pd.read_csv(mtb.config.DEFAULT.data_path / "D11" / "cty.csv"))
    out = tmp_path / "out"
    mdir = out / f"{method}_D11"
    mdir.mkdir(parents=True)
    with h5py.File(mdir / "embedding.h5", "w") as h:
        h["data"] = np.arange(n * 3, dtype=float).reshape(n, 3)
    rec = {"method": method, "status": status, "out_dir": str(mdir), "dataset": "D11",
           "category": "vertical", "modalities": ["rna", "adt"], "labels_used": ["cty.csv"]}
    return mtb.BatchResult([rec], "D11", "vertical", out_dir=out), n


def test_embedding_is_an_anndata_with_barcodes_labels_and_batch(tmp_path):
    res, n = _result(tmp_path)
    emb = res.embedding("matilda")                      # any letter case
    d11 = mtb.config.DEFAULT.data_path / "D11"
    assert emb.obsm["X_emb"].shape == (n, 3)
    assert list(emb.obs_names) == list(read_canonical(d11 / "rna.h5").obs_names)
    assert emb.obs["label"].tolist() == pd.read_csv(d11 / "cty.csv").iloc[:, -1].tolist()
    assert set(emb.obs["batch"]) == {"cty"}
    assert emb.obsm["X_emb"][1, 0] == 3.0                # the method's row order


def test_a_method_without_a_record_names_the_ones_there(tmp_path):
    res, _ = _result(tmp_path)
    with pytest.raises(KeyError, match="No record for 'totalVI'; this result has Matilda"):
        res.embedding("totalVI")


def test_a_failed_method_has_no_embedding(tmp_path):
    res, _ = _result(tmp_path, status="FAIL")
    with pytest.raises(ValueError, match="Matilda has no output: its status is FAIL"):
        res.embedding("Matilda")


def test_a_graph_method_has_no_embedding(tmp_path):
    res, _ = _result(tmp_path, method="Seurat_WNN")
    with pytest.raises(ValueError, match="Seurat_WNN writes a graph, not an embedding"):
        res.embedding("Seurat_WNN")
