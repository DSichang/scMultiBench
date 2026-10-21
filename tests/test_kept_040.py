"""Names kept through the 0.4 cut: the metric families, the ATAC kind read
from the names, and a dataset given as a folder path."""
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W


# ------------------------------------------------------------------ catalog / io / plot / env
def test_catalog_metrics_has_the_families():
    tab = mtb.catalog.metrics()
    fam = tab.groupby("family", sort=False).metric.apply(list).to_dict()
    assert fam == {"clustering": ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"],
                   "batch": ["ASW_batch", "ARI_batch", "NMI_batch", "GC", "iLISI", "kBET"],
                   "classification": ["OCA", "ACA", "Spec", "Sens", "F1"]}


def test_export_dataset_reads_the_atac_kind_from_the_names(tmp_path):
    import anndata as ad
    rna = ad.AnnData(np.ones((4, 3), dtype=np.float32))
    rna.var_names = ["g1", "g2", "g3"]
    peaks = ad.AnnData(np.ones((4, 2), dtype=np.float32))
    peaks.var_names = ["chr1:1-100", "chr2:5-90"]
    genes = rna.copy()
    out = mtb.io.export_dataset(None, tmp_path / "P", rna=rna, atac=peaks, category="vertical")
    assert (out / "atac.h5").exists()
    out = mtb.io.export_dataset(None, tmp_path / "G", rna=rna, atac=genes)
    assert (out / "atac_gas.h5").exists()
    odd = peaks.copy()
    odd.var_names = ["x", "y"]
    with pytest.raises(ValueError, match="neither peaks such as chr1:100-200 nor the RNA's "
                                         "gene names. Pass atac_kind='peak' or "
                                         "atac_kind='gene_activity'."):
        mtb.io.export_dataset(None, tmp_path / "Q", rna=rna, atac=odd)


def test_a_dataset_can_be_a_folder_path(tmp_path, monkeypatch):
    root = mtb.config.data_path
    monkeypatch.chdir(root.parent)
    rel = f"{root.name}/D11"
    assert mtb.engine.resolve.inputs_for(rel, "vertical", "Matilda") == mtb.engine.resolve.inputs_for("D11", "vertical",
                                                                        "Matilda")
    assert mtb.engine.resolve.labels_for(root / "D11") == mtb.engine.resolve.labels_for("D11")
    pd.testing.assert_frame_equal(mtb.scan(rel, "vertical", verbose=False),
                                  mtb.scan("D11", "vertical", verbose=False))
