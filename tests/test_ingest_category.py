"""export_dataset category= hint (Tomás): modality='peak' wrote
atac_peak.h5, which no vertical method resolves."""
import anndata as ad
import numpy as np
import pytest

import multibench as mtb
from multibench.engine import ingest, resolve


def _peaks(n_cell=6, n_feat=5):
    a = ad.AnnData(np.random.default_rng(0).random((n_cell, n_feat)))
    a.var_names = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(n_feat)]
    a.obs["ct"] = ["a", "b"] * (n_cell // 2)
    return a


def _genes(n_cell=6, n_feat=5):
    a = ad.AnnData(np.random.default_rng(1).random((n_cell, n_feat)))
    a.var_names = [f"G{i}" for i in range(n_feat)]
    return a


def test_peak_vertical_makes_scan_find_a_runnable_layout(tmp_path):
    rna = _genes(); rna.obs["ct"] = ["a", "b"] * 3
    ingest.export_dataset(None, tmp_path / "TWOADATA", rna=rna, atac=_peaks(),
                          atac_kind="peak", labels=rna.obs["ct"], category="vertical")
    got = resolve.inputs_for("TWOADATA", "vertical", "Matilda", modalities=["rna", "atac"],
                             data_path=tmp_path, check=True)
    assert got["atac"].endswith("atac.h5")


def test_export_dataset_vertical_writes_plain_atac_only(tmp_path):
    pk = _peaks()
    out = ingest.export_dataset(pk, tmp_path / "VP", rna=None, atac="X", atac_kind="peak",
                                labels="obs:ct", category="vertical")
    assert sorted(p.name for p in out.iterdir()) == ["atac.h5", "cty.csv"]
    g = _genes(); g.obs["ct"] = ["a", "b"] * 3
    out = ingest.export_dataset(g, tmp_path / "VG", rna=None, atac="X", atac_kind="gene_activity",
                                labels="obs:ct", category="vertical")
    assert sorted(p.name for p in out.iterdir()) == ["atac.h5", "cty.csv"]
    assert list(ingest.read(out / "atac.h5").var_names) == list(g.var_names)
    # per-batch files are what no vertical method reads: refused before any write
    g.obs["b"] = ["x"] * 3 + ["y"] * 3
    with pytest.raises(ValueError, match="Vertical methods read one rna.h5. Export "
                                         "without batch="):
        ingest.export_dataset(g, tmp_path / "VB", rna=None, atac="X",
                              atac_kind="gene_activity", batch="obs:b", category="vertical")
    assert not (tmp_path / "VB").exists()


def test_export_dataset_explicit_other_category_keeps_representation_names(tmp_path):
    pk = _peaks()
    out = ingest.export_dataset(pk, tmp_path / "DP", rna=None, atac="X", atac_kind="peak",
                                category="diagonal")
    assert sorted(p.name for p in out.iterdir()) == ["atac_peak.h5"]     # no atac.h5 link
    g = _genes()
    out = ingest.export_dataset(g, tmp_path / "CG", rna=None, atac="X", atac_kind="gene_activity",
                                category="cross", batch_index=1)
    assert sorted(p.name for p in out.iterdir()) == ["atac_gas1.h5"]
    # mosaic: the atac<i>.h5 every mosaic variant reads, whatever the kind (L06)
    with pytest.warns(UserWarning, match=r"^Every mosaic method reads peak ATAC\. Export "
                                         r"the ATAC as peaks \(atac_kind='peak'\)\.$"):
        out = ingest.export_dataset(g, tmp_path / "MG", rna=None, atac="X",
                                    atac_kind="gene_activity", category="mosaic",
                                    batch_index=2)
    assert sorted(p.name for p in out.iterdir()) == ["atac2.h5"]
    # category=None: today's behaviour (atac_peak.h5 + hard-linked atac.h5)
    out = ingest.export_dataset(pk, tmp_path / "NP", rna=None, atac="X", atac_kind="peak")
    assert sorted(p.name for p in out.iterdir()) == ["atac.h5", "atac_peak.h5"]
    with pytest.raises(ValueError, match="unknown category"):
        ingest.export_dataset(pk, tmp_path / "BAD", rna=None, atac="X", atac_kind="peak",
                              category="paired")


def test_export_dataset_mudata_forwards_category(tmp_path):
    mudata = pytest.importorskip("mudata")
    rna = _genes(); pk = _peaks()
    rna.obs_names = pk.obs_names = [f"c{i}" for i in range(6)]
    md = mudata.MuData({"rna": rna, "atac": pk})
    out = mtb.io.export_dataset(md, tmp_path / "MU", rna="rna", atac="atac", atac_kind="peak",
                                category="vertical")
    assert sorted(p.name for p in out.iterdir()) == ["atac.h5", "rna.h5"]
