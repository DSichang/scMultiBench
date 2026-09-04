"""to_canonical / export_dataset category= hint (Tomás): modality='peak' wrote
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


def test_to_canonical_vertical_writes_atac_h5_for_any_representation(tmp_path):
    d = tmp_path / "V"; d.mkdir()
    assert ingest.to_canonical(_peaks(), d, modality="peak", category="vertical").name == "atac.h5"
    d2 = tmp_path / "V2"; d2.mkdir()
    assert ingest.to_canonical(_genes(), d2, modality="gas", category="vertical").name == "atac.h5"
    d3 = tmp_path / "V3"; d3.mkdir()
    assert ingest.to_canonical(_peaks(), d3, modality="atac", category="vertical").name == "atac.h5"
    # non-ATAC modalities are untouched by the hint
    assert ingest.to_canonical(_genes(), d3, modality="rna", category="vertical").name == "rna.h5"
    # the vertical `atac` role now resolves
    assert resolve._resolve_role(d, "atac").name == "atac.h5" and (d / "atac.h5").is_file()


def test_to_canonical_other_categories_and_none_keep_todays_names(tmp_path):
    for cat in ("diagonal", "mosaic", "cross", None):
        d = tmp_path / str(cat); d.mkdir()
        assert ingest.to_canonical(_peaks(), d, modality="peak", category=cat).name == "atac_peak.h5"
        assert ingest.to_canonical(_genes(), d, modality="gas", category=cat).name == "atac_gas.h5"
        assert ingest.to_canonical(_peaks(), d, modality="atac", category=cat).name == "atac.h5"
    with pytest.raises(ValueError, match="unknown category"):
        ingest.to_canonical(_peaks(), tmp_path, modality="peak", category="vertcal")
    # an explicit file path is used verbatim whatever the category
    out = ingest.to_canonical(_peaks(), tmp_path / "mine.h5", modality="peak", category="vertical")
    assert out.name == "mine.h5"


def test_to_canonical_peak_vertical_makes_scan_find_a_runnable_layout(tmp_path):
    d = tmp_path / "TWOADATA"; d.mkdir()
    rna = _genes(); rna.obs["ct"] = ["a", "b"] * 3
    ingest.to_canonical(rna, d, modality="rna")
    ingest.to_canonical(_peaks(), d, modality="peak", category="vertical")
    ingest._write_labels(rna.obs["ct"], d / "cty.csv")
    got = resolve.inputs_for("TWOADATA", "vertical", "Matilda", modalities=["rna", "atac"],
                             data_path=tmp_path, check=True)
    assert got["atac"].endswith("atac.h5")


def test_to_canonical_docstring_says_representation_is_not_recorded():
    import inspect
    doc = inspect.getdoc(ingest.to_canonical)
    assert "recorded NOWHERE on disk" in doc and "category='vertical'" in doc


def test_export_dataset_vertical_writes_plain_atac_only(tmp_path):
    pk = _peaks()
    out = ingest.export_dataset(pk, tmp_path / "VP", rna=None, atac="X", atac_kind="peak",
                                labels="obs:ct", category="vertical")
    assert sorted(p.name for p in out.iterdir()) == ["atac.h5", "cty.csv"]
    g = _genes(); g.obs["ct"] = ["a", "b"] * 3
    out = ingest.export_dataset(g, tmp_path / "VG", rna=None, atac="X", atac_kind="gene_activity",
                                labels="obs:ct", category="vertical")
    assert sorted(p.name for p in out.iterdir()) == ["atac.h5", "cty.csv"]
    assert list(ingest.read_canonical(out / "atac.h5").var_names) == list(g.var_names)
    # batch numbering keeps the plain name with the suffix
    g.obs["b"] = ["x"] * 3 + ["y"] * 3
    out = ingest.export_dataset(g, tmp_path / "VB", rna=None, atac="X", atac_kind="gene_activity",
                                batch="obs:b", category="vertical")
    assert sorted(p.name for p in out.iterdir()) == ["atac1.h5", "atac2.h5"]


def test_export_dataset_explicit_other_category_keeps_representation_names(tmp_path):
    pk = _peaks()
    out = ingest.export_dataset(pk, tmp_path / "DP", rna=None, atac="X", atac_kind="peak",
                                category="diagonal")
    assert sorted(p.name for p in out.iterdir()) == ["atac_peak.h5"]     # no atac.h5 link
    g = _genes()
    out = ingest.export_dataset(g, tmp_path / "MG", rna=None, atac="X", atac_kind="gene_activity",
                                category="mosaic")
    assert sorted(p.name for p in out.iterdir()) == ["atac_gas.h5"]
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
