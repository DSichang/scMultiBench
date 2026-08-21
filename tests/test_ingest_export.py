"""export_dataset / from_mudata: AnnData or MuData -> canonical dataset folder
(work package D_ingest, proposal P04)."""
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from multibench.engine import ingest
from multibench.engine import resolve as _resolve

ad = pytest.importorskip("anndata")


def _h5(path):
    with h5py.File(path, "r") as f:
        d = f["matrix/data"]
        return (np.array(d), d.dtype, d.compression,
                [x.decode() for x in np.array(f["matrix/features"])],
                [x.decode() for x in np.array(f["matrix/barcodes"])])


def _cite(n_cells=60, n_genes=20, n_prot=6, seed=0):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    a = ad.AnnData(sp.random(n_cells, n_genes, density=0.2, format="csr",
                             random_state=rng, dtype=np.float64))
    a.var_names = [f"gene{i}" for i in range(n_genes)]
    a.obs_names = [f"cell{i}" for i in range(n_cells)]
    a.obsm["protein"] = rng.poisson(2.0, size=(n_cells, n_prot)).astype(float)
    a.uns["protein_names"] = [f"CD{i}" for i in range(n_prot)]
    a.obs["ct"] = pd.Categorical(rng.choice(["T", "B", "NK"], size=n_cells))
    a.obs["batch"] = np.where(np.arange(n_cells) < 25, "b1", "b2")
    return a


def test_export_dataset_vertical(tmp_path):
    a = _cite()
    root = tmp_path / "data"
    d = ingest.export_dataset(a, root / "MYCITE", rna="X", adt="obsm:protein",
                              labels="obs:ct")
    assert d == root / "MYCITE"
    assert sorted(p.name for p in d.iterdir()) == ["adt.h5", "cty.csv", "rna.h5"]
    rna, dt, comp, feats, bars = _h5(d / "rna.h5")
    assert rna.shape == (20, 60) and dt == np.float64 and comp == "gzip"
    assert feats == list(a.var_names) and bars == list(a.obs_names)
    assert np.allclose(rna, a.X.toarray().T)
    adt, _, _, pfeats, _ = _h5(d / "adt.h5")
    assert adt.shape == (6, 60) and pfeats == a.uns["protein_names"]
    assert np.allclose(adt, a.obsm["protein"].T)
    lab = pd.read_csv(d / "cty.csv")
    assert lab.columns.tolist() == ["x"] and lab["x"].tolist() == a.obs["ct"].astype(str).tolist()
    # the folder passes the orientation/existence preflight ...
    got = _resolve.inputs_for("MYCITE", "Matilda", "vertical", modalities=["rna", "adt"],
                              data_path=root, check=True)
    assert got["rna"].endswith("rna.h5") and got["adt"].endswith("adt.h5")
    # ... and scan() lists the vertical rna+adt methods for it (env-independent)
    import multibench as mtb
    sc = mtb.scan("MYCITE", "vertical", data_path=root)
    assert ((sc["method"] == "Matilda") & (sc["modalities"] == "rna+adt")).any()
    assert not sc["reason"].str.contains("input files not found").any()


def test_export_dataset_selectors_and_errors(tmp_path):
    a = _cite()
    a.layers["counts"] = a.X.copy()
    d = ingest.export_dataset(a, tmp_path / "L", rna="layer:counts")
    assert np.allclose(_h5(d / "rna.h5")[0], a.layers["counts"].toarray().T)
    with pytest.raises(ValueError, match="unknown selector"):
        ingest.export_dataset(a, tmp_path / "E", rna="foo:bar")
    with pytest.raises(ValueError, match="'mod:<name>' selectors need a MuData"):
        ingest.export_dataset(a, tmp_path / "E", rna="mod:rna")
    with pytest.raises(KeyError, match="column 'nope' not in obs"):
        ingest.export_dataset(a, tmp_path / "E", rna="X", labels="obs:nope")
    with pytest.raises(ValueError, match="use 'obs:<col>'"):
        ingest.export_dataset(a, tmp_path / "E", rna="X", labels="ct")
    with pytest.raises(ValueError, match="nothing to export"):
        ingest.export_dataset(a, tmp_path / "E", rna=None)
    with pytest.raises(ValueError, match="atac_kind= given without atac="):
        ingest.export_dataset(a, tmp_path / "E", atac_kind="peak")
    # adt='X' when obsm keys exist is the classic silent mistake -> refused
    with pytest.raises(ValueError, match="adt requested but obsm= not given"):
        ingest.export_dataset(a, tmp_path / "E", rna=None, adt="X")


def test_export_dataset_atac_kinds(tmp_path):
    n = 10
    peaks = ad.AnnData(np.ones((n, 4)))
    peaks.var_names = ["chr1_100_200", "chr1_300_400", "chr2_10_20", "chrX_5_9"]
    genes = ad.AnnData(np.ones((n, 3)))
    genes.var_names = ["GAPDH", "ACTB", "CD3E"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = ingest.export_dataset(genes, tmp_path / "GAS", rna=None, atac="X",
                                  atac_kind="gene_activity")
        assert sorted(p.name for p in d.iterdir()) == ["atac_gas.h5"]
        d2 = ingest.export_dataset(peaks, tmp_path / "PEAK", rna=None, atac="X",
                                   atac_kind="peak")
    assert sorted(p.name for p in d2.iterdir()) == ["atac.h5", "atac_peak.h5"]
    assert _h5(d2 / "atac.h5")[3] == _h5(d2 / "atac_peak.h5")[3] == list(peaks.var_names)
    # both the plain 'atac' role and the 'atac_peak' role resolve in that folder
    assert _resolve._resolve_role(d2, "atac").name == "atac.h5"
    assert _resolve._resolve_role(d2, "atac_peak").name == "atac_peak.h5"
    # gene-activity export never provides a plain atac.h5 (no silent fallback)
    assert _resolve._resolve_role(d, "atac_gas").name == "atac_gas.h5"
    assert not (d / "atac.h5").exists()
    with pytest.raises(ValueError, match="needs atac_kind="):
        ingest.export_dataset(peaks, tmp_path / "E", rna=None, atac="X")
    with pytest.raises(ValueError, match="needs atac_kind="):
        ingest.export_dataset(peaks, tmp_path / "E", rna=None, atac="X", atac_kind="gas")
    with pytest.warns(UserWarning, match="look like peaks .* not gene activity"):
        ingest.export_dataset(peaks, tmp_path / "W", rna=None, atac="X",
                              atac_kind="gene_activity")


def test_export_dataset_rewrites_existing_atac_link(tmp_path):
    # re-export into the same folder must not fail on the existing hard link
    peaks = ad.AnnData(np.ones((5, 2)))
    peaks.var_names = ["chr1_1_2", "chr1_3_4"]
    for _ in range(2):
        ingest.export_dataset(peaks, tmp_path / "P", rna=None, atac="X", atac_kind="peak")
    assert (tmp_path / "P" / "atac.h5").exists()


def test_export_dataset_batch_numbering(tmp_path):
    a = _cite()
    d = ingest.export_dataset(a, tmp_path / "B", rna="X", adt="obsm:protein",
                              labels="obs:ct", batch="obs:batch")
    names = sorted(p.name for p in d.iterdir())
    assert names == ["adt1.h5", "adt2.h5", "cty1.csv", "cty2.csv", "rna1.h5", "rna2.h5"]
    r1, _, _, _, b1 = _h5(d / "rna1.h5")
    r2, _, _, _, b2 = _h5(d / "rna2.h5")
    assert r1.shape == (20, 25) and r2.shape == (20, 35)
    assert b1 == list(a.obs_names[:25]) and b2 == list(a.obs_names[25:])
    assert np.allclose(r1, a.X[:25].toarray().T)
    assert _h5(d / "adt2.h5")[0].shape == (6, 35)
    assert pd.read_csv(d / "cty1.csv")["x"].tolist() == a.obs["ct"].astype(str).tolist()[:25]
    assert len(pd.read_csv(d / "cty2.csv")) == 35
    # the numbered layout resolves like the shipped D52 (rna1.h5 / cty1.csv)
    got = _resolve.inputs_for("B", "Matilda", "vertical", modalities=["rna", "adt"],
                              data_path=tmp_path, check=True)
    assert got["rna"].endswith("rna1.h5")


def test_export_dataset_cell_count_mismatch(tmp_path):
    a = _cite()
    a.obsm["protein"] = a.obsm["protein"]          # same cells: fine
    b = ad.AnnData(np.ones((7, 2)))
    mu = pytest.importorskip("mudata")
    m = mu.MuData({"rna": a, "adt": b})
    with pytest.raises(ValueError, match="all modalities of one dataset must cover the same cells"):
        ingest.export_dataset(m, tmp_path / "M", rna="mod:rna", adt="mod:adt")


def test_from_mudata(tmp_path):
    mu = pytest.importorskip("mudata")
    a = _cite()
    rna = ad.AnnData(a.X.copy(), obs=a.obs[["ct", "batch"]].copy())
    rna.var_names = a.var_names
    rna.obs_names = a.obs_names
    adt = ad.AnnData(a.obsm["protein"].copy())
    adt.var_names = a.uns["protein_names"]
    adt.obs_names = a.obs_names
    m = mu.MuData({"rna": rna, "adt": adt})
    d1 = ingest.from_mudata(m, tmp_path / "FM", rna="rna", adt="adt", labels="rna:ct")
    d2 = ingest.export_dataset(m, tmp_path / "ED", rna="mod:rna", adt="mod:adt",
                               labels="mod:rna.obs:ct")
    assert sorted(p.name for p in d1.iterdir()) == sorted(p.name for p in d2.iterdir()) \
        == ["adt.h5", "cty.csv", "rna.h5"]
    for name in ("rna.h5", "adt.h5"):
        x1, x2 = _h5(d1 / name), _h5(d2 / name)
        assert np.allclose(x1[0], x2[0]) and x1[3] == x2[3]
    assert _h5(d1 / "adt.h5")[3] == list(adt.var_names)
    assert pd.read_csv(d1 / "cty.csv")["x"].tolist() == pd.read_csv(d2 / "cty.csv")["x"].tolist()
    # an .h5mu path is accepted too; AnnData is rejected with a pointer
    p = tmp_path / "m.h5mu"
    m.write(p)
    d3 = ingest.from_mudata(p, tmp_path / "FP", rna="rna", labels="rna:ct")
    assert (d3 / "rna.h5").exists() and (d3 / "cty.csv").exists()
    with pytest.raises(TypeError, match="use export_dataset for AnnData"):
        ingest.from_mudata(a, tmp_path / "X")
    # MuData without a mod: selector is refused with the available names
    with pytest.raises(ValueError, match="MuData input needs a 'mod:<name>' selector"):
        ingest.export_dataset(m, tmp_path / "X", rna="X")
    # batch split through MuData obs
    d4 = ingest.from_mudata(m, tmp_path / "FB", rna="rna", adt="adt",
                            labels="rna:ct", batch="rna:batch")
    assert (d4 / "rna1.h5").exists() and (d4 / "cty2.csv").exists()
