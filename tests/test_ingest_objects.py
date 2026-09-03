"""export_dataset with objects + one master cell order (P20/P11); to_canonical's
size warning (P23) and file errors (Noor)."""
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from multibench.engine import ingest

ad = pytest.importorskip("anndata")


def _bars(path):
    with h5py.File(path, "r") as f:
        return [x.decode() for x in np.array(f["matrix/barcodes"])]


def _feats(path):
    with h5py.File(path, "r") as f:
        return [x.decode() for x in np.array(f["matrix/features"])]


def _pair(n=30, seed=0):
    rng = np.random.default_rng(seed)
    rna = ad.AnnData(rng.poisson(1.0, (n, 8)).astype(float))
    rna.obs_names = [f"cell_{i}" for i in range(n)]
    rna.var_names = [f"g{i}" for i in range(8)]
    rna.obs["ct"] = rng.choice(["A", "B"], n)
    atac = ad.AnnData(rng.poisson(1.0, (n, 5)).astype(float))
    atac.var_names = [f"chr1_{i}_{i + 9}" for i in range(5)]
    return rna, atac


def test_export_accepts_anndata_objects_and_reorders_to_master(tmp_path):
    rna, atac = _pair()
    perm = np.random.default_rng(1).permutation(rna.n_obs)
    atac_shuffled = atac[perm].copy()
    atac_shuffled.obs_names = [rna.obs_names[i] for i in perm]
    d = ingest.export_dataset(rna, tmp_path / "M", atac=atac_shuffled, atac_kind="peak",
                              labels=rna.obs["ct"])
    assert _bars(d / "rna.h5") == _bars(d / "atac.h5") == list(rna.obs_names)
    lab = pd.read_csv(d / "cty.csv")["x"].tolist()
    assert lab == rna.obs["ct"].astype(str).tolist()
    # the re-ordered matrix really is the same cells' data
    with h5py.File(d / "atac.h5") as f:
        got = np.array(f["matrix/data"]).T
    assert np.allclose(got, atac.X)


def test_export_raises_when_barcodes_differ(tmp_path):
    rna, atac = _pair()
    atac.obs_names = [f"other_{i}" for i in range(atac.n_obs)]
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(rna, tmp_path / "X", atac=atac, atac_kind="peak")
    msg = str(e.value)
    assert "atac has 30 cells but 30 barcodes are not in data" in msg
    assert "all modalities of one dataset must cover the same cells" in msg
    # a different cell count still says so (old phrase kept)
    with pytest.raises(ValueError, match="all modalities of one dataset must cover the same cells"):
        ingest.export_dataset(rna, tmp_path / "Y", atac=atac[:10].copy(), atac_kind="peak")


def test_export_from_loose_objects_without_data(tmp_path):
    rna, atac = _pair()
    atac.obs_names = list(rna.obs_names)
    ser = pd.Series(rna.obs["ct"].values, index=rna.obs_names)[::-1]   # reversed index
    d = ingest.export_dataset(None, tmp_path / "L", rna=rna, atac=atac, atac_kind="peak",
                              labels=ser, category="vertical")
    assert sorted(p.name for p in d.iterdir()) == ["atac.h5", "cty.csv", "rna.h5"]
    assert pd.read_csv(d / "cty.csv")["x"].tolist() == rna.obs["ct"].astype(str).tolist()
    # DataFrame modality (index = barcodes) and a plain list of labels
    df = pd.DataFrame(np.ones((rna.n_obs, 3)), index=rna.obs_names, columns=["p1", "p2", "p3"])
    d2 = ingest.export_dataset(None, tmp_path / "L2", rna=rna, adt=df,
                               labels=list(rna.obs["ct"]))
    assert _feats(d2 / "adt.h5") == ["p1", "p2", "p3"] and _bars(d2 / "adt.h5") == list(rna.obs_names)
    # a Series that lacks cells is refused
    with pytest.raises(ValueError, match="have no entry"):
        ingest.export_dataset(None, tmp_path / "L3", rna=rna, labels=ser.iloc[:5])
    with pytest.raises(ValueError, match="nothing to export"):
        ingest.export_dataset(None, tmp_path / "L4")
    with pytest.raises(ValueError, match="must be a selector string .* an AnnData"):
        ingest.export_dataset(rna, tmp_path / "L5", adt=42)


def test_bare_array_needs_master_and_adt_names_warns(tmp_path):
    rna, _ = _pair()
    arr = np.ones((rna.n_obs, 4))
    with pytest.raises(ValueError, match="cell barcodes are unknown"):
        ingest.export_dataset(None, tmp_path / "A", adt=arr)
    with pytest.warns(UserWarning, match="no feature names found .* feature_0..feature_3") as rec:
        ingest.export_dataset(rna, tmp_path / "A", adt=arr)
    assert "adt_names=" in str(rec[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = ingest.export_dataset(rna, tmp_path / "B", adt=arr, adt_names=["CD3", "CD4", "CD8", "CD19"])
    assert _feats(d / "adt.h5") == ["CD3", "CD4", "CD8", "CD19"]
    # the obsm form Priya hit
    rna.obsm["protein"] = arr
    with pytest.warns(UserWarning, match="adt: no feature names found"):
        ingest.export_dataset(rna, tmp_path / "C", adt="obsm:protein")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ingest.export_dataset(rna, tmp_path / "D", adt="obsm:protein", adt_names=list("abcd"))
    with pytest.raises(ValueError, match="2 feature names for 4 features"):
        ingest.export_dataset(rna, tmp_path / "E", adt="obsm:protein", adt_names=["a", "b"])


def test_to_canonical_missing_file_names_path_and_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        ingest.to_canonical("nothere.h5ad", tmp_path / "x.h5")
    assert "nothere.h5ad" in str(e.value) and f"cwd {os.getcwd()}" in str(e.value)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ingest.to_canonical("nothere.h5", tmp_path / "x.h5")


def test_to_canonical_rejects_h5_without_matrix_data(tmp_path):
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((4, 2)))
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(p, tmp_path / "x.h5")
    msg = str(e.value)
    assert "no dataset 'matrix/data'" in msg and "found keys ['data']" in msg
    assert "method OUTPUT" in msg and "evaluate" in msg


def test_to_canonical_warns_above_dense_size_threshold(tmp_path, monkeypatch):
    a = ad.AnnData(np.ones((40, 30)))
    monkeypatch.setattr(ingest, "DENSE_WARN_BYTES", 40 * 30 * 8 - 1)
    with pytest.warns(UserWarning, match="stored DENSE") as rec:
        ingest.to_canonical(a, tmp_path / "big.h5")
    msg = str(rec[0].message)
    assert "30 x 40 x 8 B" in msg and "float32" in msg and "highly-variable" in msg
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ingest.to_canonical(a, tmp_path / "small.h5", dtype="float32")   # 4 B: under the limit
    assert ingest.DENSE_WARN_BYTES == 40 * 30 * 8 - 1
    monkeypatch.undo()
    assert ingest.DENSE_WARN_BYTES == 10 ** 9
