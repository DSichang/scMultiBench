import numpy as np
import h5py
import pytest
from multibench.engine import ingest


def test_from_anndata_writes_features_by_cells(tmp_path):
    ad = pytest.importorskip("anndata")
    import numpy as np
    a = ad.AnnData(np.arange(6, dtype=float).reshape(2, 3))  # 2 cells x 3 genes
    a.var_names = ["g0", "g1", "g2"]
    a.obs_names = ["c0", "c1"]
    out = ingest.export_dataset(a, tmp_path / "DS") / "rna.h5"
    with h5py.File(out, "r") as f:
        data = np.array(f["matrix/data"])
        feats = [x.decode() if isinstance(x, bytes) else x for x in np.array(f["matrix/features"])]
        bars = [x.decode() if isinstance(x, bytes) else x for x in np.array(f["matrix/barcodes"])]
    assert data.shape == (3, 2)              # features x cells
    assert feats == ["g0", "g1", "g2"]
    assert bars == ["c0", "c1"]


def test_roundtrip_read(tmp_path):
    ad = pytest.importorskip("anndata")
    import numpy as np
    a = ad.AnnData(np.arange(6, dtype=float).reshape(2, 3))
    back = ingest.read(ingest.export_dataset(a, tmp_path / "DS") / "rna.h5")
    assert back.shape == (2, 3)              # cells x genes restored


# ---------------------------------------------------------------------------
# sparse-safe / compressed writes, selectors, validation (work package D_ingest)
# ---------------------------------------------------------------------------

def _read(path):
    with h5py.File(path, "r") as f:
        d = f["matrix/data"]
        return (np.array(d), d.dtype, d.compression,
                [x.decode() for x in np.array(f["matrix/features"])],
                [x.decode() for x in np.array(f["matrix/barcodes"])])


def _sparse_adata(n_cells=40, n_genes=25, density=0.1, seed=0):
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    X = sp.random(n_cells, n_genes, density=density, format="csr",
                  random_state=rng, dtype=np.float64)
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


def test_gzip_shrinks_sparse_output(tmp_path):
    a = _sparse_adata(n_cells=3000, n_genes=2000, density=0.08)
    out = ingest.export_dataset(a, tmp_path / "DS") / "rna.h5"
    size = out.stat().st_size
    assert size < 5_000_000, size           # dense float64 uncompressed is 48 MB
    assert _read(out)[0].shape == (2000, 3000)


def test_obsm_selects_protein_matrix(tmp_path):
    ad = pytest.importorskip("anndata")
    import pandas as pd
    n_cells, n_prot = 12, 30
    a = ad.AnnData(np.ones((n_cells, 5)))
    a.obsm["protein"] = np.arange(n_cells * n_prot, dtype=float).reshape(n_cells, n_prot)
    names = [f"CD{i}" for i in range(n_prot)]
    a.uns["protein_names"] = names
    out = ingest.export_dataset(a, tmp_path / "A", adt="obsm:protein") / "adt.h5"
    data, _, _, feats, _ = _read(out)
    assert data.shape == (n_prot, n_cells)
    assert feats == names
    assert np.allclose(data, a.obsm["protein"].T)
    # no names anywhere -> feature_0..
    del a.uns["protein_names"]
    with pytest.warns(UserWarning, match="has no feature names"):
        out = ingest.export_dataset(a, tmp_path / "B", adt="obsm:protein") / "adt.h5"
    feats2 = _read(out)[3]
    assert feats2 == [f"feature_{i}" for i in range(n_prot)]
    # DataFrame-valued obsm -> its columns
    a.obsm["prot_df"] = pd.DataFrame(np.ones((n_cells, 2)), index=a.obs_names,
                                     columns=["p1", "p2"])
    out = ingest.export_dataset(a, tmp_path / "C", adt="obsm:prot_df") / "adt.h5"
    assert _read(out)[3] == ["p1", "p2"]
    with pytest.raises(KeyError, match="obsm key 'nope' not found"):
        ingest.export_dataset(a, tmp_path / "D", adt="obsm:nope")


def test_layer_selects_layer(tmp_path):
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    a = ad.AnnData(np.zeros((3, 4)))
    a.layers["counts"] = sp.csr_matrix(np.arange(12, dtype=float).reshape(3, 4))
    out = ingest.export_dataset(a, tmp_path / "DS", rna="layer:counts") / "rna.h5"
    assert np.allclose(_read(out)[0], np.arange(12).reshape(3, 4).T)
    with pytest.raises(KeyError, match="layer 'nope' not found"):
        ingest.export_dataset(a, tmp_path / "X", rna="layer:nope")


def test_the_layer_error_lists_layers_only(tmp_path):
    """anndata 0.13 lists X among the layers, under the key None: the error
    read "available: [None]" for an object without a layer."""
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((3, 4)))
    out = tmp_path / "OUT"
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(a, out, rna="layer:counts")
    assert e.value.args[0] == "layer 'counts' not found, and the object has no layers."
    a.layers["raw"] = np.ones((3, 4))
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(a, out, rna="layer:counts")
    assert e.value.args[0] == "layer 'counts' not found; available: ['raw']"
    assert not out.exists()
    # the layer that is there is written
    ingest.export_dataset(a, out, rna="layer:raw")
    assert ingest.read(out / "rna.h5").shape == (3, 4)


def test_1d_matrix_raises(tmp_path):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((3, 2)))
    a.obsm["bad"] = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match="must be 2-D"):
        ingest.export_dataset(a, tmp_path / "X", rna=None, adt="obsm:bad")


def test_write_labels_roundtrip(tmp_path):
    import pandas as pd
    from multibench.eval import io as eio
    p = ingest._write_labels(["A", "B", "A"], tmp_path / "sub" / "cty.csv")
    assert p.exists()
    assert list(eio.read_labels(p)) == ["A", "B", "A"]
    assert len(eio.read_labels(p)) == 3
    assert pd.read_csv(p).columns.tolist() == ["x"]
    # pandas Categorical / Series input
    ingest._write_labels(pd.Series(pd.Categorical(["T", "B"])), tmp_path / "c2.csv")
    assert list(eio.read_labels(tmp_path / "c2.csv")) == ["T", "B"]
    with pytest.raises(ValueError, match="1-D"):
        ingest._write_labels(np.ones((2, 2)), tmp_path / "bad.csv")
