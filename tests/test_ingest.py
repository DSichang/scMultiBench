import numpy as np
import h5py
import pytest
from pathlib import Path
from multibench.engine import ingest


def test_canonical_passthrough(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("matrix/data", data=np.zeros((3, 5)))
    out = ingest.to_canonical(p, out=tmp_path / "y.h5", convert=False)
    assert str(out) == str(p)  # untouched


def test_from_anndata_writes_features_by_cells(tmp_path):
    ad = pytest.importorskip("anndata")
    import numpy as np
    a = ad.AnnData(np.arange(6, dtype=float).reshape(2, 3))  # 2 cells x 3 genes
    a.var_names = ["g0", "g1", "g2"]
    a.obs_names = ["c0", "c1"]
    out = ingest.to_canonical(a, out=tmp_path / "c.h5")
    with h5py.File(out, "r") as f:
        data = np.array(f["matrix/data"])
        feats = [x.decode() if isinstance(x, bytes) else x for x in np.array(f["matrix/features"])]
        bars = [x.decode() if isinstance(x, bytes) else x for x in np.array(f["matrix/barcodes"])]
    assert data.shape == (3, 2)              # features x cells
    assert feats == ["g0", "g1", "g2"]
    assert bars == ["c0", "c1"]


def test_roundtrip_read_canonical(tmp_path):
    ad = pytest.importorskip("anndata")
    import numpy as np
    a = ad.AnnData(np.arange(6, dtype=float).reshape(2, 3))
    out = ingest.to_canonical(a, out=tmp_path / "c.h5")
    back = ingest.read_canonical(out)
    assert back.shape == (2, 3)              # cells x genes restored


def test_from_csv(tmp_path):
    import pandas as pd, numpy as np
    p = tmp_path / "m.csv"  # rows=cells, cols=genes
    pd.DataFrame(np.ones((4, 2))).to_csv(p, index=False)
    out = ingest.to_canonical(p, out=tmp_path / "c.h5")
    with h5py.File(out, "r") as f:
        assert np.array(f["matrix/data"]).shape == (2, 4)  # genes x cells


def test_from_csv_labeled_roundtrip(tmp_path):
    # labeled matrix (cell barcodes in col 0, gene names in header) must not
    # crash on the string label column and should preserve names.
    import pandas as pd, numpy as np
    p = tmp_path / "labeled.csv"
    pd.DataFrame(np.ones((3, 2)), index=["cellA", "cellB", "cellC"],
                 columns=["geneX", "geneY"]).to_csv(p)
    back = ingest.read_canonical(ingest.to_canonical(p, out=tmp_path / "c.h5"))
    assert back.shape == (3, 2)
    assert list(back.obs_names) == ["cellA", "cellB", "cellC"]
    assert list(back.var_names) == ["geneX", "geneY"]


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


def test_sparse_anndata_and_h5ad_write_features_by_cells(tmp_path):
    a = _sparse_adata()
    expect = a.X.toarray().T
    out = ingest.to_canonical(a, out=tmp_path / "mem.h5")
    data, dtype, comp, feats, bars = _read(out)
    assert data.shape == (25, 40)
    assert np.allclose(data, expect)
    assert dtype == np.float64 and comp == "gzip"
    assert feats == list(a.var_names) and bars == list(a.obs_names)
    # via .h5ad on disk (X stored as csr)
    p = tmp_path / "s.h5ad"
    a.write_h5ad(p)
    out2 = ingest.to_canonical(p, out=tmp_path / "disk.h5")
    data2 = _read(out2)[0]
    assert np.allclose(data2, expect)
    # csc input and a tiny block size exercise the streaming loop boundaries
    a.X = a.X.tocsc()
    out3 = ingest.to_canonical(a, out=tmp_path / "csc.h5", block=7)
    assert np.allclose(_read(out3)[0], expect)


def test_gzip_shrinks_sparse_output(tmp_path):
    a = _sparse_adata(n_cells=3000, n_genes=2000, density=0.08)
    out = ingest.to_canonical(a, out=tmp_path / "big.h5")
    size = out.stat().st_size
    assert size < 5_000_000, size           # dense float64 uncompressed is 48 MB
    assert _read(out)[0].shape == (2000, 3000)


def test_existing_float64_contract(tmp_path):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert _read(ingest.to_canonical(a, out=tmp_path / "d.h5"))[1] == np.float64
    assert _read(ingest.to_canonical(a, out=tmp_path / "f.h5", dtype="float32"))[1] == np.float32
    # compression=None -> uncompressed (contiguous) like the pre-gzip output
    assert _read(ingest.to_canonical(a, out=tmp_path / "n.h5", compression=None))[2] is None


def test_obsm_selects_protein_matrix(tmp_path):
    ad = pytest.importorskip("anndata")
    import pandas as pd
    n_cells, n_prot = 12, 30
    a = ad.AnnData(np.ones((n_cells, 5)))
    a.obsm["protein"] = np.arange(n_cells * n_prot, dtype=float).reshape(n_cells, n_prot)
    names = [f"CD{i}" for i in range(n_prot)]
    a.uns["protein_names"] = names
    out = ingest.to_canonical(a, out=tmp_path / "adt.h5", modality="adt", obsm="protein")
    data, _, _, feats, _ = _read(out)
    assert data.shape == (n_prot, n_cells)
    assert feats == names
    assert np.allclose(data, a.obsm["protein"].T)
    # no names anywhere -> feature_0..
    del a.uns["protein_names"]
    feats2 = _read(ingest.to_canonical(a, out=tmp_path / "adt2.h5", obsm="protein"))[3]
    assert feats2 == [f"feature_{i}" for i in range(n_prot)]
    # DataFrame-valued obsm -> its columns
    a.obsm["prot_df"] = pd.DataFrame(np.ones((n_cells, 2)), index=a.obs_names,
                                     columns=["p1", "p2"])
    assert _read(ingest.to_canonical(a, out=tmp_path / "adt3.h5", obsm="prot_df"))[3] == ["p1", "p2"]
    with pytest.raises(KeyError, match="obsm key 'nope' not found"):
        ingest.to_canonical(a, out=tmp_path / "x.h5", obsm="nope")
    with pytest.raises(ValueError, match="at most one of layer= / obsm="):
        ingest.to_canonical(a, out=tmp_path / "x.h5", obsm="protein", layer="counts")


def test_adt_without_obsm_raises_when_obsm_present(tmp_path):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((4, 3)))
    a.obsm["protein"] = np.ones((4, 2))
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(a, out=tmp_path / "adt.h5", modality="adt")
    msg = str(e.value)
    assert "adt requested but obsm= not given" in msg and "['protein']" in msg
    # no obsm keys at all -> X is taken as the protein matrix (nothing to confuse)
    b = ad.AnnData(np.ones((4, 3)))
    assert ingest.to_canonical(b, out=tmp_path / "adt_b.h5", modality="adt").exists()


@pytest.mark.filterwarnings("ignore:modality='atac_peak':UserWarning")
def test_modality_validated_and_aliases(tmp_path):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((4, 3)))
    d = tmp_path / "DS"
    d.mkdir()
    assert ingest.to_canonical(a, d, modality="protein") == d / "adt.h5"
    assert ingest.to_canonical(a, d, modality="peak").name == "atac_peak.h5"
    assert ingest.to_canonical(a, d, modality="gene_activity").name == "atac_gas.h5"
    assert ingest.to_canonical(a, d, modality="gas").name == "atac_gas.h5"
    assert ingest.to_canonical(a, d, "rna") == d / "rna.h5"       # positional modality
    assert ingest.to_canonical(a, d, "rna", True) == d / "rna.h5"  # positional convert
    assert ingest.to_canonical(a, d, modality="atac") == d / "atac.h5"
    # explicit file path wins over the canonical name
    assert ingest.to_canonical(a, d / "custom.h5", modality="rna") == d / "custom.h5"
    # directory not yet existing but spelled with a trailing slash
    assert ingest.to_canonical(a, str(tmp_path / "NEW") + "/", modality="rna") == tmp_path / "NEW" / "rna.h5"
    with pytest.raises(ValueError, match="unknown modality 'bogus'"):
        ingest.to_canonical(a, d, modality="bogus")
    with pytest.raises(ValueError, match="out path required"):
        ingest.to_canonical(a)


def test_modality_none_with_out_none_writes_to_cwd(tmp_path, monkeypatch):
    ad = pytest.importorskip("anndata")
    monkeypatch.chdir(tmp_path)
    out = ingest.to_canonical(ad.AnnData(np.ones((2, 2))), modality="rna")
    assert Path(out).resolve() == (tmp_path / "rna.h5").resolve()


def test_layer_selects_layer(tmp_path):
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    a = ad.AnnData(np.zeros((3, 4)))
    a.layers["counts"] = sp.csr_matrix(np.arange(12, dtype=float).reshape(3, 4))
    out = ingest.to_canonical(a, out=tmp_path / "l.h5", layer="counts")
    assert np.allclose(_read(out)[0], np.arange(12).reshape(3, 4).T)
    with pytest.raises(KeyError, match="layer 'nope' not found"):
        ingest.to_canonical(a, out=tmp_path / "x.h5", layer="nope")


def test_1d_matrix_raises(tmp_path):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((3, 2)))
    a.obsm["bad"] = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match="must be 2-D"):
        ingest.to_canonical(a, out=tmp_path / "x.h5", obsm="bad")


def test_h5mu_requires_mod_and_reads_mod(tmp_path):
    ad = pytest.importorskip("anndata")
    mu = pytest.importorskip("mudata")
    import scipy.sparse as sp
    rna = ad.AnnData(sp.csr_matrix(np.arange(12, dtype=float).reshape(4, 3)))
    rna.var_names = ["g0", "g1", "g2"]
    adt = ad.AnnData(np.ones((4, 2)))
    adt.var_names = ["p0", "p1"]
    for x in (rna, adt):
        x.obs_names = [f"c{i}" for i in range(4)]
    m = mu.MuData({"rna": rna, "adt": adt})
    p = tmp_path / "m.h5mu"
    m.write(p)
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(p, out=tmp_path / "x.h5")
    assert "pass mod=" in str(e.value) and "rna" in str(e.value)
    out = ingest.to_canonical(p, out=tmp_path / "rna.h5", mod="rna")
    data, _, _, feats, _ = _read(out)
    assert data.shape == (3, 4) and feats == ["g0", "g1", "g2"]
    # in-memory MuData works the same way
    assert _read(ingest.to_canonical(m, out=tmp_path / "adt.h5", mod="adt"))[0].shape == (2, 4)
    with pytest.raises(KeyError, match="mod 'atac' not in MuData"):
        ingest.to_canonical(m, out=tmp_path / "x.h5", mod="atac")
    with pytest.raises(ValueError, match="mod= only applies to MuData"):
        ingest.to_canonical(rna, out=tmp_path / "x.h5", mod="rna")


def test_convert_false_passthrough_or_raise(tmp_path):
    ad = pytest.importorskip("anndata")
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("matrix/data", data=np.zeros((3, 5)))
    assert ingest.to_canonical(p, out=tmp_path / "y.h5", convert=False) == p
    with pytest.raises(ValueError, match="convert=False"):
        ingest.to_canonical(ad.AnnData(np.ones((2, 2))), out=tmp_path / "y.h5", convert=False)
    assert not (tmp_path / "y.h5").exists()


def test_peak_name_warning_for_atac_roles(tmp_path):
    ad = pytest.importorskip("anndata")
    import warnings
    peaks = ad.AnnData(np.ones((3, 4)))
    peaks.var_names = ["chr1_100_200", "chr1-300-400", "chr2:10-20", "chrX_5_9"]
    genes = ad.AnnData(np.ones((3, 2)))
    genes.var_names = ["GAPDH", "ACTB"]
    with pytest.warns(UserWarning, match="look like peaks .* not gene activity"):
        ingest.to_canonical(peaks, tmp_path / "a.h5", modality="atac_gas")
    with pytest.warns(UserWarning, match="only 0% of the features look like peaks"):
        ingest.to_canonical(genes, tmp_path / "b.h5", modality="atac_peak")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ingest.to_canonical(peaks, tmp_path / "c.h5", modality="atac_peak")
        ingest.to_canonical(genes, tmp_path / "d.h5", modality="atac_gas")
        ingest.to_canonical(peaks, tmp_path / "e.h5", modality="rna")


def test_read_canonical_sparse_flag(tmp_path):
    import scipy.sparse as sp
    a = _sparse_adata(density=0.1)
    out = ingest.to_canonical(a, out=tmp_path / "s.h5")
    back = ingest.read_canonical(out, sparse=True)
    assert sp.isspmatrix_csr(back.X) and np.allclose(back.X.toarray(), a.X.toarray())
    assert sp.isspmatrix_csr(ingest.read_canonical(out).X)             # auto: <50% dense
    assert isinstance(ingest.read_canonical(out, sparse=False).X, np.ndarray)
    dense = ingest.to_canonical(_sparse_adata(density=0.9), out=tmp_path / "d.h5")
    assert isinstance(ingest.read_canonical(dense).X, np.ndarray)       # auto: dense
    assert sp.isspmatrix_csr(ingest.read_canonical(dense, sparse=True).X)


def test_write_labels_roundtrip(tmp_path):
    import pandas as pd
    from multibench import workflow
    from multibench.eval import io as eio
    p = ingest._write_labels(["A", "B", "A"], tmp_path / "sub" / "cty.csv")
    assert p.exists()
    assert list(workflow._read_cty(p)) == ["A", "B", "A"]
    assert len(eio.read_labels(p)) == 3
    assert pd.read_csv(p).columns.tolist() == ["x"]
    # pandas Categorical / Series input
    ingest._write_labels(pd.Series(pd.Categorical(["T", "B"])), tmp_path / "c2.csv")
    assert list(workflow._read_cty(tmp_path / "c2.csv")) == ["T", "B"]
    with pytest.raises(ValueError, match="1-D"):
        ingest._write_labels(np.ones((2, 2)), tmp_path / "bad.csv")


def test_namespace_exports():
    import multibench as mtb
    for name in ("export_dataset", "to_canonical", "read_canonical", "normalize_peak_names"):
        assert name in mtb.io.__all__
        assert callable(getattr(mtb.io, name))
    for gone in ("write_labels", "from_mudata"):
        assert gone not in mtb.io.__all__ and not hasattr(mtb.io, gone)


def test_runner_style_call_still_works_on_sparse_with_obsm(tmp_path):
    # runner.run() passes (val, out=<file>) with no modality: an AnnData that
    # carries obsm keys must still convert (the adt guard is modality-gated).
    a = _sparse_adata()
    a.obsm["protein"] = np.ones((a.n_obs, 2))
    out = ingest.to_canonical(a, out=tmp_path / "inputs" / "rna.h5")
    assert _read(out)[0].shape == (25, 40)
