"""Files written later for the same cells keep the row order of the folder.

A dataset folder can be built in several calls: the matrices first, the
labels when they are ready. A label file has no barcodes, so its rows are
matched to the matrix by position. 0.3.4 wrote each call in the row order of
the object it was given; with the object reordered in between, row ``i`` of
``cty.csv`` labelled another cell than row ``i`` of ``rna.h5`` and nothing
could tell afterwards. A call now writes the rows of cells that the folder
already holds in the order of the matrix files there, and says so on stderr.
"""
import numpy as np
import pandas as pd
import pytest

from multibench.engine import ingest
from multibench.engine import resolve

ad = pytest.importorskip("anndata")


def _cells(n=90, tag="C", seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n, 20)).astype(float))
    a.obs_names = [f"{tag}{i:03d}" for i in range(n)]
    a.var_names = [f"GENE{i}" for i in range(20)]
    a.obsm["protein"] = pd.DataFrame(rng.poisson(3.0, (n, 4)).astype(float),
                                     index=a.obs_names, columns=[f"P{i}" for i in range(4)])
    # one cell type per cell, so that a label names its cell
    a.obs["celltype"] = [f"type_of_{b}" for b in a.obs_names]
    a.obs["donor"] = [f"d{i % 3}" for i in range(n)]
    return a


def _shuffled(a, seed=1):
    return a[np.random.default_rng(seed).permutation(a.n_obs)].copy()


def _bars(path):
    return list(resolve._barcodes_of(path))


def _labels(path):
    return pd.read_csv(path)["x"].tolist()


def _matrix(path):
    return ingest.read_canonical(path, sparse=False)


def _note(files, ref, out):
    return (f"export_dataset: wrote the rows of {files} in the cell order of {ref}, which "
            f"the files of {out} follow.\n")


def test_labels_added_later_follow_the_matrix(tmp_path, capsys):
    a = _cells()
    f = tmp_path / "LATER"
    ingest.export_dataset(a, f, rna="X", adt="obsm:protein")
    assert capsys.readouterr().err == ""
    later = _shuffled(a)
    assert list(later.obs_names) != list(a.obs_names)
    ingest.export_dataset(later, f, rna=None, labels="obs:celltype")
    assert capsys.readouterr().err == _note("cty.csv", "rna.h5", f)
    assert _bars(f / "rna.h5") == list(a.obs_names)
    assert _labels(f / "cty.csv") == [f"type_of_{b}" for b in _bars(f / "rna.h5")]


def test_a_second_modality_follows_the_first(tmp_path, capsys):
    """Batch 2 of a mosaic folder: the RNA first, then the ADT of the same
    cells from an object in another order."""
    a = _cells()
    f = tmp_path / "MOSAIC"
    kw = dict(batch_index=2, category="mosaic")
    ingest.export_dataset(a, f, rna="X", labels="obs:celltype", **kw)
    capsys.readouterr()
    later = _shuffled(a)
    ingest.export_dataset(later, f, rna=None, adt="obsm:protein", **kw)
    assert capsys.readouterr().err == _note("adt2.h5", "rna2.h5", f)
    assert _bars(f / "adt2.h5") == _bars(f / "rna2.h5") == list(a.obs_names)
    got = _matrix(f / "adt2.h5")
    assert np.array_equal(got.X, a.obsm["protein"].to_numpy())      # the values moved with the cells
    assert list(got.var_names) == ["P0", "P1", "P2", "P3"]


def test_a_full_rewrite_keeps_the_order(tmp_path, capsys):
    a = _cells()
    f = tmp_path / "REWRITE"
    kw = dict(rna="X", adt="obsm:protein", labels="obs:celltype")
    ingest.export_dataset(a, f, **kw)
    capsys.readouterr()
    ingest.export_dataset(_shuffled(a), f, overwrite=True, **kw)
    assert capsys.readouterr().err == _note("rna.h5, adt.h5 and cty.csv", "rna.h5", f)
    assert _bars(f / "rna.h5") == _bars(f / "adt.h5") == list(a.obs_names)
    assert np.array_equal(_matrix(f / "rna.h5").X, a.X)
    assert _labels(f / "cty.csv") == list(a.obs["celltype"])


def test_same_order_prints_nothing(tmp_path, capsys):
    """The usual case: no reference in the folder, or the same order."""
    a = _cells()
    f = tmp_path / "SAME"
    ingest.export_dataset(a, f, rna="X")
    ingest.export_dataset(a, f, rna=None, adt="obsm:protein", labels="obs:celltype")
    ingest.export_dataset(a, f, rna="X", adt="obsm:protein", labels="obs:celltype",
                          overwrite=True)
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""
    assert _labels(f / "cty.csv") == list(a.obs["celltype"])


def test_two_orders_in_the_folder_are_refused(tmp_path, capsys):
    """Two matrices of the same cells in two orders: no order to follow."""
    a = _cells()
    f = tmp_path / "TWO"
    ingest.export_dataset(a, f, rna="X")
    ingest.to_canonical(_shuffled(a), f / "adt.h5", obsm="protein")     # written apart
    before = {p.name: p.read_bytes() for p in f.iterdir()}
    for kw in (dict(rna=None, labels="obs:celltype"),
               dict(rna="X", adt="obsm:protein", labels="obs:celltype", overwrite=True)):
        with pytest.raises(ValueError) as e:
            ingest.export_dataset(a, f, **kw)
        assert str(e.value) == (
            f"rna.h5 and adt.h5 of {f} hold the same cells in different orders, so a row of "
            f"one is not the cell of the same row of the other. Export this data again in "
            f"one call, into a new folder.")
        assert {p.name: p.read_bytes() for p in f.iterdir()} == before
    assert capsys.readouterr().err == ""


def test_diagonal_sides_are_judged_apart(tmp_path, capsys):
    """The RNA cells and the ATAC cells of unpaired data have an order each."""
    rna, atac = _cells(80, "R"), _cells(60, "A", seed=2)
    f = tmp_path / "DIAG"
    kw = dict(rna="X", atac_kind="gene_activity", labels="obs:celltype", category="diagonal")
    ingest.export_dataset(rna, f, atac=atac, **kw)
    capsys.readouterr()
    ingest.export_dataset(rna, f, atac=_shuffled(atac), overwrite=True, **kw)
    assert capsys.readouterr().err == _note("atac_gas.h5 and atac_cty.csv", "atac_gas.h5", f)
    ingest.export_dataset(_shuffled(rna), f, atac=atac, overwrite=True, **kw)
    assert capsys.readouterr().err == _note("rna.h5 and rna_cty.csv", "rna.h5", f)
    assert _bars(f / "rna.h5") == list(rna.obs_names)
    assert _bars(f / "atac_gas.h5") == list(atac.obs_names)
    assert _labels(f / "rna_cty.csv") == list(rna.obs["celltype"])
    assert _labels(f / "atac_cty.csv") == list(atac.obs["celltype"])


def test_batch_column_orders_each_batch(tmp_path, capsys):
    a = _cells()
    f = tmp_path / "BATCH"
    kw = dict(batch="obs:donor", category="cross")
    ingest.export_dataset(a, f, rna="X", **kw)
    capsys.readouterr()
    ingest.export_dataset(_shuffled(a), f, rna=None, labels="obs:celltype", **kw)
    assert capsys.readouterr().err == "".join(
        _note(f"cty{b}.csv", f"rna{b}.h5", f) for b in (1, 2, 3))
    for b in (1, 2, 3):
        bars = _bars(f / f"rna{b}.h5")
        assert bars == [x for x, d in zip(a.obs_names, a.obs["donor"]) if d == f"d{b - 1}"]
        assert _labels(f / f"cty{b}.csv") == [f"type_of_{x}" for x in bars]


def test_other_cells_have_no_order_to_follow(tmp_path, capsys):
    """Replacing every file by other cells is no reordering: the new files
    are written in the order of the new object."""
    f = tmp_path / "NEW"
    kw = dict(rna="X", labels="obs:celltype", overwrite=True)
    ingest.export_dataset(_cells(tag="OLD"), f, **kw)
    new = _shuffled(_cells(tag="NEW", seed=3))
    ingest.export_dataset(new, f, **kw)
    assert capsys.readouterr().err == ""
    assert _bars(f / "rna.h5") == list(new.obs_names)
    assert _labels(f / "cty.csv") == list(new.obs["celltype"])


def test_a_sparse_matrix_is_reordered_too(tmp_path, capsys):
    import scipy.sparse as sp
    a = _cells()
    f = tmp_path / "SPARSE"
    ingest.export_dataset(a, f, rna=None, labels="obs:celltype", adt="obsm:protein")
    later = _shuffled(a)
    later.X = sp.csc_matrix(later.X)
    ingest.export_dataset(later, f, rna="X")
    assert capsys.readouterr().err == _note("rna.h5", "adt.h5", f)
    assert _bars(f / "rna.h5") == list(a.obs_names)
    assert np.array_equal(_matrix(f / "rna.h5").X, a.X)
