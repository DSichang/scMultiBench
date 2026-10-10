"""A transposed modality matrix must be rejected at preflight.

Storing matrix/data as cells x features is the easy mistake - it is the
scanpy/AnnData convention, and describe_layout only says "the matrix under
matrix/data" without stating an orientation. It does NOT silently produce a
wrong number (measured: the methods fail), but without this check it fails late,
after conda-env startup and potentially hours of compute, with a third-party
error that never mentions orientation.
"""

import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry
from multibench.engine import resolve as _resolve


def _dataset(root, name, transpose, n_cells=120, n_genes=40, n_adt=8,
             with_dimnames=True, square=False):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if square:
        n_genes = n_cells
    for fname, n_feat, prefix in (("rna.h5", n_genes, "gene"), ("adt.h5", n_adt, "prot")):
        cells_by_feat = rng.poisson(3.0, size=(n_cells, n_feat)).astype(float)
        m = cells_by_feat.T if transpose else cells_by_feat
        with h5py.File(d / fname, "w") as f:
            g = f.create_group("matrix")
            g.create_dataset("data", data=m)
            if with_dimnames:
                g.create_dataset("barcodes", data=np.array(
                    [f"c{i}" for i in range(n_cells)], dtype="S8"))
                g.create_dataset("features", data=np.array(
                    [f"{prefix}{i}" for i in range(n_feat)], dtype="S8"))
    pd.DataFrame({"x": ["A"] * n_cells}).to_csv(d / "cty.csv", index=False)
    return d


def _resolve_it(root, name):
    return _resolve.inputs_for(name, "vertical", "Matilda",
                               modalities=["rna", "adt"],
                               data_path=str(root), check=True)


def test_correct_orientation_passes(tmp_path):
    _dataset(tmp_path, "GOOD", transpose=True)
    got = _resolve_it(tmp_path, "GOOD")
    assert got["rna"].endswith("rna.h5")


def test_transposed_matrix_is_rejected_with_an_actionable_message(tmp_path):
    _dataset(tmp_path, "BAD", transpose=False)
    with pytest.raises(ValueError) as e:
        _resolve_it(tmp_path, "BAD")
    msg = str(e.value)
    assert "cells x features" in msg, msg
    assert "features x cells" in msg, msg
    assert "to_canonical" in msg, "the message must say how to fix it"


def test_square_matrix_is_left_alone(tmp_path):
    """features == cells is genuinely ambiguous; guessing would be worse."""
    _dataset(tmp_path, "SQUARE", transpose=False, square=True, n_adt=8)
    # rna.h5 is square so it cannot be judged; adt stays non-square and correct
    _dataset(tmp_path, "SQUARE2", transpose=True, square=True, n_adt=8)
    _resolve_it(tmp_path, "SQUARE2")   # must not raise


def test_file_without_dimnames_is_not_rejected(tmp_path):
    """No features/barcodes means nothing to compare - do not invent a verdict."""
    _dataset(tmp_path, "NODIMS", transpose=False, with_dimnames=False)
    _resolve_it(tmp_path, "NODIMS")    # must not raise


# --- a file that is no dataset file, or holds no matrix, under the name of one
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
TENX = ("{} is a 10x Genomics file, not a multibench dataset file. Read it with scanpy, "
        "sc.read_10x_h5(path, gex_only=False), split it by var['feature_types'] and write "
        "a dataset folder with mtb.io.export_dataset. The documentation has the steps as "
        'the recipe "A 10x .h5 file".')
PEAKS = [f"chr1:{i * 1000 + 1}-{i * 1000 + 500}" for i in range(70)]


@pytest.fixture
def ready(monkeypatch, tmp_path):
    """Every environment and a GPU, no method scripts on this machine: the
    reason of a row is what the folder check says."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)


def _tenx(path, second, n=100, genes=50):
    """A 10x Genomics feature-barcode file: a sparse matrix (matrix/indptr)
    and a group matrix/features, genes and ``second`` in one matrix."""
    sp = pytest.importorskip("scipy.sparse")
    names = [f"G{i}" for i in range(genes)] + list(second[1])
    m = sp.csc_matrix(np.random.default_rng(0).poisson(0.5, (len(names), n)).astype("int32"))
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["barcodes"] = np.array([f"AAAC{i:05d}-1".encode() for i in range(n)])
        g["data"], g["indices"], g["indptr"] = m.data, m.indices, m.indptr
        g["shape"] = np.array(m.shape, dtype="int32")
        ft = g.create_group("features")
        ft["id"] = ft["name"] = np.array([x.encode() for x in names])
        ft["feature_type"] = np.array([b"Gene Expression"] * genes
                                      + [second[0].encode()] * len(second[1]))
    return path


def _matrix(path, feats, n_cells):
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=np.ones((len(feats), n_cells)))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array(
            [f"c{i}" for i in range(n_cells)], dtype="S8"))


@pytest.mark.parametrize("other,second,methods", [
    ("adt.h5", ("Antibody Capture", [f"AB{i}" for i in range(8)]), ["Seurat_WNN", "totalVI"]),
    ("atac.h5", ("Peaks", PEAKS), ["VIMCCA", "scMM"]),
])
def test_a_10x_file_under_a_dataset_name_fails_the_file_check(
        other, second, methods, ready, tmp_path, capsys):
    """0.3.4 passed the RNA + ADT folder, and gave h5py's sentence about a
    group for the RNA + ATAC one."""
    d = tmp_path / "TENX"
    d.mkdir()
    shutil.copy(_tenx(tmp_path / "filtered_feature_bc_matrix.h5", second), d / "rna.h5")
    shutil.copy(tmp_path / "filtered_feature_bc_matrix.h5", d / other)
    kw = dict(methods=methods, data_path=tmp_path)
    short, full = TENX.format("rna.h5"), "ValueError: " + TENX.format(d / "rna.h5")
    df = mtb.scan("TENX", "vertical", verbose=False, **kw)
    assert list(df["method"]) == methods and not df["files_ok"].any()
    assert list(df["reason"]) == [short, short]
    assert list(df["files_reason"]) == [full, full]
    plan = mtb.run_all("TENX", "vertical", out_dir=tmp_path / "out", dry_run=True, **kw)
    assert not plan["files_ok"].any() and not plan["runnable"].any()
    assert list(plan["reason"]) == [short, short]
    assert list(plan["files_reason"]) == [full, full]
    mtb.scan("TENX", "vertical", per_method=True, **kw)
    for m in methods:
        with pytest.raises(ValueError) as e:
            mtb.inputs_for("TENX", "vertical", m, data_path=tmp_path, check=True)
        assert str(e.value) == TENX.format(d / "rna.h5")
        got = mtb.inputs_for("TENX", "vertical", m, data_path=tmp_path, check=False)
        assert sorted(set(got.values())) == sorted([str(d / other), str(d / "rna.h5")])
    printed = capsys.readouterr()
    assert f"[scan] {methods[0]}: {short}\n" in printed.out
    assert "Accessing a group" not in printed.out + printed.err
    assert not (tmp_path / "out").exists()
    # on the command line: the sentence of multibench convert for this file
    with pytest.MonkeyPatch.context() as m:
        m.setattr(config, "_CLI", True)
        df = mtb.scan("TENX", "vertical", verbose=False, **kw)
    assert df["reason"][0] == (
        "rna.h5 is a 10x Genomics file. multibench convert reads .h5ad files, not this "
        "file. Read it in Python with scanpy, save one .h5ad file per modality and convert "
        'those. The documentation has the steps as the recipe "A 10x .h5 file".')


def test_the_files_a_folder_method_names_are_checked_too(tmp_path):
    """scBridge is handed the folder and reads rna.h5 and atac_gas.h5 in it."""
    d = tmp_path / "DIAG"
    d.mkdir()
    _tenx(d / "rna.h5", ("Peaks", PEAKS))
    _matrix(d / "atac_gas.h5", [f"G{i}" for i in range(50)], 100)
    for name in ("rna_cty.csv", "atac_cty.csv"):
        pd.DataFrame({"x": ["A"] * 100}).to_csv(d / name, index=False)

    def check():
        return mtb.inputs_for("DIAG", "diagonal", "scBridge", data_path=tmp_path, check=True)
    with pytest.raises(ValueError) as e:
        check()
    assert str(e.value) == TENX.format(d / "rna.h5")
    _matrix(d / "rna.h5", [], 100)
    with pytest.raises(ValueError) as e:
        check()
    assert str(e.value) == ("scBridge reads rna.h5 of DIAG, which holds 0 features. Write "
                            "the folder again with mtb.io.export_dataset.")
    _matrix(d / "rna.h5", [f"G{i}" for i in range(50)], 100)
    assert check() == {"data_dir": f"{d}/"}


def test_dataset_files_keep_passing_the_file_check(ready, tmp_path):
    """What export_dataset writes, and a 2-D matrix/data next to a
    matrix/indptr, which to_canonical and mtb.run pass as well."""
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, (60, 30)).astype(float))
    a.obs_names = [f"c{i}" for i in range(60)]
    a.var_names = [f"G{i}" for i in range(30)]
    a.obs["celltype"] = ["T", "B"] * 30
    a.obsm["protein"] = pd.DataFrame(rng.poisson(3.0, (60, 8)).astype(float),
                                     index=a.obs_names, columns=[f"P{i}" for i in range(8)])
    ingest.export_dataset(a, tmp_path / "GOOD", rna="X", adt="obsm:protein",
                          labels="obs:celltype")
    kw = dict(methods=["Seurat_WNN", "totalVI"], data_path=tmp_path, verbose=False)
    df = mtb.scan("GOOD", "vertical", **kw)
    assert df["files_ok"].all() and list(df["reason"]) == ["", ""]
    with h5py.File(tmp_path / "GOOD" / "rna.h5", "a") as f:
        f["matrix/indptr"] = np.arange(3)
    assert ingest._is_tenx_h5(tmp_path / "GOOD" / "rna.h5")
    assert ingest.to_canonical(tmp_path / "GOOD" / "rna.h5") == tmp_path / "GOOD" / "rna.h5"
    df = mtb.scan("GOOD", "vertical", **kw)
    assert df["files_ok"].all() and list(df["reason"]) == ["", ""]


def test_a_matrix_file_without_features_fails_the_file_check(ready, tmp_path, capsys,
                                                            monkeypatch):
    """A split that left no feature: 0.3.4 wrote the file, and the check
    counted the method among those with their input files."""
    d = tmp_path / "ZERO"
    d.mkdir()
    _matrix(d / "rna.h5", [f"G{i}" for i in range(50)], 100)
    _matrix(d / "atac.h5", [], 100)
    with h5py.File(d / "atac.h5", "r") as f:
        assert f["matrix/data"].shape == (0, 100)
    kw = dict(methods=["VIMCCA", "scMM"], data_path=tmp_path)
    short = "atac.h5 holds 0 features. Write the folder again with mtb.io.export_dataset."
    df = mtb.scan("ZERO", "vertical", verbose=False, **kw)
    assert not df["files_ok"].any() and list(df["reason"]) == [short, short]
    assert list(df["files_reason"]) == [
        f"ValueError: {m} reads atac.h5 of ZERO, which holds 0 features. Write the folder "
        f"again with mtb.io.export_dataset." for m in ("VIMCCA", "scMM")]
    plan = mtb.run_all("ZERO", "vertical", out_dir=tmp_path / "out", dry_run=True,
                       verbose=False, **kw)
    assert not plan["runnable"].any() and list(plan["reason"]) == [short, short]
    with pytest.raises(ValueError) as e:            # the real run starts no method
        mtb.run_all("ZERO", "vertical", out_dir=tmp_path / "out", **kw)
    assert (f"  VIMCCA (rna+atac): {short}\n  scMM (rna+atac_gas): {short}\n"
            in str(e.value))
    assert not (tmp_path / "out").exists()
    capsys.readouterr()
    pm = mtb.scan("ZERO", "vertical", per_method=True, **kw)
    assert capsys.readouterr().out.splitlines() == [
        "[scan] 0 of 2 methods have their input files in ZERO. 0 of 2 can run on this "
        "computer. ZERO holds RNA and ATAC.",
        "[scan] Cells per file: rna.h5 100, atac.h5 100.",
        f"[scan] VIMCCA: {short}", f"[scan] scMM: {short}"]
    assert list(pm["reason"]) == [short, short]
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("ZERO", "vertical", "VIMCCA", data_path=tmp_path, check=True)
    assert str(e.value) == ("VIMCCA reads atac.h5 of ZERO, which holds 0 features. Write "
                            "the folder again with mtb.io.export_dataset.")
    assert mtb.inputs_for("ZERO", "vertical", "VIMCCA", data_path=tmp_path, check=False) == {
        "rna": str(d / "rna.h5"), "atac": str(d / "atac.h5")}
    # a file without cells, and the command line's name for the export
    _matrix(d / "atac.h5", PEAKS, 100)
    _matrix(d / "rna.h5", [f"G{i}" for i in range(50)], 0)
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("ZERO", "vertical", "VIMCCA", data_path=tmp_path, check=True)
    assert str(e.value) == ("VIMCCA reads rna.h5 of ZERO, which holds 0 cells. Write the "
                            "folder again with multibench convert.")
