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


# --- a 10x file, or a matrix without features or cells, under a dataset file name
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
#: what the folder check says of a 10x file after its subject ("<method> reads
#: rna.h5 of <dataset>, which", in the reason column "rna.h5"): in Python, and
#: on the command line
TENX = ("is a 10x Genomics file, not a multibench dataset file. Read it with scanpy, "
        "sc.read_10x_h5(path, gex_only=False), split it by var['feature_types'] and write "
        "a dataset folder with mtb.io.export_dataset. The documentation has the steps as "
        'the recipe "A 10x .h5 file".')
TENX_CLI = ("is a 10x Genomics file, not a multibench dataset file. Read it in Python "
            "with scanpy, save one .h5ad file per modality and convert those with "
            "multibench convert. The documentation has the steps as the recipe "
            '"A 10x .h5 file".')
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
    short = f"rna.h5 {TENX}"
    full = [f"ValueError: {m} reads rna.h5 of TENX, which {TENX}" for m in methods]
    df = mtb.scan("TENX", "vertical", verbose=False, **kw)
    assert list(df["method"]) == methods and not df["files_ok"].any()
    assert list(df["reason"]) == [short, short]
    assert list(df["files_reason"]) == full
    plan = mtb.run_all("TENX", "vertical", out_dir=tmp_path / "out", dry_run=True, **kw)
    assert not plan["files_ok"].any() and not plan["runnable"].any()
    assert list(plan["reason"]) == [short, short]
    assert list(plan["files_reason"]) == full
    mtb.scan("TENX", "vertical", per_method=True, **kw)
    for m in methods:
        with pytest.raises(ValueError) as e:
            mtb.inputs_for("TENX", "vertical", m, data_path=tmp_path, check=True)
        assert str(e.value) == f"{m} reads rna.h5 of TENX, which {TENX}"
        got = mtb.inputs_for("TENX", "vertical", m, data_path=tmp_path, check=False)
        assert sorted(set(got.values())) == sorted([str(d / other), str(d / "rna.h5")])
    # without a category and as diagonal, scan looks into an ATAC file under
    # another name than the one a method reads: on the RNA + ATAC folder 0.3.4
    # raised h5py's TypeError there, for the whole call
    for cat in (None, "diagonal"):
        df = mtb.scan("TENX", cat, data_path=tmp_path)
        assert len(df) and not df["files_ok"].any()
        assert not (df["reason"] + df["files_reason"]).str.contains("Accessing a group").any()
    plan = mtb.run_all("TENX", "diagonal", out_dir=tmp_path / "out", dry_run=True,
                       data_path=tmp_path)
    assert len(plan) and not plan["runnable"].any()
    mtb.scan("TENX", "diagonal", per_method=True, data_path=tmp_path)
    printed = capsys.readouterr()
    assert f"[scan] {methods[0]}: {short}\n" in printed.out
    assert "Accessing a group" not in printed.out + printed.err
    assert not (tmp_path / "out").exists()
    # the way out is the one that to_canonical and mtb.run give for the file itself
    assert ingest._tenx_message("f.h5") == f"f.h5 {TENX}"
    # on the command line: the same fact, then the command that writes a folder.
    # The user ran scan, so it is not the sentence about what convert reads
    with pytest.MonkeyPatch.context() as m:
        m.setattr(config, "_CLI", True)
        df = mtb.scan("TENX", "vertical", verbose=False, **kw)
    assert list(df["reason"]) == [f"rna.h5 {TENX_CLI}"] * 2
    assert list(df["files_reason"]) == [
        f"ValueError: {m} reads rna.h5 of TENX, which {TENX_CLI}" for m in methods]


def test_a_10x_atac_file_is_named_where_a_method_reads_another_atac_file(
        ready, tmp_path, capsys):
    """A 10x file under atac.h5 next to a real rna.h5 and adt.h5. 0.3.4 raised
    "Accessing a group is done with bytes or str, not <class 'slice'>" for
    scan without a category, although the RNA + ADT rows have their files."""
    d = tmp_path / "STRAY"
    d.mkdir()
    _matrix(d / "rna.h5", [f"G{i}" for i in range(50)], 100)
    _matrix(d / "adt.h5", [f"AB{i}" for i in range(8)], 100)
    _tenx(d / "atac.h5", ("Peaks", PEAKS))
    df = mtb.scan("STRAY", verbose=False, data_path=tmp_path)
    ok = df[df["files_ok"]]
    assert "totalVI" in set(ok["method"]) and set(ok["modalities"]) == {"rna+adt"}
    # diagonal: a peak method does not find atac_peak.h5 and is told what is
    # there, a gene-activity method reads atac.h5
    kw = dict(methods=["SCALEX", "Seurat_v5"], data_path=tmp_path)
    df = mtb.scan("STRAY", "diagonal", verbose=False, **kw)
    assert list(df["method"]) == ["SCALEX", "Seurat_v5"]
    reasons = [f"atac.h5 {TENX}",
               "Seurat_v5 needs peak ATAC (atac_peak.h5), and the folder has ATAC (atac.h5)."]
    assert list(df["reason"]) == reasons
    plan = mtb.run_all("STRAY", "diagonal", out_dir=tmp_path / "out", dry_run=True, **kw)
    assert list(plan["reason"]) == reasons and not plan["runnable"].any()
    pm = mtb.scan("STRAY", "diagonal", per_method=True, **kw)
    assert list(pm["reason"]) == reasons
    printed = capsys.readouterr()
    assert "Accessing a group" not in printed.out + printed.err
    assert not (tmp_path / "out").exists()
    # the two readers of feature names give no names for the group
    # matrix/features of a 10x file
    path = d / "atac.h5"
    assert _resolve._sniff_features(str(path), path.stat().st_mtime_ns) == ()
    assert _resolve._all_features(str(path), path.stat().st_mtime_ns) == ()
    assert _resolve._peak_fraction_of(path) is None


def test_the_reason_starts_with_the_file_when_the_data_path_has_a_space(
        ready, tmp_path, capsys):
    """The reason column cuts a path to its file name, and takes a path to
    end at white space: a sentence that led with the full path began
    "My Data/TENX/rna.h5 is a 10x Genomics file"."""
    base = tmp_path / "My Data"
    (base / "TENX").mkdir(parents=True)
    _tenx(base / "TENX" / "rna.h5", ("Antibody Capture", [f"AB{i}" for i in range(8)]))
    shutil.copy(base / "TENX" / "rna.h5", base / "TENX" / "adt.h5")
    (base / "ZERO").mkdir()
    _matrix(base / "ZERO" / "rna.h5", [], 100)
    _matrix(base / "ZERO" / "adt.h5", [f"AB{i}" for i in range(8)], 100)
    zero = "rna.h5 holds 0 features. Write the folder again with mtb.io.export_dataset."
    for name, short in (("TENX", f"rna.h5 {TENX}"), ("ZERO", zero)):
        kw = dict(methods=["totalVI"], data_path=base)
        df = mtb.scan(name, "vertical", verbose=False, **kw)
        assert list(df["reason"]) == [short]
        assert "My Data" not in df["files_reason"][0]
        pm = mtb.scan(name, "vertical", per_method=True, **kw)
        assert list(pm["reason"]) == [short]
        assert capsys.readouterr().out.splitlines()[-1] == f"[scan] totalVI: {short}"
        with pytest.raises(ValueError) as e:
            mtb.run_all(name, "vertical", out_dir=tmp_path / "out", **kw)
        assert f"  totalVI (rna+adt): {short}\n" in str(e.value)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(config, "_CLI", True)
        df = mtb.scan("TENX", "vertical", verbose=False, methods=["totalVI"], data_path=base)
    assert list(df["reason"]) == [f"rna.h5 {TENX_CLI}"]
    assert not (tmp_path / "out").exists()


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
    assert str(e.value) == f"scBridge reads rna.h5 of DIAG, which {TENX}"
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


def test_scan_and_inputs_for_describe_the_check_of_matrix_files_alike():
    """What the check tests, and no more: another file that is not a dataset
    file (an .h5ad copied to rna.h5) passes it and is refused by mtb.run, so
    neither docstring says that the input files "are dataset files"."""
    scan_doc = " ".join(mtb.scan.__doc__.split())
    inputs_doc = " ".join(mtb.inputs_for.__doc__.split())
    assert ("no matrix file is a 10x Genomics ``.h5`` under the name of a dataset file or "
            "holds 0 features or 0 cells,") in scan_doc
    assert ("``ValueError`` when a matrix file is a 10x Genomics ``.h5`` under the name of "
            "a dataset file, or holds 0 features or 0 cells;") in inputs_doc
    assert "are dataset files" not in scan_doc + inputs_doc


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
