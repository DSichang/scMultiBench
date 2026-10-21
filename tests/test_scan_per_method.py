"""scan: one row per method, for the check after an export.

The tutorials' own-data section calls it right after ``export_dataset``. It
answers one question per method (can it run on this folder?) in three
columns, and prints what the folder holds, the cells of each matrix file and
the full reason of each method that cannot run.

The env probe and the platform are pinned; ``_run`` is faked in the real-run
tests.
"""
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, registry
from multibench.engine import runner as R
from tests.conftest import full_scan

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
GENES = [f"GENE{i}" for i in range(40)]
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
MACOS = "Methods run only on Linux, and this computer runs macOS."


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _on_disk(ds):
    if not (config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")


def _h5(path, feats, bars, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), len(bars)))
                         .astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array(bars, dtype="S32"))


def _labels(path, n):
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(path, index=False)


def _cite(root, name="MYCITE", n=60, labels=True):
    d = root / name
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", GENES[:30], bars)
    _h5(d / "adt.h5", [f"p{i}" for i in range(8)], bars)
    if labels:
        _labels(d / "cty.csv", n)
    return root


def _donors(root, name, donors, n=40):
    """A cross folder of RNA + ADT with one batch per donor."""
    d = root / name
    d.mkdir(parents=True)
    for b in range(1, donors + 1):
        bars = [f"d{b}c{i}" for i in range(n)]
        _h5(d / f"rna{b}.h5", GENES[:30], bars)
        _h5(d / f"adt{b}.h5", [f"p{i}" for i in range(8)], bars)
        _labels(d / f"cty{b}.csv", n)
    return root


@pytest.fixture
def linux(monkeypatch, tmp_path):
    """A Linux host with every environment and a GPU; no method scripts on
    this machine, no ref. ``_run`` is faked and records the methods it ran."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    calls = []

    class _Res:
        output = np.zeros((60, 5))

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(method)
        return _Res()
    monkeypatch.setattr(W, "_run", fake_run)
    return calls


def _only(monkeypatch, methods):
    """Only the environments of ``methods`` are installed."""
    have = frozenset(envs.group_for(m) for m in methods)
    monkeypatch.setattr(W, "_installed_envs", lambda: have)


@pytest.mark.parametrize("ds,cat", [("D11", "vertical"), ("D46mini", "mosaic"),
                                    ("D52mini", "cross")])
def test_runnable_is_what_run_all_tries(ds, cat, linux, monkeypatch, tmp_path):
    _on_disk(ds)
    # some environments, no GPU: a part of the methods can run
    methods = sorted({s.id for s, *_ in W._variant_rows(cat)})
    _only(monkeypatch, methods[::2])
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    for kw in ({}, {"methods": methods[:5]}, {"modalities": "all"}):
        pm = _quiet(mtb.scan, ds, cat, verbose=False, **kw)
        linux.clear()
        _quiet(mtb.run_all, ds, cat, tmp_path / f"out{len(kw)}{list(kw)}", evaluate=False,
               verbose=False, **kw)
        assert sorted(pm[pm["runnable"]]["method"]) == sorted(set(linux)) and linux, kw
        assert len(set(linux)) == len(linux)


def _tenx(path, n=60):
    """A 10x Genomics file: a sparse matrix (matrix/indptr) with its
    features in a group."""
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"] = np.ones(n), np.zeros(n, dtype=int)
        g["indptr"] = np.arange(n + 1)
        g["shape"] = np.array([40, n])
        g["barcodes"] = np.array([f"c{i}-1" for i in range(n)], dtype="S32")
        g.create_group("features")["name"] = np.array(GENES, dtype="S32")


def test_the_first_line_names_10x_files(linux, tmp_path, capsys):
    """A 10x file copied to rna.h5 is not RNA that the folder holds."""
    def first(*a, **kw):
        mtb.config.DEFAULT.data_path = tmp_path
        pm = _quiet(mtb.scan, *a, **kw)
        return pm, capsys.readouterr().out.splitlines()[0]

    counts = ("[scan] 0 of 2 methods have their input files in {0}. 0 of 2 can run on "
              "this computer. ")
    for name, other, methods in (("TXADT", "adt.h5", ["Seurat_WNN", "totalVI"]),
                                 ("TXATAC", "atac.h5", ["VIMCCA", "scMM"])):
        (tmp_path / name).mkdir()
        _tenx(tmp_path / name / "rna.h5")
        _tenx(tmp_path / name / other)
        pm, line = first(name, "vertical", methods=methods)
        assert line == counts.format(name) + (
            f"{name} holds no dataset file: rna.h5 and {other} are 10x Genomics files "
            f"under the names of dataset files.")
        assert "holds RNA and" not in line
        # the rows listed are still those of the names in the folder
        assert pm.attrs["held"] == ["rna", other[:-len(".h5")]]
        assert list(pm["method"]) == methods and not pm["runnable"].any()
    # with the rows that are not listed: the same sentence starts the note
    mtb.config.DEFAULT.data_path = tmp_path
    _quiet(mtb.run_all, 'TXADT', 'vertical', dry_run=True)
    assert capsys.readouterr().out.splitlines()[1] == (
        "[run_all] TXADT holds no dataset file: rna.h5 and adt.h5 are 10x Genomics files "
        "under the names of dataset files.")
    # one of two: the family of the other file is held, and the 10x file is named
    _cite(tmp_path, "MIX")
    _tenx(tmp_path / "MIX" / "rna.h5")
    pm, line = first("MIX", "vertical", methods=["Seurat_WNN", "totalVI"])
    assert line == counts.format("MIX") + (
        "MIX holds ADT. rna.h5 is a 10x Genomics file under the name of a dataset file.")
    assert pm.attrs["held"] == ["rna", "adt"]
    # numbered files: the batches of the families named
    _donors(tmp_path, "TWO", 2)
    _tenx(tmp_path / "TWO" / "rna1.h5", n=40)
    _tenx(tmp_path / "TWO" / "adt2.h5", n=40)
    pm, line = first("TWO", "cross", methods=["StabMap", "sciPENN"])
    assert line == counts.format("TWO") + (
        "TWO holds RNA and ADT, in batches 1-2. rna1.h5 and adt2.h5 are 10x Genomics "
        "files under the names of dataset files.")
    # a 2-D matrix/data next to a matrix/indptr is a dataset file, as for mtb.run
    _cite(tmp_path, "BOTH")
    with h5py.File(tmp_path / "BOTH" / "rna.h5", "a") as f:
        f["matrix/indptr"] = np.arange(3)
    pm, line = first("BOTH", "vertical", methods=["Seurat_WNN", "totalVI"])
    assert line == ("[scan] 2 of 2 methods have their input files in BOTH. 2 of 2 can run "
                    "on this computer. BOTH holds RNA and ADT.")
    # a vertical folder with both ATAC forms: every vertical method reads
    # atac_peak.h5, so a 10x file under atac_gas.h5 takes no ATAC from it
    bars = [f"c{i}" for i in range(60)]
    (tmp_path / "GAS").mkdir()
    _h5(tmp_path / "GAS" / "rna.h5", GENES[:30], bars)
    _h5(tmp_path / "GAS" / "atac_peak.h5", PEAKS, bars)
    _tenx(tmp_path / "GAS" / "atac_gas.h5")
    got = mtb.engine.resolve.inputs_for("GAS", "vertical", "scMM", data_path=tmp_path, check=True)
    assert got == {"rna": str(tmp_path / "GAS" / "rna.h5"),
                   "atac": str(tmp_path / "GAS" / "atac_peak.h5")}      # X13
    pm, line = first("GAS", "vertical", methods=["VIMCCA", "scMM"])
    assert line == ("[scan] 2 of 2 methods have their input files in GAS. 2 of 2 can run "
                    "on this computer. GAS holds RNA and ATAC.")
    assert pm.attrs["held"] == ["rna", "atac"] and pm["runnable"].all()
    # as diagonal a gene-activity method reads atac_gas.h5, and the file is named
    pm, line = first("GAS", "diagonal", methods=["SCALEX", "Seurat_v5"])
    assert line == ("[scan] 1 of 2 methods has its input files in GAS. 1 of 2 can run on "
                    "this computer. GAS holds RNA and ATAC. atac_gas.h5 is a 10x Genomics "
                    "file under the name of a dataset file.")
    assert list(pm["method"]) == ["Seurat_v5", "SCALEX"]
    assert list(pm["runnable"]) == [True, False]


def test_files_of_another_layout_are_not_called_no_file(linux, tmp_path, capsys):
    """The line under it lists the matrix files, so "no file" was false."""
    def lines(*a, **kw):
        mtb.config.DEFAULT.data_path = tmp_path
        pm = _quiet(mtb.scan, *a, **kw)
        return pm, capsys.readouterr().out.splitlines()

    counts = ("[scan] 0 of 2 methods have their input files in {0}. 0 of 2 can run on "
              "this computer. ")
    # one file per batch, scanned as vertical
    _donors(tmp_path, "D1", 2)
    pm, out = lines("D1", "vertical", methods=["Matilda", "sciPENN"])
    assert out[:2] == [
        counts.format("D1") + "D1 holds no RNA, ADT or ATAC file under a name that a "
        "vertical method reads.",
        "[scan] Cells per file: rna1.h5 40, adt1.h5 40, rna2.h5 40, adt2.h5 40."]
    assert pm.attrs["held"] == []
    # one file per modality, scanned as cross
    _cite(tmp_path)
    pm, out = lines("MYCITE", "cross", methods=["StabMap", "sciPENN"])
    assert out[:2] == [
        counts.format("MYCITE") + "MYCITE holds no RNA, ADT or ATAC file under a name "
        "that a cross method reads.",
        "[scan] Cells per file: rna.h5 60, adt.h5 60."]
    assert pm.attrs["held"] == []
    # a folder with label files and no matrix file holds none, as an empty one
    (tmp_path / "LABELS").mkdir()
    _labels(tmp_path / "LABELS" / "cty.csv", 60)
    pm, out = lines("LABELS", "vertical", methods=["Matilda", "sciPENN"])
    assert out[0] == counts.format("LABELS") + "LABELS holds no RNA, ADT or ATAC file."
    assert not any(l.startswith("[scan] Cells per file") for l in out)


def test_one_method_that_is_not_listed(linux, monkeypatch, capsys):
    # D52mini holds RNA and ADT in three batches; UnitedNet's cross variant reads ATAC
    _on_disk("D52mini")
    pm = _quiet(mtb.scan, "D52mini", "cross")
    line = capsys.readouterr().out.splitlines()[0]
    assert "UnitedNet" not in set(pm["method"]) and len(pm) == 10
    assert line == ("[scan] 10 of 10 methods have their input files in D52mini. 10 of 10 "
                    "can run on this computer. D52mini holds RNA and ADT, in batches 1-3.")
    # ADT and ATAC: the methods that read RNA are not listed, without a sentence
    _on_disk("D58mini")
    _quiet(mtb.scan, "D58mini", "cross")
    assert capsys.readouterr().out.splitlines()[0].endswith(
        " D58mini holds ADT and ATAC, in batches 1-2.")


# ------------------------------------------------------------------ cells per file
def test_cells_per_file_line(linux, tmp_path, capsys):
    def second(*a, **kw):
        mtb.config.DEFAULT.data_path = tmp_path
        pm = _quiet(mtb.scan, *a, **kw)
        lines = capsys.readouterr().out.splitlines()
        return pm, [l for l in lines if l.startswith("[scan] Cells per file")]

    _cite(tmp_path, n=1718)
    pm, line = second("MYCITE", "vertical", methods=["Matilda"])
    assert line == ["[scan] Cells per file: rna.h5 1,718, adt.h5 1,718."]
    assert pm.attrs["cells"] == {"rna.h5": 1718, "adt.h5": 1718}
    # numbered: by batch number, then RNA, ADT, ATAC
    _donors(tmp_path, "THREE", 3)
    d = tmp_path / "THREE"
    _h5(d / "atac2.h5", PEAKS, [f"d2c{i}" for i in range(40)])
    pm, line = second("THREE", "cross", methods=["StabMap"])
    assert line == ["[scan] Cells per file: rna1.h5 40, adt1.h5 40, rna2.h5 40, adt2.h5 40, "
                    "atac2.h5 40, rna3.h5 40, adt3.h5 40."]
    assert list(pm.attrs["cells"]) == ["rna1.h5", "adt1.h5", "rna2.h5", "adt2.h5",
                                       "atac2.h5", "rna3.h5", "adt3.h5"]
    # more than 8 files: the first 8, and the others counted
    _donors(tmp_path, "FIVE", 5)
    pm, line = second("FIVE", "cross", methods=["StabMap"])
    assert line == ["[scan] Cells per file: rna1.h5 40, adt1.h5 40, rna2.h5 40, adt2.h5 40, "
                    "rna3.h5 40, adt3.h5 40, rna4.h5 40, adt4.h5 40 and 2 more."]
    assert len(pm.attrs["cells"]) == 10 and set(pm.attrs["cells"].values()) == {40}
    # a file whose barcodes cannot be read, and files that are not of the layout
    d = tmp_path / "MYCITE"
    with h5py.File(d / "atac.h5", "w") as f:           # a 10x-style file: no barcodes
        f.create_dataset("matrix/data", data=np.arange(5.0))
    (d / "notes.txt").write_text("x")
    _h5(d / "peak.h5", PEAKS, [f"c{i}" for i in range(7)])     # an older name
    pm, line = second("MYCITE", "vertical", methods=["Matilda"])
    assert line == ["[scan] Cells per file: rna.h5 1,718, adt.h5 1,718, atac.h5 ?."]
    assert pm.attrs["cells"] == {"rna.h5": 1718, "adt.h5": 1718, "atac.h5": None}
    # no matrix file: no line
    (tmp_path / "EMPTY").mkdir()
    pm, line = second("EMPTY", "vertical", methods=["Matilda"])
    assert line == [] and pm.attrs["cells"] == {}


def test_a_name_left_from_the_example_shows_in_the_cells(linux, tmp_path, capsys):
    """The reader's RNA next to an ADT file of other cells: two numbers."""
    _cite(tmp_path)
    _h5(tmp_path / "MYCITE" / "adt.h5", [f"p{i}" for i in range(8)],
        [f"x{i}" for i in range(45)])
    mtb.config.DEFAULT.data_path = tmp_path
    _quiet(mtb.scan, 'MYCITE', 'vertical', methods=['totalVI'])
    assert "[scan] Cells per file: rna.h5 60, adt.h5 45." in capsys.readouterr().out


