"""scan(per_method=True): one row per method, for the check after an export.

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


# ------------------------------------------------------------------ the frame
def test_columns_order_and_one_row_per_method(linux, monkeypatch):
    _only(monkeypatch, ["Matilda", "sciPENN"])
    pm = mtb.scan("D11", "vertical", per_method=True, verbose=False)
    full = full_scan("D11", "vertical", verbose=False)
    assert type(pm) is pd.DataFrame
    assert list(pm.columns) == ["method", "modalities", "runnable", "reason"]
    assert pm["method"].is_unique and set(pm["method"]) == set(full["method"])
    assert pm["runnable"].dtype == bool
    # runnable methods first, then by name
    assert list(pm["method"][:2]) == ["Matilda", "sciPENN"] and list(pm["runnable"][:2]) == [
        True, True]
    rest = list(pm["method"][2:])
    assert rest == sorted(rest) and not pm["runnable"][2:].any()
    assert list(pm.index) == list(range(len(pm)))
    # when runnable: empty, or the note that the method scripts are not fetched
    # yet (0.4: the full plan has it in caveat); else the reason of a row of the
    # full table, as it is
    by = full.set_index("method")["reason"]
    assert all(r == "" or r.startswith("The method scripts are not in ")
               for r in pm["reason"][:2])
    for m, reason in zip(pm["method"][2:], pm["reason"][2:]):
        assert reason and reason == by[m]
    assert set(pm.attrs) == {"held", "not_listed", "not_held", "cells", "not_listed_batch",
                             "not_listed_methods"}
    assert {k: pm.attrs[k] for k in full.attrs} == full.attrs


def test_the_example_of_the_tutorials(linux, monkeypatch, tmp_path, capsys):
    """Spec 2.6: the check cell on a folder that is fine, two methods."""
    _only(monkeypatch, ["Matilda", "totalVI"])
    _cite(tmp_path, n=1718)
    pm = mtb.scan("MYCITE", "vertical", methods=["Matilda", "totalVI"], data_path=tmp_path,
                  per_method=True)
    assert capsys.readouterr().out == (
        "[scan] 2 of 2 methods have their input files in MYCITE. 2 of 2 can run on this "
        "computer. MYCITE holds RNA and ADT.\n"
        "[scan] Cells per file: rna.h5 1,718, adt.h5 1,718.\n")
    reasons = pm.pop("reason").tolist()
    assert pm.to_dict("list") == {"method": ["Matilda", "totalVI"],
                                  "modalities": ["rna+adt", "rna+adt"],
                                  "runnable": [True, True]}
    # 0.4: a runnable row says when the method scripts are not fetched yet
    assert all(r == "" or r.startswith("The method scripts are not in ") for r in reasons)


@pytest.mark.parametrize("ds,cat", [("D11", "vertical"), ("D46mini", "mosaic"),
                                    ("D52mini", "cross")])
def test_runnable_is_what_run_all_tries(ds, cat, linux, monkeypatch, tmp_path):
    _on_disk(ds)
    # some environments, no GPU: a part of the methods can run
    methods = sorted({s.id for s, *_ in W._variant_rows(cat)})
    _only(monkeypatch, methods[::2])
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    for kw in ({}, {"methods": methods[:5]}, {"modalities": "all"}):
        pm = _quiet(mtb.scan, ds, cat, per_method=True, verbose=False, **kw)
        linux.clear()
        _quiet(mtb.run_all, ds, cat, tmp_path / f"out{len(kw)}{list(kw)}", evaluate=False,
               verbose=False, **kw)
        assert sorted(pm[pm["runnable"]]["method"]) == sorted(set(linux)) and linux, kw
        assert len(set(linux)) == len(linux)


# ------------------------------------------------------------------ the first line
def test_first_line(linux, monkeypatch, tmp_path, capsys):
    def first(*a, **kw):
        _quiet(mtb.scan, *a, per_method=True, **kw)
        return capsys.readouterr().out.splitlines()[0]

    _cite(tmp_path)
    kw = dict(data_path=tmp_path)
    # every environment: nothing to add
    assert first("MYCITE", "vertical", methods=["Matilda", "totalVI"], **kw) == (
        "[scan] 2 of 2 methods have their input files in MYCITE. 2 of 2 can run on this "
        "computer. MYCITE holds RNA and ADT.")
    # one method, and one that lacks its file; an environment is missing
    _only(monkeypatch, ["Matilda"])
    assert first("MYCITE", "vertical", methods=["Matilda"], **kw) == (
        "[scan] 1 of 1 method has its input files in MYCITE. 1 of 1 can run on this "
        "computer. MYCITE holds RNA and ADT.")
    assert first("MYCITE", "vertical", methods=["Matilda", "totalVI", "MIRA"], **kw) == (
        "[scan] 2 of 3 methods have their input files in MYCITE. 1 of 3 can run on this "
        "computer. MYCITE holds RNA and ADT. mtb.env.doctor() checks the environments.")
    monkeypatch.setattr(config, "_CLI", True)
    assert first("MYCITE", "vertical", methods=["totalVI"], **kw).endswith(
        " MYCITE holds RNA and ADT. multibench env doctor checks the environments.")
    monkeypatch.setattr(config, "_CLI", False)
    # numbered files: the batches
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    _donors(tmp_path, "TWO", 2)
    assert first("TWO", "cross", methods=["StabMap", "sciPENN"], **kw) == (
        "[scan] 0 of 2 methods have their input files in TWO. 0 of 2 can run on this "
        "computer. TWO holds RNA and ADT, in batches 1-2.")
    _donors(tmp_path, "ONE", 1)
    assert first("ONE", "cross", methods=["StabMap"], **kw).endswith(
        " ONE holds RNA and ADT, in batch 1.")
    # no file of any modality
    (tmp_path / "EMPTY").mkdir()
    assert first("EMPTY", "vertical", methods=["Matilda", "totalVI"], **kw) == (
        "[scan] 0 of 2 methods have their input files in EMPTY. 0 of 2 can run on this "
        "computer. EMPTY holds no RNA, ADT or ATAC file.")


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
        pm = _quiet(mtb.scan, *a, per_method=True, data_path=tmp_path, **kw)
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
    _quiet(mtb.run_all, "TXADT", "vertical", data_path=tmp_path, dry_run=True)
    assert capsys.readouterr().out.splitlines()[1] == (
        "[run_all] TXADT holds no dataset file: rna.h5 and adt.h5 are 10x Genomics files "
        "under the names of dataset files. 19 rows that read ATAC are not listed. "
        'modalities="all" lists them.')
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
    got = mtb.inputs_for("GAS", "vertical", "scMM", data_path=tmp_path, check=True)
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
        pm = _quiet(mtb.scan, *a, per_method=True, data_path=tmp_path, **kw)
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


def test_first_line_counts_the_methods_that_are_not_listed(linux, monkeypatch, capsys):
    # D11 holds no ATAC: four vertical methods read nothing else
    pm = mtb.scan("D11", "vertical", per_method=True)
    line = capsys.readouterr().out.splitlines()[0]
    every = mtb.scan("D11", "vertical", modalities="all", verbose=False)
    gone = sorted(set(every["method"]) - set(pm["method"]))
    assert len(gone) == 4 and "MIRA" in gone
    assert line == ("[scan] 14 of 14 methods have their input files in D11. 14 of 14 can "
                    "run on this computer. D11 holds RNA and ADT. 4 methods that need ATAC "
                    'are not listed. modalities="all" lists them.')
    # with them: every method, and no such sentence
    pm = mtb.scan("D11", "vertical", per_method=True, modalities="all")
    line = capsys.readouterr().out.splitlines()[0]
    assert len(pm) == 18 and line == (
        "[scan] 14 of 18 methods have their input files in D11. 14 of 18 can run on this "
        "computer. D11 holds RNA and ADT.")
    # a method named in methods= keeps its rows: it is listed, with its reason
    pm = mtb.scan("D11", "vertical", per_method=True, methods=["Matilda", "MIRA"])
    line = capsys.readouterr().out.splitlines()[0]
    assert list(pm["method"]) == ["Matilda", "MIRA"] and "not listed" not in line


def test_one_method_that_is_not_listed(linux, monkeypatch, capsys):
    # D52mini holds RNA and ADT in three batches; UnitedNet's cross variant reads ATAC
    _on_disk("D52mini")
    pm = _quiet(mtb.scan, "D52mini", "cross", per_method=True)
    line = capsys.readouterr().out.splitlines()[0]
    assert "UnitedNet" not in set(pm["method"]) and len(pm) == 10
    assert line == ("[scan] 10 of 10 methods have their input files in D52mini. 10 of 10 "
                    "can run on this computer. D52mini holds RNA and ADT, in batches 1-3. "
                    '1 method that needs ATAC is not listed. modalities="all" lists it.')
    monkeypatch.setattr(config, "_CLI", True)
    _quiet(mtb.scan, "D52mini", "cross", per_method=True)
    assert capsys.readouterr().out.splitlines()[0].endswith(
        " 1 method that needs ATAC is not listed. --modalities all lists it.")
    # ADT and ATAC: the methods that read RNA
    _on_disk("D58mini")
    monkeypatch.setattr(config, "_CLI", False)
    pm = _quiet(mtb.scan, "D58mini", "cross", per_method=True)
    assert capsys.readouterr().out.splitlines()[0].endswith(
        " D58mini holds ADT and ATAC, in batches 1-2. 6 methods that need RNA are not "
        'listed. modalities="all" lists them.')


def test_what_the_folder_holds_does_not_follow_the_methods_named(linux, capsys):
    # D22mini holds RNA, ADT and ATAC; totalVI reads the first two
    _on_disk("D22mini")
    pm = mtb.scan("D22mini", "vertical", methods=["totalVI"], per_method=True)
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "[scan] 1 of 1 method has its input files in D22mini. 1 of 1 can run on this "
        "computer. D22mini holds RNA, ADT and ATAC.",
        "[scan] Cells per file: rna.h5 3,000, adt.h5 3,000, atac.h5 3,000."]
    assert pm.attrs["held"] == ["rna", "adt", "atac"]


def test_first_line_off_linux(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: MACOS)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    _cite(tmp_path)
    pm = mtb.scan("MYCITE", "vertical", methods=["Matilda", "totalVI"], data_path=tmp_path,
                  per_method=True)
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "[scan] 2 of 2 methods have their input files in MYCITE. 0 of 2 can run on this "
        "computer. MYCITE holds RNA and ADT. Method environments run only on Linux. On "
        "this computer you can check files, score embeddings and plot.",
        "[scan] Cells per file: rna.h5 60, adt.h5 60."]
    # the platform is the only block: the table says so, and no line repeats it
    assert not pm["runnable"].any()
    assert list(pm["reason"]) == [
        "Environment matilda runs only on Linux, not on this computer.",
        "Environment scmb_scvi runs only on Linux, not on this computer."]
    assert W.LINUX_ONLY_SUMMARY.startswith(W.LINUX_ONLY_CHECK + " ")


# ------------------------------------------------------------------ cells per file
def test_cells_per_file_line(linux, tmp_path, capsys):
    def second(*a, **kw):
        pm = _quiet(mtb.scan, *a, per_method=True, data_path=tmp_path, **kw)
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
    _quiet(mtb.scan, "MYCITE", "vertical", methods=["totalVI"], data_path=tmp_path,
           per_method=True)
    assert "[scan] Cells per file: rna.h5 60, adt.h5 45." in capsys.readouterr().out


# ------------------------------------------------------------------ reasons and caveats
def test_reasons_are_printed_in_full(linux, monkeypatch, tmp_path, capsys):
    _cite(tmp_path, labels=False)
    _only(monkeypatch, ["Matilda"])
    pm = mtb.scan("MYCITE", "vertical", methods=["Matilda", "totalVI", "sciPENN"],
                  data_path=tmp_path, per_method=True)
    out = capsys.readouterr().out.splitlines()
    assert out[2:] == [
        "[scan] Matilda: cty.csv is missing.",
        "[scan] sciPENN: Environment env_sciPENN is not installed. Run multibench env "
        "install --methods sciPENN --run. Use --category vertical to install the "
        "environments of every vertical method.",
        "[scan] totalVI: Environment scmb_scvi is not installed. Run multibench env "
        "install --methods totalVI --run. Use --category vertical to install the "
        "environments of every vertical method."]
    # one line per method that cannot run, in the order of the table, whole
    assert [l.split(": ")[0].split()[-1] for l in out[2:]] == list(pm["method"])
    assert [l.split(": ", 1)[1] for l in out[2:]] == list(pm["reason"])
    assert max(map(len, pm["reason"])) > 80           # longer than a printed cell
    # two donors: the missing batch
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    _donors(tmp_path, "TWO", 2)
    mtb.scan("TWO", "cross", methods=["StabMap", "sciPENN"], data_path=tmp_path,
             per_method=True)
    assert capsys.readouterr().out.splitlines()[2:] == [
        "[scan] StabMap: rna3.h5, adt3.h5 are missing.",
        "[scan] sciPENN: rna3.h5, adt3.h5 are missing."]


def test_off_linux_the_reason_line_leaves_the_platform_out(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: MACOS)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    _cite(tmp_path, labels=False)
    pm = mtb.scan("MYCITE", "vertical", methods=["Matilda", "totalVI"], data_path=tmp_path,
                  per_method=True)
    out = capsys.readouterr().out.splitlines()
    assert out[2:] == ["[scan] Matilda: cty.csv is missing."]
    assert list(pm["reason"]) == [
        "cty.csv is missing. Environment matilda runs only on Linux, not on this "
        "computer.",
        "Environment scmb_scvi runs only on Linux, not on this computer."]


def test_folder_caveats_are_printed_and_fixed_notes_are_not(linux, tmp_path, capsys):
    _donors(tmp_path, "FIVE", 5)
    pm = mtb.scan("FIVE", "cross", methods=["StabMap", "sciPENN"], data_path=tmp_path,
                  per_method=True)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == ("[scan] 2 of 2 methods have their input files in FIVE. 2 of 2 can "
                      "run on this computer. FIVE holds RNA and ADT, in batches 1-5.")
    assert out[2:] == ["[scan] StabMap reads batches 1-3 of 5. Batches 4-5 are not used.",
                       "[scan] sciPENN reads batches 1-3 of 5. Batches 4-5 are not used."]
    assert pm["runnable"].all()
    # the note on the method itself, the same on every folder, is run_all's
    _on_disk("D27mini")
    full = full_scan("D27mini", "diagonal", methods=["online_iNMF", "SCALEX"], verbose=False)
    # 3,000 RNA + 3,000 ATAC cells: enough, so the size note is not shown (X81)
    assert "online_iNMF needs about 5,000 cells" not in " ".join(full["caveat"])
    mtb.scan("D27mini", "diagonal", methods=["online_iNMF", "SCALEX"], per_method=True)
    out = capsys.readouterr().out
    assert "needs about 5,000 cells" not in out and len(out.splitlines()) == 2
    plan = mtb.run_all("D27mini", "diagonal", methods=["online_iNMF", "SCALEX"],
                       dry_run=True)
    # 6,000 cells: no size note (X81)
    assert "online_iNMF needs about 5,000 cells" not in capsys.readouterr().out and len(plan)


def test_the_caveat_of_a_row_that_does_not_run_is_left_out(linux, tmp_path, capsys):
    # UINMF's two-batch row is inside its three-batch row, which is the one
    # that runs on three donors
    _donors(tmp_path, "THREE", 3)
    mtb.scan("THREE", "cross", methods=["UINMF"], data_path=tmp_path, per_method=True)
    assert "UINMF reads batches" not in capsys.readouterr().out
    # with that row selected, it is the row that runs
    mtb.scan("THREE", "cross", methods=["UINMF"], data_path=tmp_path, per_method=True,
             modalities=["rna1", "rna2", "adt1", "adt2"])
    assert "[scan] UINMF reads batches 1-2 of 3. Batch 3 is not used." in \
        capsys.readouterr().out.splitlines()


def test_reason_row_choice(linux, monkeypatch, tmp_path):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    # Matilda on RNA + ADT without labels, every variant: the row whose files
    # are nearest speaks, not the first of the table
    _cite(tmp_path, labels=False)
    kw = dict(methods=["Matilda"], data_path=tmp_path, verbose=False)
    full = full_scan("MYCITE", "vertical", modalities="all", **kw)
    assert list(full["modalities"]) == ["rna+adt", "rna+atac", "rna+adt+atac"]
    pm = mtb.scan("MYCITE", "vertical", modalities="all", per_method=True, **kw)
    assert pm.loc[0, "reason"] == full.loc[0, "reason"]
    assert pm.loc[0, "reason"].startswith("cty.csv is missing. ")
    # RNA + ATAC: the first row of the table lacks adt.h5, the second lacks nothing
    d = tmp_path / "MULTIOME"
    d.mkdir()
    bars = [f"c{i}" for i in range(60)]
    _h5(d / "rna.h5", GENES[:30], bars)
    _h5(d / "atac.h5", PEAKS, bars)
    _labels(d / "cty.csv", 60)
    full = full_scan("MULTIOME", "vertical", modalities="all", **kw)
    on_disk = [W._inputs_on_disk(r, "MULTIOME", tmp_path) for _, r in full.iterrows()]
    assert list(full["modalities"]) == ["rna+adt", "rna+atac", "rna+adt+atac"]
    assert on_disk == [False, True, False] and not full["runnable"].any()
    pm = mtb.scan("MULTIOME", "vertical", modalities="all", per_method=True, **kw)
    assert pm.loc[0, "reason"] == full.loc[1, "reason"]
    assert pm.loc[0, "reason"].startswith("Environment matilda is not installed.")
    # no row has its files: the one that lacks the fewest
    (d / "atac.h5").unlink()
    (d / "cty.csv").unlink()
    full = full_scan("MULTIOME", "vertical", modalities="all", **kw)
    lacking = [W._missing_inputs(r, "MULTIOME", tmp_path) for _, r in full.iterrows()]
    assert lacking == [2, 2, 3]
    pm = mtb.scan("MULTIOME", "vertical", modalities="all", per_method=True, **kw)
    assert pm.loc[0, "reason"] == full.loc[0, "reason"]         # a tie: the first row


def test_per_method_without_a_category_is_the_default_scan(tmp_path):
    # 0.4: one row per method is scan's default; per_method= only warns; without a
    # category, a dataset the benchmark scored takes its category (D11: vertical)
    with pytest.warns(FutureWarning, match=r"scan\(per_method=\) is deprecated"):
        pm = mtb.scan("D11", per_method=True, verbose=False)
    pd.testing.assert_frame_equal(pm, mtb.scan("D11", verbose=False))
    pd.testing.assert_frame_equal(pm, mtb.scan("D11", "vertical", verbose=False))
    assert pm.columns.tolist() == ["method", "modalities", "runnable", "reason"]
    with pytest.raises(FileNotFoundError):
        _quiet(mtb.scan, "NO_SUCH_FOLDER", per_method=True, data_path=tmp_path)
    assert _quiet(mtb.scan, "D11", per_method=False,
                  verbose=False).columns.tolist() == W.SCAN_COLUMNS


def test_verbose_false_prints_nothing(linux, capsys):
    pm = mtb.scan("D11", "vertical", methods=["Matilda", "MIRA"], per_method=True,
                  verbose=False)
    assert capsys.readouterr().out == ""
    assert list(pm["method"]) == ["Matilda", "MIRA"] and pm.attrs["cells"] == {
        "rna.h5": 2864, "adt.h5": 2864}
