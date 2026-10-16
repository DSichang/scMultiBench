"""convert in dataset mode says what to do with a canonical .h5.

Dataset mode (--rna, --adt, --atac, --atac-from, --labels, ...) reads an
.h5ad or .h5mu. Given a canonical rna.h5, 0.3.8 said the file "has no
dataset 'matrix/data'" although it has one. The message now says the file is
already canonical and names the option that reads it.

It gives a copy command only when that command is certainly the whole
export: one matrix as it is (selector X), no labels, no batch split, a file
name that does not depend on an ATAC kind left out, and an empty folder (the
export orders new rows by the files already there; a copy does not). Every
other case ends with "Pass the .h5ad or .h5mu it came from." Each command is
run as printed; the folder then holds the file the same export from an
.h5ad writes, and multibench scan does not name it as missing.
"""
import csv
import io
import shlex

import h5py
import numpy as np
import pytest

from multibench import cli
from multibench.engine import ingest

ad = pytest.importorskip("anndata")

N = 12
FEATURES = {"rna": [f"G{i}" for i in range(6)], "adt": [f"P{i}" for i in range(4)],
            "atac": [f"chr1:{i}000-{i}500" for i in range(1, 7)]}
SOURCE = "Pass the .h5ad or .h5mu it came from."


def _inputs(root):
    """X.h5ad and the canonical c_X.h5 of the same matrix, for rna, adt, atac."""
    rng = np.random.default_rng(0)
    for mod, feats in FEATURES.items():
        a = ad.AnnData(rng.poisson(2.0, (N, len(feats))).astype(float))
        a.obs_names = [f"cell{i}" for i in range(N)]
        a.var_names = feats
        a.obs["ct"] = ["t0", "t1"] * (N // 2)
        a.obs["bt"] = ["b0"] * (N // 2) + ["b1"] * (N // 2)
        a.write_h5ad(root / f"{mod}.h5ad")
        with h5py.File(root / f"c_{mod}.h5", "w") as f:
            f["matrix/data"] = a.X.T
            f["matrix/features"] = np.array(feats, dtype="S")
            f["matrix/barcodes"] = np.array(list(a.obs_names), dtype="S")


def _run(capsys, argv):
    rc = cli.main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _error(capsys, argv):
    rc, _, err = _run(capsys, argv)
    assert rc == 1
    return err.splitlines()[0]


def _scan_reasons(capsys, folder, category):
    argv = ["scan", folder.name, "--data-path", str(folder.parent), "--modalities", "all",
            "--format", "csv", "--columns", "method,reason"]
    rc, out, _ = _run(capsys, argv + (["--category", category] if category else []))
    assert rc == 0
    return [r["reason"] for r in csv.DictReader(io.StringIO(out))]


def _missing(reasons, name):
    """The sentences of scan reasons that say ``name`` is not in the folder."""
    return [x for r in reasons for x in r.split(". ")
            if name in x and any(k in x for k in ("missing", "not in the folder",
                                                  "none of them"))]


def _category(flags):
    return flags[flags.index("--category") + 1] if "--category" in flags else None


#: (source, flags): a copy command is printed
COPY = [
    ("rna", ["--rna", "X"]),
    ("rna", ["--rna", "X", "--category", "vertical"]),
    ("rna", ["--rna", "X", "--category", "diagonal"]),
    ("rna", ["--rna", "X", "--category", "mosaic", "--batch-index", "2"]),
    ("adt", ["--adt", "X", "--category", "vertical"]),
    ("adt", ["--adt", "X", "--category", "cross", "--batch-index", "2"]),
    ("atac", ["--atac", "X", "--atac-kind", "peak", "--category", "vertical"]),
    ("atac", ["--atac", "X", "--atac-kind", "gene_activity", "--category", "vertical"]),
    ("atac", ["--atac", "X", "--atac-kind", "peak", "--category", "diagonal"]),
    ("atac", ["--atac", "X", "--atac-kind", "gene_activity", "--category", "diagonal"]),
    ("atac", ["--atac", "X", "--atac-kind", "peak", "--category", "mosaic",
              "--batch-index", "3"]),
    ("atac", ["--atac", "X", "--atac-kind", "peak", "--category", "cross",
              "--batch-index", "1"]),
    ("atac", ["--atac", "X", "--atac-kind", "gene_activity"]),
    ("rna", ["--rna", "X", "--category", "vertical", "--dtype", "float32"]),
]


@pytest.mark.parametrize("mod, flags", COPY)
def test_the_copy_command_is_the_whole_export(tmp_path, capsys, monkeypatch, mod, flags):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in dir").mkdir()                       # a path that needs quoting
    _inputs(tmp_path / "in dir")
    src = str(tmp_path / "in dir" / f"c_{mod}.h5")
    out = tmp_path / "copy" / "DS"
    err = _error(capsys, ["convert", src, str(out)] + flags)
    head = f"error: {src} is already a canonical .h5 file, and {flags[0]} reads an .h5ad or .h5mu. "
    assert err.startswith(head + "Copy it into the folder with: multibench convert ")
    assert not out.exists()
    # the same export from the .h5ad
    ref = tmp_path / "ref" / "DS"
    rc, _, _ = _run(capsys, ["convert", str(tmp_path / "in dir" / f"{mod}.h5ad"), str(ref)]
                    + flags)
    assert rc == 0
    files = sorted(p.name for p in ref.iterdir())
    # the command, as printed
    argv = shlex.split(err[len(head) + len("Copy it into the folder with: "):])
    assert argv[:2] == ["multibench", "convert"]
    rc, _, _ = _run(capsys, argv[1:])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == files
    with h5py.File(out / files[0], "r") as f:
        assert f["matrix/data"].dtype == np.dtype("float32" if "float32" in flags else "float64")
    reasons = _scan_reasons(capsys, out, _category(flags))
    assert reasons and not _missing(reasons, files[0])


#: (source, flags, the option named): no copy command
SOURCE_CASES = [
    # labels: a copy would drop them
    ("rna", ["--rna", "X", "--category", "vertical", "--labels", "obs:ct"], "--rna"),
    ("rna", ["--labels", "obs:ct", "--category", "vertical"], "--labels"),
    # a batch split writes rna1.h5, rna2.h5, ...
    ("rna", ["--rna", "X", "--category", "cross", "--batch", "obs:bt"], "--rna"),
    # two matrices from one file
    ("rna", ["--rna", "X", "--adt", "X", "--category", "vertical"], "--rna"),
    # a feature filter or a layer: not the file as it is
    ("rna", ["--rna", "X[feature_types=Gene Expression]"], "--rna"),
    ("rna", ["--rna", "layer:counts", "--category", "vertical"], "--rna"),
    # the ATAC file name depends on a kind that was not given
    ("atac", ["--atac", "X", "--category", "diagonal"], "--atac"),
    ("atac", ["--atac", "X", "--category", "vertical"], "--atac"),
    # peak without a category writes atac.h5 and atac_peak.h5
    ("atac", ["--atac", "X", "--atac-kind", "peak"], "--atac"),
]


@pytest.mark.parametrize("mod, flags, flag", SOURCE_CASES)
def test_other_cases_ask_for_the_source(tmp_path, capsys, monkeypatch, mod, flags, flag):
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    err = _error(capsys, ["convert", f"c_{mod}.h5", "OUT"] + flags)
    assert err == (f"error: c_{mod}.h5 is already a canonical .h5 file, and {flag} reads "
                   f"an .h5ad or .h5mu. {SOURCE}")
    assert not (tmp_path / "OUT").exists()


def test_a_folder_with_files_gets_no_copy_command(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    assert _run(capsys, ["convert", "rna.h5ad", "OUT", "--rna", "X",
                         "--category", "vertical"])[0] == 0
    err = _error(capsys, ["convert", "c_adt.h5", "OUT", "--adt", "X", "--category",
                          "vertical"])
    assert err.endswith(SOURCE)


def test_atac_from_alone_gets_the_copy_command(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    err = _error(capsys, ["convert", "rna.h5ad", "OUT", "--atac-from", "c_atac.h5",
                          "--atac-kind", "peak", "--category", "diagonal"])
    assert err == ("error: c_atac.h5 is already a canonical .h5 file, and --atac-from reads "
                   "an .h5ad or .h5mu. Copy it into the folder with: multibench convert "
                   "c_atac.h5 OUT/atac_peak.h5")
    assert _run(capsys, shlex.split(err.split("with: ")[1])[1:])[0] == 0
    # the same export from an .h5ad writes this one file
    assert _run(capsys, ["convert", "rna.h5ad", "REF", "--atac-from", "atac.h5ad",
                         "--atac-kind", "peak", "--category", "diagonal"])[0] == 0
    assert sorted(p.name for p in (tmp_path / "REF").iterdir()) == ["atac_peak.h5"]
    assert sorted(p.name for p in (tmp_path / "OUT").iterdir()) == ["atac_peak.h5"]
    reasons = _scan_reasons(capsys, tmp_path / "OUT", "diagonal")
    assert reasons and not _missing(reasons, "atac_peak.h5")


@pytest.mark.parametrize("flags", [
    ["--rna", "X", "--atac-kind", "peak"],          # the RNA comes from the source
    ["--atac-kind", "peak", "--labels", "obs:ct"],
    [],                                             # no kind: the file name is not known
])
def test_atac_from_with_more_asks_for_the_source(tmp_path, capsys, monkeypatch, flags):
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    err = _error(capsys, ["convert", "rna.h5ad", "OUT", "--atac-from", "c_atac.h5",
                          "--category", "diagonal"] + flags)
    assert err == ("error: c_atac.h5 is already a canonical .h5 file, and --atac-from reads "
                   f"an .h5ad or .h5mu. {SOURCE}")


def test_a_file_that_is_not_canonical_keeps_its_message(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f["data"] = np.ones((3, 2))
    err = _error(capsys, ["convert", "x.h5", "OUT", "--rna", "X"])
    assert err.startswith("error: x.h5 has no dataset 'matrix/data'; found keys ['data']")
