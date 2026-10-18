"""export says what to do with a canonical .h5.

export (--rna, --adt, --atac, --atac-from, --labels, ...) reads an
.h5ad or .h5mu; it reads the RNA from X unless --adt or --atac reads X. Given a canonical rna.h5, 0.3.8 said the file "has no
dataset 'matrix/data'" although it has one. The message now says the file is
already canonical and names the option that reads it.

It gives a copy command only when that command is certainly the whole
export: one matrix as it is (selector X), no labels, no batch split, a file
name that does not depend on an ATAC kind left out, and an empty folder (the
export orders new rows by the files already there; a copy does not). Every
other case ends with "Pass the .h5ad or .h5mu it came from." The command is
a cp, run as printed; the folder then holds the file the same export from an
.h5ad writes, and mtb.scan does not name it as missing.
"""
import shlex
import subprocess

import h5py
import numpy as np
import pytest

import multibench as mtb
from multibench import cli

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


def _scan_reasons(folder, category):
    return list(mtb.scan(str(folder), category, modalities="all", verbose=False)["reason"])


def _cp(err):
    """Run the cp command the error prints, as printed."""
    argv = shlex.split(err.split("Copy it into the folder with: ")[1])
    assert argv[0] == "cp" and len(argv) == 3
    return subprocess.run(argv, capture_output=True, text=True)


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
]


@pytest.mark.parametrize("mod, flags", COPY)
def test_the_copy_command_is_the_whole_export(tmp_path, capsys, monkeypatch, mod, flags):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in dir").mkdir()                       # a path that needs quoting
    _inputs(tmp_path / "in dir")
    src = str(tmp_path / "in dir" / f"c_{mod}.h5")
    out = tmp_path / "copy" / "DS"
    out.mkdir(parents=True)                             # an empty folder
    err = _error(capsys, ["export", src, str(out)] + flags)
    head = f"error: {src} is already a canonical .h5 file, and {flags[0]} reads an .h5ad or .h5mu. "
    assert err.startswith(head + "Copy it into the folder with: cp ")
    assert not any(out.iterdir())
    # the same export from the .h5ad
    ref = tmp_path / "ref" / "DS"
    rc, _, _ = _run(capsys, ["export", str(tmp_path / "in dir" / f"{mod}.h5ad"), str(ref)]
                    + flags)
    assert rc == 0
    files = sorted(p.name for p in ref.iterdir())
    # the command, as printed
    assert _cp(err).returncode == 0
    assert sorted(p.name for p in out.iterdir()) == files
    reasons = _scan_reasons(out, _category(flags))
    assert reasons and not _missing(reasons, files[0])


def test_the_copy_command_works_when_the_folder_is_absent(tmp_path, capsys, monkeypatch):
    # the hint is given for an absent folder too, so the command must work there
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    err = _error(capsys, ["export", "c_rna.h5", "NEW/DS", "--category", "vertical"])
    assert err.endswith("Copy it into the folder with: cp c_rna.h5 NEW/DS/rna.h5")
    done = _cp(err)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "NEW" / "DS" / "rna.h5").is_file()


#: (source, flags, the option named): no copy command
SOURCE_CASES = [
    # labels: a copy would drop them
    ("rna", ["--rna", "X", "--category", "vertical", "--labels", "obs:ct"], "--rna"),
    # the RNA from X by default, with the labels
    ("rna", ["--labels", "obs:ct", "--category", "vertical"], "--rna"),
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
    err = _error(capsys, ["export", f"c_{mod}.h5", "OUT"] + flags)
    assert err == (f"error: c_{mod}.h5 is already a canonical .h5 file, and {flag} reads "
                   f"an .h5ad or .h5mu. {SOURCE}")
    assert not (tmp_path / "OUT").exists()


def test_a_folder_with_files_gets_no_copy_command(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    assert _run(capsys, ["export", "rna.h5ad", "OUT", "--rna", "X",
                         "--category", "vertical"])[0] == 0
    err = _error(capsys, ["export", "c_adt.h5", "OUT", "--adt", "X", "--category",
                          "vertical"])
    assert err.endswith(SOURCE)


def test_atac_from_alone_gets_the_copy_command(tmp_path, capsys, monkeypatch):
    # --atac X: the ATAC alone (the RNA is not read from X)
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    (tmp_path / "OUT").mkdir()
    flags = ["--atac", "X", "--atac-kind", "peak", "--category", "diagonal"]
    err = _error(capsys, ["export", "rna.h5ad", "OUT", "--atac-from", "c_atac.h5"] + flags)
    assert err == ("error: c_atac.h5 is already a canonical .h5 file, and --atac-from reads "
                   "an .h5ad or .h5mu. Copy it into the folder with: cp "
                   "c_atac.h5 OUT/atac_peak.h5")
    assert _cp(err).returncode == 0
    # the same export from an .h5ad writes this one file
    assert _run(capsys, ["export", "rna.h5ad", "REF", "--atac-from", "atac.h5ad"]
                + flags)[0] == 0
    assert sorted(p.name for p in (tmp_path / "REF").iterdir()) == ["atac_peak.h5"]
    assert sorted(p.name for p in (tmp_path / "OUT").iterdir()) == ["atac_peak.h5"]
    reasons = _scan_reasons(tmp_path / "OUT", "diagonal")
    assert reasons and not _missing(reasons, "atac_peak.h5")


@pytest.mark.parametrize("flags", [
    ["--atac-kind", "peak"],                        # the RNA from X by default
    ["--rna", "X", "--atac-kind", "peak"],          # the RNA comes from the source
    ["--atac-kind", "peak", "--labels", "obs:ct"],
    [],                                             # no kind: the file name is not known
])
def test_atac_from_with_more_asks_for_the_source(tmp_path, capsys, monkeypatch, flags):
    monkeypatch.chdir(tmp_path)
    _inputs(tmp_path)
    err = _error(capsys, ["export", "rna.h5ad", "OUT", "--atac-from", "c_atac.h5",
                          "--category", "diagonal"] + flags)
    assert err == ("error: c_atac.h5 is already a canonical .h5 file, and --atac-from reads "
                   f"an .h5ad or .h5mu. {SOURCE}")


def test_a_file_that_is_not_canonical_keeps_its_message(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f["data"] = np.ones((3, 2))
    err = _error(capsys, ["export", "x.h5", "OUT", "--rna", "X"])
    assert err.startswith("error: x.h5 has no dataset 'matrix/data'; found keys ['data']")
