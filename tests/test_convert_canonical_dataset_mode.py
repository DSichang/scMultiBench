"""convert in dataset mode says what to do with a canonical .h5.

Dataset mode (--rna, --adt, --atac, --labels, ...) reads an .h5ad or .h5mu.
Given a canonical rna.h5, 0.3.8 said the file "has no dataset 'matrix/data'"
although it has one. The message now says the file is already canonical and
gives the file-mode command that copies it into the folder, under the name
the dataset mode would have written; that command works.
"""
import h5py
import numpy as np
import pytest

from multibench import cli
from multibench.engine import ingest


def _canonical(path, n_feat=4, n_cells=6):
    with h5py.File(path, "w") as f:
        f["matrix/data"] = np.ones((n_feat, n_cells))
        f["matrix/features"] = np.array([f"G{i}" for i in range(n_feat)], dtype="S")
        f["matrix/barcodes"] = np.array([f"c{i}" for i in range(n_cells)], dtype="S")
    return path


def _error(capsys, argv):
    assert cli.main(argv) == 1
    return capsys.readouterr().err.splitlines()[0]


@pytest.mark.parametrize("flags, name", [
    (["--rna", "X"], "rna.h5"),
    (["--adt", "X", "--category", "vertical"], "adt.h5"),
    (["--atac", "X", "--atac-kind", "peak"], "atac_peak.h5"),
    (["--atac", "X", "--atac-kind", "peak", "--category", "vertical"], "atac.h5"),
    (["--rna", "X", "--category", "mosaic", "--batch-index", "2"], "rna2.h5"),
])
def test_the_message_gives_a_command_that_works(tmp_path, capsys, monkeypatch, flags, name):
    monkeypatch.chdir(tmp_path)
    _canonical(tmp_path / "rna.h5")
    err = _error(capsys, ["convert", "rna.h5", "OUT"] + flags)
    flag = flags[0]
    assert err == (f"error: rna.h5 is already a canonical .h5 file, and {flag} reads an "
                   f".h5ad or .h5mu. Copy it into the folder with: multibench convert "
                   f"rna.h5 OUT/{name}")
    assert not (tmp_path / "OUT").exists()
    assert cli.main(["convert", "rna.h5", f"OUT/{name}"]) == 0
    assert ingest._is_canonical_h5(tmp_path / "OUT" / name)


def test_labels_alone(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _canonical(tmp_path / "rna.h5")
    err = _error(capsys, ["convert", "rna.h5", "OUT", "--labels", "obs:ct"])
    assert err == ("error: rna.h5 is already a canonical .h5 file, and --labels reads an "
                   ".h5ad or .h5mu. Pass the .h5ad or .h5mu it came from.")


def test_atac_from(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _canonical(tmp_path / "atac.h5")
    import anndata as ad
    ad.AnnData(np.ones((6, 4))).write_h5ad(tmp_path / "rna.h5ad")
    err = _error(capsys, ["convert", "rna.h5ad", "OUT", "--rna", "X", "--atac-from", "atac.h5",
                          "--atac-kind", "peak", "--category", "diagonal"])
    assert err == ("error: atac.h5 is already a canonical .h5 file, and --atac-from reads an "
                   ".h5ad or .h5mu. Copy it into the folder with: multibench convert "
                   "atac.h5 OUT/atac_peak.h5")


def test_a_file_that_is_not_canonical_keeps_its_message(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f["data"] = np.ones((3, 2))
    err = _error(capsys, ["convert", "x.h5", "OUT", "--rna", "X"])
    assert err.startswith("error: x.h5 has no dataset 'matrix/data'; found keys ['data']")
