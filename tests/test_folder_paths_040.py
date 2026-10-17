"""Dataset arguments given as folder paths (0.4): the command line resolves them
as Python does, and a bare name that is only a folder of the working directory
gets the path to pass."""
import io
import contextlib
import os
import shutil

import pytest

import multibench as mtb
from multibench import cli


@pytest.fixture
def here(tmp_path, monkeypatch):
    """A working directory with data/MYCITE, a copy of D11 (RNA + ADT, no ATAC)."""
    shutil.copytree(mtb.config.DEFAULT.data_path / "D11", tmp_path / "data" / "MYCITE")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("path", ["data/MYCITE", "./data/MYCITE"])
def test_cli_scan_takes_a_folder_path(here, path):
    code, out, err = _cli("scan", path, "--category", "vertical", "--format", "csv")
    assert code == 0, err
    assert "Matilda" in out
    # the note on the rows that read ATAC names the folder, as the Python call does
    assert "# MYCITE holds RNA and ADT." in err and "error" not in err


def test_cli_run_all_dry_run_takes_a_folder_path(here):
    code, out, err = _cli("run-all", "data/MYCITE", "--category", "vertical",
                          "--methods", "Matilda", "--out-dir", "runs", "--dry-run")
    assert code == 0, err
    assert "Matilda" in out and "error" not in err


def test_a_bare_name_that_is_a_folder_here_names_the_path(here):
    os.rename("data/MYCITE", "MYCITE")
    with pytest.raises(FileNotFoundError) as e:
        mtb.scan("MYCITE", "vertical", verbose=False)
    assert str(e.value) == (f"The folder {mtb.config.DEFAULT.data_path / 'MYCITE'} does not "
                            f"exist. ./MYCITE exists here: pass './MYCITE'.")
    code, _, err = _cli("scan", "MYCITE", "--category", "vertical")
    assert code == 1
    assert "./MYCITE exists here: pass ./MYCITE." in err
    # the path works
    assert len(mtb.scan("./MYCITE", "vertical", verbose=False))


def test_a_registered_data_path_folder_wins_over_one_here(here, monkeypatch):
    """D11 under the data path and ./D11: the documented rule, data_path first."""
    shutil.copytree("data/MYCITE", "D11")
    got = mtb.inputs_for("D11", "vertical", "Matilda")
    assert got["rna"] == str(mtb.config.DEFAULT.data_path / "D11" / "rna.h5")
