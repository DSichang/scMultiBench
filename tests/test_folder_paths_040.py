"""Dataset arguments given as folder paths (0.4): a bare name that is only a
folder of the working directory gets the path to pass."""
import os
import shutil

import pytest

import multibench as mtb


@pytest.fixture
def here(tmp_path, monkeypatch):
    """A working directory with data/MYCITE, a copy of D11 (RNA + ADT, no ATAC)."""
    shutil.copytree(mtb.config.data_path / "D11", tmp_path / "data" / "MYCITE")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_registered_data_path_folder_wins_over_one_here(here, monkeypatch):
    """D11 under the data path and ./D11: the documented rule, data_path first."""
    shutil.copytree("data/MYCITE", "D11")
    got = mtb.engine.resolve.inputs_for("D11", "vertical", "Matilda")
    assert got["rna"] == str(mtb.config.data_path / "D11" / "rna.h5")


def test_a_bare_name_names_a_folder_of_the_data_path(here):
    os.rename('data/MYCITE', 'MYCITE')
    with pytest.raises(FileNotFoundError) as e:
        mtb.scan('MYCITE', 'vertical', verbose=False)
    msg = str(e.value)
    assert msg.startswith(f"The folder {mtb.config.data_path / 'MYCITE'} does not exist. ")
    assert msg.endswith("dataset= is the folder's path, or its name under mtb.config.data_path.")
    assert len(mtb.scan('./MYCITE', 'vertical', verbose=False))
