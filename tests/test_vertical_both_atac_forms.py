"""A vertical folder with atac_peak.h5 and atac_gas.h5 serves every RNA + ATAC method.

With one atac.h5 per folder, the methods that read peaks and those that read
gene activity needed two datasets for the same task. Each method now reads
the form method_info(m)["atac"] names.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

import multibench as mtb


def _h5(path, names):
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=np.ones((len(names), 6)))
        g.create_dataset("features", data=np.array(names, dtype="S"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(6)], dtype="S"))


@pytest.fixture
def both(tmp_path):
    d = tmp_path / "BOTH"
    d.mkdir()
    _h5(d / "rna.h5", ["G1", "G2", "G3"])
    _h5(d / "atac_peak.h5", ["chr1:1-100", "chr1:200-300"])
    _h5(d / "atac_gas.h5", ["G1", "G2"])
    (d / "cty.csv").write_text("x\n" + "a\n" * 6)
    return tmp_path


@pytest.mark.parametrize("method, mods", [
    ("MIRA", ["rna", "atac"]), ("scMVP", ["rna", "atac"]), ("Matilda", ["rna", "atac"]),
    ("UnitedNet", ["rna", "atac_gas"]), ("scMDC", ["rna", "atac_gas"]),
    ("iPOLNG", ["rna", "atac_gas"]), ("moETM", ["rna", "atac_gas"]), ("scMM", ["rna", "atac_gas"]),
])
def test_each_method_reads_the_form_it_needs(both, method, mods):
    got = mtb.inputs_for("BOTH", "vertical", method, modalities=mods, data_path=both)
    atac = next(Path(v).name for k, v in got.items() if "atac" in k)
    want = {"peak": "atac_peak.h5", "gene_activity": "atac_gas.h5"}[mtb.method_info(method)["atac"]]
    assert atac == want


def test_a_folder_with_one_atac_file_is_read_as_before(tmp_path):
    d = tmp_path / "ONE"
    d.mkdir()
    _h5(d / "rna.h5", ["G1", "G2", "G3"])
    _h5(d / "atac.h5", ["chr1:1-100", "chr1:200-300"])
    (d / "cty.csv").write_text("x\n" + "a\n" * 6)
    got = mtb.inputs_for("ONE", "vertical", "MIRA", modalities=["rna", "atac"], data_path=tmp_path)
    assert Path(got["atac"]).name == "atac.h5"


def test_a_diagonal_folder_is_not_read_as_vertical(both):
    """atac_cty.csv marks unpaired ATAC cells: no vertical method takes them."""
    (both / "BOTH" / "atac_cty.csv").write_text("x\n" + "a\n" * 6)
    sc = mtb.scan("BOTH", "vertical", data_path=both, methods=["MIRA"], verbose=False)
    assert not sc.files_ok.any()
