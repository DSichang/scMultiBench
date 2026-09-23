"""Folder resolution fixes from student study round 1 (ledger L03, L06, L55).

A per-batch folder (rna1..3.h5) scanned as vertical resolved batch 1 only
(L03); the mosaic near-miss hint named ``atac22.h5`` and the ``atac<i>`` role
did not read ``atac_peak<i>.h5`` (L06); StabMap's reference batch was not
documented or exposed (L55).
"""
import inspect
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import resolve


def _h5(path, n_feat, n_cell, feats=None):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((n_feat, n_cell)))
        f.create_dataset("matrix/features",
                         data=np.array(feats or [f"g{i}" for i in range(n_feat)], dtype="S"))
        f.create_dataset("matrix/barcodes",
                         data=np.array([f"c{i}" for i in range(n_cell)], dtype="S"))


def _cty(path, n):
    pd.DataFrame({"x": ["T"] * n}).to_csv(path, index=False)


def _per_batch_folder(d):
    d.mkdir()
    for i, n in ((1, 40), (2, 50), (3, 60)):
        _h5(d / f"rna{i}.h5", 20, n)
        _h5(d / f"adt{i}.h5", 6, n)
        _cty(d / f"cty{i}.csv", n)
    return d


# ------------------------------------------------------------------ L03
def test_per_batch_folder_fails_every_vertical_row_with_the_hint(tmp_path):
    _per_batch_folder(tmp_path / "MB")
    sc = mtb.scan("MB", "vertical", data_path=tmp_path)
    assert len(sc) > 20 and not sc["files_ok"].any()
    assert sc["files_reason"].str.contains(
        r"this folder holds per-batch files \(rna1.h5, rna2.h5, \.\.\.\); vertical methods "
        r"read one rna.h5: export without batch=").all()
    with pytest.raises(FileNotFoundError, match=r"or use category='cross' \(RNA\+ADT\)"):
        resolve.inputs_for("MB", "vertical", "totalVI", data_path=tmp_path, check=True)
    # the same folder is a valid cross folder
    got = resolve.inputs_for("MB", "cross", "totalVI", data_path=tmp_path, check=True)
    assert got["rna3"].endswith("rna3.h5")


def test_labels_for_vertical_on_a_per_batch_folder_warns_or_raises(tmp_path):
    _per_batch_folder(tmp_path / "MB")
    with pytest.warns(UserWarning, match="this folder holds per-batch files"):
        lab = mtb.labels_for("MB", "vertical", data_path=tmp_path)
    assert list(lab) == ["cty1", "cty2", "cty3"]
    with pytest.raises(ValueError, match="export without batch="):
        mtb.labels_for("MB", "vertical", data_path=tmp_path, check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mtb.labels_for("MB", "vertical", data_path=tmp_path, check=False)
        mtb.labels_for("MB", "cross", "StabMap", data_path=tmp_path)
        mtb.labels_for("MB", data_path=tmp_path)


def test_a_single_numbered_file_still_resolves(tmp_path):
    d = tmp_path / "ONE"
    d.mkdir()
    _h5(d / "rna1.h5", 20, 40)
    _h5(d / "adt1.h5", 6, 40)
    _cty(d / "cty1.csv", 40)
    got = resolve.inputs_for("ONE", "vertical", "Matilda", modalities=["rna", "adt"],
                             data_path=tmp_path, check=True)
    assert got["rna"].endswith("rna1.h5") and got["cty"].endswith("cty1.csv")
    # a numbered role never tries another batch number (rna1 -> rna11.h5)
    _h5(d / "rna11.h5", 20, 40)
    assert resolve._resolve_role(d, "rna1").name == "rna1.h5"
    (d / "rna1.h5").unlink()
    assert resolve._resolve_role(d, "rna1").name == "rna1.h5"       # unresolved, not rna11


def test_demo_folders_resolve_as_before(root):
    data = root / "data"
    d11 = mtb.scan("D11", "vertical", data_path=data)
    assert d11.loc[d11["modalities"] == "rna+adt", "files_ok"].all()
    d52 = mtb.scan("D52", "cross", data_path=data)
    assert d52["files_ok"].all()
    got = resolve.inputs_for("D52", "cross", "StabMap", data_path=data, check=True)
    assert [got[f"rna{i}"].rsplit("/", 1)[-1] for i in (1, 2, 3)] == \
        ["rna1.h5", "rna2.h5", "rna3.h5"]


# ------------------------------------------------------------------ L06
def test_numbered_atac_near_miss_names_the_right_file(tmp_path):
    d = tmp_path / "M"
    d.mkdir()
    hints = resolve._near_miss_hints(d, {"atac2": str(d / "atac2.h5")}, "mosaic")
    assert hints == []
    _h5(d / "atac_gas2.h5", 30, 40)
    hints = resolve._near_miss_hints(d, {"atac2": str(d / "atac2.h5")}, "mosaic")
    assert hints == ["atac2.h5 not found; found atac_gas2.h5 - mosaic methods read atac2.h5 "
                     "or atac_peak2.h5 (every mosaic method reads peaks)"]
    assert "atac22" not in hints[0]


def test_numbered_atac_role_reads_the_031_peak_name(tmp_path):
    d = tmp_path / "LABMOS"
    d.mkdir()
    peaks = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(30)]
    for i, n in ((1, 40), (2, 50), (3, 30)):
        _h5(d / f"rna{i}.h5", 20, n)
        _cty(d / f"cty{i}.csv", n)
    _h5(d / "adt1.h5", 6, 40)
    _h5(d / "atac_peak2.h5", 30, 50, feats=peaks)
    assert resolve._resolve_role(d, "atac2").name == "atac_peak2.h5"
    sc = mtb.scan("LABMOS", "mosaic", data_path=tmp_path)
    ok = sc[sc["method"].isin(["StabMap", "scMoMaT"])]
    assert len(ok) == 2 and ok["files_ok"].all(), ok["files_reason"].tolist()
    # atac2.h5 wins when both exist
    _h5(d / "atac2.h5", 30, 50, feats=peaks)
    assert resolve._resolve_role(d, "atac2").name == "atac2.h5"


# ------------------------------------------------------------------ L55
def test_stabmap_reference_batch_is_exposed_and_documented():
    sup = {e["category"]: e["reference_batch"] for e in mtb.method_info("StabMap")["supports"]}
    assert sup == {"cross": 3, "mosaic": 1}
    assert all(e["reference_batch"] is None for e in mtb.method_info("Matilda")["supports"])
    doc = inspect.getdoc(mtb.labels_for)
    assert "StabMap uses a fixed reference batch: batch 3 in cross, batch 1 in" in doc
    assert "Number the" in doc and "donor you want as reference accordingly" in doc
    assert "methods.yaml" not in doc and "output.cell_order" not in doc
