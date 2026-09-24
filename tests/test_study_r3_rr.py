"""Round-3 student study, package work 'rr': R3-04."""
import inspect
import re
import shutil
import subprocess
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry, runner

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


# ============================================================ R3-04 GLUE peak names
def _h5(path, feats, bars):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((len(feats), len(bars))))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))


def _diagonal(root, name, peaks):
    """A diagonal folder whose atac_peak.h5 holds ``peaks``."""
    d = root / name
    d.mkdir()
    rna, atac = [f"r{i}" for i in range(20)], [f"a{i}" for i in range(20)]
    genes = [f"G{i}" for i in range(30)]
    _h5(d / "rna.h5", genes, rna)
    _h5(d / "atac_peak.h5", peaks, atac)
    _h5(d / "atac_gas.h5", genes, atac)
    pd.DataFrame({"x": ["A"] * 20}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["A"] * 20}).to_csv(d / "atac_cty.csv", index=False)
    return d


def _scan_row(root, name, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan(name, "diagonal", methods=[method], data_path=root, verbose=False)
    return sc.iloc[0]


@pytest.mark.parametrize("method", ["GLUE", "Seurat_v3"])
def test_underscore_peaks_are_passed_as_a_rewritten_copy(tmp_path, method):
    _diagonal(tmp_path, "LUNG_us", [f"chr4_{171325400 + 10 * i}_{171325903 + 10 * i}"
                                    for i in range(60)])
    inp = mtb.inputs_for("LUNG_us", "diagonal", method, data_path=tmp_path)
    argv = mtb.run(method, "diagonal", inputs=inp, out_dir=str(tmp_path / "o"),
                   dry_run=True)
    assert str(tmp_path / "o" / "inputs" / "atac_peak_normpeaks.h5") in argv
    assert inp["atac_peak"] not in argv
    row = _scan_row(tmp_path, "LUNG_us", method)
    assert row["files_ok"]
    assert "peak names" not in row["caveat"]
    assert "inputs/atac_peak_normpeaks.h5" in row["caveat"]      # the prepared-file note


def test_dash_peaks_need_no_caveat(tmp_path):
    _diagonal(tmp_path, "LUNG_dash", [f"chr1-{100 + i}-{200 + i}" for i in range(60)])
    assert "peak names" not in _scan_row(tmp_path, "LUNG_dash", "GLUE")["caveat"]


def test_peak_ids_without_coordinates_get_the_peak_names_caveat(tmp_path):
    _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(1, 61)])
    cav = _scan_row(tmp_path, "LUNG_ids", "GLUE")["caveat"]
    assert cav.startswith("GLUE reads peak names such as chr1:100-200; atac_peak.h5 "
                          "holds other names (e.g. peak_1)")
    # one caveat for the file, not also the representation guess
    assert "holds gene activity" not in cav
    # the same content check for the other method whose peaks mtb.run renames
    cav = _scan_row(tmp_path, "LUNG_ids", "Seurat_v3")["caveat"]
    assert "Seurat_v3 reads peak names such as chr1:100-200" in cav


def test_d28_glue_caveat_does_not_name_the_dataset():
    r = mtb.scan("D28", "diagonal", methods=["GLUE"], verbose=False).iloc[0]
    assert "D28" not in r["caveat"] and "peak names" not in r["caveat"]


def test_normalize_peak_names_notes_name_the_methods_run_renames_for():
    """The Notes list exactly the methods with a renamed peak role."""
    renamed = sorted(m for m in registry.list_methods()
                     if any(v.normalize_peaks for v in registry.get(m).variants))
    assert renamed == ["GLUE", "Seurat_v3"]
    doc = " ".join(inspect.getdoc(ingest.normalize_peak_names).split())
    para = doc.split("**Inside ``mtb.run``.**", 1)[1].split("**", 1)[0]
    assert "applies this itself for GLUE and Seurat_v3," in para
