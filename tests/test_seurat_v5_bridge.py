"""Seurat_v5: ATAC cells first, and its bridge needs paired files (study round 1, L14).

main_Seurat_v5.Rmd builds its bridge object from rna.h5 plus atac_peak.h5
(lines 39-44), which Seurat accepts only for the same cells, and writes
rbind(ATAC query, RNA reference) (lines 84-86, 109). Before the fix
labels_for returned rna_cty first and scan marked an unpaired folder
file-ready with no caveat. Source-level evidence: a host run on D27 has not
confirmed the order yet (WORK_DIARY).
"""
import h5py
import numpy as np
import pandas as pd

import multibench as mtb
from multibench.engine import resolve

PEAKS = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(40)]


def _h5(path, feats, barcodes):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(1.0, size=(len(feats), len(barcodes))).astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S24"))
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S12"))


def _folder(root, name, rna_cells, atac_cells):
    d = root / name
    d.mkdir()
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], rna_cells)
    _h5(d / "atac_peak.h5", PEAKS, atac_cells)
    pd.DataFrame({"x": ["A"] * len(rna_cells)}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["A"] * len(atac_cells)}).to_csv(d / "atac_cty.csv", index=False)


def _caveat(tmp_path, name):
    df = mtb.scan(name, "diagonal", methods=["Seurat_v5"], data_path=tmp_path, verbose=False)
    row = df[df.method == "Seurat_v5"].iloc[0]
    assert row["files_ok"], row["files_reason"]
    return row["caveat"]


def test_labels_for_puts_the_atac_cells_first(root):
    assert list(mtb.labels_for("D28", "diagonal", "Seurat_v5", data_path=root / "data")) == \
        ["atac_cty", "rna_cty"]


def test_scan_flags_unpaired_bridge_files(tmp_path):
    cells = [f"c{i}" for i in range(50)]
    _folder(tmp_path, "UNPAIRED", cells, [f"a{i}" for i in range(45)])
    cav = _caveat(tmp_path, "UNPAIRED")
    assert ("Seurat_v5 builds its bridge from rna.h5 and atac_peak.h5, which need the "
            "same cells; these files hold different cells (50 and 45 cells, 0 shared)") in cav
    _folder(tmp_path, "PAIRED", cells, list(reversed(cells)))     # same cells, any order
    assert "bridge" not in _caveat(tmp_path, "PAIRED")


def test_the_caveat_is_only_for_methods_that_need_paired_files(tmp_path):
    _folder(tmp_path, "UNP", [f"c{i}" for i in range(50)], [f"a{i}" for i in range(45)])
    got = resolve.inputs_for("UNP", "diagonal", "GLUE", data_path=tmp_path)
    assert resolve._same_cells_caveat("GLUE", got) == []
    assert resolve._preflight_caveats(got, atac="peak") == []


def test_setup_hint_states_the_requirement():
    hint = mtb.method_info("Seurat_v5")["setup_hint"]
    assert "rna.h5 and atac_peak.h5" in hint and "same cells" in hint
