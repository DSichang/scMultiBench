"""Where the round-1 work packages meet (integration of wp/f1_*).

Each work package fixed its own half; these pin the joins:

- scan's ``reason`` column (built from resolved paths, L31) carries the
  per-batch hint that ``inputs_for`` gives (L03);
- scan and find_methods refuse two ATAC representations alike (L13);
- a constant placeholder argument (Seurat_WNN passes ``NULL`` for its unused
  modality) is not a modality the variant reads, in find_methods as in scan
  (L13, L32);
- scan does not repeat Seurat_v5's paired-files requirement as a setup note:
  it checks the files and reports only a mismatch (L14 with L61's setup
  caveat).
"""
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import registry


def _h5(path, n_feat, n_cell):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((n_feat, n_cell)))
        f.create_dataset("matrix/features",
                         data=np.array([f"g{i}" for i in range(n_feat)], dtype="S"))
        f.create_dataset("matrix/barcodes",
                         data=np.array([f"c{i}" for i in range(n_cell)], dtype="S"))


def test_scan_reason_names_a_per_batch_folder(tmp_path):
    d = tmp_path / "MB"
    d.mkdir()
    for i, n in ((1, 40), (2, 50)):
        _h5(d / f"rna{i}.h5", 20, n)
        _h5(d / f"adt{i}.h5", 6, n)
        pd.DataFrame({"x": ["T"] * n}).to_csv(d / f"cty{i}.csv", index=False)
    sc = mtb.scan("MB", "vertical", modalities=["rna", "adt"], data_path=tmp_path,
                  verbose=False)
    assert len(sc) and not sc["files_ok"].any()
    for reason in sc["reason"]:
        assert reason.startswith("missing rna.h5, adt.h5")
        assert "this folder holds per-batch files (rna1.h5, rna2.h5, ...)" in reason
        assert "or use category='cross' (RNA+ADT)" in reason


def test_scan_reason_has_no_batch_hint_on_a_plain_folder(root):
    sc = mtb.scan("D11", "vertical", modalities=["rna", "atac"],
                  data_path=root / "data", verbose=False)
    assert not sc["reason"].str.contains("per-batch").any()


def test_scan_refuses_two_atac_representations_like_find_methods(root):
    mods = ["rna", "atac_peak", "atac_gas"]
    with pytest.raises(ValueError, match="names two ATAC representations"):
        mtb.find_methods("diagonal", modalities=mods)
    with pytest.raises(ValueError, match="names two ATAC representations"):
        mtb.scan("D28", "diagonal", modalities=mods, data_path=root / "data",
                 verbose=False)
    with pytest.raises(ValueError, match="names two ATAC representations"):
        mtb.run_all("D28", "diagonal", modalities=mods, data_path=root / "data",
                    dry_run=True)


def test_scan_and_find_methods_agree_per_representation(root):
    for mods, atac in ((["rna", "atac_peak"], "peak"), (["rna", "atac_gas"], "gene_activity")):
        sc = mtb.scan("D28", "diagonal", modalities=mods, data_path=root / "data",
                      verbose=False)
        # scBridge is fed a folder: no modality token selects it in scan
        want = set(mtb.find_methods("diagonal", atac=atac)) - {"scBridge"}
        assert set(sc["method"]) == want, mods


def test_a_constant_placeholder_is_not_a_modality_the_variant_reads():
    wnn = registry.get("Seurat_WNN")
    by_mods = {tuple(v.when["modalities"]): v for v in wnn.variants}
    assert by_mods[("rna", "adt")].modality_types == {"rna", "adt"}
    assert by_mods[("rna", "atac")].modality_types == {"rna", "atac"}
    assert not by_mods[("rna", "adt")].consumes_atac
    # the rna+adt variant reads no ATAC, so it cannot satisfy atac='peak'
    assert "Seurat_WNN" not in mtb.find_methods("vertical", modalities=["rna", "adt"],
                                                atac="peak")
    assert "Seurat_WNN" in mtb.find_methods("vertical", modalities=["rna", "atac"],
                                            atac="peak")
    assert "Seurat_WNN" in mtb.find_methods("vertical", modalities=["rna", "adt"])
    # scBridge's constant file names still count: it reads rna.h5 and atac_gas.h5
    assert registry.get("scBridge").modality_types == {"rna", "atac"}


def test_scan_has_no_setup_note_for_the_checked_bridge_requirement(root):
    sc = mtb.scan("D28", "diagonal", methods=["Seurat_v5"], data_path=root / "data",
                  verbose=False)
    cav = sc.iloc[0]["caveat"]
    assert "setup:" not in cav
    assert "which need the same cells; these files hold different cells" in cav
    assert mtb.method_info("Seurat_v5")["setup_hint"]           # still in method_info


def test_run_off_linux_names_the_cli_flag_under_the_cli(tmp_path, monkeypatch):
    """The off-Linux refusal names --dry-run when the command line runs it."""
    from multibench import config
    from multibench.engine import envs, runner
    monkeypatch.setattr(envs, "host_platform_problem",
                        lambda: "method environments are linux-64 conda envs (packed "
                                "archives + lockfiles); this host is darwin/arm64")
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=mtb.inputs_for("D11", "vertical", "totalVI"),
                out_dir=str(tmp_path / "out"))
    assert "Preview the command with --dry-run and run it on a Linux machine." in str(e.value)
    assert "`multibench env doctor`" in str(e.value)


def test_vertical_tutorial_export_demo_writes_counts(root, tmp_path, monkeypatch):
    """The own-data cell of the vertical tutorial exports raw counts, so
    export_dataset's raw-count warning and scan's caveat stay silent (L01)."""
    import json
    import tempfile
    import warnings
    nb = json.loads((root / "notebooks" / "tutorial_vertical.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "export_dataset(demo" in "".join(c["source"]))
    monkeypatch.setattr(tempfile, "mkdtemp", lambda: str(tmp_path))
    ns = {"mtb": mtb, "CATEGORY": "vertical", "pd": pd}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        exec(compile(cell, "tutorial_vertical", "exec"), ns)
    assert not [w for w in seen if "whole numbers" in str(w.message)]
    assert not ns["sc"]["caveat"].str.contains("non-integer").any()


def test_scan_checks_label_rows_for_a_folder_fed_method(tmp_path):
    """scBridge reads rna.h5 / atac_gas.h5 and their label files from the folder:
    a label file with the wrong number of rows fails its row as it fails the
    file-role methods (L05 with the data_dir variant)."""
    d = tmp_path / "SB"
    d.mkdir()
    _h5(d / "rna.h5", 20, 50)
    _h5(d / "atac_gas.h5", 20, 40)
    pd.DataFrame({"x": ["T"] * 50}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["T"] * 10}).to_csv(d / "atac_cty.csv", index=False)
    sc = mtb.scan("SB", "diagonal", methods=["scBridge"], data_path=tmp_path,
                  verbose=False)
    row = sc.iloc[0]
    assert not row["files_ok"]
    assert "atac_cty.csv has 10 labels but atac_gas.h5 has 40 cells" in row["files_reason"]
    pd.DataFrame({"x": ["T"] * 40}).to_csv(d / "atac_cty.csv", index=False)
    sc = mtb.scan("SB", "diagonal", methods=["scBridge"], data_path=tmp_path,
                  verbose=False)
    assert sc.iloc[0]["files_ok"], sc.iloc[0]["files_reason"]
