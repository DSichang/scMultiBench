import pytest

from multibench.engine import resolve


def _make_diag(tmp_path, files):
    """Real diagonal layout is FLAT: data/<dataset>/<file>."""
    d = tmp_path / "D27"
    d.mkdir(parents=True)
    for name in files:
        (d / name).write_text("")
    return tmp_path


def test_inputs_for_flat_layout_and_peak_alias(tmp_path):
    # Seurat_v5 (diagonal) uses roles rna + atac_peak; real file is peak.h5
    data = _make_diag(tmp_path, ["rna.h5", "peak.h5", "rna_cty.csv", "peak_cty.csv"])
    inputs = resolve.inputs_for("D27", "Seurat_v5", category="diagonal", data_path=data)
    assert set(inputs) == {"rna", "atac_peak"}
    assert inputs["rna"].endswith("/D27/rna.h5")        # flat: no category folder
    assert inputs["atac_peak"].endswith("/D27/peak.h5")  # alias atac_peak -> peak.h5


def test_inputs_for_gene_activity_alias(tmp_path):
    # SCALEX (diagonal) uses atac_gas; real gene-activity file is atac.h5
    data = _make_diag(tmp_path, ["rna.h5", "atac.h5"])
    inputs = resolve.inputs_for("D27", "SCALEX", category="diagonal", data_path=data)
    assert set(inputs) == {"rna", "atac_gas"}
    assert inputs["atac_gas"].endswith("/D27/atac.h5")   # alias atac_gas -> atac.h5


def test_inputs_for_falls_back_to_canonical_name(tmp_path):
    # when no candidate file exists, fall back to <role>.h5 (best effort)
    data = _make_diag(tmp_path, ["rna.h5"])
    inputs = resolve.inputs_for("D27", "SCALEX", category="diagonal", data_path=data)
    assert inputs["atac_gas"].endswith("/D27/atac_gas.h5")


def test_inputs_for_ambiguous_mosaic_raises(tmp_path):
    # Multigrate has multiple mosaic variants; no modalities -> ValueError.
    with pytest.raises(ValueError):
        resolve.inputs_for("D1", "Multigrate", category="mosaic")


def test_inputs_for_mosaic_with_modalities_selects_variant(tmp_path):
    d = tmp_path / "D1"
    d.mkdir(parents=True)
    for n in ["rna1.h5", "rna2.h5", "atac2.h5", "atac3.h5"]:
        (d / n).write_text("")
    inputs = resolve.inputs_for("D1", "Multigrate", category="mosaic",
                                modalities=["rna1", "rna2", "atac2", "atac3"],
                                data_path=tmp_path)
    assert set(inputs) == {"rna1", "rna2", "atac2", "atac3"}


def test_labels_for_collects_cty_csvs(tmp_path):
    d = tmp_path / "D27"
    d.mkdir(parents=True)
    for n in ["rna.h5", "rna_cty.csv", "peak_cty.csv", "peak_cty_scjoint.csv"]:
        (d / n).write_text("")
    labels = resolve.labels_for("D27", data_path=tmp_path)
    assert set(labels) == {"rna_cty", "peak_cty"}  # scjoint reformat excluded
    assert labels["rna_cty"].endswith("/D27/rna_cty.csv")


def test_labels_for_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve.labels_for("NOPE", data_path=tmp_path)
