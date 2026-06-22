import pytest

from multibench.run import resolve


def _make_data(tmp_path):
    d = tmp_path / "diagonal integration" / "D27"
    d.mkdir(parents=True)
    for name in ["rna.h5", "atac_gas.h5", "atac_peak.h5"]:
        (d / name).write_text("")
    return tmp_path


def test_inputs_for_scalex_uses_gene_activity(tmp_path):
    data = _make_data(tmp_path)
    inputs = resolve.inputs_for("D27", "SCALEX", category="diagonal", data_path=data)
    assert set(inputs) == {"rna", "atac_gas"}
    assert inputs["atac_gas"].endswith("atac_gas.h5")


def test_inputs_for_seurat_uses_peak(tmp_path):
    data = _make_data(tmp_path)
    inputs = resolve.inputs_for("D27", "Seurat_v5", category="diagonal", data_path=data)
    assert "atac_peak" in inputs
    assert inputs["atac_peak"].endswith("atac_peak.h5")


def _make_mosaic_data(tmp_path):
    d = tmp_path / "mosaic integration" / "D1"
    d.mkdir(parents=True)
    for name in ["rna1.h5", "rna2.h5", "atac2.h5", "atac3.h5"]:
        (d / name).write_text("")
    return tmp_path


def test_inputs_for_ambiguous_mosaic_raises(tmp_path):
    # Multigrate has two mosaic variants (rna_adt, rna_atac); no modalities -> error.
    with pytest.raises(ValueError):
        resolve.inputs_for("D1", "Multigrate", category="mosaic")


def test_inputs_for_mosaic_with_modalities_selects_variant(tmp_path):
    data = _make_mosaic_data(tmp_path)
    inputs = resolve.inputs_for("D1", "Multigrate", category="mosaic",
                                modalities=["rna1", "rna2", "atac2", "atac3"],
                                data_path=data)
    assert set(inputs) == {"rna1", "rna2", "atac2", "atac3"}
