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


# --- P12: labels_for mirrors inputs_for; positional data_path shim -------------

def _make_labels_ds(tmp_path):
    d = tmp_path / "D27"
    d.mkdir(parents=True)
    for n in ["rna.h5", "rna_cty.csv", "peak_cty.csv"]:
        (d / n).write_text("")
    return tmp_path


def test_labels_for_accepts_method_and_category_positionally(tmp_path):
    data = _make_labels_ds(tmp_path)
    a = resolve.labels_for("D27", "Seurat_v5", "diagonal", data_path=data)
    b = resolve.labels_for("D27", data_path=data)
    assert a == b and set(a) == {"rna_cty", "peak_cty"}


def test_labels_for_positional_data_path_shim_warns(tmp_path):
    import warnings
    data = _make_labels_ds(tmp_path)
    with pytest.warns(DeprecationWarning, match="data_path= by keyword"):
        a = resolve.labels_for("D27", data)
    with pytest.warns(DeprecationWarning):
        b = resolve.labels_for("D27", str(data))
    assert a == b == resolve.labels_for("D27", data_path=data)
    # a bare method id (no separator, not a dir) is NOT mistaken for a path
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FileNotFoundError):
            resolve.labels_for("D27", "Seurat_v5", data_path=tmp_path / "nowhere")


# --- P09: inputs_for validates tokens and accepts aliases ---------------------

def test_inputs_for_protein_alias_and_bad_category(tmp_path):
    d = tmp_path / "D11"; d.mkdir()
    for n in ["rna.h5", "adt.h5", "cty.csv"]:
        (d / n).write_text("")
    a = resolve.inputs_for("D11", "totalVI", "vertical", modalities=["rna", "protein"],
                           data_path=tmp_path)
    b = resolve.inputs_for("D11", "totalVI", "vertical", modalities=["rna", "adt"],
                           data_path=tmp_path)
    assert a == b and set(a) == {"rna", "adt"}
    with pytest.raises(ValueError, match="unknown category 'crosss'"):
        resolve.inputs_for("D11", "totalVI", "crosss", data_path=tmp_path)
    with pytest.raises(ValueError, match="unknown modality 'proteinx'"):
        resolve.inputs_for("D11", "totalVI", "vertical", modalities=["rna", "proteinx"],
                           data_path=tmp_path)
    with pytest.raises(KeyError, match="did you mean 'StabMap'"):
        resolve.inputs_for("D11", "Stabmap", "cross", data_path=tmp_path)
