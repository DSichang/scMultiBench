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


# --- P01 follow-up: content preflight in inputs_for(check=True) ---------------

def _h5(path, n_feat, n_cells, feats=None):
    import h5py
    import numpy as np
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=np.ones((n_feat, n_cells)))
        g.create_dataset("features", data=np.array(
            feats or [f"g{i}" for i in range(n_feat)], dtype="S24"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(n_cells)], dtype="S8"))


def test_label_length_mismatch_reported(tmp_path):
    d = tmp_path / "D11"; d.mkdir()
    _h5(d / "rna.h5", 30, 60); _h5(d / "adt.h5", 5, 60)
    (d / "cty.csv").write_text("x\n" + "A\n" * 59)
    with pytest.raises(ValueError) as e:
        resolve.inputs_for("D11", "Matilda", "vertical", modalities=["rna", "adt"],
                           data_path=tmp_path, check=True)
    msg = str(e.value)
    assert "cty.csv has 59 labels" in msg and "60 cells" in msg and "describe_layout" in msg
    # check=None / False do not raise (best-effort paths)
    resolve.inputs_for("D11", "Matilda", "vertical", modalities=["rna", "adt"],
                       data_path=tmp_path, check=False)
    (d / "cty.csv").write_text("x\n" + "A\n" * 60)
    resolve.inputs_for("D11", "Matilda", "vertical", modalities=["rna", "adt"],
                       data_path=tmp_path, check=True)     # now fine


def test_label_partners_pairing_rule():
    lp = resolve._label_partners
    assert lp("cty", ["rna", "adt", "cty"]) == ["rna", "adt"]
    assert lp("rna_cty", ["rna", "atac_gas", "atac_peak", "rna_cty", "atac_cty"]) == ["rna"]
    assert lp("atac_cty", ["rna", "atac_gas", "atac_peak", "rna_cty", "atac_cty"]) == ["atac_gas", "atac_peak"]
    assert lp("peak_cty", ["rna", "atac_peak"]) == ["atac_peak"]
    assert lp("cty2", ["rna1", "rna2", "adt1", "atac2", "cty1", "cty2"]) == ["rna2", "atac2"]
    assert lp("source_cty", ["data_dir", "source_data"]) == []


def test_atac_gas_peak_caveat(tmp_path):
    d = tmp_path / "PK"; d.mkdir()
    _h5(d / "rna.h5", 30, 50)
    _h5(d / "atac.h5", 40, 45, feats=[f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(40)])
    got = resolve.inputs_for("PK", "Portal", "diagonal", data_path=tmp_path, check=True)
    assert got["atac_gas"].endswith("atac.h5")
    assert resolve._preflight_caveats(got) == [resolve.PEAK_IN_GAS_CAVEAT]
    assert "chr:start-end" in resolve.PEAK_IN_GAS_CAVEAT
    # a real atac_gas.h5 (gene names) -> no caveat; so does gene-named atac.h5
    _h5(d / "atac_gas.h5", 40, 45)
    assert resolve._preflight_caveats(resolve.inputs_for("PK", "Portal", "diagonal",
                                                         data_path=tmp_path)) == []
    (d / "atac_gas.h5").unlink()
    _h5(d / "atac.h5", 40, 45)
    assert resolve._preflight_caveats(resolve.inputs_for("PK", "Portal", "diagonal",
                                                         data_path=tmp_path)) == []


def test_inputs_for_check_true_rejects_data_dir_without_slices(tmp_path):
    (tmp_path / "D11").mkdir()
    with pytest.raises(FileNotFoundError, match=">=2 .h5ad"):
        resolve.inputs_for("D11", "PASTE", "cross", data_path=tmp_path, check=True)
    # default (warn-only) still hands back the directory
    got = resolve.inputs_for("D11", "PASTE", "cross", data_path=tmp_path)
    assert got["data_dir"].rstrip("/").endswith("D11")


def test_check_data_dir_wants_obsm_spatial(tmp_path):
    import anndata as ad
    import numpy as np
    from multibench.engine import registry
    v = registry.get("PASTE").select("cross", set())
    d = tmp_path / "S"; d.mkdir()
    assert resolve._check_data_dir(v, d) == (False, f"spatial registration needs >=2 .h5ad slice files; found 0 in {d}")
    for i in range(2):
        ad.AnnData(np.ones((5, 3))).write_h5ad(d / f"s{i}.h5ad")
    ok, why = resolve._check_data_dir(v, d)
    assert not ok and "obsm['spatial']" in why
    for i in range(2):
        a = ad.AnnData(np.ones((5, 3))); a.obsm["spatial"] = np.zeros((5, 2))
        a.write_h5ad(d / f"s{i}.h5ad")
    assert resolve._check_data_dir(v, d) == (True, "")
    # the workflow alias delegates here
    from multibench import workflow as W
    assert W._data_dir_usable(v, d) == (True, "")


def test_labels_for_docstring_says_how_to_hand_to_evaluate():
    import inspect
    doc = inspect.getdoc(resolve.labels_for)
    assert "list(labels_for(ds).values())" in doc and "evaluate(labels=" in doc
