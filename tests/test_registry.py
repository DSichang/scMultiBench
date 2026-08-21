from multibench.engine import registry


def test_all_methods_declared():
    specs = registry.load()
    assert len(specs) >= 40


def test_verified_methods_present():
    specs = registry.load()
    ids = {s.id for s in specs}
    for m in ["SCALEX", "Multigrate", "Seurat_v5", "scBridge"]:
        assert m in ids


def test_scalex_variant_args():
    spec = registry.get("SCALEX")
    assert spec.status == "verified"
    v = spec.select("diagonal", {"rna", "atac_gas"})
    flags = [a.flag for a in v.args]
    assert "--path1" in flags and "--save_path" in flags


def test_multigrate_variant_selection():
    spec = registry.get("Multigrate")
    adt = spec.select("mosaic", {"rna1", "rna2", "adt2", "adt3"})
    atac = spec.select("mosaic", {"rna1", "rna2", "atac2", "atac3"})
    assert adt.entrypoint != atac.entrypoint
    assert "rna_adt" in adt.entrypoint and "rna_atac" in atac.entrypoint


def test_newly_wired_methods_verified():
    new_ids = ["totalVI", "MultiVI", "Cobolt", "scMVP", "VIMCCA", "uniPort",
               "sciCAN", "VIPCCA", "Portal", "SMILE", "MultiMAP", "scMSI"]
    for mid in new_ids:
        spec = registry.get(mid)
        assert spec.status == "verified", mid
        assert spec.variants, mid


def test_seurat_is_positional_R():
    spec = registry.get("Seurat_v5")
    v = spec.select("diagonal", {"rna", "atac_peak"})
    assert v.language == "R"
    assert all(a.is_positional for a in v.args)
    # five positional slots, rna + atac_peak duplicated
    roles = [a.role for a in v.args]
    assert roles.count("rna") == 2 and roles.count("atac_peak") == 2


def test_newly_wired_R_methods_verified():
    new_ids = ["iNMF", "Conos", "online_iNMF", "UINMF"]
    for mid in new_ids:
        spec = registry.get(mid)
        assert spec.status == "verified", mid
        assert spec.language == "R", mid
        assert spec.variants, mid
        for v in spec.variants:
            assert v.language == "R", mid
            assert all(a.is_positional for a in v.args), mid


def test_uinmf_two_variants():
    spec = registry.get("UINMF")
    vert = spec.select("vertical", {"rna", "adt"})
    cross = spec.select("cross", {"rna1", "rna2", "adt1", "adt2"})
    assert vert.entrypoint.endswith("main_UINMF_vertical.Rmd")
    assert cross.entrypoint.endswith("main_UINMF_cross.Rmd")


def test_online_inmf_entrypoint_has_space():
    v = registry.get("online_iNMF").select("diagonal", {"rna", "atac_gas"})
    assert v.entrypoint == "tools_scripts/online iNMF/main_online_iNMF.Rmd"


def test_graph_output_methods_verified():
    for mid in ["scMoMaT", "MIRA", "Seurat_WNN"]:
        spec = registry.get(mid)
        assert spec.status == "verified", mid
        assert spec.variants, mid


def test_seurat_v4_stub_removed():
    assert "Seurat_v4" not in registry.list_methods()
    assert "Seurat_WNN" in registry.list_methods()


def test_seurat_wnn_const_null_roundtrip():
    v = registry.get("Seurat_WNN").select("vertical", {"rna", "adt"})
    const_args = [a for a in v.args if a.const is not None]
    assert len(const_args) == 1
    # quoted "NULL" must load as the string, NOT YAML null (None)
    assert const_args[0].const == "NULL"
    assert const_args[0].role == "atac"


def test_list_methods_filter():
    assert "SCALEX" in registry.list_methods(category="diagonal")


# --- P06: needs_labels / atac are derived or enforced, never hand-drifted ---

def test_needs_labels_derived_from_cty_roles():
    from multibench.engine.schema import is_label_role
    for spec in registry.load():
        expect = any(is_label_role(r) for v in spec.variants for r in v.roles())
        assert spec.needs_labels == expect, spec.id
    # these four took cty roles while the old hand flag said False
    for m in ("scJoint", "Seurat_v3", "UnitedNet", "scMoMaT"):
        assert registry.get(m).needs_labels, m
    assert registry.get("scBridge").needs_labels       # const cty filenames still count
    assert registry.get("Matilda").needs_labels
    assert not registry.get("SCALEX").needs_labels


def test_needs_labels_per_variant_scMoMaT():
    spec = registry.get("scMoMaT")
    by_cat = {}
    for v in spec.variants:
        by_cat.setdefault(v.when["category"], set()).add(v.needs_labels)
    assert by_cat["mosaic"] == {True}
    assert by_cat["vertical"] == {False}
    assert by_cat["cross"] == {False}


def test_methods_yaml_has_no_hand_needs_labels_key():
    import yaml
    data = yaml.safe_load(registry._YAML.read_text())
    offenders = [m["id"] for m in data["methods"] if "needs_labels" in m]
    assert offenders == [], offenders


def test_parse_method_rejects_hand_needs_labels_key():
    import pytest
    with pytest.raises(ValueError, match="needs_labels is derived"):
        registry._parse_method({"id": "X", "needs_labels": True})


def test_every_atac_consuming_method_declares_atac():
    for spec in registry.load():
        if spec.consumes_atac:
            assert spec.atac in registry.ATAC_VALUES, spec.id
        if spec.atac is not None:
            assert spec.consumes_atac, spec.id
    # the role name lies for these three: atac_gas role, peak consumers
    for m in ("moETM", "scMM", "iPOLNG"):
        assert registry.get(m).atac == "peak", m
    assert registry.get("SCALEX").atac == "gene_activity"


def test_parse_method_rejects_missing_or_stray_atac():
    import pytest
    variant = {"when": {"category": "vertical", "modalities": ["rna", "atac"]},
               "entrypoint": "tools_scripts/X/main.py",
               "args": [{"role": "rna", "flag": "--a"}, {"role": "atac", "flag": "--b"},
                        {"role": "out_dir", "flag": "--o"}],
               "output": {"kind": "embedding", "file": "embedding.h5", "dataset": "data"}}
    with pytest.raises(ValueError, match="consumes an ATAC input but declares no `atac:`"):
        registry._parse_method({"id": "X", "variants": [variant]})
    with pytest.raises(ValueError, match="valid: \\['peak', 'gene_activity'\\]"):
        registry._parse_method({"id": "X", "atac": "peaks!", "variants": [variant]})
    with pytest.raises(ValueError, match="no variant takes an ATAC input"):
        registry._parse_method({"id": "X", "atac": "peak", "variants": []})
    # the valid shape parses and derives needs_labels=False
    spec = registry._parse_method({"id": "X", "atac": "peak", "variants": [variant]})
    assert spec.atac == "peak" and spec.consumes_atac and not spec.needs_labels


# --- P09: one validator per token family -------------------------------------

def test_get_unknown_method_has_did_you_mean():
    import pytest
    with pytest.raises(KeyError) as e:
        registry.get("Stabmap")
    msg = str(e.value)
    assert "did you mean 'StabMap'" in msg and "mtb.list_methods()" in msg
    with pytest.raises(KeyError, match="unknown method 'zzz'"):
        registry.check_method("zzz")
    assert registry.check_method("StabMap") == "StabMap"


def test_list_methods_validates_category_and_task():
    import pytest
    with pytest.raises(ValueError) as e:
        registry.list_methods(category="crosss")
    assert "unknown category 'crosss'" in str(e.value)
    assert "['cross', 'diagonal', 'mosaic', 'vertical']" in str(e.value)
    with pytest.raises(ValueError) as e:
        registry.list_methods(task="xx")
    assert "unknown task 'xx'" in str(e.value)
    assert str(registry.list_tasks()) in str(e.value)
    assert registry.check_category(None) is None and registry.check_task(None) is None


def test_check_atac_values_and_aliases():
    import pytest
    assert registry.check_atac("peak") == "peak"
    assert registry.check_atac("gas") == "gene_activity"
    assert registry.check_atac("peaks") == "peak"
    assert registry.check_atac(None) is None
    with pytest.raises(ValueError, match="unknown atac representation 'binary'"):
        registry.check_atac("binary")


def test_normalize_modalities_aliases_base_and_errors():
    import pytest
    assert registry.normalize_modalities(["rna", "protein"]) == ["rna", "adt"]
    assert registry.normalize_modalities(["rna", "atac_gas", "atac_peak"], base=True) == ["rna", "atac"]
    assert registry.normalize_modalities(["rna1", "rna2", "adt1"], base=True) == ["rna", "adt"]
    assert registry.normalize_modalities(None) is None
    assert registry.normalize_modalities("protein") == ["adt"]
    with pytest.raises(ValueError) as e:
        registry.normalize_modalities(["rna", "proteinx"])
    assert "unknown modality 'proteinx'" in str(e.value)
    assert "alias: protein" in str(e.value)
    assert {"rna", "adt", "atac", "atac_gas", "atac_peak", "rna1"} <= registry.known_modalities()


# --- P13: provenance attached at load time -----------------------------------

def test_references_attached_and_benchmark_entry():
    ref = registry.benchmark_reference()
    assert ref["doi"] == "10.1038/s41592-025-02856-3"
    assert "Nature Methods" in ref["journal"]
    for s in registry.load():
        assert s.reference.get("repo_url", "").startswith("http"), s.id
        assert s.reference.get("summary"), s.id
