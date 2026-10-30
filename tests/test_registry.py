from multibench.engine import registry


def test_all_methods_declared():
    specs = registry.load()
    assert len(specs) >= 36


def test_verified_methods_present():
    specs = registry.load()
    ids = {s.id for s in specs}
    for m in ["SCALEX", "Multigrate", "Seurat_v5", "scBridge"]:
        assert m in ids


def test_scalex_variant_args():
    spec = registry.get("SCALEX")
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


# --- P06: needs_labels / atac are derived or enforced, never hand-maintained ---

def test_needs_labels_derived_from_cty_roles():
    from multibench.engine.schema import is_label_role
    for spec in registry.load():
        expect = any(is_label_role(r) for v in spec.variants for r in v.roles())
        assert spec.needs_labels == expect, spec.id
    # these three took cty roles while the old hand flag said False
    for m in ("scJoint", "Seurat_v3", "UnitedNet"):
        assert registry.get(m).needs_labels, m
    assert registry.get("scBridge").needs_labels       # const cty filenames still count
    assert registry.get("Matilda").needs_labels
    assert not registry.get("SCALEX").needs_labels


def test_scMoMaT_reads_no_label_file():
    # main_scMoMaT.py:93, the only use of --cty_path, is commented out
    spec = registry.get("scMoMaT")
    assert not spec.needs_labels
    assert not any(v.needs_labels for v in spec.variants)


def test_methods_yaml_has_no_hand_needs_labels_key():
    import yaml
    data = yaml.safe_load(registry._YAML.read_text())
    offenders = [m["id"] for m in data["methods"] if "needs_labels" in m]
    assert offenders == [], offenders
    # every wired method was run end to end (the field is for maintainers)
    assert [m["id"] for m in data["methods"] if m.get("status") != "verified"] == []


def test_every_atac_consuming_method_declares_atac():
    for spec in registry.load():
        consumes = any(v.consumes_atac for v in spec.variants)
        if consumes:
            assert spec.atac in registry.ATAC_VALUES, spec.id
        if spec.atac is not None:
            assert consumes, spec.id
    # the role name lies for these three: atac_gas role, peak consumers
    for m in ("moETM", "scMM", "iPOLNG"):
        assert registry.get(m).atac == "peak", m
    assert registry.get("SCALEX").atac == "gene_activity"


def test_parse_method_reads_the_atac_key():
    variant = {"when": {"category": "vertical", "modalities": ["rna", "atac"]},
               "entrypoint": "tools_scripts/X/main.py",
               "args": [{"role": "rna", "flag": "--a"}, {"role": "atac", "flag": "--b"},
                        {"role": "out_dir", "flag": "--o"}],
               "output": {"kind": "embedding", "file": "embedding.h5", "dataset": "data"}}
    # the valid shape parses and derives needs_labels=False
    spec = registry._parse_method({"id": "X", "atac": "peak", "variants": [variant]})
    assert spec.atac == "peak" and spec.variants[0].consumes_atac and not spec.needs_labels


def test_every_method_has_variants_and_every_size_hint_a_threshold():
    for spec in registry.load():
        assert spec.variants, spec.id
        for v in spec.variants:
            if v.size_hint:
                assert v.min_cells or v.min_cells_per_batch or v.min_genes, spec.id


# --- P09: one validator per token family -------------------------------------

def test_get_unknown_method_names_find_methods():
    import pytest
    with pytest.raises(KeyError) as e:
        registry.get("StabMapp")
    msg = str(e.value)
    assert msg == "'Unknown method StabMapp. mtb.find_methods() shows all methods.'"
    with pytest.raises(KeyError, match="Unknown method zzz"):
        registry.check_method("zzz")
    assert registry.check_method("StabMap") == "StabMap"


def test_normalize_modalities_aliases_base_and_errors():
    import pytest
    assert registry.normalize_modalities(["rna", "protein"]) == ["rna", "adt"]
    assert registry.normalize_modalities(None) is None
    assert registry.normalize_modalities("protein") == ["adt"]
    with pytest.raises(ValueError) as e:
        registry.normalize_modalities(["rna", "proteinx"])
    assert str(e.value).startswith("Unknown modality proteinx. ")
    assert "atac_peak or atac_gas" in str(e.value)
    assert {"rna", "adt", "atac", "atac_gas", "atac_peak", "rna1"} <= registry.known_modalities()


# --- P13: provenance attached at load time -----------------------------------

def test_references_attached():
    for s in registry.load():
        assert s.reference.get("repo_url", "").startswith("http"), s.id
        assert s.reference.get("summary"), s.id
