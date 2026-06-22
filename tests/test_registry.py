from multibench.run import registry


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
