from multibench import discover


def test_find_methods_by_category_and_task():
    ids = discover.find_methods(category="diagonal", task="clustering")
    assert "SCALEX" in ids


def test_find_methods_needs_labels_filter():
    # scBridge requires cell-type labels; SCALEX does not
    with_labels = discover.find_methods(category="diagonal", needs_labels=True)
    assert "scBridge" in with_labels
    assert "SCALEX" not in with_labels


def test_method_info_returns_dict_with_catalog_fields(files_dir):
    info = discover.method_info("Seurat_v5", files_dir=files_dir)
    assert info["id"] == "Seurat_v5"
    assert info["language"] == "R"
    assert "env" in info and "status" in info


def test_top_level_find_methods():
    import multibench as mtb
    assert "SCALEX" in mtb.find_methods(category="diagonal")


# --- modalities= filter ---

def test_modality_types_mapping():
    from multibench.engine import registry
    from multibench.discover import _modality_types
    assert _modality_types(registry.get("SCALEX")) == {"rna", "atac"}
    assert _modality_types(registry.get("Cobolt")) == {"rna", "atac"}
    assert _modality_types(registry.get("UINMF")) == {"rna", "adt"}


def test_find_methods_modalities_rna_adt():
    ids = discover.find_methods(modalities=["rna", "adt"])
    for m in ("totalVI", "scMSI", "VIMCCA", "UINMF"):
        assert m in ids
    # pure rna+atac method must be excluded
    assert "scMVP" not in ids


def test_find_methods_modalities_rna_atac():
    ids = discover.find_methods(modalities=["rna", "atac"])
    for m in ("SCALEX", "scMVP", "uniPort", "sciCAN", "MultiMAP"):
        assert m in ids


def test_find_methods_modalities_and_combined_with_category():
    from multibench.engine import registry
    from multibench.discover import _modality_types
    ids = discover.find_methods(category="diagonal", modalities=["rna", "atac"])
    assert "SCALEX" in ids
    for m in ids:
        s = registry.get(m)
        assert "diagonal" in s.categories
        assert {"rna", "atac"} <= _modality_types(s)


def test_find_methods_modalities_rna_ubiquitous():
    ids = discover.find_methods(modalities=["rna"])
    for m in ("SCALEX", "totalVI", "scMVP", "UINMF", "VIMCCA"):
        assert m in ids


def test_find_methods_modalities_excludes_stub():
    # MOFA2 is a declared stub with no variants -> empty modality set
    for req in (["rna"], ["rna", "atac"], ["adt"]):
        assert "MOFA2" not in discover.find_methods(modalities=req)


def test_find_methods_modalities_none_unchanged():
    # modalities=None must behave exactly like the call without the kwarg
    assert discover.find_methods() == discover.find_methods(modalities=None)


def test_find_methods_runnable_filter():
    all_ids = discover.find_methods()
    runnable = discover.find_methods(runnable=True)
    stubs = discover.find_methods(runnable=False)
    assert set(runnable) <= set(all_ids)
    assert set(runnable) | set(stubs) == set(all_ids)        # complete partition
    assert set(runnable) & set(stubs) == set()               # disjoint
    assert "SCALEX" in runnable                              # has a variant
    assert "MOFA2" in stubs and "MOFA2" not in runnable      # declared stub (not yet wired)
    assert len(runnable) < len(all_ids)
