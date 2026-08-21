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
    # A declared stub (no variants) has an empty modality set, so it must never
    # appear in a modalities= filter. Checked dynamically over any remaining
    # stubs: all 40 methods are currently wired (zero stubs), so this may be
    # vacuous now, but it stays correct if an unwired stub is ever re-added.
    for stub in discover.find_methods(runnable=False):
        for req in (["rna"], ["rna", "atac"], ["adt"]):
            assert stub not in discover.find_methods(modalities=req)


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
    assert set(stubs) == set(all_ids) - set(runnable)        # stubs == the non-runnable
    # Milestone: all 40 methods are now wired -> zero stubs, every id dispatchable.
    # (Guard: adding a new declared-but-unwired method should wire it or update this.)
    assert stubs == []
    assert set(runnable) == set(all_ids)


# --- P06: derived needs_labels / explicit atac reach discovery ---------------

def test_method_info_needs_labels_matches_inputs_for(root):
    from multibench.engine import resolve
    from multibench.engine.schema import is_label_role
    data = root / "data"
    cases = [("scJoint", "D28", "diagonal", None),
             ("SCALEX", "D28", "diagonal", None),
             ("Matilda", "D11", "vertical", ["rna", "adt"])]
    for m, ds, cat, mods in cases:
        info = discover.method_info(m)
        got = resolve.inputs_for(ds, m, cat, modalities=mods, data_path=data, check=False)
        assert info["needs_labels"] == any(is_label_role(k) for k in got), (m, got)
        assert all("needs_labels" in sup for sup in info["supports"])
    assert discover.method_info("scJoint")["needs_labels"] is True
    assert discover.method_info("Matilda")["needs_labels"] is True
    # scBridge's labels are const filenames (not resolved roles) -> still True
    assert discover.method_info("scBridge")["needs_labels"] is True
    assert discover.find_methods(category="diagonal", needs_labels=True) >= ["scBridge"]
    assert "scJoint" in discover.find_methods(category="diagonal", needs_labels=True)


def test_find_methods_atac_vertical_nonempty():
    peaks = discover.find_methods(category="vertical", atac="peak")
    for m in ("moETM", "scMM", "MIRA", "scMVP", "Seurat_WNN"):
        assert m in peaks, m
    gas = discover.find_methods(atac="gene_activity")
    for m in ("SCALEX", "scJoint", "Matilda"):
        assert m in gas, m
    assert set(peaks).isdisjoint(gas)
    # alias spellings map onto the same answer
    assert discover.find_methods(atac="peaks") == discover.find_methods(atac="peak")
    assert discover.find_methods(atac="gas") == gas


# --- P09: typos raise, aliases resolve ---------------------------------------

def test_find_methods_rejects_bad_tokens():
    import pytest
    with pytest.raises(ValueError, match="unknown category 'spatial'"):
        discover.find_methods(category="spatial")
    with pytest.raises(ValueError, match="unknown category 'crosss'"):
        discover.find_methods(category="crosss")
    with pytest.raises(ValueError, match="unknown task 'xx'"):
        discover.find_methods(task="xx")
    with pytest.raises(ValueError, match="unknown atac representation 'binary'"):
        discover.find_methods(atac="binary")
    with pytest.raises(ValueError) as e:
        discover.find_methods(modalities=["proteinx"])
    assert "unknown modality 'proteinx'" in str(e.value) and "protein" in str(e.value)


def test_find_methods_modalities_aliases():
    prot = discover.find_methods(modalities=["rna", "protein"])
    assert prot == discover.find_methods(modalities=["rna", "adt"])
    assert "totalVI" in prot
    # the tutorial cell: role tokens reduce to their base type
    diag = discover.find_methods(category="diagonal", modalities=["rna", "atac_gas"])
    assert diag == discover.find_methods(category="diagonal", modalities=["rna", "atac"])
    assert diag and "SCALEX" in diag
    assert discover.find_methods(modalities=["rna", "peak"]) == discover.find_methods(modalities=["rna", "atac"])


def test_method_info_unknown_method_points_at_mtb_list_methods():
    import pytest
    with pytest.raises(KeyError) as e:
        discover.method_info("Stabmap")
    assert "did you mean 'StabMap'" in str(e.value) and "mtb.list_methods()" in str(e.value)


def test_params_for_validates_and_aliases():
    import pytest
    p = discover.params_for("totalVI", "vertical", ["rna", "protein"])
    assert p["variant"] == "vertical:rna+adt"
    with pytest.raises(KeyError, match="did you mean 'StabMap'"):
        discover.params_for("Stabmap")
    with pytest.raises(ValueError, match="unknown category 'verticall'"):
        discover.params_for("totalVI", "verticall")
