import pytest

from multibench import discover


def test_find_methods_needs_labels_filter():
    # scBridge requires cell-type labels; SCALEX does not
    with_labels = discover.find_methods(category="diagonal", needs_labels=True)
    assert "scBridge" in with_labels
    assert "SCALEX" not in with_labels


def test_top_level_find_methods():
    import multibench as mtb
    assert "SCALEX" in mtb.find_methods(category="diagonal")


# --- modalities= filter ---

def test_modality_types_mapping():
    from multibench.engine import registry
    assert registry.get("SCALEX").modality_types == {"rna", "atac"}
    assert registry.get("Cobolt").modality_types == {"rna", "atac"}
    assert registry.get("totalVI").modality_types == {"rna", "adt"}
    assert registry.get("UINMF").modality_types == {"rna", "adt", "atac"}


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
    ids = discover.find_methods(category="diagonal", modalities=["rna", "atac"])
    assert "SCALEX" in ids
    for m in ids:
        s = registry.get(m)
        assert "diagonal" in s.categories
        assert {"rna", "atac"} <= s.modality_types


def test_find_methods_modalities_rna_ubiquitous():
    ids = discover.find_methods(modalities=["rna"])
    for m in ("SCALEX", "totalVI", "scMVP", "UINMF", "VIMCCA"):
        assert m in ids


def test_find_methods_modalities_excludes_stub():
    # A declared stub (no variants) has an empty modality set, so it must never
    # appear in a modalities= filter. Checked dynamically over any remaining
    # stubs: every method is currently wired (zero stubs), so this may be
    # vacuous now, but it stays correct if an unwired stub is ever re-added.
    from multibench.engine import registry
    for stub in [s.id for s in registry.load() if not s.variants]:
        for req in (["rna"], ["rna", "atac"], ["adt"]):
            assert stub not in discover.find_methods(modalities=req)


def test_find_methods_modalities_none_unchanged():
    # modalities=None must behave exactly like the call without the kwarg
    assert discover.find_methods() == discover.find_methods(modalities=None)


# --- P06: derived needs_labels / explicit atac reach discovery ---------------

# --- P09: typos raise, aliases resolve ---------------------------------------

def test_params_for_validates_and_aliases():
    import pytest
    p = discover.params_for("totalVI", "vertical", ["rna", "protein"])
    assert p["variant"] == "vertical:rna+adt"
    with pytest.raises(KeyError, match="Did you mean StabMap"):
        discover.params_for("StabMapp")
    with pytest.raises(ValueError, match="unknown category 'verticall'"):
        discover.params_for("totalVI", "verticall")
