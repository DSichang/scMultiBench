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


def test_method_info_default_keys():
    import multibench as mtb
    info = mtb.method_info("Matilda")
    assert set(info) == {"id", "language", "notes", "categories", "env", "atac",
                         "needs_labels", "supports", "runtime", "gpu",
                         "cpu_params", "setup_hint", "repo_url", "reference"}


# --- modalities= filter ---

def test_modality_types_mapping():
    from multibench.engine import registry
    def types(m):
        return set().union(*(v.modality_types for v in registry.get(m).variants))
    assert types("SCALEX") == {"rna", "atac"}
    assert types("Cobolt") == {"rna", "atac"}
    assert types("totalVI") == {"rna", "adt"}
    assert types("UINMF") == {"rna", "adt", "atac"}


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
        assert any({"rna", "atac"} <= v.modality_types for v in s.variants)


def test_find_methods_modalities_rna_ubiquitous():
    ids = discover.find_methods(modalities=["rna"])
    for m in ("SCALEX", "totalVI", "scMVP", "UINMF", "VIMCCA"):
        assert m in ids


def test_find_methods_modalities_none_unchanged():
    # modalities=None must behave exactly like the call without the kwarg
    assert discover.find_methods() == discover.find_methods(modalities=None)


# --- P06: derived needs_labels / explicit atac reach discovery ---------------

# --- P09: typos raise, aliases resolve ---------------------------------------

def test_params_for_validates_and_aliases():
    import pytest
    p = discover.params_for("totalVI", "vertical", ["rna", "protein"])
    assert p["variant"] == "vertical:rna+adt"
    with pytest.raises(KeyError, match="Unknown method"):
        discover.params_for("StabMapp")
    with pytest.raises(ValueError, match="unknown category 'verticall'"):
        discover.params_for("totalVI", "verticall")
