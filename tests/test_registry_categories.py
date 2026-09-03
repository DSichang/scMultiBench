"""categories derived from the variants (P09); scBridge's modalities from its
const filenames (P03); resolve_method_id (Rin)."""
import pytest
import yaml

import multibench as mtb
from multibench import discover, workflow
from multibench.engine import registry
from multibench.engine.schema import MethodSpec


def test_categories_derived_from_variants():
    for spec in registry.load():
        assert spec.categories == spec.wired_categories == \
            list(dict.fromkeys(v.when["category"] for v in spec.variants)), spec.id
    for c in ("vertical", "diagonal", "mosaic", "cross"):
        want = {s.id for s in registry.load() if any(v.when["category"] == c for v in s.variants)}
        assert set(registry.list_methods(category=c)) == want
        assert set(discover.find_methods(category=c)) == want
        assert {r[0].id for r in workflow._variant_rows(c)} == want
    # the three that had drifted
    assert "Multigrate" not in mtb.list_methods(category="cross")
    for m in ("totalVI", "sciPENN"):
        assert m not in mtb.list_methods(category="mosaic")
    assert len(mtb.list_methods(category="cross")) == 12
    assert mtb.method_info("Multigrate")["categories"] == ["mosaic", "vertical"]


def test_methods_yaml_has_no_hand_categories_key():
    data = yaml.safe_load(registry._YAML.read_text())
    assert [m["id"] for m in data["methods"] if "categories" in m] == []


def test_parse_method_rejects_hand_categories_key():
    with pytest.raises(ValueError, match="categories is derived"):
        registry._parse_method({"id": "X", "categories": ["cross"]})


def test_explicit_categories_still_accepted_by_the_dataclass():
    spec = MethodSpec(id="M", language="python", categories=["mosaic"], tasks=[])
    assert spec.categories == ["mosaic"] and spec.wired_categories == []
    assert MethodSpec(id="N", language="python").categories == []


def test_scbridge_modality_types_from_const_filenames():
    v = registry.get("scBridge").variants[0]
    assert v.modality_types == {"rna", "atac"} and v.takes_data_dir
    assert not v.modalities_unknown
    for m in ("SPIRAL", "GPSA", "PASTE", "PASTE2"):
        w = registry.get(m).variants[0]
        assert w.modality_types == set() and w.takes_data_dir and w.modalities_unknown, m
    # role-derived answers unchanged
    assert registry.get("SCALEX").variants[0].modality_types == {"rna", "atac"}
    assert registry.get("UINMF").select("vertical", {"rna", "adt"}).modality_types == {"rna", "adt"}


def test_empty_when_modalities_only_for_data_dir_variants():
    for spec in registry.load():
        for v in spec.variants:
            assert (not v.when.get("modalities")) == v.takes_data_dir, (spec.id, v.when)


def test_resolve_method_id_case_folds_with_did_you_mean():
    assert registry.resolve_method_id("scipenn") == "sciPENN"
    assert registry.resolve_method_id("totalvi") == "totalVI"
    assert registry.resolve_method_id("MOFA2") == "MOFA2"
    assert registry.resolve_method_id(" Matilda ") == "Matilda"
    with pytest.raises(KeyError) as e:
        registry.resolve_method_id("Matlida")
    assert "did you mean 'Matilda'" in str(e.value) and "mtb.list_methods()" in str(e.value)
    # strict validator is unchanged
    with pytest.raises(KeyError):
        registry.check_method("scipenn")


def test_loose_select_and_family_helper():
    from multibench.engine.schema import AmbiguousVariantError, modality_family
    assert modality_family("atac_gas") == modality_family("atac_peak") == "atac"
    assert modality_family("atac_gas2") == "atac2" and modality_family("rna1") == "rna1"
    spec = registry.get("scMM")
    assert spec.select("vertical", {"rna", "atac"}, loose=True).when["modalities"] == ["rna", "atac_gas"]
    with pytest.raises(KeyError):
        spec.select("vertical", {"rna", "atac"})            # strict, as before
    e = AmbiguousVariantError("x")
    assert isinstance(e, ValueError) and isinstance(e, KeyError) and str(e) == "x"
