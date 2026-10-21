"""categories derived from the variants (P09); scBridge's modalities from its
const filenames (P03); method ids are case-sensitive."""
import pytest
import yaml

import multibench as mtb
from multibench import discover, workflow
from multibench.engine import registry
from multibench.engine.schema import MethodSpec


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
    # scBridge is the only directory-fed variant (the docstrings say so)
    assert [s.id for s in registry.load() for w in s.variants if w.takes_data_dir] == ["scBridge"]
    # role-derived answers unchanged
    assert registry.get("SCALEX").variants[0].modality_types == {"rna", "atac"}
    assert registry.get("UINMF").select("vertical", {"rna", "adt"}).modality_types == {"rna", "adt"}


def test_empty_when_modalities_only_for_data_dir_variants():
    for spec in registry.load():
        for v in spec.variants:
            assert (not v.when.get("modalities")) == v.takes_data_dir, (spec.id, v.when)


def test_check_method_returns_the_id_in_any_case():
    # since 0.4 a method id in another letter case names that method
    assert registry.check_method("scipenn") == "sciPENN"
    with pytest.raises(KeyError):
        registry.check_method("scipen")


def test_loose_select_and_family_helper():
    from multibench.engine.schema import AmbiguousVariantError, modality_family
    assert modality_family("atac_gas") == modality_family("atac_peak") == "atac"
    assert modality_family("atac_gas2") == "atac2" and modality_family("rna1") == "rna1"
    spec = registry.get("scMM")
    # scMM's vertical variant reads the atac role since 0.4.0 (X13)
    assert spec.select("vertical", {"rna", "atac"}, loose=True).when["modalities"] == ["rna", "atac"]
    assert spec.select("vertical", {"rna", "atac_gas"}).when["modalities"] == ["rna", "atac"]
    with pytest.raises(KeyError):
        spec.select("vertical", {"rna", "atac_peak"})       # strict, as before
    e = AmbiguousVariantError("x")
    assert isinstance(e, ValueError) and isinstance(e, KeyError) and str(e) == "x"
