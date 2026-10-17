"""``params_for`` takes the modality tokens ``find_methods`` takes.

``find_methods("cross", modalities=["rna", "atac"])`` lists Multigrate, but
``params_for("Multigrate", "cross", modalities=["rna", "atac"])`` raised
KeyError: the variant's roles are numbered (rna1..3, atac1..3)."""
import pytest

import multibench as mtb


@pytest.mark.parametrize("cat, mods, key", [
    ("cross", ["rna", "atac"], "cross:rna1+rna2+rna3+atac1+atac2+atac3"),
    ("cross", ["rna", "protein"], "cross:rna1+rna2+rna3+adt1+adt2+adt3"),
    ("mosaic", ["rna", "atac"], "mosaic:rna1+rna2+atac2+atac3"),
    (None, ["rna", "atac"], "vertical:rna+atac"),
])
def test_base_tokens_select_the_variant(cat, mods, key):
    assert "Multigrate" in mtb.find_methods(cat, modalities=mods)
    assert mtb.params_for("Multigrate", cat, mods)["variant"] == key


def test_several_variants_of_those_tokens_are_ambiguous():
    with pytest.raises(mtb.AmbiguousVariantError, match=r"^StabMap has 2 mosaic variants"):
        mtb.params_for("StabMap", "mosaic", ["rna", "adt", "atac"])


def test_tokens_no_variant_reads_still_raise():
    with pytest.raises(KeyError, match="no variant"):
        mtb.params_for("totalVI", "vertical", ["atac"])
