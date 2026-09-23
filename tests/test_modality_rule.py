"""One modality rule for find_methods and recommend (study round 1, L13).

A base token (``rna``, ``adt``/``protein``, ``atac``) matches every role of that
type. A representation token (``atac_peak``/``peak``, ``atac_gas``/``gas``)
means base ``atac`` plus the ATAC representation the method reads, the same as
``atac='peak'`` / ``atac='gene_activity'``. Before the fix
``find_methods('diagonal', modalities=['rna', 'atac_peak'])`` returned SCALEX,
which reads gene activity, ``recommend(atac=...)`` was a TypeError and
``recommend(modalities=['rna', 'peak'])`` ranked all 14 diagonal methods.
"""
import warnings

import pytest

import multibench as mtb
from multibench import discover


def test_representation_token_selects_by_what_the_method_reads():
    peak = discover.find_methods("diagonal", atac="peak")
    assert discover.find_methods("diagonal", modalities=["rna", "atac_peak"]) == peak
    assert discover.find_methods("diagonal", modalities=["rna", "peak"]) == peak
    assert "SCALEX" not in peak and mtb.method_info("SCALEX")["atac"] == "gene_activity"
    gas = discover.find_methods("diagonal", atac="gene_activity")
    assert discover.find_methods("diagonal", modalities=["rna", "atac_gas"]) == gas
    assert discover.find_methods("diagonal", modalities=["rna", "gas"]) == gas
    # vertical: moETM reads peaks through its atac_gas role
    vpeak = discover.find_methods("vertical", modalities=["rna", "atac_peak"])
    assert "moETM" in vpeak and vpeak == discover.find_methods("vertical", atac="peak")


def test_base_token_matches_every_role_of_its_type():
    both = discover.find_methods("diagonal", modalities=["rna", "atac"])
    assert set(both) == set(mtb.list_methods("diagonal"))
    vert = discover.find_methods("vertical", modalities=["rna", "atac"])
    for m in ("moETM", "scMM", "iPOLNG", "Matilda", "MIRA"):      # atac_gas and atac roles
        assert m in vert, m
    # numbered roles reduce to their base type
    assert "StabMap" in discover.find_methods("mosaic", modalities=["rna", "adt"])


def test_two_representations_select_the_variants_that_read_both_files():
    both = discover.find_methods("diagonal", modalities=["rna", "atac_peak", "atac_gas"])
    assert both == ["MultiMAP", "Seurat_v3"]
    assert discover.find_methods(modalities=["gas", "peaks"]) == both
    with pytest.raises(ValueError, match="atac='gene_activity'"):
        discover.find_methods("diagonal", modalities=["rna", "atac_peak"], atac="gene_activity")
    # the same representation twice is not a conflict
    assert discover.find_methods("diagonal", modalities=["rna", "peak"], atac="peaks") == \
        discover.find_methods("diagonal", atac="peak")


def test_unknown_token_raises_with_the_vocabulary():
    with pytest.raises(ValueError, match="unknown modality 'peaks2'") as e:
        discover.find_methods(modalities=["rna", "peaks2"])
    assert "atac_peak" in str(e.value) and "protein" in str(e.value)


def _methods(df):
    return set(df["method"])


def test_recommend_takes_atac_like_find_methods():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mtb.recommend("diagonal", atac="peak", source="rerun")
        peak = set(discover.find_methods("diagonal", atac="peak"))
        assert _methods(r) and _methods(r) <= peak
        assert _methods(mtb.recommend("diagonal", modalities=["rna", "peak"], source="rerun")) == _methods(r)
        assert _methods(mtb.recommend("diagonal", modalities=["rna", "atac"], source="rerun")) > _methods(r)


def test_recommend_rejects_a_bad_atac_before_loading(tmp_path):
    with pytest.raises(ValueError, match="unknown atac representation 'binary'"):
        mtb.recommend("diagonal", atac="binary", result_path=tmp_path / "nowhere")
    with pytest.raises(ValueError, match="unknown modality 'peaks2'"):
        mtb.recommend("diagonal", modalities=["atac_peak", "peaks2"],
                      result_path=tmp_path / "nowhere")
