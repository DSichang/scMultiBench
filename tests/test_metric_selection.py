"""One metrics= vocabulary: a family, one code, or codes and families in a list
(core040 E-1, E-2)."""
import pytest

import multibench as mtb
from multibench._data.catalog import metric_selection

V = "ARI, ASW, NMI, cLISI, iASW and iF1"


def test_one_code_is_a_list_of_one():
    sel = metric_selection("ari")
    assert sel.codes == ["ARI"] and sel.explicit


def test_a_family_in_a_list_adds_its_codes_without_kbet():
    sel = metric_selection(["batch", "ARI"])
    assert sel.codes[0] == "ASW_batch" and "kBET" not in sel.codes and sel.codes[-1] == "ARI"
    assert "kBET" in metric_selection(["batch", "kBET"]).codes


def test_a_code_named_twice_is_kept_once():
    assert metric_selection(["ari", "ARI"]).codes == ["ARI"]


def test_an_unknown_token_names_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown metrics= token ARX"):
        metric_selection("ARX")


def test_load_results_and_rank_take_one_code():
    assert set(mtb.load_results("vertical", metrics="ARI")["metric"]) == {"ARI"}
    assert len(mtb.rank("vertical", metrics="NMI"))


def test_load_results_refuses_a_code_it_has_no_row_for():
    with pytest.raises(ValueError) as e:
        mtb.load_results("vertical", metrics=["ARI", "kBET"])
    assert str(e.value) == (f"The stored results of vertical have no kBET. "
                            f"Pick metrics= from {V}.")
