"""``load_results`` warns when a family selection leaves no rows.

``load_results("vertical", metrics="classification")`` returned an empty frame
without a word (``recommend`` raises for the same selection)."""
import warnings

import pytest

import multibench as mtb


def test_a_family_the_tables_lack_warns_once():
    with pytest.warns(UserWarning) as rec:
        df = mtb.load_results("vertical", metrics="classification")
    assert df.empty
    msgs = [str(w.message) for w in rec if "no classification scores" in str(w.message)]
    assert msgs == ["The re-run vertical tables have no classification scores, so no "
                    "rows are left. Leave out metrics= to get the scores they have (ARI, "
                    "ASW, NMI, cLISI, iASW, iF1)."]


def test_with_a_dataset_the_message_names_it():
    with pytest.warns(UserWarning, match=r"^The re-run vertical tables of D11 have no "
                                         r"batch scores"):
        mtb.load_results("vertical", dataset="D11", metrics="batch")


def test_a_family_with_rows_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not mtb.load_results("vertical", dataset="D11", metrics="clustering").empty
