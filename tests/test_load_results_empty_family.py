"""``load_results`` raises when ``metrics=`` names a family the stored results
lack, as ``rank``, ``plot.bubble`` and ``plot.bar`` do (one resolver,
``catalog.metrics_in``)."""
import warnings

import pytest

import multibench as mtb


def test_a_family_the_tables_lack_raises():
    with pytest.raises(ValueError) as e:
        mtb.load_results("vertical", metrics="classification")
    assert str(e.value) == ("The stored results of vertical have no classification metrics. "
                            "Pick metrics= from ARI, ASW, NMI, cLISI, iASW and iF1.")


def test_with_a_dataset_the_message_names_it():
    with pytest.raises(ValueError, match=r"^The stored results of vertical, D11 have "
                                         r"no batch metrics\. "):
        mtb.load_results("vertical", dataset="D11", metrics="batch")


def test_a_family_with_rows_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not mtb.load_results("vertical", dataset="D11", metrics="clustering").empty


def test_the_four_functions_say_the_same_for_one_list():
    stored = mtb.load_results("vertical")
    said = []
    for call in (lambda: mtb.load_results("vertical", metrics=["ARI", "batch"]),
                 lambda: mtb.rank("vertical", metrics=["ARI", "batch"]),
                 lambda: mtb.plot.bubble(stored, metrics=["ARI", "batch"]),
                 lambda: mtb.plot.bar(stored, metrics=["ARI", "batch"])):
        with pytest.raises(ValueError) as e:
            call()
        said.append(str(e.value).split(" have no ")[-1].split(" has no ")[-1])
    assert said == ["batch metrics. Pick metrics= from ARI, ASW, NMI, cLISI, iASW and iF1."] * 4
