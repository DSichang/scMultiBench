"""``run(params=)`` given a flat dict says the dict is keyed by method.

``run(..., methods=["Matilda"], params={"epochs": 3})`` raised "Unknown
method epochs"; ``run`` takes the flat form, ``run`` and ``scan`` the
nested one. The error now shows the nested form.
"""
import pytest

import multibench as mtb


def test_flat_params_show_the_nested_form():
    with pytest.raises(TypeError) as e:
        mtb.run("D11", "vertical", methods=["Matilda"], params={"epochs": 3},
                    dry_run=True, verbose=False)
    assert str(e.value) == ("run(params=) takes one dict per method, such as "
                            "params={'Matilda': {'epochs': 3}}.")


def test_a_nested_dict_with_an_unknown_method_still_gets_the_did_you_mean():
    with pytest.raises(KeyError, match="Did you mean Matilda"):
        mtb.run("D11", "vertical", params={"Matlida": {"epochs": 3}}, dry_run=True,
                    verbose=False)
