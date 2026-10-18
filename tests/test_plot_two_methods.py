"""bubble and bar draw two methods without a warning (r3, Z15): the
quickstart says that with two rows the darker one is the better of the two."""
import warnings

import pandas as pd
import pytest

import multibench as mtb



def _long(n):
    names = ["Matilda", "RNA_PCA", "totalVI"][:n]
    return pd.DataFrame([{"method": m, "metric": k, "value": v - i * 0.1, "dataset": "D11"}
                         for i, m in enumerate(names)
                         for k, v in (("ARI", 0.9), ("NMI", 0.8), ("ASW", 0.6))])


@pytest.mark.parametrize("fn", ["bubble", "bar"])
def test_two_methods_warn_once(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(mtb.plot, fn)(_long(2))
    assert not [x for x in w if "2 methods" in str(x.message)]


@pytest.mark.parametrize("fn", ["bubble", "bar"])
def test_three_methods_do_not(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(mtb.plot, fn)(_long(3))
    assert not [x for x in w if "2 methods" in str(x.message)]
