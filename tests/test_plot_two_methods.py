"""bubble and bar warn once when they compare two methods: each column is
min-max scaled, so the lower method is drawn at the bottom of every column
(Matilda 1, RNA_PCA 0 though their values were close)."""
import warnings

import pandas as pd
import pytest

import multibench as mtb

MSG = "With 2 methods each column only shows which is higher."


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
    assert [str(x.message) for x in w if "2 methods" in str(x.message)] == [MSG]


@pytest.mark.parametrize("fn", ["bubble", "bar"])
def test_three_methods_do_not(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(mtb.plot, fn)(_long(3))
    assert not [x for x in w if "2 methods" in str(x.message)]
