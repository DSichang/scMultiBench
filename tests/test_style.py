import numpy as np
import pandas as pd
from multibench.plot import style


def test_minmax_scales_to_unit_interval():
    out = style.minmax(np.array([2.0, 4.0, 6.0]))
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_minmax_constant_column_returns_ones():
    out = style.minmax(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(out, [1.0, 1.0, 1.0])


def test_minmax_constant_column_keeps_its_missing_values():
    out = style.minmax(np.array([np.nan, 0.5, 0.5]))
    assert np.isnan(out[0]) and list(out[1:]) == [1.0, 1.0]


def test_rank_ties_max():
    # values 0.1,0.1,0.9 -> ranks (tie at max) 2,2,3
    r = style.rank_max(np.array([0.1, 0.1, 0.9]))
    assert list(r) == [2.0, 2.0, 3.0]


