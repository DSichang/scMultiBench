import numpy as np
import pandas as pd
from multibench.plot import style


def test_minmax_scales_to_unit_interval():
    out = style.minmax(np.array([2.0, 4.0, 6.0]))
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_minmax_constant_column_returns_ones():
    out = style.minmax(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(out, [1.0, 1.0, 1.0])


def test_rank_ties_max():
    # values 0.1,0.1,0.9 -> ranks (tie at max) 2,2,3
    r = style.rank_max(np.array([0.1, 0.1, 0.9]))
    assert list(r) == [2.0, 2.0, 3.0]


def test_compute_overall_from_matrix():
    mat = pd.DataFrame(
        {"ARI": [0.0, 1.0], "NMI": [0.0, 1.0]}, index=["m_lo", "m_hi"]
    )
    overall = style.compute_overall(mat)
    # higher row gets overall 1.0, lower 0.0
    assert overall["m_hi"] == 1.0
    assert overall["m_lo"] == 0.0
