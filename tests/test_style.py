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


def test_overall_by_basis_formulas():
    import pytest
    # two datasets, three methods; B is absent from D2
    long = pd.DataFrame({
        "method":  ["A", "B", "C", "A", "C"],
        "metric":  ["ARI"] * 5,
        "value":   [0.1, 0.9, 0.5, 0.9, 0.1],
        "dataset": ["D1", "D1", "D1", "D2", "D2"],
    })
    parts = style.per_dataset_ranks(long)
    assert set(parts) == {"D1", "D2"}
    assert list(parts["D1"]["ARI"]) == [1.0, 3.0, 2.0]          # A, B, C ranks in D1
    assert style.coverage(parts).to_dict() == {"A": 2, "B": 1, "C": 2}
    # 'rank': mean rank (B absent -> 0 in D2): A=(1+2)/2=1.5, B=(3+0)/2=1.5, C=(2+1)/2=1.5
    mr = style.mean_rank_matrix(parts)
    assert np.allclose(mr.loc[["A", "B", "C"], "ARI"], [1.5, 1.5, 1.5])
    rank_ov = style.overall_by_basis(parts, "rank")
    assert np.allclose(rank_ov[["A", "B", "C"]], [1.0, 1.0, 1.0])   # all tied -> minmax ones
    # 'mean_overall': D1 overalls minmax([1,3,2]) = [0, 1, .5]; D2 minmax([2,1]) = [1, 0]
    # A = (0+1)/2 = .5, B = 1 (only D1), C = (.5+0)/2 = .25
    mo = style.overall_by_basis(parts, "mean_overall")
    assert np.allclose(mo[["A", "B", "C"]], [0.5, 1.0, 0.25])
    with pytest.raises(ValueError, match="overall must be one of"):
        style.overall_by_basis(parts, "median")
    assert "rank" in style.OVERALL_DOC and "mean_overall" in style.OVERALL_DOC
