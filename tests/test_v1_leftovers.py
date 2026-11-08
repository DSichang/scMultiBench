"""v1 leftovers: an unreadable .h5ad in one sentence, "1 cell", and rank on
rows of another category."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------------------------------------- u5 #6
def test_rank_refuses_rows_of_another_category():
    vertical = _quiet(mtb.load_results, "vertical")
    cross = _quiet(mtb.load_results, "cross")
    with pytest.raises(ValueError) as e:
        _quiet(mtb.rank, "cross", long_df=vertical)
    assert str(e.value) == (
        "long_df holds vertical rows: rank one category at a time, as "
        "rank('vertical', long_df=long_df[long_df['category'] == 'vertical']).")
    both = pd.concat([cross, vertical])
    with pytest.raises(ValueError) as e:
        _quiet(mtb.rank, "cross", long_df=both)
    assert str(e.value) == (
        "long_df holds vertical rows: rank one category at a time, as "
        "rank('cross', long_df=long_df[long_df['category'] == 'cross']).")
    with pytest.raises(ValueError, match=r"^long_df holds cross and vertical rows: rank one "
                                         r"category at a time, as rank\('cross', "):
        _quiet(mtb.rank, "mosaic", long_df=both)
    # rows of your own without a category rank next to the stored ones
    mine = cross[cross["method"] == "StabMap"].assign(method="mine", category=None)
    out = _quiet(mtb.rank, "cross", long_df=pd.concat([cross, mine]))
    assert "mine" in set(out["method"])


def test_one_cell_is_singular(tmp_path, capsys):
    (tmp_path / 'one.csv').write_text('a,b\n1,2\n')
    pd.DataFrame({'x': ['T', 'B', 'T']}).to_csv(tmp_path / 'lab.csv', index=False)
    with pytest.raises(ValueError, match='^labels has 1 entry for 3 cells in the embedding: '):
        _quiet(mtb.evaluate, np.zeros((3, 2)), labels=np.array(['T']), metrics=['ASW'])
