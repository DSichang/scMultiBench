"""Numeric helpers ported from the R bubble-plot code (helpers.R / scIB_knit_table.R)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1]; constant or all-NaN -> all ones (matches R behavior)."""
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or lo == hi:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def rank_max(x: np.ndarray) -> np.ndarray:
    """Rank ascending with ties assigned the maximum rank (R ties.method='max')."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rank(method="max").to_numpy()


def compute_overall(mat: pd.DataFrame) -> pd.Series:
    """overall = minmax(mean over columns of per-column max-rank)."""
    ranks = mat.apply(lambda col: rank_max(col.to_numpy()), axis=0)
    mean_rank = ranks.mean(axis=1)
    return pd.Series(minmax(mean_rank.to_numpy()), index=mat.index)
