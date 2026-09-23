"""Numeric helpers ported from the R bubble-plot code (helpers.R / scIB_knit_table.R).

This module is also the single source of truth for the cross-dataset summary
math shared by :func:`multibench.plot.bubble` and :func:`multibench.plot.bar`:
:func:`per_dataset_ranks`, :func:`mean_rank_matrix` and
:func:`overall_by_basis`. Both figures call these, so with the same
``overall=`` and the metrics of one family they order methods identically
(see the Notes of ``plot.bubble``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: values that agree to this many decimal places are ties: floating-point
#: noise (an iLISI of 2.2e-16 next to 0.0) must not decide a rank or a fill
TIE_DECIMALS = 9


def _tie_round(x) -> np.ndarray:
    """``x`` as floats rounded to :data:`TIE_DECIMALS` places (NaN kept)."""
    return np.round(np.asarray(x, dtype=float), TIE_DECIMALS)


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1]; constant or all-NaN -> all ones (matches R behaviour).

    Values are compared after rounding to :data:`TIE_DECIMALS` places, so a
    column that differs only by floating-point noise counts as constant.
    """
    x = _tie_round(x)
    lo = np.nanmin(x) if np.isfinite(x).any() else np.nan
    hi = np.nanmax(x) if np.isfinite(x).any() else np.nan
    if not np.isfinite(lo) or lo == hi:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def rank_max(x: np.ndarray) -> np.ndarray:
    """Rank ascending with ties assigned the maximum rank (R ties.method='max').

    Values that agree to :data:`TIE_DECIMALS` places share a rank.
    """
    s = pd.Series(_tie_round(x))
    return s.rank(method="max").to_numpy()


def compute_overall(mat: pd.DataFrame) -> pd.Series:
    """overall = minmax(mean over columns of per-column max-rank)."""
    ranks = mat.apply(lambda col: rank_max(col.to_numpy()), axis=0)
    mean_rank = ranks.mean(axis=1)
    return pd.Series(minmax(mean_rank.to_numpy()), index=mat.index)


# --------------------------------------------------------------------------
# cross-dataset summary math (shared by bubble and bar)
# --------------------------------------------------------------------------

#: the two ways an across-dataset "Overall" can be formed; the formulas are
#: in overall_by_basis
OVERALL_BASES = ("rank", "mean_overall")

#: the ``overall`` parameter entry written out verbatim in the docstrings of
#: plot.bubble and plot.bar (tests/test_bubble.py and tests/test_bar.py pin it)
OVERALL_DOC = """\
    overall : {"rank", "mean_overall"}
        Across-dataset *Overall*: ``"rank"`` re-ranks mean ranks (missing
        dataset = rank 0); ``"mean_overall"`` averages per-dataset Overalls
        (missing dataset skipped)."""


def per_dataset_ranks(long_df: pd.DataFrame, metrics=None) -> dict:
    """Within-dataset max-ranks: ``{dataset: DataFrame(method x metric)}``.

    For each dataset the frame is pivoted to method x metric (mean over
    duplicate rows), metrics that dataset never computed are dropped, and
    every remaining column is replaced by its max-rank (1 = worst ... n =
    best, ties share the maximum rank). A frame without a ``dataset`` column
    is treated as one dataset named ``"all"``. Datasets whose pivot is empty
    are skipped.
    """
    d = long_df.copy()
    if metrics:
        d = d[d["metric"].isin(metrics)]
    if d.empty:
        raise ValueError("no rows left after filtering to metrics=%r" % (metrics,))
    if "dataset" not in d.columns:
        d = d.assign(dataset="all")
    out = {}
    for ds, g in d.groupby("dataset", sort=True):
        mat = g.pivot_table(index="method", columns="metric", values="value",
                            aggfunc="mean")
        mat = mat.dropna(axis=1, how="all")
        if mat.empty:
            continue
        out[ds] = mat.apply(lambda col: rank_max(col.to_numpy()), axis=0)
    if not out:
        raise ValueError("no dataset had usable metrics")
    return out


def mean_rank_matrix(parts: dict) -> pd.DataFrame:
    """Average the per-dataset rank matrices from :func:`per_dataset_ranks`.

    Rows are the union of methods; a method absent from a dataset, or lacking
    a metric there, scores rank 0 (as in the paper's summary), but only for
    the metrics that dataset computed - each metric is averaged over the
    datasets that have it.
    """
    mats = list(parts.values())
    if not mats:
        raise ValueError("no per-dataset rank matrices to average")
    idx = mats[0].index
    for q in mats[1:]:
        idx = idx.union(q.index)
    aligned = [q.reindex(idx).fillna(0) for q in mats]
    stacked = pd.concat(aligned, keys=range(len(aligned)))
    return stacked.groupby(level=1).mean()


def coverage(parts: dict) -> pd.Series:
    """Number of datasets (keys of ``parts``) each method appears in."""
    counts: dict = {}
    for mat in parts.values():
        for m in mat.index:
            counts[m] = counts.get(m, 0) + 1
    return pd.Series(counts, dtype=int).sort_index()


def coverage_warnings(parts: dict, *, basis: str, incomplete_fix: str) -> list:
    """The warnings a cross-dataset summary of ``parts`` needs, as messages.

    ``parts`` is the dict from :func:`per_dataset_ranks`; ``basis`` the
    ``overall=`` in use; ``incomplete_fix`` the last sentence of the
    incomplete-matrix message (it names the caller's own remedy). Shared by
    ``plot.build_table`` (``aggregate="summary"``) and ``plot.bar``.

    - no method is scored on two of several datasets: one message that the
      summary ranks unrelated rows (it replaces the incomplete-matrix one);
    - otherwise, a method missing from some dataset: the incomplete-matrix
      message;
    - a dataset that holds one method while the frame holds several: one
      message per such dataset, since that method is ranked against nothing.
    """
    out = []
    n = len(parts)
    cov = coverage(parts)
    names = ", ".join(map(str, parts))
    if n > 1 and len(cov) and (cov <= 1).all():
        out.append(
            f"no method is scored on more than one of these {n} datasets "
            f"({names}); a summary across them ranks unrelated rows. Plot each "
            f"dataset on its own, or score the same methods on every dataset.")
    elif n > 1 and (cov < n).any():
        part = cov[cov < n].sort_values()
        out.append(
            f"summary ranks an incomplete method x dataset matrix ({n} "
            f"datasets): " + ", ".join(f"{m} seen in {c}/{n}" for m, c in part.items())
            + "; a method absent from a dataset scores rank 0 there under "
            "overall='rank' and is skipped under overall='mean_overall'. "
            + incomplete_fix)
    if len(cov) > 1:
        for ds, mat in parts.items():
            if len(mat.index) != 1:
                continue
            what = ("its Overall there is 1.0 by construction" if basis == "mean_overall"
                    else "its rank there is 1 by construction, the lowest possible")
            out.append(
                f"dataset {ds} has one method ({mat.index[0]}): {what}; plot it "
                f"with the methods scored on the same dataset")
    return out


def overall_by_basis(parts: dict, basis: str = "rank") -> pd.Series:
    """Across-dataset Overall per method under the given ``basis``.

    ``parts`` is the dict from :func:`per_dataset_ranks`. The two formulas
    (spelled out again in the Notes of ``plot.bubble`` and ``plot.bar``):

    - ``"rank"``: ``minmax(mean over metrics of max-rank(mean over datasets
      of within-dataset max-rank))``; a method absent from a dataset scores
      rank 0 there (the paper's summary rule).
    - ``"mean_overall"``: ``mean over datasets of minmax(mean over metrics of
      within-dataset max-rank)``; a dataset the method lacks is skipped.

    Raises ``ValueError`` for an unknown ``basis``.
    """
    if basis not in OVERALL_BASES:
        raise ValueError(
            f"overall must be one of {list(OVERALL_BASES)}, got {basis!r}")
    if basis == "rank":
        return compute_overall(mean_rank_matrix(parts))
    per_ds = pd.DataFrame({ds: compute_overall(mat) for ds, mat in parts.items()})
    return per_ds.mean(axis=1)
