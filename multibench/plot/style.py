"""Numeric helpers ported from the R bubble-plot code (helpers.R / scIB_knit_table.R).

This module is also the SINGLE source of truth for the cross-dataset summary
math shared by :func:`multibench.plot.bubble` and :func:`multibench.plot.bar`:
:func:`per_dataset_ranks`, :func:`mean_rank_matrix` and
:func:`overall_by_basis`. Both figures call these, so passing the same
``overall=`` to both yields the same method ordering.
"""
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


# --------------------------------------------------------------------------
# cross-dataset summary math (shared by bubble and bar)
# --------------------------------------------------------------------------

#: the two ways an across-dataset "Overall" can be formed; see OVERALL_DOC
OVERALL_BASES = ("rank", "mean_overall")

OVERALL_DOC = """\
    overall : {"rank", "mean_overall"}
        How the across-dataset *Overall* score is formed.

        * ``"rank"`` (bubble's default): ``minmax(mean over metrics of
          max-rank(mean over datasets of within-dataset max-rank))`` - the
          per-dataset ranks are averaged per metric, the mean ranks are
          RE-RANKED across methods, averaged over metrics and min-max scaled.
          A method absent from a dataset scores rank 0 there (the paper's
          summary rule), which pulls it down.
        * ``"mean_overall"`` (bar's default): ``mean over datasets of
          minmax(mean over metrics of within-dataset max-rank)`` - each
          dataset gets its own min-max-scaled overall and those are averaged
          over the datasets the method was actually run on (absence is
          skipped, not penalised).

        The two formulas can order methods differently on the same frame;
        pass the same ``overall=`` to ``plot.bubble`` and ``plot.bar`` to get
        the same ordering. The formula in use is printed on the figure."""


def per_dataset_ranks(long_df: pd.DataFrame, metrics=None) -> dict:
    """Within-dataset max-ranks: ``{dataset: DataFrame(method x metric)}``.

    For each dataset the frame is pivoted to method x metric (mean over
    duplicate rows), metrics that dataset never computed are dropped
    (``dropna(axis=1, how="all")``), and every remaining column is replaced by
    its max-rank (1 = worst ... n = best, ties share the maximum rank). A
    frame without a ``dataset`` column is treated as one dataset named
    ``"all"``. Datasets whose pivot is empty are skipped.
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

    Rows are the union of methods; a method absent from a dataset scores rank
    0 there (as in the paper's summary), but only for the metrics that dataset
    actually computed - each metric is averaged over the datasets that have it.
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


def overall_by_basis(parts: dict, basis: str = "rank") -> pd.Series:
    """Across-dataset Overall per method under the given ``basis``.

    ``parts`` is the dict from :func:`per_dataset_ranks`. See
    :data:`OVERALL_DOC` for the two formulas. Raises ``ValueError`` for an
    unknown ``basis``.
    """
    if basis not in OVERALL_BASES:
        raise ValueError(
            f"overall must be one of {list(OVERALL_BASES)}, got {basis!r}")
    if basis == "rank":
        return compute_overall(mean_rank_matrix(parts))
    per_ds = pd.DataFrame({ds: compute_overall(mat) for ds, mat in parts.items()})
    return per_ds.mean(axis=1)
