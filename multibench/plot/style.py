"""Numeric helpers ported from the R bubble-plot code (helpers.R / scIB_knit_table.R).

This module is also the single source of truth for the cross-dataset summary
math shared by :func:`multibench.plot.bubble` and :func:`multibench.plot.bar`:
:func:`per_dataset_ranks`, :func:`mean_rank_matrix` and
:func:`overall_by_basis`. Both figures call these, so with the same
``overall=`` and the metrics of one family they order methods identically
(see the Notes of ``plot.build_table``).
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
    The bubble figure draws such a column in a neutral grey
    (:func:`constant_columns`); the numbers stay as in R.
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


#: ``source`` values of the stored tables (``mtb.load_results``)
STORED_SOURCES = ("published", "rerun")


def stored_datasets(df: pd.DataFrame) -> tuple:
    """Datasets that hold rows of a stored table (``source`` published or rerun).

    ``()`` when the frame has no ``dataset`` or ``source`` column. The
    cross-dataset warnings use it to name the dataset a user's own method
    should be scored on.
    """
    if not {"dataset", "source"} <= set(df.columns):
        return ()
    mask = df["source"].isin(STORED_SOURCES)
    return tuple(sorted(map(str, df.loc[mask, "dataset"].dropna().unique())))


def _no_overlap_message(parts: dict, stored=()) -> str | None:
    """The message for datasets that share no method, else ``None``.

    With stored rows in some datasets and a user's rows in others, the
    remedy names the stored dataset(s) to score the user's method on. With
    no stored rows at all, the datasets may be two folders of the same cells
    (a peak run and a gene-activity run), so one sentence says how to merge
    them.
    """
    n = len(parts)
    cov = coverage(parts)
    if not (n > 1 and len(cov) and (cov <= 1).all()):
        return None
    names = [str(ds) for ds in parts]
    stored = [ds for ds in names if ds in set(map(str, stored))]
    own = [ds for ds in names if ds not in stored]
    if stored and own:
        remedy = (f"score your method on {' or '.join(stored)} and add it to "
                  f"that table")
    else:
        remedy = "score the same methods on every dataset"
    same_cells = ("" if stored else
                  " If these datasets hold the same cells, give their rows one "
                  "dataset name first.")
    return (f"rows come from {n} datasets ({', '.join(names)}) that share no "
            f"method, so the figure ranks unrelated rows against each other. "
            f"Plot each dataset on its own, or {remedy}.{same_cells}")


def _and(names) -> str:
    """``'A'``, ``'A and B'``, ``'A, B and C'``."""
    names = [str(n) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _lone_method_messages(parts: dict, consequence: str) -> list:
    """One message per dataset that holds one method while the frame holds several."""
    if len(coverage(parts)) <= 1:
        return []
    return [f"dataset {ds} has only one method ({mat.index[0]}), so {consequence}. "
            f"Plot it with methods scored on the same dataset."
            for ds, mat in parts.items() if len(mat.index) == 1]


def coverage_warnings(parts: dict, *, basis: str, incomplete_fix: str,
                      stored=()) -> list:
    """The warnings a cross-dataset summary of ``parts`` needs, as messages.

    ``parts`` is the dict from :func:`per_dataset_ranks`; ``basis`` the
    ``overall=`` in use; ``incomplete_fix`` the last sentence of the
    incomplete-matrix message (it names the caller's own remedy);
    ``stored`` the datasets holding stored rows (:func:`stored_datasets`).
    Shared by ``plot.build_table`` (``aggregate="summary"``) and ``plot.bar``.

    - no method is scored on two of several datasets: one message that the
      figure ranks unrelated rows (it replaces the incomplete-matrix one);
    - otherwise, a method missing from some dataset: the incomplete-matrix
      message;
    - a dataset that holds one method while the frame holds several: one
      message per such dataset, since that method is ranked against nothing.
    """
    from .. import config
    out = []
    n = len(parts)
    cov = coverage(parts)
    none_shared = _no_overlap_message(parts, stored)
    if none_shared:
        out.append(none_shared)
    elif n > 1 and (cov < n).any():
        part = cov[cov < n].sort_values()
        by_count: dict = {}
        for m, c in part.items():
            by_count.setdefault(int(c), []).append(str(m))
        seen = " ".join(
            f"{_and(ms)} {'has' if len(ms) == 1 else 'have'} scores on {c} of them."
            for c, ms in by_count.items())
        out.append(
            f"The summary ranks {n} datasets. {seen} With "
            + config.hint("overall='rank'", "--overall rank")
            + ", a missing dataset counts as rank 0, the lowest. With "
            + config.hint("overall='mean_overall'", "--overall mean_overall")
            + ", the missing dataset is left out. " + incomplete_fix)
    out += _lone_method_messages(
        parts, "its Overall there is always 1.0" if basis == "mean_overall"
        else "its rank there is always the lowest")
    return out


def dataset_mode_warnings(parts: dict, *, stored=()) -> list:
    """The warnings a figure of raw values from several datasets needs.

    ``aggregate="dataset"`` of ``plot.build_table``: rows from different
    datasets are ranked against each other. ``aggregate="summary"`` is
    suggested only when every method has rows in at least two datasets and
    no dataset holds a single method; otherwise the summary would warn in
    turn.
    """
    from .. import config
    n = len(parts)
    if n <= 1:
        return []
    lone = _lone_method_messages(
        parts, "its row is ranked against rows from other datasets")
    none_shared = _no_overlap_message(parts, stored)
    if none_shared:
        return [none_shared] + lone
    names = ", ".join(map(str, parts))
    msg = (config.hint("aggregate='dataset' but the frame",
                       "--aggregate dataset but the table")
           + f" holds {n} datasets ({names}): values are averaged per method "
           "across them and rows mix datasets. ")
    if (coverage(parts) >= 2).all() and not lone:
        msg += config.hint(
            "Pass aggregate='summary' for the paper's rank-averaged panel, or "
            "filter to one dataset.",
            "Pass --aggregate summary for the paper's rank-averaged panel, or "
            "filter with --dataset.")
    else:
        msg += config.hint("Plot each dataset on its own.",
                           "Plot each dataset on its own with --dataset.")
    return [msg] + lone


def constant_columns(mat: pd.DataFrame) -> dict:
    """``{column: value}`` for the columns whose values are all equal.

    Values are compared after rounding to :data:`TIE_DECIMALS` places; NaN
    cells are ignored and an all-NaN column is left out. A one-row frame
    lists every column that has a value.
    """
    out = {}
    for col in mat.columns:
        v = _tie_round(mat[col].to_numpy())
        v = v[np.isfinite(v)]
        if len(v) and (v == v[0]).all():
            out[col] = float(v[0])
    return out


#: metrics whose value depends on the Leiden backend of the clustering sweep
LEIDEN_METRICS = ("ARI", "NMI", "iF1")


def backend_warning(df: pd.DataFrame) -> str | None:
    """The message for igraph-scored rows plotted next to stored rows, else ``None``.

    The stored tables were clustered with leidenalg. A row is affected when
    its ``scored_with`` starts with ``igraph/`` and its metric came from the
    sweep: iF1 always, ARI and NMI when the clusters part is ``sweep``.
    Rows without ``scored_with`` (NaN) never trigger it. With a ``dataset``
    column, a row counts only when the frame holds a stored row of the same
    dataset: rows from a dataset with no stored rows are compared with no
    stored score, and the no-overlap warning covers them.
    """
    from .. import config
    if not {"source", "scored_with", "metric", "method"} <= set(df.columns):
        return None
    stored = df["source"].isin(STORED_SOURCES)
    if not stored.any():
        return None
    parts = df["scored_with"].astype("string").str.split("/")
    flavor = parts.str[0].fillna("")
    clusters = parts.str[1].fillna("")
    hit = (flavor == "igraph") & (
        (df["metric"] == "iF1")
        | (df["metric"].isin(["ARI", "NMI"]) & (clusters == "sweep")))
    if "dataset" in df.columns:
        hit &= df["dataset"].isin(df.loc[stored, "dataset"].dropna().unique())
    if not hit.any():
        return None
    names = ", ".join(sorted(map(str, df.loc[hit, "method"].unique())))
    return (f"rows for {names} were clustered with the igraph Leiden backend; "
            f"the stored tables used leidenalg, which can move ARI by up to "
            f"about 0.1. " + config.hint(
                "Set mtb.config.DEFAULT.leiden_flavor = 'leidenalg' before "
                "evaluate to compare them.",
                "Score them with multibench evaluate --leiden-flavor leidenalg "
                "to compare them."))


def overall_by_basis(parts: dict, basis: str = "rank") -> pd.Series:
    """Across-dataset Overall per method under the given ``basis``.

    ``parts`` is the dict from :func:`per_dataset_ranks`. The two formulas
    (spelled out again in the Notes of ``plot.build_table``):

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
