"""Numeric helpers ported from the R bubble-plot code (helpers.R / scIB_knit_table.R).

:func:`overall` is the one overall score of ``mtb.recommend``, ``plot.bar``
and ``plot.bubble``.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd

from .._text import _and_list

#: values that agree to this many decimal places are ties: floating-point
#: noise (an iLISI of 2.2e-16 next to 0.0) must not decide a rank or a fill
TIE_DECIMALS = 9


def _tie_round(x) -> np.ndarray:
    """``x`` as floats rounded to :data:`TIE_DECIMALS` places (NaN kept)."""
    return np.round(np.asarray(x, dtype=float), TIE_DECIMALS)


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1]; a constant vector -> ones (as in R); NaN stays NaN.

    A column whose values agree to :data:`TIE_DECIMALS` places counts as
    constant, so one that differs only by floating-point noise is all ones.
    The bubble figure draws such a column in a neutral grey
    (:func:`constant_columns`); the numbers stay as in R. Other columns are
    scaled from the unrounded values: rounding first would move a value of a
    narrow column (0.99995 to 0.99999) by up to 1e-9 / its range.
    """
    x = np.asarray(x, dtype=float)
    r = _tie_round(x)
    if not np.isfinite(r).any() or np.nanmin(r) == np.nanmax(r):
        return np.where(np.isnan(x), np.nan, 1.0)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo)


def rank_max(x: np.ndarray) -> np.ndarray:
    """Rank ascending with ties assigned the maximum rank (R ties.method='max').

    Values that agree to :data:`TIE_DECIMALS` places share a rank.
    """
    s = pd.Series(_tie_round(x))
    return s.rank(method="max").to_numpy()


# cross-dataset summary math, shared by bubble and bar

def per_dataset_ranks(long_df: pd.DataFrame, metrics=None) -> dict:
    """Within-dataset max-ranks: ``{dataset: DataFrame(method x metric)}``.

    For each dataset the frame is pivoted to method x metric (mean over
    duplicate rows), metrics that dataset never computed are dropped, and
    every remaining column is replaced by its max-rank (1 = worst ... n =
    best, ties share the maximum rank). A metric only one of three or more
    methods has is dropped too (:func:`lone_columns`); ``attrs["lone"]``
    names it. A frame without a ``dataset`` column
    is treated as one dataset named ``"all"``. Datasets whose pivot is empty
    are skipped.
    """
    rows = long_df.copy()
    if metrics:
        rows = rows[rows["metric"].isin(metrics)]
    if rows.empty:
        raise ValueError("no rows left after filtering to metrics=%r" % (metrics,))
    if "dataset" not in rows.columns:
        rows = rows.assign(dataset="all")
    out = {}
    for ds, group in rows.groupby("dataset", sort=True):
        mat = group.pivot_table(index="method", columns="metric", values="value",
                                aggfunc="mean")
        mat = mat.dropna(axis=1, how="all")
        if mat.empty:
            continue
        # a metric only one method has ranks nothing, as recommend's min-max rule
        lone = lone_columns(mat)
        mat = mat.drop(columns=lone)
        ranked = mat.apply(lambda col: rank_max(col.to_numpy()), axis=0)
        ranked.attrs["lone"] = lone
        out[ds] = ranked
    if not out:
        raise ValueError("no dataset had usable metrics")
    return out


def lone_columns(mat: pd.DataFrame) -> list:
    """Columns of a method x metric frame with a value for one method only,
    when the frame holds three or more methods (with two, a gap is an n/a
    cell); ``[]`` otherwise."""
    if len(mat.index) < 3:
        return []
    return [c for c in mat.columns if mat[c].notna().sum() < 2]


def lone_message(lone: dict) -> str | None:
    """The warning for ``{dataset or None: [metric, ...]}`` columns that one
    method has: they do not count in the Overall."""
    cols = list(dict.fromkeys(c for cs in lone.values() for c in cs))
    if not cols:
        return None
    one = len(cols) == 1
    return (f"{_and_list(cols)} {'has' if one else 'have'} a value for one method only, "
            f"so {'it does' if one else 'they do'} not count in the Overall.")


def coverage(parts: dict) -> pd.Series:
    """Number of datasets (keys of ``parts``) each method appears in."""
    counts: dict = {}
    for mat in parts.values():
        for m in mat.index:
            counts[m] = counts.get(m, 0) + 1
    return pd.Series(counts, dtype=int).sort_index()


#: ``source`` values of the stored tables (``mtb.load_results``)
STORED_SOURCES = ("stored", "rerun")


#: a value of your rows this close to a stored value is that value: re-scoring noise
SNAP = 1e-6


def stored_pairs() -> frozenset:
    """``{(dataset, method)}`` of the stored results ``mtb.load_results`` reads
    (under ``_settings._RESULT_PATH``)."""
    from .._data.results import _base_path
    base = _base_path()
    # the folders' times: tables written after a first call are read again
    stamp = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in (base, base / "rerun"))
    return _pairs_under(str(base), stamp)


@functools.lru_cache(maxsize=8)
def _pairs_under(base: str, stamp: tuple) -> frozenset:
    from .._data.results import _load_results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # load_results says them when it is called
        try:
            rows = _load_results(None)
        except FileNotFoundError:
            return frozenset()
    return frozenset(zip(rows["dataset"].astype(str), rows["method"].astype(str)))


def overall(df: pd.DataFrame, metrics=None, *, stored=None) -> pd.Series:
    """The overall score per method, best first (ties by name), of
    ``mtb.recommend``, ``plot.bar`` and ``plot.bubble``.

    Per dataset, each value is ranked against the stored rows (all rows when
    the table has none): the number of stored values it reaches, plus 0.5 when
    it equals none. The mean rank over the metrics the stored rows have is
    min-max scaled with the stored rows' range (1 when they tie), then averaged
    over the datasets. Stored rows so keep their article scores, a row equal to
    a stored row gets its score, and one better than all scores above 1.

    A stored row is marked stored (``source``) and is a row of the stored
    results (``stored``: their :func:`stored_pairs`); a copy under another
    name is your row. A value of your rows within :data:`SNAP` of the stored
    value of its method (``"Cobolt (this run)"`` is Cobolt's) on the same
    dataset and metric counts as that value.
    """
    from ..engine.registry import method_of
    if metrics is not None:
        df = df[df["metric"].isin(metrics)]
    df = df.assign(dataset=df["dataset"].astype(str) if "dataset" in df.columns else "all")
    marked = df["source"].isin(STORED_SOURCES) if "source" in df.columns else False
    if np.any(marked):
        pairs = stored_pairs() if stored is None else stored
        marked &= np.array([(d, str(m)) in pairs
                            for d, m in zip(df["dataset"], df["method"])], dtype=bool)
    df = df.assign(_stored=marked)
    have = sorted(set(df.loc[df["_stored"], "dataset"]))
    alone = sorted(set(df["dataset"]) - set(have))
    fix = " or ".join(f"dataset={d!r}" for d in have)
    if have and alone == ["user"]:
        raise ValueError(f"Your rows were scored without a dataset: score your embedding "
                         f"with {fix} to compare it with the stored results.")
    if have and alone:
        raise ValueError(
            f"Your rows are on {_and_list(alone)}, the stored results on {_and_list(have)}: "
            f"plot them apart, or score your embedding with {fix}.")
    per_ds = {}
    for ds, g in df.groupby("dataset"):
        m = g.pivot_table(index="method", columns="metric", values="value").round(TIE_DECIMALS)
        anchor = m.loc[g.loc[g["_stored"], "method"].unique()] if have else m
        if have:
            for name in m.index.difference(anchor.index):
                twin = method_of(name)          # your run of a stored method
                if twin in anchor.index:
                    ref, v = anchor.loc[twin], m.loc[name]
                    m.loc[name] = np.where(np.abs(v - ref) <= SNAP, ref, v)
        anchor = anchor.loc[:, anchor.notna().sum() >= min(2, len(anchor))]
        rank = pd.DataFrame({c: [(anchor[c] <= v).sum() + 0.5 * (v not in anchor[c].values)
                                 if pd.notna(v) else np.nan for v in m[c]] for c in anchor},
                            index=m.index)
        mean = rank.mean(axis=1)
        lo, hi = mean[anchor.index].min(), mean[anchor.index].max()
        per_ds[ds] = (mean - lo) / (hi - lo) if hi > lo else mean - lo + 1
    s = pd.DataFrame(per_ds).mean(axis=1).dropna()
    return s.sort_index(key=lambda i: i.astype(str)).sort_values(ascending=False, kind="stable")


def _no_overlap_message(parts: dict) -> str | None:
    """The message for datasets that share no method, else ``None``."""
    n = len(parts)
    per_method = coverage(parts)
    if not (n > 1 and (per_method <= 1).all()):
        return None
    names = [str(ds) for ds in parts]
    return (f"The rows come from {n} datasets, {_and_list(names)}, that share no method. "
            f"The figure ranks unrelated rows against each other. "
            f"Plot each dataset on its own.")


def _lone_method_messages(parts: dict, consequence: str) -> list:
    """One message per dataset that holds one method while the frame holds several."""
    if len(coverage(parts)) <= 1:
        return []
    return [f"Dataset {ds} has only one method, {mat.index[0]}. "
            f"{consequence[:1].upper()}{consequence[1:]}. "
            f"Plot it with methods scored on the same dataset."
            for ds, mat in parts.items() if len(mat.index) == 1]


def coverage_warnings(parts: dict, *, basis: str, incomplete_fix: str) -> list:
    """The warnings a cross-dataset summary of ``parts`` needs, as messages.

    ``parts`` is ``{dataset: method x metric frame}``, as :func:`per_dataset_ranks`
    returns, ``basis`` the ``overall=`` in use and ``incomplete_fix`` the
    caller's remedy sentence.
    One message when no method spans two datasets, else one when a method
    misses a dataset; plus one per dataset that holds a single method.
    """
    out = []
    n = len(parts)
    per_method = coverage(parts)
    none_shared = _no_overlap_message(parts)
    if none_shared:
        out.append(none_shared)
    elif n > 1 and (per_method < n).any():
        partial = per_method[per_method < n].sort_values()
        by_count: dict = {}
        for m, c in partial.items():
            by_count.setdefault(int(c), []).append(str(m))
        seen = " ".join(
            f"{_and_list(ms)} {'has' if len(ms) == 1 else 'have'} scores on {c} of them."
            for c, ms in by_count.items())
        missing = ("A missing dataset is left out." if basis == "mean_overall"
                   else "A missing dataset counts as rank 0, the lowest.")
        out.append(f"The summary ranks {n} datasets. {seen} {missing} {incomplete_fix}")
    out += _lone_method_messages(
        parts, "its Overall there is always 1.0" if basis == "mean_overall"
        else "its rank there is always the lowest")
    return out


def dataset_mode_warnings(parts: dict) -> list:
    """The warnings a figure of raw values from several datasets needs.

    The bubble figure ranks rows from different datasets against each other.
    ``mtb.plot.bar`` is suggested only when every method has rows in at least
    two datasets and no dataset holds a single method.
    """
    n = len(parts)
    if n <= 1:
        return []
    lone = _lone_method_messages(
        parts, "its row is ranked against rows from other datasets")
    none_shared = _no_overlap_message(parts)
    if none_shared:
        return [none_shared] + lone
    msg = (f"This figure averages each method over {n} datasets, {_and_list(parts)}. "
           "Its rows mix datasets. ")
    if (coverage(parts) >= 2).all() and not lone:
        msg += ("mtb.plot.bar ranks the methods within each dataset; or filter to one "
            "dataset.")
    else:
        msg += "Plot each dataset on its own."
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


#: the clusters part of a 0.4.0 ``scored_with``: the article's resolution
#: search, the user's own clusters, or none (classification only)
SCORING_PROTOCOLS = ("search", "user", "none")
#: first package version that scores with :data:`SCORING_PROTOCOL`
SCORING_VERSION = (0, 4, 0)


def _version_tuple(text) -> tuple:
    """``"0.3.9"`` -> ``(0, 3, 9)``; ``()`` when ``text`` holds no version."""
    import re
    m = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", str(text))
    return tuple(int(g or 0) for g in m.groups()) if m else ()


def old_scoring(scored_with) -> "pd.Series":
    """True where a ``scored_with`` value is missing or names a scoring older
    than 0.4.0 (another clustering protocol or version); ``"unknown"`` (a frame
    without a record, which ``to_long`` warns about) is not flagged."""
    col = pd.Series(scored_with).astype("string")
    parts = col.str.split("/")

    def old(p) -> bool:
        if not isinstance(p, list):
            return True
        if p == ["unknown"]:
            return False
        if len(p) < 3:
            return True
        v = _version_tuple(p[2])
        return p[1] not in SCORING_PROTOCOLS or not v or v < SCORING_VERSION
    return parts.map(old).astype(bool)


#: the warning for rows scored before 0.4.0 next to the stored tables
OLD_SCORING = ("The rows for {names} were scored before multibench 0.4.0. "
               "Score them again to compare them with the stored results.")
#: the same for rows that do not record their scoring
NO_SCORING = ("The rows for {names} do not record how they were scored (no "
              "scored_with). Score them with multibench 0.4.0 to compare them with "
              "the stored results.")


def old_scoring_warning(df: pd.DataFrame) -> str | None:
    """The message for rows of your own, scored before 0.4.0 or with no
    ``scored_with``, plotted next to the stored rows; else ``None``."""
    if not {"source", "metric", "method"} <= set(df.columns):
        return None
    stored = df["source"].isin(STORED_SOURCES)
    if not stored.any() or stored.all():
        return None
    own = df[~stored]
    sw = own["scored_with"] if "scored_with" in own.columns \
        else pd.Series(pd.NA, index=own.index)
    hit = old_scoring(sw).to_numpy()
    if not hit.any():
        return None
    blank = sw.isna().to_numpy()
    names = ", ".join(sorted(map(str, own.loc[hit, "method"].unique())))
    return (NO_SCORING if blank[hit].all() else OLD_SCORING).format(names=names)


def backend_warning(df: pd.DataFrame) -> str | None:
    """The message for igraph-scored rows plotted next to stored rows, else ``None``.

    Rows of your own scored before 0.4.0, or without ``scored_with``, get
    :func:`old_scoring_warning`'s message instead.

    The stored tables were clustered with leidenalg. A row is affected when
    its ``scored_with`` starts with ``igraph/`` and its metric came from
    Leiden: iF1, ARI_batch and NMI_batch always; ARI, NMI, ASW and iASW when
    the clusters part is ``search`` (ARI and NMI also ``sweep``, before 0.4).
    Rows without ``scored_with`` (NaN) never trigger the igraph check.
    """
    old = old_scoring_warning(df)
    if old:
        return old
    if not {"source", "scored_with", "metric", "method"} <= set(df.columns):
        return None
    stored = df["source"].isin(STORED_SOURCES)
    if not stored.any():
        return None
    parts = df["scored_with"].astype("string").str.split("/")
    flavor = parts.str[0].fillna("")
    clusters = parts.str[1].fillna("")
    metric = df["metric"]
    hit = (flavor == "igraph") & (
        metric.isin(["iF1", "ARI_batch", "NMI_batch"])
        | (metric.isin(["ARI", "NMI", "ASW", "iASW"]) & (clusters == "search"))
        | (metric.isin(["ARI", "NMI"]) & (clusters == "sweep")))
    if not hit.any():
        return None
    names = ", ".join(sorted(map(str, df.loc[hit, "method"].unique())))
    return (f"The rows for {names} were clustered with the igraph Leiden backend. "
            f"The stored results used leidenalg, which can move ARI by up to "
            f"about 0.1. " + ("Set mtb.config.leiden_flavor = 'leidenalg' before "
                "evaluate to compare them."))


#: colour ramp names the package defines itself, on top of matplotlib's
PINKS = "Pinks"


def colormap(name):
    """The matplotlib colormap ``name``, or ``"Pinks"``: the pink half of
    ColorBrewer's PiYG, from near-white to deep magenta, which is the
    classification family's ramp as blues and greens are the other two
    families'. A colormap object is returned as it is."""
    from matplotlib import colormaps, colors
    if not isinstance(name, str):
        return name
    if name == PINKS:
        ramp = colormaps["PiYG"](np.linspace(0.5, 0.0, 256))
        return colors.ListedColormap(ramp, name=PINKS)
    return colormaps[name]
