"""Numeric helpers ported from the R bubble-plot code (helpers.R / scIB_knit_table.R).

:func:`overall` is the one overall score of ``mtb.rank``, ``plot.bar``
and ``plot.bubble``.
"""
from __future__ import annotations

import functools
import warnings
from collections import Counter

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

def dataset_tables(long_df: pd.DataFrame) -> dict:
    """``{dataset: method x metric}`` means of a long table, for the warnings of
    a summary over datasets; a frame without ``dataset`` is one dataset
    ``"all"``, and a dataset with no value is left out."""
    groups = (long_df.groupby("dataset", sort=True) if "dataset" in long_df.columns
              else [("all", long_df)])
    return {ds: mat for ds, g in groups
            if not (mat := g.pivot_table(index="method", columns="metric",
                                         values="value")).empty}


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
    counts = Counter(m for mat in parts.values() for m in mat.index)
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
    ``mtb.rank``, ``plot.bar`` and ``plot.bubble``.

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
        raise ValueError(f"Your rows were scored without a dataset. If they hold the cells "
                         f"of {_and_list(have, 'or')}, score your embedding with {fix}; "
                         f"else plot them apart.")
    if have and alone:
        raise ValueError(
            f"Your rows are on {_and_list(alone)}, the stored results on {_and_list(have)}: "
            f"plot them apart.")
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


def coverage_warnings(parts: dict) -> list:
    """The warnings ``plot.bar``'s summary of ``parts`` needs, as messages.

    ``parts`` is ``{dataset: method x metric frame}``, as :func:`dataset_tables`
    returns. One message when no method spans two datasets, else one when a
    method misses a dataset; plus one per dataset that holds a single method.
    """
    n = len(parts)
    per_method = coverage(parts)
    none_shared = _no_overlap_message(parts)
    out = [none_shared] if none_shared else []
    if not none_shared and n > 1 and (per_method < n).any():
        partial = [str(m) for m in per_method[per_method < n].index]
        out.append(f"{_and_list(partial)} {'has' if len(partial) == 1 else 'have'} scores on "
                   f"only some of the {n} datasets. Pass methods= with the methods scored "
                   f"on every dataset.")
    out += _lone_method_messages(parts, "its Overall there is always 1.0")
    return out


def dataset_mode_warnings(parts: dict) -> list:
    """The warnings a figure of raw values from several datasets needs.

    The bubble figure ranks rows from different datasets against each other.
    """
    n = len(parts)
    if n <= 1:
        return []
    lone = _lone_method_messages(
        parts, "its row is ranked against rows from other datasets")
    none_shared = _no_overlap_message(parts)
    if none_shared:
        return [none_shared] + lone
    return [f"This figure averages each method over {n} datasets, {_and_list(parts)}. "
            f"Plot one dataset, or use mtb.plot.bar."] + lone


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
#: first package version that scores with :data:`SCORING_PROTOCOLS`
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
    names = ", ".join(sorted(map(str, own.loc[hit, "method"].unique())))
    return (f"The rows for {names} were not scored by multibench 0.4.0. Score them again "
            f"to compare them with the stored results.")


def backend_warning(df: pd.DataFrame) -> str | None:
    """The message for rows of your own clustered with igraph (their Leiden
    metrics) next to the stored rows, which used leidenalg; first
    :func:`old_scoring_warning`'s; else ``None``."""
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
        | (metric.isin(["ARI", "NMI", "ASW", "iASW"]) & (clusters == "search")))
    if not hit.any():
        return None
    names = ", ".join(sorted(map(str, df.loc[hit, "method"].unique())))
    return (f"The rows for {names} were clustered with igraph, the stored results with "
            f"leidenalg. Set mtb.config.leiden_flavor = 'leidenalg' and evaluate again to "
            f"compare them.")


#: colour ramp names the package defines itself, on top of matplotlib's
PINKS = "Pinks"


def colormap(name: str):
    """The matplotlib colormap ``name``, or ``"Pinks"``: the pink half of
    ColorBrewer's PiYG, from near-white to deep magenta, which is the
    classification family's ramp as blues and greens are the other two
    families'."""
    from matplotlib import colormaps, colors
    if name == PINKS:
        ramp = colormaps["PiYG"](np.linspace(0.5, 0.0, 256))
        return colors.ListedColormap(ramp, name=PINKS)
    return colormaps[name]
