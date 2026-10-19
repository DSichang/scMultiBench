"""Numeric helpers ported from the R bubble-plot code (helpers.R / scIB_knit_table.R).

:func:`overall` is the one overall score of ``mtb.recommend``, ``plot.bar``
and ``plot.bubble``.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from ..config import _and_list

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


def compute_overall(mat: pd.DataFrame) -> pd.Series:
    """overall = minmax(mean over columns of per-column max-rank)."""
    ranks = mat.apply(lambda col: rank_max(col.to_numpy()), axis=0)
    mean_rank = ranks.mean(axis=1)
    return pd.Series(minmax(mean_rank.to_numpy()), index=mat.index)


# cross-dataset summary math, shared by bubble and bar

#: the deprecated ``overall=`` of the figures: the 0.3 across-dataset formulas
OVERALL_BASES = ("rank", "mean_overall")


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


def mean_rank_matrix(parts: dict) -> pd.DataFrame:
    """Average the per-dataset rank matrices from :func:`per_dataset_ranks`.

    Rows are the union of methods; a method absent from a dataset, or lacking
    a metric there, scores rank 0 (as in the paper's summary), but only for
    the metrics that dataset computed - each metric is averaged over the
    datasets that have it.
    """
    mats = list(parts.values())
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
STORED_SOURCES = ("published", "stored", "rerun")


@functools.lru_cache(maxsize=1)
def _shipped_pairs() -> frozenset:
    """``{(dataset, method)}`` of the tables the package ships."""
    from ..data.results import _results_coverage
    cov = _results_coverage(source="both")
    return frozenset(zip(cov["dataset"].astype(str), cov["method"].astype(str)))


#: a value of your rows this close to a stored value is that value: re-scoring noise
SNAP = 1e-6


def overall(df: pd.DataFrame, metrics=None) -> pd.Series:
    """The overall score per method, best first (ties by name), of
    ``mtb.recommend``, ``plot.bar`` and ``plot.bubble``.

    Per dataset, each value is ranked against the stored rows (all rows when
    the table has none): the number of stored values it reaches, plus 0.5 when
    it equals none. The mean rank over the metrics the stored rows have is
    min-max scaled with the stored rows' range (1 when they tie), then averaged
    over the datasets. Stored rows so keep their article scores, a row equal to
    a stored row gets its score, and one better than all scores above 1.
    A stored row is a row of a shipped table (``source`` "stored" and its
    method and dataset); a value of your rows within :data:`SNAP` of a stored
    value counts as that value.
    """
    from .. import config
    if metrics is not None:
        df = df[df["metric"].isin(metrics)]
    df = df.assign(dataset=df["dataset"].astype(str) if "dataset" in df.columns else "all")
    stored = df["source"].isin(STORED_SOURCES) if "source" in df.columns else False
    if np.any(stored):
        pairs = _shipped_pairs()
        stored &= np.array([(d, str(m)) in pairs for d, m in zip(df["dataset"], df["method"])])
    df = df.assign(_stored=stored)
    have = sorted(set(df.loc[df["_stored"], "dataset"]))
    alone = sorted(set(df["dataset"]) - set(have))
    fix = config.hint(" or ".join(f"dataset={d!r}" for d in have),
                      " or ".join(f"multibench evaluate --dataset {d}" for d in have))
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
        own = ~m.index.isin(anchor.index)
        for c in anchor if have else ():
            ref = np.unique(anchor[c].dropna().to_numpy())
            v = m.loc[own, c].to_numpy()
            if len(ref) and len(v):
                near = ref[np.abs(np.nan_to_num(v, nan=np.inf)[:, None] - ref).argmin(axis=1)]
                m.loc[own, c] = np.where(np.abs(v - near) <= SNAP, near, v)
        anchor = anchor.loc[:, anchor.notna().sum() >= min(2, len(anchor))]
        rank = pd.DataFrame({c: [(anchor[c] <= v).sum() + 0.5 * (v not in anchor[c].values)
                                 if pd.notna(v) else np.nan for v in m[c]] for c in anchor},
                            index=m.index)
        mean = rank.mean(axis=1)
        lo, hi = mean[anchor.index].min(), mean[anchor.index].max()
        per_ds[ds] = (mean - lo) / (hi - lo) if hi > lo else mean - lo + 1
    s = pd.DataFrame(per_ds).mean(axis=1).dropna()
    return s.sort_index(key=lambda i: i.astype(str)).sort_values(ascending=False, kind="stable")


def stored_datasets(df: pd.DataFrame) -> tuple:
    """Datasets that hold rows of a stored table (``source`` published or rerun).

    ``()`` when the frame has no ``dataset`` or ``source`` column.
    """
    if not {"dataset", "source"} <= set(df.columns):
        return ()
    mask = df["source"].isin(STORED_SOURCES)
    return tuple(sorted(map(str, df.loc[mask, "dataset"].dropna().unique())))


def _no_overlap_message(parts: dict, stored=()) -> str | None:
    """The message for datasets that share no method, else ``None``. Without
    stored rows (``stored``: their datasets) the datasets may be two folders of
    the same cells, so it says how to merge them."""
    n = len(parts)
    per_method = coverage(parts)
    if not (n > 1 and (per_method <= 1).all()):
        return None
    names = [str(ds) for ds in parts]
    same_cells = ("" if stored else
                  " If these datasets hold the same cells, give their rows one "
                  "dataset name first.")
    return (f"The rows come from {n} datasets, {_and_list(names)}, that share no method. "
            f"The figure ranks unrelated rows against each other. "
            f"Plot each dataset on its own, or score the same methods on every "
            f"dataset.{same_cells}")


def _lone_method_messages(parts: dict, consequence: str) -> list:
    """One message per dataset that holds one method while the frame holds several."""
    if len(coverage(parts)) <= 1:
        return []
    return [f"Dataset {ds} has only one method, {mat.index[0]}. "
            f"{consequence[:1].upper()}{consequence[1:]}. "
            f"Plot it with methods scored on the same dataset."
            for ds, mat in parts.items() if len(mat.index) == 1]


def coverage_warnings(parts: dict, *, basis: str, incomplete_fix: str,
                      stored=()) -> list:
    """The warnings a cross-dataset summary of ``parts`` needs, as messages.

    ``parts`` is the dict from :func:`per_dataset_ranks`, ``basis`` the
    ``overall=`` in use, ``incomplete_fix`` the caller's remedy sentence and
    ``stored`` the datasets holding stored rows (:func:`stored_datasets`).
    One message when no method spans two datasets, else one when a method
    misses a dataset; plus one per dataset that holds a single method.
    """
    out = []
    n = len(parts)
    per_method = coverage(parts)
    none_shared = _no_overlap_message(parts, stored)
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
    msg = (f"This figure averages each method over {n} datasets, {_and_list(parts)}. "
           "Its rows mix datasets. ")
    if (coverage(parts) >= 2).all() and not lone:
        msg += config.hint(
            "mtb.plot.bar ranks the methods within each dataset; or filter to one "
            "dataset.",
            "multibench plot bar ranks the methods within each dataset; or filter "
            "with --dataset.")
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
    from .. import config
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
            f"about 0.1. " + config.hint(
                "Set mtb.config.DEFAULT.leiden_flavor = 'leidenalg' before "
                "evaluate to compare them.",
                "Score them with multibench evaluate --leiden-flavor leidenalg "
                "to compare them."))


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
