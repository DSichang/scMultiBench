"""Load benchmark metric tables into a tidy long DataFrame.

Two sources ship with the package (``multibench/result/``):

* ``published`` - the benchmark's own scIB metric tables, one
  ``<category>/<dataset>/<method>/metric*.csv`` per run
  (``result/scib_metric/``; mosaic has none). A few cross runs keep the
  table one level deeper, in a run-configuration subfolder
  (``D56/MOFA2/filtered3/metric.csv``, ``D56/MOFA2/kmeans/metric_kmeans.csv``,
  ``D53/MOFA2/8000HVG/metric.csv``); those are read too (``kbet/`` folders,
  which hold raw kBET output, are not metric tables and are skipped);
* ``rerun`` - the package's re-run sweeps behind the tutorial figures
  (``result/rerun/long_all_<dataset>.csv``; D11/D11s, D28/D28s, D45/D45s,
  D52/D52s), stamped with the package version they were produced by.

Every frame returned here carries the same seven columns
``metric, value, method, dataset, category, clustering, source`` so frames
from either source (or your own, via :func:`multibench.to_long`) can be
concatenated and handed to ``mtb.plot.bubble`` / ``mtb.plot.bar``.

v1 implements the scib clustering + batch loader. Other metric sets raise a
clear "not in v1" error (declared but not wired).
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from .. import config
from . import catalog

__all__ = ["load_results", "available_datasets", "results_coverage", "recommend",
           "COLUMNS", "SOURCES"]

# clustering variant -> companion filename for the per-(dataset,method) dir
_CLUSTERING_FILES = {
    "default": "metric.csv",
    "louvain": "metric_louvain.csv",
    "kmeans": "metric_kmeans.csv",
}
_CORRECTION_FILE = "metric_asw_iasw_if1.csv"

#: the four integration categories (the set ``load_results`` accepts)
_CATEGORIES = ("cross", "diagonal", "mosaic", "vertical")

#: fixed column order of every frame ``load_results`` returns
COLUMNS = ["metric", "value", "method", "dataset", "category", "clustering", "source"]

#: valid values of the ``source=`` knob
SOURCES = ("published", "rerun", "both")

#: ``task=`` values: None/"all" = no filter, else a metric family
_TASKS = (None, "all", "clustering", "batch")

# a result directory like ``Concerto_louvain`` is the method's LOUVAIN variant:
# split it into method id + clustering token instead of inventing a method id
_SUFFIX_RE = re.compile(r"^(.*)_(louvain|kmeans)$")

#: sub-directory of ``result_path`` holding the package re-run sweeps
_RERUN_DIR = "rerun"
_RERUN_GLOB = "long_all_*.csv"


def _rerun_source() -> str:
    """Provenance stamp for re-run frames that lack a ``source`` column.

    The shipped tables carry their own stamp (``rerun-<version that produced
    them>``); this fallback is for a file without one. It deliberately does NOT
    use the running package's version: that would claim the current release
    produced numbers it merely read.
    """
    return "rerun"


def _read_metric_csv(path: Path) -> pd.DataFrame:
    """Read a metric.csv (unnamed index + Value col) into long metric/value rows."""
    raw = pd.read_csv(path)
    raw.columns = ["metric", "value"] + list(raw.columns[2:])
    raw = raw[["metric", "value"]].copy()
    raw["metric"] = raw["metric"].map(catalog.canonical_metric)
    raw = raw.dropna(subset=["metric"])
    return raw


def _as_list(x) -> list | None:
    if x is None:
        return None
    if isinstance(x, (str, bytes)):
        return [x]
    return list(x)


def _base_path(result_path) -> Path:
    return Path(result_path) if result_path is not None else config.DEFAULT.result_path


def _published_missing_msg(root: Path, base: Path, category: str) -> str:
    return (
        f"no published scIB metric tables under {root}. The tables ship inside "
        f"the multibench wheel at multibench/result/scib_metric; check "
        f"result_path= / mtb.config.DEFAULT.result_path (currently {base}). "
        f"mosaic has no published tables - use source='rerun' (package sweeps) "
        f"or load your own long CSV with result_path=<file>."
        + (f" (requested category={category!r})" if category else "")
    )


# --------------------------------------------------------------------------
# published tree
# --------------------------------------------------------------------------
# sub-folders of a method result dir that are NOT run configurations holding a
# metric table (raw kBET output lives in ``kbet/benchmark_results*.csv``)
_NON_TABLE_SUBDIRS = {"kbet"}


def _find_metric_file(m_dir: Path, fname: str) -> Path | None:
    """``<m_dir>/<fname>`` or, when absent, the same file one level down in a
    run-configuration subfolder (``MOFA2/filtered3/metric.csv``,
    ``MOFA2/kmeans/metric_kmeans.csv``, ``MOFA2/8000HVG/metric.csv``).

    Several nested candidates would be ambiguous: the first in sorted order is
    used and a ``UserWarning`` names the rest, so a silent pick never goes
    unnoticed. ``None`` when nothing matches.
    """
    direct = m_dir / fname
    if direct.exists():
        return direct
    nested = sorted(p for p in m_dir.glob(f"*/{fname}")
                    if p.parent.name not in _NON_TABLE_SUBDIRS and p.is_file())
    if not nested:
        return None
    if len(nested) > 1:
        warnings.warn(
            f"{m_dir}: {len(nested)} nested {fname} tables "
            f"({[q.parent.name for q in nested]}); using {nested[0].parent.name}",
            UserWarning, stacklevel=3)
    return nested[0]


def _iter_published(root: Path, datasets: list | None, clustering: str):
    """Yield ``(ds_dir, m_dir, metric_file, method_id, row_clustering, coalesce)``
    for every loadable (dataset, method) under a category root.

    ``m_dir`` is the folder the companion files (the ASW/iASW/iF1 correction
    table) are looked up in: the method dir itself, or the nested
    run-configuration folder the metric table was found in (see
    :func:`_find_metric_file`)."""
    fname = _CLUSTERING_FILES[clustering]
    if datasets:
        ds_dirs = [root / d for d in datasets]
    else:
        ds_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for ds_dir in ds_dirs:
        if not ds_dir.is_dir():
            continue
        for m_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            m = _SUFFIX_RE.match(m_dir.name)
            if m:
                # the directory IS the variant: its metric.csv holds the
                # louvain/kmeans result, so it contributes only when that
                # clustering is requested, under the method's canonical id
                if m.group(2) != clustering:
                    continue
                mfile = _find_metric_file(m_dir, "metric.csv")
                if mfile is None:
                    continue
                yield ds_dir, mfile.parent, mfile, catalog.canonical_id(m.group(1)), m.group(2), False
                continue
            mfile = _find_metric_file(m_dir, fname)
            if mfile is None:
                continue
            yield ds_dir, mfile.parent, mfile, catalog.canonical_id(m_dir.name), clustering, True


def _load_published(category: str, datasets: list | None, clustering: str,
                    base: Path, metric_set: str) -> pd.DataFrame:
    root = base / config.metric_set_dir(metric_set) / config.category_folder(category)
    if not root.exists():
        raise FileNotFoundError(_published_missing_msg(root, base, category))
    rows: list[pd.DataFrame] = []
    for ds_dir, m_dir, mfile, method, row_clust, coalesce in _iter_published(
            root, datasets, clustering):
        df = _read_metric_csv(mfile)
        # coalesce corrected ASW/iASW/iFI when present. The correction file
        # holds corrected values for the *default* clustering only; the
        # louvain/kmeans variant files already carry their own correct
        # ASW/iASW/iFI, so only coalesce for the default clustering.
        corr = m_dir / _CORRECTION_FILE
        if coalesce and clustering == "default" and corr.exists():
            cdf = _read_metric_csv(corr).set_index("metric")["value"]
            df["value"] = df.apply(lambda r: cdf.get(r["metric"], r["value"]), axis=1)
        df["method"] = method
        df["dataset"] = ds_dir.name
        df["category"] = category
        df["clustering"] = row_clust
        df["source"] = "published"
        rows.append(df)
    if not rows:
        have = sorted(p.name for p in root.iterdir() if p.is_dir())
        if datasets:
            raise FileNotFoundError(
                f"no published {_CLUSTERING_FILES[clustering]} for "
                f"{category}/{datasets if len(datasets) > 1 else datasets[0]} "
                f"under {root}; datasets with published tables: {have}")
        raise FileNotFoundError(
            f"no {_CLUSTERING_FILES[clustering]} found under {root} "
            f"(clustering={clustering!r})")
    return pd.concat(rows, ignore_index=True)[COLUMNS]


# --------------------------------------------------------------------------
# re-run sweeps
# --------------------------------------------------------------------------
def _read_rerun_files(base: Path) -> list[tuple[Path, pd.DataFrame]]:
    root = base / _RERUN_DIR
    out = []
    for f in sorted(root.glob(_RERUN_GLOB)) if root.is_dir() else []:
        df = pd.read_csv(f)
        if not {"metric", "value", "method", "dataset", "category"}.issubset(df.columns):
            continue
        if "source" not in df.columns:
            df["source"] = _rerun_source()
        if "clustering" not in df.columns:
            df["clustering"] = "default"
        out.append((f, df[COLUMNS]))
    return out


def _load_rerun(category: str | None, datasets: list | None, base: Path) -> pd.DataFrame:
    files = _read_rerun_files(base)
    root = base / _RERUN_DIR
    if not files:
        raise FileNotFoundError(
            f"no re-run sweeps ({_RERUN_GLOB}) under {root}. They ship inside the "
            f"multibench wheel at multibench/result/rerun; check result_path= / "
            f"mtb.config.DEFAULT.result_path (currently {base}).")
    frames = [df for _, df in files]
    allf = pd.concat(frames, ignore_index=True)
    avail = (allf[["category", "dataset"]].drop_duplicates()
             .sort_values(["category", "dataset"]))
    sel = allf
    if category is not None:
        sel = sel[sel["category"] == category]
    if datasets:
        sel = sel[sel["dataset"].isin(datasets)]
    if sel.empty:
        pairs = [f"{c}/{d}" for c, d in avail.itertuples(index=False)]
        raise FileNotFoundError(
            f"no re-run sweep for {category or 'any category'}/"
            f"{(datasets if len(datasets) > 1 else datasets[0]) if datasets else 'any dataset'}; "
            f"available: {pairs}")
    return sel.reset_index(drop=True)


# --------------------------------------------------------------------------
# a user's own long CSV
# --------------------------------------------------------------------------
def _load_long_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["metric", "value", "method"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is not a long results CSV: missing column(s) {missing}; "
            f"need at least {need} (plus optional dataset/category/clustering/"
            f"source). Found columns: {list(df.columns)}. Write such a file "
            f"with mtb.to_long(...).to_csv(path, index=False) or "
            f"BatchResult.save().")
    df = df.copy()
    df["metric"] = df["metric"].map(catalog.canonical_metric)
    df = df.dropna(subset=["metric"])
    if "dataset" not in df.columns:
        df["dataset"] = path.stem
    if "category" not in df.columns:
        df["category"] = "user"
    if "clustering" not in df.columns:
        df["clustering"] = "default"
    if "source" not in df.columns:
        df["source"] = "user"
    return df[COLUMNS]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def _task_metrics(task):
    if task not in _TASKS:
        raise ValueError(
            f"unknown task {task!r}; valid: None (all metrics), 'all', "
            f"'clustering', 'batch'")
    if task in (None, "all"):
        return None
    # the family lists live in plot.bar and are pinned to eval by
    # tests/test_metric_groups.py; import lazily (plot imports nothing from
    # data at import time, but keep the dependency one-directional at load)
    from ..plot.bar import BATCH_METRICS, CLUSTERING_METRICS
    return list(CLUSTERING_METRICS if task == "clustering" else BATCH_METRICS)


def load_results(
    category: str | None = None,
    task: str | None = None,
    metric_set: str = "scib",
    dataset: str | list[str] | None = None,
    method: str | list[str] | None = None,
    metric: list[str] | str | None = None,
    clustering: str = "default",
    result_path: Path | str | None = None,
    *,
    source: str = "published",
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Benchmark metric tables as a tidy long frame.

    Parameters
    ----------
    category : {"vertical", "diagonal", "mosaic", "cross"}, optional
        Integration category. ``None`` (default) loads every category that
        has tables for the requested ``source`` (an unknown token raises
        ``ValueError`` listing the valid ones).
    task : {None, "all", "clustering", "batch"}, optional
        Restrict to a metric family: ``"clustering"`` keeps
        ``mtb.plot.CLUSTERING_METRICS`` (ARI, NMI, ASW, iASW, iF1, cLISI),
        ``"batch"`` keeps ``mtb.plot.BATCH_METRICS`` (ASW_batch, GC, iLISI,
        kBET). ``None``/``"all"`` (default) keeps every metric. Anything else
        raises ``ValueError``.
    metric_set : str
        Only ``"scib"`` is wired (others raise ``NotImplementedError``).
    dataset : str or list of str, optional
        Dataset id(s), e.g. ``"D11"`` or ``["D11", "D11s"]``. Default: all
        datasets of the category. A dataset with no table raises
        ``FileNotFoundError`` naming the available ones.
    method : str or list of str, optional
        Keep only these method(s); alias tolerant (``"mofa+"`` -> MOFA2). If
        nothing matches, the frame is EMPTY and a ``UserWarning`` lists the
        methods present (a typo is not an error here - concatenating several
        calls would otherwise break).
    metric : str or list of str, optional
        Keep only these metric code(s); alias tolerant (``"ari"`` -> ARI).
        Same empty-frame-plus-warning rule as ``method``.
    clustering : {"default", "louvain", "kmeans"}
        Which clustering variant of the published tables to read
        (``metric.csv`` / ``metric_louvain.csv`` / ``metric_kmeans.csv``). A
        result directory named ``<method>_louvain`` / ``<method>_kmeans`` IS
        that variant: it is reported under the method's canonical id with
        ``clustering`` set to the suffix, and only when that variant is
        requested (so no method id ever ends in ``_louvain``/``_kmeans``). The
        re-run sweeps are ``"default"`` only.
    result_path : path-like, optional
        A results ROOT (holding ``scib_metric/`` and/or ``rerun/``; default
        ``mtb.config.DEFAULT.result_path``, i.e. the tables shipped in the
        package) OR a single long CSV file (columns ``metric, value, method``
        at least, e.g. written by ``to_long(...).to_csv`` or
        ``BatchResult.save()``). For a file, ``source`` is ignored and the
        frame's ``source`` column is whatever the file carries (``"user"``
        if none).
    source : {"published", "rerun", "both"}, keyword-only
        ``"published"`` (default): the benchmark's scIB tables under
        ``result_path/scib_metric`` (none for mosaic). ``"rerun"``: the
        package's re-run sweeps ``result_path/rerun/long_all_<dataset>.csv``
        (vertical D11/D11s, diagonal D28/D28s, mosaic D45/D45s, cross
        D52/D52s). ``"both"``: the concatenation - tell the two apart by the
        ``source`` column (``"published"`` vs ``"rerun-<version>"``).
    methods : list of str, keyword-only
        Alias for ``method`` (a list); passing both raises ``ValueError``.

    Returns
    -------
    pandas.DataFrame
        Columns, in this order: ``metric, value, method, dataset, category,
        clustering, source``. Method ids are canonical registry tokens
        (``MOFA+`` -> ``MOFA2``, ``Seurat(WNN)`` -> ``Seurat_WNN``); metric
        codes are canonical (``iFI`` -> ``iF1``).

    Raises
    ------
    FileNotFoundError
        No table for the requested category/dataset/source, with the path
        looked at and what IS available.
    ValueError
        Unknown ``category`` / ``task`` / ``clustering`` / ``source``, or a
        ``result_path`` file without the long-format columns.
    NotImplementedError
        ``metric_set`` other than ``"scib"``.

    Examples
    --------
    >>> pub = mtb.load_results("diagonal", dataset="D28")            # published
    >>> rr = mtb.load_results("diagonal", dataset="D28", source="rerun")
    >>> both = mtb.load_results("cross", dataset="D52", source="both")
    >>> mtb.plot.bubble(both[both.source != "published"])
    """
    if metric_set != "scib":
        raise NotImplementedError(
            f"metric_set={metric_set!r} is declared but not wired in v1 "
            "(only 'scib' is implemented)."
        )
    if clustering not in _CLUSTERING_FILES:
        raise ValueError(
            f"unknown clustering {clustering!r}; valid: {sorted(_CLUSTERING_FILES)}"
        )
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    task_metrics = _task_metrics(task)
    if category is not None:
        config.category_folder(category)      # raises "unknown category ... valid: [...]"
    if methods is not None:
        if method is not None:
            raise ValueError("pass either method= or methods=, not both")
        method = methods
    datasets = _as_list(dataset)
    wanted_methods = _as_list(method)

    base = _base_path(result_path)

    if base.is_file():
        out = _load_long_csv(base)
        if category is not None and (out["category"] == category).any():
            out = out[out["category"] == category]
        if datasets:
            out = out[out["dataset"].isin(datasets)]
    else:
        cats = [category] if category is not None else list(_CATEGORIES)
        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        for cat in cats:
            got_any = False
            if source in ("published", "both"):
                try:
                    frames.append(_load_published(cat, datasets, clustering, base, metric_set))
                    got_any = True
                except FileNotFoundError as e:
                    errors.append(str(e))
            if source in ("rerun", "both"):
                try:
                    frames.append(_load_rerun(cat, datasets, base))
                    got_any = True
                except FileNotFoundError as e:
                    errors.append(str(e))
            if not got_any and category is not None:
                # a named category must resolve; say precisely what was tried
                raise FileNotFoundError(" | ".join(errors))
        if not frames:
            raise FileNotFoundError(
                f"no results for any category under {base} (source={source!r}): "
                + " | ".join(errors))
        out = pd.concat(frames, ignore_index=True)

    # ---- filters ---------------------------------------------------------
    where = f"{category or 'any category'}/{datasets if datasets else 'any dataset'}"
    if task_metrics is not None:
        out = out[out["metric"].isin(task_metrics)]
    if wanted_methods is not None:
        # alias tolerant AND case-insensitive: 'mofa+' -> MOFA2, 'scbridge' -> scBridge
        want = {catalog.canonical_id(m).lower() for m in wanted_methods}
        avail = sorted(out["method"].unique())
        out = out[out["method"].map(lambda m: catalog.canonical_id(m).lower()).isin(want)]
        if out.empty:
            warnings.warn(
                f"method {wanted_methods if len(wanted_methods) > 1 else wanted_methods[0]!r} "
                f"not present in {where} (source={source!r}; available: {avail})",
                UserWarning, stacklevel=2)
    if metric is not None:
        wanted = [catalog.canonical_metric(m) for m in _as_list(metric)]
        avail = sorted(out["metric"].unique())
        out = out[out["metric"].isin(wanted)]
        if out.empty:
            warnings.warn(
                f"metric {metric!r} not present in {where} (source={source!r}; "
                f"available: {avail})", UserWarning, stacklevel=2)
    return out[COLUMNS].reset_index(drop=True)


def available_datasets(
    category: str | None = None,
    metric_set: str = "scib",
    result_path: Path | str | None = None,
    *,
    source: str = "published",
    clustering: str = "default",
) -> list[str]:
    """Dataset ids that have loadable ``metric_set`` results.

    Parameters
    ----------
    category : str, optional
        One of the four integration categories; ``None`` (default) = the
        union across all of them, so a bare ``available_datasets()`` "just
        works". A category folder that does not exist (``mosaic`` has no
        published tables) contributes nothing - no error.
    metric_set : str
        Only ``"scib"`` is wired (``NotImplementedError`` otherwise).
    result_path : path-like, optional
        Results root (see :func:`load_results`). If the root itself does not
        exist a ``UserWarning`` is raised and ``[]`` returned, instead of a
        silent empty list.
    source : {"published", "rerun", "both"}, keyword-only
        Which tables to look at (see :func:`load_results`).
    clustering : {"default", "louvain", "kmeans"}, keyword-only
        For ``published``: list only datasets holding at least one method
        with that clustering variant's file - exactly what
        ``load_results(category, dataset=..., clustering=...)`` can load.

    Returns
    -------
    list of str
        Sorted dataset ids.
    """
    if metric_set != "scib":
        raise NotImplementedError(
            f"metric_set={metric_set!r} is not wired in v1 (only 'scib')."
        )
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    if clustering not in _CLUSTERING_FILES:
        raise ValueError(
            f"unknown clustering {clustering!r}; valid: {sorted(_CLUSTERING_FILES)}")
    base = _base_path(result_path)
    if not base.exists():
        warnings.warn(
            f"result_path {base} does not exist - nothing can be listed "
            f"(default root: {config.DEFAULT.result_path})",
            UserWarning, stacklevel=2)
        return []
    if category is not None:
        config.category_folder(category)
    cats = [category] if category is not None else list(_CATEGORIES)
    found: set[str] = set()
    if source in ("published", "both"):
        for cat in cats:
            root = base / config.metric_set_dir(metric_set) / config.category_folder(cat)
            if not root.exists():
                continue
            for ds_dir, *_ in _iter_published(root, None, clustering):
                found.add(ds_dir.name)
    if source in ("rerun", "both"):
        for _, df in _read_rerun_files(base):
            found.update(df.loc[df["category"].isin(cats), "dataset"].astype(str).unique())
    return sorted(found)


def results_coverage(
    category: str | None = None,
    *,
    source: str = "both",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """Which (category, dataset, method) combinations have results, and where from.

    One row per distinct ``category, dataset, method, clustering, source``;
    the published tree is scanned for every clustering variant (default,
    louvain, kmeans), so a method that only exists as e.g. a ``_louvain``
    directory shows up under ``clustering="louvain"``. Nothing is raised for
    a category that has no tables - it simply has no rows.

    Parameters
    ----------
    category : str, optional
        Restrict to one integration category (default: all four).
    source : {"published", "rerun", "both"}, keyword-only
        Which tables to scan (default ``"both"``).
    result_path : path-like, keyword-only
        Results root (see :func:`load_results`).

    Returns
    -------
    pandas.DataFrame
        Columns ``category, dataset, method, clustering, source``, sorted.

    Examples
    --------
    >>> cov = mtb.results_coverage("cross")
    >>> cov[cov.dataset == "D52"]           # scMoMaT (published) + 8 methods (rerun-0.2.1)
    >>> cov.groupby(["category", "source"]).method.nunique()
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    cols = ["category", "dataset", "method", "clustering", "source"]
    cats = [category] if category is not None else list(_CATEGORIES)
    if category is not None:
        config.category_folder(category)
    frames = []
    for cat in cats:
        if source in ("published", "both"):
            for clus in _CLUSTERING_FILES:
                try:
                    frames.append(load_results(cat, clustering=clus, result_path=result_path,
                                               source="published")[cols])
                except FileNotFoundError:
                    pass
        if source in ("rerun", "both"):
            try:
                frames.append(load_results(cat, result_path=result_path, source="rerun")[cols])
            except FileNotFoundError:
                pass
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out.sort_values(cols).reset_index(drop=True)


def recommend(
    category: str,
    *,
    modalities: list[str] | None = None,
    task: str | None = "clustering",
    long_df: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
    min_methods: int = 2,
    source: str = "published",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """Rank methods for a category from stored results, with coverage made explicit.

    The score is the benchmark's own rule applied per dataset
    (``overall="mean_overall"`` in :mod:`multibench.plot.style`: min-max
    scaled mean of per-metric max-ranks within each dataset, then averaged
    over the datasets the method was run on). Two honesty rules apply:

    * a dataset holding fewer than ``min_methods`` methods is DROPPED - the
      min-max of a single method is 1.0 by construction, so a lone method
      would "win" such a dataset with authority it never earned;
    * the returned ``n_datasets`` / ``n_datasets_total`` / ``coverage``
      columns say how much of the matrix each score rests on, and a
      ``UserWarning`` summarises dropped datasets and partial coverage.

    Parameters
    ----------
    category : {"vertical", "diagonal", "mosaic", "cross"}
        Integration category to rank.
    modalities : list of str, keyword-only
        Keep only methods that consume ALL of these base modalities
        (``["rna", "adt"]``), via :func:`multibench.find_methods`.
    task : {None, "all", "clustering", "batch"}, keyword-only
        Metric family to score on (default ``"clustering"``); ignored when
        ``metrics`` is given.
    long_df : pandas.DataFrame, keyword-only
        Score THIS frame (``metric, value, method, dataset``) instead of
        loading stored results - e.g. ``pd.concat([published, mine])`` to
        place your own method.
    metrics : list of str, keyword-only
        Explicit metric codes to score on (overrides ``task``).
    min_methods : int, keyword-only
        Datasets with fewer methods than this are dropped (default 2).
    source : {"published", "rerun", "both"}, keyword-only
        Which stored tables to load when ``long_df`` is not given (default
        ``"published"``; note the published tables are PARTIAL - vertical
        ships 8 of the paper's methods, cross mostly one method per dataset -
        so the ranking is over what shipped, not the paper's full matrix).
    result_path : path-like, keyword-only
        Results root (see :func:`load_results`).

    Returns
    -------
    pandas.DataFrame
        Sorted best-first; columns ``method, grand_score, n_datasets,
        n_datasets_total, coverage, needs_labels, runtime_tier, worst_sec,
        env, output_kind``. The metadata columns are ``None`` for ids that are
        not registry methods (your own method, a result-dir token).

    Raises
    ------
    ValueError
        No dataset has ``min_methods`` methods (nothing can be ranked), or
        the frame has no usable metrics.
    FileNotFoundError
        No stored results for the category/source.

    Examples
    --------
    >>> mtb.recommend("vertical", modalities=["rna", "adt"])
    >>> mtb.recommend("diagonal", source="rerun")[["method", "grand_score", "coverage", "needs_labels", "runtime_tier"]]
    """
    from ..plot import style

    config.category_folder(category)
    if long_df is None:
        long_df = load_results(category, task=None if metrics else task,
                               source=source, result_path=result_path)
    df = long_df.copy()
    if metrics is None:
        fam = _task_metrics(task)          # validates task; None = all metrics
        if fam is not None:
            df = df[df["metric"].isin(fam)]
            if df.empty:
                raise ValueError(
                    f"no {task!r} metrics ({fam}) in the frame; metrics present: "
                    f"{sorted(long_df['metric'].unique())}")
    if "dataset" not in df.columns:
        df["dataset"] = "all"
    if metrics is not None:
        metrics = [catalog.canonical_metric(m) for m in metrics]

    parts = style.per_dataset_ranks(df, metrics)
    degenerate = sorted(ds for ds, mat in parts.items() if len(mat.index) < min_methods)
    kept = {ds: mat for ds, mat in parts.items() if ds not in degenerate}
    if not kept:
        raise ValueError(
            f"no dataset in {category} holds >= {min_methods} methods "
            f"({len(parts)} dataset(s): {sorted(parts)}); a ranking over "
            f"single-method datasets is meaningless (min-max of one value is "
            f"1.0). Pass long_df= with more methods, or lower min_methods.")
    per_ds = pd.DataFrame({ds: style.compute_overall(mat) for ds, mat in kept.items()})
    grand = per_ds.mean(axis=1)
    n_ds = per_ds.notna().sum(axis=1)
    n_total = per_ds.shape[1]

    keep_methods = list(grand.index)
    if modalities is not None:
        from ..discover import find_methods
        allowed = set(find_methods(category=category, modalities=modalities))
        keep_methods = [m for m in keep_methods if catalog.canonical_id(m) in allowed]
        if not keep_methods:
            raise ValueError(
                f"no scored method in {category} consumes modalities "
                f"{list(modalities)}; methods with results: {sorted(grand.index)}; "
                f"methods matching the modalities: {sorted(allowed)}")

    from ..engine import registry, envs
    from ..workflow import runtime_hint

    rows = []
    for m in keep_methods:
        try:
            spec = registry.get(catalog.canonical_id(m))
        except KeyError:
            spec = None
        if spec is not None:
            kinds = [v.output.kind for v in spec.variants
                     if v.when.get("category") == category] or \
                    [v.output.kind for v in spec.variants]
            env = envs.group_for(spec.id)
            # per-category: a method may be supervised in one category only
            _vs = [v for v in spec.variants if v.when.get("category") == category]
            needs = any(v.needs_labels for v in _vs) if _vs else bool(spec.needs_labels)
            okind = kinds[0] if kinds else None
        else:
            env, needs, okind = None, None, None
        rt = runtime_hint(catalog.canonical_id(m)) if spec is not None else {}
        rows.append({
            "method": m,
            "grand_score": float(grand.loc[m]),
            "n_datasets": int(n_ds.loc[m]),
            "n_datasets_total": int(n_total),
            "coverage": float(n_ds.loc[m]) / n_total,
            "needs_labels": needs,
            "runtime_tier": rt.get("tier") if spec is not None else None,
            "worst_sec": rt.get("worst_sec") if spec is not None else None,
            "env": env,
            "output_kind": okind,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("grand_score", ascending=False, kind="mergesort").reset_index(drop=True)

    notes = []
    if degenerate:
        notes.append(
            f"dropped {len(degenerate)} dataset(s) with fewer than {min_methods} "
            f"methods ({', '.join(map(str, degenerate))}) - a min-max score over "
            f"one method is 1.0 by construction")
    partial = out[out["coverage"] < 1.0]
    if not partial.empty:
        notes.append(
            "grand_score averages over an incomplete method x dataset matrix: "
            + ", ".join(f"{r.method} scored on {r.n_datasets}/{r.n_datasets_total}"
                        for r in partial.itertuples())
            + " dataset(s); compare coverage before trusting the order")
    if notes:
        warnings.warn(f"recommend({category!r}): " + "; ".join(notes),
                      UserWarning, stacklevel=2)
    return out
