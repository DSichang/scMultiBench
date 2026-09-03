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

Currently the scib clustering + batch loader is the only metric set; any
other ``metric_set`` token raises ``ValueError`` listing the valid ones.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from .. import config
from . import catalog

__all__ = ["load_results", "available_datasets", "results_coverage", "recommend",
           "COLUMNS", "SOURCES", "FAMILIES", "DegenerateRerunWarning"]

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

#: ``family=`` (alias ``task=``) values: None/"all" = no filter, else a
#: metric family
FAMILIES = (None, "all", "clustering", "batch")
_TASKS = FAMILIES          # old private name, kept for callers that imported it

#: A re-run row is DEGENERATE when its ARI is below this ...
_DEGENERATE_RERUN_ARI = 0.01
#: ... while the published table scored the same (category, dataset, method)
#: above this: the re-run almost certainly failed silently (wrong label order,
#: a collapsed embedding) rather than the method being that bad.
_DEGENERATE_PUBLISHED_ARI = 0.2


class DegenerateRerunWarning(UserWarning):
    """A re-run row scored ARI ~0 where the published table scored well.

    Emitted by :func:`load_results` (``source="rerun"`` / ``"both"``) so a
    silently failed re-run (Conos on D28: ARI 0.0004 vs published 0.27) can
    never enter a ranking unnoticed. Filter with
    ``warnings.simplefilter("ignore", mtb.data.results.DegenerateRerunWarning)``
    once you have decided how to treat those rows.
    """

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
    # provenance defaults are filled PER ROW, not only for an absent column: a
    # frame concatenated from load_results (7 columns) and an older 5-column
    # to_long frame used to carry NaN clustering/source for the user's rows
    # through to_csv -> load_results, and those NaNs then hid in every plot
    for col, default in (("clustering", "default"), ("source", "user")):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df[COLUMNS]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def _resolve_family(task, family):
    """The one metric-family token from the ``task=`` / ``family=`` pair.

    ``family`` is the documented name; ``task`` is the original name, kept as
    an alias. Both given and different -> ``ValueError``.
    """
    if family is not None and task is not None and family != task:
        raise ValueError(
            f"pass either family= or task= (they are the same selector; got "
            f"family={family!r}, task={task!r})")
    return family if family is not None else task


def _family_metrics(value, kw: str = "family"):
    """Metric codes of a family token, or ``None`` for "every metric".

    Parameters
    ----------
    value
        ``None`` / ``"all"`` (no filter), ``"clustering"`` or ``"batch"``.
    kw : str
        The keyword the value arrived through (``"family"`` or ``"task"``),
        named in the error.

    Raises
    ------
    ValueError
        ``"unknown family 'bogus' (given as task=); valid: None (all metrics),
        'all', 'clustering', 'batch'"``. A :func:`multibench.list_tasks`
        token (``'dimension_reduction'``) or a dataset id in that slot gets an
        extra sentence saying what the slot means.
    """
    if value not in FAMILIES:
        hint = ""
        try:
            from ..engine.registry import list_tasks
            tasks = list_tasks()
        except Exception:
            tasks = []
        if value in tasks:
            hint = (f" - {kw}= here selects a METRIC FAMILY, not a "
                    f"mtb.list_tasks() token; 'dimension_reduction' and "
                    f"'clustering' share the 'clustering' family")
        elif re.match(r"^S?D\d+", str(value)):
            hint = (f" - the 2nd positional argument is {kw}, did you mean "
                    f"dataset={value!r}?")
        raise ValueError(
            f"unknown family {value!r} (given as {kw}=); valid: None (all "
            f"metrics), 'all', 'clustering', 'batch'{hint}")
    if value in (None, "all"):
        return None
    # the family lists live in plot.bar and are pinned to eval by
    # tests/test_metric_groups.py; import lazily (plot imports nothing from
    # data at import time, but keep the dependency one-directional at load)
    from ..plot.bar import BATCH_METRICS, CLUSTERING_METRICS
    return list(CLUSTERING_METRICS if value == "clustering" else BATCH_METRICS)


def _task_metrics(task):
    """Back-compat alias of :func:`_family_metrics` (``task=`` spelling)."""
    return _family_metrics(task, kw="task")


def _check_metric_set(metric_set: str) -> None:
    """``ValueError`` listing the valid tokens for an unknown ``metric_set``.

    ``NotImplementedError`` is reserved for a token the config declares but
    this loader does not read yet (none today - only ``"scib"`` exists).
    """
    config.metric_set_dir(metric_set)     # unknown -> ValueError with the list
    if metric_set != "scib":
        raise NotImplementedError(
            f"metric_set={metric_set!r} is declared but its tables are not "
            f"wired into load_results yet (only 'scib' is).")


def _check_methods(wanted: list, present) -> None:
    """Every requested method must be a registry id or a name in the frame.

    Raises ``KeyError`` with a did-you-mean hint (the same shape ``scan`` /
    ``method_info`` use) for anything else - a typo used to yield an empty
    frame, indistinguishable from "no rows for that method".
    """
    present = list(present)
    by_lower = {str(m).lower() for m in present}
    for m in wanted:
        cid = catalog.canonical_id(m)
        if cid.lower() in by_lower:
            continue
        try:
            catalog.canonical_id(m, strict=True)
        except KeyError:
            import difflib
            pool = sorted(set(present) | set(catalog._registry_ids()))
            hint = difflib.get_close_matches(str(m), pool, n=1, cutoff=0.6)
            raise KeyError(
                f"unknown method {m!r}"
                + (f"; did you mean {hint[0]!r}?" if hint else "")
                + "; see mtb.list_methods() (a method absent from the loaded "
                "tables but known to the registry gives an empty frame plus a "
                "UserWarning instead)") from None


def _check_metrics(wanted: list, present) -> list:
    """Canonicalise metric codes; unknown ones (not a known code, not in the
    frame) raise ``ValueError`` listing both vocabularies."""
    present = sorted(set(map(str, present)))
    out, unknown = [], []
    for m in wanted:
        c = catalog.canonical_metric(m)
        if c is None or (c not in catalog.known_metrics() and c not in present):
            unknown.append(m)
        else:
            out.append(c)
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; valid codes: {catalog.known_metrics()}"
            f"; present in this frame: {present}")
    return out


def _degenerate_rerun_rows(out: pd.DataFrame, base: Path) -> pd.DataFrame:
    """Re-run rows whose ARI is ~0 while the published table scored well.

    Returns a frame ``category, dataset, method, source, rerun_ARI,
    published_ARI`` (empty when nothing is degenerate or no published table
    exists for the re-run datasets).
    """
    cols = ["category", "dataset", "method", "source", "rerun_ARI", "published_ARI"]
    rr = out[(out["source"] != "published") & (out["metric"] == "ARI")]
    if rr.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for cat, g in rr.groupby("category"):
        try:
            pub = _load_published(str(cat), sorted(g["dataset"].astype(str).unique()),
                                  "default", base, "scib")
        except (FileNotFoundError, ValueError):
            continue
        pub = pub[pub["metric"] == "ARI"].set_index(["dataset", "method"])["value"]
        for r in g.itertuples(index=False):
            key = (str(r.dataset), str(r.method))
            if key not in pub.index:
                continue
            p = float(pub.loc[key]) if not isinstance(pub.loc[key], pd.Series) \
                else float(pub.loc[key].iloc[0])
            if float(r.value) < _DEGENERATE_RERUN_ARI and p > _DEGENERATE_PUBLISHED_ARI:
                rows.append({"category": cat, "dataset": r.dataset, "method": r.method,
                             "source": r.source, "rerun_ARI": float(r.value),
                             "published_ARI": p})
    return pd.DataFrame(rows, columns=cols)


def _warn_degenerate(out: pd.DataFrame, base: Path, stacklevel: int = 3) -> None:
    bad = _degenerate_rerun_rows(out, base)
    if bad.empty:
        return
    items = ", ".join(f"{r.method}/{r.dataset} ({r.source} ARI {r.rerun_ARI:.4f} vs "
                      f"published {r.published_ARI:.2f})" for r in bad.itertuples())
    warnings.warn(
        f"degenerate re-run row(s) - ARI < {_DEGENERATE_RERUN_ARI} where the "
        f"published table scored > {_DEGENERATE_PUBLISHED_ARI}: {items}. The "
        f"re-run most likely failed silently (a collapsed embedding or a wrong "
        f"label order), so the row says nothing about the method; drop it "
        f"before ranking (df[df.method != {bad.method.iloc[0]!r}]) or compare "
        f"with source='published'.", DegenerateRerunWarning, stacklevel=stacklevel)


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
    family: str | None = None,
) -> pd.DataFrame:
    """Benchmark metric tables as a tidy long frame.

    Parameters
    ----------
    category : {"vertical", "diagonal", "mosaic", "cross"}, optional
        Integration category. ``None`` (default) loads every category that
        has tables for the requested ``source`` (an unknown token raises
        ``ValueError`` listing the valid ones).
    task : {None, "all", "clustering", "batch"}, optional
        Alias of ``family`` (the original name of the selector; NOT a
        :func:`multibench.list_tasks` token - ``'dimension_reduction'`` here
        raises and points at ``'clustering'``). Passing both with different
        values raises ``ValueError``.
    metric_set : str
        Only ``"scib"`` exists; an unknown token raises ``ValueError``
        listing the valid ones (``NotImplementedError`` is reserved for a
        token the config declares but this loader cannot read).
    dataset : str or list of str, optional
        Dataset id(s), e.g. ``"D11"`` or ``["D11", "D11s"]``. Default: all
        datasets of the category. EVERY requested id must have a table:
        ``["D11", "D99"]`` raises ``FileNotFoundError`` naming ``D99`` and the
        datasets that are available, exactly like ``dataset="D99"`` (an
        unknown id in a list used to be dropped silently).
    method : str or list of str, optional
        Keep only these method(s); alias tolerant and case-insensitive
        (``"mofa+"`` -> MOFA2, ``"totalvi"`` -> totalVI). Every name must be
        a registry id or a method present in the loaded frame: a typo raises
        ``KeyError`` with a did-you-mean hint (``"unknown method 'Matlida';
        did you mean 'Matilda'?"``). A KNOWN method with no rows in the
        loaded tables gives an EMPTY frame and a ``UserWarning`` (a typo is
        not an error there - concatenating several calls would otherwise
        break); under the default ``source="published"`` the warning also
        says whether the re-run sweeps hold that method (``"rerun has 2
        dataset(s) - pass source='rerun'"``), because the published tables
        are partial (vertical: 7 of 18 methods).
    metric : str or list of str, optional
        Keep only these metric code(s); alias tolerant (``"ari"`` -> ARI).
        Every code must be one of :func:`multibench.catalog.known_metrics`
        or present in the frame: ``"ZZZ"`` raises ``ValueError`` listing
        both. A known code the tables lack gives the empty-frame-plus-warning
        of ``method``.
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
        ``BatchResult.save()``). A file keeps whatever ``source`` /
        ``clustering`` values it carries; a missing column or a blank cell is
        filled with ``"user"`` / ``"default"`` per row.
    source : str, keyword-only
        For a results ROOT one of ``"published"`` (default: the benchmark's
        scIB tables under ``result_path/scib_metric``; none for mosaic),
        ``"rerun"`` (the package's re-run sweeps
        ``result_path/rerun/long_all_<dataset>.csv``: vertical D11/D11s,
        diagonal D28/D28s, mosaic D45/D45s, cross D52/D52s) or ``"both"``
        (the concatenation - tell the two apart by the ``source`` column,
        ``"published"`` vs ``"rerun-<version>"``); anything else raises
        ``ValueError``. For a FILE, ``"published"`` and ``"both"`` keep every
        row, while any other value filters on the file's own ``source``
        column: ``"user"`` keeps the rows :func:`multibench.to_long` wrote,
        ``"rerun"`` matches ``rerun-<version>`` by prefix, and a value the
        file does not contain raises ``ValueError`` listing the ones present.
        With ``"rerun"``/``"both"`` a re-run row whose ARI is ~0 while the
        published table scored the same method/dataset well is reported by a
        :class:`DegenerateRerunWarning` (Conos on D28) - never silently.
    methods : list of str, keyword-only
        Alias for ``method`` (a list); passing both raises ``ValueError``.
    family : {None, "all", "clustering", "batch"}, keyword-only
        Restrict to a metric family: ``"clustering"`` keeps
        ``mtb.plot.CLUSTERING_METRICS`` (ARI, NMI, ASW, iASW, iF1, cLISI),
        ``"batch"`` keeps ``mtb.plot.BATCH_METRICS`` (ASW_batch, GC, iLISI,
        kBET). ``None``/``"all"`` (default) keeps every metric. Anything
        else raises ``ValueError`` (``"unknown family ..."``).

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
        No table for the requested category/dataset/source (any element of a
        ``dataset`` list), with the path looked at and what IS available.
    KeyError
        An unknown method name in ``method``/``methods`` (did-you-mean hint).
    ValueError
        Unknown ``category`` / ``family`` (``task``) / ``clustering`` /
        ``source`` / ``metric_set`` / metric code, or a ``result_path`` file
        without the long-format columns.

    Examples
    --------
    >>> pub = mtb.load_results("diagonal", dataset="D28")            # published
    >>> rr = mtb.load_results("diagonal", dataset="D28", source="rerun")
    >>> both = mtb.load_results("cross", dataset="D52", source="both")
    >>> mtb.plot.bubble(both[both.source != "published"])
    >>> mine = mtb.load_results(result_path="mine.csv", source="user")  # your rows only
    """
    _check_metric_set(metric_set)
    if clustering not in _CLUSTERING_FILES:
        raise ValueError(
            f"unknown clustering {clustering!r}; valid: {sorted(_CLUSTERING_FILES)}"
        )
    fam = _resolve_family(task, family)
    fam_metrics = _family_metrics(fam, kw="family" if family is not None else "task")
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
        if source not in ("published", "both"):
            # a file carries its own provenance: filter on it, loudly
            col = out["source"].astype(str)
            mask = col.str.startswith("rerun") if source == "rerun" else col == source
            if not mask.any():
                raise ValueError(
                    f"source {source!r} not in {base}; present: "
                    f"{sorted(col.unique())} (pass source='published' or "
                    f"'both' to keep every row of a file)")
            out = out[mask]
        if category is not None and (out["category"] == category).any():
            out = out[out["category"] == category]
        if datasets:
            have = sorted(out["dataset"].astype(str).unique())
            out = out[out["dataset"].astype(str).isin([str(d) for d in datasets])]
            missing = [d for d in datasets if str(d) not in set(have)]
            if missing:
                raise FileNotFoundError(
                    f"no rows for dataset {missing if len(missing) > 1 else missing[0]!r} "
                    f"in {base}; datasets in the file: {have}")
    else:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
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
        if datasets:
            # every element of a list must resolve, like the scalar form does
            have = set(out["dataset"].astype(str).unique())
            missing = [d for d in datasets if str(d) not in have]
            if missing:
                avail = available_datasets(category, result_path=base, source=source,
                                           clustering=clustering)
                raise FileNotFoundError(
                    f"no {source} results for {category or 'any category'}/"
                    f"{missing if len(missing) > 1 else missing[0]}; datasets with "
                    f"{source} tables: {avail}")

    # ---- filters ---------------------------------------------------------
    where = f"{category or 'any category'}/{datasets if datasets else 'any dataset'}"
    if fam_metrics is not None:
        out = out[out["metric"].isin(fam_metrics)]
    if wanted_methods is not None:
        avail = sorted(out["method"].unique())
        _check_methods(wanted_methods, avail)
        # alias tolerant AND case-insensitive: 'mofa+' -> MOFA2, 'scbridge' -> scBridge
        want = {catalog.canonical_id(m).lower() for m in wanted_methods}
        out = out[out["method"].map(lambda m: catalog.canonical_id(m).lower()).isin(want)]
        if out.empty:
            shown = wanted_methods if len(wanted_methods) > 1 else wanted_methods[0]
            hint = ""
            if source == "published" and not base.is_file():
                hint = _other_source_hint(category, datasets, wanted_methods, base)
            warnings.warn(
                f"no {source} rows for method {shown!r} in {where} (available: "
                f"{avail}){hint}", UserWarning, stacklevel=2)
    if metric is not None:
        avail = sorted(out["metric"].unique())
        wanted = _check_metrics(_as_list(metric), avail)
        out = out[out["metric"].isin(wanted)]
        if out.empty:
            warnings.warn(
                f"metric {metric!r} not present in {where} (source={source!r}; "
                f"available: {avail})", UserWarning, stacklevel=2)
    out = out[COLUMNS].reset_index(drop=True)
    if source in ("rerun", "both") and not base.is_file():
        _warn_degenerate(out, base)
    return out


def _other_source_hint(category, datasets, wanted_methods, base: Path) -> str:
    """'; rerun has N dataset(s) (...) - pass source="rerun"' when the re-run
    sweeps hold rows for a method the published tables lack, else ''."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cov = results_coverage(category, source="rerun", result_path=base)
    except Exception:
        return ""
    want = {catalog.canonical_id(m).lower() for m in wanted_methods}
    cov = cov[cov["method"].map(lambda m: catalog.canonical_id(m).lower()).isin(want)]
    if datasets:
        cov = cov[cov["dataset"].astype(str).isin([str(d) for d in datasets])]
    if cov.empty:
        return ("; results_coverage(source='both') lists no rows for it in any "
                "source either")
    ds = sorted(cov["dataset"].astype(str).unique())
    return (f"; rerun has {len(ds)} dataset(s) ({', '.join(ds)}) - pass "
            f"source='rerun' (or 'both')")


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
        Only ``"scib"`` exists; an unknown token raises ``ValueError``
        listing the valid ones.
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
    _check_metric_set(metric_set)
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
                # a coverage scan asks WHERE rows are, not whether they are
                # sound; the degenerate-row check belongs to load_results
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DegenerateRerunWarning)
                    frames.append(load_results(cat, result_path=result_path,
                                               source="rerun")[cols])
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
    family: str | None = None,
) -> pd.DataFrame:
    """Rank methods for a category from stored results, with coverage made explicit.

    The score is the benchmark's own rule applied per dataset
    (``overall="mean_overall"`` in :mod:`multibench.plot.style`: min-max
    scaled mean of per-metric max-ranks within each dataset, then averaged
    over the datasets the method was run on). Three honesty rules apply:

    * a dataset holding fewer than ``min_methods`` methods is DROPPED - the
      min-max of a single method is 1.0 by construction, so a lone method
      would "win" such a dataset with authority it never earned;
    * the returned ``n_datasets`` / ``n_datasets_total`` / ``coverage``
      columns say how much of the matrix each score rests on;
    * every method wired for the category (and ``modalities``) that has NO
      rows in the chosen source is still listed - appended after the scored
      rows with ``grand_score`` NaN, ``n_datasets`` 0 and ``coverage`` 0.0 -
      so "not ranked" is never mistaken for "ranked last" (the published
      tables are partial: vertical rna+adt scores 7 of 14 methods; the re-run
      sweeps cover more - ``source="rerun"``). Their ids are also in
      ``frame.attrs["not_scored"]``.

    One ``UserWarning`` with one line per finding summarises dropped
    datasets, partial coverage and the unscored methods.

    Parameters
    ----------
    category : {"vertical", "diagonal", "mosaic", "cross"}
        Integration category to rank.
    modalities : list of str, keyword-only
        Keep only methods that consume ALL of these base modalities
        (``["rna", "adt"]``), via :func:`multibench.find_methods`.
    task : {None, "all", "clustering", "batch"}, keyword-only
        Alias of ``family`` (default ``"clustering"``); ``family`` wins when
        both are given. Not a :func:`multibench.list_tasks` token.
    long_df : pandas.DataFrame, keyword-only
        Score THIS frame (``metric, value, method, dataset``) instead of
        loading stored results - e.g. ``pd.concat([published, mine])`` to
        place your own method.
    metrics : list of str, keyword-only
        Explicit metric codes to score on (overrides ``family``/``task``).
    min_methods : int, keyword-only
        Datasets with fewer methods than this are dropped (default 2).
    source : {"published", "rerun", "both"}, keyword-only
        Which stored tables to load when ``long_df`` is not given (default
        ``"published"``; the published tables are PARTIAL - see above - and
        ``"both"`` averages the 34 method/dataset/metric triples present in
        both sources).
    result_path : path-like, keyword-only
        Results root (see :func:`load_results`).
    family : {None, "all", "clustering", "batch"}, keyword-only
        Metric family to score on: ``"clustering"`` (ARI, NMI, ASW, iASW,
        iF1, cLISI) or ``"batch"`` (ASW_batch, GC, iLISI, kBET); ``None`` /
        ``"all"`` scores every metric present. The documented name of the
        ``task`` selector.

    Returns
    -------
    pandas.DataFrame
        Sorted best-first, unscored methods last; columns ``method,
        grand_score, n_datasets, n_datasets_total, coverage, needs_labels,
        runtime_tier, worst_sec, env, output_kind``. The metadata columns are
        ``None`` for ids that are not registry methods (your own method, a
        result-dir token). ``frame.attrs`` records the choices the ranking
        was made under: ``"family"`` (and ``"task"``, the same value; ``None``
        when ``metrics`` was given), ``"source"`` (``"published"`` /
        ``"rerun"`` / ``"both"``, or ``"long_df"``) and ``"not_scored"`` (the
        unscored method ids, also under ``"missing"``).

    Raises
    ------
    ValueError
        No dataset has ``min_methods`` methods (nothing can be ranked), the
        frame has none of the family's metrics (the message lists the metrics
        it does have), or an unknown ``family``/``task``.
    FileNotFoundError
        No stored results for the category/source.

    Examples
    --------
    >>> r = mtb.recommend("vertical", modalities=["rna", "adt"])
    >>> r[r.grand_score.notna()]                    # the scored rows
    >>> r.attrs["not_scored"]                       # wired but no published rows
    >>> mtb.recommend("diagonal", family="batch", source="rerun")[["method", "grand_score", "coverage"]]
    """
    from ..plot import style

    config.category_folder(category)
    fam = family if family is not None else task
    fam_kw = "family" if family is not None else "task"
    long_df_was_none = long_df is None
    if long_df is None:
        # load EVERY metric and filter locally, so the "metrics present"
        # error below can name what the frame really holds
        long_df = load_results(category, source=source, result_path=result_path)
    df = long_df.copy()
    if metrics is None:
        fam_metrics = _family_metrics(fam, kw=fam_kw)   # validates; None = all metrics
        if fam_metrics is not None:
            df = df[df["metric"].isin(fam_metrics)]
            if df.empty:
                raise ValueError(
                    f"no {fam!r} metrics ({fam_metrics}) in the frame; metrics present: "
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

    from ..discover import find_methods

    keep_methods = list(grand.index)
    if modalities is not None:
        allowed = set(find_methods(category=category, modalities=modalities))
        keep_methods = [m for m in keep_methods if catalog.canonical_id(m) in allowed]
        if not keep_methods:
            raise ValueError(
                f"no scored method in {category} consumes modalities "
                f"{list(modalities)}; methods with results: {sorted(grand.index)}; "
                f"methods matching the modalities: {sorted(allowed)}")

    # Methods wired for the category but absent from the source. The registry
    # 'clustering' tag is complete in every category and excludes the
    # registration-only cross methods (PASTE/PASTE2/SPIRAL/GPSA - coordinates,
    # not an embedding, so scIB never applies); the 'batch' tag is NOT
    # reliable (vertical: 8/18 tagged), so never gate on the requested family.
    wired = find_methods(category=category, task="clustering", modalities=modalities,
                         runnable=True)
    scored_ids = {catalog.canonical_id(m) for m in keep_methods}
    missing = sorted((m for m in wired if m not in scored_ids), key=str.lower)
    # a method with rows ONLY in dropped (< min_methods) datasets is unscored
    # for a different reason than "no rows at all" - say which
    in_frame = {catalog.canonical_id(m) for m in df["method"].unique()}
    only_dropped = [m for m in missing if m in in_frame]
    no_rows = [m for m in missing if m not in in_frame]
    label = f"source={source!r}" if long_df_was_none else "long_df"

    from ..engine import registry, envs
    from ..workflow import runtime_hint

    rows = []
    for m in keep_methods + missing:
        scored = m in grand.index
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
            "grand_score": float(grand.loc[m]) if scored else float("nan"),
            "n_datasets": int(n_ds.loc[m]) if scored else 0,
            "n_datasets_total": int(n_total),
            "coverage": float(n_ds.loc[m]) / n_total if scored else 0.0,
            "needs_labels": needs,
            "runtime_tier": rt.get("tier") if spec is not None else None,
            "worst_sec": rt.get("worst_sec") if spec is not None else None,
            "env": env,
            "output_kind": okind,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("grand_score", ascending=False, kind="mergesort",
                          na_position="last").reset_index(drop=True)
    out.attrs["family"] = None if metrics is not None else fam
    out.attrs["task"] = out.attrs["family"]
    out.attrs["source"] = source if long_df_was_none else "long_df"
    out.attrs["not_scored"] = list(missing)
    out.attrs["missing"] = list(missing)

    notes = []
    if degenerate:
        notes.append(
            f"dropped {len(degenerate)} dataset(s) with fewer than {min_methods} "
            f"methods ({', '.join(map(str, degenerate))}) - a min-max score over "
            f"one method is 1.0 by construction")
    partial = out[(out["coverage"] < 1.0) & out["grand_score"].notna()]
    if not partial.empty:
        notes.append(
            "grand_score averages over an incomplete method x dataset matrix "
            "(partial coverage: "
            + ", ".join(f"{r.method} {r.n_datasets}/{r.n_datasets_total}"
                        for r in partial.itertuples())
            + "); compare coverage before trusting the order")
    if only_dropped:
        notes.append(
            f"rows only in dropped dataset(s) for: {', '.join(only_dropped)} - "
            f"listed with grand_score NaN / coverage 0.0")
    if no_rows:
        notes.append(
            f"no rows in {label} for: {', '.join(no_rows)} - listed with "
            f"grand_score NaN / coverage 0.0"
            + (' (try source="rerun")' if source == "published" and long_df_was_none
               else ""))
    if notes:
        warnings.warn(f"recommend({category!r}):\n  - " + "\n  - ".join(notes),
                      UserWarning, stacklevel=2)
    return out
