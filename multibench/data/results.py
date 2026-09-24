"""Load benchmark metric tables into one long table (a DataFrame).

Two sources ship with the package (``multibench/result/``):

* ``published`` - scIB metric tables under ``result/scib_metric/``, one
  ``<category>/<dataset>/<method>/metric*.csv`` per run; shipped for
  vertical D11, diagonal D24/D25/D28 and cross D52 (mosaic has none). A
  table one level deeper, in a run-configuration subfolder
  (``<dataset>/MOFA2/filtered3/metric.csv``,
  ``<dataset>/MOFA2/kmeans/metric_kmeans.csv``), is read too; ``kbet/``
  folders hold raw kBET output and are skipped;
* ``rerun`` - the package's re-run sweeps behind the tutorial figures
  (``result/rerun/long_all_<dataset>.csv``; D11/D11s, D28/D28s, D45/D45s,
  D52/D52s). The files stamp their rows ``rerun-<package version>``; the
  loader reports plain ``source == "rerun"`` and keeps the version in
  ``frame.attrs["rerun_version"]``.

Every frame returned here carries the same seven columns
``metric, value, method, dataset, category, clustering, source`` so frames
from either source (or your own, via :func:`multibench.to_long`) can be
concatenated and handed to ``mtb.plot.bubble`` / ``mtb.plot.bar``. The
``source`` column holds ``"published"``, ``"rerun"`` or ``"user"`` (a long
file of your own keeps whatever it carries).

The scIB clustering and batch tables are the only metric set; ``metrics=``
selects within them with the vocabulary of :func:`multibench.evaluate`.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from .. import config
from .. import _compat
from . import catalog

__all__ = ["load_results", "available_datasets", "fetchable", "results_coverage",
           "recommend", "COLUMNS", "SOURCES", "DegenerateRerunWarning"]

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

#: a re-run row is degenerate when its ARI is below this ...
_DEGENERATE_RERUN_ARI = 0.01
#: ... while the published table scored the same (category, dataset, method)
#: above this: the re-run almost certainly failed silently (wrong label order,
#: a collapsed embedding) rather than the method being that bad.
_DEGENERATE_PUBLISHED_ARI = 0.2


class DegenerateRerunWarning(UserWarning):
    """A re-run row scored ARI ~0 where the published table scored well.

    Emitted by ``mtb.load_results`` and ``mtb.recommend`` with
    ``source="rerun"`` or ``"both"``.

    Notes
    -----
    **When it fires:**

    - a re-run row scored ARI below 0.01 where the published table scored
      the same category, dataset and method above 0.2. Such a row most
      likely comes from a failed re-run, for example a collapsed embedding
      or a wrong label order, not from the method itself;
    - never for a ``result_path`` file or a ``long_df`` frame passed to
      ``mtb.recommend``;
    - in the shipped sweeps: Conos on D28.

    **What to do:** drop the row before ranking (the message names the
    filter, here ``df[df.method != 'Conos']``); once you have decided how to
    treat those rows, silence it with
    ``warnings.simplefilter("ignore", mtb.DegenerateRerunWarning)``.
    """

# a result directory like ``Concerto_louvain`` is the method's louvain variant:
# split it into method id + clustering token instead of inventing a method id
_SUFFIX_RE = re.compile(r"^(.*)_(louvain|kmeans)$")

#: sub-directory of ``result_path`` holding the package re-run sweeps
_RERUN_DIR = "rerun"
_RERUN_GLOB = "long_all_*.csv"


def _rerun_source() -> str:
    """Provenance value of every re-run row: the plain token ``"rerun"``.

    The shipped sweep files stamp their rows ``rerun-<version that produced
    them>``; :func:`_split_rerun_tag` folds that to ``"rerun"`` (so
    ``df[df.source == "rerun"]`` works) and keeps the version in
    ``frame.attrs["rerun_version"]``. A file without a stamp gets this token
    and no version - not the running package's version, which would claim the
    current release produced numbers it merely read.
    """
    return "rerun"


#: a stamped re-run provenance value: ``rerun-<version>``
_RERUN_TAG_RE = re.compile(r"^rerun-(.+)$")


def _split_rerun_tag(source: pd.Series) -> tuple[pd.Series, set[str]]:
    """Fold ``rerun-<version>`` values to ``"rerun"``; return the versions seen.

    Any other value (``published``, ``user``, a plain ``rerun``) is returned
    unchanged.
    """
    col = source.astype(str)
    m = col.str.extract(_RERUN_TAG_RE, expand=False)
    versions = set(m.dropna().unique())
    out = source.where(m.isna(), _rerun_source())
    return out, versions


def _version_attr(versions: set[str]):
    """``attrs["rerun_version"]`` value: ``None`` (no stamped rows), the one
    version string, or a sorted tuple when files from several versions were
    loaded together."""
    if not versions:
        return None
    if len(versions) == 1:
        return next(iter(versions))
    return tuple(sorted(versions))


def _rerun_tag(version) -> str:
    """The full stamp for messages: ``rerun-0.2.1`` (``rerun`` when unknown)."""
    if version is None:
        return _rerun_source()
    if isinstance(version, tuple):
        return f"{_rerun_source()}-" + "/".join(version)
    return f"{_rerun_source()}-{version}"


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


#: appended to every "no table for <dataset>" error, for a results tree of
#: the user's own
_RESULT_PATH_HINT = " (pass result_path= for another results root)"


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
# sub-folders of a method result dir that are not run configurations holding a
# metric table (raw kBET output lives in ``kbet/benchmark_results*.csv``)
_NON_TABLE_SUBDIRS = {"kbet"}


def _find_metric_file(m_dir: Path, fname: str) -> Path | None:
    """``<m_dir>/<fname>`` or, when absent, the same file one level down in a
    run-configuration subfolder (``MOFA2/filtered3/metric.csv``,
    ``MOFA2/kmeans/metric_kmeans.csv``, ``MOFA2/8000HVG/metric.csv``).

    With several nested candidates the first in sorted order is used and a
    ``UserWarning`` lists them all. ``None`` when nothing matches.
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
                # the directory is the variant: its metric.csv holds the
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


def _published_root(base: Path, category: str) -> Path:
    """``<base>/scib_metric/<category folder>`` - the published tree of a category."""
    return base / config.metric_set_dir("scib") / config.category_folder(category)


def _load_published(category: str, datasets: list | None, clustering: str,
                    base: Path) -> pd.DataFrame:
    root = _published_root(base, category)
    if not root.exists():
        raise FileNotFoundError(_published_missing_msg(root, base, category))
    rows: list[pd.DataFrame] = []
    for ds_dir, m_dir, mfile, method, row_clust, coalesce in _iter_published(
            root, datasets, clustering):
        df = _read_metric_csv(mfile)
        # the correction file holds corrected ASW/iASW/iF1 for the default
        # clustering only; the louvain/kmeans files carry their own values
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
                f"under {root}{_RESULT_PATH_HINT}; datasets with published "
                f"tables: {have}")
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
        versions: set[str] = set()
        if "source" not in df.columns:
            df["source"] = _rerun_source()
        else:
            df["source"], versions = _split_rerun_tag(df["source"])
        if "clustering" not in df.columns:
            df["clustering"] = "default"
        df = df[COLUMNS]
        df.attrs["rerun_version"] = _version_attr(versions)
        out.append((f, df))
    return out


def _load_rerun(category: str | None, datasets: list | None, base: Path) -> pd.DataFrame:
    """The re-run rows for a category / dataset selection; the frame's
    ``attrs["rerun_version"]`` records the stamp(s) of the files that
    contributed rows."""
    files = _read_rerun_files(base)
    root = base / _RERUN_DIR
    if not files:
        raise FileNotFoundError(
            f"no re-run sweeps ({_RERUN_GLOB}) under {root}. They ship inside the "
            f"multibench wheel at multibench/result/rerun; check result_path= / "
            f"mtb.config.DEFAULT.result_path (currently {base}).")
    frames = []
    versions: set[str] = set()
    for _, df in files:
        sel = df
        if category is not None:
            sel = sel[sel["category"] == category]
        if datasets:
            sel = sel[sel["dataset"].isin(datasets)]
        if sel.empty:
            continue
        frames.append(sel)
        v = df.attrs.get("rerun_version")
        versions.update(v if isinstance(v, tuple) else ([v] if v else []))
    if not frames:
        allf = pd.concat([df for _, df in files], ignore_index=True)
        avail = (allf[["category", "dataset"]].drop_duplicates()
                 .sort_values(["category", "dataset"]))
        pairs = [f"{c}/{d}" for c, d in avail.itertuples(index=False)]
        raise FileNotFoundError(
            f"no re-run sweep for {category or 'any category'}/"
            f"{(datasets if len(datasets) > 1 else datasets[0]) if datasets else 'any dataset'}"
            f"{_RESULT_PATH_HINT}; available: {pairs}")
    out = pd.concat(frames, ignore_index=True)
    out.attrs["rerun_version"] = _version_attr(versions)
    return out


# --------------------------------------------------------------------------
# a user's own long CSV
# --------------------------------------------------------------------------
#: optional columns of a long CSV that ``load_results(result_path=<file>)``
#: keeps: ``scored_with`` is the provenance ``mtb.to_long`` writes
_KEPT_FILE_COLUMNS = ("scored_with",)


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
    # provenance defaults are filled per row, not only for an absent column: a
    # concat of a load_results frame with one lacking clustering/source leaves
    # NaN in those rows, which would survive to_csv -> load_results into plots
    for col, default in (("clustering", "default"), ("source", "user")):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df[COLUMNS + [c for c in _KEPT_FILE_COLUMNS if c in df.columns]]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def _canonical_dataset_ids(category, datasets, base, source) -> list:
    """Replace a dataset id that differs from a stored table id only in case.

    A case-insensitive filesystem (macOS, Windows) resolves the folder
    ``d52`` to ``D52``, so ``dataset="d52"`` would load and stamp the
    lower-case spelling into the frame (a concat with ``D52`` rows would then
    hold two datasets). The on-disk id wins, with one ``UserWarning``;
    anything else is returned unchanged so the usual "no table" error still
    names it.
    """
    try:
        have = _list_datasets(category, base, "both" if source == "user" else source,
                              "default")
    except Exception:
        return list(datasets)
    have = [str(h) for h in have]
    by_lower = {h.lower(): h for h in have}
    out = []
    for d in datasets:
        s = str(d)
        canon = by_lower.get(s.lower())
        if canon is not None and canon != s and s not in have:
            warnings.warn(
                f"dataset {s!r} is not a stored table id but {canon!r} is - using "
                f"that spelling (dataset ids are case-sensitive: 'D52', not 'd52')",
                UserWarning, stacklevel=4)
            out.append(canon)
        else:
            out.append(s)
    return out


def _check_methods(wanted: list, present) -> None:
    """Every requested method must be a registry id or a name in the frame.

    Raises ``KeyError`` with a did-you-mean hint (the same shape ``scan`` /
    ``method_info`` use) for anything else: an empty frame for a typo would
    be indistinguishable from "no rows for that method".
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
                                  "default", base)
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


def _warn_degenerate(out: pd.DataFrame, base: Path, stacklevel: int = 4,
                     rerun_version=None) -> None:
    """Emit :class:`DegenerateRerunWarning` for the rows
    :func:`_degenerate_rerun_rows` finds; ``rerun_version`` (the frame's
    ``attrs["rerun_version"]``) restores the full ``rerun-<version>`` stamp
    in the message, since the ``source`` column reads plain ``rerun``."""
    bad = _degenerate_rerun_rows(out, base)
    if bad.empty:
        return

    def _stamp(src) -> str:
        if src != _rerun_source():
            return str(src)
        m = _RERUN_TAG_RE.match(_rerun_tag(rerun_version))
        return f"re-run {m.group(1)}" if m else "re-run"

    items = ", ".join(f"{r.method}/{r.dataset} ({_stamp(r.source)}, ARI {r.rerun_ARI:.4f})"
                      for r in bad.itertuples())
    head = ("These re-run rows have" if len(bad) > 1 else "This re-run row has")
    warnings.warn(
        f"{head} ARI below {_DEGENERATE_RERUN_ARI}, while the published table has "
        f"above {_DEGENERATE_PUBLISHED_ARI} for the same method and dataset: "
        f"{items}. Such a "
        f"row most likely comes from a failed re-run, for example a collapsed "
        f"embedding or a wrong label order, not from the method itself. Drop it "
        f"before ranking: df[df.method != {bad.method.iloc[0]!r}].",
        DegenerateRerunWarning, stacklevel=stacklevel)


def _legacy_load_results_kwargs(kw: dict) -> dict:
    """0.2.x spellings of :func:`load_results`'s keywords: map or refuse.

    ``method=`` -> ``methods=``, ``metric=`` -> ``metrics=[...]``, ``task=`` /
    ``family=`` -> ``metrics=<token>`` (each with a ``DeprecationWarning``);
    ``metric_set=`` was removed in 0.3.0 and raises ``TypeError``.
    """
    if "metric_set" in kw:
        raise TypeError(
            "load_results() got metric_set=, removed in 0.3.0: only the scIB "
            "metric set exists - drop the argument")
    if "method" in kw:
        if kw.get("methods") is not None:
            raise ValueError("pass either method= or methods=, not both (method= is "
                             "deprecated)")
        kw["methods"] = kw.pop("method")
        _compat.warn("load_results(method=...)", "methods=", stacklevel=4)
    legacy = {n: kw.pop(n) for n in ("task", "family", "metric") if n in kw}
    if not legacy:
        return kw
    if kw.get("metrics") is not None:
        raise TypeError(
            f"load_results() got metrics= together with the deprecated "
            f"{sorted(legacy)}; pass metrics= only")
    token = None
    if "task" in legacy or "family" in legacy:
        if ("task" in legacy and "family" in legacy
                and legacy["task"] is not None and legacy["family"] is not None
                and legacy["task"] != legacy["family"]):
            raise ValueError(
                f"pass either family= or task= (they are the same selector; got "
                f"family={legacy['family']!r}, task={legacy['task']!r})")
        token = legacy.get("family") if legacy.get("family") is not None else legacy.get("task")
        for name in ("task", "family"):
            if name in legacy:
                _compat.warn(f"load_results({name}=...)", f"metrics={token!r}", stacklevel=4)
    if "metric" in legacy and legacy["metric"] is not None:
        codes = [catalog.canonical_metric(m) or m for m in _as_list(legacy["metric"])]
        _compat.warn("load_results(metric=...)", f"metrics={codes!r}", stacklevel=4)
        if token not in (None, "all"):
            # 0.2.x applied both filters: keep the codes inside the family
            fam = catalog.metric_selection(token).codes
            codes = [c for c in codes if c in fam]
        kw["metrics"] = codes
    else:
        kw["metrics"] = token
    return kw


@_compat.legacy_kwargs(_legacy_load_results_kwargs)
def load_results(
    category: str | None = None,
    *,
    dataset: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    metrics=None,
    clustering: str = "default",
    source: str = "published",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load the stored benchmark metric tables as one long table.

    Reads the published scIB tables, the package's re-run sweeps, or a long
    CSV of your own. Frames from any source concatenate with each other.

    Parameters
    ----------
    category : str, optional
        Integration category (``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``); ``None`` = every category with tables for ``source``.
    dataset : str or list of str
        Dataset id(s), e.g. ``"D11"`` or ``["D11", "D11s"]``; ``None`` = every
        dataset of the category.
    methods : str or list of str
        Method(s) to keep, alias tolerant and case-insensitive (``"mofa+"``
        -> ``MOFA2``, ``"totalvi"`` -> ``totalVI``); ``None`` = every method.
    metrics : None, str or list of str
        ``None`` / ``"all"`` = every metric; or ``"clustering"``,
        ``"batch"``, or a list of codes such as ``["ARI", "NMI"]``.
    clustering : {"default", "louvain", "kmeans"}
        Clustering variant of the published tables: ``metric.csv``,
        ``metric_louvain.csv`` or ``metric_kmeans.csv``.
    source : str
        ``"published"`` (scIB tables), ``"rerun"`` (package sweeps) or
        ``"both"``. For a ``result_path`` file: a value of its ``source``
        column; ``"published"`` / ``"both"`` keep every row.
    result_path : path-like
        A results root holding ``scib_metric/`` and/or ``rerun/``, or one
        long CSV file; ``None`` = the tables shipped in the package.

    Returns
    -------
    pandas.DataFrame
        One row per score, columns ``metric, value, method, dataset,
        category, clustering, source``, plus ``scored_with`` when a user CSV
        has it. ``attrs["rerun_version"]`` holds the re-run stamp (Notes).

    Raises
    ------
    FileNotFoundError
        No table for a requested category, dataset or source, or no ``result_path``.
    KeyError
        Unknown method name in ``methods``; the message suggests a close match,
        if any.
    ValueError
        Unknown ``category``, ``metrics``, ``clustering`` or ``source``, or a
        malformed ``result_path`` file.
    TypeError
        A positional argument after ``category``, or a retired keyword
        (Notes).

    Examples
    --------
    >>> import multibench as mtb
    >>> pub = mtb.load_results("diagonal", dataset="D28")   # published tables
    >>> # package re-runs
    >>> rr = mtb.load_results("diagonal", dataset="D28", source="rerun")
    >>> both = mtb.load_results("cross", dataset="D52", source="both",
    ...                         metrics="batch")
    >>> # your own rows
    >>> mine = mtb.load_results(result_path="mine.csv", source="user")

    Notes
    -----
    **Sources.** ``source`` picks the tables under a results root
    (default ``mtb.config.DEFAULT.result_path``, shipped in the package):

    - ``"published"`` - the scIB tables under ``result_path/scib_metric``
      (shipped: vertical D11, diagonal D24/D25/D28, cross D52; none for
      mosaic);
    - ``"rerun"`` - the package's re-run sweeps
      ``result_path/rerun/long_all_<dataset>.csv`` (shipped: vertical
      D11/D11s, diagonal D28/D28s, mosaic D45/D45s, cross D52/D52s);
    - ``"both"`` - the concatenation; the ``source`` column tells the rows
      apart.

    Re-run rows were scored by multibench 0.2.1's ``evaluate`` with the
    leidenalg backend and every cell type counted as isolated. Set
    ``mtb.config.DEFAULT.leiden_flavor = "leidenalg"`` to score comparable
    rows.

    **Your own file.** ``result_path`` may be one long CSV with at least the
    columns ``metric, value, method``, e.g. written by ``to_long(...).to_csv``
    or ``BatchResult.save()``. A file keeps whatever ``source`` /
    ``clustering`` values it carries; a missing column or a blank cell is
    filled with ``"user"`` / ``"default"`` per row. A ``scored_with`` column
    is kept, with NaN for the stored rows. ``source=`` then filters on the
    file's own ``source`` column:

    - ``"published"`` / ``"both"`` - keep every row;
    - ``"user"`` - the rows ``mtb.to_long`` wrote;
    - ``"rerun"`` - ``rerun`` and ``rerun-<version>`` rows (prefix match);
    - any value the file does not contain raises ``ValueError`` listing the
      ones present.

    **Method and metric names.** A ``methods`` entry that is neither a
    registry id nor a method in the loaded frame raises ``KeyError`` with a
    did-you-mean hint (``"unknown method 'Matlida'; did you mean
    'Matilda'?"``). Method folders of the published tables are reported by
    registry id (``MOFA+`` -> ``MOFA2``, ``Seurat(WNN)`` -> ``Seurat_WNN``);
    metric names read from the published tables or a file are canonical
    (``iFI`` -> ``iF1``).

    **Dataset ids.** Every requested id must have a table: ``["D11",
    "D99"]`` raises ``FileNotFoundError`` naming ``D99``, the datasets that
    have tables and the ``result_path=`` hint. Ids are case-sensitive; with
    a ``category``, an id that differs from a stored one only in case
    (``"d52"``) is replaced by the stored spelling, with a ``UserWarning``.

    **Metric selection.** Uses the vocabulary of ``mtb.evaluate``:

    - ``"clustering"`` - ``mtb.plot.CLUSTERING_METRICS`` (ARI, NMI, ASW,
      iASW, iF1, cLISI);
    - ``"batch"`` - ``mtb.plot.BATCH_METRICS`` (ASW_batch, GC, iLISI, kBET);
    - a list - exactly those codes (alias tolerant, ``["ari"]`` -> ARI).

    Every code must be one of ``mtb.catalog.known_metrics()`` or present in
    the frame: ``["ZZZ"]`` raises ``ValueError`` listing both. An unknown
    token raises too (``"dimension_reduction"`` is a ``list_tasks`` token,
    not a family).

    **Empty results.** A known method or metric with no rows gives an empty
    frame and a ``UserWarning``, not an error. Under ``source="published"``
    the warning also says whether the re-run sweeps hold that method
    (``"rerun has 1 dataset(s) ... pass source='rerun'"``): a published
    table need not score every method the package runs for its category.

    **Clustering variants.** A result directory named ``<method>_louvain`` /
    ``<method>_kmeans`` is that variant: it is reported under the method's
    canonical id with ``clustering`` set to the suffix, and only when that
    variant is requested. ``clustering=`` reads the published tables only;
    re-run rows are ``"default"``.

    **Warnings about the tables.** With ``"rerun"`` / ``"both"``, a re-run
    row whose ARI is ~0 while the published table scored the same method
    and dataset well emits a ``mtb.DegenerateRerunWarning`` (in the shipped
    sweeps: Conos on D28).

    When the selection holds one method in the chosen source while the
    other source holds more, a ``UserWarning`` says so (``"only one method
    (scMoMaT) in the published table for cross/D52 ... pass source='rerun'
    (or 'both')"``): a one-method table yields meaningless ranks. No warning
    when the other source has nothing more.

    **Re-run version.** The sweep files stamp their rows
    ``rerun-<version>``; the ``source`` column reads plain ``"rerun"`` and
    ``attrs["rerun_version"]`` keeps the version (``"0.2.1"``; a sorted
    tuple when files from several versions were loaded; ``None`` when no
    stamped row is present). ``pd.concat`` keeps ``attrs`` only when every
    input carries the same ones - read it before concatenating.

    **Retired keywords.** ``method=``, ``metric=``, ``task=`` and
    ``family=`` still work, with a ``DeprecationWarning``; ``metric_set=``
    raises ``TypeError``. The Changes page lists the replacements.

    See Also
    --------
    mtb.available_datasets : the dataset ids that have stored tables.

    mtb.results_coverage : which methods each source covers, per dataset.

    mtb.to_long : turns ``mtb.evaluate`` scores into this shape.

    mtb.recommend : ranks methods from these tables.

    mtb.plot.bubble : plots the frame.
    """
    if clustering not in _CLUSTERING_FILES:
        raise ValueError(
            f"unknown clustering {clustering!r}; valid: {sorted(_CLUSTERING_FILES)}"
        )
    # a token / None is validated before anything is read; a list of codes is
    # resolved after the load, against the codes the frame really holds too
    sel = catalog.metric_selection(metrics) if metrics is None or isinstance(metrics, str) \
        else None
    if category is not None:
        config.category_folder(category)      # raises "unknown category ... valid: [...]"
    datasets = _as_list(dataset)
    wanted_methods = _as_list(methods)

    base = _base_path(result_path)
    if result_path is not None and not base.exists() and (
            base.suffix.lower() in (".csv", ".tsv") or source not in SOURCES):
        # a file that is not there, not a source= the stored tables lack
        raise FileNotFoundError(
            f"result_path {str(result_path)!r} does not exist (working directory: "
            f"{Path.cwd()})")
    if datasets and category is not None and not base.is_file():
        datasets = _canonical_dataset_ids(category, datasets, base, source)
    rerun_versions: set[str] = set()

    if base.is_file():
        out = _load_long_csv(base)
        # a file keeps its own provenance values; only the stamp is parsed
        _, rerun_versions = _split_rerun_tag(out["source"])
        if source not in ("published", "both"):
            # filter on the file's own source column
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
                    frames.append(_load_published(cat, datasets, clustering, base))
                    got_any = True
                except FileNotFoundError as e:
                    errors.append(str(e))
            if source in ("rerun", "both"):
                try:
                    rr = _load_rerun(cat, datasets, base)
                    v = rr.attrs.get("rerun_version")
                    rerun_versions.update(v if isinstance(v, tuple) else ([v] if v else []))
                    frames.append(rr)
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
                avail = _list_datasets(category, base, source, clustering)
                raise FileNotFoundError(
                    f"no {source} results for {category or 'any category'}/"
                    f"{missing if len(missing) > 1 else missing[0]}{_RESULT_PATH_HINT}; "
                    f"datasets with {source} tables: {avail}")
        if source != "both":
            # a one-method table ranks nothing; say so when the other source
            # holds a real table for the same selection
            _warn_single_method(out, source, category, datasets, clustering, base)

    # ---- filters ---------------------------------------------------------
    where = f"{category or 'any category'}/{datasets if datasets else 'any dataset'}"
    if wanted_methods is not None:
        avail = sorted(out["method"].unique())
        _check_methods(wanted_methods, avail)
        # alias tolerant and case-insensitive: 'mofa+' -> MOFA2, 'scbridge' -> scBridge
        want = {catalog.canonical_id(m).lower() for m in wanted_methods}
        out = out[out["method"].map(lambda m: catalog.canonical_id(m).lower()).isin(want)]
        if out.empty:
            shown = wanted_methods if len(wanted_methods) > 1 else wanted_methods[0]
            hint = ""
            if source == "published" and not base.is_file():
                hint = _other_source_hint(category, datasets, wanted_methods, base)
            warnings.warn(
                f"no {source} rows for method {shown!r} in {where} (available: "
                f"{avail}){hint}", UserWarning, stacklevel=3)
    have_metrics = sorted(out["metric"].unique())
    if sel is None:
        sel = catalog.metric_selection(metrics, extra=have_metrics)
    if sel.codes is not None:
        out = out[out["metric"].isin(sel.codes)]
        if out.empty and sel.explicit:
            warnings.warn(
                f"metric(s) {sel.codes} not present in {where} (source={source!r}; "
                f"available: {have_metrics})", UserWarning, stacklevel=3)
    out = out[COLUMNS + [c for c in _KEPT_FILE_COLUMNS if c in out.columns]]
    out = out.reset_index(drop=True)
    out.attrs["rerun_version"] = _version_attr(rerun_versions)
    if source in ("rerun", "both") and not base.is_file():
        _warn_degenerate(out, base, rerun_version=out.attrs["rerun_version"])
    return out


def _other_source_methods(source: str, cats: list, datasets, clustering: str,
                          base: Path) -> set[str]:
    """Method ids the other stored source holds for the same selection
    (``published`` <-> ``rerun``); empty when it has none. Never raises."""
    other = "rerun" if source == "published" else "published"
    have: set[str] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cat in cats:
            try:
                if other == "rerun":
                    df = _load_rerun(cat, datasets, base)
                else:
                    df = _load_published(cat, datasets, clustering, base)
            except (FileNotFoundError, ValueError):
                continue
            have.update(df["method"].astype(str).unique())
    return have


def _warn_single_method(out: pd.DataFrame, source: str, category, datasets,
                        clustering: str, base: Path, stacklevel: int = 4) -> None:
    """One ``UserWarning`` (the CLI's "only one method in this table" text)
    when the loaded selection holds a single method while the other stored
    source holds more methods for the same category/dataset(s). Silent when
    the other source has nothing more."""
    n = out["method"].nunique()
    if n >= 2:
        return
    cats = sorted(out["category"].astype(str).unique()) or (
        [category] if category is not None else list(_CATEGORIES))
    have = _other_source_methods(source, cats, datasets, clustering, base)
    if len(have) <= n:
        return
    other = "rerun" if source == "published" else "published"
    where = category or "/".join(cats)
    if datasets:          # no dataset filter: the selection is the category
        where += f"/{datasets[0] if len(datasets) == 1 else list(datasets)}"
    warnings.warn(
        f"only one method ({out['method'].iloc[0]}) in the {source} table for "
        f"{where}; ranks and Overall bars are not meaningful with a single "
        f"method - the {other} tables hold {len(have)} methods for it "
        f"({', '.join(sorted(have, key=str.lower))}): pass source={other!r} "
        f"(or 'both')", UserWarning, stacklevel=stacklevel)


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


def _list_datasets(category, base: Path, source: str, clustering: str) -> list[str]:
    """Sorted dataset ids with a table under ``base`` for ``source`` (published
    ids only when they hold the ``clustering`` variant's file). No validation;
    a missing category folder contributes nothing."""
    cats = [category] if category is not None else list(_CATEGORIES)
    found: set[str] = set()
    if source in ("published", "both"):
        for cat in cats:
            root = _published_root(base, cat)
            if not root.exists():
                continue
            for ds_dir, *_ in _iter_published(root, None, clustering):
                found.add(ds_dir.name)
    if source in ("rerun", "both"):
        for _, df in _read_rerun_files(base):
            found.update(df.loc[df["category"].isin(cats), "dataset"].astype(str).unique())
    return sorted(found)


def available_datasets(
    category: str | None = None,
    *,
    source: str = "published",
    result_path: Path | str | None = None,
) -> list[str]:
    """Dataset ids that ship stored results, not the datasets that can be downloaded.

    Stored results are the metric tables ``mtb.load_results`` reads;
    ``mtb.data.fetchable`` lists the ids that can be downloaded.

    Parameters
    ----------
    category : str, optional
        Integration category; ``None`` = the union over all four.
    source : {"published", "rerun", "both"}
        Which stored tables to look at (see ``mtb.load_results``).
    result_path : path-like
        Results root; ``None`` = the tables shipped in the package.

    Returns
    -------
    list of str
        Sorted dataset ids.

    Raises
    ------
    ValueError
        Unknown ``category`` or ``source``.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.available_datasets()  # ['D11', 'D24', 'D25', 'D28', 'D52']
    >>> mtb.available_datasets("diagonal")  # ['D24', 'D25', 'D28']
    >>> mtb.available_datasets(source="rerun")  # ['D11', 'D11s', 'D28', ...]

    Notes
    -----
    **Downloadable datasets.** Only a few of the benchmark's datasets
    are downloadable: the release assets of ``mtb.data.fetch``, listed by
    ``mtb.data.fetchable()``. An id returned here but not there has metric
    tables to plot and rank against, and no data file this package can
    obtain.

    **What counts.** Published ids are those holding at least one method's
    default-clustering table (``metric.csv``) - what
    ``load_results(category, dataset=...)`` loads. A category folder that
    does not exist (``mosaic`` has no published tables) contributes
    nothing, without an error. A ``result_path`` that does not exist gives
    a ``UserWarning`` and ``[]``.

    **Retired keywords.** ``metric_set=`` and ``clustering=`` are not
    parameters; passing either raises ``TypeError``.

    See Also
    --------
    mtb.data.fetchable : the ids ``mtb.data.fetch`` can download.

    mtb.results_coverage : which methods have results for each dataset.

    mtb.load_results : loads the tables behind these ids.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    base = _base_path(result_path)
    if not base.exists():
        warnings.warn(
            f"result_path {base} does not exist - nothing can be listed "
            f"(default root: {config.DEFAULT.result_path})",
            UserWarning, stacklevel=2)
        return []
    if category is not None:
        config.category_folder(category)
    return _list_datasets(category, base, source, "default")


def fetchable() -> list[str]:
    """Dataset ids ``mtb.data.fetch`` can download.

    The companion of ``mtb.available_datasets``, which lists the ids with
    stored results; the two sets overlap but are not the same.

    Returns
    -------
    list of str
        Dataset ids in natural order (``D11``, ``D28``, ``D45``, ...).

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.data.fetchable()
    ['D11', 'D28', 'D45', 'D46', 'D52']

    Notes
    -----
    **Overlap with stored results:** ``D24`` has published metric tables and
    no downloadable file, ``D46`` downloads and has no stored table. The
    list is exactly the set of ids ``fetch()`` accepts.
    """
    from .fetch import AVAILABLE
    return sorted(AVAILABLE, key=catalog._dataset_sort_key)


def results_coverage(
    category: str | None = None,
    *,
    source: str = "both",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """Which (category, dataset, method) combinations have results, and where from.

    Use it to choose the ``source`` to pass to ``mtb.load_results`` or
    ``mtb.recommend``.

    Parameters
    ----------
    category : str, optional
        Integration category; ``None`` = all four.
    source : {"both", "published", "rerun"}
        Which stored tables to scan; ``"both"`` = published and re-run.
    result_path : path-like
        Results root; ``None`` = the tables shipped in the package.

    Returns
    -------
    pandas.DataFrame
        One sorted row per distinct ``category, dataset, method, clustering,
        source``; ``source`` is ``"published"`` or ``"rerun"``.

    Raises
    ------
    ValueError
        Unknown ``category`` or ``source``.

    Examples
    --------
    >>> import multibench as mtb
    >>> cov = mtb.results_coverage("cross")
    >>> cov[cov.dataset == "D52"]     # scMoMaT (published) + 8 methods (rerun)
    >>> cov.attrs["rerun_version"]    # '0.2.1'
    >>> cov.groupby(["category", "source"]).method.nunique()

    Notes
    -----
    **Clustering variants.** The published tree is scanned for every
    clustering variant (default, louvain, kmeans), so a method that only
    exists as e.g. a ``_louvain`` directory shows up under
    ``clustering="louvain"``.

    **Empty categories.** A category that has no tables raises nothing; it
    has no rows.

    **Warnings.** The degenerate-row and one-method warnings of
    ``mtb.load_results`` are silenced here.

    **Re-run version.** ``attrs["rerun_version"]`` holds the package version
    stamped on the re-run sweeps, as in ``mtb.load_results``.

    See Also
    --------
    mtb.available_datasets : just the dataset ids.

    mtb.load_results : the scores behind each row.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    cols = ["category", "dataset", "method", "clustering", "source"]
    cats = [category] if category is not None else list(_CATEGORIES)
    if category is not None:
        config.category_folder(category)
    frames = []
    versions: set[str] = set()
    # a coverage scan asks where rows are, not whether they are sound or
    # rankable: the degenerate-row and one-method warnings of load_results
    # are silenced for the per-variant probes made here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateRerunWarning)
        warnings.filterwarnings("ignore", message="only one method", category=UserWarning)
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
                    rr = load_results(cat, result_path=result_path, source="rerun")
                    v = rr.attrs.get("rerun_version")
                    versions.update(v if isinstance(v, tuple) else ([v] if v else []))
                    frames.append(rr[cols])
                except FileNotFoundError:
                    pass
    if not frames:
        out = pd.DataFrame(columns=cols)
    else:
        out = pd.concat(frames, ignore_index=True).drop_duplicates()
        out = out.sort_values(cols).reset_index(drop=True)
    out.attrs["rerun_version"] = _version_attr(versions)
    return out


#: the modalities each shipped stored dataset measured: metadata about the
#: shipped tables, so ``recommend`` can say which data a ranking describes
_STORED_DATASET_MODALITIES = {
    "D11": ("rna", "adt"), "D11s": ("rna", "adt"),
    "D24": ("rna", "atac"), "D25": ("rna", "atac"),
    "D28": ("rna", "atac"), "D28s": ("rna", "atac"),
    "D45": ("rna", "atac"), "D45s": ("rna", "atac"),
    "D52": ("rna", "adt"), "D52s": ("rna", "adt"),
}
_MODALITY_ORDER = ("rna", "adt", "atac")


def _unmeasured_note(category: str, datasets, wanted: set[str], stored: bool) -> str:
    """The recommend() warning line when a requested modality was measured by
    none of the ranked datasets; ``""`` when every one was, or when a dataset
    is not a shipped one (its modalities are unknown)."""
    datasets = [str(d) for d in datasets]
    if not datasets or any(d not in _STORED_DATASET_MODALITIES for d in datasets):
        return ""
    measured = set().union(*(_STORED_DATASET_MODALITIES[d] for d in datasets))
    missing = [m for m in _MODALITY_ORDER if m in wanted and m not in measured]
    if not missing:
        return ""
    groups: dict[tuple, list[str]] = {}
    for d in sorted(datasets, key=catalog._dataset_sort_key):
        groups.setdefault(_STORED_DATASET_MODALITIES[d], []).append(d)
    where = "; ".join(f"{', '.join(ds)} ({'+'.join(mods)})" for mods, ds in groups.items())
    data = "+".join(m.upper() for m in _MODALITY_ORDER if m in wanted)
    head = f"Stored {category} scores" if stored else f"The {category} scores in long_df"
    return (f"{head} come from {where}. None of these datasets measured "
            f"{' or '.join(missing)}, so this ranking does not describe {data} data.")


def _and(names) -> str:
    """``'A'``, ``'A and B'``, ``'A, B and C'``."""
    names = [str(n) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _runtime(method_id: str, method_info) -> dict:
    """``method_info(m)["runtime"]`` (``{"tier", "worst_sec", "observed"}``)."""
    return dict(method_info(method_id)["runtime"])


def _legacy_recommend_kwargs(kw: dict) -> dict:
    """0.2.x spellings of :func:`recommend`'s keywords: ``task=`` / ``family=``
    become ``metrics=<token>`` with a ``DeprecationWarning`` (``family`` wins
    when both are given, as before; an explicit ``metrics=`` keeps winning
    over either)."""
    legacy = {n: kw.pop(n) for n in ("task", "family") if n in kw}
    if not legacy:
        return kw
    token = legacy.get("family") if legacy.get("family") is not None else legacy.get("task")
    token = "all" if token is None else token
    for name in ("task", "family"):
        if name in legacy:
            _compat.warn(f"recommend({name}=...)", f"metrics={token!r}", stacklevel=4)
    if kw.get("metrics") is None:
        kw["metrics"] = token
    return kw


@_compat.legacy_kwargs(_legacy_recommend_kwargs)
def recommend(
    category: str,
    *,
    modalities: list[str] | None = None,
    atac: str | None = None,
    methods: list[str] | None = None,
    metrics=None,
    long_df: pd.DataFrame | None = None,
    min_methods: int = 2,
    source: str = "published",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """Rank methods from stored results, with the share of datasets each was scored on.

    Scores each method on the stored metric tables (or ``long_df``) and also
    lists every available method it could not score; the rules are in Notes.

    Parameters
    ----------
    category : str
        Integration category to rank: ``vertical``, ``diagonal``, ``mosaic``
        or ``cross``.
    modalities : list of str
        Keep only methods that consume all of these modalities, e.g.
        ``["rna", "adt"]``; tokens as in ``mtb.find_methods``. ``None`` = no
        filter.
    atac : str
        Keep only methods that read this ATAC representation: ``"peak"`` or
        ``"gene_activity"``; ``None`` = no filter.
    methods : list of str
        Rank only these methods (alias tolerant, case-insensitive), among
        themselves; ``None`` = every method.
    metrics : None, str or list of str
        ``None`` = the ``"clustering"`` family (the benchmark's main
        ranking); or ``"batch"``, ``"all"``, or a list of codes.
    long_df : pandas.DataFrame
        Frame to score (``metric, value, method, dataset``) instead of the
        stored tables, e.g. ``pd.concat([published, mine])``.
    min_methods : int
        Datasets with fewer methods than this are dropped.
    source : {"published", "rerun", "both"}
        Stored tables to load when ``long_df`` is not given; ``"both"``
        averages the scores present in both.
    result_path : path-like
        Results root; ``None`` = the tables shipped in the package.

    Returns
    -------
    pandas.DataFrame
        One row per method, best first, unscored methods last. Read
        ``method``, ``grand_score``, ``coverage`` and ``datasets``; all
        columns and ``attrs`` are in Notes.

    Raises
    ------
    ValueError
        Nothing left to rank (Notes lists the cases), or an unknown
        ``metrics`` value.
    KeyError
        Unknown method name in ``methods``; the message suggests a close match,
        if any.
    FileNotFoundError
        No stored results for the category and source.

    Warns
    -----
    UserWarning
        Partial coverage, dropped or unscored methods, or igraph-scored rows;
        one message (Notes).

    Examples
    --------
    >>> import multibench as mtb
    >>> r = mtb.recommend("vertical", modalities=["rna", "adt"])
    >>> r[["method", "grand_score", "coverage", "datasets"]]
    >>> r.attrs["not_scored"]                   # available, but no published rows
    >>> mtb.recommend("diagonal", atac="peak")  # methods that read peaks
    >>> mtb.recommend("cross", source="rerun")  # cross has one published method

    Notes
    -----
    **Score.** ``grand_score`` is the ``overall="mean_overall"`` score of
    ``mtb.plot.bar``: within each dataset, the min-max scaled mean of the
    per-metric max-ranks, then averaged over the datasets the method was run
    on.

    **Ranking rules.**

    - Only methods this package runs for the category are ranked - the set
      ``mtb.list_methods(category=...)`` lists. Other registry methods in
      the table (MOFA2 or Multigrate in a cross table) are dropped before
      the within-dataset ranks are taken, and named in the warning and in
      ``attrs["dropped_methods"]``. A name the registry does not know (your
      own method in ``long_df``) is kept.
    - A dataset holding fewer than ``min_methods`` methods is dropped. The
      min-max of a single method is always 1.0, so a lone method would win
      that dataset.
    - ``n_datasets`` / ``n_datasets_total`` / ``coverage`` say how much of
      the method x dataset matrix each score rests on.
    - Every method the package runs for the category (and ``modalities``)
      that has no rows in the chosen source is still listed, after the scored
      rows, with ``grand_score`` NaN, ``n_datasets`` 0 and ``coverage`` 0.0.
      A published table need not score every such method;
      the re-run sweeps may cover more (``source="rerun"``).

    **Columns.**

    - ``method`` - the method id;
    - ``grand_score`` - the score above; NaN when unscored;
    - ``n_datasets`` / ``n_datasets_total`` - datasets the method was scored
      on / datasets kept for the ranking;
    - ``coverage`` - ``n_datasets / n_datasets_total``;
    - ``needs_labels``, ``runtime_tier``, ``worst_sec``, ``env``,
      ``output_kind`` - registry metadata for the category; ``None`` for
      ids that are not registry methods (your own method, a result-dir
      token);
    - ``datasets`` - the dataset ids the score comes from, comma-joined;
      ``""`` when unscored.

    **Which data the ranking describes.** ``modalities`` and ``atac`` select
    methods; they do not change the datasets the scores come from. The
    shipped tables measured rna+adt (D11, D11s, D52, D52s) or rna+atac (D24,
    D25, D28, D28s, D45, D45s). When ``modalities`` or ``atac`` names a
    modality that none of the ranked datasets measured, a ``UserWarning``
    says so.

    The stored Seurat_v5 diagonal scores come from runs with a separate paired
    bridge dataset; ``mtb.scan`` marks Seurat_v5 runnable only when your RNA and
    ATAC files hold the same cells.

    **Frame attrs.** ``frame.attrs`` records the choices the ranking was
    made under:

    - ``"metrics"`` - the family token, or the list of codes;
    - ``"family"`` - the token; ``None`` when a list was given;
    - ``"source"`` - ``"published"`` / ``"rerun"`` / ``"both"``, or
      ``"long_df"``;
    - ``"not_scored"`` (also under ``"missing"``) - the unscored method ids;
    - ``"dropped_methods"`` - registry methods present in the table but not
      run by this package for the category.

    **Selections.** ``metrics="batch"`` scores ASW_batch, GC, iLISI, kBET;
    ``"all"`` every metric present; a list exactly those codes (alias
    tolerant). ``methods`` is resolved as in ``mtb.load_results``
    (``"mofa+"`` -> MOFA2, ``"totalvi"`` -> totalVI), and the unscored line
    of the warning is restricted to the same set, so a requested method
    without rows is still reported as such. ``modalities`` and ``atac``
    select methods with ``mtb.find_methods``, under the same token rule.

    **Warning.** One ``UserWarning`` with one line per finding summarises
    a requested modality the datasets did not measure, dropped methods and
    datasets, partial coverage and the unscored methods. The stored tables
    were clustered with leidenalg. A line also names the ``long_df`` methods
    clustered with igraph when ARI, NMI or iF1 ranks them against stored
    rows of their dataset.

    **Errors.** ``ValueError`` is raised when:

    - no dataset holds ``min_methods`` methods (when the other stored source
      holds more methods for the category the message says so - cross/D52:
      ``pass source='rerun'``);
    - the frame has none of the requested metrics (the message lists the
      metrics it does have);
    - every row belongs to a method the package does not run for the
      category;
    - ``methods=``, ``modalities=`` or ``atac=`` leaves no scored method;
    - an unknown modality token or ``atac`` value, or a representation
      token that contradicts ``atac`` (the rule of ``mtb.find_methods``);
    - a ``metrics`` token or code is unknown.

    **Retired keywords.** ``task=`` / ``family=`` still work as
    ``metrics=<token>``, with a ``DeprecationWarning``.

    See Also
    --------
    mtb.load_results : the stored tables it ranks.

    mtb.results_coverage : which methods each source covers.

    mtb.plot.bar : plots the same overall score per method.
    """
    from ..plot import style

    config.category_folder(category)
    from ..discover import _modality_filter, find_methods

    # the modality selection is validated before any table is read
    want_mods, atac = _modality_filter(modalities, atac)
    filtered = want_mods is not None or atac is not None
    if want_mods is None and atac is not None:
        want_mods = {"atac"}
    if metrics is None:
        metrics = "clustering"           # the benchmark's headline ranking
    sel = catalog.metric_selection(metrics)   # a token is validated before any load
    long_df_was_none = long_df is None
    if long_df is None:
        # load every metric and filter locally, so the "metrics present"
        # error below can name what the frame really holds
        long_df = load_results(category, source=source, result_path=result_path)
    df = long_df.copy()
    label = f"source={source!r}" if long_df_was_none else "long_df"
    table_noun = {"published": "published table", "rerun": "re-run sweeps",
                  "both": "stored tables"}.get(source, "stored tables") \
        if long_df_was_none else "long_df frame"

    from ..engine.registry import list_methods

    # A registry method not listed for the category (MOFA2 / Multigrate in a
    # cross table) is dropped before the per-dataset ranks are taken, so it
    # cannot shape the other methods' ranks; a name the registry does not
    # know (the user's own method) is kept.
    listed = set(list_methods(category=category))
    registry_ids = set(catalog._registry_ids())
    canon = df["method"].map(catalog.canonical_id)
    foreign = sorted({m for m, c in zip(df["method"], canon)
                      if c in registry_ids and c not in listed}, key=str.lower)
    if foreign:
        df = df[~canon.isin({catalog.canonical_id(m) for m in foreign})]
        if df.empty:
            raise ValueError(
                f"every row in {label} belongs to a method this package does "
                f"not run for {category} ({', '.join(foreign)}); "
                f"mtb.list_methods(category={category!r}) lists the rankable "
                f"ones")
    want_ids: set[str] | None = None
    if methods is not None:
        wanted = _as_list(methods)
        _check_methods(wanted, sorted(df["method"].astype(str).unique()))
        want_ids = {catalog.canonical_id(m) for m in wanted}
        want_lower = {w.lower() for w in want_ids}
        df = df[df["method"].map(lambda m: catalog.canonical_id(m).lower()).isin(want_lower)]
        if df.empty:
            raise ValueError(
                f"none of methods={list(wanted)} has rows in {label} for "
                f"{category}; methods with rows: "
                f"{sorted(long_df['method'].astype(str).unique())}")
    if sel.explicit:
        # a user's own metric name in long_df is selectable too
        sel = catalog.metric_selection(metrics, extra=df["metric"].astype(str).unique())
    if sel.codes is not None:
        df = df[df["metric"].isin(sel.codes)]
        if df.empty:
            raise ValueError(
                f"no {metrics!r} metrics ({sel.codes}) in the frame; metrics present: "
                f"{sorted(long_df['metric'].unique())}")
    if "dataset" not in df.columns:
        df["dataset"] = "all"

    parts = style.per_dataset_ranks(df, sel.codes if sel.explicit else None)
    degenerate = sorted(ds for ds, mat in parts.items() if len(mat.index) < min_methods)
    kept = {ds: mat for ds, mat in parts.items() if ds not in degenerate}
    if not kept:
        hint = ""
        if long_df_was_none and source != "both":
            # the other stored source may hold a real table for the category
            # (cross/D52: one published method, several re-run ones)
            other = "rerun" if source == "published" else "published"
            have = _other_source_methods(source, [category], None, "default",
                                         _base_path(result_path))
            if len(have) >= min_methods:
                hint = (f" The {other} tables hold {len(have)} methods for {category} "
                        f"({', '.join(sorted(have, key=str.lower))}): pass "
                        f"source={other!r} (or 'both').")
        raise ValueError(
            f"no dataset in {category} holds >= {min_methods} methods "
            f"({len(parts)} dataset(s): {sorted(parts)}); a ranking over "
            f"single-method datasets is meaningless (min-max of one value is "
            f"1.0).{hint} Otherwise pass long_df= with more methods, or lower "
            f"min_methods.")
    per_ds = pd.DataFrame({ds: style.compute_overall(mat) for ds, mat in kept.items()})
    # igraph-scored rows ranked against the leidenalg-scored stored rows of
    # their dataset: the plot functions' warning, on the ranked rows only
    backend = style.backend_warning(df[df["dataset"].isin(list(kept))])
    grand = per_ds.mean(axis=1)
    n_ds = per_ds.notna().sum(axis=1)
    n_total = per_ds.shape[1]

    keep_methods = list(grand.index)
    if filtered:
        allowed = set(find_methods(category=category, modalities=modalities, atac=atac))
        keep_methods = [m for m in keep_methods if catalog.canonical_id(m) in allowed]
        if not keep_methods:
            asked = " and ".join(
                ([f"consumes modalities {list(modalities)}"] if modalities is not None
                 else []) + ([f"reads atac={atac!r}"] if atac is not None else []))
            raise ValueError(
                f"no scored method in {category} {asked}; methods with results: "
                f"{sorted(grand.index)}; methods matching the selection: "
                f"{sorted(allowed)}")

    # Methods wired for the category but absent from the source. The registry
    # 'clustering' tag covers every embedding method of a category; the
    # 'batch' tag is incomplete (vertical), so the requested family never
    # gates this list.
    wired = find_methods(category=category, task="clustering", modalities=modalities,
                         atac=atac, runnable=True)
    if want_ids is not None:
        wired = [m for m in wired if m in want_ids]
    scored_ids = {catalog.canonical_id(m) for m in keep_methods}
    missing = sorted((m for m in wired if m not in scored_ids), key=str.lower)
    # a method with rows only in dropped (< min_methods) datasets is unscored
    # for a different reason than "no rows at all" - say which
    in_frame = {catalog.canonical_id(m) for m in df["method"].unique()}
    only_dropped = [m for m in missing if m in in_frame]
    no_rows = [m for m in missing if m not in in_frame]

    from ..engine import registry, envs
    from ..discover import method_info

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
        rt = _runtime(catalog.canonical_id(m), method_info) if spec is not None else {}
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
            "datasets": ", ".join(sorted((str(d) for d in per_ds.columns
                                          if pd.notna(per_ds.loc[m, d])),
                                         key=catalog._dataset_sort_key)) if scored else "",
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("grand_score", ascending=False, kind="mergesort",
                          na_position="last").reset_index(drop=True)
    out.attrs["metrics"] = list(sel.codes) if sel.explicit else sel.family
    out.attrs["family"] = None if sel.explicit else sel.family
    out.attrs["source"] = source if long_df_was_none else "long_df"
    out.attrs["not_scored"] = list(missing)
    out.attrs["missing"] = list(missing)
    out.attrs["dropped_methods"] = list(foreign)

    notes = []
    if filtered:
        unmeasured = _unmeasured_note(category, list(per_ds.columns), want_mods,
                                      long_df_was_none)
        if unmeasured:
            notes.append(unmeasured)
    if degenerate:
        one = len(degenerate) == 1
        notes.append(
            f"{'Dataset' if one else 'Datasets'} {_and(degenerate)} "
            f"{'has' if one else 'have'} fewer than {min_methods} methods and "
            f"{'is' if one else 'are'} left out of the ranking. A min-max score "
            f"over one method is always 1.0.")
    if foreign:
        one = len(foreign) == 1
        it, they = ("it", "it is") if one else ("them", "they are")
        notes.append(
            f"{_and(foreign)} {'has' if one else 'have'} scores in the "
            f"{table_noun}, but this package does not run {it} for {category}, "
            f"so {they} left out of the ranking. "
            f"mtb.list_methods(category={category!r}) does not list {it}.")
    partial = out[(out["coverage"] < 1.0) & out["grand_score"].notna()]
    if not partial.empty:
        notes.append(
            "Some methods have scores on only part of the datasets: "
            + ", ".join(f"{r.method} {r.n_datasets} of {r.n_datasets_total}"
                        for r in partial.itertuples())
            + ". Check the coverage column before reading the order.")
    if backend:
        notes.append(backend)
    if only_dropped:
        one = len(only_dropped) == 1
        notes.append(
            f"{_and(only_dropped)} {'has' if one else 'have'} scores only on "
            f"datasets left out of the ranking, so {'it is' if one else 'they are'} "
            f"listed last.")
    if no_rows:
        scores = ({"published": "published scores", "rerun": "re-run scores"}
                  .get(source, "stored scores") if long_df_was_none
                  else "scores in long_df")
        head = (f"{no_rows[0]} has no {scores} and is listed last"
                if len(no_rows) == 1 else
                f"These {len(no_rows)} methods have no {scores} and are listed "
                f"last: {', '.join(no_rows)}")
        notes.append(
            head + "."
            + (' Try source="rerun".' if source == "published" and long_df_was_none
               else ""))
    if notes:
        warnings.warn(f"recommend({category!r}):\n  - " + "\n  - ".join(notes),
                      UserWarning, stacklevel=3)
    return out
