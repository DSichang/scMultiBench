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
  D52/D52s). The files are stamped ``rerun-<package version that produced
  them>``; the loader reports those rows as plain ``source == "rerun"`` and
  keeps the stamp in ``frame.attrs["rerun_version"]``.

Every frame returned here carries the same seven columns
``metric, value, method, dataset, category, clustering, source`` so frames
from either source (or your own, via :func:`multibench.to_long`) can be
concatenated and handed to ``mtb.plot.bubble`` / ``mtb.plot.bar``. The
``source`` column holds ``"published"``, ``"rerun"`` or ``"user"`` (a long
file of your own keeps whatever it carries).

The scIB clustering + batch tables are the only metric set (the 0.2.x
``metric_set=`` keyword is gone); ``metrics=`` selects within them with the
same vocabulary :func:`multibench.evaluate` uses.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from .. import config
from ..eval import _compat
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
    """Provenance value of every re-run row: the plain token ``"rerun"``.

    The shipped sweep files stamp their rows ``rerun-<version that produced
    them>``; :func:`_split_rerun_tag` folds that to ``"rerun"`` (so
    ``df[df.source == "rerun"]`` works, as every doc says) and keeps the
    version in ``frame.attrs["rerun_version"]``. A file without a stamp gets
    this token and no version - deliberately NOT the running package's
    version, which would claim the current release produced numbers it merely
    read.
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
            f"{(datasets if len(datasets) > 1 else datasets[0]) if datasets else 'any dataset'}; "
            f"available: {pairs}")
    out = pd.concat(frames, ignore_index=True)
    out.attrs["rerun_version"] = _version_attr(versions)
    return out


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
def _canonical_dataset_ids(category, datasets, base, source) -> list:
    """Replace a dataset id that differs from a stored table id only in case.

    macOS / Windows filesystems open ``long_all_d52.csv`` for ``D52`` too, so
    ``dataset="d52"`` used to load and stamp the lower-case spelling into the
    frame (a concat with ``D52`` rows then held two datasets). The on-disk id
    wins, with one ``UserWarning``; anything else is returned unchanged so the
    usual "no table" error still names it.
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
    in the message, since the ``source`` column now reads plain ``rerun``."""
    bad = _degenerate_rerun_rows(out, base)
    if bad.empty:
        return

    def _stamp(src) -> str:
        return _rerun_tag(rerun_version) if src == _rerun_source() else str(src)

    items = ", ".join(f"{r.method}/{r.dataset} ({_stamp(r.source)} ARI {r.rerun_ARI:.4f} vs "
                      f"published {r.published_ARI:.2f})" for r in bad.itertuples())
    warnings.warn(
        f"degenerate re-run row(s) - ARI < {_DEGENERATE_RERUN_ARI} where the "
        f"published table scored > {_DEGENERATE_PUBLISHED_ARI}: {items}. The "
        f"re-run most likely failed silently (a collapsed embedding or a wrong "
        f"label order), so the row says nothing about the method; drop it "
        f"before ranking (df[df.method != {bad.method.iloc[0]!r}]) or compare "
        f"with source='published'.", DegenerateRerunWarning, stacklevel=stacklevel)


def _legacy_load_results_kwargs(kw: dict) -> dict:
    """0.2.x spellings of :func:`load_results`'s keywords: map or refuse.

    ``method=`` -> ``methods=``, ``metric=`` -> ``metrics=[...]``, ``task=`` /
    ``family=`` -> ``metrics=<token>`` (each with a ``DeprecationWarning``);
    ``metric_set=`` is gone and raises ``TypeError``.
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
    """Benchmark metric tables as a tidy long frame.

    Parameters
    ----------
    category : {"vertical", "diagonal", "mosaic", "cross"}, optional
        Integration category. ``None`` (default) loads every category that
        has tables for the requested ``source`` (an unknown token raises
        ``ValueError`` listing the valid ones).
    dataset : str or list of str, keyword-only
        Dataset id(s), e.g. ``"D11"`` or ``["D11", "D11s"]``. Default: all
        datasets of the category. EVERY requested id must have a table:
        ``["D11", "D99"]`` raises ``FileNotFoundError`` naming ``D99`` and the
        datasets that are available, exactly like ``dataset="D99"`` (an
        unknown id in a list used to be dropped silently).
    methods : str or list of str, keyword-only
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
        are partial (vertical: 7 of 18 methods). (``method=`` is the
        deprecated 0.2.x spelling.)
    metrics : None, str or list of str, keyword-only
        Which metrics to keep - the same vocabulary as
        :func:`multibench.evaluate`: ``None`` / ``"all"`` (default) keeps
        every metric; ``"clustering"`` keeps ``mtb.plot.CLUSTERING_METRICS``
        (ARI, NMI, ASW, iASW, iF1, cLISI); ``"batch"`` keeps
        ``mtb.plot.BATCH_METRICS`` (ASW_batch, GC, iLISI, kBET); a LIST of
        codes keeps exactly those (alias tolerant, ``["ari"]`` -> ARI). Every
        code must be one of :func:`multibench.catalog.known_metrics` or
        present in the frame: ``["ZZZ"]`` raises ``ValueError`` listing both.
        A known code the tables lack gives the empty-frame-plus-warning of
        ``methods``; an unknown token (``"dimension_reduction"`` - a
        ``list_tasks`` token, not a family) raises. (``metric=``, ``task=``
        and ``family=`` are the deprecated 0.2.x spellings.)
    clustering : {"default", "louvain", "kmeans"}, keyword-only
        Which clustering variant of the published tables to read
        (``metric.csv`` / ``metric_louvain.csv`` / ``metric_kmeans.csv``). A
        result directory named ``<method>_louvain`` / ``<method>_kmeans`` IS
        that variant: it is reported under the method's canonical id with
        ``clustering`` set to the suffix, and only when that variant is
        requested (so no method id ever ends in ``_louvain``/``_kmeans``). The
        re-run sweeps are ``"default"`` only.
    source : str, keyword-only
        For a results ROOT one of ``"published"`` (default: the benchmark's
        scIB tables under ``result_path/scib_metric``; none for mosaic),
        ``"rerun"`` (the package's re-run sweeps
        ``result_path/rerun/long_all_<dataset>.csv``: vertical D11/D11s,
        diagonal D28/D28s, mosaic D45/D45s, cross D52/D52s) or ``"both"``
        (the concatenation - tell the two apart by the ``source`` column,
        ``"published"`` vs ``"rerun"``; the sweep's package version is in
        ``frame.attrs["rerun_version"]``, e.g. ``"0.2.1"``); anything else
        raises ``ValueError``. For a FILE, ``"published"`` and ``"both"``
        keep every row, while any other value filters on the file's own
        ``source`` column: ``"user"`` keeps the rows
        :func:`multibench.to_long` wrote, ``"rerun"`` matches both ``rerun``
        and an older ``rerun-<version>`` stamp by prefix, and a value the
        file does not contain raises ``ValueError`` listing the ones present.
        With ``"rerun"``/``"both"`` a re-run row whose ARI is ~0 while the
        published table scored the same method/dataset well is reported by a
        :class:`DegenerateRerunWarning` (Conos on D28) - never silently.
        When the selected category/dataset(s) hold only ONE method in the
        chosen source while the other source holds more, a ``UserWarning``
        says so (``"only one method (scMoMaT) in the published table for
        cross/D52 ... pass source='rerun' (or 'both')"``): the default
        ``"published"`` cross tables are that sparse, and a one-method table
        yields meaningless ranks. No warning when the other source has
        nothing more.
    result_path : path-like, keyword-only
        A results ROOT (holding ``scib_metric/`` and/or ``rerun/``; default
        ``mtb.config.DEFAULT.result_path``, i.e. the tables shipped in the
        package) OR a single long CSV file (columns ``metric, value, method``
        at least, e.g. written by ``to_long(...).to_csv`` or
        ``BatchResult.save()``). A file keeps whatever ``source`` /
        ``clustering`` values it carries; a missing column or a blank cell is
        filled with ``"user"`` / ``"default"`` per row.

    Returns
    -------
    pandas.DataFrame
        Columns, in this order: ``metric, value, method, dataset, category,
        clustering, source``. Method ids are canonical registry tokens
        (``MOFA+`` -> ``MOFA2``, ``Seurat(WNN)`` -> ``Seurat_WNN``); metric
        codes are canonical (``iFI`` -> ``iF1``). ``source`` is
        ``"published"``, ``"rerun"`` or ``"user"`` (a file keeps its own
        values); ``frame.attrs["rerun_version"]`` holds the package version
        stamped on the re-run rows (``"0.2.1"``; a sorted tuple when files
        from several versions were loaded; ``None`` when no stamped row is
        present). ``attrs`` do not survive ``pd.concat`` - read it before
        concatenating.

    Raises
    ------
    FileNotFoundError
        No table for the requested category/dataset/source (any element of a
        ``dataset`` list), with the path looked at and what IS available.
    KeyError
        An unknown method name in ``methods`` (did-you-mean hint).
    ValueError
        Unknown ``category`` / ``metrics`` token or code / ``clustering`` /
        ``source``, or a ``result_path`` file without the long-format
        columns.
    TypeError
        The removed 0.2.x keyword ``metric_set``, or a positional argument
        after ``category`` (everything else is keyword-only).

    Examples
    --------
    >>> pub = mtb.load_results("diagonal", dataset="D28")            # published
    >>> rr = mtb.load_results("diagonal", dataset="D28", source="rerun")
    >>> both = mtb.load_results("cross", dataset="D52", source="both")
    >>> mtb.plot.bubble(both[both.source != "published"])
    >>> mtb.load_results("diagonal", dataset="D28", metrics="batch")   # one family
    >>> mine = mtb.load_results(result_path="mine.csv", source="user")  # your rows only
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
    if datasets and category is not None and not base.is_file():
        datasets = _canonical_dataset_ids(category, datasets, base, source)
    rerun_versions: set[str] = set()

    if base.is_file():
        out = _load_long_csv(base)
        # a file keeps its own provenance values; only the stamp is parsed
        _, rerun_versions = _split_rerun_tag(out["source"])
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
                    f"{missing if len(missing) > 1 else missing[0]}; datasets with "
                    f"{source} tables: {avail}")
        if source != "both":
            # a one-method table ranks nothing; say so when the OTHER source
            # would have given the user a real table for the same selection
            _warn_single_method(out, source, category, datasets, clustering, base)

    # ---- filters ---------------------------------------------------------
    where = f"{category or 'any category'}/{datasets if datasets else 'any dataset'}"
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
    out = out[COLUMNS].reset_index(drop=True)
    out.attrs["rerun_version"] = _version_attr(rerun_versions)
    if source in ("rerun", "both") and not base.is_file():
        _warn_degenerate(out, base, rerun_version=out.attrs["rerun_version"])
    return out


def _other_source_methods(source: str, cats: list, datasets, clustering: str,
                          base: Path) -> set[str]:
    """Method ids the OTHER stored source holds for the same selection
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
    the other source has nothing more - a source the user asked for that is
    the only one with rows is not a mistake."""
    n = out["method"].nunique()
    if n >= 2:
        return
    cats = sorted(out["category"].astype(str).unique()) or (
        [category] if category is not None else list(_CATEGORIES))
    have = _other_source_methods(source, cats, datasets, clustering, base)
    if len(have) <= n:
        return
    other = "rerun" if source == "published" else "published"
    sel = (datasets[0] if len(datasets) == 1 else list(datasets)) if datasets else "any dataset"
    where = f"{category or '/'.join(cats)}/{sel}"
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
    """Dataset ids that SHIP STORED RESULTS (metric tables ``load_results``
    can read) - NOT the datasets that can be downloaded.

    Only a handful of the benchmark's datasets are downloadable; those are
    the release assets of :func:`multibench.data.fetch`, listed by
    :func:`fetchable` (``mtb.data.results.fetchable()``). An id returned here
    but not by ``fetchable()`` has metric tables you can plot and rank
    against, and no data file this package can obtain for you.

    Parameters
    ----------
    category : str, optional
        One of the four integration categories; ``None`` (default) = the
        union across all of them, so a bare ``available_datasets()`` "just
        works". A category folder that does not exist (``mosaic`` has no
        published tables) contributes nothing - no error.
    source : {"published", "rerun", "both"}, keyword-only
        Which tables to look at (see :func:`load_results`). Published ids
        are those holding at least one method's default-clustering table
        (``metric.csv``) - what ``load_results(category, dataset=...)``
        loads.
    result_path : path-like, keyword-only
        Results root (see :func:`load_results`). If the root itself does not
        exist a ``UserWarning`` is raised and ``[]`` returned, instead of a
        silent empty list.

    Returns
    -------
    list of str
        Sorted dataset ids (result-table ids; see :func:`fetchable` for the
        downloadable ones).

    Raises
    ------
    ValueError
        An unknown ``category`` or ``source``.
    TypeError
        The removed 0.2.x keywords ``metric_set`` / ``clustering``.

    See Also
    --------
    fetchable : the ids :func:`multibench.data.fetch` can download.
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
    """Dataset ids :func:`multibench.data.fetch` can download.

    The companion of :func:`available_datasets`, which lists the ids that
    ship STORED RESULTS: the two vocabularies overlap but are not the same
    (``D12`` has published metric tables and no downloadable file; ``D46``
    downloads and has no stored table). Read
    from the fetcher's own asset table, so it cannot drift from what
    ``fetch()`` accepts.

    Returns
    -------
    list of str
        Dataset ids in natural order (``D11 < D28 < ...``).

    Examples
    --------
    >>> mtb.data.results.fetchable()
    ['D11', 'D28', 'D45', 'D46', 'D52']
    >>> set(mtb.data.results.fetchable()) <= set(mtb.available_datasets(source="both"))
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
        Columns ``category, dataset, method, clustering, source``, sorted;
        ``source`` is ``"published"`` or ``"rerun"``, and
        ``frame.attrs["rerun_version"]`` holds the package version stamped
        on the re-run sweeps (as in :func:`load_results`).

    Examples
    --------
    >>> cov = mtb.results_coverage("cross")
    >>> cov[cov.dataset == "D52"]           # scMoMaT (published) + 8 methods (rerun)
    >>> cov.attrs["rerun_version"]          # '0.2.1'
    >>> cov.groupby(["category", "source"]).method.nunique()
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    cols = ["category", "dataset", "method", "clustering", "source"]
    cats = [category] if category is not None else list(_CATEGORIES)
    if category is not None:
        config.category_folder(category)
    frames = []
    versions: set[str] = set()
    # a coverage scan asks WHERE rows are, not whether they are sound or
    # rankable: the degenerate-row and one-method checks belong to
    # load_results, not to every per-variant probe made here
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


def _runtime(method_id: str, method_info) -> dict:
    """``method_info(m)["runtime"]`` (``{"tier", "worst_sec", "observed"}``);
    falls back to the 0.2.x ``runtime_hint`` while ``method_info`` predates
    the ``runtime`` key (the discover side of the 0.3.0 cut adds it)."""
    rt = method_info(method_id).get("runtime")
    if rt is None:
        from ..workflow import runtime_hint
        rt = runtime_hint(method_id)
    return dict(rt)


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
    methods: list[str] | None = None,
    metrics=None,
    long_df: pd.DataFrame | None = None,
    min_methods: int = 2,
    source: str = "published",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """Rank methods for a category from stored results, with coverage made explicit.

    The score is the benchmark's own rule applied per dataset
    (``overall="mean_overall"`` in :mod:`multibench.plot.style`: min-max
    scaled mean of per-metric max-ranks within each dataset, then averaged
    over the datasets the method was run on). Five honesty rules apply:

    * only methods this package runs for the category are ranked - the set
      ``mtb.list_methods(category=...)`` lists. The published cross table
      also scores MOFA2 and Multigrate, which the package wires for other
      categories only; such rows are dropped before ranking (they would
      otherwise shape every other method's within-dataset rank) and named
      in the warning (``"also scored in the published table but not run by
      this package for cross: MOFA2, Multigrate"``) and in
      ``frame.attrs["dropped_methods"]``. A name the registry does not know
      at all (your own method in ``long_df``) is kept;
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
      ``frame.attrs["not_scored"]``;
    * registration methods (``output_kind == "coords"``: GPSA, PASTE,
      PASTE2, SPIRAL in cross) produce aligned coordinates, not an
      embedding, so no scIB metric applies and they are NOT rows of the
      table; the warning names them with that reason and
      ``frame.attrs["unranked_registration"]`` lists them.

    One ``UserWarning`` with one line per finding summarises dropped
    methods and datasets, partial coverage, the unscored methods and the
    unranked registration methods.

    Parameters
    ----------
    category : {"vertical", "diagonal", "mosaic", "cross"}
        Integration category to rank.
    modalities : list of str, keyword-only
        Keep only methods that consume ALL of these base modalities
        (``["rna", "adt"]``), via :func:`multibench.find_methods`.
    methods : list of str, keyword-only
        Rank only these methods (alias tolerant and case-insensitive, as in
        :func:`load_results` - ``"mofa+"`` -> MOFA2, ``"totalvi"`` ->
        totalVI); a name that is neither a registry id nor in the frame
        raises ``KeyError`` with a did-you-mean hint. The unscored and
        registration lines of the warning are restricted to the same set,
        so asking for a method without rows still tells you it has none.
        Default: every method. Note that the within-dataset ranks are
        computed among the requested methods only.
    metrics : None, str or list of str, keyword-only
        What to score on - the vocabulary of :func:`multibench.evaluate` /
        :func:`load_results`. ``None`` (default) scores the ``"clustering"``
        family (ARI, NMI, ASW, iASW, iF1, cLISI - the benchmark's headline
        ranking); ``"batch"`` scores ASW_batch, GC, iLISI, kBET; ``"all"``
        every metric present; a LIST of codes exactly those (alias
        tolerant). A family / list none of whose metrics is in the frame
        raises ``ValueError`` naming the metrics that ARE there; an unknown
        token or code raises. (``task=`` / ``family=`` are the deprecated
        0.2.x spellings.)
    long_df : pandas.DataFrame, keyword-only
        Score THIS frame (``metric, value, method, dataset``) instead of
        loading stored results - e.g. ``pd.concat([published, mine])`` to
        place your own method.
    min_methods : int, keyword-only
        Datasets with fewer methods than this are dropped (default 2).
    source : {"published", "rerun", "both"}, keyword-only
        Which stored tables to load when ``long_df`` is not given (default
        ``"published"``; the published tables are PARTIAL - see above - and
        ``"both"`` averages the 34 method/dataset/metric triples present in
        both sources).
    result_path : path-like, keyword-only
        Results root (see :func:`load_results`).

    Returns
    -------
    pandas.DataFrame
        Sorted best-first, unscored methods last; columns ``method,
        grand_score, n_datasets, n_datasets_total, coverage, needs_labels,
        runtime_tier, worst_sec, env, output_kind``. The metadata columns are
        ``None`` for ids that are not registry methods (your own method, a
        result-dir token). ``frame.attrs`` records the choices the ranking
        was made under: ``"metrics"`` (the family token, or the list of
        codes), ``"family"`` (the token; ``None`` when a list was given),
        ``"source"`` (``"published"`` / ``"rerun"`` / ``"both"``, or
        ``"long_df"``), ``"not_scored"`` (the unscored method ids, also under
        ``"missing"``), ``"dropped_methods"`` (registry methods present in
        the table but not run by this package for the category) and
        ``"unranked_registration"`` (the coords-output methods of the
        category, never scored).

    Raises
    ------
    ValueError
        No dataset has ``min_methods`` methods (nothing can be ranked), the
        frame has none of the requested metrics (the message lists the
        metrics it does have), every row belongs to a method the package
        does not run for the category, ``methods=`` leaves no rows, or an
        unknown ``metrics`` token / code.
    KeyError
        An unknown name in ``methods``.
    FileNotFoundError
        No stored results for the category/source.

    Examples
    --------
    >>> r = mtb.recommend("vertical", modalities=["rna", "adt"])
    >>> r[r.grand_score.notna()]                    # the scored rows
    >>> r.attrs["not_scored"]                       # wired but no published rows
    >>> mtb.recommend("diagonal", metrics="batch", source="rerun")[["method", "grand_score", "coverage"]]
    """
    from ..plot import style

    config.category_folder(category)
    if metrics is None:
        metrics = "clustering"           # the benchmark's headline ranking
    sel = catalog.metric_selection(metrics)   # a token is validated before any load
    long_df_was_none = long_df is None
    if long_df is None:
        # load EVERY metric and filter locally, so the "metrics present"
        # error below can name what the frame really holds
        long_df = load_results(category, source=source, result_path=result_path)
    df = long_df.copy()
    label = f"source={source!r}" if long_df_was_none else "long_df"
    table_noun = {"published": "published table", "rerun": "re-run sweeps",
                  "both": "stored tables"}.get(source, "stored tables") \
        if long_df_was_none else "long_df frame"

    from ..engine.registry import list_methods

    # Only methods this package runs for the category are ranked. A registry
    # method the registry does NOT list for it (MOFA2 / Multigrate in the
    # published cross table) is dropped BEFORE the per-dataset ranks are
    # taken, so it cannot shape the ranks of the methods that are; a name the
    # registry does not know at all (the user's own method) is kept.
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
    if want_ids is not None:
        wired = [m for m in wired if m in want_ids]
    scored_ids = {catalog.canonical_id(m) for m in keep_methods}
    missing = sorted((m for m in wired if m not in scored_ids), key=str.lower)
    # a method with rows ONLY in dropped (< min_methods) datasets is unscored
    # for a different reason than "no rows at all" - say which
    in_frame = {catalog.canonical_id(m) for m in df["method"].unique()}
    only_dropped = [m for m in missing if m in in_frame]
    no_rows = [m for m in missing if m not in in_frame]

    from ..engine import registry, envs
    from ..discover import method_info

    # registration (coords-output) methods of the category: aligned
    # coordinates, not an embedding - no scIB metric applies, never ranked
    registration = []
    for m in list_methods(category=category, runnable=True):
        vs = [v for v in registry.get(m).variants if v.when.get("category") == category]
        if vs and all(v.output.kind == "coords" for v in vs):
            registration.append(m)
    if modalities is not None:
        registration = [m for m in registration if m in allowed]
    if want_ids is not None:
        registration = [m for m in registration if m in want_ids]
    registration = sorted(registration, key=str.lower)

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
    out.attrs["unranked_registration"] = list(registration)

    notes = []
    if degenerate:
        notes.append(
            f"dropped {len(degenerate)} dataset(s) with fewer than {min_methods} "
            f"methods ({', '.join(map(str, degenerate))}) - a min-max score over "
            f"one method is 1.0 by construction")
    if foreign:
        notes.append(
            f"also scored in the {table_noun} but not run by this package for "
            f"{category}: {', '.join(foreign)} - dropped before ranking "
            f"(mtb.list_methods(category={category!r}) does not list them)")
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
    if registration:
        notes.append(
            f"registration methods (coords output: {', '.join(registration)}) "
            f"produce aligned coordinates, not an embedding - no scIB metric "
            f"applies, so they have no rows and are not ranked")
    if notes:
        warnings.warn(f"recommend({category!r}):\n  - " + "\n  - ".join(notes),
                      UserWarning, stacklevel=3)
    return out
