"""The stored results as one long table (a DataFrame), and their ranking.

The stored results are the package's own runs,
``multibench/result/rerun/long_all_<dataset>.csv``, whose rows are stamped
``rerun-<package version>`` (``source`` ``"stored"`` in the long table). The article's scIB tables
``scib_metric/<category>/<dataset>/<method>/metric*.csv`` (a table one level
deeper, in a run-configuration folder, is read too; ``kbet/`` folders hold raw
kBET output) are read through the deprecated ``source=`` and for
:class:`DegenerateRerunWarning`.

Every frame has the columns ``metric, value, method, dataset, category,
clustering, source``, so they concatenate with :func:`multibench.evaluate`'s.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from .. import config
from ..config import _and_list
from .. import _compat
from . import catalog
from ..engine.registry import _BASE_MODALITIES

__all__ = ["load_results", "fetchable", "results_coverage", "recommend", "COLUMNS",
           "SOURCES", "DegenerateRerunWarning"]

#: clustering variant -> metric file of a method folder
_CLUSTERING_FILES = {
    "default": "metric.csv",
    "louvain": "metric_louvain.csv",
    "kmeans": "metric_kmeans.csv",
}
_CORRECTION_FILE = "metric_asw_iasw_if1.csv"
#: the published ARI_batch and NMI_batch, under the names ARI and NMI, as the
#: article's figure script reads them
_BATCH_CLUSTER_FILE = "metric_ari_nmi_batch.csv"

#: the four integration categories
_CATEGORIES = ("cross", "diagonal", "mosaic", "vertical")

#: fixed column order of every frame ``load_results`` returns
COLUMNS = ["metric", "value", "method", "dataset", "category", "clustering", "source"]

#: valid values of ``source=``
SOURCES = ("published", "rerun", "both")

#: a re-run row is degenerate when its ARI is below this while the published
#: table scored the same category, dataset and method above the second value
_DEGENERATE_RERUN_ARI = 0.01
_DEGENERATE_PUBLISHED_ARI = 0.2


class DegenerateRerunWarning(UserWarning):
    """A stored score of ARI ~0 where the article's run of the method scored well.

    ``mtb.load_results`` and ``mtb.recommend`` emit it when a stored row
    scored ARI below 0.01 while the article's table scored the same method
    on the same dataset above 0.2: most likely a failed run, not the method
    (0.4.0 removed the one such shipped row, Conos on D28).
    ``warnings.simplefilter("ignore", mtb.DegenerateRerunWarning)`` silences it.
    """

#: ``Concerto_louvain`` is Concerto's louvain variant, not a method id
_SUFFIX_RE = re.compile(r"^(.*)_(louvain|kmeans)$")

#: sub-directory of ``result_path`` holding the package re-run sweeps
_RERUN_DIR = "rerun"
_RERUN_GLOB = "long_all_*.csv"


#: ``source`` of every row of the stored results (the files say ``rerun-<version>``);
#: the stamp's version goes to ``attrs["rerun_version"]``. An unstamped file gets
#: no version, not the running one, which did not produce its numbers
_RERUN_SOURCE = "stored"

#: a stamped re-run provenance value: ``rerun-<version>``
_RERUN_TAG_RE = re.compile(r"^rerun-(.+)$")


def _split_rerun_tag(source: pd.Series) -> tuple[pd.Series, set[str]]:
    """Fold ``rerun-<version>`` and ``rerun`` values to ``"stored"``; return them
    and the versions seen."""
    col = source.astype(str)
    m = col.str.extract(_RERUN_TAG_RE, expand=False)
    versions = set(m.dropna().unique())
    out = source.where(m.isna() & (col != "rerun"), _RERUN_SOURCE)
    return out, versions


def _version_attr(versions: set[str]):
    """``attrs["rerun_version"]``: ``None``, the one version, or a sorted tuple
    of several."""
    if not versions:
        return None
    if len(versions) == 1:
        return next(iter(versions))
    return tuple(sorted(versions))


def _versions_in(frame) -> tuple:
    """The versions in ``frame.attrs["rerun_version"]`` (:func:`_version_attr`)."""
    v = frame.attrs.get("rerun_version")
    return v if isinstance(v, tuple) else ((v,) if v else ())


def _read_metric_csv(path: Path) -> pd.DataFrame:
    """A metric.csv (unnamed index + Value column) as long metric/value rows."""
    raw = pd.read_csv(path)
    raw.columns = ["metric", "value"] + list(raw.columns[2:])
    raw = raw[["metric", "value"]].copy()
    raw["metric"] = raw["metric"].map(catalog._canonical_metric)
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


#: how the "no table for <dataset>" errors name each ``source=``
_SOURCE_WORD = {"published": "published tables", "rerun": "stored results",
                "both": "published tables or stored results"}


def _no_dataset_msg(source: str, category, missing, have) -> str:
    """The error for datasets without a stored table: the datasets the tables hold."""
    of = f" of {category}" if category else ""
    held = (f"hold {_and_list(have)}" if have else "hold no dataset") + (
        f", not {_and_list(missing)}." if missing else ".")
    return f"The {_SOURCE_WORD.get(source, source + ' tables')}{of} {held}"


def _joined_errors(errors) -> str:
    """The lookup errors of both sources as one text, each once."""
    return " ".join(dict.fromkeys(e.strip() for e in errors if e.strip()))


#: the results the package ships (``multibench/result``)
_SHIPPED_BASE = Path(__file__).resolve().parent.parent / "result"


def _category_unpublished(category: str) -> bool:
    """The shipped published tree exists and has no folder for ``category``
    (mosaic); a missing tree is an install problem, not this case."""
    tree = _SHIPPED_BASE / config.metric_set_dir("scib")
    return tree.is_dir() and not (tree / config.category_folder(category)).is_dir()


def _unpublished_msg(category: str) -> str:
    """The error for a category without published tables."""
    return (f"{category} has no published tables. Leave out source= to load the "
            f"stored results.")


def _published_missing_msg(root: Path, base: Path, category: str) -> str:
    if _category_unpublished(category):
        return _unpublished_msg(category)
    return (
        f"no published scIB metric tables under {root}. The tables ship inside "
        f"the multibench wheel at multibench/result/scib_metric. Check "
        f"mtb.config.DEFAULT.result_path (currently {base})."
    )


#: sub-folders of a method folder that hold no metric table (raw kBET output)
_NON_TABLE_SUBDIRS = {"kbet"}


def _find_metric_file(m_dir: Path, fname: str) -> Path | None:
    """``<m_dir>/<fname>``, else the same file in a run-configuration subfolder
    (the first in sorted order, with a warning when there are several); ``None``."""
    direct = m_dir / fname
    if direct.exists():
        return direct
    nested = sorted(p for p in m_dir.glob(f"*/{fname}")
                    if p.parent.name not in _NON_TABLE_SUBDIRS and p.is_file())
    if not nested:
        return None
    if len(nested) > 1:
        _compat.warn(
            f"{m_dir}: {len(nested)} nested {fname} tables "
            f"({[q.parent.name for q in nested]}); using {nested[0].parent.name}",
            UserWarning)
    return nested[0]


def _iter_published(root: Path, datasets: list | None, clustering: str):
    """``(ds_dir, m_dir, metric_file, method_id, row_clustering, coalesce)`` for
    every loadable (dataset, method); ``m_dir`` is the folder that holds the
    table, where the correction table is looked up."""
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
                # the folder is the variant; its metric.csv holds that clustering
                if m.group(2) != clustering:
                    continue
                mfile = _find_metric_file(m_dir, "metric.csv")
                if mfile is None:
                    continue
                yield ds_dir, mfile.parent, mfile, catalog._canonical_id(m.group(1)), m.group(2), False
                continue
            mfile = _find_metric_file(m_dir, fname)
            if mfile is None:
                continue
            yield ds_dir, mfile.parent, mfile, catalog._canonical_id(m_dir.name), clustering, True


def _published_root(base: Path, category: str) -> Path:
    """``<base>/scib_metric/<category folder>``."""
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
        # corrected ASW/iASW/iF1, for the default clustering only
        corr = m_dir / _CORRECTION_FILE
        if coalesce and clustering == "default" and corr.exists():
            cdf = _read_metric_csv(corr).set_index("metric")["value"]
            df["value"] = df.apply(lambda r: cdf.get(r["metric"], r["value"]), axis=1)
        bat = m_dir / _BATCH_CLUSTER_FILE
        if coalesce and clustering == "default" and bat.exists():
            bdf = _read_metric_csv(bat)
            bdf = bdf[bdf["metric"].isin(["ARI", "NMI"])]
            df = pd.concat([df, bdf.assign(metric=bdf["metric"] + "_batch")],
                           ignore_index=True)
        df["method"] = method
        df["dataset"] = ds_dir.name
        df["category"] = category
        df["clustering"] = row_clust
        df["source"] = "published"
        rows.append(df)
    if not rows:
        have = sorted(p.name for p in root.iterdir() if p.is_dir())
        missing = [d for d in datasets or () if d not in have]
        if missing or not have:
            raise FileNotFoundError(_no_dataset_msg("published", category, missing, have))
        # the datasets are there, without a table of this clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other = [c for c in _CLUSTERING_FILES if c != clustering
                     and next(_iter_published(root, datasets, c), None) is not None]
        raise FileNotFoundError(
            f"The published tables of {category} have no {clustering} variant."
            + (f" They have {_and_list(other)} only." if other else ""))
    return pd.concat(rows, ignore_index=True)[COLUMNS]


def _read_rerun_files(base: Path) -> list[tuple[Path, pd.DataFrame]]:
    root = base / _RERUN_DIR
    out = []
    for f in sorted(root.glob(_RERUN_GLOB)) if root.is_dir() else []:
        df = pd.read_csv(f)
        if not {"metric", "value", "method", "dataset", "category"}.issubset(df.columns):
            continue
        versions: set[str] = set()
        if "source" not in df.columns:
            df["source"] = _RERUN_SOURCE
        else:
            df["source"], versions = _split_rerun_tag(df["source"])
        if "clustering" not in df.columns:
            df["clustering"] = "default"
        df = df[COLUMNS]
        df.attrs["rerun_version"] = _version_attr(versions)
        out.append((f, df))
    return out


def _load_rerun(category: str | None, datasets: list | None, base: Path) -> pd.DataFrame:
    """The re-run rows of a category / dataset selection, with the stamps of
    the contributing files in ``attrs["rerun_version"]``."""
    files = _read_rerun_files(base)
    root = base / _RERUN_DIR
    if not files:
        raise FileNotFoundError(
            f"no stored results ({_RERUN_GLOB}) under {root}. They ship inside the "
            f"multibench wheel at multibench/result/rerun; check "
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
        versions.update(_versions_in(df))
    if not frames:
        allf = pd.concat([df for _, df in files], ignore_index=True)
        if category is not None:
            allf = allf[allf["category"] == category]
        have = sorted(allf["dataset"].astype(str).unique(), key=catalog._dataset_sort_key)
        raise FileNotFoundError(_no_dataset_msg("rerun", category, datasets or (), have))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["rerun_version"] = _version_attr(versions)
    return out


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
            f"with mtb.evaluate(...).to_csv(path, index=False) or "
            f"BatchResult.save().")
    df = df.copy()
    df["metric"] = df["metric"].map(catalog._canonical_metric)
    df = df.dropna(subset=["metric"])
    if "dataset" not in df.columns:
        df["dataset"] = path.stem
    if "category" not in df.columns:
        df["category"] = "user"
    # per row: a concat with a frame lacking clustering/source leaves NaN there
    for col, default in (("clustering", "default"), ("source", "user")):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df[COLUMNS + [c for c in _KEPT_FILE_COLUMNS if c in df.columns]]


def _canonical_dataset_ids(category, datasets, base, source) -> list:
    """Replace a dataset id that differs from a stored id only in case, with a
    warning: a case-insensitive filesystem would load ``d52`` from ``D52`` and
    stamp the frame with the wrong spelling."""
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
            _compat.warn(
                f"dataset {s!r} is not in the stored results but {canon!r} is - using "
                f"that spelling (dataset ids are case-sensitive: 'D52', not 'd52')",
                UserWarning)
            out.append(canon)
        else:
            out.append(s)
    return out


def _check_methods(wanted: list, present) -> None:
    """``KeyError`` with a did-you-mean hint for a method that is neither a
    method id nor in the frame: an empty frame would hide the typo."""
    present = list(present)
    by_lower = {str(m).lower() for m in present}
    for m in wanted:
        cid = catalog._canonical_id(m)
        if cid.lower() in by_lower:
            continue
        try:
            catalog._canonical_id(m, strict=True)
        except KeyError:
            import difflib
            pool = sorted(set(present) | set(catalog._registry_ids()))
            hint = difflib.get_close_matches(str(m), pool, n=1, cutoff=0.6)
            from ..engine import registry
            raise KeyError(registry.unknown_method_message(m, hint[0] if hint else "")
                           ) from None


def _degenerate_rerun_rows(out: pd.DataFrame, base: Path) -> pd.DataFrame:
    """Re-run rows whose ARI is ~0 where the published table scored well, as a
    frame ``category, dataset, method, source, rerun_ARI, published_ARI``."""
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


def _warn_degenerate(out: pd.DataFrame, base: Path,
                     rerun_version=None) -> None:
    """:class:`DegenerateRerunWarning` for the rows :func:`_degenerate_rerun_rows`
    finds; ``rerun_version`` puts the stamp's version in the message."""
    bad = _degenerate_rerun_rows(out, base)
    if bad.empty:
        return

    def _stamp(src) -> str:
        if src != _RERUN_SOURCE:
            return str(src)
        if rerun_version is None:
            return "stored"
        if isinstance(rerun_version, tuple):
            return "stored " + "/".join(rerun_version)
        return f"stored {rerun_version}"

    items = ", ".join(f"{r.method}/{r.dataset} ({_stamp(r.source)}, ARI {r.rerun_ARI:.4f})"
                      for r in bad.itertuples())
    head = ("These stored rows have" if len(bad) > 1 else "This stored row has")
    _compat.warn(
        f"{head} ARI below {_DEGENERATE_RERUN_ARI} where the article's run scored above "
        f"{_DEGENERATE_PUBLISHED_ARI}: {items}. It most likely comes from a failed "
        f"run. Leave it out before ranking.",
        DegenerateRerunWarning)


#: the deprecated keywords of the stored-results functions, which keep their effect
_READ_RERUN = _compat.note("leave it out to read the stored results")
_TABLE_KEYWORDS = {"source": _READ_RERUN,
                   "result_path": _compat.use("config.DEFAULT.result_path")}
#: the 0.2 keywords, removed in 0.4
_REMOVED_METRIC_KEYWORDS = {"task": ("0.4", _compat.use("metrics=")),
                            "family": ("0.4", _compat.use("metrics="))}


@_compat.keywords("load_results", private="_load_results_kept", kept={
    **_TABLE_KEYWORDS,
    "clustering": _compat.note("it picks the clustering variant of the published tables")},
    removed={**_REMOVED_METRIC_KEYWORDS,
             "method": ("0.4", _compat.use("methods=")),
             "metric": ("0.4", _compat.use("metrics=")),
             "metric_set": ("0.3.0", _compat.note("only the scIB metric set exists, so "
                                                  "drop it"))})
def load_results(
    category: str | None = None,
    *,
    dataset: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    metrics=None,
) -> pd.DataFrame:
    """Load the stored results as one long table.

    The stored results are the package's own runs of its methods on the
    benchmark datasets, scored with ``mtb.evaluate``. The table concatenates
    with the long table ``mtb.evaluate`` returns.

    Parameters
    ----------
    category : str, optional
        Integration category (``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``); ``None`` = every category.
    dataset : str or list of str
        Dataset id(s), e.g. ``"D11"`` or ``["D11", "D28"]``; ``None`` = every
        dataset of the category. ``mtb.catalog.datasets()`` lists them.
    methods : str or list of str
        Method(s) to keep, in any case or spelling (``"mofa+"`` -> ``MOFA2``);
        ``None`` = every method.
    metrics : None, str or list of str
        ``None`` / ``"all"`` = every metric; or ``"clustering"``,
        ``"batch"``, ``"classification"``, or a list of codes such as
        ``["ARI", "NMI"]``.

    Returns
    -------
    pandas.DataFrame
        One row per score: ``metric, value, method, dataset, category,
        source, scored_with``. ``attrs["rerun_version"]`` holds the package
        version that scored the stored rows.

    Raises
    ------
    FileNotFoundError
        No stored results for a requested category or dataset; the message
        names the datasets there are.
    KeyError
        Unknown method name in ``methods``, with a did-you-mean hint.
    ValueError
        Unknown ``category`` or ``metrics``.

    Warns
    -----
    UserWarning
        The selection leaves no rows, for example ``metrics="classification"``:
        the stored results have none.

    Examples
    --------
    >>> import multibench as mtb
    >>> long = mtb.load_results("vertical", dataset="D11")
    >>> mtb.load_results("diagonal", dataset="D28", metrics="clustering")

    See Also
    --------
    mtb.evaluate : scores your own embedding into this shape.
    mtb.plot.bubble : draws the table.
    """
    return _load_results_kept(category, dataset=dataset, methods=methods, metrics=metrics)


def _load_results_kept(*args, **kwargs) -> pd.DataFrame:
    """:func:`_load_results` without the ``clustering`` column when it holds only
    ``"default"``, as in the re-run tables (the deprecated published variants keep
    it), and with ``scored_with``, as the other long tables have it."""
    out = _load_results(*args, **kwargs)
    if (out["clustering"] == "default").all():
        out = out.drop(columns="clustering")
    version = out.attrs.get("rerun_version")
    if isinstance(version, str) and "scored_with" not in out.columns \
            and (out["source"] == _RERUN_SOURCE).all():
        # the stored re-runs were scored as evaluate scores: leidenalg, the search
        out["scored_with"] = f"leidenalg/search/{version}"
    return out


def _load_results(
    category: str | None = None,
    *,
    dataset: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    metrics=None,
    clustering: str | None = "default",
    source: str | None = None,
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`load_results` with the deprecated ``clustering=``, ``source=`` and
    ``result_path=``, without their warnings. ``source=None`` reads the re-run
    tables, or every row of a long CSV given as ``result_path``."""
    clustering = clustering or "default"
    if clustering not in _CLUSTERING_FILES:
        raise ValueError(
            f"unknown clustering {clustering!r}; valid: {sorted(_CLUSTERING_FILES)}"
        )
    # a token is validated before anything is read; a list after the load,
    # against the codes the frame holds too
    sel = catalog.metric_selection(metrics) if metrics is None or isinstance(metrics, str) \
        else None
    if category is not None:
        config.category_folder(category)      # validates
    datasets = _as_list(dataset)
    wanted_methods = _as_list(methods)

    base = _base_path(result_path)
    if source is None:
        source = "both" if base.is_file() else "rerun"
    if result_path is not None and not base.exists() and (
            base.suffix.lower() in (".csv", ".tsv") or source not in SOURCES):
        raise FileNotFoundError(
            f"result_path {str(result_path)!r} does not exist (working directory: "
            f"{Path.cwd()})")
    if datasets and category is not None and not base.is_file():
        datasets = _canonical_dataset_ids(category, datasets, base, source)
    rerun_versions: set[str] = set()

    if base.is_file():
        out = _load_long_csv(base)
        _, rerun_versions = _split_rerun_tag(out["source"])
        if source not in ("published", "both"):
            col = out["source"].astype(str)
            mask = (col.str.startswith("rerun") | (col == "stored")) if source == "rerun" \
                else col == source
            if not mask.any():
                raise ValueError(
                    f"source {source!r} not in {base}; present: "
                    f"{sorted(col.unique())}. Leave the source out to keep every row.")
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
                    rerun_versions.update(_versions_in(rr))
                    frames.append(rr)
                    got_any = True
                except FileNotFoundError as e:
                    errors.append(str(e))
            if not got_any and category is not None:
                raise FileNotFoundError(_joined_errors(errors))
        if not frames:
            avail = _list_datasets(None, base, source, clustering) if datasets else []
            raise FileNotFoundError(
                _no_dataset_msg(source, None, datasets, avail) if avail else
                f"No {_SOURCE_WORD.get(source, source + ' tables')} under {base}. "
                + _joined_errors(errors))
        out = pd.concat(frames, ignore_index=True)
        if datasets:
            have = set(out["dataset"].astype(str).unique())
            missing = [d for d in datasets if str(d) not in have]
            if missing:
                avail = _list_datasets(category, base, source, clustering)
                raise FileNotFoundError(_no_dataset_msg(source, category, missing, avail))
        if source == "published":
            _warn_single_method(out, source, category, datasets, clustering, base)

    where = f"{category or 'any category'}/{datasets if datasets else 'any dataset'}"
    if wanted_methods is not None:
        avail = sorted(out["method"].unique())
        _check_methods(wanted_methods, avail)
        want = {catalog._canonical_id(m).lower() for m in wanted_methods}
        out = out[out["method"].map(lambda m: catalog._canonical_id(m).lower()).isin(want)]
        if out.empty:
            shown = wanted_methods if len(wanted_methods) > 1 else wanted_methods[0]
            hint = ""
            if source == "published" and not base.is_file():
                hint = _other_source_hint(category, datasets, wanted_methods, base)
            _compat.warn(
                f"no {source} rows for method {shown!r} in {where} (available: "
                f"{avail}){hint}", UserWarning)
    have_metrics = sorted(out["metric"].unique())
    if sel is None:
        sel = catalog.metric_selection(metrics, extra=have_metrics)
    if sel.codes is not None:
        before = len(out)
        out = out[out["metric"].isin(sel.codes)]
        absent = [c for c in sel.codes if c not in have_metrics]
        if sel.explicit and absent:
            _compat.warn(f"The stored results have no {_and_list(absent)}.",
                         UserWarning)
        elif out.empty and before:
            tables = _SOURCE_WORD.get(source, source + " tables")
            of = " of " + ", ".join(x for x in [category, *map(str, datasets or ())] if x) \
                if category or datasets else ""
            _compat.warn(
                f"The {tables}{of} have no {sel.family} scores, so no rows are left. "
                f"Leave out metrics= to get the scores they have "
                f"({', '.join(have_metrics)}).", UserWarning)
    out = out[COLUMNS + [c for c in _KEPT_FILE_COLUMNS if c in out.columns]]
    out = out.reset_index(drop=True)
    out.attrs["rerun_version"] = _version_attr(rerun_versions)
    if source in ("rerun", "both") and not base.is_file():
        _warn_degenerate(out, base, rerun_version=out.attrs["rerun_version"])
    return out


def _other_source_methods(source: str, cats: list, datasets, clustering: str,
                          base: Path) -> set[str]:
    """Method ids the other stored source holds for the same selection; never
    raises."""
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
                        clustering: str, base: Path) -> None:
    """One ``UserWarning`` when the selection holds one method while the other
    stored source holds more for it."""
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
        where += f"/{_and_list(datasets)}"
    names = {"published": "published", "rerun": "re-run"}
    fix = config.hint("Leave out source= to read them.", "Leave out --source to read them.")
    _compat.warn(
        f"The {names[source]} table for {where} has one method, "
        f"{out['method'].iloc[0]}. Every rank in it is the same. The {names[other]} "
        f"tables have {len(have)} methods for {where}: "
        f"{', '.join(sorted(have, key=str.lower))}. {fix}",
        UserWarning)


#: how the one-method warning of :func:`_warn_single_method` starts (a filter pattern)
_ONE_METHOD_WARNING = r"The (published|re-run) table for .+? has one method, "


def _other_source_hint(category, datasets, wanted_methods, base: Path) -> str:
    """``". The stored results hold N datasets (...). Pass source=..."`` when the
    re-run sweeps hold rows for the methods, else a pointer to
    ``results_coverage``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cov = _results_coverage(category, source="rerun", result_path=base)
    except Exception:
        return ""
    want = {catalog._canonical_id(m).lower() for m in wanted_methods}
    cov = cov[cov["method"].map(lambda m: catalog._canonical_id(m).lower()).isin(want)]
    if datasets:
        cov = cov[cov["dataset"].astype(str).isin([str(d) for d in datasets])]
    if cov.empty:
        return ". results_coverage() lists no rows for it either."
    ds = sorted(cov["dataset"].astype(str).unique())
    return (f". The stored results hold {len(ds)} dataset{'' if len(ds) == 1 else 's'} "
            f"({', '.join(ds)}). "
            + config.hint("Leave out source= to read them.", "Leave out --source to read them."))


def _list_datasets(category, base: Path, source: str, clustering: str) -> list[str]:
    """Sorted dataset ids with a table under ``base`` for ``source`` (published
    ids only with the ``clustering`` variant's file)."""
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


def _available_datasets(
    category: str | None = None,
    *,
    source: str = "rerun",
    result_path: Path | str | None = None,
) -> list[str]:
    """Sorted dataset ids with stored results (the deprecated
    ``available_datasets``): the datasets of :func:`results_coverage` (0.3's
    default was the published tables; ``source="published"`` lists those)."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    base = _base_path(result_path)
    if not base.exists():
        _compat.warn(
            f"result_path {base} does not exist - nothing can be listed "
            f"(default root: {config.DEFAULT.result_path})",
            UserWarning)
        return []
    if category is not None:
        config.category_folder(category)
    return _list_datasets(category, base, source, "default")


available_datasets = _compat.renamed("available_datasets", "mtb.catalog.datasets()",
                                     _available_datasets)


def fetchable() -> list[str]:
    """Dataset ids ``mtb.data.fetch`` can download.

    ``mtb.results_coverage`` lists the datasets with stored results; the two
    sets overlap but are not the same (``D46`` downloads and has no stored
    scores).

    Returns
    -------
    list of str
        Dataset ids in natural order (``D11``, ``D28``, ``D45``, ...).

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.data.fetchable()
    ['D11', 'D22mini', 'D27mini', 'D27mini_vertical', 'D28', 'D37mini', 'D38mini',
     'D45', 'D45mini', 'D46', 'D46mini', 'D49mini', 'D52', 'D52mini', 'D56mini',
     'D58mini', 'D59mini']
    """
    from .fetch import AVAILABLE
    return sorted(AVAILABLE, key=catalog._dataset_sort_key)


@_compat.keywords("results_coverage", private="_results_coverage", kept=_TABLE_KEYWORDS)
def results_coverage(category: str | None = None) -> pd.DataFrame:
    """Which (category, dataset, method) combinations have stored results.

    Parameters
    ----------
    category : str, optional
        Integration category; ``None`` = all four.

    Returns
    -------
    pandas.DataFrame
        One sorted row per ``category, dataset, method``;
        ``attrs["rerun_version"]`` holds the package version that scored
        the stored rows.

    Raises
    ------
    ValueError
        Unknown ``category``.

    Examples
    --------
    >>> import multibench as mtb
    >>> cov = mtb.results_coverage("cross")
    >>> cov.groupby("dataset").method.nunique()

    See Also
    --------
    mtb.load_results : the scores behind each row.
    mtb.data.fetchable : the datasets ``mtb.data.fetch`` can download.
    """
    out = _results_coverage(category)
    # one clustering and one source since 0.4: the constant columns go
    keep = out[["category", "dataset", "method"]].drop_duplicates().reset_index(drop=True)
    keep.attrs = dict(out.attrs)
    return keep


def _results_coverage(category: str | None = None, *, source: str = "rerun",
                      result_path: Path | str | None = None) -> pd.DataFrame:
    """:func:`results_coverage` with the deprecated ``source=`` and ``result_path=``."""
    source = source or "rerun"
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; valid: {list(SOURCES)}")
    cols = ["category", "dataset", "method", "clustering", "source"]
    cats = [category] if category is not None else list(_CATEGORIES)
    if category is not None:
        config.category_folder(category)
    frames = []
    versions: set[str] = set()
    # a coverage scan asks where rows are, not whether they rank
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateRerunWarning)
        warnings.filterwarnings("ignore", message=_ONE_METHOD_WARNING, category=UserWarning)
        for cat in cats:
            if source in ("published", "both"):
                for clus in _CLUSTERING_FILES:
                    try:
                        frames.append(_load_results(cat, clustering=clus,
                                                    result_path=result_path,
                                                    source="published")[cols])
                    except FileNotFoundError:
                        pass
            if source in ("rerun", "both"):
                try:
                    rr = _load_results(cat, result_path=result_path, source="rerun")
                    versions.update(_versions_in(rr))
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


#: the modalities each shipped stored dataset measured, for recommend's error
_STORED_DATASET_MODALITIES = {
    "D11": ("rna", "adt"), "D24": ("rna", "atac"), "D25": ("rna", "atac"),
    "D28": ("rna", "atac"), "D45": ("rna", "atac"), "D52": ("rna", "adt"),
}


def _unmeasured_note(category: str, datasets, wanted: set[str], stored: bool) -> str:
    """The recommend() error when no ranked dataset measured a requested
    modality; ``""`` otherwise, or for a dataset that is not shipped."""
    datasets = [str(d) for d in datasets]
    if not datasets or any(d not in _STORED_DATASET_MODALITIES for d in datasets):
        return ""
    measured = set().union(*(_STORED_DATASET_MODALITIES[d] for d in datasets))
    missing = [m for m in _BASE_MODALITIES if m in wanted and m not in measured]
    if not missing:
        return ""
    groups: dict[tuple, list[str]] = {}
    for d in sorted(datasets, key=catalog._dataset_sort_key):
        groups.setdefault(_STORED_DATASET_MODALITIES[d], []).append(d)
    where = "; ".join(f"{', '.join(ds)} ({'+'.join(mods)})" for mods, ds in groups.items())
    data = "+".join(m.upper() for m in _BASE_MODALITIES if m in wanted)
    head = f"The stored results of {category}" if stored else f"The {category} scores in long_df"
    return (f"{head} come from {where}. None of these datasets measured "
            f"{' or '.join(missing)}, so they cannot rank methods for {data} data. "
            f"Pass scores on such data as long_df=.")


@_compat.keywords("recommend", private="_recommend", kept={
    **_TABLE_KEYWORDS, "atac": _compat.use('modalities=["atac_peak"] or ["atac_gas"]')},
    dropped=("min_methods",), removed=_REMOVED_METRIC_KEYWORDS)
def recommend(
    category: str,
    *,
    modalities: list[str] | None = None,
    methods: list[str] | None = None,
    metrics=None,
    long_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The overall score and row order of ``mtb.plot.bubble``, best first.

    Within each dataset the methods are ranked on each metric; each
    per-dataset score is the mean rank over the metrics, min-max scaled so
    that the best stored method scores 1 and the worst 0, and
    ``grand_score`` is the mean over the datasets. Your rows are ranked
    against the stored rows, so these keep their scores and order. Without
    ``long_df=``, the methods the package runs for the category without
    stored results are listed last.

    Parameters
    ----------
    category : str
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or ``cross``.
    modalities : list of str
        Keep only methods that read all of these modalities, e.g.
        ``["rna", "adt"]``, as in ``mtb.find_methods``; ``None`` = no filter.
    methods : list of str
        Rank only these methods, among themselves; ``None`` = every method.
    metrics : None, str or list of str
        ``None`` = every metric present, as ``mtb.plot.bubble``; or a family, a
        code, or a list of codes and families.
    long_df : pandas.DataFrame
        A long table to rank instead of the stored results, such as
        ``pd.concat([mtb.load_results(category), mine])``.

    Returns
    -------
    pandas.DataFrame
        One row per method, best first: ``method``, ``grand_score``,
        ``needs_labels``, ``runtime_tier``, ``worst_sec`` and ``env``.

    Raises
    ------
    ValueError
        Nothing left to rank, ``long_df`` repeating stored method/dataset
        rows, your rows on another dataset than the stored rows, or
        ``modalities`` naming data that none of the ranked datasets measured
        (pass such scores as ``long_df=``).
    KeyError
        Unknown method name in ``methods``, with a did-you-mean hint.

    Warns
    -----
    UserWarning
        One message: partial coverage, methods left out and why.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.recommend("vertical", modalities=["rna", "adt"])
    >>> free = mtb.find_methods("vertical", needs_labels=False)
    >>> mtb.recommend("vertical", methods=free)
    >>> both = pd.concat([mtb.load_results("vertical"), mine])
    >>> mtb.recommend("vertical", long_df=both)
    """
    return _recommend(category, modalities=modalities, methods=methods,
                      metrics=metrics, long_df=long_df)


def _recommend(
    category: str,
    *,
    modalities: list[str] | None = None,
    atac: str | None = None,
    methods: list[str] | None = None,
    metrics=None,
    long_df: pd.DataFrame | None = None,
    min_methods: int = 2,
    source: str = "rerun",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`recommend` with the deprecated ``source=`` and ``result_path=``."""
    source = source or "rerun"
    from ..plot import style

    config.category_folder(category)
    from ..discover import _find_methods as find_methods, _modality_filter

    want_mods, atac = _modality_filter(modalities, atac)
    filtered = want_mods is not None or atac is not None
    if want_mods is None and atac is not None:
        want_mods = {"atac"}
    sel = catalog.metric_selection(metrics)
    long_df_was_none = long_df is None
    if long_df is None:
        # every metric: the "metrics present" error names what the frame holds
        long_df = _load_results(category, source=source, result_path=result_path)
    df = long_df.copy()
    table_noun = {"published": "published table", "rerun": "stored results",
                  "both": "stored results"}.get(source, "stored results") \
        if long_df_was_none else "long_df frame"
    label = f"the {table_noun}" if long_df_was_none else "long_df"
    if not long_df_was_none and "category" in df.columns:
        # rows scored as another category would be ranked against this one's
        cats = {str(c) for c in df["category"].dropna()}
        other = sorted(cats - {category})
        if len(other) == 1 and category not in cats:
            raise ValueError(f"long_df holds {other[0]} rows: pass "
                             f"recommend({other[0]!r}, long_df=...).")
        if other:
            one = category if category in cats else other[0]
            raise ValueError(
                f"long_df holds {_and_list(other)} rows"
                + (f" next to {category} ones" if category in cats else "")
                + f": rank one category at a time, as recommend({one!r}, "
                f"long_df=long_df[long_df['category'] == {one!r}]).")

    from ..engine.registry import list_methods

    # a package method not run for the category (scMSI in a cross table) must not
    # shape the ranks; a name the registry does not know (the user's own) is kept
    listed = set(list_methods(category=category))
    registry_ids = set(catalog._registry_ids())
    canon = df["method"].map(catalog._canonical_id)
    foreign = sorted({m for m, c in zip(df["method"], canon)
                      if c in registry_ids and c not in listed}, key=str.lower)
    if foreign:
        df = df[~canon.isin({catalog._canonical_id(m) for m in foreign})]
        if df.empty:
            raise ValueError(
                f"every row in {label} belongs to a method this package does "
                f"not run for {category} ({', '.join(foreign)}); "
                f"mtb.find_methods(category={category!r}) lists the rankable "
                f"ones")
    want_ids: set[str] | None = None
    if methods is not None:
        wanted = _as_list(methods)
        _check_methods(wanted, sorted(df["method"].astype(str).unique()))
        want_ids = {catalog._canonical_id(m) for m in wanted}
        want_lower = {w.lower() for w in want_ids}
        df = df[df["method"].map(lambda m: catalog._canonical_id(m).lower()).isin(want_lower)]
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
            from ..plot.bar import CLASSIFICATION_METRICS
            if long_df_was_none and result_path is None \
                    and set(sel.codes) <= set(CLASSIFICATION_METRICS):
                raise ValueError(
                    "The stored results have no classification scores. Score your runs "
                    "with mtb.evaluate and pass them as long_df=.")
            if long_df_was_none and not sel.explicit and sel.family == "batch":
                one = ", ".join(sorted(map(str, long_df["dataset"].unique())))
                raise ValueError(
                    f"The {category} tables have no batch scores ({one} has one "
                    f"batch). Use metrics=\"clustering\".")
            raise ValueError(
                f"no {metrics!r} metrics ({sel.codes}) in the frame; metrics present: "
                f"{sorted(long_df['metric'].unique())}")
    if "dataset" not in df.columns:
        df["dataset"] = "all"
    keys = ["method", "metric", "dataset"]
    from ..plot.bubble import _duplicate_rows_message, _user_repeats
    repeated = _user_repeats(df, keys)
    if repeated.any():
        raise ValueError(_duplicate_rows_message(df[repeated], keys))

    # the methods with a score on each dataset
    scored = df[df["value"].notna()]
    per_ds = {ds: sorted(set(g["method"])) for ds, g in scored.groupby("dataset")}
    degenerate = sorted(ds for ds, ms in per_ds.items() if len(ms) < min_methods)
    kept = {ds: ms for ds, ms in per_ds.items() if ds not in degenerate}
    if not kept:
        hint = ""
        if long_df_was_none and source == "published":
            # cross/D52: one published method, several re-run ones
            have = _other_source_methods(source, [category], None, "default",
                                         _base_path(result_path))
            if len(have) >= min_methods:
                hint = (f" The stored results have {len(have)} methods for {category}: "
                        f"{', '.join(sorted(have, key=str.lower))}. Leave out source= "
                        f"to rank them.")
        sizes = {ds: len(per_ds[ds]) for ds in sorted(per_ds)}
        counts = ", ".join(f"{ds} has {k}" for ds, k in sizes.items())
        one = (" A min-max score over one method is always 1.0."
               if sizes and max(sizes.values()) == 1 else "")
        raise ValueError(
            f"No {category} dataset has {min_methods} or more methods. {counts}."
            f"{one}{hint} Otherwise pass long_df= with more methods.")
    # igraph-scored rows ranked against leidenalg-scored stored rows
    backend = style.backend_warning(df[df["dataset"].isin(list(kept))])
    n_ds = pd.Series([m for ms in kept.values() for m in ms]).value_counts()
    n_total = len(kept)
    codes = sel.codes if sel.explicit else None
    # the stored rows are those of the tables read (the deprecated result_path=)
    stored = style.stored_pairs(result_path)
    grand = style.overall(df, codes, stored=stored)   # refuses your rows on another dataset
    if degenerate:
        grand = style.overall(df[df["dataset"].isin(list(kept))], codes, stored=stored)
    keep_methods = list(grand.index) + sorted(
        {m for ms in kept.values() for m in ms} - set(grand.index), key=str)
    if filtered:
        allowed = set(find_methods(category=category, modalities=modalities, atac=atac))
        # your own methods are not package methods: the filter leaves them
        from ..engine.registry import method_of
        keep_methods = [m for m in keep_methods if method_of(m) in allowed | {None}]
        if not keep_methods:
            asked = " and ".join(
                ([f"consumes modalities {list(modalities)}"] if modalities is not None
                 else []) + ([f"reads atac={atac!r}"] if atac is not None else []))
            raise ValueError(
                f"no scored method in {category} {asked}; methods with results: "
                f"{sorted(grand.index)}; methods matching the selection: "
                f"{sorted(allowed)}")

    # methods run for the category without scores; of long_df's methods only
    in_frame = {catalog._canonical_id(m) for m in df["method"].unique()}
    wired = [m for m in find_methods(category=category, modalities=modalities, atac=atac)
             if long_df_was_none or m in in_frame]
    if want_ids is not None:
        wired = [m for m in wired if m in want_ids]
    scored_ids = {catalog._canonical_id(m) for m in keep_methods}
    missing = sorted((m for m in wired if m not in scored_ids), key=str.lower)
    # rows only in dropped datasets is another reason than no rows
    only_dropped = [m for m in missing if m in in_frame]

    from ..engine import registry, envs
    from ..discover import method_info

    rows = []
    for m in keep_methods + missing:
        scored = m in grand.index
        # '<id> (this run)' takes the chips of its method
        own = registry.method_of(m)
        spec = registry.get(own) if own else None
        if spec is not None:
            in_category = [v for v in spec.variants if v.when.get("category") == category]
            env = envs.group_for(spec.id)
            # a method may be supervised in one category only
            needs = any(v.needs_labels for v in in_category) if in_category \
                else bool(spec.needs_labels)
            runtime = method_info(spec.id)["runtime"]
            # the observed times are the package's runs: none for a run of yours
            tier, worst = ((runtime.get("tier"), runtime.get("worst_sec"))
                           if catalog._canonical_id(m) == spec.id else (None, None))
        else:
            env, needs, tier, worst = None, None, None, None
        rows.append({
            "method": m,
            "grand_score": float(grand.loc[m]) if scored else float("nan"),
            "needs_labels": needs,
            "runtime_tier": tier,
            "worst_sec": worst,
            "env": env,
        })
    out = pd.DataFrame(rows)
    out.attrs["metrics"] = list(sel.codes) if sel.explicit else sel.family
    out.attrs["source"] = source if long_df_was_none else "long_df"
    out.attrs["missing"] = list(missing)
    out.attrs["dropped_methods"] = list(foreign)

    notes = []
    if filtered:
        unmeasured = _unmeasured_note(category, list(kept), want_mods,
                                      long_df_was_none)
        if unmeasured:
            raise ValueError(unmeasured)
    if degenerate:
        one = len(degenerate) == 1
        notes.append(
            f"{'Dataset' if one else 'Datasets'} {_and_list(degenerate)} "
            f"{'has' if one else 'have'} fewer than {min_methods} methods and "
            f"{'is' if one else 'are'} left out of the ranking. A min-max score "
            f"over one method is always 1.0.")
    if foreign:
        one = len(foreign) == 1
        it, they = ("it", "it is") if one else ("them", "they are")
        notes.append(
            f"{_and_list(foreign)} {'has' if one else 'have'} scores in the "
            f"{table_noun}, but this package does not run {it} for {category}, "
            f"so {they} left out of the ranking. "
            f"mtb.find_methods(category={category!r}) does not list {it}.")
    partial = [m for m in out["method"] if m in n_ds.index and n_ds.loc[m] < n_total]
    if partial:
        notes.append(
            "Some methods have scores on only part of the datasets: "
            + ", ".join(f"{m} {int(n_ds.loc[m])} of {n_total}" for m in partial) + ".")
    if backend:
        notes.append(backend)
    if only_dropped:
        one = len(only_dropped) == 1
        notes.append(
            f"{_and_list(only_dropped)} {'has' if one else 'have'} scores only on "
            f"datasets left out of the ranking, so {'it is' if one else 'they are'} "
            f"listed last.")
    if notes:
        _compat.warn(f"recommend({category!r}):\n  - " + "\n  - ".join(notes),
                     UserWarning)
    return out
