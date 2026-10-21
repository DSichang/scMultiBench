"""The stored results as one long table (a DataFrame), and their ranking.

The stored results are the package's own runs,
``multibench/result/rerun/long_all_<dataset>.csv``, whose rows are stamped
``rerun-<package version>`` (``source`` ``"stored"`` in the long table).

Every frame has the columns ``metric, value, method, dataset, category,
source, scored_with``, so they concatenate with :func:`multibench.evaluate`'s.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import config
from ..config import _and_list
from .. import _compat
from . import catalog
from ..engine.registry import _BASE_MODALITIES

__all__ = ["load_results", "recommend"]

#: the four integration categories
_CATEGORIES = ("cross", "diagonal", "mosaic", "vertical")

#: fixed column order of every frame ``load_results`` returns
COLUMNS = ["metric", "value", "method", "dataset", "category", "clustering", "source"]

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


def _as_list(x) -> list | None:
    if x is None:
        return None
    if isinstance(x, (str, bytes)):
        return [x]
    return list(x)


def _base_path() -> Path:
    """The folder holding the stored results (``config._RESULT_PATH``)."""
    return Path(config._RESULT_PATH)


def _no_dataset_msg(category, missing, have) -> str:
    """The error for datasets without a stored table: the datasets the tables hold."""
    of = f" of {category}" if category else ""
    held = (f"hold {_and_list(have)}" if have else "hold no dataset") + (
        f", not {_and_list(missing)}." if missing else ".")
    return f"The stored results{of} {held}"


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
            f"No stored results ({_RERUN_GLOB}) under {root}. They ship inside the "
            f"multibench wheel: reinstall multibench-sc.")
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
        raise FileNotFoundError(_no_dataset_msg(category, datasets or (), have))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["rerun_version"] = _version_attr(versions)
    return out


def _canonical_dataset_ids(category, datasets, base) -> list:
    """Replace a dataset id that differs from a stored id only in case, with a
    warning: a case-insensitive filesystem would load ``d52`` from ``D52`` and
    stamp the frame with the wrong spelling."""
    try:
        have = _list_datasets(category, base)
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
    out = _load_results(category, dataset=dataset, methods=methods,
                        metrics=metrics).drop(columns="clustering")
    version = out.attrs.get("rerun_version")
    if isinstance(version, str) and (out["source"] == _RERUN_SOURCE).all():
        # the stored re-runs were scored as evaluate scores: leidenalg, the search
        out["scored_with"] = f"leidenalg/search/{version}"
    return out


def _load_results(
    category: str | None = None,
    *,
    dataset: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    metrics=None,
) -> pd.DataFrame:
    """The stored rows of a selection, with every column of :data:`COLUMNS`."""
    # a token is validated before anything is read; a list after the load,
    # against the codes the frame holds too
    sel = catalog.metric_selection(metrics) if metrics is None or isinstance(metrics, str) \
        else None
    if category is not None:
        config.category_folder(category)      # validates
    datasets = _as_list(dataset)
    wanted_methods = _as_list(methods)

    base = _base_path()
    if datasets and category is not None:
        datasets = _canonical_dataset_ids(category, datasets, base)
    rerun_versions: set[str] = set()
    cats = [category] if category is not None else list(_CATEGORIES)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for cat in cats:
        try:
            rr = _load_rerun(cat, datasets, base)
        except FileNotFoundError as e:
            if category is not None:
                raise
            errors.append(str(e).strip())
            continue
        rerun_versions.update(_versions_in(rr))
        frames.append(rr)
    if not frames:
        avail = _list_datasets(None, base) if datasets else []
        raise FileNotFoundError(
            _no_dataset_msg(None, datasets, avail) if avail else
            f"No stored results under {base}. " + " ".join(dict.fromkeys(errors)))
    out = pd.concat(frames, ignore_index=True)
    if datasets:
        have = set(out["dataset"].astype(str).unique())
        missing = [d for d in datasets if str(d) not in have]
        if missing:
            avail = _list_datasets(category, base)
            raise FileNotFoundError(_no_dataset_msg(category, missing, avail))

    where = f"{category or 'any category'}/{datasets if datasets else 'any dataset'}"
    if wanted_methods is not None:
        avail = sorted(out["method"].unique())
        _check_methods(wanted_methods, avail)
        want = {catalog._canonical_id(m).lower() for m in wanted_methods}
        out = out[out["method"].map(lambda m: catalog._canonical_id(m).lower()).isin(want)]
        if out.empty:
            shown = wanted_methods if len(wanted_methods) > 1 else wanted_methods[0]
            _compat.warn(
                f"no rerun rows for method {shown!r} in {where} (available: "
                f"{avail})", UserWarning)
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
            of = " of " + ", ".join(x for x in [category, *map(str, datasets or ())] if x) \
                if category or datasets else ""
            _compat.warn(
                f"The stored results{of} have no {sel.family} scores, so no rows are left. "
                f"Leave out metrics= to get the scores they have "
                f"({', '.join(have_metrics)}).", UserWarning)
    out = out[COLUMNS].reset_index(drop=True)
    out.attrs["rerun_version"] = _version_attr(rerun_versions)
    return out


def _list_datasets(category, base: Path) -> list[str]:
    """Sorted dataset ids with stored results under ``base``."""
    cats = [category] if category is not None else list(_CATEGORIES)
    found: set[str] = set()
    for _, df in _read_rerun_files(base):
        found.update(df.loc[df["category"].isin(cats), "dataset"].astype(str).unique())
    return sorted(found)


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
    from ..plot import style

    config.category_folder(category)
    from ..discover import find_methods, _modality_filter

    min_methods = 2
    want_mods, atac = _modality_filter(modalities)
    filtered = want_mods is not None or atac is not None
    if want_mods is None and atac is not None:
        want_mods = {"atac"}
    sel = catalog.metric_selection(metrics)
    long_df_was_none = long_df is None
    if long_df is None:
        # every metric: the "metrics present" error names what the frame holds
        long_df = _load_results(category)
    df = long_df.copy()
    table_noun = "stored results" if long_df_was_none else "long_df frame"
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
            if long_df_was_none and set(sel.codes) <= set(CLASSIFICATION_METRICS):
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
        sizes = {ds: len(per_ds[ds]) for ds in sorted(per_ds)}
        counts = ", ".join(f"{ds} has {k}" for ds, k in sizes.items())
        one = (" A min-max score over one method is always 1.0."
               if sizes and max(sizes.values()) == 1 else "")
        raise ValueError(
            f"No {category} dataset has {min_methods} or more methods. {counts}."
            f"{one} Otherwise pass long_df= with more methods.")
    # igraph-scored rows ranked against leidenalg-scored stored rows
    backend = style.backend_warning(df[df["dataset"].isin(list(kept))])
    n_ds = pd.Series([m for ms in kept.values() for m in ms]).value_counts()
    n_total = len(kept)
    codes = sel.codes if sel.explicit else None
    stored = style.stored_pairs()
    grand = style.overall(df, codes, stored=stored)   # refuses your rows on another dataset
    if degenerate:
        grand = style.overall(df[df["dataset"].isin(list(kept))], codes, stored=stored)
    keep_methods = list(grand.index) + sorted(
        {m for ms in kept.values() for m in ms} - set(grand.index), key=str)
    if filtered:
        allowed = set(find_methods(category=category, modalities=modalities))
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
    wired = [m for m in find_methods(category=category, modalities=modalities)
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
    out.attrs["source"] = "rerun" if long_df_was_none else "long_df"
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
