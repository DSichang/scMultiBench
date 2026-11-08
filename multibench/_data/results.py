"""The stored results as one long table (a DataFrame), and their ranking.

The stored results are the package's own runs,
``multibench/result/rerun/long_all_<dataset>.csv``, whose rows are stamped
``rerun-<package version>`` (``source`` ``"stored"`` in the long table).

Every frame has the columns ``metric, value, method, dataset, category,
source, scored_with``, so they concatenate with :func:`multibench.evaluate`'s.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import _settings
from ..engine import registry
from .._text import _and_list
from .. import _text
from . import catalog

__all__ = ["load_results", "rank"]

#: the four integration categories
_CATEGORIES = registry.CATEGORIES

#: fixed column order of every frame ``load_results`` returns
COLUMNS = ["metric", "value", "method", "dataset", "category", "source"]

#: sub-directory of ``result_path`` holding the package re-run sweeps
_RERUN_DIR = "rerun"
_RERUN_GLOB = "long_all_*.csv"


#: ``source`` of every row of the stored results; the files say ``rerun-<version>``,
#: and the version goes to ``attrs["rerun_version"]``
_RERUN_SOURCE = "stored"


def _as_list(x) -> list | None:
    if x is None:
        return None
    if isinstance(x, (str, bytes)):
        return [x]
    return list(x)


def _base_path() -> Path:
    """The folder holding the stored results (``_settings._RESULT_PATH``)."""
    return Path(_settings._RESULT_PATH)


def _no_dataset_msg(category, missing, have) -> str:
    """The error for datasets without a stored table: the datasets the tables hold."""
    of = f" of {category}" if category else ""
    held = (f"hold {_and_list(have)}" if have else "hold no dataset") + (
        f", not {_and_list(missing)}." if missing else ".")
    return f"The stored results{of} {held}"


def _read_rerun_files(base: Path) -> list[tuple[Path, pd.DataFrame]]:
    """``(file, rows)`` of each stored table; its one ``rerun-<version>`` stamp
    goes to ``attrs["rerun_version"]``."""
    root = base / _RERUN_DIR
    out = []
    for f in sorted(root.glob(_RERUN_GLOB)) if root.is_dir() else []:
        df = pd.read_csv(f)[COLUMNS]
        (stamp,) = df["source"].unique()
        df["source"] = _RERUN_SOURCE
        df.attrs["rerun_version"] = stamp.removeprefix("rerun-")
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
    for _, df in files:
        sel = df
        if category is not None:
            sel = sel[sel["category"] == category]
        if datasets:
            sel = sel[sel["dataset"].isin(datasets)]
        if sel.empty:
            continue
        frames.append(sel)
    if not frames:
        allf = pd.concat([df for _, df in files], ignore_index=True)
        if category is not None:
            allf = allf[allf["category"] == category]
        have = sorted(allf["dataset"].astype(str).unique(), key=catalog._dataset_sort_key)
        raise FileNotFoundError(_no_dataset_msg(category, datasets or (), have))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["rerun_version"] = _one_version(frames)
    return out


def _one_version(frames) -> str:
    """The one package version that scored the stored ``frames``."""
    (version,) = {f.attrs["rerun_version"] for f in frames}
    return version


def _check_methods(wanted: list, present) -> None:
    """``KeyError`` for a method that is neither a method id nor in the frame:
    an empty frame would hide the typo."""
    present = list(present)
    by_lower = {str(m).lower() for m in present}
    for m in wanted:
        if catalog._canonical_id(m).lower() not in by_lower:
            catalog._canonical_id(m, strict=True)


def _of_methods(df, wanted) -> pd.DataFrame:
    """The rows of ``df`` whose method is one of ``wanted``, in any spelling."""
    want = {catalog._canonical_id(m).lower() for m in wanted}
    return df[df["method"].map(lambda m: catalog._canonical_id(m).lower()).isin(want)]


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
        Unknown method name in ``methods``.
    ValueError
        Unknown ``category`` or ``metrics``, or ``metrics`` names a family or
        code the stored results have no rows of, for example
        ``metrics="classification"`` on vertical.

    Warns
    -----
    UserWarning
        ``methods`` leaves no rows.

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
    out = _load_results(category, dataset=dataset, methods=methods, metrics=metrics)
    # the stored re-runs were scored as evaluate scores: leidenalg, the search
    out["scored_with"] = f"leidenalg/search/{out.attrs['rerun_version']}"
    return out


def _load_results(
    category: str | None = None,
    *,
    dataset: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    metrics=None,
) -> pd.DataFrame:
    """The stored rows of a selection, with every column of :data:`COLUMNS`."""
    registry.check_category(category)
    datasets = _as_list(dataset)
    wanted_methods = _as_list(methods)

    base = _base_path()
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

    of = " of " + ", ".join(x for x in [category, *map(str, datasets or ())] if x) \
        if category or datasets else ""
    if wanted_methods is not None:
        _check_methods(wanted_methods, sorted(out["method"].unique()))
        out = _of_methods(out, wanted_methods)
        if out.empty:
            _text.warn(f"The stored results{of} have no rows for {_and_list(wanted_methods)}.",
                       UserWarning)
    picked = catalog.metrics_in(metrics, out["metric"].unique(),
                                table=f"The stored results{of} have")
    if picked is not None:
        out = out[out["metric"].isin(picked)]
    out = out[COLUMNS].reset_index(drop=True)
    out.attrs["rerun_version"] = _one_version(frames)
    return out


def _list_datasets(category, base: Path) -> list[str]:
    """Sorted dataset ids with stored results under ``base``."""
    cats = [category] if category is not None else list(_CATEGORIES)
    found: set[str] = set()
    for _, df in _read_rerun_files(base):
        found.update(df.loc[df["category"].isin(cats), "dataset"].astype(str).unique())
    return sorted(found)


#: the modalities each shipped stored dataset measured, for rank's error
_STORED_DATASET_MODALITIES = {
    "D11": ("rna", "adt"), "D24": ("rna", "atac"), "D25": ("rna", "atac"),
    "D28": ("rna", "atac"), "D45": ("rna", "atac"), "D52": ("rna", "adt"),
}


def _unmeasured_note(category: str, datasets, wanted: set[str], stored: bool) -> str:
    """The rank() error when no ranked dataset measured a requested
    modality; ``""`` otherwise, or for a dataset that is not shipped."""
    datasets = [str(d) for d in datasets]
    if not datasets or any(d not in _STORED_DATASET_MODALITIES for d in datasets):
        return ""
    measured = set().union(*(_STORED_DATASET_MODALITIES[d] for d in datasets))
    missing = [m for m in registry._BASE_MODALITIES if m in wanted and m not in measured]
    if not missing:
        return ""
    groups: dict[tuple, list[str]] = {}
    for d in sorted(datasets, key=catalog._dataset_sort_key):
        groups.setdefault(_STORED_DATASET_MODALITIES[d], []).append(d)
    where = "; ".join(f"{', '.join(ds)} ({'+'.join(mods)})" for mods, ds in groups.items())
    data = "+".join(m.upper() for m in registry._BASE_MODALITIES if m in wanted)
    head = f"The stored results of {category}" if stored else f"The {category} scores in long_df"
    return (f"{head} come from {where}. None of these datasets measured "
            f"{' or '.join(missing)}, so they cannot rank methods for {data} data. "
            f"Pass scores on such data as long_df=.")


def rank(
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
        Nothing left to rank, ``metrics`` naming a family or code the table
        has no rows of, ``long_df`` repeating stored method/dataset rows, your
        rows on another dataset than the stored rows, or ``modalities`` naming
        data that none of the ranked datasets measured (pass such scores as
        ``long_df=``).
    KeyError
        Unknown method name in ``methods``.

    Warns
    -----
    UserWarning
        One message: partial coverage, methods left out and why.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.rank("vertical", modalities=["rna", "adt"])
    >>> free = mtb.find_methods("vertical", needs_labels=False)
    >>> mtb.rank("vertical", methods=free)
    >>> both = pd.concat([mtb.load_results("vertical"), mine])
    >>> mtb.rank("vertical", long_df=both)
    """
    from ..discover import _modality_filter, find_methods
    from ..plot import style
    from ..plot.bubble import _refuse_repeats

    registry.check_category(category)
    want_mods, atac = _modality_filter(modalities)
    filtered = want_mods is not None or atac is not None
    if want_mods is None and atac is not None:
        want_mods = {"atac"}
    stored = long_df is None
    if stored:
        # every metric: the "metrics present" error names what the frame holds
        long_df = _load_results(category)
    elif "category" in long_df.columns:
        _refuse_other_categories(long_df, category)
    label = "the stored results" if stored else "long_df"

    df, foreign = _drop_unrun(long_df.copy(), category, label)
    want_ids: set[str] | None = None
    if methods is not None:
        wanted = _as_list(methods)
        _check_methods(wanted, sorted(df["method"].astype(str).unique()))
        want_ids = {catalog._canonical_id(m) for m in wanted}
        df = _of_methods(df, wanted)
        if df.empty:
            raise ValueError(
                f"none of methods={list(wanted)} has rows in {label} for "
                f"{category}; methods with rows: "
                f"{sorted(long_df['method'].astype(str).unique())}")
    picked = catalog.metrics_in(metrics, df["metric"].unique(), table=(
        f"The stored results of {category} have" if stored else "long_df has"))
    if picked is not None:
        df = df[df["metric"].isin(picked)]
    if "dataset" not in df.columns:
        df["dataset"] = "all"
    _refuse_repeats(df)

    kept, degenerate = _rankable(df, category, metrics, stored)
    # igraph-scored rows ranked against leidenalg-scored stored rows
    backend = style.backend_warning(df[df["dataset"].isin(list(kept))])
    n_ds = pd.Series([m for ms in kept.values() for m in ms]).value_counts()
    pairs = style.stored_pairs()
    grand = style.overall(df, stored=pairs)   # refuses your rows on another dataset
    if degenerate:
        grand = style.overall(df[df["dataset"].isin(list(kept))], stored=pairs)
    keep_methods = list(grand.index) + sorted(
        {m for ms in kept.values() for m in ms} - set(grand.index), key=str)
    if filtered:
        allowed = set(find_methods(category=category, modalities=modalities))
        # your own methods are not package methods: the filter leaves them
        keep_methods = [m for m in keep_methods if registry.method_of(m) in allowed | {None}]
        if not keep_methods:
            raise ValueError(f"No scored method in {category} reads {list(modalities)}.")

    # methods run for the category without scores; of long_df's methods only
    in_frame = None if stored else {catalog._canonical_id(m) for m in df["method"].unique()}
    missing = _unscored_methods(category, modalities, keep_methods, in_frame=in_frame,
                                want_ids=want_ids)

    out = pd.DataFrame([{"method": m,
                         "grand_score": float(grand.loc[m]) if m in grand.index else float("nan"),
                         **_chips(m, category)} for m in keep_methods + missing])
    out.attrs["metrics"] = "all" if picked is None else picked
    out.attrs["source"] = "rerun" if stored else "long_df"
    out.attrs["missing"] = list(missing)
    out.attrs["dropped_methods"] = list(foreign)

    if filtered:
        unmeasured = _unmeasured_note(category, list(kept), want_mods, stored)
        if unmeasured:
            raise ValueError(unmeasured)
    partial = {m: int(n_ds.loc[m]) for m in out["method"]
               if m in n_ds.index and n_ds.loc[m] < len(kept)}
    warning = _rank_warning(category, degenerate, foreign, partial, len(kept), backend)
    if warning:
        _text.warn(warning, UserWarning)
    return out


def _unscored_methods(category, modalities, scored, *, in_frame=None, want_ids=None) -> list:
    """The methods the package runs for ``category`` (and ``modalities``) that
    are not among ``scored``, by name; only those of ``in_frame`` and of
    ``want_ids`` when given (``long_df=`` and ``methods=``)."""
    from ..discover import find_methods
    scored_ids = {catalog._canonical_id(m) for m in scored}
    wired = [m for m in find_methods(category=category, modalities=modalities)
             if (in_frame is None or m in in_frame) and (want_ids is None or m in want_ids)]
    return sorted((m for m in wired if m not in scored_ids), key=str.lower)


def _rank_warning(category, degenerate, foreign, partial: dict, n_datasets: int,
                  backend) -> str | None:
    """rank()'s one warning, or ``None``: the ``degenerate`` datasets left out,
    the ``foreign`` methods not run for ``category``, the methods scored on part
    of the ``n_datasets`` datasets (``partial``: ``{method: datasets}``) and the
    ``backend`` note."""
    notes = []
    if degenerate:
        one = len(degenerate) == 1
        notes.append(
            f"{'Dataset' if one else 'Datasets'} {_and_list(degenerate)} "
            f"{'has' if one else 'have'} fewer than {_MIN_METHODS} methods and "
            f"{'is' if one else 'are'} left out of the ranking. A min-max score "
            f"over one method is always 1.0.")
    if foreign:
        one = len(foreign) == 1
        notes.append(f"{_and_list(foreign)} {'is' if one else 'are'} not run for {category} "
                     f"by this package and left out.")
    if partial:
        notes.append(
            "Some methods have scores on only part of the datasets: "
            + ", ".join(f"{m} {k} of {n_datasets}" for m, k in partial.items()) + ".")
    if backend:
        notes.append(backend)
    return f"rank({category!r}):\n  - " + "\n  - ".join(notes) if notes else None


def _refuse_other_categories(long_df, category) -> None:
    """Raise when ``long_df`` holds rows of another category: they would be
    ranked against this one's."""
    cats = {str(c) for c in long_df["category"].dropna()}
    other = sorted(cats - {category})
    if other:
        one = category if category in cats else other[0]
        raise ValueError(
            f"long_df holds {_and_list(other)} rows: rank one category at a time, as "
            f"rank({one!r}, long_df=long_df[long_df['category'] == {one!r}]).")


def _drop_unrun(df, category, label) -> tuple[pd.DataFrame, list]:
    """``df`` without the rows of package methods not run for ``category``
    (scMSI in a cross table), which must not shape the ranks, and their names;
    a name the registry does not know (the user's own) is kept."""
    listed = set(registry.list_methods(category=category))
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
    return df, foreign


#: the fewest methods a dataset needs to count in the ranking
_MIN_METHODS = 2


def _rankable(df, category, metrics, stored: bool) -> tuple[dict, list]:
    """``({dataset: its scored methods}, [datasets left out])``: a dataset with
    fewer than :data:`_MIN_METHODS` scored methods is left out; none left raises."""
    scored = df[df["value"].notna()]
    per_ds = {ds: sorted(set(g["method"])) for ds, g in scored.groupby("dataset")}
    degenerate = sorted(ds for ds, ms in per_ds.items() if len(ms) < _MIN_METHODS)
    kept = {ds: ms for ds, ms in per_ds.items() if ds not in degenerate}
    if kept:
        return kept, degenerate
    lone = sorted({m for ms in per_ds.values() for m in ms}, key=str)
    what = f"{metrics} scores" if isinstance(metrics, str) else "scores"
    on = _and_list(sorted(map(str, per_ds))) or category
    fix = ("Pass long_df= with the scores of more methods." if stored
           else "Add the scores of more methods to long_df.")
    who = (f"Only {_and_list(lone)} {'has' if len(lone) == 1 else 'have'}" if lone
           else "No method has")
    raise ValueError(f"{who} {what} on {on}: a ranking needs {_MIN_METHODS} or more "
                     f"methods. {fix}")


def _chips(name, category) -> dict:
    """``needs_labels``, ``runtime_tier``, ``worst_sec`` and ``env`` of a ranked
    row; ``'<id> (this run)'`` takes its method's, but not the observed times
    (they are the package's runs)."""
    from ..discover import method_info
    from ..engine import envs

    own = registry.method_of(name)
    if not own:
        return {"needs_labels": None, "runtime_tier": None, "worst_sec": None, "env": None}
    spec = registry.get(own)
    in_category = [v for v in spec.variants if v.when.get("category") == category]
    env = envs.group_for(spec.id)
    # a method may be supervised in one category only
    needs = (any(v.needs_labels for v in in_category) if in_category
             else bool(spec.needs_labels))
    runtime = method_info(spec.id)["runtime"]
    own_run = catalog._canonical_id(name) == spec.id
    return {"needs_labels": needs,
            "runtime_tier": runtime.get("tier") if own_run else None,
            "worst_sec": runtime.get("worst_sec") if own_run else None,
            "env": env}
