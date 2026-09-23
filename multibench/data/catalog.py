"""Catalog: parse files/*.csv into typed tables with canonical names."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import config

__all__ = ["methods", "datasets", "metrics", "canonical_id", "canonical_metric",
           "known_metrics", "PAPER_COLUMNS"]


def __dir__() -> list[str]:
    """Tab completion (``dir(mtb.catalog)``) shows the public API and the
    underscore names, not the module-level imports (``pd``, ``Path``, ``re``,
    ``config``, the ``annotations`` future-feature object).

    Every attribute stays accessible; only the listing changes (PEP 562).
    """
    return sorted(n for n in globals() if n in __all__ or n.startswith("_"))

# --- canonical method id + aliases -----------------------------------------
# Canonical id = the registry token. Map known display / result-dir spellings.
_ALIASES = {
    "seurat v3": "Seurat_v3",
    "seurat_v3": "Seurat_v3",
    "seurat.v3": "Seurat_v3",
    "seurat v4": "Seurat_v4",
    "seurat_v4": "Seurat_v4",
    "seurat.v4": "Seurat_v4",
    "seurat v5": "Seurat_v5",
    "seurat_v5": "Seurat_v5",
    "seurat.v5": "Seurat_v5",
    "seurat(wnn)": "Seurat_WNN",
    "seurat_wnn": "Seurat_WNN",
    "seurat.wnn": "Seurat_WNN",
    "mofa+": "MOFA2",
    "mofa2": "MOFA2",
    "online inmf": "online_iNMF",
    "online_inmf": "online_iNMF",
    "online.inmf": "online_iNMF",
    "ipolng": "iPOLNG",
}


def _registry_ids() -> list[str]:
    """Registry method ids (empty when the registry cannot be loaded)."""
    try:
        from ..engine import registry as _registry     # lazy: registry imports config
        return [s.id for s in _registry.load()]
    except Exception:       # a broken registry must not break name folding
        return []


def canonical_id(name: str, *, strict: bool = False) -> str:
    """Return the canonical method id for any known spelling.

    Parameters
    ----------
    name : str
        Any spelling of a method name (``"MOFA+"``, ``"totalvi"``, ``"Seurat(WNN)"``).
    strict : bool
        ``True`` = raise for a name that is neither an alias nor a registry id;
        ``False`` = return the folded name.

    Returns
    -------
    str
        The canonical id (an alias target or a registry id); for an unknown
        name, the input with spaces and dots collapsed to ``_``.

    Raises
    ------
    KeyError
        ``strict=True`` and the name is neither an alias nor a registry id.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.catalog.canonical_id("MOFA+"), mtb.catalog.canonical_id("totalvi")
    ('MOFA2', 'totalVI')
    >>> mtb.catalog.canonical_id("my method")          # unknown: separators folded
    'my_method'

    Notes
    -----
    **Resolution order.** The first rule that matches wins:

    1. the alias table, case-insensitive (``"MOFA+"`` -> ``"MOFA2"``,
       ``"Seurat(WNN)"`` -> ``"Seurat_WNN"``);
    2. a case-folded match against the registry ids, after collapsing spaces
       and dots to ``_`` (``"totalvi"`` -> ``"totalVI"``, ``"scmomat"`` ->
       ``"scMoMaT"``);
    3. for a name the registry does not know (a result-directory token, a
       user's own method name): the input with separators collapsed to
       ``_``, unchanged in case.

    **Strict mode.** The error is the message ``mtb.method_info`` and
    ``mtb.scan`` give, with a did-you-mean hint, e.g. ``"unknown method
    'Matlida'; did you mean 'Matilda'?; see mtb.list_methods()"``. An
    alias-table hit is returned without the registry check, even with
    ``strict=True``: ``"Seurat v4"`` -> ``"Seurat_v4"``, which is not a
    registry id. The default is lenient because result directories and user
    frames legitimately carry names the registry does not know.

    See Also
    --------
    mtb.list_methods : the registry ids.
    mtb.catalog.canonical_metric : the same normalisation for metric codes.
    """
    key = str(name).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    ids = _registry_ids()
    by_lower = {i.lower(): i for i in ids}
    folded = re.sub(r"[ .]+", "_", str(name).strip())
    if folded in ids:
        return folded
    if folded.lower() in by_lower:
        return by_lower[folded.lower()]
    if strict:
        import difflib
        hint = difflib.get_close_matches(folded, ids, n=1, cutoff=0.6)
        raise KeyError(
            f"unknown method {name!r}"
            + (f"; did you mean {hint[0]!r}?" if hint else "")
            + "; see mtb.list_methods()")
    # default: collapse separators to underscore, keep original casing token
    return folded


# --- metric code canonicalization ------------------------------------------
_METRIC_CANON = {
    "kbet": "kBET",
    "ifi": "iF1",
    "if1": "iF1",
    "ari": "ARI",
    "nmi": "NMI",
    "asw": "ASW",
    "iasw": "iASW",
    "clisi": "cLISI",
    "ilisi": "iLISI",
    "gc": "GC",
    "asw_batch": "ASW_batch",
    "pcr": "PCR",
    # raw scIB long-names -> canonical codes (some metric.csv use these)
    "ari_cluster/label": "ARI",
    "nmi_cluster/label": "NMI",
    "asw_label": "ASW",
    "isolated_label_f1": "iF1",
    "isolated_label_silhouette": "iASW",
}


def known_metrics() -> list[str]:
    """The canonical metric codes the package knows, in family order.

    Returns
    -------
    list of str
        The clustering / bio-conservation codes (``mtb.plot.CLUSTERING_METRICS``),
        the batch-correction codes (``mtb.plot.BATCH_METRICS``), then ``PCR``
        (principal-component regression, in some published tables).

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.catalog.known_metrics()
    ['ARI', 'NMI', 'ASW', 'iASW', 'iF1', 'cLISI', 'ASW_batch', 'GC', 'iLISI', 'kBET', 'PCR']

    See Also
    --------
    mtb.catalog.canonical_metric : maps any spelling to these codes; ``strict=True`` accepts only them.
    mtb.catalog.metrics : the description of each scIB metric.
    """
    seen: list[str] = []
    for v in _METRIC_CANON.values():
        if v not in seen:
            seen.append(v)
    order = ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI",
             "ASW_batch", "GC", "iLISI", "kBET"]
    return order + sorted(v for v in seen if v not in order)


def canonical_metric(code: str, *, strict: bool = False) -> str | None:
    """Return the canonical code for a metric name (``"ari"`` -> ``"ARI"``).

    Parameters
    ----------
    code : str
        A metric name in any spelling the package or scIB uses
        (``"kbet"``, ``"isolated_label_f1"``).
    strict : bool
        ``True`` = raise for a code not in ``known_metrics()``; ``False`` =
        return an unknown code unchanged (stripped).

    Returns
    -------
    str or None
        The canonical code; ``None`` for ``None``, an empty string or the
        string ``"nan"`` (a blank cell, which callers drop).

    Raises
    ------
    ValueError
        ``strict=True`` and the code is not a known metric (the message lists them).

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.catalog.canonical_metric("kbet"), mtb.catalog.canonical_metric("isolated_label_f1")
    ('kBET', 'iF1')
    >>> mtb.catalog.canonical_metric("my_score")        # unknown: kept as given
    'my_score'

    Notes
    -----
    **Spellings.** Matching ignores case and surrounding whitespace:
    ``"iFI"``, ``"if1"`` and ``"isolated_label_f1"`` all give ``"iF1"``. The
    raw scIB long names ``"ARI_cluster/label"``, ``"NMI_cluster/label"``,
    ``"ASW_label"`` and ``"isolated_label_silhouette"`` map to ``ARI``,
    ``NMI``, ``ASW`` and ``iASW``.

    **Unknown codes.** The default returns an unknown code stripped but
    otherwise unchanged, so a user frame can carry a metric the package
    does not know. With ``strict=True`` the error reads ``"unknown metric
    'nope'; valid: ['ARI', 'NMI', ...]"``.

    See Also
    --------
    mtb.catalog.known_metrics : the codes ``strict=True`` accepts.
    mtb.catalog.canonical_id : the same normalisation for method names.
    """
    if code is None:
        return None
    key = str(code).strip().lower()
    if key == "" or key == "nan":
        return None
    out = _METRIC_CANON.get(key, str(code).strip())
    if strict and out not in _METRIC_CANON.values():
        raise ValueError(f"unknown metric {code!r}; valid: {known_metrics()}")
    return out


# --- the one ``metrics=`` selector -----------------------------------------
#: family tokens the ``metrics=`` knob accepts (besides None and a list of codes)
_METRIC_TOKENS = ("all", "clustering", "batch")


class MetricSelection:
    """What a ``metrics=`` argument resolved to (see :func:`metric_selection`).

    ``family`` is ``"all"``, ``"clustering"`` or ``"batch"``: the token that
    was given, or - for an explicit list - the smallest family holding every
    code. ``codes`` is ``None`` for "no restriction" (``None`` / ``"all"``)
    and otherwise the canonical codes selected, in request order. ``explicit``
    is ``True`` when a list of codes was given.
    """
    __slots__ = ("family", "codes", "explicit")

    def __init__(self, family: str, codes, explicit: bool):
        self.family, self.codes, self.explicit = family, codes, explicit

    def __repr__(self) -> str:
        return f"MetricSelection(family={self.family!r}, codes={self.codes!r}, explicit={self.explicit})"


def metric_selection(metrics, *, extra=(), kw: str = "metrics") -> MetricSelection:
    """Resolve the ``metrics=`` knob shared by ``evaluate``, ``load_results`` and ``recommend``.

    Parameters
    ----------
    metrics
        ``None`` (no restriction), a family token (``"clustering"``,
        ``"batch"``, ``"all"``) or a list of metric codes (case/alias
        tolerant: ``["ari", "kbet"]`` -> ``ARI, kBET``).
    extra : collection of str
        Codes accepted on top of :func:`known_metrics` - the metrics present
        in a frame, so a user's own metric name in a long file can still be
        selected; they are listed in the error too.
    kw : str
        The keyword the value arrived through, named in the errors.

    Returns
    -------
    MetricSelection

    Raises
    ------
    ValueError
        A string that is not a family token (``"unknown metrics= token
        'ARI'; valid: 'all', 'clustering', 'batch' - or a list of codes"``;
        a :func:`multibench.list_tasks` token gets an extra sentence saying
        the slot selects a metric family), an unknown code in a list
        (listing the valid codes), an empty list, or a list that names the
        same metric twice after canonicalisation.
    TypeError
        Anything that is neither ``None``, a string nor a collection.
    """
    from ..plot.bar import BATCH_METRICS, CLUSTERING_METRICS   # families live in plot
    fams = {"clustering": list(CLUSTERING_METRICS), "batch": list(BATCH_METRICS)}
    if metrics is None or (isinstance(metrics, str) and metrics == "all"):
        return MetricSelection("all", None, False)
    if isinstance(metrics, str):
        if metrics in fams:
            return MetricSelection(metrics, fams[metrics], False)
        hint = ""
        try:
            from ..engine.registry import list_tasks
            tasks = list_tasks()
        except Exception:
            tasks = []
        if metrics in tasks:
            hint = (f" - {kw}= selects a metric family, not a mtb.list_tasks() token; "
                    f"'dimension_reduction' and 'clustering' share the 'clustering' "
                    f"family, and the scIB families are the only metrics computed")
        elif canonical_metric(metrics) in known_metrics():
            hint = f" - a single code goes in a list: {kw}=[{canonical_metric(metrics)!r}]"
        raise ValueError(
            f"unknown {kw}= token {metrics!r}; valid: 'all', 'clustering', 'batch' "
            f"(or a list of metric codes, e.g. {kw}=['ARI', 'NMI']){hint}")
    if isinstance(metrics, (bytes, dict)) or not hasattr(metrics, "__iter__"):
        raise TypeError(
            f"{kw}= must be None, a family token ('all', 'clustering', 'batch') or "
            f"a list of metric codes; got {type(metrics).__name__}")
    wanted = list(metrics)
    if not wanted:
        raise ValueError(
            f"{kw}=[] selects nothing; pass None for every metric, a family token, "
            f"or a non-empty list of codes")
    valid = list(known_metrics())
    present = sorted(set(map(str, extra)))
    codes, unknown = [], []
    for m in wanted:
        c = canonical_metric(m)
        if c is None or (c not in valid and c not in present):
            unknown.append(m)
        else:
            codes.append(c)
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; valid codes: {valid}"
            + (f"; present in this frame: {present}" if present else ""))
    dup = sorted({c for c in codes if codes.count(c) > 1})
    if dup:
        raise ValueError(
            f"{kw}= names the same metric twice after canonicalisation: {dup}")
    in_clu = any(c in fams["clustering"] for c in codes)
    in_bat = any(c in fams["batch"] for c in codes)
    family = "all" if (in_clu and in_bat) or not (in_clu or in_bat) else (
        "clustering" if in_clu else "batch")
    return MetricSelection(family, codes, True)


# --- tables ----------------------------------------------------------------
def _split_multivalue(cell: object) -> list[str]:
    if pd.isna(cell):
        return []
    parts = re.split(r"[\n;,]+", str(cell))
    out = []
    for p in parts:
        t = p.strip().lower().replace(" integration", "").replace("integration", "").strip()
        if t:
            out.append(t)
    return out


def methods(files_dir: Path | str | None = None) -> pd.DataFrame:
    """Table of the benchmark's methods, one row per method.

    Parameters
    ----------
    files_dir : Path | str | None
        Folder holding ``method.csv``; ``None`` = the package's shipped ``files/``.

    Returns
    -------
    pandas.DataFrame
        One row per method. Read ``method``, ``needs_labels``, ``atac`` and
        ``categories``; all columns are listed in Notes.

    Examples
    --------
    >>> import multibench as mtb
    >>> m = mtb.catalog.methods()
    >>> m[m.needs_labels][["method", "categories"]]
    >>> m[m.categories.map(lambda c: "vertical" in c)].method.tolist()

    Notes
    -----
    **Column reference.**

    - ``method`` - the name as spelled in ``method.csv``;
    - ``canonical_id`` - the registry id (``mtb.catalog.canonical_id``);
    - ``language`` - ``'python'`` or ``'r'`` (lower-cased);
    - ``deep_learning`` - ``'Yes'`` / ``'No'``, as in the CSV;
    - ``atac`` - ``'peak'``, ``'gene_activity'`` or ``None``;
    - ``output`` - ``'embedding'`` or ``'graph'``;
    - ``needs_labels`` - bool, the method needs cell-type labels;
    - ``categories`` / ``tasks`` - lists of integration categories and tasks.

    **Registry overlay.** For every registered id, ``needs_labels``,
    ``atac``, ``categories`` and ``tasks`` come from the method registry,
    the source ``mtb.method_info`` and ``mtb.scan`` read. A row without a
    registry entry keeps the CSV values. ``language``, ``deep_learning`` and
    ``output`` come from ``method.csv``.

    See Also
    --------
    mtb.list_methods : the registry method ids, optionally per category.
    mtb.method_info : everything the registry knows about one method.
    mtb.catalog.canonical_id : the normalisation behind the ``canonical_id`` column.
    """
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "method.csv")
    # normalize column access by stripping whitespace/newlines
    cols = {c: c.strip().replace("\n", " ").strip() for c in raw.columns}
    raw = raw.rename(columns=cols)
    out = pd.DataFrame()
    out["method"] = raw["Methods"].astype(str).str.strip()
    out["canonical_id"] = out["method"].map(canonical_id)
    out["language"] = raw["Programming Language"].astype(str).str.strip().str.lower()
    out["deep_learning"] = raw["Deep Learning"].astype(str).str.strip()
    out["atac"] = raw["Peak/Gene Activity"].astype(str).str.strip()
    out["output"] = raw["Output"].astype(str).str.strip()
    out["needs_labels"] = (
        raw["CellType Information Required"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    )
    out["categories"] = raw["Integration Categories"].map(_split_multivalue)
    out["tasks"] = raw["Task Categories"].map(_split_multivalue)
    # registry overlay: the derived / validated values win over the CSV prose
    from ..engine import registry as _registry
    specs = {s.id: s for s in _registry.load()}
    atac_col = out["atac"].astype(object)
    for i, cid in enumerate(out["canonical_id"]):
        spec = specs.get(cid)
        if spec is None:
            continue
        out.at[i, "needs_labels"] = bool(spec.needs_labels)
        atac_col.at[i] = spec.atac
        out.at[i, "categories"] = list(spec.categories)
        out.at[i, "tasks"] = list(spec.tasks)
    out["atac"] = atac_col
    out["needs_labels"] = out["needs_labels"].astype(bool)
    return out


def _dataset_sort_key(ds: str):
    """Natural order for dataset ids: ``D2 < D11 < D11s < SD7``."""
    m = re.match(r"^([A-Za-z]*)(\d+)(.*)$", str(ds))
    if not m:
        return (str(ds), 0, "")
    return (m.group(1).upper(), int(m.group(2)), m.group(3))


#: descriptive columns of dataset.csv, to be transcribed from the paper's
#: supplementary dataset table. The shipped file leaves them empty, and
#: datasets() returns only the ones a CSV fills (files/README_PROVENANCE.txt).
PAPER_COLUMNS = ["assay", "tissue", "n_cells", "n_batches", "source"]


def datasets(files_dir: Path | str | None = None, *,
             category: str | None = None) -> pd.DataFrame:
    """Table of the benchmark's datasets, joined with the stored results.

    Parameters
    ----------
    files_dir : Path | str | None
        Folder holding ``dataset.csv``; ``None`` = the package's shipped ``files/``.
    category : str | None
        Keep only datasets with stored results in this integration category
        (``vertical``, ``diagonal``, ``mosaic`` or ``cross``); ``None`` = all.

    Returns
    -------
    pandas.DataFrame
        One row per dataset id. Read ``dataset``, ``category`` and
        ``has_results``; all columns are listed in Notes.

    Raises
    ------
    ValueError
        Unknown ``category``; the message lists the valid ones.

    Examples
    --------
    >>> import multibench as mtb
    >>> ds = mtb.catalog.datasets()
    >>> ds[ds.has_results][["dataset", "category"]]
    >>> mtb.catalog.datasets(category="vertical").dataset.tolist()

    Notes
    -----
    **Column reference.**

    - ``dataset`` - the id (``D11``, ``SD15``, ...);
    - ``dataset name`` - a duplicate of ``dataset``, kept for one release
      for older callers;
    - ``simulated`` - bool, ids starting with ``SD``;
    - ``category`` - the integration categories whose stored results
      (published or re-run) contain the dataset, ``";"``-joined; ``None``
      when no stored results exist;
    - ``has_results`` - bool, a stored metric table (``mtb.load_results``)
      covers it.

    A ``dataset.csv`` passed through ``files_dir`` may also fill the
    descriptive columns ``assay``, ``tissue``, ``n_cells``, ``n_batches`` and
    ``source``; each one that holds a value is added. The shipped file
    leaves them empty.

    **Row set.** The rows are the union of ``dataset.csv`` and every id with
    stored results (``mtb.available_datasets(source="both")``), so
    ``D11s``/``D28s``/``D45s``/``D52s`` and ``D24`` (published tables only)
    are listed although ``dataset.csv`` does not name them. Ids missing from
    the CSV are appended after it, in natural order.

    **Subsamples.** D11s, D28s, D45s and D52s are random subsamples of the
    full datasets, used for the second re-run sweep; they cannot be fetched.
    D28s holds 60% of D28's cells.

    **When the columns are computed.** ``category`` and ``has_results`` are
    derived at each call from ``mtb.available_datasets``; a missing or
    unreadable result tree leaves them empty instead of raising.

    See Also
    --------
    mtb.available_datasets : dataset ids with stored results, per category.
    mtb.data.fetchable : dataset ids that can be downloaded.
    mtb.load_results : the stored metric tables themselves.
    """
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "dataset.csv")
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    raw = raw.dropna(how="all", axis=0)
    name_col = "dataset name" if "dataset name" in raw.columns else "dataset"
    raw = raw.assign(**{name_col: raw[name_col].astype(str).str.strip()})

    # result-tree-derived columns are filled at call time, so the table never
    # lists fewer datasets than load_results can serve; ids with results that
    # dataset.csv lacks are appended with the paper columns empty
    from . import results as _results
    cat_of: dict[str, list[str]] = {}
    for cat in ("cross", "diagonal", "mosaic", "vertical"):
        try:
            for ds in _results.available_datasets(cat, source="both"):
                cat_of.setdefault(ds, []).append(cat)
        except Exception:       # a broken/absent tree must not break the catalog
            continue
    csv_ids = list(raw[name_col])
    extra = sorted((d for d in cat_of if d not in set(csv_ids)), key=_dataset_sort_key)
    if extra:
        raw = pd.concat([raw, pd.DataFrame({name_col: extra})], ignore_index=True)

    out = pd.DataFrame()
    out["dataset"] = raw[name_col].astype(str).str.strip()
    out["dataset name"] = out["dataset"]          # back-compat alias (one release)
    out["simulated"] = out["dataset"].str.upper().str.startswith("SD")
    out["category"] = out["dataset"].map(lambda d: ";".join(cat_of[d]) if d in cat_of else None)
    out["has_results"] = out["dataset"].isin(cat_of.keys())

    # a descriptive column is shown only when the CSV fills it: the shipped
    # file leaves all five empty
    for col in PAPER_COLUMNS:
        if col in raw.columns and raw[col].notna().any():
            out[col] = raw[col].values
    if category is not None:
        config.category_folder(category)
        out = out[out["category"].fillna("").str.split(";").map(lambda cs: category in cs)]
    return out.reset_index(drop=True)


def metrics(files_dir: Path | str | None = None) -> pd.DataFrame:
    """Table describing the scIB metrics, from ``files/metric_full.csv``.

    Parameters
    ----------
    files_dir : Path | str | None
        Folder holding ``metric_full.csv``; ``None`` = the package's shipped ``files/``.

    Returns
    -------
    pandas.DataFrame
        One row per scIB metric, with the CSV's columns (``metric`` and
        ``description`` in the shipped file).

    Examples
    --------
    >>> import multibench as mtb
    >>> tab = mtb.catalog.metrics()
    >>> tab.set_index("metric").loc["kBET", "description"]

    Notes
    -----
    **Rows.** The ten scIB metrics ``ARI, NMI, ASW, iASW, iF1, cLISI,
    ASW_batch, GC, iLISI, kBET``; header whitespace is stripped. The
    canonical code vocabulary, including ``PCR`` from the published tables,
    is ``mtb.catalog.known_metrics``.

    See Also
    --------
    mtb.catalog.known_metrics : the canonical metric codes.
    mtb.evaluate : computes these metrics for an embedding.
    """
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "metric_full.csv")
    return raw.rename(columns={c: c.strip() for c in raw.columns})
