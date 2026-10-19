"""The reference tables: ``mtb.catalog.datasets()`` (which datasets) and
``mtb.catalog.metrics()`` (which metrics)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import _compat, config

__all__ = ["datasets", "metrics"]


def __dir__() -> list[str]:
    """``dir(mtb.catalog)`` lists the public API and the dunders only (PEP 562);
    the other names stay reachable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})

#: display and result-folder spellings -> method id
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
    """The registry's method ids."""
    from ..engine import registry     # lazy: registry imports config
    return [s.id for s in registry.load()]


def _canonical_id(name: str, *, strict: bool = False) -> str:
    """The method id for any known spelling (``"MOFA+"`` -> ``"MOFA2"``,
    ``"totalvi"`` -> ``"totalVI"``): the alias table, then a case-folded match
    of the ids after collapsing spaces and dots to ``_``; an unknown name comes
    back folded, or with ``strict=True`` raises the did-you-mean ``KeyError``."""
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
        from ..engine import registry
        hint = difflib.get_close_matches(folded, ids, n=1, cutoff=0.6)
        raise KeyError(registry.unknown_method_message(name, hint[0] if hint else ""))
    return folded


#: the metric codes of each ``metrics=`` family, in the paper's order. iASW and iF1
#: are isolated-label scores, which scIB files under bio conservation: a
#: single-batch dataset has them and no batch-correction result.
FAMILY_METRICS = {
    "clustering": ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"],
    "batch": ["ASW_batch", "ARI_batch", "NMI_batch", "GC", "iLISI", "kBET"],
    "classification": ["OCA", "ACA", "Spec", "Sens", "F1"],
}

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
    "ari_batch": "ARI_batch",
    "nmi_batch": "NMI_batch",
    "oca": "OCA",
    "aca": "ACA",
    "spec": "Spec",
    "sens": "Sens",
    "f1": "F1",
    # scIB's long names, in some published metric.csv
    "ari_cluster/label": "ARI",
    "nmi_cluster/label": "NMI",
    "asw_label": "ASW",
    "isolated_label_f1": "iF1",
    "isolated_label_silhouette": "iASW",
}


def _known_metrics() -> list[str]:
    """The metric codes, in family order: the rows of :func:`metrics`."""
    return [c for codes in FAMILY_METRICS.values() for c in codes]


def _canonical_metric(code: str, *, strict: bool = False) -> str | None:
    """The canonical code for a metric name in any case or scIB spelling
    (``"kbet"`` -> ``"kBET"``, ``"isolated_label_f1"`` -> ``"iF1"``); ``None``
    for a blank; an unknown code comes back stripped, or with ``strict=True``
    raises ``ValueError``."""
    if code is None:
        return None
    key = str(code).strip().lower()
    if key == "" or key == "nan":
        return None
    out = _METRIC_CANON.get(key, str(code).strip())
    if strict and out not in _METRIC_CANON.values():
        raise ValueError(f"unknown metric {code!r}; valid: {_known_metrics()}")
    return out


class MetricSelection:
    """What a ``metrics=`` argument resolved to: ``family`` (the token, or for
    a list the family holding every code, else ``"all"``), ``codes`` (``None``
    = no restriction, else the canonical codes in request order) and
    ``explicit`` (a list was given)."""
    __slots__ = ("family", "codes", "explicit")

    def __init__(self, family: str, codes, explicit: bool):
        self.family, self.codes, self.explicit = family, codes, explicit

    def __repr__(self) -> str:
        return f"MetricSelection(family={self.family!r}, codes={self.codes!r}, explicit={self.explicit})"


def _unknown_metrics(kw: str, what: str) -> str:
    """The one error text of a ``metrics=`` value that names no family or code."""
    return (f"Unknown {kw}= {what}. Pass 'all', 'clustering', 'batch' or "
            f"'classification', or metric codes such as {kw}=['ARI', 'NMI'].")


def metric_selection(metrics, *, extra=(), kw: str = "metrics") -> MetricSelection:
    """Resolve the ``metrics=`` knob shared by ``evaluate``, ``load_results`` and ``recommend``.

    Parameters
    ----------
    metrics
        ``None`` (no restriction), a family token (``"clustering"``,
        ``"batch"``, ``"classification"``, ``"all"``), one metric code, or a list
        of codes and families (case/alias tolerant: ``["ari", "kbet"]`` -> ``ARI,
        kBET``; a family in a list adds its codes, kBET only when named).
    extra : collection of str
        Codes accepted on top of :func:`_known_metrics` - the metrics present
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
        A string that is neither a family token nor a metric code (a method
        task token gets an extra sentence saying the slot selects a metric
        family), an unknown code in a list
        (listing the valid codes), an empty list, or a list that names the
        same metric twice after canonicalisation.
    TypeError
        Anything that is neither ``None``, a string nor a collection.
    """
    fams = {k: list(v) for k, v in FAMILY_METRICS.items()}
    if isinstance(metrics, str) and metrics.lower() in (*fams, "all"):
        metrics = metrics.lower()          # "Clustering" names the family too
    if metrics is None or (isinstance(metrics, str) and metrics == "all"):
        return MetricSelection("all", None, False)
    if isinstance(metrics, str):
        if metrics in fams:
            return MetricSelection(metrics, fams[metrics], False)
        code = _canonical_metric(metrics)
        if code in _known_metrics() or code in set(map(str, extra)):
            metrics = [code]               # one code: a list of one
        else:
            hint = ""
            from ..engine.registry import list_tasks
            if metrics in list_tasks():
                hint = (f" {kw}= selects a metric family, not a task. The tasks "
                        f"dimension_reduction and clustering both use the clustering "
                        f"family.")
            raise ValueError(_unknown_metrics(kw, f"token {metrics}") + hint)
    if isinstance(metrics, (bytes, dict)) or not hasattr(metrics, "__iter__"):
        got = type(metrics).__name__
        raise TypeError(
            f"{kw}= takes None, a family ('all', 'clustering', 'batch', 'classification'), "
            f"a metric code, or a list of them. It got {'an' if got[:1] in 'aeiou' else 'a'} {got}.")
    wanted = list(metrics)
    if not wanted:
        raise ValueError(
            f"{kw}=[] selects nothing. Pass None for every metric, a family token, "
            f"or a non-empty list of codes.")
    valid = _known_metrics()
    present = sorted(set(map(str, extra)))
    codes, unknown = [], []
    for m in wanted:
        if isinstance(m, str) and m.lower() in (*fams, "all"):
            # a family in the list: its codes (kBET only when named, as for the token)
            family = valid if m.lower() == "all" else fams[m.lower()]
            codes += [c for c in family if c != "kBET" and c not in codes]
            continue
        c = _canonical_metric(m)
        if c is None or (c not in valid and c not in present):
            unknown.append(m)
        else:
            codes.append(c)
    if unknown:
        raise ValueError(_unknown_metrics(kw, f"token{'s' if len(unknown) > 1 else ''} "
                                              f"{', '.join(map(str, unknown))}")
                         + (f" This frame also has {', '.join(present)}." if present else ""))
    named = [_canonical_metric(m) for m in wanted
             if not (isinstance(m, str) and m.lower() in (*fams, "all"))]
    dup = sorted({c for c in named if named.count(c) > 1})
    codes = list(dict.fromkeys(codes))
    if dup:
        raise ValueError(
            f"{kw}= names the same metric twice after canonicalisation: {dup}")
    holding = [f for f, members in fams.items() if all(c in members for c in codes)]
    return MetricSelection(holding[0] if holding else "all", codes, True)


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


_FILES_DIR = {"files_dir": _compat.note("leave it out for the package's tables")}


@_compat.keywords("catalog.methods", private="_methods", kept=_FILES_DIR,
                  positional=("files_dir",))
def methods() -> pd.DataFrame:
    """Table of the benchmark's methods, one row per method.

    Returns
    -------
    pandas.DataFrame
        One row per method: ``method`` (the id every function takes),
        ``name`` (the article's spelling, such as ``MOFA+``), ``language``,
        ``deep_learning``, ``atac`` (``'peak'`` / ``'gene_activity'`` /
        ``None``), ``output`` (``'embedding'`` / ``'graph'``),
        ``needs_labels`` and ``categories``.

    Examples
    --------
    >>> import multibench as mtb
    >>> m = mtb.catalog.methods()
    >>> m[m.needs_labels][["method", "categories"]]
    >>> m[m.categories.map(lambda c: "vertical" in c)].method.tolist()

    See Also
    --------
    mtb.find_methods : the method ids, filtered by category, modalities and more.
    mtb.method_info : everything the package knows about one method.
    """
    return _methods()


methods = _compat.renamed("catalog.methods", "mtb.find_methods() and mtb.method_info(method)",
                          methods)


def _methods(files_dir: Path | str | None = None) -> pd.DataFrame:
    """:func:`methods`, with the deprecated ``files_dir=`` (the folder holding
    ``method.csv``)."""
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "method.csv")
    cols = {c: c.strip().replace("\n", " ").strip() for c in raw.columns}
    raw = raw.rename(columns=cols)
    out = pd.DataFrame()
    out["name"] = raw["Methods"].astype(str).str.strip()
    out.insert(0, "method", out["name"].map(_canonical_id))
    out["language"] = raw["Programming Language"].astype(str).str.strip().str.lower()
    out["deep_learning"] = raw["Deep Learning"].astype(str).str.strip()
    out["atac"] = raw["Peak/Gene Activity"].astype(str).str.strip()
    out["output"] = raw["Output"].astype(str).str.strip()
    out["needs_labels"] = (
        raw["CellType Information Required"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    )
    out["categories"] = raw["Integration Categories"].map(_split_multivalue)
    # the registry's values win over the CSV prose
    from ..engine import registry
    specs = {s.id: s for s in registry.load()}
    atac_col = out["atac"].astype(object)
    for i, cid in enumerate(out["method"]):
        spec = specs.get(cid)
        if spec is None:
            continue
        out.at[i, "needs_labels"] = bool(spec.needs_labels)
        atac_col.at[i] = spec.atac
        out.at[i, "categories"] = list(spec.categories)
    out["atac"] = atac_col
    out["needs_labels"] = out["needs_labels"].astype(bool)
    return out


def _dataset_sort_key(ds: str):
    """Natural order for dataset ids: ``D2 < D11 < D11s < SD7``."""
    m = re.match(r"^([A-Za-z]*)(\d+)(.*)$", str(ds))
    if not m:
        return (str(ds), 0, "")
    return (m.group(1).upper(), int(m.group(2)), m.group(3))


#: descriptive columns of dataset.csv (the paper's dataset table); the shipped
#: file leaves them empty, and datasets() returns only the filled ones
_PAPER_COLUMNS = ["assay", "tissue", "n_cells", "n_batches", "source"]


@_compat.keywords("catalog.datasets", private="_datasets", kept=_FILES_DIR,
                  positional=("files_dir",))
def datasets(*, category: str | None = None) -> pd.DataFrame:
    """The datasets ``mtb.data.fetch`` downloads or ``mtb.load_results`` holds.

    Parameters
    ----------
    category : str | None
        ``vertical``, ``diagonal``, ``mosaic`` or ``cross``: only the datasets
        of that category; ``None`` = all.

    Returns
    -------
    pandas.DataFrame
        One row per dataset: ``dataset``, ``category`` (``;``-joined when the
        benchmark scored it in several), ``fetchable`` (``mtb.data.fetch``
        downloads it) and ``has_results`` (``mtb.load_results`` has stored
        results for it).

    Examples
    --------
    >>> import multibench as mtb
    >>> ds = mtb.catalog.datasets()
    >>> ds[ds.fetchable].dataset.tolist()
    >>> ds[ds.has_results]
    """
    return _datasets(category=category)


def _datasets(files_dir: Path | str | None = None, *,
              category: str | None = None) -> pd.DataFrame:
    """:func:`datasets`, with the deprecated ``files_dir=`` (the folder
    holding ``dataset.csv``)."""
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "dataset.csv")
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    raw = raw.dropna(how="all", axis=0)
    name_col = "dataset name" if "dataset name" in raw.columns else "dataset"
    raw = raw.assign(**{name_col: raw[name_col].astype(str).str.strip()})

    # from the result tree at each call, so no dataset load_results serves is
    # missing; the article's tables give the category of D24 and D25 too
    from . import results
    cat_of: dict[str, list[str]] = {}
    served: set[str] = set()
    for cat in results._CATEGORIES:
        try:
            for ds in results._available_datasets(cat, source="both"):
                cat_of.setdefault(ds, []).append(cat)
            served.update(results._available_datasets(cat, source="rerun"))
        except Exception:       # a broken/absent tree must not break the catalog
            continue
    csv_ids = list(raw[name_col])
    extra = sorted((d for d in cat_of if d not in set(csv_ids)), key=_dataset_sort_key)
    if extra:
        raw = pd.concat([raw, pd.DataFrame({name_col: extra})], ignore_index=True)

    from .fetch import AVAILABLE, DEMO_CATEGORIES
    for ds, cat in DEMO_CATEGORIES.items():
        cat_of.setdefault(ds, [cat])
    demos = [d for d in DEMO_CATEGORIES if d not in set(raw[name_col])]
    raw = pd.concat([raw, pd.DataFrame({name_col: sorted(demos, key=_dataset_sort_key)})],
                    ignore_index=True)
    out = pd.DataFrame()
    out["dataset"] = raw[name_col].astype(str).str.strip()
    out["category"] = out["dataset"].map(lambda d: ";".join(cat_of[d]) if d in cat_of else None)
    out["fetchable"] = out["dataset"].isin(set(AVAILABLE))
    out["has_results"] = out["dataset"].isin(served)

    for col in _PAPER_COLUMNS:
        if col in raw.columns and raw[col].notna().any():
            out[col] = raw[col].values
    # the rows a user can act on: data to fetch or stored results to compare with
    out = out[out["fetchable"] | out["has_results"]]
    if category is not None:
        config.category_folder(category)
        out = out[out["category"].fillna("").str.split(";").map(lambda cs: category in cs)]
    return out.reset_index(drop=True)


@_compat.keywords("catalog.metrics", private="_metrics", kept=_FILES_DIR,
                  positional=("files_dir",))
def metrics() -> pd.DataFrame:
    """Table describing the metrics: one row per metric code.

    Returns
    -------
    pandas.DataFrame
        One row per metric code: ``metric``, ``family`` (``clustering``,
        ``batch`` or ``classification``, the families ``metrics=`` takes) and
        ``description``.

    Examples
    --------
    >>> import multibench as mtb
    >>> tab = mtb.catalog.metrics()
    >>> tab.set_index("metric").loc["kBET", "description"]

    See Also
    --------
    mtb.evaluate : computes these metrics for an embedding.
    """
    return _metrics()


def _metrics(files_dir: Path | str | None = None) -> pd.DataFrame:
    """:func:`metrics`, with the deprecated ``files_dir=`` (the folder holding
    ``metric_full.csv``)."""
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "metric_full.csv")
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    family = {c: f for f, codes in FAMILY_METRICS.items() for c in codes}
    raw.insert(1, "family", raw["metric"].map(family))
    return raw


_ANY_SPELLING = _compat.note("every function accepts any spelling of a name")
__getattr__ = _compat.module_getattr(__name__, "catalog", old={
    "canonical_id": (_canonical_id, _ANY_SPELLING),
    "canonical_metric": (_canonical_metric, _ANY_SPELLING),
    "known_metrics": (_known_metrics, _compat.use("catalog.metrics()")),
    "PAPER_COLUMNS": (_PAPER_COLUMNS, _compat.NO_EFFECT),
})
