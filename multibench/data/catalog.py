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
    underscore names, not the imports this module merely uses.

    ``dir()`` always sorts, so the public names cannot be listed FIRST; what
    can be done (PEP 562) is to leave out the leak-through module-level
    imports (``pd``, ``Path``, ``re``, ``config`` and the ``annotations``
    future-feature object - ``mtb.catalog.annotations()`` was a puzzling
    ``TypeError`` for one re-tester). Every attribute stays accessible;
    only the listing changes.
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

    Resolution order: the alias table (``"MOFA+"`` -> ``"MOFA2"``,
    ``"Seurat(WNN)"`` -> ``"Seurat_WNN"``), then a CASE-FOLDED match against
    the registry ids (``"totalvi"`` -> ``"totalVI"``, ``"scmomat"`` ->
    ``"scMoMaT"``), then - for a name the registry does not know - the input
    with separators collapsed to ``_`` (a result-directory token, a user's own
    method name), unchanged in case.

    Parameters
    ----------
    name : str
        Any spelling of a method id.
    strict : bool, keyword-only
        ``True`` raises ``KeyError`` for a name that is not a registry id
        after aliasing and case-folding, with a did-you-mean hint (the same
        message :func:`multibench.method_info` / ``scan`` give). Default
        ``False`` returns the folded token, because result directories and
        user frames legitimately carry names the registry does not know.

    Returns
    -------
    str
        The canonical id.

    Raises
    ------
    KeyError
        ``strict=True`` and the name is unknown, e.g.
        ``"unknown method 'Matlida'; did you mean 'Matilda'?; see
        mtb.list_methods()"``.
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
        ``ARI, NMI, ASW, iASW, iF1, cLISI`` (clustering / bio conservation),
        ``ASW_batch, GC, iLISI, kBET`` (batch correction) and ``PCR``
        (principal-component regression, present in some published tables).
        This is the vocabulary ``canonical_metric(strict=True)`` validates
        against; the two scIB families are ``mtb.plot.CLUSTERING_METRICS`` /
        ``mtb.plot.BATCH_METRICS``.
    """
    seen: list[str] = []
    for v in _METRIC_CANON.values():
        if v not in seen:
            seen.append(v)
    order = ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI",
             "ASW_batch", "GC", "iLISI", "kBET"]
    return order + sorted(v for v in seen if v not in order)


def canonical_metric(code: str, *, strict: bool = False) -> str | None:
    """Canonicalize a metric short-code (``"ari"`` -> ``"ARI"``, ``"kbet"`` -> ``"kBET"``).

    Parameters
    ----------
    code : str
        A metric name in any spelling the package or scIB uses
        (``"iFI"``, ``"isolated_label_f1"`` and ``"if1"`` all -> ``"iF1"``).
    strict : bool, keyword-only
        ``True`` raises ``ValueError`` for a code that is not in
        :func:`known_metrics` after canonicalisation. Default ``False``
        returns an unknown code stripped but otherwise unchanged, so a user
        frame can carry a metric the package does not know.

    Returns
    -------
    str or None
        The canonical code; ``None`` for ``None``, an empty string or the
        string ``"nan"`` (a blank cell), which callers drop.

    Raises
    ------
    ValueError
        ``strict=True`` and the code is unknown:
        ``"unknown metric 'nope'; valid: ['ARI', 'NMI', ...]"``.
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
    """The methods table: ``method, canonical_id, language, deep_learning, atac,
    output, needs_labels, categories, tasks`` (``categories``/``tasks`` are
    lists).

    ``deep_learning`` and ``output`` come from the shipped ``method.csv``.
    ``needs_labels``, ``atac`` (``'peak'`` / ``'gene_activity'`` / ``None``),
    ``categories`` and ``tasks`` are OVERLAID from the method registry
    (``registry.get(canonical_id)``) for every row whose id is registered, so
    the table cannot disagree with ``method_info`` / ``scan`` (the CSV's hand
    columns had drifted on 37 of 40 rows). Rows without a registry entry keep
    the CSV values. ``language`` is the CSV value (lower-cased), cross-checked
    against the registry's.
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


#: columns of dataset.csv that must be transcribed from the paper's
#: supplementary dataset table. They are NOT derivable from anything in this
#: repository and are shipped EMPTY (nullable) until transcribed - see
#: files/README_PROVENANCE.txt. Nothing in the package fabricates them.
PAPER_COLUMNS = ["assay", "tissue", "n_cells", "n_batches", "source"]


def datasets(files_dir: Path | str | None = None, *,
             category: str | None = None) -> pd.DataFrame:
    """The benchmark's dataset table, joined with what the result tree knows.

    Parameters
    ----------
    files_dir : path-like, optional
        Directory holding ``dataset.csv`` (default: the package's
        ``files/``).
    category : str, keyword-only, optional
        Keep only datasets whose result tree places them in this integration
        category (``"vertical"``, ``"diagonal"``, ``"mosaic"``, ``"cross"``).

    Returns
    -------
    pandas.DataFrame
        One row per dataset id with columns

        * ``dataset`` - the id (``D11``, ``SD7``, ...); ``dataset name`` is
          kept as a duplicate column for one release (old callers read it).
          The rows are the union of ``dataset.csv`` and every id that has
          stored results (:func:`multibench.available_datasets` with
          ``source="both"``), so ``D11s``/``D28s``/``D45s``/``D52s`` (the
          re-run subsamples) and ``SD7``-``SD10``/``D24`` (published tables
          only) are listed even though the paper table does not name them;
          ids missing from the CSV are appended after it, in natural order;
        * ``simulated`` - ``bool``, ids starting with ``SD``;
        * ``category`` - the integration category whose stored results
          (published or re-run) contain the dataset, ``";"``-joined if
          several, NaN when no stored results exist; derived at call time
          from :func:`multibench.available_datasets` so it never goes stale;
        * ``has_results`` - ``bool``, whether any stored metric table
          (``load_results``) covers it;
        * ``assay, tissue, n_cells, n_batches, source`` - the paper's
          descriptive columns, nullable; empty until transcribed from the
          supplementary table (see ``files/README_PROVENANCE.txt``).
    """
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "dataset.csv")
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    raw = raw.dropna(how="all", axis=0)
    name_col = "dataset name" if "dataset name" in raw.columns else "dataset"
    raw = raw.assign(**{name_col: raw[name_col].astype(str).str.strip()})

    # result-tree-derived columns: filled at call time, so the table can never
    # list fewer datasets than load_results can serve. Ids that ship results
    # but are not in dataset.csv (the re-run subsampled D11s/D28s/D45s/D52s, the
    # simulated SD7-SD10, D24) are APPENDED with the paper columns empty.
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

    for col in PAPER_COLUMNS:
        out[col] = raw[col].values if col in raw.columns else pd.NA
    if category is not None:
        config.category_folder(category)
        out = out[out["category"].fillna("").str.split(";").map(lambda cs: category in cs)]
    return out.reset_index(drop=True)


def metrics(files_dir: Path | str | None = None) -> pd.DataFrame:
    """The metric details table shipped in ``files/metric_full.csv``.

    Parameters
    ----------
    files_dir : path-like, optional
        Directory holding ``metric_full.csv`` (default: the package's
        ``files/``).

    Returns
    -------
    pandas.DataFrame
        One row per scIB metric (``ARI, NMI, ASW, iASW, iF1, cLISI,
        ASW_batch, GC, iLISI, kBET``) with the CSV's descriptive columns
        (whitespace-stripped headers). The canonical code vocabulary,
        including ``PCR`` from the published tables, is
        :func:`known_metrics`.
    """
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "metric_full.csv")
    return raw.rename(columns={c: c.strip() for c in raw.columns})
