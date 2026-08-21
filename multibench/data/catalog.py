"""Catalog: parse files/*.csv into typed tables with canonical names."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import config

__all__ = ["methods", "datasets", "metrics", "canonical_id", "canonical_metric",
           "PAPER_COLUMNS"]

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


def canonical_id(name: str) -> str:
    """Return the canonical method id for any known spelling."""
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # default: collapse separators to underscore, keep original casing token
    return re.sub(r"[ .]+", "_", name.strip())


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


def canonical_metric(code: str) -> str | None:
    """Canonicalize a raw metric short-code; None for blank/unknown-empty."""
    if code is None:
        return None
    key = str(code).strip().lower()
    if key == "" or key == "nan":
        return None
    return _METRIC_CANON.get(key, str(code).strip())


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
          kept as a duplicate column for one release (old callers read it);
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
    out = pd.DataFrame()
    out["dataset"] = raw[name_col].astype(str).str.strip()
    out["dataset name"] = out["dataset"]          # back-compat alias (one release)
    out["simulated"] = out["dataset"].str.upper().str.startswith("SD")

    # registry-derived columns: filled from the result tree at call time
    from . import results as _results
    cat_of: dict[str, list[str]] = {}
    for cat in ("cross", "diagonal", "mosaic", "vertical"):
        try:
            for ds in _results.available_datasets(cat, source="both"):
                cat_of.setdefault(ds, []).append(cat)
        except Exception:       # a broken/absent tree must not break the catalog
            continue
    out["category"] = out["dataset"].map(lambda d: ";".join(cat_of[d]) if d in cat_of else None)
    out["has_results"] = out["dataset"].isin(cat_of.keys())

    for col in PAPER_COLUMNS:
        out[col] = raw[col].values if col in raw.columns else pd.NA
    if category is not None:
        config.category_folder(category)
        out = out[out["category"].fillna("").str.split(";").map(lambda cs: category in cs)]
    return out.reset_index(drop=True)


def metrics(files_dir: Path | str | None = None) -> pd.DataFrame:
    """Return the metric details table."""
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "metric_full.csv")
    return raw.rename(columns={c: c.strip() for c in raw.columns})
