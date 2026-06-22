"""Catalog: parse files/*.csv into typed tables with canonical names."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import config

__all__ = ["methods", "datasets", "metrics", "canonical_id", "canonical_metric"]

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
    """Return the methods table with normalized columns and list-valued cats/tasks."""
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
    return out


def datasets(files_dir: Path | str | None = None) -> pd.DataFrame:
    """Return the datasets table with a derived `simulated` flag."""
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "dataset.csv")
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    raw = raw.dropna(how="all", axis=1).dropna(how="all", axis=0)
    name_col = "dataset name"
    raw["simulated"] = raw[name_col].astype(str).str.strip().str.upper().str.startswith("SD")
    return raw


def metrics(files_dir: Path | str | None = None) -> pd.DataFrame:
    """Return the metric details table."""
    if files_dir is None:
        files_dir = config.DEFAULT.files_path
    raw = pd.read_csv(Path(files_dir) / "metric_full.csv")
    return raw.rename(columns={c: c.strip() for c in raw.columns})
