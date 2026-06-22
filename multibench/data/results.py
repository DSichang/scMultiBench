"""Load scib benchmark metric tables into a tidy long DataFrame.

v1 implements the scib clustering + batch loader. Other metric sets raise a
clear "not in v1" error (declared but not wired).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from . import catalog

# clustering variant -> companion filename for the per-(dataset,method) dir
_CLUSTERING_FILES = {
    "default": "metric.csv",
    "louvain": "metric_louvain.csv",
    "kmeans": "metric_kmeans.csv",
}
_CORRECTION_FILE = "metric_asw_iasw_if1.csv"


def _read_metric_csv(path: Path) -> pd.DataFrame:
    """Read a metric.csv (unnamed index + Value col) into long metric/value rows."""
    raw = pd.read_csv(path)
    raw.columns = ["metric", "value"] + list(raw.columns[2:])
    raw = raw[["metric", "value"]].copy()
    raw["metric"] = raw["metric"].map(catalog.canonical_metric)
    raw = raw.dropna(subset=["metric"])
    return raw


def load_results(
    category: str,
    task: str = "clustering",
    metric_set: str = "scib",
    dataset: str | None = None,
    method: str | None = None,
    metric: list[str] | str | None = None,
    clustering: str = "default",
    result_path: Path | str | None = None,
) -> pd.DataFrame:
    """Return a tidy long DataFrame: metric, value, method, dataset, category."""
    if metric_set != "scib":
        raise NotImplementedError(
            f"metric_set={metric_set!r} is declared but not wired in v1 "
            "(only 'scib' is implemented)."
        )
    if clustering not in _CLUSTERING_FILES:
        raise ValueError(
            f"unknown clustering {clustering!r}; valid: {sorted(_CLUSTERING_FILES)}"
        )

    base = Path(result_path) if result_path is not None else config.DEFAULT.result_path
    root = base / config.metric_set_dir(metric_set) / config.category_folder(category)
    if not root.exists():
        raise FileNotFoundError(f"no results at {root}")

    fname = _CLUSTERING_FILES[clustering]
    rows: list[pd.DataFrame] = []
    ds_dirs = [root / dataset] if dataset else sorted(p for p in root.iterdir() if p.is_dir())
    for ds_dir in ds_dirs:
        if not ds_dir.is_dir():
            continue
        for m_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            mfile = m_dir / fname
            if not mfile.exists():
                continue
            df = _read_metric_csv(mfile)
            # coalesce corrected ASW/iASW/iFI when present. The correction file
            # holds corrected values for the *default* clustering only; the
            # louvain/kmeans variant files already carry their own correct
            # ASW/iASW/iFI, so only coalesce for the default clustering.
            corr = m_dir / _CORRECTION_FILE
            if clustering == "default" and corr.exists():
                cdf = _read_metric_csv(corr).set_index("metric")["value"]
                df["value"] = df.apply(
                    lambda r: cdf.get(r["metric"], r["value"]), axis=1
                )
            df["method"] = catalog.canonical_id(m_dir.name)
            df["dataset"] = ds_dir.name
            df["category"] = category
            rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"no {fname} found under {root} (dataset={dataset!r})"
        )
    out = pd.concat(rows, ignore_index=True)

    if method is not None:
        out = out[out["method"] == catalog.canonical_id(method)]
    if metric is not None:
        wanted = [metric] if isinstance(metric, str) else list(metric)
        wanted = [catalog.canonical_metric(m) for m in wanted]
        out = out[out["metric"].isin(wanted)]
    return out.reset_index(drop=True)


def available_datasets(
    category: str,
    metric_set: str = "scib",
    result_path: Path | str | None = None,
) -> list[str]:
    """Dataset ids that have published ``metric_set`` results for a category.

    These are exactly the datasets ``load_results(category, dataset=...)`` can
    load — i.e. the subdirectories under the category's result folder. Returns
    ``[]`` if the category folder does not exist (e.g. ``mosaic``, which has no
    published metrics). Useful for discovering what is actually loadable before
    calling :func:`load_results`.
    """
    if metric_set != "scib":
        raise NotImplementedError(
            f"metric_set={metric_set!r} is not wired in v1 (only 'scib')."
        )
    base = Path(result_path) if result_path is not None else config.DEFAULT.result_path
    root = base / config.metric_set_dir(metric_set) / config.category_folder(category)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
