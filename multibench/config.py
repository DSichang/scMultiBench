"""Filesystem paths and on-disk token maps for multibench.

The benchmark's result tree uses space-named category folders and a singular
`scib_metric` top-level dir. Callers use clean tokens (e.g. "vertical"); this
module translates them to the real folder names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Config", "DEFAULT", "category_folder", "metric_set_dir"]

# token -> on-disk space-named folder
_CATEGORY_FOLDERS = {
    "vertical": "vertical integration",
    "diagonal": "diagonal integration",
    "mosaic": "mosaic integration",
    "cross": "cross integration",
}

# metric-set token -> top-level result dir. Only "scib" is wired in v1
# (load_results raises NotImplementedError otherwise); other metric sets are not
# yet exposed here rather than advertising tokens with no working loader.
_METRIC_SET_DIRS = {
    "scib": "scib_metric",
}

_ROOT = Path(__file__).resolve().parent.parent  # <ROOT>


def category_folder(token: str) -> str:
    """Map a category token to its space-named result folder."""
    try:
        return _CATEGORY_FOLDERS[token]
    except KeyError:
        raise ValueError(
            f"unknown category {token!r}; valid: {sorted(_CATEGORY_FOLDERS)}"
        ) from None


def metric_set_dir(token: str) -> str:
    """Map a metric-set token to its top-level result dir name."""
    try:
        return _METRIC_SET_DIRS[token]
    except KeyError:
        raise ValueError(
            f"unknown metric_set {token!r}; valid: {sorted(_METRIC_SET_DIRS)}"
        ) from None


@dataclass
class Config:
    """Resolved paths. Override fields to point at custom data locations."""

    result_path: Path = field(default_factory=lambda: _ROOT / "multibench" / "result")
    files_path: Path = field(default_factory=lambda: _ROOT / "multibench" / "files")
    repo_path: Path = field(default_factory=lambda: _ROOT / "scMultiBench_ref")
    data_path: Path = field(default_factory=lambda: _ROOT / "data")


# module-level default instance; callers may replace its fields
DEFAULT = Config()
