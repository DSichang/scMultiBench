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


def ensure_repo(path=None):
    """Return a directory that contains ``tools_scripts/``, provisioning it if needed.

    Resolution order: the given (or configured) ``repo_path``; the package root
    itself (the merged-repository layout, where ``tools_scripts/`` sits next to
    ``multibench/``); otherwise a one-time shallow clone of the public
    scMultiBench repository into the configured location - which is what makes
    method execution work on a fresh machine or Colab, where the wrapper's
    clone does not carry the 3 GB of upstream method scripts.
    """
    import subprocess
    from pathlib import Path as _P

    p = _P(path) if path else DEFAULT.repo_path
    if (p / "tools_scripts").is_dir():
        return p
    if (_ROOT / "tools_scripts").is_dir():
        return _ROOT
    print(f"method scripts not found - fetching PYangLab/scMultiBench (once) into {p} ...",
          flush=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/PYangLab/scMultiBench.git", str(p)],
                   check=True)
    return p
