"""Filesystem paths and on-disk token maps for multibench.

The benchmark's result tree uses space-named category folders and a singular
`scib_metric` top-level dir. Callers use clean tokens (e.g. "vertical"); this
module translates them to the real folder names.
"""
from __future__ import annotations

import functools
import os as _os
import shutil as _shutil
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Config", "DEFAULT"]


def __dir__() -> list[str]:
    """``dir(mtb.config)`` lists the public names (PEP 562).

    ``category_folder`` / ``metric_set_dir`` stay importable for the package's
    own modules but are internal token maps, not user-facing API, so tab
    completion does not advertise them.
    """
    return sorted(n for n in globals() if n in __all__ or n.startswith("__"))

# token -> on-disk space-named folder
_CATEGORY_FOLDERS = {
    "vertical": "vertical integration",
    "diagonal": "diagonal integration",
    "mosaic": "mosaic integration",
    "cross": "cross integration",
}

# metric-set token -> top-level result dir; only the scIB metric set exists
_METRIC_SET_DIRS = {
    "scib": "scib_metric",
}

_ROOT = Path(__file__).resolve().parent.parent
# Large client-side artefacts (datasets, the upstream clone) sit next to the
# package in a repository checkout or editable install. In a wheel install
# _ROOT is inside site-packages, which must not accumulate them, so they go to
# a per-user cache dir.
_IN_REPO = (_ROOT / "pyproject.toml").is_file()
_CACHE = (Path(_os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
          / "multibench")
_BASE = _ROOT if _IN_REPO else _CACHE

#: The environment variable that pins :attr:`Config.envs_dir`.
ENVS_DIR_VAR = "MULTIBENCH_ENVS_DIR"


@functools.lru_cache(maxsize=1)
def _conda_envs_dir() -> Path | None:
    """The first writable envs dir of the conda/mamba on PATH, or ``None``.

    One ``info --json`` subprocess per process (cached), run only when
    :attr:`Config.envs_dir` is first read without ``MULTIBENCH_ENVS_DIR`` set.
    Tests that fake or hide conda clear this cache.
    """
    exe = _shutil.which("mamba") or _shutil.which("conda")
    if exe is None:
        return None
    from .engine.envs import _envs_dir
    return _envs_dir(exe)


def _default_envs_dir() -> Path:
    """Resolve where method env prefixes live (see :attr:`Config.envs_dir`)."""
    override = _os.environ.get(ENVS_DIR_VAR)
    if override:
        return Path(override).expanduser()
    found = _conda_envs_dir()
    if found is not None:
        return found
    return _CACHE / "envs"


class _LazyEnvsDir:
    """Descriptor behind the ``envs_dir`` field: resolved on first read.

    A plain ``default_factory`` would run ``conda info --json`` every time a
    ``Config`` is built - including ``config.DEFAULT`` at import - on every
    host that has conda. The descriptor keeps the field settable
    (``cfg.envs_dir = Path(...)``) and runs the probe only when the directory
    is needed (``env_prefix``, ``install_packed``, the runner's prefix mode).
    """

    def __set_name__(self, owner, name):
        self._slot = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self          # the dataclass default: "unset", see __set__
        value = obj.__dict__.get(self._slot)
        if value is None:
            value = _default_envs_dir()
            obj.__dict__[self._slot] = value
        return value

    def __set__(self, obj, value):
        # dataclass __init__ assigns the class-level default, i.e. this very
        # descriptor, which means "not given": stay lazy
        obj.__dict__[self._slot] = None if isinstance(value, _LazyEnvsDir) else Path(value)


def category_folder(token: str) -> str:
    """Map a category token to its space-named result folder (internal).

    Parameters
    ----------
    token : str
        ``vertical`` / ``diagonal`` / ``mosaic`` / ``cross``.

    Returns
    -------
    str
        The on-disk folder name, e.g. ``"vertical integration"``.

    Raises
    ------
    ValueError
        Unknown token; the message lists the valid ones (this is the
        validator ``registry.check_category`` delegates to).
    """
    try:
        return _CATEGORY_FOLDERS[token]
    except KeyError:
        raise ValueError(
            f"unknown category {token!r}; valid: {sorted(_CATEGORY_FOLDERS)}"
        ) from None


def metric_set_dir(token: str) -> str:
    """Map a metric-set token to its top-level result dir name (internal).

    Parameters
    ----------
    token : str
        Only ``"scib"`` is wired.

    Returns
    -------
    str
        The directory name under the result root (``"scib_metric"``).

    Raises
    ------
    ValueError
        Unknown token, listing the valid ones.
    """
    try:
        return _METRIC_SET_DIRS[token]
    except KeyError:
        raise ValueError(
            f"unknown metric_set {token!r}; valid: {sorted(_METRIC_SET_DIRS)}"
        ) from None


@dataclass
class Config:
    """Resolved filesystem paths; override fields to point at custom locations.

    ``mtb.config.DEFAULT`` is the instance every function reads. Set its
    fields directly, or pass a path explicitly where a function takes
    ``data_path=`` / ``result_path=``.

    Attributes
    ----------
    result_path : pathlib.Path
        Result tables shipped with the package, read by ``mtb.load_results``.
        Default ``<package root>/multibench/result``.
    files_path : pathlib.Path
        Catalog CSVs read by ``mtb.catalog`` (``method.csv``, ``dataset.csv``,
        ``metric_full.csv``). Default ``<package root>/multibench/files``.
    repo_path : pathlib.Path
        Checkout holding the upstream ``tools_scripts/`` (the method scripts),
        cloned on first use when absent. Default ``<base>/scMultiBench_ref``.
    data_path : pathlib.Path
        Data root that holds the dataset folders: ``mtb.data.fetch`` puts
        datasets there, ``mtb.scan`` / ``mtb.run_all`` look for
        ``<data_path>/<dataset>/``. Default ``<base>/data``.
    leiden_flavor : str
        Leiden backend of the scIB clustering sweep in ``mtb.evaluate``:
        ``"igraph"`` (default, faster) or ``"leidenalg"`` (the backend of both
        stored sources).
    envs_dir : pathlib.Path
        Where the method environment prefixes live (``<envs_dir>/<env>``);
        resolved on first read (order in Notes).

    Examples
    --------
    >>> from pathlib import Path
    >>> import multibench as mtb
    >>> mtb.config.DEFAULT.data_path = Path("/scratch/data")    # a Path; a str fails in fetch / scan
    >>> mtb.config.DEFAULT.envs_dir = Path("/scratch/envs")     # before mtb.env.install(...)
    >>> mtb.config.DEFAULT.leiden_flavor = "leidenalg"          # the leidenalg backend, not igraph
    >>> cfg = mtb.config.Config(data_path=Path("/data/mine"))   # a separate instance

    Notes
    -----
    **Where ``<base>`` is.** The repository root in a checkout or editable
    install (``pyproject.toml`` next to the package), and the per-user cache
    ``~/.cache/multibench`` (``$XDG_CACHE_HOME`` honoured) for a wheel
    install, so ``site-packages`` never accumulates datasets or clones.

    **Assigning paths.** Assign ``pathlib.Path`` objects. Only ``envs_dir``
    converts a string on assignment; ``mtb.data.fetch``, ``mtb.scan`` and
    ``mtb.load_results`` use ``data_path`` / ``result_path`` as they are and
    fail on a plain string.

    **How ``envs_dir`` is resolved.** Lazily, on first read, from the first
    of:

    1. the ``MULTIBENCH_ENVS_DIR`` environment variable;
    2. the first writable envs directory of the conda/mamba found on PATH;
    3. ``~/.cache/multibench/envs`` (``$XDG_CACHE_HOME`` honoured).

    It is what ``mtb.env.install`` unpacks packed archives into and what the
    runner's prefix mode activates. The first read may run ``conda info
    --json`` (once per process, never at import); assigning a value skips
    the probe. It is settable like every other field.

    **Leiden backends.** ``"igraph"`` is scanpy's igraph implementation,
    several times faster; ``"leidenalg"`` is the backend both stored
    sources (published and re-run) were computed with.

    See Also
    --------
    mtb.env.install : provisions the method envs under ``envs_dir``.
    mtb.data.fetch : downloads reference datasets into ``data_path``.
    """

    result_path: Path = field(default_factory=lambda: _ROOT / "multibench" / "result")
    files_path: Path = field(default_factory=lambda: _ROOT / "multibench" / "files")
    repo_path: Path = field(default_factory=lambda: _BASE / "scMultiBench_ref")
    data_path: Path = field(default_factory=lambda: _BASE / "data")
    leiden_flavor: str = "igraph"
    envs_dir: Path = _LazyEnvsDir()


# module-level default instance; callers may replace its fields
DEFAULT = Config()


def ensure_repo(path=None):
    """Return a directory that contains ``tools_scripts/``, provisioning it if needed.

    Resolution order: the given (or configured) ``repo_path``; the package root
    itself (the merged-repository layout, where ``tools_scripts/`` sits next to
    ``multibench/``); otherwise a one-time shallow clone of the public
    scMultiBench repository into the configured location, so methods run on a
    fresh machine or Colab, where the package does not carry the upstream
    method scripts.
    """
    import subprocess
    from pathlib import Path as _P

    p = _P(path) if path else DEFAULT.repo_path
    if (p / "tools_scripts").is_dir():
        return p
    if (_ROOT / "tools_scripts").is_dir():
        return _ROOT
    if p.exists():
        # a directory without tools_scripts is most likely an interrupted
        # clone; refuse to guess and never delete a directory not created here
        raise RuntimeError(
            f"{p} exists but has no tools_scripts/ - remove it (or point "
            f"repo_path elsewhere) and the method scripts will be fetched "
            f"fresh")
    print(f"method scripts not found - fetching PYangLab/scMultiBench (once) into {p} ...",
          flush=True)
    part = p.with_name(p.name + ".partial")
    import shutil as _sh
    _sh.rmtree(part, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/PYangLab/scMultiBench.git", str(part)],
                   check=True)
    part.rename(p)
    return p
