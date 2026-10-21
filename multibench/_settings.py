"""``mtb.config``: the settings every multibench function reads."""
from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path

from ._text import _and_list

#: The package checkout or install root (``multibench/..``).
_ROOT = Path(__file__).resolve().parent.parent
# datasets and the scripts clone sit next to the package in a checkout; a wheel
# install keeps them out of site-packages, in a per-user cache
_IN_REPO = (_ROOT / "pyproject.toml").is_file()
_CACHE = (Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
          / "multibench")
_BASE = _ROOT if _IN_REPO else _CACHE

#: The package's own tables (the stored results, the dataset and metric lists):
#: read by the package, not settings.
_RESULT_PATH = _ROOT / "multibench" / "result"
_FILES_PATH = _ROOT / "multibench" / "files"

#: The environment variables that set the ``mtb.config`` paths. Each wins over
#: the built-in default; a value assigned in Python wins over the variable.
ENVS_DIR_VAR = "MULTIBENCH_ENVS_DIR"
DATA_PATH_VAR = "MULTIBENCH_DATA_PATH"
REPO_PATH_VAR = "MULTIBENCH_REPO_PATH"

#: The Leiden backends ``leiden_flavor`` takes.
LEIDEN_FLAVORS = ("leidenalg", "igraph")


def _from_env(var: str) -> Path | None:
    """The path in environment variable ``var`` (``~`` expanded), or ``None`` if unset or empty."""
    value = os.environ.get(var)
    return Path(value).expanduser() if value else None


@functools.lru_cache(maxsize=1)
def _conda_envs_dir() -> Path | None:
    """The first writable envs dir of the conda/mamba on PATH, or ``None``; one
    ``info --json`` subprocess per process (tests that fake conda clear the cache)."""
    exe = shutil.which("mamba") or shutil.which("conda")
    if exe is None:
        return None
    from .engine.envs import _envs_dir
    return _envs_dir(exe)


def _default_envs_dir() -> Path:
    """``$MULTIBENCH_ENVS_DIR``, else the conda envs folder, else the cache."""
    return _from_env(ENVS_DIR_VAR) or _conda_envs_dir() or _CACHE / "envs"


def _path(value) -> Path | None:
    return None if value is None else Path(value).expanduser()


class Settings:
    """The paths and settings every multibench function reads.

    ``mtb.config`` is one of these: set a field to change it, for example
    ``mtb.config.data_path = "/scratch/data"``, or set
    ``MULTIBENCH_DATA_PATH``, ``MULTIBENCH_REPO_PATH`` and
    ``MULTIBENCH_ENVS_DIR`` in the shell. A value assigned in Python wins
    over the variable; ``None`` goes back to the default.
    ``print(mtb.config)`` shows every field; a misspelt field raises
    ``AttributeError``.

    Attributes
    ----------
    data_path : pathlib.Path
        Data root that holds the dataset folders; ``mtb.data.fetch``,
        ``mtb.scan`` and ``mtb.run`` use it. Default
        ``$MULTIBENCH_DATA_PATH``, else ``<base>/data``.
    repo_path : pathlib.Path
        Folder of the method scripts (``tools_scripts/``), fetched there on
        the first run when absent or empty. Default
        ``$MULTIBENCH_REPO_PATH``, else ``<base>/scMultiBench_ref``.
    envs_dir : pathlib.Path
        Where the method environments live (``<envs_dir>/<env>``). Default
        ``$MULTIBENCH_ENVS_DIR``, else the first writable envs folder of the
        conda on ``PATH``, else ``~/.cache/multibench/envs``.
    leiden_flavor : str
        Leiden backend of the clustering in ``mtb.evaluate``: ``"leidenalg"``
        (default, the article's and the stored results') or ``"igraph"``
        (faster).

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.config.data_path = "/scratch/data"
    >>> # before mtb.env.install(...)
    >>> mtb.config.envs_dir = "/scratch/envs"
    >>> print(mtb.config)

    See Also
    --------
    mtb.env.install : installs the method envs under ``envs_dir``.
    mtb.data.fetch : downloads demo datasets into ``data_path``.
    """

    __slots__ = ("_data_path", "_repo_path", "_envs_dir", "_leiden_flavor")

    #: The settings, in the order ``print(mtb.config)`` shows them.
    FIELDS = ("data_path", "repo_path", "envs_dir", "leiden_flavor")

    def __init__(self):
        for slot in self.__slots__:
            object.__setattr__(self, slot, None)

    @property
    def data_path(self) -> Path:
        return self._data_path or _from_env(DATA_PATH_VAR) or _BASE / "data"

    @data_path.setter
    def data_path(self, value):
        self._data_path = _path(value)

    @property
    def repo_path(self) -> Path:
        return self._repo_path or _from_env(REPO_PATH_VAR) or _BASE / "scMultiBench_ref"

    @repo_path.setter
    def repo_path(self, value):
        self._repo_path = _path(value)

    @property
    def envs_dir(self) -> Path:
        # the first default is kept: probing conda on every read would be slow
        if self._envs_dir is None:
            self._envs_dir = _default_envs_dir()
        return self._envs_dir

    @envs_dir.setter
    def envs_dir(self, value):
        self._envs_dir = _path(value)

    @property
    def leiden_flavor(self) -> str:
        return self._leiden_flavor or "leidenalg"

    @leiden_flavor.setter
    def leiden_flavor(self, value):
        if value is not None and value not in LEIDEN_FLAVORS:
            raise ValueError(f"leiden_flavor takes {_and_list(map(repr, LEIDEN_FLAVORS), 'or')}, "
                             f"not {value!r}.")
        self._leiden_flavor = value

    def __setattr__(self, name, value):
        # a typo such as data_dir= would otherwise be a field nothing reads
        if name not in self.FIELDS and name not in self.__slots__:
            raise AttributeError(f"mtb.config has no setting {name!r}. Its settings are "
                                 f"{_and_list(self.FIELDS)}.")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return ("mtb.config(" + ", ".join(f"{k}={getattr(self, k)!r}" for k in self.FIELDS)
                + ")")


config = Settings()
