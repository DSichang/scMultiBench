"""The paths and settings every multibench function reads.

Change a field of ``mtb.config.DEFAULT`` to use your own data folder, env
folder or Leiden backend, for example
``mtb.config.DEFAULT.data_path = "/scratch/data"``. ``help(mtb.config.Config)``
lists the fields; ``multibench config`` prints each path and where it came
from.
"""
from __future__ import annotations

import functools
import os as _os
import re as _re
import shutil as _shutil
import socket as _socket
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

#: The environment variables that set :class:`Config` paths. Each wins over
#: the built-in default; a value assigned in Python wins over the variable.
ENVS_DIR_VAR = "MULTIBENCH_ENVS_DIR"
DATA_PATH_VAR = "MULTIBENCH_DATA_PATH"
REPO_PATH_VAR = "MULTIBENCH_REPO_PATH"
#: The commit or tag of the method scripts a fetch checks out; unset = the
#: repository's default branch. ``multibench fetch --scripts --ref`` wins.
SCRIPTS_REF_VAR = "MULTIBENCH_SCRIPTS_REF"
#: Where the method scripts are fetched from.
SCRIPTS_URL = "https://github.com/PYangLab/scMultiBench.git"

#: ``True`` while ``multibench.cli.main`` runs a command. Messages built with
#: :func:`hint` then name the shell command instead of the Python call.
_CLI = False


def hint(py: str, cli: str) -> str:
    """The ``cli`` spelling while the command line runs a command, else ``py`` (internal).

    Messages that tell the user what to call next use it, e.g.
    ``hint("mtb.env.doctor()", "multibench env doctor")``.
    """
    return cli if _CLI else py


def _from_env(var: str) -> Path | None:
    """The path in environment variable ``var`` (``~`` expanded), or ``None`` when unset or empty."""
    value = _os.environ.get(var)
    return Path(value).expanduser() if value else None


def _default_data_path() -> Path:
    """``$MULTIBENCH_DATA_PATH``, else ``<base>/data`` (see :attr:`Config.data_path`)."""
    return _from_env(DATA_PATH_VAR) or _BASE / "data"


def _default_repo_path() -> Path:
    """``$MULTIBENCH_REPO_PATH``, else ``<base>/scMultiBench_ref`` (see :attr:`Config.repo_path`)."""
    return _from_env(REPO_PATH_VAR) or _BASE / "scMultiBench_ref"


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
    override = _from_env(ENVS_DIR_VAR)
    if override is not None:
        return override
    found = _conda_envs_dir()
    if found is not None:
        return found
    return _CACHE / "envs"


class _LazyPath:
    """Descriptor behind a path field whose default is resolved on read.

    ``envs_dir`` uses it with ``cache=True``: a plain ``default_factory``
    would run ``conda info --json`` every time a ``Config`` is built,
    including ``config.DEFAULT`` at import, so the probe runs on the first
    read only and its answer is kept. ``data_path`` / ``repo_path`` use
    ``cache=False``: their environment variable is read again on every read
    until a value is assigned.

    The field stays settable (``cfg.data_path = "/scratch/data"``); a string
    is converted to ``pathlib.Path`` and ``None`` returns the field to its
    default.
    """

    def __init__(self, resolve, *, cache: bool):
        self._resolve = resolve
        self._cache = cache

    def __set_name__(self, owner, name):
        self._slot = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self          # the dataclass default: "unset", see __set__
        value = obj.__dict__.get(self._slot)
        if value is None:
            value = self._resolve()
            if self._cache:
                obj.__dict__[self._slot] = value
        return value

    def __set__(self, obj, value):
        # dataclass __init__ assigns the class-level default, i.e. this very
        # descriptor, which means "not given": stay lazy
        unset = value is None or isinstance(value, _LazyPath)
        obj.__dict__[self._slot] = None if unset else Path(value).expanduser()


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
        Only ``"scib"`` is available.

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
        cloned on first use when absent. Default ``$MULTIBENCH_REPO_PATH``,
        else ``<base>/scMultiBench_ref``.
    data_path : pathlib.Path
        Data root that holds the dataset folders; ``mtb.data.fetch``,
        ``mtb.scan`` and ``mtb.run_all`` use it. Default
        ``$MULTIBENCH_DATA_PATH``, else ``<base>/data``.
    leiden_flavor : str
        Leiden backend of the scIB clustering sweep in ``mtb.evaluate``:
        ``"igraph"`` (default, faster) or ``"leidenalg"`` (the backend of both
        stored sources).
    envs_dir : pathlib.Path
        Where the method environment prefixes live (``<envs_dir>/<env>``);
        resolved on first read (order in Notes).

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.config.DEFAULT.data_path = "/scratch/data"
    >>> # before mtb.env.install(...)
    >>> mtb.config.DEFAULT.envs_dir = "/scratch/envs"
    >>> # the backend of both stored sources, not igraph
    >>> mtb.config.DEFAULT.leiden_flavor = "leidenalg"
    >>> cfg = mtb.config.Config(data_path="/data/mine")  # a separate instance

    Notes
    -----
    **Environment variables.** Set them in the shell, a job script or a
    module file, and every process that sees them uses the paths. A value
    assigned in Python wins over the variable. ``multibench config`` prints
    each resolved path and where it came from. The variables and the fields
    they set:

    ```text
    MULTIBENCH_DATA_PATH   data_path
    MULTIBENCH_REPO_PATH   repo_path
    MULTIBENCH_ENVS_DIR    envs_dir
    ```

    **Where ``<base>`` is.** The repository root in a checkout or editable
    install (``pyproject.toml`` next to the package). For a wheel install it
    is the per-user cache ``~/.cache/multibench`` (``$XDG_CACHE_HOME``
    honoured).

    **Assigning paths.** ``data_path``, ``repo_path`` and ``envs_dir``
    accept a string and store a ``pathlib.Path``; assigning ``None`` returns
    the field to its default. Assign ``result_path`` and ``files_path`` as
    ``pathlib.Path`` objects: ``mtb.load_results`` and ``mtb.catalog`` use
    them as they are.

    **How ``envs_dir`` is resolved.** Lazily, on first read, from the first
    of:

    1. the ``MULTIBENCH_ENVS_DIR`` environment variable;
    2. the first writable envs directory of the conda/mamba found on PATH;
    3. ``~/.cache/multibench/envs`` (``$XDG_CACHE_HOME`` honoured).

    It is what ``mtb.env.install`` unpacks packed archives into and what the
    runner's prefix mode activates. The first read may run ``conda info
    --json`` (once per process, not at import); assigning a value skips
    the probe.

    **Leiden backends.** ``"igraph"`` is scanpy's igraph implementation,
    several times faster; ``"leidenalg"`` is the backend both stored
    sources (published and re-run) were computed with.

    **Method scripts.** The first run fetches them from GitHub into
    ``repo_path`` (``multibench fetch --scripts`` does it ahead). Set
    ``MULTIBENCH_SCRIPTS_REF`` to a commit or tag to fetch that version
    instead of the default branch. ``multibench config`` and every run record
    show the commit in use (``scripts_commit``).

    See Also
    --------
    mtb.env.install : provisions the method envs under ``envs_dir``.
    mtb.data.fetch : downloads reference datasets into ``data_path``.
    """

    result_path: Path = field(default_factory=lambda: _ROOT / "multibench" / "result")
    files_path: Path = field(default_factory=lambda: _ROOT / "multibench" / "files")
    repo_path: Path = _LazyPath(_default_repo_path, cache=False)
    data_path: Path = _LazyPath(_default_data_path, cache=False)
    leiden_flavor: str = "igraph"
    envs_dir: Path = _LazyPath(_default_envs_dir, cache=True)


# module-level default instance; callers may replace its fields
DEFAULT = Config()


def _base_source() -> str:
    """Where ``<base>`` is, in words (see the Config Notes)."""
    return "the repository checkout" if _IN_REPO else f"the user cache {_CACHE}"


def scripts_present(cfg: Config | None = None) -> bool:
    """Whether :func:`ensure_repo` would find the method scripts without a download (internal)."""
    return _scripts_checkout(cfg) is not None


def _scripts_checkout(cfg: Config | None = None) -> Path | None:
    """The folder holding ``tools_scripts/`` that :func:`ensure_repo` would use, or
    ``None`` when the scripts are not on this machine (no fetch)."""
    cfg = DEFAULT if cfg is None else cfg
    for p in (Path(cfg.repo_path), _ROOT):
        if (p / "tools_scripts").is_dir():
            return p
    return None


def _git_dir(repo: Path) -> Path | None:
    """``<repo>/.git`` (a worktree's ``.git`` file followed), or ``None``."""
    git = repo / ".git"
    if git.is_file():
        text = git.read_text().strip()
        if not text.startswith("gitdir:"):
            return None
        git = (repo / text.split(":", 1)[1].strip()).resolve()
    return git if git.is_dir() else None


def scripts_commit(repo=None) -> str | None:
    """The commit of the method-scripts checkout, or ``None`` (internal).

    ``repo`` = the folder holding ``tools_scripts/``; ``None`` = the one
    :func:`ensure_repo` would use, without fetching. The answer is
    ``git rev-parse HEAD``'s, read from the checkout's own ``.git`` (no git
    process, so it also works where git is absent). ``None`` when the folder
    is not the root of a git checkout, e.g. a copy of the scripts.
    """
    repo = _scripts_checkout() if repo is None else Path(repo)
    if repo is None:
        return None
    try:
        git = _git_dir(repo)
        if git is None:
            return None
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head.lower() if _re.fullmatch(r"[0-9a-fA-F]{40}", head) else None
        ref = head[4:].strip()
        common = git
        if (git / "commondir").is_file():                  # a linked worktree
            common = (git / (git / "commondir").read_text().strip()).resolve()
        for base in dict.fromkeys((git, common)):
            if (base / ref).is_file():
                return (base / ref).read_text().strip().lower()
        for base in dict.fromkeys((git, common)):
            if (base / "packed-refs").is_file():
                for line in (base / "packed-refs").read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0].lower()
    except OSError:
        return None
    return None


#: A file in the checkout's ``.git`` that records the ref a fetch was asked for.
_REF_RECORD = "multibench-scripts-ref"


def _check_ref(repo: Path, ref: str) -> None:
    """``RuntimeError`` when the scripts in ``repo`` are not at ``ref`` (internal).

    Accepted: ``ref`` is the checkout's commit or a prefix of it (7+ hex
    digits), the ref this package fetched there, or what ``git rev-parse``
    resolves it to. A folder that is not a git checkout cannot be checked and
    is accepted.
    """
    import subprocess
    head = scripts_commit(repo)
    if head is None or (_re.fullmatch(r"[0-9a-fA-F]{7,40}", ref)
                        and head.startswith(ref.lower())):
        return
    git = _git_dir(repo)
    try:
        if (git / _REF_RECORD).read_text().strip() == ref:
            return
    except OSError:
        pass
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet",
                              f"{ref}^{{commit}}"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip().lower() == head:
            return
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError(
        f"the method scripts in {repo} are at commit {head[:12]}, not {ref!r} "
        f"({SCRIPTS_REF_VAR} or --ref). Remove that folder to fetch {ref!r}, or "
        f"set {REPO_PATH_VAR} to a checkout of it.")


def run_provenance(env: str | None, repo=None) -> dict:
    """``scripts_commit``, ``env_flavor`` and ``hostname`` for a run record (internal).

    ``env_flavor`` is ``'cpu'`` / ``'gpu'`` from the environment's install
    record (``mtb.env.install`` of a packed archive), else ``'unknown'``
    (built from a lockfile, not installed, or ``env=None``).
    """
    flavor = None
    if env:
        try:
            from .engine import envs
            flavor = envs.installed_flavor(env)
        except Exception:  # noqa: BLE001 - a record field must never fail a run
            flavor = None
    return {"scripts_commit": scripts_commit(repo), "env_flavor": flavor or "unknown",
            "hostname": _socket.gethostname()}


def _sources(cfg: Config | None = None) -> list[dict]:
    """Each setting of ``cfg`` with its resolved value and where it came from (internal).

    Backs ``multibench config``. Returns one dict per setting, in the order
    ``data_path``, ``envs_dir``, ``repo_path``, ``scripts_commit`` (the
    commit of the method scripts in ``repo_path``), ``result_path``,
    ``leiden_flavor``, with the keys ``name``, ``value`` and ``source``.
    """
    cfg = DEFAULT if cfg is None else cfg
    rows = []

    def _var_or_default(name: str, var: str, default_text: str) -> str:
        if cfg.__dict__.get("_" + name) is not None:
            return "set in Python"
        if _from_env(var) is not None:
            return f"environment variable {var}"
        return default_text

    rows.append({"name": "data_path", "value": cfg.data_path,
                 "source": _var_or_default("data_path", DATA_PATH_VAR,
                                           f"default <base>/data; <base> = {_base_source()}")})
    value, var_dir = cfg.envs_dir, _from_env(ENVS_DIR_VAR)
    conda_dir = None if var_dir is not None else _conda_envs_dir()
    if var_dir is not None and value == var_dir:
        envs_src = f"environment variable {ENVS_DIR_VAR}"
    elif conda_dir is not None and value == conda_dir:
        envs_src = "the envs directory of the conda/mamba on PATH"
    elif var_dir is None and conda_dir is None and value == _CACHE / "envs":
        envs_src = "default: the user cache, no conda/mamba on PATH"
    else:
        envs_src = "set in Python"
    rows.append({"name": "envs_dir", "value": cfg.envs_dir, "source": envs_src})
    repo_src = _var_or_default("repo_path", REPO_PATH_VAR,
                               f"default <base>/scMultiBench_ref; <base> = {_base_source()}")
    rows.append({"name": "repo_path", "value": cfg.repo_path, "source": repo_src})
    checkout = _scripts_checkout(cfg)
    if checkout is None:
        commit, commit_src = "not fetched", "multibench fetch --scripts fetches the method scripts"
    else:
        sha = scripts_commit(checkout)
        commit = sha or "unknown"
        where = Path(checkout) / "tools_scripts"
        commit_src = (f"the method scripts in {where}" if sha else
                      f"{where} is not a git checkout")
    rows.append({"name": "scripts_commit", "value": commit, "source": commit_src})
    rows.append({"name": "result_path", "value": cfg.result_path,
                 "source": ("default: the tables shipped with the package"
                            if cfg.result_path == _ROOT / "multibench" / "result"
                            else "set in Python")})
    rows.append({"name": "leiden_flavor", "value": cfg.leiden_flavor,
                 "source": "default" if cfg.leiden_flavor == "igraph" else "set in Python"})
    return rows


def scripts_line(repo) -> str:
    """``'method scripts: <repo>/tools_scripts at <commit>'`` (internal).

    Printed by ``multibench fetch --scripts``; ``multibench config`` shows the
    commit as its ``scripts_commit`` row.
    """
    sha = scripts_commit(repo)
    where = Path(repo) / "tools_scripts"
    return (f"method scripts: {where} at {sha}" if sha else
            f"method scripts: {where} (not a git checkout, commit unknown)")


def ensure_repo(path=None, ref=None):
    """Return a directory that contains ``tools_scripts/``, provisioning it if needed.

    Resolution order: the given (or configured) ``repo_path``; the package root
    itself (the merged-repository layout, where ``tools_scripts/`` sits next to
    ``multibench/``); otherwise a one-time shallow fetch of the public
    scMultiBench repository into the configured location, so methods run on a
    fresh machine or Colab, where the package does not carry the upstream
    method scripts.

    ``ref`` (default ``$MULTIBENCH_SCRIPTS_REF``) is a commit or tag: a fetch
    checks it out instead of the default branch, and scripts already present
    must be at it (``RuntimeError`` otherwise). A failed fetch raises
    ``RuntimeError`` naming the offline route, and leaves nothing behind.
    """
    import subprocess
    import shutil as _sh

    ref = ref or _os.environ.get(SCRIPTS_REF_VAR) or None
    p = Path(path) if path else DEFAULT.repo_path
    for have in (p, _ROOT):
        if (have / "tools_scripts").is_dir():
            if ref:
                _check_ref(have, ref)
            return have
    if p.exists():
        # a directory without tools_scripts is most likely an interrupted
        # clone; refuse to guess and never delete a directory not created here
        raise RuntimeError(
            f"{p} exists but has no tools_scripts/ - remove it (or point "
            f"repo_path or {REPO_PATH_VAR} elsewhere) and the method scripts "
            f"will be fetched fresh")
    at = f" at {ref}" if ref else ""
    print(f"method scripts not found - fetching PYangLab/scMultiBench{at} (once) into "
          f"{p} ...", flush=True)
    part = p.with_name(p.name + ".partial")
    _sh.rmtree(part, ignore_errors=True)
    try:
        if ref is None:
            subprocess.run(["git", "clone", "--depth", "1", SCRIPTS_URL, str(part)],
                           check=True)
        else:
            # clone --branch takes branches and tags only; fetch also takes a commit
            subprocess.run(["git", "init", "-q", str(part)], check=True)
            subprocess.run(["git", "-C", str(part), "fetch", "-q", "--depth", "1",
                            SCRIPTS_URL, ref], check=True)
            subprocess.run(["git", "-C", str(part), "checkout", "-q", "--detach",
                            "FETCH_HEAD"], check=True)
            if (part / ".git").is_dir():
                (part / ".git" / _REF_RECORD).write_text(ref + "\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        _sh.rmtree(part, ignore_errors=True)
        if isinstance(e, FileNotFoundError):
            first = "git is not installed; it is needed to fetch the method scripts."
        elif ref:
            first = (f"could not fetch the method scripts at {ref!r} from github.com "
                     f"(no network, or no such commit or tag).")
        else:
            first = "could not reach github.com to fetch the method scripts."
        raise RuntimeError(
            f"{first} On a host without network, copy a scripts checkout "
            f"(`multibench fetch --scripts` on a connected machine makes one) and "
            f"set {REPO_PATH_VAR}.") from e
    part.rename(p)
    return p
