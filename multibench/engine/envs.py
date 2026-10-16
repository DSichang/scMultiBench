"""Method environments: check, plan and install the conda envs the methods run in.

Methods whose dependencies conflict run in separate envs; methods that fit
together share one. ``mtb.method_info(m)['env']`` names a method's env.

- ``mtb.env.plan(methods=[...])`` lists the envs a selection of methods
  needs.
- ``mtb.env.install([...], dry_run=False)`` installs them: a prebuilt
  archive when one exists, else a build from the pinned package list.
- ``mtb.env.status`` and ``mtb.env.doctor`` check the installed envs.
- ``mtb.env.recipe(method)`` returns the hand-written recipe of one env.

The envs are Linux-only.
"""
from __future__ import annotations

import functools
import json as _json
import platform as _platform
import shutil
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from .. import config
from . import registry

__all__ = ["status", "plan", "install", "doctor", "recipe"]


def __dir__() -> list[str]:
    """``dir(mtb.env)`` lists :data:`__all__` and the dunders (PEP 562)."""
    return sorted(n for n in globals() if n in __all__ or n.startswith("__"))


# --- host platform --------------------------------------------------------
def host_platform_problem() -> str | None:
    """Why method environments cannot be built on this host, or ``None``.

    The archives and lockfiles are linux-64: elsewhere a build fails after a
    multi-GB download, or in the solver.

    Returns
    -------
    str or None
        ``None`` on Linux (WSL included); else a sentence ending
        ``"this host is darwin/arm64"``.
    """
    if sys.platform == "linux":
        return None
    return (f"method environments are linux-64 conda envs (packed archives + "
            f"lockfiles); this host is {sys.platform}/{_platform.machine() or '?'}")


def _host_of(problem: str) -> str:
    """``'darwin/arm64'`` out of a :func:`host_platform_problem` text, so a
    patched text and the real one name the same host."""
    marker = "this host is "
    if marker in problem:
        return problem.rsplit(marker, 1)[1].strip().rstrip(".")
    return f"{sys.platform}/{_platform.machine() or '?'}"


def _host_system_name(problem: str) -> str:
    """The operating system of a :func:`host_platform_problem` text, for
    users: ``'macOS'``, ``'Windows'`` or the platform name."""
    system = _host_of(problem).split("/", 1)[0]
    return {"darwin": "macOS", "win32": "Windows", "cygwin": "Windows"}.get(system, system)


def linux_only_text(problem: str) -> str:
    """``"Method environments run only on Linux, and this computer runs macOS."``"""
    return (f"Method environments run only on Linux, and this computer runs "
            f"{_host_system_name(problem)}.")


def _require_linux(force: bool) -> None:
    """Raise ``RuntimeError`` before any download or build on a non-Linux
    host, unless ``force``."""
    problem = None if force else host_platform_problem()
    if problem:
        raise RuntimeError(
            f"{linux_only_text(problem)} Run the install on a Linux machine. "
            f"{config.hint('force=True', '--force')} tries anyway.")


#: The first sentence of the install error without conda (the docs quote it).
NO_CONDA = "Conda is not installed on this computer."


#: What the ``difficulty`` tag of ``env_specs.yaml`` means: how hard the env
#: is to build from its recipe, not how well the method works.
DIFFICULTY = {
    "easy": "modern python/torch stack; builds from the lockfile",
    "old-scvi": "pins an old scvi-tools (<0.20) / old anndata - needs its own env, "
                "cannot share the modern torch env",
    "old-tensorflow": "pins TensorFlow 1.x/2.4 - needs its own env with matching CUDA",
    "R": "an R env (Seurat/MOFA2/rliger ...); R packages installed by "
         "install.packages() are restored by the env's post-install script",
    "verified": "env built from the lockfile on a fresh machine and the method "
                "ran end-to-end on its reference dataset",
    "blocked-script": "the env builds, but the upstream script cannot run "
                      "unmodified from the public checkout; it needs extra files "
                      "(the method's setup_hint and scan's files_reason say which)",
    "unknown": "no recipe declared for the method",
}

#: The ``*`` suffix ``env status`` appends to the difficulty tag.
VERIFIED_STAR = ("* = verified_working: the env ran the method end-to-end on "
                 "its reference dataset")

#: The marks of the ``env status`` and ``env doctor`` listings.
MARK_LEGEND = ("[x]=installed  [L]=missing, lockfile ready (run multibench env "
               "install --run)  [!]=missing, no lockfile")


def env_mark(exists: bool, has_lock: bool) -> str:
    """The :data:`MARK_LEGEND` mark of an env row: ``"x"``, ``"L"`` or ``"!"``."""
    return "x" if exists else ("L" if has_lock else "!")


_SIZES_JSON = Path(__file__).resolve().parent / "packed_sizes.json"


@functools.lru_cache(maxsize=1)
def packed_sizes() -> dict:
    """Byte sizes of the published packed archives, per env.

    Read from the shipped ``engine/packed_sizes.json`` (written by
    ``tools/packed_sizes.py``); no request is made at runtime.

    Returns
    -------
    dict
        ``{env: {"archive_bytes": int | None, "unpacked_bytes": int | None}}``;
        ``None`` means "not measured yet". An unpacked size below the archive
        size cannot be right and reads ``None``.
    """
    data = _json.loads(_SIZES_JSON.read_text())
    out = {}
    for k, v in data.items():
        if k.startswith("_"):                       # metadata
            continue
        archive, unpacked = v.get("archive_bytes"), v.get("unpacked_bytes")
        if (isinstance(archive, (int, float)) and isinstance(unpacked, (int, float))
                and unpacked < archive):
            v = {**v, "unpacked_bytes": None}
        out[k] = v
    return out


def _gb(n) -> str:
    """``3215645570`` -> ``"3.2 GB"``; ``None`` / non-numeric -> ``"?"``."""
    try:
        if n is None:
            return "?"
        return f"{float(n) / 1e9:.1f} GB"
    except (TypeError, ValueError):
        return "?"


def _as_frame(rows: list[dict], as_frame: bool):
    """``rows`` as given, or as a ``pandas.DataFrame`` when ``as_frame``."""
    if not as_frame:
        return rows
    import pandas as pd
    return pd.DataFrame(rows)


# --- archive flavours (CPU-only vs CUDA builds) ----------------------------
#: The accepted ``flavor=`` values of :func:`install_packed` / :func:`install`.
FLAVORS = ("auto", "cpu", "gpu")

#: The file in an unpacked prefix that names the archive it came from.
FLAVOR_FILE = ".multibench_flavor"

#: Present when an NVIDIA driver is loaded, even without ``nvidia-smi`` on PATH.
_NVIDIA_PROC = Path("/proc/driver/nvidia/version")


def check_flavor(flavor) -> str:
    """``flavor`` when it is one of :data:`FLAVORS`.

    Returns
    -------
    str
        The value as given.

    Raises
    ------
    ValueError
        Any other value; the message lists the accepted ones.
    """
    if flavor not in FLAVORS:
        raise ValueError(f"flavor={flavor!r}: choose one of "
                         + ", ".join(repr(f) for f in FLAVORS))
    return flavor


@functools.lru_cache(maxsize=1)
def host_has_gpu() -> bool:
    """Is an NVIDIA GPU visible on this host?

    Returns
    -------
    bool
        ``True`` when a GPU is visible; ``False`` on a CPU-only host.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.host_has_gpu()

    Notes
    -----
    **Detection.** ``True`` when ``nvidia-smi -L`` runs and lists at least one
    device, or when ``/proc/driver/nvidia/version`` exists (driver loaded, tool
    not on ``PATH``). A torch install, ``CUDA_VISIBLE_DEVICES`` or a CUDA
    library on disk do not count.

    **Caching.** The result is cached per process;
    ``mtb.env.host_has_gpu.cache_clear()`` probes again.
    """
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run([smi, "-L"], capture_output=True, text=True,
                                  timeout=30)
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None and proc.returncode == 0 and any(
                line.strip() for line in proc.stdout.splitlines()):
            return True
    return _NVIDIA_PROC.exists()


def resolve_flavor(flavor: str = "auto") -> str:
    """``'cpu'`` or ``'gpu'`` for a ``flavor=`` value.

    Returns
    -------
    str
        The value as given; for ``'auto'``, ``'gpu'`` when :func:`host_has_gpu`,
        else ``'cpu'``.
    """
    check_flavor(flavor)
    if flavor == "auto":
        return "gpu" if host_has_gpu() else "cpu"
    return flavor


def archive_key(env: str, flavor: str) -> str:
    """The archive key of an env: ``'<env>-cpu'`` for ``'cpu'``, else ``'<env>'``."""
    return f"{env}-cpu" if flavor == "cpu" else env


def archive_for(env: str, flavor: str = "auto", *, manifest: dict | None = None,
                sizes: dict | None = None) -> tuple[str, str]:
    """Which published archive serves ``env`` for a flavour.

    ``manifest`` and ``sizes`` stand in for :func:`packed_manifest` and
    :func:`packed_sizes`.

    Returns
    -------
    tuple of (str, str)
        ``('<env>-cpu', 'cpu')`` when a CPU build is wanted and published (in
        the manifest with a measured size; a ``null`` size marks an archive
        not uploaded yet), else ``('<env>', 'gpu')``, which every env has.
    """
    wanted = resolve_flavor(flavor)
    if wanted == "cpu":
        key = archive_key(env, "cpu")
        manifest = packed_manifest() if manifest is None else manifest
        sizes = packed_sizes() if sizes is None else sizes
        if key in manifest and (sizes.get(key) or {}).get("archive_bytes") is not None:
            return key, "cpu"
    return env, "gpu"


def _single_build(env: str, *, manifest: dict | None = None) -> bool:
    """``True`` when ``env`` has one archive for CPU and GPU hosts."""
    manifest = packed_manifest() if manifest is None else manifest
    return archive_key(env, "cpu") not in manifest


def _cpu_fallback_warning(env: str, manifest: dict, sizes: dict) -> str:
    """The warning for a CPU request that installs the ``'<env>'`` archive."""
    size = _gb((sizes.get(env) or {}).get("archive_bytes"))
    if _single_build(env, manifest=manifest):
        return (f"{env} has a single build (the same archive for CPU and GPU hosts); "
                f"installing it ({size})")
    return (f"no CPU archive for {env}; installing the GPU build ({size}) - the CPU "
            f"archive {archive_key(env, 'cpu')} is not published yet")


def auto_flavor_note(flavor: str, *, planning: bool = False) -> str | None:
    """The stderr note for ``flavor='auto'`` on a host without a GPU, else
    ``None``: pass ``'gpu'`` when the jobs run on GPU nodes. ``planning``
    returns the advice alone (``env plan`` already names the CPU builds)."""
    if flavor != "auto" or resolve_flavor(flavor) != "cpu":
        return None
    gpu = config.hint("flavor='gpu'", "--flavor gpu")
    if planning:
        return f"# for jobs on GPU nodes, pass {gpu}"
    return (f"# installing CPU builds (no NVIDIA GPU on this host); if the jobs run on GPU "
            f"nodes, pass {gpu}")


def installed_flavor(env: str, conda: str | None = None) -> str | None:
    """Which archive flavour an installed env came from.

    Returns
    -------
    str or None
        ``'cpu'``, ``'gpu'`` or ``'single'`` from :data:`FLAVOR_FILE` (an env
        with a single build reads ``'single'``); ``None`` when the env is not
        installed or has no record, or the record is unreadable or holds
        anything else.
    """
    prefix = env_prefix(env, conda)
    if prefix is None:
        return None
    try:
        words = (prefix / FLAVOR_FILE).read_text().split()
    except OSError:
        return None
    word = words[0] if words and words[0] in ("cpu", "gpu", "single") else None
    if word in ("gpu", "single"):
        return "single" if _single_build(env) else "gpu"
    return word


_GROUPS_YAML = Path(__file__).resolve().parent / "env_groups.yaml"
# per-env lockfiles (`conda env export --no-builds`)
_LOCKS_DIR = Path(__file__).resolve().parent / "env_locks"


# --- recipes ---------------------------------------------------------------
def recipe(method: str) -> dict:
    """Return the hand-written environment recipe of a method.

    The readable alternative to the lockfile: which Python, conda and pip
    packages the method's env needs, with caveats.

    Parameters
    ----------
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.list_methods()``.

    Returns
    -------
    dict
        The recipe as a dict; ``{}`` when the method declares none. Read
        ``conda_packages``, ``pip_packages`` and ``caveats``; all keys are
        listed in Notes.

    Raises
    ------
    KeyError
        Unknown method id; the message suggests a close match, if any.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.recipe("Matilda")["conda_packages"]
    >>> mtb.env.recipe("Matilda").get("caveats")

    Notes
    -----
    **Keys.**

    - ``python_version`` - the Python pin of the create line.
    - ``conda_channels`` / ``conda_packages`` - the conda create line.
    - ``pip_packages`` / ``pip_git`` - installed with pip afterwards
      (``pip_git`` from git URLs).
    - ``package_source`` - where the method's own code comes from (PyPI,
      CRAN, vendored in ``tools_scripts/`` ...).
    - ``difficulty`` / ``verified_working`` - the same values as in
      ``mtb.env.status``.
    - ``caveats`` - free-text notes on the recipe.

    **Command line.** ``multibench env recipe METHOD`` prints the recipe as
    conda/pip commands and ``multibench env yml METHOD`` as an
    ``environment.yml``. ``mtb.run`` relies on the lockfile or packed-archive
    build (``mtb.env.install``); a recipe build can miss what it needs (see
    ``caveats``).

    See Also
    --------
    mtb.env.install : installs the env ``mtb.run`` relies on (packed archive or lockfile).

    mtb.env.status : ``has_recipe`` and ``difficulty`` per method.
    """
    return registry.get(method).env_spec


def own_env_name(method: str) -> str:
    """The method's own env name, ``scmb_<method>``: :func:`group_for`'s
    fallback for a method in no shared group."""
    return f"scmb_{method.lower()}"


def default_env_name(method: str) -> str:
    """The conda env name every entry point uses for ``method``:
    :func:`group_for` after validating the id."""
    return group_for(registry.check_method(method))


def _conda_bin(prefer: str = "mamba") -> str:
    order = [prefer, "mamba", "conda"] if prefer != "conda" else ["conda", "mamba"]
    for cand in order:
        found = shutil.which(cand)
        if found:
            return found
    return prefer


def _install_commands(spec: dict, env_name: str, conda: str | None) -> list[list[str]]:
    """argv command(s) to create an env from a spec (recipe or group) and install."""
    conda = conda or _conda_bin()
    # channels first: newer mamba rejects packages placed after `-c`
    create = [conda, "create", "-y", "-n", env_name]
    for ch in spec.get("conda_channels", []):
        create += ["-c", ch]
    if spec.get("python_version"):
        create.append(f"python={spec['python_version']}")
    create += _conda_packages(spec)
    pip_git = list(spec.get("pip_git", []))
    pip_pkgs = list(spec.get("pip_packages", [])) + pip_git
    if pip_pkgs:
        create.append("pip")
    cmds = [create]
    if pip_pkgs:
        runner = conda.replace("mamba", "conda")  # `conda run` for the pip step
        pip = [runner, "run", "-n", env_name, "pip", "install", *pip_pkgs]
        # git installs of setuptools_scm packages fail on shallow/tagless clones
        # (`git tag --points-at HEAD` -> 128); a pretend version makes them build.
        if pip_git:
            pip = ["env", "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0", *pip]
        cmds.append(pip)
    return cmds


def _conda_packages(spec: dict) -> list[str]:
    """The recipe's conda packages, minus a ``python`` pin when
    ``python_version`` already gives one."""
    pkgs = list(spec.get("conda_packages", []))
    if not spec.get("python_version"):
        return pkgs
    return [c for c in pkgs if not _is_python_pin(c)]


def _is_python_pin(package: str) -> bool:
    """``True`` for ``python``, ``python=3.7``, ``python==3.7.10``, ``python>=3``."""
    name = re.split(r"[=<>!~ ]", str(package).strip(), maxsplit=1)[0]
    return name == "python"


def _environment_yml(spec: dict, env_name: str) -> str:
    channels = list(spec.get("conda_channels", [])) or ["conda-forge"]
    lines = [f"name: {env_name}", "channels:"]
    lines += [f"  - {c}" for c in channels]
    lines.append("dependencies:")
    if spec.get("python_version"):
        lines.append(f"  - python={spec['python_version']}")
    for c in _conda_packages(spec):
        lines.append(f"  - {c}")
    pip_pkgs = list(spec.get("pip_packages", [])) + list(spec.get("pip_git", []))
    if pip_pkgs:
        lines += ["  - pip", "  - pip:"]
        lines += [f"    - {p}" for p in pip_pkgs]
    return "\n".join(lines) + "\n"


def create_commands(method: str, env_name: str | None = None,
                    conda: str | None = None) -> list[list[str]]:
    """The create+install commands of a method's hand-written recipe, into
    ``env_name`` (default :func:`default_env_name`)."""
    r = recipe(method)
    if not r:
        raise ValueError(f"no recipe declared for {method!r}")
    return _install_commands(r, env_name or default_env_name(method), conda)


def environment_yml(method: str, env_name: str | None = None) -> str:
    """A method's recipe as an ``environment.yml`` string, named ``env_name``
    (default :func:`default_env_name`)."""
    r = recipe(method)
    if not r:
        raise ValueError(f"no recipe declared for {method!r}")
    return _environment_yml(r, env_name or default_env_name(method))


# --- shared group environments --------------------------------------------
@functools.lru_cache(maxsize=1)
def _merged_groups() -> dict:
    return (yaml.safe_load(_GROUPS_YAML.read_text()) or {}).get("groups", {})


@functools.lru_cache(maxsize=1)
def _method_env() -> dict:
    """``{method: conda env}`` overrides from ``env_groups.yaml``; they win
    over group membership."""
    return (yaml.safe_load(_GROUPS_YAML.read_text()) or {}).get("method_env", {}) or {}


def groups() -> dict:
    """``{env_name: spec}`` of every env: the shared ones and each method's own."""
    merged = _merged_groups()
    covered = {m for g in merged.values() for m in g.get("members", [])}
    out = {name: {**g, "shared": True} for name, g in merged.items()}
    for method, env in _method_env().items():
        g = out.setdefault(env, {"members": [], "shared": True})
        g.setdefault("members", [])
        if method not in g["members"]:
            g["members"].append(method)
        covered.add(method)
    for s in registry.load():
        if s.id not in covered:
            out[own_env_name(s.id)] = {
                **s.env_spec, "members": [s.id], "shared": False,
            }
    return out


def group_for(method: str) -> str:
    """The env that serves this method: its override, its shared group, or
    its own ``scmb_<method>`` env."""
    override = _method_env()
    if method in override:
        return override[method]
    for name, g in _merged_groups().items():
        if method in g.get("members", []):
            return name
    return own_env_name(method)


def group_create_commands(group: str, env_name: str | None = None,
                          conda: str | None = None) -> list[list[str]]:
    """Create+install commands for a group env, into ``env_name`` (default:
    the group name)."""
    spec = groups().get(group)
    if spec is None:
        raise KeyError(f"unknown group {group!r}; see envs.groups()")
    return _install_commands(spec, env_name or group, conda)


def _check_methods(methods):
    """``methods`` as registry ids (any letter case or a display name is
    accepted); ``KeyError`` for an unknown id (else a typo gets a made-up
    ``scmb_<typo>`` env) and ``TypeError`` for a bare string."""
    registry.check_id_list(methods, "methods")
    return None if methods is None else [registry.check_method(m) for m in methods]


def _category_or_methods(category, methods, fn: str):
    """``(category, methods)``, with a list or tuple of method ids passed as the
    first argument (``plan(["Matilda"])``) moved to ``methods``."""
    if isinstance(category, (list, tuple)):
        if methods is not None:
            raise TypeError(f"{fn}() got methods twice: pass the list once, as "
                            f"methods=[...]")
        return None, list(category)
    return category, methods


def plan(category: str | list[str] | None = None, methods: list[str] | None = None, *,
         as_frame: bool = False):
    """List the conda envs a set of methods needs, one row per env.

    Methods that share an env collapse into one row.

    Parameters
    ----------
    category : str | list[str] | None
        Integration category (``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``) whose methods to cover; ``None`` = every method. A list
        of method ids here is read as ``methods``.
    methods : list[str] | None
        Method ids to cover instead of ``category``.
    as_frame : bool
        ``True`` returns a ``pandas.DataFrame`` with the same keys as columns.

    Returns
    -------
    list[dict] or pandas.DataFrame
        One row per env, the env serving the most methods first. Read
        ``env`` and ``methods``; all keys are listed in Notes.

    Raises
    ------
    ValueError
        Unknown ``category``; the message lists the valid ones.
    KeyError
        Unknown method id in ``methods``; the message suggests a close
        match, if any.
    TypeError
        ``methods`` given as a bare string.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.plan("vertical", as_frame=True)[["env", "methods"]]
    >>> mtb.env.plan(["Matilda", "totalVI", "scMoMaT"])

    Notes
    -----
    **Keys.**

    - ``env`` - the conda env name, the one ``mtb.run`` activates.
    - ``shared`` - the env serves several methods (``multibench env groups``
      lists these shared envs).
    - ``methods`` - the selected methods this env serves, sorted.
    - ``flavor`` - ``'cpu'`` / ``'gpu'`` / ``'single'`` when the env is
      installed here from a packed archive, else ``None``.

    **Selection.** ``methods`` takes precedence over ``category``.

    **Command line.** ``multibench env plan`` prints the same rows with each
    archive's download and unpacked size (``?`` = not measured) and a total.

    See Also
    --------
    mtb.env.install : builds or unpacks exactly these envs.

    mtb.env.doctor : whether each of these envs exists here.
    """
    category, methods = _category_or_methods(category, methods, "plan")
    methods = _check_methods(methods)
    if methods is None:
        methods = registry.list_methods(category=category)
    shared = _merged_groups()
    buckets: dict[str, list[str]] = {}
    for m in methods:
        buckets.setdefault(group_for(m), []).append(m)
    rows = [
        {"env": env, "shared": env in shared, "methods": sorted(ms),
         "flavor": installed_flavor(env)}
        for env, ms in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    return _as_frame(rows, as_frame)


# --- env existence / status -----------------------------------------------
PACKED_URL = "https://github.com/DSichang/scMultiBench/releases/download/envs-v1"


def _envs_dir(conda_bin: str) -> Path | None:
    """Where this conda/mamba keeps named environments, or None.

    ``info --json`` names the folders differently in conda and mamba >= 2,
    so every spelling is read, then ``<root>/envs`` of the launcher's
    ``<root>/bin/<tool>``. The first writable candidate wins: an env in a
    folder conda does not search is invisible by name.
    """
    def _paths(value):
        items = value if isinstance(value, (list, tuple)) else [value]
        return [Path(str(v)) for v in items if str(v or "").startswith("/")]

    cands: list[Path] = []
    try:
        out = subprocess.run([conda_bin, "info", "--json"],
                             capture_output=True, text=True).stdout
        info = _json.loads(out) if out.strip().startswith("{") else {}
    except Exception:  # noqa: BLE001 - any failure just means "ask elsewhere"
        info = {}
    for key in ("envs_dirs", "envs directories"):
        cands += _paths(info.get(key))
    for key in ("root_prefix", "base environment"):
        cands += [p / "envs" for p in _paths(info.get(key))]
    exe = shutil.which(conda_bin)
    if exe:
        root = Path(exe).resolve().parent.parent
        if (root / "conda-meta").is_dir() or (root / "envs").is_dir():
            cands.append(root / "envs")
    for var in ("CONDA_ROOT", "MAMBA_ROOT_PREFIX"):
        cands += [p / "envs" for p in _paths(os.environ.get(var))]

    for d in cands:
        probe = d if d.is_dir() else d.parent
        if probe.is_dir() and os.access(probe, os.W_OK):
            return d
    return None


def _find_conda() -> str | None:
    """Path of the conda (else mamba) binary on PATH, or ``None`` (unlike
    ``_conda_bin``, which returns a bare name when nothing is installed)."""
    return shutil.which("conda") or shutil.which("mamba")


@functools.lru_cache(maxsize=4)
def _conda_prefixes(conda: str) -> tuple:
    """The prefixes ``<conda> env list --json`` reports, cached per binary;
    ``()`` when the tool is absent or answers garbage. Cleared after an
    env is created."""
    try:
        out = subprocess.run([conda, "env", "list", "--json"],
                             capture_output=True, text=True, check=True).stdout
        prefixes = _json.loads(out).get("envs", [])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError,
            TypeError, AttributeError, OSError):
        return ()
    return tuple(str(p) for p in prefixes)


def _prefixes_on_disk(envs_dir: Path) -> dict[str, Path]:
    """``{env: prefix}`` for every ``<envs_dir>/<env>`` that carries ``bin/``."""
    try:
        return {p.name: p for p in sorted(envs_dir.iterdir())
                if (p / "bin").is_dir()}
    except OSError:
        return {}


#: What :data:`FLAVOR_FILE` holds in an env built from a lockfile or a recipe
#: (a packed install writes its flavour there instead).
BUILT_FROM = {"lockfile": "build", "recipe": "recipe"}


def _creation_command(prefix: Path) -> str:
    """The first ``# cmd:`` line of ``<prefix>/conda-meta/history`` (the command
    that created the env), or ``""``."""
    try:
        with open(prefix / "conda-meta" / "history", encoding="utf-8",
                  errors="replace") as fh:
            for line in fh:
                if line.startswith("# cmd:"):
                    return line[len("# cmd:"):].strip()
    except OSError:
        pass
    return ""


def _lockfile_build_record(prefix: Path, env: str) -> bool:
    """Was ``prefix`` created by a lockfile build of an earlier multibench?

    Those builds ran ``<conda> env create -n <env> -f <file>`` with ``<file>``
    one of ``.../multibench_envbuild/<env>/conda.yml`` (installed package),
    ``.../env_locks/.build/<env>/conda.yml`` (source checkout), or, before
    0.2, ``.../env_locks/<env>.yml``; conda keeps that command as the first
    ``# cmd:`` line of ``conda-meta/history``.
    """
    cmd = _creation_command(prefix)
    if not cmd:
        return False
    e = re.escape(env)
    if not re.search(r"(?:^|\s)env\s+create\s", cmd):
        return False
    if not re.search(rf"\s(?:-n|--name)(?:\s+|=){e}(?:\s|$)", cmd):
        return False
    return re.search(rf"\s(?:-f|--file)(?:\s+|=).*/(?:multibench_envbuild/{e}/conda\.yml|"
                     rf"env_locks/\.build/{e}/conda\.yml|env_locks/{e}\.yml)\s*$",
                     cmd) is not None


def made_by_multibench(prefix: Path | str, env: str | None = None) -> bool:
    """Did multibench install the env at ``prefix``?

    ``True`` when the prefix holds :data:`FLAVOR_FILE` (every install path
    writes it), or when conda's history shows a lockfile build of an earlier
    multibench (:func:`_lockfile_build_record`). An env someone made by hand
    under the same name is not a method env.
    """
    prefix = Path(prefix)
    return (prefix / FLAVOR_FILE).is_file() or _lockfile_build_record(prefix, env or prefix.name)


def _same_named(env: str, conda: str | None = None):
    """Every prefix named ``env``, lazily: ``<envs_dir>/<env>`` when it has
    ``bin/`` (no conda asked), then those ``conda env list`` reports."""
    candidate = Path(config.DEFAULT.envs_dir) / env
    if (candidate / "bin").is_dir():
        yield candidate
    conda_exe = conda or _find_conda()
    if conda_exe is not None:
        for p in _conda_prefixes(conda_exe):
            if os.path.basename(p.rstrip("/")) == env and Path(p) != candidate:
                yield Path(p)


def other_env_prefix(env: str, conda: str | None = None) -> Path | None:
    """A prefix named ``env`` that multibench did not install, or ``None``.

    Such an env is never used for a method; :func:`other_env_text` says what to do.
    """
    for p in _same_named(env, conda):
        if p.is_dir() and not made_by_multibench(p, env):
            return p
    return None


def other_env_text(env: str, prefix) -> str:
    """``"Environment matilda at /x/envs/matilda was not installed by multibench,
    so it is not used."``"""
    return (f"Environment {env} at {prefix} was not installed by multibench, so it "
            f"is not used.")


def _refuse_other_env(env: str, conda: str | None = None) -> None:
    """``RuntimeError`` before an install when an env of that name exists that
    multibench did not install."""
    other = other_env_prefix(env, conda)
    if other is not None:
        raise RuntimeError(f"{other_env_text(env, other)} Rename or remove it, "
                           f"then install again.")


def _mark_built(env: str, conda: str | None, how: str, before: set) -> None:
    """Write :data:`FLAVOR_FILE` (``BUILT_FROM[how]``) into the env a lockfile
    or recipe build just created under the name ``env``: a prefix of that
    name that was not there ``before`` the build."""
    _conda_prefixes.cache_clear()
    for p in _same_named(env, conda):
        if p not in before and p.is_dir():
            (p / FLAVOR_FILE).write_text(BUILT_FROM[how] + "\n")


def env_prefix(env: str, conda: str | None = None) -> Path | None:
    """The on-disk prefix of a method environment, or ``None``.

    Parameters
    ----------
    env : str
        Environment name, e.g. ``"matilda"`` (``mtb.method_info(m)["env"]``).
    conda : str, optional
        conda/mamba executable to ask when the env is not under ``envs_dir``;
        ``None`` = the one on ``PATH``, if any.

    Returns
    -------
    Path or None
        The environment's prefix; ``None`` when it is not installed.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.env_prefix("matilda")       # None until the env is installed

    Notes
    -----
    ``<mtb.config.DEFAULT.envs_dir>/<env>`` when it contains ``bin/`` (no
    conda needed); else the prefix ``conda env list`` reports for that name.
    Only an env multibench installed counts: every install writes the file
    ``.multibench_flavor`` into it, and an env built from a lockfile by an
    earlier version is known by the build command conda recorded. An env
    you made yourself under the same name is never used. This is what
    "installed" means everywhere, and the prefix ``mtb.run`` activates.
    """
    for p in _same_named(env, conda):
        if made_by_multibench(p, env):
            return p
    return None


def archive_urls(key: str, manifest: dict | None = None) -> list[str]:
    """The download URLs of the archive ``key``, in order: one, or the parts
    of an archive over 2 GiB; ``<PACKED_URL>/<key>.tar.gz`` without an entry."""
    manifest = packed_manifest() if manifest is None else manifest
    entry = manifest.get(key) or f"{PACKED_URL}/{key}.tar.gz"
    return [entry] if isinstance(entry, str) else list(entry)


def download_archive(urls: list[str], target: Path, *, label: str,
                     total: int | None = None, attempts: int = 4,
                     timeout: float = 60) -> int:
    """Stream ``urls`` in order into ``target``; return the bytes written.

    Prints a line at the start and per tenth of ``total`` (per 500 MB when it
    is unknown). A dropped or stalled connection or an HTTP 5xx resumes where
    it stopped, up to ``attempts`` times per URL; another HTTP error raises.
    """
    import time
    import urllib.error
    import urllib.request
    step = max(total // 10, 1) if total else 500_000_000
    # MB below a gigabyte, so a 53 MB file does not read "0.1 GB"
    fmt = _gb if (total or 0) >= 1e9 else (lambda n: f"{n / 1e6:.0f} MB")
    size = f" {fmt(total)}" if total else ""
    print(f"[env] downloading {label}{size} ...", flush=True)
    done, mark, t0 = 0, step, time.monotonic()
    with open(target, "wb") as out:
        for url in urls:
            start = done
            for attempt in range(1, attempts + 1):
                have = done - start
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={have}-"} if have else {})
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if have and getattr(resp, "status", 200) != 206:
                            # the whole file again, not the rest
                            out.seek(start)
                            out.truncate()
                            done = start
                        while chunk := resp.read(1 << 20):
                            out.write(chunk)
                            done += len(chunk)
                            if done >= mark:
                                rate = done / max(time.monotonic() - t0, 1e-3) / 1e6
                                print(f"[env]   {fmt(done)}" + (f" of {fmt(total)}" if total else "")
                                      + f" ({rate:.0f} MB/s)", flush=True)
                                while mark <= done:
                                    mark += step
                    break
                except (OSError, TimeoutError) as e:
                    final = isinstance(e, urllib.error.HTTPError) and e.code < 500
                    if final or attempt == attempts:
                        raise
                    print(f"[env]   connection lost ({type(e).__name__}); resuming",
                          flush=True)
                    time.sleep(3 * attempt)
    return done


_LAYERS_JSON = Path(__file__).resolve().parent / "packed_layers.json"

#: Folder under the envs dir that holds the unpacked layers.
LAYERS_DIR = ".layers"
#: Written by ``tools/pack_layers.py`` into an env archive that needs layers.
LAYER_MARKER = ".multibench_layers.json"
#: Carried by a layer archive: its name, file count and digest.
LAYER_INFO = ".multibench_layer.json"


@functools.lru_cache(maxsize=1)
def packed_layers() -> dict:
    """The layers of ``engine/packed_layers.json``: ``{layer: {"urls",
    "files", "digest", "archive_bytes", "unpacked_bytes", "envs"}}``.

    A layer holds files several env archives share (torch and CUDA
    libraries); it is downloaded once and hard-linked into each env.
    """
    return _json.loads(_LAYERS_JSON.read_text())


def layers_for(keys, *, layers: dict | None = None, envs_dir=None,
               missing_only: bool = True) -> dict:
    """``{layer: entry}`` the archive ``keys`` need, each layer once;
    ``missing_only`` leaves out those ``envs_dir`` already holds."""
    layers = packed_layers() if layers is None else layers
    root = Path(envs_dir or config.DEFAULT.envs_dir) / LAYERS_DIR
    keys = set(keys)
    return {name: entry for name, entry in layers.items()
            if keys & set(entry.get("envs", ()))
            and not (missing_only and (root / name / LAYER_INFO).is_file())}


def _ensure_layer(name: str, info: dict, envs_root: Path) -> Path:
    """``<envs_root>/.layers/<name>``, downloaded and unpacked on first use;
    ``RuntimeError`` when its digest is not the one ``info`` names."""
    import tarfile
    from ..data.fetch import safe_extract
    dest = envs_root / LAYERS_DIR / name
    if not (dest / LAYER_INFO).is_file():
        entry = packed_layers().get(name)
        if not entry or not entry.get("urls"):
            raise RuntimeError(f"this multibench version does not list the layer {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tgz = dest.parent / f".{name}.tar.gz.download"
        part = dest.with_name(name + ".partial")
        try:
            download_archive(list(entry["urls"]), tgz, label=f"{name} (shared layer)",
                             total=entry.get("archive_bytes"))
            print(f"[env] unpacking {name} -> {dest} ...", flush=True)
            shutil.rmtree(part, ignore_errors=True)
            shutil.rmtree(dest, ignore_errors=True)
            part.mkdir(parents=True)
            with tarfile.open(tgz) as t:
                safe_extract(t, part)
            part.rename(dest)
        finally:
            tgz.unlink(missing_ok=True)
            shutil.rmtree(part, ignore_errors=True)
    have = _json.loads((dest / LAYER_INFO).read_text())
    if have.get("digest") != info.get("digest"):
        raise RuntimeError(f"the layer {name} on disk is not the one this archive was "
                           f"packed with; delete {dest} and install again")
    return dest


def _link_layer(layer: Path, prefix: Path, info: dict) -> None:
    """Hard-link (else copy) every file of ``layer`` into ``prefix``;
    ``RuntimeError`` when a file is already there or the count is wrong."""
    n = 0
    for dirpath, _dirs, files in os.walk(layer):
        rel = Path(dirpath).relative_to(layer)
        (prefix / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if rel == Path(".") and f == LAYER_INFO:
                continue
            src, dst = Path(dirpath) / f, prefix / rel / f
            if dst.exists():
                raise RuntimeError(f"{rel / f} is in the env archive and in the layer")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            n += 1
    if n != info.get("files"):
        raise RuntimeError(f"the layer {layer.name} has {n} files, expected {info.get('files')}")


def install_packed(env: str, *, envs_dir: Path | str | None = None,
                   conda: str | None = None, force: bool = False,
                   flavor: str = "auto") -> bool:
    """Provision ``env`` from a prebuilt conda-pack archive, if one is published.

    Downloads the archive, extracts it into ``<envs_dir>/<env>`` and runs its
    own ``bin/conda-unpack``; no conda is needed. ``envs_dir`` defaults to
    that of ``conda`` when only it is given, else
    ``mtb.config.DEFAULT.envs_dir``. ``flavor`` picks the archive as in
    :func:`install`; the prefix is the same whatever the flavour, and the
    flavour installed is written to :data:`FLAVOR_FILE`. A non-Linux host
    raises ``RuntimeError`` unless ``force``.

    Returns
    -------
    bool
        ``True`` on success or when the prefix exists; ``False`` when no
        archive is published or the unpack failed (the caller then builds
        from the lockfile).
    """
    import tarfile
    import urllib.error
    import urllib.request
    import warnings

    check_flavor(flavor)                  # a typo fails before the platform check
    _require_linux(force)                 # fail closed before any bytes land
    if envs_dir is not None:
        envs_root = Path(envs_dir)
    elif conda is not None and shutil.which(conda):
        envs_root = _envs_dir(conda) or Path(config.DEFAULT.envs_dir)
    else:
        envs_root = Path(config.DEFAULT.envs_dir)
    dest = envs_root / env
    if dest.exists():
        if made_by_multibench(dest, env):
            return True
        raise RuntimeError(f"{other_env_text(env, dest)} Rename or remove it, "
                           f"then install again.")
    manifest, sizes = packed_manifest(), packed_sizes()
    key, installed = archive_for(env, flavor, manifest=manifest, sizes=sizes)
    if resolve_flavor(flavor) == "cpu" and installed == "gpu":
        warnings.warn(_cpu_fallback_warning(env, manifest, sizes), UserWarning,
                      stacklevel=2)
    urls = archive_urls(key, manifest)
    build = "single" if _single_build(env, manifest=manifest) else installed
    envs_root.mkdir(parents=True, exist_ok=True)
    tgz = envs_root / f".{key}.tar.gz.download"
    try:
        download_archive(urls, tgz, label=f"{env} ({build} build)",
                         total=(sizes.get(key) or {}).get("archive_bytes"))
    except urllib.error.HTTPError:
        tgz.unlink(missing_ok=True)
        return False
    except BaseException:
        tgz.unlink(missing_ok=True)
        raise
    print(f"[env] unpacking {env} -> {dest} ...", flush=True)
    from ..data.fetch import safe_extract
    part = dest.with_name(dest.name + ".partial")
    try:
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with tarfile.open(tgz) as t:
            safe_extract(t, part)
        if not (part / "bin").is_dir():
            raise RuntimeError("archive did not contain bin/")
        # the files this env shares with others: one download, hard links
        marker = part / LAYER_MARKER
        if marker.is_file():
            for name, info in _json.loads(marker.read_text())["layers"].items():
                _link_layer(_ensure_layer(name, info, envs_root), part, info)
        # conda-unpack writes the final path into the env: move first
        part.rename(dest)
        _conda_unpack(dest)
        # last, so a prefix that failed to unpack claims no flavour
        (dest / FLAVOR_FILE).write_text(
            ("single" if build == "single" else installed) + "\n")
        _conda_prefixes.cache_clear()
        return True
    except Exception as e:  # noqa: BLE001 - degrade to the lockfile build
        shutil.rmtree(part, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[env] prebuilt {env} failed ({type(e).__name__}: "
              f"{str(e)[:120]}); falling back to the lockfile build",
              flush=True)
        return False
    finally:
        tgz.unlink(missing_ok=True)


def _conda_unpack(prefix: Path) -> None:
    """Run ``<prefix>/bin/conda-unpack``, if there, with the env's own python
    (first on ``PATH``, or explicitly for a script that lost its executable bit)."""
    unpack = prefix / "bin" / "conda-unpack"
    if not unpack.exists():
        return
    py = prefix / "bin" / "python"
    argv = [str(unpack)] if os.access(unpack, os.X_OK) else [str(py), str(unpack)]
    env = {**os.environ,
           "PATH": f"{prefix / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
    subprocess.run(argv, check=True, capture_output=True, env=env)


def _all_env_names(conda: str | None = None) -> list[str]:
    """Names of every env here, whoever made it: the prefixes under
    ``envs_dir`` first, then those ``conda`` lists, without duplicates."""
    return list(dict.fromkeys(name for name, _ in _all_prefixes(conda)))


def _all_prefixes(conda: str | None = None) -> list[tuple[str, Path]]:
    """``(name, prefix)`` of every env under ``envs_dir``, then of those
    ``conda`` lists."""
    out = list(_prefixes_on_disk(Path(config.DEFAULT.envs_dir)).items())
    conda_exe = conda or _find_conda()
    if conda_exe is not None:
        for p in _conda_prefixes(conda_exe):
            name = os.path.basename(str(p).rstrip("/"))
            if name:
                out.append((name, Path(p)))
    return out


def installed_envs(conda: str | None = None) -> list[str]:
    """Names of the envs :func:`env_prefix` would find: those multibench
    installed (:func:`made_by_multibench`), under ``envs_dir`` first, then
    those ``conda`` lists, without duplicates."""
    return list(dict.fromkeys(name for name, p in _all_prefixes(conda)
                              if made_by_multibench(p, name)))


def _check_conda(conda) -> None:
    """Raise unless ``conda`` is ``None`` or names an executable; a method id
    or a list (``status("Matilda")``) gets the call that takes methods."""
    if conda is None:
        return
    if isinstance(conda, (list, tuple, set, frozenset)):
        raise TypeError(
            f"conda is the conda/mamba executable, not a list of methods "
            f"(got {list(conda)!r}); for their envs call "
            f"mtb.env.doctor(methods={list(conda)!r})")
    if not isinstance(conda, (str, os.PathLike)):
        raise TypeError(
            f"conda must be the conda/mamba executable (a path or a name on "
            f"PATH), got {type(conda).__name__} {conda!r}")
    name = os.fspath(conda)
    ids = {s.id.lower(): s.id for s in registry.load()}
    if str(name).lower() in ids:
        m = ids[str(name).lower()]
        raise ValueError(
            f"{name!r} is a method id, but status() takes no methods: its "
            f"only positional argument is conda, the conda/mamba executable. "
            f"For this method call mtb.env.doctor(methods=[{m!r}]), or keep "
            f"the status() row whose method is {m!r}")
    if shutil.which(name) is None:
        raise ValueError(
            f"conda={name!r} is not an executable path or a name on PATH")


def status(conda: str | None = None, *, as_frame: bool = False):
    """Report, per method, whether its conda env is installed on this machine.

    Parameters
    ----------
    conda : str | None
        conda/mamba executable, a path or a name on PATH, that lists the
        installed envs; ``None`` = conda if found, else mamba.
    as_frame : bool
        ``True`` returns a ``pandas.DataFrame`` with the same keys as columns.

    Returns
    -------
    list[dict] or pandas.DataFrame
        One row per package method. Read ``method``, ``env``, ``exists``
        and ``has_lock``; all keys are listed in Notes.

    Raises
    ------
    TypeError
        ``conda`` is not a path or a name, e.g. a list of method ids.
    ValueError
        ``conda`` is a method id, or no executable has that path or name.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.env.status(as_frame=True)
    >>> df.loc[~df.exists, ["method", "env", "has_lock"]]     # what is missing
    >>> df.loc[df.method == "Matilda"].iloc[0].to_dict()

    Notes
    -----
    **Selecting methods.** ``status`` reports every method;
    ``mtb.env.doctor(methods=[...])`` reports the envs of a selection, and
    ``multibench env status --methods ...`` prints only those methods.

    **Keys.**

    - ``method`` - the method id.
    - ``env`` - the env the package uses for the method, the name
      ``mtb.scan`` and ``mtb.run`` use (``mtb.method_info(m)['env']``).
    - ``group`` - the same name as ``env``.
    - ``own_env`` - the singleton name ``scmb_<method>`` (lower-case); the
      method also counts as installed when this env exists.
    - ``exists`` - ``env`` or ``own_env`` is installed here.
    - ``has_lock`` - a shipped lockfile can build ``env``.
    - ``difficulty`` - a tag for how hard the env is to build (below).
    - ``verified_working`` - the env ran the method end-to-end on its
      reference dataset (the ``*`` after the tag in ``multibench env status``).
    - ``has_recipe`` - the method declares a recipe (``mtb.env.recipe``).
    - ``flavor`` - ``'cpu'`` / ``'gpu'`` when the installed env came from a
      packed archive, ``'single'`` for an env with one archive for CPU and
      GPU hosts, else ``None``.

    **Difficulty tags.** How hard the env is to build from its recipe, not
    how well the method works:

    - ``easy`` - modern Python/torch stack; builds from the lockfile.
    - ``old-scvi`` / ``old-tensorflow`` - old scvi-tools or TensorFlow pins
      that need their own env.
    - ``R`` - an R env; the post-install script restores packages
      installed with ``install.packages()``.
    - ``verified`` - built from the lockfile on a fresh machine, and the
      method ran end-to-end on its reference dataset.
    - ``blocked-script`` - the env builds, but the upstream script cannot
      run unmodified from the public checkout.
    - ``unknown`` - no recipe declared.

    **What counts as installed.** What ``mtb.env.env_prefix`` finds, asking
    the ``conda`` given.

    See Also
    --------
    mtb.env.doctor : the same information per env rather than per method.

    mtb.env.install : builds or unpacks the missing envs.
    """
    _check_conda(conda)
    have = set(installed_envs(conda))
    out = []
    for s in registry.load():
        env_spec = s.env_spec
        env_name = group_for(s.id)
        own = own_env_name(s.id)
        exists = env_name in have or own in have
        out.append({
            "method": s.id, "env": env_name, "group": env_name, "own_env": own,
            "exists": exists,
            "has_lock": lockfile(env_name) is not None,
            "difficulty": env_spec.get("difficulty", "unknown"),
            "verified_working": bool(env_spec.get("verified_working", False)),
            "has_recipe": bool(env_spec),
            "flavor": (installed_flavor(env_name, conda) or installed_flavor(own, conda)
                       if exists else None),
        })
    return _as_frame(out, as_frame)


# --- lockfile-based provisioning (the reproducible install path) -----------
def lockfile(env_name: str) -> Path | None:
    """Path of the shipped lockfile ``env_locks/<env_name>.yml``, or ``None``."""
    p = _LOCKS_DIR / f"{env_name}.yml"
    return p if p.exists() else None


def required_envs(category: str | None = None,
                  methods: list[str] | None = None) -> list[str]:
    """The distinct envs (:func:`group_for`) the given methods, or all, run in."""
    if methods is None:
        methods = registry.list_methods(category=category)
    seen: list[str] = []
    for m in methods:
        e = group_for(m)
        if e not in seen:
            seen.append(e)
    return seen


def split_lock(text: str):
    """Split a lockfile into (conda-only YAML, pip requirement lines).

    The pip section is a full ``pip freeze`` closure: re-resolving it, as
    ``conda env create`` does, fails on combinations pip calls inconsistent.
    It is installed separately with ``--no-deps``, which a requirements file
    cannot express.
    """
    conda_lines, pip_lines = [], []
    pip_indent = None
    for ln in text.splitlines():
        stripped = ln.strip()
        indent = len(ln) - len(ln.lstrip())
        if stripped == "- pip:":
            pip_indent = indent
            continue
        if pip_indent is not None and stripped:
            if indent <= pip_indent:
                pip_indent = None
            else:
                pip_lines.append(stripped[2:].strip() if stripped.startswith("- ")
                                 else stripped)
                continue
        conda_lines.append(ln)
    return "\n".join(conda_lines) + "\n", pip_lines


def _materialise_split(env_name: str, lock):
    """Write the conda-only YAML and pip requirements a two-phase install needs."""
    # an installed package's folder stays untouched: build files go to the temp dir
    if (_LOCKS_DIR.parents[2] / "pyproject.toml").is_file():
        build_dir = _LOCKS_DIR / ".build" / env_name
    else:
        import tempfile
        build_dir = Path(tempfile.gettempdir()) / "multibench_envbuild" / env_name
    build_dir.mkdir(parents=True, exist_ok=True)
    conda_yaml, pip_lines = split_lock(lock.read_text(encoding="utf-8"))
    y = build_dir / "conda.yml"
    y.write_text(conda_yaml, encoding="utf-8")
    r = build_dir / "requirements.txt"
    r.write_text("\n".join(pip_lines) + "\n" if pip_lines else "", encoding="utf-8")
    return y, r, pip_lines


def post_install(env_name: str):
    """Path to the shipped post-install script of an env, or None.

    It restores what a lockfile cannot record: R packages from
    ``install.packages()``, pip packages on no index (:data:`_NOT_ON_PYPI`)
    and editable installs (matilda, scMVP).
    """
    p = _LOCKS_DIR / f"{env_name}.post.sh"
    return p if p.is_file() else None


def create_env(env_name: str, conda: str | None = None,
               dry_run: bool = True, *, force: bool = False) -> list[list[str]]:
    """Create one env from its shipped lockfile (the reproducible path).

    ``dry_run=False`` runs the commands; on a non-Linux host it raises
    ``RuntimeError`` first unless ``force``. A missing lockfile raises
    ``FileNotFoundError``.

    Returns
    -------
    list of list of str
        The commands: conda phase, pip ``--no-deps`` phase, post-install
        script when shipped.
    """
    lock = lockfile(env_name)
    if lock is None:
        raise FileNotFoundError(
            f"no lockfile for env {env_name!r} (expected {_LOCKS_DIR / (env_name + '.yml')}). "
            f"Capture it on a host where the env exists via freeze({env_name!r}), "
            f"or build from the hand recipe via create_commands()."
        )
    conda = conda or _conda_bin("conda")
    conda_yaml, req, pip_lines = _materialise_split(env_name, lock)
    cmds = [[conda, "env", "create", "-n", env_name, "-f", str(conda_yaml)]]
    if pip_lines:
        cmds.append([conda, "run", "-n", env_name,
                     "pip", "install", "--no-deps", "-r", str(req)])
    post = post_install(env_name)
    if post is not None:
        cmds.append([conda, "run", "-n", env_name, "bash", str(post)])
    if not dry_run:
        _require_linux(force)
        _build(env_name, cmds, conda, "lockfile")
    return cmds


def _build(env_name: str, cmds: list[list[str]], conda: str | None, how: str) -> None:
    """Run the build ``cmds`` of ``env_name`` and mark the new env as made by
    multibench; refused first when another env already has that name."""
    _refuse_other_env(env_name, conda)
    before = set(_same_named(env_name, conda))
    _run_all(cmds)
    _mark_built(env_name, conda, how, before)


def create_all(category: str | None = None, methods: list[str] | None = None,
               conda: str | None = None, dry_run: bool = True, *,
               force: bool = False) -> list[dict]:
    """Provision every env needed to run the methods, from lockfiles.

    With ``dry_run=False`` the missing envs that have a lockfile are built;
    the others are reported, not built.

    Parameters
    ----------
    category : str, optional
        Methods with a variant in this category; default: all.
    methods : list of str, optional
        Explicit method ids, instead of ``category``.
    conda : str, optional
        conda executable; default ``conda``.
    dry_run : bool
        ``True`` (default) plans only.
    force : bool
        Build on a non-Linux host, which raises ``RuntimeError`` otherwise.

    Returns
    -------
    list of dict
        One entry per env, largest first: ``{env, methods, exists, has_lock,
        cmds}`` (``cmds`` = the commands run, or that would run).
    """
    methods = _check_methods(methods)
    if not dry_run:
        _require_linux(force)             # before conda is even asked anything
    have = set(installed_envs(conda))
    if methods is None:
        methods = registry.list_methods(category=category)
    by_env: dict[str, list[str]] = {}
    for m in methods:
        by_env.setdefault(group_for(m), []).append(m)
    out = []
    for env, ms in sorted(by_env.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lock = lockfile(env)
        exists = env in have
        cmds: list[list[str]] = []
        if not exists and lock is not None:
            cmds = create_env(env, conda=conda, dry_run=True)
            if not dry_run:
                _build(env, cmds, conda, "lockfile")
        out.append({"env": env, "methods": sorted(ms), "exists": exists,
                    "has_lock": lock is not None, "cmds": cmds})
    return out


_PACKED_MANIFEST = Path(__file__).resolve().parent / "packed_urls.json"


def _first_url(entry):
    """The URL a plan shows: the archive's, or its first part's."""
    return entry[0] if isinstance(entry, list) and entry else entry


def packed_manifest() -> dict:
    """``{archive key: URL or list of part URLs}`` of the published archives
    (``engine/packed_urls.json``)."""
    return _json.loads(_PACKED_MANIFEST.read_text())


def install(methods: list[str] | None = None, *, category: str | None = None,
            packed: bool = True, dry_run: bool = True, conda: str | None = None,
            force: bool = False, flavor: str = "auto") -> list[dict]:
    """Install the conda envs a set of methods needs (a dry run by default).

    The Python form of ``multibench env install``. The default returns the
    plan; ``dry_run=False`` unpacks packed archives or builds from lockfiles,
    on Linux.

    Parameters
    ----------
    methods : list[str] | None
        Method ids to cover; ``None`` = every method of ``category``, or every
        method.
    category : str | None
        Integration category (``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``) whose methods to cover when ``methods`` is ``None``.
    packed : bool
        Use a prebuilt conda-pack archive where one is published, else the
        lockfile; ``False`` = lockfile builds only.
    dry_run : bool
        ``True`` = return the plan and install nothing; ``False`` = install the
        missing envs.
    conda : str | None
        conda/mamba executable; ``None`` = conda if found, else mamba. The
        packed path needs none.
    force : bool
        ``True`` = attempt a real install on a non-Linux host, which is
        refused otherwise.
    flavor : str
        Packed-archive build per env: ``'cpu'``, ``'gpu'``
        or ``'auto'`` (``'cpu'`` unless an NVIDIA GPU is visible); unused
        when ``packed=False``.

    Returns
    -------
    list[dict]
        One row per env, the env serving the most methods first. Read
        ``env``, ``state`` and ``archive_bytes``; all keys and ``state``
        values are listed in Notes.

    Raises
    ------
    ValueError
        ``flavor`` is not ``'auto'``, ``'cpu'`` or ``'gpu'``; or unknown
        ``category``.
    KeyError
        Unknown method id in ``methods``; the message suggests a close
        match, if any.
    TypeError
        ``methods`` given as a bare string.
    RuntimeError
        ``dry_run=False`` only: a non-Linux host, no conda where needed, or a
        failed build.

    Warns
    -----
    UserWarning
        A CPU install finds no CPU archive for an env; its one archive is
        installed.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.install(["Matilda"])  # dry run: the plan, sizes, URLs
    >>> mtb.env.install(["Matilda"], dry_run=False)  # packed archive, flavor='auto'
    >>> mtb.env.install(category="vertical", dry_run=False, flavor="cpu")
    >>> # lockfile build via conda
    >>> mtb.env.install(["Matilda"], packed=False, dry_run=False)

    Notes
    -----
    **Keys.**

    - ``env`` / ``methods`` - the env and the selected methods it serves.
    - ``exists`` - the env is installed when the lockfile step runs:
      ``True`` for an env already there or just unpacked (``'PACKED'``),
      ``False`` for one this call builds or cannot build.
    - ``has_lock`` - a shipped lockfile can build the env.
    - ``state`` - what happened, or would happen (values below).
    - ``cmds`` - the lockfile commands run, or that would run.
    - ``packed_url`` / ``archive_bytes`` / ``unpacked_bytes`` - the archive
      the flavour selects and its sizes; ``None`` unless ``packed`` and
      known (a size not measured yet is ``None``).
    - ``flavor`` - for an installed env, the archive it came from (``None``
      when unrecorded); for a missing env, the flavour the packed path would
      install, or ``None`` when ``packed=False``.

    **State values on a dry run.**

    - ``'have'`` - already installed.
    - ``'name taken'`` - an env of this name exists that multibench did not
      install; it is not used. Rename or remove it, then install.
    - ``'packed archive published'`` - an archive would be downloaded; the
      sizes and URL are filled in.
    - ``'no archive - lockfile build'`` / ``'no archive - NO-LOCK'`` -
      ``packed=True`` but no archive is published; the lockfile builds the
      env, or there is no lockfile either.
    - ``'build(dry-run)'`` / ``'NO-LOCK'`` - ``packed=False``; the lockfile
      builds the env, or there is none.

    **State values after a real install** (``dry_run=False``).

    - ``'PACKED'`` - unpacked from an archive.
    - ``'BUILD'`` - built from the lockfile.
    - ``'have'`` - already installed.
    - ``'NO-LOCK'`` - no lockfile and no archive unpacked; reported, not
      built.

    **Flavours.** The env name and prefix are the same whatever the flavour.

    - ``'gpu'`` - the ``'<env>'`` archive, which every env has.
    - ``'cpu'`` - the ``'<env>-cpu'`` archive (the same env without the CUDA
      libraries, several times smaller) where published; otherwise the
      ``'<env>'`` archive and one ``UserWarning``.
    - ``'auto'`` - ``'cpu'`` when ``mtb.env.host_has_gpu()`` is ``False``,
      else ``'gpu'``; when it picks the CPU builds, one line on stderr says
      so. Pass ``flavor='gpu'`` when the envs serve jobs on GPU nodes.

    The dry-run sizes and URL follow the flavour. The flavour installed is
    shown by ``mtb.env.status``, ``mtb.env.doctor`` and ``mtb.env.plan``.

    **Order of work** (``dry_run=False``). With ``packed``, each missing env
    is first tried as an archive; an HTTP error or a failed unpack falls back
    to the lockfile build. Archives unpack into ``mtb.config.Config.envs_dir``
    (or, when ``conda`` is given, that tool's envs dir) and need no conda.
    Without conda, a missing env that needs a lockfile build raises
    ``RuntimeError`` before any download, and so does an env of the same
    name that multibench did not install.

    **Command line.** ``multibench env install --methods X --packed --run``
    takes the packed path; ``multibench env install --run`` builds from
    lockfiles.

    See Also
    --------
    mtb.env.plan : the envs a set of methods needs, before installing.

    mtb.env.doctor : which of those envs exist here.

    mtb.env.status : install status per method.
    """
    check_flavor(flavor)
    methods = _check_methods(methods)
    manifest = packed_manifest() if packed else {}
    sizes = packed_sizes() if packed else {}
    unpacked: set[str] = set()
    if not dry_run:
        _require_linux(force)             # before conda is even asked anything
        missing = [r for r in doctor(category=category, methods=methods, conda=conda)
                   if not r["exists"]]
        for r in missing:          # before any download or build
            if r.get("other_env"):
                raise RuntimeError(f"{other_env_text(r['env'], r['other_env'])} Rename "
                                   f"or remove it, then install again.")
        if (conda or _find_conda()) is None:
            # only the packed path works: refuse before any download
            for r in missing:
                if packed and r["env"] in manifest:
                    continue
                if r["env"] in packed_manifest():
                    raise RuntimeError(
                        f"{NO_CONDA} Environment {r['env']} has a packed archive. Pass "
                        f"{config.hint('packed=True', '--packed')} to install it.")
                raise RuntimeError(
                    f"{NO_CONDA} Environment {r['env']} has no packed archive. Install "
                    f"conda first.")
        if packed:
            # said only when a CPU build is taken
            note = auto_flavor_note(flavor) if any(
                archive_for(r["env"], flavor, manifest=manifest, sizes=sizes)[1] == "cpu"
                for r in missing) else None
            if note:
                print(note, file=sys.stderr, flush=True)
            for r in missing:
                if install_packed(r["env"], conda=conda, force=force, flavor=flavor):
                    unpacked.add(r["env"])
    rows = create_all(category=category, methods=methods, conda=conda,
                      dry_run=dry_run, force=force)
    out = []
    for r in rows:
        env = r["env"]
        if env in unpacked:
            state = "PACKED"
        elif r["exists"]:
            state = "have"
        elif dry_run and other_env_prefix(env, conda) is not None:
            state = "name taken"
        elif not dry_run:
            state = "BUILD" if r["has_lock"] else "NO-LOCK"
        elif packed:
            if env in manifest:
                state = "packed archive published"
            else:
                state = ("no archive - lockfile build" if r["has_lock"]
                         else "no archive - NO-LOCK")
        else:
            state = "build(dry-run)" if r["has_lock"] else "NO-LOCK"
        # URL and sizes of the archive this host would fetch
        if packed:
            key, would_install = archive_for(env, flavor, manifest=manifest, sizes=sizes)
        else:
            key, would_install = env, None
        installed = installed_flavor(env, conda) if (r["exists"] or env in unpacked) else None
        size = sizes.get(key) or {}
        out.append({**r, "state": state, "packed_url": _first_url(manifest.get(key)),
                    "archive_bytes": size.get("archive_bytes"),
                    "unpacked_bytes": size.get("unpacked_bytes"),
                    "flavor": installed if (r["exists"] or env in unpacked) else would_install})
    if dry_run and packed and any(not r["exists"] and r["flavor"] == "cpu"
                                  and r["state"] != "name taken" for r in out):
        note = auto_flavor_note(flavor)
        if note:
            print(note, file=sys.stderr, flush=True)
    return out


def doctor(category: str | list[str] | None = None, methods: list[str] | None = None,
           conda: str | None = None, *, as_frame: bool = False):
    """Report, per needed env, whether it is installed and has a lockfile.

    Run it before ``mtb.run`` on a new machine; ``mtb.env.install`` installs
    the envs it reports missing.

    Parameters
    ----------
    category : str | list[str] | None
        Integration category (``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``) whose methods to check; ``None`` = every method. A list
        of method ids here is read as ``methods``.
    methods : list[str] | None
        Method ids to check instead of ``category``.
    conda : str | None
        conda/mamba executable that lists the installed envs; ``None`` =
        conda if found, else mamba.
    as_frame : bool
        ``True`` returns a ``pandas.DataFrame`` with the same keys as columns.

    Returns
    -------
    list[dict] or pandas.DataFrame
        One row per env, the env serving the most methods first. Read
        ``env``, ``exists`` and ``has_lock``; all keys are listed in Notes.

    Raises
    ------
    ValueError
        Unknown ``category``; the message lists the valid ones.
    KeyError
        Unknown method id in ``methods``; the message suggests a close
        match, if any.
    TypeError
        ``methods`` given as a bare string.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.doctor("vertical", as_frame=True)
    >>> rows = mtb.env.doctor(methods=["Matilda", "totalVI"])
    >>> [r["env"] for r in rows if not r["exists"]]

    Notes
    -----
    **Keys.** The marks in brackets are the ones ``multibench env doctor``
    prints per env and ``multibench env status`` prints per method.

    - ``env`` / ``methods`` - the env and the selected methods it serves.
    - ``exists`` - the env is installed here (``[x]``).
    - ``has_lock`` - the package ships a lockfile for the env, so
      ``multibench env install --run`` can build it (``[L]`` while missing).
      A missing env without one (``[!]``) needs a packed archive or the
      recipe (``mtb.env.recipe``).
    - ``flavor`` - ``'cpu'`` / ``'gpu'`` / ``'single'`` when the installed
      env came from a packed archive, else ``None``.
    - ``other_env`` - the path of an env of the same name that multibench
      did not install, else ``None``. It is not used; rename or remove it,
      then install.

    **Next step.** ``mtb.env.install(methods, dry_run=False)`` installs the
    missing envs, packed archives first; on the command line that is
    ``multibench env install --methods ... --packed --run``, while plain
    ``multibench env install --run`` builds from lockfiles only. Linux only.

    See Also
    --------
    mtb.env.install : builds or unpacks the envs reported missing.

    mtb.env.status : the same information per method.

    mtb.scan : its ``env_ok`` column uses the same installed-env check.
    """
    category, methods = _category_or_methods(category, methods, "doctor")
    methods = _check_methods(methods)
    have = set(installed_envs(conda))
    if methods is None:
        methods = registry.list_methods(category=category)
    by_env: dict[str, list[str]] = {}
    for m in methods:
        by_env.setdefault(group_for(m), []).append(m)
    rows = [{"env": env, "methods": sorted(ms), "exists": env in have,
             "has_lock": lockfile(env) is not None,
             "flavor": installed_flavor(env, conda) if env in have else None,
             "other_env": (None if env in have
                           else _str_or_none(other_env_prefix(env, conda)))}
            for env, ms in sorted(by_env.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    return _as_frame(rows, as_frame)


def _str_or_none(path) -> str | None:
    return None if path is None else str(path)


_LOCAL_PIP_RE = re.compile(r"@\s*file://|feedstock_root")
# `pip freeze` in a conda env reports the conda name, which pip cannot resolve
_CONDA_TO_PYPI = {
    "python-graphviz": "graphviz",
}
# on no index: restored by <env>.post.sh
_NOT_ON_PYPI = frozenset({"cobolt", "multimap"})
# conda's own tools: `pip freeze` lists them, only conda channels ship them
_CONDA_ONLY_PIP = frozenset({
    "conda", "mamba", "libmambapy", "boa",
    "conda-build", "conda-libmamba-solver", "conda-content-trust",
})
_CUDA_PIN_RE = re.compile(r"==[0-9][^\s]*\+(cu\d+)")


def sanitize_lock(text: str) -> str:
    """Make an exported lockfile rebuildable on a different machine.

    Drops the pip entries that point into the conda-forge build tree
    (``... @ file:///home/conda/feedstock_root/...``; conda already provides
    them) and those no index can supply (:data:`_CONDA_ONLY_PIP`,
    :data:`_NOT_ON_PYPI`); renames conda names to PyPI names
    (:data:`_CONDA_TO_PYPI`); and for a CUDA-local torch pin
    (``torch==2.6.0+cu118``) adds the download.pytorch.org
    ``--extra-index-url`` rather than relaxing the pin.

    The pip block is found by indentation (conda writes six spaces,
    :func:`freeze`'s fallback four). Idempotent.
    """
    out, cuda_tags = [], []
    pip_indent = None
    entry_indent = None
    for ln in text.splitlines():
        stripped = ln.strip()
        indent = len(ln) - len(ln.lstrip())
        if stripped == "- pip:":
            pip_indent, entry_indent = indent, None
            out.append(ln)
            continue
        if pip_indent is not None and stripped:
            if indent <= pip_indent:
                pip_indent = None
            else:
                if entry_indent is None:
                    entry_indent = indent
                if _LOCAL_PIP_RE.search(stripped):
                    continue
                name = re.split(r"[=<>!~\s\[]", stripped[2:].strip())[0].lower()
                if name in _CONDA_ONLY_PIP or name in _NOT_ON_PYPI:
                    continue
                if name in _CONDA_TO_PYPI:
                    ln = ln.replace(name, _CONDA_TO_PYPI[name], 1)
                m = _CUDA_PIN_RE.search(stripped)
                if m and m.group(1) not in cuda_tags:
                    cuda_tags.append(m.group(1))
        out.append(ln)

    if cuda_tags:
        pad = " " * (entry_indent if entry_indent is not None else 6)
        merged, inserted = [], False
        for ln in out:
            merged.append(ln)
            if not inserted and ln.strip() == "- pip:":
                for tag in cuda_tags:
                    url = f"https://download.pytorch.org/whl/{tag}"
                    if not any(url in x for x in out):
                        merged.append(f"{pad}- --extra-index-url {url}")
                inserted = True
        out = merged
    return "\n".join(out) + "\n"


def _has_own_python(env_name: str, conda: str | None = None) -> bool:
    """Does this env contain its own python interpreter?

    In an env without one (``scmb_r``), ``conda run -n <env> pip freeze``
    runs the base env's pip and would record the base env's packages.
    """
    conda = conda or _conda_bin("conda")
    probe = subprocess.run(
        [conda, "run", "-n", env_name, "python", "-c",
         "import sys; print(sys.prefix)"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        return False
    prefix = probe.stdout.strip()
    return bool(prefix) and Path(prefix).name == env_name


def freeze(env_name: str, conda: str | None = None,
           out_dir: Path | str | None = None) -> Path:
    """Capture an existing env to a lockfile ``env_locks/<env>.yml`` (maintainer
    tool): ``conda env export --no-builds`` without the ``prefix:`` line,
    through :func:`sanitize_lock`."""
    conda = conda or _conda_bin("conda")
    # an export of a name that does not exist succeeds, empty: it would
    # overwrite a lockfile with a stub
    if env_name not in _all_env_names(conda):
        raise FileNotFoundError(
            f"no conda env named {env_name!r} on this machine "
            f"(conda env list shows them); freeze captures existing envs only")
    dst_dir = Path(out_dir) if out_dir else _LOCKS_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _run(args):
        return subprocess.run([conda, *args], capture_output=True, text=True)

    def _has_real_deps(text: str) -> bool:
        # real = a conda package beyond python/pip, or a pip: section
        if "- pip:" in text:
            return True
        in_deps = False
        for ln in text.splitlines():
            if ln.startswith("dependencies:"):
                in_deps = True
                continue
            if in_deps and ln.lstrip().startswith("- "):
                name = re.split(r"[=<>\s]", ln.split("- ", 1)[1].strip())[0]
                if name not in ("python", "pip"):
                    return True
        return False

    # a full export, else the explicit specs when it fails or lists nothing
    export = _run(["env", "export", "-n", env_name, "--no-builds"])
    if export.returncode == 0 and _has_real_deps(export.stdout):
        body = export.stdout
    else:
        history = _run(["env", "export", "-n", env_name, "--from-history"])
        body = history.stdout if history.returncode == 0 else (
            f"name: {env_name}\nchannels:\n  - conda-forge\ndependencies:\n  - python\n")
    lines = [ln for ln in body.splitlines() if not ln.startswith("prefix:")]
    # pip packages the export missed, from `pip freeze`
    if "- pip:" not in "\n".join(lines) and _has_own_python(env_name, conda):
        pip = _run(["run", "-n", env_name, "pip", "freeze"]).stdout
        pip_pkgs = [ln.strip() for ln in pip.splitlines()
                    if ln.strip() and not ln.startswith("-e ")]
        if pip_pkgs:
            if not any(l.startswith("dependencies:") for l in lines):
                lines.append("dependencies:")
            lines += ["  - pip", "  - pip:"] + [f"    - {p}" for p in pip_pkgs]
    dst = dst_dir / f"{env_name}.yml"
    dst.write_text(sanitize_lock("\n".join(lines)))
    return dst


# --- recipe/lockfile provisioning entry points -----------------------------
def create(method: str, env_name: str | None = None, conda: str | None = None,
           dry_run: bool = True, *, force: bool = False) -> list[list[str]]:
    """Provision the env a method runs in, from its lockfile, else its recipe.

    ``env_name`` builds into another env. ``dry_run=False`` runs the
    commands; on a non-Linux host it raises ``RuntimeError`` unless ``force``.

    Returns
    -------
    list of list of str
        The argv commands.
    """
    method = registry.check_method(method)
    target = env_name or group_for(method)
    if lockfile(target) is not None:
        return create_env(target, conda=conda, dry_run=dry_run, force=force)
    cmds = create_commands(method, env_name=env_name, conda=conda)
    if not dry_run:
        _require_linux(force)
        _build(env_name or default_env_name(method), cmds, conda, "recipe")
    return cmds


def create_group(group: str, env_name: str | None = None, conda: str | None = None,
                 dry_run: bool = True, *, force: bool = False) -> list[list[str]]:
    """Provision a group env (:func:`groups`), from its lockfile, else its recipe.

    As :func:`create`; an unknown ``group`` raises ``KeyError``.

    Returns
    -------
    list of list of str
        The argv commands.
    """
    target = env_name or group
    if lockfile(target) is not None:
        return create_env(target, conda=conda, dry_run=dry_run, force=force)
    cmds = group_create_commands(group, env_name=env_name, conda=conda)
    if not dry_run:
        _require_linux(force)
        _build(env_name or group, cmds, conda, "recipe")
    return cmds


def _run_all(cmds: list[list[str]]) -> None:
    # else pip skips a package that ~/.local provides, which run() hides
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    if cmds and shutil.which(cmds[0][0]) is None:
        raise RuntimeError(
            "Neither conda nor mamba is on this machine. Method environments "
            "cannot be built here. They need Linux with conda, and mamba is "
            "recommended. The installation guide shows the steps. The method "
            "list, stored results and figures work without them."
        )
    for c in cmds:
        proc = subprocess.run(c, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"env command failed: {' '.join(c)}\nstderr tail:\n{proc.stderr[-2000:]}"
            )
    _conda_prefixes.cache_clear()
