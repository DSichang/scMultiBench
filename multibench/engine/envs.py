"""Method environments: packed archives, lockfiles, recipes and shared envs.

scMultiBench wraps ~40 separately developed tools whose pinned dependencies
conflict (TF 2.4 vs 2.8, scvi <0.20 vs latest, py3.7 vs 3.10, R vs Python), so
no single conda env can host them all. Compatible methods share an env:
``env_groups.yaml`` maps each method to the env that serves it, and ``plan()``
lists the few envs a selection of methods needs.

An env is unpacked from a prebuilt conda-pack archive (``install_packed``) or
rebuilt from its committed lockfile (``create_env``); the hand-written
``env_spec`` recipe (``create_commands`` / ``environment_yml``) is the fallback
for an env without a lockfile.
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
    """``dir(mtb.env)`` shows the five public entry points and the dunders
    (PEP 562). Everything else in this module - the env-name resolvers, the
    lockfile/recipe builders, ``create_all`` / ``install_packed`` ... - stays
    importable for the CLI and the tests but is not advertised."""
    return sorted(n for n in globals() if n in __all__ or n.startswith("__"))


# --- host platform --------------------------------------------------------
def host_platform_problem() -> str | None:
    """Why method environments cannot be built on this host, or ``None``.

    Every method env is a linux-64 conda env: the packed archives are
    conda-pack snapshots of linux-64 envs and the lockfiles pin linux-only
    packages (``libgcc-ng`` ...), so on macOS or Windows a build fails on ELF
    binaries after a multi-GB download, or in the solver.

    Returns
    -------
    str or None
        ``None`` when ``sys.platform == "linux"`` (WSL counts as Linux);
        otherwise one sentence naming the requirement and this host, e.g.
        ``"method environments are linux-64 conda envs (packed archives +
        lockfiles); this host is darwin/arm64"``. Module-level so tests can
        monkeypatch it.
    """
    if sys.platform == "linux":
        return None
    return (f"method environments are linux-64 conda envs (packed archives + "
            f"lockfiles); this host is {sys.platform}/{_platform.machine() or '?'}")


def _require_linux(force: bool) -> None:
    """Raise ``RuntimeError`` before any download or build on a non-Linux host.

    ``force=True`` (``--force`` on the CLI) skips the check.
    """
    problem = None if force else host_platform_problem()
    if problem:
        raise RuntimeError(
            f"{problem} - method envs cannot be built here. Run methods on a "
            f"Linux host (the registry, stored results, scan's file gate, "
            f"evaluate and plot all work on this machine); pass force=True / "
            f"--force to try anyway.")


#: What the ``difficulty`` tag of ``env_specs.yaml`` (shown by ``env status``
#: and :func:`status`) means. The tag describes how hard the env is to build
#: from its recipe, not how well the method works.
DIFFICULTY = {
    "easy": "modern python/torch stack, builds from the lockfile without surprises",
    "old-scvi": "pins an old scvi-tools (<0.20) / old anndata - needs its own env, "
                "cannot share the modern torch env",
    "old-tensorflow": "pins TensorFlow 1.x/2.4 - needs its own env with matching CUDA",
    "R": "an R env (Seurat/MOFA2/rliger ...); R packages installed by "
         "install.packages() are restored by the env's post-install script",
    "verified": "env built from the lockfile on a fresh machine and the method "
                "ran end-to-end on its reference dataset",
    "blocked-script": "the upstream script itself cannot run unmodified from the "
                      "public checkout (see method_info(m)['setup_hint'] and scan's "
                      "files_reason); the env builds, and the benchmark host ran the "
                      "method only with a local shim",
    "unknown": "no env_spec recipe declared for the method",
}

#: The ``*`` suffix ``env status`` appends to the difficulty tag.
VERIFIED_STAR = ("* = verified_working: the env ran the method end-to-end on "
                 "its reference dataset")

#: The symbol set of every env listing (``env status`` and ``env doctor``):
#: installed / missing-with-lockfile / missing-without-lockfile.
MARK_LEGEND = ("[x]=installed  [L]=missing, lockfile ready (run `multibench env "
               "install --run`)  [!]=missing, no lockfile")


def env_mark(exists: bool, has_lock: bool) -> str:
    """The one-character mark of :data:`MARK_LEGEND` for an env row.

    Parameters
    ----------
    exists : bool
        The env is installed here.
    has_lock : bool
        ``env_locks/<env>.yml`` is shipped, so ``env install --run`` can build it.

    Returns
    -------
    str
        ``"x"``, ``"L"`` or ``"!"``.
    """
    return "x" if exists else ("L" if has_lock else "!")


_SIZES_JSON = Path(__file__).resolve().parent / "packed_sizes.json"


@functools.lru_cache(maxsize=1)
def packed_sizes() -> dict:
    """Byte sizes of the published packed archives, per env.

    Read from the shipped ``engine/packed_sizes.json``, a snapshot that
    ``tools/packed_sizes.py`` writes from one HEAD request per URL in
    ``packed_urls.json``. No request is made at runtime: compute nodes can be
    offline and Zenodo rate-limits. Keys starting with ``_`` are metadata,
    not envs.

    Returns
    -------
    dict
        ``{env: {"archive_bytes": int | None, "unpacked_bytes": int | None}}``;
        ``None`` means "not measured yet". ``{}`` when the file is absent or
        unreadable.
    """
    if not _SIZES_JSON.is_file():
        return {}
    try:
        data = _json.loads(_SIZES_JSON.read_text())
    except Exception:  # noqa: BLE001 - a broken table means "sizes unknown"
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items()
            if not str(k).startswith("_") and isinstance(v, dict)}


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

#: One-word record of which archive an unpacked prefix came from, written
#: by :func:`install_packed` as ``<prefix>/.multibench_flavor`` and read by
#: :func:`installed_flavor` (``env status`` / ``doctor`` / ``plan``).
FLAVOR_FILE = ".multibench_flavor"

#: The kernel's NVIDIA driver record; its presence means a GPU driver is
#: loaded even where ``nvidia-smi`` is not on PATH. Module-level so tests can
#: point it at a temporary file.
_NVIDIA_PROC = Path("/proc/driver/nvidia/version")


def check_flavor(flavor) -> str:
    """``flavor`` when it is one of :data:`FLAVORS`, else ``ValueError``.

    Parameters
    ----------
    flavor : str
        ``'auto'``, ``'cpu'`` or ``'gpu'``.

    Returns
    -------
    str
        The value as given.

    Raises
    ------
    ValueError
        ``"flavor='cuda': choose one of 'auto', 'cpu', 'gpu'"`` - listing the
        accepted values, so a typo never reaches a download.
    """
    if flavor not in FLAVORS:
        raise ValueError(f"flavor={flavor!r}: choose one of "
                         + ", ".join(repr(f) for f in FLAVORS))
    return flavor


@functools.lru_cache(maxsize=1)
def host_has_gpu() -> bool:
    """Is an NVIDIA GPU visible on this host?

    ``True`` when ``nvidia-smi`` is on ``PATH`` and ``nvidia-smi -L`` exits
    0 printing at least one line, or when ``/proc/driver/nvidia/version``
    exists (the driver is loaded but the tool is not on ``PATH``). Nothing
    else counts: a torch install, ``CUDA_VISIBLE_DEVICES`` or a CUDA
    library on disk say nothing about the machine. Cached per process
    (``host_has_gpu.cache_clear()`` re-probes); module-level so tests
    monkeypatch it either way.

    Returns
    -------
    bool
        ``True`` when a GPU is visible; ``False`` on a CPU-only host - the
        ``flavor='auto'`` decision of :func:`install_packed`.
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
    """``'cpu'`` or ``'gpu'`` for a ``flavor=`` value (``'auto'`` decided here).

    Parameters
    ----------
    flavor : str
        One of :data:`FLAVORS` (``ValueError`` otherwise).

    Returns
    -------
    str
        ``'cpu'`` / ``'gpu'`` as given, or for ``'auto'``: ``'cpu'`` when
        :func:`host_has_gpu` is ``False``, else ``'gpu'``.
    """
    check_flavor(flavor)
    if flavor == "auto":
        return "gpu" if host_has_gpu() else "cpu"
    return flavor


def archive_key(env: str, flavor: str) -> str:
    """The ``packed_urls.json`` / ``packed_sizes.json`` key of an env's archive.

    ``'<env>-cpu'`` for ``'cpu'``, ``'<env>'`` for ``'gpu'`` - a pure name
    rule; :func:`archive_for` says whether that archive is published.
    """
    return f"{env}-cpu" if flavor == "cpu" else env


def archive_for(env: str, flavor: str = "auto", *, manifest: dict | None = None,
                sizes: dict | None = None) -> tuple[str, str]:
    """Which published archive serves ``env`` for a flavour: ``(key, flavour)``.

    Parameters
    ----------
    env : str
        The env name (:func:`group_for`), e.g. ``'env_sciPENN'``.
    flavor : str
        One of :data:`FLAVORS`; ``'auto'`` resolves through
        :func:`resolve_flavor`.
    manifest, sizes : dict, keyword-only, optional
        Stand-ins for :func:`packed_manifest` / :func:`packed_sizes`.

    Returns
    -------
    tuple of (str, str)
        ``('<env>-cpu', 'cpu')`` when the CPU archive is published: its key
        is in ``packed_urls.json`` and ``packed_sizes.json`` carries a
        measured ``archive_bytes`` for it. A ``null`` size marks an archive
        that is not uploaded yet (the size is recorded after the upload);
        picking it would send a CPU host to a 404 and on to a lockfile build.
        Otherwise ``('<env>', 'gpu')`` - the CUDA build every env has.
    """
    wanted = resolve_flavor(flavor)
    if wanted == "cpu":
        key = archive_key(env, "cpu")
        manifest = packed_manifest() if manifest is None else manifest
        sizes = packed_sizes() if sizes is None else sizes
        if key in manifest and (sizes.get(key) or {}).get("archive_bytes") is not None:
            return key, "cpu"
    return env, "gpu"


def _cpu_fallback_warning(env: str, manifest: dict, sizes: dict) -> str:
    """The one ``UserWarning`` a CPU request that lands on the GPU build emits."""
    size = _gb((sizes.get(env) or {}).get("archive_bytes"))
    msg = f"no CPU archive for {env}; installing the GPU build ({size})"
    key = archive_key(env, "cpu")
    if key in manifest:
        msg += (f" - {key} is listed in packed_urls.json but has no measured size "
                f"in packed_sizes.json (not published yet; tools/packed_sizes.py "
                f"records it after the upload)")
    return msg


def installed_flavor(env: str, conda: str | None = None) -> str | None:
    """Which archive flavour an installed env came from, or ``None``.

    Parameters
    ----------
    env : str
        The env name (:func:`group_for`).
    conda : str, optional
        conda/mamba executable :func:`env_prefix` may ask.

    Returns
    -------
    str or None
        The word in ``<prefix>/.multibench_flavor`` (``'cpu'`` / ``'gpu'``,
        written by :func:`install_packed`); ``None`` when the env is not
        installed, was built from a lockfile or by conda (no record), or the
        file holds anything else.
    """
    prefix = env_prefix(env, conda)
    if prefix is None:
        return None
    try:
        words = (prefix / FLAVOR_FILE).read_text().split()
    except OSError:
        return None
    return words[0] if words and words[0] in ("cpu", "gpu") else None


_GROUPS_YAML = Path(__file__).resolve().parent / "env_groups.yaml"
# Committed per-env lockfiles (`conda env export --no-builds`: versions + pip
# section). create_env/create_all rebuild envs from them under the name run()
# uses; the hand-written recipe (create_commands) is only the fallback.
_LOCKS_DIR = Path(__file__).resolve().parent / "env_locks"


# --- recipes ---------------------------------------------------------------
def recipe(method: str) -> dict:
    """Return the hand-written environment recipe of a method.

    The readable alternative to the lockfile: which Python, conda and pip
    packages the method's env needs, with caveats.

    Parameters
    ----------
    method : str
        Registry method id, e.g. ``'Matilda'``.

    Returns
    -------
    dict
        The method's ``env_spec``; ``{}`` when it declares none. Read
        ``conda_packages``, ``pip_packages`` and ``caveats``; all keys are
        listed in Notes.

    Raises
    ------
    KeyError
        Unknown method id; the message suggests the closest id.

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

    **Source.** The recipes live in ``engine/env_specs.yaml``.

    **Command line.** ``multibench env recipe METHOD`` prints the recipe as
    conda/pip commands and ``multibench env yml METHOD`` as an
    ``environment.yml``; both name the env the way ``scan`` and ``run``
    expect it. The reproducible build is the lockfile or packed archive
    (``mtb.env.install``).

    See Also
    --------
    mtb.env.install : the reproducible install path (packed archive or lockfile).
    mtb.env.status : ``has_recipe`` and ``difficulty`` per method.
    """
    return registry.get(method).env_spec or {}


def own_env_name(method: str) -> str:
    """The method's own (singleton) env name, ``scmb_<method>``.

    :func:`group_for` falls back to it when the method has no ``method_env``
    override and is in no shared group of ``env_groups.yaml``. The recipe
    builders (:func:`create_commands` / :func:`environment_yml`) use it only
    when it is passed explicitly.
    """
    return f"scmb_{method.lower()}"


def default_env_name(method: str) -> str:
    """The conda env name every entry point uses for ``method``.

    :func:`group_for` after validating the id (``KeyError`` with a
    did-you-mean hint otherwise): the ``method_env`` override from
    ``env_groups.yaml`` (``Matilda`` -> ``matilda``), else the shared group env
    the method is a member of (``SCALEX`` -> ``scmb_torch``), else its own
    ``scmb_<method>`` env (:func:`own_env_name`). ``scan()['env']``,
    ``method_info(m)['env']``, ``run()``, ``env doctor`` / ``plan`` /
    ``install`` / ``create`` and ``env recipe`` / ``yml`` all use this name,
    so a recipe pasted into a build job produces an env the package
    recognises.
    """
    registry.check_method(method)
    return group_for(method)


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
    # channels first (newer mamba rejects positional packages placed after `-c`).
    create = [conda, "create", "-y", "-n", env_name]
    for ch in spec.get("conda_channels", []):
        create += ["-c", ch]
    if spec.get("python_version"):
        create.append(f"python={spec['python_version']}")
    create += _conda_packages(spec)
    pip_git = list(spec.get("pip_git", []))
    pip_pkgs = list(spec.get("pip_packages", [])) + pip_git
    if pip_pkgs:
        create.append("pip")   # ensure pip is in the env for the pip-install step
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
    """The recipe's conda packages, minus a ``python`` pin already emitted.

    Some recipes carry ``python_version`` and also list ``python=3.7`` in
    ``conda_packages``; ``python_version`` wins and the duplicate is dropped,
    so the create line names python once.
    """
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
    """Build the create+install commands for a single method's env.

    ``env_name`` defaults to :func:`default_env_name` - the env ``scan`` /
    ``run`` / ``env doctor`` expect for ``method`` - so the commands build an
    env the package recognises; pass :func:`own_env_name` (or any name) to
    build somewhere else. The recipe is the hand-written ``env_spec``; the
    reproducible path is the lockfile (:func:`create`).
    """
    r = recipe(method)
    if not r:
        raise ValueError(f"no env_spec recipe declared for {method!r}")
    return _install_commands(r, env_name or default_env_name(method), conda)


def environment_yml(method: str, env_name: str | None = None) -> str:
    """Render an ``environment.yml`` string for a single method's recipe.

    ``env_name`` (the ``name:`` line) defaults to :func:`default_env_name`,
    the env ``scan`` / ``run`` / ``env doctor`` expect for ``method``.
    """
    r = recipe(method)
    if not r:
        raise ValueError(f"no env_spec recipe declared for {method!r}")
    return _environment_yml(r, env_name or default_env_name(method))


# --- shared group environments --------------------------------------------
@functools.lru_cache(maxsize=1)
def _merged_groups() -> dict:
    if _GROUPS_YAML.exists():
        return (yaml.safe_load(_GROUPS_YAML.read_text()) or {}).get("groups", {})
    return {}


@functools.lru_cache(maxsize=1)
def _method_env() -> dict:
    """Explicit ``{method: conda env}`` overrides from ``env_groups.yaml``.

    They take precedence over membership in ``groups`` (:func:`group_for`).
    """
    if _GROUPS_YAML.exists():
        return (yaml.safe_load(_GROUPS_YAML.read_text()) or {}).get("method_env", {}) or {}
    return {}


def groups() -> dict:
    """All install groups: shared merged envs + singleton (own-env) methods.

    Returns {env_name: spec}; merged groups serve several methods, singletons one.
    """
    merged = _merged_groups()
    covered = {m for g in merged.values() for m in g.get("members", [])}
    out = {name: {**g, "shared": True} for name, g in merged.items()}
    # method_env overrides: ensure each target env is a group too
    for method, env in _method_env().items():
        g = out.setdefault(env, {"members": [], "shared": True})
        g.setdefault("members", [])
        if method not in g["members"]:
            g["members"].append(method)
        covered.add(method)
    for s in registry.load():
        if s.id not in covered:
            out[own_env_name(s.id)] = {
                **(s.env_spec or {}), "members": [s.id], "shared": False,
            }
    return out


def group_for(method: str) -> str:
    """The env name that serves this method (an explicit override, a shared
    group, or its own ``scmb_<method>`` env). :func:`default_env_name` is the
    same answer after validating the method id."""
    override = _method_env()
    if method in override:
        return override[method]
    for name, g in _merged_groups().items():
        if method in g.get("members", []):
            return name
    return own_env_name(method)


def group_create_commands(group: str, env_name: str | None = None,
                          conda: str | None = None) -> list[list[str]]:
    """Create+install commands for a group env (shared or singleton).

    ``env_name`` overrides the target env (default: the group name), e.g. a
    scratch env to validate a recipe without touching the real one.
    """
    spec = groups().get(group)
    if spec is None:
        raise KeyError(f"unknown group {group!r}; see envs.groups()")
    return _install_commands(spec, env_name or group, conda)


def _check_methods(methods):
    """Raise ``KeyError`` for an unknown id in ``methods``.

    Without it a typo resolves through :func:`group_for` to a made-up
    ``scmb_<typo>`` env and gets a row instead of an error. Delegates to
    :func:`registry.check_method` so the error carries the same did-you-mean
    hint as every other entry point (``'Stabmap'`` -> ``'StabMap'``).
    """
    if not methods:
        return
    from . import registry
    for m in methods:
        registry.check_method(m)


def plan(category: str | None = None, methods: list[str] | None = None, *,
         as_frame: bool = False):
    """List the conda envs a set of methods needs, one row per env.

    Methods that share an env collapse into one row, so a whole category
    needs only a few envs.

    Parameters
    ----------
    category : str | None
        Integration category whose methods to cover; ``None`` = every method.
    methods : list[str] | None
        Method ids to cover instead of ``category``.
    as_frame : bool, keyword-only
        ``True`` returns a ``pandas.DataFrame`` with the same keys as columns.

    Returns
    -------
    list[dict] or pandas.DataFrame
        One row per env, largest first. Read ``env``, ``methods`` and
        ``availability``; all keys are listed in Notes.

    Raises
    ------
    ValueError
        Unknown ``category``; the message lists the four.
    KeyError
        Unknown method id in ``methods``.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.plan("vertical", as_frame=True)[["env", "methods", "availability"]]
    >>> mtb.env.plan(methods=["Matilda", "totalVI", "scMoMaT"])

    Notes
    -----
    **Keys.**

    - ``env`` - the conda env name, the one ``run`` activates.
    - ``shared`` - the env is a shared group of ``env_groups.yaml``.
    - ``methods`` - the selected methods this env serves, sorted.
    - ``availability`` - ``'public'``, or ``'benchmark-host-only'`` when
      none of those methods has a published script (SPIRAL): the env
      builds, but the method still cannot run off the benchmark host.
    - ``flavor`` - ``'cpu'`` / ``'gpu'`` when the env is installed here from
      a packed archive, else ``None``.

    **Selection.** ``methods`` takes precedence over ``category``. A method
    typo raises ``KeyError`` with a did-you-mean hint.

    **Command line.** ``multibench env plan`` prints the same rows with each
    archive's download size and unpacked size on disk (from the shipped
    ``packed_sizes.json``; ``?`` = not measured) and a total line.

    See Also
    --------
    mtb.env.install : builds or unpacks exactly these envs.
    mtb.env.doctor : whether each of these envs exists here.
    """
    _check_methods(methods)
    if methods is None:
        methods = registry.list_methods(category=category)
    shared = _merged_groups()
    buckets: dict[str, list[str]] = {}
    for m in methods:
        buckets.setdefault(group_for(m), []).append(m)
    rows = [
        {"env": env, "shared": env in shared, "methods": sorted(ms),
         "availability": ("benchmark-host-only"
                          if all(registry.get(m).availability != "public" for m in ms)
                          else "public"),
         "flavor": installed_flavor(env)}
        for env, ms in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    return _as_frame(rows, as_frame)


# --- env existence / status -----------------------------------------------
PACKED_URL = "https://github.com/DSichang/scMultiBench/releases/download/envs-v1"


def _envs_dir(conda_bin: str) -> Path | None:
    """Where this conda/mamba keeps named environments, or None.

    The key names in ``info --json`` are not stable across tools or major
    versions - conda answers ``envs_dirs``/``root_prefix``, mamba >= 2 answers
    ``"envs directories"``/``"base environment"`` - and ``info --base`` is prose
    in mamba and a bare path in conda. So: accept every spelling, and when the
    tool answers with none of them, fall back to the layout every installation
    shares regardless of schema (the launcher lives at ``<root>/bin/<tool>``).
    Candidates are tried in order and the first writable one wins, because an
    env unpacked into a directory conda does not search is invisible by name.
    """
    import json as _json
    import os

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
    """Path of the conda (else mamba) binary on PATH, or ``None``.

    The existence test every conda-free path keys on: prefix discovery,
    ``installed_envs`` and ``install`` ask this, never ``_conda_bin`` (which
    returns a bare name even when nothing is installed).
    """
    return shutil.which("conda") or shutil.which("mamba")


@functools.lru_cache(maxsize=4)
def _conda_prefixes(conda: str) -> tuple:
    """``(prefix, ...)`` that ``<conda> env list --json`` reports (cached per
    binary; ``()`` when the tool is absent or answers garbage). The cache is
    cleared by the two places this module creates envs, so an env built in
    the same process is seen."""
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


def env_prefix(env: str, conda: str | None = None) -> Path | None:
    """The on-disk prefix of a method environment, or ``None``.

    Parameters
    ----------
    env : str
        The env name (:func:`group_for`), e.g. ``'matilda'``.
    conda : str, optional
        conda/mamba executable to ask when the prefix is not under
        ``envs_dir``; default: the one on PATH, if any.

    Returns
    -------
    Path or None
        ``<config.DEFAULT.envs_dir>/<env>`` when that directory contains
        ``bin/`` (a conda-pack archive unpacked there, or a conda-built env
        in the same directory) - no conda needed; else, when a conda/mamba
        binary is found, the prefix ``conda env list`` reports for that name;
        else ``None``. This is what "installed" means everywhere
        (:func:`installed_envs`, :func:`doctor`, ``scan()['env_ok']``) and
        what the runner's prefix mode activates.
    """
    cand = Path(config.DEFAULT.envs_dir) / env
    if (cand / "bin").is_dir():
        return cand
    bin_ = conda or _find_conda()
    if bin_ is None:
        return None
    for p in _conda_prefixes(bin_):
        if os.path.basename(p.rstrip("/")) == env:
            return Path(p)
    return None


def install_packed(env: str, *, envs_dir: Path | str | None = None,
                   conda: str | None = None, force: bool = False,
                   flavor: str = "auto") -> bool:
    """Provision ``env`` from a prebuilt conda-pack archive, if one is published.

    Downloads the archive named in ``packed_urls.json`` (else
    ``<PACKED_URL>/<env>.tar.gz``), extracts it into ``<envs_dir>/<env>``
    and runs the archive's own ``bin/conda-unpack`` with ``<prefix>/bin``
    first on ``PATH`` (the archive carries its own python) to rewrite the
    embedded prefixes. No conda binary is needed at any step: conda-pack
    archives are relocatable, and the runner activates the prefix directly.
    Unlike a lockfile build there is no dependency solve, only a download.

    Parameters
    ----------
    env : str
        The conda env name (:func:`group_for`), e.g. ``'matilda'``.
    envs_dir : path, keyword-only, optional
        Where the prefix goes; default :attr:`multibench.config.Config.envs_dir`
        (``MULTIBENCH_ENVS_DIR``, else conda's envs dir, else
        ``~/.cache/multibench/envs``) - or, when only ``conda`` is given,
        that tool's envs dir.
    conda : str, keyword-only, optional
        conda/mamba executable whose envs dir to unpack into when
        ``envs_dir`` is not given. Not required.
    force : bool, keyword-only
        The archives are linux-64; on any other host ``RuntimeError`` is
        raised before the download unless ``force=True``.
    flavor : str, keyword-only
        Which archive: ``'gpu'`` - the ``'<env>'`` archive (the CUDA build
        every env has); ``'cpu'`` - the ``'<env>-cpu'`` archive (the same
        env without the CUDA libraries, 3-4x smaller) when it is published
        (:func:`archive_for`), else the GPU build with one ``UserWarning``
        ``"no CPU archive for <env>; installing the GPU build (<size>)"``;
        ``'auto'`` (default) - ``'cpu'`` when :func:`host_has_gpu` is
        ``False``, else ``'gpu'``. Whatever the flavour, the prefix is
        ``<envs_dir>/<env>`` - the env name never changes, so the runner
        and the registry are untouched - and the flavour installed is
        recorded in ``<prefix>/.multibench_flavor`` (:data:`FLAVOR_FILE`,
        one word) for ``env status`` / ``doctor``. Anything else:
        ``ValueError`` listing the three.

    Returns
    -------
    bool
        ``True`` on success (or when the prefix already exists), ``False``
        when no archive exists for this env or the unpack failed - the caller
        falls back to the lockfile build.
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
        return True
    # A shipped manifest maps env -> archive URL, so archives can live where
    # their size dictates (GitHub release assets up to 2 GiB, Zenodo beyond);
    # envs without an entry fall back to the release-asset convention.
    manifest, sizes = packed_manifest(), packed_sizes()
    key, installed = archive_for(env, flavor, manifest=manifest, sizes=sizes)
    if resolve_flavor(flavor) == "cpu" and installed == "gpu":
        warnings.warn(_cpu_fallback_warning(env, manifest, sizes), UserWarning,
                      stacklevel=2)
    url = manifest.get(key) or f"{PACKED_URL}/{key}.tar.gz"
    try:
        tgz, _ = urllib.request.urlretrieve(url)
    except urllib.error.HTTPError:
        return False
    print(f"[env] unpacking prebuilt {env} ({installed} build) -> {dest} ...", flush=True)
    from ..data.fetch import safe_extract
    part = dest.with_name(dest.name + ".partial")
    try:
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with tarfile.open(tgz) as t:
            safe_extract(t, part)
        if not (part / "bin").is_dir():
            raise RuntimeError("archive did not contain bin/")
        # the env must carry its final path before conda-unpack rewrites
        # prefixes, so move first, then unpack
        part.rename(dest)
        _conda_unpack(dest)
        # the record env status / doctor / plan show; written last, so a
        # prefix that failed to unpack never claims a flavour
        (dest / FLAVOR_FILE).write_text(installed + "\n")
        _conda_prefixes.cache_clear()
        return True
    except Exception as e:  # noqa: BLE001 - degrade to the lockfile build
        shutil.rmtree(part, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[env] prebuilt {env} failed ({type(e).__name__}: "
              f"{str(e)[:120]}); falling back to the lockfile build",
              flush=True)
        return False


def _conda_unpack(prefix: Path) -> None:
    """Run ``<prefix>/bin/conda-unpack`` (a no-op when the archive has none).

    ``<prefix>/bin`` goes first on ``PATH`` so the script's ``python``
    shebang resolves to the env's own interpreter - a bare machine (Colab
    without conda) has no other. A script that lost its executable bit is
    handed to that interpreter explicitly.
    """
    unpack = prefix / "bin" / "conda-unpack"
    if not unpack.exists():
        return
    py = prefix / "bin" / "python"
    argv = [str(unpack)] if os.access(unpack, os.X_OK) else [str(py), str(unpack)]
    env = {**os.environ,
           "PATH": f"{prefix / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
    subprocess.run(argv, check=True, capture_output=True, env=env)


def installed_envs(conda: str | None = None) -> list[str]:
    """Names of the method environments present on this machine.

    An env counts as installed when :func:`env_prefix` finds it: a
    ``<envs_dir>/<env>/bin`` prefix on disk (no conda needed - this is what
    ``install_packed`` produces on a conda-free host), or an env
    ``conda env list --json`` reports when a conda/mamba binary is found.

    Parameters
    ----------
    conda : str, optional
        conda/mamba executable to ask; default: the one on PATH, if any.

    Returns
    -------
    list of str
        Basenames of the prefixes (path-only entries such as basilisk caches
        become clean names rather than raw paths), prefixes on disk first,
        no duplicates.
    """
    names = list(_prefixes_on_disk(Path(config.DEFAULT.envs_dir)))
    bin_ = conda or _find_conda()
    if bin_ is not None:
        for p in _conda_prefixes(bin_):
            name = os.path.basename(str(p).rstrip("/"))
            if name and name not in names:
                names.append(name)
    return names


def status(conda: str | None = None, *, as_frame: bool = False):
    """Report, per method, whether its conda env is installed on this machine.

    Parameters
    ----------
    conda : str | None
        conda/mamba executable that lists the installed envs; ``None`` =
        conda if found, else mamba.
    as_frame : bool, keyword-only
        ``True`` returns a ``pandas.DataFrame`` with the same keys as columns.

    Returns
    -------
    list[dict] or pandas.DataFrame
        One row per registry method. Read ``method``, ``env``, ``exists``
        and ``has_lock``; all keys are listed in Notes.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.env.status(as_frame=True)
    >>> df.loc[~df.exists, ["method", "env", "has_lock"]]     # what is missing
    >>> df.loc[df.method == "Matilda"].iloc[0].to_dict()

    Notes
    -----
    **Keys.**

    - ``method`` - the registry id.
    - ``env`` - the env the package uses for the method, the name ``scan``
      and ``run`` use (``mtb.env.default_env_name``).
    - ``group`` - the same name as ``env``.
    - ``own_env`` - the singleton name ``scmb_<method>`` (lower-case); the
      method also counts as installed when this env exists.
    - ``exists`` - ``env`` or ``own_env`` is installed here.
    - ``has_lock`` - a shipped lockfile can build ``env`` (the ``[L]`` mark,
      shared with ``doctor``).
    - ``difficulty`` - a tag for how hard the env is to build (below).
    - ``verified_working`` - the env ran the method end-to-end on its
      reference dataset (the ``*`` after the tag in ``env status``).
    - ``has_recipe`` - the method declares a recipe (``mtb.env.recipe``).
    - ``flavor`` - ``'cpu'`` / ``'gpu'`` when the installed env came from a
      packed archive, else ``None``.

    **Difficulty tags.** They describe how hard the env is to build from its
    recipe, not how well the method works (full text in
    ``mtb.env.DIFFICULTY``):

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

    **What counts as installed.** A prefix ``<envs_dir>/<env>`` with a
    ``bin/`` directory (no conda needed), or an env that the conda/mamba
    found reports in ``env list``. Prefixes under ``envs_dir`` are probed
    whether or not ``conda`` is given.

    See Also
    --------
    mtb.env.doctor : the same information per env rather than per method.
    mtb.env.install : builds or unpacks the missing envs.
    """
    have = set(installed_envs(conda))
    out = []
    for s in registry.load():
        r = s.env_spec or {}
        grp = group_for(s.id)
        own = own_env_name(s.id)
        exists = grp in have or own in have
        out.append({
            "method": s.id, "env": grp, "group": grp, "own_env": own,
            "exists": exists,
            "has_lock": lockfile(grp) is not None,
            "difficulty": r.get("difficulty", "unknown"),
            "verified_working": bool(r.get("verified_working", False)),
            "has_recipe": bool(r),
            "flavor": (installed_flavor(grp, conda) or installed_flavor(own, conda)
                       if exists else None),
        })
    return _as_frame(out, as_frame)


# --- lockfile-based provisioning (the reproducible install path) -----------
def lockfile(env_name: str) -> Path | None:
    """Path of the committed lockfile ``env_locks/<env_name>.yml``, or ``None``.

    Written by :func:`freeze`; :func:`create_env` rebuilds the env from it.
    """
    p = _LOCKS_DIR / f"{env_name}.yml"
    return p if p.exists() else None


def required_envs(category: str | None = None,
                  methods: list[str] | None = None) -> list[str]:
    """The distinct conda envs needed to run the given methods (or all).

    The env names ``run()`` activates (:func:`group_for`), i.e. what a fresh
    machine must provision.
    """
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

    ``conda env create`` hands the pip section to ``pip install -r``, which
    re-resolves it. The section is a full ``pip freeze`` closure, so every
    transitive dependency is already pinned, and re-resolving fails with
    ``ResolutionImpossible`` whenever the recorded env holds a combination pip
    considers inconsistent - common in envs built up incrementally.

    Installing the closure with ``--no-deps`` reproduces the env as recorded.
    ``--no-deps`` cannot be expressed inside a requirements file, so the two
    halves are installed separately.
    """
    conda_lines, pip_lines = [], []
    pip_indent = None
    for ln in text.splitlines():
        stripped = ln.strip()
        indent = len(ln) - len(ln.lstrip())
        if stripped == "- pip:":
            pip_indent = indent
            continue                       # drop the header from the conda half
        if pip_indent is not None and stripped:
            if indent <= pip_indent:
                pip_indent = None          # dedented back out
            else:
                pip_lines.append(stripped[2:].strip() if stripped.startswith("- ")
                                 else stripped)
                continue
        conda_lines.append(ln)
    return "\n".join(conda_lines) + "\n", pip_lines


def _materialise_split(env_name: str, lock):
    """Write the conda-only YAML and pip requirements a two-phase install needs."""
    # an installed package stays read-only (site-packages): build files go to
    # the temp dir; a repo checkout keeps them under env_locks/.build
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
    """Path to the committed post-install script for an env, or None.

    Restores what a lockfile cannot: packages installed inside the env by a
    language-native installer (``install.packages()``, ``install_github()``),
    which ``conda env export`` never records (rliger in ``scmb_r``); pip
    packages that are on no index (:data:`_NOT_ON_PYPI`); and packages the
    working env loads from a local checkout, such as the editable installs
    :func:`freeze` skips (matilda, scMVP).
    """
    p = _LOCKS_DIR / f"{env_name}.post.sh"
    return p if p.is_file() else None


def create_env(env_name: str, conda: str | None = None,
               dry_run: bool = True, *, force: bool = False) -> list[list[str]]:
    """Create one env from its committed lockfile (the reproducible path).

    The env is built under the name ``run()`` uses.

    Parameters
    ----------
    env_name : str
        The env to build (``env_locks/<env_name>.yml`` must exist -
        ``FileNotFoundError`` otherwise, naming ``freeze`` as the fix).
    conda : str, optional
        conda executable; default ``conda``.
    dry_run : bool
        ``True`` (default) only returns the commands; ``False`` runs them.
    force : bool, keyword-only
        Lockfiles are linux-64; with ``dry_run=False`` on another host
        ``RuntimeError`` is raised before anything runs unless ``force``.

    Returns
    -------
    list of list of str
        The argv commands (conda phase, pip ``--no-deps`` phase, post-install
        script when committed), whether or not they were executed.
    """
    lock = lockfile(env_name)
    if lock is None:
        raise FileNotFoundError(
            f"no lockfile for env {env_name!r} (expected {_LOCKS_DIR / (env_name + '.yml')}). "
            f"Capture it on a host where the env exists via freeze({env_name!r}), "
            f"or build from the hand recipe via create_commands()."
        )
    conda = conda or _conda_bin("conda")
    # two phases: conda deps, then the pip closure with --no-deps (split_lock)
    conda_yaml, req, pip_lines = _materialise_split(env_name, lock)
    cmds = [[conda, "env", "create", "-n", env_name, "-f", str(conda_yaml)]]
    if pip_lines:
        cmds.append([conda, "run", "-n", env_name,
                     "pip", "install", "--no-deps", "-r", str(req)])
    # <env>.post.sh restores what the lockfile cannot record (post_install)
    post = _LOCKS_DIR / f"{env_name}.post.sh"
    if post.is_file():
        cmds.append([conda, "run", "-n", env_name, "bash", str(post)])
    if not dry_run:
        _require_linux(force)
        _run_all(cmds)
    return cmds


def create_all(category: str | None = None, methods: list[str] | None = None,
               conda: str | None = None, dry_run: bool = True, *,
               force: bool = False) -> list[dict]:
    """Provision every env needed to run the methods, from lockfiles.

    The lockfile path of ``multibench env install --run``. With
    ``dry_run=False`` the envs that are missing and have a lockfile are
    built; existing envs are skipped and envs without a lockfile are
    reported, not built.

    Parameters
    ----------
    category : str, optional
        Restrict to the methods wired for this category; default: all.
    methods : list of str, optional
        Explicit method ids (``KeyError`` with a did-you-mean hint on a typo).
    conda : str, optional
        conda executable; default ``conda``.
    dry_run : bool
        ``True`` (default) plans only - works on every host.
    force : bool, keyword-only
        Lockfiles are linux-64; ``dry_run=False`` on macOS/Windows raises
        ``RuntimeError`` before any build unless ``force=True``.

    Returns
    -------
    list of dict
        One entry per distinct env, largest first: ``{env, methods, exists,
        has_lock, cmds}`` (``cmds`` = the commands run, or that would run).
    """
    _check_methods(methods)
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
                _run_all(cmds)
        out.append({"env": env, "methods": sorted(ms), "exists": exists,
                    "has_lock": lock is not None, "cmds": cmds})
    return out


_PACKED_MANIFEST = Path(__file__).resolve().parent / "packed_urls.json"


def packed_manifest() -> dict:
    """The ``{env: archive_url}`` map shipped as ``engine/packed_urls.json`` (``{}`` if absent).

    The keys say whether an archive is published for an env; the URL is what
    ``multibench env install --packed`` (dry run) prints so a cluster user can
    check it against an egress proxy. Sizes come from the sibling snapshot
    ``engine/packed_sizes.json`` (:func:`packed_sizes`).
    """
    if not _PACKED_MANIFEST.is_file():
        return {}
    try:
        data = _json.loads(_PACKED_MANIFEST.read_text())
    except Exception:  # noqa: BLE001 - a broken manifest means "no archives"
        return {}
    return data if isinstance(data, dict) else {}


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
    category : str | None, keyword-only
        Integration category whose methods to cover when ``methods`` is
        ``None``.
    packed : bool, keyword-only
        Use a prebuilt conda-pack archive where one is published, else the
        lockfile; ``False`` = lockfile builds only.
    dry_run : bool, keyword-only
        ``True`` returns the plan and installs nothing; ``False`` installs the
        missing envs.
    conda : str | None, keyword-only
        conda/mamba executable; ``None`` = conda if found, else mamba. The
        packed path needs none.
    force : bool, keyword-only
        ``True`` = attempt a real install on a non-Linux host, which is
        refused otherwise.
    flavor : str, keyword-only
        Archive per env: ``'cpu'``, ``'gpu'`` (the CUDA build) or ``'auto'``
        (``'cpu'`` unless an NVIDIA GPU is visible).

    Returns
    -------
    list[dict]
        One row per env, largest first. Read ``env``, ``state`` and
        ``archive_bytes``; all keys and ``state`` values are listed in Notes.

    Raises
    ------
    ValueError
        ``flavor`` is not ``'auto'``, ``'cpu'`` or ``'gpu'``; or unknown
        ``category``.
    KeyError
        Unknown method id in ``methods``.
    RuntimeError
        ``dry_run=False`` only: a non-Linux host, no conda where needed, or a
        failed build.

    Warns
    -----
    UserWarning
        A CPU install finds no CPU archive for an env; the GPU build is
        installed.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.install(["Matilda"])                       # dry run: the plan, sizes, URLs
    >>> mtb.env.install(["Matilda"], dry_run=False)        # packed archive, flavor='auto'
    >>> mtb.env.install(category="vertical", dry_run=False, flavor="cpu")
    >>> mtb.env.install(["Matilda"], packed=False, dry_run=False)   # lockfile build via conda

    Notes
    -----
    **Keys.**

    - ``env`` / ``methods`` - the env and the selected methods it serves.
    - ``exists`` - the env was already installed here.
    - ``has_lock`` - a shipped lockfile can build the env.
    - ``state`` - what happened, or would happen (values below).
    - ``cmds`` - the lockfile commands run, or that would run.
    - ``packed_url`` / ``archive_bytes`` / ``unpacked_bytes`` - the archive
      the flavour selects and its sizes; ``None`` unless ``packed`` and
      known (a ``null`` in ``packed_sizes.json`` counts as unknown).
    - ``flavor`` - for an installed env, the archive it came from (``None``
      when unrecorded); for a missing env, the flavour the packed path would
      install, or ``None`` when ``packed=False``.

    **State values on a dry run.**

    - ``'have'`` - already installed.
    - ``'packed archive published'`` - an archive would be downloaded; the
      sizes and URL are filled in.
    - ``'no archive - lockfile build'`` / ``'no archive - NO-LOCK'`` -
      ``packed=True`` but no archive is published; the lockfile builds the
      env, or there is no lockfile either.
    - ``'build(dry-run)'`` / ``'NO-LOCK'`` - ``packed=False``; the lockfile
      builds the env, or there is none.

    **State values after** ``dry_run=False``: ``'PACKED'`` (unpacked from an
    archive), ``'BUILD'`` (built from the lockfile), ``'have'``, or
    ``'NO-LOCK'`` (reported, not built).

    **Flavours.** The env name and prefix are the same whatever the flavour.

    - ``'gpu'`` - the ``'<env>'`` archive, the CUDA build every env has.
    - ``'cpu'`` - the ``'<env>-cpu'`` archive (the same env without the CUDA
      libraries, several times smaller) where published; otherwise the GPU
      build and one ``UserWarning``.
    - ``'auto'`` - ``'cpu'`` when ``host_has_gpu`` is ``False`` (no
      ``nvidia-smi -L`` output and no ``/proc/driver/nvidia/version``), else
      ``'gpu'``.

    The dry-run sizes and URL follow the flavour, so a CPU host sees the CPU
    archives' download total. The flavour installed is recorded in
    ``<prefix>/.multibench_flavor`` and shown by ``status``, ``doctor`` and
    ``plan``.

    **Order of work** with ``dry_run=False``. With ``packed``, each missing
    env is first tried as an archive (its URL in ``packed_urls.json``, else
    the release-asset default under ``PACKED_URL``). An HTTP error on the
    download or a failed unpack falls back to the lockfile build; envs with
    no lockfile are reported ``'NO-LOCK'``, not built.

    **Where envs go.** Archives unpack into ``mtb.config.Config.envs_dir``
    (or, when ``conda`` is given, that tool's envs dir). The runner activates
    the prefix directly, so the packed path works without conda (Colab,
    laptops without conda).

    **Errors.** ``flavor`` is checked before anything else. With
    ``dry_run=False``:

    - Archives and lockfiles are linux-64, so a non-Linux host raises
      ``RuntimeError`` before any download unless ``force=True``.
    - Without conda/mamba, a missing env that needs a lockfile build raises
      before any download: ``"no conda/mamba on this host; <env> has a
      packed archive - pass packed=True"`` when ``packed=False`` skipped a
      published archive, ``"... has no packed archive - install conda
      first"`` otherwise.
    - A failed build command raises ``RuntimeError`` with its stderr tail.

    **Command line.** ``multibench env install --methods X --packed --run``
    takes the packed path; ``multibench env install --run`` builds from
    lockfiles (the CLI's ``--packed`` is off by default). The per-env steps
    are ``install_packed`` (one archive) and ``create_all`` (lockfile builds).

    See Also
    --------
    mtb.env.plan : the envs a set of methods needs, before installing.
    mtb.env.doctor : which of those envs exist here.
    mtb.env.status : install status per method.
    """
    check_flavor(flavor)
    _check_methods(methods)
    manifest = packed_manifest() if packed else {}
    sizes = packed_sizes() if packed else {}
    unpacked: set[str] = set()
    if not dry_run:
        _require_linux(force)             # before conda is even asked anything
        missing = [r for r in doctor(category=category, methods=methods, conda=conda)
                   if not r["exists"]]
        if (conda or _find_conda()) is None:
            # no conda here: only the packed path can provision anything, and
            # the error must be raised before any archive is downloaded
            for r in missing:
                if packed and r["env"] in manifest:
                    continue
                if r["env"] in packed_manifest():
                    raise RuntimeError(
                        f"no conda/mamba on this host; {r['env']} has a packed "
                        f"archive - pass packed=True")
                raise RuntimeError(
                    f"no conda/mamba on this host; {r['env']} has no packed "
                    f"archive - install conda first")
        if packed:
            for r in missing:
                if install_packed(r["env"], conda=conda, force=force, flavor=flavor):
                    unpacked.add(r["env"])
    # a RuntimeError (no conda here, a failed build, a non-Linux host)
    # propagates to the caller
    rows = create_all(category=category, methods=methods, conda=conda,
                      dry_run=dry_run, force=force)
    out = []
    for r in rows:
        env = r["env"]
        if env in unpacked:
            state = "PACKED"
        elif r["exists"]:
            state = "have"
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
        # the archive the flavour selects for this env (its fallback to the
        # GPU build included): URL and sizes follow it, so the printed
        # download total is what this host would fetch
        if packed:
            key, eff = archive_for(env, flavor, manifest=manifest, sizes=sizes)
        else:
            key, eff = env, None
        installed = installed_flavor(env, conda) if (r["exists"] or env in unpacked) else None
        sz = sizes.get(key) or {}
        out.append({**r, "state": state, "packed_url": manifest.get(key),
                    "archive_bytes": sz.get("archive_bytes"),
                    "unpacked_bytes": sz.get("unpacked_bytes"),
                    "flavor": installed if (r["exists"] or env in unpacked) else eff})
    return out


def doctor(category: str | None = None, methods: list[str] | None = None,
           conda: str | None = None, *, as_frame: bool = False):
    """Report, per needed env, whether it is installed and has a lockfile.

    The preflight check before running methods: one row per env the selected
    methods need.

    Parameters
    ----------
    category : str | None
        Integration category whose methods to check; ``None`` = every method.
    methods : list[str] | None
        Method ids to check instead of ``category``.
    conda : str | None
        conda/mamba executable that lists the installed envs; ``None`` =
        conda if found, else mamba.
    as_frame : bool, keyword-only
        ``True`` returns a ``pandas.DataFrame`` with the same keys as columns.

    Returns
    -------
    list[dict] or pandas.DataFrame
        One row per env, largest first. Read ``env``, ``exists`` and
        ``has_lock``; all keys are listed in Notes.

    Raises
    ------
    ValueError
        Unknown ``category``; the message lists the four.
    KeyError
        Unknown method id in ``methods``.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.doctor("vertical", as_frame=True)
    >>> [r["env"] for r in mtb.env.doctor(methods=["Matilda", "totalVI"]) if not r["exists"]]

    Notes
    -----
    **Keys.** The marks in brackets are the ones ``multibench env doctor``
    prints per env and ``env status`` prints per method.

    - ``env`` / ``methods`` - the env and the selected methods it serves.
    - ``exists`` - the env is installed here (``[x]``).
    - ``has_lock`` - ``env_locks/<env>.yml`` is shipped, so
      ``multibench env install --run`` can build it (``[L]`` while missing).
      A missing env without one (``[!]``) needs a packed archive or the
      recipe (``mtb.env.recipe``).
    - ``flavor`` - ``'cpu'`` / ``'gpu'`` when the installed env came from a
      packed archive, else ``None``.

    **Next step.** A fresh machine reports ``exists=False`` everywhere.
    ``multibench env install --run`` (Python: ``mtb.env.install`` with
    ``dry_run=False``) builds the missing envs, on Linux only
    (``host_platform_problem`` says why another host refuses).

    See Also
    --------
    mtb.env.install : builds or unpacks the envs reported missing.
    mtb.env.status : the same information per method.
    mtb.scan : its ``env_ok`` column uses the same installed-env check.
    """
    _check_methods(methods)
    have = set(installed_envs(conda))
    if methods is None:
        methods = registry.list_methods(category=category)
    by_env: dict[str, list[str]] = {}
    for m in methods:
        by_env.setdefault(group_for(m), []).append(m)
    rows = [{"env": env, "methods": sorted(ms), "exists": env in have,
             "has_lock": lockfile(env) is not None,
             "flavor": installed_flavor(env, conda) if env in have else None}
            for env, ms in sorted(by_env.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    return _as_frame(rows, as_frame)


_LOCAL_PIP_RE = re.compile(r"@\s*file://|feedstock_root")
# conda-forge and PyPI disagree on some package names, and `pip freeze` inside
# a conda env reports the conda name, which pip cannot resolve.
_CONDA_TO_PYPI = {
    "python-graphviz": "graphviz",
}
# Not on PyPI under any name: installed from git or from source in the working
# env and recorded by `pip freeze` as a bare `name==version` no index can
# satisfy. Stripped from the pip section and restored by <env>.post.sh, which
# names the source (a git URL + commit, or a path inside this repo).
_NOT_ON_PYPI = frozenset({"cobolt", "spiral", "multimap"})
# conda's own installer machinery: importable in a conda env, so `pip freeze`
# lists it, but distributed only through conda channels, so the pip install
# aborts with "No matching distribution found". Narrow on purpose: packages
# that are conda-only, not everything that also ships on conda-forge.
_CONDA_ONLY_PIP = frozenset({
    "conda", "mamba", "libmambapy", "boa",
    "conda-build", "conda-libmamba-solver", "conda-content-trust",
})
_CUDA_PIN_RE = re.compile(r"==[0-9][^\s]*\+(cu\d+)")


def sanitize_lock(text: str) -> str:
    """Make an exported lockfile rebuildable on a different machine.

    ``conda env export`` records two things that resolve only on the machine
    that produced them, so a lockfile can look complete and still fail every
    install:

    * pip entries pointing into the conda-forge build tree, e.g.
      ``argcomplete @ file:///home/conda/feedstock_root/build_artifacts/...``.
      These are conda packages pip merely observed; the conda dependency list
      already provides them. Kept, they abort the install with
      ``OSError: [Errno 2] No such file or directory``. They are dropped.

    * CUDA-local torch pins, e.g. ``torch==2.6.0+cu118``. Those wheels are
      published on download.pytorch.org and never on PyPI, so pip reports
      ``No matching distribution found``. The matching ``--extra-index-url`` is
      inserted rather than relaxing the pin, since the CUDA build is the point
      of pinning it.

    Pip entries no index can supply (:data:`_CONDA_ONLY_PIP`,
    :data:`_NOT_ON_PYPI`) are dropped as well, and a package that ``pip freeze``
    lists under its conda-forge name is renamed to its PyPI name
    (:data:`_CONDA_TO_PYPI`, ``python-graphviz`` -> ``graphviz``).

    The pip block is located by indentation rather than a fixed prefix: conda's
    own export indents entries six spaces while freeze()'s fallback path writes
    four, so a hard-coded width would skip one of the two.

    Idempotent: re-sanitising an already-clean lockfile changes nothing.
    """
    out, cuda_tags = [], []
    pip_indent = None
    entry_indent = None
    for ln in text.splitlines():
        stripped = ln.strip()
        indent = len(ln) - len(ln) + (len(ln) - len(ln.lstrip()))
        if stripped == "- pip:":
            pip_indent, entry_indent = indent, None
            out.append(ln)
            continue
        if pip_indent is not None and stripped:
            if indent <= pip_indent:
                pip_indent = None          # dedented out of the pip block
            else:
                if entry_indent is None:
                    entry_indent = indent
                if _LOCAL_PIP_RE.search(stripped):
                    continue               # unresolvable off this machine
                name = re.split(r"[=<>!~\s\[]", stripped[2:].strip())[0].lower()
                if name in _CONDA_ONLY_PIP:
                    continue               # conda-only: pip can never supply it
                if name in _NOT_ON_PYPI:
                    continue               # restored by <env>.post.sh instead
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

    ``conda run -n <env> pip freeze`` in an env that has no pip does not fail:
    it falls through to the next pip on PATH, the base environment's, and the
    base env's packages would be written into this env's lockfile. A pure R env
    such as ``scmb_r`` has no python binary, so :func:`freeze` asks this first.
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
    """Capture an existing env to a committed lockfile (maintainer tool).

    Runs ``conda env export -n <env> --no-builds``, strips the host-specific
    ``prefix:`` line, sanitises the result (:func:`sanitize_lock`) and writes
    ``env_locks/<env>.yml`` - what :func:`create_env` rebuilds from. Run on the
    host where the working env lives.
    """
    import re
    conda = conda or _conda_bin("conda")
    # `conda env export` exits 0 with an empty env for a name that does not
    # exist, so without this check a typo (or `freeze --all` on a machine
    # without the envs) would overwrite committed lockfiles with stubs.
    if env_name not in installed_envs(conda):
        raise FileNotFoundError(
            f"no conda env named {env_name!r} on this machine "
            f"(see `conda env list`); freeze captures existing envs only")
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

    # Prefer a full export; fall back to explicit specs when it fails (corrupt
    # transitive metadata) or yields nothing (an env whose packages are all pip
    # and untracked by conda history: `--no-builds` emits an empty deps list).
    exp = _run(["env", "export", "-n", env_name, "--no-builds"])
    if exp.returncode == 0 and _has_real_deps(exp.stdout):
        body = exp.stdout
    else:
        hist = _run(["env", "export", "-n", env_name, "--from-history"])
        body = hist.stdout if hist.returncode == 0 else (
            f"name: {env_name}\nchannels:\n  - conda-forge\ndependencies:\n  - python\n")
    lines = [ln for ln in body.splitlines() if not ln.startswith("prefix:")]
    # If pip-installed packages were not captured, append a pip: section from
    # `pip freeze` so the lockfile reproduces the env.
    if "- pip:" not in "\n".join(lines) and _has_own_python(env_name, conda):
        pip = _run(["run", "-n", env_name, "pip", "freeze"]).stdout
        pip_pkgs = [ln.strip() for ln in pip.splitlines()
                    if ln.strip() and not ln.startswith("-e ")]
        if pip_pkgs:
            if not any(l.startswith("dependencies:") for l in lines):
                lines.append("dependencies:")
            lines += ["  - pip", "  - pip:"] + [f"    - {p}" for p in pip_pkgs]
    dst = dst_dir / f"{env_name}.yml"
    # an export is not automatically installable elsewhere - see sanitize_lock
    dst.write_text(sanitize_lock("\n".join(lines)))
    return dst


# --- recipe/lockfile provisioning entry points -----------------------------
def create(method: str, env_name: str | None = None, conda: str | None = None,
           dry_run: bool = True, *, force: bool = False) -> list[list[str]]:
    """Provision the env a method runs in.

    Prefers the committed lockfile of the env ``run()`` uses
    (:func:`group_for`); falls back to the hand-written recipe, built under
    the same name, when that env has no lockfile.

    Parameters
    ----------
    method : str
        Registry method id.
    env_name : str, optional
        Build into this env instead of :func:`default_env_name`.
    conda : str, optional
        conda executable.
    dry_run : bool
        ``True`` (default) returns the commands without running them.
    force : bool, keyword-only
        Build even though this host is not linux-64 (``dry_run=False`` on
        macOS/Windows otherwise raises ``RuntimeError`` first).

    Returns
    -------
    list of list of str
        The argv commands.
    """
    target = env_name or group_for(method)
    if lockfile(target) is not None:
        return create_env(target, conda=conda, dry_run=dry_run, force=force)
    cmds = create_commands(method, env_name=env_name, conda=conda)
    if not dry_run:
        _require_linux(force)
        _run_all(cmds)
    return cmds


def create_group(group: str, env_name: str | None = None, conda: str | None = None,
                 dry_run: bool = True, *, force: bool = False) -> list[list[str]]:
    """Provision a shared group env.

    Prefers the committed lockfile for the env; falls back to the hand recipe.

    Parameters
    ----------
    group : str
        Group env name (see :func:`groups`); ``KeyError`` when unknown.
    env_name : str, optional
        Build into this env instead of ``group``.
    conda : str, optional
        conda executable.
    dry_run : bool
        ``True`` (default) returns the commands without running them.
    force : bool, keyword-only
        Build even though this host is not linux-64.

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
        _run_all(cmds)
    return cmds


def _run_all(cmds: list[list[str]]) -> None:
    # PYTHONNOUSERSITE matters while building, not only while running: pip
    # treats a package importable from ~/.local/lib/pythonX/site-packages as
    # satisfied and skips it, the build reports success, and the method then
    # fails with ModuleNotFoundError under run(), which sets PYTHONNOUSERSITE=1.
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    if cmds and shutil.which(cmds[0][0]) is None:
        raise RuntimeError(
            "conda/mamba not found on this machine, so method environments "
            "cannot be built here. They need Linux with conda (mamba "
            "recommended) - see the installation guide. Everything that does "
            "not run a method (the registry, stored results, figures) works "
            "without them."
        )
    for c in cmds:
        proc = subprocess.run(c, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"env command failed: {' '.join(c)}\nstderr tail:\n{proc.stderr[-2000:]}"
            )
    _conda_prefixes.cache_clear()         # the env just built must be visible
