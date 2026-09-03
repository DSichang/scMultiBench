"""Per-method environment recipes + shared "group" environments.

scMultiBench wraps ~40 separately-developed tools, each with its own (often
conflicting, pinned) dependencies — so the benchmark uses one conda env per
method. This module materialises a method's ``env_spec`` recipe into installable
conda/pip commands or an ``environment.yml``.

A single env cannot host *all* methods (TF 2.4 vs 2.8, scvi <0.20 vs latest,
py3.7 vs 3.10, R vs Python all conflict). But many methods ARE compatible, so
``env_groups.yaml`` declares a handful of **shared group envs** (e.g. one torch
env serving ~18 methods, one R env serving ~9). ``groups()``/``plan()`` let you
build the few group envs a dataset's applicable methods need, instead of one per
method.
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

from . import registry

__all__ = [
    # inspection
    "recipe", "default_env_name", "own_env_name", "group_for", "groups", "plan", "status",
    "doctor", "required_envs", "installed_envs", "lockfile",
    "host_platform_problem", "packed_sizes", "DIFFICULTY", "VERIFIED_STAR",
    # recipe view (the declared, hand-written recipe — for transparency)
    "create_commands", "environment_yml", "group_create_commands",
    # provisioning (lockfile-based: build the REAL envs run() uses)
    "create", "create_group", "create_env", "create_all", "freeze",
]


# --- host platform --------------------------------------------------------
def host_platform_problem() -> str | None:
    """Why method environments cannot be built on THIS host, or ``None``.

    Every method env is a linux-64 conda env: the packed archives are
    conda-pack snapshots of linux-64 envs and the lockfiles pin linux-only
    packages (``libgcc-ng`` ...), so on macOS or Windows a build either fails
    on ELF binaries after a multi-GB download or dies in the solver. Nothing
    enforced that before 0.3.1: the CLI happily started the download.

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

    ``force=True`` skips the check (``--force`` on the CLI) for people who
    know what they are doing - a Linux container on a Mac, say.
    """
    problem = None if force else host_platform_problem()
    if problem:
        raise RuntimeError(
            f"{problem} - method envs cannot be built here. Run methods on a "
            f"Linux host (the registry, stored results, scan's file gate, "
            f"evaluate and plot all work on this machine); pass force=True / "
            f"--force to try anyway.")


#: What the ``difficulty`` tag of ``env_specs.yaml`` (shown by ``env status``
#: and :func:`status`) means. The tag describes how hard the env is to BUILD
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

#: ONE symbol set for every env listing (``env status`` and ``env doctor``):
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

    Read from the shipped ``engine/packed_sizes.json`` (a snapshot written by
    ``tools/packed_sizes.py``, which HEAD-requests every URL in
    ``packed_urls.json``; no request is made at runtime - offline nodes and
    Zenodo rate limits make that a bad idea). Keys starting with ``_`` are
    metadata, not envs.

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

_GROUPS_YAML = Path(__file__).resolve().parent / "env_groups.yaml"
# Committed per-env lockfiles (`conda env export --no-builds`). These capture the
# ACTUAL working envs (versions + pip section) and are the reproducible install
# source: create_env/create_all rebuild a fresh machine's envs from them, by the
# real env name run() uses — unlike the hand-written `recipe`/create_commands,
# which build a method's OWN scmb_<method> env from a best-effort spec.
_LOCKS_DIR = Path(__file__).resolve().parent / "env_locks"


# --- recipes ---------------------------------------------------------------
def recipe(method: str) -> dict:
    """Return the method's environment recipe (its ``env_spec``); ``{}`` if none."""
    return registry.get(method).env_spec or {}


def own_env_name(method: str) -> str:
    """The method's OWN (singleton) env name, ``scmb_<method>``.

    Used only when a method is NOT a member of any shared group and has no
    explicit ``method_env`` override in ``env_groups.yaml`` - then
    :func:`group_for` falls back to it. It is the name the hand-written recipe
    (:func:`create_commands` / :func:`environment_yml`) builds into when you
    pass it explicitly; by default those use :func:`default_env_name`.
    """
    return f"scmb_{method.lower()}"


def default_env_name(method: str) -> str:
    """The conda env name EVERY entry point uses for ``method``.

    Identical to :func:`group_for`: the explicit ``method_env`` override from
    ``env_groups.yaml`` (``Matilda`` -> ``matilda``), else the shared group env
    the method is a member of (``UINMF`` -> ``scmb_r``), else its own
    ``scmb_<method>`` env (:func:`own_env_name`). This is the name
    ``scan()['env']``, ``method_info(m)['env']``, ``run()``'s ``conda run -n``,
    ``env doctor`` / ``env plan`` / ``env install`` / ``env create`` AND
    ``env recipe`` / ``env yml`` all agree on - so a recipe pasted into a
    build job produces an env the package recognises.

    Earlier releases returned the own ``scmb_<method>`` name even for grouped
    methods, so ``env recipe Matilda`` named ``scmb_matilda`` while everything
    else expected ``matilda``. ``method`` must be a registry id (``KeyError``
    with a did-you-mean hint otherwise).
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

    A recipe carries ``python_version`` AND (some) list ``python=3.7`` in
    ``conda_packages`` too, so the create line read ``python=3.7 python=3.7``.
    Harmless to conda, but it reads as a generator bug; the explicit
    ``python_version`` wins and the duplicate is dropped.
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
    """Explicit ``{method: real conda env}`` overrides from env_groups.yaml.

    These point at the actual installed envs (verified by import-testing) and
    take precedence over the generated ``groups`` below, whose ``scmb_*`` names
    are largely fictional / never built.
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
    # explicit method->real-env overrides: ensure each target env is a group too
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

    ``env_name`` overrides the target env (default: the group name) — handy for
    building into a scratch env to validate a recipe without touching the real one.
    """
    spec = groups().get(group)
    if spec is None:
        raise KeyError(f"unknown group {group!r}; see envs.groups()")
    return _install_commands(spec, env_name or group, conda)


def _check_methods(methods):
    """Unknown names in methods= fabricated empty plans instead of failing.

    Delegates to :func:`registry.check_method` so the KeyError carries the same
    did-you-mean hint as every other entry point (``'Stabmap'`` -> ``'StabMap'``).
    """
    if not methods:
        return
    from . import registry
    for m in methods:
        registry.check_method(m)


def plan(category: str | None = None, methods: list[str] | None = None, *,
         as_frame: bool = False):
    """Which envs to build to cover a set of methods (e.g. all for a category).

    One entry per distinct conda env, listing the methods it serves - build
    these (few) envs instead of one per method. ``multibench env plan`` prints
    it with the archive / on-disk sizes from :func:`packed_sizes`.

    Parameters
    ----------
    category : str, optional
        Restrict to the methods wired for this integration category
        (``ValueError`` listing the four on a typo); default: every method.
    methods : list of str, optional
        Explicit method ids instead of ``category`` (``KeyError`` with a
        did-you-mean hint on a typo).
    as_frame : bool, keyword-only
        ``True`` returns a ``pandas.DataFrame`` with the same columns
        instead of the list of dicts (the default, kept for compatibility).

    Returns
    -------
    list of dict or pandas.DataFrame
        ``[{env, shared, methods, availability}]``, largest env first.
        ``shared`` - the env serves several methods (an ``env_groups.yaml``
        group); ``availability`` - ``'public'``, or ``'benchmark-host-only'``
        when EVERY method the env serves needs a script that is not
        published (SPIRAL): the env builds, the method still cannot run off
        the benchmark host.
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
         # 'benchmark-host-only' when every method the env serves needs a
         # script that is not published (SPIRAL, GPSA): the env builds, the
         # method still cannot run off the benchmark host
         "availability": ("benchmark-host-only"
                          if all(registry.get(m).availability != "public" for m in ms)
                          else "public")}
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


def install_packed(env: str, conda: str | None = None, *, force: bool = False) -> bool:
    """Provision ``env`` from a prebuilt conda-pack archive, if one is published.

    Downloads the archive named in ``packed_urls.json`` (else
    ``<PACKED_URL>/<env>.tar.gz``), unpacks it into the conda envs directory
    and runs the archive's own ``bin/conda-unpack`` to rewrite the embedded
    prefixes. This turns a 10-30 minute solve-and-download into a
    download-bound couple of minutes.

    Parameters
    ----------
    env : str
        The real conda env name (:func:`group_for`), e.g. ``'matilda'``.
    conda : str, optional
        conda/mamba executable; default: mamba if found, else conda.
    force : bool, keyword-only
        The archives are linux-64; on any other host ``RuntimeError`` is
        raised BEFORE the download unless ``force=True``.

    Returns
    -------
    bool
        ``True`` on success (or when the env already exists), ``False`` when
        no archive exists for this env or the unpack failed - the caller
        falls back to the lockfile build.
    """
    import subprocess
    import tarfile
    import tempfile
    import urllib.error
    import urllib.request

    _require_linux(force)                 # fail closed before any bytes land
    bin_ = _conda_bin() if conda is None else conda
    if shutil.which(bin_) is None:
        return False
    import json as _json
    envs_root = _envs_dir(bin_)
    if envs_root is None:
        print(f"[env] could not locate the environments directory of {bin_}; "
              "falling back to the lockfile build", flush=True)
        return False
    dest = envs_root / env
    if dest.exists():
        return True
    # A shipped manifest maps env -> archive URL, so archives can live where
    # their size dictates (GitHub release assets up to 2 GiB, Zenodo beyond);
    # envs without an entry fall back to the release-asset convention.
    _manifest = {}
    _mf = Path(__file__).parent / "packed_urls.json"
    if _mf.is_file():
        try:
            _manifest = _json.loads(_mf.read_text())
        except Exception:
            _manifest = {}
    url = _manifest.get(env) or f"{PACKED_URL}/{env}.tar.gz"
    try:
        tgz, _ = urllib.request.urlretrieve(url)
    except urllib.error.HTTPError:
        return False
    print(f"[env] unpacking prebuilt {env} -> {dest} ...", flush=True)
    from ..data.fetch import safe_extract
    part = dest.with_name(dest.name + ".partial")
    try:
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with tarfile.open(tgz) as t:
            safe_extract(t, part)
        # conda-unpack's shebang expects a `python` on PATH, which a bare
        # machine may not have - run it through the env's own interpreter.
        py = part / "bin" / "python"
        unpack = part / "bin" / "conda-unpack"
        if not py.exists():
            raise RuntimeError("archive did not contain bin/python")
        # the env must carry its FINAL path before conda-unpack rewrites
        # prefixes, so move first, then unpack
        part.rename(dest)
        if (dest / "bin" / "conda-unpack").exists():
            subprocess.run([str(dest / "bin" / "python"),
                            str(dest / "bin" / "conda-unpack")],
                           check=True, capture_output=True)
        return True
    except Exception as e:  # noqa: BLE001 - degrade to the lockfile build
        shutil.rmtree(part, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[env] prebuilt {env} failed ({type(e).__name__}: "
              f"{str(e)[:120]}); falling back to the lockfile build",
              flush=True)
        return False


def installed_envs(conda: str | None = None) -> list[str]:
    """Names (basenames) of existing conda envs.

    Uses ``conda env list --json`` and returns the basename of each env prefix,
    so path-only/prefix entries (e.g. basilisk caches) become clean names rather
    than leaking raw filesystem paths.
    """
    import json
    import os
    conda = conda or _conda_bin("conda")
    try:
        out = subprocess.run([conda, "env", "list", "--json"],
                             capture_output=True, text=True, check=True).stdout
        prefixes = json.loads(out).get("envs", [])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, TypeError):
        return []
    names: list[str] = []
    for p in prefixes:
        name = os.path.basename(str(p).rstrip("/"))
        if name and name not in names:
            names.append(name)
    return names


def status(conda: str | None = None, *, as_frame: bool = False):
    """Per-method install status: is the method's env on this machine?

    Parameters
    ----------
    conda : str, optional
        conda executable used to list the installed envs.
    as_frame : bool, keyword-only
        ``True`` returns a ``pandas.DataFrame`` instead of the list of dicts.

    Returns
    -------
    list of dict or pandas.DataFrame
        One entry per registry method: ``{method, env, group, own_env,
        exists, has_lock, difficulty, verified_working, has_recipe}``.
        ``env`` and ``group`` are both the env the package uses for the
        method (:func:`default_env_name` == :func:`group_for`); ``own_env``
        is the singleton ``scmb_<method>`` name, reported installed too when
        present; ``has_lock`` says a shipped lockfile can build ``env``
        (the ``[L]`` of :data:`MARK_LEGEND`, shared with :func:`doctor`).
        ``difficulty`` is one of the :data:`DIFFICULTY` tags
        (``easy`` / ``old-scvi`` / ``old-tensorflow`` / ``R`` / ``verified``
        / ``blocked-script``; ``unknown`` without a recipe) and
        ``verified_working`` is the ``*`` of ``env status``
        (:data:`VERIFIED_STAR`).
    """
    have = set(installed_envs(conda))
    out = []
    for s in registry.load():
        r = s.env_spec or {}
        grp = group_for(s.id)
        own = own_env_name(s.id)
        out.append({
            "method": s.id, "env": grp, "group": grp, "own_env": own,
            "exists": grp in have or own in have,
            "has_lock": lockfile(grp) is not None,
            "difficulty": r.get("difficulty", "unknown"),
            "verified_working": bool(r.get("verified_working", False)),
            "has_recipe": bool(r),
        })
    return _as_frame(out, as_frame)


# --- lockfile-based provisioning (the reproducible install path) -----------
def lockfile(env_name: str) -> Path | None:
    """Path to the committed lockfile for a real env, or None if not captured.

    Lockfiles live in ``env_locks/<env_name>.yml`` (a ``conda env export
    --no-builds`` of the actual working env) and are the reproducible source
    create_env/create_all rebuild from.
    """
    p = _LOCKS_DIR / f"{env_name}.yml"
    return p if p.exists() else None


def required_envs(category: str | None = None,
                  methods: list[str] | None = None) -> list[str]:
    """The distinct real conda envs needed to run the given methods (or all).

    Exactly the env names ``run()`` shells into (via group_for) — i.e. what a
    fresh machine must provision.
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

    `conda env create` hands the whole pip section to `pip install -r`, which
    RE-RESOLVES it. But the section is a full `pip freeze` closure: every
    transitive dependency is already present and pinned. Re-resolving it is
    unnecessary, and it fails outright whenever the working env contains a
    combination pip considers inconsistent - which is common in envs built up
    incrementally over time. Four of the 29 lockfiles died with
    ResolutionImpossible for exactly this reason.

    Installing the closure with --no-deps reproduces the env as recorded instead
    of asking pip to re-derive it. That requires the two halves to be installed
    separately, because --no-deps cannot be expressed inside a requirements file.
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
    # never scribble inside an installed package (site-packages must stay
    # read-only); a repo checkout keeps the old path for inspectability
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

    Covers what a conda lockfile provably cannot: packages installed inside the
    env by a language-native installer (install.packages(), install_github())
    which conda never sees and therefore never restores.
    """
    p = _LOCKS_DIR / f"{env_name}.post.sh"
    return p if p.is_file() else None


def create_env(env_name: str, conda: str | None = None,
               dry_run: bool = True, *, force: bool = False) -> list[list[str]]:
    """Create one real env from its committed lockfile (the reproducible path).

    Builds the env under its real name (the one run() uses), so 'what you build'
    == 'what runs'.

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
    # Two phases: conda deps, then the pip closure with --no-deps. See split_lock.
    conda_yaml, req, pip_lines = _materialise_split(env_name, lock)
    cmds = [[conda, "env", "create", "-n", env_name, "-f", str(conda_yaml)]]
    if pip_lines:
        cmds.append([conda, "run", "-n", env_name,
                     "pip", "install", "--no-deps", "-r", str(req)])
    # Some packages cannot be captured by `conda env export` at all: an R package
    # installed with install.packages() lives in the env's R library but conda has
    # no record of it, so the lockfile rebuilds an env WITHOUT it. scmb_r lost
    # rliger exactly this way - the one package UINMF needs - while still
    # reporting a clean build. A committed <env>.post.sh restores those.
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
    """Provision EVERY env needed to run the methods, from lockfiles.

    One-shot 'set up a fresh machine' (``multibench env install --run``).
    With ``dry_run=False`` the envs that are MISSING and have a lockfile are
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
        ``RuntimeError`` BEFORE any build unless ``force=True``.

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


def doctor(category: str | None = None, methods: list[str] | None = None,
           conda: str | None = None, *, as_frame: bool = False):
    """Report, per conda env the selected methods need, whether it exists here and whether a lockfile can build it.

    This is the preflight behind ``multibench env doctor`` and the ``env_ok``
    gate of :func:`multibench.scan`. A fresh machine sees ``exists=False``
    everywhere; ``multibench env install --run`` (Python:
    :func:`create_all` ``(dry_run=False)``) builds the missing ones - on
    Linux only, see :func:`host_platform_problem`.

    Parameters
    ----------
    category : str, optional
        Restrict to the methods wired for this category; default: all.
    methods : list of str, optional
        Explicit method ids (``KeyError`` with a did-you-mean hint on a typo).
    conda : str, optional
        conda executable used to list the installed envs.
    as_frame : bool, keyword-only
        ``True`` returns a ``pandas.DataFrame`` instead of the list of dicts.

    Returns
    -------
    list of dict or pandas.DataFrame
        ``[{env, methods, exists, has_lock}]``, largest env first. ``exists``
        - the env is installed (the ``[x]`` of :data:`MARK_LEGEND`);
        ``has_lock`` - ``env_locks/<env>.yml`` is shipped, so ``env install
        --run`` can build it (``[L]``); neither (``[!]``) means the recipe
        path only. The same marks ``env status`` prints per method.
    """
    _check_methods(methods)
    have = set(installed_envs(conda))
    if methods is None:
        methods = registry.list_methods(category=category)
    by_env: dict[str, list[str]] = {}
    for m in methods:
        by_env.setdefault(group_for(m), []).append(m)
    rows = [{"env": env, "methods": sorted(ms), "exists": env in have,
             "has_lock": lockfile(env) is not None}
            for env, ms in sorted(by_env.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    return _as_frame(rows, as_frame)



_LOCAL_PIP_RE = re.compile(r"@\s*file://|feedstock_root")
# conda's own installer machinery. `pip freeze` inside a conda env reports these
# because they ARE importable there, but they are distributed only through conda
# channels, so pip resolves nothing and the whole install aborts. scmb_r carried
# `conda==23.3.1` and failed with "No matching distribution found for conda".
# Deliberately narrow: packages that are genuinely conda-only, not everything
# that happens to ship on conda-forge.
# conda-forge and PyPI disagree on some package names. `pip freeze` inside a
# conda env reports the CONDA name, which PyPI has never heard of: unitednet
# pinned `python-graphviz==0.8.4` and failed with "No matching distribution".
_CONDA_TO_PYPI = {
    "python-graphviz": "graphviz",
}
# Packages that are simply not on PyPI under any name - installed from git or
# from source in the working env, and recorded by `pip freeze` as a bare
# `name==version` that no index can satisfy. They are stripped from the pip
# section and restored by a committed <env>.post.sh, which can name the real
# source (a git URL + commit, or a path inside this repo).
_NOT_ON_PYPI = frozenset({"cobolt", "spiral", "multimap"})
_CONDA_ONLY_PIP = frozenset({
    "conda", "mamba", "libmambapy", "boa",
    "conda-build", "conda-libmamba-solver", "conda-content-trust",
})
_CUDA_PIN_RE = re.compile(r"==[0-9][^\s]*\+(cu\d+)")


def sanitize_lock(text: str) -> str:
    """Make an exported lockfile rebuildable on a DIFFERENT machine.

    ``conda env export`` faithfully records two things that cannot resolve
    anywhere except the machine that produced them, so a lockfile can look
    complete and still fail every install:

    * pip entries pointing into the conda-forge BUILD tree, e.g.
      ``argcomplete @ file:///home/conda/feedstock_root/build_artifacts/...``.
      These are conda packages pip merely observed; the conda dependency list
      already provides them. Kept, they abort the install with
      ``OSError: [Errno 2] No such file or directory``. They are dropped.

    * CUDA-local torch pins, e.g. ``torch==2.6.0+cu118``. Those wheels are
      published on download.pytorch.org and never on PyPI, so pip reports
      ``No matching distribution found``. The matching ``--extra-index-url`` is
      inserted rather than relaxing the pin, since the CUDA build is the point
      of pinning it.

    The pip block is located by INDENTATION rather than a fixed prefix: conda's
    own export indents entries six spaces while freeze()'s fallback path writes
    four, and a hard-coded width silently skips half the files.

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
    """Does this env contain its OWN python interpreter?

    ``conda run -n <env> pip freeze`` in an env that has no pip does not fail -
    it falls through to whatever pip is next on PATH, which is the BASE
    environment's. The captured list is then the base env's packages, written
    into this env's lockfile.

    That is not hypothetical: scmb_r is a pure R env with no python binary at
    all, and its lockfile had picked up 229 base-environment entries including
    ``conda==23.3.1`` (the base conda's own version), torch, and grpcio. On a
    fresh machine the install aborted, and had it succeeded it would have
    polluted an R env with the whole base interpreter.
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
    ``prefix:`` line, and writes ``env_locks/<env>.yml`` — exactly what
    create_env rebuilds. Run on the host where the working env lives.
    """
    import re
    conda = conda or _conda_bin("conda")
    # conda env export exits 0 with an EMPTY env for a name that does not
    # exist, so without this check a typo silently overwrites a committed
    # lockfile with a stub - and freeze --all on a machine without the envs
    # would destroy all of them while printing success.
    if env_name not in installed_envs(conda):
        raise FileNotFoundError(
            f"no conda env named {env_name!r} on this machine "
            f"(see `conda env list`); freeze captures EXISTING envs only")
    dst_dir = Path(out_dir) if out_dir else _LOCKS_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _run(args):
        return subprocess.run([conda, *args], capture_output=True, text=True)

    def _has_real_deps(text: str) -> bool:
        # real = a conda package beyond python/pip, OR a pip: section
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
    # transitive metadata) OR yields nothing (env whose pkgs are all pip and
    # untracked by conda history — `--no-builds` then emits an empty deps list).
    exp = _run(["env", "export", "-n", env_name, "--no-builds"])
    if exp.returncode == 0 and _has_real_deps(exp.stdout):
        body = exp.stdout
    else:
        hist = _run(["env", "export", "-n", env_name, "--from-history"])
        body = hist.stdout if hist.returncode == 0 else (
            f"name: {env_name}\nchannels:\n  - conda-forge\ndependencies:\n  - python\n")
    lines = [ln for ln in body.splitlines() if not ln.startswith("prefix:")]
    # If pip-installed packages weren't captured, append a pip: section from
    # `pip freeze` so the lockfile actually reproduces the env.
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

    Prefers the committed lockfile for the REAL env run() uses (group_for), so
    'what you build' == 'what runs'. Falls back to the hand-written recipe
    (its own scmb_<method> env) only when no lockfile was captured.

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
    # PYTHONNOUSERSITE is essential while BUILDING, not just while running: pip
    # treats a package already importable from ~/.local/lib/pythonX/site-packages
    # as satisfied and skips installing it into the target env. The build then
    # reports success while producing an env that only works if user-site leakage
    # is allowed - and run() correctly sets PYTHONNOUSERSITE=1, so the method
    # fails at dispatch. Six of the 29 envs were built short of 32 packages this
    # way, and VIPCCA died with ModuleNotFoundError on a package its lockfile
    # pinned.
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
