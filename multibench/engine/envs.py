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
import shutil
import subprocess
from pathlib import Path

import yaml

from . import registry

__all__ = [
    # inspection
    "recipe", "default_env_name", "group_for", "groups", "plan", "status",
    "doctor", "required_envs", "installed_envs", "lockfile",
    # recipe view (the declared, hand-written recipe — for transparency)
    "create_commands", "environment_yml", "group_create_commands",
    # provisioning (lockfile-based: build the REAL envs run() uses)
    "create", "create_group", "create_env", "create_all", "freeze",
]

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


def default_env_name(method: str) -> str:
    """The method's OWN (singleton) env name, ``scmb_<method>``.

    This is only the env name used when a method is NOT a member of any shared
    group (see :func:`group_for`). It is NOT necessarily the env ``run()``
    executes in: that is always :func:`group_for`, which returns the shared
    group env for grouped methods and falls back to this name otherwise.
    """
    return f"scmb_{method.lower()}"


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
    create += list(spec.get("conda_packages", []))
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


def _environment_yml(spec: dict, env_name: str) -> str:
    channels = list(spec.get("conda_channels", [])) or ["conda-forge"]
    lines = [f"name: {env_name}", "channels:"]
    lines += [f"  - {c}" for c in channels]
    lines.append("dependencies:")
    if spec.get("python_version"):
        lines.append(f"  - python={spec['python_version']}")
    for c in spec.get("conda_packages", []):
        lines.append(f"  - {c}")
    pip_pkgs = list(spec.get("pip_packages", [])) + list(spec.get("pip_git", []))
    if pip_pkgs:
        lines += ["  - pip", "  - pip:"]
        lines += [f"    - {p}" for p in pip_pkgs]
    return "\n".join(lines) + "\n"


def create_commands(method: str, env_name: str | None = None,
                    conda: str | None = None) -> list[list[str]]:
    """Build the create+install commands for a single method's env."""
    r = recipe(method)
    if not r:
        raise ValueError(f"no env_spec recipe declared for {method!r}")
    return _install_commands(r, env_name or default_env_name(method), conda)


def environment_yml(method: str, env_name: str | None = None) -> str:
    """Render an ``environment.yml`` string for a single method's recipe."""
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
            out[default_env_name(s.id)] = {
                **(s.env_spec or {}), "members": [s.id], "shared": False,
            }
    return out


def group_for(method: str) -> str:
    """The env name that serves this method (an explicit override, a shared
    group, or its own env)."""
    override = _method_env()
    if method in override:
        return override[method]
    for name, g in _merged_groups().items():
        if method in g.get("members", []):
            return name
    return default_env_name(method)


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


def plan(category: str | None = None, methods: list[str] | None = None) -> list[dict]:
    """Which envs to build to cover a set of methods (e.g. all for a category).

    Returns [{env, shared, methods}] — one entry per distinct env, listing the
    methods it serves. Build these (few) envs instead of one per method.
    """
    if methods is None:
        methods = registry.list_methods(category=category)
    shared = _merged_groups()
    buckets: dict[str, list[str]] = {}
    for m in methods:
        buckets.setdefault(group_for(m), []).append(m)
    return [
        {"env": env, "shared": env in shared, "methods": sorted(ms)}
        for env, ms in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]


# --- env existence / status -----------------------------------------------
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


def status(conda: str | None = None) -> list[dict]:
    """Per-method install status: difficulty, default env name, group, exists."""
    have = set(installed_envs(conda))
    out = []
    for s in registry.load():
        r = s.env_spec or {}
        env = default_env_name(s.id)
        grp = group_for(s.id)
        out.append({
            "method": s.id, "env": env, "group": grp,
            "exists": env in have or grp in have,
            "difficulty": r.get("difficulty", "unknown"),
            "verified_working": bool(r.get("verified_working", False)),
            "has_recipe": bool(r),
        })
    return out


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


def create_env(env_name: str, conda: str | None = None,
               dry_run: bool = True) -> list[list[str]]:
    """Create one real env from its committed lockfile (the reproducible path).

    Builds the env under its real name (the one run() uses), so 'what you build'
    == 'what runs'. Raises if no lockfile was captured for it.
    """
    lock = lockfile(env_name)
    if lock is None:
        raise FileNotFoundError(
            f"no lockfile for env {env_name!r} (expected {_LOCKS_DIR / (env_name + '.yml')}). "
            f"Capture it on a host where the env exists via freeze({env_name!r}), "
            f"or build from the hand recipe via create_commands()."
        )
    conda = conda or _conda_bin("conda")
    cmds = [[conda, "env", "create", "-n", env_name, "-f", str(lock)]]
    if not dry_run:
        _run_all(cmds)
    return cmds


def create_all(category: str | None = None, methods: list[str] | None = None,
               conda: str | None = None, dry_run: bool = True) -> list[dict]:
    """Provision EVERY env needed to run the methods, from lockfiles.

    One-shot 'set up a fresh machine'. Returns one entry per distinct env:
    {env, methods, exists, has_lock, cmds}. With dry_run=False, builds the envs
    that are MISSING and have a lockfile (existing envs are skipped; envs without
    a lockfile are reported, not built).
    """
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
           conda: str | None = None) -> list[dict]:
    """Preflight: per env needed to run the methods, is it present + is a lockfile
    available to build it. [{env, methods, exists, has_lock}], coverage-sorted.

    A fresh machine sees exists=False everywhere; run create_all(dry_run=False).
    """
    have = set(installed_envs(conda))
    if methods is None:
        methods = registry.list_methods(category=category)
    by_env: dict[str, list[str]] = {}
    for m in methods:
        by_env.setdefault(group_for(m), []).append(m)
    return [{"env": env, "methods": sorted(ms), "exists": env in have,
             "has_lock": lockfile(env) is not None}
            for env, ms in sorted(by_env.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


def freeze(env_name: str, conda: str | None = None,
           out_dir: Path | str | None = None) -> Path:
    """Capture an existing env to a committed lockfile (maintainer tool).

    Runs ``conda env export -n <env> --no-builds``, strips the host-specific
    ``prefix:`` line, and writes ``env_locks/<env>.yml`` — exactly what
    create_env rebuilds. Run on the host where the working env lives.
    """
    import re
    conda = conda or _conda_bin("conda")
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
    if "- pip:" not in "\n".join(lines):
        pip = _run(["run", "-n", env_name, "pip", "freeze"]).stdout
        pip_pkgs = [ln.strip() for ln in pip.splitlines()
                    if ln.strip() and not ln.startswith("-e ")]
        if pip_pkgs:
            if not any(l.startswith("dependencies:") for l in lines):
                lines.append("dependencies:")
            lines += ["  - pip", "  - pip:"] + [f"    - {p}" for p in pip_pkgs]
    dst = dst_dir / f"{env_name}.yml"
    dst.write_text("\n".join(lines) + "\n")
    return dst


# --- recipe/lockfile provisioning entry points -----------------------------
def create(method: str, env_name: str | None = None, conda: str | None = None,
           dry_run: bool = True) -> list[list[str]]:
    """Provision the env a method runs in. dry_run=True (default) returns the
    commands without running.

    Prefers the committed lockfile for the REAL env run() uses (group_for), so
    'what you build' == 'what runs'. Falls back to the hand-written recipe
    (its own scmb_<method> env) only when no lockfile was captured.
    """
    target = env_name or group_for(method)
    if lockfile(target) is not None:
        return create_env(target, conda=conda, dry_run=dry_run)
    cmds = create_commands(method, env_name=env_name, conda=conda)
    if not dry_run:
        _run_all(cmds)
    return cmds


def create_group(group: str, env_name: str | None = None, conda: str | None = None,
                 dry_run: bool = True) -> list[list[str]]:
    """Provision a shared group env. dry_run=True (default) returns the commands.

    Prefers the committed lockfile for the env; falls back to the hand recipe.
    """
    target = env_name or group
    if lockfile(target) is not None:
        return create_env(target, conda=conda, dry_run=dry_run)
    cmds = group_create_commands(group, env_name=env_name, conda=conda)
    if not dry_run:
        _run_all(cmds)
    return cmds


def _run_all(cmds: list[list[str]]) -> None:
    for c in cmds:
        proc = subprocess.run(c, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"env command failed: {' '.join(c)}\nstderr tail:\n{proc.stderr[-2000:]}"
            )
