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

_GROUPS_YAML = Path(__file__).resolve().parent / "env_groups.yaml"


# --- recipes ---------------------------------------------------------------
def recipe(method: str) -> dict:
    """Return the method's environment recipe (its ``env_spec``); ``{}`` if none."""
    return registry.get(method).env_spec or {}


def default_env_name(method: str) -> str:
    """Default conda env name for a method: ``scmb_<method>``."""
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


def groups() -> dict:
    """All install groups: shared merged envs + singleton (own-env) methods.

    Returns {env_name: spec}; merged groups serve several methods, singletons one.
    """
    merged = _merged_groups()
    covered = {m for g in merged.values() for m in g.get("members", [])}
    out = {name: {**g, "shared": True} for name, g in merged.items()}
    for s in registry.load():
        if s.id not in covered:
            out[default_env_name(s.id)] = {
                **(s.env_spec or {}), "members": [s.id], "shared": False,
            }
    return out


def group_for(method: str) -> str:
    """The env name that serves this method (a shared group, or its own env)."""
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


def group_environment_yml(group: str) -> str:
    spec = groups().get(group)
    if spec is None:
        raise KeyError(f"unknown group {group!r}; see envs.groups()")
    return _environment_yml(spec, group)


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
    conda = conda or _conda_bin("conda")
    try:
        out = subprocess.run([conda, "env", "list"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [ln.split()[0] for ln in out.splitlines()
            if ln.strip() and not ln.startswith("#")]


def env_exists(env_name: str, conda: str | None = None) -> bool:
    return env_name in installed_envs(conda)


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
            "exists": env in have or grp in have or s.env in have,
            "difficulty": r.get("difficulty", "unknown"),
            "verified_working": bool(r.get("verified_working", False)),
            "has_recipe": bool(r),
        })
    return out


def create(method: str, env_name: str | None = None, conda: str | None = None,
           dry_run: bool = True) -> list[list[str]]:
    """Create a method's env. dry_run=True (default) returns commands without running."""
    cmds = create_commands(method, env_name=env_name, conda=conda)
    if not dry_run:
        _run_all(cmds)
    return cmds


def create_group(group: str, env_name: str | None = None, conda: str | None = None,
                 dry_run: bool = True) -> list[list[str]]:
    """Create a shared group env. dry_run=True (default) returns commands."""
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
