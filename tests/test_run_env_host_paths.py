"""A variant's ``run_env`` path values apply only where the paths exist.

methods.yaml records files of the benchmark host in ``run_env``
(``LD_PRELOAD`` on the R variants, ``RETICULATE_PYTHON`` on MOFA2). Merged
over ``os.environ`` as they were, they forced a nonexistent Python on
reticulate and preloaded a missing library everywhere else. The runner now
keeps only the paths that exist and leaves the key unset when none does.

Where every path exists - the benchmark host - the child environment must be
byte-identical to the old merge. The first two tests prove that on the
registry's own values, re-rooted under ``tmp_path`` and created there, so they
hold on any machine.
"""
from __future__ import annotations

import dataclasses
import os
import re

import pytest

from multibench.engine import registry, runner


def _path_valued():
    """``(method, variant, run_env)`` for every variant whose run_env holds a
    value made of absolute paths."""
    out = []
    for m in registry.list_methods():
        for v in registry.get(m).variants:
            env = v.run_env or {}
            if any(str(x).replace(":", " ").split()
                   and all(p.startswith("/") for p in str(x).replace(":", " ").split())
                   for x in env.values()):
                out.append((m, v, env))
    return out


def _rerooted(env: dict, root) -> dict:
    """``env`` with every absolute path moved under ``root`` and created there."""
    out = {}
    for k, x in env.items():
        x = str(x)
        toks = x.replace(":", " ").split()
        if toks and all(t.startswith("/") for t in toks):
            for t in toks:
                p = f"{root}{t}"
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").close()
            x = re.sub(r"[^:\s]+", lambda mo: f"{root}{mo.group(0)}", x)
        out[k] = x
    return out


def _old_merge(run_env: dict) -> dict:
    """The child environment exactly as the runner built it before the fix."""
    env = {**os.environ, "PYTHONNOUSERSITE": "1", **{k: str(v) for k, v in run_env.items()}}
    if "MPLBACKEND" not in run_env:
        env["MPLBACKEND"] = "Agg"
    return env


class _Spawned(Exception):
    pass


def _child_env(monkeypatch, tmp_path, method: str, run_env: dict) -> dict:
    """Run ``method`` with its first variant's run_env replaced; return the env
    handed to the subprocess (Popen is stopped before anything runs)."""
    spec = registry.get(method)
    variant = dataclasses.replace(spec.variants[0], run_env=run_env)
    fake = dataclasses.replace(spec, variants=[variant])
    monkeypatch.setattr(runner.registry, "get", lambda m: fake if m == method else spec)
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(runner, "check_gpu_requirement", lambda spec: None)
    monkeypatch.setattr(runner, "run_mode", lambda env: ("conda", None))  # no env probe
    seen = {}

    def popen(cmd, **kw):
        seen["env"] = kw["env"]
        raise _Spawned

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    (tmp_path / "repo" / "tools_scripts").mkdir(parents=True, exist_ok=True)
    inputs = {role: str(tmp_path / f"{role}.h5") for role in variant.when["modalities"]}
    with pytest.raises(_Spawned):
        runner.run(method, variant.when["category"], inputs=inputs,
                   out_dir=str(tmp_path / "out"), convert=False,
                   repo_path=tmp_path / "repo")
    return seen["env"]


def test_the_registry_path_values_pass_unchanged_where_the_paths_exist(tmp_path):
    found = _path_valued()
    keys = {k for _, _, env in found for k in env}
    assert {"LD_PRELOAD", "RETICULATE_PYTHON"} <= keys, keys
    for m, v, env in found:
        env = _rerooted(env, tmp_path)
        assert runner._host_run_env(env) == {k: str(x) for k, x in env.items()}, m


def test_where_the_paths_exist_the_child_env_is_the_old_one(monkeypatch, tmp_path):
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("RETICULATE_PYTHON", raising=False)
    env = _rerooted(registry.get("MOFA2").variants[0].run_env, tmp_path)
    assert set(env) == {"LD_PRELOAD", "RETICULATE_PYTHON"}
    assert _child_env(monkeypatch, tmp_path, "MOFA2", env) == _old_merge(env)


def test_a_missing_path_leaves_the_callers_value(monkeypatch, tmp_path):
    monkeypatch.setenv("RETICULATE_PYTHON", "/opt/mine/bin/python")
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    gone = str(tmp_path / "absent")
    child = _child_env(monkeypatch, tmp_path, "MOFA2",
                       {"RETICULATE_PYTHON": f"{gone}/bin/python",
                        "LD_PRELOAD": f"{gone}/lib/libstdc++.so.6"})
    assert child["RETICULATE_PYTHON"] == "/opt/mine/bin/python"
    assert "LD_PRELOAD" not in child


@pytest.mark.parametrize("sep", [":", " "])
def test_only_the_existing_paths_of_a_list_are_applied(tmp_path, sep):
    have = tmp_path / "libhave.so"
    have.touch()
    value = sep.join([str(tmp_path / "libgone.so"), str(have)])
    assert runner._host_run_env({"LD_PRELOAD": value}) == {"LD_PRELOAD": str(have)}
    two = tmp_path / "libtwo.so"
    two.touch()
    value = sep.join([str(have), str(tmp_path / "libgone.so"), str(two)])
    assert runner._host_run_env({"LD_PRELOAD": value}) == {
        "LD_PRELOAD": sep.join([str(have), str(two)])}


def test_other_values_pass_unchanged():
    env = {"CUDA_VISIBLE_DEVICES": "", "R_FUTURE_GLOBALS_MAXSIZE": 2147483648,
           "MIXED": "/no/such/lib.so libc.so.6", "URL": "http://example.org/x"}
    assert runner._host_run_env(env) == {k: str(v) for k, v in env.items()}
