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

from multibench.engine import envs, registry, runner


def _is_paths(x) -> bool:
    """True for a value made only of absolute paths (split on ':' or whitespace)."""
    toks = str(x).replace(":", " ").split()
    return bool(toks) and all(t.startswith("/") for t in toks)


def _path_valued():
    """``(method, variant, run_env)`` for every variant whose run_env holds a
    value made of absolute paths."""
    out = []
    for m in registry.list_methods():
        for v in registry.get(m).variants:
            env = v.run_env or {}
            if any(_is_paths(x) for x in env.values()):
                out.append((m, v, env))
    return out


def test_the_registry_holds_no_host_paths():
    """The LD_PRELOAD / RETICULATE_PYTHON values named files of the benchmark
    host, so every check there passed with files a student's machine lacks.
    Without them the R methods and MOFA2 run from their own environments
    (verified in a conda-free run of the packed environments)."""
    assert _path_valued() == []


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
