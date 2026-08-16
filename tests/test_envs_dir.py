"""The envs directory must resolve across conda/mamba `info --json` schemas.

Getting this wrong is silent and expensive: `env install --packed` degrades to
a 10-30 minute lockfile build per environment (mamba 2 dropped `root_prefix`),
or worse, unpacks into a directory conda never searches.
"""
import json
import os
import stat
from pathlib import Path

import pytest

from multibench.engine import envs


def _fake_conda(tmp_path: Path, name: str, payload) -> str:
    """A conda/mamba stand-in at <root>/bin/<name> printing `payload`."""
    root = tmp_path / "root"
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "conda-meta").mkdir(exist_ok=True)
    exe = root / "bin" / name
    body = json.dumps(payload) if payload is not None else "not json at all"
    exe.write_text(f"#!/bin/sh\ncat <<'EOF'\n{body}\nEOF\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


CONDA_SCHEMA = {"root_prefix": "@ROOT@", "envs_dirs": ["@ROOT@/envs"]}
MAMBA2_SCHEMA = {"base environment": "@ROOT@", "envs directories": ["@ROOT@/envs"]}
ROOT_ONLY_CONDA = {"root_prefix": "@ROOT@"}
ROOT_ONLY_MAMBA2 = {"base environment": "@ROOT@"}


@pytest.mark.parametrize("payload", [CONDA_SCHEMA, MAMBA2_SCHEMA,
                                     ROOT_ONLY_CONDA, ROOT_ONLY_MAMBA2])
def test_every_schema_resolves_to_the_envs_dir(tmp_path, payload):
    root = tmp_path / "root"
    filled = json.loads(json.dumps(payload).replace("@ROOT@", str(root)))
    exe = _fake_conda(tmp_path, "mamba", filled)
    assert envs._envs_dir(exe) == root / "envs"


@pytest.mark.parametrize("payload", [None, {}, {"unrelated": 1}])
def test_unparseable_info_falls_back_to_the_binary_location(tmp_path, payload):
    """<root>/bin/<tool> is the one layout every installation shares."""
    root = tmp_path / "root"
    exe = _fake_conda(tmp_path, "mamba", payload)
    assert envs._envs_dir(exe) == root / "envs"


def test_unwritable_candidate_is_skipped(tmp_path):
    """An env unpacked where conda cannot write is worse than falling back."""
    root = tmp_path / "root"
    blocked = tmp_path / "blocked"
    (blocked / "envs").mkdir(parents=True)
    os.chmod(blocked / "envs", 0o500)
    exe = _fake_conda(tmp_path, "mamba", {
        "envs directories": [str(blocked / "envs"), str(root / "envs")]})
    try:
        if os.access(blocked / "envs", os.W_OK):
            pytest.skip("running as root: unwritable directories are writable")
        assert envs._envs_dir(exe) == root / "envs"
    finally:
        os.chmod(blocked / "envs", 0o700)


def test_no_conda_at_all_returns_none(tmp_path):
    assert envs._envs_dir(str(tmp_path / "nonexistent-conda")) is None
