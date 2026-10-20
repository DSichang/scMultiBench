"""Round-4 student study, package work 'ing': R4-06 (method scripts not fetched).

- An existing empty ``repo_path`` is filled in place by ``multibench fetch
  --scripts`` and by the first run; the folder itself is kept.
- A ``repo_path`` that holds other files and no ``tools_scripts/`` is refused
  with one sentence, which ``scan`` (reason, not runnable), the dry run and
  ``multibench config`` repeat.
- ``scan --strict`` fails while the method scripts are not fetched; the Python
  ``scan`` keeps such rows runnable, with the caveat.
"""
import os
import argparse
import re
import shutil
import stat
import subprocess
import warnings

import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, registry
from tests.conftest import full_scan

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
#: the fix of a folder with other files: the safe action first (review of wp/f4_int)
FIX = ("an empty or new folder, then run multibench fetch --scripts. If the folder is "
       "left over from an earlier fetch, you can remove it instead")
PY_FIX = f"Set MULTIBENCH_REPO_PATH or mtb.config.DEFAULT.repo_path to {FIX}"
CLI_FIX = f"Set MULTIBENCH_REPO_PATH to {FIX}"
OTHER = "holds other files and no method scripts."
MATILDA = ["scan", "D11", "--category", "vertical", "--methods", "Matilda",
           "--modalities", "rna,adt"]


def _upstream(path):
    """A local git repository standing in for PYangLab/scMultiBench; returns HEAD."""
    ep = path / registry.get("Matilda").variants[0].entrypoint
    ep.parent.mkdir(parents=True)
    ep.write_text("print(1)\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(path), "PATH": "/usr/bin:/bin"}
    for argv in (["git", "init", "-q"], ["git", "add", "."],
                 ["git", "commit", "-q", "-m", "x"], ["git", "tag", "v1"]):
        subprocess.run(argv, cwd=path, check=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def host(tmp_path, monkeypatch):
    """No scripts in the package root; every env installed; a GPU; no ref pinned."""
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    return tmp_path


def _use(monkeypatch, repo):
    monkeypatch.setattr(config.DEFAULT, "repo_path", repo)


@needs_git
def test_first_run_fills_an_empty_folder_at_a_ref(host, monkeypatch):
    head = _upstream(host / "upstream")
    monkeypatch.setattr(config, "SCRIPTS_URL", (host / "upstream").as_uri())
    target = host / "scripts"
    (target / config._PARTIAL / "junk").mkdir(parents=True)   # an interrupted fetch
    assert config.scripts_folder_problem(target) is None
    assert config.ensure_repo(target, ref="v1") == target
    assert config.scripts_commit(target) == head
    assert config.scripts_ref_problem(target) is None
    assert not (target / config._PARTIAL).exists()


def test_a_failed_fetch_leaves_the_empty_folder_as_it_was(host, monkeypatch):
    target = host / "scripts"
    target.mkdir()

    def dead_proxy(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)
    monkeypatch.setattr(subprocess, "run", dead_proxy)
    with pytest.raises(RuntimeError, match="^Could not reach github.com"):
        config.ensure_repo(target)
    assert target.is_dir() and not os.listdir(target)


def test_config_keeps_the_fetch_hint_for_an_empty_folder(host, monkeypatch):
    target = host / "scripts"
    target.mkdir()
    _use(monkeypatch, target)
    row = next(r for r in config._sources() if r["name"] == "scripts_commit")
    assert (row["value"], row["source"]) == (
        "not fetched", "multibench fetch --scripts fetches the method scripts")


# ================================================= (2) --strict: scripts not fetched
#: 0.4.0: scan --strict is deprecated; its first stderr line says so
STRICT_DEPRECATED = ("warning: scan --strict is deprecated and will be removed in 0.5. "
                     "Use the exit code of multibench run-all.\n")


