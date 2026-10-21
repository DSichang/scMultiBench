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
OTHER = "holds other files and no method scripts."


@pytest.fixture
def host(tmp_path, monkeypatch):
    """No scripts in the package root; every env installed; a GPU; no ref pinned."""
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    return tmp_path


def test_a_failed_fetch_leaves_the_empty_folder_as_it_was(host, monkeypatch):
    target = host / "scripts"
    target.mkdir()

    def dead_proxy(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)
    monkeypatch.setattr(subprocess, "run", dead_proxy)
    with pytest.raises(RuntimeError, match="^Could not reach github.com"):
        config.ensure_repo(target)
    assert target.is_dir() and not os.listdir(target)


# ================================================= (2) --strict: scripts not fetched
#: 0.4.0: scan --strict is deprecated; its first stderr line says so
STRICT_DEPRECATED = ("warning: scan --strict is deprecated and will be removed in 0.5. "
                     "Use the exit code of multibench run-all.\n")


