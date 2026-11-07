"""Round-4 student study, package work 'ing': R4-06 (method scripts not fetched).

- An existing empty ``repo_path`` is filled in place by the first run; the
  folder itself is kept.
- A ``repo_path`` that holds other files and no ``tools_scripts/`` is refused
  with one sentence, which ``scan`` (reason, not runnable) and the dry run
  repeat.
- ``scan`` keeps the rows runnable while the method scripts are not fetched,
  with the caveat.
"""
import os
import shutil
import subprocess

import pytest

from multibench.engine import scripts
from multibench import _settings
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
OTHER = "holds other files and no method scripts."


@pytest.fixture
def host(tmp_path, monkeypatch):
    """No scripts in the package root; every env installed; a GPU; no ref pinned."""
    monkeypatch.setattr(_settings, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.delenv(scripts.SCRIPTS_REF_VAR, raising=False)
    return tmp_path


def test_a_failed_fetch_leaves_the_empty_folder_as_it_was(host, monkeypatch):
    target = host / "scripts"
    target.mkdir()

    def dead_proxy(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)
    monkeypatch.setattr(subprocess, "run", dead_proxy)
    with pytest.raises(RuntimeError, match="^Could not reach github.com"):
        scripts.ensure_repo(target)
    assert target.is_dir() and not os.listdir(target)
