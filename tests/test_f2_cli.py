"""Fix round 2, work package 'cli': version, GPU wording and GPU use, run-all
dry run without --out-dir, plain env/evaluate texts, literal blocks in
docstrings, and the method-scripts commit in fetch/config/run records.

Every platform- or GPU-dependent test pins ``envs.host_platform_problem`` /
``envs.host_has_gpu``, so it holds on Linux and macOS alike.
"""
import inspect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, registry, runner

ROOT = Path(__file__).resolve().parents[1]
DARWIN = ("method environments are linux-64 conda envs (packed archives + "
          "lockfiles); this host is darwin/arm64")
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
GPU_VALUES = {"required", "used when present", "not used", "unknown"}


@pytest.fixture
def off_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)


# ====================================================================== M14
def test_version_is_past_the_pypi_release():
    """0.3.1 is on PyPI; a build with new commands must not reuse the number."""
    assert tuple(int(p) for p in mtb.__version__.split(".")[:3]) > (0, 3, 1)


def test_pyproject_reads_the_version_from_the_package():
    text = (ROOT / "pyproject.toml").read_text()
    project = text.split("[project]\n", 1)[1].split("\n[", 1)[0]
    assert not re.search(r"^version\s*=", project, re.M), "a second version source"
    assert 'dynamic = ["version"]' in project
    assert 'version = {attr = "multibench.__version__"}' in text


# ====================================================================== M20
def test_gpu_refusal_is_one_plain_sentence_and_a_hint():
    reason = registry.get("moETM").requires_gpu_reason
    assert reason.startswith("moETM needs an NVIDIA GPU, and this computer has none.")
    assert "tools_scripts" not in reason and not re.search(r"\.py:\d+", reason)
    assert reason == "moETM needs an NVIDIA GPU, and this computer has none."   # Z17 (r3)


def test_d46_cell_count_is_recorded():
    obs = {o["dataset"]: o for o in mtb.method_info("StabMap")["runtime"]["observed"]}
    # rna1 + rna2 + rna3 barcodes of the fetched D46 folder: 8404 + 8845 + 4167
    assert obs["D46"]["cells"] == 21416


def test_runtime_note_mentions_the_cpu_only_for_gpu_methods():
    assert "CPU-only" not in mtb.method_info("StabMap")["runtime"]["note"]
    assert "RTX 4090" in mtb.method_info("StabMap")["runtime"]["note"]
    assert "CPU-only" not in mtb.method_info("scMoMaT")["runtime"]["note"]     # Z33 (r3)


# ====================================================================== M26
PLAIN_REFUSAL = ("Method environments run only on Linux, and this computer runs macOS. "
                 "Run the install on a Linux machine. {force} tries anyway.")


def test_install_refusal_is_plain_in_python(off_linux, monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    with pytest.raises(RuntimeError) as e:
        envs.create_all(methods=["Matilda"], dry_run=False)
    # the hidden force= is not offered since 0.4.0 (X33)
    assert str(e.value) == PLAIN_REFUSAL.replace(" {force} tries anyway.", "")


def test_env_plan_fallback_line_agrees_in_number(monkeypatch):
    # Z27 (r3): no builds line; the total sums the archive each row takes
    sizes = {"a": {"archive_bytes": 10**9, "unpacked_bytes": 2 * 10**9},
             "b": {"archive_bytes": 4 * 10**9, "unpacked_bytes": 8 * 10**9},
             "b-cpu": {"archive_bytes": 10**9, "unpacked_bytes": 2 * 10**9}}
    rows = [{"env": "a", "flavor": "gpu"}, {"env": "b", "flavor": "cpu"}]
    assert envs.plan_totals(rows, sizes, layers={})[1] == \
        "2 envs: 2.0 GB to download, 4.0 GB on disk"
    rows = [{"env": "a", "flavor": "any"}, {"env": "b", "flavor": "gpu"}]
    assert envs.plan_totals(rows, sizes, layers={})[1] == \
        "2 envs: 5.0 GB to download, 10.0 GB on disk"


def test_evaluate_names_the_mudata_type():
    md = pytest.importorskip("mudata")
    import anndata as ad
    a = ad.AnnData(np.zeros((6, 2)))
    mdata = md.MuData({"rna": a})
    with pytest.raises(ValueError, match=r"The MuData has no obsm\['X_emb'\]"):
        mtb.evaluate(mdata, labels=np.array(["A", "B"] * 3), metrics=["ASW"])


# ====================================================================== M32
def _public_objects():
    sys.path.insert(0, str(Path(__file__).parent))
    from test_examples_width import _public_objects as objs
    return list(objs())


def test_no_public_docstring_line_ends_in_a_literal_block_marker():
    """The site renders ``::`` as text; Notes use fenced blocks instead."""
    bad = []
    for path, obj in _public_objects():
        for line in (inspect.getdoc(obj) or "").splitlines():
            if line.rstrip().endswith("::"):
                bad.append(f"{path}: {line.strip()}")
    assert bad == []


# ====================================================================== M33
def _git_repo(path):
    """A git checkout with tools_scripts/ and one commit; returns its HEAD."""
    (path / "tools_scripts").mkdir(parents=True)
    (path / "tools_scripts" / "x.py").write_text("print(1)\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(path), "PATH": "/usr/bin:/bin"}
    for argv in (["git", "init", "-q"], ["git", "add", "."],
                 ["git", "commit", "-q", "-m", "x"]):
        subprocess.run(argv, cwd=path, check=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@needs_git
def test_scripts_commit_matches_git_rev_parse(tmp_path):
    head = _git_repo(tmp_path / "repo")
    assert config.scripts_commit(tmp_path / "repo") == head
    (tmp_path / "plain" / "tools_scripts").mkdir(parents=True)
    assert config.scripts_commit(tmp_path / "plain") is None      # not a checkout


def test_clone_failure_names_the_offline_route(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    target = tmp_path / "fresh"

    def dead_proxy(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)
    monkeypatch.setattr(subprocess, "run", dead_proxy)
    with pytest.raises(RuntimeError) as e:
        config.ensure_repo(target)
    msg = str(e.value)
    assert msg.startswith("Could not reach github.com to fetch the method scripts.")
    assert "copy the folder" in msg and "set MULTIBENCH_REPO_PATH" in msg
    assert "returned non-zero exit status" not in msg
    assert not target.exists() and not target.with_name("fresh.partial").exists()


def test_run_result_carries_provenance(tmp_path, monkeypatch):
    import dataclasses
    names = {f.name for f in dataclasses.fields(runner.RunResult)}
    assert {"scripts_commit", "env_flavor", "hostname"} <= names
    # defaults keep the old constructor working
    r = runner.RunResult(method="m", out_dir=tmp_path, cmd=[], output=None, extra={})
    assert r.scripts_commit is None and r.env_flavor == "unknown" and r.hostname == ""


def test_run_all_records_the_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(config, "scripts_commit", lambda repo=None: "e" * 40)
    monkeypatch.setattr(envs, "installed_flavor", lambda env, conda=None: None)

    class _Res:
        output = np.zeros((2864, 5))
    monkeypatch.setattr(W, "_run", lambda **kw: _Res())
    res = mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir=str(tmp_path),
                      evaluate=False, verbose=False)
    rec = res.records[0]
    assert rec["scripts_commit"] == "e" * 40
    assert rec["env_flavor"] == "unknown"
    assert rec["hostname"]
    saved = json.loads((tmp_path / "batch_result.json").read_text())
    blob = json.dumps(saved)
    assert '"scripts_commit"' in blob and '"hostname"' in blob


