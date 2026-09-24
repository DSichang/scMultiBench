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
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, registry, runner

ROOT = Path(__file__).resolve().parents[1]
DARWIN = ("method environments are linux-64 conda envs (packed archives + "
          "lockfiles); this host is darwin/arm64")
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
GPU_VALUES = {"required", "used when present", "not used", "unknown"}


def _sub(*names):
    """The argparse sub-parser at ``names`` (e.g. ``'env', 'status'``)."""
    parser = cli.build_parser()
    for name in names:
        action = next(a for a in parser._actions
                      if isinstance(a, __import__("argparse")._SubParsersAction))
        parser = action.choices[name]
    return parser


def _opt_help(parser, flag):
    return next(a.help for a in parser._actions if flag in a.option_strings)


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


@pytest.fixture
def off_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)


def _h5(path, n_feat, n_cells, prefix="g"):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(n_feat, n_cells)).astype(float))
        g.create_dataset("features", data=np.array([f"{prefix}{i}" for i in range(n_feat)],
                                                   dtype="S12"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(n_cells)],
                                                   dtype="S12"))


def _cite(root, name="CITE", n=60):
    d = root / name
    d.mkdir(parents=True)
    _h5(d / "rna.h5", 30, n)
    _h5(d / "adt.h5", 10, n, prefix="p")
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / "cty.csv", index=False)
    return d


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
    assert reason.startswith("moETM needs an NVIDIA GPU; this computer has none.")
    assert "tools_scripts" not in reason and not re.search(r"\.py:\d+", reason)
    assert 'method_info("moETM")["requires_gpu"]' in reason


def test_gpu_refusal_names_the_cli_command_in_the_cli(tmp_path, no_gpu, monkeypatch, capsys):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    _cite(tmp_path)
    rc = cli.main(["scan", "CITE", "--category", "vertical", "--methods", "moETM",
                   "--data-path", str(tmp_path), "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0
    row = next(r for r in rows if r["modalities"] == "rna+adt")
    assert "multibench info moETM" in row["reason"]
    assert "tools_scripts" not in row["reason"] and "method_info(" not in row["reason"]


def test_scan_notes_quote_the_gpu_refusal_run_raises():
    """scan's Notes quote the env_reason of a GPU-only row; after M20 that is
    the one plain sentence, not the old CUDA/file:line text."""
    notes = " ".join(inspect.getdoc(W.scan).split())
    lead = registry.get("moETM").requires_gpu_reason.split(";")[0].replace("moETM", "<method>")
    assert f'{lead}; this computer has none.' in notes
    assert "calls CUDA unconditionally (<file>:<line>)" not in notes


def test_moetm_gpu_evidence_names_both_scripts():
    ev = mtb.method_info("moETM")["gpu_evidence"]
    assert "tools_scripts/moETM/main_moETM_rna_adt.py:109" in ev
    assert "tools_scripts/moETM/main_moETM_rna_atac.py:124" in ev


# ====================================================================== M30
def test_every_method_has_a_gpu_use_value():
    for m in mtb.list_methods():
        info = mtb.method_info(m)
        assert info["gpu"] in GPU_VALUES, (m, info["gpu"])
        assert (info["gpu"] == "required") == info["requires_gpu"], m
        if info["cpu_params"]:
            assert info["gpu"] == "used when present", m


def test_gpu_use_values_from_the_source():
    got = {m: mtb.method_info(m)["gpu"] for m in mtb.list_methods()}
    for m in ("StabMap", "UINMF", "iNMF", "online_iNMF", "Conos", "Seurat_v3",
              "Seurat_WNN", "Seurat_v5"):
        assert got[m] == "not used", m          # R methods: no CUDA call anywhere
    # MOFA2 is R but trains in mofapy2 with gpu_mode TRUE (main_MOFA2.Rmd:38)
    assert got["MOFA2"] == "used when present"
    for m in ("SCALEX", "scMoMaT", "uniPort", "Matilda", "sciPENN", "scJoint", "scMDC"):
        assert got[m] == "used when present", m
    # the package hides the GPU (CUDA_VISIBLE_DEVICES="") or passes --no_cuda
    for m in ("scMVP", "scMSI", "scMM"):
        assert got[m] == "not used", m
    assert got["totalVI"] == "unknown"


def test_d46_cell_count_is_recorded():
    obs = {o["dataset"]: o for o in mtb.method_info("StabMap")["runtime"]["observed"]}
    # rna1 + rna2 + rna3 barcodes of the fetched D46 folder: 8404 + 8845 + 4167
    assert obs["D46"]["cells"] == 21416


def test_info_prints_gpu_use_and_cells_per_observed_runtime(capsys):
    rc = cli.main(["info", "StabMap"])
    out = capsys.readouterr().out
    assert rc == 0
    assert re.search(r"^  GPU:\s+not used$", out, re.M)
    assert "209 s on D46 (21,416 cells)" in out
    assert "69 s on D52 (23,478 cells)" in out
    assert "many times longer" not in out           # no GPU, no CPU slowdown note
    rc = cli.main(["info", "moETM"])
    out = capsys.readouterr().out
    assert re.search(r"^  GPU:\s+required$", out, re.M)
    assert "many times longer" in out
    rc = cli.main(["info", "MIRA"])
    out = capsys.readouterr().out
    assert "on D12 (cells not recorded)" in out


def test_runtime_note_mentions_the_cpu_only_for_gpu_methods():
    assert "CPU-only" not in mtb.method_info("StabMap")["runtime"]["note"]
    assert "RTX 4090" in mtb.method_info("StabMap")["runtime"]["note"]
    assert "CPU-only" in mtb.method_info("scMoMaT")["runtime"]["note"]


# ====================================================================== M24
def test_run_all_dry_run_needs_no_out_dir(capsys):
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--methods", "Matilda",
                   "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "<out_dir>" in out


def test_run_all_without_dry_run_still_needs_out_dir(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["run-all", "D11", "--category", "vertical"])
    assert e.value.code == 2
    assert "the following arguments are required: --out-dir/--out" in capsys.readouterr().err


def test_run_all_out_dir_help_says_when_it_is_needed():
    assert "unless --dry-run" in _opt_help(_sub("run-all"), "--out-dir")


# ====================================================================== M26
PLAIN_REFUSAL = ("Method environments run only on Linux (this computer is darwin/arm64). "
                 "Run the install on a Linux machine; {force} tries anyway.")


def test_install_refusal_is_plain_in_python(off_linux, monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    with pytest.raises(RuntimeError) as e:
        envs.create_all(methods=["Matilda"], dry_run=False)
    assert str(e.value) == PLAIN_REFUSAL.format(force="force=True")


def test_install_refusal_is_plain_on_the_cli(off_linux, monkeypatch, capsys):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    rc = cli.main(["env", "install", "--methods", "Matilda", "--run"])
    err = capsys.readouterr().err
    assert rc == 1
    assert PLAIN_REFUSAL.format(force="--force") in err
    assert "linux-64 conda envs" not in err and "packed archives +" not in err
    assert err.count("run only on Linux") == 1          # no warning repeating it


def test_env_plan_fallback_line_agrees_in_number(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    sizes = {"a": {"archive_bytes": 10**9, "unpacked_bytes": 2 * 10**9},
             "b": {"archive_bytes": 10**9, "unpacked_bytes": 2 * 10**9}}
    one = cli._size_total_line([{"env": "a", "flavor": "gpu"}], sizes, flavor="cpu")
    assert one.startswith("# total for 1 env (GPU build; no CPU build is published for it)")
    two = cli._size_total_line([{"env": "a", "flavor": "gpu"}, {"env": "b", "flavor": "gpu"}],
                               sizes, flavor="cpu")
    assert two.startswith("# total for 2 envs (GPU builds; no CPU build is published for "
                          "these envs)")
    some = cli._size_total_line([{"env": "a", "flavor": "gpu"}, {"env": "b", "flavor": "cpu"},
                                 {"env": "c", "flavor": "gpu"}], sizes, flavor="cpu")
    assert some.startswith("# total for 3 envs (CPU builds; 2 envs have only a GPU build)")
    some = cli._size_total_line([{"env": "a", "flavor": "gpu"}, {"env": "b", "flavor": "cpu"}],
                                sizes, flavor="cpu")
    assert some.startswith("# total for 2 envs (CPU builds; 1 env has only a GPU build)")


def test_env_status_help_is_one_sentence_and_one_tag_per_line():
    p = _sub("env", "status")
    assert p.description == ("One line per method: installed or not, the environment "
                             "name and a difficulty tag.")
    text = p.format_help()
    assert "without surprises" not in text
    for tag in envs.DIFFICULTY:
        assert re.search(rf"^  {re.escape(tag)}\s+", text, re.M), tag


# ====================================================================== M27
def test_evaluate_and_run_help_texts():
    ev = _sub("evaluate")
    clus = _opt_help(ev, "--clustering")
    assert "they replace the sweep for ARI and NMI" in clus
    assert "iF1 still sweeps unless --metrics leaves it out" in clus
    meth = _opt_help(ev, "--method")
    assert "must be a registry method, which sets the label order" in meth
    dry = _opt_help(_sub("run"), "--dry-run")
    assert "environment activation included" in dry and "conda run -n" not in dry


def test_evaluate_unknown_method_on_the_dataset_route_names_labels(tmp_path, capsys):
    emb = tmp_path / "emb.npy"
    np.save(emb, np.zeros((10, 3)))
    rc = cli.main(["evaluate", "--output", str(emb), "--dataset", "D11", "--category",
                   "vertical", "--method", "PriyaNet"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "unknown method 'PriyaNet'" in err
    assert ("for your own method, pass the label files with --labels, once per file, "
            "in your embedding's cell order") in err


def test_evaluate_names_the_mudata_type():
    md = pytest.importorskip("mudata")
    import anndata as ad
    a = ad.AnnData(np.zeros((6, 2)))
    mdata = md.MuData({"rna": a})
    with pytest.raises(ValueError, match="not found in the MuData"):
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


@needs_git
def test_fetch_scripts_and_config_print_the_commit(tmp_path, monkeypatch, capsys):
    head = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "repo")
    rc = cli.main(["fetch", "--scripts"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == f"method scripts present: {tmp_path / 'repo' / 'tools_scripts'} at {head}\n"
    rc = cli.main(["config"])
    out = capsys.readouterr().out
    assert re.search(rf"^scripts_commit\s+{head}$", out, re.M), out
    assert f"(the method scripts in {tmp_path / 'repo' / 'tools_scripts'})" in out


def test_clone_failure_names_the_offline_route(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    target = tmp_path / "fresh"

    def dead_proxy(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)
    monkeypatch.setattr(subprocess, "run", dead_proxy)
    with pytest.raises(RuntimeError) as e:
        config.ensure_repo(target)
    msg = str(e.value)
    assert msg.startswith("could not reach github.com to fetch the method scripts.")
    assert "copy a scripts checkout" in msg and "set MULTIBENCH_REPO_PATH" in msg
    assert "returned non-zero exit status" not in msg
    assert not target.exists() and not target.with_name("fresh.partial").exists()


def test_scripts_ref_fetches_that_commit_or_tag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    ran = []

    def fake_run(argv, **kw):
        ran.append(list(argv))
        if argv[:2] in (["git", "init"], ["git", "clone"]):
            (Path(argv[-1]) / "tools_scripts").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "v1.2")
    config.ensure_repo(tmp_path / "a")
    fetch = next(a for a in ran if "fetch" in a)
    assert fetch[-1] == "v1.2" and "--depth" in fetch
    assert not any(a[:2] == ["git", "clone"] for a in ran)
    # the CLI flag wins over the variable
    ran.clear()
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "b")
    rc = cli.main(["fetch", "--scripts", "--ref", "0123abc"])
    assert rc == 0
    assert next(a for a in ran if "fetch" in a)[-1] == "0123abc"
    # without a ref: the unchanged default-branch clone
    ran.clear()
    monkeypatch.delenv(config.SCRIPTS_REF_VAR)
    config.ensure_repo(tmp_path / "c")
    assert ran[0][:4] == ["git", "clone", "--depth", "1"]


def test_ref_flag_needs_scripts(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["fetch", "D11", "--ref", "v1"])
    assert e.value.code == 2 and "--ref" in capsys.readouterr().err


def test_run_result_carries_provenance(tmp_path, monkeypatch):
    import dataclasses
    names = {f.name for f in dataclasses.fields(runner.RunResult)}
    assert {"scripts_commit", "env_flavor", "hostname"} <= names
    # defaults keep the old constructor working
    r = runner.RunResult(method="m", out_dir=tmp_path, cmd=[], output=None, extra={})
    assert r.scripts_commit is None and r.env_flavor == "unknown" and r.hostname == ""


class _FakePopen:
    def __init__(self, cmd, cwd, stdout, stderr, text, env=None, start_new_session=False):
        self.pid = 4242
        self.returncode = 0
        with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=np.zeros((60, 3)))

    def communicate(self):
        return "", ""


def test_run_fills_the_provenance(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(config, "scripts_commit", lambda repo=None: "f" * 40)
    monkeypatch.setattr(envs, "installed_flavor", lambda env, conda=None: "cpu")
    d = _cite(tmp_path)
    res = mtb.run("scMDC", "vertical", inputs={"rna": str(d / "rna.h5"),
                                               "adt": str(d / "adt.h5")},
                  out_dir=str(tmp_path / "o"), convert=False, cmd_template="{cmd}")
    assert res.scripts_commit == "f" * 40
    assert res.hostname == socket.gethostname()
    # a {cmd} template runs in the caller's own env: its build is not known
    assert res.env_flavor == "unknown"
    # the package's env: the flavour its install recorded
    assert config.run_provenance("scmb_scmdc")["env_flavor"] == "cpu"


def test_run_all_records_the_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(config, "scripts_commit", lambda repo=None: "e" * 40)
    monkeypatch.setattr(envs, "installed_flavor", lambda env, conda=None: None)

    class _Res:
        output = np.zeros((2864, 5))
    monkeypatch.setattr(W, "_run", lambda **kw: _Res())
    res = mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir=str(tmp_path),
                      evaluate=False, verbose=False)
    rec = res.results[0]
    assert rec["scripts_commit"] == "e" * 40
    assert rec["env_flavor"] == "unknown"
    assert rec["hostname"]
    saved = json.loads((tmp_path / "batch_result.json").read_text())
    blob = json.dumps(saved)
    assert '"scripts_commit"' in blob and '"hostname"' in blob


@needs_git
def test_present_scripts_must_be_at_the_requested_ref(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    assert config.ensure_repo(repo, ref=head[:12]) == repo          # commit prefix
    env = {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin",
           "HOME": str(repo)}
    subprocess.run(["git", "tag", "v1"], cwd=repo, check=True, env=env)
    assert config.ensure_repo(repo, ref="v1") == repo               # a local tag
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "v2")
    # the real run names the scripts folder
    with pytest.raises(RuntimeError,
                       match=rf"^the method scripts in {re.escape(str(repo))} are at "
                             rf"{head[:7]}, not v2 \(MULTIBENCH_SCRIPTS_REF\)"):
        config.ensure_repo(repo)                                     # silently other code: no
    with pytest.raises(RuntimeError, match=r"not a \(--ref\)"):
        config.ensure_repo(repo, ref="a")                            # too short to be a prefix
