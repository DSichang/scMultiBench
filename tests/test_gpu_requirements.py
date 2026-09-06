"""The GPU/CPU contract of the upstream scripts, and what the package does with it.

CPU-only builds of the method envs exist for Colab CPU runtimes and laptops,
so the registry now says, per method and from the script's own source, which
of three things is true: the script has a switch that turns CUDA off
(``cpu_params`` - scJoint's ``--use_cuda ""``, scMDC's ``--device cpu``), the
script calls CUDA unconditionally (``requires_gpu`` + ``gpu_evidence``, a
``file:line``), or neither (the script has no CUDA call, or falls back on
``torch.cuda.is_available()`` itself). On a host without an NVIDIA GPU
(``envs.host_has_gpu()`` False - this laptop, a CPU Colab) ``run`` merges the
``cpu_params`` unless the caller set the key, and refuses a ``requires_gpu``
method with ``OSError`` before launching; ``scan`` / ``run_all`` report the
same sentence in the env gate. On a GPU host nothing changes. Every test
patches ``host_has_gpu`` both ways, so it holds on ANY machine.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry, runner, schema

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())

#: the classification snapshot - a registry change must change this test
GPU_ONLY = {"scBridge", "moETM", "SMILE", "sciCAN", "UnitedNet", "SPIRAL", "iPOLNG"}
CPU_SWITCH = {"scJoint": {"use_cuda": ""}, "scMDC": {"device": "cpu"}}


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


@pytest.fixture
def all_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)


def _h5(path, n_feat, n_cells, prefix="g"):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(n_feat, n_cells)).astype(float))
        g.create_dataset("features", data=np.array([f"{prefix}{i}" for i in range(n_feat)], dtype="S12"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(n_cells)], dtype="S12"))


def _cite(root, name="CITE", n=60):
    """A vertical rna+adt dataset folder: what moETM (gpu-only) and scMDC
    (cpu-switch) both take."""
    d = root / name
    d.mkdir(parents=True)
    _h5(d / "rna.h5", 30, n)
    _h5(d / "adt.h5", 10, n, prefix="p")
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / "cty.csv", index=False)
    return d


class _FakePopen:
    """Writes the embedding the runner loads; records argv; launches nothing."""
    calls: dict = {}

    def __init__(self, cmd, cwd, stdout, stderr, text, env=None, start_new_session=False):
        _FakePopen.calls["cmd"] = cmd
        self.pid = 4242
        self.returncode = 0
        with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=np.zeros((6, 3)))

    def communicate(self):
        return "", ""

    def kill(self):
        pass

    def wait(self):
        return self.returncode


def _must_not_spawn(*a, **k):
    raise AssertionError("Popen must not be reached")


# ------------------------------------------------------------ the snapshot

def test_requires_gpu_set_is_pinned():
    got = {m for m in mtb.list_methods() if mtb.method_info(m)["requires_gpu"]}
    assert got == GPU_ONLY, got


def test_cpu_params_are_pinned():
    got = {m: mtb.method_info(m)["cpu_params"] for m in mtb.list_methods()
           if mtb.method_info(m)["cpu_params"]}
    assert got == CPU_SWITCH, got
    # exactly what the scripts need: argparse type=bool is truthy for any
    # non-empty string, so scJoint's switch is the EMPTY string
    assert mtb.method_info("scJoint")["cpu_params"] == {"use_cuda": ""}
    assert mtb.method_info("scMDC")["cpu_params"] == {"device": "cpu"}


def test_gpu_evidence_names_a_cuda_call():
    """``gpu_evidence`` is ``file:line`` of the unconditional CUDA call; when
    the reference checkout is here, that line really says cuda."""
    root = runner._repo_root_no_fetch()
    for m in mtb.list_methods():
        info = mtb.method_info(m)
        ev = info["gpu_evidence"]
        if not info["requires_gpu"]:
            assert ev is None, m
            continue
        path, _, line = ev.rpartition(":")
        assert path.startswith("tools_scripts/") and line.isdigit(), (m, ev)
        f = root / path
        if f.is_file():
            text = f.read_text(errors="replace").splitlines()[int(line) - 1]
            assert "cuda" in text.lower(), (m, ev, text)


def test_method_info_keys_and_defaults():
    for m in mtb.list_methods():
        info = mtb.method_info(m)
        assert isinstance(info["cpu_params"], dict)
        assert isinstance(info["requires_gpu"], bool)
        assert info["gpu_evidence"] is None or isinstance(info["gpu_evidence"], str)
    info = mtb.method_info("SCALEX")
    assert info["cpu_params"] == {} and info["requires_gpu"] is False
    assert info["gpu_evidence"] is None
    assert "cpu_params" in (mtb.method_info.__doc__ or "")
    assert "requires_gpu" in (mtb.method_info.__doc__ or "")


def test_scmm_and_scmvp_keep_their_existing_cpu_treatment():
    """scMM turns CUDA off through its own ``--no_cuda`` default param and
    scMVP / scMSI through ``run_env: CUDA_VISIBLE_DEVICES=""`` - both
    predate cpu_params and both scripts fall back on is_available(), so
    they need neither new key."""
    for v in registry.get("scMM").variants:
        assert v.params.get("no_cuda") is True
    for m in ("scMVP", "scMSI"):
        for v in registry.get(m).variants:
            assert v.run_env.get("CUDA_VISIBLE_DEVICES") == ""
        assert mtb.method_info(m)["cpu_params"] == {}
        assert mtb.method_info(m)["requires_gpu"] is False
    assert mtb.method_info("scMM")["cpu_params"] == {}


def test_cpu_params_keys_are_accepted_params_of_every_variant():
    """The runner emits cpu_params below the params check, so the registry
    guarantees they are params the script declares (tunable | params) -
    and params_for / run_all therefore accept an explicit override."""
    for m, cp in CPU_SWITCH.items():
        for v in registry.get(m).variants:
            assert set(cp) <= set(v.tunable) | set(v.params), (m, v.when)
    assert "use_cuda" in mtb.params_for("scJoint")["tunable"]
    assert "device" in mtb.params_for("scMDC", "vertical", ["rna", "adt"])["tunable"]
    # the cross variant runs the SAME script as the vertical ones: it inherits
    # their tunables (params.yaml has no entry for it)
    cross = registry.get("scMDC").select("cross", {"rna1", "rna2", "rna3", "adt1", "adt2", "adt3"})
    assert cross.tunable == registry.get("scMDC").select("vertical", {"rna", "adt"}).tunable
    assert "device" in mtb.params_for("scMDC", "cross")["tunable"]


# ------------------------------------------------------------ schema validation

_VARIANT = {"when": {"category": "vertical", "modalities": ["rna", "adt"]},
            "entrypoint": "tools_scripts/X/main.py",
            "args": [{"role": "rna", "flag": "--a"}, {"role": "adt", "flag": "--b"},
                     {"role": "out_dir", "flag": "--o"}],
            "output": {"kind": "embedding", "file": "embedding.h5", "dataset": "data"},
            "tunable": {"device": {"default": "cuda", "type": "str"}}}


def _parse(**extra):
    return registry._parse_method({"id": "X", "variants": [_VARIANT], **extra})


def test_schema_accepts_the_two_valid_shapes():
    spec = _parse(cpu_params={"device": "cpu"})
    assert spec.cpu_params == {"device": "cpu"} and not spec.requires_gpu
    assert spec.requires_gpu_reason == ""
    spec = _parse(requires_gpu=True, gpu_evidence="tools_scripts/X/main.py:12")
    assert spec.requires_gpu and spec.cpu_params == {}
    assert spec.requires_gpu_reason == (
        'X needs an NVIDIA GPU: the upstream script calls CUDA unconditionally '
        '(tools_scripts/X/main.py:12); see method_info(m)["requires_gpu"]')
    # absent keys -> the defaults
    spec = _parse()
    assert spec.cpu_params == {} and spec.requires_gpu is False and spec.gpu_evidence == ""


def test_schema_rejects_bad_shapes():
    with pytest.raises(ValueError, match="X: cpu_params must be a mapping"):
        _parse(cpu_params=["--device", "cpu"])
    with pytest.raises(ValueError, match="cpu_params keys must be non-empty"):
        _parse(cpu_params={"": "cpu"})
    with pytest.raises(ValueError, match=r"cpu_params\['device'\] must be a command-line value"):
        _parse(cpu_params={"device": None})
    with pytest.raises(ValueError, match="None/False are never emitted"):
        _parse(cpu_params={"device": False})
    with pytest.raises(ValueError, match="requires_gpu must be true or false"):
        _parse(requires_gpu="yes", gpu_evidence="a.py:1")
    with pytest.raises(ValueError, match="gpu_evidence must be a '<file>:<line>' string"):
        _parse(requires_gpu=True, gpu_evidence=12)
    with pytest.raises(ValueError, match="requires_gpu and cpu_params are both set"):
        _parse(requires_gpu=True, gpu_evidence="a.py:1", cpu_params={"device": "cpu"})
    with pytest.raises(ValueError, match="requires_gpu is true but gpu_evidence is empty"):
        _parse(requires_gpu=True)
    with pytest.raises(ValueError, match="is set but requires_gpu is false"):
        _parse(gpu_evidence="a.py:1")
    with pytest.raises(ValueError, match="gpu_evidence must be '<file>:<line>'"):
        _parse(requires_gpu=True, gpu_evidence="a.py")
    with pytest.raises(ValueError, match="gpu_evidence must be '<file>:<line>'"):
        _parse(requires_gpu=True, gpu_evidence="a.py:twelve")


def test_registry_rejects_cpu_params_the_script_does_not_accept():
    with pytest.raises(ValueError, match=r"'X' cpu_params names \['use_cuda'\], which the "
                                         r"vertical:rna\+adt variant does not accept"):
        _parse(cpu_params={"use_cuda": ""})


def test_validate_gpu_fields_is_the_schema_entry_point():
    schema.validate_gpu_fields("Y", {}, False, "")
    with pytest.raises(ValueError, match="^Y: "):
        schema.validate_gpu_fields("Y", {}, True, "")


# ------------------------------------------------------------ runner: cpu_params

def _cite_inputs(tmp_path):
    d = _cite(tmp_path)
    return {"rna": str(d / "rna.h5"), "adt": str(d / "adt.h5")}


def test_cpu_params_for_merges_only_without_gpu(no_gpu, monkeypatch):
    spec = registry.get("scMDC")
    merged, applied = runner.cpu_params_for(spec, None)
    assert merged == {"device": "cpu"} and applied == {"device": "cpu"}
    merged, applied = runner.cpu_params_for(spec, {"nbatch": 2})
    assert merged == {"device": "cpu", "nbatch": 2} and applied == {"device": "cpu"}
    # the caller's key wins, whatever the host - and the object is returned as is
    given = {"device": "cuda:1"}
    merged, applied = runner.cpu_params_for(spec, given)
    assert merged is given and applied == {}
    # a method without cpu_params is untouched
    given = {"epochs": 3}
    assert runner.cpu_params_for(registry.get("SCALEX"), given) == (given, {})
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    assert runner.cpu_params_for(spec, None) == (None, {})


def test_dry_run_shows_cpu_params_without_gpu(tmp_path, no_gpu):
    argv = mtb.run("scMDC", "vertical", inputs=_cite_inputs(tmp_path),
                   out_dir=str(tmp_path / "o"), dry_run=True,
                   cmd_template="conda run -n x {cmd}")
    i = argv.index("--device")
    assert argv[i + 1] == "cpu"
    # scJoint: the empty string, inside the pty wrap the variant opts into
    argv = mtb.run("scJoint", "diagonal",
                   inputs={"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"},
                   out_dir=str(tmp_path / "o"), dry_run=True,
                   cmd_template="conda run -n x {cmd}")
    assert any("--use_cuda ''" in a for a in argv), argv


def test_dry_run_unchanged_on_gpu_host(tmp_path, gpu):
    argv = mtb.run("scMDC", "vertical", inputs=_cite_inputs(tmp_path),
                   out_dir=str(tmp_path / "o"), dry_run=True,
                   cmd_template="conda run -n x {cmd}")
    assert "--device" not in argv
    argv = mtb.run("scJoint", "diagonal",
                   inputs={"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"},
                   out_dir=str(tmp_path / "o"), dry_run=True,
                   cmd_template="conda run -n x {cmd}")
    assert not any("use_cuda" in a for a in argv), argv


def test_explicit_param_wins_over_cpu_params(tmp_path, no_gpu):
    argv = mtb.run("scMDC", "vertical", inputs=_cite_inputs(tmp_path),
                   out_dir=str(tmp_path / "o"), dry_run=True, params={"device": "cuda:1"},
                   cmd_template="conda run -n x {cmd}")
    assert argv[argv.index("--device") + 1] == "cuda:1"
    assert argv.count("--device") == 1 and "cpu" not in argv


def test_real_run_merges_and_logs_one_stderr_line(tmp_path, no_gpu, monkeypatch, capsys):
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen)
    res = mtb.run("scMDC", "vertical", inputs=_cite_inputs(tmp_path),
                  out_dir=str(tmp_path / "o"), convert=False,
                  cmd_template="conda run -n x {cmd}")
    cmd = _FakePopen.calls["cmd"]
    assert cmd[cmd.index("--device") + 1] == "cpu" and res.cmd == cmd
    err = capsys.readouterr().err
    assert err.count("[run] no GPU on this host: applying scMDC cpu_params {'device': 'cpu'}") == 1


def test_real_run_is_silent_on_gpu_host_and_with_explicit_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    mtb.run("scMDC", "vertical", inputs=_cite_inputs(tmp_path),
            out_dir=str(tmp_path / "o1"), convert=False, cmd_template="conda run -n x {cmd}")
    assert "--device" not in _FakePopen.calls["cmd"]
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    mtb.run("scMDC", "vertical", inputs=_cite_inputs(tmp_path / "b"),
            out_dir=str(tmp_path / "o2"), convert=False, params={"device": "cuda:0"},
            cmd_template="conda run -n x {cmd}")
    cmd = _FakePopen.calls["cmd"]
    assert cmd[cmd.index("--device") + 1] == "cuda:0"
    assert "[run] no GPU" not in capsys.readouterr().err


# ------------------------------------------------------------ runner: requires_gpu

MOETM_REASON = ('moETM needs an NVIDIA GPU: the upstream script calls CUDA '
                'unconditionally (tools_scripts/moETM/main_moETM_rna_adt.py:109); '
                'see method_info(m)["requires_gpu"]')


def test_requires_gpu_raises_before_launching(tmp_path, no_gpu, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", _must_not_spawn)
    out = tmp_path / "o"
    with pytest.raises(OSError) as e:
        mtb.run("moETM", "vertical", inputs=_cite_inputs(tmp_path), out_dir=str(out),
                convert=False, cmd_template="conda run -n x {cmd}")
    assert str(e.value) == MOETM_REASON
    assert not out.exists()                      # nothing written
    assert registry.get("moETM").requires_gpu_reason == MOETM_REASON
    with pytest.raises(OSError, match="^moETM needs an NVIDIA GPU"):
        runner.check_gpu_requirement(registry.get("moETM"))
    runner.check_gpu_requirement(registry.get("scMDC"))   # no requirement: silent


def test_requires_gpu_method_runs_on_gpu_host(tmp_path, gpu, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen)
    res = mtb.run("moETM", "vertical", inputs=_cite_inputs(tmp_path),
                  out_dir=str(tmp_path / "o"), convert=False,
                  cmd_template="conda run -n x {cmd}")
    assert res.output.shape == (6, 3)


def test_requires_gpu_dry_run_still_previews(tmp_path, no_gpu):
    argv = mtb.run("moETM", "vertical", inputs=_cite_inputs(tmp_path),
                   out_dir=str(tmp_path / "o"), dry_run=True,
                   cmd_template="conda run -n x {cmd}")
    assert any(a.endswith("main_moETM_rna_adt.py") for a in argv)


def test_run_docstring_documents_the_contract():
    doc = mtb.run.__doc__ or ""
    assert "cpu_params" in doc and "requires_gpu" in doc and "OSError" in doc


# ------------------------------------------------------------ scan / run_all gate

def test_scan_reports_requires_gpu_in_the_env_gate(tmp_path, no_gpu, all_envs):
    _cite(tmp_path)
    df = mtb.scan("CITE", "vertical", methods=["moETM", "scMDC"], data_path=tmp_path,
                  verbose=False)
    df = df[df["modalities"] == "rna+adt"].set_index("method")
    m = df.loc["moETM"]
    assert bool(m["files_ok"]) and not bool(m["env_ok"]) and not bool(m["runnable"])
    assert m["env_reason"] == MOETM_REASON and m["reason"] == MOETM_REASON
    # the preview is still there - and it shows nothing to switch (no cpu_params)
    assert "main_moETM_rna_adt.py" in m["command"]
    s = df.loc["scMDC"]
    assert bool(s["runnable"]) and s["reason"] == ""
    assert "--device cpu" in s["command"]


def test_scan_unchanged_on_gpu_host(tmp_path, gpu, all_envs):
    _cite(tmp_path)
    df = mtb.scan("CITE", "vertical", methods=["moETM", "scMDC"], data_path=tmp_path,
                  verbose=False)
    df = df[df["modalities"] == "rna+adt"].set_index("method")
    assert bool(df.loc["moETM", "runnable"]) and df.loc["moETM", "env_reason"] == ""
    assert "--device" not in df.loc["scMDC", "command"]


def test_scan_joins_missing_env_and_missing_gpu(tmp_path, no_gpu, monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    _cite(tmp_path)
    df = mtb.scan("CITE", "vertical", methods=["moETM"], data_path=tmp_path, verbose=False)
    r = df[df["modalities"] == "rna+adt"].iloc[0]
    assert not bool(r["env_ok"])
    assert r["env_reason"].startswith("conda env ") and "not installed" in r["env_reason"]
    assert r["env_reason"].endswith("; " + MOETM_REASON)
    assert r["reason"] == r["env_reason"]


def test_run_all_never_dispatches_a_requires_gpu_method(tmp_path, no_gpu, all_envs, monkeypatch):
    _cite(tmp_path)
    monkeypatch.setattr(W, "_run", _must_not_spawn)
    plan = mtb.run_all("CITE", "vertical", methods=["moETM"], data_path=tmp_path,
                       out_dir=str(tmp_path / "res"), dry_run=True, verbose=False)
    row = plan[plan["modalities"] == "rna+adt"].iloc[0]
    assert not bool(row["runnable"]) and row["reason"] == MOETM_REASON
    with pytest.raises(ValueError, match="nothing is runnable") as e:
        mtb.run_all("CITE", "vertical", methods=["moETM"], data_path=tmp_path,
                    out_dir=str(tmp_path / "res"), verbose=False)
    assert "needs an NVIDIA GPU" in str(e.value)


def test_scan_docstring_documents_the_gate():
    assert "requires_gpu" in (mtb.scan.__doc__ or "")


def test_runtime_hint_names_its_host():
    """The observed times were taken on the GPU benchmark host; a CPU-only
    user must be told so next to the numbers."""
    import multibench as mtb
    rt = mtb.method_info("scMoMaT")["runtime"]
    assert rt["host"] == "gpu" and "RTX 4090" in rt["note"] and "CPU-only" in rt["note"]
    assert set(rt) >= {"tier", "worst_sec", "observed", "host", "note"}
