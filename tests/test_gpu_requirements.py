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
method with ``OSError`` before launching; ``scan`` / ``run`` report the
same sentence in the env gate. On a GPU host nothing changes. Every test
patches ``host_has_gpu`` both ways, so it holds on ANY machine.
"""
import os

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry, runner
from tests.conftest import full_scan

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())

#: the classification snapshot - a registry change must change this test
GPU_ONLY = {"scBridge", "moETM", "SMILE", "sciCAN", "UnitedNet", "iPOLNG"}
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


def _must_not_spawn(*a, **k):
    raise AssertionError("Popen must not be reached")


# ------------------------------------------------------------ the snapshot

def test_cpu_params_keys_are_accepted_params_of_every_variant():
    """The runner emits cpu_params below the params check. Since r3 (X64) they are
    the package's switch, not tunable; the scripts declare them
    (tests/test_params_api.py reads the scripts)."""
    for m, cp in CPU_SWITCH.items():
        for v in registry.get(m).variants:
            assert not set(cp) & set(v.tunable), (m, v.when)
    assert "use_cuda" not in mtb.params_for("scJoint", "diagonal", ["rna", "atac_gas"])["tunable"]
    assert "device" not in mtb.params_for("scMDC", "vertical", ["rna", "adt"])["tunable"]
    # the cross variant runs the SAME script as the vertical ones: it inherits
    # their tunables (params.yaml has no entry for it)
    cross = registry.get("scMDC").select("cross", {"rna1", "rna2", "rna3", "adt1", "adt2", "adt3"})
    assert cross.tunable == registry.get("scMDC").select("vertical", {"rna", "adt"}).tunable
    # scMDC has two cross variants (RNA + ADT, RNA + ATAC), both on that script
    for second in ("adt", "atac"):
        mods = [f"{r}{i}" for r in ("rna", second) for i in (1, 2, 3)]
        assert "sigma1" in mtb.params_for("scMDC", "cross", mods)["tunable"]


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
    # Z17 (r3): no lookup sentence after the fact
    assert spec.requires_gpu_reason == 'X needs an NVIDIA GPU, and this computer has none.'
    # absent keys -> the defaults
    spec = _parse()
    assert spec.cpu_params == {} and spec.requires_gpu is False and spec.gpu_evidence == ""


def _gpu_field_problems(d: dict) -> list:
    """What is wrong with the GPU keys of one methods.yaml entry."""
    cpu, req = d.get("cpu_params") or {}, d.get("requires_gpu", False)
    ev, use = d.get("gpu_evidence", "") or "", d.get("gpu_use", "") or ""
    bad = []
    if not isinstance(cpu, dict):
        return ["cpu_params is not a mapping"]
    for k, v in cpu.items():
        if not isinstance(k, str) or not k:
            bad.append(f"cpu_params key {k!r}")
        # None/False are never emitted, so they cannot turn CUDA off
        if v is None or v is False or not isinstance(v, (str, int, float, bool)):
            bad.append(f"cpu_params[{k!r}] = {v!r}")
    if not isinstance(req, bool):
        bad.append("requires_gpu is not a bool")
    if not isinstance(ev, str):
        return bad + ["gpu_evidence is not a string"]
    if req and cpu:
        bad.append("requires_gpu and cpu_params both set")
    if bool(req) != bool(ev.strip()):
        bad.append("gpu_evidence must be set exactly when requires_gpu")
    if req and not all(e.strip().rpartition(":")[0] and e.strip().rpartition(":")[2].isdigit()
                       for e in ev.split(",")):
        bad.append(f"gpu_evidence {ev!r} is not '<file>:<line>'")
    if use not in ("", "used when present", "not used"):
        bad.append(f"gpu_use {use!r}")
    if use and req:
        bad.append("gpu_use set with requires_gpu")
    if cpu and use not in ("", "used when present"):
        bad.append(f"gpu_use {use!r} contradicts cpu_params")
    return bad


def test_methods_yaml_gpu_keys_are_consistent():
    data = yaml.safe_load(registry._YAML.read_text())
    bad = {m["id"]: _gpu_field_problems(m) for m in data["methods"]}
    assert {k: v for k, v in bad.items() if v} == {}


def test_the_gpu_key_check_catches_bad_shapes():
    for d in ({"cpu_params": ["--device", "cpu"]}, {"cpu_params": {"": "cpu"}},
              {"cpu_params": {"device": None}}, {"cpu_params": {"device": False}},
              {"requires_gpu": "yes", "gpu_evidence": "a.py:1"},
              {"requires_gpu": True, "gpu_evidence": 12},
              {"requires_gpu": True, "gpu_evidence": "a.py:1", "cpu_params": {"device": "cpu"}},
              {"requires_gpu": True}, {"gpu_evidence": "a.py:1"},
              {"requires_gpu": True, "gpu_evidence": "a.py"},
              {"requires_gpu": True, "gpu_evidence": "a.py:twelve"},
              {"gpu_use": "sometimes"}, {"gpu_use": "not used", "cpu_params": {"device": "cpu"}}):
        assert _gpu_field_problems(d), d
    assert _gpu_field_problems({"cpu_params": {"device": "cpu"}}) == []


def test_registry_rejects_cpu_params_the_script_does_not_accept():
    # r3: the CUDA switch left tunable, so the check reads the scripts themselves
    from tests.test_params_api import test_the_keys_left_out_are_script_arguments
    test_the_keys_left_out_are_script_arguments()


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
    argv = mtb.engine.runner.preview("scMDC", "vertical", inputs=_cite_inputs(tmp_path), out_dir=str(tmp_path / "o"))[0]
    i = argv.index("--device")
    assert argv[i + 1] == "cpu"
    # scJoint: the empty string, inside the pty wrap the variant opts into
    argv = mtb.engine.runner.preview("scJoint", "diagonal", inputs={"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}, out_dir=str(tmp_path / "o"))[0]
    assert any("--use_cuda ''" in a for a in argv), argv


def test_dry_run_unchanged_on_gpu_host(tmp_path, gpu):
    argv = mtb.engine.runner.preview("scMDC", "vertical", inputs=_cite_inputs(tmp_path), out_dir=str(tmp_path / "o"))[0]
    assert "--device" not in argv
    argv = mtb.engine.runner.preview("scJoint", "diagonal", inputs={"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}, out_dir=str(tmp_path / "o"))[0]
    assert not any("use_cuda" in a for a in argv), argv


def test_explicit_param_wins_over_cpu_params(tmp_path, no_gpu):
    # r3 (X64): the CUDA switch is the package's, so params= cannot set it
    with pytest.raises(KeyError, match=r"scMDC does not accept \['device'\]"):
        mtb.engine.runner.preview("scMDC", "vertical", inputs=_cite_inputs(tmp_path),
                                  out_dir=str(tmp_path / "o"), params={"device": "cuda:1"})


# ------------------------------------------------------------ runner: requires_gpu

MOETM_REASON = 'moETM needs an NVIDIA GPU, and this computer has none.'


def test_requires_gpu_dry_run_still_previews(tmp_path, no_gpu):
    argv = mtb.engine.runner.preview("moETM", "vertical", inputs=_cite_inputs(tmp_path), out_dir=str(tmp_path / "o"))[0]
    assert any(a.endswith("main_moETM_rna_adt.py") for a in argv)


def test_run_docstring_documents_the_contract():
    # the method runner is internal; mtb.run calls it
    assert "mtb.run" in mtb.engine.runner.run.__doc__
    doc = mtb.engine.runner.run.__doc__ or ""
    assert "OSError" in doc and "needs a GPU this host lacks" in " ".join(doc.split())


# ------------------------------------------------------------ scan / run gate

def test_scan_reports_requires_gpu_in_the_env_gate(tmp_path, no_gpu, all_envs):
    _cite(tmp_path)
    df = full_scan("CITE", "vertical", methods=["moETM", "scMDC"], data_path=tmp_path,
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
    df = full_scan("CITE", "vertical", methods=["moETM", "scMDC"], data_path=tmp_path,
                  verbose=False)
    df = df[df["modalities"] == "rna+adt"].set_index("method")
    assert bool(df.loc["moETM", "runnable"]) and df.loc["moETM", "env_reason"] == ""
    assert "--device" not in df.loc["scMDC", "command"]


def test_scan_joins_missing_env_and_missing_gpu(tmp_path, no_gpu, monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    _cite(tmp_path)
    df = full_scan("CITE", "vertical", methods=["moETM"], data_path=tmp_path, verbose=False)
    r = df[df["modalities"] == "rna+adt"].iloc[0]
    assert not bool(r["env_ok"])
    assert r["env_reason"].startswith("Environment ") and "not installed" in r["env_reason"]
    assert r["env_reason"].endswith(". " + MOETM_REASON)
    assert r["reason"] == r["env_reason"]


def test_run_all_never_dispatches_a_requires_gpu_method(tmp_path, no_gpu, all_envs, monkeypatch):
    _cite(tmp_path)
    monkeypatch.setattr(W, "_run", _must_not_spawn)
    plan = mtb.scan(os.path.join(tmp_path, 'CITE'), 'vertical', methods=['moETM'], verbose=False)
    row = plan[plan["modalities"] == "rna+adt"].iloc[0]
    assert not bool(row["runnable"]) and row["reason"] == MOETM_REASON
    with pytest.raises(ValueError, match=r"^None of the requested methods \(moETM\) can "
                                         r"run on CITE \(vertical\)\.") as e:
        mtb.run(os.path.join(tmp_path, 'CITE'), 'vertical', methods=['moETM'], out_dir=str(tmp_path / 'res'), verbose=False)
    assert "needs an NVIDIA GPU" in str(e.value)


def test_scan_docstring_documents_the_gate():
    assert "skip this host's GPU test" in " ".join((mtb.scan.__doc__ or "").split())


