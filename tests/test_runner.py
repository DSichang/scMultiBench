from pathlib import Path

import numpy as np
import h5py
from multibench.engine import io as rio
from multibench.engine.schema import OutputSpec


def test_load_embedding_output(tmp_path):
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((10, 5)))
    spec = OutputSpec(kind="embedding", file="embedding.h5", dataset="data")
    arr = rio.load_output(tmp_path, spec)
    assert arr.shape == (10, 5)


def test_load_graph_output(tmp_path):
    p = tmp_path / "knn_indices.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.arange(12).reshape(4, 3))
    spec = OutputSpec(kind="graph", file="knn_indices.h5", dataset="data")
    arr = rio.load_output(tmp_path, spec)
    assert arr.shape == (4, 3)
    assert int(arr[0, 0]) == 0 and int(arr[3, 2]) == 11


def test_load_labels_output(tmp_path):
    p = tmp_path / "Prediction.csv"
    p.write_text("B\nT\nNK\n")
    spec = OutputSpec(kind="labels", file="Prediction.csv")
    labels = rio.load_output(tmp_path, spec)
    assert list(labels) == ["B", "T", "NK"]


def test_wrap_cmd_template():
    from multibench.engine import runner
    wrapped = runner.wrap_command(["python", "x.py", "--a", "1"],
                                  cmd_template="conda run -n scalex {cmd}")
    assert wrapped[:4] == ["conda", "run", "-n", "scalex"]
    assert wrapped[-1] == "1"


def test_run_invokes_subprocess_and_loads_output(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    import numpy as np, h5py

    calls = {}

    class FakePopen:
        # the runner spawns via Popen(start_new_session=True) so a timeout
        # can kill the whole process tree; the fake mirrors that interface
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            calls["cmd"] = cmd
            calls["cwd"] = cwd
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

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)

    res = mtb.run(method="SCALEX", category="diagonal",
                  inputs={"rna": str(tmp_path/"a.h5"), "atac_gas": str(tmp_path/"b.h5")},
                  out_dir=str(tmp_path/"out"), convert=False,
                  cmd_template="conda run -n scalex {cmd}")
    assert "conda" in calls["cmd"][0]
    assert res.output.shape == (6, 3)


def test_run_surfaces_stdout_and_stderr_on_failure(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    import pytest

    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            self.pid = 4242
            self.returncode = 1

        def communicate(self):
            return "boom-out", "boom-err"

        def kill(self):
            pass

        def wait(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)

    with pytest.raises(RuntimeError) as excinfo:
        mtb.run(method="SCALEX", category="diagonal",
                inputs={"rna": str(tmp_path/"a.h5"), "atac_gas": str(tmp_path/"b.h5")},
                out_dir=str(tmp_path/"out"), convert=False,
                cmd_template="conda run -n scalex {cmd}")
    msg = str(excinfo.value)
    assert "boom-err" in msg
    assert "boom-out" in msg


# --- P12: run() preflights the conda env before touching the disk -----------

def _fake_popen_factory(calls):
    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            calls["cmd"] = cmd
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
    return FakePopen


def test_run_preflight_raises_when_env_missing(tmp_path, monkeypatch):
    import pytest
    import multibench as mtb
    from multibench.engine import runner

    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])

    def must_not_spawn(*a, **k):
        raise AssertionError("Popen must not be reached when the env is missing")
    monkeypatch.setattr(runner.subprocess, "Popen", must_not_spawn)

    out = tmp_path / "out"
    with pytest.raises(EnvironmentError) as e:
        mtb.run(method="SCALEX", category="diagonal",
                inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                out_dir=str(out), convert=False)
    msg = str(e.value)
    assert "is not installed" in msg
    assert "multibench env install --methods SCALEX" in msg
    assert "mtb.env.doctor()" in msg
    assert not out.exists(), "preflight must fail before anything is written"


def test_run_preflight_skipped_with_cmd_template(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    calls = {}
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(calls))
    res = mtb.run(method="SCALEX", category="diagonal",
                  inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                  out_dir=str(tmp_path / "out"), convert=False,
                  cmd_template="conda run -n scalex {cmd}")
    assert calls["cmd"][:4] == ["conda", "run", "-n", "scalex"]
    assert res.output.shape == (6, 3)


def test_run_preflight_skipped_when_probe_empty(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    calls = {}
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(calls))
    res = mtb.run(method="SCALEX", category="diagonal",
                  inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                  out_dir=str(tmp_path / "out"), convert=False)
    # default template still targets the method's env; no false block
    assert "run" in calls["cmd"] and runner.envs.group_for("SCALEX") in calls["cmd"]
    assert res.output.shape == (6, 3)
