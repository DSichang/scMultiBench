"""``run`` creates ``<out_dir>/inputs/`` only for a method fed modality files.

A file-role method gets ``inputs/`` for its canonical copies; a ``data_dir``
method (scBridge) takes the dataset folder plus bare filenames, so it gets no
empty ``inputs/``.
"""
from pathlib import Path

import numpy as np
import h5py
import pytest

import multibench as mtb
from multibench.engine import runner


class _FakePopen:
    """Records argv; never runs anything."""
    calls: dict = {}

    def __init__(self, cmd, cwd, stdout, stderr, text, env=None, start_new_session=False):
        _FakePopen.calls["cmd"] = cmd
        _FakePopen.calls["cwd"] = cwd
        self.pid = 4242
        self.returncode = 0

    def communicate(self):
        return "", ""

    def kill(self):
        pass

    def wait(self):
        return self.returncode


def test_file_role_method_keeps_its_inputs_dir(tmp_path, monkeypatch):
    """A file-role method gets inputs/."""
    class FakePopen(_FakePopen):
        def __init__(self, cmd, cwd, **kw):
            super().__init__(cmd, cwd, None, None, True)
            with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
                f.create_dataset("data", data=np.zeros((6, 3)))
    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    out = tmp_path / "out"
    res = mtb.run(method="SCALEX", category="diagonal",
                  inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                  out_dir=str(out), convert=False, cmd_template="conda run -n scalex {cmd}")
    assert (out / "inputs").is_dir()
    assert res.extra == {}


def test_data_dir_method_gets_no_empty_inputs_dir(tmp_path, monkeypatch):
    """scBridge takes a data_dir plus const filenames: nothing to stage, so no inputs/."""
    class FakePopen(_FakePopen):
        def __init__(self, cmd, cwd, **kw):
            super().__init__(cmd, cwd, None, None, True)
            self.returncode = 1                  # stop before any output is loaded
    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["scmb_torch"])
    # scBridge's script calls CUDA unconditionally (registry requires_gpu), so
    # on a GPU-less host run() refuses it before Popen; this test is about the
    # inputs/ layout, so pretend the host has a GPU (tests/test_gpu_requirements.py
    # covers the refusal)
    monkeypatch.setattr(runner.envs, "host_has_gpu", lambda: True)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="scBridge failed"):
        mtb.run("scBridge", "diagonal", inputs={"data_dir": str(tmp_path)}, out_dir=str(out))
    assert out.is_dir() and not (out / "inputs").exists()
