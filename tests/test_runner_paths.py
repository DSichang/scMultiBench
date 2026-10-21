"""run() hands the child ABSOLUTE paths and a separator-terminated out_dir (P01)."""
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import multibench as mtb
from multibench.engine import runner


def _popen_honouring_save_path(calls):
    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            calls["cmd"], calls["cwd"] = cmd, cwd
            self.pid, self.returncode = 4242, 0
            # like a real script: write where --save_path says, relative to cwd
            save = cmd[cmd.index("--save_path") + 1]
            target = Path(os.path.join(cwd, save))
            target.mkdir(parents=True, exist_ok=True)
            with h5py.File(target / "embedding.h5", "w") as f:
                f.create_dataset("data", data=np.zeros((6, 3)))

        def communicate(self):
            return "", ""

        def kill(self):
            pass

        def wait(self):
            return self.returncode
    return FakePopen


def test_normalize_paths_absolutizes_and_marks_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "S").mkdir(parents=True)
    obj = object()
    vals, out = runner.normalize_paths(
        {"rna": "data/a.h5", "data_dir": "data/S", "adt": obj, "x": Path("data/S")}, "out/x")
    assert vals["rna"] == str(tmp_path / "data" / "a.h5")
    assert vals["data_dir"] == os.path.join(str(tmp_path / "data" / "S"), "")
    assert vals["x"].endswith(os.sep)               # any existing directory
    assert vals["adt"] is obj                       # in-memory objects untouched
    assert out == os.path.join(str(tmp_path / "out" / "x"), "")


def test_run_keeps_in_memory_anndata_inputs(tmp_path, monkeypatch, linux_with_envs):
    ad = pytest.importorskip("anndata")
    monkeypatch.chdir(tmp_path)
    calls = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _popen_honouring_save_path(calls))
    rna = ad.AnnData(np.ones((6, 4)))
    atac = ad.AnnData(np.ones((6, 5)))
    res = mtb.engine.runner.run('SCALEX', 'diagonal', inputs={'rna': rna, 'atac_gas': atac}, out_dir='out/y')
    cmd = calls["cmd"]
    p1 = cmd[cmd.index("--path1") + 1]
    assert os.path.isabs(p1) and p1 == str(tmp_path / "out" / "y" / "inputs" / "rna.h5")
    assert Path(p1).is_file() and res.output.shape == (6, 3)


def test_run_help_has_no_v1_or_reserved_wording():
    doc = mtb.engine.runner.run.__doc__
    assert "v1" not in doc and "reserved" not in doc
