"""run() hands the child ABSOLUTE paths and a separator-terminated out_dir (P01)."""
import os
from pathlib import Path

import h5py
import numpy as np

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


def test_run_hands_the_child_absolute_input_paths(tmp_path, monkeypatch, linux_with_envs):
    from multibench.engine import ingest
    monkeypatch.chdir(tmp_path)
    calls = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _popen_honouring_save_path(calls))
    bars = [f"c{i}" for i in range(6)]
    for name, n in (("rna.h5", 4), ("atac_gas.h5", 5)):
        ingest._write_canonical(tmp_path / "data" / name, np.ones((6, n)),
                                [f"G{i}" for i in range(n)], bars, dtype="float64",
                                compression=None, block=1024)
    out = mtb.engine.runner.run('SCALEX', 'diagonal', out_dir='out/y',
                                inputs={'rna': 'data/rna.h5', 'atac_gas': 'data/atac_gas.h5'})
    cmd = calls["cmd"]
    p1 = cmd[cmd.index("--path1") + 1]
    assert os.path.isabs(p1) and p1 == str(tmp_path / "data" / "rna.h5")
    assert out.shape == (6, 3)


def test_run_help_has_no_v1_or_reserved_wording():
    doc = mtb.engine.runner.run.__doc__
    assert "v1" not in doc and "reserved" not in doc
