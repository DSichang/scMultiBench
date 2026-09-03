"""Directory-fed registration methods get a staged, recorded slice order.

The upstream scripts load ``glob.glob(data_dir + "*.h5ad")`` - directory
order, sorted by nobody - and write ``aligned_slice_<i>.h5ad`` for the i-th
file, while PASTE drops every ``obs`` column at load: nothing in the output
said which input slice ``aligned_slice_3`` came from. ``run`` now stages the
directory as zero-padded symlinks under ``<out_dir>/inputs/`` and writes
``<out_dir>/slices_manifest.json`` in the order the glob returns there. It
also settles the empty ``inputs/`` a data_dir-only method used to get.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import h5py
import pytest

import multibench as mtb
from multibench.engine import runner


def _slices(root, names):
    import anndata as ad
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        a = ad.AnnData(X=np.zeros((4, 3), dtype=np.float32))
        a.obsm["spatial"] = np.zeros((4, 2))
        a.write_h5ad(root / n)
    return root


class _FakePopen:
    """Records argv and what a script's glob would see; never runs anything."""
    calls: dict = {}

    def __init__(self, cmd, cwd, stdout, stderr, text, env=None, start_new_session=False):
        _FakePopen.calls["cmd"] = cmd
        _FakePopen.calls["cwd"] = cwd
        self.pid = 4242
        self.returncode = 0
        if "--data_dir" in cmd:
            dd = cmd[cmd.index("--data_dir") + 1]
            _FakePopen.calls["seen"] = [Path(p).name for p in glob.glob(dd + "*.h5ad")]

    def communicate(self):
        return "", ""

    def kill(self):
        pass

    def wait(self):
        return self.returncode


@pytest.fixture
def fake_popen(monkeypatch):
    _FakePopen.calls = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _FakePopen)
    return _FakePopen.calls


# ----------------------------------------------------------------- stage_slices
def test_stage_slices_links_are_sorted_and_zero_padded(tmp_path):
    src = _slices(tmp_path / "src", ["b.h5ad", "a.h5ad", "slice_10.h5ad", "slice_2.h5ad"])
    (src / "notes.txt").write_text("ignored")
    man = runner.stage_slices(src, tmp_path / "staged")
    links = sorted(p.name for p in (tmp_path / "staged").iterdir())
    assert links == ["00_a.h5ad", "01_b.h5ad", "02_slice_10.h5ad", "03_slice_2.h5ad"]
    for link in (tmp_path / "staged").iterdir():
        assert link.is_symlink()
        assert os.readlink(link) == str(src / link.name.split("_", 1)[1])
    assert man["n_slices"] == 4
    assert man["data_dir"].endswith(os.sep) and man["staged_dir"].endswith(os.sep)


def test_stage_slices_manifest_follows_the_glob_the_script_makes(tmp_path):
    src = _slices(tmp_path / "src", [f"slice_{i}.h5ad" for i in (2, 0, 1)])
    man = runner.stage_slices(src, tmp_path / "staged")
    seen = [Path(p).name for p in glob.glob(man["staged_dir"] + "*.h5ad")]
    assert [s["staged"] for s in man["slices"]] == seen
    assert [s["index"] for s in man["slices"]] == list(range(len(seen)))
    for s in man["slices"]:
        assert s["output"] == f"aligned_slice_{s['index']}.h5ad"
        assert Path(s["source"]).name == s["staged"].split("_", 1)[1]
        assert Path(s["source"]).parent == src
    assert "glob.glob(data_dir + '*.h5ad')" in man["order"]
    if seen != sorted(seen):
        assert "trust this list" in man["order"]


def test_stage_slices_restage_removes_stale_links(tmp_path):
    src = _slices(tmp_path / "src", ["x.h5ad", "y.h5ad", "z.h5ad"])
    runner.stage_slices(src, tmp_path / "staged")
    (src / "z.h5ad").unlink()
    man = runner.stage_slices(src, tmp_path / "staged")
    assert sorted(p.name for p in (tmp_path / "staged").iterdir()) == ["00_x.h5ad", "01_y.h5ad"]
    assert man["n_slices"] == 2


def test_stage_slices_pads_to_the_slice_count(tmp_path):
    src = _slices(tmp_path / "src", [f"s{i:03d}.h5ad" for i in range(101)])
    runner.stage_slices(src, tmp_path / "staged")
    names = sorted(p.name for p in (tmp_path / "staged").iterdir())
    assert names[0] == "000_s000.h5ad" and names[-1] == "100_s100.h5ad"


# ----------------------------------------------------------------- run()
def test_run_stages_data_dir_and_writes_manifest(tmp_path, fake_popen, monkeypatch):
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["paste_envi"])
    src = _slices(tmp_path / "MYVISIUM", [f"slice_{i}.h5ad" for i in (1, 0, 2)])
    out = tmp_path / "out"
    res = mtb.run("PASTE", "cross", inputs={"data_dir": str(src)}, out_dir=str(out))
    cmd = fake_popen["cmd"]
    staged = cmd[cmd.index("--data_dir") + 1]
    assert staged == os.path.join(str(out / "inputs"), "")       # the staged dir, with separator
    assert sorted(p.name for p in (out / "inputs").iterdir()) == \
        ["00_slice_0.h5ad", "01_slice_1.h5ad", "02_slice_2.h5ad"]
    man = json.loads((out / runner.SLICES_MANIFEST).read_text())
    assert [s["staged"] for s in man["slices"]] == fake_popen["seen"]   # what the script loaded
    assert {Path(s["source"]).name for s in man["slices"]} == {"slice_0.h5ad", "slice_1.h5ad", "slice_2.h5ad"}
    assert res.extra[runner.SLICES_MANIFEST] == man
    assert man["data_dir"] == os.path.join(str(src), "")


def test_run_stages_for_every_coords_method(tmp_path, fake_popen, monkeypatch):
    monkeypatch.setattr(runner.envs, "installed_envs",
                        lambda conda=None: ["paste_envi", "scmb_gpsa2"])
    src = _slices(tmp_path / "V", ["a.h5ad", "b.h5ad"])
    for m in ("PASTE2", "GPSA"):
        out = tmp_path / f"out_{m}"
        mtb.run(m, "cross", inputs={"data_dir": str(src)}, out_dir=str(out))
        assert (out / runner.SLICES_MANIFEST).is_file()
        assert (out / "inputs" / "00_a.h5ad").is_symlink()


def test_file_role_method_keeps_its_inputs_dir(tmp_path, monkeypatch):
    """Unchanged for file-role methods: inputs/ is created, no manifest."""
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
    assert not (out / runner.SLICES_MANIFEST).exists()
    assert runner.SLICES_MANIFEST not in res.extra


def test_data_dir_only_non_slice_method_gets_no_empty_inputs_dir(tmp_path, monkeypatch):
    """scBridge takes a data_dir plus const filenames: nothing to stage, so no inputs/."""
    class FakePopen(_FakePopen):
        def __init__(self, cmd, cwd, **kw):
            super().__init__(cmd, cwd, None, None, True)
            self.returncode = 1                  # stop before any output is loaded
    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["scmb_torch"])
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="scBridge failed"):
        mtb.run("scBridge", "diagonal", inputs={"data_dir": str(tmp_path)}, out_dir=str(out))
    assert out.is_dir() and not (out / "inputs").exists()
