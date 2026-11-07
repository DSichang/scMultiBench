"""A script that writes next to the files it reads (scMM, scMVP, scJoint)
reads links in its run's ``inputs/`` folder: a data folder others share, or
one without write access, is only read, and two runs on one dataset write
apart."""
import os
import shlex
from pathlib import Path

import h5py
import numpy as np
import pytest

from multibench import workflow
from multibench.engine import registry, runner
from multibench.engine.schema import is_label_role

CROSS_RNA_ADT = ["rna1", "rna2", "rna3", "adt1", "adt2", "adt3"]
CROSS_RNA_ATAC = ["rna1", "rna2", "rna3", "atac1", "atac2", "atac3"]
WRITERS = [("scMM", "vertical", ["rna", "adt"]), ("scMM", "vertical", ["rna", "atac"]),
           ("scMM", "cross", CROSS_RNA_ADT), ("scMM", "cross", CROSS_RNA_ATAC),
           ("scMVP", "vertical", ["rna", "atac"]),
           ("scJoint", "diagonal", ["rna", "atac_gas", "rna_cty"]),
           ("scJoint", "diagonal", ["rna1", "rna2", "rna3", "atac_gas1", "atac_gas2",
                                    "atac_gas3", "rna_cty1", "rna_cty2", "rna_cty3"])]


def _h5(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((3, 2)))


def _data(tmp_path, roles) -> dict:
    """``{role: path}``: an ``.h5`` per modality, a ``.csv`` per label file."""
    data = tmp_path / "data"
    data.mkdir()
    inputs = {}
    for role in roles:
        if is_label_role(role):
            path = data / f"{role}.csv"
            path.write_text("x\n")
        else:
            path = data / f"{role}.h5"
            _h5(path)
        inputs[role] = str(path)
    return inputs


def _script_argv(cmd: list) -> list:
    """The method's own argv, out of the pseudo-terminal wrap (scJoint)."""
    return shlex.split(cmd[cmd.index("-c") + 1]) if "script" in cmd else cmd


def test_every_variant_of_these_methods_writes_beside_its_inputs():
    for method in ("scMM", "scMVP", "scJoint"):
        assert all(v.writes_beside_inputs for v in registry.get(method).variants), method


@pytest.mark.parametrize("method,category,roles", WRITERS)
def test_the_script_gets_links_to_the_original_files(tmp_path, monkeypatch, method,
                                                     category, roles):
    seen = {}

    class Popen:
        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd
            self.pid, self.returncode = 4242, 1         # stop before an output is loaded

        def communicate(self):
            return "", ""

    monkeypatch.setattr(runner.subprocess, "Popen", Popen)
    monkeypatch.setattr(runner.envs, "env_prefix", lambda env, conda=None: tmp_path / "envs")
    inputs = _data(tmp_path, roles)
    out = tmp_path / "out" / f"{method}_D1"
    with pytest.raises(RuntimeError, match=f"{method} failed"):
        runner.run(method, category, inputs=inputs, out_dir=str(out))
    given = [a for a in _script_argv(seen["cmd"]) if a.endswith((".h5", ".csv"))]
    assert len(given) == len(roles)
    for path in given:
        assert Path(path).parent == out / "inputs"
        assert os.readlink(path) == inputs[Path(path).stem]
        assert os.path.samefile(path, inputs[Path(path).stem])
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == sorted(
        Path(p).name for p in inputs.values())


def test_the_link_step_creates_and_replaces_links(tmp_path, monkeypatch):
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    first.write_text("first")
    second.write_text("second")
    link = tmp_path / "inputs" / "rna.h5"
    link.parent.mkdir()
    monkeypatch.chdir(tmp_path)
    runner._link_input("first.h5", link)
    assert os.readlink(link) == str(first) and link.read_text() == "first"
    runner._link_input(second, link)                     # an earlier run's link
    assert os.readlink(link) == str(second) and link.read_text() == "second"
    second.unlink()                                      # a link to a moved file
    runner._link_input(first, link)
    assert os.readlink(link) == str(first) and link.read_text() == "first"


def test_scan_shows_the_links_and_says_run_makes_them():
    inputs = {"rna": "/data/D1/rna.h5", "adt": "/data/D1/adt.h5"}
    argv, note = runner.preview("scMM", "vertical", inputs=inputs,
                                out_dir=Path(runner.OUT_DIR_PLACEHOLDER) / "scMM_D1")
    assert "<out_dir>/scMM_D1/inputs/rna.h5" in argv
    assert "<out_dir>/scMM_D1/inputs/adt.h5" in argv
    assert not set(inputs.values()) & set(argv)
    assert note == ("scMM reads inputs/rna.h5, inputs/adt.h5. mtb.run writes those files "
                    "first, so start scMM with mtb.run.")
    assert workflow._run_caveat(f"Another caveat. {note}") == "Another caveat."


def test_another_script_reads_the_files_themselves():
    inputs = {"rna": "/data/D28/rna.h5", "atac_gas": "/data/D28/atac_gas.h5"}
    variant = registry.get("SCALEX").select("diagonal", set(inputs))
    plan = runner._plan_inputs(variant, inputs, Path("/o/inputs"))
    assert {r: s["value"] for r, s in plan.items()} == inputs
    assert runner._prepared_note(plan, "/o", "SCALEX") is None
