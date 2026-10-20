"""Fix round 4, work package 'cli'.

R4-03: ``run-all`` exits 3 when a finished run lists a method in failures,
and the ``-> FAIL`` / ``-> TIMEOUT`` progress line ends with the error.
R4-05: ``evaluate`` refuses ``--labels`` files typed in an order that
contradicts the method named by ``--method``; the count error speaks of the
row order of ``--embedding``, not batches.
R4-07: the ``run-all`` help names the files it saves and the config default.
R4-08: an env with one archive is "a single build" in ``env plan`` /
``env install`` and the install warning; the total is short lines with no
semicolon inside parentheses.

The run-all tests run the real runner in prefix mode against stand-in env
prefixes (a ``bin/python`` stub), as ``test_f3_cli`` does.
"""
import os
import re
import stat
import sys
import textwrap
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, runner
from tests.test_f3_cli import _clear_env_caches, _h5, _rendered_help, _WRITER, N
from _serve import serve_archives

pytest.importorskip("scib")


# ====================================================================== R4-03
_FAILING = textwrap.dedent('''\
    import sys
    print("loading the inputs")
    sys.stderr.write("Traceback (most recent call last):\\n  File x\\n")
    sys.stderr.write("ValueError: stand-in method failed on purpose\\n")
    sys.exit(1)
    ''')

_SLOW = "import time\ntime.sleep(20)\n"


def _standin(tmp_path, monkeypatch, script):
    """MYCITE (rna/adt/cty, 120 cells) and a ``matilda`` prefix whose python runs ``script``."""
    data = tmp_path / "data"
    d = data / "MYCITE"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", 30, "g")
    _h5(d / "adt.h5", 6, "p")
    pd.DataFrame({"x": ["A"] * (N // 2) + ["B"] * (N // 2)}).to_csv(d / "cty.csv",
                                                                    index=False)
    envs_dir = tmp_path / "envs"
    py_file = tmp_path / "method.py"
    py_file.write_text(script)
    py = envs_dir / "matilda" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{py_file}" "$@"\n')
    py.chmod(py.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (envs_dir / "matilda" / envs.FLAVOR_FILE).write_text("build\n")   # made by multibench
    monkeypatch.setattr(config.DEFAULT, "envs_dir", envs_dir)
    monkeypatch.delenv(config.ENVS_DIR_VAR, raising=False)
    monkeypatch.setattr(envs.shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(config._shutil, "which", lambda *a, **k: None)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setenv(runner.RUN_MODE_VAR, "prefix")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    _clear_env_caches()
    return data


@pytest.fixture
def failing(tmp_path, monkeypatch):
    yield _standin(tmp_path, monkeypatch, _FAILING)
    _clear_env_caches()


@pytest.fixture
def working(tmp_path, monkeypatch):
    yield _standin(tmp_path, monkeypatch, _WRITER)
    _clear_env_caches()


def _status_line(err):
    return next(l for l in err.splitlines() if l.startswith("[run_all]   -> "))


def test_python_run_all_status_line_ends_with_the_error(failing, tmp_path, capsys):
    res = mtb.run_all(str(failing / "MYCITE"), "vertical", out_dir=tmp_path / "py",
                      methods=["Matilda"], modalities=["rna", "adt"])
    out = capsys.readouterr().out
    assert len(res.failures) == 1                       # the Python call does not raise
    assert _status_line(out).endswith(") ValueError: stand-in method failed on purpose")


def test_error_tail_is_the_last_line_that_says_something():
    r_error = ("RuntimeError: StabMap failed (exit 1).\nstdout tail:\n\nstderr tail:\n"
               "Error in library(StabMap) : there is no package called 'StabMap'\n"
               "Execution halted\n")
    assert W._error_tail(r_error) == ("Error in library(StabMap) : there is no package "
                                      "called 'StabMap'")
    long = "OSError: " + "x" * 400
    tail = W._error_tail("first\n" + long + "\n\n")
    assert len(tail) == 200 and tail.startswith("...") and tail.endswith("x" * 50)
    assert W._error_tail("TimeoutError: over 1 s") == "TimeoutError: over 1 s"
    assert W._error_tail(None) == "" and W._error_tail("\n \n") == ""


# ====================================================================== R4-05
def _diag_folder(root, name="D28", n=40):
    """A diagonal folder: RNA cells (types r0/r1), then ATAC cells (a0/a1)."""
    d = root / name
    d.mkdir(parents=True)
    rna = np.array(["t0"] * (n // 2) + ["t1"] * (n // 2))
    atac = np.array(["t2"] * (n // 2) + ["t3"] * (n // 2))
    pd.DataFrame({"x": rna}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": atac}).to_csv(d / "atac_cty.csv", index=False)
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(8 * i, 1, size=(n // 2, 4)) for i in range(4)])
    return d, emb


def test_install_warning_and_unpack_line_for_a_single_build(tmp_path, monkeypatch):
    manifest = {"scmb_r": "https://x/scmb_r.tar.gz",
                "scmb_torch": "https://x/scmb_torch.tar.gz",
                "scmb_torch-cpu": "https://x/scmb_torch-cpu.tar.gz"}
    sizes = {"scmb_r": {"archive_bytes": 900_000_000}}
    msg = envs._cpu_fallback_warning("scmb_r", manifest, sizes)
    assert msg == ("scmb_r has a single build (the same archive for CPU and GPU hosts); "
                   "installing it (0.9 GB)")
    assert "GPU build" not in msg
    # an env whose CPU archive is listed but not uploaded keeps the two-build words
    assert envs._cpu_fallback_warning("scmb_torch", manifest, {}) == (
        "no CPU archive for scmb_torch; installing the GPU build (?) - the CPU archive "
        "scmb_torch-cpu is not published yet")
    assert envs._single_build("scmb_r", manifest=manifest)
    assert not envs._single_build("scmb_torch", manifest=manifest)


def test_single_build_install_warns_in_its_own_words_and_unpacks_as_single(
        tmp_path, monkeypatch, capsys):
    import urllib.request

    from tests.test_prefix_mode import _tiny_archive
    manifest = {"scmb_r": "https://x/scmb_r.tar.gz"}
    monkeypatch.setattr(envs, "packed_layers", lambda: {})
    monkeypatch.setattr(envs, "packed_manifest", lambda: dict(manifest))
    monkeypatch.setattr(envs, "packed_sizes",
                        lambda: {"scmb_r": {"archive_bytes": 900_000_000}})
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    tgz = _tiny_archive(tmp_path / "a.tar.gz")
    serve_archives(monkeypatch, lambda url: (str(tgz), None))
    with pytest.warns(UserWarning) as rec:
        assert envs.install_packed("scmb_r", envs_dir=tmp_path / "envs") is True
    assert [str(r.message) for r in rec] == [
        "scmb_r has a single build (the same archive for CPU and GPU hosts); "
        "installing it (0.9 GB)"]
    out = capsys.readouterr().out
    assert "[env] downloading scmb_r (single build) 900 MB ..." in out
    assert "gpu build" not in out.lower()


