"""Integration of fix round 4: two package defects the docs work packages found.

- The ``-> FAIL`` line of ``run`` (R4-03) ended with R's trailing warning
  or call trace instead of the error, because R prints both after the error.
- ``BatchResult.rescore`` scored FAIL and TIMEOUT records again, so a method
  that failed to run came back as ``RUN_OK_EVAL_FAILED``, ``CHAIN_OK`` or
  ``RUN_OK_NO_EMBEDDING`` and left ``failures``.

The run tests use the stand-in env prefixes of ``tests/test_f4_cli.py``.
"""
import os
import textwrap

import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from tests.test_f3_cli import _WRITER
from tests.test_f4_cli import _clear_env_caches, _standin, _status_line

pytest.importorskip("scib")

# What Rscript 4.x prints for a missing package loaded inside a function after a
# warning (checked with Rscript on macOS): the message on its own line, then the
# call trace, then the warning block, then "Execution halted".
_R_STDERR = (
    "Error in library(StabMap) : \n"
    "  there is no package called ‘StabMap’\n"
    "Calls: main -> suppressPackageStartupMessages -> withCallingHandlers -> library\n"
    "In addition: Warning message:\n"
    "package ‘SeuratObject’ was built under R version 4.5.2 \n"
    "Execution halted\n")
_R_CAUSE = "Error in library(StabMap) : there is no package called ‘StabMap’"


def test_error_tail_names_the_r_error_not_its_warnings():
    wrapped = "RuntimeError: StabMap failed (exit 1).\nstdout tail:\n\nstderr tail:\n" + _R_STDERR
    assert W._error_tail(wrapped) == _R_CAUSE
    # several warnings, and an error message on the same line as "Error in"
    many = ("Error in readRDS(f) : cannot open the connection\n"
            "Calls: main -> readRDS\n"
            "In addition: Warning messages:\n"
            "1: package ‘Seurat’ was built under R version 4.5.2\n"
            "2: In gzfile(file, \"rb\") : cannot open compressed file\n"
            "Execution halted\n")
    assert W._error_tail(many) == "Error in readRDS(f) : cannot open the connection"
    # an R error without warnings or call trace keeps its round-4 result
    assert W._error_tail("Error in f() : boom\nExecution halted\n") == "Error in f() : boom"
    # Python tails are unchanged: the last line
    assert W._error_tail("Traceback (most recent call last):\n  File x\n"
                         "ValueError: bad input") == "ValueError: bad input"
    # a warning block with nothing before it is still shown
    assert W._error_tail("In addition: Warning message:\nsomething") == "something"


_R_FAILING = textwrap.dedent('''\
    import sys
    sys.stderr.write(%r)
    sys.exit(1)
    ''') % _R_STDERR


# ---- rescore keeps what did not run ---------------------------------------
_FAIL_AFTER_WRITING = _WRITER + "sys.exit(1)\n"


# ---- restored from before the 0.4 cut: kept behaviour
@pytest.fixture
def r_failing(tmp_path, monkeypatch):
    yield _standin(tmp_path, monkeypatch, _R_FAILING)
    _clear_env_caches()


def _run_py(data, out):
    """The run of the stand-in Matilda on MYCITE, as a job script calls it."""
    return mtb.run(os.path.join(data, "MYCITE"), "vertical", out, methods=["Matilda"],
                   modalities=["rna", "adt"])


def test_run_all_fail_line_ends_with_the_r_error(r_failing, tmp_path, capsys):
    res = _run_py(r_failing, tmp_path / "out")
    out = capsys.readouterr().out
    assert res.failures["method"].tolist() == ["Matilda"], out
    assert _status_line(out).endswith(") " + _R_CAUSE), out


@pytest.fixture
def fail_after_writing(tmp_path, monkeypatch):
    yield _standin(tmp_path, monkeypatch, _FAIL_AFTER_WRITING)
    _clear_env_caches()


@pytest.fixture
def fail_early(tmp_path, monkeypatch):
    from tests.test_f4_cli import _FAILING
    yield _standin(tmp_path, monkeypatch, _FAILING)
    _clear_env_caches()


def _rescored(out):
    res = mtb.load_batch(out)
    assert res.summary.loc[0, "status"] == "failed"
    batch = pd.Series(["b1", "b2"] * 60)
    return res, res.rescore(batch=batch.to_numpy())


def test_rescore_keeps_a_fail_record_without_output(fail_early, tmp_path):
    out = tmp_path / "out"
    assert not _run_py(fail_early, out).failures.empty
    res, new = _rescored(out)
    rec = new.records[0]
    assert rec["status"] == "FAIL"
    assert rec["error"] == res.records[0]["error"]
    assert new.failures["status"].tolist() == ["failed"]


def test_rescore_keeps_a_fail_record_whose_output_exists(fail_after_writing, tmp_path):
    """The method wrote embedding.h5 and then exited 1: still a failed run."""
    out = tmp_path / "out"
    assert not _run_py(fail_after_writing, out).failures.empty
    assert list(out.rglob("embedding.h5"))
    res, new = _rescored(out)
    assert new.records[0]["status"] == "FAIL"
    assert not new.records[0].get("metrics")
    assert new.failures["method"].tolist() == ["Matilda"]
    new.save(tmp_path / "rescored")
    assert pd.read_csv(tmp_path / "rescored" / "failures.csv")["status"].tolist() == ["failed"]

