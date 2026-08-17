"""The LISI helper probe must name the real cause, not hide it.

scib calls its prebuilt binary without check=True and without capturing stderr,
so a binary that cannot start surfaces as a FileNotFoundError on an output file
that was never written - once per metric, cause discarded. These tests pin the
probe's verdicts so that failure mode cannot come back silently.
"""
import stat
import sys
import types
from pathlib import Path

import pytest

from multibench.eval import scib as mscib


def _fake_scib(tmp_path, monkeypatch, script: str | None):
    """Install a fake `scib` package whose knn_graph.o is `script` (or absent)."""
    pkg = tmp_path / "scib"
    (pkg / "knn_graph").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    if script is not None:
        exe = pkg / "knn_graph" / "knn_graph.o"
        exe.write_text(script)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    (pkg / "knn_graph" / "knn_graph.cpp").write_text("// source")
    mod = types.ModuleType("scib")
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "scib", mod)
    mscib._lisi_helper_problem.cache_clear()
    return pkg


HEALTHY = "#!/bin/sh\necho 'usage: knn_graph.o matrixfile, output_prefix, k'\nexit 0\n"
LOADER_FAILURE = ("#!/bin/sh\n"
                  ">&2 echo './knn_graph.o: error while loading shared libraries: "
                  "libstdc++.so.6: version GLIBCXX_3.4.29 not found'\nexit 127\n")
SILENT_CRASH = "#!/bin/sh\nexit 1\n"


def test_healthy_binary_reports_no_problem(tmp_path, monkeypatch):
    _fake_scib(tmp_path, monkeypatch, HEALTHY)
    assert mscib._lisi_helper_problem() is None


def test_loader_failure_is_reported_verbatim(tmp_path, monkeypatch):
    _fake_scib(tmp_path, monkeypatch, LOADER_FAILURE)
    problem = mscib._lisi_helper_problem()
    assert problem and "GLIBCXX_3.4.29 not found" in problem


def test_silent_crash_is_still_a_problem(tmp_path, monkeypatch):
    """Exit 1 with no output is exactly what scib swallows today."""
    _fake_scib(tmp_path, monkeypatch, SILENT_CRASH)
    assert mscib._lisi_helper_problem()


def test_missing_binary_is_named(tmp_path, monkeypatch):
    _fake_scib(tmp_path, monkeypatch, None)
    problem = mscib._lisi_helper_problem()
    assert problem and "missing" in problem


def test_non_executable_binary_is_repaired(tmp_path, monkeypatch):
    pkg = _fake_scib(tmp_path, monkeypatch, HEALTHY)
    exe = pkg / "knn_graph" / "knn_graph.o"
    exe.chmod(0o644)
    mscib._lisi_helper_problem.cache_clear()
    assert mscib._lisi_helper_problem() is None
    assert exe.stat().st_mode & stat.S_IEXEC


@pytest.mark.parametrize("metric", ["cLISI", "iLISI"])
def test_lisi_metrics_are_gated_on_the_probe(metric):
    assert metric in mscib._LISI_METRICS
