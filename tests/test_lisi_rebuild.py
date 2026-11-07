"""scib's LISI helper is rebuilt when it cannot execute (Sam's macOS case) or is missing.

scib ships a Linux x86-64 knn_graph.o; on macOS exec fails with
`Exec format error`, so cLISI/iLISI recorded NaN and the warning told the user
to run g++ by hand. With a compiler on PATH and knn_graph.cpp shipped, the
probe now runs scib's own build line once and proceeds. The build writes a
file of its own and replaces the helper only when it succeeds, so a failed
build, or two processes building at once (a shared environment), never leave
the helper missing. These tests use a fake scib package and a fake compiler
script, never the real one.
"""
import os
import stat
import sys
import types

import pytest

from multibench.eval import scib as mscib


@pytest.fixture(autouse=True)
def _probe_the_real_scib_afterwards():
    """The probe is cached: a fake scib's answer must not outlive its test, or
    a later test scores cLISI and iLISI as NaN."""
    yield
    mscib._lisi_helper_problem.cache_clear()

HEALTHY = "#!/bin/sh\necho 'usage: knn_graph.o matrixfile, output_prefix, k'\nexit 0\n"


def _fake_scib(tmp_path, monkeypatch, *, exe_bytes, cpp=True):
    pkg = tmp_path / "scib"
    (pkg / "knn_graph").mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    exe = pkg / "knn_graph" / "knn_graph.o"
    exe.write_bytes(exe_bytes)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    src = pkg / "knn_graph" / "knn_graph.cpp"
    if cpp:
        src.write_text("int main() { return 0; }\n")
    elif src.exists():
        src.unlink()
    mod = types.ModuleType("scib")
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "scib", mod)
    # import_scib imports scib.metrics: without this the file passes only after
    # another test has imported the real one
    monkeypatch.setitem(sys.modules, "scib.metrics", types.ModuleType("scib.metrics"))
    mscib._lisi_helper_problem.cache_clear()
    return exe


def _build_line(exe):
    """What the fake compiler logs: scib's build line, written to this process's
    own file next to the helper."""
    return f"-std=c++11 -O3 -o {exe.parent / f'.knn_graph.o.{os.getpid()}'} {exe.parent / 'knn_graph.cpp'}"


def _leftovers(exe):
    return [f.name for f in exe.parent.iterdir() if f.name.startswith(".")]


def _fake_compiler(tmp_path, monkeypatch, name="g++", *, succeed=True):
    """A `g++` on PATH that writes a HEALTHY helper to the -o target (or fails)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "cxx.log"
    script = bindir / name
    if succeed:
        script.write_text(
            "#!/bin/sh\n"
            f"echo \"$@\" >> {log}\n"
            "out=''\nwhile [ $# -gt 0 ]; do if [ \"$1\" = '-o' ]; then out=\"$2\"; shift; fi; shift; done\n"
            "printf '%s' '" + HEALTHY.replace("'", "'\\''") + "' > \"$out\"\nchmod +x \"$out\"\nexit 0\n")
    else:
        script.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\n>&2 echo 'error: no such instruction'\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(bindir))
    return log


# an ELF header that the kernel refuses to exec here (Exec format error on
# macOS and on any non-x86-64 Linux): what scib actually ships
FOREIGN_BINARY = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56


def test_foreign_binary_is_rebuilt_once_and_then_healthy(tmp_path, monkeypatch, capsys):
    exe = _fake_scib(tmp_path, monkeypatch, exe_bytes=FOREIGN_BINARY)
    log = _fake_compiler(tmp_path, monkeypatch)
    assert mscib._lisi_helper_problem() is None
    # scib's own build line, once, then the helper replaced
    assert log.read_text().strip().splitlines() == [_build_line(exe)]
    assert exe.read_text() == HEALTHY and _leftovers(exe) == []
    assert capsys.readouterr().err == ""                   # a repair that worked says nothing
    mscib._lisi_helper_problem.cache_clear()
    assert mscib._lisi_helper_problem() is None            # now just runs
    assert len(log.read_text().strip().splitlines()) == 1  # no second compile


def test_clang_is_accepted_when_gxx_is_absent(tmp_path, monkeypatch):
    _fake_scib(tmp_path, monkeypatch, exe_bytes=FOREIGN_BINARY)
    _fake_compiler(tmp_path, monkeypatch, name="clang++")
    assert mscib._find_cxx().endswith("clang++")
    assert mscib._lisi_helper_problem() is None


def test_failed_rebuild_keeps_the_original_problem_and_the_gxx_hint(tmp_path, monkeypatch):
    exe = _fake_scib(tmp_path, monkeypatch, exe_bytes=FOREIGN_BINARY)
    _fake_compiler(tmp_path, monkeypatch, succeed=False)
    problem = mscib._lisi_helper_problem()
    assert problem and "cannot be executed here" in problem
    assert "rebuilding from source failed" in problem and "no such instruction" in problem
    assert exe.read_bytes() == FOREIGN_BINARY               # untouched
    assert _leftovers(exe) == []
    # the metric-level fallback still prints the manual build line;
    # exercise the warning text through the same helper compute() uses
    msg = mscib._lisi_fallback_message("cLISI", problem)
    assert "g++ -std=c++11 -O3 -o" in msg and str(exe) in msg and "Recording NaN" in msg


def test_no_compiler_or_no_source_is_explained(tmp_path, monkeypatch):
    _fake_scib(tmp_path, monkeypatch, exe_bytes=FOREIGN_BINARY)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    problem = mscib._lisi_helper_problem()
    assert "no C++ compiler (g++, c++, clang++) on PATH" in problem
    _fake_scib(tmp_path, monkeypatch, exe_bytes=FOREIGN_BINARY, cpp=False)
    _fake_compiler(tmp_path, monkeypatch)
    assert "knn_graph.cpp is not shipped" in mscib._lisi_helper_problem()


def test_loader_failure_is_rebuilt(tmp_path, monkeypatch):
    """A binary the loader rejects (scib 1.1.7 needs glibc 2.38; Colab's
    Ubuntu 22.04 has 2.35) is rebuilt against this system's libraries."""
    exe = _fake_scib(tmp_path, monkeypatch,
                     exe_bytes=b"#!/bin/sh\n>&2 echo 'GLIBC_2.38 not found'\nexit 127\n")
    log = _fake_compiler(tmp_path, monkeypatch)
    assert mscib._lisi_helper_problem() is None
    assert log.read_text().strip().splitlines() == [_build_line(exe)]


def test_a_missing_helper_is_built(tmp_path, monkeypatch):
    # e.g. removed by a failed build of an older multibench in a shared environment
    exe = _fake_scib(tmp_path, monkeypatch, exe_bytes=FOREIGN_BINARY)
    exe.unlink()
    log = _fake_compiler(tmp_path, monkeypatch)
    assert mscib._lisi_helper_problem() is None
    assert log.read_text().strip().splitlines() == [_build_line(exe)]
    assert exe.read_text() == HEALTHY


def test_a_missing_helper_without_a_compiler_is_named(tmp_path, monkeypatch):
    exe = _fake_scib(tmp_path, monkeypatch, exe_bytes=FOREIGN_BINARY)
    exe.unlink()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    problem = mscib._lisi_helper_problem()
    assert "is missing from the scib installation" in problem
    assert "no C++ compiler" in problem


def test_a_rebuild_that_does_not_help_reports_the_new_problem(tmp_path, monkeypatch):
    _fake_scib(tmp_path, monkeypatch,
                     exe_bytes=b"#!/bin/sh\n>&2 echo 'GLIBCXX_3.4.29 not found'\nexit 127\n")
    log = _fake_compiler(tmp_path, monkeypatch, succeed=False)
    problem = mscib._lisi_helper_problem()
    assert "GLIBCXX_3.4.29 not found" in problem and "rebuilding from source failed" in problem
    assert log.exists()
