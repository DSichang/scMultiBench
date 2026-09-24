"""Fix round 4, work package 'cli'.

R4-03: ``run-all`` exits 3 when a finished run lists a method in failures,
and the ``-> FAIL`` / ``-> TIMEOUT`` progress line ends with the error.
R4-05: ``evaluate`` refuses ``--labels`` files typed in an order that
contradicts the method named by ``--method``; the count error speaks of the
row order of ``--output``, not batches.
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
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, runner
from tests.test_f3_cli import _clear_env_caches, _h5, _rendered_help, _sub, _WRITER, N

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


@pytest.fixture
def slow(tmp_path, monkeypatch):
    yield _standin(tmp_path, monkeypatch, _SLOW)
    _clear_env_caches()


def _run_all(data, out, *extra):
    return cli.main(["run-all", "MYCITE", "--category", "vertical", "--methods", "Matilda",
                     "--modalities", "rna,adt", "--data-path", str(data),
                     "--out-dir", str(out), "--format", "csv", *extra])


def _status_line(err):
    return next(l for l in err.splitlines() if l.startswith("[run_all]   -> "))


def test_run_all_exits_3_when_a_method_fails_and_says_why(failing, tmp_path, capsys):
    out = tmp_path / "out"
    rc = _run_all(failing, out)
    err = capsys.readouterr().err
    assert rc == 3, err
    fails = pd.read_csv(out / "failures.csv")
    assert fails["method"].tolist() == ["Matilda"] and fails["status"].tolist() == ["FAIL"]
    assert (out / "summary.csv").exists() and (out / "batch_result.json").exists()
    # the status line ends with the last line of the error, not after the status
    line = _status_line(err)
    assert re.fullmatch(r"\[run_all\]   -> FAIL \([0-9.]+s\) ValueError: stand-in method "
                        r"failed on purpose", line), line
    assert f"# 1 of 1 method failed: Matilda (FAIL). See {out / 'failures.csv'}." in err
    assert err.rstrip().endswith("See " + str(out / "failures.csv") + ".")


def test_run_all_exits_0_when_every_method_is_chain_ok(working, tmp_path, capsys):
    rc = _run_all(working, tmp_path / "out")
    err = capsys.readouterr().err
    assert rc == 0, err
    assert mtb.load_batch(tmp_path / "out").summary.loc[0, "status"] == "CHAIN_OK"
    assert "failed" not in err
    # a CHAIN_OK line still ends with the ARI
    assert re.fullmatch(r"\[run_all\]   -> CHAIN_OK \([0-9.]+s\) [0-9.]+", _status_line(err))


def test_run_all_timeout_exits_3_and_the_line_names_the_limit(slow, tmp_path, capsys):
    rc = _run_all(slow, tmp_path / "out", "--timeout", "1")
    err = capsys.readouterr().err
    assert rc == 3, err
    line = _status_line(err)
    assert line.startswith("[run_all]   -> TIMEOUT (") and "1" in line.split(")", 1)[1]
    assert "# 1 of 1 method failed: Matilda (TIMEOUT)." in err


def test_python_run_all_status_line_ends_with_the_error(failing, tmp_path, capsys):
    res = mtb.run_all("MYCITE", "vertical", out_dir=tmp_path / "py", methods=["Matilda"],
                      modalities=["rna", "adt"], data_path=failing)
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


def test_exit_code_3_is_documented_in_both_helps(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    top = " ".join(capsys.readouterr().out.split())
    # worded by this run: failures.csv holds the merged folder (review of wp/f4_int)
    assert ("3 run-all finished but a method of this run failed or was skipped (a line "
            "on stderr names it)") in top
    assert "``3`` ``run-all``" in cli.__doc__
    with pytest.raises(SystemExit):
        cli.main(["run-all", "--help"])
    text = capsys.readouterr().out
    lines = [l for l in text.splitlines() if "exit code 3" in l.lower()]
    assert len(lines) == 1, text


def test_dry_run_still_exits_0_with_a_blocked_row(working, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(W, "_run", lambda *a, **k: pytest.fail("a method was started"))
    assert _run_all(working, tmp_path / "out", "--dry-run") == 0
    capsys.readouterr()


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


@pytest.fixture
def diag(tmp_path, monkeypatch):
    data = tmp_path / "data"
    d, emb = _diag_folder(data)
    monkeypatch.setattr(config.DEFAULT, "data_path", data)
    np.save(tmp_path / "emb.npy", emb)
    return d, tmp_path / "emb.npy"


def _evaluate(emb, *label_files, method="uniPort", dataset="D28", extra=()):
    argv = ["evaluate", "--output", str(emb)]
    for f in label_files:
        argv += ["--labels", str(f)]
    if method is not None:
        argv += ["--method", method, "--dataset", dataset, "--category", "diagonal"]
    return cli.main(argv + ["--metrics", "ASW", *extra])


def test_labels_in_the_order_the_method_contradicts_exit_1(diag, capsys):
    d, emb = diag
    rc = _evaluate(emb, d / "rna_cty.csv", d / "atac_cty.csv")
    err = capsys.readouterr().err
    assert rc == 1
    assert ("error: uniPort puts the cells of atac_cty before rna_cty. Repeat --labels "
            "in that order, or drop --labels to read them in that order.") in err
    # the order uniPort names is accepted
    assert _evaluate(emb, d / "atac_cty.csv", d / "rna_cty.csv") == 0
    capsys.readouterr()


def test_labels_in_the_method_order_pass_and_unknown_methods_are_not_checked(diag, capsys):
    d, emb = diag
    assert _evaluate(emb, d / "rna_cty.csv", d / "atac_cty.csv", method="SCALEX") == 0
    # a name the package does not know (a stored re-run) is not checked
    assert _evaluate(emb, d / "atac_cty.csv", d / "rna_cty.csv", method="SCALEX_rerun") == 0
    # without --method there is nothing to check against
    assert _evaluate(emb, d / "atac_cty.csv", d / "rna_cty.csv", method=None) == 0
    capsys.readouterr()


def test_label_files_with_other_names_are_not_checked(diag, tmp_path, capsys):
    d, emb = diag
    own = tmp_path / "own"
    own.mkdir()
    (own / "rna_types.csv").write_text((d / "rna_cty.csv").read_text())
    (own / "atac_types.csv").write_text((d / "atac_cty.csv").read_text())
    assert _evaluate(emb, own / "rna_types.csv", own / "atac_types.csv") == 0
    capsys.readouterr()


def test_folder_outside_the_data_path_is_checked_by_file_names(tmp_path, monkeypatch,
                                                                  capsys):
    monkeypatch.setattr(config.DEFAULT, "data_path", tmp_path / "empty")
    d, emb = _diag_folder(tmp_path / "elsewhere", name="MYDIAG")
    np.save(tmp_path / "emb.npy", emb)
    rc = _evaluate(tmp_path / "emb.npy", d / "rna_cty.csv", d / "atac_cty.csv",
                   dataset="MYDIAG")
    err = capsys.readouterr().err
    assert rc == 1
    assert ("error: uniPort puts the cells of atac_cty before rna_cty. Repeat --labels "
            "in that order.") in err
    assert "drop --labels" not in err          # the folder is not under the data path


def test_count_error_speaks_of_the_row_order_of_output(tmp_path, capsys):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "emb.npy", rng.normal(size=(90, 4)))
    folder = tmp_path / "LABMOS"
    folder.mkdir()
    for i in range(3):
        pd.DataFrame({"x": ["a"] * 15 + ["b"] * 15}).to_csv(folder / f"cty{i + 1}.csv",
                                                           index=False)
    assert cli.main(["evaluate", "--output", str(tmp_path / "emb.npy"),
                     "--labels", str(folder / "cty1.csv"), "--metrics", "ASW"]) == 1
    err = capsys.readouterr().err
    assert "batch order" not in err
    assert ("For several label files (cty1.csv, cty2.csv, cty3.csv), repeat --labels in "
            "the row order of --output, or pass --dataset, --category and --method.") in err


# ====================================================================== R4-07
def test_run_all_description_names_the_saved_files_and_the_plot_command():
    desc = " ".join(_sub("run-all").description.split())
    assert ("evaluate each output and save summary.csv, long.csv, failures.csv and "
            "batch_result.json under --out-dir. multibench plot bubble --input OUT draws "
            "the figure.") in desc
    assert "and figure under" not in desc


def test_run_all_data_path_and_dry_run_help():
    p = _sub("run-all")
    data = " ".join(next(a.help for a in p._actions if "--data-path" in a.option_strings)
                    .split())
    assert data.endswith("(default: see `multibench config`)"), data
    dry = " ".join(next(a.help for a in p._actions if "--dry-run" in a.option_strings)
                   .split())
    # the ledger's wording ('as multibench scan prints it') wraps to three lines
    assert dry == ("print the plan (one row per method variant, as in multibench scan) "
                   "and the commands; nothing runs")
    for flag in ("--data-path", "--dry-run"):
        lines = _rendered_help(p, flag)
        if "  " not in lines[0].strip():      # the flag alone; the help starts below
            lines = lines[1:]
        assert len(lines) <= 2, (flag, lines)
    text = p.format_help()
    assert "mtb." not in text.split("--dry-run", 1)[1].split("--assume-gpu", 1)[0]
    assert "package data path" not in text


# ====================================================================== R4-08
@pytest.fixture
def linux_cpu(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(envs, "installed_flavor", lambda env, conda=None: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


def _parens_with_semicolon(text):
    return [m for m in re.findall(r"\(([^()]*)\)", text) if ";" in m]


def test_stabmap_cpu_plan_calls_its_env_a_single_build(linux_cpu, capsys):
    assert cli.main(["env", "plan", "--methods", "StabMap", "--flavor", "cpu"]) == 0
    err = capsys.readouterr().err
    assert "GPU build" not in err
    lines = err.splitlines()
    assert lines[0] == "# total for 1 env: 0.9 GB download"
    assert lines[1] == ("# This env has a single build (the same archive for CPU and "
                        "GPU hosts).")
    assert not _parens_with_semicolon(err)


def test_mosaic_auto_plan_has_short_lines_without_semicolons_in_parentheses(linux_cpu,
                                                                           capsys):
    assert cli.main(["env", "plan", "--category", "mosaic"]) == 0
    err = capsys.readouterr().err
    assert not _parens_with_semicolon(err), err
    lines = err.splitlines()
    assert lines[0] == "# total for 6 envs: 15.6 GB download"
    assert lines[1] == ("# CPU builds, because this host has no NVIDIA GPU. 4 envs have a "
                        "single build (the same archive for CPU and GPU hosts).")
    assert lines[2].startswith("# size on disk not recorded for 4 of 6 envs")
    assert "GPU build" not in err and "only a GPU" not in err


def test_gpu_plan_says_gpu_build_only_for_envs_that_have_both(linux_cpu, capsys):
    assert cli.main(["env", "plan", "--methods", "StabMap,scMoMaT", "--flavor", "gpu"]) == 0
    err = capsys.readouterr().err
    assert err.splitlines()[:2] == [
        "# total for 2 envs: 5.5 GB download",
        "# GPU build. 1 env has a single build (the same archive for CPU and GPU hosts)."]


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


def test_flavor_token_is_left_out_for_a_single_build(monkeypatch):
    assert cli._flavor_token("gpu", "scmb_r") == ""
    assert cli._flavor_token("gpu", "scmb_torch") == " flavor=gpu"
    assert cli._flavor_token("cpu", "scmb_torch") == " flavor=cpu"
    assert cli._flavor_token(None, "scmb_torch") == ""


def test_single_build_install_warns_in_its_own_words_and_unpacks_as_single(
        tmp_path, monkeypatch, capsys):
    import urllib.request

    from tests.test_prefix_mode import _tiny_archive
    manifest = {"scmb_r": "https://x/scmb_r.tar.gz"}
    monkeypatch.setattr(envs, "packed_manifest", lambda: dict(manifest))
    monkeypatch.setattr(envs, "packed_sizes",
                        lambda: {"scmb_r": {"archive_bytes": 900_000_000}})
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    tgz = _tiny_archive(tmp_path / "a.tar.gz")
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url: (str(tgz), None))
    with pytest.warns(UserWarning) as rec:
        assert envs.install_packed("scmb_r", envs_dir=tmp_path / "envs") is True
    assert [str(r.message) for r in rec] == [
        "scmb_r has a single build (the same archive for CPU and GPU hosts); "
        "installing it (0.9 GB)"]
    out = capsys.readouterr().out
    assert "[env] unpacking prebuilt scmb_r (single build) -> " in out
    assert "gpu build" not in out.lower()


def test_env_status_shows_no_flavour_for_a_single_build(monkeypatch, capsys):
    rows = [{"method": "StabMap", "env": "scmb_r", "exists": True, "has_lock": True,
             "difficulty": "easy", "verified_working": True, "flavor": "gpu"},
            {"method": "scMoMaT", "env": "scmb_torch", "exists": True, "has_lock": True,
             "difficulty": "easy", "verified_working": True, "flavor": "cpu"}]
    monkeypatch.setattr(envs, "status", lambda: [dict(r) for r in rows])
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    assert cli.main(["env", "status", "--methods", "StabMap,scMoMaT"]) == 0
    out = {l.split()[1]: l for l in capsys.readouterr().out.splitlines()}
    assert "flavor=" not in out["StabMap"]
    assert out["scMoMaT"].endswith(" flavor=cpu")
