"""Fix round 3, work package 'cli': ``run-all --batch`` (R3-09), help texts that
name `multibench config` and the two ``scan --strict`` rules (R3-11), and the
``env plan`` / ``env install`` size total in two short lines (R3-13).

The run-all test runs the real runner in prefix mode against a stand-in env
prefix (a ``bin/python`` stub that writes an embedding), so the path from the
command line to ``run_all(batch=)`` and the saved summary is exercised end to
end without conda or a method.
"""
import argparse
import os
import stat
import sys
import textwrap

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.data import results
from multibench.engine import envs, runner

N = 120


def _sub(*names):
    """The argparse sub-parser at ``names`` (e.g. ``'env', 'plan'``)."""
    parser = cli.build_parser()
    for name in names:
        action = next(a for a in parser._actions
                      if isinstance(a, argparse._SubParsersAction))
        parser = action.choices[name]
    return parser


def _opt_help(parser, flag):
    return next(a.help for a in parser._actions if flag in a.option_strings)


def _rendered_help(parser, flag, columns=80):
    """The lines ``--help`` prints for ``flag`` at a terminal ``columns`` wide."""
    old = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = str(columns)
    try:
        text = parser.format_help()
    finally:
        if old is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = old
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(flag))
    out = [lines[start]]
    for l in lines[start + 1:]:
        if not l.startswith(" " * 10) or l.lstrip().startswith("-"):
            break
        out.append(l)
    return out


# ====================================================================== R3-09
def _h5(path, n_feat, prefix):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(n_feat, N)).astype(float))
        g.create_dataset("features", data=np.array([f"{prefix}{i}" for i in range(n_feat)],
                                                   dtype="S12"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(N)], dtype="S12"))


_WRITER = textwrap.dedent('''\
    import sys
    from pathlib import Path
    import h5py, numpy as np
    out = Path(sys.argv[sys.argv.index("--save_path") + 1])
    out.mkdir(parents=True, exist_ok=True)
    emb = np.random.default_rng(7).normal(size=(%d, 4))
    emb[%d:, 0] += 8.0
    with h5py.File(out / "embedding.h5", "w") as f:
        f.create_dataset("data", data=emb.T)
    (out / "predict.csv").write_text("x\\n" + "A\\n" * %d + "B\\n" * %d)
    ''') % (N, N // 2, N // 2, N // 2)


@pytest.fixture
def standin(tmp_path, monkeypatch):
    """MYCITE (rna/adt/cty, 120 cells) and a stand-in ``matilda`` env prefix.

    The prefix's ``bin/python`` runs a small writer with this interpreter: it
    writes ``embedding.h5`` (two clusters matching the labels) where the
    runner's ``--save_path`` points. conda is hidden; the runner uses the
    prefix.
    """
    data = tmp_path / "data"
    d = data / "MYCITE"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", 30, "g")
    _h5(d / "adt.h5", 6, "p")
    pd.DataFrame({"x": ["A"] * (N // 2) + ["B"] * (N // 2)}).to_csv(d / "cty.csv",
                                                                    index=False)
    envs_dir = tmp_path / "envs"
    writer = tmp_path / "writer.py"
    writer.write_text(_WRITER)
    py = envs_dir / "matilda" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{writer}" "$@"\n')
    py.chmod(py.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setattr(config.DEFAULT, "envs_dir", envs_dir)
    monkeypatch.delenv(config.ENVS_DIR_VAR, raising=False)
    monkeypatch.setattr(envs.shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(config._shutil, "which", lambda *a, **k: None)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setenv(runner.RUN_MODE_VAR, "prefix")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    _clear_env_caches()
    yield data
    _clear_env_caches()


def _clear_env_caches():
    """Drop the per-process env probes, so the stand-in prefix is seen (and forgotten)."""
    for fn in (W._installed_envs, config._conda_envs_dir, envs._conda_prefixes):
        fn.cache_clear()


def _run_all(data, out, *extra):
    return cli.main(["run-all", "MYCITE", "--category", "vertical", "--methods", "Matilda",
                     "--modalities", "rna,adt", "--data-path", str(data),
                     "--out-dir", str(out), "--format", "csv", *extra])


def test_run_all_parser_accepts_batch_and_its_help_is_two_lines():
    args = cli.build_parser().parse_args(
        ["run-all", "D11", "--category", "vertical", "--batch", "b.csv", "--dry-run"])
    assert args.batch == "b.csv" and args.dry_run
    assert cli.build_parser().parse_args(
        ["run-all", "D11", "--category", "vertical", "--dry-run"]).batch is None
    lines = _rendered_help(_sub("run-all"), "--batch")
    # R5-02 added the barcode clause: three lines
    assert len(lines) <= 3, lines
    text = " ".join(" ".join(lines).split())
    # the order the ids follow: the label files', as run_all puts them per method
    assert "evaluate --batch" in text and "cells in the order of the label files" in text
    assert "a first column of barcodes is aligned by barcode" in text
    assert "(default: one batch per label file)" in text


def test_run_all_batch_reaches_the_summary(standin, tmp_path, capsys):
    bp = tmp_path / "batch.csv"
    pd.DataFrame({"x": np.tile(["s1", "s2"], N // 2)}).to_csv(bp, index=False)
    rc = _run_all(standin, tmp_path / "out", "--batch", str(bp))
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    sm = mtb.load_batch(tmp_path / "out").summary
    assert sm.loc[0, "status"] == "CHAIN_OK", sm.loc[0].to_dict()
    assert sm.loc[0, "batch_source"] == "user" and sm.loc[0, "n_batches"] == 2
    assert "ASW_batch" in sm.columns and pd.notna(sm.loc[0, "ASW_batch"])
    # without --batch the same folder is one batch, as before
    rc = _run_all(standin, tmp_path / "out1")
    capsys.readouterr()
    sm = mtb.load_batch(tmp_path / "out1").summary
    assert rc == 0 and sm.loc[0, "n_batches"] == 1 and sm.loc[0, "batch_source"] is None


def test_run_all_batch_missing_file_exits_1_before_any_method(standin, tmp_path, capsys,
                                                             monkeypatch):
    monkeypatch.setattr(W, "_run", lambda *a, **k: pytest.fail("a method was started"))
    missing = tmp_path / "nope.csv"
    for extra in ([], ["--dry-run"]):
        rc = _run_all(standin, tmp_path / "out", "--batch", str(missing), *extra)
        err = capsys.readouterr().err
        assert rc == 1, extra
        assert f"error: --batch file {missing} does not exist" in err
        assert "Traceback" not in err
    assert not (tmp_path / "out").exists()


def test_run_all_dry_run_accepts_batch(standin, tmp_path, capsys):
    bp = tmp_path / "batch.csv"
    pd.DataFrame({"x": np.tile(["s1", "s2"], N // 2)}).to_csv(bp, index=False)
    rc = _run_all(standin, tmp_path / "out", "--batch", str(bp), "--dry-run")
    cap = capsys.readouterr()
    assert rc == 0 and "Matilda" in cap.out
    assert not (tmp_path / "out").exists()
    # the dry run reads the file, so one the real run could not read fails the check
    bad = tmp_path / "wide.csv"
    pd.DataFrame({"a": ["1", "1"], "b": ["s1", "s2"], "c": ["u", "v"]}).to_csv(bad, index=False)
    rc = _run_all(standin, tmp_path / "out", "--batch", str(bad), "--dry-run")
    assert rc == 1 and "error: " in capsys.readouterr().err


# ====================================================================== R3-11
def test_scan_strict_help_names_both_rules():
    text = " ".join(_opt_help(_sub("scan"), "--strict").split())
    assert text.startswith("exit 1 when no requested row is runnable; with --methods, "
                           "when any named method has none")
    assert "nothing requested is runnable" not in text


def test_help_defaults_point_to_multibench_config_not_python():
    for path, flag in ((("scan",), "--data-path"), (("run-all",), "--leiden-flavor"),
                       (("evaluate",), "--leiden-flavor")):
        text = _opt_help(_sub(*path), flag)
        assert "default: see `multibench config`" in text, (path, flag)
        assert "mtb." not in text, (path, flag)


def test_evaluate_description_and_load_results_name_scored_with():
    desc = " ".join(_sub("evaluate").description.split())
    assert "(metric,value,method,dataset,category,clustering,source,scored_with)" in desc
    doc = results.load_results.__doc__
    returns = doc.split("Returns\n")[1].split("Raises\n")[0]
    assert "scored_with" in returns
    assert len(returns.split()) - 3 <= 30        # minus the header rule and the type


# ====================================================================== R3-13
SIZES = {"scmb_r": {"archive_bytes": 916_953_088, "unpacked_bytes": None},
         "scmb_torch": {"archive_bytes": 4_537_398_808, "unpacked_bytes": None},
         "scmb_torch-cpu": {"archive_bytes": 922_099_761, "unpacked_bytes": 2_642_831_784}}
DU = ("# Unpacked envs are larger than the download. Check with du after the first "
      "install.")


def _total_lines(err):
    """The ``# total`` line and the builds / unknown-size lines after it."""
    lines = err.splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("# total"))
    out = [lines[i]]
    for l in lines[i + 1:i + 5]:
        if " build" not in l and "not measured" not in l and "not recorded" not in l \
                and l != DU:
            break
        out.append(l)
    return out


@pytest.fixture
def tables(monkeypatch):
    envs.packed_sizes.cache_clear()
    monkeypatch.setattr(envs, "packed_sizes", lambda: SIZES)
    monkeypatch.setattr(envs, "packed_manifest",
                        lambda: {k: f"https://x/{k}.tar.gz" for k in SIZES})
    monkeypatch.setattr(envs, "installed_flavor", lambda env, conda=None: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


def test_plan_gpu_total_is_short_lines_with_the_du_advice(tables, capsys):
    assert cli.main(["env", "plan", "--methods", "StabMap,scMoMaT", "--flavor", "gpu"]) == 0
    err = capsys.readouterr().err
    assert _total_lines(err) == [
        "# total for 2 envs: 5.5 GB download",
        "# 1 of 2 envs has the GPU build. 1 env has a single build (the same archive for "
        "CPU and GPU hosts).",
        "# The size on disk of scmb_r and scmb_torch is not measured.", DU]
    assert "sizes are those recorded" not in err and "summed the" not in err
    assert "--flavor gpu" not in err                  # asked for GPU builds already
    assert all(";" not in l for l in _total_lines(err))


def test_plan_auto_on_a_cpu_host_states_the_reason_and_the_advice_once(tables, monkeypatch,
                                                                      capsys):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    assert cli.main(["env", "plan", "--methods", "StabMap,scMoMaT"]) == 0
    err = capsys.readouterr().err
    assert _total_lines(err) == [
        "# total for 2 envs: 1.8 GB download",
        "# 1 of 2 envs has the CPU build, because this host has no NVIDIA GPU. 1 env has a "
        "single build (the same archive for CPU and GPU hosts).",
        "# 2.6 GB on disk for scmb_torch. scmb_r is not measured.", DU]
    assert err.count("NVIDIA GPU") == 1 and err.count("--flavor gpu") == 1
    assert "# for jobs on GPU nodes, pass --flavor gpu" in err
    assert all(";" not in l for l in _total_lines(err))


def test_install_dry_run_states_the_no_gpu_fact_once(tables, monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(envs, "create_all", lambda **kw: [
        {"env": "scmb_r", "methods": ["StabMap"], "exists": False, "has_lock": True},
        {"env": "scmb_torch", "methods": ["scMoMaT"], "exists": False, "has_lock": True}])
    assert cli.main(["env", "install", "--packed", "--methods", "StabMap,scMoMaT"]) == 0
    err = capsys.readouterr().err
    assert _total_lines(err) == [
        "# total for 2 envs: 1.8 GB to download",
        "# 1 of 2 envs has the CPU build. 1 env has a single build (the same archive for "
        "CPU and GPU hosts).",
        "# 2.6 GB on disk for scmb_torch. scmb_r is not measured.", DU]
    assert err.count("NVIDIA GPU") == 1 and err.count("--flavor gpu") == 1
    assert all(";" not in l for l in _total_lines(err))


def test_size_total_known_sizes_are_one_line():
    sizes = {"a": {"archive_bytes": 3 * 10**9, "unpacked_bytes": 9 * 10**9}}
    both = {"a": "u", "a-cpu": "u"}                 # a GPU and a CPU build
    assert cli._size_total_line([{"env": "a", "flavor": "gpu"}], sizes, flavor="gpu",
                                manifest=both) == \
        "# total for 1 env: 3.0 GB download, 9.0 GB on disk\n# This env has the GPU build."
    assert cli._size_total_line([{"env": "a"}], sizes) == \
        "# total for 1 env: 3.0 GB download, 9.0 GB on disk"


@pytest.mark.parametrize("n_known", [0, 1, 5])
def test_size_total_with_unknown_disk_gives_the_du_advice(n_known):
    rows = [{"env": e} for e in "abcdef"]
    sizes = {e: {"archive_bytes": 2 * 10**9,
                 "unpacked_bytes": 3 * 10**9 if i < n_known else None}
             for i, e in enumerate("abcdef")}
    text = cli._size_total_line(rows, sizes, flavor="gpu", manifest={})
    lines = text.splitlines()
    assert len(lines) == 4 and lines[0] == "# total for 6 envs: 12.0 GB download"
    assert lines[1] == ("# All 6 envs have a single build (the same archive for CPU and "
                        "GPU hosts).")
    assert lines[3] == DU and all(";" not in l for l in lines)
    assert "not measured." in lines[2]
    assert "on disk" not in lines[0]                 # an incomplete column is not a total
