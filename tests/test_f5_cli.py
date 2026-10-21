"""Fix round 5, work package 'cli'.

R5-06: ``load_results`` for a category with no published tables (mosaic)
starts its error with that fact and the fix; the wheel path and
``result_path`` advice stay for a category whose tables should be there.
``multibench plot --input ... --category mosaic`` names ``--source rerun``.
R5-07: ``multibench evaluate --name`` is the row name; ``--method`` then
must be a package method, and it sets the label order and its check.
R5-10: printed messages read as short sentences: no backticks in CLI text,
the prepared-files and peak-name notes start with the method, the scan
reason is sentences with their own subjects, counts without '(s)'.
"""
import ast
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import scripts
from multibench import config
from multibench.engine import envs
from multibench._data import results as R
from tests.test_f4_cli import _diag_folder
from tests.conftest import full_scan

pytest.importorskip("scib")

_PKG = Path(mtb.__file__).resolve().parent


# ====================================================================== R5-10
from multibench import workflow as W                                  # noqa: E402
from multibench.engine import envs, registry, runner                  # noqa: E402
from tests.test_review_f3_int import _diagonal, _quiet, pinned        # noqa: E402,F401

DARWIN = "method environments are linux-64 conda envs; this host is darwin/arm64"


def _strings(node):
    """The literal text of a string argument (a plain or an f-string)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    for sub in ast.walk(node):
        if isinstance(sub, ast.JoinedStr):
            for part in sub.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield part.value


def test_peak_name_note_and_warning_end_with_the_fix(tmp_path, pinned, capsys):
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    d = root / "LUNG_ids"
    inputs = {"rna": str(d / "rna.h5"), "atac_peak": str(d / "atac_peak.h5")}
    note = ("GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds other names, "
            "for example peak_0. Rename them to chr:start-end.")
    with pytest.warns(UserWarning) as rec:
        with pytest.raises(Exception):
            mtb.engine.runner.run("GLUE", "diagonal", inputs=inputs, out_dir=tmp_path / "o2")
    assert note in [str(w.message) for w in rec]


def test_scan_reason_is_sentences_with_their_own_subjects(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    sc = _quiet(full_scan, "LUNG_ids", "diagonal", methods=["GLUE"], data_path=root,
                verbose=False).iloc[0]
    assert sc["reason"] == (
        f"Environment {sc['env']} runs only on Linux, not on this computer. GLUE reads peak "
        f"names such as chr1:100-200. atac_peak.h5 holds other names, for example peak_0. "
        f"Rename them to chr:start-end.")        # 0.4.0: no deprecated override offered
    moetm = _quiet(full_scan, "D11", "vertical", methods=["moETM"], verbose=False).iloc[0]
    # 0.4.0: off Linux the GPU sentence is left out (the OS already blocks)
    assert moetm["reason"] == (
        f"Environment {moetm['env']} runs only on Linux, not on this computer.")
    assert ";" not in moetm["reason"]


def test_count_lines_say_rows_without_plural_brackets(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    df = full_scan("D11", "vertical", modalities="all")
    out = capsys.readouterr().out
    n, k = len(df), int(df["files_ok"].sum())
    assert out == (f"[scan] {k} of {n} rows have their input files. 0 of {n} have their "
                   f"environment installed. mtb.env.install(dry_run=True) checks the "
                   f"environments.\n")
    mtb.scan("D11", "vertical", modalities="all")
    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("[scan] 14 of 18 methods have their input files in D11. ")
    with pytest.raises(ValueError) as e:
        mtb.run("D11", "vertical", out_dir=tmp_path, verbose=False, modalities="all")
    assert f"\nThe first 3 of {n} blocked rows:\n" in str(e.value)
    for text in (out, line, str(e.value)):
        assert "(s)" not in text and "variant" not in text


def test_platform_line_is_two_sentences(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    with pytest.raises(ValueError) as e:
        mtb.run("D11", "vertical", out_dir=tmp_path, verbose=False)
    assert str(e.value).splitlines()[1] == (
        "Methods run only on Linux, and this computer runs macOS. On this computer you "
        "can check files, score embeddings and plot. Run the methods on a Linux machine.")


def test_env_plan_size_lines_name_the_envs():
    # Z27 (r3): one total line; an unmeasured env makes the disk figure a lower bound
    sizes = {"scmb_torch": {"archive_bytes": 9 * 10**8, "unpacked_bytes": 26 * 10**8},
             "scmb_r": {"archive_bytes": 9 * 10**8}}
    _, text = envs.plan_totals([{"env": "scmb_torch"}, {"env": "scmb_r"}], sizes, layers={})
    assert text == "2 envs: 1.8 GB to download, at least 2.6 GB on disk"


def test_fetch_progress_line_is_a_sentence(tmp_path, monkeypatch, capsys):
    import subprocess

    def fake_run(argv, check=True, **kw):
        target = argv[-1] if argv[:2] == ["git", "clone"] else argv[3]
        (tmp_path / "fresh.partial" / "tools_scripts").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    fresh = tmp_path / "fresh"
    try:
        scripts.ensure_repo(fresh)
    except Exception:  # noqa: BLE001 - only the progress line is checked here
        pass
    assert capsys.readouterr().out.startswith(
        f"Fetching the method scripts from PYangLab/scMultiBench into {fresh} ...")


def test_error_tail_is_clipped_at_a_word_boundary():
    long = "RuntimeError: " + " ".join(["could not reach github.com"] * 20)
    tail = W._error_tail(long, width=60)
    assert tail.startswith("...") and len(tail) <= 60
    assert tail[3:].split(" ", 1)[0] in {"could", "not", "reach", "github.com"}


# ---- the Python part of tests that also ran the command line
def test_prepared_note_starts_with_the_method(monkeypatch):
    counts = 'GLUE needs raw counts. rna.h5 holds non-integer values.'
    assert runner._prepared_at('GLUE reads peak names such as chr1:100-200.') == -1
