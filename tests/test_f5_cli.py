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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench.data import results as R
from tests.test_f4_cli import _diag_folder

pytest.importorskip("scib")

_PKG = Path(mtb.__file__).resolve().parent


# ====================================================================== R5-06
def test_mosaic_error_starts_with_the_category_and_the_fix():
    with pytest.raises(FileNotFoundError) as e:
        mtb.load_results("mosaic")
    msg = str(e.value)
    assert msg.startswith("mosaic has no published tables. Pass source='rerun' ")
    assert " - " not in msg and "package sweeps" not in msg
    assert "multibench/result/scib_metric" not in msg          # not an install problem


def test_wrong_result_path_keeps_the_path_advice_without_mosaic(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        mtb.load_results("vertical", result_path=tmp_path)
    msg = str(e.value)
    assert msg.startswith(f"no published scIB metric tables under {tmp_path}")
    assert "result_path=" in msg
    assert "mosaic" not in msg and " - " not in msg


def test_the_condition_is_read_from_the_shipped_tree(tmp_path, monkeypatch):
    # a package without a vertical tree: vertical then reads as unpublished
    tree = tmp_path / "result" / config.metric_set_dir("scib")
    (tree / config.category_folder("mosaic")).mkdir(parents=True)
    monkeypatch.setattr(R, "_SHIPPED_BASE", tmp_path / "result")
    assert R._category_unpublished("vertical")
    assert not R._category_unpublished("mosaic")
    with pytest.raises(FileNotFoundError, match=r"^vertical has no published tables\."):
        mtb.load_results("vertical", result_path=tmp_path / "result")


def _mosaic_rows(path):
    pd.DataFrame({"metric": ["ARI", "NMI"], "value": [0.5, 0.6], "method": "Mine",
                  "dataset": "LAB", "category": "mosaic"}).to_csv(path, index=False)
    return path


def test_plot_overlay_on_mosaic_names_source_rerun_first(tmp_path, capsys):
    own = _mosaic_rows(tmp_path / "mine.csv")
    rc = cli.main(["plot", "bar", "--input", str(own), "--category", "mosaic",
                   "--out", str(tmp_path / "o.png")])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith(
        "error: mosaic has no published tables. Pass --source rerun to plot your rows "
        "with the package's re-run tables, or drop --category to plot your rows alone.")
    assert "--result-path" not in err and "scib_metric" not in err


def test_plot_stored_mosaic_without_input_names_source_rerun(tmp_path, capsys):
    rc = cli.main(["plot", "bubble", "--category", "mosaic",
                   "--out", str(tmp_path / "o.png")])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("error: mosaic has no published tables. Pass --source rerun "
                          "to plot the package's re-run tables.")


def test_plot_wrong_result_path_keeps_the_flag_advice(tmp_path, capsys):
    rc = cli.main(["plot", "bar", "--category", "vertical", "--result-path", str(tmp_path),
                   "--out", str(tmp_path / "o.png")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "--result-path" in err and "mosaic" not in err


# ====================================================================== R5-07
@pytest.fixture
def diag(tmp_path, monkeypatch):
    data = tmp_path / "data"
    d, emb = _diag_folder(data)
    monkeypatch.setattr(config.DEFAULT, "data_path", data)
    np.save(tmp_path / "emb.npy", emb)
    return d, tmp_path / "emb.npy"


def _evaluate(emb, *label_files, method="uniPort", name=None, out=None):
    argv = ["evaluate", "--output", str(emb)]
    for f in label_files:
        argv += ["--labels", str(f)]
    argv += ["--method", method, "--dataset", "D28", "--category", "diagonal",
             "--metrics", "ASW"]
    if name is not None:
        argv += ["--name", name]
    if out is not None:
        argv += ["--out", str(out)]
    return cli.main(argv)


def test_name_keeps_the_order_check_of_the_package_method(diag, capsys):
    d, emb = diag
    rc = _evaluate(emb, d / "rna_cty.csv", d / "atac_cty.csv", name="uniPort_rerun")
    err = capsys.readouterr().err
    assert rc == 1
    assert "error: uniPort puts the cells of atac_cty before rna_cty." in err


def test_name_without_labels_reads_the_labels_in_the_method_order(diag, tmp_path, capsys):
    d, emb = diag
    out = tmp_path / "mine.csv"
    assert _evaluate(emb, method="SCALEX", name="SCALEX_rerun", out=out) == 0
    err = capsys.readouterr().err
    assert "# labels: rna_cty.csv, atac_cty.csv from D28, in SCALEX's cell order" in err
    df = pd.read_csv(out)
    assert set(df["method"]) == {"SCALEX_rerun"}


def test_name_needs_a_package_method(diag, capsys):
    d, emb = diag
    rc = _evaluate(emb, d / "rna_cty.csv", d / "atac_cty.csv", method="uniPort_rerun",
                   name="mine")
    err = capsys.readouterr().err
    assert rc == 1
    assert ("error: With --name, --method must be a package method. uniPort_rerun is "
            "not one. Did you mean uniPort? multibench list shows the methods.") in err


def test_name_needs_method(diag, capsys):
    d, emb = diag
    with pytest.raises(SystemExit) as e:
        cli.main(["evaluate", "--output", str(emb), "--labels", str(d / "rna_cty.csv"),
                  "--name", "x", "--dataset", "D28", "--category", "diagonal"])
    assert e.value.code == 2
    assert "missing --method" in capsys.readouterr().err


def test_unknown_method_with_labels_is_still_only_a_row_name(diag, tmp_path, capsys):
    d, emb = diag
    out = tmp_path / "mine.csv"
    assert _evaluate(emb, d / "rna_cty.csv", d / "atac_cty.csv", method="SCALEX_rerun",
                     out=out) == 0
    capsys.readouterr()
    assert set(pd.read_csv(out)["method"]) == {"SCALEX_rerun"}


def test_evaluate_help_names_both_flags():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    pe = sub.choices["evaluate"]
    helps = {a.option_strings[0]: a.help for a in pe._actions if a.option_strings}
    assert helps["--method"].startswith(
        "package method whose label order is used, or with --labels any row name. It "
        "is also the row name unless --name is given")
    assert helps["--name"].startswith("row name in the long table, e.g. SCALEX_rerun")


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


def test_no_cli_variant_of_config_hint_has_a_backtick():
    found = []
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "hint" and any("`" in t for t in _strings(node.args[1])):
                found.append(f"{path.relative_to(_PKG)}:{node.lineno}")
    assert found == []


def test_no_help_text_of_any_parser_has_a_backtick():
    """The --help of every command and subcommand, as a shell prints it."""
    from tests.test_content_scope import _walk_parsers
    bad = [(name, line.strip()) for name, p in _walk_parsers(cli.build_parser())
           for line in p.format_help().splitlines() if "`" in line]
    assert bad == []


def test_no_runtime_string_of_cli_py_has_a_backtick():
    """Docstrings and argparse help texts are reference text; everything else
    in cli.py is printed at run time."""
    tree = ast.parse((_PKG / "cli.py").read_text())
    skip = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            skip.add(id(body[0].value))
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("help", "description", "epilog"):
                    skip.update(id(n) for n in ast.walk(kw.value))
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.endswith(("_HELP", "_EPILOG"))
                for t in node.targets):
            skip.update(id(n) for n in ast.walk(node.value))
    bad = [(n.lineno, n.value[:60]) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str) and "`" in n.value
           and id(n) not in skip]
    assert bad == []


def _plan(*files):
    return {f"r{i}": {"value": f"/o/{f}", "convert": True, "normpeaks_from": None}
            for i, f in enumerate(files)}


def test_prepared_note_starts_with_the_method(monkeypatch):
    py = runner._prepared_note(_plan("inputs/atac_peak_normpeaks.h5"), "/o", "GLUE")
    assert py == ("GLUE reads inputs/atac_peak_normpeaks.h5. mtb.run writes that file first, "
                  "so start GLUE with mtb.run or mtb.run_all. The printed command alone "
                  "fails in a job script.")
    monkeypatch.setattr(config, "_CLI", True)
    cli_note = runner._prepared_note(_plan("inputs/atac_peak_normpeaks.h5"), "/o", "GLUE")
    assert cli_note == ("GLUE reads inputs/atac_peak_normpeaks.h5. multibench run writes "
                        "that file first, so start GLUE with multibench run or multibench "
                        "run-all. The printed command alone fails in a job script.")
    # the matchers find it at the start of a caveat or after another sentence
    counts = "GLUE needs raw counts. rna.h5 holds non-integer values."
    assert runner._prepared_at(cli_note) == 0
    assert runner._prepared_at(counts + " " + cli_note) == len(counts + " ")
    assert runner._prepared_at("GLUE reads peak names such as chr1:100-200.") == -1
    assert W._run_caveat(counts + " " + cli_note) == counts


def test_peak_name_note_and_warning_end_with_the_fix(tmp_path, pinned, capsys):
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    d = root / "LUNG_ids"
    inputs = {"rna": str(d / "rna.h5"), "atac_peak": str(d / "atac_peak.h5")}
    note = ("GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds other names, "
            "for example peak_0. Rename them to chr:start-end.")
    _quiet(mtb.run, "GLUE", "diagonal", inputs=inputs, out_dir=tmp_path / "o", dry_run=True)
    assert f"# {note}\n" in capsys.readouterr().err
    with pytest.warns(UserWarning) as rec:
        with pytest.raises(Exception):
            mtb.run("GLUE", "diagonal", inputs=inputs, out_dir=tmp_path / "o2")
    assert note in [str(w.message) for w in rec]


def test_scan_reason_is_sentences_with_their_own_subjects(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    sc = _quiet(mtb.scan, "LUNG_ids", "diagonal", methods=["GLUE"], data_path=root,
                verbose=False).iloc[0]
    assert sc["reason"] == (
        f"Environment {sc['env']} runs only on Linux, not on this computer. GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds other "
        f"names, for example peak_0. Rename them to chr:start-end, or pass "
        f"allow_atac_mismatch=True to run GLUE anyway.")
    moetm = _quiet(mtb.scan, "D11", "vertical", methods=["moETM"], verbose=False).iloc[0]
    assert moetm["reason"] == (
        f"Environment {moetm['env']} runs only on Linux, not on this computer. moETM needs an NVIDIA GPU, and this computer has none. See "
        f"method_info(\"moETM\")[\"requires_gpu\"].")
    assert ";" not in moetm["reason"]


def test_scan_env_reason_on_linux_names_the_command_without_backticks(monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    rc = cli.main(["scan", "D11", "--category", "vertical", "--methods", "Matilda",
                   "--modalities", "rna,adt", "--format", "json"])
    import json
    row = json.loads(capsys.readouterr().out)[0]
    assert rc == 0
    assert row["reason"] == (
        "conda env 'matilda' is not installed. Run multibench env install --methods "
        "Matilda --packed --run (or --category vertical). See multibench env doctor.")


def test_count_lines_say_rows_without_plural_brackets(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    df = mtb.scan("D11", "vertical")
    out = capsys.readouterr().out
    n, k = len(df), int(df["files_ok"].sum())
    assert out == (f"[scan] {k} of {n} rows have their input files. 0 of {n} have their "
                   f"environment installed.\n")
    mtb.run_all("D11", "vertical", out_dir=tmp_path, dry_run=True)
    line = next(l for l in capsys.readouterr().out.splitlines() if "Dry run" in l)
    assert line.startswith(f"[run_all] Dry run: 0 of {n} requested rows can run on D11 "
                           f"(vertical). {n} are blocked. The table's reason column")
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir=tmp_path, verbose=False)
    assert f"\nThe first 3 of {n} blocked rows:\n" in str(e.value)
    for text in (out, line, str(e.value)):
        assert "(s)" not in text and "variant" not in text


def test_cli_dry_run_headers_are_sentences(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--methods", "Matilda",
                   "--modalities", "rna,adt", "--dry-run"])
    cap = capsys.readouterr()
    assert rc == 0
    assert cap.err.startswith(
        "# Dry run. Nothing was executed. 0 of 1 method can run on D11 (vertical). The "
        "commands below are what multibench run would execute. Rows with files_ok False "
        "have none.\n")
    assert ("# Commands of the 1 row whose input files are in place. [env missing] marks "
            "a row whose environment is not installed. [use multibench run] marks a "
            "command that reads a file multibench run writes first.") in cap.out
    for text in (cap.out, cap.err):
        assert "`" not in text and "(s)" not in text


def test_platform_line_is_two_sentences(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir=tmp_path, verbose=False)
    assert str(e.value).splitlines()[1] == (
        "Methods run only on Linux (this computer is darwin/arm64). On this computer you "
        "can check files, score embeddings and plot. Run the methods on a Linux machine.")


def test_strict_counts_are_labelled(monkeypatch, capsys):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    rc = cli.main(["scan", "D45", "--category", "mosaic", "--methods", "SMILE,Cobolt",
                   "--strict"])
    first = capsys.readouterr().err.splitlines()[0]
    assert rc == 1
    assert first.startswith("error: --strict: 0 of 2 rows are runnable. Rows whose "
                            "environment is not ready: 2. Rows that need a GPU this host "
                            "lacks: 1.")


def test_env_plan_size_lines_name_the_envs():
    sizes = {"scmb_torch": {"archive_bytes": 9 * 10**8, "unpacked_bytes": 26 * 10**8},
             "scmb_r": {"archive_bytes": 9 * 10**8}}
    text = cli._size_total_line([{"env": "scmb_torch"}, {"env": "scmb_r"}], sizes)
    assert text.splitlines()[1:] == [
        "# 2.6 GB on disk for scmb_torch. scmb_r is not measured.",
        "# Unpacked envs are larger than the download. Check with du after the first "
        "install."]
    gpu = cli._size_total_line([{"env": "a", "flavor": "gpu"}],
                               {"a": {"archive_bytes": 1, "unpacked_bytes": 1}},
                               flavor="gpu", manifest={"a": "u", "a-cpu": "u"})
    assert gpu.splitlines()[1] == "# This env has the GPU build."


def test_fetch_progress_line_is_a_sentence(tmp_path, monkeypatch, capsys):
    import subprocess

    def fake_run(argv, check=True, **kw):
        target = argv[-1] if argv[:2] == ["git", "clone"] else argv[3]
        (tmp_path / "fresh.partial" / "tools_scripts").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    fresh = tmp_path / "fresh"
    try:
        config.ensure_repo(fresh)
    except Exception:  # noqa: BLE001 - only the progress line is checked here
        pass
    assert capsys.readouterr().out.startswith(
        f"Fetching the method scripts from PYangLab/scMultiBench into {fresh} ...")


def test_error_tail_is_clipped_at_a_word_boundary():
    long = "RuntimeError: " + " ".join(["could not reach github.com"] * 20)
    tail = W._error_tail(long, width=60)
    assert tail.startswith("...") and len(tail) <= 60
    assert tail[3:].split(" ", 1)[0] in {"could", "not", "reach", "github.com"}
