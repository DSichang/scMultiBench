"""Fix round 4, work package 'plot': figure key and warning texts.

R4-12: the chip key says what '?' and 'DR' mean without the word 'registry';
two own datasets that share no method get a sentence for the same-cells case;
the incomplete-matrix warning is short sentences; the na='raise' error does not
offer to hide itself.
R4-13: recommend's warning lines, DegenerateRerunWarning, evaluate's
reordered-dict error, the prepared-file note, the batch= advice and the
run_all messages are short plain sentences.
"""
import importlib
import inspect
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config

B = importlib.import_module("multibench.plot.bubble")
style = importlib.import_module("multibench.plot.style")

matplotlib.use("Agg")


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def _mine(method, dataset, base=0.5, metrics=("ARI", "NMI", "ASW", "cLISI")):
    return pd.DataFrame([
        {"metric": m, "value": base + 0.01 * i, "method": method, "dataset": dataset,
         "category": "vertical", "source": "user"}
        for i, m in enumerate(metrics)])


# --- R4-12 (a): chip key ------------------------------------------------------

def test_chip_key_names_no_registry_and_spells_out_dr():
    assert "registry" not in B.CHIP_KEY
    assert "? = not a package method, such as your own" in B.CHIP_KEY
    assert "DR = dimension reduction" in B.CHIP_KEY
    # the family name stays: build_table's columns carry it
    assert B.FAMILIES[0][0] == "DR and clustering"
    notes = inspect.getdoc(mtb.plot.bubble)
    assert "registry" not in notes and "not a package" in notes


# --- R4-12 (b): two folders of the same cells --------------------------------

def _two_own_folders():
    # S2: the peak run and the gene-activity run of one Multiome sample
    return pd.concat([_mine("PeakA", "MYMULTIOME"), _mine("PeakB", "MYMULTIOME", 0.6),
                      _mine("GasA", "MYMULTIOME_GA"), _mine("GasB", "MYMULTIOME_GA", 0.4)],
                     ignore_index=True)


@pytest.mark.parametrize("fn,kw", [(mtb.plot.build_table, {}),
                                   (mtb.plot.build_table, {"aggregate": "summary"}),
                                   (mtb.plot.bar, {})])
def test_two_own_datasets_sharing_no_method_mention_the_same_cells(fn, kw):
    _, msgs = _messages(fn, _two_own_folders(), **kw)
    (msg,) = [m for m in msgs if m.startswith("rows come from")]
    assert msg.endswith("Plot each dataset on its own, or score the same methods on "
                        "every dataset. If these datasets hold the same cells, give "
                        "their rows one dataset name first."), msg


def test_stored_rows_keep_the_no_overlap_text():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stored = mtb.load_results("vertical", dataset="D11", source="rerun")
    _, msgs = _messages(mtb.plot.build_table,
                        pd.concat([stored, _mine("RNA PCA", "MYCITE")], ignore_index=True))
    (msg,) = [m for m in msgs if m.startswith("rows come from")]
    assert "same cells" not in msg
    assert msg.endswith("score your method on D11 and add it to that table.")


# --- R4-12 (c): incomplete-matrix warning -------------------------------------

def _incomplete():
    return pd.concat([_mine("A", "D1"), _mine("B", "D1", 0.6), _mine("C", "D1", 0.4),
                      _mine("A", "D2", 0.3), _mine("B", "D2", 0.45)], ignore_index=True)


def test_incomplete_matrix_warning_is_short_sentences():
    _, msgs = _messages(mtb.plot.build_table, _incomplete(), aggregate="summary")
    (msg,) = [m for m in msgs if m.startswith("The summary ranks")]
    assert msg == ("The summary ranks 2 datasets. C has scores on 1 of them. With "
                   "overall='rank', a missing dataset counts as rank 0, the lowest. "
                   "With overall='mean_overall', the missing dataset is left out. "
                   "Pass require_complete=True to keep only the methods scored on "
                   "every dataset.")
    assert ";" not in msg and "under" not in msg
    _, msgs = _messages(mtb.plot.bar, _incomplete())
    (msg,) = [m for m in msgs if m.startswith("The summary ranks")]
    assert ";" not in msg
    assert msg.endswith("Filter long_df to the methods scored on every dataset.")


def test_incomplete_matrix_groups_methods_by_count():
    parts = {"D1": pd.DataFrame(index=["A", "B", "C", "E"]),
             "D2": pd.DataFrame(index=["A", "B"]),
             "D3": pd.DataFrame(index=["A", "C"])}
    (msg,) = style.coverage_warnings(parts, basis="rank", incomplete_fix="Fix.")
    assert ("The summary ranks 3 datasets. E has scores on 1 of them. B and C have "
            "scores on 2 of them.") in msg


def test_incomplete_matrix_cli_spelling(tmp_path, capsys):
    p = tmp_path / "in.csv"
    _incomplete().to_csv(p, index=False)
    assert cli.main(["plot", "bubble", "--input", str(p), "--aggregate", "summary",
                     "--out", str(tmp_path / "f.png")]) == 0
    err = capsys.readouterr().err
    assert "With --overall rank, a missing dataset counts as rank 0, the lowest." in err
    assert "Pass --require-complete to keep only the methods scored on every dataset." in err


# --- R4-12 (d): the n/a error -------------------------------------------------

def _with_gap():
    df = pd.concat([_mine("A", "X"), _mine("B", "X", 0.6), _mine("C", "X", 0.4)],
                   ignore_index=True)
    return df[~((df["method"] == "B") & (df["metric"] == "NMI"))]


def test_na_raise_does_not_offer_to_hide_the_error():
    with pytest.raises(ValueError) as e:
        mtb.plot.build_table(_with_gap(), na="raise")
    msg = str(e.value)
    assert "hide this message" not in msg
    assert msg.endswith("Pass na='warn' to draw the figure with these gaps.")
    # the warning keeps its own last sentence
    _, msgs = _messages(mtb.plot.build_table, _with_gap())
    (warn,) = [m for m in msgs if "has no" in m]
    assert warn.endswith("Pass na='skip' to hide this message.")
    assert warn.rsplit(". ", 1)[0] == msg.rsplit(". ", 1)[0]


def test_na_raise_cli_spelling(tmp_path, capsys):
    p = tmp_path / "in.csv"
    _with_gap().to_csv(p, index=False)
    rc = cli.main(["plot", "bubble", "--input", str(p), "--na", "raise",
                   "--out", str(tmp_path / "f.png")])
    err = capsys.readouterr().err
    assert rc != 0
    assert "Pass --na warn to draw the figure with these gaps." in err
    assert "hide this message" not in err


# --- R4-12 (e): extended summaries --------------------------------------------

def test_plot_extended_summaries_are_plain():
    bar = inspect.getdoc(mtb.plot.bar).split("\n\n")
    assert bar[1].startswith("Parameters")
    assert "summary view" not in inspect.getdoc(mtb.plot.bar)
    bt = inspect.getdoc(mtb.plot.build_table).split("\n\n")
    assert bt[1] == "Returns the numbers ``mtb.plot.bubble`` draws, without drawing."


# --- R4-13 (a): recommend's warning lines -------------------------------------

def _recommend(*a, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = mtb.recommend(*a, **kw)
    msgs = [str(w.message) for w in rec if str(w.message).startswith("recommend(")]
    return r, (msgs[0] if msgs else "")


def _no_log_joins(msg):
    """No clause joined with ' - ', no ';' inside parentheses (a line's own
    bullet marker, as in recommend's list, is not a join)."""
    import re
    for line in msg.splitlines():
        body = re.sub(r"^\s*- ", "", line)
        assert " - " not in body, line
        assert not re.search(r"\([^)]*;[^)]*\)", body), line


def test_recommend_unscored_methods_line_is_plain(result_dir):
    r, msg = _recommend("vertical", result_path=result_dir)
    unscored = r[r.grand_score.isna()].method.tolist()
    assert (f"These {len(unscored)} methods have no published scores and are listed "
            f"last: {', '.join(unscored)}. Try source=\"rerun\".") in msg
    _no_log_joins(msg)
    _, msg = _recommend("vertical", modalities=["rna", "adt"], source="rerun",
                        result_path=result_dir)
    assert "Seurat_WNN has no re-run scores and is listed last." in msg
    assert "Try source" not in msg


def test_recommend_coverage_and_dropped_lines_are_plain(result_dir, layout_tree):
    _, msg = _recommend("diagonal", source="both", result_path=result_dir)
    assert ("Some methods have scores on only part of the datasets: Seurat_v5 3 of 4, "
            "GLUE 3 of 4. Check the coverage column before reading the order.") in msg
    _no_log_joins(msg)
    _, msg = _recommend("cross", result_path=layout_tree)
    assert ("scMoMaT and UINMF have scores only on datasets left out of the ranking, "
            "so they are listed last.") in msg
    assert ("MOFA2 and Multigrate have scores in the published table, but this package "
            "does not run them for cross, so they are left out of the ranking. "
            "mtb.list_methods(category='cross') does not list them.") in msg
    assert "A min-max score over one method is always 1.0." in msg
    _no_log_joins(msg)


# --- R4-13 (c): DegenerateRerunWarning ----------------------------------------

def test_degenerate_rerun_warning_head_is_a_sentence(result_dir):
    with pytest.warns(mtb.DegenerateRerunWarning) as rec:
        mtb.load_results("diagonal", dataset="D28", source="rerun", result_path=result_dir)
    msg = str(rec[0].message)
    assert msg.startswith(
        "This re-run row has ARI below 0.01, while the published table has above 0.2 "
        "for the same method and dataset: Conos/D28 (re-run 0.2.1, ARI 0.0004). Such a "
        "row most likely comes from a failed re-run")
    _no_log_joins(msg)


# --- R4-13 (d): evaluate's reordered-dict error -------------------------------

def test_reordered_label_dict_error_is_short_sentences(tmp_path):
    import numpy as np
    p1, p2 = tmp_path / "cty1.csv", tmp_path / "cty2.csv"
    pd.DataFrame({"x": ["a", "b"] * 5}).to_csv(p1, index=False)
    pd.DataFrame({"x": ["a", "b"] * 5}).to_csv(p2, index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(np.random.default_rng(0).normal(size=(20, 3)),
                     labels={"cty2": str(p2), "cty1": str(p1)}, metrics=["ASW"],
                     verbose=False)
    assert str(e.value) == (
        "labels: the keys ['cty2', 'cty1'] are in neither the method's cell order nor "
        "the default order. Pass the dict from mtb.labels_for(dataset, category, method) "
        "unchanged, a list of paths in cell order, or label_order=[...]. The default "
        "order is cty1, cty2, ... by number, with rna before adt before atac. It is not "
        "alphabetical.")


# --- R4-13 (b): run_all's nothing-runnable error and dry-run count line --------

@pytest.fixture
def no_envs(monkeypatch):
    W = importlib.import_module("multibench.workflow")
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


def test_nothing_runnable_tail_is_plain(no_envs, root):
    for kw, where in (({}, "mtb.scan('D11', 'vertical') shows every row."),
                      ({"methods": ["Matilda"]},
                       "mtb.scan('D11', 'vertical', methods=['Matilda']) shows these rows.")):
        with pytest.raises(ValueError) as e:
            mtb.run_all("D11", "vertical", out_dir="/tmp/unused", verbose=False,
                        data_path=root / "data", **kw)
        tail = str(e.value).splitlines()[-1]
        assert tail == (f"{where} Its files_ok and env_ok columns say which check "
                        f"failed. mtb.env.doctor() checks the environments.")


def test_dry_run_count_line_is_plain(no_envs, root, capsys, tmp_path):
    mtb.run_all("D11", "vertical", out_dir=tmp_path, dry_run=True, verbose=True,
                data_path=root / "data")
    (line,) = [ln for ln in capsys.readouterr().out.splitlines()
               if ln.startswith("[run_all] dry run:")]
    assert " blocked. The table's reason column says why, and its files_ok and env_ok " \
           "columns say which check failed. mtb.env.doctor() checks the environments." in line
    _no_log_joins(line)


# --- R4-13 (e): the prepared-file note ----------------------------------------

def test_prepared_file_note_names_the_method(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan("D28", "diagonal", data_path=root / "data", verbose=False)
    cav = sc.set_index("method").loc["GLUE", "caveat"]
    assert ("the command reads inputs/atac_peak_normpeaks.h5. mtb.run writes that file "
            "first, so start GLUE with mtb.run or mtb.run_all. The printed command alone "
            "fails in a job script.") in cav
    assert "shell line" not in cav


def test_prepared_file_note_cli_spelling(monkeypatch):
    from multibench.engine import runner
    monkeypatch.setattr(config, "_CLI", True)
    plan = {"atac": {"value": "/o/inputs/a.h5", "convert": True, "normpeaks_from": None},
            "rna": {"value": "/o/inputs/b.h5", "convert": True, "normpeaks_from": None}}
    assert runner._prepared_note(plan, "/o", "GLUE") == (
        "the command reads inputs/a.h5, inputs/b.h5. `multibench run` writes those "
        "files first, so start GLUE with `multibench run` or `multibench run-all`. The "
        "printed command alone fails in a job script.")


# --- R4-13 (f): the batch advice names run_all too ----------------------------

def test_batch_advice_names_run_all_and_evaluate(monkeypatch):
    from multibench.engine import resolve
    py = resolve._one_file_advice("vertical", has_adt=True)
    assert py == ("vertical methods read one rna.h5: export without batch= and pass the "
                  "batch column to run_all(batch=...) or evaluate(batch=...), or use "
                  "category='cross' (RNA+ADT)")
    monkeypatch.setattr(config, "_CLI", True)
    assert resolve._one_file_advice("diagonal") == (
        "diagonal methods read one rna.h5 and one ATAC file: convert without --batch "
        "and pass the batch column to run-all --batch or evaluate --batch")


def test_batch_with_atac_warning_names_run_all():
    from multibench.engine import ingest
    with pytest.warns(UserWarning, match=r"pass the batch column to run_all\(batch=\.\.\.\) "
                                         r"or evaluate\(batch=\.\.\.\)"):
        ingest._check_batch_args("obs:batch", None, None, adt=None, atac="X")
    for fn in (mtb.io.export_dataset, mtb.labels_for):
        flat = " ".join(inspect.getdoc(fn).split())
        assert "``mtb.run_all(batch=...)`` or ``mtb.evaluate(batch=...)``" in flat, fn
