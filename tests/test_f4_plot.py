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
