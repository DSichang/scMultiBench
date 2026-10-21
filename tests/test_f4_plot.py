"""Fix round 4, work package 'plot': figure key and warning texts.

R4-12: the chip key says what '?' and 'DR' mean without the word 'registry';
two own datasets that share no method get one action (c4);
the incomplete-matrix warning is short sentences; the na='raise' error does not
offer to hide itself.
R4-13: recommend's warning lines, DegenerateRerunWarning, evaluate's
reordered-dict error, the prepared-file note, the batch= advice and the
run messages are short plain sentences.
"""
import os
import importlib
import inspect
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
# the table behind a bubble figure
from multibench.plot.bubble import _build_table  # noqa: E402
from multibench import config
from tests.conftest import full_scan

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


# --- R4-12 (a): the chip wording ----------------------------------------------

def test_chip_note_names_no_registry():
    # the family name stays: build_table's columns carry it
    assert B.FAMILIES[0][0] == "Clustering"
    notes = " ".join(inspect.getdoc(mtb.plot.bubble).split())
    assert "registry" not in notes


# --- R4-12 (b): two folders of the same cells --------------------------------

def _two_own_folders():
    # S2: the peak run and the gene-activity run of one Multiome sample
    return pd.concat([_mine("PeakA", "MYMULTIOME"), _mine("PeakB", "MYMULTIOME", 0.6),
                      _mine("GasA", "MYMULTIOME_GA"), _mine("GasB", "MYMULTIOME_GA", 0.4)],
                     ignore_index=True)


@pytest.mark.parametrize("fn,kw", [(_build_table, {}), (mtb.plot.bar, {})])
def test_two_own_datasets_sharing_no_method_get_one_action(fn, kw):
    # c4 (u3): one action, no same-cells sentence
    _, msgs = _messages(fn, _two_own_folders(), **kw)
    (msg,) = [m for m in msgs if m.startswith("The rows come from")]
    assert msg.endswith("against each other. Plot each dataset on its own."), msg


# --- R4-12 (c): incomplete-matrix warning -------------------------------------

def _incomplete():
    return pd.concat([_mine("A", "D1"), _mine("B", "D1", 0.6), _mine("C", "D1", 0.4),
                      _mine("A", "D2", 0.3), _mine("B", "D2", 0.45)], ignore_index=True)


def test_incomplete_matrix_groups_methods_by_count():
    parts = {"D1": pd.DataFrame(index=["A", "B", "C", "E"]),
             "D2": pd.DataFrame(index=["A", "B"]),
             "D3": pd.DataFrame(index=["A", "C"])}
    (msg,) = style.coverage_warnings(parts, basis="rank", incomplete_fix="Fix.")
    assert ("The summary ranks 3 datasets. E has scores on 1 of them. B and C have "
            "scores on 2 of them.") in msg


# --- R4-12 (d): the n/a error -------------------------------------------------

# --- R4-12 (e): extended summaries --------------------------------------------

# --- R4-13 (a): recommend's warning lines -------------------------------------

def _recommend(*a, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = mtb.recommend(*a, **kw)
    msgs = [str(w.message) for w in rec if str(w.message).startswith("recommend(")
            and "is deprecated" not in str(w.message)]
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
    # c1 (C17): the unscored rows are last, without a line every call
    r, msg = _recommend("vertical")
    assert r.grand_score.isna().any() and r.grand_score.iloc[-1:].isna().all()
    assert "listed last" not in msg
    _, msg = _recommend("vertical", modalities=["rna", "adt"])
    assert "listed last" not in msg


# --- R4-13 (c): DegenerateRerunWarning ----------------------------------------

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
        "The label keys cty2 and cty1 are not in the default order. Pass a list of paths "
        "in cell order, or dataset=, category= and method=. The default "
        "order is cty1, cty2, ... by number, with rna before adt before atac. It is not "
        "alphabetical.")


# --- R4-13 (b): run's nothing-runnable error and dry-run count line --------

@pytest.fixture
def no_envs(monkeypatch):
    W = importlib.import_module("multibench.workflow")
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


def test_nothing_runnable_tail_is_plain(no_envs, root):
    # D11 holds no ATAC file: the rows that read one are not listed, and, since
    # c3 (C55), no line says so
    for kw, where in (
            ({}, "mtb.scan('D11', 'vertical') says why."),
            ({"methods": ["Matilda"]},
             "mtb.scan('D11', 'vertical', methods=['Matilda']) says why.")):
        with pytest.raises(ValueError) as e:
            mtb.run(os.path.join(root / 'data', 'D11'), 'vertical', out_dir='/tmp/unused', verbose=False, **kw)
        lines = str(e.value).splitlines()
        assert lines[-1] == f"{where} mtb.env.install(dry_run=True) checks the environments."
        assert "not listed" not in str(e.value)


def test_scan_count_line_is_plain(no_envs, root, capsys, tmp_path):
    mtb.scan(os.path.join(root / 'data', 'D11'), 'vertical', verbose=True)
    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("[scan] 14 of 14 methods have their input files in D11. ")
    assert "doctor" not in line
    _no_log_joins(line)


# --- R4-13 (e): the prepared-file note ----------------------------------------

def test_prepared_file_note_names_the_method(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = full_scan("D28", "diagonal", data_path=root / "data", verbose=False)
    # GLUE's one-batch row; its several-batch row lacks its files in D28
    cav = sc.set_index(["method", "modalities"]).loc[("GLUE", "rna+atac_peak"), "caveat"]
    assert ("GLUE reads inputs/atac_peak_normpeaks.h5. mtb.run writes that file "
            "first, so start GLUE with mtb.run. The printed command alone "
            "fails in a job script.") in cav
    assert "shell line" not in cav


# --- R4-13 (f): the batch advice names run too ----------------------------

def test_batch_with_atac_warning_names_run_all():
    from multibench.engine import ingest
    with pytest.warns(UserWarning) as rec:
        ingest._check_batch_args("obs:batch", None, None, adt=None, atac="X")
    assert [str(w.message) for w in rec] == [
        "batch= with atac=: only mosaic methods read numbered ATAC files. For mosaic pass "
        "category='mosaic'; otherwise export without batch= and pass the column itself, as "
        'run(batch=adata.obs["donor"])']
