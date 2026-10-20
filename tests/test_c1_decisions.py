"""Core round c1 (0.4.0) owner decisions: anchored ranks, one metrics default, the
summary's five statuses and own clusters."""
import importlib
import json
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow
from multibench.engine import registry, resolve
from multibench.eval import io as eio, pipeline, scib as escib
from multibench.plot import style
from multibench.plot.bubble import _build_table

bar_module = importlib.import_module("multibench.plot.bar")


# --------------------------------------------------------------- anchored ranks
_LAB = {"vertical": "D11", "diagonal": "D28", "mosaic": "D45", "cross": "D52"}


def _added(stored):
    """Rows of three methods not in ``stored``: a copy of its best method, a copy of
    its worst and random values, on its datasets."""
    rng = np.random.default_rng(3)
    order = list(mtb.recommend(stored["category"].iloc[0], long_df=stored).method)
    best = stored[stored.method == order[0]].assign(method="Best copy", source="user")
    worst = stored[stored.method == order[-1]].assign(method="Worst copy", source="user")
    rand = stored[stored.method == order[1]].assign(
        method="Mine", source="user", value=lambda d: rng.random(len(d)))
    out = pd.concat([best, worst, rand], ignore_index=True)
    out.attrs["copied"] = {"Best copy": order[0], "Worst copy": order[-1]}
    return out


def _bar_order(df, metrics):
    import matplotlib.pyplot as plt
    fig = bar_module._bar(df, metrics=metrics)
    names = [t.get_text() for t in fig.axes[0].get_yticklabels()][::-1]
    plt.close(fig)
    return names


@pytest.mark.parametrize("category", list(_LAB))
@pytest.mark.parametrize("family", [None, "clustering", "batch"])
def test_added_rows_keep_the_stored_order(category, family):
    if family == "batch" and category == "vertical":
        pytest.skip("the vertical table has one batch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stored = mtb.load_results(category)
        added = _added(stored)
        both = pd.concat([stored, added], ignore_index=True)
        names = set(stored["method"])
        orders = {
            "recommend": (list(mtb.recommend(category, metrics=family, long_df=stored).method),
                          list(mtb.recommend(category, metrics=family, long_df=both).method)),
            "bubble": (_build_table(stored, metrics=family).methods,
                       _build_table(both, metrics=family).methods),
            "bar": (_bar_order(stored, family), _bar_order(both, family)),
        }
    for name, (alone, mixed) in orders.items():
        assert [m for m in mixed if m in names] == alone, name
    # the three agree on where the added rows go
    assert orders["recommend"][1] == orders["bubble"][1] == orders["bar"][1]
    # a copy of a stored method lands next to it
    mixed = orders["bubble"][1]
    if family is None:
        for copy, of in added.attrs["copied"].items():
            assert abs(mixed.index(copy) - mixed.index(of)) <= 2, (copy, mixed)


def test_an_added_row_is_placed_by_its_rank_against_the_stored_rows_only():
    stored = mtb.load_results("vertical")
    first = list(mtb.recommend("vertical", long_df=stored).method)[0]
    top = stored[stored.method == first].assign(method="Top", source="user",
                                                value=lambda d: d.value + 1)
    other = stored[stored.method == first].assign(method="Other", source="user",
                                                  value=lambda d: d.value + 2)
    # Top beats every stored method; with Other next to it, it still comes first
    # among the stored methods, and Other does not push it below them
    mixed = _build_table(pd.concat([stored, top, other]), metrics="clustering").methods
    assert set(mixed[:2]) == {"Other", "Top"} and mixed[2] == first


# --------------------------------------------------------------- C04 and C17
@pytest.mark.parametrize("category", list(_LAB))
def test_recommend_and_the_figures_share_one_default(category):
    stored = mtb.load_results(category)
    rec = [m for m in mtb.recommend(category).method if m in set(stored.method)]
    assert rec == _build_table(stored).methods


def test_recommend_says_nothing_about_methods_without_scores():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mtb.recommend("vertical")
    assert out["grand_score"].isna().any()             # still listed, last


def test_recommend_with_long_df_ranks_its_methods_only():
    stored = mtb.load_results("vertical")
    out = mtb.recommend("vertical", long_df=stored)
    assert set(out.method) == set(stored.method)


# --------------------------------------------------------------- C38 summary
SUMMARY = ["method", "status", "ARI", "NMI", "run_sec", "caveat", "reason"]


@pytest.mark.parametrize("code, status", [
    ("CHAIN_OK", "scored"), ("CHAIN_OK_GRAPH_METHOD", "scored"), ("RUN_OK", "ran"),
    ("RUN_OK_EVAL_FAILED", "ran"), ("RUN_OK_NO_EMBEDDING", "ran"),
    ("RUN_OK_NO_LABEL_MATCH", "ran"), ("FAIL", "failed"), ("SKIPPED", "skipped"),
    ("TIMEOUT", "timeout"), ("scored", "scored")])
def test_the_nine_record_codes_map_onto_five_statuses(code, status):
    assert workflow._status(code) == status


def test_the_summary_columns_and_reasons():
    recs = [{"method": "A", "status": "CHAIN_OK", "run_sec": 1.0,
             "metrics": {"ARI": 0.5, "NMI": 0.6}},
            {"method": "B", "status": "RUN_OK_EVAL_FAILED", "error": "ValueError: x"},
            {"method": "C", "status": "RUN_OK_NO_EMBEDDING"},
            {"method": "D", "status": "SKIPPED", "error": "needs labels", "requested": True}]
    s = mtb.BatchResult(recs, "D11", "vertical").summary
    assert list(s.columns) == SUMMARY
    assert s.status.tolist() == ["scored", "ran", "ran", "skipped"]
    assert s.reason.tolist() == ["", "ValueError: x", "It writes a graph, not an embedding.",
                                 "needs labels"]
    f = mtb.BatchResult(recs, "D11", "vertical").failures
    assert f.status.tolist() == ["ran", "skipped"]
    assert repr(mtb.BatchResult(recs, "D11", "vertical")) == \
        "<BatchResult vertical/D11: 1 scored, 2 ran, 1 skipped>"


def test_an_empty_summary_has_the_columns():
    assert list(mtb.BatchResult([], "D11", "vertical").summary.columns) == \
        ["method", "status", "run_sec", "caveat", "reason"]


def _tree(tmp_path, dims=6):
    """A 0.4.0 run_all folder of D11: one record with the old CHAIN_OK code and a
    random embedding."""
    out = tmp_path / "out"
    (out / "Matilda_D11").mkdir(parents=True)
    n = len(pd.read_csv(Path(mtb.config.DEFAULT.data_path) / "D11" / "cty.csv"))
    with h5py.File(out / "Matilda_D11" / "embedding.h5", "w") as f:
        f["data"] = np.random.default_rng(0).normal(size=(n, dims))
    rec = {"method": "Matilda", "category": "vertical", "dataset": "D11",
           "modalities": ["rna", "adt"], "output_kind": "embedding", "status": "CHAIN_OK",
           "out_dir": str(out / "Matilda_D11"), "run_sec": 3.0,
           "metrics": {"ARI": 0.9, "NMI": 0.8, "cLISI": 0.7}}
    (out / "batch_result.json").write_text(json.dumps(
        {"dataset": "D11", "category": "vertical", "records": [rec]}))
    pd.DataFrame([{"method": "Matilda", "status": "CHAIN_OK"}]).to_csv(
        out / "summary.csv", index=False)
    return out


def test_load_batch_reads_an_old_tree_and_shows_the_new_statuses(tmp_path):
    res = mtb.load_batch(_tree(tmp_path))
    assert res.summary["status"].tolist() == ["scored"]
    assert res.records[0]["status"] == "CHAIN_OK"          # the record keeps the detail


def test_rescore_merges_the_named_metrics_and_save_writes_them(tmp_path):
    out = _tree(tmp_path)
    new = mtb.load_batch(out).rescore(metrics=["ARI"], verbose=False)
    assert new.records[0]["metrics"]["NMI"] == 0.8          # kept
    assert new.records[0]["metrics"]["ARI"] != 0.9          # scored again
    assert sorted(new.long["metric"]) == ["ARI", "NMI", "cLISI"]
    new.save()
    back = pd.read_csv(out / "summary.csv")
    assert {"ARI", "NMI", "cLISI"} <= set(back.columns) and back.status.tolist() == ["scored"]
    assert sorted(pd.read_csv(out / "long.csv")["metric"]) == ["ARI", "NMI", "cLISI"]


# --------------------------------------------------------------- C39 own clusters
def test_the_clustering_warning_names_sklearn():
    with pytest.warns(FutureWarning, match=r"sklearn\.metrics\.adjusted_rand_score"
                                           r"\(labels, clusters\)"):
        mtb.evaluate(np.eye(4), ["a", "b", "a", "b"], clustering=[0, 1, 0, 1],
                     metrics=["ARI"], verbose=False)
