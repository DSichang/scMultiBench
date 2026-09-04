"""evaluate() input validation and alignment, pinned by the 0.3.0 user study.

Four silent-wrong-answer paths are closed here:

* ``metrics=`` asking for a batch metric without batch labels used to run
  the full pipeline and return an EMPTY frame (Chen: ``only={'GC'}`` under the
  default task -> 14 s, ``Empty DataFrame``);
* a labels/batch/clustering ``Series`` was consumed POSITIONALLY, its index
  ignored - a shuffled ``obs['celltype']`` scored ARI 0.002 with no warning
  (Tomas);
* a multi-entry ``labels_for()`` dict was refused with a hint recommending the
  ALPHABETICAL order, which on D28 (atac before rna) scored a perfect
  embedding at ARI 0.001 (Marcus);
* the metric names evaluate() emits must be the canonical spelling
  ``load_results`` uses, so ``pd.concat([published, mine])`` never carries two
  spellings of one metric (Yuki, Elena).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench.eval import io
from multibench.eval.pipeline import evaluate, to_long

pytest.importorskip("scib")


def _toy(n_per=45, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.1, size=(n_per, 5))
    b = rng.normal(5, 0.1, size=(n_per, 5))
    emb = np.vstack([a, b])
    ct = np.array(["B"] * n_per + ["T"] * n_per)
    bat = np.array(["s1", "s2"] * n_per)
    ids = pd.Index([f"cell{i}" for i in range(2 * n_per)])
    return emb, ct, bat, ids


def _ari(df):
    return float(df.loc["ARI", "Value"])


def _no_alignment_warning(rec):
    return [w for w in rec if "non-default index" in str(w.message)]


# ------------------------------------------------------------------ I1: metrics=
def test_batch_metric_without_batch_labels_raises_not_empty():
    emb, ct, bat, _ = _toy()
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=ct, metrics=["GC"])
    msg = str(exc.value)
    assert "batch labels required for batch metric(s) ['GC']: pass batch=<vector>" in msg
    # several offenders are all named
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=ct, metrics=["GC", "iLISI"])
    assert "['GC', 'iLISI']" in str(exc.value)
    # the family tokens need them too
    with pytest.raises(ValueError, match="batch labels required for metrics='batch'"):
        evaluate(emb, labels=ct, metrics="batch")
    with pytest.raises(ValueError, match="batch labels required for metrics='all'"):
        evaluate(emb, labels=ct, metrics="all")


def test_metrics_none_computes_every_applicable_family():
    from multibench.plot import BATCH_METRICS, CLUSTERING_METRICS
    emb, ct, bat, _ = _toy()
    clu = evaluate(emb, labels=ct)                       # no batch: clustering family
    assert set(clu.index) == set(CLUSTERING_METRICS)
    both = evaluate(emb, labels=ct, batch=bat)           # batch given: both families
    assert set(both.index) == set(CLUSTERING_METRICS) | (set(BATCH_METRICS) - {"kBET"})


def test_kbet_is_computed_only_when_named():
    emb, ct, bat, _ = _toy()
    assert "kBET" not in evaluate(emb, labels=ct, batch=bat, metrics="batch").index
    assert "kBET" not in evaluate(emb, labels=ct, batch=bat, metrics=["GC"]).index
    assert list(evaluate(emb, labels=ct, batch=bat, metrics=["kBET"]).index) == ["kBET"]


def test_only_valid_requests_still_work_and_unknown_lists_valid_names():
    emb, ct, bat, _ = _toy()
    df = evaluate(emb, labels=ct, batch=bat, metrics=["GC", "ARI"])
    assert set(df.index) == {"GC", "ARI"}
    assert _ari(df) == pytest.approx(1.0)
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=ct, metrics=["XYZ"])
    assert "unknown metric(s) ['XYZ']" in str(exc.value) and "GC" in str(exc.value)


def test_batch_under_clustering_task_warns_instead_of_silently_ignoring():
    emb, ct, bat, _ = _toy()
    with pytest.warns(UserWarning, match=r"batch= was given but metrics=\['ASW'\] computes no batch metric"):
        df = evaluate(emb, labels=ct, batch=bat, metrics=["ASW"])
    assert list(df.index) == ["ASW"]
    # no such warning when batch is used
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        evaluate(emb, labels=ct, batch=bat, metrics=["ASW", "GC"])
    assert not [w for w in rec if "batch= was given" in str(w.message)]


def test_evaluate_never_returns_an_empty_frame():
    """Every public path either raises or returns at least one metric row."""
    emb, ct, bat, _ = _toy()
    for kw in ({"metrics": "clustering"}, {"metrics": "batch", "batch": bat},
               {"metrics": "all", "batch": bat}, {"metrics": ["ASW"]},
               {"metrics": None, "batch": bat}):
        df = evaluate(emb, labels=ct, **kw)
        assert not df.empty


# -------------------------------------------------- I2: index alignment
def test_labels_series_aligned_by_index_against_dataframe_output():
    emb, ct, _, ids = _toy()
    df = pd.DataFrame(emb, index=ids)
    ser = pd.Series(ct, index=ids)
    shuffled = ser.sample(frac=1, random_state=0)
    assert not shuffled.index.equals(ids)          # the test is real
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = evaluate(df, labels=shuffled, metrics=["ARI"])
    assert _ari(got) == pytest.approx(1.0)         # positional would be ~0
    assert not _no_alignment_warning(rec)


def test_labels_series_aligned_by_obs_names_against_anndata_output():
    ad = pytest.importorskip("anndata")
    emb, ct, bat, ids = _toy()
    a = ad.AnnData(np.zeros((len(ct), 2)))
    a.obs_names = ids
    a.obsm["X_emb"] = emb
    a.obs["celltype"] = ct
    shuffled = pd.Series(ct, index=ids).sample(frac=1, random_state=1)
    got = evaluate(a, labels=shuffled, metrics=["ARI"])
    assert _ari(got) == pytest.approx(1.0)
    # batch= follows the same rule: a shuffled batch Series is re-aligned
    shuffled_batch = pd.Series(bat, index=ids).sample(frac=1, random_state=2)
    ref = evaluate(a, labels="celltype", batch=bat, metrics=["GC", "ASW_batch"])
    got = evaluate(a, labels="celltype", batch=shuffled_batch,
                   metrics=["GC", "ASW_batch"])
    pd.testing.assert_frame_equal(got.sort_index(), ref.sort_index())
    # clustering= too (and an obs column name works for clustering, like labels)
    a.obs["leiden"] = pd.Categorical(ct)
    c1 = evaluate(a, labels="celltype", clustering="leiden", metrics=["ARI"])
    c2 = evaluate(a, labels="celltype",
                  clustering=pd.Series(ct, index=ids).sample(frac=1, random_state=3),
                  metrics=["ARI"])
    assert _ari(c1) == pytest.approx(1.0) and _ari(c2) == pytest.approx(1.0)


def test_labels_series_missing_or_extra_ids_raise_naming_them():
    emb, ct, _, ids = _toy()
    df = pd.DataFrame(emb, index=ids)
    ser = pd.Series(ct, index=ids)
    with pytest.raises(ValueError) as exc:
        evaluate(df, labels=ser.iloc[:-2], metrics=["ARI"])
    msg = str(exc.value)
    assert "cannot align by cell id" in msg and "2 of the output's 90 cells are missing" in msg
    assert "cell88" in msg and "cell89" in msg and "labels.to_numpy()" in msg
    extra = pd.concat([ser, pd.Series(["B"], index=["ghost"])])
    with pytest.raises(ValueError) as exc:
        evaluate(df, labels=extra, metrics=["ARI"])
    assert "1 id(s) the output lacks" in str(exc.value) and "ghost" in str(exc.value)
    # a dims x cells DataFrame has row ids that are not cells: disjoint sets -> hint
    with pytest.raises(ValueError, match="is the output transposed"):
        evaluate(pd.DataFrame(emb.T, columns=ids, index=[f"d{i}" for i in range(5)]),
                 labels=ser, metrics=["ARI"])
    # same rule for batch=
    with pytest.raises(ValueError, match="batch: cannot align by cell id"):
        evaluate(df, labels=ser, batch=pd.Series(["s1"] * 89, index=ids[:-1]),
                 metrics=["GC"])


def test_labels_series_with_ids_against_bare_array_is_positional_with_warning():
    emb, ct, _, ids = _toy()
    ser = pd.Series(ct, index=ids)
    with pytest.warns(UserWarning) as rec:
        got = evaluate(emb, labels=ser, metrics=["ARI"])
    msgs = [str(w.message) for w in rec if "non-default index" in str(w.message)]
    assert len(msgs) == 1, msgs
    assert ("labels Series has a non-default index; matched positionally because "
            "the embedding carries no cell ids - pass labels.to_numpy() to silence, "
            "or an AnnData/DataFrame with cell ids to align") in msgs[0]
    assert _ari(got) == pytest.approx(1.0)         # positional and in order: fine
    # batch= gets its own (one) warning
    with pytest.warns(UserWarning, match="batch Series has a non-default index"):
        evaluate(emb, labels=ct, batch=pd.Series(["s1", "s2"] * 45, index=ids),
                 metrics=["GC"])


def test_labels_series_with_default_index_is_positional_and_silent():
    emb, ct, _, ids = _toy()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a = evaluate(emb, labels=pd.Series(ct), metrics=["ARI"])
        b = evaluate(pd.DataFrame(emb, index=ids), labels=pd.Series(ct), metrics=["ARI"])
        c = evaluate(emb, labels=ct, metrics=["ARI"])
    assert not _no_alignment_warning(rec)
    assert _ari(a) == _ari(b) == _ari(c) == pytest.approx(1.0)


def test_align_vector_unit():
    ids = pd.Index(["a", "b", "c"])
    s = pd.Series([1, 2, 3], index=["c", "a", "b"])
    assert io.align_vector(s, ids).tolist() == [2, 3, 1]
    # same order already -> no-op, even with duplicated ids
    dup = pd.Index(["a", "a", "b"])
    assert io.align_vector(pd.Series([1, 2, 3], index=dup), dup).tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="duplicated id"):
        io.align_vector(pd.Series([1, 2, 3], index=["a", "a", "b"]), ids)
    with pytest.raises(ValueError, match="output carries 1 duplicated cell id"):
        io.align_vector(s, dup)
    # DataFrame forms
    frame = pd.DataFrame({"ct": [1, 2, 3], "other": [0, 0, 0]}, index=["c", "a", "b"])
    assert io.align_vector(frame, ids, column="ct").tolist() == [2, 3, 1]
    assert io.align_vector(frame[["ct"]], ids).tolist() == [2, 3, 1]
    with pytest.raises(ValueError, match="pass one column"):
        io.align_vector(frame, ids)


# ------------------------------------------------ I3: dict + label_order
def _write_cty(path, values):
    pd.DataFrame({"x": values}).to_csv(path, index=False)


def test_multi_entry_dict_error_names_keys_and_the_stacking_rule(tmp_path):
    emb, ct, _, _ = _toy()
    p1, p2 = tmp_path / "atac_cty.csv", tmp_path / "rna_cty.csv"
    _write_cty(p1, ct[30:]); _write_cty(p2, ct[:30])
    d = {"atac_cty": str(p1), "rna_cty": str(p2)}      # alphabetical, as labels_for used to
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=d, metrics=["ARI"])
    msg = str(exc.value)
    assert "got a dict with 2 label files ['atac_cty', 'rna_cty']" in msg
    assert "label_order=" in msg
    assert "stacking order" in msg and "cty1, cty2" in msg and "rna before adt before atac" in msg
    assert "NOT alphabetical" in msg
    assert "mtb.labels_for(dataset, method=<method>, category=<category>)" in msg
    # the same rule reaches as_vector() directly (batch= given as a dict)
    with pytest.raises(ValueError, match="batch: got a dict with 2 label files"):
        io.as_vector(d, what="batch")
    # metrics='all' does not trip over the batch check BEFORE explaining the dict
    with pytest.raises(ValueError, match="stacking order"):
        evaluate(emb, metrics="all", labels=d)


def test_label_order_fixes_the_dict_order_and_feeds_the_file_batch(tmp_path):
    emb, ct, _, _ = _toy()
    p_rna, p_atac = tmp_path / "rna_cty.csv", tmp_path / "atac_cty.csv"
    _write_cty(p_rna, ct[:60]); _write_cty(p_atac, ct[60:])
    d = {"atac_cty": str(p_atac), "rna_cty": str(p_rna)}
    got = evaluate(emb, labels=d, label_order=["rna_cty", "atac_cty"],
                   metrics=["ARI", "GC"])
    ref = evaluate(emb, labels=[p_rna, p_atac], metrics=["ARI", "GC"])
    pd.testing.assert_frame_equal(got.sort_index(), ref.sort_index())
    assert _ari(got) == pytest.approx(1.0)
    # trusting the dict's own order is spelled label_order=list(d): here it is
    # the WRONG (alphabetical) order, and the score shows it - explicitly asked for
    wrong = evaluate(emb, labels=d, label_order=list(d), metrics=["ARI"])
    assert _ari(wrong) < 0.5
    # a one-entry dict needs no order (unchanged behaviour)
    one = tmp_path / "cty.csv"; _write_cty(one, ct)
    assert _ari(evaluate(emb, labels={"cty": str(one)}, metrics=["ARI"])) == pytest.approx(1.0)
    # a subset selects those files
    sub = evaluate(emb[:60], labels=d, label_order=["rna_cty"], metrics=["ARI"])
    assert _ari(sub) == pytest.approx(1.0)


def test_label_order_validation_messages(tmp_path):
    emb, ct, _, _ = _toy()
    p1, p2 = tmp_path / "cty1.csv", tmp_path / "cty2.csv"
    _write_cty(p1, ct[:30]); _write_cty(p2, ct[30:])
    d = {"cty1": str(p1), "cty2": str(p2)}
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=d, label_order=["cty1", "cty3"], metrics=["ARI"])
    assert "label_order names key(s) ['cty3'] that are not in the labels dict" in str(exc.value)
    assert "['cty1', 'cty2']" in str(exc.value)
    with pytest.raises(ValueError, match="label_order repeats a key"):
        evaluate(emb, labels=d, label_order=["cty1", "cty1"], metrics=["ARI"])
    with pytest.raises(ValueError, match="label_order= is empty"):
        evaluate(emb, labels=d, label_order=[], metrics=["ARI"])
    with pytest.raises(TypeError, match="label_order= must be a list of keys"):
        evaluate(emb, labels=d, label_order="cty1", metrics=["ARI"])
    with pytest.raises(TypeError, match="label_order= applies only when labels is a dict"):
        evaluate(emb, labels=[p1, p2], label_order=["cty1", "cty2"], metrics=["ARI"])
    with pytest.raises(TypeError):                 # keyword-only: no positional slot
        evaluate(emb, None, "clustering", d, None, None, "scib", False, {"ARI"},
                 "X_emb", None, ["cty1", "cty2"])


# ------------------------------------ I4: canonical names end to end
_CANONICAL = ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI", "ASW_batch", "GC", "iLISI", "kBET"]


def test_evaluate_emits_canonical_metric_names_only():
    from multibench.data import catalog
    emb, ct, bat, _ = _toy()
    df = evaluate(emb, metrics="all", labels=ct, batch=bat)
    assert set(df.index) <= set(_CANONICAL), sorted(df.index)
    assert all(catalog.canonical_metric(m) == m for m in df.index)
    assert list(df.columns) == ["Value"]


def test_evaluate_to_long_concat_with_published_builds_the_bubble_table():
    import matplotlib
    matplotlib.use("Agg")
    import multibench as mtb
    from multibench import plot as mplot

    emb, ct, bat, _ = _toy()
    mine_wide = evaluate(emb, metrics="all", labels=ct, batch=bat)
    mine = to_long(mine_wide, method="MyMethod", dataset="D52", category="cross")
    published = mtb.load_results("cross", dataset="D52", source="rerun")
    both = pd.concat([published, mine], ignore_index=True)
    # one spelling per metric after canonicalisation - no duplicate columns
    canon = both["metric"].map(mtb.data.catalog.canonical_metric)
    assert set(canon) == set(both["metric"]), "to_long/evaluate names were not canonical"
    assert both.groupby(["method", "metric"]).size().max() == 1
    tab = mplot.build_table(both)
    assert "MyMethod" in tab.methods
    wide = both.pivot_table(index="method", columns="metric", values="value")
    assert len(wide.columns) == len(set(published["metric"]) | set(mine["metric"]))
    assert wide.loc["MyMethod", "ARI"] == pytest.approx(float(mine_wide.loc["ARI", "Value"]))
