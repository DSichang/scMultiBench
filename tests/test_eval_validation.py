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
from multibench.eval import pipeline
from multibench.eval.pipeline import LONG_COLUMNS, evaluate
from tests.test_study_r4_wf import N as N_CITE, _quiet, cite  # noqa: F401

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
    """ARI from evaluate's long table."""
    return float(df.set_index("metric").loc["ARI", "value"])


def _metrics(df):
    return df["metric"].tolist()


def _no_alignment_warning(rec):
    return [w for w in rec if "is matched by position" in str(w.message)]


# ------------------------------------------------------------------ I1: metrics=
def test_batch_metric_without_batch_labels_raises_not_empty():
    emb, ct, bat, _ = _toy()
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=ct, metrics=["GC"])
    msg = str(exc.value)
    assert "metrics=['GC'] needs batch labels for GC. Pass batch=<vector>" in msg
    # several offenders are all named
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=ct, metrics=["GC", "iLISI"])
    assert "['GC', 'iLISI']" in str(exc.value)
    # the family tokens need them too
    with pytest.raises(ValueError, match="metrics='batch' needs batch labels"):
        evaluate(emb, labels=ct, metrics="batch")
    with pytest.raises(ValueError, match="metrics='all' needs batch labels"):
        evaluate(emb, labels=ct, metrics="all")


def test_metrics_none_computes_every_applicable_family():
    from multibench.plot import BATCH_METRICS, CLASSIFICATION_METRICS, CLUSTERING_METRICS
    emb, ct, bat, _ = _toy()
    clu = evaluate(emb, labels=ct)                       # no batch: clustering family
    assert set(_metrics(clu)) == set(CLUSTERING_METRICS)
    # batches given: the batch family, and classification (no category: any)
    both = evaluate(emb, labels=ct, batch=bat)
    assert set(_metrics(both)) == (set(CLUSTERING_METRICS) | (set(BATCH_METRICS) - {"kBET"})
                               | set(CLASSIFICATION_METRICS))


def test_kbet_is_computed_only_when_named():
    emb, ct, bat, _ = _toy()
    assert "kBET" not in _metrics(evaluate(emb, labels=ct, batch=bat, metrics="batch"))
    assert "kBET" not in _metrics(evaluate(emb, labels=ct, batch=bat, metrics=["GC"]))
    assert _metrics(evaluate(emb, labels=ct, batch=bat, metrics=["kBET"])) == ["kBET"]


def test_only_valid_requests_still_work_and_unknown_lists_valid_names():
    emb, ct, bat, _ = _toy()
    df = evaluate(emb, labels=ct, batch=bat, metrics=["GC", "ARI"])
    assert set(_metrics(df)) == {"GC", "ARI"}
    assert _ari(df) == pytest.approx(1.0)
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=ct, metrics=["XYZ"])
    assert str(exc.value).startswith("Unknown metrics= token XYZ. Pass 'all'")


def test_batch_under_clustering_task_warns_instead_of_silently_ignoring():
    emb, ct, bat, _ = _toy()
    with pytest.warns(UserWarning, match=r"batch= changes nothing here, because "
                                         r"metrics=\['ASW'\] has no batch metric"):
        df = evaluate(emb, labels=ct, batch=bat, metrics=["ASW"])
    assert _metrics(df) == ["ASW"]
    # no such warning when batch is used
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        evaluate(emb, labels=ct, batch=bat, metrics=["ASW", "GC"])
    assert not [w for w in rec if "batch= changes nothing" in str(w.message)]


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


def test_labels_series_missing_or_extra_ids_raise_naming_them():
    emb, ct, _, ids = _toy()
    df = pd.DataFrame(emb, index=ids)
    ser = pd.Series(ct, index=ids)
    with pytest.raises(ValueError) as exc:
        evaluate(df, labels=ser.iloc[:-2], metrics=["ARI"])
    msg = str(exc.value)
    assert "cannot align by cell id" in msg and "2 of the output's 90 cells are missing" in msg
    # c4: to_numpy() only fits labels with one entry per cell
    assert "cell88" in msg and "cell89" in msg and "labels.to_numpy()" not in msg
    # c3 (E08): ids the output lacks are left out when every cell has its label
    extra = pd.concat([ser, pd.Series(["B"], index=["ghost"])])
    assert _ari(evaluate(df, labels=extra, metrics=["ARI"])) == \
        _ari(evaluate(df, labels=ser, metrics=["ARI"]))
    # a dims x cells DataFrame has row ids that are not cells: disjoint sets -> hint
    with pytest.raises(ValueError, match=r"^The labels' cell ids do not match the "
                                         r"embedding's \(embedding e\.g\. d0, labels e\.g\. "
                                         r"cell0\)\."):
        evaluate(pd.DataFrame(emb.T, columns=ids, index=[f"d{i}" for i in range(5)]),
                 labels=ser, metrics=["ARI"])
    # same rule for batch=
    with pytest.raises(ValueError, match="batch: cannot align by cell id"):
        evaluate(df, labels=ser, batch=pd.Series(["s1"] * 89, index=ids[:-1]),
                 metrics=["GC"])


def test_rescore_keeps_its_text_for_a_label_file_without_every_cell(cite, tmp_path):
    """rescore has no missing_labels, so its error for a label file that lacks
    cells of the dataset gives no advice about it."""
    import multibench as mtb
    data, _, _ = cite
    mtb.config.DEFAULT.data_path = data
    _quiet(mtb.run_all, 'MYCITE', 'vertical', tmp_path / 'a', methods=['Matilda'], modalities=['rna', 'adt'], verbose=False)
    res = mtb.load_batch(tmp_path / "a")
    pd.DataFrame({"barcode": [f"c{i}" for i in range(N_CITE - 30)],
                  "x": ["A"] * 60 + ["B"] * 30}).to_csv(tmp_path / "part.csv", index=False)
    with pytest.raises(ValueError) as exc:
        res.rescore(labels=str(tmp_path / "part.csv"))
    assert str(exc.value) == (
        "labels: part.csv has no row for 30 of the 120 cells of MYCITE (first: "
        "['c90', 'c91', 'c92']). Give a row for every cell.")
    assert "missing_labels" not in str(exc.value)


def test_labels_series_with_ids_against_bare_array_is_positional_with_warning():
    emb, ct, _, ids = _toy()
    ser = pd.Series(ct, index=ids)
    with pytest.warns(UserWarning) as rec:
        got = evaluate(emb, labels=ser, metrics=["ARI"])
    msgs = [str(w.message) for w in rec if "is matched by position" in str(w.message)]
    assert len(msgs) == 1, msgs
    # R5-01: the warning says how to put the vector in order, not how to silence it
    assert msgs[0] == ("The labels Series is matched by position, because the embedding "
                       "has no cell ids. Check that it follows the embedding rows, or "
                       "pass an AnnData whose obs_names are the barcodes.")
    assert _ari(got) == pytest.approx(1.0)         # positional and in order: fine
    # batch= gets its own (one) warning
    with pytest.warns(UserWarning, match="The batch Series is matched by position"):
        evaluate(emb, labels=ct, batch=pd.Series(["s1", "s2"] * 45, index=ids),
                 metrics=["GC"])


def test_labels_series_with_default_index_is_positional_and_silent():
    emb, ct, _, ids = _toy()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a = evaluate(emb, labels=pd.Series(ct), metrics=["ARI"])
        c = evaluate(emb, labels=ct, metrics=["ARI"])
    assert not _no_alignment_warning(rec)
    assert _ari(a) == _ari(c) == pytest.approx(1.0)
    # c3 (E05): next to an output with cell ids, the ids decide
    with pytest.raises(ValueError, match="labels is a Series without cell ids"):
        evaluate(pd.DataFrame(emb, index=ids), labels=pd.Series(ct), metrics=["ARI"])


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
    assert msg.startswith("The label keys atac_cty and rna_cty are not in "
                          "the default order. ")
    # 0.4.0: evaluate's own keywords, not the deprecated label_order= / labels_for
    assert "Pass a list of paths in cell order, or dataset=, category= and method=." in msg
    assert "label_order=" not in msg and "labels_for" not in msg
    assert "The default order is cty1, cty2, ... by number" in msg
    assert "rna before adt before atac" in msg
    assert "not alphabetical" in msg
    # R4-13: short sentences, no parenthesis holding clauses
    assert "(" not in msg and ";" not in msg
    # the same rule reaches as_vector() directly (batch= given as a dict)
    with pytest.raises(ValueError, match="batch: got a dict with 2 label files"):
        io.as_vector(d, what="batch")
    # metrics='all' does not trip over the batch check BEFORE explaining the dict
    with pytest.raises(ValueError, match="are not in the default order"):
        evaluate(emb, metrics="all", labels=d)


# ------------------------------------ I4: canonical names end to end
_CANONICAL = ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI", "ASW_batch", "ARI_batch",
              "NMI_batch", "GC", "iLISI", "kBET",
              "OCA", "ACA", "Spec", "Sens", "F1"]


