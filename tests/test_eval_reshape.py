import matplotlib
matplotlib.use("Agg")

import pandas as pd

from multibench.eval import to_long
import multibench as mtb


def test_to_long_shape_and_canonicalization():
    df = pd.DataFrame({"Value": [0.8, 0.6]}, index=["ARI", "NMI"])
    out = to_long(df, method="SCALEX", dataset="D27", category="diagonal")
    assert list(out.columns) == ["metric", "value", "method", "dataset", "category",
                                 "clustering", "source"]
    assert set(out["clustering"]) == {"default"} and set(out["source"]) == {"user"}
    # values preserved
    vals = dict(zip(out["metric"], out["value"]))
    assert vals["ARI"] == 0.8
    assert vals["NMI"] == 0.6
    # method/dataset/category broadcast
    assert set(out["method"]) == {"SCALEX"}
    assert set(out["dataset"]) == {"D27"}
    assert set(out["category"]) == {"diagonal"}
    # metric canonicalized (ARI/NMI canonicalize to themselves)
    assert set(out["metric"]) == {"ARI", "NMI"}


def test_to_long_canonicalizes_lowercase_codes():
    df = pd.DataFrame({"Value": [0.5, 0.4]}, index=["ari", "kbet"])
    out = to_long(df, method="GLUE", dataset="D27", category="diagonal")
    assert set(out["metric"]) == {"ARI", "kBET"}


def test_evaluate_to_plot_handoff_end_to_end(tmp_path):
    w = pd.DataFrame({"Value": [0.8, 0.6]}, index=["ARI", "NMI"])
    l1 = to_long(w, method="SCALEX", dataset="D27", category="diagonal")
    l2 = to_long(w, method="GLUE", dataset="D27", category="diagonal")
    long_df = pd.concat([l1, l2], ignore_index=True)
    out = tmp_path / "f.png"
    mtb.plot.bubble(long_df, metrics=["ARI", "NMI"], save=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_to_long_rejects_colliding_metric_names():
    import pytest
    df = pd.DataFrame({"Value": [0.5, 0.4]}, index=["ari", "ARI"])
    with pytest.raises(ValueError, match="collide after canonicalisation"):
        to_long(df, method="M", dataset="D", category="vertical")
    # blank / unknown-empty names are dropped, not kept as NaN rows
    df = pd.DataFrame({"Value": [0.5, 0.1]}, index=["nmi", ""])
    out = to_long(df, method="M", dataset="D", category="vertical")
    assert out["metric"].tolist() == ["NMI"]
    assert list(out.columns) == ["metric", "value", "method", "dataset", "category",
                                 "clustering", "source"]
    # a named index (e.g. read back from CSV) still becomes 'metric'
    df = pd.DataFrame({"Value": [0.5]}, index=pd.Index(["ARI"], name="Metric"))
    assert to_long(df, method="M", dataset="D", category="vertical")["metric"].tolist() == ["ARI"]


# --- P08: full 7-column schema, provenance overrides, input validation ------
import pytest


def test_to_long_columns_pin_results_columns():
    from multibench.data.results import COLUMNS
    from multibench.eval.pipeline import LONG_COLUMNS
    w = pd.DataFrame({"Value": [0.5]}, index=["ARI"])
    assert to_long(w, method="M", dataset="D", category="vertical").columns.tolist() == COLUMNS == LONG_COLUMNS


def test_to_long_provenance_override_and_keyword_only_call():
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    out = to_long(w, method="M", dataset="D", category="vertical",
                  clustering="louvain", source="mine")
    assert set(out.clustering) == {"louvain"} and set(out.source) == {"mine"}
    kw = to_long(w, method="M", dataset="D", category="vertical")
    assert kw.source.tolist() == ["user", "user"]
    with pytest.raises(TypeError):                   # everything after value_df is keyword-only
        to_long(w, "M", "D", "vertical")
    with pytest.raises(TypeError):
        to_long(w, "M", dataset="D", category="vertical")


def test_to_long_dataset_and_category_default_to_placeholders():
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    out = to_long(w, method="M")
    assert out.columns.tolist() == ["metric", "value", "method", "dataset", "category",
                                    "clustering", "source"]
    assert set(out.dataset) == {"all"} and set(out.category) == {"user"}


def test_to_long_rejects_already_long_frame():
    w = pd.DataFrame({"Value": [0.5]}, index=["ARI"])
    long = to_long(w, method="M", dataset="D", category="vertical")
    with pytest.raises(ValueError, match="already long frame"):
        to_long(long, method="M2", dataset="D", category="vertical")


def test_to_long_rejects_wide_one_row_frame_with_hint():
    wide = pd.DataFrame([[0.5, 0.6]], columns=["ARI", "NMI"])       # RangeIndex row
    with pytest.raises(ValueError, match=r"expects evaluate\(\)'s frame") as e:
        to_long(wide, method="M", dataset="D", category="vertical")
    assert "df.T.set_axis(['Value'], axis=1)" in str(e.value)
    # the hint is accepted, for a RangeIndex row and a named row alike
    ok = to_long(wide.T.set_axis(["Value"], axis=1), method="M", dataset="D", category="vertical")
    assert ok.metric.tolist() == ["ARI", "NMI"]
    named = pd.DataFrame([[0.5, 0.6]], columns=["ARI", "NMI"], index=["M"])
    with pytest.raises(ValueError, match="expects evaluate"):
        to_long(named, method="M", dataset="D", category="vertical")
    assert to_long(named.T.set_axis(["Value"], axis=1), method="M", dataset="D", category="vertical").value.tolist() == [0.5, 0.6]
    # an empty frame names the expected shape too, not a KeyError
    with pytest.raises(ValueError, match="expects evaluate"):
        to_long(pd.DataFrame(), method="M", dataset="D", category="vertical")


def test_to_long_accepts_csv_readback_and_series(tmp_path):
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=pd.Index(["ARI", "NMI"], name="metric"))
    f = tmp_path / "wide.csv"
    w.to_csv(f)                                    # header 'metric,Value' (a named index)
    back = pd.read_csv(f)                          # columns ['metric', 'Value'], RangeIndex
    out = to_long(back, method="M", dataset="D", category="vertical")
    assert out.columns.tolist().count("metric") == 1
    assert out.metric.tolist() == ["ARI", "NMI"] and out.value.tolist() == [0.5, 0.6]
    ser = pd.Series({"ARI": 0.5, "NMI": 0.6})
    assert to_long(ser, method="M", dataset="D", category="vertical").metric.tolist() == ["ARI", "NMI"]


def test_to_long_all_blank_names_raises():
    w = pd.DataFrame({"Value": [0.5, 0.1]}, index=["", ""])
    with pytest.raises(ValueError, match="no metric name in the index canonicalises"):
        to_long(w, method="M", dataset="D", category="vertical")


# --- CSV read-back: metric names are strings or to_long raises -------------
def _evaluate_frame():
    """A real ``mtb.evaluate`` frame (its index is unnamed)."""
    import numpy as np
    lab = np.repeat(["a", "b"], 20)
    emb = np.random.default_rng(0).normal(size=(40, 3)) + (lab == "b")[:, None] * 5.0
    return mtb.evaluate(emb, labels=lab, clustering=lab, metrics=["ARI", "NMI"])


def test_to_long_accepts_plain_read_csv_of_evaluate_frame(tmp_path):
    wide = _evaluate_frame()
    f = tmp_path / "m.csv"
    wide.to_csv(f)                                 # header ',Value': the index is unnamed
    back = pd.read_csv(f)
    assert back.columns.tolist() == ["Unnamed: 0", "Value"]
    out = to_long(back, method="M", dataset="D", category="vertical")
    assert out.metric.tolist() == ["ARI", "NMI"]
    assert out.value.tolist() == wide["Value"].tolist()
    assert out.columns.tolist() == ["metric", "value", "method", "dataset", "category",
                                    "clustering", "source"]
    # the documented read-back gives the same rows
    same = to_long(pd.read_csv(f, index_col=0), method="M", dataset="D", category="vertical")
    pd.testing.assert_frame_equal(out, same)


def test_to_long_accepts_cli_evaluate_out_read_back_plainly(tmp_path):
    from multibench import cli
    import numpy as np
    lab = np.repeat(["a", "b"], 20)
    emb = tmp_path / "e.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(40, 3)) + (lab == "b")[:, None] * 5.0)
    labels = tmp_path / "cty.csv"
    pd.DataFrame({"x": lab}).to_csv(labels, index=False)
    f = tmp_path / "wide.csv"
    assert cli.main(["evaluate", "--output", str(emb), "--labels", str(labels),
                     "--clustering", str(labels), "--metrics", "ARI,NMI", "--out", str(f)]) == 0
    out = to_long(pd.read_csv(f), method="M", dataset="D", category="vertical")
    assert out.metric.tolist() == ["ARI", "NMI"]


FIX = "pd.read_csv(path, index_col=0)"


@pytest.mark.parametrize("frame", [
    # saved twice with the index: the row numbers come first, the names second
    pd.DataFrame({"Unnamed: 0": [0, 1], "Unnamed: 0.1": ["ARI", "NMI"], "Value": [0.5, 0.6]}),
    # an unnamed column that holds no names
    pd.DataFrame({"Unnamed: 0": [0, 1], "Value": [0.5, 0.6]}),
    # the read-back shape plus a column evaluate never writes
    pd.DataFrame({"Unnamed: 0": ["ARI", "NMI"], "Value": [0.5, 0.6], "note": ["x", "y"]}),
    # wide.to_csv(path, index_label="Metric") read back plainly
    pd.DataFrame({"Metric": ["ARI", "NMI"], "Value": [0.5, 0.6]}),
    # no names anywhere
    pd.DataFrame({"Value": [0.5, 0.6]}),
    pd.Series([0.5, 0.6]),
    # rows filtered after a plain read-back of the named-column form
    pd.DataFrame({"Metric": ["ARI", "NMI", "ASW"], "Value": [0.5, 0.6, 0.7]}).iloc[[0, 2]],
], ids=["saved-twice", "unnamed-ints", "extra-column", "other-label", "range-index",
        "range-series", "filtered-rows"])
def test_to_long_raises_instead_of_numbering_metrics(frame):
    with pytest.raises(ValueError, match="metric names") as e:
        to_long(frame, method="M", dataset="D", category="vertical")
    assert FIX in str(e.value)


def test_to_long_unnamed_column_is_used_only_when_the_index_holds_no_names():
    frame = pd.DataFrame({"Unnamed: 0": ["x", "y"], "Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    assert to_long(frame, method="M", dataset="D", category="vertical").metric.tolist() == ["ARI", "NMI"]


def test_to_long_unnamed_read_back_drops_a_blank_name(tmp_path):
    f = tmp_path / "m.csv"
    pd.DataFrame({"Value": [0.5, 0.1]}, index=["ARI", ""]).to_csv(f)
    back = pd.read_csv(f)                          # the blank name reads back as NaN
    assert back.columns.tolist() == ["Unnamed: 0", "Value"]
    assert to_long(back, method="M", dataset="D", category="vertical").metric.tolist() == ["ARI"]


def test_to_long_named_fix_recovers_a_file_saved_twice(tmp_path):
    once, twice = tmp_path / "once.csv", tmp_path / "twice.csv"
    pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"]).to_csv(once)
    pd.read_csv(once).to_csv(twice)                # ',Unnamed: 0,Value' then '0,ARI,0.5'
    with pytest.raises(ValueError, match="metric names"):
        to_long(pd.read_csv(twice), method="M", dataset="D", category="vertical")
    back = pd.read_csv(twice, index_col=0)         # the fix the message names
    out = to_long(back, method="M", dataset="D", category="vertical")
    assert out.metric.tolist() == ["ARI", "NMI"] and out.value.tolist() == [0.5, 0.6]
