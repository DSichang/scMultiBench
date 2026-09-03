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
    assert to_long(df, "M", "D", "vertical")["metric"].tolist() == ["ARI"]


# --- P08: full 7-column schema, provenance overrides, input validation ------
import pytest


def test_to_long_columns_pin_results_columns():
    from multibench.data.results import COLUMNS
    from multibench.eval.pipeline import LONG_COLUMNS
    w = pd.DataFrame({"Value": [0.5]}, index=["ARI"])
    assert to_long(w, "M", "D", "vertical").columns.tolist() == COLUMNS == LONG_COLUMNS


def test_to_long_provenance_override_and_positional_call():
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    out = to_long(w, "M", "D", "vertical", clustering="louvain", source="mine")
    assert set(out.clustering) == {"louvain"} and set(out.source) == {"mine"}
    pos = to_long(w, "M", "D", "vertical")           # positional still works
    assert pos.source.tolist() == ["user", "user"]
    with pytest.raises(TypeError):                   # provenance is keyword-only
        to_long(w, "M", "D", "vertical", "louvain")


def test_to_long_needs_labels_column():
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    assert "needs_labels" not in to_long(w, "M", "D", "vertical").columns
    out = to_long(w, "M", "D", "vertical", needs_labels=True)
    assert out.columns.tolist()[-1] == "needs_labels" and out.needs_labels.dtype == bool
    assert out.needs_labels.all() and len(out.columns) == 8


def test_to_long_rejects_already_long_frame():
    w = pd.DataFrame({"Value": [0.5]}, index=["ARI"])
    long = to_long(w, "M", "D", "vertical")
    with pytest.raises(ValueError, match="already long frame"):
        to_long(long, "M2", "D", "vertical")


def test_to_long_rejects_wide_one_row_frame_with_hint():
    wide = pd.DataFrame([[0.5, 0.6]], columns=["ARI", "NMI"])       # RangeIndex row
    with pytest.raises(ValueError, match=r"expects evaluate\(\)'s frame") as e:
        to_long(wide, "M", "D", "vertical")
    assert "df.T.set_axis(['Value'], axis=1)" in str(e.value)
    # the hint is accepted, for a RangeIndex row and a named row alike
    ok = to_long(wide.T.set_axis(["Value"], axis=1), "M", "D", "vertical")
    assert ok.metric.tolist() == ["ARI", "NMI"]
    named = pd.DataFrame([[0.5, 0.6]], columns=["ARI", "NMI"], index=["M"])
    with pytest.raises(ValueError, match="expects evaluate"):
        to_long(named, "M", "D", "vertical")
    assert to_long(named.T.set_axis(["Value"], axis=1), "M", "D", "vertical").value.tolist() == [0.5, 0.6]
    # an empty frame names the expected shape too, not a KeyError
    with pytest.raises(ValueError, match="expects evaluate"):
        to_long(pd.DataFrame(), "M", "D", "vertical")


def test_to_long_accepts_csv_readback_and_series(tmp_path):
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=pd.Index(["ARI", "NMI"], name="metric"))
    f = tmp_path / "wide.csv"
    w.to_csv(f)                                    # what `multibench evaluate --out` writes
    back = pd.read_csv(f)                          # columns ['metric', 'Value'], RangeIndex
    out = to_long(back, "M", "D", "vertical")
    assert out.columns.tolist().count("metric") == 1
    assert out.metric.tolist() == ["ARI", "NMI"] and out.value.tolist() == [0.5, 0.6]
    ser = pd.Series({"ARI": 0.5, "NMI": 0.6})
    assert to_long(ser, "M", "D", "vertical").metric.tolist() == ["ARI", "NMI"]


def test_to_long_all_blank_names_raises():
    w = pd.DataFrame({"Value": [0.5, 0.1]}, index=["", ""])
    with pytest.raises(ValueError, match="no metric name in the index canonicalises"):
        to_long(w, "M", "D", "vertical")
