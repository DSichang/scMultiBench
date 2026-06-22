import matplotlib
matplotlib.use("Agg")

import pandas as pd

from multibench.eval import to_long
import multibench as mtb


def test_to_long_shape_and_canonicalization():
    df = pd.DataFrame({"Value": [0.8, 0.6]}, index=["ARI", "NMI"])
    out = to_long(df, method="SCALEX", dataset="D27", category="diagonal")
    assert list(out.columns) == ["metric", "value", "method", "dataset", "category"]
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
