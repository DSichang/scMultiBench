"""mtb.evaluate returns the long table and reads a dataset's labels (0.4.0, O1/O2)."""
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.eval import pipeline

LONG = ["metric", "value", "method", "dataset", "category", "source", "scored_with"]


def _folder(tmp_path, n=40):
    """A cross-style folder of two label files and a random embedding of its cells."""
    ds = tmp_path / "TINY"
    ds.mkdir()
    rng = np.random.default_rng(0)
    for b in (1, 2):
        pd.DataFrame({"x": rng.choice(["T", "B"], n)}).to_csv(ds / f"cty{b}.csv", index=False)
    return ds, rng.normal(size=(2 * n, 4))


def test_evaluate_returns_the_long_table(tmp_path):
    ds, emb = _folder(tmp_path)
    labels = [str(ds / "cty1.csv"), str(ds / "cty2.csv")]
    out = mtb.evaluate(emb, labels, metrics="ARI", verbose=False)
    assert list(out.columns) == LONG
    # c1 (C16): no category given, none written
    assert out[["metric", "method", "dataset", "category", "source"]].iloc[0].tolist() == \
        ["ARI", "embedding", "user", None, "user"]
    assert out["scored_with"].iloc[0].startswith("leidenalg/search/")


def test_method_dataset_and_category_name_the_rows(tmp_path):
    ds, emb = _folder(tmp_path)
    out = mtb.evaluate(emb, [str(ds / "cty1.csv"), str(ds / "cty2.csv")], method="Mine",
                       dataset="MYDATA", category="cross", metrics=["ARI", "NMI"],
                       verbose=False)
    assert out["metric"].tolist() == ["ARI", "NMI"]
    assert set(out["method"]) == {"Mine"} and set(out["dataset"]) == {"MYDATA"}
    assert set(out["category"]) == {"cross"}


def test_dataset_reads_the_label_files_and_its_name_fills_the_column(tmp_path):
    ds, emb = _folder(tmp_path)
    by_path = mtb.evaluate(emb, dataset=ds, category="cross", metrics="ARI", verbose=False)
    files = mtb.evaluate(emb, [str(ds / "cty1.csv"), str(ds / "cty2.csv")], dataset="TINY",
                         category="cross", metrics="ARI", verbose=False)
    assert set(by_path["dataset"]) == {"TINY"}          # the folder's name, not the path
    assert by_path["value"].tolist() == files["value"].tolist()


def test_a_package_method_reads_the_labels_in_its_cell_order():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # no default-order warning either
        own = pipeline._dataset_labels("D52mini", "cross", "StabMap", None)
        mine = pipeline._dataset_labels("D52mini", "cross", "Mine", None)
    assert list(own) == ["cty3", "cty1", "cty2"]
    assert list(mine) == ["cty1", "cty2", "cty3"]


def test_a_name_close_to_a_reordering_method_gets_one_did_you_mean():
    with pytest.warns(UserWarning, match=r"Did you mean method='StabMap'\? StabMap puts the "
                                         r"cells of D52mini in another order"):
        pipeline._dataset_labels("D52mini", "cross", "StabMapp", None)


def test_no_labels_and_no_dataset_says_what_to_pass():
    with pytest.raises(ValueError, match="evaluate needs labels=, one cell type per cell, "
                                         "or dataset="):
        mtb.evaluate(np.zeros((4, 2)))


def test_clusters_never_shipped():
    with pytest.raises(TypeError):
        mtb.evaluate(np.zeros((4, 2)), ["a", "b", "a", "b"], clusters=[0, 1, 0, 1])


def test_one_metric_code_and_a_family_in_a_list(tmp_path):
    ds, emb = _folder(tmp_path)
    labels = [str(ds / "cty1.csv"), str(ds / "cty2.csv")]
    one = mtb.evaluate(emb, labels, metrics="ari", verbose=False)
    both = mtb.evaluate(emb, labels, metrics=["clustering", "ARI"], verbose=False)
    assert one["metric"].tolist() == ["ARI"]
    assert both["metric"].tolist()[:2] == ["ARI", "NMI"]
    assert both["metric"].is_unique


def test_the_leiden_line_gives_no_skip_advice(tmp_path, capsys):
    ds, emb = _folder(tmp_path, n=1100)                 # over the 2,000-cell notice
    mtb.evaluate(emb, [str(ds / "cty1.csv"), str(ds / "cty2.csv")], metrics="ARI")
    err = capsys.readouterr().err
    assert "with the leidenalg Leiden backend." in err
    assert "To skip it" not in err


def test_to_long_is_deprecated_and_takes_the_long_table(tmp_path):
    ds, emb = _folder(tmp_path)
    out = mtb.evaluate(emb, [str(ds / "cty1.csv"), str(ds / "cty2.csv")], metrics="ARI",
                       verbose=False)
    with pytest.warns(FutureWarning, match=r"^to_long is deprecated and will be removed in "
                                           r"0\.5\. Use mtb\.evaluate"):
        again = mtb.to_long(out, method="X", dataset="D11")
    assert again[["method", "dataset", "category"]].iloc[0].tolist() == ["X", "D11", None]
    assert again["value"].tolist() == out["value"].tolist()
