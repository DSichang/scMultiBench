"""catalog.metrics() and catalog.datasets() say what the package computes and
ships (study round 1, L37 and L56).

Before the fix catalog.metrics() described iASW as 'for rare/isolated cell
types' and iF1 as 'for the most isolated label(s)' while evaluate scores
every cell type, ARI named 'Louvain/Leiden' clusters, and only some rows said
which direction is better. catalog.datasets() carried five columns that were
empty in all rows.
"""
import warnings

import pandas as pd

import multibench as mtb
from multibench.data import catalog


def _desc():
    return mtb.catalog.metrics().set_index("metric")["description"]


def test_isolated_label_metrics_describe_every_cell_type():
    d = _desc()
    for code in ("iASW", "iF1"):
        assert "rare" not in d[code] and "most isolated" not in d[code], d[code]
    assert "Every cell type is treated as an isolated label" in d["iF1"]
    assert "Every cluster is treated as an isolated label" in d["iASW"]


def test_every_metric_states_direction_and_the_clustering_used():
    d = _desc()
    assert set(d.index) == {"ARI", "NMI", "ASW", "iASW", "iF1", "cLISI",
                            "ASW_batch", "ARI_batch", "NMI_batch", "GC", "iLISI", "kBET",
                            "OCA", "ACA", "Spec", "Sens", "F1"}
    assert all(text.endswith("higher = better.") for text in d), d.to_dict()
    for code in ("ARI", "NMI"):
        assert "Leiden clusters, one per cell type" in d[code]
        assert "Louvain" not in d[code]


def test_the_classification_rows_follow_the_family_order():
    # the rows are the codes of known_metrics, in its order (PCR left it in 0.4)
    rows = list(mtb.catalog.metrics()["metric"])
    assert rows == catalog.known_metrics()
    d = _desc()
    assert d["OCA"].startswith("Overall classification accuracy: the fraction of query cells")
    assert d["ACA"].startswith("Average classification accuracy: the fraction of each cell type")
    for code in ("Spec", "Sens", "F1"):
        assert "averaged over cell types" in d[code], code


def test_datasets_drops_the_always_empty_columns():
    df = catalog.datasets()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # PAPER_COLUMNS is deprecated
        paper_columns = catalog.PAPER_COLUMNS
    for col in paper_columns:
        assert col not in df.columns, col
    # 0.4.0: the demo subsets join the rows; every row is fetchable, so no such column
    assert list(df.columns) == ["dataset", "category", "has_results"]


def test_datasets_keeps_a_descriptive_column_a_csv_fills(tmp_path):
    pd.DataFrame({"dataset": ["D11", "D28"], "assay": ["CITE-seq", None],
                  "tissue": [None, None]}).to_csv(tmp_path / "dataset.csv", index=False)
    df = catalog.datasets(tmp_path).set_index("dataset")
    assert "assay" in df.columns and "tissue" not in df.columns
    assert df.loc["D11", "assay"] == "CITE-seq"
