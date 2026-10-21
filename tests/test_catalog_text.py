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


def test_datasets_keeps_a_descriptive_column_a_csv_fills(tmp_path, monkeypatch):
    pd.DataFrame({"dataset": ["D11", "D28"], "assay": ["CITE-seq", None],
                  "tissue": [None, None]}).to_csv(tmp_path / "dataset.csv", index=False)
    monkeypatch.setattr(catalog.config, "_FILES_PATH", tmp_path)
    df = catalog.datasets().set_index("dataset")
    assert "assay" in df.columns and "tissue" not in df.columns
    assert df.loc["D11", "assay"] == "CITE-seq"
