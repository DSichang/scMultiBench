"""catalog.metrics() and catalog.datasets() say what the package computes and
ships (study round 1, L37 and L56).

Before the fix catalog.metrics() described iASW as 'for rare/isolated cell
types' and iF1 as 'for the most isolated label(s)' while evaluate scores
every cell type, ARI named 'Louvain/Leiden' clusters, and only some rows said
which direction is better. catalog.datasets() carried five columns that were
empty in all rows.
"""
import pandas as pd

import multibench as mtb
from multibench.data import catalog


def _desc():
    return mtb.catalog.metrics().set_index("metric")["description"]


def test_isolated_label_metrics_describe_every_cell_type():
    d = _desc()
    for code in ("iASW", "iF1"):
        assert "rare" not in d[code] and "most isolated" not in d[code], d[code]
        assert "Every cell type is treated as an isolated label" in d[code]


def test_every_metric_states_direction_and_the_clustering_used():
    d = _desc()
    assert set(d.index) == {"ARI", "NMI", "ASW", "iASW", "iF1", "cLISI",
                            "ASW_batch", "GC", "iLISI", "kBET"}
    assert all(text.endswith("higher = better.") for text in d), d.to_dict()
    for code in ("ARI", "NMI"):
        assert "Leiden clusters from the optimal-resolution sweep" in d[code]
        assert "Louvain" not in d[code]


def test_datasets_drops_the_always_empty_columns():
    df = catalog.datasets()
    for col in catalog.PAPER_COLUMNS:
        assert col not in df.columns, col
    assert list(df.columns) == ["dataset", "dataset name", "simulated", "category",
                                "has_results"]


def test_datasets_keeps_a_descriptive_column_a_csv_fills(tmp_path):
    pd.DataFrame({"dataset": ["D11", "D28"], "assay": ["CITE-seq", None],
                  "tissue": [None, None]}).to_csv(tmp_path / "dataset.csv", index=False)
    df = catalog.datasets(tmp_path).set_index("dataset")
    assert "assay" in df.columns and "tissue" not in df.columns
    assert df.loc["D11", "assay"] == "CITE-seq"


def test_datasets_notes_say_what_the_subsamples_are():
    doc = " ".join(catalog.datasets.__doc__.split())
    assert "random subsamples of the full datasets" in doc and "cannot be fetched" in doc
    assert "D28s holds 60% of D28's cells" in doc
