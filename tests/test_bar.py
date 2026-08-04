"""Tests for the cross-dataset summary bar chart."""
import pandas as pd
import pytest
import multibench as mtb


def _long(datasets=("D1", "D2"), metrics=("ARI", "NMI")):
    rows = []
    for ds in datasets:
        for i, m in enumerate(["A", "B", "C"]):
            for k, met in enumerate(metrics):
                rows.append({"metric": met, "value": 0.1 * (i + 1) + 0.05 * k,
                             "method": m, "dataset": ds, "category": "vertical"})
    return pd.DataFrame(rows)


def test_bar_returns_a_figure_and_ranks_methods():
    fig = mtb.plot.bar(_long())
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()][-1] == "C"   # best on top


def test_bar_aggregates_across_datasets():
    one = mtb.plot.bar(_long(datasets=("D1",)))
    two = mtb.plot.bar(_long(datasets=("D1", "D2", "D3")))
    assert "1 dataset" in one.axes[0].get_title()
    assert "3 datasets" in two.axes[0].get_title()


def test_bar_group_selects_metric_family():
    fig = mtb.plot.bar(_long(metrics=("ARI", "NMI")), group="clustering")
    assert "clustering" in fig.axes[0].get_title()


def test_bar_group_batch_errors_clearly_without_batch_metrics():
    """A single-batch design has no batch metrics; say so instead of drawing nothing."""
    with pytest.raises(ValueError) as e:
        mtb.plot.bar(_long(metrics=("ARI", "NMI")), group="batch")
    assert "multi-batch" in str(e.value)


def test_bar_top_n():
    fig = mtb.plot.bar(_long(), top=2)
    assert len(fig.axes[0].get_yticklabels()) == 2
