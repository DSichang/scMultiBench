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
    # one dataset: the title names it (0.4.0, one stored table per category)
    assert one.axes[0].get_title(loc="left") == "overall — D1"
    assert "3 datasets" in two.axes[0].get_title(loc="left")


def test_bar_group_selects_metric_family():
    fig = mtb.plot.bar(_long(metrics=("ARI", "NMI")), metrics="clustering")
    assert "clustering" in fig.axes[0].get_title(loc="left")


def test_bar_group_batch_errors_clearly_without_batch_metrics():
    """A single-batch design has no batch metrics; say so instead of drawing nothing."""
    with pytest.raises(ValueError) as e:
        mtb.plot.bar(_long(metrics=("ARI", "NMI")), metrics="batch")
    assert "multi-batch" in str(e.value)


def test_bar_methods_selects_the_bars():
    # methods= replaces the deprecated top= (0.4.0)
    fig = mtb.plot.bar(_long(), methods=["B", "C"])
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["B", "C"]


def test_bar_xlabel_names_dataset_or_formula():
    one = mtb.plot.bar(_long(datasets=("D1",)))
    assert one.axes[0].get_xlabel() == "overall score, scaled 0-1 (D1)"
    two = mtb.plot.bar(_long())
    assert "mean of per-dataset overall" in two.axes[0].get_xlabel()
    # the deprecated overall= (0.4.0) keeps its effect
    with pytest.warns(FutureWarning, match=r"plot\.bar\(overall=\) is deprecated"):
        rk = mtb.plot.bar(_long(), overall="rank")
    assert "rank of mean ranks" in rk.axes[0].get_xlabel()
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="overall must be one of"):
        mtb.plot.bar(_long(), overall="median")


def test_bar_doc_names_the_recommend_order():
    # 0.4.0: the overall= text left the docstring with the keyword; the bars
    # are in recommend's order
    import inspect
    assert "overall" not in inspect.signature(mtb.plot.bar).parameters
    norm = " ".join(mtb.plot.bar.__doc__.split())
    assert "in the order of ``mtb.recommend(category, long_df=long_df, metrics=...)``" in norm
