"""0.3.0 public-surface cut, compare side: ``load_results`` / ``recommend``
share ``evaluate``'s ``metrics=`` vocabulary; ``method=`` / ``metric=`` /
``task=`` / ``family=`` are deprecated aliases that warn and map;
``metric_set`` is gone (TypeError); ``available_datasets`` lost ``metric_set``
and ``clustering``. One test per alias, one per removed keyword.
"""
import inspect
import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.data import results
from multibench.data.catalog import metric_selection


def _quiet(fn, *a, **k):
    """Silence the coverage / degenerate-row UserWarnings only - a
    DeprecationWarning must still reach the enclosing ``pytest.warns``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


# ---------------------------------------------------------------- signatures
def test_compare_signatures_are_the_contract():
    lr = inspect.signature(results.load_results).parameters
    assert list(lr) == ["category", "dataset", "methods", "metrics", "clustering",
                        "source", "result_path"]
    assert all(lr[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(lr)[1:])
    ad = inspect.signature(results.available_datasets).parameters
    assert list(ad) == ["category", "source", "result_path"]
    rc = inspect.signature(results.results_coverage).parameters
    assert list(rc) == ["category", "source", "result_path"]
    re_ = inspect.signature(results.recommend).parameters
    assert list(re_) == ["category", "modalities", "methods", "metrics", "long_df",
                         "min_methods", "source", "result_path"]
    assert all(re_[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(re_)[1:])
    for fn in (results.load_results, results.available_datasets):
        assert "metric_set" in fn.__doc__ and "Parameters" in fn.__doc__
    assert "task=" in results.recommend.__doc__ and "Parameters" in results.recommend.__doc__


# ---------------------------------------------------------------- the shared parser
def test_metric_selection_is_one_vocabulary():
    assert metric_selection(None).codes is None and metric_selection("all").codes is None
    clu = metric_selection("clustering")
    assert clu.family == "clustering" and clu.codes == list(mtb.plot.CLUSTERING_METRICS)
    assert not clu.explicit
    lst = metric_selection(["ari", "kbet"])
    assert lst.codes == ["ARI", "kBET"] and lst.family == "all" and lst.explicit
    assert metric_selection(["gc"]).family == "batch"
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['nope'\]; valid codes: \['ARI'"):
        metric_selection(["ARI", "nope"])
    with pytest.raises(ValueError, match="present in this frame: \\['MyScore'\\]"):
        metric_selection(["MyScore", "nope"], extra=["MyScore"])
    assert metric_selection(["MyScore"], extra=["MyScore"]).codes == ["MyScore"]
    with pytest.raises(ValueError, match="metrics=\\[\\] selects nothing"):
        metric_selection([])
    with pytest.raises(ValueError, match="names the same metric twice"):
        metric_selection(["ARI", "ari"])
    with pytest.raises(ValueError, match="a single code goes in a list: metrics=\\['ARI'\\]"):
        metric_selection("ARI")
    with pytest.raises(TypeError, match="must be None, a family token"):
        metric_selection(3)


def test_load_results_and_evaluate_share_the_vocabulary(result_dir):
    df = results.load_results("diagonal", dataset="D28", metrics=["ari", "GC"],
                              result_path=result_dir)
    assert set(df.metric) == {"ARI", "GC"}
    # a user's own metric name in a long file is selectable
    long = pd.DataFrame({"metric": ["ARI", "MyScore"], "value": [0.5, 0.7],
                         "method": ["M", "M"], "dataset": ["D", "D"],
                         "category": ["vertical", "vertical"]})
    with pytest.raises(ValueError, match="unknown metric"):
        results.recommend("vertical", long_df=long, metrics=["Nope"])


# ---------------------------------------------------------------- deprecated aliases
def test_deprecated_method_maps_to_methods(result_dir):
    new = results.load_results("diagonal", dataset="D27", methods=["scBridge"],
                               result_path=result_dir)
    with pytest.warns(DeprecationWarning, match=r"load_results\(method=\.\.\.\) is deprecated since 0.3.0.*use methods="):
        old = results.load_results("diagonal", dataset="D27", method="scBridge",
                                   result_path=result_dir)
    pd.testing.assert_frame_equal(old, new)
    with pytest.raises(ValueError, match="either method= or methods="):
        _quiet(results.load_results, "diagonal", dataset="D27", method="scBridge",
               methods=["scBridge"], result_path=result_dir)


def test_deprecated_metric_maps_to_metrics(result_dir):
    new = results.load_results("diagonal", dataset="D27", metrics=["ARI", "NMI"],
                               result_path=result_dir)
    with pytest.warns(DeprecationWarning, match=r"load_results\(metric=\.\.\.\) is deprecated.*use metrics=\['ARI', 'NMI'\]"):
        old = results.load_results("diagonal", dataset="D27", metric=["ARI", "NMI"],
                                   result_path=result_dir)
    pd.testing.assert_frame_equal(old, new)
    # a scalar was accepted too
    with pytest.warns(DeprecationWarning):
        one = results.load_results("diagonal", dataset="D27", metric="ari", result_path=result_dir)
    assert set(one.metric) == {"ARI"}


def test_deprecated_load_results_task_maps_to_metrics_token(result_dir):
    new = results.load_results("diagonal", dataset="D28", metrics="batch", result_path=result_dir)
    with pytest.warns(DeprecationWarning, match=r"load_results\(task=\.\.\.\) is deprecated.*use metrics='batch'"):
        old = results.load_results("diagonal", dataset="D28", task="batch", result_path=result_dir)
    pd.testing.assert_frame_equal(old, new)
    with pytest.raises(ValueError, match="pass either family= or task="):
        _quiet(results.load_results, "diagonal", dataset="D28", task="batch",
               family="clustering", result_path=result_dir)
    with pytest.raises(TypeError, match="metrics= together with the deprecated"):
        _quiet(results.load_results, "diagonal", dataset="D28", task="batch",
               metrics="batch", result_path=result_dir)


def test_deprecated_load_results_family_maps_to_metrics_token(result_dir):
    new = results.load_results("diagonal", dataset="D28", metrics="clustering",
                               result_path=result_dir)
    with pytest.warns(DeprecationWarning, match=r"load_results\(family=\.\.\.\) is deprecated.*use metrics='clustering'"):
        old = results.load_results("diagonal", dataset="D28", family="clustering",
                                   result_path=result_dir)
    pd.testing.assert_frame_equal(old, new)


def test_deprecated_recommend_task_maps_to_metrics_token(result_dir):
    new = _quiet(results.recommend, "diagonal", metrics="batch", source="rerun",
                 result_path=result_dir)
    with pytest.warns(DeprecationWarning, match=r"recommend\(task=\.\.\.\) is deprecated.*use metrics='batch'"):
        old = _quiet(results.recommend, "diagonal", task="batch", source="rerun",
                     result_path=result_dir)
    pd.testing.assert_frame_equal(old, new)
    # task=None meant "every metric"
    every = _quiet(results.recommend, "diagonal", metrics="all", source="rerun",
                   result_path=result_dir)
    with pytest.warns(DeprecationWarning, match="use metrics='all'"):
        old_none = _quiet(results.recommend, "diagonal", task=None, source="rerun",
                          result_path=result_dir)
    pd.testing.assert_frame_equal(old_none, every)


def test_deprecated_recommend_family_maps_to_metrics_token(result_dir):
    new = _quiet(results.recommend, "diagonal", metrics="batch", source="rerun",
                 result_path=result_dir)
    with pytest.warns(DeprecationWarning, match=r"recommend\(family=\.\.\.\) is deprecated.*use metrics='batch'"):
        old = _quiet(results.recommend, "diagonal", family="batch", source="rerun",
                     result_path=result_dir)
    pd.testing.assert_frame_equal(old, new)
    # family won over task in 0.2.x; both warn
    with pytest.warns(DeprecationWarning):
        both = _quiet(results.recommend, "diagonal", task="clustering", family="batch",
                      source="rerun", result_path=result_dir)
    pd.testing.assert_frame_equal(both, new)


# ---------------------------------------------------------------- removed keywords
def test_removed_load_results_metric_set_is_a_type_error(result_dir):
    with pytest.raises(TypeError, match="metric_set=, removed in 0.3.0: only the scIB metric set exists"):
        results.load_results("diagonal", metric_set="scib", result_path=result_dir)


def test_removed_available_datasets_metric_set_and_clustering_are_type_errors(result_dir):
    with pytest.raises(TypeError, match="unexpected keyword argument 'metric_set'"):
        results.available_datasets("vertical", metric_set="scib", result_path=result_dir)
    with pytest.raises(TypeError, match="unexpected keyword argument 'clustering'"):
        results.available_datasets("vertical", clustering="louvain", result_path=result_dir)


def test_load_results_positional_selectors_fail_loudly(result_dir):
    # 0.2.x: load_results(category, task, metric_set, dataset, ...)
    with pytest.raises(TypeError):
        results.load_results("diagonal", None, "scib", "D28", result_path=result_dir)
    with pytest.raises(TypeError):
        results.recommend("diagonal", ["rna", "atac"], result_path=result_dir)


# ---------------------------------------------------------------- callers inside the package
def test_recommend_default_scores_the_clustering_family(result_dir):
    r = _quiet(results.recommend, "diagonal", result_path=result_dir)
    assert r.attrs["metrics"] == "clustering" and r.attrs["family"] == "clustering"
    assert "task" not in r.attrs
    assert "runtime_tier" in r.columns and r.runtime_tier.notna().any()


def test_results_coverage_and_degenerate_detection_still_run(result_dir):
    cov = results.results_coverage("diagonal", result_path=result_dir)
    assert set(cov.source) == {"published", "rerun"}
    with pytest.warns(results.DegenerateRerunWarning, match="Conos/D28"):
        results.load_results("diagonal", dataset="D28", source="rerun", result_path=result_dir)
