"""The compare side's public surface: ``load_results`` / ``recommend`` share
``evaluate``'s ``metrics=`` vocabulary; ``method=`` / ``metric=`` / ``task=`` /
``family=`` (0.4) and ``metric_set`` (0.3.0) are removed, with a TypeError
naming the replacement; ``available_datasets`` lost ``metric_set`` and
``clustering``. The 0.4 deprecated keywords (``source``, ``result_path``,
``clustering``) leave the signatures and keep working.
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
    assert list(lr) == ["category", "dataset", "methods", "metrics"]
    assert all(lr[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(lr)[1:])
    ad = inspect.signature(results.available_datasets).parameters
    assert list(ad) == ["category", "source", "result_path"]
    rc = inspect.signature(results.results_coverage).parameters
    assert list(rc) == ["category"]
    re_ = inspect.signature(results.recommend).parameters
    # atac= is deprecated since 0.4.0 (X44): modalities=["atac_peak"] does the same;
    # min_methods= is deprecated in 0.4.0 (it had no effect)
    assert list(re_) == ["category", "modalities", "methods", "metrics", "long_df"]
    assert all(re_[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(re_)[1:])
    assert "Parameters" in results.load_results.__doc__
    assert "Parameters" in results.recommend.__doc__


# ---------------------------------------------------------------- the shared parser
def test_metric_selection_is_one_vocabulary():
    assert metric_selection(None).codes is None and metric_selection("all").codes is None
    clu = metric_selection("clustering")
    assert clu.family == "clustering" and clu.codes == list(mtb.plot.CLUSTERING_METRICS)
    assert not clu.explicit
    lst = metric_selection(["ari", "kbet"])
    assert lst.codes == ["ARI", "kBET"] and lst.family == "all" and lst.explicit
    assert metric_selection(["gc"]).family == "batch"
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['nope'\]; valid: clustering, batch, classification, all, or the codes \['ARI'"):
        metric_selection(["ARI", "nope"])
    with pytest.raises(ValueError, match="present in this frame: \\['MyScore'\\]"):
        metric_selection(["MyScore", "nope"], extra=["MyScore"])
    assert metric_selection(["MyScore"], extra=["MyScore"]).codes == ["MyScore"]
    with pytest.raises(ValueError, match="metrics=\\[\\] selects nothing"):
        metric_selection([])
    with pytest.raises(ValueError, match="names the same metric twice"):
        metric_selection(["ARI", "ari"])
    # 0.4.0: one code is accepted alone, and families go inside a list
    one = metric_selection("ARI")
    assert one.codes == ["ARI"] and one.explicit
    assert metric_selection(["clustering", "ARI"]).codes == clu.codes
    mixed = metric_selection(["clustering", "kBET"])
    assert mixed.codes == clu.codes + ["kBET"] and mixed.family == "all"
    with pytest.raises(TypeError, match=r"^metrics= takes None, a family \('all', "
                                        r"'clustering', 'batch', 'classification'\), a "
                                        r"metric code, or a list of them\. It got an int\.$"):
        metric_selection(3)


def test_load_results_and_evaluate_share_the_vocabulary(result_dir):
    df = results.load_results("diagonal", dataset="D28", metrics=["ari", "GC"])
    assert set(df.metric) == {"ARI", "GC"}
    # a user's own metric name in a long file is selectable
    long = pd.DataFrame({"metric": ["ARI", "MyScore"], "value": [0.5, 0.7],
                         "method": ["M", "M"], "dataset": ["D", "D"],
                         "category": ["vertical", "vertical"]})
    with pytest.raises(ValueError, match="unknown metric"):
        results.recommend("vertical", long_df=long, metrics=["Nope"])


# ---------------------------------------------------------------- removed keywords
@pytest.mark.parametrize("fn, name, value, message", [
    ("load_results", "method", "scBridge", "load_results(method=) was removed in 0.4. "
                                           "Use methods=."),
    ("load_results", "metric", ["ARI"], "load_results(metric=) was removed in 0.4. "
                                        "Use metrics=."),
    ("load_results", "task", "batch", "load_results(task=) was removed in 0.4. Use metrics=."),
    ("load_results", "family", "batch", "load_results(family=) was removed in 0.4. "
                                        "Use metrics=."),
    ("load_results", "metric_set", "scib", "load_results(metric_set=) was removed in 0.3.0; "
                                           "only the scIB metric set exists, so drop it."),
    ("recommend", "task", "batch", "recommend(task=) was removed in 0.4. Use metrics=."),
    ("recommend", "family", "batch", "recommend(family=) was removed in 0.4. Use metrics=."),
])
def test_a_removed_keyword_is_a_type_error_naming_the_replacement(fn, name, value, message):
    with pytest.raises(TypeError) as e:
        getattr(results, fn)("diagonal", **{name: value})
    assert str(e.value) == message


def test_removed_available_datasets_metric_set_and_clustering_are_type_errors(result_dir):
    with pytest.raises(TypeError, match="unexpected keyword argument 'metric_set'"):
        results.available_datasets("vertical", metric_set="scib")
    with pytest.raises(TypeError, match="unexpected keyword argument 'clustering'"):
        results.available_datasets("vertical", clustering="louvain")


def test_load_results_positional_selectors_fail_loudly(result_dir):
    # 0.2.x: load_results(category, task, metric_set, dataset, ...)
    with pytest.raises(TypeError):
        results.load_results("diagonal", None, "scib", "D28")
    with pytest.raises(TypeError):
        results.recommend("diagonal", ["rna", "atac"])


# ---------------------------------------------------------------- callers inside the package
def test_recommend_default_scores_the_clustering_family(result_dir):
    r = _quiet(results.recommend, "diagonal")
    assert r.attrs["metrics"] == "clustering" and r.attrs["family"] == "clustering"
    assert "task" not in r.attrs
    assert "runtime_tier" in r.columns and r.runtime_tier.notna().any()


def test_results_coverage_and_degenerate_detection_still_run(result_dir):
    cov = results.results_coverage("diagonal")
    # category, dataset, method: the constant clustering and source left (X42)
    assert list(cov.columns) == ["category", "dataset", "method"]
    d28 = results.load_results("diagonal", dataset="D28")
    bad = d28.iloc[:1].assign(metric="ARI", value=0.001, method="Conos")
    with pytest.warns(results.DegenerateRerunWarning, match="Conos/D28"):
        results._warn_degenerate(bad, result_dir)
