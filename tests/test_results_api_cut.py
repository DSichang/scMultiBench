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


# ---------------------------------------------------------------- the shared parser
def test_metric_selection_is_one_vocabulary():
    assert metric_selection(None).codes is None and metric_selection("all").codes is None
    clu = metric_selection("clustering")
    assert clu.family == "clustering" and clu.codes == list(mtb.plot.CLUSTERING_METRICS)
    assert not clu.explicit
    lst = metric_selection(["ari", "kbet"])
    assert lst.codes == ["ARI", "kBET"] and lst.family == "all" and lst.explicit
    assert metric_selection(["gc"]).family == "batch"
    with pytest.raises(ValueError, match=r"^Unknown metrics= token nope\. Pass 'all'"):
        metric_selection(["ARI", "nope"])
    with pytest.raises(ValueError, match="This frame also has MyScore"):
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
    with pytest.raises(ValueError, match="Unknown metrics= token Nope"):
        results.recommend("vertical", long_df=long, metrics=["Nope"])


def test_load_results_positional_selectors_fail_loudly(result_dir):
    # 0.2.x: load_results(category, task, metric_set, dataset, ...)
    with pytest.raises(TypeError):
        results.load_results("diagonal", None, "scib", "D28")
    with pytest.raises(TypeError):
        results.recommend("diagonal", ["rna", "atac"])


# ---------------------------------------------------------------- callers inside the package
def test_recommend_default_scores_every_metric_present(result_dir):
    # c1 (C04): the default of the figures
    r = _quiet(results.recommend, "diagonal")
    assert r.attrs["metrics"] == "all" and "family" not in r.attrs
    assert "task" not in r.attrs
    assert "runtime_tier" in r.columns and r.runtime_tier.notna().any()


