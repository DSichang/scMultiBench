"""0.3.0 public-surface cut, score side: ``evaluate`` has ONE metric-selection
knob (``metrics=``); ``task=`` / ``family=`` / ``only=`` are deprecated aliases
that warn and map onto it, ``slow_metrics`` / ``column`` / ``metric_set`` are
gone and fail with a TypeError naming the replacement; ``to_long`` lost
``needs_labels``. One test per alias, one per removed keyword.
"""
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench.eval import evaluate, to_long


def _toy(n_per=45, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, size=(n_per, 5)) + np.array([8, 0, 0, 0, 0])
    b = rng.normal(0, 1, size=(n_per, 5)) - np.array([8, 0, 0, 0, 0])
    emb = np.vstack([a, b])
    ct = np.array(["B"] * n_per + ["T"] * n_per)
    bat = np.array(["s1", "s2"] * n_per)
    return emb, ct, bat


def _quiet(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return evaluate(**kw)


# ---------------------------------------------------------------- signature
def test_evaluate_signature_is_the_contract():
    params = inspect.signature(evaluate).parameters
    assert list(params) == ["output", "labels", "category", "batch", "metrics",
                            "clustering", "obsm", "label_order", "verbose"]
    assert params["labels"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("category", "batch", "metrics", "clustering", "obsm", "label_order", "verbose"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert params["metrics"].default is None and params["verbose"].default is True
    assert "metrics" in evaluate.__doc__ and "slow_metrics" in evaluate.__doc__


def test_to_long_signature_is_the_contract():
    params = inspect.signature(to_long).parameters
    assert list(params) == ["value_df", "method", "dataset", "category", "clustering", "source"]
    assert all(params[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(params)[1:])
    assert params["dataset"].default is None and params["category"].default is None


# ---------------------------------------------------------------- deprecated aliases
def test_deprecated_task_maps_to_metrics_token():
    emb, ct, bat = _toy()
    new = evaluate(emb, labels=ct, batch=bat, metrics="batch")
    with pytest.warns(DeprecationWarning, match=r"evaluate\(task=\.\.\.\) is deprecated since 0.3.0.*use metrics='batch'"):
        old = evaluate(emb, labels=ct, batch=bat, task="batch")
    pd.testing.assert_frame_equal(old, new)
    # the paper's alias of the clustering family maps too
    with pytest.warns(DeprecationWarning, match="use metrics='clustering'"):
        dr = evaluate(emb, labels=ct, clustering=ct, task="dimension_reduction")
    assert set(dr.index) == set(evaluate(emb, labels=ct, clustering=ct, metrics="clustering").index)


def test_deprecated_family_maps_to_metrics_token():
    emb, ct, bat = _toy()
    new = evaluate(emb, labels=ct, batch=bat, metrics="batch")
    with pytest.warns(DeprecationWarning, match=r"evaluate\(family=\.\.\.\) is deprecated.*use metrics='batch'"):
        old = evaluate(emb, labels=ct, batch=bat, family="batch")
    pd.testing.assert_frame_equal(old, new)


def test_deprecated_only_maps_to_metrics_list():
    emb, ct, bat = _toy()
    new = evaluate(emb, labels=ct, batch=bat, metrics=["ASW", "GC"])
    with pytest.warns(DeprecationWarning, match=r"evaluate\(only=\.\.\.\) is deprecated.*use metrics=\['ASW', 'GC'\]"):
        old = evaluate(emb, labels=ct, batch=bat, only={"ASW", "GC"})
    pd.testing.assert_frame_equal(old.sort_index(), new.sort_index())
    # a bare string was always a mistake (set('ARI') == {'A', 'R', 'I'})
    with pytest.raises(TypeError, match="only= must be a collection of metric names"):
        _quiet(output=emb, labels=ct, only="ARI")
    # 0.2.x refused a code outside the family; the mapping still does
    with pytest.raises(ValueError, match="GC not in the 'clustering' family: pass metrics=\\['GC'\\] alone"):
        _quiet(output=emb, labels=ct, batch=bat, task="clustering", only={"GC"})
    # the alias and the new knob cannot be mixed
    with pytest.raises(TypeError, match="metrics= together with the deprecated"):
        _quiet(output=emb, labels=ct, only={"ASW"}, metrics=["ASW"])


# ---------------------------------------------------------------- removed keywords
def test_removed_slow_metrics_is_a_type_error_naming_metrics():
    emb, ct, bat = _toy()
    with pytest.raises(TypeError, match=r"slow_metrics=, removed in 0.3.0: pass metrics=\[\.\.\.\] without cLISI/iLISI"):
        evaluate(emb, labels=ct, batch=bat, slow_metrics=True)


def test_removed_column_is_a_type_error_naming_the_series():
    emb, ct, bat = _toy()
    with pytest.raises(TypeError, match="column=, removed in 0.3.0: pass the Series/column itself"):
        evaluate(emb, labels=pd.DataFrame({"celltype": ct, "b": bat}), column="celltype")


def test_removed_metric_set_is_a_type_error():
    emb, ct, _ = _toy()
    with pytest.raises(TypeError, match="metric_set=, removed in 0.3.0: only the scIB metric set exists"):
        evaluate(emb, labels=ct, metric_set="scib")


def test_removed_to_long_needs_labels_is_a_type_error():
    w = pd.DataFrame({"Value": [0.5]}, index=["ARI"])
    with pytest.raises(TypeError, match="needs_labels"):
        to_long(w, method="M", dataset="D", category="vertical", needs_labels=True)


def test_old_positional_order_fails_loudly():
    emb, ct, _ = _toy()
    # 0.2.x: evaluate(output, category, task, labels, ...) - category is now keyword-only
    with pytest.raises(TypeError):
        evaluate(emb, "vertical", labels=ct)
