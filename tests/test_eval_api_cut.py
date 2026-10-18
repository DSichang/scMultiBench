"""The score side's public surface: ``evaluate`` has ONE metric-selection knob
(``metrics=``); ``task=`` / ``family=`` / ``only=`` (0.4) and ``slow_metrics`` /
``column`` / ``metric_set`` (0.3.0) are removed and fail with a TypeError naming
the replacement; ``to_long`` lost ``needs_labels``. One test per removed keyword.
"""
import inspect

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


# ---------------------------------------------------------------- signature
def test_evaluate_signature_is_the_contract():
    params = inspect.signature(evaluate).parameters
    # 0.4.0: dataset= and method= fill the long table; clustering=, label_order=
    # and missing_labels= are deprecated and left out of the signature
    assert list(params) == ["embedding", "labels", "dataset", "method", "category",
                            "batch", "metrics", "obsm", "verbose"]
    assert params["labels"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in list(params)[2:]:
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert params["metrics"].default is None and params["verbose"].default is True
    assert params["dataset"].default is None and params["method"].default == "embedding"
    assert "metrics" in evaluate.__doc__ and "kBET" in evaluate.__doc__


def test_to_long_signature_is_the_contract():
    params = inspect.signature(to_long).parameters
    # clustering= and source= are deprecated since 0.4.0 (X36)
    assert list(params) == ["value_df", "method", "dataset", "category"]
    assert all(params[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(params)[1:])
    assert params["dataset"].default is None and params["category"].default is None


# ---------------------------------------------------------------- removed keywords
@pytest.mark.parametrize("name, value, message", [
    ("task", "batch", "evaluate(task=) was removed in 0.4. Use metrics=."),
    ("family", "batch", "evaluate(family=) was removed in 0.4. Use metrics=."),
    ("only", {"ASW", "GC"}, "evaluate(only=) was removed in 0.4. Use metrics=."),
    ("slow_metrics", True, "evaluate(slow_metrics=) was removed in 0.3.0; pass "
                           "metrics=[...] without cLISI and iLISI."),
    ("column", "celltype", "evaluate(column=) was removed in 0.3.0; pass the column itself "
                           "as labels= or batch=."),
    ("metric_set", "scib", "evaluate(metric_set=) was removed in 0.3.0; only the scIB "
                           "metric set exists, so drop it."),
])
def test_a_removed_keyword_is_a_type_error_naming_the_replacement(name, value, message):
    emb, ct, bat = _toy()
    with pytest.raises(TypeError) as e:
        evaluate(emb, labels=ct, batch=bat, **{name: value})
    assert str(e.value) == message


def test_removed_to_long_needs_labels_is_a_type_error():
    w = pd.DataFrame({"Value": [0.5]}, index=["ARI"])
    with pytest.raises(TypeError, match="needs_labels"):
        to_long(w, method="M", dataset="D", category="vertical", needs_labels=True)


def test_old_positional_order_fails_loudly():
    emb, ct, _ = _toy()
    # 0.2.x: evaluate(output, category, task, labels, ...) - category is now keyword-only
    with pytest.raises(TypeError):
        evaluate(emb, "vertical", labels=ct)
