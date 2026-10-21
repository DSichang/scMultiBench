"""The score side's public surface: ``evaluate`` has ONE metric-selection knob
(``metrics=``); ``task=`` / ``family=`` / ``only=`` (0.4) and ``slow_metrics`` /
``column`` / ``metric_set`` (0.3.0) are removed and fail with a TypeError naming
the replacement; ``to_long`` lost ``needs_labels``. One test per removed keyword.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from multibench.eval import evaluate


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


def test_old_positional_order_fails_loudly():
    emb, ct, _ = _toy()
    # 0.2.x: evaluate(output, category, task, labels, ...) - category is now keyword-only
    with pytest.raises(TypeError):
        evaluate(emb, "vertical", labels=ct)
