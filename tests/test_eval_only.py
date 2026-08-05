"""`only=` restricts which metrics are computed, and candidate screening uses it.

Ranking candidate label orderings needs ARI alone, but every candidate used to
run the complete metric suite. On D52 cross that meant 6 permutations x
iF1/cLISI/iLISI/ASW_batch/GC over 23,478 cells - the actual cause of the cross
tutorial timing out. Screening is only safe if it changes neither the winner nor
the reported numbers, which is what these tests pin.
"""

import numpy as np
import pytest

from multibench import workflow as W
from multibench.eval import scib as escib


def _blobs(n_per=70, n_labels=4, dims=6, n_batches=2, seed=1):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 10, size=(n_labels, dims))
    emb, lab, bat = [], [], []
    for i in range(n_labels):
        emb.append(centres[i] + rng.normal(0, 1.0, size=(n_per, dims)))
        lab += [f"t{i}"] * n_per
        bat += [f"b{j % n_batches}" for j in range(n_per)]
    return np.vstack(emb), np.array(lab), np.array(bat)


def test_only_restricts_the_returned_metrics():
    emb, lab, bat = _blobs()
    got = escib.compute(emb, lab, None, bat, group="all", only={"ARI", "ASW"})
    assert set(got.index) == {"ARI", "ASW"}


def test_only_does_not_change_the_metric_value():
    """A screened ARI must equal the ARI from a full evaluation, or screening
    would rank candidates on a different quantity than the one reported."""
    emb, lab, bat = _blobs()
    full = escib.compute(emb, lab, None, bat, group="clustering")
    slim = escib.compute(emb, lab, None, bat, group="clustering", only={"ARI"})
    assert float(slim["Value"]["ARI"]) == pytest.approx(float(full["Value"]["ARI"]),
                                                        abs=1e-12)


def test_only_skips_the_leiden_sweep_when_nothing_needs_it():
    """ASW needs no clustering; asking for it alone must not pay for the sweep."""
    import time
    emb, lab, bat = _blobs(n_per=250, n_labels=6, dims=12)
    t0 = time.time()
    escib.compute(emb, lab, None, bat, group="clustering", only={"ASW"})
    t_slim = time.time() - t0
    t0 = time.time()
    escib.compute(emb, lab, None, bat, group="clustering", only={"ARI"})
    t_ari = time.time() - t0
    assert t_slim < t_ari, f"ASW-only ({t_slim:.2f}s) did not skip the sweep ({t_ari:.2f}s)"


def test_best_order_still_returns_the_full_metric_set():
    """Screening must not leak into the reported result: the winner is scored
    with everything, not just the ARI used for ranking."""
    emb, lab, bat = _blobs()
    rng = np.random.default_rng(0)
    wrong = rng.permutation(lab)
    cands = [(["good.csv"], lab, bat), (["scrambled.csv"], wrong, bat)]
    names, val, spread = W._evaluate_best_order(emb, "vertical", cands)
    assert names == ["good.csv"], f"screening picked {names}"
    assert len(spread) == 2
    assert {"ARI", "NMI", "ASW", "iASW", "iF1"} <= set(val.index), sorted(val.index)


def test_best_order_single_candidate_skips_screening():
    """One candidate has nothing to disambiguate; it must not pay for two passes."""
    emb, lab, bat = _blobs()
    calls = []
    real = W._evaluate

    def counting(*a, **k):
        calls.append(k.get("only"))
        return real(*a, **k)

    W._evaluate = counting
    try:
        names, val, spread = W._evaluate_best_order(emb, "vertical",
                                                    [(["only.csv"], lab, bat)])
    finally:
        W._evaluate = real
    assert len(calls) == 1, f"single candidate triggered {len(calls)} evaluations"
    assert calls[0] is None, "single candidate should go straight to full metrics"
    assert names == ["only.csv"] and len(spread) == 1
