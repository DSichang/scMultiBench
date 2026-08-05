"""The plotting layer's metric families must match what the eval layer computes.

These two lived apart and drifted: plot/bar.py filed iASW and iF1 under batch
correction while eval.scib.compute() emitted them for group="clustering". The
same number was therefore labelled a different family depending on which module
you asked, and a single-batch dataset like D11 - which legitimately has iASW/iF1
and no batches whatsoever - rendered as if it carried batch-correction results.

Pinning them to each other is what stops that drifting again.
"""

import numpy as np

from multibench.eval import scib as escib
from multibench.plot import BATCH_METRICS, CLUSTERING_METRICS


def _blobs(n_per=60, n_labels=4, dims=6, n_batches=2, seed=5):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 10, size=(n_labels, dims))
    emb, lab, bat = [], [], []
    for i in range(n_labels):
        emb.append(centres[i] + rng.normal(0, 1.0, size=(n_per, dims)))
        lab += [f"t{i}"] * n_per
        bat += [f"b{j % n_batches}" for j in range(n_per)]
    return np.vstack(emb), np.array(lab), np.array(bat)


def test_clustering_family_matches_what_compute_emits():
    emb, lab, bat = _blobs()
    got = escib.compute(emb, lab, None, bat, group="clustering")
    assert set(got.index) == set(CLUSTERING_METRICS), (
        f"compute emits {sorted(got.index)}, "
        f"CLUSTERING_METRICS says {sorted(CLUSTERING_METRICS)}")


def test_batch_family_matches_what_compute_emits():
    emb, lab, bat = _blobs()
    got = escib.compute(emb, lab, None, bat, group="batch")
    # kBET is opt-in (slow_metrics=True), so it is declared but not emitted here
    assert set(got.index) == set(BATCH_METRICS) - {"kBET"}, (
        f"compute emits {sorted(got.index)}, "
        f"BATCH_METRICS minus kBET says {sorted(set(BATCH_METRICS) - {'kBET'})}")


def test_families_are_disjoint():
    overlap = set(CLUSTERING_METRICS) & set(BATCH_METRICS)
    assert not overlap, f"a metric cannot be in both families: {overlap}"


def test_isolated_label_scores_are_bio_conservation_not_batch():
    """Explicit: scib files isolated-label scores under bio conservation, and a
    single-batch dataset can have them while having no batch metrics at all."""
    for m in ("iASW", "iF1"):
        assert m in CLUSTERING_METRICS, f"{m} should be a clustering metric"
        assert m not in BATCH_METRICS, f"{m} should not be a batch metric"
