"""Pin the package's isolated-label F1 to scib's own implementation.

The article's iF1 scores one clustering (the clusters searched to the number of
cell types). scib's ``isolated_labels_f1`` scores a precomputed clustering when
``obs[f"{cluster_key}_{resolution}"]`` holds it for every resolution it is
given; this compares the two directly.
"""

import numpy as np
import pytest
import scib.metrics as me

from multibench.eval import scib as escib


def _toy(n_per_label=90, n_labels=5, n_batches=2, dims=8, seed=0):
    """Well-separated blobs, so the clustering is stable and F1 is non-trivial."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 12, size=(n_labels, dims))
    emb, lab, bat = [], [], []
    for i in range(n_labels):
        emb.append(centres[i] + rng.normal(0, 1.0, size=(n_per_label, dims)))
        lab += [f"type{i}"] * n_per_label
        bat += [f"b{j % n_batches}" for j in range(n_per_label)]
    return np.vstack(emb), np.array(lab), np.array(bat)


def _both(emb, lab, bat, clusters):
    iso = len(set(bat)) + 1        # our convention: every label counts as isolated
    a1 = escib._build_adata(emb, lab, clusters, bat)
    a2 = escib._build_adata(emb, lab, clusters, bat)
    a1.obs["cluster_1.0"] = a1.obs["cluster"]          # scib's precomputed layout
    ref = me.isolated_labels_f1(a1, batch_key="batch", label_key="celltype",
                                embed="X_emb", cluster_key="cluster", resolutions=[1.0],
                                iso_threshold=iso, verbose=False)
    fast = escib._isolated_labels_f1(a2, label_key="celltype", batch_key="batch",
                                     cluster_key="cluster", iso_threshold=iso)
    return float(ref), float(fast)


def _noisy(lab, seed=1, share=0.2):
    """The labels with ``share`` of them moved to a random other label."""
    rng = np.random.default_rng(seed)
    out = lab.copy()
    pick = rng.random(len(lab)) < share
    out[pick] = rng.choice(np.unique(lab), pick.sum())
    return out


def test_fast_isolated_f1_matches_scib():
    emb, lab, bat = _toy()
    ref, fast = _both(emb, lab, bat, _noisy(lab))
    assert np.isfinite(ref) and np.isfinite(fast)
    assert fast == pytest.approx(ref, abs=1e-9), f"scib={ref!r} fast={fast!r}"


def test_fast_isolated_f1_matches_scib_uneven_labels():
    """Uneven label sizes and clusters that merge two labels."""
    rng = np.random.default_rng(3)
    dims = 8
    sizes = [140, 90, 45, 20, 12]
    centres = rng.normal(0, 12, size=(len(sizes), dims))
    emb, lab, bat = [], [], []
    for i, n in enumerate(sizes):
        emb.append(centres[i] + rng.normal(0, 1.0, size=(n, dims)))
        lab += [f"type{i}"] * n
        bat += [f"b{j % 3}" for j in range(n)]
    lab = np.array(lab)
    merged = np.where(lab == "type4", "type3", lab)
    ref, fast = _both(np.vstack(emb), lab, np.array(bat), merged)
    assert fast == pytest.approx(ref, abs=1e-9), f"scib={ref!r} fast={fast!r}"


def test_fast_isolated_f1_leaves_no_scratch_columns():
    emb, lab, bat = _toy(n_per_label=40, n_labels=3)
    adata = escib._build_adata(emb, lab, lab, bat)
    before = set(adata.obs.columns)
    escib._isolated_labels_f1(adata, label_key="celltype", batch_key="batch",
                              cluster_key="cluster", iso_threshold=len(set(bat)) + 1)
    assert set(adata.obs.columns) == before
