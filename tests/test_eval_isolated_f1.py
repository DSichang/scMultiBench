"""Pin our fast isolated-label F1 to scib's own implementation.

We replaced ``scib.metrics.isolated_labels_f1`` with an equivalent that clusters
each resolution once instead of once per label (scib repeats the identical
Leiden sweep for every isolated label). That is only acceptable if it returns
the SAME number, so this compares the two directly rather than trusting the
argument that they must agree.
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


def _both(emb, lab, bat):
    iso = len(set(bat)) + 1        # our convention: every label counts as isolated
    a1 = escib._build_adata(emb, lab, lab, bat)
    a2 = escib._build_adata(emb, lab, lab, bat)
    import scanpy as sc
    sc.pp.neighbors(a1, use_rep="X_emb")
    ref = me.isolated_labels_f1(a1, batch_key="batch", label_key="celltype",
                                embed="X_emb", iso_threshold=iso, verbose=False)
    fast = escib._isolated_labels_f1(a2, label_key="celltype", batch_key="batch",
                                     embed="X_emb", iso_threshold=iso)
    return float(ref), float(fast)


def test_fast_isolated_f1_matches_scib():
    emb, lab, bat = _toy()
    ref, fast = _both(emb, lab, bat)
    assert np.isfinite(ref) and np.isfinite(fast)
    assert fast == pytest.approx(ref, abs=1e-9), f"scib={ref!r} fast={fast!r}"


def test_fast_isolated_f1_matches_scib_uneven_labels():
    """Uneven label sizes: the per-label argmax is likelier to pick different
    resolutions for different labels, which is exactly what must be preserved."""
    rng = np.random.default_rng(3)
    dims = 8
    sizes = [140, 90, 45, 20, 12]
    centres = rng.normal(0, 12, size=(len(sizes), dims))
    emb, lab, bat = [], [], []
    for i, n in enumerate(sizes):
        emb.append(centres[i] + rng.normal(0, 1.0, size=(n, dims)))
        lab += [f"type{i}"] * n
        bat += [f"b{j % 3}" for j in range(n)]
    ref, fast = _both(np.vstack(emb), np.array(lab), np.array(bat))
    assert fast == pytest.approx(ref, abs=1e-9), f"scib={ref!r} fast={fast!r}"


def test_fast_isolated_f1_leaves_no_scratch_columns():
    """The helper adds one obs column per resolution; it must clean them up."""
    emb, lab, bat = _toy(n_per_label=40, n_labels=3)
    adata = escib._build_adata(emb, lab, lab, bat)
    before = set(adata.obs.columns)
    escib._isolated_labels_f1(adata, label_key="celltype", batch_key="batch",
                              embed="X_emb", iso_threshold=len(set(bat)) + 1)
    assert set(adata.obs.columns) == before
