"""Summary bar chart: one bar per method, aggregated ACROSS datasets.

The bubble chart answers "how did each method do on THIS dataset". This answers
"how does each method do overall", which is the summary the benchmark reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import compute_overall, minmax, rank_max

# scIB metric families, so a summary can be split the way the benchmark reports it
CLUSTERING_METRICS = ["ARI", "NMI", "ASW", "cLISI"]
BATCH_METRICS = ["iASW", "iF1", "ASW_batch", "GC", "iLISI", "kBET"]


def _score_per_dataset(long_df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Per (method, dataset) overall score, using the benchmark's rank rule."""
    d = long_df.copy()
    if metrics:
        d = d[d["metric"].isin(metrics)]
    if d.empty:
        raise ValueError("no rows left after filtering to metrics=%r" % (metrics,))
    out = {}
    for ds, g in d.groupby("dataset"):
        mat = g.pivot_table(index="method", columns="metric", values="value", aggfunc="mean")
        mat = mat.dropna(axis=1, how="all")
        if mat.empty:
            continue
        out[ds] = compute_overall(mat)
    if not out:
        raise ValueError("no dataset had usable metrics")
    return pd.DataFrame(out)


def bar(long_df: pd.DataFrame, *, metrics=None, group: str | None = None,
        top: int | None = None, title: str | None = None, cmap: str = "Blues",
        show_datasets: bool = True, save: str | None = None):
    """Bar chart of each method's overall score, averaged across datasets.

    Parameters
    ----------
    long_df : tidy frame (``metric, value, method, dataset, category``) - the same
        frame :meth:`BatchResult.long` produces. Concatenate several datasets'
        frames to summarise across them.
    metrics : restrict to these metric names. Ignored when ``group`` is given.
    group : ``"clustering"`` or ``"batch"`` - shorthand for that metric family, so
        you can produce the benchmark's two summary panels.
    top : keep only the N best methods.
    show_datasets : overlay one dot per dataset behind each bar, so a method that
        is uniformly good is distinguishable from one that averages well by
        winning on a single dataset.

    Returns a matplotlib ``Figure``. Bars are the MEAN across datasets of a
    per-dataset overall score, where that score is the benchmark's own rule
    (min-max scaled mean of per-metric max-ranks). Because it is rank-based, a
    score is only meaningful relative to the other methods in the same figure.
    """
    import matplotlib.pyplot as plt

    if group:
        fam = {"clustering": CLUSTERING_METRICS, "batch": BATCH_METRICS}
        if group not in fam:
            raise ValueError(f"group must be 'clustering' or 'batch', got {group!r}")
        metrics = fam[group]
        avail = set(long_df["metric"].unique())
        if not (set(metrics) & avail):
            raise ValueError(
                f"no {group!r} metrics present. Found {sorted(avail)}. "
                "Batch metrics need a multi-batch dataset - a single-batch design "
                "has none to compute.")

    per_ds = _score_per_dataset(long_df, metrics)
    mean = per_ds.mean(axis=1).sort_values(ascending=True)   # ascending -> best on top
    if top:
        mean = mean.tail(top)
    per_ds = per_ds.loc[mean.index]

    n = len(mean)
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.34 * n + 1.1)))
    colors = plt.get_cmap(cmap)(0.35 + 0.55 * minmax(mean.to_numpy()))
    ax.barh(range(n), mean.to_numpy(), color=colors, edgecolor="#333", linewidth=0.6, zorder=2)

    if show_datasets and per_ds.shape[1] > 1:
        for i, m in enumerate(mean.index):
            vals = per_ds.loc[m].dropna().to_numpy()
            ax.scatter(vals, np.full(len(vals), i), s=16, facecolor="white",
                       edgecolor="#444", linewidth=0.7, zorder=3)

    ax.set_yticks(range(n)); ax.set_yticklabels(mean.index, fontsize=9)
    ax.set_xlabel("overall score (mean across datasets)", fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", color="#e8e8e8", zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    nds = per_ds.shape[1]
    ttl = title or (f"overall — {nds} dataset" + ("s" if nds != 1 else ""))
    if group:
        ttl += f"  ({group} metrics)"
    ax.set_title(ttl, fontsize=11, loc="left")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig
