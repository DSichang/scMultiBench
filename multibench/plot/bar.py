"""Summary bar chart: one bar per method, aggregated ACROSS datasets.

The bubble chart answers "how did each method do on THIS dataset". This answers
"how does each method do overall", which is the summary the benchmark reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import style
from .style import compute_overall, minmax, rank_max

# scIB metric families, so a summary can be split the way the benchmark reports it
# These must agree with the groups eval.scib.compute() actually emits - see
# tests/test_metric_groups.py, which pins them together. iASW and iF1 are
# ISOLATED-LABEL scores: scib files them under bio conservation, alongside
# ARI/NMI/ASW/cLISI, not under batch correction. They used to sit in
# BATCH_METRICS here while compute() emitted them for group="clustering", so the
# same number was labelled a different family depending on which module you
# asked - and a single-batch dataset like D11, which legitimately has iASW/iF1
# and no batches at all, rendered as though it had batch-correction results.
CLUSTERING_METRICS = ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"]
BATCH_METRICS = ["ASW_batch", "GC", "iLISI", "kBET"]


def _score_per_dataset(long_df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Per (method, dataset) overall score, using the benchmark's rank rule
    (``style.compute_overall`` of the within-dataset max-rank matrix). Thin
    wrapper over :func:`multibench.plot.style.per_dataset_ranks`, the math
    shared with ``plot.bubble``."""
    parts = style.per_dataset_ranks(long_df, metrics)
    return pd.DataFrame({ds: compute_overall(mat) for ds, mat in parts.items()})


def bar(long_df: pd.DataFrame, *, metrics=None, group: str | None = None,
        top: int | None = None, title: str | None = None, cmap: str = "Blues",
        show_datasets: bool = True, save: str | None = None,
        overall: str = "mean_overall"):
    """Bar chart of each method's overall score, aggregated across datasets.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Tidy frame (``metric, value, method, dataset, category``) - the same
        frame :func:`multibench.load_results`, :func:`multibench.to_long` and
        the ``BatchResult.long`` property produce. Concatenate several
        datasets' frames to summarise across them. An empty frame raises
        ``ValueError`` saying so (``load_results`` may have returned nothing).
    metrics : list of str, optional
        Restrict to these metric codes (case/alias tolerant, ``"ari"`` ->
        ``"ARI"``; an unknown code raises ``ValueError`` listing the metrics
        present). Ignored when ``group`` is given.
    group : {"clustering", "batch"}, optional
        Shorthand for that metric family, so you can produce the benchmark's
        two summary panels.
    top : int, optional
        Keep only the N best methods.
    title : str, optional
        Figure title (default names the number of datasets).
    cmap : str
        Matplotlib colormap for the bars (``"Greens"`` is forced for
        ``group="batch"`` to match the paper's family colours).
    show_datasets : bool
        Overlay one dot per dataset behind each bar, so a method that is
        uniformly good is distinguishable from one that averages well by
        winning on a single dataset (only meaningful under
        ``overall="mean_overall"``; the dots are omitted under ``"rank"``).
    save : str, optional
        Path to write the figure to (``fig.savefig``).
{OVERALL_DOC}
        The default here is ``"mean_overall"``; ``plot.bubble`` defaults to
        ``"rank"``. Pass the same value to both to get the same ordering.

    Returns
    -------
    matplotlib.figure.Figure
        Under ``overall="mean_overall"`` whiskers are the SD of the
        per-dataset scores (methods present in a single dataset get none -
        there is no spread to show) and the x label names the single dataset
        or the formula. Because the score is rank-based, it is only
        meaningful relative to the other methods in the same figure. A
        method lacking a metric is compared on the metrics it has (the
        within-dataset rank matrix skips NaN cells).

    Raises
    ------
    ValueError
        An empty or column-less frame, an unknown ``metrics`` code, a bad
        ``group`` / ``overall``, or ``group="batch"`` on a frame without
        batch metrics.
    """
    import matplotlib.pyplot as plt
    from .bubble import _resolve
    from ..data import catalog

    need = {"method", "metric", "value"}
    have = set(getattr(long_df, "columns", []))
    if not need.issubset(have):
        raise ValueError(
            f"bar() needs a tidy long frame with columns ['method', 'metric', "
            f"'value']; missing {sorted(need - have)}")
    if len(long_df) == 0:
        raise ValueError(
            "long_df is empty (0 rows) - nothing to plot; load_results(...) may "
            "have returned nothing, see its UserWarning (e.g. a method with no "
            "rows under source='published' - try source='rerun')")
    if metrics is not None and not group:
        metrics = _resolve(metrics, long_df["metric"].dropna().unique().tolist(),
                           "metric", catalog.canonical_metric)
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
    if overall not in style.OVERALL_BASES:
        raise ValueError(
            f"overall must be one of {list(style.OVERALL_BASES)}, got {overall!r}")

    parts = style.per_dataset_ranks(long_df, metrics)
    per_ds = pd.DataFrame({ds: compute_overall(mat) for ds, mat in parts.items()})
    # best-first with a stable tie-break (same as plot.bubble), then reversed
    # so the best method is drawn on top
    mean = style.overall_by_basis(parts, overall).sort_values(
        ascending=False, kind="mergesort")[::-1]
    if top:
        mean = mean.tail(top)
    per_ds = per_ds.reindex(mean.index)

    n = len(mean)
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.34 * n + 1.1)))
    if group == "batch":
        cmap = "Greens"        # match the paper's family colours
    colors = plt.get_cmap(cmap)(0.35 + 0.55 * minmax(mean.to_numpy()))
    # error bar = SD across datasets; a method present in one dataset has no
    # spread to show, so it gets a bar without a whisker rather than a fake one.
    # Under overall="rank" the bar is not a mean of per-dataset scores, so no
    # whisker/dots apply.
    if overall == "mean_overall":
        sd = per_ds.std(axis=1, ddof=0).reindex(mean.index)
        nds_per = per_ds.notna().sum(axis=1).reindex(mean.index)
        xerr = np.where(nds_per.to_numpy() > 1, sd.to_numpy(), np.nan)
    else:
        xerr = np.full(n, np.nan)
    ax.barh(range(n), mean.to_numpy(), color=colors, edgecolor="#333",
            linewidth=0.6, zorder=2,
            xerr=np.where(np.isnan(xerr), 0.0, xerr),
            error_kw=dict(ecolor="#333", elinewidth=1.0, capsize=3, capthick=1.0))

    if show_datasets and overall == "mean_overall" and per_ds.shape[1] > 1:
        for i, m in enumerate(mean.index):
            vals = per_ds.loc[m].dropna().to_numpy()
            ax.scatter(vals, np.full(len(vals), i), s=16, facecolor="white",
                       edgecolor="#444", linewidth=0.7, zorder=3)

    ax.set_yticks(range(n)); ax.set_yticklabels(mean.index, fontsize=9)
    nds = per_ds.shape[1]
    if nds == 1:
        xlabel = f"overall score ({per_ds.columns[0]})"
    else:
        how = ("mean of per-dataset overall" if overall == "mean_overall"
               else "rank of mean ranks")
        xlabel = f"overall score ({how}, {nds} datasets)"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", color="#e8e8e8", zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ttl = title or (f"overall — {nds} dataset" + ("s" if nds != 1 else ""))
    if group:
        ttl += f"  ({group} metrics)"
    ax.set_title(ttl, fontsize=11, loc="left")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig


bar.__doc__ = bar.__doc__.replace("{OVERALL_DOC}", style.OVERALL_DOC)
