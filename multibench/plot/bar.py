"""Summary bar chart: one bar per method, aggregated across datasets."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import style
from .. import config as _config
from .style import compute_overall, minmax, rank_max

# scIB metric families, so a summary can be split the way the benchmark reports
# it. They must agree with the groups eval.scib.compute() emits
# (tests/test_metric_groups.py pins them together). iASW and iF1 are
# isolated-label scores: scIB files them under bio conservation with
# ARI/NMI/ASW/cLISI, not under batch correction, so a single-batch dataset has
# them without having any batch-correction result.
CLUSTERING_METRICS = ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"]
BATCH_METRICS = ["ASW_batch", "GC", "iLISI", "kBET"]


def bar(long_df: pd.DataFrame, *, metrics=None, group: str | None = None,
        top: int | None = None, title: str | None = None, cmap: str = "Blues",
        show_datasets: bool = True, save: str | None = None,
        overall: str = "mean_overall"):
    """Draw each method's overall score across datasets as a horizontal bar.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long table with ``method``, ``metric``, ``value`` columns and an
        optional ``dataset`` (absent = one dataset), e.g. from
        ``mtb.load_results``.
    metrics : list of str | None
        Metric codes to score; ``None`` = every metric. Ignored when
        ``group`` is given.
    group : {None, "clustering", "batch"}
        Score one metric family only (the benchmark's two summary panels);
        ``None`` = the ``metrics`` selection.
    top : int | None
        Keep the ``top`` best methods, scored against all of them; ``None``
        = every method.
    title : str | None
        Figure title; ``None`` = the number of datasets. With ``group``,
        ``(<group> metrics)`` is appended.
    cmap : str
        Matplotlib colormap for the bars; ``group="batch"`` always uses
        ``"Greens"`` (the paper's family colour).
    show_datasets : bool
        Overlay one dot per dataset's score on each bar; drawn only under
        ``overall="mean_overall"`` with several datasets.
    save : str | None
        File to write the figure to (140 dpi, tight bounding box).
    overall : {"mean_overall", "rank"}
        Across-dataset *Overall*: ``"rank"`` re-ranks mean ranks (missing
        dataset = rank 0); ``"mean_overall"`` averages per-dataset Overalls
        (missing dataset skipped).

    Returns
    -------
    matplotlib.figure.Figure
        One horizontal bar per method, best on top.

    Raises
    ------
    ValueError
        ``long_df`` is empty or lacks ``method`` / ``metric`` / ``value``.
    ValueError
        Unknown ``metrics`` code, or an invalid ``group`` / ``overall``.
    ValueError
        ``group`` names a family with no metric in the frame.

    Warns
    -----
    UserWarning
        Several datasets and a method missing from some of them.
    UserWarning
        A dataset holds one method, or no method spans two datasets.
    UserWarning
        Rows scored with the igraph Leiden backend are shown with stored rows
        of their dataset.

    Examples
    --------
    >>> import multibench as mtb
    >>> # every dataset of the category that has a stored table
    >>> df = mtb.load_results("diagonal")
    >>> fig = mtb.plot.bar(df, group="clustering", top=10, save="clustering.png")
    >>> fig = mtb.plot.bar(df, group="batch")
    >>> fig = mtb.plot.bar(df, overall="rank")       # bubble's default formula

    Notes
    -----
    **Whiskers, dots and x label.**

    - Whiskers (``overall="mean_overall"``): the SD of the per-dataset
      scores; a method present in one dataset gets none - there is no spread
      to show.
    - Dots (``overall="mean_overall"``, ``show_datasets``): one dot per
      dataset score.
    - Under ``overall="rank"`` the bar is not a mean of per-dataset scores,
      so neither whiskers nor dots are drawn.
    - X label: the single dataset, or the formula and the number of
      datasets.

    **Reading the score.** It is rank-based, so it is only meaningful
    relative to the other methods in the same figure. A method lacking a
    metric is compared on the metrics it has under ``"mean_overall"``;
    under ``"rank"`` the missing cell is rank 0 in that dataset.

    **Input.** Concatenate several datasets' frames to summarise across
    them; ``mtb.to_long`` and the ``BatchResult.long`` property give the same
    frame for your own runs. ``metrics`` is case- and alias-tolerant
    (``"ari"`` -> ``"ARI"``).

    **Shared rules.** The Overall formulas, the tie-break shared with
    ``mtb.plot.bubble`` and the warnings are those of
    ``mtb.plot.build_table`` with ``aggregate="summary"``; its Notes give
    them.

    **Errors.** An unknown ``metrics`` code gets a did-you-mean hint and the
    list of metrics present. Batch metrics need a multi-batch dataset: a
    single-batch design has none to compute, which is what the
    ``group="batch"`` error says. An empty frame's error points at
    ``mtb.load_results``, which may have returned nothing (see its
    ``UserWarning``).

    See Also
    --------
    mtb.plot.bubble : per-dataset (or rank-averaged) bubble table, same ``overall=`` formulas.
    mtb.load_results : stored metric tables as a long frame.
    """
    import matplotlib.pyplot as plt
    from .bubble import _resolve
    from ..data import catalog

    need = {"method", "metric", "value"}
    have = set(getattr(long_df, "columns", []))
    if not need.issubset(have):
        raise ValueError(
            f"bar() needs a long table with columns ['method', 'metric', "
            f"'value']; missing {sorted(need - have)}")
    if len(long_df) == 0:
        raise ValueError(
            "long_df has no rows to plot. load_results may have returned none. "
            "Its UserWarning says why, for example a method with no rows under "
            "source='published'. Then try source='rerun'.")
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
    # the same cross-dataset checks as bubble(aggregate="summary"): bar
    # always ranks within each dataset first
    for msg in style.coverage_warnings(
            parts, basis=overall,
            incomplete_fix=_config.hint(
                "Filter long_df to the methods scored on every dataset.",
                "Pass --methods with the methods scored on every dataset."),
            stored=style.stored_datasets(long_df)):
        warnings.warn(msg, UserWarning, stacklevel=2)
    shown = long_df if not metrics else long_df[long_df["metric"].isin(metrics)]
    backend = style.backend_warning(shown)
    if backend:
        warnings.warn(backend, UserWarning, stacklevel=2)
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
