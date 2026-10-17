"""Summary bar chart: one bar per method, aggregated across datasets."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import style
from .. import _compat
from .. import config as _config
from ..data.catalog import FAMILY_METRICS
from .style import compute_overall, minmax

CLUSTERING_METRICS = FAMILY_METRICS["clustering"]
BATCH_METRICS = FAMILY_METRICS["batch"]
CLASSIFICATION_METRICS = FAMILY_METRICS["classification"]
#: the colour ramp of each family, as in the paper's figures
FAMILY_CMAPS = {"clustering": "Blues", "batch": "Greens", "classification": "Pinks"}


@_compat.keywords("plot.bar", group="metrics", cmap=None, show_datasets=None)
def bar(long_df: pd.DataFrame, *, metrics=None, top: int | None = None,
        title: str | None = None, save: str | None = None, overall: str = "rank"):
    """Draw each method's overall score across datasets as a horizontal bar.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long table with ``method``, ``metric``, ``value`` columns and an
        optional ``dataset`` (absent = one dataset), e.g. from
        ``mtb.load_results``.
    metrics : str, list of str | None
        Metric codes to score, or a family (``"clustering"``, ``"batch"``,
        ``"classification"``, ``"all"``) drawn in its colour, as the
        benchmark's summary panels; ``None`` = every metric.
    top : int | None
        Keep the ``top`` best methods, scored against all of them; ``None``
        = every method.
    title : str | None
        Figure title; ``None`` = the number of datasets. With a family,
        ``(<family> metrics)`` is appended.
    save : str | None
        File to write the figure to (140 dpi, tight bounding box).
    overall : {"rank", "mean_overall"}
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
        ``long_df`` lacks ``method`` / ``metric`` / ``value``, or is empty
        (``mtb.load_results`` may have returned nothing; see its warning).
    ValueError
        Unknown ``metrics`` code (the error suggests a close match and lists
        the metrics present), or an invalid ``overall``.
    ValueError
        ``metrics`` names a family with no metric in the frame: batch metrics
        need a multi-batch dataset, and the stored tables have no
        classification scores.

    Warns
    -----
    UserWarning
        A method missing from some datasets, a dataset with one method, or a
        figure of two methods.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.load_results("diagonal")
    >>> fig = mtb.plot.bar(df, metrics="clustering", top=10, save="clustering.png")

    See Also
    --------
    mtb.plot.bubble : per-dataset (or rank-averaged) bubble table, same ``overall=`` formulas.
    mtb.load_results : stored metric tables as a long frame.
    """
    import matplotlib.pyplot as plt
    from .bubble import _resolve
    from ..data import catalog

    need = {"method", "metric", "value"}
    have = set(long_df.columns)
    if not need.issubset(have):
        raise ValueError(
            f"bar() needs a long table with columns ['method', 'metric', "
            f"'value']; missing {sorted(need - have)}")
    if len(long_df) == 0:
        raise ValueError(
            "long_df has no rows to plot. load_results may have returned none; "
            "its UserWarning says why.")
    family = metrics if isinstance(metrics, str) and metrics in FAMILY_METRICS else None
    if family:
        metrics = FAMILY_METRICS[family]
        present = set(long_df["metric"].unique())
        if not (set(metrics) & present):
            why = ("The stored tables have none. "
                   + _config.hint("mtb.evaluate", "multibench evaluate")
                   + " computes them on multi-batch data." if family == "classification" else
                   "Batch metrics need a multi-batch dataset - a single-batch design "
                   "has none to compute.")
            raise ValueError(f"no {family!r} metrics present. Found {sorted(present)}. {why}")
    elif metrics is not None:
        from .bubble import _expand_families
        metrics = _expand_families(metrics, long_df["metric"].dropna().unique().tolist())
        metrics = _resolve(metrics, long_df["metric"].dropna().unique().tolist(),
                           "metric", catalog._canonical_metric)
    if overall not in style.OVERALL_BASES:
        raise ValueError(
            f"overall must be one of {list(style.OVERALL_BASES)}, got {overall!r}")

    parts = style.per_dataset_ranks(long_df, metrics)
    # bar always ranks within each dataset: the checks of bubble(aggregate="summary")
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
    if len({m for mat in parts.values() for m in mat.index}) == 2:
        warnings.warn(style.TWO_METHODS, UserWarning, stacklevel=2)
    per_ds = pd.DataFrame({ds: compute_overall(mat) for ds, mat in parts.items()})
    # best-first with a stable tie-break (same as plot.bubble), then reversed
    # so the best method is drawn on top
    mean = style.overall_by_basis(parts, overall).sort_values(
        ascending=False, kind="mergesort")[::-1]
    if top:
        mean = mean.tail(top)
    per_ds = per_ds.reindex(mean.index)

    n = len(mean)
    # matplotlib's own defaults, whatever the session's rcParams are:
    # importing scib (mtb.evaluate) switches them to seaborn's "talk" sizes
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.34 * n + 1.1)))
        colors = style.colormap(FAMILY_CMAPS.get(family, "Blues"))(
            0.35 + 0.55 * minmax(mean.to_numpy()))
        # whisker = SD across datasets; none for a method in one dataset, and none
        # under overall="rank", whose bar is not a mean of per-dataset scores
        if overall == "mean_overall":
            sd = per_ds.std(axis=1, ddof=0).reindex(mean.index)
            n_scored = per_ds.notna().sum(axis=1).reindex(mean.index)
            xerr = np.where(n_scored.to_numpy() > 1, sd.to_numpy(), np.nan)
        else:
            xerr = np.full(n, np.nan)
        ax.barh(range(n), mean.to_numpy(), color=colors, edgecolor="#333",
                linewidth=0.6, zorder=2, xerr=xerr,
                error_kw=dict(ecolor="#333", elinewidth=1.0, capsize=3, capthick=1.0))

        if overall == "mean_overall" and per_ds.shape[1] > 1:
            for i, m in enumerate(mean.index):
                vals = per_ds.loc[m].dropna().to_numpy()
                ax.scatter(vals, np.full(len(vals), i), s=16, facecolor="white",
                           edgecolor="#444", linewidth=0.7, zorder=3)

        ax.set_yticks(range(n)); ax.set_yticklabels(mean.index, fontsize=9)
        n_datasets = per_ds.shape[1]
        if n_datasets == 1:
            xlabel = f"overall score ({per_ds.columns[0]})"
        else:
            how = ("mean of per-dataset overall" if overall == "mean_overall"
                   else "rank of mean ranks")
            xlabel = f"overall score ({how}, {n_datasets} datasets)"
        ax.set_xlabel(xlabel, fontsize=9)
        # the scores lie in [0, 1]; a whisker (mean +/- SD) can reach past either end
        reach = np.nan_to_num(xerr)
        ax.set_xlim(min(0, (mean.to_numpy() - reach).min() - 0.02),
                    max(1, (mean.to_numpy() + reach).max()) + 0.02)
        ax.grid(axis="x", color="#e8e8e8", zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        heading = title or (f"overall — {n_datasets} dataset"
                            + ("s" if n_datasets != 1 else ""))
        if family:
            heading += f"  ({family} metrics)"
        ax.set_title(heading, fontsize=11, loc="left")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig
