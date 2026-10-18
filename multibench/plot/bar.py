"""Summary bar chart: one bar per method, aggregated across datasets."""
from __future__ import annotations


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


@_compat.keywords("plot.bar", renamed={"group": "metrics"}, private="_bar", kept={
    "cmap": _compat.note("leave it out for the metric family's colours"),
    "show_datasets": _compat.note(
        'leave it out to draw the dataset dots of overall="mean_overall"'),
    "top": _compat.use("methods="),
    "overall": _compat.use("the default")})
def bar(long_df: pd.DataFrame, *, metrics=None, methods=None, title: str | None = None,
        save: str | None = None):
    """Draw each method's overall score as a horizontal bar, best on top.

    Within each dataset the methods are ranked on each metric; the bar is the
    across-dataset score, in the order of ``mtb.recommend(category,
    long_df=long_df)``.

    Parameters
    ----------
    long_df : pandas.DataFrame
        A long table (``method``, ``metric``, ``value`` and optional
        ``dataset``), as ``mtb.load_results``, ``mtb.evaluate`` and
        ``BatchResult.long`` return.
    metrics : str, list of str | None
        A family (``"clustering"``, ``"batch"``, ``"classification"``,
        ``"all"``) drawn in its colour, a code, or codes; ``None`` = every
        metric.
    methods : list of str | None
        Methods (bars) to show, ranked among themselves; ``None`` = every
        method in the frame.
    title : str | None
        Figure title; ``None`` = the dataset, or the number of datasets.
    save : str | None
        File to write the figure to.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        ``long_df`` lacks ``method`` / ``metric`` / ``value``, is empty,
        repeats a ``(method, dataset, metric)`` row, or ``metrics`` /
        ``methods`` names a value it does not hold.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.load_results("diagonal")
    >>> fig = mtb.plot.bar(df, metrics="clustering", save="clustering.png")
    """
    return _bar(long_df, metrics=metrics, methods=methods, title=title, save=save)


def _bar(long_df: pd.DataFrame, *, metrics=None, methods=None, top: int | None = None,
         title: str | None = None, save: str | None = None, overall: str = "rank",
         cmap: str = "Blues", show_datasets: bool = True):
    """:func:`bar`, with the deprecated ``top``, ``overall``, ``cmap`` (the colours
    when no metric family is drawn) and ``show_datasets``."""
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
            "long_df has no rows to plot. If it came from load_results, its "
            "UserWarning says why.")
    if methods is not None:
        from ..data import catalog as _catalog
        methods = _resolve(methods, long_df["method"].unique().tolist(), "method",
                           _catalog._canonical_id)
        long_df = long_df[long_df["method"].isin(methods)]
    family = (metrics.lower() if isinstance(metrics, str)
              and metrics.lower() in FAMILY_METRICS else None)
    if family:
        metrics = FAMILY_METRICS[family]
        present = set(long_df["metric"].unique())
        if not (set(metrics) & present):
            why = ("The stored results have none. "
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

    from .bubble import _duplicate_rows_message, _user_repeats
    keys = ["method", "metric"] + (["dataset"] if "dataset" in long_df.columns else [])
    picked = long_df if not metrics else long_df[long_df["metric"].isin(metrics)]
    repeated = _user_repeats(picked, keys)
    if repeated.any():
        raise ValueError(_duplicate_rows_message(picked[repeated], keys))
    parts = style.per_dataset_ranks(long_df, metrics)
    lone = style.lone_message({ds: mat.attrs.get("lone", []) for ds, mat in parts.items()})
    if lone:
        _compat.warn(lone, UserWarning)
    # bar always ranks within each dataset: the checks of bubble(aggregate="summary")
    for msg in style.coverage_warnings(
            parts, basis=overall,
            incomplete_fix=_config.hint(
                "Pass methods= with the methods scored on every dataset.",
                "Pass --methods with the methods scored on every dataset."),
            stored=style.stored_datasets(long_df)):
        _compat.warn(msg, UserWarning)
    shown = long_df if not metrics else long_df[long_df["metric"].isin(metrics)]
    backend = style.backend_warning(shown)
    if backend:
        _compat.warn(backend, UserWarning)
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
        colors = style.colormap(FAMILY_CMAPS.get(family, cmap))(
            0.35 + 0.55 * minmax(mean.to_numpy()))
        # whisker = SD across datasets; none for a method in one dataset, and none
        # under overall="rank", whose bar is not a mean of per-dataset scores
        if overall == "mean_overall":
            sd = per_ds.std(axis=1, ddof=0).reindex(mean.index)
            n_scored = per_ds.notna().sum(axis=1).reindex(mean.index)
            xerr = np.where(n_scored.to_numpy() > 1, sd.to_numpy(), np.nan)
        else:
            xerr = np.full(n, np.nan)
        # no whisker artist at all when no method has a spread (one dataset)
        ax.barh(range(n), mean.to_numpy(), color=colors, edgecolor="#333",
                linewidth=0.6, zorder=2,
                xerr=xerr if np.isfinite(xerr).any() else None,
                error_kw=dict(ecolor="#333", elinewidth=1.0, capsize=3, capthick=1.0))

        if show_datasets and overall == "mean_overall" and per_ds.shape[1] > 1:
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
        heading = title or (f"overall — {per_ds.columns[0]}" if n_datasets == 1
                            else f"overall — {n_datasets} datasets")
        if family:
            heading += f"  ({family} metrics)"
        ax.set_title(heading, fontsize=11, loc="left")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig
