"""Summary bar chart: one bar per method, aggregated across datasets."""
from __future__ import annotations


import pandas as pd

from . import style
from .. import _text
from .._data.catalog import FAMILY_METRICS
from .style import minmax

CLUSTERING_METRICS = FAMILY_METRICS["clustering"]
BATCH_METRICS = FAMILY_METRICS["batch"]
CLASSIFICATION_METRICS = FAMILY_METRICS["classification"]
#: the colour ramp of each family, as in the paper's figures
FAMILY_CMAPS = {"clustering": "Blues", "batch": "Greens", "classification": "Pinks"}


def bar(long_df: pd.DataFrame, *, metrics=None, methods=None, title: str | None = None,
        save: str | None = None):
    """Draw each method's overall score as a horizontal bar, best on top.

    Within each dataset the methods are ranked on each metric; the bar is the
    across-dataset score, in the order of ``mtb.recommend(category,
    long_df=long_df, metrics=...)`` with the same metrics.

    Parameters
    ----------
    long_df : pandas.DataFrame
        A long table (``method``, ``metric``, ``value`` and optional
        ``dataset``), as ``mtb.load_results``, ``mtb.evaluate`` and
        ``RunResult.long`` return.
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
        repeats a ``(method, dataset, metric)`` row, holds your rows on
        another dataset than the stored rows, or ``metrics`` / ``methods``
        names a value it does not hold.

    Warns
    -----
    UserWarning
        A method missing from some datasets, a dataset with one method, a
        metric only one method has, rows scored before 0.4.0, or a figure of
        two methods.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.load_results("diagonal")
    >>> fig = mtb.plot.bar(df, metrics="clustering", save="clustering.png")
    """
    import matplotlib.pyplot as plt
    from .bubble import _resolve
    from .._data import catalog

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
        from .._data import catalog as _catalog
        methods = _resolve(methods, long_df["method"].unique().tolist(), "method",
                           _catalog._canonical_id)
        long_df = long_df[long_df["method"].isin(methods)]
    family = (metrics.lower() if isinstance(metrics, str)
              and metrics.lower() in FAMILY_METRICS else None)
    if family:
        metrics = FAMILY_METRICS[family]
        present = set(long_df["metric"].unique())
        if not (set(metrics) & present):
            why = ("The stored results have none. mtb.evaluate computes them on "
                   "multi-batch data." if family == "classification" else
                   "Batch metrics need a multi-batch dataset - a single-batch design "
                   "has none to compute.")
            raise ValueError(f"no {family!r} metrics present. Found {sorted(present)}. {why}")
    elif metrics is not None:
        from .bubble import _expand_families
        metrics = _expand_families(metrics, long_df["metric"].dropna().unique().tolist())
        metrics = _resolve(metrics, long_df["metric"].dropna().unique().tolist(),
                           "metric", catalog._canonical_metric)

    from .bubble import _duplicate_rows_message, _user_repeats
    keys = ["method", "metric"] + (["dataset"] if "dataset" in long_df.columns else [])
    picked = long_df if not metrics else long_df[long_df["metric"].isin(metrics)]
    repeated = _user_repeats(picked, keys)
    if repeated.any():
        raise ValueError(_duplicate_rows_message(picked[repeated], keys))
    mean = style.overall(long_df, metrics)       # refuses your rows on another dataset
    # the methods and metrics of each dataset, for the warnings
    groups = picked.groupby("dataset") if "dataset" in picked.columns else [("all", picked)]
    parts = {ds: mat for ds, g in groups
             if not (mat := g.pivot_table(index="method", columns="metric",
                                          values="value")).empty}
    lone = style.lone_message({ds: style.lone_columns(mat) for ds, mat in parts.items()})
    if lone:
        _text.warn(lone, UserWarning)
    for msg in style.coverage_warnings(
            parts, basis="mean_overall",
            incomplete_fix="Pass methods= with the methods scored on every dataset."):
        _text.warn(msg, UserWarning)
    backend = style.backend_warning(picked)
    if backend:
        _text.warn(backend, UserWarning)
    datasets = list(parts)
    mean = mean[::-1]                       # the best method is drawn on top

    n = len(mean)
    # matplotlib's own defaults, whatever the session's rcParams are:
    # importing scib (mtb.evaluate) switches them to seaborn's "talk" sizes
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.34 * n + 1.1)))
        colors = style.colormap(FAMILY_CMAPS.get(family, "Blues"))(
            0.35 + 0.55 * minmax(mean.to_numpy()))
        ax.barh(range(n), mean.to_numpy(), color=colors, edgecolor="#333",
                linewidth=0.6, zorder=2)

        ax.set_yticks(range(n)); ax.set_yticklabels(mean.index, fontsize=9)
        n_datasets = len(datasets)
        if n_datasets == 1:
            xlabel = f"overall score ({datasets[0]})"
        else:
            xlabel = f"overall score (mean of per-dataset overall, {n_datasets} datasets)"
        ax.set_xlabel(xlabel, fontsize=9)
        # the stored scores lie in [0, 1]; your rows can reach past either end
        ax.set_xlim(min(0, mean.to_numpy().min() - 0.02), max(1, mean.to_numpy().max()) + 0.02)
        ax.grid(axis="x", color="#e8e8e8", zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        heading = title or (f"overall — {datasets[0]}" if n_datasets == 1
                            else f"overall — {n_datasets} datasets")
        if family:
            heading += f"  ({family} metrics)"
        ax.set_title(heading, fontsize=11, loc="left")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig
