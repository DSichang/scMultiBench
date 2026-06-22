"""scIB-style bubble table: prep + matplotlib render."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import style


@dataclass
class BubbleTable:
    matrix: pd.DataFrame      # normalized value matrix (method x metric), sorted by overall
    raw: pd.DataFrame         # raw value matrix (same shape/order)
    overall: pd.Series        # overall score per method
    ranks: pd.DataFrame       # max-rank per metric (for circle radius)


def build_table(
    long_df: pd.DataFrame,
    metrics: list[str] | None = None,
    methods: list[str] | None = None,
    order: list[str] | None = None,
    aggregate: str = "dataset",
) -> BubbleTable:
    """Pivot tidy long results to a normalized method x metric matrix + overall."""
    df = long_df.copy()
    if methods is not None:
        df = df[df["method"].isin(methods)]

    if aggregate == "summary":
        # rank within each dataset per metric, then average ranks across datasets
        parts = []
        for _, g in df.groupby("dataset"):
            piv = g.pivot_table(index="method", columns="metric", values="value", aggfunc="mean")
            if metrics is not None:
                piv = piv[[m for m in metrics if m in piv.columns]]
            ranks = piv.apply(lambda col: style.rank_max(col.to_numpy()), axis=0)
            parts.append(ranks)
        all_index = parts[0].index
        for q in parts[1:]:
            all_index = all_index.union(q.index)
        raw = sum(p.reindex(all_index).fillna(0) for p in parts) / len(parts)
    else:
        # average over datasets if multiple present (individual panel = single dataset)
        raw = df.pivot_table(index="method", columns="metric", values="value", aggfunc="mean")
        if metrics is not None:
            raw = raw[[m for m in metrics if m in raw.columns]]

    norm = raw.apply(lambda col: style.minmax(col.to_numpy()), axis=0)
    ranks = raw.apply(lambda col: style.rank_max(col.to_numpy()), axis=0)
    overall = style.compute_overall(raw)

    if order is not None:
        idx = [m for m in order if m in raw.index]
    else:
        idx = overall.sort_values(ascending=False).index.tolist()
    return BubbleTable(
        matrix=norm.loc[idx],
        raw=raw.loc[idx],
        overall=overall.loc[idx],
        ranks=ranks.loc[idx],
    )


def render(tbl: BubbleTable, cmap: str = "Blues", title: str | None = None):
    """Render the bubble table; circle radius ~ rank, fill ~ value. Returns a Figure."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    from matplotlib import cm, colors

    methods = list(tbl.matrix.index)
    cols = ["overall"] + list(tbl.matrix.columns)
    n_rows, n_cols = len(methods), len(cols)

    fig = Figure(figsize=(1.1 * n_cols + 2, 0.5 * n_rows + 1))
    ax = fig.subplots()
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # value matrix incl. overall as the first column
    val = tbl.matrix.copy()
    val.insert(0, "overall", tbl.overall)
    # radius source: ranks rescaled to [0.15, 0.85]; overall uses its own rank
    rank = tbl.ranks.copy()
    rank.insert(0, "overall", style.rank_max(tbl.overall.to_numpy()))

    def radius(col: str) -> np.ndarray:
        r = rank[col].to_numpy(dtype=float)
        return style.minmax(r) * (0.85 - 0.15) + 0.15

    for j, c in enumerate(cols):
        rads = radius(c)
        for i, m in enumerate(methods):
            v = val.loc[m, c]
            fill = mapper.to_rgba(0.0 if pd.isna(v) else float(v))
            ax.add_patch(Circle((j + 0.5, n_rows - i - 0.5), rads[i] * 0.45, facecolor=fill,
                                 edgecolor="#444444", linewidth=0.25))

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(cols, rotation=30, ha="left", fontsize=8)
    ax.set_yticks([n_rows - i - 0.5 for i in range(n_rows)])
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    fig.colorbar(mapper, ax=ax, fraction=0.03, pad=0.02, label="value")
    fig.tight_layout()
    return fig


def plot_bubble(long_df, *, metrics=None, methods=None, order=None,
                aggregate="dataset", cmap="Blues", title=None, save=None):
    """Build + render a bubble table from a long results DataFrame."""
    tbl = build_table(long_df, metrics=metrics, methods=methods, order=order, aggregate=aggregate)
    fig = render(tbl, cmap=cmap, title=title)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return fig
