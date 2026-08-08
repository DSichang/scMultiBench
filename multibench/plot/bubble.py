"""Paper-style bubble table: methods as rows, metrics as columns, grouped by task.

Layout follows the benchmark paper's figures: each task family (DR & clustering,
batch correction) is a colour-banded block - blues and greens respectively -
preceded by an *Overall* column that ranks methods within that family; the top
three per column carry their rank number inside the marker. Methods missing a
value simply have no marker there, matching how the paper shows inapplicable
metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import style
from .bar import BATCH_METRICS, CLUSTERING_METRICS

# column order within each family, as in the paper's panels
FAMILIES = [
    ("DR and clustering", ["cLISI", "ARI", "ASW", "iASW", "iF1", "NMI"], "Blues"),
    ("Batch correction", ["ASW_batch", "GC", "iLISI", "kBET"], "Greens"),
]


@dataclass
class FamilyBlock:
    label: str                # header text
    cmap: str                 # matplotlib colormap name
    raw: pd.DataFrame         # raw value matrix (method x metric), row-aligned
    norm: pd.DataFrame        # per-column min-max normalized values
    ranks: pd.DataFrame       # per-column max-ranks (1 = best is HIGHEST rank number)
    overall: pd.Series        # family overall score per method


@dataclass
class BubbleTable:
    methods: list             # row order, best first
    blocks: list              # list[FamilyBlock], in FAMILIES order
    # kept for backward compatibility with callers that inspect the table
    matrix: pd.DataFrame      # all families' normalized columns, concatenated
    raw: pd.DataFrame
    overall: pd.Series        # combined overall used for the row order


def _pivot(df: pd.DataFrame, aggregate: str) -> pd.DataFrame:
    if aggregate == "summary":
        parts = []
        for _, g in df.groupby("dataset"):
            piv = g.pivot_table(index="method", columns="metric", values="value",
                                aggfunc="mean")
            parts.append(piv.apply(lambda col: style.rank_max(col.to_numpy()), axis=0))
        idx = parts[0].index
        for q in parts[1:]:
            idx = idx.union(q.index)
        return sum(p.reindex(idx).fillna(0) for p in parts) / len(parts)
    return df.pivot_table(index="method", columns="metric", values="value",
                          aggfunc="mean")


def build_table(long_df: pd.DataFrame, metrics=None, methods=None, order=None,
                aggregate: str = "dataset") -> BubbleTable:
    """Pivot tidy long results into per-family matrices plus a combined row order."""
    df = long_df.copy()
    if methods is not None:
        df = df[df["method"].isin(methods)]
    if metrics is not None:
        df = df[df["metric"].isin(metrics)]
    raw_all = _pivot(df, aggregate)

    blocks = []
    for label, fam_metrics, cmap in FAMILIES:
        cols = [m for m in fam_metrics if m in raw_all.columns]
        if not cols:
            continue
        raw = raw_all[cols]
        if raw.notna().sum().sum() == 0:
            continue
        blocks.append(FamilyBlock(
            label=label, cmap=cmap, raw=raw,
            norm=raw.apply(lambda col: style.minmax(col.to_numpy()), axis=0),
            ranks=raw.apply(lambda col: style.rank_max(col.to_numpy()), axis=0),
            overall=style.compute_overall(raw),
        ))
    # anything not in a known family still gets shown, neutrally coloured
    leftover = [c for c in raw_all.columns
                if not any(c in f[1] for f in FAMILIES)]
    if leftover:
        raw = raw_all[leftover]
        blocks.append(FamilyBlock(
            label="Other", cmap="Purples", raw=raw,
            norm=raw.apply(lambda col: style.minmax(col.to_numpy()), axis=0),
            ranks=raw.apply(lambda col: style.rank_max(col.to_numpy()), axis=0),
            overall=style.compute_overall(raw),
        ))
    if not blocks:
        raise ValueError("no known metrics found in long_df")

    combined = pd.concat([b.overall for b in blocks], axis=1).mean(axis=1)
    if order is not None:
        idx = [m for m in order if m in raw_all.index]
    else:
        idx = combined.sort_values(ascending=False).index.tolist()

    for b in blocks:
        b.raw = b.raw.reindex(idx)
        b.norm = b.norm.reindex(idx)
        b.ranks = b.ranks.reindex(idx)
        b.overall = b.overall.reindex(idx)

    return BubbleTable(
        methods=idx, blocks=blocks,
        matrix=pd.concat([b.norm for b in blocks], axis=1),
        raw=pd.concat([b.raw for b in blocks], axis=1),
        overall=combined.loc[idx],
    )


def render(tbl: BubbleTable, cmap: str | None = None, title: str | None = None):
    """Render the paper-style table. Returns a matplotlib Figure.

    ``cmap`` overrides the FIRST family's colormap only (compatibility with the
    old single-colormap signature); the batch family stays green like the paper.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    from matplotlib import cm, colors

    methods = tbl.methods
    n_rows = len(methods)

    # x layout: [method labels] [gap] per family: [overall][metrics...] [gap]
    xs, col_meta = 0.0, []   # col_meta: (x, family_idx, kind, name)
    for fi, b in enumerate(tbl.blocks):
        col_meta.append((xs, fi, "overall", "Rank"))
        xs += 1.15
        for mname in b.raw.columns:
            col_meta.append((xs, fi, "metric", mname))
            xs += 1.0
        xs += 0.55                                    # gap between families
    total_w = xs - 0.55

    fig = Figure(figsize=(0.62 * total_w + 2.6, 0.44 * n_rows + 1.7))
    ax = fig.subplots()
    norm01 = colors.Normalize(vmin=0.0, vmax=1.0)
    mappers = {fi: cm.ScalarMappable(norm=norm01,
                                     cmap=(cmap if (fi == 0 and cmap) else b.cmap))
               for fi, b in enumerate(tbl.blocks)}

    # zebra row stripes, like the paper
    for i in range(n_rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((-0.35, n_rows - i - 1), total_w + 0.7, 1.0,
                                   facecolor="#ebebeb", edgecolor="none", zorder=0))

    def rank_label(ax_, x, y, rank_pos, fill):
        lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        ax_.text(x, y, str(rank_pos), ha="center", va="center", zorder=5,
                 fontsize=7.2, fontweight="bold",
                 color="white" if lum < 0.55 else "#222222")

    for x, fi, kind, name in col_meta:
        b = tbl.blocks[fi]
        mp = mappers[fi]
        if kind == "overall":
            vals = b.overall
            # dense best-first positions for the numbers 1..3
            pos = vals.rank(ascending=False, method="min")
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    continue
                y = n_rows - i - 0.5
                fill = mp.to_rgba(float(v))
                ax.add_patch(Rectangle((x + 0.18, y - 0.32), 0.64, 0.64,
                                       facecolor=fill, edgecolor="#444444",
                                       linewidth=0.5, zorder=3))
                if pos.loc[m] <= 3:
                    rank_label(ax, x + 0.5, y, int(pos.loc[m]), fill)
        else:
            colvals = b.raw[name]
            colnorm = b.norm[name]
            pos = colvals.rank(ascending=False, method="min")
            for i, m in enumerate(methods):
                v = colvals.loc[m]
                if pd.isna(v):
                    continue
                y = n_rows - i - 0.5
                fill = mp.to_rgba(float(colnorm.loc[m]))
                ax.add_patch(Circle((x + 0.5, y), 0.34, facecolor=fill,
                                    edgecolor="#444444", linewidth=0.5, zorder=3))
                if pos.loc[m] <= 3:
                    rank_label(ax, x + 0.5, y, int(pos.loc[m]), fill)

    # family header bands
    for fi, b in enumerate(tbl.blocks):
        xs_f = [x for x, f, k, _ in col_meta if f == fi]
        kinds = [k for x, f, k, _ in col_meta if f == fi]
        x0, x1 = xs_f[0], xs_f[-1] + 1.0
        band = mappers[fi].to_rgba(0.28)
        # the Overall chip and the family band, side by side like the paper
        ax.add_patch(FancyBboxPatch((x0 + 0.06, n_rows + 0.25), 1.0, 0.62,
                                    boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=mappers[fi].to_rgba(0.55),
                                    edgecolor="none", zorder=2))
        ax.text(x0 + 0.56, n_rows + 0.56, "Overall", ha="center", va="center",
                fontsize=7.6, color="white", fontweight="bold", zorder=3)
        if len(xs_f) > 1:
            ax.add_patch(FancyBboxPatch((xs_f[1] + 0.04, n_rows + 0.25),
                                        x1 - xs_f[1] - 0.08, 0.62,
                                        boxstyle="round,pad=0.02,rounding_size=0.18",
                                        facecolor=band, edgecolor="none", zorder=2))
            ax.text((xs_f[1] + x1) / 2, n_rows + 0.56, b.label, ha="center",
                    va="center", fontsize=8.2, color="#1a1a1a", zorder=3)

    # column tick labels, slanted like the paper
    for x, fi, kind, name in col_meta:
        ax.text(x + 0.5, -0.18, name, rotation=38, ha="right", va="top", fontsize=8)

    ax.set_xlim(-0.35, total_w + 0.35)
    ax.set_ylim(-1.7, n_rows + 1.1)
    ax.set_yticks([n_rows - i - 0.5 for i in range(n_rows)])
    ax.set_yticklabels(methods, fontsize=8.6)
    ax.set_xticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    if title:
        ax.set_title(title, fontsize=10, pad=14)
    fig.tight_layout()
    return fig


def plot_bubble(long_df, *, metrics=None, methods=None, order=None,
                aggregate="dataset", cmap=None, title=None, save=None):
    """Build + render the paper-style bubble table from a long results frame."""
    tbl = build_table(long_df, metrics=metrics, methods=methods, order=order,
                      aggregate=aggregate)
    fig = render(tbl, cmap=cmap, title=title)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return fig
