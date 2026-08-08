"""Paper-style bubble table: methods as rows, metrics as columns, grouped by task.

Layout follows the benchmark paper's figures: each task family (DR & clustering,
batch correction) is a colour-banded block - blues and greens respectively -
preceded by an *Overall* column drawn as a vertical bar whose LENGTH and colour
both encode the family score; the top three per column carry their rank number.
With ``aggregate="summary"`` (several datasets, the paper's panel c) every
metric marker becomes such a bar too - the value is then the metric's rank
averaged across datasets; with a single dataset the metric markers are circles
(panel b). Methods missing a value simply have no marker there.
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
    aggregate: str = "dataset"  # "dataset" -> circles; "summary" -> bars (paper c)


def _pivot(df: pd.DataFrame, aggregate: str) -> pd.DataFrame:
    if aggregate == "summary":
        # Rank methods within each dataset per metric, then average the ranks -
        # but average each METRIC only over the datasets that computed it. A
        # naive sum of the per-dataset frames aligns columns and turns a metric
        # into all-NaN as soon as ONE dataset lacks it (a single-batch dataset
        # has no batch metrics), which silently erased whole families.
        parts = []
        for _, g in df.groupby("dataset"):
            piv = g.pivot_table(index="method", columns="metric", values="value",
                                aggfunc="mean")
            piv = piv.dropna(axis=1, how="all")   # a metric this dataset never computed
            ranks = piv.apply(lambda col: style.rank_max(col.to_numpy()), axis=0)
            parts.append(ranks)
        idx = parts[0].index
        for q in parts[1:]:
            idx = idx.union(q.index)
        # a method absent from a dataset scores 0 there (as in the paper's
        # summary), but only for metrics that dataset actually computed
        aligned = [q.reindex(idx).fillna(0) for q in parts]
        stacked = pd.concat(aligned, keys=range(len(aligned)))
        return stacked.groupby(level=1).mean()
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
        aggregate=aggregate,
    )


def _method_language(name):
    try:
        from ..engine import registry
        return (registry.get(name).language or "python").lower()
    except Exception:
        return None


def render(tbl: BubbleTable, cmap: str | None = None, title: str | None = None,
           show_language: bool = True):
    """Render the paper-style table. Returns a matplotlib Figure.

    ``cmap`` overrides the FIRST family's colormap only (compatibility with the
    old single-colormap signature); the batch family stays green like the paper.
    ``show_language`` adds the paper's language chip (Py / R) beside each method.
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
            ax.add_patch(Rectangle((-1.15, n_rows - i - 1), total_w + 1.5, 1.0,
                                   facecolor="#ebebeb", edgecolor="none", zorder=0))

    # language chip per method (paper's metadata column), left of the grid
    if show_language:
        langs = {m: _method_language(m) for m in methods}
        if any(langs.values()):
            for i, m in enumerate(methods):
                lang = langs.get(m)
                if not lang:
                    continue
                label, colr = ("Py", "#3572A5") if lang.startswith("py") else ("R", "#777777")
                y = n_rows - i - 0.5
                ax.add_patch(Circle((-0.75, y), 0.24, facecolor=colr,
                                    edgecolor="none", zorder=3))
                ax.text(-0.75, y, label, ha="center", va="center", zorder=4,
                        fontsize=6.4, fontweight="bold", color="white")

    def rank_label(ax_, x, y, rank_pos, fill):
        lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        ax_.text(x, y, str(rank_pos), ha="center", va="center", zorder=5,
                 fontsize=7.2, fontweight="bold",
                 color="white" if lum < 0.55 else "#222222")

    def vbar(ax_, x, y_row, frac, fill, width):
        """Vertical bar whose LENGTH and colour both encode the score, as in the
        paper: full row height = best, a sliver = worst."""
        h = 0.10 + 0.66 * max(0.0, min(1.0, frac))
        bottom = y_row - 0.38
        ax_.add_patch(Rectangle((x - width / 2, bottom), width, h,
                                facecolor=fill, edgecolor="#444444",
                                linewidth=0.5, zorder=3))
        return bottom + h

    for x, fi, kind, name in col_meta:
        b = tbl.blocks[fi]
        mp = mappers[fi]
        as_bars = (kind == "overall") or (tbl.aggregate == "summary")
        if kind == "overall":
            colvals, colnorm = b.overall, style.minmax(b.overall.to_numpy())
            colnorm = pd.Series(colnorm, index=b.overall.index)
        else:
            colvals, colnorm = b.raw[name], b.norm[name]
        pos = colvals.rank(ascending=False, method="min")
        for i, m in enumerate(methods):
            v = colvals.loc[m]
            if pd.isna(v):
                continue
            y = n_rows - i - 0.5
            frac = float(colnorm.loc[m])
            fill = mp.to_rgba(frac)
            if as_bars:
                width = 0.64 if kind == "overall" else 0.55
                top = vbar(ax, x + 0.5, y, frac, fill, width)
                if pos.loc[m] <= 3:
                    if frac >= 0.30:            # label fits inside the bar
                        rank_label(ax, x + 0.5, top - 0.16, int(pos.loc[m]), fill)
                    else:                        # tiny bar: sit the label above it
                        rank_label(ax, x + 0.5, top + 0.14, int(pos.loc[m]),
                                   (1, 1, 1, 1))
            else:
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

    legend_bits = ["numbers = top-3 per column", "Rank bar: length + colour = family overall"]
    if tbl.aggregate == "summary":
        legend_bits.insert(0, "bar length + colour = rank averaged across datasets")
    else:
        legend_bits.insert(0, "circle colour = value (scaled per column)")
    ax.text(-0.35, -1.45, "  |  ".join(legend_bits), fontsize=7.4,
            color="#555555", ha="left", va="center")

    ax.set_xlim(-1.15, total_w + 0.35)
    ax.set_ylim(-1.95, n_rows + 1.1)
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
