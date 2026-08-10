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
    spread: pd.DataFrame | None = None       # SD of per-dataset ranks (summary mode)
    overall_spread: pd.Series | None = None  # SD of per-dataset family overalls


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
        # return the PRE-alignment parts: whiskers must reflect spread over the
        # datasets a method actually ran in, not the structural zeros the bar
        # mean uses for absences
        return stacked.groupby(level=1).mean(), parts
    return df.pivot_table(index="method", columns="metric", values="value",
                          aggfunc="mean"), None


def build_table(long_df: pd.DataFrame, metrics=None, methods=None, order=None,
                aggregate: str = "dataset") -> BubbleTable:
    """Pivot tidy long results into per-family matrices plus a combined row order."""
    df = long_df.copy()
    if methods is not None:
        df = df[df["method"].isin(methods)]
    if metrics is not None:
        df = df[df["metric"].isin(metrics)]
    raw_all, _parts = _pivot(df, aggregate)

    blocks = []
    for label, fam_metrics, cmap in FAMILIES:
        cols = [m for m in fam_metrics if m in raw_all.columns]
        if not cols:
            continue
        raw = raw_all[cols]
        if raw.notna().sum().sum() == 0:
            continue
        spread = overall_spread = None
        if _parts is not None:
            # Whiskers: SD over the datasets a method ACTUALLY ran in (>=2),
            # per metric over the datasets that computed it. The bar's length
            # keeps the paper's absent-scores-0 convention; the whisker
            # deliberately does not, because a spread made of structural zeros
            # would measure absence, not performance variability.
            have = [q[[c for c in cols if c in q.columns]] for q in _parts]
            have = [q for q in have if q.shape[1]]
            cat_ranks = pd.concat(have, keys=range(len(have)))
            n_ds = cat_ranks.groupby(level=1).count()
            spread = (cat_ranks.groupby(level=1).std(ddof=0)
                      .where(n_ds >= 2).reindex(raw.index))
            per_ds_overall = pd.concat(
                [style.compute_overall(q) for q in have], axis=1)
            overall_spread = (per_ds_overall.std(axis=1, ddof=0)
                              .where(per_ds_overall.notna().sum(axis=1) >= 2)
                              .reindex(raw.index))
        blocks.append(FamilyBlock(
            label=label, cmap=cmap, raw=raw,
            norm=raw.apply(lambda col: style.minmax(col.to_numpy()), axis=0),
            ranks=raw.apply(lambda col: style.rank_max(col.to_numpy()), axis=0),
            overall=style.compute_overall(raw),
            spread=spread, overall_spread=overall_spread,
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
    """Render in the Shiny app's scIB knit-table format. Returns a Figure.

    Faithful to scIB_knit_table.R: metric CIRCLES whose radius encodes the
    within-column rank (sqrt-scaled to [0.15, 0.85]) and whose fill encodes the
    min-max-scaled value; each family's Overall as a HORIZONTAL bar whose length
    is the min-max-scaled mean rank; alternating #DDDDDD row stripes; column
    titles slanted 30 degrees above the table; a Score colour-ramp legend and a
    Rank circle-size legend underneath. With ``aggregate="summary"`` every
    metric becomes a horizontal bar too, exactly like the Shiny summary tables,
    and each bar carries an SD whisker across the datasets that computed it
    (methods or metrics present in a single dataset get no whisker).
    ``cmap`` overrides the first family's palette; no rank numbers are drawn -
    marker size carries the rank, as in the Shiny output.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    from matplotlib import cm, colors

    methods = tbl.methods
    n_rows = len(methods)
    ROW_H, R_MAX = 1.0, 0.5          # row height; circle max radius = row/2

    # ---- x layout: per family [overall][metrics...] with a family gap -----
    xs, col_meta = 0.0, []
    for fi, b in enumerate(tbl.blocks):
        col_meta.append((xs, fi, "overall", "Overall"))
        xs += 1.5                                     # bar column is wider
        for mname in b.raw.columns:
            col_meta.append((xs, fi, "metric", mname))
            xs += 1.1
        xs += 0.5
    total_w = xs - 0.5

    fig = Figure(figsize=(0.6 * total_w + 2.8, 0.42 * n_rows + 2.9))
    ax = fig.subplots()
    norm01 = colors.Normalize(vmin=0.0, vmax=1.0)
    mappers = {fi: cm.ScalarMappable(norm=norm01,
                                     cmap=(cmap if (fi == 0 and cmap) else b.cmap))
               for fi, b in enumerate(tbl.blocks)}

    # ---- alternating row stripes (#DDDDDD, like the Shiny output) ---------
    for i in range(n_rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((-1.3, n_rows - i - 1), total_w + 1.8, ROW_H,
                                   facecolor="#DDDDDD", edgecolor="none", zorder=0))

    # ---- language chips ----------------------------------------------------
    if show_language:
        for i, m in enumerate(methods):
            lang = _method_language(m)
            if not lang:
                continue
            label, colr = ("Py", "#3572A5") if lang.startswith("py") else ("R", "#777777")
            y = n_rows - i - 0.5
            ax.add_patch(Circle((-0.85, y), 0.24, facecolor=colr,
                                edgecolor="none", zorder=3))
            ax.text(-0.85, y, label, ha="center", va="center", zorder=4,
                    fontsize=6.4, fontweight="bold", color="white")

    def whisker(x_tip, y, half, x_min, x_max):
        """SD whisker centred on the bar tip, as error bars ON the bar.

        half is capped and both ends are clipped to the column's own
        [x_min, x_max] span so a large cross-dataset SD can never draw
        across neighbouring columns.
        """
        if not np.isfinite(half) or half <= 0:
            return
        half = min(float(half), 0.45)
        x0 = max(x_min, x_tip - half)
        x1 = min(x_max, x_tip + half)
        if x1 <= x0:
            return
        ax.plot([x0, x1], [y, y], color="#333333", linewidth=0.9,
                zorder=5, gid="whisker")
        for xc in (x0, x1):
            ax.plot([xc, xc], [y - 0.10, y + 0.10], color="#333333",
                    linewidth=0.9, zorder=5, gid="whisker")

    def rank_radius(colvals):
        """sqrt of the max-rank, rescaled to [0.15, 0.85] per column (scIB rule)."""
        r = colvals.rank(ascending=True, method="max") / colvals.notna().sum()
        r = np.sqrt(r.to_numpy(dtype=float))
        lo, hi = np.nanmin(r), np.nanmax(r)
        if not np.isfinite(lo) or hi == lo:
            # Constant column (e.g. a single method): everyone is tied at the
            # top rank, so draw the LARGEST bubble - mirrors minmax()'s
            # constant -> all-ones rule rather than collapsing to the minimum.
            return pd.Series(0.85, index=colvals.index)
        return pd.Series(0.15 + 0.70 * (r - lo) / (hi - lo), index=colvals.index)

    # ---- markers -----------------------------------------------------------
    for x, fi, kind, name in col_meta:
        b, mp = tbl.blocks[fi], mappers[fi]
        if kind == "overall":
            vals = b.overall
            length = style.minmax(vals.to_numpy())          # ranked overall, 0..1
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    continue
                y0 = n_rows - i - 1
                L = 1.24 * max(0.04, float(length[i]))
                ax.add_patch(Rectangle((x + 0.08, y0 + 0.12), L, ROW_H - 0.24,
                                       facecolor=mp.to_rgba(float(v)),
                                       edgecolor="#333333", linewidth=0.5, zorder=3))
                # No whisker on Overall: each dataset's overall is a WITHIN-
                # dataset relative score over that dataset's own method pool,
                # so a cross-dataset SD of it would not compare like with like.
        elif tbl.aggregate == "summary":
            vals, normv = b.raw[name], b.norm[name]
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    continue
                y0 = n_rows - i - 1
                L = 0.95 * max(0.04, float(normv.loc[m]))
                ax.add_patch(Rectangle((x + 0.06, y0 + 0.16), L, ROW_H - 0.32,
                                       facecolor=mp.to_rgba(float(normv.loc[m])),
                                       edgecolor="#333333", linewidth=0.4, zorder=3))
                if b.spread is not None and name in b.spread.columns:
                    sd = b.spread[name].loc[m]
                    lo = np.nanmin(vals.to_numpy()); hi = np.nanmax(vals.to_numpy())
                    if pd.notna(sd) and hi > lo:
                        whisker(x + 0.06 + L, y0 + ROW_H / 2,
                                0.95 * float(sd) / (hi - lo),
                                x + 0.06, x + 1.04)
        else:
            vals, normv = b.raw[name], b.norm[name]
            rad = rank_radius(vals)
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    continue
                y = n_rows - i - 0.5
                ax.add_patch(Circle((x + 0.55, y), R_MAX * float(rad.loc[m]) * 0.9,
                                    facecolor=mp.to_rgba(float(normv.loc[m])),
                                    edgecolor="#333333", linewidth=0.4, zorder=3))

    # ---- column titles: slanted 30deg ABOVE the table, with tick marks -----
    for x, fi, kind, name in col_meta:
        cx = x + (0.7 if kind == "overall" else 0.55)
        ax.plot([cx, cx], [n_rows + 0.05, n_rows + 0.22], color="#333333",
                linewidth=0.6, zorder=2)
        ax.text(cx - 0.05, n_rows + 0.30, name, rotation=30, ha="left",
                va="bottom", fontsize=8, rotation_mode="anchor")

    # ---- family bands above the slanted titles ----------------------------
    band_y = n_rows + 2.05
    for fi, b in enumerate(tbl.blocks):
        xs_f = [x for x, f, k, _ in col_meta if f == fi]
        x0, x1 = xs_f[0], xs_f[-1] + 1.1
        ax.add_patch(FancyBboxPatch((x0 + 0.05, band_y), x1 - x0 - 0.35, 0.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor=mappers[fi].to_rgba(0.30),
                                    edgecolor="none", zorder=2))
        ax.text((x0 + x1 - 0.3) / 2, band_y + 0.3, b.label, ha="center",
                va="center", fontsize=8.6, color="#1a1a1a", zorder=3)

    # ---- legends: Score ramps + Rank size, under the table (scIB layout) ---
    ly = -1.1
    ax.text(-0.9, ly, "Score", fontsize=8, fontweight="bold", va="center")
    for fi, b in enumerate(tbl.blocks):
        xoff = 1.2 + fi * 4.6
        for k in range(40):
            ax.add_patch(Rectangle((xoff + k * 0.07, ly - 0.28), 0.07, 0.56,
                                   facecolor=mappers[fi].to_rgba(k / 39),
                                   edgecolor="none", zorder=2))
        ax.text(xoff - 0.12, ly, "Low", fontsize=6.6, ha="right", va="center")
        ax.text(xoff + 40 * 0.07 + 0.12, ly, "High", fontsize=6.6, ha="left",
                va="center")
    if tbl.aggregate != "summary":
        ly2 = ly - 1.35
        ax.text(-0.9, ly2, "Rank", fontsize=8, fontweight="bold", va="center")
        rr = np.linspace(0.15, 0.85, 5)
        xoff = 1.5
        for k, r in enumerate(rr):
            ax.add_patch(Circle((xoff + k * 1.0, ly2), R_MAX * r * 0.9,
                                facecolor="#bbbbbb", edgecolor="#333333",
                                linewidth=0.4, zorder=2))
        ax.text(xoff - 0.65, ly2 - 0.62, str(n_rows), fontsize=6.6, ha="center")
        ax.text(xoff + 4.0, ly2 - 0.62, "1", fontsize=6.6, ha="center")

    ax.set_xlim(-1.35, total_w + 0.45)
    ax.set_ylim((-2.9 if tbl.aggregate != "summary" else -1.8), band_y + 1.0)
    ax.set_yticks([n_rows - i - 0.5 for i in range(n_rows)])
    ax.set_yticklabels(methods, fontsize=8.6)
    ax.set_xticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    if title:
        ax.set_title(title, fontsize=10, pad=10)
    fig.tight_layout()
    return fig


def plot_bubble(long_df, *, metrics=None, methods=None, order=None,
                aggregate="dataset", cmap=None, title=None, save=None):
    """Build + render the Shiny-format table from a long results frame."""
    tbl = build_table(long_df, metrics=metrics, methods=methods, order=order,
                      aggregate=aggregate)
    fig = render(tbl, cmap=cmap, title=title)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return fig
