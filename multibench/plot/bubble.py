"""Paper-style bubble table: methods as rows, metrics as columns, grouped by task.

Layout follows the benchmark paper's figures: each task family (clustering,
batch correction, classification) is a colour-banded block - blues, greens and
pinks respectively - preceded by an *Overall* column drawn as a horizontal bar whose length and
colour both encode the family score. The metric markers are circles (the
paper's panel b). A missing value is drawn as a dash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import style
from .. import _text
from .._text import _and_list

from .._data.catalog import FAMILY_METRICS as _FAMILY_METRICS

# column order within each family: catalog.metrics()'s
FAMILIES = [
    ("Clustering", list(_FAMILY_METRICS["clustering"]), "Blues"),
    ("Batch correction", list(_FAMILY_METRICS["batch"]), "Greens"),
    ("Classification", list(_FAMILY_METRICS["classification"]), style.PINKS),
]

#: the suffix that tells your rows from stored results of the same method
THIS_RUN = " (this run)"

#: glyph drawn in a cell whose metric was not computed for that method
NA_MARK = "\u2013"

#: second line of the Score legend title: the fill is scaled within each column
SCORE_SCALE_NOTE = "(scaled per column)"

#: the legend of the chips left of the rows
LANGUAGE_NOTE = "Py / R: the method's language"

#: fill of a column whose rows all hold the same value (and of every marker in
#: a one-method figure): the colour ramp would draw it at its "High" end
CONSTANT_FILL = "#bfbfbf"


@dataclass
class FamilyBlock:
    """One metric family of a :class:`BubbleTable` (a colour-banded column block).

    ``raw`` / ``norm`` are method x metric frames in the table's row order:
    the metric means and their per-column min-max values; ``overall`` is the
    family's Overall score per method.
    """
    label: str
    cmap: str
    raw: pd.DataFrame
    norm: pd.DataFrame
    overall: pd.Series


@dataclass
class BubbleTable:
    """The numbers behind a bubble figure (internal: ``mtb.plot.bubble`` builds
    and draws it).

    Attributes
    ----------
    methods : list of str
        Row order: best first, as ``mtb.rank`` orders them.
    blocks : list of FamilyBlock
        One block per metric family present, in paper order: clustering
        (blues), batch correction (greens), classification (pinks), then
        "Other" (purples) for metrics outside the three.
    datasets : tuple of str
        Dataset ids in the frame, sorted; ``()`` without a ``dataset`` column.
    method_datasets : dict or None
        ``{method: [dataset, ...]}`` from the frame; ``None`` without a
        ``dataset`` column.
    """
    methods: list
    blocks: list
    datasets: tuple = ()
    method_datasets: object = None


def _and_distinct(values) -> str:
    """``_and_list`` of the distinct ``str`` forms of ``values``, in first-seen order."""
    return _and_list(dict.fromkeys(map(str, values)))


def _unknown_names(unknown, available) -> str:
    """The error for ``unknown`` names, then the names the table has."""
    one = len(set(map(str, unknown))) == 1
    have = sorted({str(a) for a in available})
    return (f"{_and_distinct(unknown)} {'is' if one else 'are'} not in the table. "
            + (f"The table has {_and_distinct(have)}." if have else "The table is empty."))


def _resolve(requested, available, canon) -> list:
    """The frame's own spelling of each requested name: matched exactly, else by
    ``canon`` form in any case. A name given twice counts once; an unknown name
    raises."""
    requested = [requested] if isinstance(requested, str) else list(requested)
    available = list(available)

    def key(x):
        return str(canon(x)).lower()
    by_key = {}
    for a in available:
        by_key.setdefault(key(a), a)          # frame spelling wins
    out, unknown = [], []
    for r in requested:
        hit = r if r in available else by_key.get(key(r))
        if hit is None:
            unknown.append(r)
        else:
            out.append(hit)
    if unknown:
        raise ValueError(_unknown_names(unknown, available))
    return list(dict.fromkeys(out))


def _expand_families(metrics, available) -> list:
    """``metrics`` with each family word (``"clustering"``, ``"batch"``,
    ``"classification"``, ``"all"``, as ``mtb.evaluate`` takes them) replaced by
    the frame's codes of that family, in the figure's column order; other
    names are kept for ``_resolve``. A family the frame has no code of
    raises ``ValueError``."""
    from .._data import catalog
    paper = [c for _, codes, _ in FAMILIES for c in codes]

    def in_figure_order(names):
        return sorted(names, key=lambda a: (
            paper.index(catalog._canonical_metric(a))
            if catalog._canonical_metric(a) in paper else len(paper), str(a)))

    requested = [metrics] if isinstance(metrics, str) else list(metrics)
    available = [a for a in available if a is not None]
    out = []
    for r in requested:
        word = str(r).strip().lower()
        if word == "all":
            out += in_figure_order(available)
        elif word in _FAMILY_METRICS:
            codes = _FAMILY_METRICS[word]
            got = [a for a in available if catalog._canonical_metric(a) in codes]
            if not got:
                raise ValueError(f"The table has no {word} metrics. It has "
                                 f"{_and_distinct(sorted(map(str, available)))}.")
            out += in_figure_order(got)
        else:
            out.append(r)
    return list(dict.fromkeys(out))


def _need_long(long_df: pd.DataFrame, fn: str) -> None:
    """``ValueError`` when ``long_df`` is not a long table (method, metric, value)."""
    absent = sorted({"method", "metric", "value"} - set(long_df.columns))
    if absent:
        raise ValueError(f"{fn}() needs a long table with columns "
                         f"['method', 'metric', 'value']; missing {absent}.")


def _build_table(long_df: pd.DataFrame, *, metrics=None, methods=None) -> BubbleTable:
    """The :class:`BubbleTable` of ``long_df``, as :func:`bubble` draws it."""
    from .._data import catalog

    _need_long(long_df, "bubble")
    df = long_df.dropna(subset=["metric"])
    if methods is not None:
        methods = _resolve(methods, df["method"].unique().tolist(), catalog._canonical_id)
        df = df[df["method"].isin(methods)]
    if metrics is not None:
        names = df["metric"].unique().tolist()
        metrics = _resolve(_expand_families(metrics, names), names, catalog._canonical_metric)
        df = df[df["metric"].isin(metrics)]
    if df.empty:
        raise ValueError("no rows left after applying methods=/metrics= filters"
                         if methods is not None or metrics is not None
                         else "The table has no rows.")

    _refuse_repeats(df)
    score = style.overall(df)       # refuses your rows on another dataset

    has_ds = "dataset" in df.columns
    datasets = tuple(sorted(map(str, df["dataset"].dropna().unique()))) if has_ds else ()
    method_datasets = ({m: sorted(map(str, g["dataset"].dropna().unique()))
                        for m, g in df.groupby("method")} if has_ds else None)
    if len(datasets) > 1:
        # the per-dataset view, only to see which datasets share methods; the
        # figure itself shows raw values
        for msg in style.dataset_mode_warnings(style.dataset_tables(df)):
            _text.warn(msg, UserWarning)
    raw_all = df.pivot_table(index="method", columns="metric", values="value",
                             aggfunc="mean")
    backend = style.backend_warning(df)
    if backend:
        _text.warn(backend, UserWarning)
    # rows without a score (no metric the ranking uses) come last
    rows = list(score.index) + sorted(set(raw_all.index) - set(score.index), key=str)

    def block(label, cmap, cols):
        raw = raw_all[cols].reindex(rows)
        return FamilyBlock(
            label=label, cmap=cmap, raw=raw,
            norm=raw.apply(lambda col: style.minmax(col.to_numpy()), axis=0),
            overall=style.overall(df, cols).reindex(rows))

    blocks = []
    for label, codes, cmap in FAMILIES:
        cols = [m for m in (codes if metrics is None else metrics)
                if m in codes and m in raw_all.columns]
        if cols:
            blocks.append(block(label, cmap, cols))
    known = {m for _, codes, _ in FAMILIES for m in codes}
    leftover = [c for c in (raw_all.columns if metrics is None else metrics)
                if c not in known and c in raw_all.columns]
    if leftover:
        blocks.append(block("Other", "Purples", leftover))
    if not blocks:
        raise ValueError("The table has no values.")

    # a family some rows lack (classification of your rows next to stored rows
    # without it) is drawn, but counts in no row's Overall; its n/a cells need
    # no words
    missing = _na_missing([b for b in blocks if b.raw.notna().any(axis=1).all()] or blocks)
    if missing:
        _text.warn(_na_message(missing), UserWarning)

    raw = pd.concat([b.raw for b in blocks], axis=1)
    # a column only one method has a value in is not "equal for every method"
    msg = _no_comparison_message(rows, style.constant_columns(raw.loc[:, raw.notna().sum() > 1]))
    if msg:
        _text.warn(msg, UserWarning)

    return BubbleTable(methods=rows, blocks=blocks, datasets=datasets,
                       method_datasets=method_datasets)


def _no_comparison_message(methods, constant: dict) -> str | None:
    """The warning for grey fills: one method, or the ``constant`` columns;
    ``None`` when there is none."""
    if len(methods) == 1:
        return (f"Only one method, {methods[0]}: nothing to rank it against, so the "
                f"fills are grey. Plot it with other methods.")
    if not constant:
        return None
    one = len(constant) == 1
    values = _and_list([f"{c} is {v:.3f}" for c, v in constant.items()])
    return (f"{values} for every method. {'That column is' if one else 'Those columns are'} "
            f"grey. Leave {'it' if one else 'them'} out with metrics=.")


def _na_missing(blocks) -> list:
    """The n/a cells per row, in row order, as ``[(method, codes)]``; codes in
    column order."""
    out = []
    for m in blocks[0].raw.index:
        codes = [c for b in blocks for c in b.raw.columns if pd.isna(b.raw.loc[m, c])]
        if codes:
            out.append((m, codes))
    return out


def _na_message(missing: list) -> str:
    """The n/a warning: the rows with n/a cells, then what to do about it."""
    have = "has" if len(missing) == 1 else "have"
    codes = list(dict.fromkeys(c for _, cs in missing for c in cs))
    return (f"{_and_distinct(m for m, _ in missing)} {have} no {_and_list(codes, 'or')}. "
            f"Pass metrics= without {_and_list(codes)} to compare on the same metrics.")


def _by_id(df):
    """``df`` with each method name that is a package method's id or alias in
    that id (``"cobolt"`` is Cobolt), so a repeat is found whatever its case."""
    from ..engine import registry

    def canonical(name):
        try:
            return registry.check_method(str(name))
        except KeyError:
            return name
    return df.assign(method=df["method"].map(canonical)) if "method" in df.columns else df


def _user_repeats(df, keys):
    """Rows that repeat ``keys`` where one of the repeats is not a stored row
    (your rows next to the stored table, or your rows twice); repeats among
    stored rows are left alone."""
    df = _by_id(df)
    repeated = df.duplicated(keys, keep=False)
    if not repeated.any() or "source" not in df.columns:
        return repeated
    stored = df["source"].isin(style.STORED_SOURCES)
    all_stored = stored.groupby([df[k] for k in keys]).transform("all")
    return repeated & ~all_stored


def _refuse_repeats(df) -> None:
    """Raise for rows that repeat a method, metric and dataset (:func:`_user_repeats`)."""
    keys = ["method", "metric"] + (["dataset"] if "dataset" in df.columns else [])
    repeated = _user_repeats(df, keys)
    if repeated.any():
        raise ValueError(_duplicate_rows_message(df[repeated], keys))


def _duplicate_rows_message(dup, keys) -> str:
    """The error for rows that repeat ``keys``: the fix when your rows collide
    with stored ones, else one example repeat."""
    dup = _by_id(dup)
    stored = (dup["source"].isin(style.STORED_SOURCES)
              if "source" in dup.columns else pd.Series(False, index=dup.index))
    mixed = dup.assign(_st=stored.to_numpy()).groupby(keys)["_st"].agg(
        lambda s: s.any() and not s.all())
    mixed = mixed[mixed].index.to_frame(index=False)
    if len(mixed):
        where = f" on {_and_distinct(mixed['dataset'])}" if "dataset" in keys else ""
        return (f"Your rows and the stored results both have "
                f"{_and_distinct(mixed['method'])}{where}. Give your rows another "
                f"name, e.g. rows.assign(method=rows.method + {THIS_RUN!r}).")
    what = _and_list(keys)
    example = ", ".join(str(dup[k].iloc[0]) for k in keys)
    return (f"{len(dup)} rows repeat the same {what}, for example {example}. The figure "
            "does not average them. Remove the repeats, or give each version its own "
            "method name.")


def _method_language(name):
    """``"python"`` or ``"r"`` (``"StabMap (this run)"`` and ``"scJoint (ep40)"``
    are the method's, as evaluate reads them); None for a name the registry
    does not know."""
    from ..engine import registry
    own = registry.method_of(name)
    return None if own is None else registry.get(own).language.lower()


def render(tbl: BubbleTable, title: str | None = None):
    """Draw a :class:`BubbleTable` in the Shiny app's scIB knit-table format.

    Parameters
    ----------
    tbl : BubbleTable
        From :func:`_build_table`.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The layout follows scIB_knit_table.R; the Notes of ``bubble`` say what
    each mark encodes.
    """
    import matplotlib.style
    gaps = [0.0] * (len(tbl.blocks) - 1)
    # matplotlib's own defaults, whatever the session's rcParams are:
    # importing scib (mtb.evaluate) switches them to seaborn's "talk" sizes
    with matplotlib.style.context("default"):
        for _ in range(4):
            fig = _draw(tbl, title, gaps)
            short = _fit_pills(fig, fig.axes[0])
            if not any(s > 1e-6 for s in short):
                break
            # two widened pills would touch: more room between those families
            gaps = [g + s for g, s in zip(gaps, short)]
    return fig


#: row height; circle max radius = row/2
ROW_H, R_MAX = 1.0, 0.5
#: the Score ramp of each family under the table: first x, step, width
RAMP_X0, RAMP_STEP, RAMP_W = 1.2, 4.6, 40 * 0.07
#: room for the "High" label after a ramp: about 0.5 units in a tall figure, up
#: to 0.7 in a one-row figure with a title
HIGH_W = 0.7
#: x of the first Rank legend circle; the next ones are 1.0 apart
RANK_X0 = 1.5


def _draw(tbl, title, gaps):
    """The body of :func:`render`; ``gaps`` widens the space between families
    (one data-unit amount per pair of neighbouring families)."""
    from matplotlib import cm, colors
    from matplotlib.patches import Rectangle
    from ._figure import BubbleFigure

    methods = tbl.methods
    n_rows = len(methods)
    layout, total_w = _x_layout(tbl.blocks, gaps)

    # the legends under a table of few metrics are wider than the table, so the
    # axes end at the right edge of the widest one
    legend_right = RAMP_X0 + (len(tbl.blocks) - 1) * RAMP_STEP + RAMP_W + 0.12 + HIGH_W
    # at most 5 legend circles, sized as the plot's own: 0.85 * sqrt(rank / n)
    n_legend = min(5, n_rows)
    legend_radii = np.maximum(
        0.85 * np.sqrt(np.linspace(1, n_rows, n_legend) / n_rows), 0.12)
    legend_right = max(legend_right,
                       RANK_X0 + (n_legend - 1) * 1.0 + R_MAX * legend_radii[-1] * 0.9 + 0.15)
    x_right = max(total_w + 0.45, legend_right)

    # the width follows the x-range: 0.6 inch per unit past the table's own
    fig = BubbleFigure(figsize=(0.6 * (x_right - 0.45) + 2.8, 0.42 * n_rows + 2.9))
    ax = fig.subplots()
    norm01 = colors.Normalize(vmin=0.0, vmax=1.0)
    mappers = [cm.ScalarMappable(norm=norm01, cmap=style.colormap(b.cmap))
               for b in tbl.blocks]

    # alternating row stripes, like the Shiny output
    for i in range(0, n_rows, 2):
        ax.add_patch(Rectangle((-1.3, n_rows - i - 1), total_w + 1.8, ROW_H,
                               facecolor="#DDDDDD", edgecolor="none", zorder=0))
    langs = [_method_language(m) for m in methods]
    _draw_chips(ax, langs)
    _draw_cells(ax, tbl, layout, mappers)
    band_y = n_rows + 2.05
    _draw_headers(ax, tbl, layout, mappers, band_y)
    y_bottom = _draw_legends(ax, mappers, legend_radii, unknown=not all(langs))

    # a figure that mixes datasets gets a dataset cue in each row label
    labels = list(methods)
    if len(tbl.datasets) > 1:
        datasets_of = tbl.method_datasets
        labels = [f"{m} · {datasets_of[m][0]}" if len(datasets_of[m]) == 1
                  else f"{m} · {len(datasets_of[m])} ds" for m in methods]

    ax.set_xlim(-1.35, x_right)
    ax.set_ylim(y_bottom, band_y + 1.0)
    ax.set_yticks([n_rows - i - 0.5 for i in range(n_rows)])
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    if title:
        ax.set_title(title, fontsize=10, pad=10)
    fig.tight_layout()
    _fit_width(fig, ax)
    return fig


def _x_layout(blocks, gaps):
    """``[(x, family index, "overall" or "metric", column name)]`` and the
    table width: per family [overall][metrics...], with a gap between families."""
    x_next, layout = 0.0, []
    for fi, b in enumerate(blocks):
        if fi:
            x_next += 0.5 + gaps[fi - 1]
        layout.append((x_next, fi, "overall", "Overall"))
        x_next += 1.5                                 # bar column is wider
        for mname in b.raw.columns:
            layout.append((x_next, fi, "metric", mname))
            x_next += 1.1
    return layout, x_next


def _draw_chips(ax, langs):
    """The language chip left of each row (``langs``: :func:`_method_language` per row)."""
    from matplotlib.patches import Circle

    n_rows = len(langs)
    for i, lang in enumerate(langs):
        label, colour = (("?", "#aaaaaa") if not lang else ("Py", "#3572A5")
                         if lang.startswith("py") else ("R", "#777777"))
        y = n_rows - i - 0.5
        ax.add_patch(Circle((-0.85, y), 0.24, facecolor=colour,
                            edgecolor="none", zorder=3))
        ax.text(-0.85, y, label, ha="center", va="center", zorder=4,
                fontsize=6.4, fontweight="bold", color="white")


def _draw_cells(ax, tbl, layout, mappers):
    """The Overall bars and the metric circles (a dash for a missing value)."""
    from matplotlib import colors
    from matplotlib.patches import Circle, Rectangle

    methods = tbl.methods
    n_rows = len(methods)

    def rank_radius(colvals):
        """radius = 0.85 * sqrt(rank / n), n = methods scored in the column."""
        r = style.rank_max(colvals.to_numpy()) / colvals.notna().sum()
        return pd.Series(np.maximum(0.85 * np.sqrt(r), 0.12), index=colvals.index)

    # a column whose rows all hold one value gets the neutral fill
    constant = style.constant_columns(pd.concat([b.raw for b in tbl.blocks], axis=1))
    flat_overall = {fi for fi, b in enumerate(tbl.blocks)
                    if style.constant_columns(b.overall.to_frame("o"))}

    def fill(mapper, value, flat):
        return colors.to_rgba(CONSTANT_FILL) if flat else mapper.to_rgba(value)

    floor = min(0.30, 1.2 / n_rows)      # shortest bar, visible with few methods

    for x, fi, kind, name in layout:
        b, mapper = tbl.blocks[fi], mappers[fi]
        if kind == "overall":
            # bar length: family Overall min-max scaled across the rows
            length = style.minmax(b.overall.to_numpy())
            for i, m in enumerate(methods):
                v = b.overall.loc[m]
                if pd.isna(v):
                    continue
                bar_len = 1.24 * max(floor, float(length[i]))
                ax.add_patch(Rectangle((x + 0.08, n_rows - i - 1 + 0.12), bar_len, ROW_H - 0.24,
                                       facecolor=fill(mapper, float(v), fi in flat_overall),
                                       edgecolor="#333333", linewidth=0.5, zorder=3))
            continue
        vals, scaled_values = b.raw[name], b.norm[name]
        radii = rank_radius(vals)
        for i, m in enumerate(methods):
            y = n_rows - i - 0.5
            if pd.isna(vals.loc[m]):
                ax.text(x + 0.55, y, NA_MARK, color="#999999",
                        ha="center", va="center", fontsize=7, zorder=3)
                continue
            ax.add_patch(Circle((x + 0.55, y), R_MAX * float(radii.loc[m]) * 0.9,
                                facecolor=fill(mapper, float(scaled_values.loc[m]),
                                               name in constant),
                                edgecolor="#333333", linewidth=0.4, zorder=3))


def _draw_headers(ax, tbl, layout, mappers, band_y):
    """The column names over the table and a family pill over each block."""
    from matplotlib.patches import FancyBboxPatch

    n_rows = len(tbl.methods)
    for x, fi, kind, name in layout:
        tick_x = x + (0.7 if kind == "overall" else 0.55)
        ax.plot([tick_x, tick_x], [n_rows + 0.05, n_rows + 0.22], color="#333333",
                linewidth=0.6, zorder=2)
        ax.text(tick_x - 0.05, n_rows + 0.30, name, rotation=30, ha="left",
                va="bottom", fontsize=8, rotation_mode="anchor")
    for fi, b in enumerate(tbl.blocks):
        family_xs = [x for x, f, _, _ in layout if f == fi]
        x0, x1 = family_xs[0], family_xs[-1] + 1.1
        # gid="family": _fit_pills finds the pill and its header
        ax.add_patch(FancyBboxPatch((x0 + 0.05, band_y), x1 - x0 - 0.35, 0.6,
                                    boxstyle=f"round,pad={PILL_BOX_PAD},rounding_size=0.18",
                                    facecolor=mappers[fi].to_rgba(0.30),
                                    edgecolor="none", zorder=2, gid="family",
                                    clip_on=False))
        ax.text((x0 + x1 - 0.3) / 2, band_y + 0.3, b.label, ha="center",
                va="center", fontsize=8.6, color="#1a1a1a", zorder=3, gid="family")


def _draw_legends(ax, mappers, legend_radii, *, unknown) -> float:
    """The Score ramps, the Rank sizes and the language note under the table,
    as in the scIB layout (each artist with gid="legend"); returns the lowest y."""
    from matplotlib.patches import Circle, Rectangle

    score_y = -1.1
    ax.text(-0.9, score_y, "Score", fontsize=8, fontweight="bold", va="center",
            gid="legend")
    ax.text(-0.9, score_y - 0.5, SCORE_SCALE_NOTE, fontsize=6.4, va="center",
            gid="legend")
    for fi, mapper in enumerate(mappers):
        xoff = RAMP_X0 + fi * RAMP_STEP
        for k in range(40):
            ax.add_patch(Rectangle((xoff + k * 0.07, score_y - 0.28), 0.07, 0.56,
                                   facecolor=mapper.to_rgba(k / 39),
                                   edgecolor="none", zorder=2, gid="legend"))
        ax.text(xoff - 0.12, score_y, "Low", fontsize=6.6, ha="right", va="center",
                gid="legend")
        ax.text(xoff + RAMP_W + 0.12, score_y, "High", fontsize=6.6, ha="left",
                va="center", gid="legend")
    rank_y = score_y - 1.35
    ax.text(-0.9, rank_y, "Rank", fontsize=8, fontweight="bold", va="center",
            gid="legend")
    n_legend = len(legend_radii)
    for k, r in enumerate(legend_radii):
        ax.add_patch(Circle((RANK_X0 + k * 1.0, rank_y), R_MAX * r * 0.9,
                            facecolor="#bbbbbb", edgecolor="#333333",
                            linewidth=0.4, zorder=2, gid="legend"))
    if n_legend > 1:
        ax.text(RANK_X0, rank_y - 0.62, "worst", fontsize=6.6,
                ha="center", gid="legend")
    ax.text(RANK_X0 + (n_legend - 1) * 1.0, rank_y - 0.62, "best", fontsize=6.6,
            ha="center", gid="legend")

    # room under the legends: the Rank labels, or the Score legend's note
    y_bottom = -3.4
    # what the chips left of the rows mean
    ax.text(-0.9, y_bottom - 0.05, LANGUAGE_NOTE + ("; ?: not a package method"
                                                    if unknown else ""),
            fontsize=6.4, va="center", gid="legend")
    return y_bottom - 0.6


#: data units the rounded box of a family pill reaches past its bounds
PILL_BOX_PAD = 0.02
#: least room, in points, between a family header and each end of its pill,
#: and between two neighbouring pills
PILL_PAD_PT, PILL_GAP_PT = 3.0, 2.0


def _fit_pills(fig, ax, pad_pt=PILL_PAD_PT, gap_pt=PILL_GAP_PT) -> list:
    """Widen, about its centre, each family pill narrower than its header; return
    per pair of neighbouring pills the data units their gap lacks (0 when none).
    It measures the drawn headers, so call it on a finished layout."""
    fig.draw_without_rendering()
    pills = [p for p in ax.patches if p.get_gid() == "family"]
    heads = [t for t in ax.texts if t.get_gid() == "family"]
    inv = ax.transData.inverted()
    per_px = abs(inv.transform((1.0, 0.0))[0] - inv.transform((0.0, 0.0))[0])
    pad = pad_pt * fig.dpi / 72.0 * per_px
    for pill, head in zip(pills, heads):
        ext = head.get_window_extent()
        tx0 = inv.transform((ext.x0, ext.y0))[0]
        tx1 = inv.transform((ext.x1, ext.y1))[0]
        x, w = pill.get_x(), pill.get_width()
        c = x + w / 2
        half = max(c - (tx0 - pad), (tx1 + pad) - c) - PILL_BOX_PAD
        if 2 * half > w:
            pill.set_x(c - half)
            pill.set_width(2 * half)
    gap = gap_pt * fig.dpi / 72.0 * per_px
    return [max(0.0, float((a.get_x() + a.get_width() + PILL_BOX_PAD) + gap
                - (b.get_x() - PILL_BOX_PAD)))
            for a, b in zip(pills, pills[1:])]


def _fit_width(fig, ax, pad_pt=3.0, rounds=4):
    """Widen ``fig`` until the texts of ``ax`` fit; the height stays. With the
    axes' equal aspect, ``tight_layout`` alone leaves the legends and row labels
    of a figure of few metrics past the figure edges."""
    pad = pad_pt * fig.dpi / 72.0
    for _ in range(rounds):
        fig.draw_without_rendering()
        arts = [t for t in ax.texts if t.get_visible() and t.get_text()]
        arts += [t for t in ax.get_yticklabels() if t.get_text()]
        if ax.title.get_text():
            arts.append(ax.title)
        boxes = [a.get_window_extent() for a in arts]
        over = (max(0.0, fig.bbox.x0 + pad - min(b.x0 for b in boxes))
                + max(0.0, max(b.x1 for b in boxes) - (fig.bbox.x1 - pad)))
        if over < 1.0:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + over / fig.dpi, h)
        fig.tight_layout()


def bubble(long_df, *, metrics=None, methods=None, title=None, save=None):
    """Draw the paper's bubble table: methods as rows, metrics as columns.

    Metrics are grouped into families, each led by an *Overall* bar. Rows run
    best first, in the order of ``mtb.rank(category, long_df=long_df,
    metrics=...)`` with the same metrics.

    Parameters
    ----------
    long_df : pandas.DataFrame
        A long table (``method``, ``metric``, ``value`` and optional
        ``dataset``), as ``mtb.load_results``, ``mtb.evaluate`` and
        ``RunResult.long`` return; stored and your own rows can be concatenated.
    metrics : str, list of str | None
        A family (``"clustering"``, ``"batch"``, ``"classification"``,
        ``"all"``), a code, or codes in their column order; ``None`` = every
        metric in the frame.
    methods : list of str | None
        Methods (rows) to show; ``None`` = every method in the frame.
    title : str | None
        Figure title; ``None`` = ``"<category> <dataset>"`` when the table
        holds one of each; ``""`` = no title.
    save : str or path-like | None
        File to write the figure to; the suffix picks the format (``.pdf``,
        ``.png``, ``.svg``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        ``long_df`` lacks ``method`` / ``metric`` / ``value``, repeats a
        ``(method, dataset, metric)`` row, holds your rows on another dataset
        than the stored rows, or ``metrics`` / ``methods`` names a value it
        does not hold.

    Warns
    -----
    UserWarning
        ``n/a`` cells, several datasets averaged per method, a column with the
        same value in every row, or a figure of one method.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.load_results("vertical", dataset="D11")
    >>> fig = mtb.plot.bubble(df, metrics="clustering", save="d11.pdf")
    """
    tbl = _build_table(long_df, metrics=metrics, methods=methods)
    if title is None and isinstance(long_df, pd.DataFrame):
        one = [str(long_df[c].iloc[0]) for c in ("category", "dataset")
               if c in long_df.columns and long_df[c].nunique() == 1]
        title = " ".join(one) if len(one) == 2 else None
    fig = render(tbl, title=title or None)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return fig
