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
from .. import _compat

# column order within each family, as in the paper's panels
FAMILIES = [
    ("DR and clustering", ["cLISI", "ARI", "ASW", "iASW", "iF1", "NMI"], "Blues"),
    ("Batch correction", ["ASW_batch", "GC", "iLISI", "kBET"], "Greens"),
]

#: glyph drawn in a cell whose metric was not computed for that method
NA_MARK = "\u2013"

#: legend line explaining the chips left of each row (drawn when show_language)
CHIP_KEY = ("Py / R = language \u00b7 L = consumes cell-type labels (supervised) "
            "\u00b7 ? = not a registry method")


@dataclass
class FamilyBlock:
    """One metric family of a :class:`BubbleTable` (a colour-banded column block).

    ``raw`` / ``norm`` / ``ranks`` are method x metric frames in the table's
    row order; ``ranks`` holds per-column MAX-ranks (``n`` = best, ties share
    the higher number - R's ``ties.method="max"``), ``norm`` per-column
    min-max values, ``overall`` the family's Overall score per method.
    """
    label: str                # header text
    cmap: str                 # matplotlib colormap name
    raw: pd.DataFrame         # raw value matrix (method x metric), row-aligned
    norm: pd.DataFrame        # per-column min-max normalized values
    ranks: pd.DataFrame       # per-column max-ranks (1 = best is HIGHEST rank number)
    overall: pd.Series        # family overall score per method


@dataclass
class BubbleTable:
    """The numbers behind a bubble figure (what :func:`build_table` returns).

    Rows are ``methods`` (best first); the per-family matrices live in
    ``blocks`` and are also exposed concatenated in figure order as
    ``matrix`` (= ``norm``, per-column min-max values), ``raw`` and
    ``ranks`` (per-column max-ranks, ``n`` = best; the legend's 1 = best is
    ``methods.index(name) + 1`` for the overall position).
    """
    methods: list             # row order, best first
    blocks: list              # list[FamilyBlock], in FAMILIES order
    # kept for backward compatibility with callers that inspect the table
    matrix: pd.DataFrame      # all families' normalized columns, concatenated
    raw: pd.DataFrame
    overall: pd.Series        # combined overall used for the row order
    aggregate: str = "dataset"  # "dataset" -> circles; "summary" -> bars (paper c)
    overall_basis: str = "rank"  # formula behind the Overall bars (style.OVERALL_DOC)
    datasets: tuple = ()        # dataset ids present in the frame (sorted)
    coverage: object = None     # Series: datasets per method (summary only)
    method_datasets: object = None  # dict: method -> list of datasets it has rows in
    category: object = None     # the one category the frame came from, else None (drives the supervised badge)
    needs_labels: object = None  # dict method -> bool from an optional 'needs_labels' column; overrides the registry lookup
    na_cells: object = None     # list[str]: per-method report of n/a cells (see build_table(na=))

    @property
    def norm(self) -> pd.DataFrame:
        """All families' per-column min-max values (method x metric), row order
        = ``methods``, columns in figure order; an alias of ``matrix``."""
        return self.matrix

    @property
    def ranks(self) -> pd.DataFrame:
        """All families' per-column max-ranks concatenated in figure order
        (row order = ``methods``).

        Max-rank: ``n`` = best, ties share the higher number (R
        ``ties.method="max"``); a method with no value in a column is NaN
        there and the column's ``n`` counts only the scored methods. The
        legend on the figure prints 1 = best; the overall position of a
        method is ``methods.index(name) + 1``.
        """
        return pd.concat([b.ranks for b in self.blocks], axis=1)


def _pivot(df: pd.DataFrame, aggregate: str) -> pd.DataFrame:
    """method x metric matrix: raw means (``"dataset"``) or the mean of the
    within-dataset max-ranks (``"summary"``; absent method = rank 0, each
    metric averaged over the datasets that computed it). The summary math
    lives in :mod:`multibench.plot.style` and is shared with ``plot.bar``."""
    if aggregate == "summary":
        return style.mean_rank_matrix(style.per_dataset_ranks(df))
    return df.pivot_table(index="method", columns="metric", values="value",
                          aggfunc="mean")


def _resolve(requested, available, kind: str, canon) -> list:
    """Map user-supplied selector names onto the frame's own spellings.

    Resolution order per requested name: exact match -> same canonical form
    (``catalog.canonical_metric`` / ``catalog.canonical_id``) -> case-
    insensitive match. The FRAME's spelling is always returned, so a frame
    built with ``to_long(method="Seurat v4")`` keeps its own label. Unknown
    names raise ``ValueError`` with a did-you-mean hint and the list of values
    present in the frame; a name listed twice (after canonicalisation) raises.
    """
    requested = [requested] if isinstance(requested, str) else list(requested)
    available = list(available)
    by_exact = set(available)
    by_canon = {}
    by_lower = {}
    for a in available:
        by_canon.setdefault(canon(a), a)          # frame spelling wins
        by_lower.setdefault(str(a).lower(), a)
    out, unknown = [], []
    for r in requested:
        if r in by_exact:
            out.append(r)
        elif canon(r) in by_canon:
            out.append(by_canon[canon(r)])
        elif str(r).lower() in by_lower:
            out.append(by_lower[str(r).lower()])
        else:
            unknown.append(r)
    if unknown:
        import difflib
        hints = {u: difflib.get_close_matches(str(u), [str(a) for a in available],
                                             n=1, cutoff=0.6) for u in unknown}
        did = "; ".join(f"{u!r}: did you mean {h[0]!r}?" for u, h in hints.items() if h)
        raise ValueError(
            f"unknown {kind}(s) {unknown}" + (f" ({did})" if did else "")
            + f"; available in this frame: {sorted(map(str, available))}")
    dup = sorted({x for x in out if out.count(x) > 1})
    if dup:
        raise ValueError(
            f"{kind}s listed more than once after canonicalisation: {dup}")
    return out


#: accepted values of ``build_table(na=)`` / ``bubble(na=)``
NA_POLICIES = ("skip", "warn", "raise")


def build_table(long_df: pd.DataFrame, *, metrics=None, methods=None, order=None,
                aggregate: str = "dataset", require_complete: bool = False,
                overall: str = "rank", na: str = "warn") -> BubbleTable:
    """Pivot tidy long results into per-family matrices plus a combined row order.

    Missing-metric rule. A method may lack a value for some metric (a cell
    ``n/a``, drawn as a dash). Under ``aggregate="dataset"`` that cell is
    simply absent: the family *Overall* averages the ranks of the metrics
    the method HAS (``mean`` skips NaN - a method scored on 3 of 4 metrics
    is compared on those 3), and a column's ranks count only the methods
    scored in that column. Under ``aggregate="summary"`` the paper's rule
    applies instead: an ``n/a`` cell within a dataset is rank 0 there. Both
    are silent by arithmetic, so ``na`` decides how loudly to say it.

    Parameters
    ----------
    long_df : DataFrame
        Tidy frame with at least ``method, metric, value``; an optional
        ``dataset`` column is used by ``aggregate="summary"`` and for the
        duplicate check; an optional boolean ``needs_labels`` column
        (add it to your own frame) overrides the registry's
        supervised badge per method. Rows whose ``metric`` is NaN are dropped.
    metrics : list of str, optional
        Metric codes to keep, drawn in THIS order within each family block
        (family blocks themselves stay in paper order). Spellings are
        canonicalised against the frame (``"ari"`` -> ``"ARI"``); an unknown
        code raises ``ValueError`` listing the metrics present.
    methods : list of str, optional
        Methods to keep (case/alias tolerant, resolved against the frame's
        own labels); unknown -> ``ValueError`` with a did-you-mean hint.
    order : list of str, optional
        Row order for the listed methods; every other method is appended
        best-first. Reorders ONLY - it never filters (use ``methods=``);
        unknown names raise ``ValueError``.
    aggregate : {"dataset", "summary"}
        ``"dataset"``: raw values (circles); several datasets in the frame
        are averaged per method with a ``UserWarning``. ``"summary"``: the
        paper's panel c - within-dataset max-ranks averaged across datasets
        (bars). Anything else -> ``ValueError``.
    require_complete : bool, keyword-only
        With ``aggregate="summary"``: keep only methods present in EVERY
        dataset of the frame (``ValueError`` if that leaves nothing). The
        drop is never silent: one ``UserWarning`` names each dropped method
        and the datasets it lacks (``"require_complete=True dropped 1
        method(s) ...: MyRandom (missing D52s)"``). Default ``False`` keeps
        all methods and warns when the matrix is incomplete.
    overall : {"rank", "mean_overall"}, keyword-only
        Formula for the per-family Overall under ``aggregate="summary"``
        (see :data:`multibench.plot.style.OVERALL_DOC`). Default ``"rank"``
        reproduces the paper; ``"mean_overall"`` is what ``plot.bar`` uses
        by default.
    na : {"warn", "skip", "raise"}, keyword-only
        What to do when a method has ``n/a`` cells (see the rule above).
        ``"warn"`` (default): one ``UserWarning`` naming each such method
        (``"YukiNet: DR and clustering Overall over 3 of 4 metrics (cLISI
        n/a)"``); ``"skip"``: silent; ``"raise"``: ``ValueError`` with the
        same text. The report is also stored on the table as ``na_cells``.

    Returns
    -------
    BubbleTable
        ``methods`` (row order), ``blocks`` (one ``FamilyBlock`` per family
        with ``raw / norm / ranks / overall``), ``matrix`` (= ``norm``),
        ``raw``, ``ranks`` (max-ranks, ``n`` = best, concatenated over the
        blocks in figure order), ``overall`` (combined, drives the row
        order), ``aggregate``, ``overall_basis``, ``datasets``, ``coverage``
        (datasets per method, summary only), ``method_datasets``,
        ``category``, ``needs_labels`` (per-method override dict) and
        ``na_cells``.

    Raises
    ------
    ValueError
        Missing columns, unknown selector names, duplicate
        ``(method[, dataset], metric)`` rows (they would otherwise be
        silently averaged), bad ``aggregate`` / ``overall`` / ``na``, an
        empty complete intersection, an inconsistent ``needs_labels`` column
        (both ``True`` and ``False`` rows for one method), or ``n/a`` cells
        under ``na="raise"``.
    """
    import warnings
    from ..data import catalog

    need = {"method", "metric", "value"}
    have = set(getattr(long_df, "columns", []))
    if not need.issubset(have):
        missing = sorted(need - have)
        hint = ""
        if "Value" in have or getattr(long_df, "index", None) is not None and getattr(long_df.index, "name", None) == "metric":
            hint = (" This looks like evaluate()'s wide output - convert it "
                    "with mtb.eval.to_long(df, method=..., dataset=..., "
                    "category=...) first.")
        raise ValueError(
            f"bubble() needs a tidy long frame with columns "
            f"['method', 'metric', 'value']; missing {missing}.{hint}")
    if aggregate not in ("dataset", "summary"):
        raise ValueError(
            f"aggregate must be 'dataset' or 'summary', got {aggregate!r}")
    if overall not in style.OVERALL_BASES:
        raise ValueError(
            f"overall must be one of {list(style.OVERALL_BASES)}, got {overall!r}")
    if na not in NA_POLICIES:
        raise ValueError(f"na must be one of {list(NA_POLICIES)}, got {na!r}")

    df = long_df.copy()
    df = df.dropna(subset=["metric"])
    if methods is not None:
        methods = _resolve(methods, df["method"].unique().tolist(), "method",
                           catalog.canonical_id)
        df = df[df["method"].isin(methods)]
    if metrics is not None:
        metrics = _resolve(metrics, df["metric"].unique().tolist(), "metric",
                           catalog.canonical_metric)
        df = df[df["metric"].isin(metrics)]
    if df.empty:
        raise ValueError("no rows left after applying methods=/metrics= filters")
    # resolved on the FILTERED frame: a method removed by methods= cannot
    # raise on an inconsistency the plot would never show
    needs_labels = _frame_needs_labels(df)

    keys = ["method", "metric"] + (["dataset"] if "dataset" in df.columns else [])
    dmask = df.duplicated(keys, keep=False)
    if dmask.any():
        raise ValueError(
            f"{int(dmask.sum())} duplicate rows for the same {tuple(keys)} "
            f"(e.g. {df[dmask][keys].drop_duplicates().head(3).to_dict('records')}); "
            "bubble() would silently average them - deduplicate, or name the "
            "variants distinctly (as sweep() does).")

    datasets = (tuple(sorted(map(str, df["dataset"].dropna().unique())))
                if "dataset" in df.columns else ())
    method_datasets = None
    if "dataset" in df.columns:
        method_datasets = {m: sorted(map(str, g["dataset"].dropna().unique()))
                           for m, g in df.groupby("method")}

    cov = None
    parts = None
    if aggregate == "summary":
        parts = style.per_dataset_ranks(df)
        n = len(parts)
        cov = style.coverage(parts)
        if require_complete:
            keep = cov[cov == n].index
            if len(keep) == 0:
                raise ValueError(
                    f"no method has results on all {n} datasets "
                    f"({', '.join(map(str, parts))}); coverage: "
                    f"{cov.to_dict()}")
            if len(keep) < len(cov):
                # the drop is what the caller asked for; the NAMES are not
                # something to hide (the method a student just added is the
                # one most likely to be on a single dataset)
                dropped = cov[cov < n].sort_values()
                lacks = {m: [ds for ds, mat in parts.items() if m not in mat.index]
                         for m in dropped.index}
                warnings.warn(
                    f"require_complete=True dropped {len(dropped)} method(s) "
                    f"not present on all {n} datasets ({', '.join(map(str, parts))}): "
                    + ", ".join(f"{m} (missing {', '.join(map(str, lacks[m]))})"
                                for m in dropped.index)
                    + "; pass require_complete=False to keep them (a missing "
                    "dataset then scores rank 0 under overall='rank').",
                    UserWarning, stacklevel=2)
                df = df[df["method"].isin(keep)]
                parts = style.per_dataset_ranks(df)
                cov = style.coverage(parts)
        elif n > 1 and (cov < n).any():
            part = cov[cov < n].sort_values()
            warnings.warn(
                f"summary ranks an incomplete method x dataset matrix ({n} "
                f"datasets): " + ", ".join(f"{m} seen in {c}/{n}" for m, c in part.items())
                + "; a method absent from a dataset scores rank 0 there under "
                "overall='rank' and is skipped under overall='mean_overall'. "
                "Pass require_complete=True to restrict to the complete "
                "intersection.", UserWarning, stacklevel=2)
        raw_all = style.mean_rank_matrix(parts)
    else:
        if len(datasets) > 1:
            warnings.warn(
                f"aggregate='dataset' but the frame holds {len(datasets)} datasets "
                f"({', '.join(datasets)}): values are averaged per method across "
                "them and rows mix datasets. Pass aggregate='summary' for the "
                "paper's rank-averaged panel, or filter to one dataset.",
                UserWarning, stacklevel=2)
        raw_all = df.pivot_table(index="method", columns="metric", values="value",
                                 aggfunc="mean")

    def _block(label, cmap, raw):
        if aggregate == "summary" and overall == "mean_overall":
            sub = {}
            for ds, p in parts.items():
                cols_p = [c for c in raw.columns if c in p.columns]
                if cols_p:
                    sub[ds] = p[cols_p]
            ov = style.overall_by_basis(sub, "mean_overall").reindex(raw.index)
        else:
            ov = style.compute_overall(raw)
        return FamilyBlock(
            label=label, cmap=cmap, raw=raw,
            norm=raw.apply(lambda col: style.minmax(col.to_numpy()), axis=0),
            ranks=raw.apply(lambda col: style.rank_max(col.to_numpy()), axis=0),
            overall=ov,
        )

    blocks = []
    known = {m for f in FAMILIES for m in f[1]}
    for label, fam_metrics, cmap in FAMILIES:
        if metrics is not None:
            cols = [m for m in metrics if m in fam_metrics and m in raw_all.columns]
        else:
            cols = [m for m in fam_metrics if m in raw_all.columns]
        if not cols:
            continue
        raw = raw_all[cols]
        if raw.notna().sum().sum() == 0:
            continue
        blocks.append(_block(label, cmap, raw))
    # anything not in a known family still gets shown, neutrally coloured
    if metrics is not None:
        leftover = [c for c in metrics if c not in known and c in raw_all.columns]
    else:
        leftover = [c for c in raw_all.columns if c not in known]
    if leftover:
        blocks.append(_block("Other", "Purples", raw_all[leftover]))
    if not blocks:
        raise ValueError("no known metrics found in long_df")

    combined = pd.concat([b.overall for b in blocks], axis=1).mean(axis=1)
    # stable sort: tied methods keep index (alphabetical) order, the same
    # tie-break plot.bar uses, so the two figures agree under the same overall=
    ranked = combined.sort_values(ascending=False, kind="mergesort").index.tolist()
    if order is not None:
        order = _resolve(order, raw_all.index.tolist(), "method", catalog.canonical_id)
        idx = order + [m for m in ranked if m not in order]
    else:
        idx = ranked

    for b in blocks:
        b.raw = b.raw.reindex(idx)
        b.norm = b.norm.reindex(idx)
        b.ranks = b.ranks.reindex(idx)
        b.overall = b.overall.reindex(idx)

    na_cells = _na_report(blocks, parts, aggregate)
    if na_cells and na != "skip":
        if aggregate == "summary":
            rule = ("an n/a cell within a dataset is rank 0 there (the paper's "
                    "summary rule)")
        else:
            rule = ("the family Overall averages the ranks of the metrics a "
                    "method HAS and a column's ranks count only the methods "
                    "scored in it")
        msg = ("n/a cells: " + "; ".join(na_cells) + f" - {rule}. Pass na='skip' "
               "to silence this, na='raise' to refuse an incomplete frame.")
        if na == "raise":
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    return BubbleTable(
        methods=idx, blocks=blocks,
        matrix=pd.concat([b.norm for b in blocks], axis=1),
        raw=pd.concat([b.raw for b in blocks], axis=1),
        overall=combined.loc[idx],
        aggregate=aggregate,
        overall_basis=overall,
        datasets=datasets,
        category=_frame_category(long_df),
        coverage=cov,
        method_datasets=method_datasets,
        needs_labels=needs_labels,
        na_cells=na_cells,
    )


def _na_report(blocks, parts, aggregate: str) -> list:
    """One line per method with n/a cells, in row order.

    ``aggregate="dataset"``: ``"<method>: <family> Overall over k of n metrics
    (<codes> n/a)"`` per family block. ``"summary"``: ``"<method>: <codes> n/a
    in <dataset> -> rank 0 there"`` from the per-dataset rank matrices.
    """
    lines = []
    if aggregate == "summary":
        for ds, mat in (parts or {}).items():
            for m in mat.index:
                na = [c for c in mat.columns if pd.isna(mat.loc[m, c])]
                if na:
                    lines.append(f"{m}: {', '.join(map(str, na))} n/a in {ds} -> rank 0 there")
        return lines
    for b in blocks:
        n = b.raw.shape[1]
        whole = []
        for m in b.raw.index:
            na = [c for c in b.raw.columns if pd.isna(b.raw.loc[m, c])]
            if na and len(na) < n:
                lines.append(f"{m}: {b.label} Overall over {n - len(na)} of {n} "
                             f"metrics ({', '.join(map(str, na))} n/a)")
            elif na:
                whole.append(str(m))
        if whole:
            # a family absent for a method (a frame mixing single-batch and
            # multi-batch datasets): no Overall bar there, one line for all
            lines.append(f"no {b.label} metric (no {b.label} Overall) for: "
                         f"{', '.join(whole)}")
    return lines


def _frame_needs_labels(df) -> dict:
    """``{method: bool}`` from an optional ``needs_labels`` column.

    NaN means "no override"; a method carrying both ``True`` and ``False``
    rows raises ``ValueError`` naming it, because the badge cannot be both.
    """
    if "needs_labels" not in df.columns:
        return {}
    out = {}
    for m, g in df.groupby("method"):
        vals = set(bool(v) for v in g["needs_labels"].dropna())
        if len(vals) > 1:
            raise ValueError(
                f"needs_labels differs between rows of method {m!r}: "
                f"{sorted(vals)} - a method is supervised or not; fix the column")
        if vals:
            out[str(m)] = vals.pop()
    return out


def _method_language(name):
    try:
        from ..engine import registry
        return (registry.get(name).language or "python").lower()
    except Exception:
        return None


def _frame_category(long_df) -> str | None:
    """The one category a long frame holds, else None (mixed or absent)."""
    try:
        cats = long_df["category"].dropna().unique()
        return str(cats[0]) if len(cats) == 1 else None
    except Exception:
        return None


def _single_category(tbl) -> str | None:
    """The one category a table was built from, or None when mixed/unknown.

    Recorded on the table at build time: ``tbl.raw`` is the pivoted
    method x metric matrix and carries no category column.
    """
    return getattr(tbl, "category", None)


def _method_needs_labels(name, category=None) -> bool:
    """Does this method consume cell-type labels - for ``category`` when given.

    A method can be supervised in one category and not another (scMoMaT: mosaic
    yes, vertical no), so the badge on a single-category figure must follow the
    variants of THAT category; without a category it falls back to 'any'.
    """
    try:
        from ..engine import registry
        spec = registry.get(name)
        variants = getattr(spec, "variants", None) or []
        if category and variants:
            vs = [v for v in variants if v.when.get("category") == category]
            if vs:
                return any(v.needs_labels for v in vs)
        return bool(spec.needs_labels)
    except Exception:
        return False


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
    exactly as in the paper's summary panels (no error bars there, and none
    here).
    ``cmap`` overrides the first family's palette; no rank numbers are drawn -
    marker size carries the rank, as in the Shiny output.

    Parameters
    ----------
    tbl : BubbleTable
        From :func:`build_table`.
    cmap : str, optional
        Matplotlib colormap overriding the FIRST family's palette.
    title : str, optional
        Figure title.
    show_language : bool
        Draw the Py / R / ? chips and the ``L`` badge left of each row plus
        the one-line key (:data:`CHIP_KEY`) under the legends; the badge
        honours ``tbl.needs_labels`` (an explicit override) before the
        registry lookup for ``tbl.category``.

    Returns
    -------
    matplotlib.figure.Figure
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
                # not a registry method (user's own method, a sweep variant,
                # a result-dir token): say so with '?' rather than a blank
                label, colr = "?", "#aaaaaa"
            elif lang.startswith("py"):
                label, colr = "Py", "#3572A5"
            else:
                label, colr = "R", "#777777"
            y = n_rows - i - 0.5
            ax.add_patch(Circle((-0.85, y), 0.24, facecolor=colr,
                                edgecolor="none", zorder=3))
            ax.text(-0.85, y, label, ha="center", va="center", zorder=4,
                    fontsize=6.4, fontweight="bold", color="white")
            # an explicit needs_labels column beats the registry (the only
            # way to badge a method the registry does not know)
            ov = (getattr(tbl, "needs_labels", None) or {}).get(m)
            if ov if ov is not None else _method_needs_labels(m, _single_category(tbl)):
                # supervised: the method consumed cell-type labels, so its
                # clustering scores are not comparable with unsupervised rows
                ax.add_patch(Circle((-0.38, y), 0.20, facecolor="#b8860b",
                                    edgecolor="none", zorder=3))
                ax.text(-0.38, y, "L", ha="center", va="center", zorder=4,
                        fontsize=5.8, fontweight="bold", color="white")

    def rank_radius(colvals):
        """radius = 0.85 * sqrt(rank / n), n = methods SCORED in the column.

        n/a cells are excluded from both rank and n, so a column with 5 of 6
        methods scored ranks 1..5 and the best circle is still the full size.

        No min-max stretch: stretching the sqrt to a fixed [0.15, 0.85] made
        the worst of THREE methods as small as the worst of twenty-eight -
        with few methods there is no need for the full dynamic range. Under
        this mapping the worst of 3 is a clearly visible 0.49, while at
        n = 28 the smallest is ~0.16, i.e. the familiar large-field look.
        A single method (or an all-tied column) is rank fraction 1 -> 0.85.
        """
        n = colvals.notna().sum()
        r = colvals.rank(ascending=True, method="max") / max(int(n), 1)
        rad = 0.85 * np.sqrt(r.to_numpy(dtype=float))
        return pd.Series(np.maximum(rad, 0.12), index=colvals.index)
        return pd.Series(0.15 + 0.70 * (r - lo) / (hi - lo), index=colvals.index)

    # ---- markers -----------------------------------------------------------
    n_na = 0          # NaN metric cells drawn as a dash (legend added if any)
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
                _floor = min(0.30, 1.2 / max(n_rows, 1))
                L = 1.24 * max(_floor, float(length[i]))
                ax.add_patch(Rectangle((x + 0.08, y0 + 0.12), L, ROW_H - 0.24,
                                       facecolor=mp.to_rgba(float(v)),
                                       edgecolor="#333333", linewidth=0.5, zorder=3))

        elif tbl.aggregate == "summary":
            vals, normv = b.raw[name], b.norm[name]
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    ax.text(x + 0.5, n_rows - i - 0.5, NA_MARK, color="#999999",
                            ha="center", va="center", fontsize=7, zorder=3)
                    n_na += 1
                    continue
                y0 = n_rows - i - 1
                # Adaptive floor: with n methods the shortest bar is at least
                # ~1.2/n of the column (capped at 0.30) - visible in a small
                # comparison, indistinguishable from the paper's look at scale.
                _floor = min(0.30, 1.2 / max(n_rows, 1))
                L = 0.95 * max(_floor, float(normv.loc[m]))
                ax.add_patch(Rectangle((x + 0.06, y0 + 0.16), L, ROW_H - 0.32,
                                       facecolor=mp.to_rgba(float(normv.loc[m])),
                                       edgecolor="#333333", linewidth=0.4, zorder=3))
        else:
            vals, normv = b.raw[name], b.norm[name]
            rad = rank_radius(vals)
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    ax.text(x + 0.55, n_rows - i - 0.5, NA_MARK, color="#999999",
                            ha="center", va="center", fontsize=7, zorder=3)
                    n_na += 1
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
        # One legend bubble per rank actually present - a fixed 5 would show
        # sizes that cannot occur when fewer methods are plotted.
        n_bub = max(1, min(5, n_rows))
        # legend sizes follow the plot's own mapping: 0.85 * sqrt(rank / n)
        _fracs = (np.linspace(1, n_rows, n_bub) / n_rows) if n_bub > 1 else np.array([1.0])
        rr = np.maximum(0.85 * np.sqrt(_fracs), 0.12)
        xoff = 1.5
        for k, r in enumerate(rr):
            ax.add_patch(Circle((xoff + k * 1.0, ly2), R_MAX * r * 0.9,
                                facecolor="#bbbbbb", edgecolor="#333333",
                                linewidth=0.4, zorder=2))
        if n_bub > 1:
            ax.text(xoff - 0.65, ly2 - 0.62, str(n_rows), fontsize=6.6,
                    ha="center")
        ax.text(xoff + (n_bub - 1) * 1.0, ly2 - 0.62, "1", fontsize=6.6,
                ha="center")

    # ---- how the rows were ordered (so bubble and bar can be reconciled) ---
    y_bottom = -2.9 if tbl.aggregate != "summary" else -1.8
    if tbl.aggregate == "summary":
        basis = getattr(tbl, "overall_basis", "rank")
        what = ("minmax of re-ranked mean ranks" if basis == "rank"
                else "mean of per-dataset overall")
        note = (f"Overall = overall='{basis}' ({what}); "
                "rows ordered by mean of family Overall")
    else:
        note = ("Overall = minmax(mean metric rank); "
                "rows ordered by mean of family Overall")
    y_text = y_bottom - 0.55
    ax.text(-0.9, y_text, note, fontsize=6.4, ha="left", va="center",
            color="#666666")
    if n_na:
        y_text -= 0.45
        ax.text(-0.9, y_text, f"{NA_MARK} = n/a (metric not computed for that method"
                + ("; Overall averages the metrics the method has)"
                   if tbl.aggregate != "summary" else "; rank 0 in that dataset)"),
                fontsize=6.4, ha="left", va="center", color="#666666")
    if show_language:
        # the chips left of the row labels are otherwise unexplained on paper
        y_text -= 0.45
        ax.text(-0.9, y_text, CHIP_KEY, fontsize=6.4, ha="left", va="center",
                color="#666666")
    y_bottom = y_text - 0.35

    # ---- row labels: add a dataset cue when one figure mixes datasets -----
    labels = list(methods)
    md = getattr(tbl, "method_datasets", None) or {}
    if tbl.aggregate == "dataset" and len(getattr(tbl, "datasets", ())) > 1 and md:
        labels = []
        for m in methods:
            ds = md.get(m, [])
            labels.append(f"{m} \u00b7 {ds[0]}" if len(ds) == 1
                          else f"{m} \u00b7 {len(ds)} ds")

    ax.set_xlim(-1.35, total_w + 0.45)
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
    return fig


def bubble(long_df, *, metrics=None, methods=None, order=None,
           aggregate="dataset", cmap=None, title=None, save=None,
           show_language=True, require_complete=False, overall="rank",
           na="warn"):
    """Paper-style bubble table of a tidy long results frame (build + render).

    Methods are rows (best first), metrics are columns grouped into the
    benchmark's task families, each family preceded by an *Overall* bar.
    ``mtb.plot.bubble(df)`` is the one call most users need; the two halves are
    available as ``mtb.plot.build_table`` (numbers) and ``mtb.plot.render``
    (figure) when you want to audit the ranks before drawing.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Tidy frame with columns ``method, metric, value`` (what
        :func:`multibench.to_long`, :func:`multibench.load_results` and the
        ``BatchResult.long`` property return); ``dataset``, ``category``
        and a boolean ``needs_labels`` column are optional - ``dataset`` is
        needed for ``aggregate="summary"``, ``category`` drives the
        supervised badge, ``needs_labels`` overrides it. Rows whose
        ``metric`` is NaN are dropped. Concatenate frames
        (``pd.concat([published, mine])``) to compare your method with the
        benchmark's.
    metrics : list of str, optional
        Metric codes to show, drawn in THIS order within each family block
        (the blocks themselves keep the paper's order: DR & clustering, then
        batch correction, then "Other"). Case/alias tolerant (``"ari"`` ->
        ``"ARI"``); an unknown code raises ``ValueError`` naming the metrics
        present in the frame. Default: every metric in the frame.
    methods : list of str, optional
        Methods to show, resolved against the frame's own labels (exact,
        canonical or case-insensitive match); unknown -> ``ValueError`` with
        a did-you-mean hint. Default: all.
    order : list of str, optional
        Row order for the listed methods; any method not listed is appended
        below them, best-first. This only REORDERS - it never filters rows
        (use ``methods=``) - and ranks are still computed on the whole
        (filtered) frame. Unknown names raise ``ValueError``.
    aggregate : {"dataset", "summary"}
        ``"dataset"`` (default): one dataset's raw values; metric markers are
        circles. If the frame holds several datasets their values are averaged
        per method and a ``UserWarning`` says so (row labels then carry the
        dataset id). ``"summary"``: the paper's across-dataset panel -
        within-dataset max-ranks averaged over datasets, drawn as bars; a
        ``UserWarning`` lists methods missing from some datasets. Any other
        value raises ``ValueError``.
    cmap : str, optional
        Matplotlib colormap name overriding the FIRST family's palette
        (default blues / greens / purples per family).
    title : str, optional
        Figure title.
    save : str or path-like, optional
        If given, ``fig.savefig(save, bbox_inches="tight")`` - the suffix
        picks the format (``.pdf``, ``.png``, ``.svg``).
    show_language : bool
        Draw the Py / R language chip left of each row (``?`` for methods not
        in the registry, e.g. your own) and an ``L`` badge for methods that
        consume cell-type labels (supervised; their scores are not comparable
        with unsupervised rows), plus a one-line key under the legends
        explaining the chips. The badge follows the variants of the frame's
        single ``category`` (scMoMaT is supervised in mosaic only); an
        optional boolean ``needs_labels`` column in the frame (add it to
        your own frame) overrides the registry lookup
        per method, which is the only way to badge a method the registry
        does not know. Default ``True``.
    require_complete : bool
        With ``aggregate="summary"``: restrict to methods present in EVERY
        dataset instead of warning about the incomplete matrix; a
        ``UserWarning`` names each dropped method and the datasets it lacks,
        and ``ValueError`` is raised if no method is complete. Default
        ``False``.
    na : {"warn", "skip", "raise"}
        How to report ``n/a`` cells (a method lacking a metric). Under
        ``aggregate="dataset"`` the family Overall averages the metrics the
        method HAS and a column's ranks count only the scored methods; under
        ``"summary"`` an ``n/a`` cell is rank 0 in that dataset. ``"warn"``
        (default) says so once per figure in a ``UserWarning`` naming each
        method (``"YukiNet: DR and clustering Overall over 3 of 4 metrics
        (cLISI n/a)"``), ``"skip"`` is silent, ``"raise"`` refuses.
{OVERALL_DOC}

    Encoding
    --------
    * metric circle (``aggregate="dataset"``): RADIUS = within-column rank
      fraction (largest = best, ``0.85 * sqrt(rank / n)`` with ``n`` = the
      methods SCORED in that column, n/a cells excluded), FILL = min-max
      scaled value along the family's colour ramp.
    * metric bar (``aggregate="summary"``): LENGTH and fill = min-max scaled
      mean rank.
    * family *Overall* bar: length and fill = the family score (see
      ``overall``), min-max scaled across the rows. Rows are ordered by the
      MEAN of the family Overall scores. The *Rank* legend counts 1 = best,
      whereas ``BubbleTable.ranks`` / ``FamilyBlock.ranks`` store max-ranks
      (``n`` = best).
    * a cell whose metric was not computed for that method shows a dash
      (``n/a`` in the legend) instead of a marker; that metric is left out
      of the method's Overall (``aggregate="dataset"``) or counts as rank 0
      (``"summary"``) - see ``na``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        Missing columns (with a hint when the frame looks like
        ``evaluate()``'s wide output), unknown ``metrics`` / ``methods`` /
        ``order`` names, duplicate ``(method[, dataset], metric)`` rows (they
        would be silently averaged - deduplicate or name variants distinctly),
        bad ``aggregate`` / ``overall``, or an empty complete intersection.

    Examples
    --------
    >>> pub = mtb.load_results("vertical", dataset="D11")
    >>> mine = mtb.to_long(metrics, method="MyMethod", dataset="D11", category="vertical")
    >>> fig = mtb.plot.bubble(pd.concat([pub, mine]), metrics=["ARI", "NMI", "ASW"],
    ...                       title="D11 + MyMethod", save="d11.pdf")
    """
    tbl = build_table(long_df, metrics=metrics, methods=methods, order=order,
                      aggregate=aggregate, require_complete=require_complete,
                      overall=overall, na=na)
    fig = render(tbl, cmap=cmap, title=title, show_language=show_language)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return fig


# the `overall` parameter is documented once, in style.OVERALL_DOC, and
# spliced into every docstring that accepts it (bubble, build_table, bar)
bubble.__doc__ = bubble.__doc__.replace("{OVERALL_DOC}", style.OVERALL_DOC)

#: deprecated 0.2.x name of :func:`bubble` (DeprecationWarning; removed in 0.4)
plot_bubble = _compat.deprecated_alias("mtb.plot.plot_bubble", "mtb.plot.bubble", bubble)
