"""Paper-style bubble table: methods as rows, metrics as columns, grouped by task.

Layout follows the benchmark paper's figures: each task family (DR & clustering,
batch correction) is a colour-banded block - blues and greens respectively -
preceded by an *Overall* column drawn as a horizontal bar whose length and
colour both encode the family score. With ``aggregate="summary"`` (several
datasets, the paper's panel c) every metric marker becomes such a bar too - the
value is then the metric's rank averaged across datasets; with a single dataset
the metric markers are circles (panel b). A missing value is drawn as a dash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import style
from .bar import BATCH_METRICS, CLUSTERING_METRICS
from .. import _compat
from .. import config as _config

# column order within each family, as in the paper's panels
FAMILIES = [
    ("DR and clustering", ["cLISI", "ARI", "ASW", "iASW", "iF1", "NMI"], "Blues"),
    ("Batch correction", ["ASW_batch", "GC", "iLISI", "kBET"], "Greens"),
]

#: glyph drawn in a cell whose metric was not computed for that method
NA_MARK = "\u2013"

#: second line of the Score legend title: the fill is scaled within each column
SCORE_SCALE_NOTE = "(scaled per column)"

#: legend line explaining the chips left of each row (drawn when show_language)
CHIP_KEY = ("Py / R = language \u00b7 L = consumes cell-type labels (supervised) "
            "\u00b7 ? = not a registry method")


@dataclass
class FamilyBlock:
    """One metric family of a :class:`BubbleTable` (a colour-banded column block).

    ``raw`` / ``norm`` / ``ranks`` are method x metric frames in the table's
    row order; ``ranks`` holds per-column max-ranks (``n`` = best, ties share
    the higher number - R's ``ties.method="max"``), ``norm`` per-column
    min-max values, ``overall`` the family's Overall score per method.
    """
    label: str                # header text
    cmap: str                 # matplotlib colormap name
    raw: pd.DataFrame         # raw value matrix (method x metric), row-aligned
    norm: pd.DataFrame        # per-column min-max scaled values
    ranks: pd.DataFrame       # per-column max-ranks (n = best)
    overall: pd.Series        # family overall score per method


@dataclass
class BubbleTable:
    """The numbers behind a bubble figure, as ``mtb.plot.build_table`` returns.

    Rows are ``methods``; the per-family matrices live in ``blocks``.
    ``mtb.plot.bubble`` draws the same figure from the long table.

    Attributes
    ----------
    methods : list of str
        Row order: best first, or the ``order`` methods first;
        ``methods.index(name) + 1`` is the row's position (1 = top).
    blocks : list of FamilyBlock
        One block per metric family present, in paper order, each with its
        own ``raw``, ``norm``, ``ranks`` and ``overall``.
    matrix : pandas.DataFrame
        Per-column min-max values (method x metric) of all families, in
        figure order; the same object as ``norm``.
    raw : pandas.DataFrame
        Unscaled matrix (method x metric) in figure order: metric means, or
        mean within-dataset max-ranks under ``"summary"``.
    overall : pandas.Series
        Combined Overall per method (mean of the family Overalls); it sets
        the row order unless ``order`` is given.
    aggregate : str
        ``"dataset"`` (metric markers are circles) or ``"summary"`` (bars,
        the paper's panel c).
    overall_basis : str
        Formula behind the family Overall under ``"summary"``: ``"rank"`` or
        ``"mean_overall"`` (the ``overall=`` of ``mtb.plot.bubble``).
    datasets : tuple of str
        Dataset ids in the frame, sorted; ``()`` without a ``dataset`` column.
    coverage : pandas.Series or None
        Number of datasets each method has rows in; ``"summary"`` only,
        else ``None``.
    method_datasets : dict or None
        ``{method: [dataset, ...]}`` from the frame; ``None`` without a
        ``dataset`` column.
    category : str or None
        The frame's single ``category`` value, else ``None`` (mixed or
        absent); drives the supervised ``L`` badge.
    needs_labels : dict or None
        ``{method: bool}`` from an optional ``needs_labels`` column, empty
        without it; overrides the registry for the ``L`` badge.
    na_cells : list of str or None
        One line per method with ``n/a`` cells, in row order: the text of
        the ``na="warn"`` warning.

    Examples
    --------
    >>> import multibench as mtb
    >>> tbl = mtb.plot.build_table(mtb.load_results("vertical", dataset="D11"))
    >>> tbl.methods[:3]                  # the best three methods
    >>> tbl.ranks.loc[tbl.methods[0]]    # the best method's max-rank per metric
    >>> tbl.blocks[0].overall            # first family's Overall per method

    Notes
    -----
    **Ranks.** ``ranks`` and ``FamilyBlock.ranks`` hold per-column
    max-ranks: ``n`` = best, ties share the higher number (R's
    ``ties.method="max"``). A method with no value in a column is NaN there,
    and the column's ``n`` counts only the scored methods. The figure's
    *Rank* legend counts 1 = best instead.

    **Ties.** Values that agree to 9 decimal places count as equal in
    ``ranks``, ``norm`` and the Overall scores. An iLISI of ``2.2e-16``
    therefore ties with ``0.0``.

    **Scaled values.** ``matrix`` and ``norm`` are per-column min-max values
    in [0, 1]; a constant (or all-NaN) column is all ones, as in the R code.

    **Blocks.** Paper order: DR and clustering (blues), batch correction
    (greens), then "Other" (purples) for any metric outside the two.

    **Drawing.** ``mtb.plot.bubble`` builds this table from the same long
    table and draws it.

    See Also
    --------
    mtb.plot.build_table : builds a table from a long table.
    mtb.plot.bubble : builds the table and draws the figure in one call.
    """
    methods: list             # row order: order= methods first, then best first
    blocks: list              # list[FamilyBlock], in FAMILIES order
    # kept for backward compatibility with callers that inspect the table
    matrix: pd.DataFrame      # all families' min-max scaled columns
    raw: pd.DataFrame
    overall: pd.Series        # combined overall used for the row order
    aggregate: str = "dataset"  # "dataset" -> circles; "summary" -> bars (paper c)
    overall_basis: str = "rank"  # formula behind the Overall bars (style.overall_by_basis)
    datasets: tuple = ()        # dataset ids present in the frame (sorted)
    coverage: object = None     # Series: datasets per method (summary only)
    method_datasets: object = None  # dict: method -> list of datasets it has rows in
    category: object = None     # the frame's single category, else None
    needs_labels: object = None  # dict: method -> bool ('needs_labels' column)
    na_cells: object = None     # list[str]: n/a report (see build_table(na=))

    @property
    def norm(self) -> pd.DataFrame:
        """All families' per-column min-max values, in figure order.

        An alias of ``matrix``: rows are ``methods``, columns the metrics in
        the order the figure draws them.

        Returns
        -------
        pandas.DataFrame
            Method x metric values scaled to [0, 1] per column.
        """
        return self.matrix

    @property
    def ranks(self) -> pd.DataFrame:
        """All families' per-column max-ranks, in figure order.

        Rows are ``methods``, columns the metrics in the order the figure
        draws them; the class Notes give the max-rank convention.

        Returns
        -------
        pandas.DataFrame
            Method x metric max-ranks: ``n`` = best, NaN where the method has
            no value for that metric.
        """
        return pd.concat([b.ranks for b in self.blocks], axis=1)


def _pivot(df: pd.DataFrame, aggregate: str) -> pd.DataFrame:
    """method x metric matrix: raw means (``"dataset"``) or the mean of the
    within-dataset max-ranks (``"summary"``; absent method or n/a cell =
    rank 0, each metric averaged over the datasets that computed it). The
    summary math lives in :mod:`multibench.plot.style` and is shared with
    ``plot.bar``."""
    if aggregate == "summary":
        return style.mean_rank_matrix(style.per_dataset_ranks(df))
    return df.pivot_table(index="method", columns="metric", values="value",
                          aggfunc="mean")


def _resolve(requested, available, kind: str, canon) -> list:
    """Map user-supplied selector names onto the frame's own spellings.

    Resolution order per requested name: exact match -> same canonical form
    (``catalog.canonical_metric`` / ``catalog.canonical_id``) -> case-
    insensitive match. The frame's spelling is always returned, so a frame
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
    """Compute the ranks, scores and row order behind a bubble figure.

    The numeric half of ``mtb.plot.bubble``: audit the ranks and the row
    order before (or instead of) drawing.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long table with ``method``, ``metric``, ``value`` columns; optional
        ``dataset``, ``category`` and ``needs_labels`` columns (Notes).
    metrics : list of str | None
        Metric codes to keep, in this order within each family; ``None`` =
        every metric in the frame.
    methods : list of str | None
        Methods (rows) to keep; ``None`` = every method in the frame.
    order : list of str | None
        Methods to put first, in this order; the rest follow best first.
        Reorders only - filter with ``methods``.
    aggregate : {"dataset", "summary"}
        ``"dataset"``: raw metric values of one dataset (several are averaged
        per method). ``"summary"``: within-dataset max-ranks averaged across
        datasets (the paper's panel c).
    require_complete : bool
        With ``aggregate="summary"``: keep only the methods present in every
        dataset.
    overall : {"rank", "mean_overall"}
        Formula for each family's Overall under ``aggregate="summary"``:
        ``"rank"`` (the paper's panel rule) or ``"mean_overall"`` (bar's
        default); see ``mtb.plot.bubble``.
    na : {"warn", "skip", "raise"}
        How to report ``n/a`` cells (a method lacking a metric): ``"warn"``,
        ``"skip"`` (silent, nothing is dropped) or ``"raise"``; also stored
        as ``na_cells``.

    Returns
    -------
    BubbleTable
        Read ``methods`` (row order: best first unless ``order`` is given),
        ``ranks`` (max-ranks, ``n`` = best) and ``overall``; all fields are
        under ``mtb.plot.BubbleTable``.

    Raises
    ------
    ValueError
        ``long_df`` lacks ``method`` / ``metric`` / ``value``, or ``metrics`` /
        ``methods`` / ``order`` names an absent value.
    ValueError
        Duplicate ``(method[, dataset], metric)`` rows, or an invalid
        ``aggregate`` / ``overall`` / ``na``.
    ValueError
        No method is complete under ``require_complete=True``, or ``n/a``
        cells under ``na="raise"``.

    Warns
    -----
    UserWarning
        ``n/a`` cells under ``na="warn"``, or several datasets under
        ``aggregate="dataset"``.
    UserWarning
        ``aggregate="summary"`` on an incomplete method x dataset matrix.
    UserWarning
        ``require_complete=True`` dropped methods; each is named with the
        datasets it lacks.
    UserWarning
        Under ``"summary"``: a dataset holds one method, or no method spans
        two datasets.

    Examples
    --------
    >>> import multibench as mtb
    >>> tbl = mtb.plot.build_table(mtb.load_results("vertical", dataset="D11"))
    >>> tbl.methods                      # rows, best first
    >>> tbl.ranks                        # max-ranks per metric (n = best)
    >>> multi = mtb.load_results("diagonal", dataset=["D24", "D25", "D28"])
    >>> tbl = mtb.plot.build_table(multi, aggregate="summary",
    ...                            require_complete=True)
    >>> tbl.coverage                     # datasets per method

    Notes
    -----
    **Missing-metric rule.** A method may lack a value for some metric (an
    ``n/a`` cell, drawn as a dash):

    - ``aggregate="dataset"``: the cell is simply absent. The family
      *Overall* averages the ranks of the metrics the method has (a method
      scored on 3 of 4 metrics is compared on those 3), and a column's ranks
      count only the methods scored in it.
    - ``aggregate="summary"``: the cell is rank 0 in that dataset (the
      paper's rule), in the metric columns and in the ``overall="rank"``
      Overall; ``overall="mean_overall"`` skips it.

    Neither rule is visible in the numbers, so ``na`` sets how it is
    reported; the warning reads like ``"YukiNet: DR and clustering Overall
    over 3 of 4 metrics (cLISI n/a)"``.

    **Row order.** The combined Overall is the mean of the family Overalls,
    sorted best first with a stable sort, so tied methods keep alphabetical
    order - the tie-break ``mtb.plot.bar`` uses. Ranks and scores always
    come from the whole filtered frame; ``order`` only moves rows.

    **require_complete.** One ``UserWarning`` names each dropped method and
    the datasets it lacks (``"require_complete=True dropped 1 method(s) ...:
    MyRandom (missing D52s)"``). It has no effect under
    ``aggregate="dataset"``.

    **A new dataset.** A summary compares methods only on the datasets they
    share. Under ``"summary"``, a ``UserWarning`` names a dataset that holds
    one method, and says so when no method spans two of the datasets. Plot
    such a dataset on its own, or score the same methods on it.

    **Input columns.** ``dataset`` groups rows for ``aggregate="summary"``
    (absent = one dataset) and is part of the duplicate-row key. A boolean
    ``needs_labels`` column overrides the registry's supervised badge per
    method (NaN = no override); a single ``category`` value makes the badge
    follow that category's variants. Rows whose ``metric`` is NaN are
    dropped.

    **Name matching.** ``metrics``, ``methods`` and ``order`` match the
    frame exactly, by canonical form (``"ari"`` -> ``"ARI"``) or
    case-insensitively; the frame's own spelling is kept. The family blocks
    always stay in paper order; metrics outside the two paper families form
    a neutral purple "Other" block.

    **Errors.** A frame that looks like ``mtb.evaluate``'s wide output gets a
    hint to convert it with ``mtb.to_long`` first; an unknown name gets a
    did-you-mean hint and the values present. Duplicate rows raise
    ``ValueError``: deduplicate, or name the variants distinctly (as
    ``mtb.sweep`` does). A method with both ``True`` and ``False``
    ``needs_labels`` rows, or an empty frame, also raises ``ValueError``.

    See Also
    --------
    mtb.plot.bubble : builds the table and draws the figure in one call.
    mtb.plot.BubbleTable : what is returned, field by field.
    """
    import warnings
    from ..data import catalog

    need = {"method", "metric", "value"}
    have = set(getattr(long_df, "columns", []))
    if not need.issubset(have):
        missing = sorted(need - have)
        hint = ""
        if "Value" in have or getattr(long_df, "index", None) is not None and getattr(long_df.index, "name", None) == "metric":
            hint = (" This looks like evaluate()'s output - convert it "
                    "with mtb.to_long(df, method=..., dataset=..., "
                    "category=...) first.")
        raise ValueError(
            f"bubble() needs a long table with columns "
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
    # resolved on the filtered frame: a method removed by methods= cannot
    # raise on an inconsistency the plot would never show
    needs_labels = _frame_needs_labels(df)

    keys = ["method", "metric"] + (["dataset"] if "dataset" in df.columns else [])
    dmask = df.duplicated(keys, keep=False)
    if dmask.any():
        raise ValueError(
            f"{int(dmask.sum())} duplicate rows for the same {tuple(keys)} "
            f"(e.g. {df[dmask][keys].drop_duplicates().head(3).to_dict('records')}); "
            "bubble() does not average them - deduplicate, or name the "
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
                # name what was dropped: a newly added method is the one
                # most likely to cover a single dataset
                dropped = cov[cov < n].sort_values()
                lacks = {m: [ds for ds, mat in parts.items() if m not in mat.index]
                         for m in dropped.index}
                warnings.warn(
                    _config.hint("require_complete=True", "--require-complete")
                    + f" dropped {len(dropped)} method(s) "
                    f"not present on all {n} datasets ({', '.join(map(str, parts))}): "
                    + ", ".join(f"{m} (missing {', '.join(map(str, lacks[m]))})"
                                for m in dropped.index)
                    + "; " + _config.hint("pass require_complete=False",
                                          "leave out --require-complete")
                    + " to keep them (a missing dataset then scores rank 0 under "
                    + _config.hint("overall='rank'", "--overall rank") + ").",
                    UserWarning, stacklevel=2)
                df = df[df["method"].isin(keep)]
                parts = style.per_dataset_ranks(df)
                cov = style.coverage(parts)
        for msg in style.coverage_warnings(
                parts, basis=overall,
                incomplete_fix=_config.hint(
                    "Pass require_complete=True to restrict to the complete "
                    "intersection.",
                    "Pass --require-complete to keep only the methods scored on "
                    "every dataset.")):
            warnings.warn(msg, UserWarning, stacklevel=2)
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
    # tie-break plot.bar uses
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
                    "method has and a column's ranks count only the methods "
                    "scored in it")
        from .. import config
        msg = ("n/a cells: " + "; ".join(na_cells) + f" - {rule}. " + config.hint(
            "Pass na='skip' to silence this, na='raise' to refuse an incomplete frame.",
            "Pass --na skip to silence this, --na raise to refuse an incomplete table."))
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
    variants of that category; without a category it falls back to 'any'.
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
    """Draw a :class:`BubbleTable` in the Shiny app's scIB knit-table format.

    Parameters
    ----------
    tbl : BubbleTable
        From :func:`build_table`.
    cmap : str, optional
        Matplotlib colormap overriding the first family's palette.
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

    Notes
    -----
    Follows scIB_knit_table.R: metric circles whose radius encodes the
    within-column rank (``0.85 * sqrt(rank / n)``) and whose fill encodes the
    min-max-scaled value; each family's Overall as a horizontal bar whose
    length is the min-max-scaled family score; alternating #DDDDDD row
    stripes; column titles slanted 30 degrees above the table; a Score
    colour-ramp legend (scaled per column) and a Rank circle-size legend
    underneath. With ``aggregate="summary"`` every metric becomes a
    horizontal bar too, as in the Shiny summary tables and the paper's
    summary panels (no error bars).
    No rank numbers are drawn - marker size carries the rank.
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
        """radius = 0.85 * sqrt(rank / n), n = methods scored in the column.

        n/a cells are excluded from both rank and n, so a column with 5 of 6
        methods scored ranks 1..5 and the best circle is still the full size.

        The sqrt is not min-max stretched to a fixed range: that would draw
        the worst of 3 methods as small as the worst of 28. Here the worst of
        3 is 0.49 and the smallest at n = 28 is ~0.16; a single method (or an
        all-tied column) is 0.85.
        """
        n = colvals.notna().sum()
        r = colvals.rank(ascending=True, method="max") / max(int(n), 1)
        rad = 0.85 * np.sqrt(r.to_numpy(dtype=float))
        return pd.Series(np.maximum(rad, 0.12), index=colvals.index)

    # ---- markers -----------------------------------------------------------
    n_na = 0          # NaN metric cells drawn as a dash (legend added if any)
    for x, fi, kind, name in col_meta:
        b, mp = tbl.blocks[fi], mappers[fi]
        if kind == "overall":
            vals = b.overall
            # bar length: family Overall min-max scaled across the rows
            length = style.minmax(vals.to_numpy())
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

    # ---- column titles: slanted 30deg above the table, with tick marks -----
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
    # the fill is min-max scaled within each column: the lightest fill is the
    # lowest value of its column, not zero. A second line keeps the legend
    # layout (one line this long would run into the "Low" label).
    ax.text(-0.9, ly - 0.5, SCORE_SCALE_NOTE, fontsize=6.4, va="center")
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
        # min(5, n) legend bubbles - a fixed 5 would show sizes that cannot
        # occur when fewer methods are plotted
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
    """Draw the paper-style bubble table: methods as rows, metrics as columns.

    Metrics are grouped into task families, each led by an *Overall* bar;
    rows run best first unless ``order`` is given. ``mtb.plot.build_table``
    returns the same numbers without drawing.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long table with ``method``, ``metric``, ``value`` and optional
        ``dataset`` columns, as from ``mtb.load_results``, ``mtb.to_long`` or
        the ``BatchResult.long`` property (more in Notes).
    metrics : list of str | None
        Metric codes to show, in this order within each family; ``None`` =
        every metric in the frame.
    methods : list of str | None
        Methods (rows) to show; ``None`` = every method in the frame.
    order : list of str | None
        Methods to put first, in this order; the rest follow best first.
        Reorders only - filter with ``methods``.
    aggregate : {"dataset", "summary"}
        ``"dataset"``: one dataset's values, drawn as circles. ``"summary"``:
        within-dataset ranks averaged across datasets, drawn as bars (the
        paper's panel c).
    cmap : str | None
        Matplotlib colormap for the first family; ``None`` = blues, greens
        and purples per family.
    title : str | None
        Figure title; ``None`` = no title.
    save : str or path-like | None
        File to write the figure to (tight bounding box); the suffix picks
        the format, e.g. ``.pdf``, ``.png``, ``.svg``.
    show_language : bool
        Draw the ``Py`` / ``R`` chip and the ``L`` (supervised) badge left of
        each row, plus a key line (Notes).
    require_complete : bool
        With ``aggregate="summary"``: keep only the methods present in every
        dataset.
    overall : {"rank", "mean_overall"}
        Across-dataset *Overall*: ``"rank"`` re-ranks mean ranks (missing
        dataset = rank 0); ``"mean_overall"`` averages per-dataset Overalls
        (missing dataset skipped). Applies under ``aggregate="summary"``.
    na : {"warn", "skip", "raise"}
        How to report ``n/a`` cells (a method lacking a metric): warn once,
        stay silent, or raise.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (one axes), already saved when ``save`` is given.

    Raises
    ------
    ValueError
        ``long_df`` lacks ``method`` / ``metric`` / ``value``, or ``metrics`` /
        ``methods`` / ``order`` names an absent value.
    ValueError
        Duplicate ``(method[, dataset], metric)`` rows, or an invalid
        ``aggregate`` / ``overall`` / ``na``.
    ValueError
        No method is complete under ``require_complete=True``, or ``n/a``
        cells under ``na="raise"``.

    Warns
    -----
    UserWarning
        ``n/a`` cells under ``na="warn"``, or several datasets under
        ``aggregate="dataset"``.
    UserWarning
        ``aggregate="summary"`` on an incomplete method x dataset matrix.
    UserWarning
        ``require_complete=True`` dropped methods; each is named with the
        datasets it lacks.
    UserWarning
        Under ``"summary"``: a dataset holds one method, or no method spans
        two datasets.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.load_results("vertical", dataset="D11")
    >>> fig = mtb.plot.bubble(df, metrics=["ARI", "NMI", "ASW"], title="D11",
    ...                       save="d11.pdf")
    >>> multi = mtb.load_results("diagonal", dataset=["D24", "D25", "D28"])
    >>> fig = mtb.plot.bubble(multi, aggregate="summary", require_complete=True)

    Notes
    -----
    **Reading the figure.** What each mark encodes:

    - Metric circle (``aggregate="dataset"``): radius = within-column rank,
      ``0.85 * sqrt(rank / n)`` with ``n`` = the methods scored in that column
      (n/a cells excluded), so the largest is the best; fill = the min-max
      scaled value on the family's colour ramp. The lightest fill is the
      lowest value of that column in this figure, not zero.
    - Metric bar (``aggregate="summary"``): length and fill = the min-max
      scaled mean rank.
    - *Score* legend: the colour ramp, labelled "(scaled per column)":
      ``Low`` and ``High`` are the lowest and highest value of each column.
    - Family *Overall* bar: length = the family score min-max scaled across
      the rows; fill = the score itself (the two differ only under
      ``overall="mean_overall"``).
    - Rows: ordered by the mean of the family Overall scores, best first.
      ``order`` moves rows but never changes a rank or a score.
    - *Rank* legend: 1 = best, whereas ``BubbleTable.ranks`` /
      ``FamilyBlock.ranks`` store max-ranks (``n`` = best).
    - Footnote: the Overall formula in use, the ``n/a`` rule when a dash is
      drawn, and the chip key.

    **Overall formulas.** ``overall=`` sets the family *Overall* under
    ``aggregate="summary"``; under ``"dataset"`` it is always ``minmax(mean
    over metrics of max-rank)``. The two can order methods differently on
    the same frame.

    - ``"rank"`` (bubble's default): ``minmax(mean over metrics of
      max-rank(mean over datasets of within-dataset max-rank))`` - the
      per-dataset ranks are averaged per metric, re-ranked across methods,
      averaged over metrics and min-max scaled. A method absent from a
      dataset scores rank 0 there (the paper's summary rule), which pulls it
      down.
    - ``"mean_overall"`` (bar's default): ``mean over datasets of
      minmax(mean over metrics of within-dataset max-rank)`` - each dataset
      gets its own min-max-scaled overall, and these are averaged; a dataset
      the method lacks is skipped.

    **Bubble and bar.** ``mtb.plot.bar`` uses the same formulas and the same
    tie-break (alphabetical within a tie). With ``aggregate="summary"``, the
    same ``overall=`` and the metrics of one family (e.g. against
    ``bar(group="clustering")``), both figures order methods identically.
    Across both families they can differ: bubble averages the family
    Overalls, bar scores all metrics together.

    **Missing cells.** A metric not computed for a method shows a dash
    (``n/a``) instead of a marker.

    - ``aggregate="dataset"``: the family Overall averages the metrics the
      method has; a column's ranks count only the scored methods.
    - ``aggregate="summary"``: the cell is rank 0 in that dataset for the
      metric bar (the paper's rule); the family Overall counts it as rank 0
      under ``overall="rank"`` and skips it under ``"mean_overall"``.

    ``na="warn"`` names each method in one warning per figure, e.g.
    ``"YukiNet: DR and clustering Overall over 3 of 4 metrics (cLISI n/a)"``.

    **Input columns.** Only ``method``, ``metric`` and ``value`` are
    required.

    - ``dataset`` - groups rows for ``aggregate="summary"`` (absent = one
      dataset) and is part of the duplicate-row key. A ``"dataset"`` figure
      that mixes datasets averages them per method and adds a dataset cue to
      each row label (``Name · D11`` or ``Name · 3 ds``).
    - ``category`` - a single value makes the ``L`` badge follow that
      category's variants.
    - ``needs_labels`` (bool) - overrides the registry's ``L`` badge per
      method; the only way to badge a method the registry does not know.
      NaN = no override.
    - Rows whose ``metric`` is NaN are dropped.

    To draw your own runs next to the stored table, concatenate the frames:
    ``pd.concat([mtb.load_results("vertical", dataset="D11"), res.long])``,
    with ``res`` from ``mtb.run_all``.

    **A new dataset.** The stored tables hold only the demo datasets, so
    rows from your own dataset have nothing stored to rank against. Plot
    that dataset on its own, or score your method on the demo dataset of
    its category and add that row. Under ``"summary"``, a ``UserWarning``
    names a dataset that holds one method, and says so when no method spans
    two datasets.

    **Name matching.** ``metrics``, ``methods`` and ``order`` match the
    frame exactly, by canonical form (``"ari"`` -> ``"ARI"``) or
    case-insensitively; the frame's own spelling is kept. The family blocks
    always stay in paper order.

    **Chips and badge** (``show_language``).

    - Chip: ``Py`` / ``R`` = the registry method's language; ``?`` = not a
      registry method (your own, a sweep variant).
    - ``L`` badge: the method consumes cell-type labels (supervised), so its
      clustering scores are not comparable with unsupervised rows. It
      follows the variants of the frame's single ``category`` (scMoMaT is
      supervised in mosaic only); without one, the registry's method-level
      flag.

    **Errors.** A frame that looks like ``mtb.evaluate``'s wide output gets a
    hint to convert it with ``mtb.to_long`` first; an unknown name gets a
    did-you-mean hint and the values present. Duplicate rows raise
    ``ValueError``: deduplicate, or name the variants distinctly (as
    ``mtb.sweep`` does). A method with both ``True`` and ``False``
    ``needs_labels`` rows, or an empty frame, also raises ``ValueError``.

    See Also
    --------
    mtb.plot.build_table : the numbers behind the figure, to audit first.
    mtb.plot.bar : one bar per method across datasets; same ``overall=`` formulas.
    mtb.to_long : reshape ``mtb.evaluate``'s scores into a long table.
    mtb.load_results : stored metric tables as a long table.
    """
    tbl = build_table(long_df, metrics=metrics, methods=methods, order=order,
                      aggregate=aggregate, require_complete=require_complete,
                      overall=overall, na=na)
    fig = render(tbl, cmap=cmap, title=title, show_language=show_language)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    return fig


# The `overall` parameter text is style.OVERALL_DOC, written out verbatim in
# the docstrings of bubble and bar (a runtime splice is invisible to the
# static docs build); tests/test_bubble.py and tests/test_bar.py pin parity.
# The two formulas are in both Notes and in style.overall_by_basis.

#: deprecated 0.2.x name of :func:`bubble` (DeprecationWarning; removed in 0.4)
plot_bubble = _compat.deprecated_alias("mtb.plot.plot_bubble", "mtb.plot.bubble", bubble)
