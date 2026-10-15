"""Paper-style bubble table: methods as rows, metrics as columns, grouped by task.

Layout follows the benchmark paper's figures: each task family (DR & clustering,
batch correction, classification) is a colour-banded block - blues, greens and
pinks respectively - preceded by an *Overall* column drawn as a horizontal bar whose length and
colour both encode the family score. With ``aggregate="summary"`` (several
datasets, the paper's panel c) every metric marker becomes such a bar too - the
value is then the metric's rank averaged across datasets; with a single dataset
the metric markers are circles (panel b). A missing value is drawn as a dash.
"""
from __future__ import annotations

from dataclasses import dataclass

import functools
import numpy as np
import pandas as pd

from . import style
from .. import _compat
from .. import config as _config
from ..config import _and_list

# column order within each family, as in the paper's panels
FAMILIES = [
    ("DR and clustering", ["cLISI", "ARI", "ASW", "iASW", "iF1", "NMI"], "Blues"),
    ("Batch correction", ["ASW_batch", "GC", "iLISI", "kBET"], "Greens"),
    ("Classification", ["OCA", "ACA", "Spec", "Sens", "F1"], style.PINKS),
]

#: glyph drawn in a cell whose metric was not computed for that method
NA_MARK = "\u2013"

#: second line of the Score legend title: the fill is scaled within each column
SCORE_SCALE_NOTE = "(scaled per column)"

#: fill of a column whose rows all hold the same value (and of every marker in
#: a one-method figure): the colour ramp would draw it at its "High" end
CONSTANT_FILL = "#bfbfbf"


@dataclass
class FamilyBlock:
    """One metric family of a :class:`BubbleTable` (a colour-banded column block).

    ``raw`` / ``norm`` / ``ranks`` are method x metric frames in the table's
    row order; ``ranks`` holds per-column max-ranks (``n`` = best, ties share
    the higher number - R's ``ties.method="max"``), ``norm`` per-column
    min-max values, ``overall`` the family's Overall score per method.
    """
    label: str
    cmap: str
    raw: pd.DataFrame
    norm: pd.DataFrame
    ranks: pd.DataFrame
    overall: pd.Series


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
        One block per metric family present, in paper order: DR and
        clustering (blues), batch correction (greens), classification
        (pinks), then "Other" (purples) for metrics outside the three. Each
        has its own ``raw``, ``norm``, ``ranks`` and ``overall``.
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
    na_cells : list of str or None
        The ``n/a`` cells: one line per family and method (per dataset and
        method under ``"summary"``).

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
    and the column's ``n`` counts only the scored methods.

    **Ties.** Values that agree to 9 decimal places count as equal in
    ``ranks``, ``norm`` and the Overall scores. An iLISI of ``2.2e-16``
    therefore ties with ``0.0``.

    **Scaled values.** ``matrix`` and ``norm`` are per-column min-max values
    in [0, 1]; a constant (or all-NaN) column is all ones, as in the R code.
    The figure draws a column whose rows all hold one value in grey, not at
    the ``High`` end of the Score legend.

    See Also
    --------
    mtb.plot.build_table : builds a table from a long table.
    mtb.plot.bubble : builds the table and draws the figure in one call.
    """
    methods: list
    blocks: list
    matrix: pd.DataFrame
    raw: pd.DataFrame
    overall: pd.Series
    aggregate: str = "dataset"
    overall_basis: str = "rank"
    datasets: tuple = ()
    coverage: object = None
    method_datasets: object = None
    na_cells: object = None

    @property
    def norm(self) -> pd.DataFrame:
        """All families' per-column min-max values; an alias of ``matrix``."""
        return self.matrix

    @property
    def ranks(self) -> pd.DataFrame:
        """All families' per-column max-ranks (``n`` = best), in figure order."""
        return pd.concat([b.ranks for b in self.blocks], axis=1)


def _and_distinct(values) -> str:
    """``_and_list`` of the distinct ``str`` forms of ``values``, in first-seen order."""
    return _and_list(dict.fromkeys(map(str, values)))


def _unknown_names(unknown, available, kind: str) -> str:
    """The error for ``unknown`` names of ``kind``: a did-you-mean from the
    table (for a method also from the registry), then the names the table has."""
    import difflib
    pool = [str(a) for a in available]
    hints = []
    for u in unknown:
        h = difflib.get_close_matches(str(u), pool, n=1, cutoff=0.6)
        if not h and kind == "method":
            from ..engine import registry
            h = [c for c in (registry.closest_method(str(u)),) if c]
        hints += h
    text = f"Unknown {kind if len(unknown) == 1 else kind + 's'} {_and_distinct(unknown)}."
    if hints:
        text += f" Did you mean {_and_distinct(hints)}?"
    have = sorted(set(pool))
    return text + (f" The table has {_and_distinct(have)}." if have else " The table is empty.")


def _resolve(requested, available, kind: str, canon) -> list:
    """The frame's own spelling of each requested name, matched exactly, by
    ``canon`` form, then case-insensitively; unknown or repeated names raise."""
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
        raise ValueError(_unknown_names(unknown, available, kind))
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

    Returns the numbers ``mtb.plot.bubble`` draws, without drawing.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long table with ``method``, ``metric``, ``value`` columns; optional
        ``dataset`` column (Notes).
    metrics : list of str | None
        Metric codes to keep, in this order within each family; ``None`` =
        every metric in the frame.
    methods : list of str | None
        Methods (rows) to keep; ``None`` = every method in the frame.
    order : list of str | None
        Methods to put first, in this order. The rest follow, best first. To
        drop methods, use ``methods``.
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
        default); formulas in Notes.
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
        A dataset holds one method, or no method spans two datasets.
    UserWarning
        A column has the same value in every row, or there is one method.
    UserWarning
        Rows scored with the igraph Leiden backend are shown with stored rows
        of their dataset.

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
    **Missing cells.** A metric not computed for a method is an ``n/a``
    cell, drawn as a dash.

    - ``aggregate="dataset"``: the family *Overall* averages the ranks of
      the metrics the method has, and a column's ranks count only the
      methods scored in it.
    - ``aggregate="summary"``: the cell is rank 0 in that dataset (the
      paper's rule), in the metric columns and in the ``overall="rank"``
      Overall; ``overall="mean_overall"`` skips it.

    ``na`` sets how this is reported. ``na_cells`` lists the cells, e.g.
    ``"YukiNet: DR and clustering Overall over 3 of 4 metrics (cLISI n/a)"``.

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

    **Row order.** The combined Overall is the mean of the family Overalls,
    sorted best first with a stable sort, so tied methods keep alphabetical
    order. Ranks and scores always come from the whole filtered frame;
    ``order`` only moves rows.

    **Input columns.** Only ``method``, ``metric`` and ``value`` are
    required.

    - ``dataset`` - groups rows for ``aggregate="summary"`` (absent = one
      dataset) and is part of the duplicate-row key. A ``"dataset"`` figure
      that mixes datasets averages them per method and adds a dataset cue to
      each row label (``Name · D11`` or ``Name · 3 ds``).
    - Rows whose ``metric`` is NaN are dropped.

    To draw your own runs next to the stored table, concatenate the frames:
    ``pd.concat([mtb.load_results("vertical", dataset="D11"), res.long])``,
    with ``res`` from ``mtb.run_all``.

    **Name matching.** ``metrics``, ``methods`` and ``order`` match the
    frame exactly, by canonical form (``"ari"`` -> ``"ARI"``) or
    case-insensitively; the frame's own spelling is kept. The family blocks
    always stay in paper order; metrics outside the three families form a
    neutral purple "Other" block.

    See Also
    --------
    mtb.plot.bubble : builds the table and draws the figure in one call.
    mtb.plot.BubbleTable : what is returned, field by field.
    """
    import warnings
    from ..data import catalog

    need = {"method", "metric", "value"}
    have = set(long_df.columns)
    if not need.issubset(have):
        missing = sorted(need - have)
        hint = ""
        if "Value" in have or long_df.index.name == "metric":
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

    keys = ["method", "metric"] + (["dataset"] if "dataset" in df.columns else [])
    repeated = df.duplicated(keys, keep=False)
    if repeated.any():
        raise ValueError(_duplicate_rows_message(df[repeated], keys))

    datasets = (tuple(sorted(map(str, df["dataset"].dropna().unique())))
                if "dataset" in df.columns else ())
    method_datasets = None
    if "dataset" in df.columns:
        method_datasets = {m: sorted(map(str, g["dataset"].dropna().unique()))
                           for m, g in df.groupby("method")}

    stored = style.stored_datasets(df)
    datasets_per_method = None
    parts = None
    if aggregate == "summary":
        parts = style.per_dataset_ranks(df)
        n = len(parts)
        datasets_per_method = style.coverage(parts)
        if require_complete:
            keep = datasets_per_method[datasets_per_method == n].index
            if len(keep) == 0:
                raise ValueError(
                    f"No method has results on all {n} datasets "
                    f"({', '.join(map(str, parts))}). "
                    + _config.hint("Pass require_complete=False.",
                                   "Leave out --require-complete."))
            if len(keep) < len(datasets_per_method):
                dropped = datasets_per_method[datasets_per_method < n].sort_values()
                groups = {}
                for m in dropped.index:
                    lacks = tuple(ds for ds, mat in parts.items() if m not in mat.index)
                    groups.setdefault(lacks, []).append(str(m))
                n_dropped = len(dropped)
                warnings.warn(
                    _config.hint("require_complete=True", "--require-complete")
                    + f" dropped {n_dropped} method{'' if n_dropped == 1 else 's'}: "
                    + " ".join(f"{_and_distinct(ms)} lack{'s' if len(ms) == 1 else ''} "
                               f"{_and_distinct(ds)}." for ds, ms in groups.items())
                    + " " + _config.hint("Pass require_complete=False",
                                         "Leave out --require-complete")
                    + " to keep them.",
                    UserWarning, stacklevel=2)
                df = df[df["method"].isin(keep)]
                parts = style.per_dataset_ranks(df)
                datasets_per_method = style.coverage(parts)
        for msg in style.coverage_warnings(
                parts, basis=overall,
                incomplete_fix=_config.hint(
                    "Pass require_complete=True to keep only the methods scored "
                    "on every dataset.",
                    "Pass --require-complete to keep only the methods scored on "
                    "every dataset."),
                stored=stored):
            warnings.warn(msg, UserWarning, stacklevel=2)
        raw_all = style.mean_rank_matrix(parts)
    else:
        if len(datasets) > 1:
            # the same per-dataset view as the summary, only to see which
            # datasets share methods; the figure itself shows raw values
            for msg in style.dataset_mode_warnings(style.per_dataset_ranks(df),
                                                   stored=stored):
                warnings.warn(msg, UserWarning, stacklevel=2)
        raw_all = df.pivot_table(index="method", columns="metric", values="value",
                                 aggfunc="mean")
    backend = style.backend_warning(df)
    if backend:
        warnings.warn(backend, UserWarning, stacklevel=2)

    def _block(label, cmap, raw):
        if aggregate == "summary" and overall == "mean_overall":
            family_ranks = {}
            for ds, p in parts.items():
                shared = [c for c in raw.columns if c in p.columns]
                if shared:
                    family_ranks[ds] = p[shared]
            family_overall = style.overall_by_basis(family_ranks, "mean_overall")
            family_overall = family_overall.reindex(raw.index)
        else:
            family_overall = style.compute_overall(raw)
        return FamilyBlock(
            label=label, cmap=cmap, raw=raw,
            norm=raw.apply(lambda col: style.minmax(col.to_numpy()), axis=0),
            ranks=raw.apply(lambda col: style.rank_max(col.to_numpy()), axis=0),
            overall=family_overall,
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
        blocks.append(_block(label, cmap, raw_all[cols]))
    if metrics is not None:
        leftover = [c for c in metrics if c not in known and c in raw_all.columns]
    else:
        leftover = [c for c in raw_all.columns if c not in known]
    if leftover:
        blocks.append(_block("Other", "Purples", raw_all[leftover]))
    if not blocks:
        raise ValueError("no known metrics found in long_df")

    combined = pd.concat([b.overall for b in blocks], axis=1).mean(axis=1)
    # stable sort: tied methods keep alphabetical order, as in plot.bar
    ranked = combined.sort_values(ascending=False, kind="mergesort").index.tolist()
    if order is not None:
        order = _resolve(order, raw_all.index.tolist(), "method", catalog.canonical_id)
        rows = order + [m for m in ranked if m not in order]
    else:
        rows = ranked

    for b in blocks:
        b.raw = b.raw.reindex(rows)
        b.norm = b.norm.reindex(rows)
        b.ranks = b.ranks.reindex(rows)
        b.overall = b.overall.reindex(rows)

    na_cells = _na_report(blocks, parts, aggregate)
    if na_cells and na != "skip":
        msg = _na_message(_na_missing(blocks, parts, aggregate), df, aggregate, na)
        if na == "raise":
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    drawn = pd.concat([b.raw for b in blocks], axis=1)
    msg = _no_comparison_message(rows, style.constant_columns(drawn), aggregate,
                                 partial=set(drawn.columns[drawn.isna().any()]))
    if msg:
        warnings.warn(msg, UserWarning, stacklevel=2)

    return BubbleTable(
        methods=rows, blocks=blocks,
        matrix=pd.concat([b.norm for b in blocks], axis=1),
        raw=pd.concat([b.raw for b in blocks], axis=1),
        overall=combined.loc[rows],
        aggregate=aggregate,
        overall_basis=overall,
        datasets=datasets,
        coverage=datasets_per_method,
        method_datasets=method_datasets,
        na_cells=na_cells,
    )


def _no_comparison_message(methods, constant: dict, aggregate: str = "dataset",
                           partial=frozenset()) -> str | None:
    """The warning for grey fills: one method, or the ``constant`` columns
    (``partial`` ones lack some values); ``None`` when there is none."""
    if len(methods) == 1:
        return (f"Only one method, {methods[0]}, is in this figure. With nothing "
                f"to rank it against, the fills are grey. Plot it with methods "
                f"scored on the same dataset.")
    if not constant:
        return None
    which = "That column is" if len(constant) == 1 else "Those columns are"
    if aggregate == "summary":
        # columns that share a mean rank are named together
        groups = {}
        for c, v in constant.items():
            groups.setdefault(f"{round(v, 2):g}", []).append(str(c))
        same = " ".join(f"Every method has mean rank {v} in {_and_list(cs)}."
                        for v, cs in groups.items())
        return f"{same} {which} grey."
    sentences = []
    for cols, who in (([c for c in constant if c not in partial], "every method"),
                      ([c for c in constant if c in partial], "every method with a value")):
        if cols:
            sentences.append(_and_list([f"{c} is {constant[c]:.3f}" for c in cols])
                             + f" for {who}.")
    return f"{' '.join(sentences)} {which} grey."


def _na_report(blocks, parts, aggregate: str) -> list:
    """The ``na_cells`` lines: one per family and method with n/a cells (per
    dataset and method under ``"summary"``)."""
    lines = []
    if aggregate == "summary":
        for ds, mat in parts.items():
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
            # methods with no metric of the family (single- next to multi-batch
            # datasets) get no Overall bar: one line for all
            lines.append(f"no {b.label} metric (no {b.label} Overall) for: "
                         f"{', '.join(whole)}")
    return lines


def _na_missing(blocks, parts, aggregate: str) -> list:
    """The n/a cells per row, in row order, as ``[(method, [(dataset, codes)],
    families)]``: dataset ``None`` under ``"dataset"``, where ``families`` are
    those the method has no value in; codes in column order."""
    columns = [c for b in blocks for c in b.raw.columns]
    pos = {c: i for i, c in enumerate(columns)}

    def _sorted(codes):
        return sorted(codes, key=lambda c: pos.get(c, len(pos)))

    out = []
    for m in blocks[0].raw.index:
        if aggregate == "summary":
            per_ds = [(ds, _sorted(c for c in mat.columns if pd.isna(mat.loc[m, c])))
                      for ds, mat in parts.items() if m in mat.index]
            per_ds = [(ds, codes) for ds, codes in per_ds if codes]
            if per_ds:
                out.append((m, per_ds, []))
            continue
        codes, whole = [], []
        for b in blocks:
            na = [c for c in b.raw.columns if pd.isna(b.raw.loc[m, c])]
            codes += na
            if na and len(na) == b.raw.shape[1]:
                whole.append(b.label)
        if codes:
            out.append((m, [(None, codes)], whole))
    return out


#: most rows the ``n/a`` warning names before it counts the rest
NA_ROWS_SHOWN = 3


def _na_message(missing: list, df: pd.DataFrame, aggregate: str,
                na: str = "warn") -> str:
    """The ``na="warn"`` / ``na="raise"`` text: one sentence per row (at most
    ``NA_ROWS_SHOWN``), the rule the figure applies, then the other policy."""
    stored_sources = set(style.STORED_SOURCES)
    has_ds = "dataset" in df.columns
    sentences = []
    for m, per_ds, whole in missing[:NA_ROWS_SHOWN]:
        rows = df[df["method"] == m]
        src = (set(rows["source"].dropna().astype(str)) if "source" in rows.columns
               else set())
        stored = bool(src) and src <= stored_sources
        who = ("The stored row" if stored else
               "Your row" if src and not src & stored_sources else "Row")
        where = []
        if aggregate != "summary" and has_ds:
            where += sorted(map(str, rows["dataset"].dropna().unique()))
        if stored:
            where += sorted(src)
        name = f"{who} {m}" + (f" ({', '.join(where)})" if where else "")
        if aggregate == "summary" and has_ds:
            # datasets that lack the same metrics are named together
            by_codes: dict = {}
            for ds, codes in per_ds:
                by_codes.setdefault(tuple(codes), []).append(str(ds))
            what = ", and no ".join(f"{_and_list(codes, 'or')} on {_and_list(dss)}"
                                    for codes, dss in by_codes.items())
        else:
            what = _and_list(dict.fromkeys(c for _, codes in per_ds for c in codes), "or")
        text = f"{name} has no {what}."
        if whole:
            text += f" It has no {' or '.join(whole)} Overall."
        sentences.append(text)
    more = len(missing) - NA_ROWS_SHOWN
    if more > 0:
        sentences.append(f"... and {more} more row{'s' if more > 1 else ''}.")
    if aggregate == "summary":
        sentences.append("In the summary, a missing value counts as the lowest rank on "
                         "that dataset.")
    elif na != "raise":
        # the Overall rule describes the figure, which na="raise" does not draw
        sentences.append("Its Overall uses the metrics it has." if len(missing) == 1
                         else "Each row's Overall uses the metrics it has.")
    if na == "raise":
        sentences.append(_config.hint("Pass na='warn' to draw the figure with these gaps.",
                                      "Pass --na warn to draw the figure with these gaps."))
    else:
        sentences.append(_config.hint("Pass na='skip' to hide this message.",
                                      "Pass --na skip to hide this message."))
    return " ".join(sentences)


def _duplicate_rows_message(dup, keys) -> str:
    """The error for rows that repeat ``keys``: the fix when your rows collide
    with stored ones, else one example repeat."""
    stored = (dup["source"].astype("string").str.match(r"^(published|rerun)").fillna(False)
              if "source" in dup.columns else pd.Series(False, index=dup.index))
    mixed = dup.assign(_st=stored.to_numpy()).groupby(keys)["_st"].agg(
        lambda s: s.any() and not s.all())
    mixed = mixed[mixed].index.to_frame(index=False)
    if len(mixed):
        where = f" on {_and_distinct(mixed['dataset'])}" if "dataset" in keys else ""
        first = str(mixed["method"].iloc[0])
        return (f"Your rows and the stored table both have "
                f"{_and_distinct(mixed['method'])}{where}. "
                f"Give your rows another name, such as {first}_rerun, "
                + _config.hint("with to_long(method=...).", "with evaluate --name."))
    what = _and_list(keys)
    example = ", ".join(str(dup[k].iloc[0]) for k in keys)
    return (f"{len(dup)} rows repeat the same {what}, for example {example}. The figure "
            "does not average them. Remove the repeats, or give each version its own "
            "method name.")


def _method_language(name):
    """``"python"`` or ``"r"``; None for a name the registry does not know."""
    from ..engine import registry
    try:
        return registry.get(name).language.lower()
    except KeyError:
        return None


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
        Draw the Py / R / ? chips left of each row.

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
            fig = _draw(tbl, cmap, title, show_language, gaps)
            short = _fit_pills(fig, fig.axes[0])
            if not any(s > 1e-6 for s in short):
                break
            # two widened pills would touch: more room between those families
            gaps = [g + s for g, s in zip(gaps, short)]
    return fig


@functools.cache
def _figure_class():
    """A ``Figure`` with ``_repr_png_``. Built without pyplot, the figure is
    never shown twice, and a notebook without ``%matplotlib inline`` still
    shows the image rather than ``<Figure size ...>``."""
    from matplotlib.figure import Figure

    class BubbleFigure(Figure):
        def _repr_png_(self):
            import io
            buf = io.BytesIO()
            self.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            return buf.getvalue()

    return BubbleFigure


def _draw(tbl, cmap, title, show_language, gaps):
    """The body of :func:`render`; ``gaps`` widens the space between families
    (one data-unit amount per pair of neighbouring families)."""
    Figure = _figure_class()
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    from matplotlib import cm, colors

    methods = tbl.methods
    n_rows = len(methods)
    ROW_H, R_MAX = 1.0, 0.5          # row height; circle max radius = row/2

    # x layout: per family [overall][metrics...], with a gap between families
    x_next, layout = 0.0, []
    for fi, b in enumerate(tbl.blocks):
        if fi:
            x_next += 0.5 + gaps[fi - 1]
        layout.append((x_next, fi, "overall", "Overall"))
        x_next += 1.5                                 # bar column is wider
        for mname in b.raw.columns:
            layout.append((x_next, fi, "metric", mname))
            x_next += 1.1
    total_w = x_next

    # the legends under a table of few metrics are wider than the table, so the
    # axes end at the right edge of the widest one
    RAMP_X0, RAMP_STEP, RAMP_W = 1.2, 4.6, 40 * 0.07   # Score ramp per family
    # room for the "High" label after a ramp: about 0.5 units in a tall
    # figure, up to 0.7 in a one-row figure with a title
    HIGH_W = 0.7
    RANK_X0 = 1.5                    # first Rank circle; the next ones 1.0 apart
    legend_right = RAMP_X0 + (len(tbl.blocks) - 1) * RAMP_STEP + RAMP_W + 0.12 + HIGH_W
    if tbl.aggregate != "summary":
        # at most 5 legend circles, sized as the plot's own: 0.85 * sqrt(rank / n)
        n_legend = min(5, n_rows)
        legend_radii = np.maximum(
            0.85 * np.sqrt(np.linspace(1, n_rows, n_legend) / n_rows), 0.12)
        legend_right = max(legend_right,
                           RANK_X0 + (n_legend - 1) * 1.0 + R_MAX * legend_radii[-1] * 0.9 + 0.15)
    x_right = max(total_w + 0.45, legend_right)

    # the width follows the x-range: 0.6 inch per unit past the table's own
    fig = Figure(figsize=(0.6 * (x_right - 0.45) + 2.8, 0.42 * n_rows + 2.9))
    ax = fig.subplots()
    norm01 = colors.Normalize(vmin=0.0, vmax=1.0)
    mappers = {fi: cm.ScalarMappable(norm=norm01, cmap=style.colormap(
                   cmap if (fi == 0 and cmap) else b.cmap))
               for fi, b in enumerate(tbl.blocks)}

    # alternating row stripes, like the Shiny output
    for i in range(n_rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((-1.3, n_rows - i - 1), total_w + 1.8, ROW_H,
                                   facecolor="#DDDDDD", edgecolor="none", zorder=0))

    if show_language:
        for i, m in enumerate(methods):
            lang = _method_language(m)
            if not lang:
                label, colour = "?", "#aaaaaa"
            elif lang.startswith("py"):
                label, colour = "Py", "#3572A5"
            else:
                label, colour = "R", "#777777"
            y = n_rows - i - 0.5
            ax.add_patch(Circle((-0.85, y), 0.24, facecolor=colour,
                                edgecolor="none", zorder=3))
            ax.text(-0.85, y, label, ha="center", va="center", zorder=4,
                    fontsize=6.4, fontweight="bold", color="white")

    def rank_radius(colvals):
        """radius = 0.85 * sqrt(rank / n), n = methods scored in the column."""
        n = colvals.notna().sum()
        r = style.rank_max(colvals.to_numpy()) / n
        radii = 0.85 * np.sqrt(r)
        return pd.Series(np.maximum(radii, 0.12), index=colvals.index)

    # a column whose rows all hold one value gets the neutral fill
    constant = style.constant_columns(pd.concat([b.raw for b in tbl.blocks], axis=1))
    flat_overall = {fi for fi, b in enumerate(tbl.blocks)
                    if style.constant_columns(b.overall.to_frame("o"))}

    def _fill(mapper, value, flat):
        return colors.to_rgba(CONSTANT_FILL) if flat else mapper.to_rgba(value)

    floor = min(0.30, 1.2 / n_rows)      # shortest bar, visible with few methods

    for x, fi, kind, name in layout:
        b, mapper = tbl.blocks[fi], mappers[fi]
        if kind == "overall":
            vals = b.overall
            # bar length: family Overall min-max scaled across the rows
            length = style.minmax(vals.to_numpy())
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    continue
                y0 = n_rows - i - 1
                bar_len = 1.24 * max(floor, float(length[i]))
                ax.add_patch(Rectangle((x + 0.08, y0 + 0.12), bar_len, ROW_H - 0.24,
                                       facecolor=_fill(mapper, float(v), fi in flat_overall),
                                       edgecolor="#333333", linewidth=0.5, zorder=3))

        elif tbl.aggregate == "summary":
            scaled_values = b.norm[name]
            # a summary table has no empty cell: a missing score is rank 0
            for i, m in enumerate(methods):
                y0 = n_rows - i - 1
                bar_len = 0.95 * max(floor, float(scaled_values.loc[m]))
                ax.add_patch(Rectangle((x + 0.06, y0 + 0.16), bar_len, ROW_H - 0.32,
                                       facecolor=_fill(mapper, float(scaled_values.loc[m]),
                                                       name in constant),
                                       edgecolor="#333333", linewidth=0.4, zorder=3))
        else:
            vals, scaled_values = b.raw[name], b.norm[name]
            radii = rank_radius(vals)
            for i, m in enumerate(methods):
                v = vals.loc[m]
                if pd.isna(v):
                    ax.text(x + 0.55, n_rows - i - 0.5, NA_MARK, color="#999999",
                            ha="center", va="center", fontsize=7, zorder=3)
                    continue
                y = n_rows - i - 0.5
                ax.add_patch(Circle((x + 0.55, y), R_MAX * float(radii.loc[m]) * 0.9,
                                    facecolor=_fill(mapper, float(scaled_values.loc[m]),
                                                    name in constant),
                                    edgecolor="#333333", linewidth=0.4, zorder=3))

    for x, fi, kind, name in layout:
        tick_x = x + (0.7 if kind == "overall" else 0.55)
        ax.plot([tick_x, tick_x], [n_rows + 0.05, n_rows + 0.22], color="#333333",
                linewidth=0.6, zorder=2)
        ax.text(tick_x - 0.05, n_rows + 0.30, name, rotation=30, ha="left",
                va="bottom", fontsize=8, rotation_mode="anchor")

    band_y = n_rows + 2.05
    for fi, b in enumerate(tbl.blocks):
        family_xs = [x for x, f, k, _ in layout if f == fi]
        x0, x1 = family_xs[0], family_xs[-1] + 1.1
        # gid="family": _fit_pills finds the pill and its header
        ax.add_patch(FancyBboxPatch((x0 + 0.05, band_y), x1 - x0 - 0.35, 0.6,
                                    boxstyle=f"round,pad={PILL_BOX_PAD},rounding_size=0.18",
                                    facecolor=mappers[fi].to_rgba(0.30),
                                    edgecolor="none", zorder=2, gid="family",
                                    clip_on=False))
        ax.text((x0 + x1 - 0.3) / 2, band_y + 0.3, b.label, ha="center",
                va="center", fontsize=8.6, color="#1a1a1a", zorder=3, gid="family")

    # Score ramps and Rank sizes under the table, as in the scIB layout
    # gid="legend" marks every artist of the Score and Rank legends
    score_y = -1.1
    ax.text(-0.9, score_y, "Score", fontsize=8, fontweight="bold", va="center",
            gid="legend")
    ax.text(-0.9, score_y - 0.5, SCORE_SCALE_NOTE, fontsize=6.4, va="center",
            gid="legend")
    for fi, b in enumerate(tbl.blocks):
        xoff = RAMP_X0 + fi * RAMP_STEP
        for k in range(40):
            ax.add_patch(Rectangle((xoff + k * 0.07, score_y - 0.28), 0.07, 0.56,
                                   facecolor=mappers[fi].to_rgba(k / 39),
                                   edgecolor="none", zorder=2, gid="legend"))
        ax.text(xoff - 0.12, score_y, "Low", fontsize=6.6, ha="right", va="center",
                gid="legend")
        ax.text(xoff + RAMP_W + 0.12, score_y, "High", fontsize=6.6, ha="left",
                va="center", gid="legend")
    if tbl.aggregate != "summary":
        rank_y = score_y - 1.35
        ax.text(-0.9, rank_y, "Rank", fontsize=8, fontweight="bold", va="center",
                gid="legend")
        xoff = RANK_X0
        for k, r in enumerate(legend_radii):
            ax.add_patch(Circle((xoff + k * 1.0, rank_y), R_MAX * r * 0.9,
                                facecolor="#bbbbbb", edgecolor="#333333",
                                linewidth=0.4, zorder=2, gid="legend"))
        if n_legend > 1:
            ax.text(xoff, rank_y - 0.62, str(n_rows), fontsize=6.6,
                    ha="center", gid="legend")
        ax.text(xoff + (n_legend - 1) * 1.0, rank_y - 0.62, "1", fontsize=6.6,
                ha="center", gid="legend")

    # room under the legends: the Rank labels, or the Score legend's note
    y_bottom = -3.4 if tbl.aggregate != "summary" else -2.0

    # a figure that mixes datasets gets a dataset cue in each row label
    labels = list(methods)
    if tbl.aggregate == "dataset" and len(tbl.datasets) > 1:
        datasets_of = tbl.method_datasets
        labels = [f"{m} \u00b7 {datasets_of[m][0]}" if len(datasets_of[m]) == 1
                  else f"{m} \u00b7 {len(datasets_of[m])} ds" for m in methods]

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
        Methods to put first, in this order. The rest follow, best first. To
        drop methods, use ``methods``.
    aggregate : {"dataset", "summary"}
        ``"dataset"``: one dataset's values, drawn as circles. ``"summary"``:
        within-dataset ranks averaged across datasets, drawn as bars (the
        paper's panel c).
    cmap : str | None
        Matplotlib colormap for the first family; ``None`` = blues, greens,
        pinks and purples per family.
    title : str | None
        Figure title; ``None`` = no title.
    save : str or path-like | None
        File to write the figure to (tight bounding box); the suffix picks
        the format, e.g. ``.pdf``, ``.png``, ``.svg``.
    show_language : bool
        Draw the ``Py`` / ``R`` chip left of each row (Notes).
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
        As ``mtb.plot.build_table``.

    Warns
    -----
    UserWarning
        As ``mtb.plot.build_table``.

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
    The Notes of ``build_table`` give the Overall formulas, missing cells,
    row order, input columns and name matching.

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
    - *Rank* legend: 1 = best, whereas ``BubbleTable.ranks`` /
      ``FamilyBlock.ranks`` store max-ranks (``n`` = best).
    - Grey fill: every row of that column holds the same value, or the
      figure has one method, so there is nothing to compare.

    **Chips** (``show_language``). ``Py`` / ``R`` = language; ``?`` = a name
    the package does not know, such as your own or a renamed re-run.

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


#: deprecated 0.2.x name of :func:`bubble` (DeprecationWarning; removed in 0.4)
plot_bubble = _compat.deprecated_alias("mtb.plot.plot_bubble", "mtb.plot.bubble", bubble)
