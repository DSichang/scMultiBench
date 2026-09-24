"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import contextvars
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from .. import _compat, config
from ..data import catalog

#: the seven columns of the long table (pinned to
#: multibench.data.results.COLUMNS by tests/test_eval_reshape.py)
LONG_COLUMNS = ["metric", "value", "method", "dataset", "category", "clustering", "source"]

#: the ``attrs`` keys evaluate() sets on its frame and to_long() carries over
#: (and joins into ``scored_with``); evaluate() also records ``scib_version``
PROVENANCE_ATTRS = ("leiden_flavor", "clustering", "multibench_version")

#: ``scored_with`` of rows whose frame carries no record of how it was scored
UNKNOWN_SCORING = "unknown"

#: to_long's warning for such a frame (a hand-made frame, a wide CSV read back)
NO_SCORING_RECORD = (
    "these scores carry no record of how they were scored, so scored_with is "
    "\"unknown\". A wide CSV read back loses that record. To keep it, save the "
    "mtb.to_long(...) table instead of the wide one.")


def _scib_version() -> str | None:
    """The installed scib version, or ``None`` when it cannot be read."""
    try:
        from importlib.metadata import version
        return version("scib")
    except Exception:  # noqa: BLE001 - not installed, or no metadata
        try:
            import scib
            return getattr(scib, "__version__", None)
        except Exception:  # noqa: BLE001
            return None


def _scored_with(attrs) -> str | None:
    """``"<leiden flavor>/<clusters>/<version>"`` from evaluate()'s ``attrs``,
    e.g. ``"leidenalg/sweep/0.3.2"``; ``None`` when the frame carries none
    (a hand-made frame or a CSV read back). ``none`` fills a part that did
    not apply (no sweep ran, no ARI/NMI)."""
    if not attrs or attrs.get("multibench_version") is None:
        return None
    return "/".join(str(attrs.get(k) or "none") for k in PROVENANCE_ATTRS)


def to_long(value_df, *, method: str, dataset: str | None = None,
            category: str | None = None, clustering: str = "default",
            source: str = "user") -> pd.DataFrame:
    """Reshape ``mtb.evaluate``'s scores into the long table of ``mtb.load_results``.

    The result has the columns of ``mtb.load_results`` and concatenates with
    the stored tables.

    Parameters
    ----------
    value_df : pandas.DataFrame or pandas.Series
        What ``mtb.evaluate`` returns (metrics as the index, one column
        ``Value``), a Series indexed by metric, or that frame's CSV read back
        (forms in Notes).
    method : str
        Method id written into every row; your own name is fine.
    dataset : str, optional
        Dataset id written into every row; ``None`` writes ``"all"``.
    category : str, optional
        Integration category written into every row; ``None`` writes
        ``"user"``.
    clustering : str
        Value of the ``clustering`` column.
    source : str
        Value of the ``source`` column; ``source="user"`` in
        ``mtb.load_results`` selects these rows again.

    Returns
    -------
    pandas.DataFrame
        One row per metric, with the columns ``metric, value, method,
        dataset, category, clustering, source, scored_with``. ``scored_with``
        is ``"unknown"`` for a frame not from ``mtb.evaluate``.

    Raises
    ------
    ValueError
        ``value_df`` is already long, lacks ``Value`` or string metric names,
        or repeats a name.

    Warns
    -----
    UserWarning
        ``value_df`` has no record of how it was scored; ``scored_with`` is
        ``"unknown"``.

    Examples
    --------
    >>> import multibench as mtb, pandas as pd
    >>> wide = mtb.evaluate(emb, labels=labels, metrics=["ARI", "NMI"])
    >>> mine = mtb.to_long(wide, method="MyMethod", dataset="D11",
    ...                    category="vertical")
    >>> stored = mtb.load_results("vertical", dataset="D11", source="rerun")
    >>> pd.concat([stored, mine]).to_csv("all.csv", index=False)
    >>> mtb.load_results(result_path="all.csv", source="user")   # your rows only

    Notes
    -----
    **Metric names.** Names are canonicalised (``ari`` -> ``ARI``, ``kbet``
    -> ``kBET``); rows whose name is blank are dropped, and a name the
    package does not know is kept as written.

    **CSV read-back.** ``pd.read_csv(path, index_col=0)`` reads
    ``wide.to_csv(path)`` back. A plain ``pd.read_csv(path)`` works too: its
    metric column ``Unnamed: 0`` is taken as the names only when the columns
    are exactly ``Unnamed: 0, Value``, it holds strings and the index none.
    So does a ``metric`` column (``wide.to_csv(path, index_label="metric")``).
    Metric names that are not strings, like row numbers ``0, 1, ...``, raise
    instead of being scored.

    **Column values.** ``dataset=None`` writes ``"all"``. ``category=None``
    writes ``"user"``, the value ``load_results`` gives a user file without a
    category column. The published tables use
    ``clustering`` values ``"louvain"`` / ``"kmeans"`` for their variants.

    **Provenance.** The ``scored_with`` column of a frame from
    ``mtb.evaluate`` reads like ``"leidenalg/sweep/0.3.2"``: the Leiden
    backend, where the clusters came from (``sweep`` or ``user``) and the
    package version; ``none`` fills a part that did not apply. The same
    three values and ``scib_version`` are in ``attrs``.

    Any other frame, such as a hand-made one or a wide CSV read back, has no
    such record. Its rows get ``scored_with = "unknown"`` and a
    ``UserWarning``. To keep the record, save the long table:
    ``mtb.to_long(...).to_csv(path, index=False)``.

    The stored tables have no ``scored_with`` column, so it is NaN for their
    rows after ``pd.concat``. ``load_results(result_path=...)`` keeps the
    column when it reads the CSV back.

    **Plot badges.** The bubble figure shows ``?`` for a name the package
    does not know, such as your own or a renamed re-run. To badge such a row
    as supervised, add a boolean ``needs_labels`` column.

    **Errors.** All are ``ValueError``:

    - an already long frame (columns ``metric, value, method``) - pass it to
      the plot / ``load_results`` consumers directly;
    - no ``Value`` column - the message names the expected shape and, for a
      wide one-row frame, the ``df.T.set_axis(['Value'], axis=1)`` fix;
    - a metric name that is not a string (row numbers from a plain
      ``pd.read_csv`` of another shape) - the message names
      ``pd.read_csv(path, index_col=0)``;
    - every metric name blank;
    - two names that collapse onto one canonical metric (``ari`` and
      ``ARI``).

    See Also
    --------
    mtb.evaluate : computes the wide frame this function reshapes.

    mtb.load_results : the stored scores, in the same long shape.

    mtb.plot.bubble : plots a long frame.
    """
    if isinstance(value_df, pd.Series):
        value_df = value_df.to_frame("Value")
    cols = [str(c) for c in getattr(value_df, "columns", [])]
    if {"metric", "value", "method"} <= set(cols):
        raise ValueError(
            "to_long() got an already long frame (columns metric, value, method"
            f"{', ...' if len(cols) > 3 else ''}); pass it to mtb.plot.bubble / "
            "pd.concat / load_results consumers directly - to_long reshapes "
            "evaluate()'s wide frame (metrics as the index, one column 'Value')")
    if "Value" not in cols:
        idx = list(map(str, list(value_df.index)[:5]))
        raise ValueError(
            f"to_long expects evaluate()'s frame: metrics as index, one column "
            f"'Value'; got columns {cols} (index {idx}) - for a wide one-row "
            f"frame use df.T.set_axis(['Value'], axis=1)")
    if "metric" in cols:
        # pd.read_csv of evaluate's frame saved with index_label="metric": the
        # metric names are a column, not the index - reset_index below would
        # otherwise prepend the RangeIndex as a second 'metric' column
        value_df = value_df.set_index("metric")
    elif (cols == ["Unnamed: 0", "Value"]
          and all(map(_name_or_blank, value_df["Unnamed: 0"]))
          and any(isinstance(n, str) for n in value_df["Unnamed: 0"])
          and not any(isinstance(n, str) for n in value_df.index)):
        # pd.read_csv(path) of evaluate's frame saved with .to_csv(path): the
        # index is unnamed, so pandas names its blank header cell 'Unnamed: 0'
        # and numbers the rows - the names are that column, not the index
        value_df = value_df.set_index("Unnamed: 0")
    out = value_df.rename(columns={"Value": "value"}).copy()
    out = out.reset_index()                      # the index column comes first,
    out = out.rename(columns={out.columns[0]: "metric"})   # whatever it was named
    if not all(map(_name_or_blank, out["metric"])):
        # row numbers (a plain read_csv of any other saved shape) would pass
        # canonical_metric unchanged and score metrics '0', '1', ...
        raise ValueError(
            f"to_long needs the metric names as strings (evaluate()'s index or a "
            f"'metric' column); got {out['metric'].head(5).tolist()} with columns {cols} "
            f"- read a CSV saved with wide.to_csv(path) back with "
            f"pd.read_csv(path, index_col=0)")
    out["metric"] = out["metric"].map(catalog.canonical_metric)
    out = out.dropna(subset=["metric"])
    if out.empty and len(value_df):
        raise ValueError(
            f"no metric name in the index canonicalises to a known code: "
            f"{list(map(str, value_df.index))[:10]} - to_long expects "
            f"evaluate()'s frame (metrics as index, one column 'Value')")
    dup = out["metric"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"metric names collide after canonicalisation: "
            f"{sorted(set(out.loc[dup, 'metric']))} - the input names two rows that "
            f"map to the same canonical metric (e.g. 'ari' and 'ARI'); drop one")
    out["method"] = method
    out["dataset"] = "all" if dataset is None else dataset
    out["category"] = "user" if category is None else category
    out["clustering"] = clustering
    out["source"] = source
    stamp = _scored_with(getattr(value_df, "attrs", None))
    out["scored_with"] = UNKNOWN_SCORING if stamp is None else stamp
    out = out[LONG_COLUMNS + ["scored_with"]].reset_index(drop=True)
    if stamp is None:
        warnings.warn(NO_SCORING_RECORD, UserWarning, stacklevel=2)
    else:
        out.attrs = {k: value_df.attrs.get(k) for k in PROVENANCE_ATTRS}
        if "scib_version" in value_df.attrs:
            out.attrs["scib_version"] = value_df.attrs["scib_version"]
    return out


def _name_or_blank(v) -> bool:
    """A metric name (``str``) or a blank cell (``None``/NaN, which to_long drops)."""
    return isinstance(v, str) or (pd.api.types.is_scalar(v) and pd.isna(v))


def _metric_families() -> tuple[list, list]:
    """``(CLUSTERING_METRICS, BATCH_METRICS)`` - the two scIB families evaluate()
    can produce, in the canonical spelling ``load_results``/``to_long`` use."""
    # the plotting layer owns the metric families
    from ..plot.bar import BATCH_METRICS, CLUSTERING_METRICS
    return list(CLUSTERING_METRICS), list(BATCH_METRICS)


def _known_metrics() -> list:
    clu, bat = _metric_families()
    return clu + bat


def _validate_category(category):
    if category is None:
        return
    from ..workflow import list_categories  # lazy: workflow imports this module
    valid = list_categories()
    if category not in valid:
        raise ValueError(
            f"unknown category {category!r}; valid: {sorted(valid)}")


def _is_label_path_list(x) -> bool:
    return (isinstance(x, (list, tuple)) and len(x) > 0
            and all(io._is_label_file(e) for e in x))


def _cell_ids(output):
    """Cell ids carried by ``output``, or ``None`` when it has none.

    An AnnData carries ``obs_names``; a DataFrame carries its index unless that
    index is a plain ``RangeIndex`` (the default, i.e. no ids). Everything else
    (ndarray, sparse matrix, file path) has no ids, so labels can only be
    matched positionally.
    """
    if io._is_anndata(output):
        return pd.Index(output.obs_names)
    if isinstance(output, pd.DataFrame) and not isinstance(output.index, pd.RangeIndex):
        return output.index
    return None


def _carries_ids(x) -> bool:
    """A Series / DataFrame whose index is not a ``RangeIndex`` carries cell ids."""
    return (isinstance(x, (pd.Series, pd.DataFrame))
            and not isinstance(x.index, pd.RangeIndex))


def _obs_or_vector(x, adata, *, what, column=None, ids=None):
    """Coerce one label-like argument to a 1-D array in the output's cell order.

    * a ``str`` naming an obs column of ``adata`` -> that column (already in
      the output's order);
    * a ``Series``/``DataFrame`` with a non-default index, when the output
      carries cell ids (``ids`` given) -> aligned by id via
      :func:`multibench.eval.io.align_vector` (raises on missing/extra ids);
    * a ``Series``/``DataFrame`` with a non-default index, when the output is
      a bare array -> positional, with a ``UserWarning`` saying so;
    * a CSV whose first column holds the output's cell ids -> aligned by
      that column (:func:`multibench.eval.io.by_id_column`);
    * anything else -> :func:`multibench.eval.io.as_vector` (positional).
    """
    if adata is not None and isinstance(x, str):
        if x in adata.obs.columns:
            return np.asarray(pd.Series(adata.obs[x]).to_numpy())
        if not Path(x).is_file():
            raise ValueError(
                f"{what}={x!r} is neither an obs column of the AnnData (obs "
                f"columns: {list(map(str, adata.obs.columns))}) nor an existing "
                f"file")
    if isinstance(x, (str, Path)) and column is None:
        vals, first = io.read_labels_ids(x, what=what, pick=_pick(what))
        unique = ids is not None and pd.Index(ids).is_unique
        return io.by_id_column(
            vals, first, ids if unique else None, what=what, name=Path(x).name,
            target="the output", order="the rows of the output",
            no_ids=_no_ids(ids), stacklevel=5)[0]
    if _carries_ids(x):
        if ids is not None:
            return io.align_vector(x, ids, what=what, column=column)
        warnings.warn(
            f"The {what} {type(x).__name__} is matched by position, because the "
            f"embedding has no cell ids. Check that it follows the embedding rows, "
            f"or pass an AnnData whose obs_names are the barcodes.",
            # _obs_or_vector <- evaluate <- its keyword wrapper <- the caller
            UserWarning, stacklevel=4)
    return io.as_vector(x, what=what, column=column)


def _pick(what: str):
    """How to fix a per-cell file with several columns and no chosen column.

    The ``pick`` of :func:`multibench.eval.io.read_labels_ids`: a function of
    the pandas call that reads one column of the file as a Series. The
    command line names the flag that chooses the column.
    """
    def fix(example: str) -> str:
        py = f"Pass one column as a Series, for example {example}"
        if what == "labels":
            return config.hint(f"{py}, or name the label column x.",
                               "Choose one with --column, or name the label column x.")
        cli = ("Choose one with --batch-column." if what == "batch"
               else "Keep the cell ids and one column in the file.")
        return config.hint(f"{py}.", cli)
    return fix


def _one_column_file(path, column: str, stack, *, what: str = "batch") -> str:
    """A copy of the per-cell file ``path`` with only ``column`` (as ``x``)
    after its first column, in a folder that ``stack`` (an ``ExitStack``)
    removes. ``multibench evaluate/run-all --batch-column`` pass it on; the
    copy keeps the file name, so the messages name the file typed."""
    import tempfile
    path = io._require_file(path, f"{_FLAGS[what]} file")
    vals, first = io.read_labels_ids(path, column, what=what)
    out = Path(stack.enter_context(tempfile.TemporaryDirectory())) / path.name
    cols = {"x": vals} if first is None else {"id": first.to_numpy(), "x": vals}
    pd.DataFrame(cols).to_csv(out, index=False, sep=io._sep_for(out))
    return str(out)


def _no_ids(ids) -> str:
    return ("the output repeats cell ids" if ids is not None
            else "the output has no cell ids")


def _stack_label_files(files, ids):
    """Several label files, stacked in the given order, and each row's file (1, 2, ...).

    When every file has a first column of cell ids, the stacked rows are
    aligned to ``ids`` by those columns, as a single file is
    (:func:`multibench.eval.io.by_id_column`); the file of origin moves with
    its rows. Otherwise the rows stay in file order, and a first column that
    looks like cell ids gets the positional warning.
    """
    read = [io.read_labels_ids(f, what="labels", pick=_pick("labels")) for f in files]
    vals = np.concatenate([np.asarray(v) for v, _ in read])
    origin = np.concatenate([np.full(len(v), i + 1) for i, (v, _) in enumerate(read)])
    names = [Path(f).name for f in files]
    unique = ids is not None and pd.Index(ids).is_unique
    order = "the rows of the output"
    if all(first is not None for _, first in read):
        name = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        # a caller of evaluate is 5 frames up: by_id_column, this, evaluate, its wrapper
        pos, _ = io.by_id_column(np.arange(len(vals)), pd.concat(
            [first for _, first in read], ignore_index=True), ids if unique else None,
            what="labels", name=name, target="the output", order=order,
            no_ids=_no_ids(ids), stacklevel=5, n_files=len(names))
        return vals[pos], origin[pos]
    for (v, first), name in zip(read, names):
        io.by_id_column(v, first, None, what="labels", name=name, target="the output",
                        order=order, stacklevel=5, no_ids=(
                            "the other label files have none" if unique
                            else _no_ids(ids)))
    return vals, origin


def _labels_from_dict(d: dict, label_order) -> list:
    """Turn a ``{name: path}`` label dict into the list of paths in cell order.

    One entry needs no order. Several entries need ``label_order`` (keys of
    ``d``; a subset selects those files) unless ``d`` is what ``labels_for``
    returned, in the order it returned it (a method variant's cell order,
    recorded on the dict), or any dict in the default order
    (``_label_sort_key``). Any other order must be explicit: a guess would
    score the embedding against misordered labels without any error.
    """
    if label_order is None:
        if len(d) == 1:
            return [next(iter(d.values()))]
        from ..engine.resolve import LabelFiles, _label_sort_key
        keys = list(d)
        if isinstance(d, LabelFiles) and d.in_stacking_order():
            return [d[k] for k in keys]
        if keys == sorted(keys, key=_label_sort_key):
            return [d[k] for k in keys]
        # name the method's cell order only for a labels_for dict that had one
        # (a method order other than the default) and was then reordered
        own = list(getattr(d, "stacking_order", ()))
        if isinstance(d, LabelFiles) and own != sorted(own, key=_label_sort_key):
            where = "are in neither the method's cell order nor the default order"
        else:
            where = "are not in the default order"
        raise ValueError(
            f"labels: the keys {keys} {where}. Pass the dict from "
            f"mtb.labels_for(dataset, category, method) unchanged, a list of "
            f"paths in cell order, or label_order=[...]. The default order is "
            f"cty1, cty2, ... by number, with rna before adt before atac. It is "
            f"not alphabetical.")
    if isinstance(label_order, str) or not isinstance(label_order, (list, tuple)):
        raise TypeError(
            f"label_order= must be a list of keys of the labels dict, e.g. "
            f"label_order={list(map(str, d))!r}; got {type(label_order).__name__}")
    order = list(label_order)
    if not order:
        raise ValueError(
            f"label_order= is empty; list the keys of the labels dict in the "
            f"method's cell order, e.g. {list(map(str, d))!r}")
    unknown = [k for k in order if k not in d]
    if unknown:
        raise ValueError(
            f"label_order names key(s) {unknown!r} that are not in the labels "
            f"dict; its keys are {list(map(str, d))!r}")
    if len(set(order)) != len(order):
        raise ValueError(f"label_order repeats a key: {order!r}")
    return [d[k] for k in order]


#: the ``multibench evaluate`` flag that fills each argument of evaluate()
_FLAGS = {"labels": "--labels", "clustering": "--clustering", "batch": "--batch"}
#: what one value of each argument is called in the command-line messages
_UNITS = {"labels": "labels", "clustering": "cluster ids", "batch": "batch ids"}
#: the dataset whose label files ``multibench evaluate --dataset`` read
#: (``labels_for``), so a count error names them, not a --labels never typed
_CLI_LABELS_FROM = contextvars.ContextVar("cli_labels_from", default=None)


def _label_files(given) -> list:
    """The label file(s) behind an argument of evaluate(), as paths; ``[]``
    for anything that is not a file path or a list of them."""
    if isinstance(given, (str, Path)) and io._is_label_file(Path(given)) \
            and Path(given).is_file():
        return [Path(given)]
    if _is_label_path_list(given):
        return [Path(p) for p in given]
    return []


def _batch_files(path: Path, n_cells: int) -> list:
    """``cty1.csv, cty2.csv, ...`` next to ``path`` (itself one of them), in
    number order, when their labels add up to ``n_cells``; else ``[]``."""
    import re
    m = re.fullmatch(r"(.*?)\d+(\.\w+)", path.name)
    if not m:
        return []
    head, suffix = m.groups()
    pattern = re.compile(re.escape(head) + r"(\d+)" + re.escape(suffix))
    found = sorted((int(pattern.fullmatch(q.name).group(1)), q)
                   for q in path.parent.iterdir() if pattern.fullmatch(q.name))
    try:
        total = sum(len(io.read_labels(q)) for _, q in found)
    except Exception:  # noqa: BLE001 - a file that does not read: no suggestion
        return []
    return [q for _, q in found] if len(found) > 1 and total == n_cells else []


def _count_error(what: str, n: int, n_cells: int, given) -> str:
    """evaluate()'s error for ``n`` values of ``what`` against ``n_cells`` rows.

    On the command line it names the flag and the file(s) the user typed;
    for too few labels it adds the per-batch fix (the sibling files when
    they add up to the cell count).
    """
    from . import scib as escib
    files = _label_files(given)
    names = f" ({', '.join(f.name for f in files)})" if files else ""
    dataset = _CLI_LABELS_FROM.get() if what == "labels" else None
    if dataset:
        cli = (f"the label files of {dataset}{names} hold {n:,} labels for "
               f"{n_cells:,} cells in --output. Check that --output holds the "
               f"embedding of --method on {dataset}.")
        return config.hint(escib.count_error(what, n, n_cells), cli)
    msg = config.hint(
        escib.count_error(what, n, n_cells),
        f"{_FLAGS[what]} gave {n:,} {_UNITS[what]}{names} for {n_cells:,} cells "
        f"in --output.")
    if what != "labels" or n >= n_cells:
        return msg
    batches = _batch_files(files[0], n_cells) if len(files) == 1 else []
    order = f" ({', '.join(f.name for f in batches)})" if batches else ""
    return msg + " " + config.hint(
        "For a folder with one label file per batch, pass "
        "mtb.labels_for(dataset, category, method).",
        f"For several label files{order}, repeat --labels in the row order of "
        f"--output, or pass --dataset, --category and --method.")


def _plan_metrics(metrics, *, has_batch: bool, batch_given: bool):
    """Turn the ``metrics=`` argument into what :func:`multibench.eval.scib.compute` needs.

    Parameters
    ----------
    metrics
        The ``metrics`` argument of :func:`evaluate` (``None`` / family token /
        list of codes).
    has_batch : bool
        Whether a batch vector is available (given, or derived from a list of
        label files), i.e. whether batch metrics are computable.
    batch_given : bool
        Whether ``batch=`` itself was passed (drives the "batch changes
        nothing" warning).

    Returns
    -------
    tuple
        ``(group, only, slow)`` - the compute group (``"clustering"`` /
        ``"batch"`` / ``"all"``), the set of codes to compute (``None`` =
        every metric of the group) and whether kBET is among them.

    Raises
    ------
    ValueError
        A family token / code that needs batch labels when none are
        available, or a code evaluate() cannot compute (``PCR``).
    """
    clu, bat = _metric_families()
    # a family token computes kBET only when it is named
    family_bat = [c for c in bat if c != "kBET"]
    sel = catalog.metric_selection(metrics)
    if sel.explicit:
        outside = [c for c in sel.codes if c not in clu and c not in bat]
        if outside:
            raise ValueError(
                f"evaluate() cannot compute {outside}; it computes the scIB "
                f"families only: {clu + bat}")
        want_bat = [c for c in sel.codes if c in bat]
        want_clu = [c for c in sel.codes if c in clu]
        if want_bat and not has_batch:
            raise ValueError(
                f"{_metrics_spelling(metrics)} needs batch labels for "
                f"{_and_list(want_bat)}. "
                + config.hint("Pass batch=<vector>, or labels as a list of two or more "
                              "files. Each file then counts as one batch.",
                              "Pass --batch CSV, or two or more --labels files. Each "
                              "file then counts as one batch."))
        group = "all" if (want_bat and want_clu) else ("batch" if want_bat else "clustering")
        only, slow = set(sel.codes), "kBET" in sel.codes
    elif sel.family == "all":
        if metrics is None:
            # every applicable metric: the batch family joins in when a batch
            # vector exists to score it against
            group = "all" if has_batch else "clustering"
        else:
            if not has_batch:
                raise ValueError(
                    f"{_metrics_spelling(metrics)} needs batch labels for "
                    f"{_and_list(family_bat)}. "
                    + config.hint("Pass batch=<vector>, or metrics='clustering'.",
                                  "Pass --batch CSV, or --metrics clustering."))
            group = "all"
        only, slow = None, False
    elif sel.family == "batch":
        if not has_batch:
            raise ValueError(
                f"{_metrics_spelling(metrics)} needs batch labels for "
                f"{_and_list(family_bat)}. "
                + config.hint("Pass batch=<vector>, or labels as a list of two or "
                              "more files.",
                              "Pass --batch CSV, or two or more --labels files."))
        group, only, slow = "batch", None, False
    else:
        group, only, slow = "clustering", None, False
    if group == "clustering" and batch_given:
        warnings.warn(
            config.hint("batch= changes nothing here, because ",
                        "--batch changes nothing here, because ")
            + f"{_metrics_spelling(metrics)} has no batch metric. Add "
            + _and_list(family_bat, "or")
            + config.hint(", or pass metrics='all'.",
                          " to --metrics, or pass --metrics all."),
            UserWarning, stacklevel=4)
    return group, only, slow


def _metrics_spelling(metrics) -> str:
    """``metrics=['ARI', 'NMI']`` in Python, ``--metrics ARI,NMI`` on the command line."""
    if metrics is not None and not isinstance(metrics, str):
        codes = [str(m) for m in metrics]
        return config.hint(f"metrics={codes!r}", f"--metrics {','.join(codes)}")
    return config.hint(f"metrics={metrics!r}", f"--metrics {metrics}")


def _and_list(items, last: str = "and") -> str:
    """``['a', 'b', 'c']`` -> ``'a, b and c'``."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} {last} {items[-1]}"


def _legacy_evaluate_kwargs(kw: dict) -> dict:
    """0.2.x spellings of :func:`evaluate`'s keywords: map or refuse.

    ``task=`` / ``family=`` become ``metrics=<token>`` and ``only=[...]``
    becomes ``metrics=[...]`` (each with a ``DeprecationWarning``);
    ``slow_metrics``, ``column`` and ``metric_set`` are gone and raise
    ``TypeError`` naming the replacement.
    """
    removed = {
        "slow_metrics": "pass metrics=[...] without cLISI/iLISI (kBET is computed "
                        "only when it is named in that list)",
        "column": "pass the Series/column itself as labels= / batch= / clustering=",
        "metric_set": "only the scIB metric set exists - drop the argument",
    }
    for name, fix in removed.items():
        if name in kw:
            raise TypeError(f"evaluate() got {name}=, removed in 0.3.0: {fix}")
    legacy = {n: kw.pop(n) for n in ("task", "family", "only") if n in kw}
    if not legacy:
        return kw
    if "metrics" in kw:
        raise TypeError(
            f"evaluate() got metrics= together with the deprecated "
            f"{sorted(legacy)}; pass metrics= only")
    token = legacy.get("family", legacy.get("task"))
    if "family" in legacy and "task" in legacy and legacy["family"] != legacy["task"]:
        token = legacy["family"]          # family won over task in 0.2.x
    if token == "dimension_reduction":
        token = "clustering"              # one metric group in the paper
    for name in ("task", "family"):
        if name in legacy:
            _compat.warn(f"evaluate({name}=...)", f"metrics={token!r}", stacklevel=4)
    if "only" in legacy:
        only = legacy["only"]
        if isinstance(only, str):
            raise TypeError(
                f"only= must be a collection of metric names, e.g. only={{{only!r}}}; "
                f"got the string {only!r} (and only= is deprecated: pass metrics=[...])")
        codes = [catalog.canonical_metric(m) or m for m in only]
        if isinstance(only, (set, frozenset)):
            codes = sorted(codes)          # a set has no order to preserve
        _compat.warn("evaluate(only=...)", f"metrics={codes!r}", stacklevel=4)
        if token not in (None, "all"):
            clu, bat = _metric_families()
            fam = clu if token == "clustering" else bat
            bad = [c for c in codes if c in (clu + bat) and c not in fam]
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} not in the {token!r} family: pass "
                    f"metrics={codes!r} alone (the family token is deprecated)")
        kw["metrics"] = codes
    else:
        kw["metrics"] = token
    return kw


@_compat.legacy_kwargs(_legacy_evaluate_kwargs)
def evaluate(
    output,
    labels=None,
    *,
    category: str | None = None,
    batch=None,
    metrics=None,
    clustering=None,
    obsm: str = "X_emb",
    label_order=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute scIB metrics for a run output (an embedding) against cell-type labels.

    Pass the embedding a run produced and one label per cell. Reshape the
    result with ``mtb.to_long`` to plot it or combine it with
    ``mtb.load_results``.

    Parameters
    ----------
    output
        Embedding, cells x dims: array, DataFrame, sparse matrix, AnnData or
        MuData (``.obsm[obsm]``; ``labels``/``batch`` may name ``.obs``
        columns), or file path (Notes).
    labels
        Cell types, one per cell (required): CSV path(s), a
        ``mtb.labels_for`` dict, a 1-D array-like, or an ``.obs`` column name
        (forms in Notes).
    category : str, optional
        Integration category, one of ``mtb.list_categories()``; validated,
        otherwise unused (the metrics do not depend on it). Lets a call mirror
        ``mtb.run``.
    batch
        Batch labels, one per cell, in the forms ``labels`` accepts; needed
        for the batch metrics unless ``labels`` lists two or more files.
    metrics : None, str or list of str
        ``None`` = every applicable metric except kBET; or ``"clustering"``,
        ``"batch"``, ``"all"``, or a list of codes such as ``["ARI", "NMI"]``.
    clustering
        Precomputed cluster assignment, in the forms ``labels`` accepts or an
        ``.h5`` path; ``None`` = derive one with the Leiden sweep.
    obsm : str
        ``.obsm`` key of an AnnData, MuData or ``.h5ad`` ``output``, such as
        ``'X_pca'``; ``'X'`` means ``.X``.
    label_order : list of str
        Keys of a multi-entry ``labels`` dict, in the method's cell order; a
        subset selects those files. ``None`` works for an unchanged
        ``mtb.labels_for`` dict (Notes).
    verbose : bool
        ``True`` prints one stderr line when the Leiden sweep starts on more
        than 2,000 cells; ``False`` never prints.

    Returns
    -------
    pandas.DataFrame
        One row per metric, indexed by the canonical metric name (``ARI``,
        ``NMI``, ...), with one column ``Value``. ``attrs`` records how the
        scores were computed (Notes).

    Raises
    ------
    ValueError
        Missing or misaligned labels, an unknown category or metric, or batch
        metrics without batch labels.
    FileNotFoundError
        An ``output`` or label path does not exist.
    TypeError
        An unsupported input type, or a retired keyword (Notes).

    Examples
    --------
    >>> import multibench as mtb
    >>> labels = mtb.labels_for("D11")                    # {'cty': '.../cty.csv'}
    >>> scores = mtb.evaluate(res.output, labels=labels)  # res = mtb.run(...)
    >>> # ASW and cLISI need no Leiden sweep
    >>> mtb.evaluate(res.output, labels=labels, metrics=["ASW", "cLISI"])
    >>> mtb.evaluate(adata, labels="celltype", obsm="X_pca")
    >>> mtb.evaluate(mdata, labels="celltype", batch="sample", obsm="X_joint",
    ...              metrics="all")

    Notes
    -----
    **Label forms.** ``labels`` may be:

    - a CSV path (header row; column ``x``, the only column, or the last of
      two when the first is a barcode index);
    - a list of CSV paths, concatenated in that order (multi-batch datasets:
      ``[cty1, cty2, cty3]``);
    - a ``{name: path}`` dict, such as ``mtb.labels_for`` returns (order
      rules under **Label dicts.**);
    - a 1-D ``ndarray`` / ``Series`` / ``Categorical`` / list, or a
      single-column DataFrame;
    - when ``output`` is an AnnData or MuData, the name of an ``.obs``
      column.

    A multi-column CSV / DataFrame raises: pass the one column
    (``df["celltype"]``) itself. ``batch`` and ``clustering`` take the same
    forms except a multi-entry dict; for ``clustering``, a path that is not
    ``.csv`` / ``.tsv`` / ``.txt`` is read as an h5 from
    ``/obs/cluster_leiden``.

    **Label dicts.** A dict with several entries needs a known order:

    - a dict from ``mtb.labels_for``, unchanged, is used as it is, in the
      order ``labels_for`` gave it (with ``category`` and ``method``, that
      method's cell order);
    - any other dict goes in as is only in the default order
      (``cty1, cty2, ...`` numerically; ``rna`` before ``adt`` before
      ``atac``) and otherwise needs ``label_order=`` naming its keys;
    - ``label_order=list(d)`` trusts the dict's own order. A one-entry dict
      needs no ``label_order``.

    **Metric selection.** ``None`` computes the clustering family (``ARI,
    NMI, ASW, iASW, iF1, cLISI``), plus ``ASW_batch, GC, iLISI`` when a batch
    vector is available (``batch=`` or a list of label files). kBET is never
    included by default: it shells out to R and takes hours on large
    datasets.

    - ``"clustering"`` / ``"batch"`` - that family; ``"all"`` - both
      (``"batch"`` and ``"all"`` need the batch vector or raise);
    - a list of codes - exactly those (case/alias tolerant, ``["ari"]``
      works); the Leiden sweep runs only when one of them needs it (ARI,
      NMI, iF1), and ``"kBET"`` in the list turns kBET on.

    Valid codes: ``mtb.plot.CLUSTERING_METRICS + mtb.plot.BATCH_METRICS``.
    An unknown code raises ``ValueError`` listing the valid ones; a bare
    code string (``metrics="ARI"``) raises and points at the list form.

    **Cost and Leiden backend.** ``ARI``, ``NMI`` and ``iF1`` need the scIB
    optimal-resolution Leiden sweep (10 resolutions on a kNN graph of the
    embedding). ``mtb.config.DEFAULT.leiden_flavor`` sets its backend:
    ``"igraph"`` (default) or ``"leidenalg"``, the classic backend scib
    itself runs. With ``leidenalg`` the sweep takes tens of seconds for a
    few thousand cells and minutes for ~10^4; ``igraph`` is several times
    faster.

    To skip it, name only metrics that do not need it in ``metrics=[...]``
    (``ASW``, ``iASW``, ``cLISI``, the batch family). ``clustering=``
    removes the need for ``ARI`` / ``NMI`` only; ``iF1`` always sweeps.

    **Metric definitions.** All metrics are scaled so that higher is better.
    Most lie between 0 and 1; ARI can be slightly below 0, which means a
    random clustering.

    Each metric is a scib function on the embedding. The package requires
    ``scib >= 1.1``; ``attrs["scib_version"]`` records the installed version.

    The metrics look at different neighbourhoods:

    - the Leiden sweep (ARI, NMI, iF1) and GC use scanpy's default
      neighbour graph (15 neighbours);
    - cLISI and iLISI: scib builds scanpy's 15-neighbour graph from the
      embedding. For each cell it takes the 90 cells with the shortest
      paths on that graph, where the length of an edge is its connectivity
      weight (``obsp["connectivities"]``), not its distance. Perplexity 30.
      A cell with fewer than 90 reachable cells counts as one label;
    - ASW, iASW and ASW_batch use distances in the embedding;
    - kBET builds its own neighbour graph.

    The sweep clusters at 10 resolutions (0.2 to 2.0) and keeps the one with
    the highest NMI. ``iso_threshold`` = number of batches + 1 counts every
    cell type as isolated; scib's default counts only types found in few
    batches.

    The scib call behind each code:

    ```text
    code       scib function         arguments
    ARI, NMI   ari, nmi              Leiden clusters, best sweep resolution
    ASW        silhouette            cell-type labels; rescaled to 0-1 by scib
    iASW       isolated_labels_asw   iso_threshold = number of batches + 1
    iF1        isolated_labels_f1    same threshold; best F1 over the sweep
    cLISI      clisi_graph           type_="embed"; 90 nearest by path over
                                     the connectivity weights of the
                                     15-neighbour graph; scaled to 0-1
    ASW_batch  silhouette_batch      1 - |batch silhouette| per cell type
    GC         graph_connectivity    on the 15-neighbour graph
    iLISI      ilisi_graph           type_="embed"; 90 nearest by path over
                                     the connectivity weights of the
                                     15-neighbour graph; scaled to 0-1
    kBET       kBET                  computed only when named in metrics=[...]
    ```

    cLISI and iLISI take the median m of the per-cell scores and scale it:
    cLISI = (L - m)/(L - 1), iLISI = (m - 1)/(B - 1), with L cell types and
    B batches.

    The re-run tables were scored by multibench 0.2.1's ``evaluate`` with
    these definitions and the leidenalg backend. The published tables were
    computed by the benchmark; see the paper's Methods.

    **Batch.** The batch family (``ASW_batch, GC, iLISI, kBET``) needs batch
    labels. When ``labels`` is a list (or dict) of two or more files and
    ``batch`` is omitted, the file of origin (1, 2, ...) serves as the
    batch, the same rule ``mtb.run_all`` applies. ``batch=`` given with a
    ``metrics`` selection that has no batch metric changes nothing, and a
    ``UserWarning`` says so.

    **Cell order.** Arrays, lists and label files without a cell-id column
    are matched by position to the rows of ``output``. A ``Series`` /
    ``DataFrame`` with a non-default index is aligned by cell id when
    ``output`` carries ids (an AnnData, or a DataFrame with a non-default
    index): rows are reindexed to the output's order, and a missing or
    extra id raises ``ValueError`` naming the first ones.

    A CSV whose first column holds the output's cell ids, as
    ``obs[["batch"]].to_csv(path)`` writes it, is aligned by that column in
    the same way, and so is a list of such files. A first column where only
    some values are cell ids raises ``ValueError``.

    A first column of text that holds none of the cell ids is matched by
    position, with a ``UserWarning``. Numbers there, such as R's row
    numbers, are matched by position unless they are exactly the output's
    cell ids.

    When ``output`` is a bare array there is nothing to align against: the
    Series, or a CSV with text in its first column, is matched positionally
    and a ``UserWarning`` says so.

    **Result shape.** The frame has the ``metric.csv`` shape: index =
    metric, one column ``Value``, never empty. ``mtb.to_long`` makes the
    long frame (lowercase ``value``) that ``load_results`` returns.

    **Provenance.** ``attrs`` records how the scores were computed:

    - ``leiden_flavor`` - the backend of the Leiden sweep; ``None`` when no
      sweep ran;
    - ``clustering`` - where the clusters ARI and NMI scored came from:
      ``"sweep"`` or ``"user"``; ``None`` without ARI and NMI;
    - ``multibench_version`` and ``scib_version`` - the installed versions.

    ``mtb.to_long`` writes the first three to a ``scored_with`` column.

    **Output formats.** A file path may be ``.h5`` (dataset ``data``, the
    benchmark's ``embedding.h5``), ``.h5ad`` (read as an AnnData),
    ``.npy``, or ``.csv`` / ``.tsv``. A dims x cells array is transposed
    against the label count, with a warning.

    **Errors.** ``ValueError`` covers:

    - missing labels, or a batch metric / family requested without batch
      labels;
    - an unknown category, ``metrics`` token or code;
    - a count that differs from the embedding's cells (``'labels has M
      entries for N cells in the embedding.'``), and cell-id mismatches when
      aligning;
    - ambiguous label files, and a multi-entry labels dict (other than an
      unchanged ``labels_for`` one) out of the default order without
      ``label_order``;
    - unknown, repeated or no keys in ``label_order``;
    - an ``.h5`` output without dataset ``data``.

    ``FileNotFoundError`` names the missing path and the working directory.
    ``TypeError`` covers unsupported input types (non-array ``labels``,
    ``label_order`` with non-dict ``labels``, ...).

    **Retired keywords.** ``task=``, ``family=`` and ``only=`` still work
    as spellings of ``metrics=``, with a ``DeprecationWarning``;
    ``slow_metrics=``, ``column=`` and ``metric_set=`` raise ``TypeError``
    naming the replacement. The Changes page lists them.

    See Also
    --------
    mtb.labels_for : the label files of a dataset, in the method's cell order.

    mtb.to_long : reshapes the result into the long results frame.

    mtb.run_all : runs and scores every runnable method on a dataset.
    """
    _validate_category(category)
    if labels is None:
        raise ValueError("metrics require `labels` (cty / ground-truth cell types).")
    # a labels_for() dict becomes the list of paths in cell order first, so the
    # file-of-origin batch rule below sees it like any other list of files
    if isinstance(labels, dict):
        labels = _labels_from_dict(labels, label_order)
    elif label_order is not None:
        raise TypeError(
            f"label_order= applies only when labels is a dict ({{name: path}}, "
            f"as returned by mtb.labels_for); labels is a "
            f"{type(labels).__name__} - pass the files as a list in that order "
            f"instead")
    batch_from_files = batch is None and _is_label_path_list(labels) and len(labels) > 1
    group, only, slow = _plan_metrics(
        metrics, has_batch=batch is not None or batch_from_files,
        batch_given=batch is not None)

    if isinstance(output, (str, Path)) and Path(output).suffix.lower() == ".h5ad":
        import anndata as ad      # read once; keeps obs_names and obs columns
        output = ad.read_h5ad(output)
    adata = output if io._is_anndata(output) else None
    ids = _cell_ids(output)
    emb = np.asarray(io.as_matrix(output, obsm=obsm))
    if _is_label_path_list(labels):
        # several label files: stacked in the given order (or aligned by their
        # id columns); the file of origin can serve as the batch below
        ct, origin = _stack_label_files(labels, ids)
    else:
        ct, origin = _obs_or_vector(labels, adata, what="labels", ids=ids), None
    # Orient to cells x dims. read_embedding() already does this for h5 inputs;
    # do the same for everything else so a caller can pass run().output (dims x
    # cells for many methods) directly. Transpose only when the label count
    # says the cells are on the other axis.
    if emb.ndim == 2 and emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        warnings.warn(
            f"output was (dims x cells) {emb.shape}; transposed to (cells x dims)")
        emb = emb.T
    if clustering is None:
        cl = None
    elif isinstance(clustering, (str, Path)) \
            and Path(clustering).suffix.lower() not in io._LABEL_FILE_SUFFIXES \
            and not (adata is not None and isinstance(clustering, str)
                     and clustering in adata.obs.columns):
        cl = io.read_clustering(clustering)        # the benchmark's h5 layout
    else:
        cl = _obs_or_vector(clustering, adata, what="clustering", ids=ids)
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ids)
    elif batch_from_files and group in {"batch", "all"}:
        # batch = which label file each cell came from (1-based, as run_all)
        ba = origin
    else:
        # clustering metrics need a batch_key but it is a no-op there, so a
        # constant vector is acceptable when no batch labels are supplied.
        ba = np.zeros(emb.shape[0], dtype=int)

    from . import scib as escib
    n_cells = emb.shape[0]
    for what, values, given in (("labels", ct, labels), ("clustering", cl, clustering),
                                ("batch", ba, batch)):
        if values is not None and len(values) != n_cells:
            raise ValueError(_count_error(what, len(values), n_cells, given))
    out = escib.compute(emb, ct, cl, ba, group=group, slow_metrics=slow, only=only,
                        verbose=None if verbose else False)
    # compute() already emits canonical names; canonicalise anyway, so a frame
    # concatenated with the published tables never carries a second spelling
    # of a metric
    out.index = pd.Index([catalog.canonical_metric(m) for m in out.index],
                         name=out.index.name)
    from .. import __version__
    out.attrs = {k: out.attrs.get(k) for k in PROVENANCE_ATTRS[:-1]}
    out.attrs["multibench_version"] = __version__
    out.attrs["scib_version"] = _scib_version()
    if out.empty:
        # unreachable after _plan_metrics; kept so a future metric family
        # mismatch fails loudly instead of handing back a (0, 1) frame
        raise ValueError(
            f"evaluate(metrics={metrics!r}) selected no metric - nothing to return")
    return out
