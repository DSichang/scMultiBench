"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import contextvars
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from .. import _compat, config
from ..config import _and_list
from ..data import catalog
from ..data.results import COLUMNS as _RESULT_COLUMNS

LONG_COLUMNS = list(_RESULT_COLUMNS)  # own copy: a caller that edits one list must not change the other

#: the ``attrs`` keys evaluate() sets and to_long() joins into ``scored_with``
PROVENANCE_ATTRS = ("leiden_flavor", "clustering", "multibench_version")

#: ``attrs`` keys of a frame scored with ``missing_labels="drop"``: cells scored, cells given
DROP_ATTRS = ("cells_scored", "cells_total")

#: ``scored_with`` of rows whose frame carries no record of how it was scored
UNKNOWN_SCORING = "unknown"

#: to_long's warning for such a frame
NO_SCORING_RECORD = (
    "These scores carry no record of how they were scored. Their scored_with is "
    "\"unknown\". A wide CSV read back loses that record. To keep it, save the "
    "mtb.to_long(...) table instead of the wide one.")


def _scib_version() -> str | None:
    """The installed scib version, or ``None`` when it cannot be read."""
    try:
        from importlib.metadata import version
        return version("scib")
    except Exception:  # noqa: BLE001 - no metadata: ask the module
        try:
            import scib
            return getattr(scib, "__version__", None)
        except Exception:  # noqa: BLE001
            return None


def _scored_with(attrs) -> str | None:
    """``"leidenalg/search/0.4.0"`` (plus ``"/270 of 300 cells"`` after a drop)
    from evaluate()'s ``attrs``; ``None`` when the frame carries none."""
    if not attrs or attrs.get("multibench_version") is None:
        return None
    parts = [str(attrs.get(k) or "none") for k in PROVENANCE_ATTRS]
    if all(attrs.get(k) is not None for k in DROP_ATTRS):
        # no thousands separators: a CSV column
        parts.append("{} of {} cells".format(*(attrs[k] for k in DROP_ATTRS)))
    return "/".join(parts)


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
        ``"user"``, the value ``load_results`` gives a user file without a
        category column.
    clustering : str
        Value of the ``clustering`` column. The published tables use
        ``"louvain"`` / ``"kmeans"`` for their variants.
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
        ``value_df`` is already long (pass it to the plot / ``load_results``
        consumers directly), lacks ``Value`` (for a wide one-row frame the
        message gives the ``df.T.set_axis(['Value'], axis=1)`` fix) or string
        metric names, has only blank names, or names one metric twice
        (``ari`` and ``ARI``).

    Warns
    -----
    UserWarning
        ``value_df`` has no record of how it was scored. Its ``scored_with`` is
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

    **Provenance.** The ``scored_with`` column of a frame from
    ``mtb.evaluate`` reads like ``"leidenalg/search/0.4.0"``: the Leiden
    backend, where the clusters came from (``search`` or ``user``;
    ``sweep`` before 0.4, the best-NMI resolution of a sweep) and the
    package version; ``none`` fills a part that did not apply. The same
    three values and ``scib_version`` are in ``attrs``.

    Scores of the labelled cells only (``missing_labels="drop"`` left cells
    out) have a fourth part, such as ``"igraph/search/0.4.0/270 of 300
    cells"``, and ``cells_scored`` and ``cells_total`` in ``attrs``.

    Any other frame, such as a hand-made one or a wide CSV read back, gets
    ``scored_with = "unknown"`` and a ``UserWarning``. To keep the record,
    save the long table: ``mtb.to_long(...).to_csv(path, index=False)``.

    The stored tables have no ``scored_with`` column, so it is NaN for their
    rows after ``pd.concat``. ``load_results(result_path=...)`` keeps the
    column when it reads the CSV back.

    **Plot chips.** The bubble figure shows ``?`` for a name the package
    does not know, such as your own or a renamed re-run.

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
        # saved with index_label="metric": reset_index would add a second 'metric' column
        value_df = value_df.set_index("metric")
    elif (cols == ["Unnamed: 0", "Value"]
          and all(map(_name_or_blank, value_df["Unnamed: 0"]))
          and any(isinstance(n, str) for n in value_df["Unnamed: 0"])
          and not any(isinstance(n, str) for n in value_df.index)):
        # a plain pd.read_csv of wide.to_csv(path): the names are column 'Unnamed: 0'
        value_df = value_df.set_index("Unnamed: 0")
    out = value_df.rename(columns={"Value": "value"}).copy()
    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "metric"})   # the index, whatever its name
    if not all(map(_name_or_blank, out["metric"])):
        # row numbers would pass canonical_metric as metrics '0', '1', ...
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
        for k in ("scib_version",) + DROP_ATTRS:
            if k in value_df.attrs:
                out.attrs[k] = value_df.attrs[k]
    return out


def _name_or_blank(v) -> bool:
    """A metric name (``str``) or a blank cell (``None``/NaN, which to_long drops)."""
    return isinstance(v, str) or (pd.api.types.is_scalar(v) and pd.isna(v))


def _metric_families() -> tuple[list, list, list]:
    """``(CLUSTERING_METRICS, BATCH_METRICS, CLASSIFICATION_METRICS)``, the
    families evaluate() can produce."""
    # the plotting layer owns the metric families
    from ..plot.bar import BATCH_METRICS, CLASSIFICATION_METRICS, CLUSTERING_METRICS
    return list(CLUSTERING_METRICS), list(BATCH_METRICS), list(CLASSIFICATION_METRICS)


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
    """Cell ids of ``output``: AnnData ``obs_names``, a DataFrame's index unless
    a ``RangeIndex``; ``None`` for anything else."""
    if io._is_anndata(output):
        return pd.Index(output.obs_names)
    if isinstance(output, pd.DataFrame) and not isinstance(output.index, pd.RangeIndex):
        return output.index
    return None


def _carries_ids(x) -> bool:
    """A Series / DataFrame whose index is not a ``RangeIndex`` carries cell ids."""
    return (isinstance(x, (pd.Series, pd.DataFrame))
            and not isinstance(x.index, pd.RangeIndex))


def _obs_column_name(x, adata) -> str | None:
    """The obs column ``x`` names, with or without ``obs:`` (a column named
    ``x`` itself wins), or ``None``."""
    if x in adata.obs.columns:
        return x
    if x.startswith("obs:") and x[4:] in adata.obs.columns:
        return x[4:]
    return None


def _mod_obs_column(x, adata):
    """The values of a MuData's ``"<mod>:<col>"`` (``mdata[mod].obs[col]``, as
    ``export_dataset`` reads it) in the MuData's cell order, or ``None``."""
    mods = getattr(adata, "mod", None)
    if not isinstance(mods, dict) or ":" not in x:
        return None
    mod, col = x.split(":", 1)
    if mod not in mods or col not in mods[mod].obs.columns:
        return None
    return pd.Series(mods[mod].obs[col]).reindex(adata.obs_names).to_numpy()


def _obs_choices(adata) -> list[str]:
    """The obs column names an argument can give: the object's own, then a
    MuData's ``<mod>:<col>``."""
    names = list(map(str, adata.obs.columns))
    for mod, a in (getattr(adata, "mod", None) or {}).items():
        names += [f"{mod}:{c}" for c in map(str, a.obs.columns)]
    return names


def _column_without_obs(x, adata) -> bool:
    """``x`` can only be an obs column name, and the output has no obs. The
    command line takes files only and keeps its file errors."""
    return adata is None and not config._CLI and io._names_no_file(x)


def _no_column_error(what: str, x: str) -> str:
    col = x[4:] if x.startswith("obs:") else x
    return (f"{what}={x!r} is not a file, and the output has no obs. Pass the values, as "
            f"{what}=adata.obs[{col!r}].")


class _Table:
    """A ``.csv`` / ``.tsv`` embedding whose first column gave the cell ids: its
    file name, and the arguments put in its row order (``moved``)."""

    def __init__(self, name: str):
        self.name, self.moved = name, []


def _same_cells(keys, ids, *, what: str, name: str, table: _Table, fix: str) -> bool:
    """Whether the cell ids ``keys`` of an argument are in another order than
    ``ids``, the barcodes of ``table``; ``ValueError`` when they name other cells.

    Ids of which none is a cell of the embedding are not an order: ``False``,
    and the caller matches by position with a warning.
    """
    keys, ids = pd.Index([str(k) for k in keys]), pd.Index([str(i) for i in ids])
    if not keys.isin(ids).any():
        return False
    extra = keys[~keys.isin(ids)].unique()
    missing = ids.difference(keys, sort=False)
    if len(extra) or len(missing):
        ix, a, b = (missing, table.name, name) if len(missing) else (extra, name, table.name)
        raise ValueError(
            f"{what}: {len(ix):,} {'cell' if len(ix) == 1 else 'cells'} of {a} "
            f"{'is' if len(ix) == 1 else 'are'} not in {b}, for example {ix[0]}. {fix}")
    return not keys.equals(ids)


def _shares_no_id(keys, ids) -> bool:
    """No id of ``keys`` is one of ``ids``."""
    return not pd.Index([str(k) for k in keys]).isin([str(i) for i in ids]).any()


def _file_fix(plural: bool = False) -> str:
    return "Write them for the same cells." if plural else "Write both for the same cells."


def _file_carries_ids(first, ids) -> bool:
    """The first column of a per-cell file holds cell ids: text that is unique
    or names at least one cell. Repeated text naming no cell is a batch column."""
    if first is None or not io._looks_like_ids(first):
        return False
    return bool(first.is_unique or pd.Index(first.astype(str)).isin(ids).any())


def _obs_or_vector(x, adata, *, what, column=None, ids=None, as_written=False,
                   table=None, cells=None):
    """One label-like argument as a 1-D array in the output's cell order: an obs
    column, a Series aligned by id, a file aligned by its id column, else by position.

    ``as_written`` reads files for a message naming the missing words;
    ``table`` (a ``.csv`` embedding's barcodes) collects the arguments put in
    its row order; ``cells`` words the error of ``io._align_vector``.
    """
    if adata is not None and isinstance(x, str):
        col = _obs_column_name(x, adata)
        if col is not None:
            return np.asarray(pd.Series(adata.obs[col]).to_numpy())
        in_mod = _mod_obs_column(x, adata)
        if in_mod is not None:
            return np.asarray(in_mod)
        if not Path(x).is_file():
            raise ValueError(
                f"{what}={x!r} is neither an obs column of the "
                f"{type(adata).__name__} (obs columns: {_obs_choices(adata)}) nor an "
                f"existing file")
    if _column_without_obs(x, adata):
        raise FileNotFoundError(_no_column_error(what, x))
    if isinstance(x, (str, Path)) and column is None:
        vals, first = io.read_labels_ids(x, what=what, pick=_pick(what),
                                         as_written=as_written)
        unique = ids is not None and pd.Index(ids).is_unique
        name = Path(x).name
        if table is not None and unique and _file_carries_ids(first, ids) \
                and _same_cells(first, ids, what=what, name=name, table=table,
                                fix=_file_fix()):
            table.moved.append(name)
        return io.by_id_column(
            vals, first, ids if unique else None, what=what, name=name,
            target="the output", order="the rows of the output",
            no_ids=_no_ids(ids), stacklevel=5, drop=what == "labels")[0]
    if _carries_ids(x):
        other = table is not None and _shares_no_id(x.index, ids)
        if ids is not None and not other:
            name = f"the {what} {type(x).__name__}"
            if table is not None and x.index.is_unique and pd.Index(ids).is_unique \
                    and _same_cells(x.index, ids, what=what, name=name, table=table,
                                    fix=f"Pass {what} for the cells of {table.name}."):
                table.moved.append(name)
            return io._align_vector(x, ids, what=what, column=column, cells=cells)
        warnings.warn(
            f"The {what} {type(x).__name__} is matched by position, because "
            + (f"none of its ids is a cell of {table.name}. Check that it follows the "
               f"rows of {table.name}." if other else
               "the embedding has no cell ids. Check that it follows the embedding rows, "
               "or pass an AnnData whose obs_names are the barcodes."),
            # _obs_or_vector <- evaluate <- its keyword wrapper <- the caller
            UserWarning, stacklevel=4)
    if what == "batch" and io._numbers_rows(x, ids):
        warnings.warn(io._by_position(what),
                      UserWarning, stacklevel=4)
    return io.as_vector(x, what=what, column=column, as_written=as_written)


def _pick(what: str):
    """The ``pick`` of ``io.read_labels_ids``: how to choose one column of a
    per-cell file, in Python or on the command line."""
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
    after its first column, in a folder ``stack`` removes; for ``--batch-column``.
    The copy keeps the file name and the missing words, so messages name them."""
    import tempfile
    path = io._require_file(path, f"{_FLAGS[what]} file")
    vals, first = io.read_labels_ids(path, column, what=what, as_written=True)
    out = Path(stack.enter_context(tempfile.TemporaryDirectory())) / path.name
    cols = {"x": vals} if first is None else {"id": first.to_numpy(), "x": vals}
    pd.DataFrame(cols).to_csv(out, index=False, sep=io._sep_for(out))
    return str(out)


def _no_ids(ids) -> str:
    return ("the output repeats cell ids" if ids is not None
            else "the output has no cell ids")


def _stack_label_files(files, ids, as_written=False, table=None):
    """Several label files, stacked in order, and each row's file (1, 2, ...).

    When every file has a first column of cell ids, the rows are aligned to
    ``ids`` by it; otherwise they stay in file order.
    """
    read = [io.read_labels_ids(f, what="labels", pick=_pick("labels"),
                               as_written=as_written) for f in files]
    vals = np.concatenate([np.asarray(v) for v, _ in read])
    origin = np.concatenate([np.full(len(v), i + 1) for i, (v, _) in enumerate(read)])
    names = [Path(f).name for f in files]
    unique = ids is not None and pd.Index(ids).is_unique
    order = "the rows of the output"
    if all(first is not None for _, first in read):
        name = _and_list(names)
        keys = pd.concat([first for _, first in read], ignore_index=True)
        if table is not None and unique and _file_carries_ids(keys, ids) \
                and _same_cells(keys, ids, what="labels", table=table, **(
                    dict(name=name, fix=_file_fix()) if len(names) == 1 else
                    dict(name="the label files", fix=_file_fix(plural=True)))):
            table.moved.extend(names)
        # a caller of evaluate is 5 frames up: by_id_column, this, evaluate, its wrapper
        pos, _ = io.by_id_column(np.arange(len(vals)), pd.concat(
            [first for _, first in read], ignore_index=True), ids if unique else None,
            what="labels", name=name, target="the output", order=order,
            no_ids=_no_ids(ids), stacklevel=5, n_files=len(names), drop=True)
        return vals[pos], origin[pos]
    for (v, first), name in zip(read, names):
        io.by_id_column(v, first, None, what="labels", name=name, target="the output",
                        order=order, stacklevel=5, no_ids=(
                            "the other label files have none" if unique
                            else _no_ids(ids)))
    return vals, origin


def _labels_from_dict(d: dict, label_order) -> list:
    """The paths of a ``{name: path}`` label dict in cell order.

    A guess at any other order would score misordered labels without an error,
    so several entries need ``label_order``, an unchanged ``labels_for`` dict,
    or the default order (``_label_sort_key``).
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
        # name the method's cell order only for a reordered labels_for dict that had one
        if isinstance(d, LabelFiles) and \
                list(d.stacking_order) != sorted(d.stacking_order, key=_label_sort_key):
            where = "are in neither the method's cell order nor the default order"
        else:
            where = "are not in the default order"
        raise ValueError(
            f"The label keys {_and_list(keys)} {where}. Pass the dict from "
            f"mtb.labels_for(dataset, category, method) unchanged, a list of "
            f"paths in cell order, or label_order=[...]. The default order is "
            f"cty1, cty2, ... by number, with rna before adt before atac. It is "
            f"not alphabetical.")
    if isinstance(label_order, str) or not isinstance(label_order, (list, tuple)):
        raise TypeError(
            f"label_order= takes a list of keys of the labels dict, such as "
            f"label_order={list(map(str, d))!r}. It got a "
            f"{type(label_order).__name__}.")
    order = list(label_order)
    if not order:
        raise ValueError(
            f"label_order= is empty. List the keys of the labels dict in the "
            f"method's cell order, such as {list(map(str, d))!r}.")
    unknown = list(dict.fromkeys(k for k in order if k not in d))
    if unknown:
        what = ("which is not a key" if len(unknown) == 1
                else "which are not keys")
        raise ValueError(
            f"label_order names {_and_list(unknown)}, {what} of the labels dict. "
            f"Its keys are {_and_list(d)}.")
    if len(set(order)) != len(order):
        times = {k: order.count(k) for k in dict.fromkeys(order) if order.count(k) > 1}
        raise ValueError("label_order names " + _and_list(
            f"{k} {'twice' if n == 2 else f'{n} times'}" for k, n in times.items()) + ".")
    return [d[k] for k in order]


#: the ``multibench evaluate`` flag that fills each argument of evaluate()
_FLAGS = {"labels": "--labels", "clustering": "--clustering", "batch": "--batch"}
#: what one value of each argument is called in the command-line messages
_UNITS = {"labels": "labels", "clustering": "cluster ids", "batch": "batch ids"}
#: the dataset whose label files ``multibench evaluate --dataset`` read, so the
#: errors name them, not a --labels never typed
_CLI_LABELS_FROM = contextvars.ContextVar("cli_labels_from", default=None)


def _label_files(given) -> list:
    """The label file(s) behind an argument of evaluate(), as paths; ``[]`` for
    anything else."""
    if isinstance(given, dict) and len(given) == 1:
        given = next(iter(given.values()))
    if isinstance(given, (str, Path)) and io._is_label_file(Path(given)) \
            and Path(given).is_file():
        return [Path(given)]
    if _is_label_path_list(given):
        return [Path(p) for p in given]
    return []


def _named_files(given) -> str:
    """`` (cty1.csv, cty2.csv)``: the files an argument named; ``""`` for values
    in memory."""
    files = _label_files(given)
    return f" ({', '.join(f.name for f in files)})" if files else ""


def _n_values(labels, adata) -> int | None:
    """The number of label values, read by position and silently; ``None`` when
    that read fails (the read with the cell ids then raises)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if _is_label_path_list(labels):
                return len(_stack_label_files(labels, None)[0])
            return len(_obs_or_vector(labels, adata, what="labels"))
        except Exception:  # noqa: BLE001 - the second read says what is wrong
            return None


def _as_written(given, adata, ids, *, what):
    """An argument of evaluate() that names files, read again in cell order with
    each missing value as the word the file holds; ``None`` for values in memory."""
    if not _label_files(given):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # the first read said them
        if what == "labels" and _is_label_path_list(given):
            return _stack_label_files(given, ids, as_written=True)[0]
        return _obs_or_vector(given, adata, what=what, ids=ids, as_written=True)


def _measured_warning(given, adata, reason: str, distinct: int, cells: int) -> str:
    """The warning for labels that look like a measured number per cell, such as
    ``nCount_RNA``; names the column, Series or file(s) when known."""
    arg = config.hint("labels", "--labels")
    files = [f.name for f in _label_files(given)]
    if adata is not None and isinstance(given, str) \
            and _obs_column_name(given, adata) is not None:
        src = config.hint(f"labels={given!r}", f"--labels {given}")
    elif files:
        src = f"{arg} ({files[0]}" + (f" and {len(files) - 1} more)" if len(files) > 1 else ")")
    elif isinstance(given, pd.Series) and given.name is not None:
        src = f"The labels Series {str(given.name)!r}"
    else:
        src = arg
    why = ("holds numbers that are not whole numbers" if reason == "fraction"
           else f"has {distinct:,} different labels for {cells:,} cells")
    return f"{src} {why}. Check that it holds the cell types."


def _no_label_error(shown, miss, given) -> str:
    """evaluate()'s error for labels missing where ``miss`` is true. ``shown``
    are the labels as written."""
    n, n_cells = int(miss.sum()), len(miss)
    dataset = _CLI_LABELS_FROM.get()
    head = config.hint("labels", f"the label files of {dataset}" if dataset
                       else "--labels") + _named_files(given)
    example = io._missing_example(shown)
    if n == n_cells:
        return (f"{head}: none of the {n_cells:,} cells has a label{example}. Pass the "
                f"column or file that holds the cell types.")
    drop = config.hint('missing_labels="drop"', "--missing-labels drop")
    return (f"{head}: {n:,} of the {n_cells:,} cells have no label{example}. Pass {drop} "
            f"to score the other {n_cells - n:,}.")


def _no_batch_error(shown, miss, given) -> str:
    """evaluate()'s error for a batch missing where ``miss`` is true. ``shown``
    is the batch as written."""
    head = config.hint("batch", "--batch") + _named_files(given)
    return (f"{head}: {int(miss.sum()):,} of the {len(miss):,} cells have no batch"
            f"{io._missing_example(shown)}. Give every cell a batch.")


def _n_batches(ba) -> int:
    """How many batches a batch vector names; a missing value is not one."""
    ba = np.asarray(ba)
    return len(set(ba[~io.missing_mask(ba)].astype(str).tolist()))


def _dropped_warning(n_scored: int, n_cells: int, batches=None) -> str:
    """evaluate()'s warning after ``missing_labels="drop"``. ``batches`` is
    ``(batches with a labelled cell, batches given)`` when a batch vector is known."""
    msg = f"Scored the {n_scored:,} labelled cells of {n_cells:,}"
    b, n_batches = batches or (0, 0)
    if b == 1 < n_batches:
        msg += f", all in one batch of {n_batches}"
    elif b < n_batches:
        msg += f", from {b} of the {n_batches} batches"
    return msg + "."


def _batch_files(path: Path, n_cells: int) -> list:
    """``cty1.csv, cty2.csv, ...`` next to ``path``, in number order, when their
    labels add up to ``n_cells``; else ``[]``."""
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


def _diagonal_side(files) -> tuple | None:
    """``("RNA", "ATAC")`` when every file is ``rna_cty<N>.csv`` and no folder of
    theirs holds an ATAC label file (``atac_cty`` or ``peak_cty``), the reverse
    for ``atac_cty<N>.csv``; else ``None``."""
    import re
    sides = {m.group(1) if m else None
             for m in (re.fullmatch(r"(rna|atac)_cty\d*", f.stem) for f in files)}
    if len(sides) != 1 or None in sides:
        return None
    side = sides.pop()
    other = re.compile(r"(atac|peak)_cty\d*" if side == "rna" else r"rna_cty\d*")
    if any(other.fullmatch(q.stem) and q.suffix.lower() in io._LABEL_FILE_SUFFIXES
           for folder in {f.parent for f in files} for q in folder.iterdir()):
        return None
    return ("RNA", "ATAC") if side == "rna" else ("ATAC", "RNA")


def _count_error(what: str, n: int, n_cells: int, given, category=None) -> str:
    """evaluate()'s error for ``n`` values of ``what`` against ``n_cells`` rows.

    On the command line it names the flag and the files typed. Too few labels
    get the per-batch fix, or, for one side of a diagonal folder, the "No
    labels" recipe.
    """
    from . import scib as escib
    files, names = _label_files(given), _named_files(given)
    dataset = _CLI_LABELS_FROM.get() if what == "labels" else None
    side = _diagonal_side(files) if what == "labels" and n < n_cells else None
    if dataset:
        cli = (f"the label files of {dataset}{names} hold {n:,} labels for "
               f"{n_cells:,} cells in --output. ")
        if side and category == "diagonal":
            cli += (f'{dataset} labels the {side[0]} cells only, see the recipe '
                    f'"No labels" in the docs.')
        else:
            cli += f"Check that --output holds the embedding of --method on {dataset}."
        return config.hint(escib.count_error(what, n, n_cells), cli)
    msg = config.hint(
        escib.count_error(what, n, n_cells),
        f"{_FLAGS[what]} gave {n:,} {_UNITS[what]}{names} for {n_cells:,} cells "
        f"in --output.")
    if what != "labels" or n >= n_cells:
        return msg
    batches = _batch_files(files[0], n_cells) if len(files) == 1 else []
    if side and not batches:
        return (f'{msg} {files[0].name} labels the {side[0]} cells only, see the recipe '
                f'"No labels" in the docs.')
    order = f" ({', '.join(f.name for f in batches)})" if batches else ""
    return msg + " " + config.hint(
        "For a folder with one label file per batch, pass "
        "mtb.labels_for(dataset, category, method).",
        f"For several label files{order}, repeat --labels in the row order of "
        f"--output, or pass --dataset, --category and --method.")


def _plan_metrics(metrics, *, has_batch: bool, batch_given: bool):
    """``(group, only, slow, cls)`` for ``scib.compute`` from ``metrics=``.

    ``group`` is ``"clustering"`` / ``"batch"`` / ``"all"`` or ``None`` (no scIB
    metric), ``only`` the codes (``None`` = the whole group), ``slow`` whether
    kBET is named, and ``cls`` ``None`` or ``(how, codes)``: ``how`` is
    ``"default"`` (scored when the data allow), ``"all"`` (NaN with a warning
    when not) or ``"explicit"`` (an error when not).
    """
    clu, bat, cls = _metric_families()
    # a family token computes kBET only when it is named
    family_bat = [c for c in bat if c != "kBET"]
    sel = catalog.metric_selection(metrics)
    want_cls = None
    if sel.explicit:
        outside = [c for c in sel.codes if c not in clu + bat + cls]
        if outside:
            raise ValueError(
                f"evaluate() cannot compute {outside}; it computes "
                f"{clu + bat + cls}")
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
        codes = [c for c in cls if c in sel.codes]        # in family order, as scIB's
        if codes:
            want_cls = ("explicit", codes)
        group = ("all" if (want_bat and want_clu) else "batch" if want_bat
                 else "clustering" if want_clu else None)
        only, slow = set(sel.codes) - set(cls), "kBET" in sel.codes
    elif sel.family == "all":
        if metrics is None:
            # the batch family needs a batch vector; classification is decided later
            group = "all" if has_batch else "clustering"
            want_cls = ("default", cls) if has_batch else None
        else:
            if not has_batch:
                raise ValueError(
                    f"{_metrics_spelling(metrics)} needs batch labels for "
                    f"{_and_list(family_bat)}. "
                    + config.hint("Pass batch=<vector>, or metrics='clustering'.",
                                  "Pass --batch CSV, or --metrics clustering."))
            group, want_cls = "all", ("all", cls)
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
    elif sel.family == "classification":
        group, only, slow, want_cls = None, None, False, ("explicit", cls)
    else:
        group, only, slow = "clustering", None, False
    if want_cls and want_cls[0] == "explicit" and not has_batch:
        raise ValueError(_classification_error(metrics, "no batch"))
    if group == "clustering" and batch_given and not want_cls:
        warnings.warn(
            config.hint("batch= changes nothing here, because ",
                        "--batch changes nothing here, because ")
            + f"{_metrics_spelling(metrics)} has no batch metric. Add "
            + _and_list(family_bat, "or")
            + config.hint(", or pass metrics='all'.",
                          " to --metrics, or pass --metrics all."),
            UserWarning, stacklevel=4)
    return group, only, slow, want_cls


def _classification_error(metrics, why: str, what: str = "output") -> str:
    """evaluate()'s error when the classification codes cannot be computed;
    ``why`` is ``"no batch"``, ``"one batch"``, ``"graph"``, ``"not embedding"``
    or ``"none scored"``."""
    head = _metrics_spelling(metrics)
    if why == "graph":
        from .classify import _graph_text
        return _graph_text(what) + " " + config.hint("Pass metrics='clustering'.",
                                                     "Pass --metrics clustering.")
    if why == "not embedding":
        return f"{head} needs an embedding; {what} is not one. Pass the method's embedding file."
    if why == "none scored":
        return (f"{head} found no batch to learn from: each needs two cell types that "
                f"another batch has. Check labels and batch.")
    fix = config.hint("Pass batch=<vector>, or labels as a list of two or more files.",
                      "Pass --batch CSV, or two or more --labels files.")
    if why == "one batch":
        return f"{head} needs two or more batches; every cell is in one batch. {fix}"
    return f"{head} needs two or more batches. {fix}"


def _classification_warning(why: str, what: str = "output") -> str:
    """evaluate()'s one warning when classification scores it was not asked for
    explicitly cannot be computed."""
    head = "OCA, ACA, Spec, Sens and F1 are NaN"
    if why == "graph":
        who = "output is a graph" if what == "output" else f"{what} writes a graph"
        return (f"{who}, not an embedding, so OCA, ACA, Spec, Sens and F1 are NaN. "
                + config.hint("Pass metrics='clustering' to leave them out.",
                              "Pass --metrics clustering to leave them out."))
    if why == "not embedding":
        return f"{head}: {what} is not an embedding. Pass the method's embedding file."
    if why == "none scored":
        return (f"{head}: no batch has two cell types that another batch has. "
                f"Check labels and batch.")
    return f"{head}: every cell is in one batch. " + config.hint(
        "Pass batch= with two or more batches.", "Pass --batch with two or more batches.")


def _classification_scores(request, emb, ct, cb, *, category, not_embedding, graph,
                            metrics):
    """``{code: value}`` of the classification rows, ``{}`` when left out: the
    ``mean`` row of ``classification_metrics(classify(...))`` with seed 0.

    ``cb`` is the batch of each scored cell. Under ``metrics=None`` they are
    scored for two or more batches outside ``"vertical"``; when they cannot be,
    an explicit request raises and any other gives NaN and one warning.
    """
    how, codes = request
    n_batches = 0 if cb is None else _n_batches(cb)
    if how == "default" and (n_batches < 2 or category == "vertical"):
        return {}
    why = ("graph" if graph else "not embedding" if not_embedding
           else "one batch" if n_batches < 2 else None)
    what = graph or not_embedding or "output"
    mean = None
    if why is None:
        from .classify import classification_metrics, classify
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred = classify(emb, labels=np.asarray(ct, dtype=object),
                            batch=np.array([str(b) for b in cb], dtype=object), seed=0)
        table = classification_metrics(pred)
        if "mean" in table.index:
            mean = table.loc["mean"]
            for w in caught:      # a skipped reference batch: said once, as classify says it
                warnings.warn(w.message, w.category, stacklevel=4)
            one_type = [str(r) for r in table.index[:-1] if np.isnan(table.loc[r, "Spec"])]
            if "Spec" in codes and one_type:
                # specificity needs cells of a second true cell type
                warnings.warn(
                    f"Spec is NaN: when batch {_and_list(one_type, 'or')} is the reference, "
                    f"the cells it predicts are all of one cell type. Compare methods on "
                    f"the other metrics.", UserWarning, stacklevel=4)
        else:
            why = "none scored"
    if why is not None:
        if how == "explicit":
            raise ValueError(_classification_error(metrics, why, what))
        # evaluate <- its keyword wrapper <- the caller
        warnings.warn(_classification_warning(why, what),
                      UserWarning, stacklevel=4)
        return {c: float("nan") for c in codes}
    return {c: float(mean[c]) for c in codes}


def _origin_names(labels, origin):
    """The batch of each cell when several label files give it: the file's stem,
    as ``mtb.classify`` names it, else its number."""
    stems = [Path(p).stem for p in labels]
    names = stems if len(set(stems)) == len(stems) else \
        [str(i + 1) for i in range(len(labels))]
    return np.asarray(names, dtype=object)[np.asarray(origin) - 1]


def _metrics_spelling(metrics) -> str:
    """``metrics=['ARI', 'NMI']`` in Python, ``--metrics ARI,NMI`` on the
    command line."""
    if metrics is not None and not isinstance(metrics, str):
        codes = [str(m) for m in metrics]
        return config.hint(f"metrics={codes!r}", f"--metrics {','.join(codes)}")
    return config.hint(f"metrics={metrics!r}", f"--metrics {metrics}")


def _legacy_evaluate_kwargs(kw: dict) -> dict:
    """0.2.x keywords of :func:`evaluate`: ``task=`` / ``family=`` / ``only=``
    become ``metrics=`` with a ``DeprecationWarning``; the removed ones raise."""
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
            clu, bat, cls = _metric_families()
            fam = {"clustering": clu, "batch": bat, "classification": cls}.get(token, [])
            bad = [c for c in codes if c in (clu + bat + cls) and c not in fam]
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
    missing_labels: str = "error",
) -> pd.DataFrame:
    """Compute the benchmark's metrics for a run output (an embedding) against cell-type labels.

    Pass the embedding a run produced and one label per cell. Reshape the
    result with ``mtb.to_long`` to plot it or combine it with
    ``mtb.load_results``.

    Parameters
    ----------
    output
        Embedding, cells x dims: array, DataFrame, sparse matrix, AnnData or
        MuData (``.obsm[obsm]``; ``labels``/``batch`` may name ``.obs``
        columns), or a file: ``.h5`` (dataset ``data``, the benchmark's
        ``embedding.h5``), ``.h5ad``, ``.npy``, ``.csv`` / ``.tsv``. A dims x
        cells array is transposed against the label count, with a warning.
    labels
        Cell types, one per cell (required): CSV path(s), a
        ``mtb.labels_for`` dict, a 1-D array-like, or an ``.obs`` column
        name, with or without the ``obs:`` prefix (forms in Notes).
    category : str, optional
        Integration category, one of ``mtb.list_categories()``; validated,
        otherwise unused (the metrics do not depend on it). Lets a call mirror
        ``mtb.run``.
    batch
        Batch labels, one per cell, in the forms ``labels`` accepts; needed
        for the batch metrics unless ``labels`` lists two or more files.
        Every cell needs a batch.
    metrics : None, str or list of str
        ``None`` = every applicable metric except kBET; or ``"clustering"``,
        ``"batch"``, ``"classification"``, ``"all"``, or a list of codes such
        as ``["ARI", "NMI"]``.
    clustering
        Precomputed cluster assignment, in the forms ``labels`` accepts or an
        ``.h5`` path, used for ARI, NMI, ASW, iASW and iF1; ``None`` = Leiden
        clusters, one per cell type.
    obsm : str
        ``.obsm`` key of an AnnData, MuData or ``.h5ad`` ``output``, such as
        ``'X_pca'``; ``'X'`` means ``.X``.
    label_order : list of str
        Keys of a multi-entry ``labels`` dict, in the method's cell order; a
        subset selects those files. ``None`` works for an unchanged
        ``mtb.labels_for`` dict (Notes).
    verbose : bool
        ``True`` prints one stderr line when the Leiden clustering starts on more
        than 2,000 cells, and one when labels, batches or clusters are put
        in the row order of a ``.csv`` / ``.tsv`` output by barcode;
        ``False`` never prints.
    missing_labels : {"error", "drop"}
        What to do when a cell has no label. ``"error"`` raises. ``"drop"``
        scores the labelled cells only and warns how many (Notes).

    Returns
    -------
    pandas.DataFrame
        One row per metric, indexed by the canonical metric name (``ARI``,
        ``NMI``, ...), with one column ``Value``, never empty. ``attrs``
        records how the scores were computed (Notes). ``mtb.to_long`` makes
        the long frame (lowercase ``value``) that ``load_results`` returns.

    Raises
    ------
    ValueError
        Missing or misaligned labels, an unknown category or metric, or batch
        or classification metrics without batch labels. A cell without a label (unless
        ``missing_labels="drop"``), no cell with a label, or a cell without a
        batch in ``batch``. Also a count that differs from the embedding's
        cells, a ``missing_labels`` other than ``"error"`` / ``"drop"``,
        ambiguous label files, a labels dict out of order without
        ``label_order``, unknown, repeated or no keys in ``label_order``, and
        an ``.h5`` output without dataset ``data``.
    FileNotFoundError
        An ``output`` or label path does not exist (the message names it and
        the working directory), or a column is named for an ``output`` that
        has no ``.obs``.
    TypeError
        An unsupported input type, ``label_order`` with labels that are not
        a dict, or a retired keyword (Notes).

    Examples
    --------
    >>> import multibench as mtb
    >>> labels = mtb.labels_for("D11")                    # {'cty': '.../cty.csv'}
    >>> scores = mtb.evaluate(res.output, labels=labels)  # res = mtb.run(...)
    >>> # cLISI needs no Leiden clustering
    >>> mtb.evaluate(res.output, labels=labels, metrics=["cLISI"])
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
      column, with or without the ``obs:`` prefix (``"celltype"`` or
      ``"obs:celltype"``); for a MuData also ``"<mod>:<column>"``, a column
      of one modality (``"rna:celltype"`` = ``mdata["rna"].obs["celltype"]``),
      as ``export_dataset`` reads it.

    ``"obs:<column>"`` and ``"<column>"`` name a column of the object you
    pass. ``export_dataset`` and ``evaluate`` take both. ``run_all`` and
    ``rescore`` are given a folder, not the object, so they take the column
    itself, for example ``batch=adata.obs["donor"]``.

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

    **Missing labels.** A label is missing when it is NaN, ``None``,
    ``pd.NA``, blank text, or text that a label file reads as missing:
    ``NA``, ``None``, ``nan``, ``null``, ``N/A`` and the like, with or
    without blanks around them. A missing label is not a cell type, so it
    is never scored as one. A cell type that really has such a name needs
    another name.

    - ``missing_labels="error"`` raises ``ValueError``. The message says
      how many cells have no label and how the missing values are written;
    - ``missing_labels="drop"`` leaves those cells out of the embedding,
      the labels, ``batch`` and ``clustering``, and scores the rest. The
      scores equal those of a call with the labelled cells only. A
      ``UserWarning`` gives the two cell counts, and says when the labelled
      cells come from fewer batches;
    - when no cell has a label, both raise.

    The values of ``batch`` are missing by the same rule. A missing batch
    always raises ``ValueError``; with ``"drop"``, only the cells that are
    scored need a batch.

    **Metric selection.** ``None`` computes the clustering family (``ARI,
    NMI, ASW, iASW, iF1, cLISI``), plus ``ASW_batch, ARI_batch, NMI_batch,
    GC, iLISI`` when a batch
    vector is available (``batch=`` or a list of label files), plus the
    classification family (``OCA, ACA, Spec, Sens, F1``) when the cells are
    in two or more batches and ``category`` is not ``"vertical"``. kBET is
    never included by default: it shells out to R and takes hours on large
    datasets.

    - ``"clustering"`` / ``"batch"`` / ``"classification"`` - that family;
      ``"all"`` - all three (``"batch"``, ``"classification"`` and ``"all"``
      need the batch vector or raise);
    - a list of codes - exactly those (case/alias tolerant, ``["ari"]``
      works); Leiden runs only when one of them needs it (ARI, NMI, ASW,
      iASW, iF1, ARI_batch, NMI_batch), and ``"kBET"`` in the list turns
      kBET on.

    Valid codes: ``mtb.plot.CLUSTERING_METRICS + mtb.plot.BATCH_METRICS +
    mtb.plot.CLASSIFICATION_METRICS``.
    An unknown code raises ``ValueError`` listing the valid ones; a bare
    code string (``metrics="ARI"``) raises and points at the list form.

    **Cost and Leiden backend.** The protocol is the article's. ARI, NMI,
    ASW, iASW and iF1 score Leiden clusters searched to the number of cell
    types; ARI_batch and NMI_batch, clusters searched to the number of
    batches. ``mtb.config.DEFAULT.leiden_flavor`` sets the backend:
    ``"igraph"`` (default) or ``"leidenalg"``, scanpy's default before 1.10.
    With ``leidenalg`` the clustering takes tens of seconds for a few
    thousand cells and minutes for ~10^4; ``igraph`` is several times
    faster.

    To skip it, name only metrics that do not need it in ``metrics=[...]``
    (``cLISI``, ``ASW_batch``, ``GC``, ``iLISI``, ``kBET``). ``clustering=``
    replaces the cell-type clusters only.

    **Metric definitions.** All metrics are scaled so that higher is better.
    Most lie between 0 and 1; ARI can be slightly below 0, which means a
    random clustering.

    Each metric is a scib function on the embedding. The package requires
    ``scib >= 1.1``; ``attrs["scib_version"]`` records the installed version.

    The metrics look at different neighbourhoods:

    - the cluster search (ARI, NMI, ASW, iASW, iF1, ARI_batch, NMI_batch) uses
      scanpy's default 15-neighbour graph without ``use_rep``, as the
      article: the first 50 principal components of an embedding wider than
      50, NaN read as 0;
    - GC uses the 15-neighbour graph of the embedding;
    - cLISI and iLISI: scib builds scanpy's 15-neighbour graph from the
      embedding. For each cell it takes the 90 cells with the shortest
      paths on that graph, where the length of an edge is its connectivity
      weight (``obsp["connectivities"]``), not its distance. Perplexity 30.
      A cell with fewer than 90 reachable cells counts as one label;
    - ASW, iASW and ASW_batch use distances in the embedding;
    - kBET builds its own neighbour graph.

    The search bisects the Leiden resolution on [0, 3] for at most 30 steps
    until it gives the number wanted (``sinfonia.get_N_clusters``); when no
    step does, the last step's clusters are scored, with a ``UserWarning``.
    ``iso_threshold`` = number of batches + 1 counts every label as
    isolated; scib's default counts only labels found in few batches.

    The scib call behind each code:

    ```text
    code       scib function         arguments
    ARI, NMI   ari, nmi              clusters, one per cell type
    ASW        silhouette            those clusters; rescaled to 0-1 by scib
    iASW       isolated_labels_asw   those clusters; iso_threshold = number
                                     of batches + 1
    iF1        isolated_labels_f1    cell types, same threshold; best F1
                                     of one of those clusters
    cLISI      clisi_graph           type_="embed"; 90 nearest by path over
                                     the connectivity weights of the
                                     15-neighbour graph; scaled to 0-1
    ASW_batch  silhouette_batch      1 - |batch silhouette| per cell type
    ARI_batch  ari                   1 - |ARI| of clusters, one per batch,
                                     against the batches
    NMI_batch  nmi                   1 - |NMI|, the same way
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
    the leidenalg backend and an earlier protocol: the clusters of the best
    NMI of a 10-resolution sweep, iF1 over that sweep, ASW and iASW on the
    cell types, no ARI_batch or NMI_batch. The published tables were
    computed by the benchmark; see the paper's Methods.

    **Batch.** When ``labels`` lists two or more files (or a dict of them)
    and ``batch`` is omitted, the file of origin (1, 2, ...) serves as the
    batch, as in ``mtb.run_all``. ``batch=`` with a selection that has no
    batch metric changes nothing, and a ``UserWarning`` says so.

    **Classification.** The five values are the ``mean`` row of
    ``mtb.classification_metrics(mtb.classify(...))`` with seed 0: each batch
    in turn is the reference. They need an embedding and two or more
    batches: the output of a graph method (its neighbour index, or the UMAP
    scMoMaT writes) has none. When they cannot be computed,
    ``"classification"`` or a list of their codes raises ``ValueError``;
    ``"all"``, and ``None`` with two or more batches, give NaN and one
    ``UserWarning``. ``Spec`` is NaN, with a ``UserWarning``, when the cells
    a reference batch predicts are all of one cell type.

    **Cell order.** Arrays, lists and label files without a cell-id column
    are matched by position to the rows of ``output``. When ``output``
    carries cell ids (an AnnData, a DataFrame with a non-default index, or a
    ``.csv`` / ``.tsv`` file whose first column holds text, the barcodes),
    a ``Series`` / ``DataFrame`` with a non-default index is aligned by id,
    and so is a CSV whose first column holds cell ids (as
    ``obs[["batch"]].to_csv(path)`` writes it), or a list of such files. A
    missing or extra id raises ``ValueError`` naming the first ones (for a
    ``.csv`` / ``.tsv`` output, their number and one of them), and so does a
    first column where only some values are cell ids.

    A file, or a Series against a ``.csv`` / ``.tsv`` output, whose ids name
    none of the cells is matched by position with a ``UserWarning``; such a
    Series raises for an AnnData. Numbers in a first column, such as R's row
    numbers, are matched by position unless they are exactly the output's
    cell ids. A bare-array ``output`` has nothing to align against: the
    Series, or a CSV with text in its first column, is matched by position
    with a ``UserWarning``.

    A ``.csv`` / ``.tsv`` output in another row order is matched by barcode,
    and a line on stderr says so unless ``verbose=False``. Its first column
    of numbers is left out without a header (row numbers, no ids) and is a
    column of the embedding with one.

    **Provenance.** ``attrs`` records how the scores were computed:

    - ``leiden_flavor`` - the Leiden backend; ``None`` when Leiden did not
      run;
    - ``clustering`` - where the clusters of ARI, NMI, ASW, iASW and iF1 came
      from: ``"search"`` or ``"user"``; ``None`` without those metrics;
    - ``multibench_version`` and ``scib_version`` - the installed versions.

    ``mtb.to_long`` writes the first three to a ``scored_with`` column.
    When ``missing_labels="drop"`` left cells out, ``attrs`` also holds
    ``cells_scored`` and ``cells_total``, and ``scored_with`` ends with
    them (``.../270 of 300 cells``).

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
    if not isinstance(missing_labels, str) or missing_labels not in ("error", "drop"):
        raise ValueError(
            f'missing_labels must be "error" or "drop", got {missing_labels!r}.')
    _validate_category(category)
    if labels is None:
        raise ValueError("metrics require `labels` (cty / ground-truth cell types).")
    if isinstance(labels, dict) and not labels:
        raise ValueError(
            "labels is an empty dict. Pass the cell types, as "
            "labels=adata.obs['celltype'].")
    # a dict becomes a list of paths first, for the file-of-origin batch rule
    if isinstance(labels, dict):
        labels = _labels_from_dict(labels, label_order)
    elif label_order is not None:
        raise TypeError(
            f"label_order= applies only when labels is a dict ({{name: path}}, "
            f"as returned by mtb.labels_for); labels is a "
            f"{type(labels).__name__} - pass the files as a list in that order "
            f"instead")
    batch_from_files = batch is None and _is_label_path_list(labels) and len(labels) > 1
    group, only, slow, want_cls = _plan_metrics(
        metrics, has_batch=batch is not None or batch_from_files,
        batch_given=batch is not None)
    if want_cls:
        from .classify import _graph_writer, _not_embedding
        not_embedding = _not_embedding(output)     # a label file or a list
        graph = _graph_writer(output)              # a file only graph methods write

    # named in the error for a Series that lacks some cells
    cells = "adata.obs_names" if io._is_anndata(output) else None
    if isinstance(output, (str, Path)) and Path(output).suffix.lower() == ".h5ad":
        import anndata as ad      # read once; keeps obs_names and obs
        output = ad.read_h5ad(output)
    adata = output if io._is_anndata(output) else None
    table = None
    if isinstance(output, (str, Path)) and Path(output).suffix.lower() in {".csv", ".tsv"}:
        file = io._require_file(output, "output")
        emb, ids = io._read_csv_embedding(file)
        if ids is not None and emb.shape[0] != emb.shape[1] \
                and _n_values(labels, adata) == emb.shape[1]:
            ids = None           # dims x cells: the first column names the dimensions
        table = None if ids is None else _Table(file.name)
    else:
        ids = _cell_ids(output)
        emb = np.asarray(io.as_matrix(output, obsm=obsm))
    if _is_label_path_list(labels):
        ct, origin = _stack_label_files(labels, ids, table=table)
    else:
        ct, origin = _obs_or_vector(labels, adata, what="labels", ids=ids,
                                    table=table, cells=cells), None
    # many run outputs are dims x cells; the label count says which axis holds the cells
    if emb.ndim == 2 and emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        warnings.warn(
            f"output was (dims x cells) {emb.shape}; transposed to (cells x dims)")
        emb = emb.T
    if want_cls and not (graph or not_embedding):
        from .classify import _is_neighbour_index
        if emb.ndim == 2 and (emb.shape[0] == emb.shape[1] or _is_neighbour_index(emb)):
            graph = "output"                       # cells x cells, or a kNN index
    if clustering is None:
        cl = None
    elif isinstance(clustering, (str, Path)) \
            and Path(clustering).suffix.lower() not in io._LABEL_FILE_SUFFIXES \
            and not (adata is not None and isinstance(clustering, str)
                     and _obs_column_name(clustering, adata) is not None):
        if _column_without_obs(clustering, adata) \
                and Path(clustering).suffix.lower() not in {".h5", ".hdf5", ".hdf"}:
            # "leiden" next to an array; "clusters.h5" keeps the missing-file error
            raise FileNotFoundError(_no_column_error("clustering", clustering))
        cl = io.read_clustering(clustering)        # the benchmark's h5 layout
    else:
        cl = _obs_or_vector(clustering, adata, what="clustering", ids=ids, table=table,
                            cells=cells)
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ids, table=table,
                            cells=cells)
    elif batch_from_files and group in {"batch", "all"}:
        # the label file of each cell (1, 2, ...), as run_all
        ba = origin
    else:
        # the clustering metrics need a batch key but do not use it
        ba = np.zeros(emb.shape[0], dtype=int)
    cb = None             # the batches classification learns from and predicts
    if want_cls:
        cb = ba if batch is not None else \
            _origin_names(labels, origin) if batch_from_files else None
    if verbose and table is not None and table.moved:
        import sys
        moved = table.moved[0] + (f" and {len(table.moved) - 1} more"
                                  if len(table.moved) > 1 else "")
        print(f"evaluate: matched {moved} to the rows of {table.name} by barcode.",
              file=sys.stderr, flush=True)

    from . import scib as escib
    n_cells = emb.shape[0]
    for what, values, given in (("labels", ct, labels), ("clustering", cl, clustering),
                                ("batch", ba, batch)):
        if values is not None and len(values) != n_cells:
            raise ValueError(_count_error(what, len(values), n_cells, given, category))
    # checked here: compute() casts labels to str, and "nan" would become a cell type
    miss, dropped = io.missing_mask(ct), None
    keep = ~miss
    if miss.any():
        if missing_labels == "error" or miss.all():
            shown = _as_written(labels, adata, ids, what="labels")
            raise ValueError(_no_label_error(
                ct if shown is None else shown, miss, labels))
        # ba is zeros when no batch metric is scored: count the file of origin then
        known = ba if batch is not None else origin if batch_from_files else None
        batches = None if known is None else (
            _n_batches(np.asarray(known)[keep]), _n_batches(known))
        emb, ct, ba = emb[keep], np.asarray(ct)[keep], np.asarray(ba)[keep]
        cl = None if cl is None else np.asarray(cl)[keep]
        cb = None if cb is None else np.asarray(cb)[keep]
        dropped = (int(keep.sum()), n_cells, batches)
    if batch is not None:
        # after the drop: only the cells that are scored need a batch
        no_batch = io.missing_mask(ba)
        if no_batch.any():
            shown = _as_written(batch, adata, ids, what="batch")
            raise ValueError(_no_batch_error(
                ba if shown is None else np.asarray(shown)[keep], no_batch, batch))
    odd = io._labels_look_measured(ct)
    if odd:
        # evaluate <- its keyword wrapper <- the caller
        warnings.warn(_measured_warning(labels, adata, *odd), UserWarning, stacklevel=3)
    if group is None:           # classification only: no scIB metric
        out = pd.DataFrame(columns=["Value"], dtype=float)
        out.attrs = {"leiden_flavor": None, "clustering": None}
    else:
        out = escib.compute(emb, ct, cl, ba, group=group, slow_metrics=slow, only=only,
                            verbose=None if verbose else False)
    if want_cls:
        rows = _classification_scores(want_cls, emb, ct, cb, category=category,
                                      not_embedding=not_embedding, graph=graph,
                                      metrics=metrics)
        if rows:
            attrs = out.attrs
            out = pd.concat([out, pd.DataFrame({"Value": rows})]) if len(out) \
                else pd.DataFrame({"Value": rows})
            out.attrs = attrs
    from .. import __version__
    out.attrs = {k: out.attrs.get(k) for k in PROVENANCE_ATTRS[:-1]}
    out.attrs["multibench_version"] = __version__
    out.attrs["scib_version"] = _scib_version()
    if out.empty:
        # unreachable while every family code is one compute() scores; loud if not
        raise ValueError(
            f"evaluate(metrics={metrics!r}) selected no metric - nothing to return")
    if dropped:
        out.attrs.update(zip(DROP_ATTRS, dropped[:2]))
        # evaluate <- its keyword wrapper <- the caller
        warnings.warn(_dropped_warning(*dropped), UserWarning, stacklevel=3)
    return out
