"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import contextvars
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from .. import _compat, config
from ..config import _and_list
from ..data import catalog
from ..data.results import COLUMNS as _RESULT_COLUMNS

#: the long table's columns; the stored tables' constant ``clustering`` is not written
LONG_COLUMNS = [c for c in _RESULT_COLUMNS if c != "clustering"]

#: the ``attrs`` keys evaluate() sets and to_long() joins into ``scored_with``
PROVENANCE_ATTRS = ("leiden_flavor", "clustering", "multibench_version")

#: ``attrs`` keys of a frame scored with ``missing_labels="drop"``: cells scored, cells given
DROP_ATTRS = ("cells_scored", "cells_total")

#: ``scored_with`` of rows whose frame carries no record of how it was scored
UNKNOWN_SCORING = "unknown"

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


def _long_table(value_df, *, method: str, dataset: str | None = None,
                category: str | None = None) -> pd.DataFrame:
    """The long table of a wide score frame (metrics as the index, one column
    ``Value``), its CSV, or a Series indexed by metric: one row per metric with
    the columns of ``mtb.load_results`` plus ``scored_with``. ``dataset=None``
    writes ``"all"``, ``category=None`` leaves the column empty."""
    if category is not None:
        config.category_folder(category)        # "verticall" is refused here
    if isinstance(value_df, (str, Path)):
        # evaluate()'s frame saved with to_csv: the metrics are its first column
        value_df = pd.read_csv(io._require_file(value_df, "metric CSV"), index_col=0)
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
    out["metric"] = out["metric"].map(catalog._canonical_metric)
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
    out["category"] = category
    out["source"] = "user"
    stamp = _scored_with(getattr(value_df, "attrs", None))
    out["scored_with"] = UNKNOWN_SCORING if stamp is None else stamp
    out = out[LONG_COLUMNS + ["scored_with"]].reset_index(drop=True)
    if stamp is not None:
        out.attrs = {k: value_df.attrs.get(k) for k in PROVENANCE_ATTRS}
        for k in ("scib_version",) + DROP_ATTRS:
            if k in value_df.attrs:
                out.attrs[k] = value_df.attrs[k]
    return out


@_compat.keywords("to_long", dropped=("clustering", "source"))
def to_long(value_df, *, method: str, dataset: str | None = None,
            category: str | None = None) -> pd.DataFrame:
    """Deprecated: ``mtb.evaluate(..., method=, dataset=)`` returns the long table."""
    _compat.deprecated("to_long", _compat.use("mtb.evaluate(..., method=, dataset=)"))
    cols = {str(c) for c in getattr(value_df, "columns", [])}
    if not {"metric", "value", "method"} <= cols:
        out = _long_table(value_df, method=method, dataset=dataset, category=category)
        # as in 0.3, until to_long goes in 0.5
        return out if category is not None else out.assign(category="user")
    # what evaluate returns in 0.4: only the named columns change
    out = value_df.copy()
    out["method"] = method
    if dataset is not None:
        out["dataset"] = dataset
    if category is not None:
        config.category_folder(category)
        out["category"] = category
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
    from ..workflow import CATEGORIES as valid  # lazy: workflow imports this module
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
    return (f"{what}={x!r} is not a file, and the embedding has no obs. Pass the values, as "
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
            from ..engine.ingest import _case_hint
            raise ValueError(
                f"{what}={x!r} is neither an obs column of the "
                f"{type(adata).__name__} (obs columns: {_obs_choices(adata)}) nor an "
                f"existing file." + _case_hint(x.removeprefix("obs:"), _obs_choices(adata)))
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
        if what == "labels" and first is None and _has_cell_ids(ids) and unique:
            # no ids: the dataset folder's matrix names the cells, else by position
            first = _folder_ids([x], [len(vals)], ids)
            if first is None:
                _position_warning([name], [x])
        return io.by_id_column(
            vals, first, ids if unique else None, what=what, name=name,
            target="the embedding", order="the rows of the embedding",
            no_ids=_no_ids(ids), drop=what == "labels")[0]
    if _carries_ids(x):
        other = table is not None and _shares_no_id(x.index, ids)
        if ids is not None and not other:
            name = f"the {what} {type(x).__name__}"
            if table is not None and x.index.is_unique and pd.Index(ids).is_unique \
                    and _same_cells(x.index, ids, what=what, name=name, table=table,
                                    fix=f"Pass {what} for the cells of {table.name}."):
                table.moved.append(name)
            return io._align_vector(x, ids, what=what, column=column, cells=cells)
        _compat.warn(
            f"The {what} {type(x).__name__} is matched by position, because "
            + (f"none of its ids is a cell of {table.name}. Check that it follows the "
               f"rows of {table.name}." if other else
               "the embedding has no cell ids. Check that it follows the embedding rows, "
               "or pass an AnnData whose obs_names are the barcodes."),
            # _obs_or_vector <- evaluate <- its keyword wrapper <- the caller
            UserWarning)
    if what == "batch" and io._numbers_rows(x, ids):
        _compat.warn(io._by_position(what),
                     UserWarning)
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


def _has_cell_ids(ids) -> bool:
    """Whether the embedding names its cells (barcodes, not row numbers)."""
    return ids is not None and io._looks_like_ids(pd.Series(list(ids), dtype=object))


#: the label files :func:`_position_warning` named in this session
_POSITION_NOTED: set = set()


def _by_position_note(names: list) -> str:
    """The warning for files without cell ids next to an embedding with them."""
    if len(names) == 1:
        return (f"{names[0]} has no cell ids, so its rows are matched to the embedding "
                f"by position.")
    return (f"{_and_list(names)} have no cell ids, so their rows are matched to the "
            f"embedding by position.")


def _folder_ids(files, counts, ids):
    """The cell ids of label files that have none, from the matrix next to each
    one with as many cells (``cty2.csv`` -> ``rna2.h5``, ...), when together they
    are the embedding's cells; else ``None``."""
    import h5py
    from ..workflow import _label_data_files      # workflow imports this module
    out = []
    for f, n in zip(files, counts):
        for m in (x for x in _label_data_files(Path(f)) if x.is_file()):
            try:
                with h5py.File(m, "r") as h:
                    bars = [b.decode() if isinstance(b, bytes) else str(b)
                            for b in h["matrix/barcodes"][:]]
            except (OSError, KeyError):
                continue
            if len(bars) == n:
                out += bars
                break
        else:
            return None
    keys = pd.Series(out, dtype=object)
    if not keys.is_unique or set(keys) != {str(i) for i in ids}:
        return None
    return keys


def _position_warning(names: list, paths) -> None:
    """:func:`_by_position_note`, once per set of files in a session."""
    key = tuple(os.path.abspath(str(p)) for p in paths)
    if key not in _POSITION_NOTED:
        _POSITION_NOTED.add(key)
        _compat.warn(_by_position_note(names), UserWarning)


def _no_ids(ids) -> str:
    return ("the embedding repeats cell ids" if ids is not None
            else "the embedding has no cell ids")


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
    order = "the rows of the embedding"
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
            what="labels", name=name, target="the embedding", order=order,
            no_ids=_no_ids(ids), n_files=len(names), drop=True)
        return vals[pos], origin[pos]
    if all(first is None for _, first in read) and _has_cell_ids(ids) and unique:
        # no ids: the dataset folder's matrices name the cells, else by position
        keys = _folder_ids(files, [len(v) for v, _ in read], ids)
        if keys is not None:
            pos, _ = io.by_id_column(np.arange(len(vals)), keys, ids, what="labels",
                                     name=_and_list(names), target="the embedding",
                                     order=order, no_ids=_no_ids(ids),
                                     n_files=len(names), drop=True)
            return vals[pos], origin[pos]
        _position_warning(names, files)
    for (v, first), name in zip(read, names):
        io.by_id_column(v, first, None, what="labels", name=name, target="the embedding",
                        order=order, no_ids=(
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
            f"The label keys {_and_list(keys)} {where}. Pass a list of paths in "
            f"cell order, or dataset=, category= and method=. The default order is "
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
_FLAGS = {"labels": "--labels", "clusters": "--clusters", "batch": "--batch"}
#: what one value of each argument is called in the command-line messages
_UNITS = {"labels": "labels", "clusters": "cluster ids", "batch": "batch ids"}
#: the dataset whose label files ``multibench evaluate --dataset`` read, so the
#: errors name them, not a --labels never typed
_CLI_LABELS_FROM = contextvars.ContextVar("cli_labels_from", default=None)
#: the keys of a labels dict, in the order of its files (label_order's spelling)
_LABEL_KEYS = contextvars.ContextVar("label_keys", default=None)


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
    keep = config.hint("Score the other cells only, as adata[adata.obs[col].notna()].",
                       "Score the other cells only: leave the unlabelled ones out of "
                       "the embedding and the label file.")
    return (f"{head}: {n:,} of the {n_cells:,} cells have no label{example}. {keep}")


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


def _files_that_fit(files: list, n_cells: int) -> list:
    """The first label files whose cells add up to ``n_cells`` (a method that
    read batches 1-2 of 3), else ``[]``."""
    total, out = 0, []
    for f in files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                total += len(io.read_labels(f))
        except Exception:  # noqa: BLE001 - only a hint: no fit
            return []
        out.append(f)
        if total == n_cells:
            return out
        if total > n_cells:
            return []
    return []


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
               f"{n_cells:,} cells in --embedding. ")
        if side and category == "diagonal":
            cli += (f'{dataset} labels the {side[0]} cells only, see the recipe '
                    f'"No labels" in the docs.')
        else:
            cli += f"Check that --embedding holds the embedding of --method on {dataset}."
        return config.hint(escib.count_error(what, n, n_cells), cli)
    msg = config.hint(
        escib.count_error(what, n, n_cells),
        f"{_FLAGS[what]} gave {n:,} {_UNITS[what]}{names} for {n_cells:,} cells "
        f"in --embedding.")
    if what == "labels" and n > n_cells and len(files) > 1:
        fit = _files_that_fit(files, n_cells)
        names = _LABEL_KEYS.get()
        if fit:
            keys = (names[:len(fit)] if names and len(names) == len(files)
                    else [f.stem for f in fit])
            return (f"The embedding has {n_cells:,} cells = {'+'.join(keys)}. "
                    + config.hint(f"Pass the label files {_and_list(f.name for f in fit)} "
                                  f"only.",
                                  "Pass --labels for " + " and ".join(
                                      f.name for f in fit) + " only."))
    if what != "labels" or n >= n_cells:
        return msg
    batches = _batch_files(files[0], n_cells) if len(files) == 1 else []
    if side and not batches:
        return (f'{msg} {files[0].name} labels the {side[0]} cells only, see the recipe '
                f'"No labels" in the docs.')
    order = f" ({', '.join(f.name for f in batches)})" if batches else ""
    return msg + " " + config.hint(
        "For a folder with one label file per batch, pass dataset=, category= "
        "and method=.",
        f"For several label files{order}, repeat --labels in the row order of "
        f"--embedding, or pass --dataset, --category and --method.")


def _plan_metrics(metrics, *, has_batch: bool):
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
    return group, only, slow, want_cls


def _unused_batch_warning(metrics) -> None:
    """The warning for a batch that no selected metric uses."""
    _compat.warn(
        config.hint("batch= changes nothing here, because ",
                    "--batch changes nothing here, because ")
        + f"{_metrics_spelling(metrics)} has no batch metric. Add "
        + _and_list([c for c in _metric_families()[1] if c != "kBET"], "or")
        + config.hint(", or pass metrics='all'.",
                      " to --metrics, or pass --metrics all."),
        UserWarning)


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
                _compat.warn(w.message, w.category)
            one_type = [str(r) for r in table.index[:-1] if np.isnan(table.loc[r, "Spec"])]
            if "Spec" in codes and one_type:
                # specificity needs cells of a second true cell type
                _compat.warn(
                    f"Spec is NaN: when batch {_and_list(one_type, 'or')} is the reference, "
                    f"the cells it predicts are all of one cell type. Compare methods on "
                    f"the other metrics.", UserWarning)
        else:
            why = "none scored"
    if why is not None:
        if how == "explicit":
            raise ValueError(_classification_error(metrics, why, what))
        # evaluate <- its keyword wrapper <- the caller
        _compat.warn(_classification_warning(why, what),
                     UserWarning)
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


_METRICS_KEYWORD = ("0.4", _compat.use("metrics="))


@_compat.keywords("evaluate", private="_evaluate", renamed={"output": "embedding"},
                  kept={"clustering": _compat.note(
                            "for your own clusters use "
                            "sklearn.metrics.adjusted_rand_score(labels, clusters)"),
                        "label_order": _compat.use("evaluate(..., dataset=, method=)"),
                        "missing_labels": _compat.note(
                            "keep the cells that have a label, as adata[adata.obs[col].notna()]")},
                  removed={"task": _METRICS_KEYWORD, "family": _METRICS_KEYWORD,
                           "only": _METRICS_KEYWORD,
                           "slow_metrics": ("0.3.0", _compat.note(
                               "pass metrics=[...] without cLISI and iLISI")),
                           "column": ("0.3.0", _compat.note(
                               "pass the column itself as labels= or batch=")),
                           "metric_set": ("0.3.0", _compat.note(
                               "only the scIB metric set exists, so drop it"))})
def evaluate(embedding, labels=None, *, dataset: str | None = None,
             method: str = "embedding", category: str | None = None, batch=None,
             metrics=None, obsm: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Score an embedding with the benchmark's metrics.

    The protocol is the article's: Leiden clusters searched to the number of
    cell types for ARI, NMI, ASW, iASW and iF1, and to the number of batches
    for ARI_batch and NMI_batch.

    Parameters
    ----------
    embedding
        Cells x dims: an array, DataFrame, AnnData or MuData (``.obsm[obsm]``),
        or a file: ``.h5`` (the benchmark's ``embedding.h5``), ``.h5ad``,
        ``.npy``, ``.csv`` / ``.tsv``.
    labels
        One cell type per cell: CSV path(s), a 1-D array-like, or an ``.obs``
        column name (``"celltype"``, ``"rna:celltype"`` for a MuData
        modality). ``None`` = the label files of ``dataset``.
    dataset : str | None
        The dataset folder, a name or a path; its name fills the ``dataset``
        column. With ``labels=None`` its label files are read, in the cell
        order of ``method`` when that is a package method.
    method : str
        Fills the ``method`` column: a package method id or your own name.
    category : str | None
        ``vertical``, ``diagonal``, ``mosaic`` or ``cross``; fills the
        ``category`` column. With two or more batches, all but ``vertical``
        add the classification metrics.
    batch
        One batch per cell, in the forms ``labels`` takes; needed for the
        batch metrics unless ``labels`` lists two or more files.
    metrics : None, str or list of str
        ``None`` = every metric the inputs allow except kBET; a family
        (``"clustering"``, ``"batch"``, ``"classification"``, ``"all"``), a
        code (``"ARI"``) or a list of codes and families.
    obsm : str | None
        ``.obsm`` key of an AnnData, MuData or ``.h5ad`` embedding; ``'X'``
        means ``.X``. ``None`` = ``X_emb``.
    verbose : bool
        Print a line when the Leiden search starts on more than 2,000 cells
        or labels are matched by barcode.

    Returns
    -------
    pandas.DataFrame
        The long table: ``metric, value, method, dataset, category, source,
        scored_with``, one row per metric. ``mtb.plot.bubble`` draws it, next
        to ``mtb.load_results`` rows or alone.

    Raises
    ------
    ValueError
        Missing or misaligned labels or batches, an unknown category or
        metric, or batch or classification metrics without batch labels.
    FileNotFoundError
        An embedding, label file or dataset folder that does not exist.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.evaluate("out/Matilda_D11/embedding.h5", dataset="D11",
    ...              category="vertical", method="Matilda")
    >>> mtb.evaluate(adata, labels="celltype", batch="sample", obsm="X_pca",
    ...              method="PCA")
    """
    return _evaluate(embedding, labels, dataset=dataset, method=method,
                     category=category, batch=batch, metrics=metrics, obsm=obsm,
                     verbose=verbose)


def _evaluate(embedding, labels=None, *, dataset=None, method="embedding", category=None,
              batch=None, metrics=None, obsm=None, verbose=True, clustering=None,
              label_order=None, missing_labels="error", data_path=None) -> pd.DataFrame:
    """:func:`evaluate` with the deprecated ``clustering=``, ``label_order=`` and
    ``missing_labels=``; ``data_path`` is the folder that holds ``dataset``."""
    if not isinstance(missing_labels, str) or missing_labels not in ("error", "drop"):
        raise ValueError(
            f'missing_labels must be "error" or "drop", got {missing_labels!r}.')
    if not isinstance(method, str) or not method.strip():
        raise TypeError(f"method= takes a name, such as method='Matilda'; got {method!r}.")
    name = None
    if dataset is not None:
        from ..engine import resolve
        name, data_path = resolve.split_dataset(dataset, data_path)
        if labels is None:
            labels = _dataset_labels(name, category, method, data_path)
    elif labels is None:
        raise ValueError("evaluate needs labels=, one cell type per cell, or dataset= "
                         "to read the label files of a dataset folder.")
    wide = _scores(embedding, labels, category=category, batch=batch, metrics=metrics,
                   clusters=clustering, obsm=obsm, label_order=label_order,
                   verbose=verbose, missing_labels=missing_labels)
    return _long_table(wide, method=method, dataset="user" if name is None else name,
                       category=category)


def _dataset_labels(dataset, category, method, data_path):
    """The label files of ``dataset``: in ``method``'s cell order when it names a
    package method (``registry.method_of``), which needs ``category``; else the
    folder's, with one did-you-mean warning when ``method`` is close to a
    package method that puts the cells in another order."""
    from ..engine import registry, resolve
    own = registry.method_of(method)
    if own is not None and category is None:
        raise ValueError(config.hint(
            f"evaluate(dataset={dataset!r}, method={method!r}) needs category= to read "
            f"{own}'s cell order.",
            f"--method {method!r} needs --category to read {own}'s cell order."))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=resolve.DEFAULT_ORDER_WARNING)
        try:
            labels = resolve.labels_for(dataset, category, own, data_path=data_path)
        except KeyError as e:
            if own is None or not e.args:
                raise
            raise KeyError(str(e.args[0]) + config.hint(
                " For your own embedding, pass the label files as labels=.",
                " For your own embedding, pass the label files with --labels.")) from None
    if not labels:
        raise ValueError(f"{dataset} has no label file. "
                         + config.hint("Pass labels=.", "Pass --labels."))
    near = registry.closest_method(method) if own is None and category else None
    if near:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                theirs = list(resolve.labels_for(dataset, category, near,
                                                 data_path=data_path, check=False))
            except Exception:  # noqa: BLE001 - no variant here: nothing to suggest
                theirs = list(labels)
        if theirs != list(labels):
            _compat.warn(config.hint(f"Did you mean method={near!r}? ", f"Did you mean "
                                     f"--method {near}? ")
                         + f"{near} puts the cells of {dataset} in another order; "
                         f"{method!r} reads the folder's order.", UserWarning)
    return labels


def _scores(
    embedding,
    labels=None,
    *,
    category: str | None = None,
    batch=None,
    metrics=None,
    clusters=None,
    obsm: str | None = None,
    label_order=None,
    verbose: bool = True,
    missing_labels: str = "error",
) -> pd.DataFrame:
    """The scores of :func:`evaluate` as a wide frame: one row per metric, one
    column ``Value``; ``attrs`` records how they were scored. ``clusters``
    replaces the Leiden clusters of ARI, NMI, ASW, iASW and iF1."""
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
    _LABEL_KEYS.set(None)
    if isinstance(labels, dict):
        _LABEL_KEYS.set([str(k) for k in (labels if label_order is None else label_order)])
        labels = _labels_from_dict(labels, label_order)
    elif label_order is not None:
        raise TypeError(
            f"label_order= applies only when labels is a dict ({{name: path}}, "
            f"as returned by mtb.labels_for); labels is a "
            f"{type(labels).__name__} - pass the files as a list in that order "
            f"instead")
    batch_from_files = batch is None and _is_label_path_list(labels) and len(labels) > 1
    group, only, slow, want_cls = _plan_metrics(
        metrics, has_batch=batch is not None or batch_from_files)
    if want_cls:
        from .classify import _graph_writer, _not_embedding
        not_embedding = _not_embedding(embedding)  # a label file or a list
        graph = _graph_writer(embedding)           # a file only graph methods write

    # named in the error for a Series that lacks some cells
    cells = "adata.obs_names" if io._is_anndata(embedding) else None
    if isinstance(embedding, (str, Path)) and Path(embedding).suffix.lower() == ".h5ad":
        import anndata as ad      # read once; keeps obs_names and obs
        embedding = ad.read_h5ad(embedding)
    adata = embedding if io._is_anndata(embedding) else None
    table = None
    if isinstance(embedding, (str, Path)) and Path(embedding).suffix.lower() in {".csv", ".tsv"}:
        file = io._require_file(embedding, "embedding")
        emb, ids = io._read_csv_embedding(file)
        if ids is not None and emb.shape[0] != emb.shape[1] \
                and _n_values(labels, adata) == emb.shape[1]:
            ids = None           # dims x cells: the first column names the dimensions
        table = None if ids is None else _Table(file.name)
    else:
        ids = _cell_ids(embedding)
        emb = np.asarray(io.as_matrix(embedding, obsm=obsm))
    if _is_label_path_list(labels):
        ct, origin = _stack_label_files(labels, ids, table=table)
    else:
        ct, origin = _obs_or_vector(labels, adata, what="labels", ids=ids,
                                    table=table, cells=cells), None
    # many run outputs are dims x cells; the label count says which axis holds the cells
    if emb.ndim == 2 and emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        _compat.warn(
            f"embedding was (dims x cells) {emb.shape}; transposed to (cells x dims)")
        emb = emb.T
    if want_cls and not (graph or not_embedding):
        from .classify import _is_neighbour_index
        if emb.ndim == 2 and (emb.shape[0] == emb.shape[1] or _is_neighbour_index(emb)):
            graph = "output"                       # cells x cells, or a kNN index
    if clusters is None:
        cl = None
    elif isinstance(clusters, (str, Path)) \
            and Path(clusters).suffix.lower() not in io._LABEL_FILE_SUFFIXES \
            and not (adata is not None and isinstance(clusters, str)
                     and _obs_column_name(clusters, adata) is not None):
        if _column_without_obs(clusters, adata) \
                and Path(clusters).suffix.lower() not in {".h5", ".hdf5", ".hdf"}:
            # "leiden" next to an array; "clusters.h5" keeps the missing-file error
            raise FileNotFoundError(_no_column_error("clusters", clusters))
        cl = io.read_clustering(clusters)          # the benchmark's h5 layout
    else:
        cl = _obs_or_vector(clusters, adata, what="clusters", ids=ids, table=table,
                            cells=cells)
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ids, table=table,
                            cells=cells)
        if group == "clustering" and not want_cls:
            # after the batch is read: a wrong column gets its error alone
            _unused_batch_warning(metrics)
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
    for what, values, given in (("labels", ct, labels), ("clusters", cl, clusters),
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
    if group in ("batch", "all") and len(set(np.asarray(ba).astype(str).tolist())) < 2:
        # batch metrics on one batch: NaN, or a GC of 1.0
        if metrics is not None:
            raise ValueError(_classification_error(metrics, "one batch"))
        group = "clustering"
    odd = io._labels_look_measured(ct)
    if odd:
        # evaluate <- its keyword wrapper <- the caller
        _compat.warn(_measured_warning(labels, adata, *odd), UserWarning)
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
        _compat.warn(_dropped_warning(*dropped), UserWarning)
    return out
