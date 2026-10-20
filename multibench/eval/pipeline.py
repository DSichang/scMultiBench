"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import contextvars
import warnings
from pathlib import Path
from typing import NamedTuple

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
    """``x`` can only be an obs column name, and the output has no obs."""
    return adata is None and io._names_no_file(x)


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


def _file_fix(plural: bool = False) -> str:
    return "Write them for the same cells." if plural else "Write both for the same cells."


def _file_carries_ids(first, ids) -> bool:
    """The first column of a per-cell file holds cell ids: text that is unique
    or names at least one cell. Repeated text naming no cell is a batch column."""
    if first is None or not io._looks_like_ids(first):
        return False
    return bool(first.is_unique or pd.Index(first.astype(str)).isin(ids).any())


def _obs_or_vector(x, adata, *, what, column=None, ids=None, as_written=False,
                   table=None, cells=None, read=(), others=None):
    """One label-like argument as a 1-D array in the output's cell order: an obs
    column, a Series aligned by id, a file aligned by its id column, else by position.

    ``as_written`` reads files for a message naming the missing words;
    ``table`` (a ``.csv`` embedding's barcodes) collects the arguments put in
    its row order; ``cells`` words the error of ``io._align_vector``; ``read``
    and ``others`` name the files whose barcodes name the cells (:func:`_folder_ids`).
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
                f"{what}={x!r}" + " is neither an obs "
                f"column of the {type(adata).__name__} (obs columns: {_obs_choices(adata)}) "
                f"nor an existing file." + _case_hint(x.removeprefix("obs:"),
                                                      _obs_choices(adata)))
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
        if what == "labels" and first is None and _has_cell_ids(ids):
            # no ids: the dataset folder's matrix names the cells
            first = _folder_ids([x], [len(vals)], ids, read, others)
        if _has_cell_ids(ids) and not _file_carries_ids(first, ids):
            by_position = "" if len(vals) != len(ids) else ", or " + f"pass {what}=pd.read_csv(...).iloc[:, 0].to_numpy()" + " to match by position"
            raise ValueError(f"{name} has no cell ids, and the embedding has: put the barcodes "
                             f"in its first column{by_position}.")
        return io.by_id_column(
            vals, first, ids if unique else None, what=what, name=name,
            target="the embedding", order="the rows of the embedding",
            no_ids=_no_ids(ids), drop=what == "labels")[0]
    if _carries_ids(x):
        if ids is not None:
            name = f"the {what} {type(x).__name__}"
            if table is not None and x.index.is_unique and pd.Index(ids).is_unique \
                    and _same_cells(x.index, ids, what=what, name=name, table=table,
                                    fix=f"Pass {what} for the cells of {table.name}."):
                table.moved.append(name)
            return io._align_vector(x, ids, what=what, column=column, cells=cells)
        _compat.warn(
            f"The {what} {type(x).__name__} is matched by position, because the embedding "
            f"has no cell ids. Check that it follows the embedding rows, or pass an AnnData "
            f"whose obs_names are the barcodes.",
            # _obs_or_vector <- evaluate <- its keyword wrapper <- the caller
            UserWarning)
    if io._numbers_rows(x, ids) and _has_cell_ids(ids):
        raise ValueError(f"{what} is a Series without cell ids, and the embedding has: index "
                         f"it by the barcodes, or pass {what}.to_numpy() to match by position.")
    return io.as_vector(x, what=what, column=column, as_written=as_written)


def _pick(what: str):
    """The ``pick`` of ``io.read_labels_ids``: how to choose one column of a
    per-cell file."""
    def fix(example: str) -> str:
        py = f"Pass one column, for example {example}"
        if what == "labels":
            return f"{py}, or name the label column x."
        return f"{py}."
    return fix


def _has_cell_ids(ids) -> bool:
    """Whether the embedding names its cells (barcodes, not row numbers)."""
    return ids is not None and io._looks_like_ids(pd.Series(list(ids), dtype=object))


def _file_barcodes(files, counts=None, read=()):
    """The cell ids of label files that have none, from the matrix next to each
    one with as many cells (``cty2.csv`` -> ``rna2.h5``, ...; the files a method
    read, ``read``, first); ``None`` when one has no such matrix. Where barcodes
    repeat, each is named by its label file, ``cty2:<barcode>``, as
    ``BatchResult.embedding`` names it."""
    raw = _raw_barcodes(files, counts, read)
    if raw is None:
        return None
    out, stems = raw
    keys = pd.Series(out, dtype=object)
    if not keys.is_unique:
        keys = pd.Series([f"{s}:{b}" for s, b in zip(stems, out)], dtype=object)
    return keys if keys.is_unique else None


def _raw_barcodes(files, counts=None, read=()):
    """``(barcodes, label file stem of each)`` of :func:`_file_barcodes`, the
    barcodes as the matrices write them; ``None`` when a file has no matrix."""
    import h5py
    from ..workflow import _label_data_files      # workflow imports this module
    out, stems = [], []
    for i, f in enumerate(files):
        n = counts[i] if counts is not None else len(io.read_labels(f))
        for m in (x for x in _label_data_files(Path(f), read) if x.is_file()):
            try:
                with h5py.File(m, "r") as h:
                    bars = [b.decode() if isinstance(b, bytes) else str(b)
                            for b in h["matrix/barcodes"][:]]
            except (OSError, KeyError):
                continue
            if len(bars) == n:
                out += bars
                stems += [Path(f).stem] * n
                break
        else:
            return None
    return out, stems


def _folder_ids(files, counts, ids, read=(), others=None):
    """:func:`_file_barcodes` of ``files`` when the embedding's cells ``ids`` are
    distinct cells among them (all of them, or a subset); ``None`` when the
    folder does not name its cells. The files a method read (``read``) name the
    cells first, as ``BatchResult.embedding`` names them, then the folder's
    default files. Any other ids raise: by position, the labels would be a
    guess. ``others()``: ``{read files: [package methods]}`` of the dataset,
    to name the method whose output names the cells so."""
    ids = [str(i) for i in ids]
    reads = list(dict.fromkeys((frozenset(read), frozenset())))
    found = [k for k in (_file_barcodes(files, counts, r) for r in reads) if k is not None]
    unique = len(set(ids)) == len(ids)
    for keys in found:
        if unique and set(ids) <= set(keys):
            return keys
    if not found:
        return None
    raise ValueError(_folder_ids_error(files, counts, ids, reads, found, others))


def _folder_ids_error(files, counts, ids, reads, found, others) -> str:
    """The error of :func:`_folder_ids`: what is wrong with ``ids``, then the one
    fix that works - the package method whose output names the cells so, an
    array when the cells are in the folder's order, else the folder's names."""
    names = _and_list([Path(f).name for f in files])
    have = set().union(*map(set, found))
    seen, dup = set(), None
    for i in ids:
        if i in seen:
            dup = i
            break
        seen.add(i)
    miss = [i for i in ids if i not in have]
    if dup is not None:
        head = f"The embedding repeats cell ids, such as {dup!r}"
    elif len(miss) == len(ids):
        head = f"None of the embedding's cell ids is a cell of {names}, such as {miss[0]!r}"
    else:
        head = (f"{len(miss):,} of the embedding's {len(ids):,} cell ids are not cells of "
                f"{names}, such as {miss[0]!r}")
    other = others() if others else {}
    if dup is None:
        made = [m for read, ms in other.items() for m in ms
                if (k := _file_barcodes(files, counts, read)) is not None
                and set(ids) <= set(k)]
        if made:
            m = made[0]
            named = (f"{m}'s output" if len(made) == 1 else
                     f"the output of {_and_list(made, 'or')}" if len(made) <= 3 else
                     f"the output of {', '.join(made[:3])} or {len(made) - 3} more")
            return (f"{head}. These cells are named as {named}: " + (f"pass method= the method that made it, such as method={m!r} "
                f"(or '{m} (this run)')."))
    for read in dict.fromkeys([*reads, *other]):
        raw = _raw_barcodes(files, counts, read)
        if raw is not None and len(raw[0]) == len(ids) and list(raw[0]) == ids:
            return (f"{head}. The cells are in the folder's order: " + ("pass the embedding as an array, as np.asarray(adata.obsm[key]) or "
                "adata.X, to match them by position."))
    return f"{head}. Name the embedding's cells as the folder does, such as {found[0].iloc[0]!r}."


def _no_ids(ids) -> str:
    return ("the embedding repeats cell ids" if ids is not None
            else "the embedding has no cell ids")


def _stack_label_files(files, ids, as_written=False, table=None, read=(), others=None):
    """Several label files, stacked in order, and each row's file (1, 2, ...).

    When every file has a first column of cell ids, the rows are aligned to
    ``ids`` by it; else by the barcodes of the folder's matrices (``read``, the
    files a method read, first: :func:`_folder_ids`, which refuses ids that are
    not distinct cells of the folder); otherwise they stay in file order.
    """
    rows = [io.read_labels_ids(f, what="labels", pick=_pick("labels"),
                               as_written=as_written) for f in files]
    vals = np.concatenate([np.asarray(v) for v, _ in rows])
    origin = np.concatenate([np.full(len(v), i + 1) for i, (v, _) in enumerate(rows)])
    names = [Path(f).name for f in files]
    unique = ids is not None and pd.Index(ids).is_unique
    order = "the rows of the embedding"
    if all(first is not None for _, first in rows):
        name = _and_list(names)
        keys = pd.concat([first for _, first in rows], ignore_index=True)
        if table is not None and unique and _file_carries_ids(keys, ids) \
                and _same_cells(keys, ids, what="labels", table=table, **(
                    dict(name=name, fix=_file_fix()) if len(names) == 1 else
                    dict(name="the label files", fix=_file_fix(plural=True)))):
            table.moved.extend(names)
        # a caller of evaluate is 5 frames up: by_id_column, this, evaluate, its wrapper
        pos, _ = io.by_id_column(np.arange(len(vals)), pd.concat(
            [first for _, first in rows], ignore_index=True), ids if unique else None,
            what="labels", name=name, target="the embedding", order=order,
            no_ids=_no_ids(ids), n_files=len(names), drop=True)
        return vals[pos], origin[pos]
    if all(first is None for _, first in rows) and _has_cell_ids(ids):
        # no ids: the dataset folder's matrices name the cells; without them, by position
        keys = _folder_ids(files, [len(v) for v, _ in rows], ids, read, others)
        if keys is not None:
            pos, _ = io.by_id_column(np.arange(len(vals)), keys, ids, what="labels",
                                     name=_and_list(names), target="the embedding",
                                     order=order, no_ids=_no_ids(ids),
                                     n_files=len(names), drop=True)
            return vals[pos], origin[pos]
    for (v, first), name in zip(rows, names):
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


#: the keys of a labels dict, in the order of its files (label_order's spelling)
_LABEL_KEYS = contextvars.ContextVar("label_keys", default=None)


class _Cells(NamedTuple):
    """The cells ``evaluate(dataset=)`` expects in the embedding."""
    dataset: str
    #: its label files, in the cell order of ``method``
    files: list
    #: the package method, or ``None`` for your own
    method: str | None
    #: labels= is these files (read from the dataset)
    from_files: bool
    #: a benchmark dataset: the embedding holds every cell
    whole: bool
    #: the file stems in ``method``'s order when the folder's order differs
    moved: tuple = ()
    #: the names of the files ``method`` read, whose barcodes name the cells
    read: frozenset = frozenset()
    #: the data root ``dataset`` is read from
    data_path: object = None


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


def _as_written(given, adata, ids, *, what, read=(), others=None):
    """An argument of evaluate() that names files, read again in cell order with
    each missing value as the word the file holds; ``None`` for values in memory.
    ``read``: as :func:`_stack_label_files`."""
    if not _label_files(given):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # the first read said them
        if what == "labels" and _is_label_path_list(given):
            return _stack_label_files(given, ids, as_written=True, read=read,
                                      others=others)[0]
        return _obs_or_vector(given, adata, what=what, ids=ids, as_written=True, read=read,
                              others=others)


def _measured_warning(given, adata, reason: str, distinct: int, cells: int) -> str:
    """The warning for labels that look like a measured number per cell, such as
    ``nCount_RNA``; names the column, Series or file(s) when known."""
    arg = "labels"
    files = [f.name for f in _label_files(given)]
    if adata is not None and isinstance(given, str) \
            and _obs_column_name(given, adata) is not None:
        src = f"labels={given!r}"
    elif files:
        src = f"{arg} ({files[0]}" + (f" and {len(files) - 1} more)" if len(files) > 1 else ")")
    elif isinstance(given, pd.Series) and given.name is not None:
        src = f"The labels Series {str(given.name)!r}"
    else:
        src = arg
    why = ("holds numbers that are not whole numbers" if reason == "fraction"
           else f"has {distinct:,} different labels for {cells:,} cells")
    return f"{src} {why}. Check that it holds the cell types."


def _no_label_error(shown, miss, given, dataset=None) -> str:
    """evaluate()'s error for labels missing where ``miss`` is true. ``shown``
    are the labels as written; ``dataset`` the dataset whose label files they are."""
    n, n_cells = int(miss.sum()), len(miss)
    head = "labels" + _named_files(given)
    example = io._missing_example(shown)
    if n == n_cells:
        return (f"{head}: none of the {n_cells:,} cells has a label{example}. Pass the "
                f"column or file that holds the cell types.")
    keep = "Score the other cells only, as adata[adata.obs[col].notna()]."
    return (f"{head}: {n:,} of the {n_cells:,} cells have no label{example}. {keep}")


def _no_batch_error(shown, miss, given) -> str:
    """evaluate()'s error for a batch missing where ``miss`` is true. ``shown``
    is the batch as written."""
    head = "batch" + _named_files(given)
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


def _n(n: int, one: str, many: str | None = None) -> str:
    """``1 cell``, ``2,500 cells``: ``n`` and the word in its number."""
    return f"{n:,} {one if n == 1 else many or one + 's'}"


def _count_error(what: str, n: int, n_cells: int, given, category=None,
                 cells: _Cells | None = None) -> str:
    """evaluate()'s error for ``n`` values of ``what`` against ``n_cells`` rows.

    Too few labels get the per-batch fix, or, for one side of a diagonal folder, the "No
    labels" recipe. ``cells``: the labels are those of a dataset folder.
    """
    files, names = _label_files(given), _named_files(given)
    side = _diagonal_side(files) if what == "labels" and n < n_cells else None
    if cells is not None and what == "labels":
        dataset = cells.dataset
        head = (f"The label files of {dataset}{names} hold {_n(n, 'cell')}; the "
                           f"embedding has {_n(n_cells, 'row')}.")
        if side and category == "diagonal":
            return (f'{head} {dataset} labels the {side[0]} cells only, see the recipe '
                    f'"No labels" in the docs.')
        if cells.method:
            return f"{head} Check that the embedding is the one of method= on {dataset}."
        return (f"{head} To score other cells, pass their labels and a dataset name of "
                f"your own, such as dataset='{dataset}_subset', and plot them apart from "
                f"the stored results.")
    fact = f"{what} has {_n(n, 'entry', 'entries')} for {_n(n_cells, 'cell')} in the embedding"
    if what == "labels" and n > n_cells and len(files) > 1:
        fit = _files_that_fit(files, n_cells)
        names = _LABEL_KEYS.get()
        if fit:
            keys = (names[:len(fit)] if names and len(names) == len(files)
                    else [f.stem for f in fit])
            return (f"The embedding has {_n(n_cells, 'cell')} = {'+'.join(keys)}. "
                    f"Pass the label files {_and_list(f.name for f in fit)} only.")
    if what != "labels" or n >= n_cells:
        one = {"labels": "label", "clusters": "cluster id", "batch": "batch"}[what]
        return f"{fact}: give one {one} per cell."
    batches = _batch_files(files[0], n_cells) if len(files) == 1 else []
    if side and not batches:
        return (f'{fact}. {files[0].name} labels the {side[0]} cells only, see the recipe '
                f'"No labels" in the docs.')
    order = f" ({', '.join(f.name for f in batches)})" if batches else ""
    return (f"{fact}. For a folder with one label file per batch, pass dataset=, category= "
        f"and method=." if files else
        f"labels has {_n(n, 'entry', 'entries')} for {_n(n_cells, 'embedding row')}: "
        f"give one label per row.")


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
                + ("Pass batch=<vector>, or labels as a list of two or more "
                              "files. Each file then counts as one batch."))
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
                    + "Pass batch=<vector>, or metrics='clustering'.")
            group, want_cls = "all", ("all", cls)
        only, slow = None, False
    elif sel.family == "batch":
        if not has_batch:
            raise ValueError(
                f"{_metrics_spelling(metrics)} needs batch labels for "
                f"{_and_list(family_bat)}. "
                + ("Pass batch=<vector>, or labels as a list of two or "
                              "more files."))
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
        "batch= changes nothing here, because "
        + f"{_metrics_spelling(metrics)} has no batch metric. Add "
        + _and_list([c for c in _metric_families()[1] if c != "kBET"], "or")
        + ", or pass metrics='all'.",
        UserWarning)


def _classification_error(metrics, why: str, what: str = "output") -> str:
    """evaluate()'s error when the classification codes cannot be computed;
    ``why`` is ``"no batch"``, ``"one batch"``, ``"graph"``, ``"not embedding"``
    or ``"none scored"``."""
    head = _metrics_spelling(metrics)
    if why == "graph":
        from .classify import _graph_text
        return _graph_text(what) + " " + "Pass metrics='clustering'."
    if why == "not embedding":
        return f"{head} needs an embedding; {what} is not one. Pass the method's embedding file."
    if why == "none scored":
        return (f"{head} found no batch to learn from: each needs two cell types that "
                f"another batch has. Check labels and batch.")
    fix = "Pass batch=<vector>, or labels as a list of two or more files."
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
                + "Pass metrics='clustering' to leave them out.")
    if why == "not embedding":
        return f"{head}: {what} is not an embedding. Pass the method's embedding file."
    if why == "none scored":
        return (f"{head}: no batch has two cell types that another batch has. "
                f"Check labels and batch.")
    return f"{head}: every cell is in one batch. " + "Pass batch= with two or more batches."


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
    """``metrics=['ARI', 'NMI']``."""
    if metrics is not None and not isinstance(metrics, str):
        codes = [str(m) for m in metrics]
        return f"metrics={codes!r}"
    return f"metrics={metrics!r}"


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
              label_order=None, missing_labels="error", data_path=None,
              from_dataset=False) -> pd.DataFrame:
    """:func:`evaluate` with the deprecated ``clustering=``, ``label_order=`` and
    ``missing_labels=``; ``data_path`` is the folder that holds ``dataset``;
    ``from_dataset``: ``labels`` are copies of the dataset's label files (the
    command line's ``--column``)."""
    if not isinstance(missing_labels, str) or missing_labels not in ("error", "drop"):
        raise ValueError(
            f'missing_labels must be "error" or "drop", got {missing_labels!r}.')
    if not isinstance(method, str) or not method.strip():
        raise TypeError(f"method= takes a name, such as method='Matilda'; got {method!r}.")
    name = cells = None
    if dataset is not None:
        from ..engine import resolve
        from ..workflow import known_category
        name, data_path = resolve.split_dataset(dataset, data_path)
        bench = known_category(name)             # a benchmark dataset has one
        category = bench if category is None else category
        from_files = labels is None or from_dataset
        if labels is None:
            labels = _dataset_labels(name, category, method, data_path)
        cells = _expected_cells(name, category, method, data_path,
                                labels if from_files else None, whole=bench is not None)
    elif labels is None:
        raise ValueError("evaluate needs labels=, one cell type per cell, or dataset= "
                         "to read the label files of a dataset folder.")
    wide = _scores(embedding, labels=labels, category=category, batch=batch,
                   metrics=metrics, clusters=clustering, obsm=obsm, label_order=label_order,
                   verbose=verbose, missing_labels=missing_labels, cells=cells)
    return _long_table(wide, method=method, dataset="user" if name is None else name,
                       category=category)


def _gives_ids(x, adata) -> bool:
    """Whether an argument of evaluate names its cells: a Series or DataFrame
    indexed by cell, or files whose first column holds cell ids. An obs
    column follows the embedding and names none."""
    if _carries_ids(x):
        return True
    if adata is not None and isinstance(x, str) and _obs_column_name(x, adata) is not None:
        return False
    files = _label_files(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the read that scores says them
        try:
            firsts = [io.read_labels_ids(f, what="labels", pick=_pick("labels"))[1]
                      for f in files]
        except (ValueError, KeyError, OSError):
            return False
    return bool(files) and all(f is not None and io._looks_like_ids(f) for f in firsts)


def _follows_cells(labels, adata, cells: _Cells) -> bool:
    """Whether ``labels`` can be put in the cells' order: an obs column of the
    embedding, ids, or the dataset's label files in that order."""
    if adata is not None and isinstance(labels, str) and (
            _obs_column_name(labels, adata) is not None
            or _mod_obs_column(labels, adata) is not None):
        return True
    files = _file_list(labels) if isinstance(labels, dict) else _label_files(labels)
    if files and [f.name for f in files] == [f.name for f in cells.files]:
        return True
    return _gives_ids(labels, adata)


def _cells_ids(cells: _Cells, shape: tuple):
    """The barcodes of ``cells`` in the method's order, when they are the cells
    of an embedding of ``shape`` (cells x dims, or dims x cells); else ``None``."""
    keys = _file_barcodes(cells.files, read=cells.read)
    return None if keys is None or len(keys) not in shape[:2] else pd.Index(keys)


def _file_list(labels) -> list:
    """The label files of ``labels``, a ``labels_for`` dict, a list or one path."""
    if isinstance(labels, (str, Path)):
        return [Path(labels)]
    return [Path(p) for p in (labels.values() if isinstance(labels, dict) else labels)]


def _expected_cells(dataset, category, method, data_path, read, *,
                    whole: bool) -> _Cells | None:
    """The cells evaluate(dataset=) expects, or ``None`` when nothing is known of
    them: your own labels for your own method on your own folder, or a dataset
    folder that is not here. ``read``: the label files evaluate read from the
    dataset, ``None`` when labels= was given."""
    from ..engine import registry, resolve
    own = registry.method_of(method)
    from_files = read is not None
    if from_files:
        files = _file_list(read)
    elif (not whole and own is None) or category is None:
        return None
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # said when labels= is left out
                files = _file_list(_dataset_labels(dataset, category, method, data_path))
        except FileNotFoundError:
            return None                              # not fetched: nothing to compare
    moved = ()
    if own is not None and not from_files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            folder = _file_list(resolve.labels_for(dataset, category, None,
                                                   data_path=data_path, check=False))
        if [f.name for f in folder] != [f.name for f in files]:
            moved = tuple(f.stem for f in files)
    from ..workflow import _method_files
    return _Cells(dataset, files, own, from_files, whole, moved,
                  frozenset(_method_files(dataset, category, own, None, data_path)),
                  data_path)


def _method_reads(cells: _Cells, category) -> dict:
    """``{names of the files read: [package methods]}`` of the methods of
    ``category`` on ``cells.dataset``; for an error that names the method whose
    output an embedding's cell names come from."""
    from ..engine.registry import list_methods
    from ..workflow import _method_files
    out: dict = {}
    for m in list_methods(category=category) if category else ():
        read = frozenset(_method_files(cells.dataset, category, m, None, cells.data_path))
        if read:
            out.setdefault(read, []).append(m)
    return out


def _dataset_labels(dataset, category, method, data_path):
    """The label files of ``dataset``: in ``method``'s cell order when it names a
    package method (``registry.method_of``), which needs ``category``; else the
    folder's, with one did-you-mean warning when ``method`` is close to a
    package method that puts the cells in another order."""
    from ..engine import registry, resolve
    from ..workflow import known_category
    own = registry.method_of(method)
    cats = registry.get(own).categories if own is not None else []
    if own is not None and category is None:
        raise ValueError((f"evaluate(dataset={dataset!r}, method={method!r}) needs category= to read "
            f"{own}'s cell order")
            + f": {_and_list(cats, 'or')}.")
    if own is not None and category not in cats:
        from ..engine.schema import no_category_message
        known = known_category(dataset)
        if known in cats:
            raise ValueError(f"{own} does not run on {category} data; {dataset} is {known}: "
                             + f"pass category={known!r}.")
        raise ValueError(no_category_message(own, cats, category) + " For your own embedding, pass the label files as labels=.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=resolve.DEFAULT_ORDER_WARNING)
        try:
            labels = resolve.labels_for(dataset, category, own, data_path=data_path)
        except KeyError as e:
            if own is None or not e.args:
                raise
            raise KeyError(str(e.args[0]) + " For your own embedding, pass the label files as labels=.") from None
    if not labels:
        raise ValueError(f"{dataset} has no label file. "
                         + "Pass labels=.")
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
            _compat.warn(f"Did you mean method={near!r}? "
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
    cells: _Cells | None = None,
) -> pd.DataFrame:
    """The scores of :func:`evaluate` as a wide frame: one row per metric, one
    column ``Value``; ``attrs`` records how they were scored. ``clusters``
    replaces the Leiden clusters of ARI, NMI, ASW, iASW and iF1; ``cells`` are
    the cells of ``evaluate(dataset=)``."""
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
    obs_ids = "adata.obs_names" if io._is_anndata(embedding) else None
    if isinstance(embedding, (str, Path)) and Path(embedding).suffix.lower() == ".h5ad":
        embedding = io._read_h5ad(embedding, "embedding")   # once; keeps obs_names, obs
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
    if cells is not None and cells.moved and not _follows_cells(labels, adata, cells):
        stems = ", ".join(cells.moved)
        raise ValueError(
            "labels= is matched to the cells by position, and "
            + f"{cells.method} puts the cells of {cells.dataset} in the order {stems}. "
            + f"Leave labels= out to read the label files in that order.")
    lab_ids, ba_ids = ids, ids
    # the files whose barcodes name the cells: the method's, else the folder's default
    read = cells.read if cells is not None else ()
    others = (lambda: _method_reads(cells, category)) if cells is not None else None
    if ids is None and table is None and cells is not None:
        # an embedding without ids holds the dataset's cells in the method's order
        lab_ids, ba_ids = (_cells_ids(cells, emb.shape) if _gives_ids(x, adata) else None
                           for x in (labels, batch))
    if _is_label_path_list(labels):
        ct, origin = _stack_label_files(labels, lab_ids, table=table, read=read,
                                        others=others)
    else:
        ct, origin = _obs_or_vector(labels, adata, what="labels", ids=lab_ids,
                                    table=table, cells=obs_ids, read=read,
                                    others=others), None
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
                            cells=obs_ids, read=read)
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ba_ids, table=table,
                            cells=obs_ids, read=read)
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
        # diagonal: the RNA and ATAC sides of the folder, whatever batch= is
        sides = category == "diagonal" and origin is not None and len(set(origin)) > 1
        cb = _origin_names(labels, origin) if batch_from_files or sides else \
            ba if batch is not None else None
    if verbose and table is not None and table.moved:
        import sys
        moved = table.moved[0] + (f" and {len(table.moved) - 1} more"
                                  if len(table.moved) > 1 else "")
        print(f"evaluate: matched {moved} to the rows of {table.name} by barcode.",
              file=sys.stderr, flush=True)

    from . import scib as escib
    n_cells = emb.shape[0]
    if cells is not None and cells.whole:
        n_held = sum(len(io.read_labels(f)) for f in cells.files)
        if n_held != n_cells:
            raise ValueError(_count_error("labels", n_held, n_cells, cells.files, category,
                                          cells))
    from_files = cells if cells is not None and cells.from_files else None
    for what, values, given in (("labels", ct, labels), ("clusters", cl, clusters),
                                ("batch", ba, batch)):
        if values is not None and len(values) != n_cells:
            raise ValueError(_count_error(what, len(values), n_cells, given, category,
                                          from_files))
    # checked here: compute() casts labels to str, and "nan" would become a cell type
    miss, dropped = io.missing_mask(ct), None
    keep = ~miss
    if miss.any():
        if missing_labels == "error" or miss.all():
            shown = _as_written(labels, adata, lab_ids, what="labels", read=read,
                                others=others)
            raise ValueError(_no_label_error(
                ct if shown is None else shown, miss, labels,
                from_files and from_files.dataset))
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
            shown = _as_written(batch, adata, ba_ids, what="batch", read=read,
                                others=others)
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
