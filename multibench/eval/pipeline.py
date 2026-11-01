"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import io
from .. import _text
from ..engine import registry
from .._text import _and_list
from .._data import catalog
from .._data.results import COLUMNS as _RESULT_COLUMNS

#: the long table's columns; the stored tables' constant ``clustering`` is not written
LONG_COLUMNS = list(_RESULT_COLUMNS)

#: the ``attrs`` keys evaluate() sets and the long table joins into ``scored_with``
PROVENANCE_ATTRS = ("leiden_flavor", "clustering", "multibench_version")

#: ``scored_with`` of rows whose frame carries no record of how it was scored
UNKNOWN_SCORING = "unknown"

def _scib_version() -> str | None:
    """The installed scib version, or ``None`` when it cannot be read."""
    try:
        from importlib.metadata import version
        return version("scib")
    except Exception:  # noqa: BLE001 - no metadata: ask the module
        try:
            from .scib import import_scib
            return getattr(import_scib(), "__version__", None)
        except Exception:  # noqa: BLE001
            return None


def _scored_with(attrs) -> str | None:
    """``"leidenalg/search/0.4.0"`` from evaluate()'s ``attrs``; ``None`` when
    the frame carries none."""
    if not attrs or attrs.get("multibench_version") is None:
        return None
    return "/".join(str(attrs.get(k) or "none") for k in PROVENANCE_ATTRS)


def _long_table(value_df, *, method: str, dataset: str | None = None,
                category: str | None = None) -> pd.DataFrame:
    """The long table of a wide score frame (metrics as the index, one column
    ``Value``): one row per metric with the columns of ``mtb.load_results``
    plus ``scored_with``. ``dataset=None`` writes ``"all"``, ``category=None``
    leaves the column empty."""
    if category is not None:
        registry.check_category(category)        # "verticall" is refused here
    out = value_df.rename(columns={"Value": "value"}).reset_index()
    out = out.rename(columns={out.columns[0]: "metric"})   # the index, whatever its name
    out["metric"] = out["metric"].map(catalog._canonical_metric)
    out = out.dropna(subset=["metric"])
    out["method"] = method
    out["dataset"] = "all" if dataset is None else dataset
    out["category"] = category
    out["source"] = "user"
    stamp = _scored_with(getattr(value_df, "attrs", None))
    out["scored_with"] = UNKNOWN_SCORING if stamp is None else stamp
    out = out[LONG_COLUMNS + ["scored_with"]].reset_index(drop=True)
    if stamp is not None:
        out.attrs = {k: value_df.attrs.get(k) for k in PROVENANCE_ATTRS}
        if "scib_version" in value_df.attrs:
            out.attrs["scib_version"] = value_df.attrs["scib_version"]
    return out


def _metric_families() -> tuple[list, list, list]:
    """The clustering, batch and classification codes evaluate() can produce."""
    # the plotting layer owns the metric families
    from ..plot.bar import BATCH_METRICS, CLASSIFICATION_METRICS, CLUSTERING_METRICS
    return list(CLUSTERING_METRICS), list(BATCH_METRICS), list(CLASSIFICATION_METRICS)


def _cell_ids(output):
    """Cell ids of ``output``: AnnData ``obs_names``, a DataFrame's index unless
    a ``RangeIndex``, the ``barcodes`` a run wrote into an ``.h5`` file;
    ``None`` for anything else."""
    if io._is_anndata(output):
        return pd.Index(output.obs_names)
    if isinstance(output, pd.DataFrame) and not isinstance(output.index, pd.RangeIndex):
        return output.index
    if isinstance(output, (str, Path)):
        return io.read_cell_names(output)
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


def _read_output(output, obsm, what: str):
    """``(emb, ids, adata)`` of an embedding argument: the matrix, its cell ids
    and the AnnData it is (else ``None``). ``what`` names a missing file."""
    if isinstance(output, (str, Path)) and Path(output).suffix.lower() == ".h5ad":
        output = io._read_h5ad(output, what)   # once; keeps obs_names, obs
    adata = output if io._is_anndata(output) else None
    if not (isinstance(output, (str, Path)) and Path(output).suffix.lower() in {".csv", ".tsv"}):
        ids = _cell_ids(output)
        return np.asarray(io.as_matrix(output, obsm=obsm, what=what)), ids, adata
    emb, ids = io._read_csv_embedding(io._require_file(output, what))
    return emb, ids, adata


def _file_carries_ids(first, ids) -> bool:
    """The first column of a per-cell file holds cell ids: text that is unique
    or names at least one cell. Repeated text naming no cell is a batch column."""
    if first is None or not io._looks_like_ids(first):
        return False
    return bool(first.is_unique or pd.Index(first.astype(str)).isin(ids).any())


def _obs_or_vector(x, adata, *, what, column=None, ids=None, read=(), others=None):
    """One label-like argument as a 1-D array in the output's cell order: an obs
    column, a Series aligned by id, a file aligned by its id column, else by position.

    ``read`` and ``others`` name the files whose barcodes name the cells
    (:func:`_folder_ids`).
    """
    if adata is not None and isinstance(x, str):
        col = _obs_column_name(x, adata)
        if col is not None:
            return np.asarray(pd.Series(adata.obs[col]).to_numpy())
        in_mod = _mod_obs_column(x, adata)            # already an array
        if in_mod is not None:
            return in_mod
        if not Path(x).is_file():
            raise ValueError(
                f"{what}={x!r} is neither an obs column of the {type(adata).__name__} "
                f"(obs columns: {_obs_choices(adata)}) nor an existing file.")
    if adata is None and io._names_no_file(x):    # an obs column name, and no obs
        raise FileNotFoundError(
            f"{what}={x!r} is not a file, and the embedding has no obs. Pass the values, as "
            f"{what}=adata.obs[{x.removeprefix('obs:')!r}].")
    if isinstance(x, (str, Path)) and column is None:
        vals, first = io.read_labels_ids(x, what=what)
        unique = ids is not None and pd.Index(ids).is_unique
        name = Path(x).name
        if what == "labels" and first is None and _has_cell_ids(ids):
            # no ids: the dataset folder's matrix names the cells
            first = _folder_ids([x], [len(vals)], ids, read, others)
        if _has_cell_ids(ids) and not _file_carries_ids(first, ids):
            by_position = "" if len(vals) != len(ids) else f", or pass {what}=pd.read_csv(...).iloc[:, 0].to_numpy() to match by position"
            raise ValueError(f"{name} has no cell ids, and the embedding has: put the barcodes "
                             f"in its first column{by_position}.")
        return io.by_id_column(
            vals, first, ids if unique else None, what=what, name=name,
            target="the embedding", order="the rows of the embedding",
            no_ids=_no_ids(ids))[0]
    if _carries_ids(x):
        if ids is not None:
            return io._align_vector(x, ids, what=what, column=column)
        _text.warn(
            f"The {what} {type(x).__name__} is matched by position: the embedding has no "
            f"cell ids.", UserWarning)
    if io._numbers_rows(x, ids) and _has_cell_ids(ids):
        raise ValueError(f"{what} is a Series without cell ids, and the embedding has: index "
                         f"it by the barcodes, or pass {what}.to_numpy() to match by position.")
    return io.as_vector(x, what=what, column=column)


def _has_cell_ids(ids) -> bool:
    """Whether the embedding names its cells (barcodes, not row numbers)."""
    return ids is not None and io._looks_like_ids(pd.Series(list(ids), dtype=object))


def _file_barcodes(files, counts=None, read=()):
    """The cell ids of label files that have none, from the matrix next to each
    one with as many cells (``cty2.csv`` -> ``rna2.h5``, ...; the files a method
    read, ``read``, first); ``None`` when one has no such matrix. Where barcodes
    repeat, each is named by its label file, ``cty2:<barcode>``, as
    ``RunResult.embedding`` names it."""
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
    from ..engine.resolve import _barcodes_of
    from ..workflow import _label_data_files      # workflow imports this module
    out, stems = [], []
    for i, f in enumerate(files):
        n = counts[i] if counts is not None else len(io.read_labels(f))
        bars = next((b for b in map(_barcodes_of, _label_data_files(Path(f), read))
                     if b is not None and len(b) == n), None)
        if bars is None:
            return None
        out += [str(b) for b in bars]
        stems += [Path(f).stem] * n
    return out, stems


def _folder_ids(files, counts, ids, read=(), others=None):
    """:func:`_file_barcodes` of ``files`` when the embedding's cells ``ids`` are
    distinct cells among them (all of them, or a subset); ``None`` when the
    folder does not name its cells. The files a method read (``read``) name the
    cells first, as ``RunResult.embedding`` names them, then the folder's
    default files. Any other ids raise: by position, the labels would be a
    guess. ``others()``: the file sets the dataset's package methods read,
    whose barcodes may name the cells (scJoint's ATAC cells by ``atac_gas.h5``)."""
    ids = [str(i) for i in ids]
    reads = list(dict.fromkeys((frozenset(read), frozenset())))
    found = [k for k in (_file_barcodes(files, counts, r) for r in reads) if k is not None]
    unique = len(set(ids)) == len(ids)
    for keys in found:
        if unique and set(ids) <= set(keys):
            return keys
    for r in (others() if others and unique else ()):
        keys = _file_barcodes(files, counts, r)
        if keys is not None and set(ids) <= set(keys):
            return keys
    if not found:
        return None
    raise ValueError(_folder_ids_error(files, counts, ids, reads, found, others))


def _folder_ids_error(files, counts, ids, reads, found, others) -> str:
    """The error of :func:`_folder_ids`: what is wrong with ``ids``, then the one
    fix that works - an array when the cells are in the folder's order, else the
    folder's names."""
    names = _and_list([Path(f).name for f in files])
    have = set().union(*map(set, found))
    repeats = pd.Index(ids)[pd.Index(ids).duplicated()]
    dup = repeats[0] if len(repeats) else None
    miss = [i for i in ids if i not in have]
    if dup is not None:
        head = f"The embedding repeats cell ids, such as {dup!r}"
    elif len(miss) == len(ids):
        head = f"The embedding's cell ids are not the cell names of {names}, such as {miss[0]!r}"
    else:
        head = (f"{len(miss):,} of the embedding's {len(ids):,} cell ids are not the cell "
                f"names of {names}, such as {miss[0]!r}")
    for read in dict.fromkeys([*reads, *(others() if others else ())]):
        raw = _raw_barcodes(files, counts, read)
        if raw is not None and len(raw[0]) == len(ids) and list(raw[0]) == ids:
            return (f"{head}. The cells are in the folder's order: pass the embedding as "
                    f"an array, as np.asarray(adata.obsm[key]) or adata.X, to match them "
                    f"by position.")
    return f"{head}. Name the embedding's cells as the folder does, such as {found[0].iloc[0]!r}."


def _no_ids(ids) -> str:
    return ("the embedding repeats cell ids" if ids is not None
            else "the embedding has no cell ids")


def _stack_label_files(files, ids, read=(), others=None):
    """Several label files, stacked in order, and each row's file (1, 2, ...).

    When every file has a first column of cell ids, the rows are aligned to
    ``ids`` by it; else by the barcodes of the folder's matrices (``read``, the
    files a method read, first: :func:`_folder_ids`, which refuses ids that are
    not distinct cells of the folder); otherwise they stay in file order.
    """
    rows = [io.read_labels_ids(f, what="labels") for f in files]
    vals = np.concatenate([np.asarray(v) for v, _ in rows])
    origin = np.concatenate([np.full(len(v), i + 1) for i, (v, _) in enumerate(rows)])
    names = [Path(f).name for f in files]
    unique = ids is not None and pd.Index(ids).is_unique
    order = "the rows of the embedding"
    if all(first is not None for _, first in rows):
        name = _and_list(names)
        keys = pd.concat([first for _, first in rows], ignore_index=True)
        pos, _ = io.by_id_column(np.arange(len(vals)), keys, ids if unique else None,
                                 what="labels", name=name, target="the embedding",
                                 order=order, no_ids=_no_ids(ids))
        return vals[pos], origin[pos]
    if all(first is None for _, first in rows) and _has_cell_ids(ids):
        # no ids: the dataset folder's matrices name the cells; without them, by position
        keys = _folder_ids(files, [len(v) for v, _ in rows], ids, read, others)
        if keys is not None:
            pos, _ = io.by_id_column(np.arange(len(vals)), keys, ids, what="labels",
                                     name=_and_list(names), target="the embedding",
                                     order=order, no_ids=_no_ids(ids))
            return vals[pos], origin[pos]
    for (v, first), name in zip(rows, names):
        io.by_id_column(v, first, None, what="labels", name=name, target="the embedding",
                        order=order, no_ids=(
                            "the other label files have none" if unique
                            else _no_ids(ids)))
    return vals, origin


def _labels_from_dict(d: dict) -> list:
    """The paths of a ``{name: path}`` label dict in cell order.

    A guess at any other order would score misordered labels without an error,
    so several entries need an unchanged ``labels_for`` dict or the default
    order (``_label_sort_key``).
    """
    if len(d) == 1:
        return [next(iter(d.values()))]
    from ..engine.resolve import LabelFiles, _label_sort_key
    keys = list(d)
    if (isinstance(d, LabelFiles) and d.in_stacking_order()) \
            or keys == sorted(keys, key=_label_sort_key):
        return [d[k] for k in keys]
    raise ValueError(
        f"The label keys {_and_list(keys)} are not in the default order: cty1, cty2, "
        f"... by number, and rna before adt before atac. Pass a list of paths in cell order.")


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
    if io._is_label_path_list(given):
        return [Path(p) for p in given]
    return []


def _named_files(given) -> str:
    """`` (cty1.csv, cty2.csv)``: the files an argument named; ``""`` for values
    in memory."""
    files = _label_files(given)
    return f" ({', '.join(f.name for f in files)})" if files else ""


def _measured_warning(distinct: int, cells: int) -> str:
    """The warning for labels that look like a measured number per cell, such as
    ``nCount_RNA``."""
    return (f"The labels have {distinct:,} different values for {cells:,} cells. Check "
            f"that they are the cell types.")


def _no_label_error(shown, miss, given) -> str:
    """evaluate()'s error for labels missing where ``miss`` is true. ``shown``
    are the labels."""
    n, n_cells = int(miss.sum()), len(miss)
    head = "labels" + _named_files(given)
    example = io._missing_example(shown)
    if n == n_cells:
        return (f"{head}: none of the {n_cells:,} cells has a label{example}. Pass the "
                f"column or file that holds the cell types.")
    return (f"{head}: {n:,} of the {n_cells:,} cells have no label{example}. Score the "
            f"other cells only, as adata[adata.obs[col].notna()].")


def _no_batch_error(shown, miss, given) -> str:
    """evaluate()'s error for a batch missing where ``miss`` is true. ``shown``
    is the batch."""
    head = "batch" + _named_files(given)
    return (f"{head}: {int(miss.sum()):,} of the {len(miss):,} cells have no batch"
            f"{io._missing_example(shown)}. Give every cell a batch.")


def _n_batches(ba) -> int:
    """How many batches a batch vector names; a missing value is not one."""
    ba = np.asarray(ba)
    return len(set(ba[~io.missing_mask(ba)].astype(str).tolist()))


def _n(n: int, one: str, many: str | None = None) -> str:
    """``1 cell``, ``2,500 cells``: ``n`` and the word in its number."""
    return f"{n:,} {one if n == 1 else many or one + 's'}"


def _count_error(what: str, n: int, n_cells: int, cells: _Cells | None = None) -> str:
    """evaluate()'s error for ``n`` values of ``what`` against ``n_cells`` rows;
    ``cells``: the labels are the label files of a dataset folder."""
    if cells is not None and what == "labels":
        fix = (f"Check that the embedding is the one of method= on {cells.dataset}."
               if cells.method else "For other cells, pass labels= and another name, "
               f"such as dataset='{cells.dataset}_subset'.")
        return (f"The label files of {cells.dataset} hold {_n(n, 'cell')}; the embedding "
                f"has {_n(n_cells, 'row')}. {fix}")
    one = {"labels": "label", "clusters": "cluster id", "batch": "batch"}[what]
    return (f"{what} has {_n(n, 'entry', 'entries')} for {_n(n_cells, 'cell')} in the "
            f"embedding: give one {one} per cell.")


#: the fix for batch metrics without a batch
_FILES_FIX = "Pass batch=<vector>, or labels as a list of two or more files."


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

    def needs_batch(codes, fix):
        return ValueError(f"{_metrics_spelling(metrics)} needs batch labels for "
                          f"{_and_list(codes)}. {fix}")

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
            raise needs_batch(want_bat, _FILES_FIX + " Each file then counts as one batch.")
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
                raise needs_batch(family_bat, "Pass batch=<vector>, or metrics='clustering'.")
            group, want_cls = "all", ("all", cls)
        only, slow = None, False
    elif sel.family == "batch":
        if not has_batch:
            raise needs_batch(family_bat, _FILES_FIX)
        group, only, slow = "batch", None, False
    elif sel.family == "classification":
        group, only, slow, want_cls = None, None, False, ("explicit", cls)
    else:
        group, only, slow = "clustering", None, False
    if want_cls and want_cls[0] == "explicit" and not has_batch:
        raise ValueError(_classification_error(metrics, "no batch"))
    return group, only, slow, want_cls


def _classification_reason(why: str, what: str = "output") -> str:
    """Why the classification codes cannot be computed, and the fix: ``why`` is
    ``"no batch"``, ``"one batch"``, ``"graph"`` or ``"none scored"``."""
    if why == "graph":
        from .classify import _graph_text
        return _graph_text(what) + " Pass metrics='clustering'."
    if why == "none scored":
        return "no batch has two cell types that another batch has. Check labels and batch."
    if why == "one batch":
        return f"every cell is in one batch. {_FILES_FIX}"
    return f"it needs two or more batches. {_FILES_FIX}"


def _classification_error(metrics, why: str, what: str = "output") -> str:
    """evaluate()'s error when the codes ``metrics=`` names cannot be computed."""
    reason = _classification_reason(why, what)
    return reason if why == "graph" else \
        f"{_metrics_spelling(metrics)} cannot be computed: {reason}"


def _classification_scores(request, emb, ct, cb, *, category, graph, metrics):
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
    why = "graph" if graph else "one batch" if n_batches < 2 else None
    what = graph or "output"
    mean = None
    if why is None:
        from .classify import classification_metrics, classify
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred = classify(emb, ct, cb, seed=0)
        table = classification_metrics(pred)
        if "mean" in table.index:
            mean = table.loc["mean"]
            for w in caught:      # a skipped reference batch: said once, as classify says it
                _text.warn(w.message, w.category)
            one_type = [str(r) for r in table.index[:-1] if np.isnan(table.loc[r, "Spec"])]
            if "Spec" in codes and one_type:
                # specificity needs cells of a second true cell type
                _text.warn(
                    f"Spec is NaN when batch {_and_list(one_type, 'or')} is the reference: "
                    f"the cells it predicts have one true cell type.", UserWarning)
        else:
            why = "none scored"
    if why is not None:
        if how == "explicit":
            raise ValueError(_classification_error(metrics, why, what))
        _text.warn("OCA, ACA, Spec, Sens and F1 are NaN: " + _classification_reason(why, what),
                   UserWarning)
        return {c: float("nan") for c in codes}
    return {c: float(mean[c]) for c in codes}


def _origin_names(labels, origin, names=None):
    """The batch of each cell when several label files give it: ``names[i]``
    for file ``i + 1``, by default the file's stem, else its number."""
    if names is None:
        stems = [Path(p).stem for p in labels]
        names = stems if len(set(stems)) == len(stems) else \
            [str(i + 1) for i in range(len(labels))]
    return np.asarray(names, dtype=object)[np.asarray(origin) - 1]


def _metrics_spelling(metrics) -> str:
    """``metrics=['ARI', 'NMI']``."""
    if metrics is not None and not isinstance(metrics, str):
        metrics = [str(m) for m in metrics]
    return f"metrics={metrics!r}"


def evaluate(embedding, labels=None, *, dataset: str | None = None,
             method: str = "embedding", category: str | None = None, batch=None,
             metrics=None, obsm: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Score an embedding with the benchmark's metrics.

    The protocol is the paper's: Leiden clusters searched to the number of
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
        column. With ``labels=None`` its label files are read and matched to
        the cells by the names ``run`` writes into ``embedding.h5``; for a
        file of a ``run`` folder without them, in the cell order of the
        method its run record names; else in the order of ``method`` when
        that is a package method.
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
        Print a line when the Leiden search starts on more than 2,000 cells.

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
    if isinstance(embedding, (str, Path)):
        embedding = Path(embedding).expanduser()
    name = cells = None
    if dataset is not None:
        from ..engine import resolve
        from ..workflow import known_category, own_batches_only, _run_record_for, _dataset_folder
        name, data_path = resolve.split_dataset(dataset, None)
        bench = known_category(name)             # a benchmark dataset has one
        if batch is not None:
            own_batches_only(name)
        # a file of a run folder: its record says whose cell order it has
        record = _run_record_for(embedding, name)
        order = record["method"] if record else method
        if category is None:
            category = record["category"] if record else bench
        from_files = labels is None
        # the label files the run read (UINMF on D52 reads two of three batches)
        used = [_dataset_folder(name, data_path) / f for f in record["labels_used"]] \
            if record and labels is None else []
        if used and all(f.is_file() for f in used):
            labels = used
        elif labels is None:
            labels = _dataset_labels(name, category, order, data_path)
        cells = _expected_cells(name, category, order, data_path,
                                labels if from_files else None, whole=bench is not None)
    elif labels is None:
        raise ValueError("evaluate needs labels=, one cell type per cell, or dataset= "
                         "to read the label files of a dataset folder.")
    wide = _scores(embedding, labels=labels, category=category, batch=batch,
                   metrics=metrics, obsm=obsm, verbose=verbose, cells=cells)
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
            firsts = [io.read_labels_ids(f, what="labels")[1]
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
    from ..engine import resolve
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
                                                   data_path=data_path))
        if [f.name for f in folder] != [f.name for f in files]:
            moved = tuple(f.stem for f in files)
    from ..workflow import _method_files
    return _Cells(dataset, files, own, from_files, whole, moved,
                  frozenset(_method_files(dataset, category, own, None, data_path)),
                  data_path)


def _method_reads(cells: _Cells, category) -> list:
    """The sets of file names the package methods of ``category`` read from
    ``cells.dataset``, whose barcodes may name an embedding's cells."""
    from ..engine.registry import list_methods
    from ..workflow import _method_files
    reads = (frozenset(_method_files(cells.dataset, category, m, None, cells.data_path))
             for m in (list_methods(category=category) if category else ()))
    return list(dict.fromkeys(r for r in reads if r))


def _dataset_labels(dataset, category, method, data_path):
    """The label files of ``dataset``: in ``method``'s cell order when it names a
    package method (``registry.method_of``), which needs ``category``; else the
    folder's, with one did-you-mean warning when ``method`` is close to a
    package method that puts the cells in another order."""
    from ..engine import resolve
    own = registry.method_of(method)
    cats = registry.get(own).categories if own is not None else []
    if own is not None and category is None:
        raise ValueError(f"evaluate(dataset={dataset!r}, method={method!r}) needs category= "
                         f"to read {own}'s cell order: {_and_list(cats, 'or')}.")
    if own is not None and category not in cats:
        from ..engine.schema import no_category_message
        raise ValueError(no_category_message(own, cats, category))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=resolve.DEFAULT_ORDER_WARNING)
        labels = resolve.labels_for(dataset, category, own, data_path=data_path)
    if not labels:
        raise ValueError(f"{dataset} has no label file. Pass labels=.")
    near = registry.closest_method(method) if own is None and category else None
    if near:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                theirs = list(resolve.labels_for(dataset, category, near,
                                                 data_path=data_path))
            except Exception:  # noqa: BLE001 - no variant here: nothing to suggest
                theirs = list(labels)
        if theirs != list(labels):
            _text.warn(f"Did you mean method={near!r}? {near} puts the cells of {dataset} "
                       f"in another order; {method!r} reads the folder's order.", UserWarning)
    return labels


def _check_finite(emb) -> None:
    """``ValueError`` for an embedding with NaN or infinite values, naming the rows."""
    try:
        bad = ~np.isfinite(np.asarray(emb, dtype=float)).all(axis=1)
    except (TypeError, ValueError):
        return                     # not numbers: the scoring says what it is
    if bad.any():
        rows = np.flatnonzero(bad)
        raise ValueError(
            f"{len(rows):,} of the {len(bad):,} rows of the embedding hold NaN or "
            f"infinite values, such as row {rows[0]}: drop those cells from the "
            f"embedding and the labels.")


def _cell_labels(labels, emb, adata, ids, read=(), others=None):
    """``(emb, labels, origin)``: the labels in the cell order of the embedding,
    the label file of each cell (``None`` unless several files give them), and
    the embedding as cells x dims (many run outputs are dims x cells; the label
    count says which axis holds the cells)."""
    if io._is_label_path_list(labels):
        ct, origin = _stack_label_files(labels, ids, read=read, others=others)
    else:
        ct, origin = _obs_or_vector(labels, adata, what="labels", ids=ids, read=read,
                                    others=others), None
    if emb.ndim == 2 and emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        emb = emb.T
    return emb, ct, origin


def _scores(
    embedding,
    labels=None,
    *,
    category: str | None = None,
    batch=None,
    metrics=None,
    clusters=None,
    obsm: str | None = None,
    verbose: bool = True,
    cells: _Cells | None = None,
) -> pd.DataFrame:
    """The scores of :func:`evaluate` as a wide frame: one row per metric, one
    column ``Value``; ``attrs`` records how they were scored. ``clusters``
    replaces the Leiden clusters of ARI, NMI, ASW, iASW and iF1; ``cells`` are
    the cells of ``evaluate(dataset=)``."""
    registry.check_category(category)
    # a dict becomes a list of paths first, for the file-of-origin batch rule
    if isinstance(labels, dict):
        labels = _labels_from_dict(labels)
    batch_from_files = batch is None and io._is_label_path_list(labels) and len(labels) > 1
    group, only, slow, want_cls = _plan_metrics(
        metrics, has_batch=batch is not None or batch_from_files)
    if want_cls:
        from .classify import _graph_writer
        graph = _graph_writer(embedding)           # a file only graph methods write
    emb, ids, adata = _read_output(embedding, obsm, "embedding")
    if cells is not None and cells.moved and not _follows_cells(labels, adata, cells):
        stems = ", ".join(cells.moved)
        raise ValueError(
            f"labels= is matched to the cells by position, and {cells.method} puts "
            f"the cells of {cells.dataset} in the order {stems}. Leave labels= out to "
            f"read the label files in that order.")
    lab_ids, ba_ids = ids, ids
    # the files whose barcodes name the cells: the method's, else the folder's default
    read = cells.read if cells is not None else ()
    others = (lambda: _method_reads(cells, category)) if cells is not None else None
    if ids is None and cells is not None:
        # an embedding without ids holds the dataset's cells in the method's order
        lab_ids, ba_ids = (_cells_ids(cells, emb.shape) if _gives_ids(x, adata) else None
                           for x in (labels, batch))
    emb, ct, origin = _cell_labels(labels, emb, adata, lab_ids, read, others)
    _check_finite(emb)
    if want_cls and not graph:
        from .classify import _is_neighbour_index
        if emb.ndim == 2 and (emb.shape[0] == emb.shape[1] or _is_neighbour_index(emb)):
            graph = "output"                       # cells x cells, or a kNN index
    # the clusters of run's label-order screening, an array in the embedding's rows
    cl = None if clusters is None else np.asarray(clusters)
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ba_ids, read=read)
    elif batch_from_files and group in {"batch", "all"}:
        # the label file of each cell (1, 2, ...), as run
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

    n_cells = emb.shape[0]
    if cells is not None and cells.whole:
        n_held = sum(len(io.read_labels(f)) for f in cells.files)
        if n_held != n_cells:
            raise ValueError(_count_error("labels", n_held, n_cells, cells))
    from_files = cells if cells is not None and cells.from_files else None
    for what, values in (("labels", ct), ("clusters", cl), ("batch", ba)):
        if values is not None and len(values) != n_cells:
            raise ValueError(_count_error(what, len(values), n_cells, from_files))
    # checked here: compute() casts labels to str, and "nan" would become a cell type
    miss = io.missing_mask(ct)
    if miss.any():
        raise ValueError(_no_label_error(ct, miss, labels))
    if batch is not None:
        no_batch = io.missing_mask(ba)
        if no_batch.any():
            raise ValueError(_no_batch_error(np.asarray(ba), no_batch, batch))
    if group in ("batch", "all") and len(set(np.asarray(ba).astype(str).tolist())) < 2:
        # batch metrics on one batch: NaN, or a GC of 1.0
        if metrics is not None:
            raise ValueError(_classification_error(metrics, "one batch"))
        group = "clustering"
    odd = io._labels_look_measured(ct)
    if odd:
        _text.warn(_measured_warning(*odd), UserWarning)
    if group is None:           # classification only: no scIB metric
        out = pd.DataFrame(columns=["Value"], dtype=float)
    else:
        from .scib import compute
        out = compute(emb, ct, cl, ba, group=group, slow_metrics=slow, only=only,
                      verbose=None if verbose else False)
    attrs = {k: out.attrs.get(k) for k in PROVENANCE_ATTRS[:-1]}
    if want_cls:
        rows = _classification_scores(want_cls, emb, ct, cb, category=category,
                                      graph=graph,
                                      metrics=metrics)
        if rows:
            out = pd.concat([out, pd.DataFrame({"Value": rows})]) if len(out) \
                else pd.DataFrame({"Value": rows})
    from .. import __version__
    out.attrs = {**attrs, "multibench_version": __version__, "scib_version": _scib_version()}
    if out.empty:
        # unreachable while every family code is one compute() scores; loud if not
        raise ValueError(
            f"evaluate(metrics={metrics!r}) selected no metric - nothing to return")
    return out
