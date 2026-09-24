"""Readers and coercers for evaluation inputs: embedding, labels, clustering.

Three groups:

* ``read_*`` - file readers for the benchmark's on-disk formats (``embedding.h5``
  with dataset ``data``; ``*cty*.csv`` label files; ``/obs/cluster_leiden`` in
  an h5).
* ``as_matrix`` / ``as_vector`` - coercers that accept whatever a user is
  likely to hold in memory (ndarray, DataFrame, Series, Categorical, list,
  AnnData, a path of any supported suffix, a list of label files) and return
  plain numpy arrays. :func:`multibench.evaluate` is built on these.
* ``align_vector`` - reorders an indexed Series/DataFrame to an output's cell
  ids (AnnData ``obs_names`` / DataFrame index), so a label Series is matched
  by barcode rather than by position whenever both sides carry ids.
  ``by_id_column`` does the same for a label CSV whose first column holds
  the cell ids.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

#: Suffixes that mark a path as a CSV-like label file. evaluate() reads a
#: clustering= path (str or Path) with any other suffix as an h5
#: (read_clustering), and as_vector() takes a list of strings without these
#: suffixes as label values.
_LABEL_FILE_SUFFIXES = {".csv", ".tsv", ".txt"}


def _require_file(path: Path | str, what: str = "file") -> Path:
    """``Path(path)`` when it is an existing file; else ``FileNotFoundError``
    naming the path and the working directory (a relative path that resolves
    from the repository but not from a notebook's cwd is the usual cause)."""
    p = Path(path)
    if p.is_dir():
        raise FileNotFoundError(
            f"{what} {p} is a directory, not a file (cwd {Path.cwd()})")
    if not p.is_file():
        raise FileNotFoundError(
            f"{what} {p} does not exist (cwd {Path.cwd()}"
            + (f"; resolved {p.resolve()}" if not p.is_absolute() else "") + ")")
    return p


def read_embedding(path: Path | str) -> np.ndarray:
    """Read an embedding from an HDF5 file's dataset ``data``; orient as (cells, dims).

    Parameters
    ----------
    path : str or Path
        The benchmark's ``embedding.h5`` layout: one top-level dataset
        ``data`` (either orientation).

    Returns
    -------
    numpy.ndarray
        2-D array, cells x dims. Orientation auto-detection assumes there are
        more cells than embedding dimensions; square (cells == dims) or
        tall-thin embeddings cannot be auto-disambiguated and may come back
        transposed.

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist (the message names it and the cwd).
    ValueError
        The file has no dataset ``data``: the message lists the keys found
        and, when they are the canonical input layout (a ``matrix`` group
        holding ``data``/``barcodes``/``features``), says that this is an
        input matrix, not a method output.
    """
    path = _require_file(path, "output")
    with h5py.File(path, "r") as f:
        if "data" not in f or not isinstance(f["data"], h5py.Dataset):
            keys = sorted(f.keys())
            hint = ""
            if "matrix" in keys:
                hint = (" - this looks like a canonical input matrix (matrix/data, "
                        "matrix/barcodes, matrix/features: rna.h5 / adt.h5 / "
                        "atac.h5), not an embedding; evaluate() wants a method "
                        "output such as out/<method>/embedding.h5 (or pass an "
                        "AnnData / .npy / .csv embedding)")
            raise ValueError(
                f"{path} has no dataset 'data'; found keys {keys}{hint}")
        X = np.asarray(f["data"])
    if X.ndim != 2:
        raise ValueError(
            f"{path}: dataset 'data' is {X.ndim}-D {X.shape}; an embedding is "
            f"2-D (cells x dims)")
    if X.shape[0] < X.shape[1]:
        X = X.T
    return X


def _sep_for(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def read_labels(path: Path | str, column: str | None = None) -> np.ndarray:
    """Read a cell-type (or batch) label vector from a CSV/TSV with a header row.

    The benchmark's label files (``cty.csv``, ``rna_cty.csv``, ``cty1.csv`` ...)
    are one column with header ``"x"`` - R's ``write.csv(x)`` layout, possibly
    with a leading row-number column. This reader also accepts the pandas
    ``obs``-style export, i.e. ``adata.obs[["celltype"]].to_csv(p)`` (barcode
    index column + label column).

    Column choice, in order:

    1. ``column`` when given (must exist; error lists the columns otherwise);
    2. the column named ``x`` when present;
    3. the only column when the file has one;
    4. the last column when the file has exactly two and the first is all
       unique or has no header (an index / barcode column);
    5. otherwise the file is ambiguous and a ``ValueError`` lists the
       columns after the cell ids and asks for ``column=``: a silent pick
       could score the wrong column without any error. When the first of
       two columns is an index that repeats a barcode (text without a
       header, or 90% distinct), the ``ValueError`` names the repeated ids
       instead.

    Parameters
    ----------
    path : str or Path
        CSV (``,``) or TSV (``.tsv`` -> ``\\t``) file with a header row.
    column : str, optional
        Name of the column holding the labels.

    Returns
    -------
    numpy.ndarray
        1-D array of the raw label values (strings or numbers, as written);
        not integer codes. Every consumer in the package casts to ``str``.

    Notes
    -----
    The first column that rules 2 and 4 leave out may hold cell ids.
    Callers that know the cells' ids align the file by it
    (:func:`read_labels_ids`, :func:`by_id_column`).
    """
    return read_labels_ids(path, column)[0]


def read_labels_ids(path: Path | str, column: str | None = None, *,
                    what: str | None = None, pick=None):
    """:func:`read_labels`, plus the first column it left out.

    Returns ``(values, first)``. ``first`` is the file's first column (a
    ``pandas.Series``) when ``column`` or rule 2 picked another column, or
    rule 4 dropped the first of two columns; it may hold cell ids. Otherwise
    ``first`` is ``None``.

    ``what`` ("labels", "batch", "clustering") starts the messages of the
    errors. ``pick`` ends an ambiguous file's error: a sentence, or a
    function of the Python expression that reads one column of the file as
    a Series (:func:`_series_example`), for callers that have no ``column``
    argument (default: pass ``column=``).
    """
    path = _require_file(path, "labels file")
    d = pd.read_csv(path, sep=_sep_for(path))
    head = f"{what}: {path.name}" if what else str(path)
    first = d.iloc[:, 0] if d.shape[1] > 1 else None
    if column is not None:
        if column not in d.columns:
            kind, names = _data_columns(d)
            after = f" after the {kind}" if kind else ""
            have = (f"Its only column{after} is {names[0]}." if len(names) == 1 else
                    f"Its columns{after} are {', '.join(names)}.")
            raise ValueError(f"{head} has no column named {column!r}. {have}")
        other = first is not None and str(d.columns[0]) != str(column)
        return d[column].to_numpy(), (first if other else None)
    if "x" in d.columns:
        extra = first is not None and str(d.columns[0]) != "x"
        return d["x"].to_numpy(), (first if extra else None)
    if d.shape[1] == 1:
        return d.iloc[:, 0].to_numpy(), None
    no_header = _no_header(d.columns[0])
    if d.shape[1] == 2 and first.is_unique:
        # index/barcode column + one label column: the obs-style export
        return d.iloc[:, -1].to_numpy(), first
    # barcodes with a repeat, not a sample or batch column: an index written
    # without a header (pandas' and R's to_csv), or text that is 90% distinct
    index_like = no_header or (len(first) >= 10 and first.nunique() >= 0.9 * len(first))
    if d.shape[1] == 2 and _looks_like_ids(first) and index_like:
        dup = pd.Index(first[first.duplicated()].astype(str)).unique()
        raise ValueError(
            f"{what or path}: the first column of {path.name} repeats "
            f"{_n_ids(len(dup))} (first: {_first(dup, 3)}). Give each cell one row.")
    if d.shape[1] == 2 and no_header:
        # an index without a header that repeats numbers, such as the
        # RangeIndex of pd.concat([...]).to_csv(path): the other column
        return d.iloc[:, -1].to_numpy(), first
    kind, names = _data_columns(d)
    if callable(pick):
        # barcodes become the Series index; row numbers would not match any cell
        pick = pick(_series_example(path, names, index=kind == "cell ids", what=what))
    after = f" after the {kind}" if kind else ""
    raise ValueError(f"{head} has several columns{after}: {', '.join(names)}. "
                     f"{pick or 'Pass column=<name>.'}")


def _no_header(name) -> bool:
    """A column that had no header: pandas reads it as ``Unnamed: <i>``."""
    return str(name).startswith("Unnamed: ")


def _data_columns(d: pd.DataFrame) -> tuple[str | None, list]:
    """The columns of a per-cell file that may hold values, for messages.

    Returns ``(kind, names)``. When the first column is an index (no
    header, or unique text such as a barcode column), ``kind`` says what it
    holds (``"cell ids"`` or ``"row numbers"``) and ``names`` are the
    columns after it; otherwise ``kind`` is ``None`` and ``names`` are all
    the columns. pandas' ``Unnamed: <i>`` is not listed.
    """
    cols = [str(c) for c in d.columns]
    first = d.iloc[:, 0]
    ids = _looks_like_ids(first)
    if d.shape[1] > 1 and (_no_header(cols[0]) or (ids and first.is_unique)):
        kind = "cell ids" if ids else "row numbers"
        return kind, [c for c in cols[1:] if not _no_header(c)] or cols[1:]
    return None, [c for c in cols if not _no_header(c)] or cols


#: words in a column name that suggest what the column holds, by ``what``;
#: the example of an ambiguous file's error takes the first such column
_COLUMN_HINTS = {"labels": ("celltype", "cell_type", "cell type", "label", "cty"),
                 "batch": ("batch", "sample", "donor"),
                 "clustering": ("cluster", "leiden", "louvain")}


def _series_example(path: Path, names: list, *, index: bool, what: str | None) -> str:
    """The pandas call that reads one column of ``path`` as a Series:
    ``pd.read_csv("obs.csv", index_col=0)["sample"]``. The column is the
    first of ``names`` whose name suggests ``what``, else the last one."""
    hints = _COLUMN_HINTS.get(what or "", ())
    col = next((n for n in names if any(h in n.lower() for h in hints)), names[-1])
    args = [json.dumps(str(path), ensure_ascii=False)]
    if _sep_for(path) == "\t":
        args.append('sep="\\t"')
    if index:
        args.append("index_col=0")
    return f"pd.read_csv({', '.join(args)})[{json.dumps(col, ensure_ascii=False)}]"


def _n_ids(n: int) -> str:
    """``'1 id'`` or ``'2,864 ids'``."""
    return "1 id" if n == 1 else f"{n:,} ids"


def _looks_like_ids(col: pd.Series) -> bool:
    """Text that is not a number: R's row numbers ``1..n`` and pandas'
    ``0..n-1`` do not look like cell ids."""
    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        return False
    return bool(pd.to_numeric(col, errors="coerce").isna().any())


def by_id_column(values, first, ids, *, what: str, name: str, target: str,
                 order: str, no_ids: str, stacklevel: int = 3, n_files: int = 1):
    """Put a per-cell file in the order of ``ids`` by its first column.

    ``values`` and ``first`` come from :func:`read_labels_ids`; ``ids`` are
    the target's cell ids (``None`` when it has none, ``no_ids`` says why).
    Returns ``(vector, aligned)``:

    - a first column of text whose values are all cells of the target: the
      file is aligned by it, and a repeated id or a cell without a row
      raises ``ValueError``;
    - text where some values are cells and others are not: ``ValueError``
      naming a few of the others;
    - text with no cell, or a target without ids: the file is matched by
      position (``aligned`` False), with a ``UserWarning`` when the column
      is unique text (a repeated column such as a sample column is not an
      id column). ``order`` names the order the rows must then follow;
    - numbers: R's row numbers ``1..n`` or pandas' ``0..n-1``. They align
      the file only when they are exactly the target's ids in some order
      (an AnnData with the default ``obs_names``); otherwise the file is
      matched by position, silently, as before.

    ``n_files`` > 1 when ``name`` names several stacked files: the warnings
    then speak of them in the plural.
    """
    vals = np.asarray(values)
    if first is None:
        return vals, False
    text = _looks_like_ids(first)
    by_position = (f"The file is matched by position. Check that its rows follow {order}"
                   if n_files == 1 else
                   f"These files are matched by position. Check that their rows follow "
                   f"{order}")
    if ids is None:
        if text and first.is_unique:
            warnings.warn(f"The first column of {name} looks like cell ids, but "
                          f"{no_ids}. {by_position}.", UserWarning,
                          stacklevel=stacklevel)
        return vals, False
    keys = pd.Index([str(v) for v in first])
    ids = pd.Index([str(i) for i in ids])
    known = keys.isin(ids)
    if not text:
        if known.all() and keys.is_unique and len(keys) == len(ids):
            return np.asarray(pd.Series(vals, index=keys).reindex(ids).to_numpy()), True
        return vals, False
    if not known.any():
        if first.is_unique:
            warnings.warn(f"The first column of {name} looks like cell ids, but none "
                          f"is a cell of {target}. {by_position}, or use the barcodes "
                          f"of {target} in that column.", UserWarning,
                          stacklevel=stacklevel)
        return vals, False
    if not known.all():
        other = keys[~known]
        raise ValueError(
            f"{what}: {len(other):,} of the {len(keys):,} ids in the first column of "
            f"{name} are not cells of {target} (first: {_first(other, 3)}). Use the "
            f"barcodes of {target}, or drop that column when the rows follow {order}.")
    if not keys.is_unique:
        dup = keys[keys.duplicated()].unique()
        raise ValueError(
            f"{what}: the first column of {name} repeats {_n_ids(len(dup))} (first: "
            f"{_first(dup, 3)}). Give each cell one row.")
    missing = ids.difference(keys, sort=False)
    if len(missing):
        raise ValueError(
            f"{what}: {name} has no row for {len(missing):,} of the {len(ids):,} "
            f"cells of {target} (first: {_first(missing, 3)}). Give a row for every "
            f"cell.")
    return np.asarray(pd.Series(vals, index=keys).reindex(ids).to_numpy()), True


def read_clustering(path: Path | str) -> np.ndarray:
    """Read a precomputed clustering from an h5 file's ``/obs/cluster_leiden``.

    Parameters
    ----------
    path : str or Path
        The benchmark's clustered-output layout (bytes or ints under
        ``/obs/cluster_leiden``).

    Returns
    -------
    numpy.ndarray
        1-D integer cluster ids.

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist (names the path and the cwd).
    ValueError
        No ``/obs/cluster_leiden`` in the file (lists the keys found).
    """
    path = _require_file(path, "clustering file")
    with h5py.File(path, "r") as f:
        if "obs" not in f or "cluster_leiden" not in f["obs"]:
            raise ValueError(
                f"{path} has no dataset '/obs/cluster_leiden'; found keys "
                f"{sorted(f.keys())} - pass the clustering as a label CSV or a "
                f"1-D array instead")
        raw = np.asarray(f["/obs/cluster_leiden"]).flatten()
    decoded = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x for x in raw]
    return np.asarray(decoded).astype(int)


# ------------------------------------------------------------------ coercers
def _is_label_file(e) -> bool:
    if isinstance(e, Path):
        return True
    return isinstance(e, str) and Path(e).suffix.lower() in _LABEL_FILE_SUFFIXES


def _is_anndata(x) -> bool:
    # duck-typed so anndata stays an evaluation-only import
    return hasattr(x, "obsm") and hasattr(x, "obs") and hasattr(x, "n_obs")


def as_vector(x, *, what: str = "labels", column: str | None = None) -> np.ndarray:
    """Coerce a label-like input to a 1-D numpy array of length n_cells.

    Accepted forms:

    * ``str`` / ``Path`` - a label file, read with :func:`read_labels`;
    * ``list``/``tuple`` of paths (every element a ``Path`` or a ``str`` ending
      in ``.csv``/``.tsv``/``.txt``) - read each and concatenate in the given
      order, e.g. ``[cty1, cty2, cty3]`` for a multi-batch dataset;
    * ``dict`` with one entry (what :func:`multibench.labels_for` returns for a
      single-label dataset) - that file. Several entries raise here, because
      this coercer does not know the method's cell order; pass the paths
      as a list in that order. (:func:`multibench.evaluate` takes a
      multi-entry dict as is when it is an unchanged ``labels_for`` dict or
      in the default order, and needs ``label_order=`` otherwise.)
    * ``numpy.ndarray`` (1-D, or ``(n, 1)``), ``pandas.Series``,
      ``pandas.Categorical``, ``pandas.Index``, or a list/tuple of scalars;
    * a single-column ``pandas.DataFrame`` (or a wider one with ``column=``).

    Parameters
    ----------
    x
        The label-like input.
    what : str
        Name used in error messages (``'labels'``, ``'batch'``, ``'clustering'``).
    column : str, optional
        Column to take when ``x`` is a file path or a multi-column DataFrame.

    Returns
    -------
    numpy.ndarray
        1-D array. Values are returned as-is (not re-coded).
    """
    if isinstance(x, (str, Path)):
        return np.asarray(read_labels(x, column=column))
    if isinstance(x, dict):
        if len(x) == 1:
            return np.asarray(read_labels(next(iter(x.values())), column=column))
        raise ValueError(_multi_dict_message(what, x, label_order_hint=(what == "labels")))
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_label_file(e) for e in x):
        return np.concatenate([np.asarray(read_labels(p, column=column)) for p in x])
    if isinstance(x, pd.DataFrame):
        if column is not None:
            if column not in x.columns:
                raise ValueError(
                    f"{what}: DataFrame has no column {column!r}; columns are "
                    f"{[str(c) for c in x.columns]}")
            return np.asarray(x[column].to_numpy())
        if x.shape[1] == 1:
            return np.asarray(x.iloc[:, 0].to_numpy())
        raise ValueError(
            f"{what}: DataFrame has {x.shape[1]} columns "
            f"{[str(c) for c in x.columns]}; pass one column (e.g. df['celltype']) "
            f"or column=<name>")
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(
                f"{what} must be 1-D (one value per cell); got shape {x.shape}")
        return arr
    if isinstance(x, (pd.Series, pd.Categorical, pd.Index, list, tuple)):
        return np.asarray(pd.Series(x).to_numpy())
    raise TypeError(
        f"{what} must be a CSV path, list of paths, or 1-D array-like of length "
        f"n_cells; got {type(x).__name__}")


def _multi_dict_message(what: str, d: dict, *, label_order_hint: bool) -> str:
    """The error for a ``{name: path}`` dict with several entries.

    A dict fixes no cell order, and the order is the method's cell order -
    the order in which the method concatenated its input cells - which is not
    alphabetical: for most methods ``cty1 < cty2 < ...`` numerically, and
    ``rna`` before ``atac``. A wrong order raises no error and invalidates
    every score, so the message names the keys, states the rule and points
    at the helper that returns the files in that order.
    """
    keys = [str(k) for k in d]
    fix = (f"pass label_order=[...] with these keys in that order (label_order="
           f"list(d) trusts the dict's order)" if label_order_hint
           else "pass the paths as a list in that order")
    return (
        f"{what}: got a dict with {len(d)} label files {keys}; a dict does not "
        f"fix the cell order, and the order must be the method's cell order "
        f"(the order in which the method concatenated its input cells, which "
        f"is not alphabetical; for most methods numbered files ascending, cty1, "
        f"cty2, ..., and rna before atac). "
        f"{fix}, or a list of paths in cell order. "
        f"mtb.labels_for(dataset, method=<method>, category=<category>) returns "
        f"the files in that order.")


def _first(ix, n: int = 5) -> list:
    return [str(v) for v in list(ix[:n])]


def align_vector(x, ids, *, what: str = "labels", column: str | None = None) -> np.ndarray:
    """Align an indexed ``Series``/``DataFrame`` to the output's cell ids.

    A Series indexed by cell barcode is matched by barcode, not by position
    (the pandas convention). ``evaluate()`` calls this when ``output`` carries
    ids (an AnnData's ``obs_names``, or a DataFrame with a non-default index)
    and ``x`` carries a non-default index.

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame
        Labels indexed by cell id. A DataFrame must have one column, or
        ``column=`` names the one to take.
    ids : pandas.Index
        The output's cell ids, in the output's row order.
    what : str
        Argument name for error messages (``'labels'``, ``'batch'``,
        ``'clustering'``).
    column : str, optional
        Column of a multi-column DataFrame to take.

    Returns
    -------
    numpy.ndarray
        ``x`` reordered to ``ids`` (a no-op when the indexes already agree
        element-wise, which also covers duplicated ids in that case).

    Raises
    ------
    ValueError
        ``x`` (or ``ids``) has duplicate ids so the mapping is ambiguous; some
        of the output's ids are missing from ``x``; ``x`` has ids the output
        lacks. The message names the first few offenders and the positional
        escape hatch (``{what}.to_numpy()``).
    """
    if isinstance(x, pd.DataFrame):
        if column is not None:
            if column not in x.columns:
                raise ValueError(
                    f"{what}: DataFrame has no column {column!r}; columns are "
                    f"{[str(c) for c in x.columns]}")
            x = x[column]
        elif x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise ValueError(
                f"{what}: DataFrame has {x.shape[1]} columns "
                f"{[str(c) for c in x.columns]}; pass one column (e.g. "
                f"df['celltype']) or column=<name>")
    ids = pd.Index(ids)
    if len(x.index) == len(ids) and x.index.equals(ids):
        return np.asarray(pd.Series(x).to_numpy())       # same order already
    if not x.index.is_unique:
        dup = x.index[x.index.duplicated()]
        raise ValueError(
            f"{what}: the Series index has {len(dup)} duplicated id(s) (first: "
            f"{_first(dup)}); cannot align by cell id - pass {what}.to_numpy() "
            f"to match positionally")
    if not ids.is_unique:
        dup = ids[ids.duplicated()]
        raise ValueError(
            f"output carries {len(dup)} duplicated cell id(s) (first: "
            f"{_first(dup)}); cannot align {what} by cell id - pass "
            f"{what}.to_numpy() to match positionally")
    missing = ids.difference(x.index, sort=False)
    extra = x.index.difference(ids, sort=False)
    if len(missing) or len(extra):
        hint = ""
        if len(missing) == len(ids):
            hint = (" The two id sets are disjoint: is the output transposed "
                    "(evaluate expects cells x dims), or indexed differently?")
        raise ValueError(
            f"{what}: cannot align by cell id - {len(missing)} of the output's "
            f"{len(ids)} cells are missing from {what} (first: {_first(missing)}) "
            f"and {what} has {len(extra)} id(s) the output lacks (first: "
            f"{_first(extra)}). Pass {what} for exactly the output's cells, or "
            f"{what}.to_numpy() to match positionally.{hint}")
    return np.asarray(x.reindex(ids).to_numpy())


def _anndata_matrix(adata, obsm: str) -> np.ndarray:
    if obsm == "X":
        X = adata.X
        return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
    try:
        X = adata.obsm[obsm]
    except KeyError:
        kind = type(adata).__name__            # AnnData, or MuData for a MuData
        raise ValueError(
            f"obsm={obsm!r} not found in the {kind}; available obsm keys: "
            f"{sorted(adata.obsm.keys())} (or obsm='X' for .X)") from None
    X = X.toarray() if hasattr(X, "toarray") else X
    return np.asarray(X, dtype=float)


def _read_csv_matrix(path: Path) -> np.ndarray:
    d = pd.read_csv(path, sep=_sep_for(path))
    if d.shape[1] > 1 and (str(d.columns[0]).startswith("Unnamed")
                           or d.iloc[:, 0].dtype == object):
        # a written index (pandas' default "Unnamed: 0" or a barcode column)
        d = d.iloc[:, 1:]
    return d.to_numpy(dtype=float)


def as_matrix(output, *, obsm: str = "X_emb") -> np.ndarray:
    """Coerce a run output / embedding to a 2-D float numpy array.

    Accepted forms:

    * ``numpy.ndarray`` (returned as-is, no copy) or a scipy sparse matrix;
    * ``pandas.DataFrame`` - ``to_numpy(float)`` (index is ignored);
    * ``AnnData`` - ``adata.obsm[obsm]`` (or ``.X`` when ``obsm='X'``);
    * ``str`` / ``Path``, dispatched on suffix: ``.h5``/``.hdf5`` -> dataset
      ``data`` via :func:`read_embedding`; ``.h5ad`` -> ``obsm[obsm]``;
      ``.npy`` -> ``numpy.load``; ``.csv``/``.tsv`` -> numeric table (a
      leading index/barcode column is dropped). Any other suffix is tried as
      HDF5 and otherwise rejected.

    Orientation is not decided here: :func:`read_embedding` orients HDF5
    files, and :func:`multibench.evaluate` orients everything else against the
    label count. A path that does not exist raises ``FileNotFoundError`` naming
    it and the working directory; an ``.h5`` without dataset ``data`` raises
    ``ValueError`` as in :func:`read_embedding`.

    Parameters
    ----------
    output
        The embedding in any of the forms above.
    obsm : str
        Key of ``.obsm`` to use for AnnData / ``.h5ad`` inputs (``'X'`` = ``.X``).

    Returns
    -------
    numpy.ndarray
        2-D array.
    """
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "toarray") and not isinstance(output, pd.DataFrame):
        return np.asarray(output.toarray(), dtype=float)
    if isinstance(output, pd.DataFrame):
        return output.to_numpy(dtype=float)
    if _is_anndata(output):
        return _anndata_matrix(output, obsm)
    if isinstance(output, (str, Path)):
        p = _require_file(output, "output")
        suf = p.suffix.lower()
        if suf in {".h5", ".hdf5", ".hdf"}:
            return read_embedding(p)
        if suf == ".h5ad":
            import anndata as ad
            return _anndata_matrix(ad.read_h5ad(p), obsm)
        if suf == ".npy":
            return np.asarray(np.load(p), dtype=float)
        if suf in {".csv", ".tsv"}:
            return _read_csv_matrix(p)
        if h5py.is_hdf5(p):
            return read_embedding(p)
        raise ValueError(
            f"output path {p} has unrecognised suffix {suf!r}; expected "
            f".h5/.h5ad/.npy/.csv/.tsv (or an HDF5 file with dataset 'data')")
    raise TypeError(
        "output must be a (cells x dims) ndarray/DataFrame/AnnData or a path to "
        f".h5/.h5ad/.npy/.csv; got {type(output).__name__}")
