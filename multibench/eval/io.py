"""Readers and coercers for evaluation inputs: embedding, labels, clustering.

Two layers:

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
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

#: Suffixes that mean "a CSV-like label file" when a string is handed to
#: as_vector(); anything else that is a str/Path is treated as an h5 by
#: read_clustering(), and a list of such strings is treated as label VALUES.
_LABEL_FILE_SUFFIXES = {".csv", ".tsv", ".txt"}


def _require_file(path: Path | str, what: str = "file") -> Path:
    """``Path(path)`` when it is an existing file; else ``FileNotFoundError``
    naming the path AND the working directory (a relative path that resolves
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
        MORE cells than embedding dimensions; square (cells == dims) or
        tall-thin embeddings cannot be auto-disambiguated and may come back
        transposed.

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist (the message names it and the cwd).
    ValueError
        The file has no dataset ``data``: the message lists the keys found
        and, when they are the canonical INPUT layout (a ``matrix`` group
        holding ``data``/``barcodes``/``features``), says that this is an
        input matrix, not a method output.
    """
    path = _require_file(path, "output")
    with h5py.File(path, "r") as f:
        if "data" not in f or not isinstance(f["data"], h5py.Dataset):
            keys = sorted(f.keys())
            hint = ""
            if "matrix" in keys:
                hint = (" - this looks like a canonical INPUT matrix (matrix/data, "
                        "matrix/barcodes, matrix/features: rna.h5 / adt.h5 / "
                        "atac.h5), not an embedding; evaluate() wants a method "
                        "OUTPUT such as out/<method>/embedding.h5 (or pass an "
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

    1. ``column`` when given (must exist; error lists the header otherwise);
    2. the column named ``x`` when present;
    3. the only column when the file has one;
    4. the LAST column when the file has exactly two and the first is all
       unique (an index / barcode column);
    5. otherwise the file is ambiguous and a ``ValueError`` asks for
       ``column=``. (Silently taking a column here is how an obs-style export
       used to yield ARI 0.0 without a word.)

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
    """
    path = _require_file(path, "labels file")
    d = pd.read_csv(path, sep=_sep_for(path))
    cols = [str(c) for c in d.columns]
    if column is not None:
        if column not in d.columns:
            raise ValueError(
                f"{path}: no column named {column!r}; columns are {cols}")
        return d[column].to_numpy()
    if "x" in d.columns:
        return d["x"].to_numpy()
    if d.shape[1] == 1:
        return d.iloc[:, 0].to_numpy()
    first_unique = d.shape[1] > 1 and d.iloc[:, 0].is_unique
    if d.shape[1] == 2 and first_unique:
        # index/barcode column + one label column: the obs-style export
        return d.iloc[:, -1].to_numpy()
    hint = (" The first column looks like cell barcodes (all unique); write the "
            "CSV with index=False to drop it." if first_unique else "")
    raise ValueError(
        f"{path}: {d.shape[1]} columns {cols}; cannot tell which holds the "
        f"labels - pass column=<name>.{hint}")


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
      in ``.csv``/``.tsv``/``.txt``) - read each and concatenate IN THE GIVEN
      ORDER, e.g. ``[cty1, cty2, cty3]`` for a multi-batch dataset;
    * ``dict`` with ONE entry (what :func:`multibench.labels_for` returns for a
      single-label dataset) - that file. Several entries raise HERE, because
      this coercer does not know the method's stacking order; pass the paths
      as a list in that order. (:func:`multibench.evaluate` is more lenient:
      it takes a multi-entry dict as is when its insertion order IS the
      stacking order - what ``labels_for`` returns - and needs
      ``label_order=`` only for a dict in any other order.)
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

    A dict fixes no cell order, and the order is the method's STACKING order -
    the order in which the method concatenated its input cells - which is not
    alphabetical: ``cty1 < cty2 < ...`` numerically, and ``rna`` before
    ``atac``. The old hint (``list(d.values())``) recommended the alphabetical
    order and scored a perfect D28 embedding at ARI 0.001 - hence this message
    names the keys, states the rule and points at the helper that returns the
    files in that order.
    """
    keys = [str(k) for k in d]
    fix = (f"pass label_order=[...] with these keys in that order (label_order="
           f"list(d) trusts the dict's order)" if label_order_hint
           else "pass the paths as a list in that order")
    return (
        f"{what}: got a dict with {len(d)} label files {keys}; a dict does not "
        f"fix the cell order, and the order MUST be the method's stacking order "
        f"(the order the method concatenated its input cells: numbered files "
        f"ascending, cty1, cty2, ...; rna before atac) - NOT alphabetical. "
        f"{fix}, or a list of paths in cell order. "
        f"mtb.labels_for(dataset, method=<method>, category=<category>) returns "
        f"the files in that order.")


def _first(ix, n: int = 5) -> list:
    return [str(v) for v in list(ix[:n])]


def align_vector(x, ids, *, what: str = "labels", column: str | None = None) -> np.ndarray:
    """Align an indexed ``Series``/``DataFrame`` to the output's cell ids.

    This is the pandas contract every scverse user expects: a Series indexed by
    cell barcode is matched BY BARCODE, not by position. ``evaluate()`` calls
    this when ``output`` carries ids (an AnnData's ``obs_names``, or a
    DataFrame with a non-default index) and ``x`` carries a non-default index.

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
        raise ValueError(
            f"obsm={obsm!r} not found in the AnnData; available obsm keys: "
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

    Orientation is NOT decided here: :func:`read_embedding` orients files, and
    :func:`multibench.evaluate` orients everything else against the label count.
    A path that does not exist raises ``FileNotFoundError`` naming it and the
    working directory; an ``.h5`` without dataset ``data`` raises
    ``ValueError`` listing the keys found (and says so when they are the
    canonical INPUT layout ``matrix/...``).

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
