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

``missing_mask`` / ``missing_words`` say which per-cell values count as
missing: NaN, blank text, and text that a label file reads back as missing.
"""
from __future__ import annotations

import io as _io
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .. import _compat

#: suffixes of a CSV-like label file; evaluate() reads a clusters= path with
#: another suffix as an h5, and as_vector() takes other strings as label values
_LABEL_FILE_SUFFIXES = {".csv", ".tsv", ".txt"}


def _require_file(path: Path | str, what: str = "file") -> Path:
    """``Path(path)`` when it is an existing file; else ``FileNotFoundError``
    naming ``what``, the path and, for a relative path, where it resolves from the
    working directory (a notebook's cwd is the usual surprise)."""
    p = Path(path)
    where = ("" if p.is_absolute() else
             f" From the working directory {Path.cwd()}, the path resolves to "
             f"{p.resolve()}.")
    if p.is_dir():
        raise FileNotFoundError(f"The {what} {p} is a folder, not a file.{where}")
    if not p.is_file():
        raise FileNotFoundError(f"The {what} {p} does not exist.{where}")
    return p


def _read_h5ad(path, what: str = "file"):
    """The AnnData of the ``.h5ad`` file ``path`` (``what`` names it when it is
    missing); a damaged or non-``.h5ad`` file raises one sentence, not h5py's."""
    import anndata as ad
    p = _require_file(path, what)
    try:
        return ad.read_h5ad(p)
    except OSError:
        raise ValueError(f"{p.name} cannot be read: it is not an .h5ad file, "
                         f"or it is damaged.") from None


def _refuse_h5mu(path) -> None:
    """``ValueError`` for an ``.h5mu`` path: it holds one AnnData per modality,
    and evaluate takes one of them."""
    if Path(path).suffix.lower() == ".h5mu":
        raise ValueError(f"{Path(path).name} is an .h5mu: pass one modality, e.g. "
                         f"mudata.read(path)['rna'], with obsm='...'.")


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
    _refuse_h5mu(path)
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


def _read_table(src, sep):
    """The one pandas call of the label reader; :func:`missing_words` runs it
    too, so what counts as missing follows the pandas version."""
    return pd.read_csv(src, sep=sep)


def _read_text(src, sep):
    """The same table as text: nothing read as missing, no number parsed. For
    messages only."""
    return pd.read_csv(src, sep=sep, dtype=str, na_filter=False)


def _null_and_text(values):
    """``(null, text)``: where pandas holds a missing value, and every value as
    stripped text."""
    s = pd.Series(np.asarray(values, dtype=object))
    return s.isna().to_numpy(), s.astype(str).str.strip()


def _csv_field(text: str) -> str:
    """``text`` as one CSV field, quoted when it holds a comma, quote or line break."""
    if any(c in text for c in ',"\r\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def missing_words(values) -> list[str]:
    """The distinct texts in ``values`` that the label reader reads as missing, sorted.

    The texts, without the blanks around them, are written to a one-column
    CSV in memory and read back with :func:`_read_table`: a text that comes
    back as NaN is one a label file cannot hold as a label (``NA``,
    ``None``, ``nan``, ``null``, ``N/A``, ... with pandas' defaults). NaN,
    ``None``, ``pd.NA`` and blank text are missing without being words.
    """
    null, text = _null_and_text(values)
    text = text[~null]
    uniq = sorted(set(text[text != ""].tolist()))
    if not uniq:
        return []
    # not DataFrame.to_csv: up to Python 3.10 its writer leaves a lone carriage
    # return bare (read as a row end), and 3.10 refuses a NUL
    buf = _io.StringIO("x\n" + "".join(f"{_csv_field(u)}\n" for u in uniq))
    back = _read_table(buf, ",")
    if len(back) != len(uniq):
        # rows split or joined: the answers would belong to other texts
        raise RuntimeError(
            f"Cannot tell which of {len(uniq):,} distinct values a label file reads "
            f"as missing: written to a CSV and read back, they are {len(back):,} rows.")
    return [u for u, gone in zip(uniq, back["x"].isna().tolist()) if gone]


def missing_mask(values) -> np.ndarray:
    """``True`` where a per-cell value (a label, a batch) is missing.

    A value is missing when pandas holds it as one (NaN, ``None``,
    ``pd.NA``), when its text is blank, or when its text without the blanks
    around it is one of :func:`missing_words`. Every check of per-cell
    values uses this one definition, so a label that passes it and is
    written to a label file is read back as a label.
    """
    null, text = _null_and_text(values)
    words = missing_words(values)
    return null | (text == "").to_numpy() | text.isin(words).to_numpy()


def _missing_example(values) -> str:
    """``, for example 'NA'``: one text among ``values`` that a label file reads
    as missing; ``""`` when the missing entries are NaN or blank."""
    words = missing_words(values)
    return f", for example {words[0]!r}" if words else ""


def _as_written(col: pd.Series, d: pd.DataFrame, path: Path) -> np.ndarray:
    """Column ``col`` of the frame ``d`` read from ``path``, with each value read
    as missing replaced by the word the file holds there; an empty field stays
    NaN. For messages only."""
    vals, null = col.to_numpy(), col.isna().to_numpy()
    if not null.any():
        return vals
    text = _read_text(path, _sep_for(path))
    if text.shape != d.shape:
        return vals
    text = text.iloc[:, d.columns.get_loc(col.name)].str.strip()
    named = null & text.isin(missing_words(text[null])).to_numpy()
    out = vals.astype(object)
    out[named] = text[named].to_numpy()
    return out


def _names_no_file(x) -> bool:
    """A string that is not a file and does not look like one (``"donor"``,
    ``"obs:donor"``; not ``"missing.csv"``)."""
    return (isinstance(x, str) and not Path(x).is_file()
            and Path(x).suffix.lower() not in _LABEL_FILE_SUFFIXES
            and "/" not in x and os.sep not in x)


def read_labels(path: Path | str, column: str | None = None, *,
                as_written: bool = False) -> np.ndarray:
    """Read a cell-type (or batch) label vector from a CSV/TSV with a header row.

    The benchmark's label files (``cty.csv``, ``cty1.csv`` ...) are R's
    ``write.csv(x)`` layout: column ``"x"``, possibly after a row-number
    column. ``adata.obs[["celltype"]].to_csv(p)`` (barcodes + labels) works too.

    Column choice, in order:

    1. ``column`` when given (must exist; error lists the columns otherwise);
    2. the column named ``x`` when present;
    3. the only column when the file has one;
    4. the last column when the file has exactly two and the first is all
       unique (an index / barcode column), or has no header and holds the
       row numbers of ``pd.concat([...]).to_csv(path)`` (runs of 0..k-1);
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
    as_written : bool
        ``True`` returns each value read as missing as the word the file
        holds there (``'NA'``, not NaN), for a message that names it
        (:func:`read_labels_ids`).

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
    return read_labels_ids(path, column, as_written=as_written)[0]


def read_labels_ids(path: Path | str, column: str | None = None, *,
                    what: str | None = None, pick=None, as_written: bool = False):
    """:func:`read_labels`, plus the first column it left out.

    Returns ``(values, first)``. ``first`` is the file's first column (a
    ``pandas.Series``) when ``column`` or rule 2 picked another column, or
    rule 4 dropped the first of two columns; it may hold cell ids. Otherwise
    ``first`` is ``None``.

    ``what`` ("labels", "batch", "clustering") starts the error messages.
    ``pick`` ends an ambiguous file's error: a sentence, or a function of the
    pandas call that reads one column as a Series (default: pass ``column=``).
    ``as_written=True`` returns each value read as missing (``NA``, ``None``,
    ...) as the word the file holds, for messages only.
    """
    path = _require_file(path, f"{what} file"
                         if what else "labels file")
    d = _read_table(path, _sep_for(path))
    head = (f"The {what} file {path.name}"
            if what else str(path))
    first = d.iloc[:, 0] if d.shape[1] > 1 else None

    def values(col):
        return _as_written(col, d, path) if as_written else col.to_numpy()

    if column is not None:
        if column not in d.columns:
            kind, names = _data_columns(d)
            after = f" after the {kind}" if kind else ""
            have = (f"Its only column{after} is {names[0]}." if len(names) == 1 else
                    f"Its columns{after} are {', '.join(names)}.")
            raise ValueError(f"{head} has no column named {column!r}. {have}")
        other = first is not None and str(d.columns[0]) != str(column)
        return values(d[column]), (first if other else None)
    if "x" in d.columns:
        extra = first is not None and str(d.columns[0]) != "x"
        return values(d["x"]), (first if extra else None)
    if d.shape[1] == 1:
        return values(d.iloc[:, 0]), None
    no_header = _no_header(d.columns[0])
    if d.shape[1] == 2 and first.is_unique:
        return values(d.iloc[:, -1]), first
    # barcodes with a repeat, not a sample or batch column: an index written
    # without a header (pandas' and R's to_csv), or text that is 90% distinct
    index_like = no_header or (len(first) >= 10 and first.nunique() >= 0.9 * len(first))
    if d.shape[1] == 2 and _looks_like_ids(first) and index_like:
        dup = pd.Index(first[first.duplicated()].astype(str)).unique()
        raise ValueError(
            f"{what or path}: the first column of {path.name} repeats "
            f"{_n_ids(len(dup))} (first: {_first(dup, 3)}). Give each cell one row.")
    if d.shape[1] == 2 and no_header and _stacked_row_numbers(first):
        return values(d.iloc[:, -1]), first
    if d.shape[1] == 2 and no_header and pd.api.types.is_integer_dtype(first):
        # stacked row numbers, reordered after the concat
        raise ValueError(f"{head} has row numbers that repeat out of order, so its rows "
                         f"may not follow the cells. Save it in cell order.")
    kind, names = _data_columns(d)
    if callable(pick):
        # barcodes become the Series index; row numbers would not match any cell
        pick = pick(_series_example(path, names, index=kind == "cell ids", what=what))
    after = f" after the {kind}" if kind else ""
    raise ValueError(f"{head} has several columns{after}: {', '.join(names)}. "
                     f"{pick or 'Pass column=<name>.'}")


def _stacked_row_numbers(col: pd.Series) -> bool:
    """Runs of ``0, 1, ..., k-1``: the row numbers ``pd.concat([...]).to_csv``
    writes. Numbers in any other order are not."""
    if not pd.api.types.is_integer_dtype(col) or not len(col):
        return False
    v = col.to_numpy()
    return bool(v[0] == 0 and np.all((np.diff(v) == 1) | (v[1:] == 0)))


def _no_header(name) -> bool:
    """A column that had no header: pandas reads it as ``Unnamed: <i>``."""
    return str(name).startswith("Unnamed: ")


def _data_columns(d: pd.DataFrame) -> tuple[str | None, list]:
    """``(kind, names)``: the columns of a per-cell file that may hold values,
    for messages; ``kind`` is ``"cell ids"`` / ``"row numbers"`` when the first
    column is an index, else ``None``."""
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
    """The pandas call that reads one column of ``path``, the column whose name
    suggests ``what``, else the last: a Series indexed by the cell ids when the
    file has them, else an array, which is matched by position."""
    hints = _COLUMN_HINTS.get(what or "", ())
    col = next((n for n in names if any(h in n.lower() for h in hints)), names[-1])
    args = [json.dumps(str(path), ensure_ascii=False)]
    if _sep_for(path) == "\t":
        args.append('sep="\\t"')
    if index:
        args.append("index_col=0")
    return (f"pd.read_csv({', '.join(args)})[{json.dumps(col, ensure_ascii=False)}]"
            + ("" if index else ".to_numpy()"))


def _n_ids(n: int) -> str:
    """``'1 id'`` or ``'2,864 ids'``."""
    return "1 id" if n == 1 else f"{n:,} ids"


def _looks_like_ids(col: pd.Series) -> bool:
    """Text that is not a number: R's ``1..n`` and pandas' ``0..n-1`` row
    numbers are not cell ids."""
    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        return False
    return bool(pd.to_numeric(col, errors="coerce").isna().any())


def by_id_column(values, first, ids, *, what: str, name: str, target: str,
                 order: str, no_ids: str, n_files: int = 1):
    """Put a per-cell file in the order of ``ids`` by its first column.

    ``values`` and ``first`` come from :func:`read_labels_ids`; ``ids`` are
    the target's cell ids (``None`` when it has none, ``no_ids`` says why).
    Returns ``(vector, aligned)``:

    - a first column of text whose values are all cells of the target: the
      file is aligned by it, and a repeated id or a cell without a row
      raises ``ValueError``;
    - unique text with a row for every cell and rows of other cells: the
      file is aligned and the other rows are left out;
    - unique text where some or all values are not cells, and some cell
      has no row: ``ValueError`` naming a few of them;
    - repeated text with no cell (a sample column), or a target without ids:
      the file is matched by position (``aligned`` False), with a
      ``UserWarning`` when the column is unique text. ``order`` names the
      order the rows must then follow;
    - numbers: R's row numbers ``1..n`` or pandas' ``0..n-1``. They align
      the file only when they are exactly the target's ids in some order
      (an AnnData with the default ``obs_names``); otherwise the file is
      matched by position, silently.

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
            _compat.warn(f"The first column of {name} looks like cell ids, but "
                         f"{no_ids}. {by_position}.", UserWarning)
        return vals, False
    keys = pd.Index([str(v) for v in first])
    ids = pd.Index([str(i) for i in ids])
    known = keys.isin(ids)
    if not text:
        if known.all() and keys.is_unique and len(keys) == len(ids):
            return np.asarray(pd.Series(vals, index=keys).reindex(ids).to_numpy()), True
        return vals, False
    if not known.any() and not first.is_unique:
        return vals, False                  # a repeated column, such as a sample column
    if keys.is_unique and ids.isin(keys).all():
        # every cell has its row; the rows of other cells are left out
        return np.asarray(pd.Series(vals, index=keys).reindex(ids).to_numpy()), True
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
        mark = ""
        raise ValueError(
            f"{what}: {name} has no row for {len(missing):,} of the {len(ids):,} "
            f"cells of {target} (first: {_first(missing, 3)}). Give a row for every "
            f"cell.{mark}")
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


def _is_label_file(e) -> bool:
    if isinstance(e, Path):
        return True
    return isinstance(e, str) and Path(e).suffix.lower() in _LABEL_FILE_SUFFIXES


def _is_anndata(x) -> bool:
    # duck-typed so anndata stays an evaluation-only import
    return hasattr(x, "obsm") and hasattr(x, "obs") and hasattr(x, "n_obs")


def as_vector(x, *, what: str = "labels", column: str | None = None,
              as_written: bool = False) -> np.ndarray:
    """Coerce a label-like input to a 1-D numpy array of length n_cells.

    Accepted forms:

    * ``str`` / ``Path`` - a label file, read with :func:`read_labels`;
    * ``list``/``tuple`` of paths (every element a ``Path`` or a ``str`` ending
      in ``.csv``/``.tsv``/``.txt``) - read each and concatenate in the given
      order, e.g. ``[cty1, cty2, cty3]`` for a multi-batch dataset;
    * ``dict`` with one entry (:func:`multibench.labels_for` for a
      single-label dataset) - that file. Several entries raise: this
      coercer does not know the method's cell order.
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
    as_written : bool
        ``True`` reads files as :func:`read_labels_ids` does with
        ``as_written=True``, for a message that names the missing words;
        values that are not read from a file are not changed.

    Returns
    -------
    numpy.ndarray
        1-D array. Values are returned as-is (not re-coded).
    """
    if isinstance(x, (str, Path)):
        return np.asarray(read_labels(x, column=column, as_written=as_written))
    if isinstance(x, dict):
        if len(x) == 1:
            return np.asarray(read_labels(next(iter(x.values())), column=column,
                                          as_written=as_written))
        raise ValueError(_multi_dict_message(what, x, label_order_hint=(what == "labels")))
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_label_file(e) for e in x):
        return np.concatenate([np.asarray(read_labels(p, column=column,
                                                      as_written=as_written)) for p in x])
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
    """The error for a ``{name: path}`` dict with several entries: a dict fixes
    no cell order, and a wrong order invalidates every score without an error."""
    keys = [str(k) for k in d]
    return (
        f"{what}: got a dict with {len(d)} label files {keys}; a dict does not "
        f"fix the cell order, and the order must be the method's cell order "
        f"(the order in which the method concatenated its input cells, which "
        f"is not alphabetical; for most methods numbered files ascending, cty1, "
        f"cty2, ..., and rna before atac). Pass the paths as a list in that order, "
        f"or dataset=, category= and method=, which read them in that order.")


def _first(ix, n: int = 5) -> list:
    return [str(v) for v in list(ix[:n])]


def align_vector(x, ids, *, what: str = "labels", column: str | None = None) -> np.ndarray:
    """Align an indexed ``Series``/``DataFrame`` to the output's cell ids.

    A Series indexed by cell barcode is matched by barcode, not by position
    (the pandas convention). ``evaluate()`` matches the same way, through the
    private ``_align_vector``, when ``output`` carries ids (an AnnData's
    ``obs_names``, or a DataFrame with a non-default index) and ``x`` carries
    a non-default index.

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
    return _align_vector(x, ids, what=what, column=column, cells="adata.obs_names")


def _align_vector(x, ids, *, what: str, column: str | None, cells: str | None):
    """:func:`align_vector` for ``evaluate``. ``cells`` is the expression for
    the output's cell ids (``adata.obs_names``), ``None`` when there is none."""
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
        return np.asarray(pd.Series(x).to_numpy())
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
    if len(missing) and len(missing) == len(ids):
        head = (f"The {what}{chr(39) if what.endswith('s') else chr(39) + 's'} cell ids "
                f"do not match the embedding's (embedding e.g. {ids[0]}, {what} e.g. "
                f"{x.index[0]}).")
        # a 10x suffix such as -1 on one side only: the one fix
        from ..engine.ingest import suffix_fix
        fix = suffix_fix(what, list(ids), list(x.index))
        if fix:
            raise ValueError(head + fix)
        raise ValueError(f"{head} Pass {what} indexed like the embedding, or "
                         f"{what}.to_numpy() to match by position.")
    if len(missing):
        # ids the output lacks are left out: every cell has its value
        hint = (" Score the cells that have a label only."
                if what == "labels" and not len(extra) else "")
        by_position = (f", or {what}.to_numpy() to match positionally"
                       if len(x) == len(ids) else "")
        raise ValueError(
            f"{what}: cannot align by cell id - {len(missing)} of the output's "
            f"{len(ids)} cells are missing from {what} (first: {_first(missing)}). "
            f"Pass {what} for every cell of the output{by_position}.{hint}")
    return np.asarray(x.reindex(ids).to_numpy())


def _by_position(what: str) -> str:
    """The warning for a ``batch`` Series with a plain ``RangeIndex`` matched by
    position to cells that have barcodes."""
    return (f"{what} is a Series without barcodes and is matched to the cells by position. "
            f"To match by barcode, pass a Series indexed by barcode.")


def _numbers_rows(x, ids) -> bool:
    """``x`` is a Series with a plain ``RangeIndex`` and ``ids`` are barcodes,
    not ``None`` and not the row numbers themselves."""
    if not isinstance(x, pd.Series) or not isinstance(x.index, pd.RangeIndex) \
            or ids is None:
        return False
    return [str(i) for i in ids] != [str(i) for i in range(len(ids))]


def _labels_look_measured(values) -> tuple[str, int, int] | None:
    """``(reason, distinct, cells)`` when labels look like a number measured
    per cell: ``"distinct"`` (20+ cells, more labels than half the cells) or
    ``"fraction"`` (numbers that are not whole); else ``None``."""
    vals = np.asarray(values)
    n = len(vals)
    if not n:
        return None
    distinct = int(pd.Series(vals.astype(object)).nunique(dropna=True))
    numbers = vals.dtype.kind == "f" or (vals.dtype == object and pd.api.types.infer_dtype(
        vals, skipna=True) in ("floating", "mixed-integer-float"))
    if numbers:
        num = pd.to_numeric(pd.Series(vals.astype(object)), errors="coerce").to_numpy(float)
        num = num[np.isfinite(num)]
        if len(num) and bool((num != np.round(num)).any()):
            return "fraction", distinct, n
    if n >= 20 and distinct > n / 2:
        return "distinct", distinct, n
    return None


def _own_dtype(x) -> np.ndarray:
    """``x`` as an array at its own float dtype (float32 stays float32, as the
    article's scripts read ``embedding.h5``); any other dtype as float."""
    x = np.asarray(x.toarray() if hasattr(x, "toarray") else x)
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(float)


def _anndata_matrix(adata, obsm: str | None) -> np.ndarray:
    key = "X_emb" if obsm is None else obsm
    has_x = getattr(adata, "X", None) is not None
    if (obsm == "X" and has_x) or (obsm != "X" and key in adata.obsm):
        return _own_dtype(adata.X if obsm == "X" else adata.obsm[key])
    kind = type(adata).__name__            # AnnData, or MuData for a MuData
    keys = sorted(map(str, adata.obsm.keys()))
    picks = [f"obsm={k!r}" for k in keys[:1] if len(keys) == 1]
    picks = picks or (["obsm= with one of them"]
                      if keys else [])
    picks += ["obsm='X' for .X"] if has_x and obsm != "X" else []
    what = "no X" if obsm == "X" else f"no obsm[{key!r}]"
    raise ValueError(f"The {kind} has {what}; its obsm keys are {', '.join(keys) or 'none'}"
                     + (f": pass {' or '.join(picks)}." if picks else "."))


def _read_csv_embedding(path: Path):
    """A ``.csv`` / ``.tsv`` embedding as ``(matrix, ids)``.

    A first column without a header (``Unnamed: 0``) or of text is not part
    of the matrix; its values are the ids when one is not a number, else
    ``None``. A first column of numbers with a header is data.
    """
    path = Path(path)
    d = pd.read_csv(path, sep=_sep_for(path))
    ids = None
    if d.shape[1] > 1 and (str(d.columns[0]).startswith("Unnamed")
                           or d.iloc[:, 0].dtype == object):
        if _looks_like_ids(d.iloc[:, 0]):
            ids = pd.Index(d.iloc[:, 0].astype(str))
        d = d.iloc[:, 1:]
    return d.to_numpy(dtype=float), ids


def as_matrix(output, *, obsm: str | None = None) -> np.ndarray:
    """Coerce a run output / embedding to a 2-D float numpy array.

    Accepted forms:

    * ``numpy.ndarray`` (returned as-is, no copy) or a scipy sparse matrix;
    * ``pandas.DataFrame`` - ``to_numpy(float)`` (index is ignored);
    * ``AnnData`` - ``adata.obsm[obsm]`` (or ``.X`` when ``obsm='X'``);
    * ``str`` / ``Path``, dispatched on suffix: ``.h5``/``.hdf5`` -> dataset
      ``data`` via :func:`read_embedding`; ``.h5ad`` -> ``obsm[obsm]``;
      ``.npy`` -> ``numpy.load``; ``.csv``/``.tsv`` -> numeric table (a
      leading index/barcode column is not part of the matrix;
      :func:`multibench.evaluate` keeps the barcodes as the cell ids). Any
      other suffix is tried as HDF5 and otherwise rejected.

    Orientation is not decided here: :func:`read_embedding` orients HDF5
    files, and :func:`multibench.evaluate` orients everything else against the
    label count. A path that does not exist raises ``FileNotFoundError`` naming
    it and the working directory; an ``.h5`` without dataset ``data`` raises
    ``ValueError`` as in :func:`read_embedding`.

    Parameters
    ----------
    output
        The embedding in any of the forms above.
    obsm : str | None
        Key of ``.obsm`` to use for AnnData / ``.h5ad`` inputs (``'X'`` = ``.X``);
        ``None`` = ``X_emb``.

    Returns
    -------
    numpy.ndarray
        2-D array.
    """
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "toarray") and not isinstance(output, pd.DataFrame):
        return _own_dtype(output)
    if isinstance(output, pd.DataFrame):
        return _own_dtype(output.to_numpy())
    if _is_anndata(output):
        return _anndata_matrix(output, obsm)
    if isinstance(output, (str, Path)):
        p = _require_file(output, "output")
        _refuse_h5mu(p)
        suf = p.suffix.lower()
        if suf in {".h5", ".hdf5", ".hdf"}:
            return read_embedding(p)
        if suf == ".h5ad":
            return _anndata_matrix(_read_h5ad(p, "output"), obsm)
        if suf == ".npy":
            return _own_dtype(np.load(p))
        if suf in {".csv", ".tsv"}:
            return _read_csv_embedding(p)[0]
        if h5py.is_hdf5(p):
            return read_embedding(p)
        raise ValueError(
            f"output path {p} has unrecognised suffix {suf!r}; expected "
            f".h5/.h5ad/.npy/.csv/.tsv (or an HDF5 file with dataset 'data')")
    raise TypeError(
        "output must be a (cells x dims) ndarray/DataFrame/AnnData or a path to "
        f".h5/.h5ad/.npy/.csv; got {type(output).__name__}")
