"""Write your data as a dataset folder (:func:`export_dataset`); read one of its
files (:func:`read`).

Canonical layout: matrix/data (features x cells), matrix/features, matrix/barcodes.
Label files are single-column CSVs: a header ``x``, then one label per cell.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from .. import _text

from .._settings import config
from .registry import ATAC_VALUES as _ATAC_KINDS, check_category, check_id_list
from . import resolve
from .h5 import (_PEAK_RE, _cell_order, _comparable, _has_fraction, _is_canonical_h5,
                 _is_tenx_h5, _peak_fraction, _strings)

#: :func:`export_dataset` warns when ``matrix/data`` (stored dense, features x
#: cells) would exceed this many bytes uncompressed on disk.
DENSE_WARN_BYTES = 10 ** 9
#: Roles whose matrix must hold raw counts; gene-activity scores are not checked.
_COUNT_ROLES = ("rna", "adt", "atac_peak")
# '<selector>[<var column>=<value>]'
_VAR_FILTER_RE = re.compile(r"^(?P<base>.*?)\[(?P<col>[^\[\]=]+)=(?P<val>[^\[\]]*)\]$")


def _warn_not_counts(X, name: str) -> None:
    """``UserWarning`` when sampled stored values of ``X`` (cells x features)
    are not whole numbers, so not raw counts; no verdict for storage
    that cannot be read."""
    import scipy.sparse as sp
    try:
        if sp.issparse(X):
            vals = X.data if X.format in ("csr", "csc", "coo") else X.tocsr().data
        else:
            arr = np.asarray(X)
            if arr.ndim != 2 or arr.size == 0:
                return
            vals = arr[np.unique(np.linspace(0, arr.shape[0] - 1,
                                             min(arr.shape[0], 500)).astype(int))]
        fraction = _has_fraction(vals)
    except Exception:                   # backed or exotic storage: no verdict
        return
    if fraction:
        _text.warn(f"The {name} values are not whole numbers, so they are not raw counts. "
                   f"Export raw counts.", UserWarning)


def _matrix_key(data, spec):
    """A key equal for two modality arguments that read one matrix; ``data``
    passed again as an object is its ``X``."""
    if isinstance(spec, str):
        mod, base, var_filter = _parse_selector(spec)
        return "selector", mod, base or "X", var_filter
    if spec is data:
        return "selector", None, "X", None
    return "object", id(spec)


def _check_one_matrix_each(data, specs, given) -> None:
    """``ValueError`` when two modalities read the same matrix
    (``rna='X', atac='X'``). ``specs`` are the arguments as resolved, ``given``
    as written."""
    seen = {}
    for what, spec in specs.items():
        if spec is None:
            continue
        first = seen.setdefault(_matrix_key(data, spec), what)
        if first != what:
            pair = " and ".join(f"{w}={given[w]!r}" if isinstance(given[w], str) else f"{w}="
                                for w in (first, what))
            raise ValueError(f"{pair} name the same matrix. Select each modality's own "
                             f"matrix, or pass {first}=None.")


def _split_var_filter(spec: str):
    """``'X[feature_types=Peaks]'`` -> ``('X', ('feature_types', 'Peaks'))``;
    a selector without a filter -> ``(spec, None)``."""
    m = _VAR_FILTER_RE.match(spec)
    if not m:
        return spec, None
    return m.group("base"), (m.group("col").strip(), m.group("val").strip())


def _parse_selector(spec: str):
    """``'mod:rna.layer:c[col=v]'`` -> ``('rna', 'layer:c', ('col', 'v'))``;
    the modality is ``None`` without ``mod:``."""
    base, var_filter = _split_var_filter(spec)
    mod = None
    if base.startswith("mod:"):
        mod, _, base = base[4:].partition(".")
    return mod, base, var_filter


def _filter_var(adata, var_filter, *, what, spec):
    """The features of ``adata`` whose ``var[col]`` equals ``value`` (a view)."""
    col, val = var_filter
    if col not in adata.var.columns:
        raise KeyError(f"{what}={spec!r}: var column {col!r} not found; available: "
                       f"{list(adata.var.columns)}")
    vals = adata.var[col].astype(str).to_numpy()
    mask = vals == val
    if not mask.any():
        levels = sorted(set(vals.tolist()))
        shown = levels[:20] + (["..."] if len(levels) > 20 else [])
        raise ValueError(f"{what}={spec!r}: no feature has var[{col!r}] == {val!r}; "
                         f"the values of var[{col!r}] are {shown}")
    return adata[:, mask]


def _tenx_message(path) -> str:
    return f'{path} is a 10x Genomics file. See the recipe "A 10x .h5 file" in the docs.'


def _is_mudata(obj) -> bool:
    # duck-typed: mudata is an optional import
    return isinstance(getattr(obj, "mod", None), dict) and type(obj).__name__ == "MuData"


def _check_canonical(path) -> None:
    """``ValueError`` unless ``path`` is a canonical ``.h5``."""
    p = Path(path)
    if _is_canonical_h5(p):
        return
    if _is_tenx_h5(p):
        raise ValueError(_tenx_message(p))
    raise ValueError(f"{p} is not a canonical .h5: it has no 2-D matrix/data (features x "
                     f"cells). Write the folder with mtb.io.export_dataset.")


def _no_names_message(what: str, source: str, n_feat: int) -> str:
    """The warning for a matrix written with the fallback names ``feature_<i>``."""
    return (f"The {what} matrix {source} has no feature names; they are written as "
            f"feature_0 to feature_{n_feat - 1}. Pass the names with {what}_names=."
            if what == "adt" else
            f"feature_0 to feature_{n_feat - 1}.")


def _shown(spec) -> str:
    """How a message shows the value of an argument: a selector as it was
    written, an object by its type."""
    return repr(spec) if isinstance(spec, str) else type(spec).__name__


#: the name a 10x Genomics file gives the features of each modality in
#: ``var['feature_types']``
_TENX_TYPE = {"rna": "Gene Expression", "adt": "Antibody Capture", "atac": "Peaks"}   # wording only


def _check_one_feature_type(adata, *, arg, spec) -> None:
    """``ValueError`` when ``var['feature_types']`` holds a 10x type name and
    another type: one file holds one modality. ``spec`` is the selector, or
    ``None`` for an object."""
    if "feature_types" not in adata.var.columns:
        return
    counts = adata.var["feature_types"].astype(str).value_counts()
    if len(counts) < 2 or not set(counts.index) & set(_TENX_TYPE.values()):
        return
    types = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    head = f"features of {len(types)} types, such as {types[0]!r} and {types[1]!r}."
    one = _TENX_TYPE.get(arg)
    one = one if one in counts.index else "<type>"
    if spec is not None and _split_var_filter(spec)[1] is None:
        raise ValueError(f"{arg}={spec!r} selects {head} Add a filter, as "
                         f"{arg}='{spec}[feature_types={one}]'.")
    shown = repr(spec) if spec is not None else type(adata).__name__
    raise ValueError(f"{arg}={shown} holds {head} Pass one type, as {arg}=adata[:, "
                     f"adata.var['feature_types'] == {one!r}].")


def _pick_matrix(adata, *, layer=None, obsm=None, feature_names=None, role, arg, spec):
    """``(X, feature_names)`` of ``X``, a layer or an ``obsm`` entry of ``adata``.
    An ``obsm`` matrix has no var: its names come from ``feature_names``, a
    DataFrame's columns or ``uns['<key>_names']``, else ``feature_0..`` with a
    warning."""
    if obsm is not None:
        if obsm not in adata.obsm:
            raise KeyError(f"obsm key {obsm!r} not found; available: {list(adata.obsm)}")
        X = adata.obsm[obsm]
        names = None
        if hasattr(X, "columns"):                       # DataFrame
            names = [str(c) for c in X.columns]
            X = X.to_numpy()
        n_feat = X.shape[1] if getattr(X, "ndim", 0) == 2 else None
        if feature_names is not None:
            names = [str(v) for v in feature_names]
            if n_feat is not None and len(names) != n_feat:
                raise ValueError(f"{len(names)} feature names for {n_feat} features "
                                 f"in obsm[{obsm!r}]")
            return X, names
        uns_names = adata.uns.get(f"{obsm}_names")
        if names is None and uns_names is not None and n_feat is not None \
                and len(uns_names) == n_feat:
            names = [str(v) for v in uns_names]
        if names is None and n_feat is not None:
            names = [f"feature_{i}" for i in range(n_feat)]
            if n_feat:                  # 0 features have no names, and are refused
                _text.warn(_no_names_message(role, f"obsm[{obsm!r}]", n_feat), UserWarning)
        return X, names
    _check_one_feature_type(adata, arg=arg, spec=spec)
    if layer is not None:
        if layer not in adata.layers:
            # anndata 0.13 lists X among the layers, under the key None
            layers = [k for k in adata.layers if k is not None]
            raise KeyError(f"layer {layer!r} not found" + (
                f"; available: {layers}" if layers else ", and the object has no layers."))
        X = adata.layers[layer]
    else:
        X = adata.X
    if feature_names is not None:
        names = [str(v) for v in feature_names]
        if len(names) != X.shape[1]:
            raise ValueError(f"{len(names)} feature names for {X.shape[1]} features")
        return X, names
    return X, [str(v) for v in adata.var_names]


def _adt_obsm_keys(adata) -> list:
    """The ``obsm`` keys that can hold a protein matrix: the ones that do not
    start with ``X_``, where scanpy keeps embeddings (``X_pca``, ``X_umap``)."""
    return [k for k in adata.obsm if not str(k).startswith("X_")]


def _warn_dense_size(out: Path, n_feat: int, n_cell: int, dtype) -> None:
    """Warn when the dense ``matrix/data`` exceeds :data:`DENSE_WARN_BYTES`:
    every reader densifies it, however well gzip shrinks the file."""
    itemsize = int(np.dtype(dtype).itemsize)
    size = int(n_feat) * int(n_cell) * itemsize
    if size > DENSE_WARN_BYTES:
        _text.warn(f"{out.name} is {size / 1e9:.1f} GB uncompressed, stored dense. Keep "
                   f"fewer features, such as the highly variable genes, before the export.",
                   UserWarning)


#: the ATAC role of each ``atac_kind=``
_ATAC_ROLES = {"peak": "atac_peak", "gene_activity": "atac_gas"}


def _atac_role(feats, rna_feats) -> str:
    """The ATAC role the feature names show, for ``atac_kind=None``: peaks
    (``chr1:100-200``) or the RNA's gene names; neither raises."""
    if _peak_fraction(feats) > 0.5:
        return "atac_peak"
    genes = set(rna_feats or ())
    if genes and sum(1 for x in feats if x in genes) > 0.5 * len(feats):
        return "atac_gas"
    what = ("neither peaks such as chr1:100-200 nor the RNA's gene names" if genes
            else "not peaks such as chr1:100-200")
    raise ValueError(
        f"The ATAC feature names are {what}. Pass "
        + "atac_kind='peak' or atac_kind='gene_activity'.")


def _write_canonical(out: Path, X, feats, bars, *, dtype, compression, block):
    """Stream ``X`` (cells x features) to ``out`` as features x cells, one
    block of features at a time; CSC makes each block a cheap column slice."""
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.tocsc()
    else:
        X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"matrix must be 2-D (cells x features), got shape {X.shape}")
    n_cell, n_feat = X.shape
    if len(feats) != n_feat:
        raise ValueError(f"{len(feats)} feature names for {n_feat} features")
    if len(bars) != n_cell:
        raise ValueError(f"{len(bars)} barcodes for {n_cell} cells")
    block = max(1, int(block))
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        d = f.create_dataset("matrix/data", shape=(n_feat, n_cell), dtype=dtype,
                             chunks=True if compression else None,
                             compression=compression)
        for i in range(0, n_feat, block):
            blk = X[:, i:i + block]
            if sp.issparse(blk):
                blk = blk.toarray()
            d[i:i + block, :] = np.asarray(blk).T.astype(dtype, copy=False)
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))
    return out


def read(path: Path | str):
    """Read a canonical ``.h5`` (a dataset file or an ``embedding.h5``) as an AnnData.

    ``matrix/data`` (features x cells) becomes ``.X`` (cells x features, CSR
    when fewer than half of the entries are non-zero), ``matrix/features``
    the ``var_names`` and ``matrix/barcodes`` the ``obs_names``. A method's
    ``embedding.h5`` goes to ``.obsm["X_emb"]`` of an AnnData without features.

    Parameters
    ----------
    path : Path | str
        Canonical ``.h5`` file.

    Returns
    -------
    anndata.AnnData

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist.
    ValueError
        ``path`` is a 10x Genomics ``.h5`` file; the message says how to read it.

    Examples
    --------
    >>> import multibench as mtb
    >>> rna = mtb.io.read(mtb.config.data_path / "D11" / "rna.h5")
    """
    import anndata as ad
    import scipy.sparse as sp
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{os.fspath(path)} not found. Demo datasets are under "
            f"mtb.config.data_path ({config.data_path}).")
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path.name} is not an .h5 file.")
    if not _is_canonical_h5(path) and _is_tenx_h5(path):
        # its 1-D matrix/data would fail below with a line about unpacking
        raise ValueError(_tenx_message(path))
    with h5py.File(path, "r") as f:
        if "matrix/data" not in f and isinstance(f.get("data"), h5py.Dataset):
            # a method's embedding.h5: the embedding goes to .obsm["X_emb"]
            from ..eval.io import read_cell_names, read_embedding
            emb = read_embedding(path)
            a = ad.AnnData(np.zeros((emb.shape[0], 0)))
            a.obsm["X_emb"] = emb
            names = read_cell_names(path)
            if names is not None and len(names) == emb.shape[0]:
                a.obs_names = names
            return a
        if "matrix/data" not in f:
            raise ValueError(f"{Path(path).name} has no matrix/data: it is not a "
                             f"canonical .h5. Its keys: {sorted(f.keys())}.")
        data = np.array(f["matrix/data"]).T
        X = np.asarray(data, dtype=float)
        sparse = X.size > 0 and np.count_nonzero(X) / X.size < 0.5
        a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
        if "matrix/features" in f:
            a.var_names = _strings(np.array(f["matrix/features"]))
        if "matrix/barcodes" in f:
            a.obs_names = _strings(np.array(f["matrix/barcodes"]))
    return a


def _normalize_peak_names(src, dst):
    """Copy the canonical ``.h5`` ``src`` to ``dst`` with ATAC peak names rewritten to
    ``chr:start-end``, the spelling Signac's ``CreateChromatinAssay(sep=c(":","-"))``
    expects: a name ``<chr><sep><start><sep><end>`` with ``_``/``-``/``:`` then
    ``_``/``-`` as separators; gene symbols pass unchanged. ``run`` applies it for
    GLUE and Seurat_v3. Returns ``dst``."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        new = np.array([f"{m.group(1)}:{m.group(2)}-{m.group(3)}"
                        if (m := _PEAK_RE.match(x)) else x
                        for x in _strings(f["matrix/features"][:])], dtype="S")
        del f["matrix/features"]
        f.create_dataset("matrix/features", data=new)
    return dst


# --------------------------------------------------------------------------
# dataset-level export
# --------------------------------------------------------------------------

def _write_labels(labels, path: Path | str) -> Path:
    """Write 1-D ``labels`` as the benchmark's label CSV (header ``x``, one
    label per line, as strings) and return ``path``."""
    from ..eval import io as eio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(labels)
    if vals.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {vals.shape}")
    # not DataFrame.to_csv: up to Python 3.10 its writer leaves a lone carriage
    # return bare, which the reader takes for a row end. "" because the reader
    # skips an empty line.
    rows = [eio._csv_field(str(v)) or '""' for v in vals]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(os.linesep.join(["x", *rows]) + os.linesep)
    return path


def _select(data, spec, *, what):
    """``(adata, _pick_matrix kwargs)`` of a selector: ``'X'``,
    ``'obsm:<key>'``, ``'layer:<key>'``, each after an optional
    ``'mod:<name>.'``, then an optional ``[<var column>=<value>]``."""
    rest, var_filter = _split_var_filter(spec)
    adata = data
    if rest.startswith("mod:"):
        if not _is_mudata(data):
            raise ValueError(f"{what}={spec!r}: 'mod:<name>' selectors need a MuData "
                             f"input, got {type(data).__name__}")
        name, _, rest = rest[4:].partition(".")
        if name not in data.mod:
            raise KeyError(f"{what}={spec!r}: mod {name!r} not in MuData; found: {list(data.mod)}")
        adata = data.mod[name]
        rest = rest or "X"
    elif _is_mudata(data):
        raise ValueError(f"{what}={spec!r}: MuData input needs a 'mod:<name>' selector "
                         f"(found mods: {list(data.mod)})")
    if rest == "X":
        kw = {}
    elif rest.startswith("obsm:"):
        kw = {"obsm": rest[5:]}
    elif rest.startswith("layer:"):
        kw = {"layer": rest[6:]}
    else:
        raise ValueError(f"{what}={spec!r}"
                         + ": unknown selector; use X, obsm:<key>, layer:<key> or "
                         "mod:<name>[.obsm:<key>|.layer:<key>], optionally followed by "
                         "[<var column>=<value>].")
    if var_filter is not None:
        if "obsm" in kw:
            raise ValueError(f"{what}={spec!r}: a [<column>=<value>] filter selects "
                             f"features by adata.var; an obsm matrix has no var")
        adata = _filter_var(adata, var_filter, what=what, spec=spec)
    return adata, kw


def _mudata_selector(mdata, spec):
    """A bare modality name (``'rna'``, ``'rna[feature_types=...]'``) of a
    MuData -> the full ``'mod:<name>...'`` selector; anything else unchanged."""
    if not isinstance(spec, str):
        return spec
    base, var_filter = _split_var_filter(spec)
    if base in mdata.mod:
        return f"mod:{base}" + (spec[len(base):] if var_filter else "")
    return spec


def _obs_column(obs, col, *, what, spec, where):
    """``obs[col]``; ``where`` names ``obs`` in the error."""
    if col not in obs.columns:
        cols = [str(c) for c in obs.columns]
        more = f" and {len(cols) - 5:,} more" if len(cols) > 5 else ""
        raise KeyError(f"{what}={spec!r}: {where} has no column {col!r}. Columns: "
                       f"{', '.join(cols[:5]) or 'none'}{more}.")
    return obs[col]


def _select_obs(data, spec, *, what):
    """The Series an obs selector names: ``'obs:<col>'`` or ``'<col>'``
    (``data.obs``), ``'mod:<name>.obs:<col>'``, or for a MuData
    ``'<name>:<col>'`` (``mdata[name].obs``, then ``mdata.obs['<name>:<col>']``)."""
    mu = _is_mudata(data)
    if spec.startswith("mod:"):
        if not mu:
            raise ValueError(f"{what}={spec!r}: 'mod:<name>' needs a MuData input")
        name, _, rest = spec[4:].partition(".")
        if name not in data.mod:
            raise KeyError(f"{what}={spec!r}: mod {name!r} not in MuData; found: {list(data.mod)}")
        if not rest.startswith("obs:"):
            raise ValueError(f"{what}={spec!r}: use 'obs:<col>' (or 'mod:<name>.obs:<col>')")
        return _obs_column(data.mod[name].obs, rest[4:], what=what, spec=spec,
                           where=f"mdata[{name!r}].obs")
    where = "mdata.obs" if mu else "the AnnData's obs"
    if spec.startswith("obs:"):
        return _obs_column(data.obs, spec[4:], what=what, spec=spec, where=where)
    if mu:
        name, _, col = spec.partition(":")
        if name in data.mod and col:
            obs = data.mod[name].obs
            if col not in obs.columns and spec in data.obs.columns:
                return data.obs[spec]
            return _obs_column(obs, col, what=what, spec=spec, where=f"mdata[{name!r}].obs")
    return _obs_column(data.obs, spec, what=what, spec=spec, where=where)


def _link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _is_anndata(obj) -> bool:
    return hasattr(obj, "X") and hasattr(obj, "obs") and hasattr(obj, "obs_names")


def _as_modality(data, spec, *, what, master, feature_names=None):
    """``(adata, kw)`` of a modality argument: a selector against ``data``, an
    AnnData, a DataFrame (cells x features) or a 2-D array in ``master`` order."""
    import anndata as ad
    import scipy.sparse as sp

    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"{what}={spec!r} selects from data, which is None. Pass the "
                             f"AnnData as data.")
        return _select(data, spec, what=what)
    if _is_anndata(spec):
        return spec, {}
    if hasattr(spec, "columns") and hasattr(spec, "index"):        # DataFrame
        a = ad.AnnData(spec.to_numpy(dtype=float))
        a.obs_names = [str(x) for x in spec.index]
        a.var_names = [str(c) for c in spec.columns]
        return a, {}
    if sp.issparse(spec) or isinstance(spec, np.ndarray):
        if getattr(spec, "ndim", 0) != 2:
            raise ValueError(f"{what}= array must be 2-D (cells x features), got shape "
                             f"{getattr(spec, 'shape', None)}")
        if master is None:
            raise ValueError(f"{what}= is an array without cell barcodes. Pass an AnnData "
                             f"or DataFrame.")
        a = ad.AnnData(spec if sp.issparse(spec) else np.asarray(spec, dtype=float))
        a.obs_names = list(master)
        if feature_names is None:
            n_feat = spec.shape[1]
            if n_feat:                  # 0 features have no names, and are refused
                _text.warn(_no_names_message(what, "array", n_feat), UserWarning)
            feature_names = [f"feature_{i}" for i in range(n_feat)]
        a.var_names = [str(x) for x in feature_names]
        return a, {}
    raise ValueError(f"{what}= takes a selector such as 'X', an AnnData, a DataFrame or an "
                     f"array, not {type(spec).__name__}.")


def _check_unique_barcodes(names) -> None:
    """``ValueError`` when cell barcodes repeat: they could not be told apart,
    nor matched between modalities."""
    import pandas as pd
    idx = pd.Index([str(x) for x in names])
    dup = idx[idx.duplicated()].unique()
    if len(dup):
        raise ValueError(f"{len(dup):,} barcodes repeat, e.g. {dup[0]}. Make them unique "
                         f"(prefix the sample).")


def _strip_to_match(longer: list, shorter: list) -> str | None:
    """The suffix or prefix whose removal from every name of ``longer`` gives
    the names of ``shorter``, else ``None``."""
    a, b = longer[0], shorter[0]
    if a.startswith(b) and a != b:
        cut = a[len(b):]
        if all(x.endswith(cut) for x in longer) and \
                {x[:-len(cut)] for x in longer} == set(shorter):
            return cut
    if a.endswith(b) and a != b:
        cut = a[:len(a) - len(b)]
        if all(x.startswith(cut) for x in longer) and \
                {x[len(cut):] for x in longer} == set(shorter):
            return cut
    return None


def suffix_fix(what: str, cells: list, given: list) -> str:
    """`` They match after removing '-1' from the batch ids: pass ...`` when a 10x
    suffix (or prefix) on one side only is all that keeps the ids ``given`` from
    the ``cells``, else ``""``. evaluate, export_dataset and run name it."""
    cells, given = [str(i) for i in cells], [str(i) for i in given]
    if not cells or not given:
        return ""
    cut = _strip_to_match(cells, given)
    if cut and cells[0].endswith(cut):
        return (f" They match after adding {cut!r} to the {what} ids: pass "
                f"{what}.set_axis({what}.index + {cut!r}).")
    cut = _strip_to_match(given, cells)
    if cut and given[0].endswith(cut):
        return (f" They match after removing {cut!r} from the {what} ids: pass "
                f"{what}.set_axis({what}.index.str.removesuffix({cut!r})).")
    return ""


def _align_cells(a, *, master, master_name, role, category=None):
    """``a`` in the master cell order. Other barcodes raise, and so do repeated
    ones (:func:`_check_unique_barcodes`)."""
    names = [str(x) for x in a.obs_names]
    master = [str(x) for x in master]
    if names == master:
        return a
    pair = {role, master_name}
    # the RNA is read from data, unless rna= says otherwise
    ours, theirs = ("RNA" if master_name == "data" else master_name.upper()), role.upper()
    if master_name == "data":
        # the parameter name alone reads as "the data"
        master_name = "the object passed as data"
    unique = len(set(names)) == len(names) and len(set(master)) == len(master)
    if unique and set(names) == set(master):
        return a[master]
    if unique:
        in_master, in_names = set(master), set(names)
        stray = [x for x in names if x not in in_master]
        lack = [x for x in master if x not in in_names]
        same = "All modalities of one dataset must hold the same cells."
        if len(stray) == len(names):            # no cell in common
            head = (f"{ours} and {theirs} share no barcode ({ours} e.g. {master[0]}, "
                    f"{theirs} e.g. {names[0]}).")
            for longer, shorter, who in ((master, names, ours), (names, master, theirs)):
                cut = _strip_to_match(longer, shorter)
                if cut:
                    raise ValueError(f"{head} They match after removing {cut!r} from the "
                                     f"{who} names.")
            raise ValueError(head + (" Make the names match, or pass category='diagonal' for "
                                     "different cells." if "atac" in pair and category is None
                                     else " Make the names match."))
        if stray:
            head = (f"{role} has {len(names):,} cells, and {len(stray):,} of them are not "
                    f"in {master_name}, for example {stray[:3]}.")
        else:
            head = (f"{role} has {len(names):,} cells and {master_name} has "
                    f"{len(master):,}. {len(lack):,} cells of {master_name} are not in "
                    f"{role}.")
        raise ValueError(f"{head} {same} Subset each modality to the shared barcodes first.")
    _check_unique_barcodes(master)
    _check_unique_barcodes(names)
    return a


def _as_obs_vector(data, spec, *, what, master):
    """Labels or batch as a 1-D array in master order, from an obs selector, a
    Series aligned by index (a RangeIndex by position) or a sequence."""
    import pandas as pd

    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"{what}={spec!r} is a selector but data is None; pass a "
                             f"Series/array instead")
        ser = _select_obs(data, spec, what=what)
        src_names = [str(x) for x in ser.index]
        if master is not None and src_names != [str(x) for x in master]:
            if len(set(src_names)) != len(src_names):
                # by position every row would go to another cell
                raise ValueError(
                    f"{what}={spec!r}: the cell names are not unique, so the {what} cannot "
                    f"be matched to the cells. Give each cell a unique name first.")
            if set(src_names) == set(master):
                pos = {k: i for i, k in enumerate(src_names)}
                ser = ser.iloc[[pos[str(x)] for x in master]]
        vals = np.asarray(ser)
    elif isinstance(spec, pd.Series):
        if master is None:
            return np.asarray(spec)
        idx = [str(x) for x in spec.index]
        m = [str(x) for x in master]
        if idx == m:
            return np.asarray(spec)
        if len(set(idx)) == len(idx) and set(m) <= set(idx):
            pos = {k: i for i, k in enumerate(idx)}
            return np.asarray(spec)[[pos[x] for x in m]]
        if isinstance(spec.index, pd.RangeIndex) and len(spec) == len(m):
            # a filtered or sorted column would go to other cells
            from ..eval import io as eio
            _text.warn(eio._by_position(what), UserWarning)
            return np.asarray(spec)
        lack = [x for x in m if x not in set(idx)]
        fix = suffix_fix(what, m, idx) if len(lack) == len(m) else ""
        raise ValueError(
            f"{what}= Series index does not match the cell barcodes: {len(lack)} of "
            f"{len(m)} cells have no entry ({lack[:5]}{'...' if len(lack) > 5 else ''})."
            + (fix or " Index it by obs_names."))
    else:
        vals = np.asarray(spec)
        if vals.ndim != 1:
            raise ValueError(f"{what}= must be 1-D, got shape {vals.shape}")
    if master is not None and len(vals) != len(master):
        raise ValueError(f"{what} has {len(vals)} entries for {len(master)} cells: give one "
                         f"{'label' if what == 'labels' else what} per cell.")
    return vals


def export_dataset(data, dataset_dir: Path | str, *, rna="X",
                   adt=None, atac=None,
                   atac_kind: str | None = None, labels=None,
                   batch=None,
                   category: str | None = None,
                   adt_names: list | None = None,
                   batch_index: int | None = None,
                   overwrite: bool = False) -> Path:
    """Write an AnnData / MuData (or loose objects) as a canonical dataset folder.

    Produces the dataset folder layout (``rna.h5``, ``adt.h5``, ATAC files,
    ``cty.csv``), so ``mtb.scan`` and ``mtb.run`` work on your own data. Give
    raw counts. The files of each category are listed at
    https://dsichang.github.io/scMultiBench/tutorials/run/#folder-layout.

    Parameters
    ----------
    data : AnnData, MuData or None
        Object the selector strings refer to; ``None`` when every modality is
        passed as an object. With a MuData, name each modality (``rna='rna'``).
    dataset_dir : Path | str
        Folder to create; its name is the dataset id for ``mtb.scan`` /
        ``mtb.run``, and its parent their ``data_path``.
    rna : str, AnnData, DataFrame, array or None
        Raw-count matrix for RNA: a selector against ``data`` (``'layer:counts'``,
        ``'X[feature_types=Gene Expression]'``) or an AnnData, DataFrame or
        array; ``None`` = no RNA.
    adt : str, AnnData, DataFrame, array or None
        Raw-count matrix for protein (ADT), same forms as ``rna`` (e.g.
        ``'obsm:protein'``); ``None`` = no ADT.
    atac : str, AnnData, DataFrame, array or None
        ATAC matrix, same forms as ``rna`` (e.g. ``'X[feature_types=Peaks]'``);
        ``None`` = no ATAC.
    atac_kind : str or None
        What ``atac`` holds: ``'peak'`` or ``'gene_activity'``; decides the
        ATAC filename. ``None`` = read from the feature names: peaks
        (``chr1:100-200``) or the RNA's gene names.
    labels : str, Series, sequence or None
        Cell-type labels: a column, named ``'obs:<col>'`` or ``'<col>'`` (a
        MuData's global ``.obs``), ``'<mod>:<col>'`` (``mdata[mod].obs``), a
        Series indexed by barcode, or a sequence.
    batch : str, Series, sequence or None
        Batch per cell, same forms as ``labels``; splits the files per batch
        (``rna1.h5``, ``rna2.h5``, ...) for mosaic or cross. Every cell
        needs a batch.
    category : str or None
        Integration category the folder is for (``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``); sets the ATAC and label filenames.
    adt_names : list or None
        Protein names for the ADT matrix; they override any it carries and
        are needed when it has none.
    batch_index : int or None
        Write the whole object as batch ``N`` (``rna<N>.h5``, ``cty<N>.csv``);
        mosaic, cross or diagonal in several batches, one call per batch.
    overwrite : bool
        ``True`` = replace files already in ``dataset_dir``; ``False`` = raise
        before writing anything.

    Returns
    -------
    pathlib.Path
        ``dataset_dir``.

    Raises
    ------
    TypeError
        ``data`` is not an AnnData, MuData or ``None``.
    ValueError
        Missing or conflicting arguments, a missing label or batch, or cells
        that do not pair across modalities.
    KeyError
        A selector names a ``mod``, ``obsm``, ``layer``, ``obs`` or ``var`` column that is absent.
    FileExistsError
        A file this call would write exists and ``overwrite`` is ``False``, or the folder holds files of other data.

    Warns
    -----
    UserWarning
        Values that are not whole numbers, or feature names that are
        missing or repeated.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.io.export_dataset(adata, "data/MYCITE", rna="layer:counts",
    ...                       adt="obsm:protein", labels="obs:celltype")
    >>> mtb.io.export_dataset(mdata, "data/MYMULTIOME", rna="rna", atac="atac",
    ...                       atac_kind="peak", labels="obs:celltype",
    ...                       category="vertical")

    See Also
    --------
    mtb.scan : confirm the folder is found and which methods can run on it.
    """
    if data is not None and not (_is_anndata(data) or _is_mudata(data)):
        raise TypeError(f"data= must be an AnnData or MuData, got {type(data).__name__}.")
    out = Path(dataset_dir).expanduser()
    category = check_category(category)
    if data is None and isinstance(rna, str) and rna == "X":
        rna = None                      # no data to select from: no RNA
    given = {"rna": rna, "adt": adt, "atac": atac}      # as written, for messages
    # a bare modality name of a MuData selects mdata.mod[name].X; full
    # 'mod:<name>[...]' selectors pass through
    specs = ({what: _mudata_selector(data, x) for what, x in given.items()}
             if _is_mudata(data) else dict(given))
    _check_export_args(data, specs, given, atac_kind=atac_kind, labels=labels, batch=batch,
                       batch_index=batch_index, category=category)
    check_id_list(adt_names, "adt_names")
    diagonal = category == "diagonal"
    sides, masters, master = _modality_sides(data, specs, atac_kind=atac_kind,
                                             adt_names=adt_names, category=category)
    if not masters:                     # labels only
        if master is None:
            # no object names the cells: a Series indexed by barcode does
            master = _series_cells(labels, batch)
        if master is not None:
            masters["rna"] = master
    if any(len(m) == 0 for m in masters.values()):
        raise ValueError(_NO_CELLS)
    labels_by_side = _labels_by_side(data, labels, sides, masters, diagonal=diagonal)
    groups = _batch_groups(data, batch, batch_index, masters.get("rna"))
    # every matrix is selected and checked before any write
    prepared = _prepare_matrices(data, sides, given, adt_names=adt_names)
    plan, units = _plan_files(out, prepared, labels_by_side, masters, groups, category)
    existing = [pth.name for pth, _, _, _ in plan if pth.exists() or pth.is_symlink()]
    found = _leftovers(out, {pth.name for pth, _, _, _ in plan}, units, diagonal=diagonal,
                       by_column=batch is not None, batch_index=batch_index)
    if found:
        n_labels = None if prepared else sum(len(v) for v in labels_by_side.values())
        raise FileExistsError(_leftover_message(out, found, n_labels))
    if existing and not overwrite:
        raise FileExistsError(
            f"{out} already holds {_text._and_list(existing)}. Pass overwrite=True to "
            f"replace {'it' if len(existing) == 1 else 'them'}.")
    _write_plan(out, plan, _row_order(out, units, diagonal=diagonal))
    return out


def _check_export_args(data, specs, given, *, atac_kind, labels, batch, batch_index,
                       category) -> None:
    """The checks of :func:`export_dataset` that need no matrix."""
    rna, adt, atac = specs["rna"], specs["adt"], specs["atac"]
    if atac_kind is not None and atac_kind not in _ATAC_KINDS:
        raise ValueError(f"atac_kind= must be 'peak' or 'gene_activity', got {atac_kind!r}.")
    if rna is None and adt is None and atac is None and labels is None:
        raise ValueError("nothing to export: give at least one of rna=, adt=, atac=, labels=")
    _check_batch_args(batch, batch_index, category)
    _check_one_matrix_each(data, specs, given)
    if category == "diagonal" and labels is not None and rna is None and adt is None \
            and atac is None:
        raise ValueError("category='diagonal' with labels= needs rna= or atac=.")
    if category in ("mosaic", "cross") and batch is None and batch_index is None:
        raise ValueError(f"category={category!r} methods read one file per batch: pass "
                         f"batch= (the batch column) or batch_index=.")


def _modality_sides(data, specs, *, atac_kind, adt_names, category):
    """``(sides, masters, master)``: ``sides`` maps a side (``'rna'``, and
    ``'atac'`` for the ATAC cells of diagonal data) to its modalities as
    ``(role, adata, kw, what, source)``, each in the side's cell order, and
    ``masters`` maps a side to that order. ``master`` is the paired cell order."""
    diagonal = category == "diagonal"
    sides = {"rna": [], "atac": []}
    master, master_name = _first_master(
        data, [("rna", specs["rna"]), ("adt", specs["adt"])]
        + ([] if diagonal else [("atac", specs["atac"])]), diagonal=diagonal)
    for what in ("rna", "adt"):
        spec = specs[what]
        if spec is not None:
            a, kw = _as_modality(data, spec, what=what, master=master,
                                 feature_names=adt_names if what == "adt" else None)
            sides["rna"].append((what, a, kw, what, _source_name(spec, what)))
    atac = specs["atac"]
    if atac is not None:
        atac_master = master
        if diagonal:
            atac_master = _names_of(atac) or (
                [str(x) for x in data.obs_names] if _is_anndata(data) and not _is_mudata(data)
                else None)
        a, kw = _as_modality(data, atac, what="atac", master=atac_master)
        role = _ATAC_ROLES.get(atac_kind, "atac")     # "atac": from the feature names
        sides["atac" if diagonal else "rna"].append(
            (role, a, kw, "atac", _source_name(atac, "atac")))
    masters = {}
    for side, mats in sides.items():
        if not mats:
            continue
        if side == "rna" and master is not None:
            order, order_name = master, master_name
        else:
            order, order_name = [str(x) for x in mats[0][1].obs_names], mats[0][3]
        _check_unique_barcodes(order)
        masters[side] = order
        sides[side] = [(role, _align_cells(a, master=order, master_name=order_name, role=what,
                                              category=category),
                        kw, what, src) for role, a, kw, what, src in mats]
    return sides, masters, master


def _labels_by_side(data, labels, sides, masters, *, diagonal) -> dict:
    """``{side: labels}``: one vector per side (diagonal) or one for the paired cells."""
    if labels is None:
        return {}
    if diagonal:
        return _diagonal_labels(data, labels, sides)
    vec = _as_obs_vector(data, labels, what="labels", master=masters.get("rna"))
    if len(vec) == 0:
        raise ValueError(_NO_CELLS)     # labels alone, and none of them
    _check_labels_complete(vec, spec=labels, where="the cells")
    return {"rna": vec}


def _batch_groups(data, batch, batch_index, master) -> list:
    """``[(batch number or '', row indices or None)]``, one per batch the call writes."""
    if batch is not None:
        values = _as_obs_vector(data, batch, what="batch", master=master)
        _check_batch_complete(values, spec=batch)
        keys = sorted(set(values.tolist()), key=str)
        return [(str(i + 1), np.flatnonzero(values == k)) for i, k in enumerate(keys)]
    if batch_index is not None:
        return [(str(int(batch_index)), None)]
    return [("", None)]


def _prepare_matrices(data, sides, given, *, adt_names) -> list:
    """``[(side, role, X, feats, bars, what)]``: each matrix selected, with
    the content checks run and the ATAC role settled."""
    import pandas as pd

    prepared = []
    for side, mats in sides.items():
        for role, a, kw, what, src in mats:
            if role == "adt" and not kw and _adt_obsm_keys(a):
                raise ValueError(_adt_in_x_message(
                    given["adt"], _adt_obsm_keys(a),
                    own=not isinstance(given["adt"], str) and given["adt"] is not data))
            X, feats = _pick_matrix(a, layer=kw.get("layer"), obsm=kw.get("obsm"),
                                    feature_names=adt_names if role == "adt" else None,
                                    role=role, arg=what, spec=given[what]
                                    if isinstance(given[what], str) else None)
            if feats is not None and len(feats) == 0:
                raise ValueError(f"The {what} matrix holds 0 features for "
                                 f"{len(a.obs_names):,} cells. Check {what}=.")
            names = pd.Index([str(f) for f in feats or ()])
            dup = names[names.duplicated()].unique()
            if len(dup):
                _text.warn(f"{what}={_shown(given[what])} has {len(dup):,} repeated feature "
                           f"{'name' if len(dup) == 1 else 'names'}, for example {dup[0]}. "
                           f"Make them unique first.", UserWarning)
            if what == "rna" and feats is not None and _peak_fraction(feats) > 0.5:
                raise ValueError(f"rna= holds peaks ({feats[0]}): pass it as atac=.")
            if what == "atac":
                # the RNA is prepared first (side 'rna', then the ATAC)
                rna_feats = next((f for _, r, _, f, _, _ in prepared if r == "rna"), None)
                if role == "atac":
                    role = _atac_role(feats, rna_feats)
            if role in _COUNT_ROLES:
                _warn_not_counts(X, what)
            prepared.append((side, role, X, feats, [str(x) for x in a.obs_names], what))
    return prepared


def _plan_files(out: Path, prepared, labels_by_side, masters, groups, category):
    """``(plan, units)``: the files the call writes, as ``(path, kind,
    payload, unit)``, and ``{unit: (barcodes or None, number of cells)}``;
    a unit is ``(side, batch number or '')``."""
    diagonal = category == "diagonal"
    plan, units = [], {}
    for number, mask in groups:
        for side, role, X, feats, bars, what in prepared:
            sub = X if mask is None else X[mask]
            sub_bars = bars if mask is None else [bars[i] for i in mask]
            path = out / _export_filename(role, number, category)
            plan.append((path, "matrix", (sub, feats, sub_bars), (side, number)))
            if role == "atac_peak" and category is None:
                plan.append((out / f"atac{number}.h5", "link", path, (side, number)))
        for side, vec in labels_by_side.items():
            name = (f"{side}_cty{number}.csv" if diagonal else f"cty{number}.csv")
            plan.append((out / name, "labels", vec if mask is None else np.asarray(vec)[mask],
                         (side, number)))
        for side in dict.fromkeys(list(masters) + list(labels_by_side)):
            m = masters.get(side)
            if m is None:               # labels alone, without barcodes
                units[(side, number)] = (None, len(labels_by_side[side]) if mask is None
                                         else len(mask))
            else:
                units[(side, number)] = (m if mask is None else [m[i] for i in mask],
                                         len(m) if mask is None else len(mask))
    return plan, units


def _write_plan(out: Path, plan, follow) -> None:
    """Write the planned files; the rows of a unit in ``follow`` (:func:`_row_order`)
    keep the order the folder already holds them in."""
    dtype = "float64"
    plan = [(pth, kind, _in_order(kind, payload, follow[unit][0]) if unit in follow
             else payload, unit) for pth, kind, payload, unit in plan]
    for pth, kind, payload, _ in plan:
        if kind == "matrix":
            _warn_dense_size(pth, payload[0].shape[1], payload[0].shape[0], dtype)
    out.mkdir(parents=True, exist_ok=True)
    for pth, kind, payload, _ in plan:
        if pth.exists() or pth.is_symlink():
            pth.unlink()                # also breaks an old hard link
        if kind == "matrix":
            sub, feats, sub_bars = payload
            _write_canonical(pth, sub, feats, sub_bars, dtype=dtype,
                             compression="gzip", block=1024)
        elif kind == "link":
            _link_or_copy(payload, pth)
        else:
            _write_labels(payload, pth)
    for unit, (_, refs) in follow.items():
        files = [pth.name for pth, _, _, u in plan if u == unit]
        # a file this call replaced is no file that the new one follows
        ref = next((name for name in refs if name not in files), None)
        names = files[0] + (f" and {len(files) - 1} more" if len(files) > 1 else "")
        print(f"export_dataset: wrote {names} in the cell order "
              + (f"{refs[0]} had before." if ref is None else f"of {ref}."),
              file=sys.stderr, flush=True)


_NO_CELLS = ("The data holds 0 cells, so there is nothing to write. Check the filter that "
             "selects the cells.")


def _check_batch_complete(vals, *, spec) -> None:
    """Refuse missing batch values (``eval.io.missing_mask``): ``'nan'`` would
    become a batch of its own."""
    from ..eval import io as eio

    miss = eio.missing_mask(vals)
    if miss.any():
        raise ValueError(
            f"batch={_shown(spec)}"
            + f": {int(miss.sum()):,} of the {len(vals):,} cells have no batch"
            f"{eio._missing_example(vals)}. Give every cell a batch.")


def _adt_in_x_message(spec, keys, own: bool = False) -> str:
    """The error for ``adt`` reading ``X`` of an object whose ``obsm`` may hold
    the protein matrix. ``own``: the ADT came as its own object."""
    head = (f"adt={_shown(spec)} takes X, and the object also holds "
            + (f"obsm[{keys[0]!r}]." if len(keys) == 1 else f"the obsm entries {keys}."))
    if own:
        return f"{head} If X is the protein matrix, delete them from obsm first."
    key = keys[0] if len(keys) == 1 else "<key>"
    return f"{head} If the protein matrix is there, pass adt='obsm:{key}'."


#: file stems of the dataset layout, in the order messages list them
_LAYOUT_STEMS = ("rna", "adt", "atac", "atac_peak", "atac_gas", "cty", "rna_cty", "atac_cty")


def _layout_unit(name: str, diagonal: bool):
    """``(side, batch number or '', is a matrix)`` of a layout file, else
    ``None``; the files of one unit hold the same cells. Paired data has the
    one side ``'rna'``; in a diagonal folder a plain ``cty*.csv`` has none."""
    m = resolve._LAYOUT_FILE_RE.match(name)
    if not m:
        return None
    matrix = m.group(1) is not None
    number = m.group(1) if matrix else m.group(2)
    if not diagonal:
        return "rna", number, matrix
    if matrix or "_cty" in name:
        return ("atac" if name.startswith("atac") else "rna"), number, matrix
    return None, number, matrix


def _layout_files(out: Path, diagonal: bool):
    """``(path, unit)`` of each layout file in ``out``, by name (:func:`_layout_unit`)."""
    for p in sorted(out.iterdir(), key=lambda p: p.name):
        unit = _layout_unit(p.name, diagonal)
        if unit is not None and p.is_file():
            yield p, unit


def _layout_order(name: str):
    """Sort key of a layout file name: by batch number, then by
    :data:`_LAYOUT_STEMS`."""
    stem = name.rsplit(".", 1)[0]
    digits = stem[len(stem.rstrip("0123456789")):]
    return int(digits or 0), _LAYOUT_STEMS.index(stem[:len(stem) - len(digits)])


def _holds_other_cells(names, mine) -> bool:
    """True when ``names`` hold a cell not among ``mine``, compared as
    :func:`_cell_order` compares; ``False`` for non-unique barcodes."""
    judged = False
    for a, b in _comparable(names, mine):
        judged = True
        if set(a) <= set(b):
            return False
    return judged


def _folder_readers():
    """The resolver's barcode and label-row readers with their caches cleared:
    a file rewritten within one tick of the file system clock keeps its mtime."""
    resolve._sniff_barcode_list.cache_clear()
    resolve._count_label_rows.cache_clear()
    return (resolve._barcodes_of,
            lambda p: resolve._count_label_rows(str(p), p.stat().st_mtime_ns))


def _leftovers(out: Path, planned, units, *, diagonal, by_column, batch_index):
    """The names of the layout files of ``out`` the call does not write that
    hold other data: other batches, cells or label counts. A method reads
    every layout file, so such a file would join the dataset.
    ``units`` maps each unit the call writes to its barcodes and cell count.
    Files without unique barcodes, or unreadable label files, are not judged."""
    found = []
    if not out.is_dir():
        return found
    barcodes_of, n_rows = _folder_readers()
    side_cells, side_n = {}, {}         # what the call writes on a side, over its units
    for (side, _), (bars, n) in units.items():
        side_n[side] = side_n.get(side, 0) + n
        if bars is not None:
            side_cells.setdefault(side, []).extend(bars)
    for p, (side, number, matrix) in _layout_files(out, diagonal):
        if side is None or p.name in planned:
            continue
        if (side, number) in units:     # a unit the call writes: other cells or rows
            bars, n = units[(side, number)]
            if matrix:
                old = barcodes_of(p)
                if old is None:
                    continue
                named = bars is not None and len(set(bars)) == len(bars)
                if (_cell_order(old, bars)[0] == "other") if named else (len(old) != n):
                    found.append(p.name)
            elif n_rows(p) not in (None, n):
                found.append(p.name)
        elif number and by_column:      # a batch column gives every batch
            found.append(p.name)
        elif batch_index is not None:   # other batches are the build
            pass
        elif matrix:                    # another unit with a cell the call does not write
            old = barcodes_of(p)
            if old is not None and side in side_cells \
                    and _holds_other_cells(old, side_cells[side]):
                found.append(p.name)
        elif not number and by_column and side in side_n:   # unnumbered labels, batch split
            if n_rows(p) not in (None, side_n[side]):
                found.append(p.name)
    return found


def _leftover_message(out: Path, found, n_labels=None) -> str:
    """The error for the files ``found`` by :func:`_leftovers`; ``n_labels``
    for a call that writes labels only."""
    files = sorted(found, key=_layout_order)
    more = len(files) - 1
    names = files[0] + (f" and {more} more {'file' if more == 1 else 'files'}" if more else "")
    if n_labels is not None:
        return (f"The {n_labels:,} labels do not match the cells of {names} in {out}. To add "
                f"labels, run the call that wrote the folder again with labels= and "
                f"overwrite=True.")
    return f"{out} holds {names} of other data. Export to a new folder."


def _row_order(out: Path, units, *, diagonal) -> dict:
    """``{unit: (index, reference file names)}`` for the units whose cells the
    folder holds in another row order; ``index`` puts the call's rows in that
    order, as a label file has no barcodes to pair it by. ``ValueError``
    when two files of a unit disagree."""
    follow = {}
    if not out.is_dir():
        return follow
    barcodes_of, _ = _folder_readers()
    held = {}                           # unit -> [(file name, barcodes)]
    for p, unit in _layout_files(out, diagonal):
        if not unit[2] or unit[:2] not in units:
            continue
        bars, old = units[unit[:2]][0], barcodes_of(p)
        if bars is not None and old is not None \
                and _cell_order(bars, old)[0] in ("same", "order"):
            held.setdefault(unit[:2], []).append((p.name, old))
    for unit, refs in held.items():
        refs.sort(key=lambda r: _layout_order(r[0]))
        ref, first = refs[0]
        for name, bars in refs[1:]:
            if _cell_order(bars, first)[0] != "same":
                raise ValueError(
                    f"{ref} and {name} of {out} hold the same cells in different orders. "
                    + "Export this data again in one call, into a new folder.")
        kind, idx = _cell_order(units[unit][0], first)
        if kind == "order":
            follow[unit] = (idx, [name for name, _ in refs])
    return follow


def _in_order(kind: str, payload, idx):
    """The payload of one planned file with its rows taken in ``idx`` order."""
    import scipy.sparse as sp

    if kind == "matrix":
        sub, feats, sub_bars = payload
        sub = sub.tocsr()[idx] if sp.issparse(sub) else np.asarray(sub)[idx]
        return sub, feats, [sub_bars[i] for i in idx]
    if kind == "labels":
        return np.asarray(payload)[idx]
    return payload                      # a link


def _check_batch_args(batch, batch_index, category) -> None:
    """Refuse the batch / batch_index combinations no method can read."""
    if batch is not None and batch_index is not None:
        raise ValueError("Pass batch= or batch_index=, not both.")
    if batch_index is not None:
        if isinstance(batch_index, bool) or not isinstance(batch_index, (int, np.integer)) \
                or batch_index < 1:
            raise ValueError(f"batch_index= must be a positive integer (1 writes rna1.h5), "
                             f"got {batch_index!r}")
        if category not in ("mosaic", "cross", "diagonal"):
            raise ValueError("batch_index= writes numbered files, which only mosaic, cross "
                             "and diagonal methods read: pass category=.")
    if batch is not None and category in ("vertical", "diagonal"):
        raise ValueError("batch= writes per-batch files (rna1.h5, rna2.h5, ...). "
                         + resolve._one_file_advice(category))


def _names_of(spec):
    """Cell barcodes an AnnData / DataFrame argument carries, else ``None``."""
    if _is_anndata(spec):
        return [str(x) for x in spec.obs_names]
    if hasattr(spec, "index") and hasattr(spec, "columns"):
        return [str(x) for x in spec.index]
    return None


def _series_cells(*specs):
    """The cells of a labels-only call given no object: the index of the
    first ``labels`` / ``batch`` Series, unless it is a ``RangeIndex`` or not
    unique (then ``None``: the call is taken row by row)."""
    import pandas as pd

    named = [[str(x) for x in s.index] for s in specs
             if isinstance(s, pd.Series) and not isinstance(s.index, pd.RangeIndex)]
    if not named or any(len(set(names)) != len(names) for names in named):
        return None
    return named[0]


def _first_master(data, specs, *, diagonal):
    """``(barcodes, name)`` of the paired cell order: ``data.obs_names``, else
    the first modality object's. A diagonal MuData has no single order (its
    global ``obs_names`` joins both sides); the RNA side then sets it."""
    if data is not None and hasattr(data, "obs_names") and not (diagonal and _is_mudata(data)):
        return [str(x) for x in data.obs_names], "data"
    for what, spec in specs:
        names = _names_of(spec) if spec is not None else None
        if names is not None:
            return names, what
    return None, None


def _source_name(spec, what) -> str:
    """How an error names the object a modality came from."""
    if isinstance(spec, str):
        if spec.startswith("mod:"):
            return f"mdata[{_parse_selector(spec)[0]!r}]"
        return "data"
    return what


def _diagonal_labels(data, spec, sides) -> dict:
    """``{side: labels}`` of a diagonal export, each side from its own cells:
    a column name is read from each object's own ``.obs`` (a MuData's global
    ``.obs`` when a modality lacks it)."""
    import pandas as pd

    present = [side for side in ("rna", "atac") if sides[side]]
    if not isinstance(spec, (str, pd.Series)) and len(present) > 1:
        raise ValueError("With category='diagonal' and both rna= and atac=, pass labels= "
                         "as 'obs:<col>' or a Series indexed by barcode.")
    mu = _is_mudata(data)
    col_each = None                     # the column read from each side's own obs
    if isinstance(spec, str):
        if spec.startswith("obs:"):
            col_each = spec[4:]
        elif mu and not spec.startswith("mod:"):
            name, sep, col = spec.partition(":")
            if sep and name in data.mod and col:
                col_each = col
            elif not sep:
                col_each = spec
        elif not mu and not spec.startswith("mod:"):
            col_each = spec                 # '<col>' names the column as 'obs:<col>' does
    out = {}
    for side in present:
        role, a, kw, what, src = sides[side][0]
        bars = [str(x) for x in a.obs_names]
        if col_each is not None and col_each in a.obs.columns:
            out[side] = np.asarray(a.obs[col_each])
        else:
            if col_each is not None:
                if not (spec.startswith("obs:") and mu and col_each in data.obs.columns):
                    _obs_column(a.obs, col_each, what="labels", spec=spec,
                                where=f"the obs of the {side.upper()} object ({src})")
                ser = data.obs[col_each]
            elif isinstance(spec, str):
                ser = _select_obs(data, spec, what="labels")
            else:
                ser = spec
            out[side] = _as_obs_vector(None, ser, what="labels", master=bars)
        _check_labels_complete(out[side], spec=spec, where=f"the {side.upper()} cells ({src})")
    return out


def _check_labels_complete(vals, *, spec, where: str) -> None:
    """Refuse missing labels (``eval.io.missing_mask``): ``mtb.evaluate`` does
    not score them."""
    from ..eval import io as eio

    n = int(eio.missing_mask(vals).sum())
    if n:
        words = eio.missing_words(vals)
        forms = f"NaN, None, '' or text such as {words[0]!r}" if words else "NaN, None or ''"
        raise ValueError(
            f"labels={_shown(spec)}: {n} of the {len(vals)} labels for {where} are missing "
            f"({forms}). Keep only the labelled cells first.")


def _export_filename(role: str, number: str, category: str | None) -> str:
    """On-disk name of one modality file of :func:`export_dataset`."""
    if role.startswith("atac") and category in ("vertical", "mosaic"):
        return f"atac{number}.h5"
    return f"{role}{number}.h5"
