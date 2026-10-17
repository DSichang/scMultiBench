"""classify(): predict the cell type of each query cell from a reference batch;
classification_metrics(): score those predictions."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from .. import _compat
from ..config import _and_list
from ..plot.bar import CLASSIFICATION_METRICS as _CLASSIFICATION_METRICS
from .pipeline import (_Table, _as_written, _cell_ids, _count_error,
                       _is_label_path_list, _labels_from_dict, _n_values,
                       _no_batch_error, _no_label_error, _obs_or_vector,
                       _stack_label_files)

#: the columns of the frame classify() returns
COLUMNS = ["reference", "batch", "cell", "label", "predicted"]


@_compat.keywords("classify", renamed={"output": "embedding"})
def classify(embedding, labels=None, *, batch=None, reference=None,
             obsm: str = "X_emb", label_order=None, seed: int = 0) -> pd.DataFrame:
    """Predict the cell type of each query cell from a reference batch.

    One batch is the reference: a small neural network learns the cell
    types from the embedding of its cells, then predicts the cells of all the
    other batches, the query.

    Parameters
    ----------
    embedding
        Embedding, cells x dims, in the forms ``mtb.evaluate`` takes: a run
        output file (``embedding.h5``, ``.h5ad``, ``.csv``), an AnnData, a
        DataFrame or an array.
    labels
        Cell types, one per cell, in the forms ``mtb.evaluate`` takes. A
        ``mtb.labels_for`` dict also gives the batches: each file is one
        batch, named by its key (``cty1``, ``cty2``, ...).
    batch
        Batch of each cell, in the forms ``labels`` takes; needed unless
        ``labels`` lists two or more files.
    reference : str, optional
        The batch to learn from. ``None`` = every batch in turn, as in the
        article.
    obsm : str
        ``.obsm`` key of an AnnData or ``.h5ad`` ``embedding``; ``'X'`` means
        ``.X``.
    label_order : list of str
        Keys of a ``labels`` dict in the method's cell order, as in
        ``mtb.evaluate``; ``None`` works for an unchanged ``mtb.labels_for``
        dict.
    seed : int
        Seed of the network; the same seed and the same input, with the
        cells in the same order, give the same predictions.

    Returns
    -------
    pandas.DataFrame
        One row per query cell and reference, with the columns
        ``reference``, ``batch`` (the cell's own batch), ``cell`` (the cell
        id, or the row number when the embedding has none), ``label`` (the
        true cell type) and ``predicted``.

    Raises
    ------
    ValueError
        Fewer than two batches, an input that is not an embedding, a
        ``reference`` that is not a batch, and the label and batch errors of
        ``mtb.evaluate``.

    Warns
    -----
    UserWarning
        Cells without a label are left out; a reference batch with fewer
        than two cell types is skipped.

    Examples
    --------
    >>> import multibench as mtb
    >>> labels = mtb.labels_for("D52", "cross", "StabMap")
    >>> pred = mtb.classify(res.output, labels=labels)    # res = mtb.run(...)

    See Also
    --------
    mtb.labels_for : the label files of a dataset, in the method's cell order.
    mtb.evaluate : scores the same embedding with the scIB metrics.
    """
    if labels is None:
        raise ValueError(
            "classify needs labels. Pass labels=mtb.labels_for(dataset, category, method).")
    if isinstance(labels, dict) and not labels:
        raise ValueError(
            "labels is an empty dict. Pass labels=mtb.labels_for(dataset, category, method).")
    names = None
    if isinstance(labels, dict):
        names = [str(k) for k in (labels if label_order is None else label_order)]
        labels = _labels_from_dict(labels, label_order)
    elif label_order is not None:
        raise TypeError("label_order= applies only when labels is a dict. "
                        "Pass the files as a list in cell order.")
    if _is_label_path_list(labels) and names is None:
        stems = [Path(p).stem for p in labels]
        names = stems if len(set(stems)) == len(stems) else \
            [str(i + 1) for i in range(len(labels))]
    _check_embedding(embedding)

    cells = "adata.obs_names" if io._is_anndata(embedding) else None
    if isinstance(embedding, (str, Path)) and Path(embedding).suffix.lower() == ".h5ad":
        import anndata as ad
        embedding = ad.read_h5ad(embedding)
    adata = embedding if io._is_anndata(embedding) else None
    table = None
    if isinstance(embedding, (str, Path)) and Path(embedding).suffix.lower() in {".csv", ".tsv"}:
        file = io._require_file(embedding, "output")
        emb, ids = io._read_csv_embedding(file)
        if ids is not None and emb.shape[0] != emb.shape[1] \
                and _n_values(labels, adata) == emb.shape[1]:
            ids = None           # dims x cells: the first column names the dimensions
        table = None if ids is None else _Table(file.name)
    else:
        ids = _cell_ids(embedding)
        emb = np.asarray(io.as_matrix(embedding, obsm=obsm))
    if emb.ndim != 2:
        raise ValueError(_NOT_EMBEDDING.format(what="output"))
    if _is_label_path_list(labels):
        ct, origin = _stack_label_files(labels, ids, table=table)
    else:
        ct, origin = _obs_or_vector(labels, adata, what="labels", ids=ids,
                                    table=table, cells=cells), None
    if emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        emb = emb.T                    # dims x cells, as evaluate orients it
    n_cells = emb.shape[0]
    if len(ct) != n_cells:
        raise ValueError(_count_error("labels", len(ct), n_cells, labels))
    if emb.shape[1] == n_cells or _is_neighbour_index(emb):
        raise ValueError(_graph_error("output"))     # cells x cells, or a kNN index
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ids, table=table, cells=cells)
        if len(ba) != n_cells:
            raise ValueError(_count_error("batch", len(ba), n_cells, batch))
    elif origin is not None:
        ba = np.asarray(names, dtype=object)[np.asarray(origin) - 1]
    else:
        raise ValueError(
            "classify needs the batch of each cell. Pass batch=, or "
            "labels=mtb.labels_for(dataset, category, method).")

    miss = io.missing_mask(ct)
    if miss.all():
        shown = _as_written(labels, adata, ids, what="labels")
        raise ValueError(_no_label_error(ct if shown is None else shown, miss, labels))
    keep = ~miss
    has_batch = ~io.missing_mask(ba)
    if (keep & ~has_batch).any():
        shown = _as_written(batch, adata, ids, what="batch")
        raise ValueError(_no_batch_error(
            np.asarray(ba if shown is None else shown)[keep], ~has_batch[keep], batch))
    ba = np.array([str(b) for b in ba], dtype=object)
    ct = np.array([str(c) for c in ct], dtype=object)
    batches = list(pd.unique(ba[has_batch]))
    if len(batches) < 2:
        raise ValueError(
            f"classify needs at least two batches; every cell is in batch {batches[0]}. "
            f"Pass batch= with two or more batches.")
    if reference is not None and str(reference) not in batches:
        raise ValueError(
            f"reference={reference!r} is not a batch. Pass one of "
            f"{_and_list(batches, 'or')}.")
    if miss.any():
        _compat.warn(f"{int(miss.sum()):,} of the {n_cells:,} cells have no label; "
                     f"they are left out.", UserWarning)
    cell = np.asarray(ids, dtype=object) if ids is not None else np.arange(n_cells)
    frames = []
    for ref in batches if reference is None else [str(reference)]:
        train = keep & (ba == ref)
        types = pd.unique(ct[train])
        if len(types) < 2:
            _compat.warn(
                f"Reference batch {ref} has " + ("no labelled cell" if len(types) == 0
                                                 else "one cell type")
                + "; skipped.", UserWarning)
            continue
        query = keep & (ba != ref) & np.isin(ct, types)
        if not query.any():
            _compat.warn(f"No other batch has a cell type of reference batch {ref}; "
                         f"skipped.", UserWarning)
            continue
        predicted = _mlp(emb[train], ct[train], seed).predict(emb[query])
        frames.append(pd.DataFrame({
            "reference": ref, "batch": ba[query], "cell": cell[query],
            "label": ct[query], "predicted": np.asarray(predicted, dtype=object)}))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)[COLUMNS]


_NOT_EMBEDDING = ("classify needs an embedding; {what} is not one. "
                  "Pass the method's embedding file.")


def _graph_text(who: str) -> str:
    """What is wrong with the output of a graph method: ``who`` is the method,
    or ``"output"`` when the method is not known."""
    if who == "output":
        return "output is a graph, not an embedding, so it has no classification scores."
    return f"{who} writes a graph, not an embedding, so it has no classification scores."


def _check_embedding(output) -> None:
    """``ValueError`` for an output that is a graph or labels, not an embedding:
    a file a method writes as its graph or label output, or a list of labels."""
    who = _graph_writer(output)
    if who is not None:
        raise ValueError(_graph_error(who))
    what = _not_embedding(output)
    if what is not None:
        raise ValueError(_NOT_EMBEDDING.format(what=what))


def _graph_error(who: str) -> str:
    return _graph_text(who) + " Score it with mtb.evaluate(metrics='clustering')."


def _not_embedding(output) -> str | None:
    """The name of an output that is labels, not an embedding (``"output"``
    for a list, the file name for a label file); ``None`` for anything else."""
    if isinstance(output, (list, tuple)):
        return "output"
    if isinstance(output, (str, Path)) and Path(output).name in _label_outputs():
        return Path(output).name
    return None


def _graph_writer(output) -> str | None:
    """The method that writes ``output`` when it is a file only graph methods
    write; ``"output"`` when several do; ``None`` for anything else."""
    if not isinstance(output, (str, Path)):
        return None
    methods = _graph_files().get(Path(output).name)
    if not methods:
        return None
    return methods[0] if len(methods) == 1 else "output"


def _registry_files() -> dict:
    """``{file name: [(output kind, variant output kind, method), ...]}`` over
    every output and extra output of the registry."""
    from ..engine import registry
    files: dict = {}
    for spec in registry.load():
        for v in spec.variants:
            for o in [v.output, *v.extra_outputs]:
                files.setdefault(o.file, []).append((o.kind, v.output.kind, spec.id))
    return files


def _label_outputs() -> set:
    """File names the registry gives to label outputs, and to no embedding."""
    return {f for f, uses in _registry_files().items()
            if {k for k, _, _ in uses} == {"labels"}}


def _graph_files() -> dict:
    """``{file name: [method, ...]}`` for files that only graph variants write."""
    return {f: sorted({m for _, _, m in uses}) for f, uses in _registry_files().items()
            if all(vk == "graph" for _, vk, _ in uses)}


def _is_neighbour_index(emb) -> bool:
    """Whether ``emb`` is a nearest-neighbour index (scMoMaT's graph), not an
    embedding: fewer columns than cells, whole numbers from 0 to the number of
    cells, no value twice in a row. Embedding values are not whole numbers."""
    a = np.asarray(emb)
    if a.ndim != 2 or not 2 <= a.shape[1] < a.shape[0]:
        return False
    if a.dtype.kind == "f":
        if not (np.isfinite(a).all() and (a == np.round(a)).all()):
            return False
    elif a.dtype.kind not in "iu":
        return False
    if a.min() < 0 or a.max() > a.shape[0]:
        return False
    s = np.sort(a, axis=1)
    return bool((s[:, 1:] != s[:, :-1]).all())


def _mlp(x, y, seed: int):
    """The benchmark's classifier fitted on ``x`` and ``y``: Linear(dims, 128),
    ReLU, Linear(128, types); Adam, learning rate 0.01, batches of 64, 10 epochs."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam",
                        alpha=0.0, batch_size=min(64, len(x)), learning_rate_init=0.01,
                        max_iter=10, shuffle=True, early_stopping=False, n_iter_no_change=10**9,
                        random_state=seed)
    with warnings.catch_warnings():
        # 10 epochs is the benchmark's setting, not an unfinished fit
        warnings.simplefilter("ignore", ConvergenceWarning)
        return clf.fit(np.asarray(x, dtype=float), y)


#: the columns of the frame classification_metrics() returns, in the article's order
METRICS = list(_CLASSIFICATION_METRICS)


def classification_metrics(predictions) -> pd.DataFrame:
    """Score the predicted cell types of ``mtb.classify`` with the benchmark's metrics.

    Parameters
    ----------
    predictions : pandas.DataFrame
        The frame ``mtb.classify`` returns, or any frame with the columns
        ``label`` (true cell type) and ``predicted``; with a ``reference``
        column, each reference batch is scored on its own.

    Returns
    -------
    pandas.DataFrame
        One row per reference batch (index ``reference``, in input order)
        and a last row ``mean``, the mean over the references; a frame
        without ``reference`` gives one row. The columns, in [0, 1]:

        - ``OCA``: overall classification accuracy, the fraction of cells
          predicted correctly.
        - ``ACA``: average classification accuracy, the fraction of the
          cells of a type predicted correctly, averaged over the types.
        - ``Spec``: specificity, the fraction of the cells of other types
          not predicted as a type, averaged over the types.
        - ``Sens``: sensitivity, the fraction of the cells of a type
          predicted as that type, averaged over the types.
        - ``F1``: F1-score, 2 x the cells correctly predicted as a type /
          (its cells + the cells predicted as it), averaged over the types.

    Raises
    ------
    TypeError
        ``predictions`` is not a DataFrame.
    ValueError
        ``predictions`` lacks ``label`` or ``predicted``.

    Examples
    --------
    >>> labels = mtb.labels_for("D52", "cross", "StabMap")
    >>> pred = mtb.classify(res.output, labels=labels)    # res = mtb.run(...)
    >>> mtb.classification_metrics(pred)

    Notes
    -----
    The types are those true or predicted in a reference's query. ``ACA``
    and ``Sens`` average over the types that are true, ``Spec`` and ``F1``
    over all of them, so ``Spec`` is NaN when a query has one true type; the
    ``mean`` row is then NaN too. A row with a missing ``label`` or
    ``predicted`` is left out.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("classification_metrics needs a DataFrame. "
                        "Pass the frame mtb.classify returns.")
    missing = [c for c in ("label", "predicted") if c not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions has no {' and no '.join(missing)} column. "
            f"Pass the frame mtb.classify returns.")
    if "reference" not in predictions.columns:
        return pd.DataFrame([_metrics(predictions)], columns=METRICS)
    codes, refs = pd.factorize(predictions["reference"], use_na_sentinel=False)
    table = pd.DataFrame([_metrics(predictions[codes == i]) for i in range(len(refs))],
                         index=pd.Index(refs, name="reference"), columns=METRICS)
    if len(table):
        table.loc["mean"] = table.mean(skipna=False)
    return table


def _metrics(frame: pd.DataFrame) -> list:
    """The five metrics of one query, from the confusion table of every type
    true or predicted."""
    keep = ~(io.missing_mask(frame["label"].to_numpy(dtype=object))
             | io.missing_mask(frame["predicted"].to_numpy(dtype=object)))
    truth = frame["label"].to_numpy(dtype=object)[keep]
    pred = frame["predicted"].to_numpy(dtype=object)[keep]
    codes, levels = pd.factorize(np.concatenate([truth, pred]))
    k, n = len(levels), len(truth)
    cm = np.bincount(codes[:n] * k + codes[n:], minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        oca = tp.sum() / cm.sum()
        aca = np.diag(cm / cm.sum(axis=1, keepdims=True))     # NaN: never true
        sens = tp / (tp + fn)                                  # NaN: never true
        spec = tn / (tn + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return [float(oca), _mean(aca[~np.isnan(aca)]), _mean(spec),
            _mean(sens[~np.isnan(sens)]), _mean(f1)]


def _mean(x) -> float:
    """``mean(x)`` as in R: NaN for no value or a NaN value."""
    return float(np.mean(x)) if len(x) else float("nan")
