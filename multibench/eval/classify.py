"""The classification metrics of ``mtb.evaluate(metrics="classification")``:
classify() predicts the cell type of each query cell from a reference batch;
classification_metrics() scores those predictions."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .. import _text
from .._data.catalog import FAMILY_METRICS

#: the columns of the frame classify() returns
COLUMNS = ["reference", "batch", "cell", "label", "predicted"]


def classify(emb, labels, batch, seed: int = 0) -> pd.DataFrame:
    """Predict the cell type of each query cell, each batch in turn the reference.

    ``emb`` (cells x dims), ``labels`` and ``batch`` (one per cell) are the
    arrays ``evaluate`` has checked: no missing value, two or more batches.
    A small neural network learns the cell types of the reference batch and
    predicts the cells of the other batches; the same seed and input give the
    same predictions.

    Returns one row per query cell and reference, with the columns
    ``reference``, ``batch``, ``cell`` (the row number), ``label`` and
    ``predicted``. A query cell of a type the reference lacks is left out;
    ``attrs["left_out"]`` counts them per reference. A reference batch with
    one cell type is skipped with a ``UserWarning``.
    """
    emb = np.asarray(emb)
    ct = np.array([str(c) for c in labels], dtype=object)
    ba = np.array([str(b) for b in batch], dtype=object)
    cell = np.arange(len(ct))
    frames = []
    left_out: dict = {}
    for ref in pd.unique(ba):
        train = ba == ref
        types = pd.unique(ct[train])
        if len(types) < 2:
            _text.warn(f"Reference batch {ref} has one cell type; skipped.", UserWarning)
            continue
        query = (ba != ref) & np.isin(ct, types)
        left_out[ref] = int(((ba != ref) & ~np.isin(ct, types)).sum())
        if not query.any():
            _text.warn(f"No other batch has a cell type of reference batch {ref}; "
                       f"skipped.", UserWarning)
            continue
        predicted = _mlp(emb[train], ct[train], seed).predict(emb[query])
        frames.append(pd.DataFrame({
            "reference": ref, "batch": ba[query], "cell": cell[query],
            "label": ct[query], "predicted": np.asarray(predicted, dtype=object)}))
    out = (pd.concat(frames, ignore_index=True)[COLUMNS] if frames
           else pd.DataFrame(columns=COLUMNS))
    out.attrs["left_out"] = left_out
    return out


def _graph_text(who: str) -> str:
    """What is wrong with the output of a graph method: ``who`` is the method,
    or ``"output"`` when the method is not known."""
    who = "output is" if who == "output" else f"{who} writes"
    return f"{who} a graph, not an embedding, so it has no classification scores."


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
    """``{file name: [(variant output kind, method), ...]}`` over every output
    and extra output of the registry."""
    from ..engine import registry
    files: dict = {}
    for spec in registry.load():
        for v in spec.variants:
            for o in [v.output, *v.extra_outputs]:
                files.setdefault(o.file, []).append((v.output.kind, spec.id))
    return files


def _graph_files() -> dict:
    """``{file name: [method, ...]}`` for files that only graph variants write."""
    return {f: sorted({m for _, m in uses}) for f, uses in _registry_files().items()
            if all(vk == "graph" for vk, _ in uses)}


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


#: the columns of the frame classification_metrics() returns, in the paper's order
METRICS = list(FAMILY_METRICS["classification"])


def classification_metrics(predictions) -> pd.DataFrame:
    """Score the frame :func:`classify` returns with the benchmark's metrics.

    Returns one row per reference batch (index ``reference``, in input
    order) and a last row ``mean``, the mean over the references, which is
    what ``mtb.evaluate(metrics="classification")`` reports. The columns, in
    [0, 1]:

    - ``OCA``: overall classification accuracy, the fraction of cells
      predicted correctly.
    - ``ACA``: average classification accuracy, the fraction of the cells of
      a type predicted correctly, averaged over the types.
    - ``Spec``: specificity, the fraction of the cells of other types not
      predicted as a type, averaged over the types.
    - ``Sens``: sensitivity, the fraction of the cells of a type predicted as
      that type, averaged over the types.
    - ``F1``: F1-score, 2 x the cells correctly predicted as a type / (its
      cells + the cells predicted as it), averaged over the types.

    The types are those true or predicted in a reference's query. ``ACA`` and
    ``Sens`` average over the types that are true, ``Spec`` and ``F1`` over all
    of them, so ``Spec`` is NaN when a query has one true type; the ``mean``
    row is then NaN too.
    """
    refs = pd.unique(predictions["reference"])
    table = pd.DataFrame([_metrics(predictions[predictions["reference"] == r]) for r in refs],
                         index=pd.Index(refs, name="reference"), columns=METRICS)
    if len(table):
        table.loc["mean"] = table.mean(skipna=False)
    return table


def _metrics(frame: pd.DataFrame) -> list:
    """The five metrics of one query, from the confusion table of every type
    true or predicted."""
    truth = frame["label"].to_numpy(dtype=object)
    pred = frame["predicted"].to_numpy(dtype=object)
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
