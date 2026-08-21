"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from ..data import catalog

# "DR and clustering" are ONE metric group in the benchmark paper, so a
# dimension_reduction request is served by the clustering metrics.
_SCIB_TASKS = {"clustering", "dimension_reduction", "batch", "all"}


def to_long(value_df, method, dataset, category):
    """Convert an evaluate() wide frame (index=metric, column 'Value') to the
    long frame used by load_results/plot.bubble: metric,value,method,dataset,category."""
    out = value_df.rename(columns={"Value": "value"}).copy()
    out = out.reset_index().rename(columns={"index": "metric"})
    out["metric"] = out["metric"].map(catalog.canonical_metric)
    out["method"] = method
    out["dataset"] = dataset
    out["category"] = category
    return out[["metric", "value", "method", "dataset", "category"]]


def _known_metrics() -> list:
    # the plotting layer owns the metric families; imported lazily so that the
    # eval layer does not pull matplotlib in at import time
    from ..plot.bar import BATCH_METRICS, CLUSTERING_METRICS
    return list(CLUSTERING_METRICS) + list(BATCH_METRICS)


def _validate_only(only):
    if only is None:
        return None
    if isinstance(only, str):
        # set("ARI") == {'A','R','I'} silently matched nothing and returned an
        # empty frame; a bare string is always a mistake here
        raise TypeError(
            f"only= must be a collection of metric names, e.g. only={{{only!r}}}; "
            f"got the string {only!r}")
    only = set(only)
    known = _known_metrics()
    unknown = only - set(known)
    if unknown:
        raise ValueError(
            f"unknown metric(s) {sorted(unknown)}; choose from {known}")
    return only


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


def _obs_or_vector(x, adata, *, what, column=None):
    """A str naming an obs column of ``adata`` -> that column; else as_vector."""
    if adata is not None and isinstance(x, str):
        if x in adata.obs.columns:
            return np.asarray(pd.Series(adata.obs[x]).to_numpy())
        if not Path(x).is_file():
            raise ValueError(
                f"{what}={x!r} is neither an obs column of the AnnData (obs "
                f"columns: {list(map(str, adata.obs.columns))}) nor an existing "
                f"file")
    return io.as_vector(x, what=what, column=column)


def evaluate(
    output,
    category: str | None = None,
    task: str = "clustering",
    labels=None,
    clustering=None,
    batch=None,
    metric_set: str = "scib",
    slow_metrics: bool = False,
    only=None,
    *,
    obsm: str = "X_emb",
    column: str | None = None,
) -> pd.DataFrame:
    """Compute scIB metrics for a run output (an embedding) against cell-type labels.

    Returns a ``metric.csv``-shaped DataFrame (index = metric name, one column
    ``Value``); reshape with :func:`multibench.to_long` for plotting.

    Parameters
    ----------
    output
        The embedding, (cells x dims). Any of: ``numpy.ndarray`` (dims x cells
        is auto-transposed against the label count, with a warning),
        ``pandas.DataFrame``, ``AnnData`` (uses ``.obsm[obsm]``), a scipy sparse
        matrix, or a path - ``.h5`` (dataset ``data``, the benchmark's
        ``embedding.h5``), ``.h5ad``, ``.npy``, ``.csv``/``.tsv``.
    category : str, optional
        One of :func:`multibench.list_categories` (``vertical``, ``diagonal``,
        ``mosaic``, ``cross``). Validated when given; metrics do not depend on
        it in v1, so it may be omitted.
    task : str
        ``'clustering'`` (default; also serves ``'dimension_reduction'`` - one
        metric group in the paper), ``'batch'``, or ``'all'``.
    labels
        Ground-truth cell types, one per cell, in the row order of ``output``.
        Any of: a CSV path (header row; column ``x`` / the only column / the
        last of two with a barcode index - see :func:`multibench.eval.io.read_labels`),
        a LIST of CSV paths concatenated in that order (multi-batch datasets:
        ``[cty1, cty2, cty3]``), a one-entry dict as returned by
        :func:`multibench.labels_for`, a 1-D ``ndarray``/``Series``/
        ``Categorical``/list, a single-column DataFrame, or - when ``output``
        is an AnnData - the name of an ``obs`` column.
    clustering
        Optional precomputed cluster assignment (same forms as ``labels``; an
        ``.h5`` path is read from ``/obs/cluster_leiden``). When omitted the
        scIB optimal-resolution Leiden sweep derives one from the embedding.
    batch
        Batch labels, one per cell (same forms as ``labels``; obs column name
        for AnnData). Required for ``task='batch'``/``'all'`` EXCEPT when
        ``labels`` is a list of two or more files, in which case the file of
        origin (1, 2, ...) is used as the batch - the same rule
        :func:`multibench.run_all` applies.
    metric_set : str
        Only ``'scib'`` is wired in v1.
    slow_metrics : bool
        Also compute kBET (shells out to R; hours on large datasets).
    only
        Collection of metric names to compute (e.g. ``{'ARI', 'NMI'}``);
        everything else is skipped, including the Leiden sweep when no
        requested metric needs it. Unknown names raise; a bare string raises.
        Valid names: ``mtb.plot.CLUSTERING_METRICS + mtb.plot.BATCH_METRICS``.
    obsm : str
        ``.obsm`` key to use when ``output`` is an AnnData / ``.h5ad``
        (default ``'X_emb'``; ``'X'`` means ``.X``).
    column : str, optional
        Column to take when ``labels`` is a CSV/DataFrame with several columns.

    Raises
    ------
    ValueError
        missing labels/batch, unknown category or metric name, length
        mismatches (``'input length mismatch: emb has N cells, celltype has M'``),
        ambiguous label files.
    TypeError
        unsupported input types (``only='ARI'``, non-array ``labels``, ...).
    """
    if metric_set != "scib" or task not in _SCIB_TASKS:
        raise NotImplementedError(
            f"evaluate(task={task!r}, metric_set={metric_set!r}) is not wired in v1; "
            f"only scib clustering/batch are supported."
        )
    if labels is None:
        raise ValueError("metrics require `labels` (cty / ground-truth cell types).")
    # `clustering` is optional: when omitted it is derived from the embedding
    # inside scib.compute() via optimal-resolution Leiden.
    batch_from_files = batch is None and _is_label_path_list(labels) and len(labels) > 1
    if task in {"batch", "all"} and batch is None and not batch_from_files:
        raise ValueError("batch labels required for batch/all metrics")
    _validate_category(category)
    only = _validate_only(only)

    adata = output if io._is_anndata(output) else None
    emb = np.asarray(io.as_matrix(output, obsm=obsm))
    if _is_label_path_list(labels):
        # several label files: concatenate IN THE GIVEN ORDER; remember the
        # sizes so the file of origin can serve as the batch below
        parts = [np.asarray(io.read_labels(p, column=column)) for p in labels]
        ct = np.concatenate(parts)
    else:
        parts = None
        ct = _obs_or_vector(labels, adata, what="labels", column=column)
    # Orient to cells x dims. read_embedding() already does this for h5 inputs;
    # do the same for everything else so a caller can pass run().output (dims x
    # cells for many methods) directly. Use the label count as the truth: only
    # transpose when it resolves the cell-axis mismatch (never ambiguously).
    if emb.ndim == 2 and emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        warnings.warn(
            f"output was (dims x cells) {emb.shape}; transposed to (cells x dims)")
        emb = emb.T
    if clustering is None:
        cl = None
    elif isinstance(clustering, (str, Path)) and not io._is_label_file(clustering):
        cl = io.read_clustering(clustering)        # the benchmark's h5 layout
    else:
        cl = _obs_or_vector(clustering, adata, what="clustering")
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch")
    elif batch_from_files and task in {"batch", "all"}:
        # batch = which label FILE each cell came from (1-based, as run_all)
        ba = np.concatenate([np.full(len(v), i + 1) for i, v in enumerate(parts)])
    else:
        # clustering metrics need a batch_key but it is a no-op there, so a
        # constant vector is acceptable when no batch labels are supplied.
        ba = np.zeros(emb.shape[0], dtype=int)

    from . import scib as escib  # imported lazily so non-scib paths don't require it
    # "DR and clustering" are one metric group in the benchmark paper,
    # so dimension_reduction is evaluated with the clustering metrics.
    group = "clustering" if task == "dimension_reduction" else task
    return escib.compute(emb, ct, cl, ba, group=group,
                         slow_metrics=slow_metrics, only=only)
