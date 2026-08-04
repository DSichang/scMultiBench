"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

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


def evaluate(
    output,
    category: str,
    task: str = "clustering",
    labels=None,
    clustering=None,
    batch=None,
    metric_set: str = "scib",
) -> pd.DataFrame:
    """Compute metrics for a run output. v1 supports metric_set='scib' only.

    Note: the ``category`` parameter is currently reserved. It is accepted for
    API stability / future per-category metric dispatch but is not yet used in
    v1.
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
    if task in {"batch", "all"} and batch is None:
        raise ValueError("batch labels required for batch/all metrics")

    emb = io.read_embedding(output) if not isinstance(output, np.ndarray) else output
    ct = io.read_labels(labels) if not isinstance(labels, np.ndarray) else labels
    # Orient a raw ndarray to cells x dims. read_embedding() already does this for
    # file inputs; do the same for an ndarray so a caller can pass run().output
    # (dims x cells for many methods) directly. Use the label count as the truth:
    # only transpose when it resolves the cell-axis mismatch (never ambiguously).
    emb = np.asarray(emb)
    if emb.ndim == 2 and emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        emb = emb.T
    cl = (
        None if clustering is None
        else clustering if isinstance(clustering, np.ndarray)
        else io.read_clustering(clustering)
    )
    # clustering metrics need a batch_key but it is a no-op there, so a constant
    # vector is acceptable when no batch labels are supplied (task="clustering").
    ba = np.zeros(emb.shape[0], dtype=int) if batch is None else (
        io.read_labels(batch) if not isinstance(batch, np.ndarray) else batch
    )

    from . import scib as escib  # imported lazily so non-scib paths don't require it
    # "DR and clustering" are one metric group in the benchmark paper,
    # so dimension_reduction is evaluated with the clustering metrics.
    group = "clustering" if task == "dimension_reduction" else task
    return escib.compute(emb, ct, cl, ba, group=group)
