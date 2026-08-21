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
    long frame used by load_results/plot.bubble: metric,value,method,dataset,category.

    Metric names are canonicalised (``ari`` -> ``ARI``, ``kbet`` -> ``kBET``);
    rows whose name canonicalises to nothing (blank) are dropped, and two rows
    that collapse onto the SAME canonical name (``ari`` and ``ARI`` both
    present) raise ``ValueError`` - a silent duplicate would double-count that
    metric in every downstream rank.
    """
    out = value_df.rename(columns={"Value": "value"}).copy()
    out = out.reset_index()                      # the index column comes first,
    out = out.rename(columns={out.columns[0]: "metric"})   # whatever it was named
    out["metric"] = out["metric"].map(catalog.canonical_metric)
    out = out.dropna(subset=["metric"])
    dup = out["metric"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"metric names collide after canonicalisation: "
            f"{sorted(set(out.loc[dup, 'metric']))} - the input names two rows that "
            f"map to the same canonical metric (e.g. 'ari' and 'ARI'); drop one")
    out["method"] = method
    out["dataset"] = dataset
    out["category"] = category
    return out[["metric", "value", "method", "dataset", "category"]]


def _metric_families() -> tuple[list, list]:
    """``(CLUSTERING_METRICS, BATCH_METRICS)`` - the two scIB families evaluate()
    can produce, in the canonical spelling ``load_results``/``to_long`` use."""
    # the plotting layer owns the metric families; imported lazily so that the
    # eval layer does not pull matplotlib in at import time
    from ..plot.bar import BATCH_METRICS, CLUSTERING_METRICS
    return list(CLUSTERING_METRICS), list(BATCH_METRICS)


def _known_metrics() -> list:
    clu, bat = _metric_families()
    return clu + bat


def _validate_only(only, task: str = "all", slow_metrics: bool = True):
    """Normalise ``only=`` to a set and refuse a request that could select nothing.

    ``evaluate(task='clustering', only={'GC'})`` used to run the whole
    clustering pipeline (Leiden sweep included) and then return an EMPTY
    (0, 1) frame - every requested name was filtered out silently because GC
    is a batch metric. A request that the chosen ``task`` cannot serve is an
    error here, before anything is computed.

    Parameters
    ----------
    only
        ``None`` (everything), or a collection of metric names.
    task
        The ``task`` evaluate() was called with; ``'dimension_reduction'`` is
        the clustering family.
    slow_metrics
        Whether kBET will be computed; ``only={'kBET'}`` without it would
        select nothing.

    Returns
    -------
    set or None
        The validated set of names, or ``None`` when ``only`` was ``None``.

    Raises
    ------
    TypeError
        ``only`` is a bare string (``set('ARI') == {'A', 'R', 'I'}``).
    ValueError
        an unknown name (lists the valid ones); a batch metric under
        ``task='clustering'``; a clustering metric under ``task='batch'``;
        kBET without ``slow_metrics=True``.
    """
    if only is None:
        return None
    if isinstance(only, str):
        # set("ARI") == {'A','R','I'} silently matched nothing and returned an
        # empty frame; a bare string is always a mistake here
        raise TypeError(
            f"only= must be a collection of metric names, e.g. only={{{only!r}}}; "
            f"got the string {only!r}")
    only = set(only)
    clu, bat = _metric_families()
    known = clu + bat
    unknown = only - set(known)
    if unknown:
        raise ValueError(
            f"unknown metric(s) {sorted(unknown)}; choose from {known}")
    group = "clustering" if task == "dimension_reduction" else task
    if group == "clustering":
        bad = [m for m in bat if m in only]
        if bad:
            verb = "is a batch metric" if len(bad) == 1 else "are batch metrics"
            raise ValueError(
                f"{', '.join(bad)} {verb}: pass task=\"all\" (or \"batch\") and "
                f"batch=<vector> (task={task!r} computes only {clu})")
    elif group == "batch":
        bad = [m for m in clu if m in only]
        if bad:
            verb = "is a clustering metric" if len(bad) == 1 else "are clustering metrics"
            raise ValueError(
                f"{', '.join(bad)} {verb}: pass task=\"all\" (or \"clustering\") "
                f"(task='batch' computes only {bat})")
    if "kBET" in only and not slow_metrics:
        raise ValueError(
            "kBET is computed only with slow_metrics=True (it shells out to R and "
            "takes hours on large datasets); pass slow_metrics=True or drop it "
            "from only=")
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


def _cell_ids(output):
    """Cell ids carried by ``output``, or ``None`` when it has none.

    An AnnData carries ``obs_names``; a DataFrame carries its index unless that
    index is a plain ``RangeIndex`` (the default, i.e. no ids). Everything else
    (ndarray, sparse matrix, file path) has no ids, so labels can only be
    matched positionally.
    """
    if io._is_anndata(output):
        return pd.Index(output.obs_names)
    if isinstance(output, pd.DataFrame) and not isinstance(output.index, pd.RangeIndex):
        return output.index
    return None


def _carries_ids(x) -> bool:
    """A Series / DataFrame whose index is not a ``RangeIndex`` carries cell ids."""
    return (isinstance(x, (pd.Series, pd.DataFrame))
            and not isinstance(x.index, pd.RangeIndex))


def _obs_or_vector(x, adata, *, what, column=None, ids=None):
    """Coerce one label-like argument to a 1-D array in the output's cell order.

    * a ``str`` naming an obs column of ``adata`` -> that column (already in
      the output's order);
    * a ``Series``/``DataFrame`` with a non-default index, when the output
      carries cell ids (``ids`` given) -> ALIGNED by id via
      :func:`multibench.eval.io.align_vector` (raises on missing/extra ids);
    * a ``Series``/``DataFrame`` with a non-default index, when the output is
      a bare array -> positional, with a ``UserWarning`` saying so;
    * anything else -> :func:`multibench.eval.io.as_vector` (positional).
    """
    if adata is not None and isinstance(x, str):
        if x in adata.obs.columns:
            return np.asarray(pd.Series(adata.obs[x]).to_numpy())
        if not Path(x).is_file():
            raise ValueError(
                f"{what}={x!r} is neither an obs column of the AnnData (obs "
                f"columns: {list(map(str, adata.obs.columns))}) nor an existing "
                f"file")
    if _carries_ids(x):
        if ids is not None:
            return io.align_vector(x, ids, what=what, column=column)
        warnings.warn(
            f"{what} {type(x).__name__} has a non-default index; matched "
            f"positionally because the embedding carries no cell ids - pass "
            f"{what}.to_numpy() to silence, or an AnnData/DataFrame with cell "
            f"ids to align", UserWarning, stacklevel=3)
    return io.as_vector(x, what=what, column=column)


def _labels_from_dict(d: dict, label_order) -> list:
    """Turn a ``{name: path}`` label dict into the list of paths in cell order.

    One entry needs no order. Several entries need ``label_order`` (keys of
    ``d``; a subset selects those files) - a dict has no cell order of its
    own, and the order is the method's stacking order, so guessing here is
    exactly the silent-wrong-score the user study hit.
    """
    if label_order is None:
        if len(d) == 1:
            return [next(iter(d.values()))]
        # Accept the dict when its insertion order IS the benchmark's stacking
        # order (what labels_for returns); any other order must be explicit,
        # because guessing here is exactly the silent wrong score the user
        # study hit.
        from ..engine.resolve import _label_sort_key
        keys = list(d)
        if keys == sorted(keys, key=_label_sort_key):
            return [d[k] for k in keys]
        raise ValueError(
            f"labels: got a dict with {len(d)} label files {keys} whose order is "
            f"not the method's stacking order (cty1, cty2, ... numerically; rna "
            f"before adt before atac - NOT alphabetical); pass a list of paths "
            f"in cell order, or label_order=[...] naming the keys in that order, "
            f"or mtb.labels_for(dataset, method=<method>, category=<category>) "
            f"which returns them already in stacking order")
    if isinstance(label_order, str) or not isinstance(label_order, (list, tuple)):
        raise TypeError(
            f"label_order= must be a list of keys of the labels dict, e.g. "
            f"label_order={list(map(str, d))!r}; got {type(label_order).__name__}")
    order = list(label_order)
    if not order:
        raise ValueError(
            f"label_order= is empty; list the keys of the labels dict in the "
            f"method's stacking order, e.g. {list(map(str, d))!r}")
    unknown = [k for k in order if k not in d]
    if unknown:
        raise ValueError(
            f"label_order names key(s) {unknown!r} that are not in the labels "
            f"dict; its keys are {list(map(str, d))!r}")
    if len(set(order)) != len(order):
        raise ValueError(f"label_order repeats a key: {order!r}")
    return [d[k] for k in order]


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
    label_order=None,
) -> pd.DataFrame:
    """Compute scIB metrics for a run output (an embedding) against cell-type labels.

    Returns a ``metric.csv``-shaped DataFrame (index = metric name in the
    canonical spelling ``load_results`` uses - ``ARI, NMI, ASW, iASW, iF1,
    cLISI, ASW_batch, GC, iLISI, kBET`` - one column ``Value``); reshape with
    :func:`multibench.to_long` for plotting. The frame is never empty: a
    request that would select no metric raises instead (see ``only``).

    Parameters
    ----------
    output
        The embedding, (cells x dims). Any of: ``numpy.ndarray`` (dims x cells
        is auto-transposed against the label count, with a warning),
        ``pandas.DataFrame``, ``AnnData`` (uses ``.obsm[obsm]``), a scipy sparse
        matrix, or a path - ``.h5`` (dataset ``data``, the benchmark's
        ``embedding.h5``), ``.h5ad`` (read as an AnnData), ``.npy``,
        ``.csv``/``.tsv``. An AnnData (``obs_names``) or a DataFrame with a
        non-default index CARRIES CELL IDS, which ``labels``/``batch``/
        ``clustering`` given as an indexed Series are aligned against (below).
    category : str, optional
        One of :func:`multibench.list_categories` (``vertical``, ``diagonal``,
        ``mosaic``, ``cross``). Validated when given; metrics do not depend on
        it in v1, so it may be omitted.
    task : str
        ``'clustering'`` (default; also serves ``'dimension_reduction'`` - one
        metric group in the paper) computes ``ARI, NMI, ASW, iASW, iF1,
        cLISI``; ``'batch'`` computes ``ASW_batch, GC, iLISI`` (+ ``kBET``
        with ``slow_metrics``); ``'all'`` both. A ``batch=`` given under
        ``task='clustering'`` changes nothing and triggers a ``UserWarning``
        pointing at ``task='all'``.
    labels
        Ground-truth cell types, one per cell. Any of: a CSV path (header row;
        column ``x`` / the only column / the last of two with a barcode index -
        see :func:`multibench.eval.io.read_labels`), a LIST of CSV paths
        concatenated in that order (multi-batch datasets: ``[cty1, cty2,
        cty3]``), a dict as returned by :func:`multibench.labels_for` (one
        entry plugs in directly; several entries need ``label_order=``), a 1-D
        ``ndarray``/``Series``/``Categorical``/list, a single-column DataFrame,
        or - when ``output`` is an AnnData - the name of an ``obs`` column.

        ORDER. Arrays/lists/files are matched POSITIONALLY to the rows of
        ``output``. A ``Series``/``DataFrame`` with a non-default index is
        aligned BY CELL ID when ``output`` carries ids (AnnData / DataFrame
        with a non-default index): rows are reindexed to the output's order,
        and a missing or extra id raises ``ValueError`` naming the first ones.
        When ``output`` is a bare array there is nothing to align against: the
        Series is matched positionally and a ``UserWarning`` says so (pass
        ``labels.to_numpy()`` to silence it).
    clustering
        Optional precomputed cluster assignment (same forms and the same
        alignment rule as ``labels``; an ``.h5`` path is read from
        ``/obs/cluster_leiden``). When omitted the scIB optimal-resolution
        Leiden sweep derives one from the embedding.
    batch
        Batch labels, one per cell (same forms and the same alignment rule as
        ``labels``; obs column name for AnnData). Required for
        ``task='batch'``/``'all'`` EXCEPT when ``labels`` is a list of two or
        more files (or a dict with ``label_order``), in which case the file of
        origin (1, 2, ...) is used as the batch - the same rule
        :func:`multibench.run_all` applies.
    metric_set : str
        Only ``'scib'`` is wired in v1.
    slow_metrics : bool
        Also compute kBET (shells out to R; hours on large datasets).
    only
        Collection of metric names to compute (e.g. ``{'ARI', 'NMI'}``);
        everything else is skipped, including the Leiden sweep when no
        requested metric needs it. Validated against what ``task`` can
        produce: an unknown name raises listing the valid ones; a batch metric
        under ``task='clustering'`` raises (``'GC is a batch metric: pass
        task="all" (or "batch") and batch=<vector>'``); a clustering metric
        under ``task='batch'`` raises; ``kBET`` without ``slow_metrics=True``
        raises; a bare string raises ``TypeError``. Valid names:
        ``mtb.plot.CLUSTERING_METRICS + mtb.plot.BATCH_METRICS``.
    obsm : str
        ``.obsm`` key to use when ``output`` is an AnnData / ``.h5ad``
        (default ``'X_emb'``; ``'X'`` means ``.X``).
    column : str, optional
        Column to take when ``labels`` is a CSV/DataFrame with several columns.
    label_order : list of str, keyword-only
        Only for a ``labels`` dict with several entries: the dict keys in the
        order the method stacked the cells (the benchmark's convention is
        numbered files ascending - ``cty1, cty2, ...`` - and ``rna`` before
        ``atac``; :func:`multibench.labels_for` ``(ds, method=, category=)``
        returns the files in that order, so ``label_order=list(d)`` trusts
        it). A subset of the keys selects those files only. Unknown or
        repeated keys raise; ``label_order`` with a non-dict ``labels`` raises
        ``TypeError``.

    Raises
    ------
    ValueError
        missing labels/batch, unknown category or metric name, a metric the
        task cannot produce, length mismatches (``'input length mismatch: emb
        has N cells, celltype has M'``), cell-id mismatches when aligning,
        ambiguous label files, a multi-entry labels dict without
        ``label_order``.
    TypeError
        unsupported input types (``only='ARI'``, non-array ``labels``,
        ``label_order`` with non-dict labels, ...).
    """
    if metric_set != "scib" or task not in _SCIB_TASKS:
        raise NotImplementedError(
            f"evaluate(task={task!r}, metric_set={metric_set!r}) is not wired in v1; "
            f"only scib clustering/batch are supported."
        )
    if labels is None:
        raise ValueError("metrics require `labels` (cty / ground-truth cell types).")
    # a labels_for() dict becomes the list of paths in cell order FIRST, so the
    # file-of-origin batch rule below sees it like any other list of files
    if isinstance(labels, dict):
        labels = _labels_from_dict(labels, label_order)
    elif label_order is not None:
        raise TypeError(
            f"label_order= applies only when labels is a dict ({{name: path}}, "
            f"as returned by mtb.labels_for); labels is a "
            f"{type(labels).__name__} - pass the files as a list in that order "
            f"instead")
    # `clustering` is optional: when omitted it is derived from the embedding
    # inside scib.compute() via optimal-resolution Leiden.
    batch_from_files = batch is None and _is_label_path_list(labels) and len(labels) > 1
    if task in {"batch", "all"} and batch is None and not batch_from_files:
        raise ValueError("batch labels required for batch/all metrics")
    if task not in {"batch", "all"} and batch is not None:
        warnings.warn(
            f"batch= was given but task={task!r} computes no batch metric "
            f"({_metric_families()[1]}); pass task='all' (or 'batch') to "
            f"compute them - batch changes nothing under task={task!r}",
            UserWarning, stacklevel=2)
    _validate_category(category)
    only = _validate_only(only, task=task, slow_metrics=slow_metrics)

    if isinstance(output, (str, Path)) and Path(output).suffix.lower() == ".h5ad":
        import anndata as ad      # read ONCE; keeps obs_names and obs columns
        output = ad.read_h5ad(output)
    adata = output if io._is_anndata(output) else None
    ids = _cell_ids(output)
    emb = np.asarray(io.as_matrix(output, obsm=obsm))
    if _is_label_path_list(labels):
        # several label files: concatenate IN THE GIVEN ORDER; remember the
        # sizes so the file of origin can serve as the batch below
        parts = [np.asarray(io.read_labels(p, column=column)) for p in labels]
        ct = np.concatenate(parts)
    else:
        parts = None
        ct = _obs_or_vector(labels, adata, what="labels", column=column, ids=ids)
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
    elif isinstance(clustering, (str, Path)) and not io._is_label_file(clustering) \
            and not (adata is not None and isinstance(clustering, str)
                     and clustering in adata.obs.columns):
        cl = io.read_clustering(clustering)        # the benchmark's h5 layout
    else:
        cl = _obs_or_vector(clustering, adata, what="clustering", ids=ids)
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ids)
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
    out = escib.compute(emb, ct, cl, ba, group=group,
                        slow_metrics=slow_metrics, only=only)
    # The names compute() emits ARE the canonical ones, but make that a
    # property of evaluate() rather than a coincidence: a frame that reaches
    # to_long()/pd.concat with the published tables must never carry a second
    # spelling of the same metric.
    out.index = pd.Index([catalog.canonical_metric(m) for m in out.index],
                         name=out.index.name)
    if out.empty:
        # unreachable after _validate_only; kept so a future metric family
        # mismatch fails loudly instead of handing back a (0, 1) frame
        raise ValueError(
            f"evaluate(task={task!r}, only={sorted(only) if only else None}) "
            f"selected no metric - nothing to return")
    return out
