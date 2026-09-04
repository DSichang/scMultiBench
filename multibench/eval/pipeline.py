"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, _compat
from ..data import catalog

#: the seven columns of the tidy long frame (pinned to
#: multibench.data.results.COLUMNS by tests/test_eval_reshape.py)
LONG_COLUMNS = ["metric", "value", "method", "dataset", "category", "clustering", "source"]


def to_long(value_df, *, method: str, dataset: str | None = None,
            category: str | None = None, clustering: str = "default",
            source: str = "user") -> pd.DataFrame:
    """Reshape :func:`evaluate`'s wide frame into the tidy long results frame.

    The long frame has the same seven columns :func:`multibench.load_results`
    returns - ``metric, value, method, dataset, category, clustering, source``
    - so ``pd.concat([mtb.load_results(...), to_long(...)])`` -> ``to_csv`` ->
    ``load_results(result_path=<file>)`` keeps every row's provenance
    (``source="user"`` selects your rows again).

    Parameters
    ----------
    value_df : pandas.DataFrame or pandas.Series
        What :func:`evaluate` returns: metric names as the index, one column
        ``Value``. Also accepted: a Series indexed by metric, and the CSV
        read-back of that frame (``pd.read_csv(out)`` with a ``metric``
        column and a ``Value`` column). Metric names are canonicalised
        (``ari`` -> ``ARI``, ``kbet`` -> ``kBET``); rows whose name is blank
        are dropped.
    method : str, keyword-only
        Method id written into every row (your own name is fine; the bubble
        figure shows ``?`` for a name the registry does not know).
    dataset : str, keyword-only, optional
        Dataset id written into every row. ``None`` (default) writes
        ``"all"`` - the placeholder the plotting layer uses for a frame
        without datasets - so the column is never blank.
    category : str, keyword-only, optional
        Integration category written into every row (``"vertical"``,
        ``"diagonal"``, ``"mosaic"``, ``"cross"``). ``None`` (default) writes
        ``"user"``, the value ``load_results`` gives a user file without a
        category column.
    clustering : str, keyword-only
        Value of the ``clustering`` column (default ``"default"``; the
        published tables use ``"louvain"`` / ``"kmeans"`` for their variants).
    source : str, keyword-only
        Value of the ``source`` column (default ``"user"``, which is what
        ``load_results(result_path=file, source="user")`` filters on).

    Returns
    -------
    pandas.DataFrame
        Exactly the seven columns ``metric, value, method, dataset,
        category, clustering, source``. (The 0.2.x ``needs_labels=`` keyword
        was removed; to badge a method of your own as supervised add a
        boolean ``needs_labels`` column to the frame yourself.)

    Raises
    ------
    ValueError
        ``value_df`` is already a long frame (columns ``metric, value,
        method``: pass it to the plot / ``load_results`` consumers directly);
        it has no ``Value`` column (the message names the expected shape and,
        for a wide one-row frame, the ``df.T.set_axis(['Value'], axis=1)``
        fix); no metric name canonicalises to anything; or two names collapse
        onto the SAME canonical metric (``ari`` and ``ARI`` both present) - a
        silent duplicate would double-count that metric in every downstream
        rank.

    Examples
    --------
    >>> wide = mtb.evaluate(emb, labels=labels, metrics=["ARI", "NMI"])
    >>> mine = mtb.to_long(wide, method="MyMethod", dataset="D11", category="vertical")
    >>> pd.concat([mtb.load_results("vertical", dataset="D11", source="rerun"), mine]).to_csv("all.csv", index=False)
    >>> mtb.load_results(result_path="all.csv", source="user")      # your rows only
    """
    if isinstance(value_df, pd.Series):
        value_df = value_df.to_frame("Value")
    cols = [str(c) for c in getattr(value_df, "columns", [])]
    if {"metric", "value", "method"} <= set(cols):
        raise ValueError(
            "to_long() got an already long frame (columns metric, value, method"
            f"{', ...' if len(cols) > 3 else ''}); pass it to mtb.plot.bubble / "
            "pd.concat / load_results consumers directly - to_long reshapes "
            "evaluate()'s WIDE frame (metrics as the index, one column 'Value')")
    if "Value" not in cols:
        idx = list(map(str, list(value_df.index)[:5]))
        raise ValueError(
            f"to_long expects evaluate()'s frame: metrics as index, one column "
            f"'Value'; got columns {cols} (index {idx}) - for a wide one-row "
            f"frame use df.T.set_axis(['Value'], axis=1)")
    if "metric" in cols:
        # the CSV read-back of evaluate's frame (pd.read_csv(out)): the metric
        # names are a column, not the index - reset_index below would
        # otherwise prepend the RangeIndex as a SECOND 'metric' column
        value_df = value_df.set_index("metric")
    out = value_df.rename(columns={"Value": "value"}).copy()
    out = out.reset_index()                      # the index column comes first,
    out = out.rename(columns={out.columns[0]: "metric"})   # whatever it was named
    out["metric"] = out["metric"].map(catalog.canonical_metric)
    out = out.dropna(subset=["metric"])
    if out.empty and len(value_df):
        raise ValueError(
            f"no metric name in the index canonicalises to a known code: "
            f"{list(map(str, value_df.index))[:10]} - to_long expects "
            f"evaluate()'s frame (metrics as index, one column 'Value')")
    dup = out["metric"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"metric names collide after canonicalisation: "
            f"{sorted(set(out.loc[dup, 'metric']))} - the input names two rows that "
            f"map to the same canonical metric (e.g. 'ari' and 'ARI'); drop one")
    out["method"] = method
    out["dataset"] = "all" if dataset is None else dataset
    out["category"] = "user" if category is None else category
    out["clustering"] = clustering
    out["source"] = source
    return out[list(LONG_COLUMNS)].reset_index(drop=True)


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


def _plan_metrics(metrics, *, has_batch: bool, batch_given: bool):
    """Turn the ``metrics=`` knob into what :func:`multibench.eval.scib.compute` needs.

    Parameters
    ----------
    metrics
        The ``metrics`` argument of :func:`evaluate` (``None`` / family token /
        list of codes).
    has_batch : bool
        Whether a batch vector is available (given, or derived from a list of
        label files), i.e. whether batch metrics are computable.
    batch_given : bool
        Whether ``batch=`` itself was passed (drives the "batch changes
        nothing" warning).

    Returns
    -------
    tuple
        ``(group, only, slow)`` - the compute group (``"clustering"`` /
        ``"batch"`` / ``"all"``), the set of codes to compute (``None`` =
        every metric of the group) and whether kBET is among them.

    Raises
    ------
    ValueError
        A family token / code that needs batch labels when none are
        available, or a code evaluate() cannot compute (``PCR``).
    """
    clu, bat = _metric_families()
    sel = catalog.metric_selection(metrics)
    if sel.explicit:
        outside = [c for c in sel.codes if c not in clu and c not in bat]
        if outside:
            raise ValueError(
                f"evaluate() cannot compute {outside}; it computes the scIB "
                f"families only: {clu + bat}")
        want_bat = [c for c in sel.codes if c in bat]
        want_clu = [c for c in sel.codes if c in clu]
        if want_bat and not has_batch:
            raise ValueError(
                f"batch labels required for batch metric(s) {want_bat}: pass "
                f"batch=<vector> (or labels as a list of two or more files, whose "
                f"file of origin then serves as the batch)")
        group = "all" if (want_bat and want_clu) else ("batch" if want_bat else "clustering")
        only, slow = set(sel.codes), "kBET" in sel.codes
    elif sel.family == "all":
        if metrics is None:
            # every applicable metric: the batch family joins in when a batch
            # vector exists to score it against
            group = "all" if has_batch else "clustering"
        else:
            if not has_batch:
                raise ValueError(
                    "batch labels required for metrics='all' (batch family "
                    f"{bat}): pass batch=<vector>, or metrics='clustering'")
            group = "all"
        only, slow = None, False
    elif sel.family == "batch":
        if not has_batch:
            raise ValueError(
                f"batch labels required for metrics='batch' ({bat}): pass "
                f"batch=<vector> (or labels as a list of two or more files)")
        group, only, slow = "batch", None, False
    else:
        group, only, slow = "clustering", None, False
    if group == "clustering" and batch_given:
        warnings.warn(
            f"batch= was given but metrics={metrics!r} computes no batch metric "
            f"({bat}); pass metrics='all' (or 'batch', or name a batch metric in "
            f"the list) to compute them - batch changes nothing here",
            UserWarning, stacklevel=4)
    return group, only, slow


def _legacy_evaluate_kwargs(kw: dict) -> dict:
    """0.2.x spellings of :func:`evaluate`'s keywords: map or refuse.

    ``task=`` / ``family=`` become ``metrics=<token>`` and ``only=[...]``
    becomes ``metrics=[...]`` (each with a ``DeprecationWarning``);
    ``slow_metrics``, ``column`` and ``metric_set`` are gone and raise
    ``TypeError`` naming the replacement.
    """
    removed = {
        "slow_metrics": "pass metrics=[...] without cLISI/iLISI (kBET is computed "
                        "only when it is named in that list)",
        "column": "pass the Series/column itself as labels= / batch= / clustering=",
        "metric_set": "only the scIB metric set exists - drop the argument",
    }
    for name, fix in removed.items():
        if name in kw:
            raise TypeError(f"evaluate() got {name}=, removed in 0.3.0: {fix}")
    legacy = {n: kw.pop(n) for n in ("task", "family", "only") if n in kw}
    if not legacy:
        return kw
    if "metrics" in kw:
        raise TypeError(
            f"evaluate() got metrics= together with the deprecated "
            f"{sorted(legacy)}; pass metrics= only")
    token = legacy.get("family", legacy.get("task"))
    if "family" in legacy and "task" in legacy and legacy["family"] != legacy["task"]:
        token = legacy["family"]          # family won over task in 0.2.x
    if token == "dimension_reduction":
        token = "clustering"              # one metric group in the paper
    for name in ("task", "family"):
        if name in legacy:
            _compat.warn(f"evaluate({name}=...)", f"metrics={token!r}", stacklevel=4)
    if "only" in legacy:
        only = legacy["only"]
        if isinstance(only, str):
            raise TypeError(
                f"only= must be a collection of metric names, e.g. only={{{only!r}}}; "
                f"got the string {only!r} (and only= is deprecated: pass metrics=[...])")
        codes = [catalog.canonical_metric(m) or m for m in only]
        if isinstance(only, (set, frozenset)):
            codes = sorted(codes)          # a set has no order to preserve
        _compat.warn("evaluate(only=...)", f"metrics={codes!r}", stacklevel=4)
        if token not in (None, "all"):
            clu, bat = _metric_families()
            fam = clu if token == "clustering" else bat
            bad = [c for c in codes if c in (clu + bat) and c not in fam]
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} not in the {token!r} family: pass "
                    f"metrics={codes!r} alone (the family token is deprecated)")
        kw["metrics"] = codes
    else:
        kw["metrics"] = token
    return kw


@_compat.legacy_kwargs(_legacy_evaluate_kwargs)
def evaluate(
    output,
    labels=None,
    *,
    category: str | None = None,
    batch=None,
    metrics=None,
    clustering=None,
    obsm: str = "X_emb",
    label_order=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute scIB metrics for a run output (an embedding) against cell-type labels.

    Returns a ``metric.csv``-shaped DataFrame (index = metric name in the
    canonical spelling ``load_results`` uses - ``ARI, NMI, ASW, iASW, iF1,
    cLISI, ASW_batch, GC, iLISI, kBET`` - one column ``Value``); reshape with
    :func:`multibench.to_long` for plotting. The frame is never empty: a
    request that would select no metric raises instead.

    COST. ``ARI``, ``NMI`` and ``iF1`` need the scIB optimal-resolution
    Leiden sweep (10 resolutions on a kNN graph of the embedding): tens of
    seconds for a few thousand cells, minutes for ~10^4. Pass ``clustering=``
    (a precomputed assignment) or ``metrics=[...]`` naming metrics that do
    not need it (``ASW``, ``iASW``, ``cLISI``, the batch family) to skip it.
    One line on stderr announces the sweep when it starts on more than 2,000
    cells (``verbose``).

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
    labels
        Ground-truth cell types, one per cell. Any of: a CSV path (header row;
        column ``x`` / the only column / the last of two with a barcode index -
        see :func:`multibench.eval.io.read_labels`), a LIST of CSV paths
        concatenated in that order (multi-batch datasets: ``[cty1, cty2,
        cty3]``), a dict as returned by :func:`multibench.labels_for` - it
        goes in AS IS when its insertion order is the method's stacking order
        (``cty1, cty2, ...`` numerically; ``rna`` before ``adt`` before
        ``atac``), which is the order ``labels_for`` returns; a dict in ANY
        OTHER order needs ``label_order=`` (a one-entry dict has no order to
        get wrong) - a 1-D ``ndarray``/``Series``/``Categorical``/list, a
        single-column DataFrame, or - when ``output`` is an AnnData - the
        name of an ``obs`` column. A multi-column CSV/DataFrame raises: pass
        the one column (``df["celltype"]``) itself.

        ORDER. Arrays/lists/files are matched POSITIONALLY to the rows of
        ``output``. A ``Series``/``DataFrame`` with a non-default index is
        aligned BY CELL ID when ``output`` carries ids (AnnData / DataFrame
        with a non-default index): rows are reindexed to the output's order,
        and a missing or extra id raises ``ValueError`` naming the first ones.
        When ``output`` is a bare array there is nothing to align against: the
        Series is matched positionally and a ``UserWarning`` says so (pass
        ``labels.to_numpy()`` to silence it).
    category : str, keyword-only, optional
        One of :func:`multibench.list_categories` (``vertical``, ``diagonal``,
        ``mosaic``, ``cross``). Validated when given and otherwise unused -
        the metrics do not depend on it - so it may be omitted; it is
        accepted so a call can mirror ``run()``'s arguments.
    batch : keyword-only, optional
        Batch labels, one per cell (same forms and the same alignment rule as
        ``labels``; obs column name for AnnData). Needed for the batch family
        (``ASW_batch, GC, iLISI, kBET``) EXCEPT when ``labels`` is a list of
        two or more files (or a dict with ``label_order``), in which case the
        file of origin (1, 2, ...) is used as the batch - the same rule
        :func:`multibench.run_all` applies. Given together with a
        ``metrics`` selection that has no batch metric it changes nothing,
        and a ``UserWarning`` says so.
    metrics : None, str or list of str, keyword-only
        THE metric-selection knob. ``None`` (default) computes every
        applicable metric: the clustering family, plus the batch family when
        a batch vector is available (``batch=`` or a list of label files);
        kBET is never included by default - it shells out to R and takes
        hours on large datasets. ``"clustering"`` computes ``ARI, NMI, ASW,
        iASW, iF1, cLISI``; ``"batch"`` computes ``ASW_batch, GC, iLISI``;
        ``"all"`` both (each needs the batch vector or raises). A LIST of
        codes computes exactly those (case/alias tolerant, ``["ari"]``
        works), including the Leiden sweep only when a requested metric
        needs it (ARI, NMI, iF1 do - ``["ASW"]`` on 10^4 cells returns in
        seconds); ``"kBET"`` in the list turns kBET on. An unknown code
        raises ``ValueError`` listing the valid ones; a bare code string
        (``metrics="ARI"``) raises and points at the list form. Valid codes:
        ``mtb.plot.CLUSTERING_METRICS + mtb.plot.BATCH_METRICS``.
        (``task=``, ``family=`` and ``only=`` are the deprecated 0.2.x
        spellings of this knob; ``slow_metrics=`` is gone.)
    clustering : keyword-only, optional
        Optional precomputed cluster assignment (same forms and the same
        alignment rule as ``labels``; an ``.h5`` path is read from
        ``/obs/cluster_leiden``). When omitted the scIB optimal-resolution
        Leiden sweep derives one from the embedding - the expensive step (10
        resolutions; minutes for ~10^4 cells); passing one skips it for
        ``ARI``/``NMI`` (``iF1`` still sweeps unless excluded via ``metrics``).
    obsm : str, keyword-only
        ``.obsm`` key to use when ``output`` is an AnnData / ``.h5ad``
        (default ``'X_emb'``; ``'X'`` means ``.X``).
    label_order : list of str, keyword-only
        Only for a ``labels`` dict with several entries: the dict keys in the
        order the method stacked the cells (the benchmark's convention is
        numbered files ascending - ``cty1, cty2, ...`` - and ``rna`` before
        ``atac``; :func:`multibench.labels_for` ``(ds, category, method)``
        returns the files in that order, so ``label_order=list(d)`` trusts
        it). A subset of the keys selects those files only. Unknown or
        repeated keys raise; ``label_order`` with a non-dict ``labels`` raises
        ``TypeError``.
    verbose : bool, keyword-only
        ``True`` (default) prints one line on stderr when the Leiden
        resolution sweep starts on more than 2,000 cells - the point at
        which it takes long enough to look like a hang (``"scIB clustering
        metrics: Leiden resolution sweep over 11,014 cells ..."``); ``False``
        never prints.

    Raises
    ------
    ValueError
        missing labels, a batch metric / family requested without batch
        labels, unknown category, ``metrics`` token or code, length
        mismatches (``'input length mismatch: emb has N cells, celltype has
        M'``), cell-id mismatches when aligning, ambiguous label files, a
        multi-entry labels dict in a non-stacking order without
        ``label_order``, an ``.h5`` output without dataset ``data``.
    FileNotFoundError
        an ``output`` / label path that does not exist (the message names
        the path and the working directory).
    TypeError
        unsupported input types (non-array ``labels``, ``label_order`` with
        non-dict labels, ...), and the removed 0.2.x keywords
        ``slow_metrics`` / ``column`` / ``metric_set``.
    """
    _validate_category(category)
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
    group, only, slow = _plan_metrics(
        metrics, has_batch=batch is not None or batch_from_files,
        batch_given=batch is not None)

    if isinstance(output, (str, Path)) and Path(output).suffix.lower() == ".h5ad":
        import anndata as ad      # read ONCE; keeps obs_names and obs columns
        output = ad.read_h5ad(output)
    adata = output if io._is_anndata(output) else None
    ids = _cell_ids(output)
    emb = np.asarray(io.as_matrix(output, obsm=obsm))
    if _is_label_path_list(labels):
        # several label files: concatenate IN THE GIVEN ORDER; remember the
        # sizes so the file of origin can serve as the batch below
        parts = [np.asarray(io.read_labels(p)) for p in labels]
        ct = np.concatenate(parts)
    else:
        parts = None
        ct = _obs_or_vector(labels, adata, what="labels", ids=ids)
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
    elif batch_from_files and group in {"batch", "all"}:
        # batch = which label FILE each cell came from (1-based, as run_all)
        ba = np.concatenate([np.full(len(v), i + 1) for i, v in enumerate(parts)])
    else:
        # clustering metrics need a batch_key but it is a no-op there, so a
        # constant vector is acceptable when no batch labels are supplied.
        ba = np.zeros(emb.shape[0], dtype=int)

    from . import scib as escib  # imported lazily so non-scib paths don't require it
    out = escib.compute(emb, ct, cl, ba, group=group, slow_metrics=slow, only=only,
                        verbose=None if verbose else False)
    # The names compute() emits ARE the canonical ones, but make that a
    # property of evaluate() rather than a coincidence: a frame that reaches
    # to_long()/pd.concat with the published tables must never carry a second
    # spelling of the same metric.
    out.index = pd.Index([catalog.canonical_metric(m) for m in out.index],
                         name=out.index.name)
    if out.empty:
        # unreachable after _plan_metrics; kept so a future metric family
        # mismatch fails loudly instead of handing back a (0, 1) frame
        raise ValueError(
            f"evaluate(metrics={metrics!r}) selected no metric - nothing to return")
    return out
