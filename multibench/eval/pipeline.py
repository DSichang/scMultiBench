"""evaluate(): turn a run output into a metric.csv-shaped DataFrame."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from .. import _compat
from ..data import catalog

#: the seven columns of the tidy long frame (pinned to
#: multibench.data.results.COLUMNS by tests/test_eval_reshape.py)
LONG_COLUMNS = ["metric", "value", "method", "dataset", "category", "clustering", "source"]


def to_long(value_df, *, method: str, dataset: str | None = None,
            category: str | None = None, clustering: str = "default",
            source: str = "user") -> pd.DataFrame:
    """Reshape ``mtb.evaluate``'s wide frame into the tidy long results frame.

    The result has the seven columns of ``mtb.load_results``: your scores
    concatenate with the stored ones, keep their provenance through a CSV
    round trip, and plot with ``mtb.plot.bubble``.

    Parameters
    ----------
    value_df : pandas.DataFrame or pandas.Series
        What ``mtb.evaluate`` returns (metrics as the index, one column
        ``Value``), a Series indexed by metric, or that frame read back with
        ``pd.read_csv``.
    method : str, keyword-only
        Method id written into every row; your own name is fine.
    dataset : str, keyword-only, optional
        Dataset id written into every row; ``None`` writes ``"all"``.
    category : str, keyword-only, optional
        Integration category written into every row; ``None`` writes
        ``"user"``.
    clustering : str, keyword-only
        Value of the ``clustering`` column.
    source : str, keyword-only
        Value of the ``source`` column; ``source="user"`` in
        ``mtb.load_results`` selects these rows again.

    Returns
    -------
    pandas.DataFrame
        One row per metric, with exactly the columns ``metric, value, method,
        dataset, category, clustering, source``.

    Raises
    ------
    ValueError
        ``value_df`` is already long, has no ``Value`` column, no metric
        names, or duplicate ones.

    Examples
    --------
    >>> import multibench as mtb, pandas as pd
    >>> wide = mtb.evaluate(emb, labels=labels, metrics=["ARI", "NMI"])
    >>> mine = mtb.to_long(wide, method="MyMethod", dataset="D11", category="vertical")
    >>> stored = mtb.load_results("vertical", dataset="D11", source="rerun")
    >>> pd.concat([stored, mine]).to_csv("all.csv", index=False)
    >>> mtb.load_results(result_path="all.csv", source="user")      # your rows only

    Notes
    -----
    **Metric names.** Names are canonicalised (``ari`` -> ``ARI``, ``kbet``
    -> ``kBET``); rows whose name is blank are dropped, and a name the
    package does not know is kept as written.

    **Column values.** ``dataset=None`` writes ``"all"``, the placeholder the
    plotting layer uses for a frame without datasets, so the column is never
    blank. ``category=None`` writes ``"user"``, the value ``load_results``
    gives a user file without a category column. The published tables use
    ``clustering`` values ``"louvain"`` / ``"kmeans"`` for their variants.

    **Plot badges.** The bubble figure shows ``?`` for a method name the
    registry does not know. To badge a method of your own as supervised, add
    a boolean ``needs_labels`` column to the frame.

    **Errors.** All are ``ValueError``:

    - an already long frame (columns ``metric, value, method``) - pass it to
      the plot / ``load_results`` consumers directly;
    - no ``Value`` column - the message names the expected shape and, for a
      wide one-row frame, the ``df.T.set_axis(['Value'], axis=1)`` fix;
    - every metric name blank;
    - two names that collapse onto one canonical metric (``ari`` and
      ``ARI``): a silent duplicate would double-count that metric in every
      downstream rank.

    See Also
    --------
    mtb.evaluate : computes the wide frame this function reshapes.

    mtb.load_results : the stored scores, in the same long shape.

    mtb.plot.bubble : plots a long frame.
    """
    if isinstance(value_df, pd.Series):
        value_df = value_df.to_frame("Value")
    cols = [str(c) for c in getattr(value_df, "columns", [])]
    if {"metric", "value", "method"} <= set(cols):
        raise ValueError(
            "to_long() got an already long frame (columns metric, value, method"
            f"{', ...' if len(cols) > 3 else ''}); pass it to mtb.plot.bubble / "
            "pd.concat / load_results consumers directly - to_long reshapes "
            "evaluate()'s wide frame (metrics as the index, one column 'Value')")
    if "Value" not in cols:
        idx = list(map(str, list(value_df.index)[:5]))
        raise ValueError(
            f"to_long expects evaluate()'s frame: metrics as index, one column "
            f"'Value'; got columns {cols} (index {idx}) - for a wide one-row "
            f"frame use df.T.set_axis(['Value'], axis=1)")
    if "metric" in cols:
        # the CSV read-back of evaluate's frame (pd.read_csv(out)): the metric
        # names are a column, not the index - reset_index below would
        # otherwise prepend the RangeIndex as a second 'metric' column
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
    # the plotting layer owns the metric families
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
      carries cell ids (``ids`` given) -> aligned by id via
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
    ``d``; a subset selects those files) unless ``d`` is what ``labels_for``
    returned, in the order it returned it (a method variant's stacking
    order, recorded on the dict), or any dict in the default order
    (``_label_sort_key``). Any other order must be explicit: a guess would
    score the embedding against misordered labels without any error.
    """
    if label_order is None:
        if len(d) == 1:
            return [next(iter(d.values()))]
        from ..engine.resolve import LabelFiles, _label_sort_key
        keys = list(d)
        if isinstance(d, LabelFiles) and d.in_stacking_order():
            return [d[k] for k in keys]
        if keys == sorted(keys, key=_label_sort_key):
            return [d[k] for k in keys]
        raise ValueError(
            f"labels: got a dict with {len(d)} label files {keys} that is not "
            f"an unchanged mtb.labels_for dict and is not in the default "
            f"stacking order (cty1, cty2, ... numerically; rna before adt before "
            f"atac - NOT alphabetical); pass the dict "
            f"mtb.labels_for(dataset, method=<method>, category=<category>) "
            f"returns, unchanged (it is in that method's stacking order), a list "
            f"of paths in cell order, or label_order=[...] naming the keys in "
            f"that order")
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
    """Turn the ``metrics=`` argument into what :func:`multibench.eval.scib.compute` needs.

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

    Pass the embedding a run produced and one label per cell. Reshape the
    result with ``mtb.to_long`` to plot it or combine it with
    ``mtb.load_results``.

    Parameters
    ----------
    output
        The embedding, cells x dims: an array, DataFrame, sparse matrix,
        AnnData (``.obsm[obsm]``) or a file path (formats in Notes).
    labels
        Cell types, one per cell: CSV path(s), a ``mtb.labels_for`` dict, a
        1-D array-like, or an ``obs`` column name (forms in Notes).
    category : str, keyword-only, optional
        One of ``mtb.list_categories()``; validated, otherwise unused (the
        metrics do not depend on it). Lets a call mirror ``mtb.run``.
    batch : keyword-only, optional
        Batch labels, one per cell, in the forms ``labels`` accepts; needed
        for the batch metrics unless ``labels`` lists two or more files.
    metrics : None, str or list of str, keyword-only
        ``None`` = every applicable metric except kBET; or ``"clustering"``,
        ``"batch"``, ``"all"``, or a list of codes such as ``["ARI", "NMI"]``.
    clustering : keyword-only, optional
        Precomputed cluster assignment, in the forms ``labels`` accepts or an
        ``.h5`` path; ``None`` = derive one with the Leiden sweep.
    obsm : str, keyword-only
        ``.obsm`` key used when ``output`` is an AnnData or ``.h5ad``;
        ``'X'`` means ``.X``.
    label_order : list of str, keyword-only
        Keys of a multi-entry ``labels`` dict in the method's stacking order
        (a subset selects those files); ``None`` = the dict's order where
        safe (Notes).
    verbose : bool, keyword-only
        ``True`` prints one stderr line when the Leiden sweep starts on more
        than 2,000 cells; ``False`` never prints.

    Returns
    -------
    pandas.DataFrame
        One row per metric, indexed by the canonical metric name (``ARI``,
        ``NMI``, ...), with one column ``Value`` - the ``metric.csv`` shape.
        Never empty.

    Raises
    ------
    ValueError
        Missing or misaligned labels, an unknown category or metric, or batch
        metrics without batch labels.
    FileNotFoundError
        An ``output`` or label path does not exist.
    TypeError
        An unsupported input type, or a retired keyword (Notes).

    Examples
    --------
    >>> import multibench as mtb
    >>> labels = mtb.labels_for("D11")                      # {'cty': '.../D11/cty.csv'}
    >>> scores = mtb.evaluate(res.output, labels=labels)    # res = mtb.run(...)
    >>> mtb.evaluate(res.output, labels=labels, metrics=["ASW", "cLISI"])   # no Leiden sweep
    >>> mtb.evaluate(adata, labels="celltype", batch="batch", metrics="all")

    Notes
    -----
    **Label forms.** ``labels`` may be:

    - a CSV path (header row; column ``x``, the only column, or the last of
      two when the first is a barcode index);
    - a list of CSV paths, concatenated in that order (multi-batch datasets:
      ``[cty1, cty2, cty3]``);
    - a dict from ``mtb.labels_for``, unchanged - it goes in AS IS, in the
      order ``labels_for`` gave it (with ``category`` and ``method``, that
      method's stacking order); any other dict goes in as is only in the
      default order (``cty1, cty2, ...`` numerically; ``rna`` before ``adt``
      before ``atac``) and otherwise needs ``label_order=`` naming its keys;
      ``label_order=list(d)`` trusts the dict's own order, and a one-entry
      dict has no order to get wrong;
    - a 1-D ``ndarray`` / ``Series`` / ``Categorical`` / list, or a
      single-column DataFrame;
    - when ``output`` is an AnnData, the name of an ``obs`` column.

    A multi-column CSV / DataFrame raises: pass the one column
    (``df["celltype"]``) itself. ``batch`` and ``clustering`` take the same
    forms except a multi-entry dict; for ``clustering``, a path that is not
    a CSV is read as an h5 from ``/obs/cluster_leiden``.

    **Metric selection.** ``None`` computes the clustering family (``ARI,
    NMI, ASW, iASW, iF1, cLISI``), plus ``ASW_batch, GC, iLISI`` when a batch
    vector is available (``batch=`` or a list of label files). kBET is never
    included by default: it shells out to R and takes hours on large
    datasets.

    - ``"clustering"`` / ``"batch"`` - that family; ``"all"`` - both
      (``"batch"`` and ``"all"`` need the batch vector or raise);
    - a list of codes - exactly those (case/alias tolerant, ``["ari"]``
      works); the Leiden sweep runs only when one of them needs it (ARI,
      NMI, iF1), and ``"kBET"`` in the list turns kBET on.

    Valid codes: ``mtb.plot.CLUSTERING_METRICS + mtb.plot.BATCH_METRICS``.
    An unknown code raises ``ValueError`` listing the valid ones; a bare
    code string (``metrics="ARI"``) raises and points at the list form.

    **Cost.** ``ARI``, ``NMI`` and ``iF1`` need the scIB optimal-resolution
    Leiden sweep (10 resolutions on a kNN graph of the embedding): tens of
    seconds for a few thousand cells, minutes for ~10^4. To skip it, name
    only metrics that do not need it in ``metrics=[...]`` (``ASW``,
    ``iASW``, ``cLISI``, the batch family). ``clustering=`` removes the need
    for ``ARI`` / ``NMI`` only; ``iF1`` always sweeps.

    **Leiden backend.** ``mtb.config.DEFAULT.leiden_flavor``: ``"igraph"``
    (default; several times faster) or ``"leidenalg"`` (the classic backend
    scib itself runs).

    **Batch.** The batch family (``ASW_batch, GC, iLISI, kBET``) needs batch
    labels. When ``labels`` is a list (or dict) of two or more files and
    ``batch`` is omitted, the file of origin (1, 2, ...) serves as the
    batch, the same rule ``mtb.run_all`` applies. ``batch=`` given with a
    ``metrics`` selection that has no batch metric changes nothing, and a
    ``UserWarning`` says so.

    **Cell order.** Arrays, lists and files are matched positionally to the
    rows of ``output``. A ``Series`` / ``DataFrame`` with a non-default
    index is aligned by cell id when ``output`` carries ids (an AnnData, or
    a DataFrame with a non-default index): rows are reindexed to the
    output's order, and a missing or extra id raises ``ValueError`` naming
    the first ones.

    When ``output`` is a bare array there is nothing to align against: the
    Series is matched positionally and a ``UserWarning`` says so (pass
    ``labels.to_numpy()`` to silence it).

    **Output formats.** A file path may be ``.h5`` (dataset ``data``, the
    benchmark's ``embedding.h5``), ``.h5ad`` (read as an AnnData),
    ``.npy``, or ``.csv`` / ``.tsv``. A dims x cells array is transposed
    against the label count, with a warning.

    **Errors.** ``ValueError`` covers:

    - missing labels, or a batch metric / family requested without batch
      labels;
    - an unknown category, ``metrics`` token or code;
    - length mismatches (``'input length mismatch: emb has N cells,
      celltype has M'``) and cell-id mismatches when aligning;
    - ambiguous label files, and a multi-entry labels dict (other than an
      unchanged ``labels_for`` one) out of the default order without
      ``label_order``;
    - unknown, repeated or no keys in ``label_order``;
    - an ``.h5`` output without dataset ``data``.

    ``FileNotFoundError`` names the missing path and the working directory.
    ``TypeError`` covers unsupported input types (non-array ``labels``,
    ``label_order`` with non-dict ``labels``, ...).

    **Retired keywords.** ``task=``, ``family=`` and ``only=`` still work
    as spellings of ``metrics=``, with a ``DeprecationWarning``;
    ``slow_metrics=``, ``column=`` and ``metric_set=`` raise ``TypeError``
    naming the replacement. The API overview lists them.

    See Also
    --------
    mtb.labels_for : the label files of a dataset, in stacking order.

    mtb.to_long : reshapes the result into the long results frame.

    mtb.run_all : runs and scores every runnable method on a dataset.
    """
    _validate_category(category)
    if labels is None:
        raise ValueError("metrics require `labels` (cty / ground-truth cell types).")
    # a labels_for() dict becomes the list of paths in cell order first, so the
    # file-of-origin batch rule below sees it like any other list of files
    if isinstance(labels, dict):
        labels = _labels_from_dict(labels, label_order)
    elif label_order is not None:
        raise TypeError(
            f"label_order= applies only when labels is a dict ({{name: path}}, "
            f"as returned by mtb.labels_for); labels is a "
            f"{type(labels).__name__} - pass the files as a list in that order "
            f"instead")
    batch_from_files = batch is None and _is_label_path_list(labels) and len(labels) > 1
    group, only, slow = _plan_metrics(
        metrics, has_batch=batch is not None or batch_from_files,
        batch_given=batch is not None)

    if isinstance(output, (str, Path)) and Path(output).suffix.lower() == ".h5ad":
        import anndata as ad      # read once; keeps obs_names and obs columns
        output = ad.read_h5ad(output)
    adata = output if io._is_anndata(output) else None
    ids = _cell_ids(output)
    emb = np.asarray(io.as_matrix(output, obsm=obsm))
    if _is_label_path_list(labels):
        # several label files: concatenate in the given order; remember the
        # sizes so the file of origin can serve as the batch below
        parts = [np.asarray(io.read_labels(p)) for p in labels]
        ct = np.concatenate(parts)
    else:
        parts = None
        ct = _obs_or_vector(labels, adata, what="labels", ids=ids)
    # Orient to cells x dims. read_embedding() already does this for h5 inputs;
    # do the same for everything else so a caller can pass run().output (dims x
    # cells for many methods) directly. Transpose only when the label count
    # says the cells are on the other axis.
    if emb.ndim == 2 and emb.shape[0] != len(ct) and emb.shape[1] == len(ct):
        warnings.warn(
            f"output was (dims x cells) {emb.shape}; transposed to (cells x dims)")
        emb = emb.T
    if clustering is None:
        cl = None
    elif isinstance(clustering, (str, Path)) \
            and Path(clustering).suffix.lower() not in io._LABEL_FILE_SUFFIXES \
            and not (adata is not None and isinstance(clustering, str)
                     and clustering in adata.obs.columns):
        cl = io.read_clustering(clustering)        # the benchmark's h5 layout
    else:
        cl = _obs_or_vector(clustering, adata, what="clustering", ids=ids)
    if batch is not None:
        ba = _obs_or_vector(batch, adata, what="batch", ids=ids)
    elif batch_from_files and group in {"batch", "all"}:
        # batch = which label file each cell came from (1-based, as run_all)
        ba = np.concatenate([np.full(len(v), i + 1) for i, v in enumerate(parts)])
    else:
        # clustering metrics need a batch_key but it is a no-op there, so a
        # constant vector is acceptable when no batch labels are supplied.
        ba = np.zeros(emb.shape[0], dtype=int)

    from . import scib as escib
    out = escib.compute(emb, ct, cl, ba, group=group, slow_metrics=slow, only=only,
                        verbose=None if verbose else False)
    # compute() already emits canonical names; canonicalise anyway, so a frame
    # concatenated with the published tables never carries a second spelling
    # of a metric
    out.index = pd.Index([catalog.canonical_metric(m) for m in out.index],
                         name=out.index.name)
    if out.empty:
        # unreachable after _plan_metrics; kept so a future metric family
        # mismatch fails loudly instead of handing back a (0, 1) frame
        raise ValueError(
            f"evaluate(metrics={metrics!r}) selected no metric - nothing to return")
    return out
