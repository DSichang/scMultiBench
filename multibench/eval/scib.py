"""scIB metric computation (clustering + batch), after upstream
``evaluation_pipelines/scib_metrics/scib_metrics.py``.

The protocol is the paper's: Leiden clusters searched to the number of cell
types (and of batches, for ARI_batch and NMI_batch); ASW, iASW and iF1 on those
clusters.
"""
from __future__ import annotations

import contextlib
import functools
import io
import warnings

import numpy as np
import pandas as pd

from .. import _text


def import_scib():
    """``scib``, with ``scib.metrics``, imported under the caller's warning
    filters: scib 1.1.7 calls ``warnings.simplefilter("default")`` on import
    (``_package_tools.py``), which shows every warning of the session from then
    on, such as Colab's own DeprecationWarnings in the output of ``run``."""
    import sys
    if "scib.metrics" not in sys.modules:
        with warnings.catch_warnings():
            import scib.metrics  # noqa: F401
    return sys.modules["scib"]


#: metrics that run scib's prebuilt LISI helper binary
_LISI_METRICS = ("cLISI", "iLISI")

#: C++ compilers tried, in order, when the shipped LISI helper cannot execute
_CXX_CANDIDATES = ("g++", "c++", "clang++")

#: the flags of the build line in scib's ``knn_graph/README.md``
_CXX_FLAGS = ("-std=c++11", "-O3")

#: above this many cells the Leiden clustering is announced on stderr
#: (``verbose=None``): it then takes long enough to look like a hang
_SWEEP_NOTICE_CELLS = 2000

#: ``sinfonia.get_N_clusters``, which the paper clusters with: the Leiden
#: resolution is bisected on this interval for at most this many steps
_SEARCH_RANGE = (0.0, 3.0)
_SEARCH_STEPS = 30

#: scanpy's ``settings.N_PCS``: ``sc.pp.neighbors`` without ``use_rep``, as
#: SINFONIA calls it, runs on the first 50 principal components of a wider ``X``
_N_PCS = 50

#: the metrics scored on the clusters searched to the number of cell types
#: (or on the clustering the caller passes)
_CLUSTER_METRICS = ("ARI", "NMI", "ASW", "iASW", "iF1")
#: the metrics scored on the clusters searched to the number of batches
_BATCH_CLUSTER_METRICS = ("ARI_batch", "NMI_batch")


def _find_cxx() -> str | None:
    """Path of the first C++ compiler on PATH, or None."""
    import shutil
    return next(filter(None, map(shutil.which, _CXX_CANDIDATES)), None)


def _probe_lisi_binary(exe) -> str | None:
    """Run scib's LISI helper once with no arguments: None when it starts, else
    the problem (``"cannot be executed"`` for an ``OSError`` at exec time)."""
    import os
    import subprocess

    if not os.access(exe, os.X_OK):
        try:
            exe.chmod(exe.stat().st_mode | 0o111)
        except OSError:
            return f"{exe} is not executable and its mode cannot be changed"
    try:
        p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    except OSError as exc:
        return f"{exe} cannot be executed here ({type(exc).__name__}: {exc})"
    except subprocess.TimeoutExpired:
        return None                      # it started, which is all the probe checks
    # a bare call prints the usage line; matching its text would turn a wording
    # change into two NaN metrics. A loader failure writes stderr only.
    if p.returncode == 0 or (p.stdout or "").strip():
        return None
    err = (p.stderr or "").strip().splitlines()
    return err[0] if err else f"{exe} exited {p.returncode} with no output"


def _rebuild_lisi_binary(exe, cpp, cxx: str) -> str | None:
    """Compile ``cpp`` to ``exe`` with scib's own build line; None on success,
    else why. scib looks for the binary in its own ``knn_graph/``. The build
    writes a file of its own and replaces ``exe`` only when it succeeds, so two
    processes rebuilding at once, or a failed build, never remove ``exe``."""
    import os
    import subprocess

    shown = " ".join([cxx, *_CXX_FLAGS, "-o", str(exe), str(cpp)])
    tmp = exe.with_name(f".{exe.name}.{os.getpid()}")
    try:
        p = subprocess.run([cxx, *_CXX_FLAGS, "-o", str(tmp), str(cpp)],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        tmp.unlink(missing_ok=True)
        return f"{shown} failed ({type(exc).__name__}: {exc})"
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        err = (p.stderr or p.stdout or "").strip().splitlines()
        return f"{shown} exited {p.returncode}" + (f": {err[-1]}" if err else "")
    os.replace(tmp, exe)
    return None


def _lisi_fallback_message(problem: str) -> str:
    """The warning for the LISI metrics recorded as NaN: the cause and scib's
    own ``g++`` build line. One text for both, so a call warns once."""
    from pathlib import Path

    scib = import_scib()
    kg = Path(scib.__file__).parent / "knn_graph"
    return (f"cLISI and iLISI need scib's LISI helper binary, which cannot run "
            f"here: {problem}. Recording NaN. To fix it, rebuild the "
            f"binary from the source scib ships (needs a C++ compiler):\n"
            f"    g++ {' '.join(_CXX_FLAGS)} -o {kg / 'knn_graph.o'} {kg / 'knn_graph.cpp'}")


@functools.lru_cache(maxsize=1)
def _lisi_helper_problem() -> str | None:
    """Why scib's LISI helper cannot run here, or None when it can.

    scib runs the prebuilt binary without checking it, so a binary that does
    not start (scib ships a Linux x86-64 executable linked against glibc 2.38)
    surfaces only as a missing output file, once per metric. One probe names
    the cause; with a C++ compiler and the shipped ``knn_graph.cpp`` a helper
    that cannot run, or is missing, is rebuilt once and probed again.
    """
    from pathlib import Path

    try:
        scib = import_scib()
    except Exception as exc:  # noqa: BLE001 - absence is the caller's problem
        return f"scib is not importable ({type(exc).__name__})"
    kg = Path(scib.__file__).parent / "knn_graph"
    exe = kg / "knn_graph.o"
    problem = (_probe_lisi_binary(exe) if exe.is_file()
               else f"{exe} is missing from the scib installation")
    if problem is None:
        return None
    cpp = kg / "knn_graph.cpp"
    cxx = _find_cxx()
    if not cpp.is_file():
        return f"{problem}; {cpp} is not shipped, so it cannot be rebuilt here"
    if cxx is None:
        return (f"{problem}; no C++ compiler ({', '.join(_CXX_CANDIDATES)}) on "
                f"PATH to rebuild it")
    failed = _rebuild_lisi_binary(exe, cpp, cxx)
    if failed:
        return f"{problem}; rebuilding from source failed: {failed}"
    return _probe_lisi_binary(exe)


#: metrics whose scIB definition is bounded by [0, 1] (scaled LISI, the ASW
#: family, graph connectivity); ``compute`` clips them to that range
BOUNDED_METRICS = ("ASW", "iASW", "cLISI", "ASW_batch", "ARI_batch", "NMI_batch",
                   "GC", "iLISI")

#: distance from 0 (and, for BOUNDED_METRICS, from 1) below which a value is
#: floating-point residue and recorded as the bound: an iLISI of 2.2e-16 next to
#: 0.0 would otherwise rank as a real difference
_SNAP = 1e-12


def _tidy_value(name: str, value) -> float:
    """``value`` as a float, ``|value| < 1e-12`` as ``0.0``, and a
    :data:`BOUNDED_METRICS` value clipped to [0, 1] with ``1 - 1e-12`` as ``1.0``."""
    v = float(value)
    if np.isnan(v):
        return v
    if abs(v) < _SNAP:
        return 0.0
    if name in BOUNDED_METRICS:
        if abs(v - 1.0) < _SNAP:
            return 1.0
        return min(max(v, 0.0), 1.0)
    return v


@functools.cache
def igraph_flavor_available() -> bool:
    """Can this scanpy run ``sc.tl.leiden(flavor="igraph")``? scanpy accepts
    ``flavor=`` from 1.10 on and the backend needs the ``igraph`` package."""
    try:
        import igraph  # noqa: F401
    except Exception:  # noqa: BLE001 - an igraph that fails to import means "no"
        return False
    return _scanpy_has_flavor_kw()


@functools.cache
def _scanpy_has_flavor_kw() -> bool:
    """Does this scanpy accept ``sc.tl.leiden(flavor=...)`` (>= 1.10)? Older
    ones fail on the keyword."""
    from importlib.metadata import version
    from packaging.version import Version
    # by version, not by signature: tests replace sc.tl.leiden with recorders
    return Version(version("scanpy")) >= Version("1.10")


_fallback_warned = False


def _resolve_flavor(flavor=None) -> str:
    """The Leiden backend that will run: ``flavor``, else ``config.leiden_flavor``,
    downgraded to ``"leidenalg"`` (one warning) where igraph cannot run."""
    global _fallback_warned
    if flavor is None:
        from .._settings import config
        flavor = config.leiden_flavor
    if flavor == "igraph" and not igraph_flavor_available():
        if not _fallback_warned:
            from importlib.metadata import version
            _text.warn(
                f"scanpy {version('scanpy')} cannot run flavor='igraph' "
                f"(needs scanpy>=1.10 and the igraph package); using leidenalg for "
                f"this process - same metrics, slower clustering", UserWarning)
            _fallback_warned = True
        return "leidenalg"
    return flavor


def _neighbors(adata, rep: str) -> None:
    """``sc.pp.neighbors`` on ``adata.obsm[rep]``, silencing the
    ``SparseEfficiencyWarning`` scipy raises on some embeddings. When numba
    cannot type the distances scanpy hands to umap-learn (umap-learn 0.6.0 with
    scanpy 1.12.5, on 8,192 cells or more), the error names the release and
    what to install."""
    import scanpy as sc
    from scipy.sparse import SparseEfficiencyWarning
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            sc.pp.neighbors(adata, use_rep=rep)
    except Exception as e:
        # known by numba's words, not by the release: another release with the
        # same fault is named the same way
        if type(e).__name__ != "TypingError" or "knn_dists" not in str(e):
            raise
        from importlib.metadata import version
        here = version("umap-learn")
        raise RuntimeError(
            f"umap-learn {here} cannot build the neighbour graph of these "
            f"{adata.n_obs:,} cells. Run pip install \"umap-learn!={here}\", then "
            f"score again.") from e


def _leiden(adata, resolution: float, key_added: str, flavor: str) -> None:
    """One Leiden clustering into ``adata.obs[key_added]``: scanpy's igraph
    backend with its documented settings, or leidenalg as scib runs it."""
    import scanpy as sc
    if flavor == "igraph":
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added,
                     flavor="igraph", n_iterations=2, directed=False)
        return
    with warnings.catch_warnings():
        # leidenalg is an explicit choice here: silence scanpy's notice that
        # igraph is the future default (a FutureWarning from scanpy 1.11)
        warnings.filterwarnings("ignore", message=".*igraph.*implementation of leiden.*",
                                category=UserWarning)
        warnings.filterwarnings("ignore", message=".*default backend for leiden will be igraph.*",
                                category=FutureWarning)
        # spelled out, so a change of scanpy's default cannot switch the backend
        kw = {"flavor": "leidenalg"} if _scanpy_has_flavor_kw() else {}
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added, **kw)


def _build_adata(emb, celltype, cluster, batch):
    import anndata as ad      # lazily: an evaluation-only dependency
    from .io import _own_dtype
    adata = ad.AnnData(_own_dtype(emb))
    adata.obsm["X_emb"] = adata.X
    adata.obs["celltype"] = pd.Categorical(np.asarray(celltype).astype(str))
    if cluster is not None:
        adata.obs["cluster"] = pd.Categorical(np.asarray(cluster))
    # as strings: rpy2 turns this into an R factor for kBET and refuses other categories
    adata.obs["batch"] = pd.Categorical(np.asarray(batch).astype(str))
    return adata


def _clustering_graph(emb):
    """An AnnData with the kNN graph the paper clusters on (SINFONIA.py):
    NaN read as 0, then ``sc.pp.neighbors`` with scanpy's defaults, which use
    the first 50 principal components of an embedding wider than 50."""
    import anndata as ad
    import scanpy as sc
    x = np.array(emb)
    x = x.astype(x.dtype if np.issubdtype(x.dtype, np.floating) else float)
    x[np.isnan(x)] = 0
    adata = ad.AnnData(x)
    rep = "X"
    if x.shape[1] > _N_PCS:
        sc.pp.pca(adata, n_comps=_N_PCS)
        rep = "X_pca"
    _neighbors(adata, rep)
    return adata


def _search_n_clusters(adata, n: int, key: str, flavor: str, seen: dict | None = None) -> None:
    """Leiden into ``adata.obs[key]`` at the resolution that gives ``n`` clusters,
    bisected as ``sinfonia.get_N_clusters`` does; the last step's clustering
    when no step gave ``n``. ``seen`` holds ``{resolution: clusters}`` of
    ``adata``: Leiden on one graph at one resolution gives one clustering, so a
    resolution two searches visit is clustered once."""
    lo, hi = _SEARCH_RANGE
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(_SEARCH_STEPS):
            res = lo + (hi - lo) / 2
            if seen is not None and res in seen:
                adata.obs[key] = seen[res]
            else:
                _leiden(adata, res, key, flavor)
                if seen is not None:
                    seen[res] = adata.obs[key].copy()
            found = adata.obs[key].nunique()
            if found > n:
                hi = res
            elif found < n:
                lo = res
            else:
                return


def _n_clusters(graph, n: int, flavor: str, seen: dict | None = None) -> np.ndarray:
    """The searched clusters of ``graph`` as strings: those of the last step when no
    resolution gave ``n``, as the paper's search (``seen``: see
    :func:`_search_n_clusters`)."""
    key = "_mb_search"
    _search_n_clusters(graph, n, key, flavor, seen)
    out = np.asarray(graph.obs[key].astype(str))
    del graph.obs[key]
    return out


def leiden_n_clusters(emb, n: int) -> np.ndarray:
    """Cluster an embedding with Leiden at the resolution that gives ``n`` clusters.

    The paper's clustering: a kNN graph with scanpy's defaults (on the
    first 50 principal components when the embedding has more than 50
    dimensions), and the resolution bisected on [0, 3] for at most 30 steps
    until there are ``n`` clusters, as ``sinfonia.get_N_clusters``.

    Parameters
    ----------
    emb : array-like
        Embedding, cells x dims. NaN counts as 0.
    n : int
        Number of clusters wanted, such as the number of cell types.

    Returns
    -------
    numpy.ndarray
        One cluster id (a string) per cell; those of the last step when no
        resolution gave ``n``.

    The Leiden backend is ``config.leiden_flavor``.
    """
    return _n_clusters(_clustering_graph(emb), int(n), _resolve_flavor())


def _isolated_labels_f1(adata, label_key, batch_key, cluster_key, iso_threshold):
    """Isolated-label F1 of the clustering ``adata.obs[cluster_key]``: for each
    isolated label, the best F1 of one cluster, averaged over labels. scib's
    ``isolated_labels_f1`` with that clustering precomputed (pinned in
    ``tests/test_eval_isolated_f1.py``)."""
    from sklearn.metrics import f1_score
    import_scib()
    from scib.metrics.isolated_labels import get_isolated_labels

    labels = get_isolated_labels(adata, label_key, batch_key, iso_threshold,
                                 verbose=False)
    if len(labels) == 0:
        return float("nan")
    col = adata.obs[cluster_key]
    scores = []
    for label in labels:
        y_true = (adata.obs[label_key] == label).values
        # the argument order of scib's max_f1
        scores.append(max([0.0] + [f1_score((col == c).values, y_true)
                                   for c in col.unique()]))
    return float(np.mean(scores))


def _searched_clusters(adata, emb, ct_metrics: list, batch_metrics: list, *,
                       verbose: bool | None) -> str | None:
    """The paper's Leiden clusters into ``adata.obs``: ``"cluster"``, searched to
    the number of cell types, when ``ct_metrics`` names a metric scored on it,
    and ``"batch_cluster"``, searched to the number of batches, when
    ``batch_metrics`` does. Returns the Leiden backend, or ``None`` when no
    search runs; ``verbose`` as in :func:`compute`."""
    if not (ct_metrics or batch_metrics):
        return None
    flavor = _resolve_flavor()
    n = adata.n_obs
    if verbose or (verbose is None and n > _SWEEP_NOTICE_CELLS):
        import sys
        print(f"Clustering {n:,} cells for {_text._and_list(ct_metrics + batch_metrics)} "
              f"with the {flavor} Leiden backend.", file=sys.stderr, flush=True)
    graph = _clustering_graph(emb)
    seen: dict = {}             # both searches cluster this graph
    if ct_metrics:
        adata.obs["cluster"] = pd.Categorical(_n_clusters(
            graph, adata.obs["celltype"].nunique(), flavor, seen))
    if batch_metrics:
        adata.obs["batch_cluster"] = pd.Categorical(_n_clusters(
            graph, adata.obs["batch"].nunique(), flavor, seen))
    return flavor


def _score(out: dict, only, name: str, fn) -> None:
    """``out[name] = fn()`` (see :func:`_tidy_value`) unless ``only`` leaves
    ``name`` out; NaN with a warning when it fails (the LISI binary does not
    load on every platform)."""
    if only is not None and name not in only:
        return
    if name in _LISI_METRICS:
        problem = _lisi_helper_problem()
        if problem:
            if not any(m in out for m in _LISI_METRICS):     # once per embedding
                _text.warn(_lisi_fallback_message(problem))
            out[name] = float("nan")
            return
    try:
        # scib's progress prints and third-party deprecation warnings
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = _tidy_value(name, fn())
    except Exception as exc:  # noqa: BLE001 - report and continue
        _text.warn(
            f"scib metric {name!r} could not be computed "
            f"({type(exc).__name__}: {str(exc)[:160]}); recording NaN."
        )
        out[name] = float("nan")


def compute(emb, celltype, cluster, batch, group: str = "clustering",
            slow_metrics: bool = False, only=None, *,
            verbose: bool | None = None) -> pd.DataFrame:
    """Compute scIB metrics for one embedding.

    Parameters
    ----------
    emb : array-like
        Embedding, cells x dims.
    celltype : array-like
        Cell-type label per cell.
    cluster : array-like or None
        Precomputed cluster assignment for ARI, NMI, ASW, iASW and iF1;
        ``None`` derives one with :func:`leiden_n_clusters` (one cluster per cell type).
    batch : array-like
        Batch label per cell (a constant vector is fine for clustering-only).
    group : {"clustering", "batch", "all"}
        Metric family.
    slow_metrics : bool
        Also compute kBET (shells out to R).
    only : collection of str, optional
        Compute only the named metrics, e.g. ``only={"ARI"}``; Leiden runs
        only when one of them needs it (ARI, NMI, ASW, iASW, iF1, ARI_batch,
        NMI_batch).
    verbose : bool, keyword-only, optional
        Print one stderr line when the Leiden clustering starts. ``None``
        (default): only for embeddings with more than 2,000 cells; ``True``
        always; ``False`` never. The Leiden backend is ``config.leiden_flavor``.

    Returns
    -------
    pandas.DataFrame
        ``metric.csv``-shaped: index = metric, one column ``Value``.
        ``attrs["leiden_flavor"]`` is the Leiden backend (``None`` when
        Leiden did not run); ``attrs["clustering"]`` is where the clusters
        of ARI, NMI, ASW, iASW and iF1 came from: ``"user"``, ``"search"``, or
        ``None`` without those metrics. ARI_batch and NMI_batch need two or
        more batches.

    Notes
    -----
    Values within ``1e-12`` of 0 are recorded as ``0.0``. The metrics
    bounded by [0, 1] (``ASW``, ``iASW``, ``cLISI``, ``ASW_batch``,
    ``ARI_batch``, ``NMI_batch``, ``GC``, ``iLISI``) are clipped to that
    range, with values within ``1e-12`` of 1 recorded as ``1.0``. An iLISI of
    ``2.2e-16`` is therefore ``0.0``.
    """
    me = import_scib().metrics

    if only is not None:
        only = set(only)

    adata = _build_adata(emb, celltype, cluster, batch)
    _neighbors(adata, "X_emb")

    want_clu = group in ("clustering", "all")
    want_bat = group in ("batch", "all")

    def wanted(names):
        return [m for m in names if only is None or m in only]

    n_batches = int(adata.obs["batch"].nunique())
    # the paper's clusterings: one per cell type, and one per batch for
    # ARI_batch and NMI_batch (scored only with two or more batches)
    flavor = _searched_clusters(
        adata, emb, wanted(_CLUSTER_METRICS) if want_clu and cluster is None else [],
        wanted(_BATCH_CLUSTER_METRICS) if want_bat and n_batches >= 2 else [],
        verbose=verbose)

    out: dict[str, float] = {}
    score = functools.partial(_score, out, only)
    if want_clu:
        score("ARI", lambda: me.ari(adata, cluster_key="cluster", label_key="celltype"))
        score("NMI", lambda: me.nmi(adata, cluster_key="cluster", label_key="celltype"))
        score("ASW", lambda: me.silhouette(adata, label_key="cluster", embed="X_emb"))
        # every label counts as isolated: scib's default scores none when every
        # one is in every batch, and n_batches + 1 clears its check
        iso_threshold = n_batches + 1
        score("iASW", lambda: me.isolated_labels_asw(
            adata, batch_key="batch", label_key="cluster", embed="X_emb",
            iso_threshold=iso_threshold))
        score("iF1", lambda: _isolated_labels_f1(
            adata, label_key="celltype", batch_key="batch", cluster_key="cluster",
            iso_threshold=iso_threshold))
        # type_="knn": scib walks the 15-NN graph computed above; "embed" would recompute it
        score("cLISI", lambda: me.clisi_graph(adata, label_key="celltype", type_="knn"))
    if want_bat:
        score("ASW_batch", lambda: me.silhouette_batch(
            adata, batch_key="batch", label_key="celltype", embed="X_emb"))
        if n_batches >= 2:
            score("ARI_batch", lambda: 1 - abs(me.ari(
                adata, cluster_key="batch_cluster", label_key="batch")))
            score("NMI_batch", lambda: 1 - abs(me.nmi(
                adata, cluster_key="batch_cluster", label_key="batch")))
        score("GC", lambda: me.graph_connectivity(adata, label_key="celltype"))
        score("iLISI", lambda: me.ilisi_graph(adata, batch_key="batch", type_="knn"))
        # opt-in: kBET runs in R and takes hours at 10-30k cells
        if slow_metrics:
            score("kBET", lambda: me.kBET(adata, batch_key="batch", label_key="celltype",
                                          type_="embed", embed="X_emb"))

    frame = pd.DataFrame.from_dict(out, orient="index", columns=["Value"])
    scored_clusters = want_clu and any(m in out for m in _CLUSTER_METRICS)
    frame.attrs["leiden_flavor"] = flavor
    frame.attrs["clustering"] = (None if not scored_clusters
                                 else "user" if cluster is not None else "search")
    return frame
