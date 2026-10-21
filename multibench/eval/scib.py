"""scIB metric computation (clustering + batch), after upstream
``evaluation_pipelines/scib_metrics/scib_metrics.py``.

The protocol is the article's: Leiden clusters searched to the number of cell
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

from .. import _compat

#: metrics that run scib's prebuilt LISI helper binary
_LISI_METRICS = ("cLISI", "iLISI")

#: C++ compilers tried, in order, when the shipped LISI helper cannot execute
_CXX_CANDIDATES = ("g++", "c++", "clang++")

#: the flags of the build line in scib's ``knn_graph/README.md``
_CXX_FLAGS = ("-std=c++11", "-O3")

#: above this many cells the Leiden clustering is announced on stderr
#: (``verbose=None``): it then takes long enough to look like a hang
_SWEEP_NOTICE_CELLS = 2000

#: ``sinfonia.get_N_clusters``, which the article clusters with: the Leiden
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
    for name in _CXX_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


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
    """Compile ``cpp`` to ``exe`` in place with scib's own build line; None on
    success, else why. scib looks for the binary in its own ``knn_graph/``."""
    import subprocess
    import sys

    cmd = [cxx, *_CXX_FLAGS, "-o", str(exe), str(cpp)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{' '.join(cmd)} failed ({type(exc).__name__}: {exc})"
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip().splitlines()
        return f"{' '.join(cmd)} exited {p.returncode}" + (f": {err[-1]}" if err else "")
    print(f"multibench: rebuilt scib's LISI helper {exe.name} from source with "
          f"{cxx} (the shipped binary could not run here)", file=sys.stderr)
    return None


def _lisi_fallback_message(name: str, problem: str) -> str:
    """The warning for a LISI metric recorded as NaN: the cause and scib's own
    ``g++`` build line."""
    from pathlib import Path

    import scib
    kg = Path(scib.__file__).parent / "knn_graph"
    return (f"scib metric {name!r} needs scib's LISI helper binary, which "
            f"cannot run here: {problem}. Recording NaN. To fix it, rebuild the "
            f"binary from the source scib ships (needs a C++ compiler):\n"
            f"    g++ -std=c++11 -O3 -o {kg / 'knn_graph.o'} {kg / 'knn_graph.cpp'}")


@functools.lru_cache(maxsize=1)
def _lisi_helper_problem() -> str | None:
    """Why scib's LISI helper cannot run here, or None when it can.

    scib runs the prebuilt binary without checking it, so a binary that does
    not start (scib ships a Linux x86-64 executable linked against glibc 2.38)
    surfaces only as a missing output file, once per metric. One probe names
    the cause; with a C++ compiler and the shipped ``knn_graph.cpp`` the
    helper is rebuilt once in place and probed again.
    """
    from pathlib import Path

    try:
        import scib
    except Exception as exc:  # noqa: BLE001 - absence is the caller's problem
        return f"scib is not importable ({type(exc).__name__})"
    kg = Path(scib.__file__).parent / "knn_graph"
    exe = kg / "knn_graph.o"
    if not exe.is_file():
        return f"{exe} is missing from the scib installation"
    problem = _probe_lisi_binary(exe)
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


#: Leiden backends ``leiden_n_clusters`` / ``compute`` accept
LEIDEN_FLAVORS = ("igraph", "leidenalg")

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


_igraph_support: bool | None = None     # probed once per process; tests may set it


def igraph_flavor_available() -> bool:
    """Can this scanpy run ``sc.tl.leiden(flavor="igraph")``?

    scanpy accepts ``flavor=`` from 1.10 on and the backend needs the
    ``igraph`` package. The answer is probed once per process.
    """
    global _igraph_support
    if _igraph_support is None:
        try:
            import igraph  # noqa: F401
            _igraph_support = _scanpy_has_flavor_kw()
        except Exception:  # noqa: BLE001 - an igraph that fails to import means "no"
            _igraph_support = False
    return _igraph_support


_fallback_warned = False
_flavor_kw: bool | None = None


def _scanpy_has_flavor_kw() -> bool:
    """Does this scanpy accept ``sc.tl.leiden(flavor=...)`` (>= 1.10)? Probed
    once. Older ones fail on the keyword."""
    global _flavor_kw
    if _flavor_kw is None:
        try:
            from importlib.metadata import version
            from packaging.version import Version
            # by version, not by signature: tests replace sc.tl.leiden with
            # recorders whose signature says nothing about the real backend
            _flavor_kw = Version(version("scanpy")) >= Version("1.10")
        except Exception:  # noqa: BLE001
            _flavor_kw = False
    return _flavor_kw


def _resolve_flavor(flavor) -> str:
    """The Leiden backend that will run: ``flavor``, else the configured
    default, downgraded to ``"leidenalg"`` (one warning) where igraph cannot run."""
    global _fallback_warned
    if flavor is None:
        from .. import config
        flavor = config.DEFAULT.leiden_flavor
    if flavor not in LEIDEN_FLAVORS:
        raise ValueError(
            f"unknown leiden flavor {flavor!r}; valid: {'|'.join(LEIDEN_FLAVORS)}")
    if flavor == "igraph" and not igraph_flavor_available():
        if not _fallback_warned:
            from importlib.metadata import version
            _compat.warn(
                f"scanpy {version('scanpy')} cannot run flavor='igraph' "
                f"(needs scanpy>=1.10 and the igraph package); using leidenalg for "
                f"this process - same metrics, slower clustering", UserWarning)
            _fallback_warned = True
        return "leidenalg"
    return flavor


def _neighbors(adata, rep: str) -> None:
    """``sc.pp.neighbors`` on ``adata.obsm[rep]``, silencing the
    ``SparseEfficiencyWarning`` scipy raises on some embeddings."""
    import scanpy as sc
    from scipy.sparse import SparseEfficiencyWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        sc.pp.neighbors(adata, use_rep=rep)


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
    """An AnnData with the kNN graph the article clusters on (SINFONIA.py):
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


def _search_n_clusters(adata, n: int, key: str, flavor: str) -> bool:
    """Leiden into ``adata.obs[key]`` at the resolution that gives ``n`` clusters,
    bisected as ``sinfonia.get_N_clusters`` does. False when no step gave ``n``:
    the last step's clustering is then kept."""
    lo, hi = _SEARCH_RANGE
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(_SEARCH_STEPS):
            res = lo + (hi - lo) / 2
            _leiden(adata, res, key, flavor)
            found = adata.obs[key].nunique()
            if found > n:
                hi = res
            elif found < n:
                lo = res
            else:
                return True
    return False


def _n_clusters(graph, n: int, flavor: str, per: str | None = None) -> np.ndarray:
    """The searched clusters of ``graph`` as strings: those of the last step when no
    resolution gave ``n``, as the article's search. ``per`` is kept for callers."""
    key = "_mb_search"
    _search_n_clusters(graph, n, key, flavor)
    out = np.asarray(graph.obs[key].astype(str))
    del graph.obs[key]
    return out


def leiden_n_clusters(emb, n: int, *, flavor=None) -> np.ndarray:
    """Cluster an embedding with Leiden at the resolution that gives ``n`` clusters.

    The article's clustering: a kNN graph with scanpy's defaults (on the
    first 50 principal components when the embedding has more than 50
    dimensions), and the resolution bisected on [0, 3] for at most 30 steps
    until there are ``n`` clusters, as ``sinfonia.get_N_clusters``.

    Parameters
    ----------
    emb : array-like
        Embedding, cells x dims. NaN counts as 0.
    n : int
        Number of clusters wanted, such as the number of cell types.
    flavor : {None, "igraph", "leidenalg"}
        Leiden backend. ``None`` (default) reads ``config.DEFAULT.leiden_flavor``
        (``"igraph"``: scanpy's igraph backend with ``n_iterations=2``,
        ``directed=False``; ``"leidenalg"``: scanpy's default before 1.10).

    Returns
    -------
    numpy.ndarray
        One cluster id (a string) per cell; those of the last step when no
        resolution gave ``n``.

    Raises
    ------
    ValueError
        ``flavor`` is neither ``"igraph"`` nor ``"leidenalg"``.
    """
    flavor = _resolve_flavor(flavor)
    return _n_clusters(_clustering_graph(emb), int(n), flavor)


def _isolated_labels_f1(adata, label_key, batch_key, cluster_key, iso_threshold):
    """Isolated-label F1 of the clustering ``adata.obs[cluster_key]``: for each
    isolated label, the best F1 of one cluster, averaged over labels. scib's
    ``isolated_labels_f1`` with that clustering precomputed (pinned in
    ``tests/test_eval_isolated_f1.py``)."""
    from sklearn.metrics import f1_score
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


def count_error(what: str, n: int, n_cells: int) -> str:
    """``'labels has 3,000 entries for 7,500 cells in the embedding.'``

    ``what`` is the argument of ``mtb.evaluate`` that holds ``n`` values:
    ``labels``, ``clusters`` or ``batch``.
    """
    return f"{what} has {n:,} entries for {n_cells:,} cells in the embedding."


def compute(emb, celltype, cluster, batch, group: str = "clustering",
            slow_metrics: bool = False, only=None, *,
            verbose: bool | None = None, flavor=None) -> pd.DataFrame:
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
        always; ``False`` never.
    flavor : {None, "igraph", "leidenalg"}
        Leiden backend. ``None`` (default) reads
        ``config.DEFAULT.leiden_flavor``; see :func:`leiden_n_clusters`.

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
    range, with values within ``1e-12`` of 1 recorded as ``1.0``. An iLISI of ``2.2e-16`` is therefore ``0.0``.
    """
    if only is not None:
        only = set(only)
    if group not in {"clustering", "batch", "all"}:
        raise ValueError(f"unknown group {group!r}; valid: clustering|batch|all")

    try:
        import scanpy as sc
        import scib.metrics as me
    except ImportError as e:
        raise RuntimeError(
            "metrics need scib and scanpy, which are not installed here - "
            "run: pip install scib scanpy"
        ) from e

    n = np.asarray(emb).shape[0]
    # named after evaluate()'s arguments
    for what, values in (("labels", celltype), ("clustering", cluster),
                         ("batch", batch)):
        if values is not None and len(np.asarray(values)) != n:
            raise ValueError(count_error(what, len(np.asarray(values)), n))

    adata = _build_adata(emb, celltype, cluster, batch)
    _neighbors(adata, "X_emb")

    want_clu = group in ("clustering", "all")
    want_bat = group in ("batch", "all")

    def wanted(names):
        return [m for m in names if only is None or m in only]

    n_batches = int(adata.obs["batch"].nunique())
    # the article's clusterings: one per cell type, and one per batch for
    # ARI_batch and NMI_batch (scored only with two or more batches)
    search_ct = want_clu and cluster is None and bool(wanted(_CLUSTER_METRICS))
    search_bat = want_bat and n_batches >= 2 and bool(wanted(_BATCH_CLUSTER_METRICS))
    leiden_ran = search_ct or search_bat
    if leiden_ran:
        flavor = _resolve_flavor(flavor)
        if verbose or (verbose is None and n > _SWEEP_NOTICE_CELLS):
            import sys
            from .. import config
            needs = ((wanted(_CLUSTER_METRICS) if search_ct else [])
                     + (wanted(_BATCH_CLUSTER_METRICS) if search_bat else []))
            names = config._and_list(needs)
            print(f"Clustering {n:,} cells for {names} with the {flavor} Leiden "
                  f"backend.", file=sys.stderr, flush=True)
    if search_ct or search_bat:
        graph = _clustering_graph(emb)
        if search_ct:
            adata.obs["cluster"] = pd.Categorical(_n_clusters(
                graph, adata.obs["celltype"].nunique(), flavor, "cell type"))
        if search_bat:
            adata.obs["batch_cluster"] = pd.Categorical(_n_clusters(
                graph, n_batches, flavor, "batch"))
    out: dict[str, float] = {}

    def _safe(name, fn):
        """Compute one metric; NaN with a warning when it fails (the LISI binary
        does not load on every platform)."""
        if only is not None and name not in only:
            return
        if name in _LISI_METRICS:
            problem = _lisi_helper_problem()
            if problem:
                _compat.warn(_lisi_fallback_message(name, problem))
                out[name] = float("nan")
                return
        try:
            # scib's progress prints and third-party deprecation warnings
            with contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = _tidy_value(name, fn())
        except Exception as exc:  # noqa: BLE001 - report and continue
            _compat.warn(
                f"scib metric {name!r} could not be computed "
                f"({type(exc).__name__}: {str(exc)[:160]}); recording NaN."
            )
            out[name] = float("nan")

    if want_clu:
        _safe("ARI", lambda: me.ari(adata, cluster_key="cluster", label_key="celltype"))
        _safe("NMI", lambda: me.nmi(adata, cluster_key="cluster", label_key="celltype"))
        _safe("ASW", lambda: me.silhouette(adata, label_key="cluster", embed="X_emb"))
        # every label counts as isolated: scib's default scores none when every
        # one is in every batch, and n_batches + 1 clears its check
        iso_threshold = n_batches + 1
        _safe("iASW", lambda: me.isolated_labels_asw(adata, batch_key="batch", label_key="cluster",
                                                     embed="X_emb", iso_threshold=iso_threshold))
        _safe("iF1", lambda: _isolated_labels_f1(adata, label_key="celltype",
                                                 batch_key="batch", cluster_key="cluster",
                                                 iso_threshold=iso_threshold))
        _safe("cLISI", lambda: me.clisi_graph(adata, label_key="celltype", type_="embed", use_rep="X_emb"))
    if want_bat:
        _safe("ASW_batch", lambda: me.silhouette_batch(adata, batch_key="batch", label_key="celltype", embed="X_emb"))
        if n_batches >= 2:
            _safe("ARI_batch", lambda: 1 - abs(me.ari(adata, cluster_key="batch_cluster", label_key="batch")))
            _safe("NMI_batch", lambda: 1 - abs(me.nmi(adata, cluster_key="batch_cluster", label_key="batch")))
        _safe("GC", lambda: me.graph_connectivity(adata, label_key="celltype"))
        _safe("iLISI", lambda: me.ilisi_graph(adata, batch_key="batch", type_="embed", use_rep="X_emb"))
        # opt-in: kBET runs in R and takes hours at 10-30k cells
        if slow_metrics:
            _safe("kBET", lambda: me.kBET(adata, batch_key="batch", label_key="celltype", type_="embed", embed="X_emb"))

    frame = pd.DataFrame.from_dict(out, orient="index", columns=["Value"])
    scored_clusters = want_clu and any(m in out for m in _CLUSTER_METRICS)
    frame.attrs["leiden_flavor"] = flavor if leiden_ran else None
    frame.attrs["clustering"] = (None if not scored_clusters
                                 else "user" if cluster is not None else "search")
    return frame
