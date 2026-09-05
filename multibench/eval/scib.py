"""scib metric computation (clustering + batch), ported from qc/scib_metrics."""
from __future__ import annotations

import contextlib
import functools
import io
import warnings

import numpy as np
import pandas as pd

# Metrics that shell out to scib's prebuilt LISI helper binary.
_LISI_METRICS = ("cLISI", "iLISI")

#: C++ compilers tried, in order, when the shipped LISI helper cannot execute
_CXX_CANDIDATES = ("g++", "c++", "clang++")

#: the exact build line scib's own README gives for knn_graph.o
_CXX_FLAGS = ("-std=c++11", "-O3")

#: embeddings above this many cells announce the Leiden sweep on stderr when
#: ``verbose`` is None (the sweep then takes long enough to look like a hang)
_SWEEP_NOTICE_CELLS = 2000


def _find_cxx() -> str | None:
    """Path of the first C++ compiler on PATH, or None."""
    import shutil
    for name in _CXX_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _probe_lisi_binary(exe) -> str | None:
    """Run scib's LISI helper once with no arguments; None when it starts.

    Returns the problem as a string otherwise. ``"cannot be executed"`` is the
    verdict for an ``OSError`` at exec time (``Exec format error``: the
    shipped binary is Linux x86-64 and this is macOS or another architecture)
    - the case a rebuild from source fixes.
    """
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
        return None                      # it started; that is all we asked
    # Evidence that it ran at all: a clean exit, or anything on stdout (the
    # binary answers a bare invocation with its usage line). Judging by the
    # usage TEXT would turn a future wording change into a false alarm that
    # silently drops two metrics; judging by "did it produce anything" does
    # not, while still catching the loader failure (exit 127, stderr only)
    # and the silent crash (non-zero, no output) that scib discards today.
    if p.returncode == 0 or (p.stdout or "").strip():
        return None
    err = (p.stderr or "").strip().splitlines()
    return err[0] if err else f"{exe} exited {p.returncode} with no output"


def _rebuild_lisi_binary(exe, cpp, cxx: str) -> str | None:
    """Compile ``cpp`` to ``exe`` in place with ``cxx``; None on success, else why.

    This is scib's own build line (``g++ -std=c++11 -O3 -o knn_graph.o
    knn_graph.cpp``) run once, writing into scib's own ``knn_graph/`` folder
    so scib finds the binary where it looks for it.
    """
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
    """The warning for a LISI metric recorded as NaN: the cause and the manual
    build line (scib's own ``g++`` command), for when the automatic rebuild
    could not run or failed."""
    import pathlib as _pl
    import scib as _scib
    kg = _pl.Path(_scib.__file__).parent / "knn_graph"
    return (f"scib metric {name!r} needs scib's LISI helper binary, which "
            f"cannot run here: {problem}. Recording NaN. To fix it, rebuild the "
            f"binary from the source scib ships (needs a C++ compiler):\n"
            f"    g++ -std=c++11 -O3 -o {kg / 'knn_graph.o'} {kg / 'knn_graph.cpp'}")


@functools.lru_cache(maxsize=1)
def _lisi_helper_problem() -> str | None:
    """Why scib's LISI helper cannot run here, or None when it can.

    scib runs a prebuilt C++ binary with neither ``check=True`` nor any capture
    of its stderr, so a binary that fails to start is discovered much later - as
    a FileNotFoundError on an output file that was never written, once per
    metric, with the actual cause discarded. A loader failure (an older
    libstdc++ ahead of the system one, a foreign architecture) is invisible that
    way. Probing the binary once turns it into one message naming the cause.

    When the binary CANNOT BE EXECUTED at all (``Exec format error`` - scib
    ships a Linux x86-64 executable, so this is every macOS install), a C++
    compiler is on PATH (``g++`` / ``c++`` / ``clang++``) and scib ships
    ``knn_graph.cpp`` next to it, the helper is rebuilt in place ONCE with
    scib's own build line and probed again; cLISI/iLISI then compute instead
    of recording NaN. If the rebuild fails, the original problem is returned
    with the compiler's verdict appended, and the caller's warning still
    prints the ``g++`` line for a manual fix.

    Healthy behaviour, verified on Linux x86-64: exits 0 and prints its usage
    line when called with no arguments.
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
    if problem is None or "cannot be executed" not in problem:
        return problem
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




#: Leiden backends ``leiden_sweep`` / ``compute`` accept
LEIDEN_FLAVORS = ("igraph", "leidenalg")


_igraph_support: bool | None = None     # probed once per process; tests may set it


def igraph_flavor_available() -> bool:
    """Can this scanpy run ``sc.tl.leiden(flavor="igraph")``?

    scanpy accepts ``flavor=`` from 1.10 on (older releases forward the
    unknown keyword to leidenalg, which raises ``TypeError``), and the backend
    needs the ``igraph`` package. The answer is probed once per process.
    """
    global _igraph_support
    if _igraph_support is None:
        try:
            import inspect
            import igraph  # noqa: F401
            import scanpy as sc
            _igraph_support = "flavor" in inspect.signature(sc.tl.leiden).parameters
        except Exception:  # noqa: BLE001 - any import/probe failure means "no"
            _igraph_support = False
    return _igraph_support


_fallback_warned = False


def _resolve_flavor(flavor) -> str:
    """The Leiden backend that will actually run: ``flavor`` itself, else the
    configured default, downgraded to ``"leidenalg"`` when this scanpy cannot
    run the igraph backend.

    ``None`` reads ``config.DEFAULT.leiden_flavor`` (``"igraph"`` when the
    field is absent, e.g. an older ``Config``). Anything outside
    :data:`LEIDEN_FLAVORS` raises rather than silently running the slow
    backend under a misspelt name. ``"igraph"`` on a scanpy older than 1.10
    (or without the igraph package) falls back to ``"leidenalg"`` with ONE
    ``UserWarning`` per process - the metrics are the same, only slower.
    """
    global _fallback_warned
    if flavor is None:
        from .. import config
        flavor = getattr(config.DEFAULT, "leiden_flavor", "igraph")
    if flavor not in LEIDEN_FLAVORS:
        raise ValueError(
            f"unknown leiden flavor {flavor!r}; valid: {'|'.join(LEIDEN_FLAVORS)}")
    if flavor == "igraph" and not igraph_flavor_available():
        if not _fallback_warned:
            import scanpy as sc
            warnings.warn(
                f"scanpy {getattr(sc, '__version__', '?')} cannot run flavor='igraph' "
                f"(needs scanpy>=1.10 and the igraph package); using leidenalg for "
                f"this process - same metrics, slower sweep", UserWarning, stacklevel=3)
            _fallback_warned = True
        return "leidenalg"
    return flavor


def _leiden(adata, resolution: float, key_added: str, flavor: str) -> None:
    """One Leiden clustering, on the backend ``flavor`` names.

    ``"igraph"`` is scanpy's igraph backend (``n_iterations=2``,
    ``directed=False`` - the settings scanpy documents for it; 7.7x faster
    than leidenalg on 20k cells). ``"leidenalg"`` is the classic backend
    scib's own ``cluster_optimal_resolution`` runs. Both write the labels to
    ``adata.obs[key_added]``.
    """
    import scanpy as sc
    if flavor == "igraph":
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added,
                     flavor="igraph", n_iterations=2, directed=False)
        return
    with warnings.catch_warnings():
        # scanpy nags every leidenalg call to switch to igraph; here leidenalg
        # is an explicit choice (config.DEFAULT.leiden_flavor / flavor=), so
        # that one message is noise. Only it is silenced - anything else
        # scanpy has to say still surfaces.
        warnings.filterwarnings("ignore", message=".*igraph.*implementation of leiden.*",
                                category=UserWarning)
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added,
                     flavor="leidenalg")


def _build_adata(emb, celltype, cluster, batch):
    # anndata is an evaluation-only dependency: importing it lazily keeps
    # `import multibench` working on environments (e.g. Colab) that have
    # no anndata installed and only use discovery + plotting.
    import anndata as ad
    adata = ad.AnnData(np.asarray(emb, dtype=float))
    adata.obsm["X_emb"] = adata.X
    adata.obs["celltype"] = pd.Categorical(np.asarray(celltype).astype(str))
    if cluster is not None:
        adata.obs["cluster"] = pd.Categorical(np.asarray(cluster))
    # kBET converts this to an R factor via rpy2, which refuses non-string
    # categories ("Converting pandas Category series to R factor is only
    # possible when categories are strings"). Integer batch ids are the
    # natural thing for a caller to pass, so coerce here rather than making
    # every caller remember.
    adata.obs["batch"] = pd.Categorical(np.asarray(batch).astype(str))
    return adata



def leiden_sweep(emb, *, flavor=None):
    """Run the scIB optimal-resolution Leiden sweep ONCE, reusably.

    ``cluster_optimal_resolution`` clusters the embedding at 10 resolutions and
    keeps whichever maximises NMI against a label vector. The clustering at a
    given resolution depends only on the embedding - the label vector enters
    solely through the argmax - so ranking N candidate label orderings needs one
    sweep, not N. On D52 cross (6 candidate orderings, 23,478 cells) the per-
    candidate sweeps cost ~250s each and dominated the whole evaluation.

    Parameters
    ----------
    emb : array-like
        Embedding, cells x dims.
    flavor : {"igraph", "leidenalg"}, keyword-only, optional
        Leiden backend. ``None`` (default) reads ``config.DEFAULT.leiden_flavor``
        (``"igraph"``: scanpy's igraph backend with ``n_iterations=2``,
        ``directed=False``; ``"leidenalg"``: the classic backend scib itself
        runs).

    Returns
    -------
    tuple
        ``(adata, keys)``. The caller assigns ``adata.obs["celltype"]`` and
        scores with scib's OWN ``nmi``/``ari`` against each key, so the
        selection protocol stays identical to ``cluster_optimal_resolution``'s
        rather than being reimplemented.

    Raises
    ------
    ValueError
        ``flavor`` is neither ``"igraph"`` nor ``"leidenalg"``.
    """
    import scanpy as sc
    from scib.metrics.clustering import get_resolutions

    flavor = _resolve_flavor(flavor)
    # anndata is an evaluation-only dependency: importing it lazily keeps
    # `import multibench` working on environments (e.g. Colab) that have
    # no anndata installed and only use discovery + plotting.
    import anndata as ad
    adata = ad.AnnData(np.asarray(emb, dtype=float))
    adata.obsm["X_emb"] = adata.X
    sc.pp.neighbors(adata, use_rep="X_emb")
    keys = []
    for res in get_resolutions(n=10, max=2):
        key = f"_mb_res_{res}"
        _leiden(adata, res, key, flavor)
        keys.append(key)
    return adata, keys


def _isolated_labels_f1(adata, label_key, batch_key, embed, iso_threshold,
                        precomputed_keys=None, flavor=None):
    """Isolated-label F1, identical to scib's but without the per-label re-clustering.

    ``scib.metrics.isolated_labels_f1`` calls ``cluster_optimal_resolution`` once
    per isolated label, and every call recomputes the kNN graph and a full Leiden
    resolution sweep. The clustering at a given resolution does not depend on
    which label is being scored - only the F1 target does. Under our convention
    that EVERY label is isolated, scib therefore repeats the same 10 clusterings
    once per label: on a 28-label dataset that is 280 Leiden runs where 10 suffice,
    which is ~90 min on 23k cells.

    Each resolution is clustered once here, then each label takes its max F1 over
    all resolutions - the same quantity scib's per-label optimisation returns.
    ``tests/test_eval_isolated_f1.py`` pins this to scib's own result.
    """
    import scanpy as sc
    from sklearn.metrics import f1_score
    from scib.metrics.clustering import get_resolutions
    from scib.metrics.isolated_labels import get_isolated_labels

    labels = get_isolated_labels(adata, label_key, batch_key, iso_threshold,
                                 verbose=False)
    if len(labels) == 0:
        return float("nan")

    if precomputed_keys:
        # the caller already ran the identical 10-resolution sweep on this
        # adata (same graph, same resolutions) - re-running it here doubled
        # the whole evaluation
        keys = list(precomputed_keys)
        _owned = False
    else:
        flavor = _resolve_flavor(flavor)
        sc.pp.neighbors(adata, use_rep=embed)
        resolutions = get_resolutions(n=10, max=2)
        keys = []
        for res in resolutions:
            key = f"_mb_isof1_{res}"
            _leiden(adata, res, key, flavor)
            keys.append(key)
        _owned = True

    try:
        scores = []
        for label in labels:
            y_true = (adata.obs[label_key] == label).values
            best = 0.0
            for key in keys:
                col = adata.obs[key]
                for cluster in col.unique():
                    # argument order mirrors scib's max_f1 exactly; F1 is
                    # symmetric under swapping y_true/y_pred, but match it anyway
                    f1 = f1_score((col == cluster).values, y_true)
                    if f1 > best:
                        best = f1
            scores.append(best)
    finally:
        if _owned:
            for key in keys:
                if key in adata.obs:
                    del adata.obs[key]

    return float(np.mean(scores))

def compute(emb, celltype, cluster, batch, group: str = "clustering",
            slow_metrics: bool = False, only=None, *,
            verbose: bool | None = None, flavor=None) -> pd.DataFrame:
    """Compute scib metrics for one embedding.

    Parameters
    ----------
    emb : array-like
        Embedding, cells x dims.
    celltype : array-like
        Cell-type label per cell.
    cluster : array-like or None
        Precomputed cluster assignment; ``None`` derives one with the scIB
        optimal-resolution Leiden sweep (10 resolutions, argmax NMI).
    batch : array-like
        Batch label per cell (a constant vector is fine for clustering-only).
    group : {"clustering", "batch", "all"}
        Metric family.
    slow_metrics : bool
        Also compute kBET (shells out to R).
    only : collection of str, optional
        Restrict the computation to the named metrics, e.g. ``only={"ARI"}``.
        Everything not named is skipped rather than computed and discarded,
        and the Leiden sweep is skipped too when no requested metric needs it
        (ARI, NMI, iF1 do). This exists because ranking candidate label
        orderings needs ARI alone, and paying for iF1/cLISI/iLISI once per
        candidate made that ranking cost more than the entire rest of the
        evaluation.
    verbose : bool, keyword-only, optional
        Print one stderr line when the Leiden sweep starts. ``None``
        (default): only for embeddings with more than 2,000 cells; ``True``
        always; ``False`` never.
    flavor : {"igraph", "leidenalg"}, keyword-only, optional
        Leiden backend for the resolution sweep. ``None`` (default) reads
        ``config.DEFAULT.leiden_flavor``; see :func:`leiden_sweep`.

    Returns
    -------
    pandas.DataFrame
        ``metric.csv``-shaped: index = metric, one column ``Value``.
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
    n_ct = len(np.asarray(celltype))
    if n_ct != n:
        raise ValueError(
            f"input length mismatch: emb has {n} cells, celltype has {n_ct}"
        )
    if cluster is not None:
        n_cl = len(np.asarray(cluster))
        if n_cl != n:
            raise ValueError(
                f"input length mismatch: emb has {n} cells, cluster has {n_cl}"
            )
    if batch is not None:
        n_ba = len(np.asarray(batch))
        if n_ba != n:
            raise ValueError(
                f"input length mismatch: emb has {n} cells, batch has {n_ba}"
            )

    adata = _build_adata(emb, celltype, cluster, batch)
    sc.pp.neighbors(adata, use_rep="X_emb")

    # When clustering metrics are requested but no precomputed clustering was
    # supplied, derive one from the embedding with scIB optimal-resolution
    # Leiden: sweep resolutions and keep the assignment that maximises NMI
    # vs. the cell-type labels. This is the standard scib clustering protocol
    # and is what lets evaluate() run directly on a method's embedding output.
    _needs_clustering = only is None or bool(only & {"ARI", "NMI"})
    _needs_isof1 = group in ("clustering", "all") and (only is None or "iF1" in only)
    _sweep_keys = []
    if group in ("clustering", "all") and (
            (cluster is None and _needs_clustering) or _needs_isof1):
        # ONE 10-resolution Leiden sweep serves both consumers: the optimal-
        # resolution cluster choice (argmax NMI vs the labels - the same
        # protocol as scib.cluster_optimal_resolution) and the isolated-label
        # F1, which needs every resolution's assignment. Running the two
        # independently used to double the whole evaluation. Quietly: scanpy
        # narrates each resolution otherwise.
        from scib.metrics.clustering import get_resolutions
        flavor = _resolve_flavor(flavor)
        if verbose or (verbose is None and n > _SWEEP_NOTICE_CELLS):
            import sys
            needs = [m for m in ("ARI", "NMI", "iF1") if only is None or m in only]
            print(f"scIB clustering metrics: Leiden resolution sweep (10 "
                  f"resolutions, flavor={flavor}) over {n:,} cells for "
                  f"{', '.join(needs)} - typically 30-60 s per 3,000 cells with "
                  f"leidenalg, several times faster with igraph; pass "
                  f"clustering= or metrics=[...] without ARI/NMI/iF1 to skip it",
                  file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for res in get_resolutions(n=10, max=2):
                key = f"_mb_res_{res}"
                _leiden(adata, res, key, flavor)
                _sweep_keys.append(key)
            if cluster is None and _needs_clustering:
                best_key, best_nmi = None, -1.0
                for k in _sweep_keys:
                    v = float(me.nmi(adata, cluster_key=k, label_key="celltype"))
                    if v > best_nmi:
                        best_nmi, best_key = v, k
                adata.obs["cluster"] = adata.obs[best_key].values
    out: dict[str, float] = {}

    def _safe(name, fn):
        """Compute one metric defensively: record NaN (with a warning) if it fails.

        Some scib metrics (notably the LISI graph metrics) rely on a prebuilt
        binary that may not load on every platform (macOS arm64, older glibc).
        Degrading gracefully lets evaluate() still return every metric that does
        compute, instead of failing the whole evaluation on one optional metric.
        """
        if only is not None and name not in only:
            return
        if name in _LISI_METRICS:
            problem = _lisi_helper_problem()
            if problem:
                warnings.warn(_lisi_fallback_message(name, problem))
                out[name] = float("nan")
                return
        try:
            # scib prints per-chunk progress from inside some metrics (the LISI
            # family especially) and emits third-party deprecation warnings;
            # neither carries information for the caller, so both are swallowed
            # here. Our own could-not-compute warning below stays visible.
            with contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = float(fn())
        except Exception as exc:  # noqa: BLE001 - report and continue
            warnings.warn(
                f"scib metric {name!r} could not be computed "
                f"({type(exc).__name__}: {str(exc)[:160]}); recording NaN."
            )
            out[name] = float("nan")

    want_clu = group in ("clustering", "all")
    want_bat = group in ("batch", "all")

    if want_clu:
        _safe("ARI", lambda: me.ari(adata, cluster_key="cluster", label_key="celltype"))
        _safe("NMI", lambda: me.nmi(adata, cluster_key="cluster", label_key="celltype"))
        _safe("ASW", lambda: me.silhouette(adata, label_key="celltype", embed="X_emb"))
        # Isolated-label convention: treat EVERY cell type as isolated and score
        # them all. scib's default picks only types confined to few batches, and
        # returns NOTHING when every type appears in every batch (it short-circuits
        # on iso_threshold == n_batches) - so a well-balanced dataset silently got
        # no iASW/iF1 at all. n_batches + 1 clears that check and admits every label.
        _iso = int(adata.obs["batch"].nunique()) + 1
        _safe("iASW", lambda: me.isolated_labels_asw(adata, batch_key="batch", label_key="celltype",
                                                     embed="X_emb", iso_threshold=_iso))
        _safe("iF1", lambda: _isolated_labels_f1(adata, label_key="celltype",
                                                 batch_key="batch", embed="X_emb",
                                                 iso_threshold=_iso,
                                                 precomputed_keys=_sweep_keys or None,
                                                 flavor=flavor))
        _safe("cLISI", lambda: me.clisi_graph(adata, label_key="celltype", type_="embed", use_rep="X_emb"))
    if want_bat:
        _safe("ASW_batch", lambda: me.silhouette_batch(adata, batch_key="batch", label_key="celltype", embed="X_emb"))
        _safe("GC", lambda: me.graph_connectivity(adata, label_key="celltype"))
        _safe("iLISI", lambda: me.ilisi_graph(adata, batch_key="batch", type_="embed", use_rep="X_emb"))
        # kBET shells out to R once per method and dominates the runtime of a
        # sweep (hours per dataset at 10-30k cells), so it is opt-in. Everything
        # it needs IS installed - pass slow_metrics=True to compute it.
        if slow_metrics:
            _safe("kBET", lambda: me.kBET(adata, batch_key="batch", label_key="celltype", type_="embed", embed="X_emb"))

    return pd.DataFrame.from_dict(out, orient="index", columns=["Value"])
