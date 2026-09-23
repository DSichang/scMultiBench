"""Discovery: filter methods by need; enrich method_info with catalog metadata; cite."""
from __future__ import annotations

import functools
import inspect
import os
from pathlib import Path
from urllib.parse import quote as _quote

from .engine import registry, upstream, envs
from .engine import resolve as _resolve
from .engine.runner import _AUX_ROLES  # noqa: F401  (re-exported for back-compat)
from .engine.schema import AmbiguousVariantError, is_label_role
from .engine.schema import base_modality as _base_modality  # noqa: F401  (tests import it from here)


def _modality_types(spec) -> set[str]:
    """Union of base modality types consumed across all of a method's variants.

    Auxiliary roles (data_dir, source/target data & cty, out_dir) and label roles
    are excluded. Methods with no variants (declared stubs) yield an empty set.
    (Thin alias of ``MethodSpec.modality_types``, kept for callers/tests.)
    """
    return spec.modality_types


#: representation role tokens -> the ``atac=`` value they stand for
_REPRESENTATION_TOKENS = {"atac_peak": "peak", "atac_gas": "gene_activity"}


def _representations(toks) -> set[str]:
    """The ATAC representations (``peak``, ``gene_activity``) that normalised
    modality tokens name."""
    return {_REPRESENTATION_TOKENS[t] for t in toks if t in _REPRESENTATION_TOKENS}


#: the two ATAC representation roles; naming both selects the variants that
#: read both files (MultiMAP and Seurat_v3 read atac_peak.h5 and atac_gas.h5)
_BOTH_FILES = frozenset(_REPRESENTATION_TOKENS)


def _modality_filter(modalities, atac) -> tuple[set[str] | None, str | None]:
    """The one modality rule of ``find_methods`` and ``recommend``.

    Returns ``(base types every matching variant must consume, atac)``. A base
    token (``rna``, ``adt``/``protein``, ``atac``, or a numbered role such as
    ``rna1``) matches every role of that type. A representation token
    (``atac_peak``/``peak``/``peaks``, ``atac_gas``/``gas``/``gene_activity``)
    means base ``atac`` plus the representation the method reads
    (``method_info(m)['atac']``), the same as ``atac='peak'`` /
    ``atac='gene_activity'``: the role name alone does not say what the script
    reads (moETM takes ``atac_gas`` and reads peaks). Both representation
    tokens together select the variants that read both files
    (:func:`_reads_both_files`); ``atac`` is then left as given.

    Raises ``TypeError`` for a bare string, ``ValueError`` for an unknown
    token or ``atac`` value, or for a representation that contradicts
    ``atac=``.
    """
    if isinstance(modalities, str):
        raise TypeError(
            f"modalities must be a list of modality tokens, got the string "
            f"{modalities!r} - did you mean modalities=[{modalities!r}]?")
    atac = registry.check_atac(atac)
    if modalities is None:
        return None, atac
    toks = registry.normalize_modalities(modalities)
    reps = sorted(_representations(toks))
    if len(reps) == 1:
        if atac is not None and atac != reps[0]:
            raise ValueError(
                f"modalities={list(modalities)} selects methods that read "
                f"{reps[0].replace('_', ' ')} but atac={atac!r}; pass one of the two")
        atac = reps[0]
    return {_base_modality(t) for t in toks}, atac


def _reads_both_files(modalities) -> bool:
    """Whether ``modalities`` names both ATAC representations (``atac_peak``
    and ``atac_gas``, in any spelling): then only a variant with both roles
    matches."""
    if modalities is None or isinstance(modalities, str):
        return False
    return _BOTH_FILES <= set(registry.normalize_modalities(modalities))


def _variant_matches(v, category, want, needs_labels, atac) -> bool:
    """Whether one variant satisfies every per-variant filter at once.

    ``v`` may be ``None`` for a declared-but-unwired stub (no variants): it
    satisfies only the filters that ask for nothing a variant could supply
    (``needs_labels`` None/False, no ``modalities``, no ``atac``).
    """
    if v is None:
        return want is None and atac is None and not needs_labels
    if category and v.when.get("category") != category:
        return False
    if want is not None and not want <= v.modality_types:
        return False
    if needs_labels is not None and v.needs_labels != needs_labels:
        return False
    return True


def find_methods(category: str | None = None, *, task: str | None = None,
                 needs_labels: bool | None = None,
                 atac: str | None = None,
                 modalities: list[str] | set[str] | None = None,
                 runnable: bool | None = None,
                 tunable: bool | None = None) -> list[str]:
    """Return the method ids that match every filter you pass.

    A method matches when one of its variants meets ``category``,
    ``modalities``, ``needs_labels`` and ``atac`` together.

    Parameters
    ----------
    category : str | None
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``; ``None`` = any.
    task : str | None
        A task from ``mtb.list_tasks()``, e.g. ``"clustering"``; ``None`` = any.
    needs_labels : bool | None
        ``True`` = a matching variant needs cell-type labels; ``False`` = one
        runs without them; ``None`` = no filter.
    atac : str | None
        ATAC representation the script expects: ``"peak"`` or
        ``"gene_activity"``; ``None`` = no filter.
    modalities : list[str] | set[str] | None
        Modalities a variant must consume, e.g. ``["rna", "adt"]``;
        ``"atac_peak"`` / ``"atac_gas"`` also select by ATAC representation.
        ``None`` = no filter.
    runnable : bool | None
        ``True`` = methods wired with at least one variant (not a check of
        your host: see ``mtb.scan``); ``False`` = declared stubs; ``None`` =
        both.
    tunable : bool | None
        ``True`` = methods with command-line hyperparameters that
        ``run(params=...)`` can set; ``False`` = the rest (settings fixed in
        the script); ``None`` = both.

    Returns
    -------
    list[str]
        Method ids in registry order.

    Raises
    ------
    ValueError
        Unknown ``category``, ``task``, ``atac`` or modality token, or a
        token that contradicts ``atac``.
    TypeError
        ``modalities`` is a bare string, not a list.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.find_methods("vertical", modalities=["rna", "adt"])
    >>> mtb.find_methods("vertical", modalities=["rna", "adt"], needs_labels=False)
    >>> mtb.find_methods("diagonal", modalities=["rna", "atac_peak"])
    >>> mtb.find_methods(atac="gene_activity")
    >>> mtb.find_methods(tunable=True)

    Notes
    -----
    **Per-variant matching.** ``task``, ``runnable`` and ``tunable`` are
    method-level; the other four filters hold per variant. Two consequences:

    - ``find_methods('vertical', modalities=['rna', 'adt'], needs_labels=False)``
      keeps scMoMaT: its vertical rna+adt variant takes no labels; only its
      mosaic variant does.
    - ``find_methods('vertical', modalities=['rna', 'atac'])`` drops
      Multigrate: rna+atac exists only as a mosaic variant, so
      ``inputs_for(..., 'vertical', modalities=['rna', 'atac'])`` would raise.

    **Modality tokens.** A method matches when one variant reads at least
    the named modalities. ``mtb.scan`` and ``mtb.run_all`` use a stricter
    rule: a row's modalities must be exactly the named combination.

    A base token matches every role of its type: ``atac`` matches the
    ``atac``, ``atac_gas`` and ``atac_peak`` roles, ``rna`` matches ``rna1``,
    and ``protein`` is ``adt``.

    A representation token selects by what the method reads. ``atac_peak``
    (also ``peak``) is ``atac`` plus ``atac='peak'``; ``atac_gas`` (also
    ``gas``, ``gene_activity``) is ``atac`` plus ``atac='gene_activity'``.
    Both tokens together select the variants that read both files
    (MultiMAP, Seurat_v3). A representation that contradicts ``atac=``
    raises ``ValueError``.

    **Directory-fed methods.** scBridge is fed a directory and is judged by
    the bare filenames its variant names: ``rna.h5`` / ``atac_gas.h5`` make it
    an rna+atac method.

    **ATAC representation.** ``atac`` is what the upstream script reads,
    which is not always what its role name suggests: moETM, scMM and iPOLNG
    take the role ``atac_gas`` but read peaks. MultiMAP and Seurat_v3 read
    peaks and gene activity; they are listed under ``peak``.

    Only a variant with an ATAC input satisfies ``atac``: Multigrate reads
    peaks only in its mosaic rna+atac variant, so
    ``find_methods('vertical', atac='peak')`` omits it. Accepted spellings:
    ``peak`` / ``peaks`` and ``gene_activity`` / ``gene-activity`` / ``gas``,
    in any case.

    **Labels.** ``needs_labels`` here is per variant.
    ``method_info(m)['needs_labels']`` is the method-level flag (any variant
    needs labels); the per-variant answer is
    ``method_info(m)['supports'][i]['needs_labels']``.

    **Stubs.** A declared stub (a method with no variant) is dropped by any
    filter that asks something of a variant: ``category``, ``modalities``,
    ``atac``, ``needs_labels=True`` or ``tunable=True``.

    See Also
    --------
    mtb.list_methods : the same ids filtered by ``category`` only.

    mtb.list_tasks : the vocabulary of ``task=``.

    mtb.method_info : the per-variant ``supports`` table these filters are read from.

    mtb.scan : which of these methods can actually run on a given dataset.
    """
    registry.check_category(category)
    registry.check_task(task)
    want, atac = _modality_filter(modalities, atac)
    both = _reads_both_files(modalities)
    out = []
    for s in registry.load():
        if category and category not in s.categories:
            continue
        if task and task not in s.tasks:
            continue
        if runnable is not None and bool(s.variants) != runnable:
            continue
        if tunable is not None:
            has = any(v.tunable for v in s.variants)
            if has != tunable:
                continue
        # `atac` is declared once per method but only variants that take an
        # ATAC input can honour it; fold that into the per-variant test.
        if atac and s.atac != atac:
            continue
        cands = s.variants or [None]
        hits = [v for v in cands
                if _variant_matches(v, category, want, needs_labels, atac)
                and (not atac or v is None or v.consumes_atac)
                and (not both or v is not None and _BOTH_FILES <= {
                    r.rstrip("0123456789") for r in v.when.get("modalities") or []})]
        if not hits:
            continue
        out.append(s.id)
    return out


def _find_methods_filters_raise(fn):
    """Keyword guard of ``list_methods``: a ``find_methods`` filter passed to
    it raises a ``TypeError`` that names ``find_methods``; any other unknown
    keyword gets Python's own message. The signature stays ``(category)``."""
    @functools.wraps(fn)
    def guard(*args, **kwargs):
        extra = [k for k in kwargs if k != "category"]
        if extra:
            filters = set(inspect.signature(find_methods).parameters) - {"category"}
            unknown = [k for k in extra if k not in filters]
            if unknown:
                raise TypeError(
                    f"list_methods() got an unexpected keyword argument {unknown[0]!r}")
            keys = ", ".join(f"{k}=..." for k in extra)
            raise TypeError(
                f"list_methods() only takes category since 0.3.0; {keys} are find_methods "
                f"filters - use find_methods(category, {keys})")
        return fn(*args, **kwargs)
    return guard


@_find_methods_filters_raise
def list_methods(category: str | None = None) -> list[str]:
    """Return the registry method ids, optionally restricted to one category.

    Parameters
    ----------
    category : str | None
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``; ``None`` = every method.

    Returns
    -------
    list[str]
        Method ids in registry order.

    Raises
    ------
    ValueError
        Unknown ``category``; the message lists the valid ones.
    TypeError
        A keyword other than ``category``; see **Other keywords** in Notes.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.list_methods()                 # every registry id
    >>> mtb.list_methods("vertical")       # ids with a vertical variant

    Notes
    -----
    **Category membership.** A method is listed under a category when it has
    a variant wired for that category - the same set ``scan``, ``run_all``
    and ``find_methods(category=)`` dispatch.

    **Other keywords.** A ``find_methods`` filter such as ``task=`` or
    ``runnable=`` raises a ``TypeError`` that names the call to use
    instead: ``find_methods(category, task=...)``. Any other keyword raises
    Python's own ``unexpected keyword argument`` message.

    See Also
    --------
    mtb.find_methods : filter by task, modalities, labels, ATAC representation and more.

    mtb.list_categories : the four category tokens with a description of each.
    """
    return registry.list_methods(category)


def _effective(v) -> dict:
    """Upstream argparse defaults overlaid with what the wrapper actually emits.

    ``tunable[k]['default']`` stays the upstream default (documented contract);
    ``defaults`` is what the package passes on the command line; ``effective`` is
    the merge - the value the script will really run with when the caller
    passes no ``params``.
    """
    eff = {k: (t or {}).get("default") for k, t in v.tunable.items()}
    eff.update(dict(v.params))
    return eff


def _reference_batch(variant):
    """Batch ``N`` of a variant's ``reference`` const ``data<N>``, else ``None``."""
    ref = next((a.const for a in variant.args
                if getattr(a, "role", None) == "reference" and a.const), None)
    ref = str(ref) if ref is not None else ""
    return int(ref[4:]) if ref.startswith("data") and ref[4:].isdigit() else None


def method_info(method: str, *, verbose: bool = False) -> dict:
    """Return everything known about one method as a flat dict.

    Parameters
    ----------
    method : str
        Registry method id, e.g. ``"Matilda"``; see ``mtb.list_methods()``.
    verbose : bool
        ``True`` adds ``notes_long``, the long upstream-audit notes.

    Returns
    -------
    dict
        One flat record. Read ``supports`` (category/modality combinations it
        runs), ``params`` (what ``run(params=...)`` can change), ``runtime``
        (to size a sweep) and ``needs_labels``; all keys are in Notes.

    Raises
    ------
    KeyError
        Unknown method id (the message suggests a close match), or a declared stub.

    Examples
    --------
    >>> import multibench as mtb
    >>> info = mtb.method_info("Matilda")
    >>> # one entry per variant: category, modalities, labels ...
    >>> info["supports"]
    >>> info["runtime"]["tier"], info["runtime"]["worst_sec"]
    >>> info["params"]["vertical:rna+adt"]["tunable"]

    Notes
    -----
    **Key reference.** The dict merges the registry spec, the provenance
    record (repository, version, paper) and the observed runtime.

    - ``id`` - the registry id.
    - ``language`` - ``'python'`` or ``'R'``.
    - ``categories`` / ``tasks`` - the categories it has a variant for and
      the tasks it serves.
    - ``env`` - the conda env ``run`` executes it in (a shared group env or
      the method's own).
    - ``atac`` - the ATAC representation the script expects (``'peak'`` /
      ``'gene_activity'``), or ``None``.
    - ``needs_labels`` - method-level label flag; see **Labels** below.
    - ``status`` - see **Status**.
    - ``setup_hint`` - free-text setup advice, or ``''`` when there is none.
    - ``variants`` - the distinct upstream entrypoints, in order.
    - ``driver`` - the package-side wrapper actually executed, or ``None``
      when the upstream script runs directly.
    - ``scripts_url`` - the method's ``tools_scripts`` folder in the
      scMultiBench repository, percent-encoded.
    - ``repo_url`` / ``version`` - the upstream repository and the version
      the benchmark ran.
    - ``reference`` - ``{doi, title, authors, journal, year}`` or ``None``;
      ``mtb.cite`` formats it.
    - ``notes`` - a short summary of the method.
    - ``supports`` - one entry per variant: ``category``, ``modalities``,
      ``output_kind``, ``n_tunable``, ``needs_labels``, ``labels`` (the
      label roles the variant reads, e.g. ``['cty']`` / ``['rna_cty']`` /
      ``[]``) and ``reference_batch`` (the batch a variant uses as its fixed
      reference, e.g. 3 for StabMap in cross; ``None`` elsewhere).
    - ``params`` - keyed per variant as ``'category:mods'``, each with
      ``defaults``, ``tunable`` and ``effective`` (see ``mtb.params_for``).
    - ``fixed_in_script`` / ``upstream_knobs`` / ``upstream_url`` - what the
      script pins and what its library documents (see ``mtb.params_for``).
    - ``runtime`` - observed cost; see **Runtime**.
    - ``cpu_params`` / ``requires_gpu`` / ``gpu_evidence`` - see **GPU and
      CPU**.
    - ``notes_long`` (``verbose=True`` only) - the raw upstream-knob audit
      prose, ``None`` for methods outside the audit.
    - Not in this dict: the paper-only catalog columns (``deep_learning``,
      ``output``); read them from ``mtb.catalog.methods()``.

    **Runtime.** ``runtime`` is ``{"tier", "worst_sec", "observed", "host",
    "note"}``, what this method has been observed to cost:

    - ``tier`` - ``fast`` (<5 min), ``medium`` (5-30 min), ``slow``
      (30 min-2 h), ``very_slow`` (>2 h) or ``unknown`` (never measured:
      ``worst_sec`` is ``None``, ``observed`` empty).
    - ``worst_sec`` - the slowest observation, in seconds.
    - ``observed`` - one ``{dataset, cells, sec, source}`` per measurement;
      ``cells`` is ``None`` when not recorded; ``source`` is ``manual``,
      ``summary_csv`` (the shipped re-run sweeps) or ``recorded`` (the
      recorded end-to-end runs).
    - ``host`` / ``note`` - ``'gpu'`` and a sentence saying the times come
      from the GPU benchmark host.

    These are measurements, not predictions: use them to choose a sensible
    ``run_all(timeout=...)``, not to promise a finish time.

    **Labels.** ``needs_labels`` is the method-level flag: ``True`` when
    any variant takes a cell-type-label (``cty``) role as a required input.
    It is not per category: scMoMaT is True because its mosaic variant takes
    ``cty1..3``, while its vertical and cross variants take no labels. For
    the per-variant answer read ``supports[i]['needs_labels']``;
    ``find_methods(needs_labels=...)`` filters per variant.

    **Status.** ``status`` is the registry's wiring status. ``'verified'``
    means the command template was cross-checked against the upstream entrypoint
    and the method was executed end to end on a reference dataset;
    ``'declared'`` = wired but not run.

    **GPU and CPU.** ``cpu_params``, ``requires_gpu`` and ``gpu_evidence``
    are the GPU/CPU contract of the upstream script, read from its source:

    - ``cpu_params`` - the command-line values that turn CUDA off in a script
      that has it on by default (``{}`` for most methods): ``{'use_cuda': ''}``
      for scJoint (its argparse ``--use_cuda`` is ``type=bool``, so only the
      empty string is false), ``{'device': 'cpu'}`` for scMDC. On a host
      without an NVIDIA GPU (``mtb.env.host_has_gpu()`` False) ``run`` merges
      them into ``params`` unless the caller set the key.
    - ``requires_gpu`` - ``True`` for a script that calls CUDA
      unconditionally (no flag, no ``torch.cuda.is_available()`` fallback).
      On a GPU-less host ``run`` refuses such a method with ``OSError``
      before launching and ``scan`` reports it ``env_ok=False``.
    - ``gpu_evidence`` - the ``file:line`` of that CUDA call, else ``None``.

    None of the three says anything about the CPU archive of the method's env
    (``mtb.env.install(..., flavor=...)``).

    See Also
    --------
    mtb.params_for : the hyperparameters of one variant, with upstream defaults.

    mtb.find_methods : filter methods by category, modalities, labels, ATAC representation.

    mtb.cite : the paper to cite for a method.

    mtb.env.recipe : the hand-written environment recipe of a method.
    """
    s = registry.get(method)
    # scripts_url, variants and params all come from the variants
    if not s.variants:
        raise KeyError(f"{method}: no variants (declared stub)")
    ref = s.reference or {}
    info = {
        "id": s.id, "language": s.language, "categories": s.categories,
        # `env` is the conda env run() actually executes in: the shared group
        # env, or the method's own scmb_<method> env (see engine.envs.group_for).
        "tasks": s.tasks, "env": envs.group_for(s.id), "atac": s.atac,
        "needs_labels": s.needs_labels, "status": s.status,
        "setup_hint": s.setup_hint,
        # distinct entrypoints, in declaration order (several variants may share
        # one script)
        "variants": list(dict.fromkeys(v.entrypoint for v in s.variants)),
        # package-relative wrapper the runner executes instead of the entrypoint
        # (it source()s/imports the unmodified upstream script); None = the
        # upstream script itself is run
        "driver": next((v.driver for v in s.variants if v.driver), None),
        # folder of this method's unmodified upstream scripts in the
        # scMultiBench repository, percent-encoded ('online iNMF' has a space)
        "scripts_url": ("https://github.com/PYangLab/scMultiBench/tree/main/"
                        + _quote("/".join(s.variants[0].entrypoint.split("/")[:2]),
                                 safe="/")),
        # provenance (engine/references.yaml): the upstream repository / docs, the
        # version the benchmark ran, and the paper to cite (see mtb.cite)
        "repo_url": ref.get("repo_url") or upstream.knobs_for(s.id)["upstream_url"],
        "version": ref.get("version"),
        "reference": dict(ref["reference"]) if ref.get("reference") else None,
        # What this method can actually be dispatched for. Methods like Multigrate
        # support several integration categories, each with its own modality
        # combination - this is the list to pass to run()/inputs_for().
        "supports": [{"category": v.when.get("category"),
                      "modalities": list(v.when.get("modalities", [])),
                      "output_kind": v.output.kind,
                      "n_tunable": len(v.tunable),
                      "needs_labels": v.needs_labels,
                      "labels": [r for r in v.roles() if is_label_role(r)],
                      "reference_batch": _reference_batch(v)}
                     for v in s.variants],
        # what the caller may pass to run(params=...): see params_for()
        "params": {_variant_key(v): {"defaults": dict(v.params),
                                     "tunable": dict(v.tunable),
                                     "effective": _effective(v)}
                   for v in s.variants},
    }
    # An empty `tunable` means the script exposes nothing on its command line,
    # not that the method has no hyperparameters: also report what the script
    # pins and what its library documents.
    up = upstream.knobs_for(s.id)
    info["fixed_in_script"] = up["fixed_in_script"]
    info["upstream_knobs"] = up["upstream_knobs"]
    info["upstream_url"] = up["upstream_url"]
    # `notes` is the curated one-line summary; verbose=True adds the long
    # audit notes as notes_long.
    info["notes"] = ref.get("summary") or None
    # observed cost (engine/runtimes.yaml); the tier scan() reports per row
    info["runtime"] = _runtime_hint(s.id)
    # the GPU/CPU contract of the upstream script (methods.yaml, read from
    # its source): what run() merges on a GPU-less host, and what it refuses
    info["cpu_params"] = dict(s.cpu_params)
    info["requires_gpu"] = bool(s.requires_gpu)
    info["gpu_evidence"] = s.gpu_evidence or None
    if verbose:
        info["notes_long"] = up["notes"]
    return info


# ------------------------------------------------------------ observed runtimes
_RUNTIMES_YAML = Path(__file__).resolve().parent / "engine" / "runtimes.yaml"


@functools.lru_cache(maxsize=1)
def _runtimes() -> dict:
    """Observed per-method runtimes (reference data; see engine/runtimes.yaml)."""
    if not _RUNTIMES_YAML.exists():
        return {}
    import yaml
    with open(_RUNTIMES_YAML) as fh:
        return yaml.safe_load(fh) or {}


def _runtime_hint(method: str) -> dict:
    """``method_info(method)["runtime"]``: ``{"tier", "worst_sec", "observed", "host", "note"}``.

    ``method`` must be a registry id: a typo raises ``KeyError`` with a
    did-you-mean hint. A known but never-measured method (e.g. a declared
    stub) returns tier ``"unknown"`` with ``worst_sec=None`` and an empty
    ``observed`` list. See :func:`method_info` for the meaning of the fields.
    """
    registry.check_method(method)
    rt = dict(_runtimes().get(method, {"tier": "unknown", "worst_sec": None,
                                       "observed": []}))
    # every observation in runtimes.yaml was taken on the GPU benchmark host
    rt["host"] = "gpu"
    rt["note"] = ("times observed on the benchmark host (NVIDIA RTX 4090); on a "
                  "CPU-only host expect training methods to take many times longer")
    return rt


def _variant_key(v) -> str:
    """Stable human-readable key for a variant: 'category:mod1+mod2'."""
    mods = "+".join(v.when.get("modalities", [])) or "-"
    return f"{v.when.get('category')}:{mods}"


def params_for(method: str, category: str | None = None,
               modalities: list[str] | set[str] | None = None, *,
               dataset: str | None = None,
               data_path: Path | str | None = None) -> dict:
    """Return the hyperparameters of one method variant.

    Call it before ``run(params=...)``, ``run_all(params=...)`` or ``sweep`` to
    see what a method accepts and what it runs with when you pass nothing.

    Parameters
    ----------
    method : str
        Registry method id, e.g. ``"Matilda"``.
    category : str | None
        Integration category of the variant: ``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``; ``None`` = infer it from the other arguments.
    modalities : list[str] | set[str] | None
        Modality tokens of the variant, e.g. ``["rna", "adt"]``; ``None`` =
        infer it from the other arguments.
    dataset : str | None
        Dataset folder that settles an ambiguous selection: the one variant
        whose input files it holds.
    data_path : Path | str | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``.

    Returns
    -------
    dict
        ``method``, ``variant`` and the hyperparameters. Read ``tunable`` (what
        the script accepts) and ``effective`` (what a run uses); all keys are
        listed in Notes.

    Raises
    ------
    KeyError
        Unknown method, a declared stub, or no variant for ``category`` /
        ``modalities``.
    AmbiguousVariantError
        Several variants fit; the message spells out the call that selects one.
    ValueError
        Unknown ``category`` or modality token.

    Examples
    --------
    >>> import multibench as mtb
    >>> p = mtb.params_for("Matilda", "vertical", ["rna", "adt"])
    >>> # upstream default vs what a run uses
    >>> p["tunable"]["device"]["default"], p["effective"]["device"]
    >>> mtb.params_for("Matilda", dataset="D11")  # the folder picks the variant
    >>> mtb.params_for("scBridge", "diagonal")  # a data_dir variant: no modalities

    Notes
    -----
    **Key reference.**

    - ``method`` / ``variant`` - the registry id and the selected variant as
      ``'category:mods'`` (``mods`` is ``-`` for a ``data_dir`` variant,
      e.g. ``'diagonal:-'``).
    - ``defaults`` - parameters the package emits on every run. Override them
      with ``run(..., params={...})``; the override is merged over these.
    - ``tunable`` - the parameters the upstream script accepts on its command
      line, as ``{name: {"default": ..., "type": ...}}``. The ``default`` here
      is the upstream argparse default, not necessarily what a wrapper run
      uses.
    - ``effective`` - ``tunable`` defaults overlaid with ``defaults``: the
      value each knob really takes when you pass no ``params``.
    - ``fixed_in_script`` - the values the script pins, each with the
      ``file:line`` that pins it.
    - ``upstream_knobs`` - what the wrapped library documents (with its own
      defaults), unreachable without editing the script.
    - ``upstream_url`` - the upstream source or docs page those knobs were
      read from, or ``None``.

    **Methods with nothing to tune.** An empty ``tunable`` means the upstream
    script exposes no hyperparameters on its command line. The package runs
    each script unchanged, so such a method cannot be tuned through the
    wrapper. Its settings are reported under ``fixed_in_script`` and
    ``upstream_knobs`` (both empty for methods outside the upstream audit).

    **Variant selection.** ``category`` and ``modalities`` select the variant
    exactly like ``run``. Either may be omitted when the rest leaves one
    variant. A ``data_dir`` variant (scBridge's) has no modality
    tokens: leave ``modalities`` out and select it by ``category``, or by
    nothing when it is the method's only variant.

    **Dataset tie-break.** When the selection is still ambiguous, ``dataset``
    picks the one variant whose input files are all present in
    ``<data_path>/<dataset>``: ``params_for('Matilda', dataset='D11')`` is the
    rna+adt variant. A folder that settles nothing changes nothing, and the
    ambiguity error is raised as usual.

    **Modality spellings.** ``protein`` is accepted for ``adt``, and ``atac``
    for either ATAC representation role (``atac_gas`` / ``atac_peak``).

    **Ambiguity error.** ``mtb.AmbiguousVariantError`` derives from both
    ``ValueError`` and ``KeyError``. Its message spells out the selecting
    call, e.g. ``params_for('Matilda', 'vertical', ['rna', 'adt'])``.

    See Also
    --------
    mtb.method_info : the same ``params`` block for every variant at once.

    mtb.sweep : run one method over a range of one of these hyperparameters.

    mtb.AmbiguousVariantError : raised when several variants fit the selection.
    """
    s = registry.get(method)
    registry.check_category(category)
    modalities = registry.normalize_modalities(modalities)
    if not s.variants:
        raise KeyError(f"{method}: no variants (declared stub); nothing to tune")

    def _example(v):
        return (f"params_for({method!r}, {v.when.get('category')!r}, "
                f"{list(v.when.get('modalities', []))})")

    ds_dir = None
    if dataset is not None:
        from . import config
        root = data_path if data_path is not None else config.DEFAULT.data_path
        ds_dir = Path(os.path.abspath(os.fspath(root))) / dataset

    def _by_folder(cands):
        """The single candidate the dataset folder satisfies, else None."""
        if ds_dir is None or not ds_dir.is_dir():
            return None
        ok = [x for x in cands if _resolve._variant_satisfiable(x, ds_dir, s.id)]
        return ok[0] if len(ok) == 1 else None

    if category is None and modalities is None:
        if len(s.variants) > 1:
            v = _by_folder(s.variants)
            if v is None:
                raise AmbiguousVariantError(
                    f"{method} has {len(s.variants)} variants - pass category and "
                    f"modalities, e.g. {_example(s.variants[0])}; available: "
                    f"{[_variant_key(x) for x in s.variants]}")
        else:
            v = s.variants[0]
    elif modalities is None:
        # category alone is enough whenever it selects exactly one variant. This is
        # the only way to reach a data_dir variant (scBridge),
        # which has no modalities to pass.
        cands = [x for x in s.variants if x.when.get("category") == category]
        if not cands:
            raise KeyError(
                f"{method}: no {category!r} variant; available: "
                f"{[_variant_key(x) for x in s.variants]}")
        if len(cands) > 1:
            v = _by_folder(cands)
            if v is None:
                raise AmbiguousVariantError(
                    f"{method} has {len(cands)} {category!r} variants - also pass "
                    f"modalities, e.g. {_example(cands[0])}; available: "
                    f"{[_variant_key(x) for x in cands]}")
        else:
            v = cands[0]
    elif category is None:
        from .engine.schema import modality_family
        want = {modality_family(m) for m in modalities}
        cands = [x for x in s.variants
                 if set(x.when.get("modalities", [])) == set(modalities)]
        if not cands:
            cands = [x for x in s.variants
                     if {modality_family(m) for m in x.when.get("modalities", [])} == want]
        if not cands:
            raise KeyError(
                f"{method}: no variant takes modalities {sorted(modalities)}; "
                f"available: {[_variant_key(x) for x in s.variants]}")
        if len(cands) > 1:
            v = _by_folder(cands)
            if v is None:
                raise AmbiguousVariantError(
                    f"{method}: modalities {sorted(modalities)} match {len(cands)} "
                    f"variants - also pass category, e.g. {_example(cands[0])}; "
                    f"available: {[_variant_key(x) for x in cands]}")
        else:
            v = cands[0]
    else:
        v = s.select(category, set(modalities), loose=True)
    up = upstream.knobs_for(s.id)
    return {"method": s.id, "variant": _variant_key(v),
            "defaults": dict(v.params), "tunable": dict(v.tunable),
            "effective": _effective(v),
            "fixed_in_script": up["fixed_in_script"],
            "upstream_knobs": up["upstream_knobs"],
            "upstream_url": up["upstream_url"]}


# ------------------------------------------------------------------ citations
_CITE_FORMATS = ("bibtex", "text")


def _bibtex_key(tag: str, year) -> str:
    return f"{tag.replace(' ', '_')}_{year}" if year else tag.replace(" ", "_")


def _format_entry(tag: str, ref: dict, fmt: str) -> str:
    doi = ref.get("doi")
    authors = ref.get("authors") or ""
    title = ref.get("title") or ""
    journal = ref.get("journal") or ""
    year = ref.get("year") or ""
    if fmt == "bibtex":
        fields = [("author", authors.replace(", ", " and ") if authors else ""),
                  ("title", title), ("journal", journal), ("year", year)]
        if ref.get("volume"):
            fields.append(("volume", ref["volume"]))
        if ref.get("pages"):
            fields.append(("pages", ref["pages"]))
        fields.append(("doi", doi))
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v not in (None, ""))
        return f"@article{{{_bibtex_key(tag, year)},\n{body}\n}}"
    line = f"{authors}. {title}. {journal} ({year})."
    if doi:
        line += f" https://doi.org/{doi}"
    return line


def cite(*methods, fmt: str = "text") -> str:
    """Return citation text for the benchmark and, optionally, the methods you ran.

    Parameters
    ----------
    *methods : str | list[str]
        Method ids, one per argument or as one list; ``"all"`` = every
        registry method; none = the benchmark only.
    fmt : str
        ``"text"`` (one line per entry) or ``"bibtex"`` (one ``@article`` per
        entry).

    Returns
    -------
    str
        The benchmark entry first, then one entry per method in the order
        given.

    Raises
    ------
    ValueError
        ``fmt`` is neither ``"text"`` nor ``"bibtex"``; the message lists both.
    KeyError
        Unknown method id; the message suggests a close match, if any.
    TypeError
        Several ids are given and one is not a string.

    Examples
    --------
    >>> import multibench as mtb
    >>> print(mtb.cite())                                  # the benchmark only
    >>> print(mtb.cite("Matilda", "MOFA2"))
    >>> print(mtb.cite(["Matilda", "MOFA2"], fmt="bibtex"))
    >>> res = mtb.load_batch("out/")
    >>> print(mtb.cite(list(res.summary.method)))          # everything a sweep ran

    Notes
    -----
    **Two spellings.** Both forms return the same text, and ``cite(None)`` is
    ``cite()``:

    - ``cite('Matilda', 'MOFA2')`` - one id per argument, like the CLI
      ``multibench cite Matilda MOFA2``.
    - ``cite(['Matilda', 'MOFA2'])`` - one list or tuple.

    **Formats.**

    - ``"text"`` - one ``Authors. Title. Journal (year). https://doi.org/...``
      line per entry.
    - ``"bibtex"`` - one ``@article{<id>_<year>, ...}`` per entry, separated
      by a blank line; the benchmark's key is ``scMultiBench_<year>``.

    **Methods without a reference.** A method without a verified reference
    is emitted as a ``% <id>: no verified reference; see <repo_url>``
    comment (bibtex) or the same line without ``%`` (text).

    See Also
    --------
    mtb.method_info : carries the same ``reference``, ``repo_url`` and ``version`` per method.
    """
    args = list(methods)
    if fmt not in _CITE_FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; valid: {list(_CITE_FORMATS)}")
    if not args:
        ids: list[str] = []
    elif len(args) == 1 and not isinstance(args[0], str):
        ids = [] if args[0] is None else [registry.check_method(m) for m in args[0]]
    elif len(args) == 1 and args[0] == "all":
        ids = registry.list_methods()
    else:
        bad = [a for a in args if not isinstance(a, str)]
        if bad:
            raise TypeError(
                f"cite(): method ids must be strings, got {type(bad[0]).__name__}; "
                f"pass one list (cite(['Matilda', 'MOFA2'])) or one id per argument")
        ids = [registry.check_method(m) for m in args]
    parts = [_format_entry("scMultiBench", registry.benchmark_reference(), fmt)]
    for m in ids:
        s = registry.get(m)
        ref = (s.reference or {}).get("reference")
        if ref:
            parts.append(_format_entry(m, ref, fmt))
        else:
            url = (s.reference or {}).get("repo_url") or "(no repo_url)"
            note = f"{m}: no verified reference; see {url}"
            parts.append(("% " + note) if fmt == "bibtex" else note)
    return "\n\n".join(parts) if fmt == "bibtex" else "\n".join(parts)
