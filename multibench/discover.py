"""Discovery: filter methods by need; enrich method_info with catalog metadata; cite."""
from __future__ import annotations

import functools
import inspect
import os
from pathlib import Path
from urllib.parse import quote as _quote

from .config import _and_list
from .engine import registry, upstream, envs
from .engine import resolve as _resolve
from .engine.schema import AmbiguousVariantError, is_label_role
from .engine.schema import base_modality as _base_modality

#: representation role tokens -> the ``atac=`` value they stand for
_REPRESENTATION_TOKENS = {"atac_peak": "peak", "atac_gas": "gene_activity"}


def _representations(toks) -> set[str]:
    """The ATAC representations (``peak``, ``gene_activity``) the normalised tokens name."""
    return {_REPRESENTATION_TOKENS[t] for t in toks if t in _REPRESENTATION_TOKENS}


#: naming both ATAC roles selects the variants that read both files (MultiMAP, Seurat_v3)
_BOTH_FILES = frozenset(_REPRESENTATION_TOKENS)


def _modality_filter(modalities, atac) -> tuple[set[str] | None, str | None]:
    """``(base types every matching variant must consume, atac)``: the modality rule
    of ``find_methods`` and ``recommend`` (see the ``find_methods`` Notes)."""
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
    """Whether ``modalities`` names both ATAC representations, in any spelling."""
    if modalities is None or isinstance(modalities, str):
        return False
    return _BOTH_FILES <= set(registry.normalize_modalities(modalities))


def _variant_matches(v, category, want, needs_labels, atac) -> bool:
    """Whether variant ``v`` satisfies every per-variant filter; ``v=None`` (a stub)
    satisfies only filters that ask nothing of a variant."""
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
        ``True`` = methods with a variant the package runs (not a check of
        your host: see ``mtb.scan``); ``False`` = declared stubs; ``None`` =
        both.
    tunable : bool | None
        ``True`` = methods with command-line hyperparameters that
        ``run(params=...)`` can set; ``False`` = the rest (settings fixed in
        the script); ``None`` = both.

    Returns
    -------
    list[str]
        Method ids in the package's order.

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
      looks at the label need of the vertical rna+adt variant alone, not at
      the method's other variants.
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

    **Labels.** ``method_info(m)['needs_labels']`` is the method-level flag (any variant
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
        if tunable is not None and any(v.tunable for v in s.variants) != tunable:
            continue
        # atac is declared per method, but only a variant with an ATAC input honours it
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
    """Keyword guard of ``list_methods``: a ``find_methods`` filter raises a
    ``TypeError`` naming ``find_methods``; another keyword gets Python's message."""
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
    """Return the method ids, optionally restricted to one category.

    Parameters
    ----------
    category : str | None
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``; ``None`` = every method.

    Returns
    -------
    list[str]
        Method ids in the package's order.

    Raises
    ------
    ValueError
        Unknown ``category``; the message lists the valid ones.
    TypeError
        A keyword other than ``category``. For a ``find_methods`` filter such as
        ``task=`` the message names the call to use: ``find_methods(category, task=...)``.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.list_methods()                 # every method id
    >>> mtb.list_methods("vertical")       # ids with a vertical variant

    Notes
    -----
    A method is listed under a category when it has a variant for that
    category - the same set ``scan``, ``run_all`` and
    ``find_methods(category=)`` dispatch.

    See Also
    --------
    mtb.find_methods : filter by task, modalities, labels, ATAC representation and more.

    mtb.list_categories : the four category tokens with a description of each.
    """
    return registry.list_methods(category)


def _effective(v) -> dict:
    """The value each knob runs with when the caller passes no ``params``: upstream
    argparse defaults overlaid with what the package passes."""
    eff = {k: (t or {}).get("default") for k, t in v.tunable.items()}
    eff.update(dict(v.params))
    return eff


def _reference_batch(variant):
    """Batch ``N`` of a variant's ``reference`` const ``data<N>``, else ``None``."""
    ref = next((a.const for a in variant.args if a.role == "reference" and a.const), None)
    ref = str(ref) if ref is not None else ""
    return int(ref[4:]) if ref.startswith("data") and ref[4:].isdigit() else None


def method_info(method: str, *, verbose: bool = False) -> dict:
    """Return everything known about one method as a flat dict.

    Parameters
    ----------
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.list_methods()``.
    verbose : bool
        ``True`` adds ``notes_long``, the long upstream-audit notes.

    Returns
    -------
    dict
        One flat record. Start with ``supports``, ``params``, ``runtime`` and
        ``needs_labels``. Notes describes every key.

    Raises
    ------
    KeyError
        Unknown method id. The message suggests a close match.

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
    **Key reference.** The dict merges the method definition, the provenance
    record (repository, version, paper) and the observed runtime.

    - ``id`` - the method id.
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
    - ``supports`` - the category and modality combinations the method
      runs, one entry per variant: ``category``, ``modalities``,
      ``output_kind``, ``n_tunable``, ``needs_labels``, ``labels`` (the
      label roles the variant reads, e.g. ``['cty']`` / ``['rna_cty']`` /
      ``[]``) and ``reference_batch`` (the batch a variant uses as its fixed
      reference, e.g. 3 for StabMap in cross; ``None`` elsewhere).
    - ``params`` - what ``run(params=...)`` can change, keyed per variant
      as ``'category:mods'``, each with ``defaults``, ``tunable`` and
      ``effective`` (see ``mtb.params_for``).
    - ``fixed_in_script`` / ``upstream_knobs`` / ``upstream_url`` - what the
      script pins and what its library documents (see ``mtb.params_for``).
    - ``runtime`` - observed cost, to size a sweep; see **Runtime**.
    - ``gpu`` / ``cpu_params`` / ``requires_gpu`` / ``gpu_evidence`` - see
      **GPU and CPU**.
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
      from the GPU benchmark host; for a method that uses a GPU it adds that
      a CPU-only host takes longer.

    Use them to set ``run_all(timeout=...)``.

    **Labels.** ``needs_labels`` is the method-level flag: ``True`` when
    any variant takes a cell-type-label (``cty``) role as a required input.
    It is not per category. For
    the per-variant answer read ``supports[i]['needs_labels']``;
    ``find_methods(needs_labels=...)`` filters per variant.

    **Status.** ``status`` says how the method was checked. ``'verified'``
    means the command template was cross-checked against the upstream entrypoint
    and the method was executed end to end on a reference dataset;
    ``'declared'`` = available but not run end to end. A method listed but
    not yet runnable raises ``KeyError``.

    **GPU and CPU.** ``gpu``, ``cpu_params``, ``requires_gpu`` and
    ``gpu_evidence`` are the GPU/CPU contract of the upstream script, read
    from its source:

    - ``gpu`` - how the method uses an NVIDIA GPU: ``'required'`` (see
      ``requires_gpu``), ``'used when present'`` (it runs on the CPU
      otherwise), ``'not used'``, or ``'unknown'`` (not checked yet).
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
    - ``gpu_evidence`` - the ``file:line`` of that CUDA call (one per script,
      joined by ``', '``), else ``None``.

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
    method = s.id
    if not s.variants:
        raise KeyError(f"{method}: no variants (declared stub)")
    ref = s.reference
    info = {
        "id": s.id, "language": s.language, "categories": s.categories,
        "tasks": s.tasks, "env": envs.group_for(s.id), "atac": s.atac,
        "needs_labels": s.needs_labels, "status": s.status,
        "setup_hint": s.setup_hint,
        "variants": list(dict.fromkeys(v.entrypoint for v in s.variants)),
        "driver": next((v.driver for v in s.variants if v.driver), None),
        # percent-encoded: 'online iNMF' has a space
        "scripts_url": ("https://github.com/PYangLab/scMultiBench/tree/main/"
                        + _quote("/".join(s.variants[0].entrypoint.split("/")[:2]),
                                 safe="/")),
        "repo_url": ref.get("repo_url") or upstream.knobs_for(s.id)["upstream_url"],
        "version": ref.get("version"),
        "reference": dict(ref["reference"]) if ref.get("reference") else None,
        "supports": [{"category": v.when.get("category"),
                      "modalities": list(v.when.get("modalities", [])),
                      "output_kind": v.output.kind,
                      "n_tunable": len(v.tunable),
                      "needs_labels": v.needs_labels,
                      "labels": [r for r in v.roles() if is_label_role(r)],
                      "reference_batch": _reference_batch(v)}
                     for v in s.variants],
        "params": {_variant_key(v): {"defaults": dict(v.params),
                                     "tunable": dict(v.tunable),
                                     "effective": _effective(v)}
                   for v in s.variants},
    }
    up = upstream.knobs_for(s.id)
    info["fixed_in_script"] = up["fixed_in_script"]
    info["upstream_knobs"] = up["upstream_knobs"]
    info["upstream_url"] = up["upstream_url"]
    info["notes"] = ref.get("summary") or None
    info["runtime"] = _runtime_hint(s.id)
    info["gpu"] = s.gpu
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
    """Observed per-method runtimes (engine/runtimes.yaml)."""
    import yaml
    with open(_RUNTIMES_YAML) as fh:
        return yaml.safe_load(fh) or {}


def _runtime_hint(method: str) -> dict:
    """``method_info(method)["runtime"]`` (see its Notes); a typo raises ``KeyError``."""
    method = registry.check_method(method)
    rt = dict(_runtimes().get(method, {"tier": "unknown", "worst_sec": None,
                                       "observed": []}))
    # every observation was taken on the GPU benchmark host
    rt["host"] = "gpu"
    rt["note"] = "times observed on the benchmark host (NVIDIA RTX 4090)"
    if registry.get(method).gpu != "not used":
        rt["note"] += ("; on a CPU-only host expect training methods to take many "
                       "times longer")
    return rt


def _variant_key(v) -> str:
    """``'category:mod1+mod2'`` (``-`` for no modalities)."""
    return registry._variant_key(v.when)


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
        Method id, e.g. ``"Matilda"``; see ``mtb.list_methods()``.
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

    - ``method`` / ``variant`` - the method id and the selected variant as
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
    rna+adt variant. When the folder holds the files of several variants of
    one category and their modality sets are nested, the one with the most
    modalities is picked, as ``mtb.inputs_for`` does. A folder that settles
    nothing changes nothing, and the ambiguity error is raised as usual.

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
    method = s.id
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
        """The one candidate the dataset folder satisfies (of nested ones the largest,
        as in ``inputs_for``), else None."""
        if ds_dir is None or not ds_dir.is_dir():
            return None
        ok = _resolve._drop_nested(
            [x for x in cands if _resolve._variant_satisfiable(x, ds_dir, s.id)])
        return ok[0] if len(ok) == 1 else None

    if category is None and modalities is None:
        if len(s.variants) > 1:
            v = _by_folder(s.variants)
            if v is None:
                keys = _and_list(_variant_key(x) for x in s.variants)
                raise AmbiguousVariantError(
                    f"{method} has {len(s.variants)} variants, {keys}. Pass the "
                    f"category and modalities of one, for example "
                    f"{_example(s.variants[0])}.")
        else:
            v = s.variants[0]
    elif modalities is None:
        # the only way to reach a data_dir variant (scBridge), which has no modalities
        cands = [x for x in s.variants if x.when.get("category") == category]
        if not cands:
            from .engine.schema import no_category_message
            raise KeyError(no_category_message(method, s.wired_categories, category))
        if len(cands) > 1:
            v = _by_folder(cands)
            if v is None:
                mods = _and_list("+".join(x.when.get("modalities", []))
                                          for x in cands)
                raise AmbiguousVariantError(
                    f"{method} has {len(cands)} {category} variants, {mods}. Pass the "
                    f"modalities too, for example {_example(cands[0])}.")
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
            keys = _and_list(_variant_key(x) for x in s.variants)
            raise KeyError(
                f"No variant of {method} takes the modalities "
                f"{_and_list(sorted(modalities))}. Its variants: {keys}.")
        if len(cands) > 1:
            v = _by_folder(cands)
            if v is None:
                keys = _and_list(_variant_key(x) for x in cands)
                raise AmbiguousVariantError(
                    f"The modalities {_and_list(sorted(modalities))} match "
                    f"{len(cands)} variants of {method}, {keys}. Pass the category "
                    f"too, for example {_example(cands[0])}.")
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
        method; none = the benchmark only.
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
    >>> res = mtb.load_batch("out/")
    >>> ran = res.summary.query("status not in ['SKIPPED', 'FAIL', 'TIMEOUT']")
    >>> print(mtb.cite(ran.method, fmt="bibtex"))      # methods that finished

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
        ref = s.reference.get("reference")
        if ref:
            parts.append(_format_entry(m, ref, fmt))
        else:
            url = s.reference.get("repo_url") or "(no repo_url)"
            note = f"{m}: no verified reference; see {url}"
            parts.append(("% " + note) if fmt == "bibtex" else note)
    return "\n\n".join(parts) if fmt == "bibtex" else "\n".join(parts)
