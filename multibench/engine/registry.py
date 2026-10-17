"""Load methods.yaml into MethodSpec objects; the validators of user-facing
tokens (method, category, task, atac, modalities): a typo raises with the
valid vocabulary instead of matching nothing.
"""
from __future__ import annotations

import difflib
import functools
from pathlib import Path

import yaml

from .. import config
from .schema import (ArgSpec, MethodSpec, OutputSpec, Variant, base_modality,
                     is_label_role)

_YAML = Path(__file__).resolve().parent / "methods.yaml"
# the tunable hyperparams, generated from the upstream argparse
_PARAMS_YAML = Path(__file__).resolve().parent / "params.yaml"
# per-method provenance, and the benchmark's own citation under `_benchmark`
_REFERENCES = Path(__file__).resolve().parent / "references.yaml"

# what `atac:` may say in methods.yaml
ATAC_VALUES = ("peak", "gene_activity")

# caller spellings -> registry tokens
MODALITY_ALIASES = {
    "protein": "adt",
    "peak": "atac_peak", "peaks": "atac_peak",
    "gas": "atac_gas", "gene_activity": "atac_gas",
}
_BASE_MODALITIES = ("rna", "adt", "atac")


@functools.lru_cache(maxsize=1)
def _tunable_map() -> dict:
    """{method_id: {"category:mods": {param: {default,type}}}} from params.yaml."""
    return yaml.safe_load(_PARAMS_YAML.read_text()) or {}


def _variant_key(when: dict) -> str:
    mods = "+".join(when.get("modalities", [])) or "-"
    return f"{when.get('category')}:{mods}"


def _parse_variant(d: dict, method_id: str | None = None) -> Variant:
    v = Variant(
        when=d["when"],
        entrypoint=d["entrypoint"],
        language=d.get("language", "python"),
        args=[ArgSpec(**a) for a in d.get("args", [])],
        output=OutputSpec(**d["output"]),
        params=d.get("params", {}),
        tunable=(d.get("tunable")
                 or _tunable_map().get(method_id, {}).get(_variant_key(d["when"]), {})
                 or {}),
        run_env=d.get("run_env", {}),
        cwd_at_script=d.get("cwd_at_script", False),
        pty=d.get("pty", False),
        driver=d.get("driver"),
        normalize_peaks=d.get("normalize_peaks", []),
        extra_outputs=[OutputSpec(**o) for o in d.get("extra_outputs", [])],
        helpers=list(d.get("helpers", []) or []),
        downloads=list(d.get("downloads", []) or []),
        size_hint=d.get("size_hint", "") or "",
    )
    # labels_for orders label files by this: a bad declaration fails at load
    try:
        v.stacked_roles()
    except ValueError as e:
        raise ValueError(f"methods.yaml: {method_id!r} {_variant_key(v.when)}: {e}") from None
    return v


def _parse_method(d: dict) -> MethodSpec:
    if "needs_labels" in d:
        raise ValueError(
            f"methods.yaml: {d.get('id')!r} declares needs_labels, but needs_labels "
            f"is derived from the variants' label roles (cty/label) - remove the key")
    if "categories" in d:
        raise ValueError(
            f"methods.yaml: {d.get('id')!r} declares categories, but categories is "
            f"derived from the variants' when.category - remove the key")
    variants = [_parse_variant(v, d["id"]) for v in d.get("variants", [])]
    # params.yaml has no entry for a variant that runs the same script as
    # another (scMDC, scMM cross): it takes that variant's tunables
    for v in variants:
        if not v.tunable:
            for other in variants:
                if other.tunable and other.entrypoint == v.entrypoint:
                    v.tunable = dict(other.tunable)
                    break
    spec = MethodSpec(
        id=d["id"], language=d.get("language", "python"),
        tasks=d.get("tasks", []),
        atac=d.get("atac"),
        setup_hint=d.get("setup_hint", ""), status=d.get("status", "declared"),
        variants=variants,
        env_spec=d.get("env_spec", {}) or {},
        # any present value is handed over, so validate_gpu_fields refuses a wrong shape
        cpu_params=d["cpu_params"] if d.get("cpu_params") is not None else {},
        requires_gpu=d.get("requires_gpu", False),
        gpu_evidence=d.get("gpu_evidence", "") or "",
        gpu_use=d.get("gpu_use", "") or "",
    )
    # a cpu_params flag the script does not accept would make argparse refuse the run
    for v in spec.variants:
        accepted = set(v.tunable) | set(v.params)
        bad = [k for k in spec.cpu_params if k not in accepted]
        if bad:
            raise ValueError(
                f"methods.yaml: {spec.id!r} cpu_params names {bad}, which the "
                f"{_variant_key(v.when)} variant does not accept (its params: "
                f"{sorted(accepted)}); cpu_params must be command-line params the "
                f"upstream script declares")
    if spec.atac is not None and spec.atac not in ATAC_VALUES:
        raise ValueError(
            f"methods.yaml: {spec.id!r} has atac={spec.atac!r}; valid: {list(ATAC_VALUES)}")
    if spec.consumes_atac and spec.atac is None:
        raise ValueError(
            f"methods.yaml: {spec.id!r} consumes an ATAC input but declares no "
            f"`atac:` key; set it to one of {list(ATAC_VALUES)} (the representation "
            f"the upstream script expects - check the script, not the role name)")
    if spec.atac is not None and not spec.consumes_atac:
        raise ValueError(
            f"methods.yaml: {spec.id!r} declares atac={spec.atac!r} but no variant "
            f"takes an ATAC input; remove the key or wire the variant")
    return spec


_ENV_SPECS = Path(__file__).resolve().parent / "env_specs.yaml"


@functools.lru_cache(maxsize=1)
def _references() -> dict:
    """Parsed engine/references.yaml; a missing file raises."""
    if not _REFERENCES.is_file():
        raise FileNotFoundError(
            f"{_REFERENCES} is missing - the package data is incomplete "
            f"(engine/references.yaml must ship with multibench)")
    return yaml.safe_load(_REFERENCES.read_text()) or {}


def benchmark_reference() -> dict:
    """The benchmark's own citation: ``{doi, title, authors, journal, year, ...}``."""
    return dict(_references().get("_benchmark") or {})


@functools.lru_cache(maxsize=1)
def load() -> list[MethodSpec]:
    data = yaml.safe_load(_YAML.read_text())
    specs = [_parse_method(m) for m in data["methods"]]
    env_specs = yaml.safe_load(_ENV_SPECS.read_text()) or {}
    for s in specs:
        if not s.env_spec and s.id in env_specs:
            s.env_spec = env_specs[s.id] or {}
    refs = _references()
    for s in specs:
        s.reference = dict(refs.get(s.id) or {})
    return specs


# --------------------------------------------------------------- validators
def check_method(method_id: str) -> str:
    """The registry id of a method name: an id in any letter case
    (``"totalvi"``) or a display name (``"MOFA+"``, ``"Seurat(WNN)"``, as
    ``catalog._canonical_id`` reads them). Anything else raises
    ``KeyError`` with :func:`unknown_method_message`."""
    ids = [s.id for s in load()]
    if method_id in ids:
        return method_id
    if isinstance(method_id, str):
        from ..data import catalog          # lazy: catalog imports registry lazily
        canon = catalog._canonical_id(method_id)
        if canon in ids:
            return canon
    raise KeyError(unknown_method_message(method_id))


def unknown_method_message(method_id, hint: str | None = None) -> str:
    """``"Unknown method X. Did you mean Y? ... shows all methods."``; ``hint``
    defaults to :func:`closest_method`."""
    hint = closest_method(method_id) if hint is None else hint
    return (f"Unknown method {method_id}."
            + (f" Did you mean {hint}?" if hint else "")
            + " " + config.hint("mtb.find_methods()", "multibench find")
            + " shows all methods.")


def closest_method(method_id: str) -> str | None:
    """The registry id closest to ``method_id``, or ``None`` when none is close."""
    ids = [s.id for s in load()]
    hint = difflib.get_close_matches(str(method_id), ids, n=1, cutoff=0.6)
    if hint:
        return hint[0]
    return {i.lower(): i for i in ids}.get(str(method_id).lower())


def check_id_list(value, name: str):
    """Return ``value``; raise ``TypeError`` if it is a bare string, which
    would be iterated character by character."""
    if isinstance(value, str):
        raise TypeError(
            f"{name} must be a list of ids, got the string {value!r} - did you "
            f"mean {name}=[{value!r}]?")
    return value


def check_category(category: str | None) -> str | None:
    """Validate a category token; ``None`` passes through. Raises the
    ``ValueError`` of ``config.category_folder``."""
    if category is None:
        return None
    config.category_folder(category)
    return category


def check_task(task: str | None) -> str | None:
    """Validate a task token against :func:`list_tasks`; ``None`` passes through."""
    if task is None:
        return None
    valid = list_tasks()
    if task not in valid:
        raise ValueError(f"unknown task {task!r}; valid: {valid}")
    return task


def check_atac(atac: str | None) -> str | None:
    """The canonical ``atac=`` token (``peak`` / ``gene_activity``) for a value
    or its aliases ``peaks`` / ``gas`` / ``gene-activity``; ``None`` passes through."""
    if atac is None:
        return None
    canon = {"peak": "peak", "peaks": "peak",
             "gene_activity": "gene_activity", "gene-activity": "gene_activity",
             "gas": "gene_activity"}.get(str(atac).lower())
    if canon is None:
        raise ValueError(f"unknown atac representation {atac!r}; valid: {list(ATAC_VALUES)}")
    return canon


@functools.lru_cache(maxsize=1)
def known_modalities() -> set[str]:
    """Every modality token a variant declares, plus the three base types."""
    toks = {m for s in load() for v in s.variants for m in v.when.get("modalities", [])}
    return toks | set(_BASE_MODALITIES)


def normalize_modalities(modalities, *, base: bool = False) -> list[str]:
    """Map caller modality spellings (:data:`MODALITY_ALIASES`) to registry
    tokens, in order, without duplicates; ``base=True`` reduces each to its
    base type. An unknown token raises ``ValueError``; ``None`` -> ``None``.
    """
    if modalities is None:
        return None
    if isinstance(modalities, str):
        modalities = [modalities]
    known = known_modalities()
    out: list[str] = []
    for tok in modalities:
        t = MODALITY_ALIASES.get(str(tok).lower(), str(tok))
        if t not in known and not is_label_role(t):
            raise ValueError(
                f"Unknown modality {tok}. The tokens are rna, adt (or protein) and "
                f"atac. atac_peak (or peak, peaks) and atac_gas (or gas, "
                f"gene_activity) name the ATAC form a method reads. rna1, adt1, "
                f"atac2 ... name numbered batches.")
        if base:
            t = base_modality(t)
        if t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------- lookups
def get(method_id: str) -> MethodSpec:
    method_id = check_method(method_id)
    return next(s for s in load() if s.id == method_id)


def list_tasks() -> list[str]:
    """List the task names that methods declare, sorted.

    Returns
    -------
    list[str]
        Sorted task names, the values ``mtb.find_methods(task=...)`` accepts.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.list_tasks()  # ['batch', 'clustering', 'dimension_reduction']
    >>> mtb.find_methods(task="dimension_reduction")

    See Also
    --------
    mtb.find_methods : filter methods by one of these tasks.
    """
    return sorted({t for s in load() for t in s.tasks})


def list_methods(category: str | None = None, task: str | None = None,
                 runnable: bool | None = None) -> list[str]:
    """Registry method ids in registry order, filtered by a validated
    ``category`` and ``task``; ``runnable=True`` keeps the methods with a
    variant, ``False`` those without."""
    check_category(category)
    check_task(task)
    out = []
    for s in load():
        if category and category not in s.categories:
            continue
        if task and task not in s.tasks:
            continue
        if runnable is not None and bool(s.variants) != runnable:
            continue
        out.append(s.id)
    return out
