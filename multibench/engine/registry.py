"""Load methods.yaml into MethodSpec objects; selection helpers; token validators.

This module is the chokepoint every entry point goes through (discover, resolve,
runner, envs, workflow, cli), so the validators for user-facing tokens live here:
:func:`check_method`, :func:`check_category`, :func:`check_task`,
:func:`check_atac`, :func:`normalize_modalities`. A typo raises with the valid
vocabulary instead of silently matching nothing.
"""
from __future__ import annotations

import difflib
import functools
from pathlib import Path

import yaml

from .. import config
from .schema import ArgSpec, MethodSpec, OutputSpec, Variant, base_modality, is_label_role

_YAML = Path(__file__).resolve().parent / "methods.yaml"
# doc-only tunable hyperparams, auto-generated from upstream argparse
_PARAMS_YAML = Path(__file__).resolve().parent / "params.yaml"
# curated provenance: repo_url / version / summary / reference per method,
# plus the benchmark's own citation under `_benchmark`
_REFERENCES = Path(__file__).resolve().parent / "references.yaml"

# What `atac:` may say in methods.yaml (the representation the upstream method
# expects). find_methods(atac=) / check_atac accept exactly these.
ATAC_VALUES = ("peak", "gene_activity")

# Caller-friendly modality spellings -> registry tokens. `protein` is what
# CITE-seq users say for ADT; the rest are the representation words used in
# describe_layout / the `atac:` key.
MODALITY_ALIASES = {
    "protein": "adt",
    "peak": "atac_peak", "peaks": "atac_peak",
    "gas": "atac_gas", "gene_activity": "atac_gas",
}
_BASE_MODALITIES = ("rna", "adt", "atac")


@functools.lru_cache(maxsize=1)
def _tunable_map() -> dict:
    """{method_id: {"category:mods": {param: {default,type}}}} from params.yaml."""
    if not _PARAMS_YAML.exists():
        return {}
    with open(_PARAMS_YAML) as fh:
        return yaml.safe_load(fh) or {}


def _variant_key(when: dict) -> str:
    mods = "+".join(when.get("modalities", [])) or "-"
    return f"{when.get('category')}:{mods}"


def _parse_variant(d: dict, method_id: str | None = None) -> Variant:
    return Variant(
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
    )


def _parse_method(d: dict) -> MethodSpec:
    if "needs_labels" in d:
        # The flag used to be hand-maintained and drifted (scJoint/Seurat_v3/
        # UnitedNet/scMoMaT all took cty roles while declaring False). It is now
        # derived from the variants' roles; refuse the key so it cannot come back.
        raise ValueError(
            f"methods.yaml: {d.get('id')!r} declares needs_labels, but needs_labels "
            f"is derived from the variants' label roles (cty/label) - remove the key")
    spec = MethodSpec(
        id=d["id"], language=d.get("language", "python"),
        categories=d.get("categories", []), tasks=d.get("tasks", []),
        atac=d.get("atac"),
        setup_hint=d.get("setup_hint", ""), status=d.get("status", "declared"),
        variants=[_parse_variant(v, d["id"]) for v in d.get("variants", [])],
        env_spec=d.get("env_spec", {}) or {},
    )
    # `atac:` is explicit on purpose (role names lie: moETM/scMM/iPOLNG take
    # `atac_gas` but consume peaks) - so make its ABSENCE loud rather than a
    # silent None that find_methods(atac=...) and describe_layout would omit.
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
    """Parsed engine/references.yaml. Missing file -> loud error, not {}: a wheel
    that silently ships without it would return None for every repo_url/DOI."""
    if not _REFERENCES.is_file():
        raise FileNotFoundError(
            f"{_REFERENCES} is missing - the package data is incomplete "
            f"(engine/references.yaml must ship with multibench)")
    return yaml.safe_load(_REFERENCES.read_text()) or {}


def benchmark_reference() -> dict:
    """The benchmark's own citation: ``{doi, title, authors, journal, year, ...}``
    (from engine/references.yaml ``_benchmark``, mirroring scMultiBench's
    CITATION.cff)."""
    return dict(_references().get("_benchmark") or {})


@functools.lru_cache(maxsize=1)
def load() -> list[MethodSpec]:
    data = yaml.safe_load(_YAML.read_text())
    specs = [_parse_method(m) for m in data["methods"]]
    # Per-method environment recipes live in a separate file so methods.yaml
    # stays focused on the run contract; attach them here.
    if _ENV_SPECS.exists():
        env_specs = yaml.safe_load(_ENV_SPECS.read_text()) or {}
        for s in specs:
            if not s.env_spec and s.id in env_specs:
                s.env_spec = env_specs[s.id] or {}
    # Provenance (repo_url / version / summary / paper reference) likewise.
    refs = _references()
    for s in specs:
        s.reference = dict(refs.get(s.id) or {})
    return specs


# --------------------------------------------------------------- validators
def check_method(method_id: str) -> str:
    """Return ``method_id`` if it is a registry id; else raise ``KeyError``.

    The message names the closest known id ("did you mean 'StabMap'?") and
    points at ``mtb.list_methods()``.
    """
    ids = [s.id for s in load()]
    if method_id in ids:
        return method_id
    hint = difflib.get_close_matches(str(method_id), ids, n=1, cutoff=0.6)
    if not hint:
        # case-insensitive exact match is a better hint than fuzzy distance
        lower = {i.lower(): i for i in ids}
        if str(method_id).lower() in lower:
            hint = [lower[str(method_id).lower()]]
    raise KeyError(
        f"unknown method {method_id!r}"
        + (f"; did you mean {hint[0]!r}?" if hint else "")
        + "; see mtb.list_methods()")


def check_category(category: str | None) -> str | None:
    """Validate a category token (``vertical``/``diagonal``/``mosaic``/``cross``).

    ``None`` passes through. Otherwise raises ``ValueError`` "unknown category
    ...; valid: [...]" - the same validator (and message) scan/run_all/load_results
    already use via ``config.category_folder``.
    """
    if category is None:
        return None
    config.category_folder(category)   # raises ValueError with the valid list
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
    """Validate an ``atac=`` filter value (``peak`` / ``gene_activity``).

    Accepts the describe_layout spellings ``peaks``/``gas``/``gene-activity`` as
    aliases and returns the canonical token; ``None`` passes through.
    """
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
    """Every modality token a variant declares (``rna``, ``adt``, ``atac``,
    ``atac_gas``, ``atac_peak``, ``rna1``..., ``adt2``..., ``atac3``...) plus
    the three base types."""
    toks = {m for s in load() for v in s.variants for m in v.when.get("modalities", [])}
    return toks | set(_BASE_MODALITIES)


def normalize_modalities(modalities, *, base: bool = False) -> list[str]:
    """Map caller modality spellings to registry tokens, validating each.

    * aliases: ``protein`` -> ``adt``; ``peak``/``peaks`` -> ``atac_peak``;
      ``gas``/``gene_activity`` -> ``atac_gas``
    * ``base=True`` further reduces every token to its base type
      (``atac_gas`` -> ``atac``, ``rna1`` -> ``rna``) - what ``find_methods``
      compares against.

    Order is kept, duplicates dropped. Unknown tokens raise ``ValueError`` naming
    the accepted vocabulary. ``None`` -> ``None``.
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
                f"unknown modality {tok!r}; known: rna, adt (alias: protein), atac "
                f"(aliases: peak, gas/gene_activity; role tokens: atac_gas, atac_peak, "
                f"rna1/adt1/atac2 ... for numbered batches)")
        if base:
            t = base_modality(t)
        if t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------- lookups
def get(method_id: str) -> MethodSpec:
    check_method(method_id)
    for s in load():
        if s.id == method_id:
            return s
    raise KeyError(f"unknown method {method_id!r}; see mtb.list_methods()")  # unreachable


def list_tasks() -> list[str]:
    """Return the sorted set of tasks declared across all method specs."""
    return sorted({t for s in load() for t in s.tasks})


def list_methods(category: str | None = None, task: str | None = None,
                 runnable: bool | None = None) -> list[str]:
    """Registry method ids, optionally filtered.

    ``category``/``task`` are validated (``ValueError`` listing the valid tokens
    on a typo). ``runnable=True`` keeps only methods with at least one declared
    variant (i.e. usable by ``inputs_for``/``run``); ``runnable=False`` keeps only
    the declared-but-unwired stubs. ``None`` (default) returns all.
    """
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
