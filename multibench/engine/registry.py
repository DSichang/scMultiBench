"""Load methods.yaml into MethodSpec objects; selection helpers."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

from .schema import ArgSpec, MethodSpec, OutputSpec, Variant

_YAML = Path(__file__).resolve().parent / "methods.yaml"


def _parse_variant(d: dict) -> Variant:
    return Variant(
        when=d["when"],
        entrypoint=d["entrypoint"],
        language=d.get("language", "python"),
        args=[ArgSpec(**a) for a in d.get("args", [])],
        output=OutputSpec(**d["output"]),
        params=d.get("params", {}),
        run_env=d.get("run_env", {}),
        extra_outputs=[OutputSpec(**o) for o in d.get("extra_outputs", [])],
    )


def _parse_method(d: dict) -> MethodSpec:
    return MethodSpec(
        id=d["id"], language=d.get("language", "python"),
        categories=d.get("categories", []), tasks=d.get("tasks", []),
        atac=d.get("atac"), needs_labels=d.get("needs_labels", False),
        setup_hint=d.get("setup_hint", ""), status=d.get("status", "declared"),
        variants=[_parse_variant(v) for v in d.get("variants", [])],
        env_spec=d.get("env_spec", {}) or {},
    )


_ENV_SPECS = Path(__file__).resolve().parent / "env_specs.yaml"


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
    return specs


def get(method_id: str) -> MethodSpec:
    for s in load():
        if s.id == method_id:
            return s
    raise KeyError(f"unknown method {method_id!r}; see registry.list_methods()")


def list_tasks() -> list[str]:
    """Return the sorted set of tasks declared across all method specs."""
    return sorted({t for s in load() for t in s.tasks})


def list_methods(category: str | None = None, task: str | None = None,
                 runnable: bool | None = None) -> list[str]:
    """Registry method ids, optionally filtered.

    ``runnable=True`` keeps only methods with at least one declared variant
    (i.e. usable by ``inputs_for``/``run``); ``runnable=False`` keeps only the
    declared-but-unwired stubs. ``None`` (default) returns all.
    """
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
