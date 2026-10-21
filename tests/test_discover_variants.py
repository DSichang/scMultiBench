"""find_methods filters per variant.

Workshop findings (Priya, Tomás, Aisha, Chen, Elena): needs_labels was the
method-level OR over all variants so scMoMaT vanished from
find_methods(vertical, rna+adt, needs_labels=False); category/modalities were
not required to hold on the same variant so Multigrate showed up for vertical
rna+atac.
"""
import pytest

import multibench as mtb
from multibench import discover
from multibench.engine import envs, registry, schema


def test_per_variant_semantics_on_a_synthetic_spec(monkeypatch):
    """A method whose ONLY mosaic variant needs labels but whose vertical one
    does not behaves per category."""
    from multibench.engine.schema import ArgSpec, MethodSpec, OutputSpec, Variant

    def var(cat, mods, roles):
        return Variant(when={"category": cat, "modalities": mods},
                       entrypoint="tools_scripts/X/main.py", language="python",
                       args=[ArgSpec(role=r, flag=f"--{r}") for r in roles],
                       output=OutputSpec(kind="embedding", file="e.h5"))
    spec = MethodSpec(id="Fake", language="python", categories=["vertical", "mosaic"],
                      tasks=["clustering"], atac=None, status="declared",
                      variants=[var("vertical", ["rna", "adt"], ["rna", "adt"]),
                                var("mosaic", ["rna1", "adt2"], ["rna1", "adt2", "cty1"])])
    monkeypatch.setattr(registry, "load", lambda: [spec])
    monkeypatch.setattr(registry, "check_category", lambda c: c)
    assert discover.find_methods(category="vertical", needs_labels=False) == ["Fake"]
    assert discover.find_methods(category="vertical", needs_labels=True) == []
    assert discover.find_methods(category="mosaic", needs_labels=True) == ["Fake"]
    assert discover.find_methods(category="mosaic", needs_labels=False) == []
    # no category: 'any variant' on both sides
    assert discover.find_methods(needs_labels=True) == ["Fake"]
    assert discover.find_methods(needs_labels=False) == ["Fake"]
    assert discover.find_methods(category="vertical", modalities=["rna", "adt"],
                                 needs_labels=False) == ["Fake"]
    assert discover.find_methods(category="mosaic", modalities=["rna", "adt"],
                                 needs_labels=False) == []


def test_find_methods_docstring_states_per_variant_rule():
    import inspect
    doc = inspect.getdoc(discover.find_methods)
    assert "one of its variants" in doc
    doc2 = inspect.getdoc(discover.method_info)
    assert "any variant" in doc2


