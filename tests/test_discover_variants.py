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


# ----------------------------------------------------------------- H1 per-variant
def test_scmomat_is_unsupervised_in_its_vertical_variants():
    ids = discover.find_methods(category="vertical", modalities=["rna", "adt"],
                                needs_labels=False)
    assert "scMoMaT" in ids
    # the method-level flag is still 'any variant' (the mosaic variant takes cty1..3)
    info = discover.method_info("scMoMaT")
    assert info["needs_labels"] is True
    per = {(s["category"], tuple(s["modalities"])): s["needs_labels"] for s in info["supports"]}
    assert per[("vertical", ("rna", "adt"))] is False
    assert per[("mosaic", ("rna1", "rna2", "rna3", "adt1", "atac2"))] is True
    # and needs_labels=True for mosaic still finds it
    assert "scMoMaT" in discover.find_methods(category="mosaic", needs_labels=True)
    assert "scMoMaT" not in discover.find_methods(category="vertical", needs_labels=True)


def test_category_and_modalities_must_hold_on_one_variant():
    # Multigrate has rna+adt and rna+atac variants in vertical and in mosaic
    assert "Multigrate" in discover.find_methods(category="vertical", modalities=["rna", "atac"])
    assert "Multigrate" in discover.find_methods(category="vertical", modalities=["rna", "adt"])
    assert "Multigrate" in discover.find_methods(category="mosaic", modalities=["rna", "atac"])
    # sciPENN reads RNA + ADT in both of its categories and no ATAC in either
    assert "sciPENN" not in discover.find_methods(category="vertical", modalities=["rna", "atac"])
    # every id returned for a (category, modalities) pair has a variant inputs_for can select
    for cat in ("vertical", "mosaic", "cross"):
        for mods in (["rna", "atac"], ["rna", "adt"], ["adt", "atac"], ["rna", "adt", "atac"]):
            want = sorted(
                m for m in discover.list_methods()
                if any(s["category"] == cat
                       and set(mods) <= {registry.base_modality(x) for x in s["modalities"]}
                       for s in discover.method_info(m)["supports"]))
            assert sorted(discover.find_methods(category=cat, modalities=mods)) == want, (cat, mods)


def test_atac_filter_is_judged_on_variants_that_consume_atac():
    assert "Multigrate" in discover.find_methods(category="vertical", atac="peak")
    assert "Multigrate" in discover.find_methods(category="mosaic", atac="peak")
    assert "Matilda" in discover.find_methods(category="vertical", atac="peak")
    # a method that declares an ATAC form matches only where a variant reads ATAC
    for cat in ("vertical", "diagonal", "mosaic", "cross"):
        for m in discover.find_methods(category=cat, atac="peak"):
            assert any(s["category"] == cat
                       and any(registry.base_modality(x) == "atac" for x in s["modalities"])
                       for s in discover.method_info(m)["supports"]) or m == "scBridge", (cat, m)


def test_the_two_per_variant_rules_on_a_synthetic_spec(monkeypatch):
    """No method of the registry has ATAC in one category only any more, so
    the two rules above are pinned on a spec that does: vertical rna+adt,
    mosaic rna+atac, ``atac: peak``."""
    from multibench.engine.schema import ArgSpec, MethodSpec, OutputSpec, Variant

    def var(cat, mods):
        return Variant(when={"category": cat, "modalities": mods},
                       entrypoint="tools_scripts/X/main.py", language="python",
                       args=[ArgSpec(role=r, flag=f"--{r}") for r in mods],
                       output=OutputSpec(kind="embedding", file="e.h5"))
    spec = MethodSpec(id="Fake", language="python", categories=["vertical", "mosaic"],
                      tasks=["clustering"], atac="peak", status="declared",
                      variants=[var("vertical", ["rna", "adt"]),
                                var("mosaic", ["rna1", "rna2", "atac2", "atac3"])])
    monkeypatch.setattr(registry, "load", lambda: [spec])
    monkeypatch.setattr(registry, "check_category", lambda c: c)
    monkeypatch.setattr(registry, "check_task", lambda t: t)
    assert discover.find_methods(category="vertical", modalities=["rna", "atac"]) == []
    assert discover.find_methods(category="vertical", modalities=["rna", "adt"]) == ["Fake"]
    assert discover.find_methods(category="mosaic", modalities=["rna", "atac"]) == ["Fake"]
    assert discover.find_methods(category="vertical", atac="peak") == []
    assert discover.find_methods(category="mosaic", atac="peak") == ["Fake"]


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
    monkeypatch.setattr(registry, "check_task", lambda t: t)
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
    assert "per variant" in doc and "scMoMaT" in doc
    doc2 = inspect.getdoc(discover.method_info)
    assert "method-level" in doc2 and "any variant" in doc2


# ----------------------------------------------------------------- removed API
def test_availability_api_is_gone():
    """No availability axis: every entrypoint is a tools_scripts/ path. The
    old keyword fails with Python's own TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'available'"):
        mtb.find_methods(available=True)
    assert "availability" not in discover.method_info("SCALEX")
    assert "availability" not in envs.plan(methods=["SCALEX"])[0]
    assert not hasattr(schema, "AVAILABILITY")
    assert not hasattr(registry.get("SCALEX"), "availability")


def test_method_info_status_doc():
    import inspect
    doc = inspect.getdoc(discover.method_info)
    assert "cross-checked against the upstream entrypoint" in doc
    assert "end to end" in doc
    assert discover.method_info("VIMCCA")["status"] == "verified"
