"""find_methods filters per VARIANT; availability in method_info.

Workshop findings (Priya, Tomás, Aisha, Chen, Elena): needs_labels was the
method-level OR over all variants so scMoMaT vanished from
find_methods(vertical, rna+adt, needs_labels=False); category/modalities were
not required to hold on the same variant so Multigrate showed up for vertical
rna+atac; nothing marked a 'verified' method whose script is not public.
"""
import copy
from pathlib import Path

import multibench as mtb
from multibench import discover
from multibench.engine import registry, resolve
from multibench.engine.schema import AVAILABILITY


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
    # Multigrate: vertical is rna+adt only; rna+atac exists only as a mosaic variant
    assert "Multigrate" not in discover.find_methods(category="vertical", modalities=["rna", "atac"])
    assert "Multigrate" in discover.find_methods(category="vertical", modalities=["rna", "adt"])
    assert "Multigrate" in discover.find_methods(category="mosaic", modalities=["rna", "atac"])
    # every id returned for a (category, modalities) pair has a variant inputs_for can select
    for m in discover.find_methods(category="vertical", modalities=["rna", "atac"]):
        sup = discover.method_info(m)["supports"]
        assert any(s["category"] == "vertical"
                   and {"rna", "atac"} <= {registry.base_modality(x) for x in s["modalities"]}
                   for s in sup), m


def test_atac_filter_is_judged_on_variants_that_consume_atac():
    # Multigrate declares atac: peak for its mosaic rna+atac variant only
    assert "Multigrate" not in discover.find_methods(category="vertical", atac="peak")
    assert "Multigrate" in discover.find_methods(category="mosaic", atac="peak")
    assert "Matilda" in discover.find_methods(category="vertical", atac="gene_activity")


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
    assert "per VARIANT" in doc and "scMoMaT" in doc and "available" in doc
    doc2 = inspect.getdoc(discover.method_info)
    assert "METHOD-level" in doc2 and "ANY variant" in doc2


# ----------------------------------------------------------------- H6 availability
def _with_host_only(monkeypatch, method="SCALEX"):
    """The live registry with ``method``'s entrypoints made absolute."""
    specs = [copy.deepcopy(s) for s in registry.load()]
    spec = next(s for s in specs if s.id == method)
    for v in spec.variants:
        v.entrypoint = "/benchmark/host/" + v.entrypoint
    monkeypatch.setattr(registry, "load", lambda: specs)
    return spec


def test_availability_is_derived_from_absolute_entrypoints(monkeypatch):
    assert AVAILABILITY == ("public", "benchmark-host-only")
    host_only = {m for m in mtb.list_methods()
                 if registry.get(m).availability == "benchmark-host-only"}
    absolute = {m for m in mtb.list_methods()
                for v in registry.get(m).variants if Path(v.entrypoint).is_absolute()}
    # every registered entrypoint is repo-relative
    assert host_only == absolute == set()
    for m in mtb.list_methods():
        assert discover.method_info(m)["availability"] == registry.get(m).availability
    # the rule, on a synthetic spec: an absolute entrypoint makes it host-only
    spec = _with_host_only(monkeypatch)
    assert spec.availability == "benchmark-host-only"
    assert not any(v.is_public for v in spec.variants)
    assert discover.method_info("SCALEX")["availability"] == "benchmark-host-only"
    # status is unchanged (availability is a separate axis)
    assert discover.method_info("SCALEX")["status"] == "verified"


def test_find_methods_available_keyword(monkeypatch):
    assert discover.find_methods(available=False) == []
    assert discover.find_methods(available=True) == discover.find_methods()
    assert discover.find_methods(available=None) == discover.find_methods()
    # top-level alias takes the keyword too
    assert mtb.find_methods(available=False) == discover.find_methods(available=False)
    _with_host_only(monkeypatch)
    assert discover.find_methods(available=False) == ["SCALEX"]
    assert "SCALEX" not in discover.find_methods(available=True)
    assert set(discover.find_methods(available=True)) | {"SCALEX"} == set(mtb.list_methods())


def test_host_only_sentence_only_for_host_only_methods(monkeypatch):
    for m in ("SCALEX", "Matilda"):
        pub = discover.method_info(m, verbose=True)
        assert pub["availability"] == "public"
        assert not pub["notes_long"] or "benchmark-host-only" not in pub["notes_long"]
    _with_host_only(monkeypatch)
    info = discover.method_info("SCALEX", verbose=True)
    assert info["availability"] == "benchmark-host-only"
    assert "benchmark-host-only" in info["notes_long"]
    assert "/benchmark/host/tools_scripts/SCALEX/" in info["notes_long"]


def test_benchmark_host_only_reason_text(tmp_path):
    why = resolve.benchmark_host_only_reason("/benchmark/host/no/such/main_M.py")
    assert why.startswith("benchmark-host-only: script not published")
    assert "absolute path on the benchmark host" in why
    assert "/benchmark/host/no/such/main_M.py" in why
    assert resolve.BENCHMARK_HOST_ONLY in why
    # relative entrypoints and existing absolute paths are not flagged
    assert resolve.benchmark_host_only_reason("tools_scripts/SCALEX/main.py") == ""
    p = tmp_path / "main.py"; p.write_text("print(1)")
    assert resolve.benchmark_host_only_reason(str(p)) == ""


def test_method_info_status_doc():
    import inspect
    doc = inspect.getdoc(discover.method_info)
    assert "cross-checked against the upstream entrypoint" in doc
    assert "end to end" in doc
    assert discover.method_info("VIMCCA")["status"] == "verified"
