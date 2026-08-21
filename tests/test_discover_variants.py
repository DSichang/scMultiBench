"""find_methods filters per VARIANT; availability + verification in method_info.

Workshop findings (Priya, Tomás, Aisha, Chen, Elena): needs_labels was the
method-level OR over all variants so scMoMaT vanished from
find_methods(vertical, rna+adt, needs_labels=False); category/modalities were
not required to hold on the same variant so Multigrate showed up for vertical
rna+atac; SPIRAL/GPSA were 'verified' with nothing saying their scripts are
not public; status='verified' hid the verification evidence.
"""
from pathlib import Path

import pytest

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
def test_availability_is_derived_from_absolute_entrypoints():
    assert AVAILABILITY == ("public", "benchmark-host-only")
    host_only = {m for m in mtb.list_methods()
                 if registry.get(m).availability == "benchmark-host-only"}
    absolute = {m for m in mtb.list_methods()
                for v in registry.get(m).variants if Path(v.entrypoint).is_absolute()}
    assert host_only == absolute == {"SPIRAL", "GPSA"}
    for m in mtb.list_methods():
        assert discover.method_info(m)["availability"] == registry.get(m).availability
    assert discover.method_info("PASTE")["availability"] == "public"
    # status is unchanged (availability is a separate axis)
    assert discover.method_info("SPIRAL")["status"] == "verified"


def test_find_methods_available_keyword():
    assert set(discover.find_methods(available=False)) == {"SPIRAL", "GPSA"}
    assert set(discover.find_methods(task="registration", available=True)) == {"PASTE", "PASTE2"}
    assert set(discover.find_methods(available=True)) | {"SPIRAL", "GPSA"} == set(mtb.list_methods())
    assert discover.find_methods(available=None) == discover.find_methods()
    # top-level alias takes the keyword too
    assert mtb.find_methods(available=False) == discover.find_methods(available=False)


def test_benchmark_host_only_sentence_in_notes_long():
    info = discover.method_info("GPSA", verbose=True)
    assert "benchmark-host-only" in info["notes_long"]
    assert "not published" in info["notes_long"] and "main_GPSA.py" in info["notes_long"]
    pub = discover.method_info("PASTE", verbose=True)
    assert not pub["notes_long"] or "benchmark-host-only" not in pub["notes_long"]


def test_benchmark_host_only_reason_text(tmp_path):
    why = resolve.benchmark_host_only_reason("/media/disk2/no/such/main_SPIRAL_ori.py")
    assert why.startswith("benchmark-host-only: script not published")
    assert "absolute path on the benchmark host" in why
    assert "/media/disk2/no/such/main_SPIRAL_ori.py" in why
    assert resolve.BENCHMARK_HOST_ONLY in why
    # relative entrypoints and existing absolute paths are not flagged
    assert resolve.benchmark_host_only_reason("tools_scripts/PASTE/main.py") == ""
    p = tmp_path / "main.py"; p.write_text("print(1)")
    assert resolve.benchmark_host_only_reason(str(p)) == ""


# ----------------------------------------------------------------- H7 verification
def test_verification_record_is_exposed_with_verbose(root):
    info = discover.method_info("VIMCCA", verbose=True)
    rec = info["verification"]
    assert isinstance(rec, list) and len(rec) == 1
    r = rec[0]
    assert set(r) == {"dataset", "category", "status", "wall_s", "ARI", "baseline", "verdict", "note"}
    assert r["dataset"] == "D11" and r["category"] == "vertical"
    assert r["status"] == "CHAIN_OK" and r["verdict"] == "DRIFT"
    assert abs(r["ARI"] - 0.6953) < 1e-9 and abs(r["baseline"] - 0.599) < 1e-9
    assert r["wall_s"] == 41
    # a no-embedding method carries None for the scores, not 0 or ''
    wnn = discover.method_info("Seurat_WNN", verbose=True)["verification"][0]
    assert wnn["status"] == "RUN_OK_NO_EMBEDDING" and wnn["ARI"] is None and wnn["baseline"] is None
    # non-verbose: not present; status values untouched
    assert "verification" not in discover.method_info("VIMCCA")
    assert discover.method_info("VIMCCA")["status"] == "verified"
    # every verified method has a record
    for m in mtb.list_methods():
        if discover.method_info(m)["status"] == "verified":
            assert discover.verification_for(m), m
    with pytest.raises(KeyError):
        discover.verification_for("NoSuchMethod")


def test_verification_tsv_is_shipped_and_identical_to_the_notebook_record(root):
    shipped = root / "multibench" / "files" / discover.VERIFICATION_TSV
    assert shipped.is_file()
    src = root / "notebooks" / "results" / "final_verification.tsv"
    if src.is_file():      # repo checkout: the two copies must not drift
        assert shipped.read_bytes() == src.read_bytes()
    import inspect
    doc = inspect.getdoc(discover.method_info)
    assert "cross-checked against the upstream entrypoint" in doc
    assert "end to end" in doc and "DRIFT" in doc and "RUN_OK_NO_EMBEDDING" in doc
