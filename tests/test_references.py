"""method_info provenance (repo_url / reference / notes / driver / variants) and mtb.cite.

DOIs in engine/references.yaml were each resolved against Crossref when added;
the tests here pin shape and reachability, not truth - do not add a DOI from
memory (see the header of references.yaml).
"""
import re

import pytest

import multibench as mtb
from multibench.engine import registry

# Methods with no Crossref-verified DOI yet. Shrink this list as references are
# verified; a method appearing here WITH a reference, or missing from here
# WITHOUT one, fails loudly.
UNREFERENCED = {"VIMCCA", "VIPCCA", "scMSI"}

DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


def test_every_method_has_repo_url_summary_and_verified_reference_shape():
    for m in registry.list_methods():
        info = mtb.method_info(m)
        assert info["repo_url"].startswith("http"), m
        assert info["version"], m
        assert info["notes"] and len(info["notes"]) <= 300 and " I " not in info["notes"], m
        if m in UNREFERENCED:
            assert info["reference"] is None, f"{m} now has a reference - drop it from UNREFERENCED"
        else:
            ref = info["reference"]
            assert ref and DOI_RE.fullmatch(ref["doi"]), (m, ref)
            for k in ("title", "authors", "journal", "year"):
                assert ref.get(k), (m, k)


def test_variants_deduplicated():
    for m in registry.list_methods():
        v = mtb.method_info(m)["variants"]
        assert len(v) == len(set(v)), m
    assert mtb.method_info("StabMap")["variants"] == ["tools_scripts/StabMap/main_StabMap.Rmd"]


def test_driver_surfaced():
    assert mtb.method_info("StabMap")["driver"] == "engine/drivers/run_stabmap.R"
    assert mtb.method_info("Matilda")["driver"] == "engine/drivers/run_matilda.py"
    assert mtb.method_info("GLUE")["driver"] is None


def test_notes_long_only_verbose():
    info = mtb.method_info("totalVI")
    assert "notes_long" not in info
    v = mtb.method_info("totalVI", verbose=True)
    assert len(v["notes_long"]) > 1000
    assert mtb.method_info("Multigrate", verbose=True)["notes_long"] is None   # unaudited
    # the other keys are unchanged by verbose
    assert {k: val for k, val in v.items() if k != "notes_long"} == info


def test_method_info_keeps_existing_keys():
    info = mtb.method_info("scMDC")
    for k in ("id", "language", "categories", "tasks", "env", "atac", "needs_labels",
              "status", "setup_hint", "variants", "scripts_url", "supports", "params",
              "fixed_in_script", "upstream_knobs", "upstream_url", "notes",
              "driver", "repo_url", "version", "reference"):
        assert k in info, k


def test_effective_merges_wrapper_defaults_over_upstream():
    p = mtb.params_for("scMDC", "vertical", ["rna", "adt"])
    assert p["tunable"]["nbatch"]["default"] == 2       # upstream default, untouched
    assert p["effective"]["nbatch"] == 1                 # what the wrapper really runs
    assert p["effective"]["lr"] == 1.0
    eff = mtb.method_info("scMDC")["params"]["vertical:rna+adt"]["effective"]
    assert eff["nbatch"] == 1 and eff["lr"] == 1.0


def test_cite_bibtex_and_text():
    from multibench.discover import cite
    s = cite(["StabMap", "GLUE"])
    assert s.count("@article{") == 3
    assert "10.1038/s41592-025-02856-3" in s
    assert "10.1038/s41587-023-01766-z" in s and "10.1038/s41587-022-01284-4" in s
    t = cite(fmt="text")
    assert t.count("\n") == 0 and "https://doi.org/10.1038/s41592-025-02856-3" in t
    with pytest.raises(ValueError, match="unknown fmt 'ris'"):
        cite(fmt="ris")
    with pytest.raises(KeyError, match="unknown method 'NoSuchMethod'"):
        cite(["NoSuchMethod"])
    # an unverified method is flagged, not silently dropped
    u = cite(["VIMCCA"])
    assert "% VIMCCA: no verified reference" in u and "scbean" in u
    allb = cite("all")
    assert allb.count("@article{") == 1 + len(registry.list_methods()) - len(UNREFERENCED)
    assert cite("StabMap") == cite(["StabMap"])
