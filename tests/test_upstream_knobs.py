"""The upstream-knob table must stay honest and reachable.

`params_for` lists what a script takes on its command line. These tests pin
the two keys that carry the rest (what the script still fixes, and what its
library offers that the script does not take), and the shape of the evidence
that makes them checkable.
"""
import re

import pytest

import multibench as mtb
from multibench.engine import upstream

AUDITED = sorted(upstream._table())


def test_the_audit_is_present_and_covers_the_zero_tunable_methods():
    assert len(AUDITED) >= 25, AUDITED
    for m in ("GLUE", "totalVI", "StabMap", "Seurat_v5", "MOFA2"):
        assert m in AUDITED


@pytest.mark.parametrize("method", AUDITED)
def test_every_fixed_setting_cites_a_checkable_source(method):
    """file:line, so a reader can verify the claim against upstream.

    One upstream directory is literally named "online iNMF", so the path half
    may contain spaces - only the ':<line>' suffix is structural.
    """
    for f in upstream.knobs_for(method)["fixed_in_script"]:
        assert re.fullmatch(r".+\.(py|R|Rmd):\d+", f["source"]), (method, f)
        assert f["name"] and f["value"]


@pytest.mark.parametrize("method", AUDITED)
def test_every_upstream_knob_says_what_it_controls(method):
    for k in upstream.knobs_for(method)["upstream_knobs"]:
        assert k["name"] and k["effect"]


def test_params_for_separates_parameters_from_what_stays_fixed():
    p = mtb.params_for("totalVI", "vertical")
    assert "n_latent" in p["tunable"]
    assert p["fixed_in_script"], "what the script still fixes"
    # a library setting that became a parameter is no longer listed as unreachable
    names = {k["name"] for k in p["upstream_knobs"]}
    assert "protein_dispersion" in names and not names & set(p["tunable"])


def test_no_library_setting_is_also_a_parameter():
    # a knob is named with its function ("k.nn (FindMultiModalNeighbors)") or
    # with two spellings ("seed / random_state"): the setting names are compared
    from multibench.engine import registry
    names = _load_generator().knob_names
    assert names("n_neighbors (calc_post_graph)") == {"n_neighbors"}
    assert names("seed / random_state") == {"seed", "random_state"}
    for m in AUDITED:
        tunable = {k for v in registry.get(m).variants for k in v.tunable}
        knobs = set().union(*(names(k["name"]) for k in upstream.knobs_for(m)["upstream_knobs"]))
        assert not tunable & knobs, m


def test_unaudited_method_returns_empty_lists_not_an_error():
    out = upstream.knobs_for("NoSuchMethod")
    assert out == {"fixed_in_script": [], "upstream_knobs": [],
                   "upstream_url": None}


# --- the generator's verification step --------------------------------------
# The table above is only as honest as the check that admits entries to it.

def _load_generator():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "tools" / "gen_upstream_knobs.py"
    spec = importlib.util.spec_from_file_location("gen_upstream_knobs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def clone(tmp_path):
    script = tmp_path / "tools_scripts" / "M" / "main_M.py"
    script.parent.mkdir(parents=True)
    script.write_text("import scanpy as sc\n"
                      "\n"
                      "sc.pp.normalize_total(adata, target_sum=1e4)\n"
                      "model = build(\n"
                      "    n_latent=20,\n"
                      ")\n")
    return tmp_path


def _entry(line, evidence):
    return {"name": "x", "value": "1", "evidence": evidence,
            "source": f"tools_scripts/M/main_M.py:{line}"}


def test_generator_keeps_an_entry_whose_cited_line_holds_the_evidence(clone):
    gen = _load_generator()
    ev = "sc.pp.normalize_total(adata,  target_sum=1e4)"   # whitespace differs
    assert gen.is_verified(_entry(3, ev), clone)


def test_generator_drops_an_entry_citing_a_blank_line(clone):
    """A blank line is a substring of every evidence string, so it must not
    count as containing it - that is how a citation of the wrong file at the
    right line number used to pass."""
    gen = _load_generator()
    ev = "sc.pp.normalize_total(adata, target_sum=1e4)"
    assert not gen.is_verified(_entry(2, ev), clone)


def test_generator_drops_an_entry_citing_a_trivial_line(clone):
    """The same hole one character wide: a lone ')' occurs inside the evidence
    without being the cited code."""
    gen = _load_generator()
    ev = "sc.pp.normalize_total(adata, target_sum=1e4)"
    assert not gen.is_verified(_entry(6, ev), clone)


def test_generator_drops_a_missing_file_a_bad_line_and_empty_evidence(clone):
    gen = _load_generator()
    ev = "sc.pp.normalize_total(adata, target_sum=1e4)"
    gone = dict(_entry(3, ev), source="tools_scripts/M/other.py:3")
    assert not gen.is_verified(gone, clone)
    assert not gen.is_verified(_entry(99, ev), clone)
    assert not gen.is_verified(_entry(3, ""), clone)
