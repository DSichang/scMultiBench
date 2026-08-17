"""The upstream-knob table must stay honest and reachable.

`params_for` reporting an empty `tunable` is accurate but misleading on its own:
it says the SCRIPT exposes nothing, not that the method is parameterless. These
tests pin the two keys that carry the rest of the truth, and the shape of the
evidence that makes them checkable.
"""
import re

import pytest

import multibench as mtb
from multibench.engine import upstream

AUDITED = upstream.audited_methods()


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


def test_method_info_carries_the_audit():
    info = mtb.method_info("GLUE")
    assert info["fixed_in_script"] and info["upstream_knobs"]
    assert info["upstream_url"]


def test_params_for_explains_an_empty_tunable():
    p = mtb.params_for("totalVI", "vertical")
    assert p["tunable"] == {}, "totalVI's script takes only paths"
    assert p["fixed_in_script"], "so the answer must say what it pins instead"
    assert any(k["name"] == "n_latent" for k in p["upstream_knobs"])


def test_unaudited_method_returns_empty_lists_not_an_error():
    out = upstream.knobs_for("NoSuchMethod")
    assert out == {"fixed_in_script": [], "upstream_knobs": [],
                   "upstream_url": None, "notes": None}
