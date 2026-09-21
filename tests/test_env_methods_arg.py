"""``mtb.env.plan`` / ``install`` / ``doctor`` take ``methods`` as a list.

A bare string was iterated character by character: ``methods='Matilda'``
failed with ``unknown method 'M'``. It now raises the ``TypeError``
``mtb.scan`` / ``mtb.run_all`` raise for the same mistake, with the same
message naming the list spelling. The list spelling is unchanged.
"""
import pytest

import multibench as mtb

CALLS = {
    "plan": lambda m: mtb.env.plan(methods=m),
    "install": lambda m: mtb.env.install(m),                # the default dry run
    "install(methods=)": lambda m: mtb.env.install(methods=m),
    "doctor": lambda m: mtb.env.doctor(methods=m),
}


def _scan_message(value):
    with pytest.raises(TypeError) as e:
        mtb.scan("D11", "vertical", methods=value)
    return str(e.value)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_bare_string_raises_the_type_error_scan_raises(name):
    with pytest.raises(TypeError) as e:
        CALLS[name]("Matilda")
    msg = str(e.value)
    assert "must be a list of ids" in msg and "methods=['Matilda']" in msg
    assert "unknown method 'M'" not in msg
    assert msg == _scan_message("Matilda")


@pytest.mark.parametrize("name", sorted(CALLS))
def test_the_list_spelling_selects_the_method(name):
    rows = CALLS[name](["Matilda"])
    assert [r["methods"] for r in rows] == [["Matilda"]]
