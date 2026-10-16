"""``mtb.env.plan`` and ``doctor`` take a list of method ids as the first
argument, as ``install`` does; it raised ``TypeError: unhashable type: 'list'``."""
import pytest

import multibench as mtb


@pytest.mark.parametrize("fn", ["plan", "doctor"])
def test_a_list_first_is_the_methods(fn):
    f = getattr(mtb.env, fn)
    assert f(["Matilda", "totalVI"]) == f(methods=["Matilda", "totalVI"])
    assert f(("Matilda",)) == f(methods=["Matilda"])
    assert f("vertical") == f(category="vertical")


@pytest.mark.parametrize("fn", ["plan", "doctor"])
def test_methods_twice_is_refused(fn):
    with pytest.raises(TypeError, match=rf"^{fn}\(\) got methods twice"):
        getattr(mtb.env, fn)(["Matilda"], methods=["totalVI"])
