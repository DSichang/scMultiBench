"""The pages after the owner's decisions on 0.3.5: the 10x type names, the
three new warnings and the help text, and the donors recipe.

The sentences quoted from messages are compared with what the package says.
"""
import os
from pathlib import Path

import pytest


_ROOT = os.environ.get("SCMULTIBENCH_DOCS")
DOCS = Path(_ROOT) if _ROOT and Path(_ROOT).is_dir() else None
pytestmark = pytest.mark.skipif(DOCS is None, reason="SCMULTIBENCH_DOCS not set")


def _flat(page):
    return " ".join((DOCS / page).read_text().split())


def test_the_donors_recipe_names_batch():
    recipe = _flat("tutorials/run.md").split("{ #recipe-donors }", 1)[1]
    assert "`batch=` takes the column itself, not its name." in recipe
    assert 'batch=adata.obs["donor"])' in recipe
    assert "The summary then has the batch metrics" in recipe
    assert ("A cross method reads a fixed number of batches: three for RNA + ADT") in recipe
    assert ("With another number of donors, treat the data as a vertical task") in recipe
