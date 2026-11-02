"""The pages after the owner's decisions on 0.3.5: the 10x type names, the
three new warnings and the help text, and the donor sentence of the
Quickstart tab.

The sentences quoted from messages are compared with what the package says.
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import ingest
from multibench.eval import io
from multibench.eval import scib as escib

_ROOT = os.environ.get("SCMULTIBENCH_DOCS")
DOCS = Path(_ROOT) if _ROOT and Path(_ROOT).is_dir() else None
pytestmark = pytest.mark.skipif(DOCS is None, reason="SCMULTIBENCH_DOCS not set")


def _flat(page):
    return " ".join((DOCS / page).read_text().split())


def test_the_quickstart_tab_names_batch_for_donors():
    page = _flat("quickstart.md")
    tab = page.split("### Your own data", 1)[1].split("10x Multiome,", 1)[0]
    assert ("With several donors or samples in the file, add `batch=adata.obs[\"donor\"]` to "
            "`run`. The summary then has the batch metrics. The recipe "
            "[Another number of donors](tutorials/run.md#recipe-donors) has the steps.") in tab
    # the recipe the bullet and the tab point at says the same
    recipe = _flat("tutorials/run.md").split("{ #recipe-donors }", 1)[1]
    assert "`batch=` takes the column itself, not its name." in recipe
    assert 'batch=adata.obs["donor"])' in recipe
    assert "The summary then has the batch metrics" in recipe
    assert ("A cross method reads a fixed number of batches: three for RNA + ADT") in recipe
    assert ("With another number of donors, treat the data as a vertical task") in recipe
