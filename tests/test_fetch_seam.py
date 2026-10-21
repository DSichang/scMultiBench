"""``mtb.data.fetch`` through its single network seam (``fetch._download``),
and the ``load_batch(methods=)`` filter, without the network."""
import fnmatch
import importlib
import json
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb

# `multibench.data.fetch` the attribute is the fetch() FUNCTION; this is the module
F = importlib.import_module("multibench.data.fetch")

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Matilda", "MOFA2", "scMoMaT"]
DATASET = "D11"


def test_load_batch_methods_is_keyword_only():
    import inspect
    p = inspect.signature(mtb.load_batch).parameters["methods"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


