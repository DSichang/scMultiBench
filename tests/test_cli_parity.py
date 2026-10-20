"""CLI parity with the Python API (work package F_cli, proposal P10).

scan / layout / export / cite / find filters / plot bubble|bar from stored or
own tables / evaluate --method/--dataset + npy/csv / flag aliases /
help strings on every parser and argument / exit codes.
"""
import argparse
import importlib

import numpy as np
import pandas as pd
import pytest

import multibench
from tests.conftest import full_scan


LONG_COLUMNS = ['metric', 'value', 'method', 'dataset', 'category', 'source', 'scored_with']


def test_packed_manifest_reads_shipped_file():
    from multibench.engine import envs
    mf = envs.packed_manifest()
    assert isinstance(mf, dict) and len(mf) > 0
    # one URL, or the URLs of an archive's parts (each under GitHub's 2 GiB)
    for key, v in mf.items():
        urls = [v] if isinstance(v, str) else v
        assert urls and all(u.startswith("https://") for u in urls), key
        if len(urls) > 1:
            assert urls == [f"{urls[0][:-4]}.{i:03d}" for i in range(1, len(urls) + 1)], key
