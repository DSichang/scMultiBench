"""The 0.4.0 public surface: 19 top-level names, 29 Python entries, 14 commands and 50
visible options (core040 TARGET), and the names new on the branch that never ship."""
import argparse

import numpy as np
import pytest

import multibench as mtb

TOP = {"config", "plot", "catalog", "io", "env", "data", "find_methods", "method_info",
       "params_for", "cite", "scan", "run_all", "load_batch", "BatchResult", "evaluate",
       "load_results", "recommend", "describe_layout", "__version__"}
MEMBERS = {"io": ["export_dataset", "read_canonical"], "env": ["install"],
           "plot": ["bubble", "bar"], "data": ["fetch", "write_example"],
           "catalog": ["datasets", "metrics"], "config": ["DEFAULT"]}
COMMANDS = ["find", "params", "info", "fetch", "config", "scan", "layout", "export", "cite",
            "plot", "run-all", "evaluate", "env", "env install"]


def test_top_level_is_the_nineteen_names():
    assert set(mtb.__all__) == TOP and len(mtb.__all__) == 19
    assert set(dir(mtb)) - {n for n in dir(mtb) if n.startswith("__")} == TOP - {"__version__"}


def test_each_namespace_lists_its_members():
    for ns, names in MEMBERS.items():
        mod = getattr(mtb, ns)
        assert list(mod.__all__) == names, ns
        assert {n for n in dir(mod) if not n.startswith("__")} == set(names), ns


