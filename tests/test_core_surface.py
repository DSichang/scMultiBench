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


def test_python_names_new_on_the_branch_are_gone():
    with pytest.raises(TypeError):
        mtb.evaluate(np.zeros((4, 2)), ["a", "b", "a", "b"], clusters=[0, 1, 0, 1])
    with pytest.raises(TypeError):
        mtb.eval.classify.classify(embedding=np.zeros((4, 2)), labels=["a", "b"] * 2)
    import inspect
    from multibench import workflow
    assert "dry_run" not in inspect.signature(workflow.sweep).parameters
