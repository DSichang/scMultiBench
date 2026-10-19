"""The 0.4.0 public surface: 19 top-level names, 29 Python entries, 14 commands and 50
visible options (core040 TARGET), and the names new on the branch that never ship."""
import argparse

import numpy as np
import pytest

import multibench as mtb
from multibench import cli

TOP = {"config", "plot", "catalog", "io", "env", "data", "find_methods", "method_info",
       "params_for", "cite", "scan", "run_all", "load_batch", "BatchResult", "evaluate",
       "load_results", "recommend", "describe_layout", "__version__"}
MEMBERS = {"io": ["export_dataset", "read_canonical"], "env": ["install"],
           "plot": ["bubble", "bar"], "data": ["fetch", "write_example"],
           "catalog": ["datasets", "metrics"], "config": ["DEFAULT"]}
COMMANDS = ["find", "params", "info", "fetch", "config", "scan", "layout", "export", "cite",
            "plot", "run-all", "evaluate", "env", "env install"]


def _visible(parser, path=()):
    """``(commands, options)`` that ``--help`` lists, recursively."""
    cmds, opts = [], []
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            shown = {c.dest for c in a._choices_actions}
            for name, sub in a.choices.items():
                if name in shown:
                    cmds.append(" ".join(path + (name,)))
                    c, o = _visible(sub, path + (name,))
                    cmds += c
                    opts += o
        elif a.option_strings and a.dest != "help" and a.help != argparse.SUPPRESS:
            opts.append((" ".join(path), a.option_strings[0]))
    return cmds, opts


def test_top_level_is_the_nineteen_names():
    assert set(mtb.__all__) == TOP and len(mtb.__all__) == 19
    assert set(dir(mtb)) - {n for n in dir(mtb) if n.startswith("__")} == TOP - {"__version__"}


def test_each_namespace_lists_its_members():
    for ns, names in MEMBERS.items():
        mod = getattr(mtb, ns)
        assert list(mod.__all__) == names, ns
        assert {n for n in dir(mod) if not n.startswith("__")} == set(names), ns


def test_the_command_line_has_fourteen_commands_and_fifty_three_options():
    cmds, opts = _visible(cli.build_parser())
    assert cmds == COMMANDS
    # c2 (C53, D17): scan --assume-gpu, fetch --scripts and env install --build are shown
    assert len(opts) == 53


@pytest.mark.parametrize("hidden", [("run-all", "--out")])
def test_the_cluster_and_maintainer_flags_parse_hidden(hidden):
    _, opts = _visible(cli.build_parser())
    assert hidden not in opts
    argv = hidden[0].split() + (["D11"] if hidden[0] in ("scan", "run-all") else [])
    argv += [hidden[1]] + (["o"] if hidden[1] == "--out" else [])
    argv += ["--category", "vertical"] if hidden[0] == "run-all" else []
    cli.build_parser().parse_args(argv)              # no SystemExit


@pytest.mark.parametrize("argv", [
    ["run", "--method", "Matilda", "--category", "vertical", "--cmd-template", "{cmd}"],
    ["evaluate", "e.h5", "--labels", "c.csv", "--clusters", "k.csv"],
])
def test_flags_new_on_the_branch_are_gone(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_python_names_new_on_the_branch_are_gone():
    with pytest.raises(TypeError):
        mtb.evaluate(np.zeros((4, 2)), ["a", "b", "a", "b"], clusters=[0, 1, 0, 1])
    with pytest.raises(TypeError):
        mtb.eval.classify.classify(embedding=np.zeros((4, 2)), labels=["a", "b"] * 2)
    import inspect
    from multibench import workflow
    assert "dry_run" not in inspect.signature(workflow.sweep).parameters
