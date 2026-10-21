"""CLI fixes from the round-1 virtual-student study (ledger L07, L22-L24,
L26-L29, L32, L41, L52, L61 for cli.py / envs.py / config.py).

Each test fails on the code before its fix: the subcommand, flag or message
it checks did not exist or read differently.
"""
import importlib
import json
import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench
from multibench import _settings
import multibench as mtb
from multibench import config, workflow
from multibench.engine import envs, ingest, registry, resolve, runner
from multibench.eval import pipeline


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())


# ================================================================ L26 {env_cmd}
def _inputs():
    return {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}


# ================================================================ L27 total line
def test_total_line_does_not_sum_an_incomplete_column_as_a_total():
    rows = [{"env": e} for e in "abcdef"]
    sizes = {e: {"archive_bytes": 2_000_000_000, "unpacked_bytes": None} for e in "abcdef"}
    sizes["a"]["unpacked_bytes"] = sizes["b"]["unpacked_bytes"] = 2_050_000_000
    totals, line = envs.plan_totals(rows, sizes, layers={})
    assert line == "6 envs: 12.0 GB to download, at least 4.1 GB on disk"
    assert totals["total_disk_bytes"] is None and "packed_sizes.json" not in line
    assert envs.plan_totals(rows, {e: {"archive_bytes": 1} for e in "abcdef"},
                            layers={})[1] == "6 envs: 0.0 GB to download"


def test_nothing_runnable_lists_rows_with_files_in_place_first(no_envs, tmp_path):
    with pytest.raises(ValueError) as e:
        mtb.run("D11", "vertical", out_dir=str(tmp_path), verbose=False,
                    modalities="all")
    body = str(e.value).split("blocked rows:\n", 1)[1]
    lines = [l for l in body.splitlines() if l.startswith("  ")]
    # D11 is CITE-seq: the rna+adt rows have their files; MIRA (rna+atac) does not
    assert len(lines) == 3 and all("(rna+adt)" in l for l in lines), lines
    assert [l.split()[0] for l in lines] == sorted(l.split()[0] for l in lines)


def test_cpu_fallback_warning_names_no_internal_file():
    msg = envs._cpu_fallback_warning("scmb_torch", {"scmb_torch-cpu": "u"}, {})
    assert msg == ("no CPU archive for scmb_torch; installing the GPU build (?) - the CPU "
                   "archive scmb_torch-cpu is not published yet")


def test_data_and_repo_path_variables(monkeypatch, tmp_path):
    for var, field in ((_settings.DATA_PATH_VAR, "data_path"), (_settings.REPO_PATH_VAR, "repo_path")):
        monkeypatch.delenv(var, raising=False)
        setattr(config, field, None)
        default = getattr(config, field)
        assert default.parent == _settings._BASE
        monkeypatch.setenv(var, str(tmp_path / field))
        cfg = config
        assert getattr(cfg, field) == tmp_path / field           # the variable wins
        setattr(cfg, field, str(tmp_path / "mine"))              # an assignment wins over it
        assert getattr(cfg, field) == tmp_path / "mine"
        assert isinstance(getattr(cfg, field), Path)
        setattr(cfg, field, None)                                # back to the variable
        assert getattr(cfg, field) == tmp_path / field
        monkeypatch.delenv(var)
        assert getattr(cfg, field) == default                    # unset: the default again
    config.data_path = "x"
    assert config.data_path == Path("x")


def test_config_docstring_lists_the_variables():
    doc = type(config).__doc__
    for var in ("MULTIBENCH_DATA_PATH", "MULTIBENCH_REPO_PATH", "MULTIBENCH_ENVS_DIR"):
        assert var in doc
