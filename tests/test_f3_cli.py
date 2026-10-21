"""Fix round 3, work package 'cli': ``run-all --batch`` (R3-09), help texts that
name `multibench config` (R3-11; ``scan --strict`` is hidden since 0.4.0), and the
``env plan`` / ``env install`` size total in two short lines (R3-13).

The run-all test runs the real runner in prefix mode against a stand-in env
prefix (a ``bin/python`` stub that writes an embedding), so the path from the
command line to ``run(batch=)`` and the saved summary is exercised end to
end without conda or a method.
"""
import argparse
import os
import re
import stat
import sys
import textwrap

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import _settings
from multibench import config
from multibench import workflow as W
from multibench._data import results
from multibench.engine import envs, runner

N = 120


def _rendered_help(parser, flag, columns=80):
    """The lines ``--help`` prints for ``flag`` at a terminal ``columns`` wide."""
    old = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = str(columns)
    try:
        text = parser.format_help()
    finally:
        if old is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = old
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(flag))
    out = [lines[start]]
    for l in lines[start + 1:]:
        if not l.startswith(" " * 10) or l.lstrip().startswith("-"):
            break
        out.append(l)
    return out


# ====================================================================== R3-09
def _h5(path, n_feat, prefix):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(n_feat, N)).astype(float))
        g.create_dataset("features", data=np.array([f"{prefix}{i}" for i in range(n_feat)],
                                                   dtype="S12"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(N)], dtype="S12"))


_WRITER = textwrap.dedent('''\
    import sys
    from pathlib import Path
    import h5py, numpy as np
    out = Path(sys.argv[sys.argv.index("--save_path") + 1])
    out.mkdir(parents=True, exist_ok=True)
    emb = np.random.default_rng(7).normal(size=(%d, 4))
    emb[%d:, 0] += 8.0
    with h5py.File(out / "embedding.h5", "w") as f:
        f.create_dataset("data", data=emb.T)
    (out / "predict.csv").write_text("x\\n" + "A\\n" * %d + "B\\n" * %d)
    ''') % (N, N // 2, N // 2, N // 2)


def _clear_env_caches():
    """Drop the per-process env probes, so the stand-in prefix is seen (and forgotten)."""
    for fn in (_settings._conda_envs_dir, envs._conda_prefixes):
        fn.cache_clear()


# ====================================================================== R3-13
SIZES = {"scmb_r": {"archive_bytes": 916_953_088, "unpacked_bytes": None},
         "scmb_torch": {"archive_bytes": 4_537_398_808, "unpacked_bytes": None},
         "scmb_torch-cpu": {"archive_bytes": 922_099_761, "unpacked_bytes": 2_642_831_784}}


def test_size_total_known_sizes_are_one_line():
    sizes = {"a": {"archive_bytes": 3 * 10**9, "unpacked_bytes": 9 * 10**9}}
    assert envs.plan_totals([{"env": "a", "flavor": "gpu"}], sizes, layers={})[1] == \
        "1 env: 3.0 GB to download, 9.0 GB on disk"
    assert envs.plan_totals([{"env": "a"}], sizes, layers={})[1] == \
        "1 env: 3.0 GB to download, 9.0 GB on disk"


@pytest.mark.parametrize("n_known", [0, 1, 5])
def test_size_total_with_unknown_disk_gives_the_du_advice(n_known):
    # Z27 (r3): an unmeasured env makes the disk figure a lower bound, in one line
    rows = [{"env": e} for e in "abcdef"]
    sizes = {e: {"archive_bytes": 2 * 10**9,
                 "unpacked_bytes": 3 * 10**9 if i < n_known else None}
             for i, e in enumerate("abcdef")}
    totals, line = envs.plan_totals(rows, sizes, layers={})
    assert line == "6 envs: 12.0 GB to download" + (
        f", at least {3 * n_known}.0 GB on disk" if n_known else "")
    assert totals["total_disk_bytes"] is None and ";" not in line
