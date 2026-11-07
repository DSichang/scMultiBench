"""Fix round 3, work package 'cli': the size total of an install plan
(``envs.plan_totals``) in one short line (R3-13).
"""
import textwrap

import h5py
import numpy as np
import pytest

from multibench import _settings
from multibench.engine import envs

N = 120


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
