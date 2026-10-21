"""Student study round 2, package items M17, M19, M23, M28 and M29.

M17: the diagonal ATAC lines of ``describe_layout`` name each method once.
M19: messages printed by the command line name ``multibench ...`` commands,
not Python calls; JSON output keeps ``/`` unescaped.
M23: a vertical folder holding ``atac_peak.h5`` gets a reason that says how
to fix it.
M28: ``assume_gpu`` / ``--assume-gpu`` let a GPU-less login node check a job
for a GPU node; ``--strict`` counts the GPU test on its own.
M29: one ``--labels`` spelling for a ``.h5mu`` batch, and an error that
names it.
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config, workflow as W
from multibench.engine import envs, ingest
from tests.conftest import full_scan

ad = pytest.importorskip("anndata")

ALL_ENVS = frozenset(envs.group_for(m) for m in mtb.find_methods())
CATS = ("vertical", "diagonal", "mosaic", "cross")


def _cite_h5ad(path, n=60):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(n, 20)).astype(float))
    a.var_names = [f"g{i}" for i in range(20)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type"] = rng.choice(["T", "B"], n)
    a.obsm["protein"] = pd.DataFrame(rng.poisson(4.0, size=(n, 5)).astype(float),
                                     index=a.obs_names, columns=[f"CD{i}" for i in range(5)])
    a.write_h5ad(path)
    return a


def test_python_overwrite_refusal_names_the_keyword(tmp_path):
    a = _cite_h5ad(tmp_path / "cite.h5ad")
    ingest.export_dataset(a, tmp_path / "D", adt="obsm:protein", labels="obs:cell_type")
    with pytest.raises(FileExistsError) as ei:
        ingest.export_dataset(a, tmp_path / "D", adt="obsm:protein", labels="obs:cell_type")
    assert "Pass overwrite=True to replace them." in str(ei.value)
    assert "--overwrite" not in str(ei.value)


# ------------------------------------------------------------------- M23
def _mu_tc(root):
    """A vertical RNA + peaks folder written without category: atac_peak.h5."""
    rng = np.random.default_rng(0)
    n = 60
    cells = [f"c{i}" for i in range(n)]
    rna = ad.AnnData(rng.poisson(1.0, size=(n, 30)).astype(float))
    rna.obs_names, rna.var_names = cells, [f"g{i}" for i in range(30)]
    atac = ad.AnnData(rng.poisson(0.5, size=(n, 40)).astype(float))
    atac.obs_names = cells
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(40)]
    d = root / "MU_TC"
    d.mkdir()
    ingest._to_canonical(rna, d, modality='rna')
    ingest._to_canonical(atac, d, modality='peak')
    pd.DataFrame({"x": rng.choice(["T", "B"], n)}).to_csv(d / "cty.csv", index=False)
    return d


def test_vertical_peak_file_under_the_diagonal_name_says_how_to_fix(tmp_path, gas_matilda):
    _mu_tc(tmp_path)
    # the real Matilda reads peaks and gets scMVP's advice; `gas_matilda`
    # stands for a vertical method that reads gene activity
    df = full_scan("MU_TC", "vertical", data_path=tmp_path, verbose=False)
    df = df.set_index(["method", "modalities"])
    peak = df.loc[("scMVP", "rna+atac"), "reason"]
    assert ('scMVP reads atac.h5 for vertical. Rename atac_peak.h5 to atac.h5, or write it '
            'with category="vertical".') in peak
    assert "needs peak ATAC" not in peak
    gas = df.loc[("Matilda", "rna+atac"), "reason"]
    assert ("Matilda needs gene-activity ATAC (atac.h5), and the folder has peaks "
            "(atac_peak.h5).") in gas
    assert "ename" not in gas
    # the peak methods whose role was atac_gas (atac since 0.4.0, X13): the reason and
    # files_reason give one rule, 'vertical reads atac.h5', and never atac_gas.h5 as the
    # rename target
    for m in ("moETM", "scMM", "iPOLNG"):
        row = df.loc[(m, "rna+atac")]
        for col in ("reason", "files_reason"):
            assert "ename atac_peak.h5 to atac.h5" in row[col], (m, col, row[col])
            assert "vertical reads atac_gas.h5" not in row[col], (m, col)
            assert "reads atac_gas.h5 for vertical" not in row[col], (m, col)
            assert "to atac_gas.h5" not in row[col], (m, col)


def test_export_dataset_atac_lines_leave_out_cross():
    doc = ingest.export_dataset.__doc__
    assert "``'cross'``: ``atac_peak" not in doc


# ------------------------------------------------------------------- M28
@pytest.fixture
def login_node(monkeypatch):
    """Every env installed, no GPU on this host."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


def test_scan_assume_gpu_keeps_the_row_and_says_so(login_node):
    kw = dict(methods=["SMILE"], verbose=False)
    plain = full_scan("D45", "mosaic", **kw).iloc[0]
    assert not plain["runnable"] and "NVIDIA GPU" in plain["env_reason"]
    row = full_scan("D45", "mosaic", assume_gpu=True, **kw).iloc[0]
    assert row["runnable"] and row["env_ok"] and row["env_reason"] == ""
    assert row["caveat"].startswith("SMILE needs an NVIDIA GPU. This check assumes the job "
                                    "runs on a GPU node.")


def test_scan_assume_gpu_previews_the_gpu_node_command(login_node, tmp_path):
    from tests.test_gpu_requirements import _cite
    _cite(tmp_path)
    mtb.config.DEFAULT.data_path = tmp_path
    kw = dict(methods=['scMDC'], verbose=False)
    cpu = full_scan("CITE", "vertical", **kw)
    gpu = full_scan("CITE", "vertical", assume_gpu=True, **kw)
    assert "--device cpu" in cpu[cpu.modalities == "rna+adt"]["command"].iloc[0]
    assert "--device" not in gpu[gpu.modalities == "rna+adt"]["command"].iloc[0]


def test_scan_assume_gpu_is_a_no_op_on_a_gpu_host(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    row = full_scan("D45", "mosaic", methods=["SMILE"], assume_gpu=True,
                   verbose=False).iloc[0]
    assert row["runnable"] and "GPU node" not in row["caveat"]


def _mdata_with_labels_in_rna_obs():
    mu = pytest.importorskip("mudata")
    rng = np.random.default_rng(0)
    n = 40
    rna = ad.AnnData(rng.poisson(1.0, size=(n, 20)).astype(float))
    rna.obs_names = [f"c{i}" for i in range(n)]
    rna.var_names = [f"g{i}" for i in range(20)]
    rna.obs["cell_type"] = rng.choice(["T", "B"], n)
    atac = ad.AnnData(rng.poisson(0.5, size=(n, 30)).astype(float))
    atac.obs_names = rna.obs_names
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(30)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mu.MuData({"rna": rna, "atac": atac})


def test_global_obs_miss_names_the_modality_spelling(tmp_path):
    m = _mdata_with_labels_in_rna_obs()
    with pytest.raises(KeyError) as ei:
        ingest.export_dataset(m, tmp_path / "B", rna="rna", atac="atac", atac_kind="peak",
                              labels="obs:cell_type", category="mosaic", batch_index=2)
    msg = str(ei.value)
    assert "mdata['rna'].obs has 'cell_type': use labels='rna:cell_type'" in msg


