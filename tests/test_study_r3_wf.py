"""Study round 3, work package wf: R3-01, R3-02 and the workflow part of R3-17.

R3-01: a gene-activity token (``atac_gas`` / ``gas`` / ``gene_activity``)
selected moETM, scMM and iPOLNG in scan and run_all, although they read peaks
(their role is only named ``atac_gas``). The near-miss text of the
gene-activity rows named a looser rule than the peak rows.

R3-02: run_all attempted a row whose ATAC file held the other representation,
printed only the unused-batch note and kept no caveat in the records.

The env probe is pinned (every env installed, a GPU present) so the verdicts
hold on any host; ``_run`` is faked in the real-run tests.

Matilda, scMDC and UnitedNet were the vertical methods that read gene
activity. They are registered as reading peaks now, as the benchmark ran
them, and no vertical method reads gene activity. The package still checks
both directions, so every test of this file runs with the three set back to
gene activity (``three_methods_read_gene_activity``). The messages pinned
here are what the package says about such a method, not about the real three.
"""
import inspect
import json
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import discover
from multibench import workflow as W
from multibench.engine import envs, registry
from tests.conftest import full_scan

#: the caveat of a gene-activity method given peaks in atac.h5 (after the method)
GAS_CAV = ("needs gene-activity ATAC. The features of atac.h5 look like chr:start-end, "
           "so it holds peaks.")

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
GENES = [f"GENE{i}" for i in range(40)]
GAS_METHODS = {"Matilda", "scMDC", "UnitedNet"}       # stand-ins: see the module docstring
# the other vertical methods with an RNA + ATAC variant, which read peaks
PEAK_METHODS = {"MIRA", "Seurat_WNN", "VIMCCA", "iPOLNG", "moETM", "scMM", "scMVP",
                "MOFA2", "Multigrate", "UINMF", "scMoMaT"}


@pytest.fixture(autouse=True)
def three_methods_read_gene_activity(monkeypatch):
    for m in GAS_METHODS:
        assert registry.get(m).atac == "peak", m
        monkeypatch.setattr(registry.get(m), "atac", "gene_activity")


@pytest.fixture(autouse=True)
def every_env_and_a_gpu(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _h5(path, feats, n=60):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), n)).astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(n)], dtype="S12"))


def _vertical(root, name, atac_file, feats, n=60):
    d = root / name
    d.mkdir()
    # gene activity is named by the RNA's genes, as in real data
    _h5(d / "rna.h5", GENES[:30], n)
    _h5(d / atac_file, feats, n)
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / "cty.csv", index=False)
    return root


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def test_peak_token_still_keeps_the_peak_methods_behind_an_atac_gas_role(tmp_path):
    root = _vertical(tmp_path, "MU_PK", "atac.h5", PEAKS)
    for call in (full_scan, lambda *a, **k: mtb.run_all(*a, dry_run=True, **k)):
        df = _quiet(call, "MU_PK", "vertical", modalities=["rna", "atac_peak"],
                    data_path=root, verbose=False)
        assert set(df["method"]) == PEAK_METHODS
        assert {"moETM", "scMM", "iPOLNG"} <= set(df.loc[df.runnable, "method"])


def test_gene_activity_near_miss_states_the_vertical_rule(tmp_path):
    root = _vertical(tmp_path, "MU_TC", "atac_peak.h5", PEAKS)
    df = _quiet(full_scan, "MU_TC", "vertical", modalities=["rna", "atac"], data_path=root,
                verbose=False)
    for m in ("scMDC", "UnitedNet", "Matilda"):
        text = df.loc[df.method == m, "files_reason"].iloc[0]
        assert ("Vertical methods read atac.h5. method_info(\"" + m + "\")[\"atac\"] says "
                "which ATAC " + m + " needs.") in text, (m, text)
        assert "The folder holds atac_peak.h5" in text, (m, text)
        assert "atac_gas.h5 or atac.h5" not in text


def test_scan_blocks_peak_methods_given_gene_activity(tmp_path):
    root = _vertical(tmp_path, "MU_GA", "atac.h5", GENES)
    df = _quiet(full_scan, "MU_GA", "vertical", modalities=["rna", "atac"], data_path=root,
                verbose=False).set_index("method")
    for m in ("moETM", "scMM", "iPOLNG", "scMVP"):
        assert not df.loc[m, "runnable"] and df.loc[m, "files_ok"], m
        assert df.loc[m, "reason"].startswith(f"{m} needs peak ATAC, and atac.h5 holds gene "
                                              "activity")
    assert df.loc[sorted(GAS_METHODS), "runnable"].all()


class _Res:
    def __init__(self, out):
        self.output = out


def test_old_records_without_a_caveat_still_summarise():
    rec = {"method": "Matilda", "status": "RUN_OK", "run_sec": 1.0}
    sm = W.BatchResult([rec], "D11", "vertical").summary
    assert list(sm.columns[-2:]) == ["caveat", "reason"] and sm["caveat"].isna().all()


@pytest.mark.parametrize("dataset,category", [("D11", "vertical"), ("D28", "diagonal"),
                                              ("D45", "mosaic"), ("D46", "mosaic"),
                                              ("D52", "cross")])
def test_demo_datasets_have_no_row_blocked_by_the_atac_kind(dataset, category):
    df = _quiet(full_scan, dataset, category, verbose=False)
    assert not W._is_wrong_atac(df["reason"]).any(), df.loc[
        W._is_wrong_atac(df["reason"]), ["method", "reason"]]


