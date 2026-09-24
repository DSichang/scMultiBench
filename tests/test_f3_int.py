"""Round-3 integration: where the work packages wf (R3-02) and rr (R3-04, R3-10) meet.

- GLUE's scan caveat now ends with the prepared-file note (rr); the run
  record and the run log keep the caveat without it (wf's ``_run_caveat``).
- A row that is both of the wrong ATAC kind (wf) and under a mismatched
  ``MULTIBENCH_SCRIPTS_REF`` (rr) is blocked, and its reason keeps both
  sentences.

The env probe is pinned (every env installed, a GPU present) and ``_run`` is
faked, as in the package tests.
"""
import json
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


@pytest.fixture(autouse=True)
def every_env_and_a_gpu(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _h5(path, feats, bars):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((len(feats), len(bars))))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))


def _diagonal(root, name):
    """A diagonal folder whose peaks carry D28's chr_start_end spelling."""
    d = root / name
    d.mkdir()
    rna, atac = [f"r{i}" for i in range(20)], [f"a{i}" for i in range(20)]
    genes = [f"G{i}" for i in range(30)]
    _h5(d / "rna.h5", genes, rna)
    _h5(d / "atac_peak.h5", [f"chr4_{100 + 10 * i}_{105 + 10 * i}" for i in range(60)], atac)
    _h5(d / "atac_gas.h5", genes, atac)
    pd.DataFrame({"x": ["A"] * 20}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["A"] * 20}).to_csv(d / "atac_cty.csv", index=False)


class _Res:
    def __init__(self, out):
        self.output = out


def test_glue_record_keeps_the_caveat_without_the_prepared_file_note(tmp_path, monkeypatch,
                                                                   capsys):
    _diagonal(tmp_path, "LUNG_us")
    monkeypatch.setattr(W, "_run", lambda method, category, inputs, out_dir, params=None:
                        _Res(np.zeros((40, 5))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = mtb.scan("LUNG_us", "diagonal", methods=["GLUE"], data_path=tmp_path,
                       verbose=False).iloc[0]
        res = mtb.run_all("LUNG_us", "diagonal", tmp_path / "out", methods=["GLUE"],
                          data_path=tmp_path, evaluate=False)
    assert row["runnable"] and "the command reads inputs/atac_peak_normpeaks.h5" in row["caveat"]
    kept = res.summary["caveat"].iloc[0]
    assert kept.startswith("setup: GLUE needs the GENCODE v43 human annotation")
    assert "the command reads" not in kept and "normpeaks" not in kept
    blob = json.loads((tmp_path / "out" / "batch_result.json").read_text())
    assert blob["records"][0]["caveat"] == kept
    assert f"[run_all]   GLUE {kept}\n" in capsys.readouterr().out


def test_wrong_kind_and_scripts_ref_keep_both_reasons(tmp_path, monkeypatch):
    """Under a mismatched ref a wrong-kind row names both problems, and run_all
    still refuses before any method starts."""
    d = tmp_path / "MU_PEAK"
    d.mkdir()
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], [f"c{i}" for i in range(60)])
    _h5(d / "atac.h5", [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)],
        [f"c{i}" for i in range(60)])
    pd.DataFrame({"x": ["A", "B"] * 30}).to_csv(d / "cty.csv", index=False)
    wrong = "method scripts are at 0000000, not deadbeef (MULTIBENCH_SCRIPTS_REF)."
    monkeypatch.setattr(config, "scripts_ref_problem", lambda repo=None: wrong)
    monkeypatch.setattr(W, "_run", lambda *a, **k: pytest.fail("no method may start"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan("MU_PEAK", "vertical", methods=None, modalities=["rna", "atac"],
                      data_path=tmp_path, verbose=False).set_index("method")
        r = sc.loc["Matilda"]
        # the ref is its own blocker, like the wrong ATAC kind: the files are fine
        assert not r["runnable"] and r["files_ok"]
        assert r["reason"].startswith(wrong[:-1])
        assert "needs gene-activity ATAC; atac.h5 holds peaks. To run Matilda anyway" \
            in r["reason"]
        with pytest.raises(ValueError, match="nothing is runnable"):
            mtb.run_all("MU_PEAK", "vertical", tmp_path / "out",
                        modalities=["rna", "atac"], data_path=tmp_path, verbose=False)
