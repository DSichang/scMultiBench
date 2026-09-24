"""Study round 3, work package wf: R3-01, R3-02 and the workflow part of R3-17.

R3-01: a gene-activity token (``atac_gas`` / ``gas`` / ``gene_activity``)
selected moETM, scMM and iPOLNG in scan and run_all, although they read peaks
(their role is only named ``atac_gas``). The near-miss text of the
gene-activity rows named a looser rule than the peak rows.

R3-02: run_all attempted a row whose ATAC file held the other representation,
printed only the unused-batch note and kept no caveat in the records.

The env probe is pinned (every env installed, a GPU present) so the verdicts
hold on any host; ``_run`` is faked in the real-run tests.
"""
import inspect
import json
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, discover
from multibench import workflow as W
from multibench.engine import envs, registry

#: the caveat of a gene-activity method given peaks in atac.h5 (after the method)
GAS_CAV = ("needs gene-activity ATAC. The features of atac.h5 look like chr:start-end, "
           "so it holds peaks.")

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
GENES = [f"GENE{i}" for i in range(40)]
GAS_METHODS = {"Matilda", "scMDC", "UnitedNet"}
PEAK_METHODS = {"MIRA", "Seurat_WNN", "VIMCCA", "iPOLNG", "moETM", "scMM", "scMVP"}


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


# ----------------------------------------------------------------- R3-01
@pytest.mark.parametrize("tok", ["gene_activity", "gas", "atac_gas"])
def test_gene_activity_token_keeps_only_the_gene_activity_methods(tmp_path, tok):
    root = _vertical(tmp_path, "MU_GA", "atac.h5", GENES)
    want = set(discover.find_methods("vertical", atac="gene_activity"))
    assert want == GAS_METHODS
    df = _quiet(mtb.scan, "MU_GA", "vertical", modalities=["rna", tok], data_path=root,
                verbose=False)
    assert set(df["method"]) == want
    plan = _quiet(mtb.run_all, "MU_GA", "vertical", modalities=["rna", tok],
                  data_path=root, dry_run=True, verbose=False)
    assert set(plan["method"]) == want
    assert plan["runnable"].all()


def test_peak_token_still_keeps_the_peak_methods_behind_an_atac_gas_role(tmp_path):
    root = _vertical(tmp_path, "MU_PK", "atac.h5", PEAKS)
    for call in (mtb.scan, lambda *a, **k: mtb.run_all(*a, dry_run=True, **k)):
        df = _quiet(call, "MU_PK", "vertical", modalities=["rna", "atac_peak"],
                    data_path=root, verbose=False)
        assert set(df["method"]) == PEAK_METHODS
        assert {"moETM", "scMM", "iPOLNG"} <= set(df.loc[df.runnable, "method"])


def test_gene_activity_near_miss_states_the_vertical_rule(tmp_path):
    root = _vertical(tmp_path, "MU_TC", "atac_peak.h5", PEAKS)
    df = _quiet(mtb.scan, "MU_TC", "vertical", modalities=["rna", "atac"], data_path=root,
                verbose=False)
    for m in ("scMDC", "UnitedNet", "Matilda"):
        text = df.loc[df.method == m, "files_reason"].iloc[0]
        assert ("Vertical methods read atac.h5. method_info(\"" + m + "\")[\"atac\"] says "
                "which ATAC " + m + " needs.") in text, (m, text)
        assert "The folder holds atac_peak.h5" in text, (m, text)
        assert "atac_gas.h5 or atac.h5" not in text


def test_modality_token_notes_state_the_exception():
    for fn in (mtb.scan, mtb.run_all):
        flat = " ".join(inspect.getdoc(fn).split())
        assert ("moETM, scMM and iPOLNG read peaks through a role named ``atac_gas``, so "
                "``atac_peak`` selects them and ``atac_gas`` does not") in flat, fn.__name__


# ----------------------------------------------------------------- R3-02
def test_scan_blocks_the_other_atac_kind_unless_allowed(tmp_path):
    root = _vertical(tmp_path, "MU_PEAK", "atac.h5", PEAKS)
    for call in (mtb.scan, lambda *a, **k: mtb.run_all(*a, dry_run=True, **k)):
        df = _quiet(call, "MU_PEAK", "vertical", modalities=["rna", "atac"],
                    data_path=root, verbose=False).set_index("method")
        for m in GAS_METHODS:
            r = df.loc[m]
            assert not r.runnable and r.files_ok and r.env_ok, m
            assert r.reason == (f"{m} needs gene-activity ATAC, and atac.h5 holds peaks. "
                                f"Export the ATAC as gene activity, or pass "
                                f"allow_atac_mismatch=True to run {m} anyway."), r.reason
            assert r.caveat.startswith(f"{m} {GAS_CAV}")
        assert df.loc[["moETM", "scMM", "iPOLNG", "scMVP"], "runnable"].all()
    # R4-01: naming the method no longer runs it; allow_atac_mismatch does
    named = _quiet(mtb.run_all, "MU_PEAK", "vertical", methods=["Matilda"],
                   modalities=["rna", "atac"], data_path=root, dry_run=True, verbose=False)
    assert not named["runnable"].any()
    allowed = _quiet(mtb.run_all, "MU_PEAK", "vertical", methods=["Matilda"],
                     modalities=["rna", "atac"], data_path=root, dry_run=True,
                     verbose=False, allow_atac_mismatch=True)
    assert allowed["runnable"].all() and allowed["reason"].eq("").all()
    assert allowed["caveat"].str.startswith("Matilda needs gene-activity ATAC").all()


def test_scan_blocks_peak_methods_given_gene_activity(tmp_path):
    root = _vertical(tmp_path, "MU_GA", "atac.h5", GENES)
    df = _quiet(mtb.scan, "MU_GA", "vertical", modalities=["rna", "atac"], data_path=root,
                verbose=False).set_index("method")
    for m in ("moETM", "scMM", "iPOLNG", "scMVP"):
        assert not df.loc[m, "runnable"] and df.loc[m, "files_ok"], m
        assert df.loc[m, "reason"].startswith(f"{m} needs peak ATAC, and atac.h5 holds gene "
                                              "activity")
    assert df.loc[sorted(GAS_METHODS), "runnable"].all()


def test_the_cli_spelling_of_the_reason(tmp_path, monkeypatch):
    from multibench import config
    root = _vertical(tmp_path, "MU_PEAK", "atac.h5", PEAKS)
    monkeypatch.setattr(config, "_CLI", True)
    df = _quiet(mtb.scan, "MU_PEAK", "vertical", methods=None, data_path=root,
                verbose=False)
    r = df[(df.method == "Matilda") & (df.modalities == "rna+atac")].iloc[0]
    assert r.reason.endswith("or pass --allow-atac-mismatch to run Matilda anyway.")


class _Res:
    def __init__(self, out):
        self.output = out


def test_real_run_skips_it_prints_every_caveat_and_keeps_it(tmp_path, monkeypatch, capsys):
    root = _vertical(tmp_path, "MU_PEAK", "atac.h5", PEAKS)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(method)
        return _Res(np.zeros((60, 5)))
    monkeypatch.setattr(W, "_run", fake_run)
    out = tmp_path / "out"
    _quiet(mtb.run_all, "MU_PEAK", "vertical", out, modalities=["rna", "atac"],
           data_path=root, evaluate=False)
    log = capsys.readouterr().out
    assert not GAS_METHODS & set(calls)
    assert ("[run_all] skipping Matilda: Matilda needs gene-activity ATAC, and atac.h5 "
            "holds peaks.") in log
    # allowed: it runs, and the caveat reaches the log, the summary and the files
    res = _quiet(mtb.run_all, "MU_PEAK", "vertical", tmp_path / "named",
                 methods=["Matilda"], modalities=["rna", "atac"], data_path=root,
                 evaluate=False, allow_atac_mismatch=True)
    log = capsys.readouterr().out
    assert calls[-1] == "Matilda"
    assert f"[run_all]   Matilda {GAS_CAV}" in log
    sm = res.summary
    # R5-03: reason follows caveat as the last column
    assert list(sm.columns[-2:]) == ["caveat", "reason"]
    assert sm["caveat"].iloc[0].startswith(f"Matilda {GAS_CAV}")
    disk = pd.read_csv(tmp_path / "named" / "summary.csv")
    assert list(disk.columns)[-2:] == ["caveat", "reason"]
    assert disk["caveat"].iloc[0].startswith("Matilda needs gene-activity ATAC")
    blob = json.loads((tmp_path / "named" / "batch_result.json").read_text())
    assert blob["records"][0]["caveat"].startswith("Matilda needs gene-activity ATAC")
    assert mtb.load_batch(tmp_path / "named").summary["caveat"].iloc[0] == sm["caveat"].iloc[0]


def test_dry_run_prints_the_whole_caveat(tmp_path, capsys):
    root = _vertical(tmp_path, "MU_PEAK", "atac.h5", PEAKS)
    _quiet(mtb.run_all, "MU_PEAK", "vertical", methods=["Matilda"], data_path=root,
           dry_run=True, allow_atac_mismatch=True)
    assert (f"[run_all] Matilda {GAS_CAV}"
            in capsys.readouterr().out)


def test_scan_strict_counts_the_wrong_atac_kind_apart(tmp_path, capsys):
    root = _vertical(tmp_path, "MU_PEAK", "atac.h5", PEAKS)
    base = ["scan", "MU_PEAK", "--category", "vertical", "--data-path", str(root),
            "--modalities", "rna,atac_gas", "--strict"]
    rc = _quiet(cli.main, base)
    err = capsys.readouterr().err
    assert rc == 1 and "Rows with the wrong ATAC kind: 3." in err, err
    # R4-01: --methods naming them still fails; --allow-atac-mismatch passes
    rc = _quiet(cli.main, base + ["--methods", "Matilda,scMDC"])
    cap = capsys.readouterr()
    assert rc == 1 and "Rows with the wrong ATAC kind: 2." in cap.err, cap.err
    rc = _quiet(cli.main, base + ["--methods", "Matilda,scMDC", "--allow-atac-mismatch"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err


def test_old_records_without_a_caveat_still_summarise():
    rec = {"method": "Matilda", "status": "RUN_OK", "run_sec": 1.0}
    sm = W.BatchResult([rec], "D11", "vertical").summary
    assert list(sm.columns[-2:]) == ["caveat", "reason"] and sm["caveat"].isna().all()


@pytest.mark.parametrize("dataset,category", [("D11", "vertical"), ("D28", "diagonal"),
                                              ("D45", "mosaic"), ("D46", "mosaic"),
                                              ("D52", "cross")])
def test_demo_datasets_have_no_row_blocked_by_the_atac_kind(dataset, category):
    df = _quiet(mtb.scan, dataset, category, verbose=False)
    assert not W._is_wrong_atac(df["reason"]).any(), df.loc[
        W._is_wrong_atac(df["reason"]), ["method", "reason"]]


def test_run_all_notes_say_it_skips_the_other_representation():
    flat = " ".join(inspect.getdoc(mtb.run_all).split())
    assert "``run_all`` skips a method given the other representation, also when " \
           "``methods=`` names it. With ``allow_atac_mismatch=True`` the method runs" in flat
    assert "runs without an error and gives a wrong embedding; ``mtb.scan``" not in flat
    flat = " ".join(inspect.getdoc(mtb.scan).split())
    assert "``allow_atac_mismatch=True`` keeps such a row runnable, with its caveat." \
        in flat


# ----------------------------------------------------------------- R3-17
def test_generated_sounding_passages_are_rewritten():
    import importlib
    from multibench.data import results
    bar = importlib.import_module("multibench.plot.bar")
    docs = {
        "summary": inspect.getdoc(W.BatchResult.summary),
        "failures": inspect.getdoc(W.BatchResult.failures),
        "Degenerate": inspect.getdoc(results.DegenerateRerunWarning),
        "coverage": inspect.getdoc(results.results_coverage),
        "recommend": inspect.getdoc(results.recommend),
        "available": inspect.getdoc(results.available_datasets),
        "find_methods": inspect.getdoc(discover.find_methods),
        "list_methods": inspect.getdoc(discover.list_methods),
        "method_info": inspect.getdoc(discover.method_info),
        "metric_set_dir": inspect.getdoc(mtb.config.metric_set_dir),
        "bar_src": inspect.getsource(bar.bar),
        "degenerate_src": inspect.getsource(results._warn_degenerate),
    }
    gone = ["wired", "headline", "like with like", "Stored is not downloadable",
            "treat that row with suspicion", "worth fixing", "Two methods can both",
            "where rows are", "usually comes from a failed re-run"]
    for name, text in docs.items():
        flat = " ".join(text.split())
        for phrase in gone:
            assert phrase not in flat, (name, phrase)
    s = " ".join(docs["summary"].split())
    assert ("scMoMaT also writes a UMAP, which is scored: ``CHAIN_OK_GRAPH_METHOD``. "
            "Seurat_WNN writes only a neighbour graph: ``RUN_OK_NO_EMBEDDING``") in s
    assert "Near 1, one order clearly fits. Below about 0.5, two orders scored alike" in s
    f = " ".join(docs["failures"].split())
    assert ("no label file has as many cells as the output, so nothing was scored. Check "
            "the folder with ``mtb.inputs_for(check=True)`` and ``mtb.labels_for``.") in f


def test_run_records_keep_the_caveat_without_the_notes_on_starting(tmp_path, monkeypatch,
                                                                   capsys):
    """The scripts-fetch note is about starting the method, which run_all does."""
    from multibench import config
    root = _vertical(tmp_path, "MU_PEAK", "atac.h5", PEAKS)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_checkout")
    plan = _quiet(mtb.scan, "MU_PEAK", "vertical", methods=["Matilda"],
                  modalities=["rna", "atac"], data_path=root, verbose=False)
    assert "The method scripts are not in" in plan["caveat"].iloc[0]
    monkeypatch.setattr(W, "_run", lambda method, category, inputs, out_dir, params=None:
                        _Res(np.zeros((60, 5))))
    res = _quiet(mtb.run_all, "MU_PEAK", "vertical", tmp_path / "out", methods=["Matilda"],
                 modalities=["rna", "atac"], data_path=root, evaluate=False,
                 allow_atac_mismatch=True)
    log = capsys.readouterr().out
    cav = res.results[0]["caveat"]
    assert cav == f"Matilda {GAS_CAV}", cav
    assert "The method scripts are not in" not in log
    assert W._run_caveat("Matilda reads inputs/a.h5. mtb.run writes that file first") == ""
    assert W._run_caveat("Matilda needs raw counts. rna.h5 holds non-integer values. "
                         "Matilda reads inputs/a.h5. mtb.run writes that file first") == (
        "Matilda needs raw counts. rna.h5 holds non-integer values.")
