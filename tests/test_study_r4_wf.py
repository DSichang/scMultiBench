"""Study round 4, work package wf: R4-01, R4-02, R4-04, R4-11 and R4-14.

R4-01: naming a method in methods= (--methods) switched the ATAC checks of
scan off, so a cluster gate that picks one method per job passed a folder of
the other representation. An explicit allow_atac_mismatch replaces it; mtb.run
notes and warns.

R4-02: run(batch=) and rescore(batch=) matched a Series by position.

R4-04: a real run dropped every blocked row except the wrong-ATAC ones
without a word; they are now logged and recorded as SKIPPED.

R4-11: summary showed caveat None for a record saved before the field.

R4-14: reference entries without justification tails; the modality-token
rule of scan names what atac_peak alone keeps.

The env probe is pinned where a verdict depends on it; ``_run`` is faked in
the real-run tests.
"""
import argparse
import inspect
import json
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config, discover
from multibench import workflow as W
from multibench.engine import envs, registry
from multibench.engine import runner as R
from tests.conftest import full_scan

#: the caveat of a gene-activity method given peaks in atac.h5 (after the method)
GAS_CAV = ("needs gene-activity ATAC. The features of atac.h5 look like chr:start-end, "
           "so it holds peaks.")
#: the caveat of a peak method given gene activity in atac2.h5 (after the method)
GP_CAV = ("needs peak ATAC. The features of atac2.h5 do not look like chr:start-end, "
          "so it seems to hold gene activity.")

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
GENES = [f"GENE{i}" for i in range(40)]


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Every env installed, a GPU, no method scripts on this machine, no ref."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _h5(path, feats, bars, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), len(bars)))
                         .astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array(bars, dtype="S32"))


def _labels(path, n):
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(path, index=False)


def _gasmos(root, name="GASMOS", n=40):
    """A D46-like mosaic folder whose batch-2 ATAC holds gene activity."""
    d = root / name
    d.mkdir(parents=True)
    # gene activity is named by the RNA's genes, as in real data
    genes = GENES[:30]
    for b in (1, 2, 3):
        bars = [f"b{b}c{i}" for i in range(n)]
        _h5(d / f"rna{b}.h5", genes, bars)
        _labels(d / f"cty{b}.csv", n)
        if b == 1:
            _h5(d / "adt1.h5", [f"p{i}" for i in range(8)], bars)
        if b == 2:
            _h5(d / "atac2.h5", GENES, bars)
    return root


def _vertical(root, name, atac_feats, n=60):
    d = root / name
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", GENES[:30], bars)
    _h5(d / "atac.h5", atac_feats, bars)
    _labels(d / "cty.csv", n)
    return root


class _Res:
    def __init__(self, out):
        self.output = out


def test_is_wrong_atac_reads_the_blocking_atac_reasons_not_other_prose():
    kind = ("Matilda needs gene-activity ATAC, and atac.h5 holds peaks. Export the ATAC "
            "as gene activity.")
    names = ("GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds other "
             "names, for example peak_0. Rename them to chr:start-end.")
    assert W._is_wrong_atac(kind) and W._is_wrong_atac(names)
    assert W._is_wrong_atac("StabMap needs peak ATAC, and atac2.h5 holds gene activity. "
                            "Export the ATAC as peaks.")
    assert not W._is_wrong_atac("conda env 'x' is not installed")
    assert not W._is_wrong_atac("moETM needs an NVIDIA GPU.")
    assert not W._is_wrong_atac("missing atac.h5")
    s = pd.Series([kind, names, "missing adt.h5", None])
    assert list(W._is_wrong_atac(s)) == [True, True, False, False]


def test_run_dry_run_notes_and_real_run_warns(tmp_path, pinned, monkeypatch, capsys,
                                              gas_matilda):
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    inp = mtb.engine.resolve.inputs_for("MU_PEAK", "vertical", "Matilda", modalities=["rna", "atac"],
                         data_path=root)
    _quiet(mtb.engine.runner.run, "Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"),
           dry_run=True)
    err = capsys.readouterr().err
    assert re.search(r"^# Matilda needs gene-activity ATAC\. The features of atac\.h5 look "
                     r"like chr:start-end, so it holds peaks\.$", err, re.M), err

    class Stop(Exception):
        pass

    def stop(spec):
        raise Stop
    monkeypatch.setattr(R, "check_gpu_requirement", stop)
    with pytest.warns(UserWarning, match=r"^Matilda needs gene-activity ATAC\. The features "
                                         r"of atac\.h5 look like chr:start-end, so it holds "
                                         r"peaks\.$"):
        with pytest.raises(Stop):
            mtb.engine.runner.run("Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"))
    # the right representation: no note
    root2 = _vertical(tmp_path, "MU_GA", GENES)
    inp2 = mtb.engine.resolve.inputs_for("MU_GA", "vertical", "Matilda", modalities=["rna", "atac"],
                          data_path=root2)
    _quiet(mtb.engine.runner.run, "Matilda", "vertical", inputs=inp2, out_dir=str(tmp_path / "o2"),
           dry_run=True)
    assert "needs gene-activity ATAC" not in capsys.readouterr().err


# ======================================================================= R4-02
N = 120


@pytest.fixture
def cite(tmp_path, monkeypatch):
    """A CITE-seq folder (barcodes c0..c119) and a fake run whose embedding
    separates the two label groups and shifts the cells of batch 2."""
    d = tmp_path / "data" / "MYCITE"
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(N)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], bars)
    _h5(d / "adt.h5", [f"p{i}" for i in range(6)], bars)
    pd.DataFrame({"x": ["A"] * (N // 2) + ["B"] * (N // 2)}).to_csv(d / "cty.csv",
                                                                    index=False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    batch = np.tile([1, 2], N // 2)

    def fake_run(method, category, inputs, out_dir, params=None):
        rng = np.random.default_rng(7)
        emb = rng.normal(size=(N, 4))
        emb[N // 2:, 0] += 8.0
        emb[batch == 2, 1] += 8.0                  # a strong batch effect
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    shuffled = pd.Series(batch, index=bars).sample(frac=1.0, random_state=3)
    return tmp_path / "data", batch, shuffled


def _batch_metrics(res):
    # iLISI is None where the LISI backend is unavailable; ASW_batch always exists
    row = res.summary.iloc[0]
    return (row["status"],) + tuple(None if pd.isna(row.get(k)) else round(float(row[k]), 4)
                                    for k in ("ASW_batch", "GC", "iLISI"))


def test_run_all_aligns_a_barcode_indexed_series(cite, tmp_path):
    data, batch, shuffled = cite
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['Matilda'], modalities=['rna', 'adt'], verbose=False)
    ordered = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "a", batch=batch, **kw)
    series = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "b", batch=shuffled,
                    **kw)
    assert _batch_metrics(ordered)[0] == "scored"
    assert _batch_metrics(ordered)[1] is not None
    assert _batch_metrics(series) == _batch_metrics(ordered)
    frame = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "c",
                   batch=shuffled.to_frame("sample"), **kw)
    assert _batch_metrics(frame) == _batch_metrics(ordered)


def test_rescore_aligns_a_barcode_indexed_series(cite, tmp_path):
    data, batch, shuffled = cite
    mtb.config.DEFAULT.data_path = data
    res = _quiet(mtb.run, 'MYCITE', 'vertical', tmp_path / 'a', methods=['Matilda'], modalities=['rna', 'adt'], verbose=False)
    back = mtb.load_batch(tmp_path / "a")
    assert _batch_metrics(_quiet(back.rescore, batch=shuffled)) == \
        _batch_metrics(_quiet(res.rescore, batch=batch))


def test_a_series_with_foreign_ids_raises(cite, tmp_path):
    data, batch, shuffled = cite
    bad = shuffled.rename(index=lambda b: "x" + b)
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['Matilda'], modalities=['rna', 'adt'], verbose=False)
    for extra in ({}, {"dry_run": True}):
        with pytest.raises(ValueError, match=r"batch: 120 ids are not cells of MYCITE "
                                             r"\(first: \['xc"):
            _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "x", batch=bad,
                   **kw, **extra)
    assert not (tmp_path / "x").exists()
    res = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "a", **kw)
    with pytest.raises(ValueError, match="are not cells of MYCITE"):
        _quiet(res.rescore, batch=bad)
    with pytest.raises(ValueError, match=r"have no id in batch"):
        _quiet(res.rescore, batch=shuffled.iloc[:-5])


def test_without_usable_barcodes_the_series_is_positional_with_a_warning(cite, tmp_path):
    data, batch, shuffled = cite
    for f in ("rna.h5", "adt.h5"):
        with h5py.File(data / "MYCITE" / f, "a") as h:
            del h["matrix/barcodes"]
            h["matrix/barcodes"] = np.array(["same"] * N, dtype="S8")
    series = pd.Series(batch, index=[f"c{i}" for i in range(N)])
    with pytest.warns(UserWarning, match=r"The batch Series is matched by position, "
                                         r"because the files of MYCITE have no usable "
                                         r"cell ids"):
        vec = W._batch_vector(series, "MYCITE", data)
    assert list(vec) == list(batch)


def test_arrays_lists_and_range_indexed_series_stay_positional(cite):
    data, batch, _ = cite
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for x in (batch, list(batch)):
            assert list(W._batch_vector(x, "MYCITE", data)) == list(batch)
    # a Series that numbers its rows is positional too, and since the owner's
    # decision on 0.3.5 it says so: the cells of MYCITE have barcodes
    with pytest.warns(UserWarning) as rec:
        assert list(W._batch_vector(pd.Series(batch), "MYCITE", data)) == list(batch)
    assert [str(w.message) for w in rec] == [
        "batch is a Series without barcodes and is matched to the cells by position. To "
        "match by barcode, pass a Series indexed by barcode."]


def test_dataset_ids_follow_the_label_files_one_data_file_each(tmp_path):
    root = _gasmos(tmp_path)
    ids = W._dataset_cell_ids("GASMOS", root)
    assert ids[:2] == ["b1c0", "b1c1"] and ids[40] == "b2c0" and len(ids) == 120
    s = pd.Series([int(i[1]) for i in ids], index=ids).iloc[::-1]
    assert list(W._batch_vector(s, "GASMOS", root)) == [1] * 40 + [2] * 40 + [3] * 40


# ======================================================================= R4-04
def _skip_setup(monkeypatch, tmp_path):
    """Stand-in envs on a host without a GPU; MIRA's script check fails."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    real = W._missing_script

    def missing(variant, *, method=None):
        if method == "MIRA":
            return ("MIRA's script imports logger.py, which the public scMultiBench "
                    "repository does not include")
        return real(variant, method=method)
    monkeypatch.setattr(W, "_missing_script", missing)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(method)
        return _Res(np.zeros((60, 5)))
    monkeypatch.setattr(W, "_run", fake_run)
    return calls


def test_named_blocked_methods_are_logged_and_recorded(tmp_path, monkeypatch, capsys,
                                                       as_gene_activity):
    # UnitedNet stands for a method that needs a GPU and reads the other ATAC form
    as_gene_activity("UnitedNet")
    calls = _skip_setup(monkeypatch, tmp_path)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    mtb.config.DEFAULT.data_path = root
    res = _quiet(mtb.run, 'MU_PEAK', 'vertical', tmp_path / 'out', methods=['scMVP', 'UnitedNet', 'MIRA'], evaluate=False)
    log = capsys.readouterr().out
    assert calls == ["scMVP"]
    assert "[run] skipping UnitedNet: " in log and "[run] skipping MIRA: " in log
    sm = res.summary.set_index("method")
    assert sm.loc["UnitedNet", "status"] == "skipped" == sm.loc["MIRA", "status"]
    fails = res.summary[res.summary.status != "scored"].set_index("method")
    assert set(fails.index) == {"scMVP", "UnitedNet", "MIRA"}
    assert "logger.py" in fails.loc["MIRA", "reason"]
    assert "needs an NVIDIA GPU" in fails.loc["UnitedNet", "reason"]
    # the export fix ends the reason, after the GPU sentence; no override (0.4.0)
    assert fails.loc["UnitedNet", "reason"].endswith("Export the ATAC as gene activity.")
    assert "allow_atac_mismatch" not in fails.loc["UnitedNet", "reason"]
    disk = pd.read_csv(tmp_path / "out" / "summary.csv").set_index("method")
    assert disk.loc["MIRA", "status"] == "skipped"
    blob = json.loads((tmp_path / "out" / "batch_result.json").read_text())
    rec = next(r for r in blob["records"] if r["method"] == "MIRA")
    assert rec["status"] == "SKIPPED" and rec["requested"] is True
    back = mtb.load_batch(tmp_path / "out")
    assert set(back.summary.query("status == 'skipped'")["method"]) == {"UnitedNet", "MIRA"}
    assert "2 skipped" in repr(back)
    # rescore keeps a SKIPPED record as it is (scMVP's fake run left no output,
    # and a tree with no output to score raises: MIRA alone)
    again = _quiet(mtb.load_batch(tmp_path / "out", methods=["MIRA"]).rescore,
                   metrics=["ARI"])
    assert {r["method"]: r["status"] for r in again.records}["MIRA"] == "SKIPPED"
    with pytest.raises(FileNotFoundError, match="^No output of "):
        _quiet(back.rescore, metrics=["ARI"])


def test_unnamed_gpu_rows_are_skipped(tmp_path, monkeypatch, capsys):
    _skip_setup(monkeypatch, tmp_path)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    mtb.config.DEFAULT.data_path = root
    res = _quiet(mtb.run, 'MU_PEAK', 'vertical', tmp_path / 'out', modalities=['rna', 'atac_peak'], evaluate=False)
    log = capsys.readouterr().out
    sm = res.summary.set_index("method")
    for m in ("moETM", "iPOLNG"):
        assert sm.loc[m, "status"] == "skipped", m
        assert f"[run] skipping {m}: " in log
    assert sm.loc["MIRA", "status"] == "skipped"


def test_rows_without_files_are_counted_not_recorded(tmp_path, monkeypatch, capsys):
    _skip_setup(monkeypatch, tmp_path)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    mtb.config.DEFAULT.data_path = root
    res = _quiet(mtb.run, 'MU_PEAK', 'vertical', tmp_path / 'out', evaluate=False, modalities='all')
    log = capsys.readouterr().out
    m = re.search(r"^\[run\] (\d+) more rows need files this folder does not "
                  r"have\. mtb\.scan shows them\.$", log, re.M)
    assert m, log
    statuses = {r["method"]: r["status"] for r in res.records}
    assert "totalVI" not in statuses          # rna+adt only: adt.h5 is not there


def test_a_named_method_with_a_runnable_mosaic_variant_is_not_a_failure(
        tmp_path, monkeypatch, capsys):
    _skip_setup(monkeypatch, tmp_path)
    res = _quiet(mtb.run, "D45", "mosaic", tmp_path / "out", methods=["Multigrate"],
                 evaluate=False, modalities="all")
    log = capsys.readouterr().out
    assert [r["status"] for r in res.records] == ["RUN_OK"]
    # Multigrate's three other mosaic rows are for patterns D45 does not hold
    assert ("[run] 3 more rows need files this folder does not have. mtb.scan shows "
            "them.") in log
    assert "skipping Multigrate" not in log


def test_a_skipped_record_never_replaces_an_earlier_run(tmp_path):
    out = tmp_path / "out"
    ok = {"method": "Matilda", "status": "CHAIN_OK", "metrics": {"ARI": 0.5},
          "scripts_commit": "abc", "env_flavor": "gpu", "hostname": "h"}
    W.BatchResult([ok], "D11", "vertical").save(out)
    skip = {"method": "Matilda", "status": "SKIPPED", "error": "env", "requested": True}
    other = {"method": "totalVI", "status": "SKIPPED", "error": "env", "requested": False}
    _quiet(W.BatchResult([skip, other], "D11", "vertical").save, out)
    back = {r["method"]: r["status"] for r in mtb.load_batch(out).records}
    assert back == {"Matilda": "CHAIN_OK", "totalVI": "SKIPPED"}
    assert {m: r["hostname"] for m, r in W._saved_records(out).items()} == {"Matilda": "h"}
    # a later run of the method replaces a SKIPPED record
    ran = {"method": "totalVI", "status": "RUN_OK"}
    _quiet(W.BatchResult([ran], "D11", "vertical").save, out)
    assert {r["method"]: r["status"] for r in mtb.load_batch(out).records}["totalVI"] \
        == "RUN_OK"


def test_summary_notes_list_skipped():
    doc = " ".join(inspect.getdoc(W.BatchResult.summary).split())
    assert "skipped or timeout" in doc and "why the method has no scores" in doc


# ======================================================================= R4-11
def test_old_records_show_nan_and_new_ones_keep_their_text():
    old = {"method": "UINMF", "status": "CHAIN_OK", "n_batches": 2}
    new = {"method": "StabMap", "status": "CHAIN_OK", "caveat": ""}
    cav = {"method": "sciPENN", "status": "CHAIN_OK", "caveat": "reads batches 1-2 of 3"}
    sm = W.BatchResult([old, new, cav], "D52", "cross").summary.set_index("method")
    v = sm.loc["UINMF", "caveat"]
    assert v is not None and isinstance(v, float) and np.isnan(v)
    assert sm.loc["StabMap", "caveat"] == ""
    assert sm.loc["sciPENN", "caveat"] == "reads batches 1-2 of 3"
    none = W.BatchResult([dict(old, caveat=None)], "D52", "cross").summary
    assert isinstance(none["caveat"].iloc[0], float)


def test_modality_token_notes_say_what_atac_peak_alone_keeps(pinned):
    # atac_peak alone also keeps MultiMAP and Seurat_v3; with atac_gas, only those two
    one = _quiet(full_scan, "D28", "diagonal", modalities=["rna", "atac_peak"],
                 verbose=False)
    assert {"MultiMAP", "Seurat_v3"} <= set(one["method"])
    both = _quiet(full_scan, "D28", "diagonal", modalities=["rna", "atac_peak", "atac_gas"],
                  verbose=False)
    assert set(both["method"]) == {"MultiMAP", "Seurat_v3"}


def test_docstrings_name_the_override_not_methods():
    from multibench.engine import ingest
    for fn in (mtb.scan, mtb.run, ingest.export_dataset):
        flat = ' '.join(inspect.getdoc(fn).split())
        assert 'name it in' not in flat and 'Name the method in' not in flat
        assert 'named in ``methods=``, or given' not in flat
    for fn in (mtb.scan, mtb.run):
        assert 'allow_atac_mismatch' not in ' '.join(inspect.getdoc(fn).split()), fn.__name__
        assert 'allow_atac_mismatch' not in inspect.signature(fn).parameters
