"""Study round 4, work package wf: R4-01, R4-02, R4-04, R4-11 and R4-14.

R4-01: naming a method in methods= (--methods) switched the ATAC checks of
scan off, so a cluster gate that picks one method per job passed a folder of
the other representation. An explicit allow_atac_mismatch replaces it; mtb.run
notes and warns.

R4-02: run_all(batch=) and rescore(batch=) matched a Series by position.

R4-04: a real run dropped every blocked row except the wrong-ATAC ones
without a word; they are now logged and recorded as SKIPPED.

R4-11: summary showed caveat None for a record saved before the field.

R4-14: reference entries without justification tails; the modality-token
rule of scan names what atac_peak alone keeps.

The env probe is pinned where a verdict depends on it; ``_run`` is faked in
the real-run tests.
"""
import inspect
import json
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config, discover
from multibench import workflow as W
from multibench.engine import envs, registry
from multibench.engine import runner as R

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
    genes = [f"g{i}" for i in range(30)]
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
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], bars)
    _h5(d / "atac.h5", atac_feats, bars)
    _labels(d / "cty.csv", n)
    return root


def _diagonal(root, name, peaks, n=20):
    d = root / name
    d.mkdir(parents=True)
    genes = [f"GENE{i}" for i in range(60)]
    _h5(d / "rna.h5", genes, [f"r{i}" for i in range(n)])
    _h5(d / "atac_peak.h5", peaks, [f"a{i}" for i in range(n)])
    _h5(d / "atac_gas.h5", genes, [f"a{i}" for i in range(n)])
    _labels(d / "rna_cty.csv", n)
    _labels(d / "atac_cty.csv", n)
    return root


class _Res:
    def __init__(self, out):
        self.output = out


# ======================================================================= R4-01
def test_named_methods_no_longer_bypass_the_atac_check(tmp_path, pinned):
    root = _gasmos(tmp_path)
    df = _quiet(mtb.scan, "GASMOS", "mosaic", methods=["StabMap", "scMoMaT"],
                data_path=root, verbose=False).set_index("method")
    for m in ("StabMap", "scMoMaT"):
        assert not df.loc[m, "runnable"] and df.loc[m, "files_ok"], m
        assert df.loc[m, "reason"] == (
            f"needs peak ATAC; atac2.h5 holds gene activity. Export the ATAC as peaks, "
            f"or pass allow_atac_mismatch=True to run {m} anyway."), df.loc[m, "reason"]
    ok = _quiet(mtb.scan, "GASMOS", "mosaic", methods=["StabMap", "scMoMaT"],
                data_path=root, verbose=False, allow_atac_mismatch=True)
    assert ok["runnable"].all() and ok["reason"].eq("").all()
    assert ok["caveat"].str.startswith("expects peaks; atac2.h5 holds gene activity").all()


def test_strict_gate_with_methods_fails_and_the_flag_passes_it(tmp_path, pinned, capsys):
    root = _gasmos(tmp_path)
    base = ["scan", "GASMOS", "--category", "mosaic", "--data-path", str(root),
            "--methods", "StabMap,scMoMaT", "--strict", "--assume-gpu"]
    rc = _quiet(cli.main, base)
    err = capsys.readouterr().err
    assert rc == 1 and "wrong ATAC kind in 2" in err, err
    for m in ("StabMap", "scMoMaT"):
        assert re.search(rf"^  {m}: needs peak ATAC; atac2.h5 holds gene activity\. Export "
                         r"the ATAC as peaks, or pass --allow-atac-mismatch to run "
                         rf"{m} anyway\.$", err, re.M), err
    rc = _quiet(cli.main, base + ["--allow-atac-mismatch", "--format", "csv"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    df = pd.read_csv(__import__("io").StringIO(cap.out))
    assert df["caveat"].str.startswith("expects peaks; atac2.h5 holds gene activity").all()


def test_run_all_named_matilda_on_peaks_is_blocked_unless_allowed(tmp_path, pinned, capsys):
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    plan = _quiet(mtb.run_all, "MU_PEAK", "vertical", methods=["Matilda"],
                  modalities=["rna", "atac"], data_path=root, dry_run=True, verbose=False)
    assert not plan["runnable"].any()
    assert plan["reason"].iloc[0].startswith("needs gene-activity ATAC; atac.h5 holds "
                                              "peaks. Export the ATAC as gene activity")
    capsys.readouterr()
    plan = _quiet(mtb.run_all, "MU_PEAK", "vertical", methods=["Matilda"],
                  modalities=["rna", "atac"], data_path=root, dry_run=True,
                  allow_atac_mismatch=True)
    assert plan["runnable"].all() and plan["reason"].eq("").all()
    assert plan["caveat"].iloc[0].startswith("expects gene activity; atac.h5 holds peaks")
    assert "[run_all] Matilda expects gene activity; atac.h5 holds peaks" in \
        capsys.readouterr().out


def test_real_run_all_with_the_override_runs_and_keeps_the_caveat(tmp_path, pinned,
                                                                  monkeypatch, capsys):
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(method)
        return _Res(np.zeros((60, 5)))
    monkeypatch.setattr(W, "_run", fake_run)
    with pytest.raises(ValueError, match="nothing is runnable"):
        _quiet(mtb.run_all, "MU_PEAK", "vertical", tmp_path / "blocked",
               methods=["Matilda"], modalities=["rna", "atac"], data_path=root,
               evaluate=False)
    assert calls == []
    res = _quiet(mtb.run_all, "MU_PEAK", "vertical", tmp_path / "named",
                 methods=["Matilda"], modalities=["rna", "atac"], data_path=root,
                 evaluate=False, allow_atac_mismatch=True)
    assert calls == ["Matilda"]
    assert "[run_all]   Matilda expects gene activity; atac.h5 holds peaks" in \
        capsys.readouterr().out
    assert res.summary["caveat"].iloc[0].startswith("expects gene activity")


def test_named_glue_with_peak_0_names_is_blocked(tmp_path, pinned):
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    peak_methods = discover.find_methods("diagonal", atac="peak")
    df = _quiet(mtb.scan, "LUNG_ids", "diagonal", methods=peak_methods, data_path=root,
                verbose=False).set_index("method")
    assert not df.loc["GLUE", "runnable"] and df.loc["GLUE", "files_ok"]
    assert df.loc["GLUE", "reason"] == (
        "reads peak names such as chr1:100-200; atac_peak.h5 holds other names (e.g. "
        "peak_0). Rename them to chr:start-end, or pass allow_atac_mismatch=True to run "
        "GLUE anyway.")
    assert W._is_wrong_atac(df["reason"]).loc[["GLUE", "Seurat_v3"]].all()


def test_is_wrong_atac_reads_the_override_not_the_prose():
    assert W._is_wrong_atac("anything; pass allow_atac_mismatch=True to run X anyway.")
    assert W._is_wrong_atac("anything; pass --allow-atac-mismatch to run X anyway.")
    assert not W._is_wrong_atac("conda env 'x' is not installed")
    s = pd.Series(["a --allow-atac-mismatch b", "missing adt.h5", None])
    assert list(W._is_wrong_atac(s)) == [True, False, False]


def test_run_dry_run_notes_and_real_run_warns(tmp_path, pinned, monkeypatch, capsys):
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    inp = mtb.inputs_for("MU_PEAK", "vertical", "Matilda", modalities=["rna", "atac"],
                         data_path=root)
    _quiet(mtb.run, "Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"),
           dry_run=True)
    err = capsys.readouterr().err
    assert re.search(r"^# Matilda expects gene activity; atac\.h5 holds peaks", err, re.M), err

    class Stop(Exception):
        pass

    def stop(spec):
        raise Stop
    monkeypatch.setattr(R, "check_gpu_requirement", stop)
    with pytest.warns(UserWarning, match=r"^Matilda expects gene activity; atac\.h5 holds "
                                         r"peaks"):
        with pytest.raises(Stop):
            mtb.run("Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"))
    # the right representation: no note
    root2 = _vertical(tmp_path, "MU_GA", GENES)
    inp2 = mtb.inputs_for("MU_GA", "vertical", "Matilda", modalities=["rna", "atac"],
                          data_path=root2)
    _quiet(mtb.run, "Matilda", "vertical", inputs=inp2, out_dir=str(tmp_path / "o2"),
           dry_run=True)
    assert "expects" not in capsys.readouterr().err


def test_cli_run_dry_run_prints_the_note(tmp_path, pinned, capsys):
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    d = root / "LUNG_ids"
    rc = _quiet(cli.main, ["run", "--method", "GLUE", "--category", "diagonal",
                           "--input", f"rna={d / 'rna.h5'}",
                           "--input", f"atac_peak={d / 'atac_peak.h5'}",
                           "--out-dir", str(tmp_path / "o"), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "# GLUE reads peak names such as chr1:100-200; atac_peak.h5 holds other names " \
           "(e.g. peak_0)" in err, err


def test_layout_lines_name_the_override_in_short_sentences(monkeypatch):
    head = ("A method whose ATAC file holds the other representation gives a wrong "
            "embedding.")
    txt = mtb.describe_layout("mosaic")
    lines = txt.splitlines()
    i = lines.index(head)
    assert lines[i + 1:i + 3] == [
        "mtb.scan and mtb.run_all skip such a method unless allow_atac_mismatch=True.",
        "mtb.run only warns, so check method_info(m)['atac'] first."]
    assert "Named in" not in txt
    monkeypatch.setattr(config, "_CLI", True)
    lines = mtb.describe_layout("vertical").splitlines()
    i = lines.index(head)
    assert lines[i + 1:i + 3] == [
        "multibench scan and run-all skip such a method unless --allow-atac-mismatch "
        "is given.",
        "multibench run only warns, so check multibench info METHOD first."]
    assert not any("`" in l or ";" in l for l in lines[i:i + 3])


def test_docstrings_name_the_override_not_methods():
    from multibench.engine import ingest
    for fn in (mtb.scan, mtb.run_all, ingest.export_dataset):
        flat = " ".join(inspect.getdoc(fn).split())
        assert "allow_atac_mismatch=True" in flat, fn.__name__
        assert "name it in" not in flat and "Name the method in" not in flat
        assert "named in ``methods=``, or given" not in flat
    for fn in (mtb.scan, mtb.run_all):
        assert "allow_atac_mismatch" in inspect.signature(fn).parameters
    for sub in ("scan", "run-all"):
        parser = cli.build_parser()
        action = next(a for a in parser._actions if hasattr(a, "choices") and a.choices
                      and sub in a.choices)
        assert any("--allow-atac-mismatch" in a.option_strings
                   for a in action.choices[sub]._actions), sub


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
    kw = dict(methods=["Matilda"], modalities=["rna", "adt"], data_path=data,
              verbose=False)
    ordered = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "a", batch=batch, **kw)
    series = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "b", batch=shuffled,
                    **kw)
    assert _batch_metrics(ordered)[0] == "CHAIN_OK"
    assert _batch_metrics(ordered)[1] is not None
    assert _batch_metrics(series) == _batch_metrics(ordered)
    frame = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "c",
                   batch=shuffled.to_frame("sample"), **kw)
    assert _batch_metrics(frame) == _batch_metrics(ordered)


def test_rescore_aligns_a_barcode_indexed_series(cite, tmp_path):
    data, batch, shuffled = cite
    res = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "a", methods=["Matilda"],
                 modalities=["rna", "adt"], data_path=data, verbose=False)
    back = mtb.load_batch(tmp_path / "a")
    assert _batch_metrics(_quiet(back.rescore, batch=shuffled)) == \
        _batch_metrics(_quiet(res.rescore, batch=batch))


def test_a_series_with_foreign_ids_raises(cite, tmp_path):
    data, batch, shuffled = cite
    bad = shuffled.rename(index=lambda b: "x" + b)
    kw = dict(methods=["Matilda"], modalities=["rna", "adt"], data_path=data,
              verbose=False)
    for extra in ({}, {"dry_run": True}):
        with pytest.raises(ValueError, match=r"batch: 120 id\(s\) are not cells of MYCITE "
                                             r"\(first: \['xc"):
            _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "x", batch=bad,
                   **kw, **extra)
    assert not (tmp_path / "x").exists()
    res = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "a", **kw)
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
    with pytest.warns(UserWarning, match=r"batch Series has a non-default index, but the "
                                         r"files of MYCITE have no usable cell ids"):
        vec = W._batch_vector(series, "MYCITE", data)
    assert list(vec) == list(batch)


def test_arrays_lists_and_range_indexed_series_stay_positional(cite):
    data, batch, _ = cite
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for x in (batch, list(batch), pd.Series(batch)):
            assert list(W._batch_vector(x, "MYCITE", data)) == list(batch)


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


def test_named_blocked_methods_are_logged_and_recorded(tmp_path, monkeypatch, capsys):
    calls = _skip_setup(monkeypatch, tmp_path)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    res = _quiet(mtb.run_all, "MU_PEAK", "vertical", tmp_path / "out",
                 methods=["scMVP", "UnitedNet", "MIRA"], data_path=root, evaluate=False)
    log = capsys.readouterr().out
    assert calls == ["scMVP"]
    assert "[run_all] skipping UnitedNet: " in log and "[run_all] skipping MIRA: " in log
    sm = res.summary.set_index("method")
    assert sm.loc["UnitedNet", "status"] == "SKIPPED" == sm.loc["MIRA", "status"]
    fails = res.failures.set_index("method")
    assert set(fails.index) == {"UnitedNet", "MIRA"}
    assert "logger.py" in fails.loc["MIRA", "error"]
    assert "needs an NVIDIA GPU" in fails.loc["UnitedNet", "error"]
    assert "to run UnitedNet anyway; UnitedNet needs an NVIDIA GPU" in \
        fails.loc["UnitedNet", "error"]
    disk = pd.read_csv(tmp_path / "out" / "summary.csv").set_index("method")
    assert disk.loc["MIRA", "status"] == "SKIPPED"
    blob = json.loads((tmp_path / "out" / "batch_result.json").read_text())
    rec = next(r for r in blob["records"] if r["method"] == "MIRA")
    assert rec["status"] == "SKIPPED" and rec["requested"] is True
    back = mtb.load_batch(tmp_path / "out")
    assert set(back.failures["method"]) == {"UnitedNet", "MIRA"}
    assert "2 skipped" in repr(back)
    # rescore keeps a SKIPPED record as it is
    again = _quiet(back.rescore, metrics=["ARI"])
    assert {r["method"]: r["status"] for r in again.records}["MIRA"] == "SKIPPED"


def test_unnamed_gpu_rows_are_skipped_but_not_failures(tmp_path, monkeypatch, capsys):
    _skip_setup(monkeypatch, tmp_path)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    res = _quiet(mtb.run_all, "MU_PEAK", "vertical", tmp_path / "out",
                 modalities=["rna", "atac_peak"], data_path=root, evaluate=False)
    log = capsys.readouterr().out
    sm = res.summary.set_index("method")
    for m in ("moETM", "iPOLNG"):
        assert sm.loc[m, "status"] == "SKIPPED", m
        assert f"[run_all] skipping {m}: " in log
    assert sm.loc["MIRA", "status"] == "SKIPPED"
    assert res.failures.empty, res.failures


def test_rows_without_files_are_counted_not_recorded(tmp_path, monkeypatch, capsys):
    _skip_setup(monkeypatch, tmp_path)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    res = _quiet(mtb.run_all, "MU_PEAK", "vertical", tmp_path / "out", data_path=root,
                 evaluate=False)
    log = capsys.readouterr().out
    m = re.search(r"^\[run_all\] (\d+) other variants need files this folder does not "
                  r"have; see mtb\.scan$", log, re.M)
    assert m, log
    statuses = {r["method"]: r["status"] for r in res.records}
    assert "totalVI" not in statuses          # rna+adt only: adt.h5 is not there
    assert res.failures.empty


def test_a_named_method_with_a_runnable_mosaic_variant_is_not_a_failure(
        tmp_path, monkeypatch, capsys):
    _skip_setup(monkeypatch, tmp_path)
    res = _quiet(mtb.run_all, "D45", "mosaic", tmp_path / "out", methods=["Multigrate"],
                 evaluate=False)
    log = capsys.readouterr().out
    assert [r["status"] for r in res.records] == ["RUN_OK"]
    assert res.failures.empty
    assert "[run_all] 1 other variant needs files this folder does not have" in log
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
    assert W._earlier_provenance(out) == {"Matilda": {"scripts_commit": "abc",
                                                      "env_flavor": "gpu",
                                                      "hostname": "h"}}
    # a later run of the method replaces a SKIPPED record
    ran = {"method": "totalVI", "status": "RUN_OK"}
    _quiet(W.BatchResult([ran], "D11", "vertical").save, out)
    assert {r["method"]: r["status"] for r in mtb.load_batch(out).records}["totalVI"] \
        == "RUN_OK"


def test_summary_notes_list_skipped():
    doc = " ".join(inspect.getdoc(W.BatchResult.summary).split())
    assert "- ``SKIPPED`` - blocked before the run" in doc
    doc = " ".join(inspect.getdoc(W.BatchResult.failures).split())
    assert "``SKIPPED``, for a method that ``methods=`` named" in doc


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
    doc = " ".join(inspect.getdoc(W.BatchResult.summary).split())
    assert "the record was saved before this column existed" in doc


# ======================================================================= R4-14
def test_reference_entries_lose_their_justification_tails():
    from multibench.data import results
    docs = {
        "long": inspect.getdoc(W.BatchResult.long),
        "plot": inspect.getdoc(W.BatchResult.plot),
        "summary": inspect.getdoc(W.BatchResult.summary),
        "method_info": inspect.getdoc(discover.method_info),
        "load_results": inspect.getdoc(results.load_results),
        "available": inspect.getdoc(results.available_datasets),
        "create_all": inspect.getdoc(envs.create_all),
    }
    flat = {k: " ".join(v.split()) for k, v in docs.items()}
    assert "still plots" not in flat["long"]
    assert "Otherwise the record contributes its ``metrics`` dict." in flat["long"]
    assert "Read it next to ``summary`` -" not in flat["plot"]
    assert ("so with few methods a small gap fills the whole colour scale. Check the "
            "values in ``summary``.") in flat["plot"]
    assert "not a difference, because" not in flat["summary"]
    assert "``(best - runner_up) / best``" in flat["summary"]
    assert "not predictions" not in flat["method_info"]
    assert "Use them to set ``run_all(timeout=...)``." in flat["method_info"]
    assert "meaningless ranks" not in flat["load_results"]
    assert docs["available"].splitlines()[0] == "Dataset ids that have stored results."
    assert "not the datasets that can be downloaded" not in flat["available"]
    assert "Only a few of the benchmark's datasets" not in flat["available"]
    assert ("An id listed here but not by ``mtb.data.fetchable()`` has metric tables and "
            "no data file to download.") in flat["available"]
    assert "wired" not in flat["create_all"]
    assert "Methods with a variant in this category; default: all." in flat["create_all"]


def test_modality_token_notes_say_what_atac_peak_alone_keeps(pinned):
    for fn in (mtb.scan, mtb.run_all):
        flat = " ".join(inspect.getdoc(fn).split())
        assert ("Base tokens keep a row whose modalities are exactly that combination"
                in flat), fn.__name__
        assert "Representation tokens select by what the method reads" in flat
        assert "A row is kept when its modalities are exactly" not in flat
    flat = " ".join(inspect.getdoc(mtb.scan).split())
    assert ("``atac_peak`` alone also keeps MultiMAP and Seurat_v3, which read both "
            "files") in flat
    assert "``atac_peak`` together with ``atac_gas`` keeps only those two" in flat
    # the facts the text states
    one = _quiet(mtb.scan, "D28", "diagonal", modalities=["rna", "atac_peak"],
                 verbose=False)
    assert {"MultiMAP", "Seurat_v3"} <= set(one["method"])
    both = _quiet(mtb.scan, "D28", "diagonal", modalities=["rna", "atac_peak", "atac_gas"],
                  verbose=False)
    assert set(both["method"]) == {"MultiMAP", "Seurat_v3"}
