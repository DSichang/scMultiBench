"""run and rescore do not score a missing label, nor train on one (0.3.5).

A label file with ``NA`` or an empty field gave a ``CHAIN_OK`` record up to
0.3.4: the missing labels were scored as one more cell type. A real run now
stops before its first method, a dry run says so, and ``rescore`` records the
reason per record. A method that reads the label file as an input stops the
run with ``evaluate=False`` as well.

A synthetic CITE-seq folder and a fake ``_run`` that writes ``embedding.h5``
keep these independent of any conda env.
"""
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, registry
from multibench.eval import io as eio

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
N = 120
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _h5(path, feats, bars):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), len(bars)))
                         .astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array(bars, dtype="S32"))


def _write(path, labels):
    pd.DataFrame({"x": labels}).to_csv(path, index=False)


def _with_gaps(n=N, na=7, empty=2):
    """``n`` labels of two types: ``na`` say NA, ``empty`` are empty fields."""
    lab = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2), dtype=object)
    lab[:na] = "NA"
    lab[na:na + empty] = None
    return lab


LABELS = np.array(["A"] * (N // 2) + ["B"] * (N // 2), dtype=object)


class _Res:
    def __init__(self, out):
        self.output = out


@pytest.fixture
def host(monkeypatch, tmp_path):
    """A Linux host with every environment and a GPU, no method scripts; the
    fake ``_run`` writes an embedding of N cells and records the method."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(method)
        emb = np.random.default_rng(7).normal(size=(N, 4))
        emb[N // 2:, 0] += 8.0                       # two clean clusters = the labels
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    return calls


@pytest.fixture
def cite(tmp_path, host):
    """``<tmp>/data/MYCITE``: RNA + ADT of N cells with complete labels."""
    d = tmp_path / "data" / "MYCITE"
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(N)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], bars)
    _h5(d / "adt.h5", [f"p{i}" for i in range(6)], bars)
    _write(d / "cty.csv", LABELS)
    return tmp_path / "data"


def _run(data, tmp_path, methods, **kw):
    mtb.config.data_path = data
    return _quiet(mtb.run, 'MYCITE', 'vertical', tmp_path / 'out', methods=methods, verbose=False, **kw)


def _refused(data, tmp_path, methods, **kw):
    with pytest.raises(ValueError) as e:
        _run(data, tmp_path, methods, **kw)
    return str(e.value)


HEAD = "cty.csv of MYCITE has 9 of 120 cells without a label, for example 'NA'"
FIX = ". Write the folder again with a label for every cell."
#: the whole message of a run that scores; it is the same for every method
M1 = HEAD + FIX
#: evaluate=False was passed: the message names a method that reads the label files
M1_MATILDA = HEAD + ", and Matilda reads the label files" + FIX


# ------------------------------------------------------------------ a real run
def test_run_all_stops_before_any_method(cite, tmp_path, host):
    _write(cite / "MYCITE" / "cty.csv", _with_gaps())
    before = {p.name: p.read_bytes() for p in (cite / "MYCITE").iterdir()}
    assert _refused(cite, tmp_path, ["totalVI"]) == M1
    assert host == [] and not (tmp_path / "out").exists()
    assert {p.name: p.read_bytes() for p in (cite / "MYCITE").iterdir()} == before
    # every method of the category: the same stop
    with pytest.raises(ValueError) as e:
        mtb.config.data_path = cite
        _quiet(mtb.run, 'MYCITE', 'vertical', tmp_path / 'out', verbose=False)
    assert str(e.value) == M1
    assert host == [] and not (tmp_path / "out").exists()


def test_complete_labels_run_as_before(cite, tmp_path, host):
    res = _run(cite, tmp_path, ["Matilda", "totalVI"])
    assert host == ["Matilda", "totalVI"]
    assert [r["status"] for r in res.records] == ["CHAIN_OK", "CHAIN_OK"]
    assert all(r["metrics"]["ARI"] > 0.9 for r in res.records)


#: (the label of one cell, the example the message gives of it): a text that
#: reads as missing is named, NaN and a blank are not
FORMS = [
    (np.nan, ""),
    ("", ""),                                               # an empty field of the file
    ("NA", ", for example 'NA'"),
    ("NA ", ", for example 'NA'"),
    ("None", ", for example 'None'"),
    ("nan", ", for example 'nan'"),
]


@pytest.mark.parametrize("gap,example,with_ids", [
    *[(*form, False) for form in FORMS], *[(*form, True) for form in FORMS],
    # blank text: a file of barcodes and labels keeps it (see the next test)
    (" ", "", True)])
def test_every_form_of_a_missing_label_stops_the_run(cite, tmp_path, host, gap, example,
                                                     with_ids):
    lab = LABELS.copy()
    lab[5] = gap
    if with_ids:                # the obs-style export: barcodes, then the labels
        pd.DataFrame({"celltype": lab}, index=[f"c{i}" for i in range(N)]).to_csv(
            cite / "MYCITE" / "cty.csv")
    else:
        _write(cite / "MYCITE" / "cty.csv", lab)
    assert _refused(cite, tmp_path, ["totalVI"]) == (
        f"cty.csv of MYCITE has 1 of 120 cells without a label{example}. Write the "
        "folder again with a label for every cell.")
    assert host == []


def test_a_blank_line_of_a_one_column_file_is_no_row(cite, tmp_path, host):
    """Unchanged: the reader drops the line, so the labels are one too few and
    fit no embedding. The run is not stopped for a missing label."""
    lab = LABELS.copy()
    lab[5] = " "
    _write(cite / "MYCITE" / "cty.csv", lab)
    assert len(eio.read_labels(cite / "MYCITE" / "cty.csv")) == N - 1
    assert W._label_gaps("MYCITE", cite) == {}
    res = _run(cite, tmp_path, ["totalVI"])
    assert [r["status"] for r in res.records] == ["RUN_OK_NO_LABEL_MATCH"]


def test_labels_that_are_not_missing_run(cite, tmp_path, host):
    lab = LABELS.copy()
    lab[:4] = ["na", "none", "Unknown", "-"]
    _write(cite / "MYCITE" / "cty.csv", lab)
    res = _run(cite, tmp_path, ["totalVI"])
    assert [r["status"] for r in res.records] == ["CHAIN_OK"]


def test_the_way_out_follows_the_methods(cite, tmp_path, host):
    """A run that scores prints one way out, the same whatever the methods:
    write the folder again. It names neither evaluate=False nor a method."""
    _write(cite / "MYCITE" / "cty.csv", _with_gaps())
    # Matilda reads cty.csv as an input; sciPENN and totalVI do not
    assert _refused(cite, tmp_path, ["Matilda", "sciPENN"]) == M1
    assert _refused(cite, tmp_path, ["totalVI", "sciPENN"]) == M1
    assert "evaluate=False" not in M1 and "Matilda" not in M1
    assert host == []
    plan, private = W._scan("MYCITE", "vertical", methods=["Matilda", "sciPENN"],
                            data_path=cite, verbose=False)
    assert dict(zip(plan["method"], private["_label_inputs"])) == {
        "Matilda": ("cty.csv",), "sciPENN": ()}
    # the answer is read from the inputs, not from needs_labels alone
    assert list(plan["needs_labels"]) == [True, False]


def test_two_methods_that_read_the_label_file(cite, tmp_path, host):
    d = cite / "MYCITE"
    _h5(d / "atac.h5", PEAKS, [f"c{i}" for i in range(N)])
    _write(d / "cty.csv", _with_gaps())
    methods = ["Matilda", "UnitedNet", "totalVI"]
    assert _refused(cite, tmp_path, methods) == M1
    assert _refused(cite, tmp_path, methods, evaluate=False) == (
        HEAD + ", and Matilda and 1 more method read the label files" + FIX)
    assert host == [] and not (tmp_path / "out").exists()


# ------------------------------------------------------------------ a dry run
def test_dry_run_prints_the_same_fact(cite, tmp_path, host, capsys):
    _write(cite / "MYCITE" / "cty.csv", _with_gaps())
    mtb.config.data_path = cite
    kw = dict(dry_run=True)
    plan = mtb.run("MYCITE", "vertical", methods=["Matilda", "sciPENN"], **kw)
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "[run] " + M1
    assert len(plan) == 2 and plan["runnable"].all() and host == []
    mtb.run("MYCITE", "vertical", methods=["totalVI"], **kw)
    assert capsys.readouterr().out.splitlines()[-1] == "[run] " + M1
    # evaluate=False: only a method that reads the file stops the run
    mtb.run("MYCITE", "vertical", methods=["Matilda", "sciPENN"], evaluate=False, **kw)
    assert capsys.readouterr().out.splitlines()[-1] == "[run] " + M1_MATILDA
    mtb.run("MYCITE", "vertical", methods=["totalVI"], evaluate=False, **kw)
    assert "without a label" not in capsys.readouterr().out
    # verbose=False prints nothing, and nothing is raised
    mtb.run("MYCITE", "vertical", methods=["Matilda"], verbose=False, **kw)
    assert capsys.readouterr().out == ""


def test_dry_run_of_complete_labels_prints_no_such_line(cite, capsys):
    mtb.run(os.path.join(cite, 'MYCITE'), 'vertical', methods=['Matilda'], dry_run=True)
    out = capsys.readouterr().out
    assert "without a label" not in out and out.startswith("[run] Dry run: 1 of 1 ")


def test_a_dry_run_off_linux_names_the_method_that_would_read_the_file(
        cite, monkeypatch, capsys):
    # no environment can be installed here; the rows would run on Linux
    from multibench.engine import runner as R
    monkeypatch.setattr(R, "linux_only_sentence",
                        lambda: "Methods run only on Linux, and this computer runs macOS.")
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    _write(cite / "MYCITE" / "cty.csv", _with_gaps())
    plan = mtb.run(os.path.join(cite, 'MYCITE'), 'vertical', methods=['Matilda', 'totalVI'], dry_run=True)
    assert not plan["runnable"].any()
    assert capsys.readouterr().out.splitlines()[-1] == "[run] " + M1
    # the method is named when the run does not score
    plan = mtb.run(os.path.join(cite, 'MYCITE'), 'vertical', methods=['Matilda', 'totalVI'], dry_run=True, evaluate=False)
    assert not plan["runnable"].any()
    assert capsys.readouterr().out.splitlines()[-1] == "[run] " + M1_MATILDA


# ------------------------------------------------------------------ evaluate=False
def test_evaluate_false_runs_methods_that_do_not_read_labels(cite, tmp_path, host):
    _write(cite / "MYCITE" / "cty.csv", _with_gaps())
    res = _run(cite, tmp_path, ["totalVI", "sciPENN"], evaluate=False)
    assert sorted(host) == ["sciPENN", "totalVI"]
    assert {r["method"]: r["status"] for r in res.records} == {"totalVI": "RUN_OK",
                                                               "sciPENN": "RUN_OK"}


def test_evaluate_false_stops_methods_that_read_labels(cite, tmp_path, host):
    _write(cite / "MYCITE" / "cty.csv", _with_gaps())
    assert _refused(cite, tmp_path, ["Matilda", "sciPENN"], evaluate=False) == M1_MATILDA
    assert host == [] and not (tmp_path / "out").exists()


def test_a_method_that_is_fed_the_folder_reads_its_label_files(tmp_path, host):
    # scBridge is given the folder and names rna_cty.csv and atac_cty.csv itself
    d = tmp_path / "data" / "UNPAIRED"
    d.mkdir(parents=True)
    genes = [f"GENE{i}" for i in range(40)]
    _h5(d / "rna.h5", genes, [f"r{i}" for i in range(60)])
    _h5(d / "atac_gas.h5", genes, [f"a{i}" for i in range(60)])
    _write(d / "rna_cty.csv", _with_gaps(60, na=3, empty=0))
    _write(d / "atac_cty.csv", ["A", "B"] * 30)
    mtb.config.data_path = tmp_path / "data"
    kw = dict(methods=['scBridge'], verbose=False)
    plan, private = _quiet(W._scan, "UNPAIRED", "diagonal", **kw)
    assert list(plan["modalities"]) == ["(data_dir)"] and plan["runnable"].all()
    assert list(private["_label_inputs"]) == [("rna_cty.csv", "atac_cty.csv")]
    with pytest.raises(ValueError) as e:
        _quiet(mtb.run, "UNPAIRED", "diagonal", tmp_path / "out", evaluate=False, **kw)
    assert str(e.value) == (
        "rna_cty.csv of UNPAIRED has 3 of 60 cells without a label, for example 'NA', "
        "and scBridge reads the label files. Write the folder again with a label for "
        "every cell.")
    assert host == []


def test_only_the_files_a_method_reads_stop_a_run_without_scores(tmp_path, host):
    # three donors; UINMF's selected variant reads no label file, scMoMaT reads all three
    d = tmp_path / "data" / "DONORS"
    d.mkdir(parents=True)
    for b in (1, 2, 3):
        bars = [f"d{b}c{i}" for i in range(40)]
        _h5(d / f"rna{b}.h5", [f"g{i}" for i in range(30)], bars)
        _h5(d / f"adt{b}.h5", [f"p{i}" for i in range(6)], bars)
        _write(d / f"cty{b}.csv", ["A", "B"] * 20)
    _write(d / "cty2.csv", _with_gaps(40, na=1200 // 400, empty=0))
    mtb.config.data_path = tmp_path / "data"
    kw = dict(verbose=False)
    res = _quiet(mtb.run, "DONORS", "cross", tmp_path / "out", methods=["totalVI"],
                 evaluate=False, **kw)
    assert [r["status"] for r in res.records] == ["RUN_OK"]
    with pytest.raises(ValueError) as e:
        _quiet(mtb.run, "DONORS", "cross", tmp_path / "out2", methods=["totalVI"], **kw)
    assert str(e.value) == (
        "cty2.csv of DONORS has 3 of 40 cells without a label, for example 'NA'. Write "
        "the folder again with a label for every cell.")
    assert not (tmp_path / "out2").exists()


# ------------------------------------------------------------------ rescore
M2 = ("ValueError: 9 of the 120 labels in {files} are missing, for example 'NA'. Give "
      "every cell a label, then score again.")


def test_rescore_records_the_error(cite, tmp_path, host, monkeypatch):
    res = _run(cite, tmp_path, ["Matilda", "totalVI"])
    assert [r["status"] for r in res.records] == ["CHAIN_OK", "CHAIN_OK"]
    _write(cite / "MYCITE" / "cty.csv", _with_gaps())
    # the ranking would take the error of an order and go on: it is not reached
    monkeypatch.setattr(W, "_evaluate_best_order", lambda *a, **k: 1 / 0)
    again = res.rescore()
    for rec in again.records:
        assert rec["status"] == "RUN_OK_EVAL_FAILED"
        assert rec["error"] == M2.format(files="cty.csv")
        assert not rec.get("metrics") and rec.get("_long") is None
        assert "labels_used" not in rec
    assert list(again.summary["status"]) == ["ran"] * 2
    assert "ARI" not in again.summary.columns
    # the first result is untouched, and complete labels score again
    assert [r["status"] for r in res.records] == ["CHAIN_OK", "CHAIN_OK"]
    monkeypatch.undo()
    _write(cite / "MYCITE" / "cty.csv", LABELS)
    assert [r["status"] for r in again.rescore().records] == ["CHAIN_OK", "CHAIN_OK"]


def test_several_label_files_are_not_ranked(tmp_path, host, monkeypatch):
    d = tmp_path / "data" / "DONORS"
    d.mkdir(parents=True)
    for b in (1, 2, 3):
        bars = [f"d{b}c{i}" for i in range(40)]
        _h5(d / f"rna{b}.h5", [f"g{i}" for i in range(30)], bars)
        _h5(d / f"adt{b}.h5", [f"p{i}" for i in range(6)], bars)
        # r3 (Y12): files that differ, so that there are several orders to rank
        _write(d / f"cty{b}.csv", (["A"] * (20 + b) + ["B"] * (20 - b)))
    mtb.config.data_path = tmp_path / "data"
    kw = dict(verbose=False)
    res = _quiet(mtb.run, "DONORS", "cross", tmp_path / "out", methods=["totalVI"], **kw)
    (rec,) = res.records
    assert rec["status"] == "CHAIN_OK" and len(rec["label_order_candidates"]) > 1
    lab = np.array(["A"] * 20 + ["B"] * 20, dtype=object)
    lab[[3, 30]] = "None"
    _write(d / "cty2.csv", lab)
    ranked = []
    real = W._evaluate_best_order
    monkeypatch.setattr(W, "_evaluate_best_order",
                        lambda *a, **k: ranked.append(1) or real(*a, **k))
    (rec,) = res.rescore().records
    assert rec["status"] == "RUN_OK_EVAL_FAILED" and ranked == []
    # c2 (D28): the files in the order the record was scored with
    a, b, c = res.records[0]["labels_used"]
    assert rec["error"] == (
        f"ValueError: 2 of the 120 labels in {a}, {b} and {c} are missing, "
        "for example 'None'. Give every cell a label, then score again.")
    assert "label_order_candidates" not in rec and "metrics" not in rec
    # a real run on this folder stops before the method
    host.clear()
    with pytest.raises(ValueError) as e:
        _quiet(mtb.run, "DONORS", "cross", tmp_path / "out2", methods=["totalVI"], **kw)
    assert str(e.value) == ("cty2.csv of DONORS has 2 of 40 cells without a label, for "
                            "example 'None'. Write the folder again with a label for "
                            "every cell.")
    assert host == []


def test_labels_passed_to_rescore(cite, tmp_path, host):
    res = _run(cite, tmp_path, ["totalVI"])
    gaps = _with_gaps()
    passed = M2.format(files="the labels you passed")
    # in the rows of the embedding, and by barcode
    for labels in (gaps, list(gaps), pd.Series(gaps, index=[f"c{i}" for i in range(N)])):
        (rec,) = res.rescore(labels=labels).records
        assert (rec["status"], rec["error"]) == ("RUN_OK_EVAL_FAILED", passed)
        assert "metrics" not in rec
    # a file: the same record
    sheet = tmp_path / "labels.csv"
    pd.DataFrame({"celltype": gaps}, index=[f"c{i}" for i in range(N)]).to_csv(sheet)
    (rec,) = res.rescore(labels=sheet).records
    assert (rec["status"], rec["error"]) == ("RUN_OK_EVAL_FAILED", passed)
    # complete labels from a file score as the same labels in memory
    pd.DataFrame({"celltype": LABELS}, index=[f"c{i}" for i in range(N)]).to_csv(sheet)
    (from_file,) = res.rescore(labels=sheet).records
    (in_memory,) = res.rescore(labels=pd.Series(LABELS, index=[f"c{i}" for i in range(N)])
                               ).records
    assert from_file["status"] == "CHAIN_OK" and from_file["metrics"] == in_memory["metrics"]
    (rec,) = res.rescore(labels=LABELS[::-1].copy()).records
    assert rec["status"] == "CHAIN_OK"


def test_the_error_of_a_record_counts_with_separators(tmp_path, host):
    cands = [(["cty.csv"], np.array(["A"] * 1200 + [None] * 1300, dtype=object), None)]
    assert W._unlabelled_error(cands, "X", tmp_path, user=True) == (
        "ValueError: 1,300 of the 2,500 labels in the labels you passed are missing. "
        "Give every cell a label, then score again.")
    assert W._unlabelled_error([(["cty.csv"], np.array(["A", "B"]), None)], "X", tmp_path,
                               user=True) is None


# ------------------------------------------------------------------ one list of files
def test_label_files_are_one_list(cite, tmp_path, host, monkeypatch):
    d = cite / "MYCITE"
    _write(d / "rna_cty_scjoint.csv", LABELS)          # a tool's reformat: not a label file
    (d / "notes.csv").write_text("x\n1\n")
    assert [p.name for p in W._label_files("MYCITE", cite)] == ["cty.csv"]
    _write(d / "cty2.csv", LABELS[:10])
    assert [p.name for p in W._label_files("MYCITE", cite)] == ["cty.csv", "cty2.csv"]
    # the default data root, and a folder that is not there
    assert [p.name for p in W._label_files("D11")] == ["cty.csv"]
    assert W._label_files("NO_SUCH", cite) == []
    # the scoring and the check before the run read this one list
    assert [names for names, _, _ in W._label_candidates("MYCITE", N, cite)] == [["cty.csv"]]
    assert set(W._label_gaps("MYCITE", cite)) == set()
    _write(d / "cty2.csv", _with_gaps(10, na=1, empty=0))
    assert {k: v[:2] for k, v in W._label_gaps("MYCITE", cite).items()} == {
        "cty2.csv": (1, 10)}
    monkeypatch.setattr(W, "_label_files", lambda dataset, data_path=None: [d / "cty2.csv"])
    assert W._label_candidates("MYCITE", N, cite) == []
    assert [names for names, _, _ in W._label_candidates("MYCITE", 10, cite)] == [
        ["cty2.csv"]]
    _, private = W._scan("MYCITE", "vertical", methods=["Matilda"], data_path=cite,
                         verbose=False)
    assert list(private["_label_inputs"]) == [()]      # cty.csv is not in the list now
