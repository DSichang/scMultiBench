"""Review of fix round 6: the findings applied in the integration worktree.

- The saved batch is in the dataset's cell order. rescore no longer uses it
  as given when the dataset folder is not found: it drops it, with a note
  and a warning. Against labels in embedding row order it follows the
  stored ``labels_used`` order, not ``labels_for``.
- ``load_batch(data_path=)`` that does not hold the dataset raises; the
  moved-folder messages name the roots tried; the root found is recorded as
  an absolute path, and so is a re-pointed ``out_dir``.
- No unsaved-batch warning when ``metrics=`` has no batch metric.
- A headerless first column that repeats numbers is read as an index only
  in the ``pd.concat([...]).to_csv()`` pattern.
- Messages as sentences: the run_all "more rows" line, ``files_reason``,
  the modality note, the no-variant error, the missing dataset folder, the
  label-count errors, the scBridge data_dir error, the several-columns
  error, the plot unknown-name errors, the overlay error for several
  methods, the scan count line for one method, the run-all dry-run header,
  and the ``scan --strict`` scripts-ref fix without ``--methods``.
"""
import inspect
import json
import re
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench
import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs
from multibench.plot import build_table
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_study_r4_wf import ALL_ENVS, _h5, _quiet, _Res, cite  # noqa: F401
from tests.test_study_r6_msg import _lung_ids, needs_git, pinned, wrong_ref
from tests.test_study_r6_wf import KW, _shuffled_labels, rel  # noqa: F401

#: 0.4: data_path= of load_batch is deprecated, so the fix names the config
MOVED = ("Set mtb.config.DEFAULT.data_path to the folder that holds it, or run rescore "
         "from the folder where run_all ran.")
from tests.conftest import full_scan

pytest.importorskip("scib")

SIZES = {1: 40, 2: 50, 3: 60}
#: the order the fake method stacks the batches in: neither the dataset's
#: (1, 2, 3) nor labels_for's for StabMap (3, 1, 2)
EMB_ORDER = (2, 3, 1)


def _donor(b, n):
    """Two donors inside each file, so the batch is not the file of origin."""
    return {1: np.where(np.arange(n) < 30, "dA", "dB"), 2: np.full(n, "dB"),
            3: np.where(np.arange(n) % 2 == 0, "dA", "dB")}[b]


@pytest.fixture
def stacked(tmp_path, monkeypatch):
    """A cross folder of three batches, a StabMap-like fake run that stacks
    them in EMB_ORDER with a donor effect, and run_all(batch=<donor Series>)
    from ``tmp_path`` with the relative data_path 'data'."""
    d = tmp_path / "data" / "MYCROSS"
    d.mkdir(parents=True)
    rng = np.random.default_rng(11)
    labs, bars = {}, []
    for b, n in SIZES.items():
        ids = [f"b{b}_{i}" for i in range(n)]
        bars += ids
        _h5(d / f"rna{b}.h5", [f"g{i}" for i in range(30)], ids)
        _h5(d / f"adt{b}.h5", [f"p{i}" for i in range(6)], ids)
        labs[b] = rng.choice(["T", "B", "NK", "Mono"], n)
        pd.DataFrame({"x": labs[b]}).to_csv(d / f"cty{b}.csv", index=False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)

    def fake_run(method, category, inputs, out_dir, params=None):
        lab = np.concatenate([labs[b] for b in EMB_ORDER])
        don = np.concatenate([(_donor(b, SIZES[b]) == "dB").astype(int)
                              for b in EMB_ORDER])
        emb = np.random.default_rng(7).normal(size=(len(lab), 6))
        for j, t in enumerate(["B", "NK", "Mono"]):
            emb[:, j] += 7.0 * (lab == t)
        emb[:, 4] += 4.0 * don
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    monkeypatch.chdir(tmp_path)
    donor = np.concatenate([_donor(b, n) for b, n in SIZES.items()])
    batch = pd.Series(donor, index=bars).sample(frac=1.0, random_state=2)
    res = _quiet(mtb.run_all, "MYCROSS", "cross", "out", data_path="data",
                 methods=["StabMap"], batch=batch, verbose=False)
    emb_labels = np.concatenate([labs[b] for b in EMB_ORDER])
    return res, emb_labels, tmp_path


def _batch_cols(res):
    row = res.summary.iloc[0]
    rec = res.records[0]
    return {**{k: (None if k not in row or pd.isna(row[k]) else row[k])
               for k in ("ASW_batch", "iLISI")},
            **{k: rec.get(k) for k in ("batch_source", "n_batches")}}


def _rescore(res, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = res.rescore(**kw)
    return out, [str(w.message) for w in rec]


# ============================================================ saved batch
def test_positional_labels_place_the_saved_batch_in_the_stored_order(stacked):
    res, emb_labels, _ = stacked
    rec = res.records[0]
    assert rec["labels_used"] == [f"cty{b}.csv" for b in EMB_ORDER]
    assert rec["batch_source"] == "user" and rec["n_batches"] == 2
    new, caught = _rescore(mtb.load_batch("out"), labels=emb_labels)
    assert "+".join(new.records[0]["labels_used"]) == "(user labels)"
    assert _batch_cols(new) == _batch_cols(res), caught


def test_a_saved_batch_is_dropped_when_the_dataset_folder_is_not_found(stacked):
    res, emb_labels, root = stacked
    shutil.move(str(root / "data"), str(root / "gone"))
    back = mtb.load_batch("out")
    new, caught = _rescore(back, labels=emb_labels)
    rec = new.records[0]
    assert rec["status"] == "CHAIN_OK" and "ARI" in rec["metrics"]
    for k in ("ASW_batch", "iLISI", "GC"):
        assert k not in rec["metrics"], k
    assert rec.get("batch_source") is None and rec["n_batches"] == 1
    note = (f"The saved batch is not used, because the dataset folder MYCROSS is not "
            f"found in {root / 'data'}. {MOVED}")
    assert rec["note"] == note
    assert caught.count(note) == 1, caught
    # the file still travels with the result, so the right directory recovers it
    new.save(root / "again")
    shutil.move(str(root / "gone"), str(root / "data"))
    back = mtb.load_batch(root / "again")
    assert back.records[0]["batch_file"] == res.records[0]["batch_file"]
    assert _batch_cols(_quiet(back.rescore)) == _batch_cols(res)


# ========================================================== data folders
def test_a_data_path_that_does_not_hold_the_dataset_raises(rel):
    _, root = rel
    with pytest.raises(ValueError) as e:
        mtb.load_batch("out", data_path=root / "data" / "MYCITE")
    assert str(e.value) == (
        f"{root / 'data' / 'MYCITE' / 'MYCITE'} is not a folder. data_path= is the "
        f"folder that holds MYCITE, here {root / 'data'}.")
    with pytest.raises(ValueError, match=r"^\S+nowhere/MYCITE is not a folder\. "
                                         r"data_path= is the folder that holds MYCITE\.$"):
        mtb.load_batch("out", data_path=root / "nowhere")
    # 0.4: data_path= is deprecated, so the docstring no longer lists this error


def test_the_folder_found_is_recorded_as_an_absolute_path(rel, monkeypatch):
    res, root = rel
    shutil.move(str(root / "data"), str(root / "gone"))
    shutil.move(str(root / "out"), str(root / "moved_out"))
    monkeypatch.chdir(root / "nb")
    back = mtb.load_batch("../moved_out", data_path="../gone")
    rec = back.records[0]
    gone = str((root / "gone").resolve())
    assert rec["data_path"] == rec["data_root"] == gone
    assert rec["out_dir"] == str((root / "moved_out" / "Matilda_MYCITE").resolve())
    _quiet(back.rescore, metrics=["ARI"]).save("../ct2")
    saved = json.loads((root / "ct2" / "batch_result.json").read_text())["records"][0]
    assert saved["data_path"] == saved["data_root"] == gone
    assert Path(saved["out_dir"]).is_absolute()
    # read from yet another directory, the saved result still finds its data
    monkeypatch.chdir(root)
    again = _quiet(mtb.load_batch("ct2").rescore, labels=_shuffled_labels())
    assert "+".join(again.records[0]["labels_used"]) == "cty.csv"


def test_no_unsaved_batch_warning_without_a_batch_metric(cite, tmp_path):
    data, batch, shuffled = cite
    _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "out", data_path=data,
           batch=shuffled, **KW)
    p = tmp_path / "out" / "batch_result.json"
    blob = json.loads(p.read_text())
    for r in blob["records"]:                      # a folder from before 0.3.2
        (tmp_path / "out" / r.pop("batch_file")).unlink()
    p.write_text(json.dumps(blob))
    old = mtb.load_batch(tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # the toy cells give no resolution with one cluster per cell type
        warnings.filterwarnings("ignore", message="No Leiden resolution gave")
        warnings.filterwarnings("ignore", message="ARI_batch and NMI_batch use")
        new = old.rescore(metrics=["ARI"])
    assert new.records[0]["status"] == "CHAIN_OK"
    with pytest.warns(UserWarning, match="which older versions did not save"):
        old.rescore(metrics=["ARI", "iLISI"])


# ================================================= stacked row numbers
def test_row_numbers_out_of_order_are_not_an_index(tmp_path):
    from multibench.eval import io as eio
    a = pd.DataFrame({"celltype": ["T", "B", "NK", "T"]})
    b = pd.DataFrame({"celltype": ["B", "T", "NK"]})
    pd.concat([a, b]).to_csv(tmp_path / "cat.csv")
    assert list(eio.read_labels(tmp_path / "cat.csv")) == ["T", "B", "NK", "T", "B", "T",
                                                           "NK"]
    pd.concat([a, b]).sort_values("celltype").to_csv(tmp_path / "sorted.csv")
    with pytest.raises(ValueError) as e:
        mtb.evaluate(np.random.default_rng(0).normal(size=(7, 3)),
                     labels=str(tmp_path / "sorted.csv"))
    assert str(e.value) == ("The labels file sorted.csv has row numbers that repeat out "
                            "of order, so its rows may not follow the cells. Save it in "
                            "cell order.")


def test_files_reason_parts_are_joined_as_sentences(tmp_path, pinned, monkeypatch):
    d = tmp_path / "data" / "CITE"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], [f"c{i}" for i in range(20)])
    script = "MIRA's script imports logger.py. mtb.method_info('MIRA')['setup_hint'] shows how."
    monkeypatch.setattr(W, "_missing_script", lambda v, method=None: script)
    row = _quiet(full_scan, "CITE", "vertical", methods=["MIRA"], data_path=tmp_path / "data",
                 verbose=False).iloc[0]
    assert row["files_reason"].startswith(f"{script} FileNotFoundError: MIRA (vertical) "
                                          f"needs atac.h5"), row["files_reason"]
    assert "; " not in row["files_reason"] and "; " not in row["reason"]


def test_scan_count_line_for_one_method_without_files(tmp_path, pinned, capsys):
    d = tmp_path / "data" / "CITE"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], [f"c{i}" for i in range(20)])
    full_scan("CITE", "vertical", methods=["MIRA"], data_path=tmp_path / "data")
    assert capsys.readouterr().out.startswith(
        "[scan] 0 of 1 method has its input files. 1 of 1 has its environment installed.")
    assert W._have_their(0, 14) == "have their" and W._have_their(1, 14) == "has its"


def test_the_no_variant_and_missing_folder_errors_are_sentences(tmp_path):
    with pytest.raises(ValueError) as e:
        # (Matilda has a variant for the three modalities; VIMCCA has none)
        _quiet(full_scan, "D11", "vertical", methods=["VIMCCA"],
               modalities=["rna", "atac", "adt"], verbose=False)
    assert str(e.value) == ("VIMCCA does not read rna+atac+adt in vertical data. "
                            "mtb.method_info('VIMCCA')['supports'] lists what VIMCCA "
                            "reads.")
    assert ["rna", "adt", "atac"] in [s["modalities"]
                                      for s in mtb.method_info("Matilda")["supports"]]
    (tmp_path / "D11").mkdir()
    (tmp_path / "D28").mkdir()
    with pytest.raises(FileNotFoundError) as e:
        full_scan("MYCITE", data_path=tmp_path)
    assert str(e.value) == (f"The folder {tmp_path / 'MYCITE'} does not exist. {tmp_path} "
                            f"holds D11 and D28. dataset= is the folder's path, or its "
                            f"name under mtb.config.DEFAULT.data_path. "
                            f"mtb.describe_layout() shows the layout.")


def test_the_label_count_error_is_sentences(tmp_path):
    d = tmp_path / "LAB"
    d.mkdir()
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], [f"c{i}" for i in range(20)])
    _h5(d / "adt.h5", [f"p{i}" for i in range(6)], [f"c{i}" for i in range(20)])
    pd.DataFrame({"x": ["A"] * 15}).to_csv(d / "cty.csv", index=False)
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("LAB", "vertical", "Matilda", modalities=["rna", "adt"],
                       data_path=tmp_path, check=True)
    assert str(e.value) == ("cty.csv has 15 labels, but rna.h5 has 20 cells. Give each "
                            "cell one label, in the order of the cells. "
                            "mtb.describe_layout('vertical') shows the files.")


def test_the_scbridge_error_names_the_method_once(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        mtb.inputs_for("LUNG_ids", "diagonal", "scBridge", data_path=_lung_ids(tmp_path),
                       check=True)
    assert str(e.value) == (f"scBridge needs atac_gas.h5, and {tmp_path / 'LUNG_ids'} has "
                            f"no such file.")
    assert not str(e.value).startswith("scBridge/")


# =========================================================== docstrings
def test_docstrings_quote_the_current_texts():
    flat = {k: " ".join(inspect.getdoc(o).split()) for k, o in {
        "run_all": mtb.run_all, "save": W.BatchResult.save, "run": mtb.run,
        "rescore": W.BatchResult.rescore, "load_batch": mtb.load_batch}.items()}
    assert "nothing is runnable" not in flat["run_all"]
    # 0.4: the Notes of run_all, run and rescore that quoted these texts are cut
    assert "no 'cross' variant matches" not in flat["run_all"]
    assert "and the ``batch`` vector" not in flat["run_all"]
    assert "all four files" not in flat["save"]
    assert "every file in the list above is rewritten from the merged set" in flat["save"]
    assert "see its Notes" not in flat["rescore"]
    assert ("each record's output folder is looked for next to ``batch_result.json`` when "
            "its recorded path is gone") in flat["load_batch"]


# ============================================================ docs pages


@needs_docs
def test_the_evaluate_page_names_no_vague_definitions():
    text = _flat("tutorials/evaluate.md")
    assert "with the same metric definitions" not in text
