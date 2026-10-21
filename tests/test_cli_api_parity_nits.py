"""CLI/API parity nits from the re-test, each with its evidence.

* ``multibench scan D52`` without ``--category`` scans all four, like Python;
* ``inputs_for('D52', 'StabMap', 'cross')`` (the 0.2 order) is refused, never accepted;
* a lower-case dataset id is canonicalised to the on-disk folder name;
* ``run(dry_run=True)`` returns the ``command`` column the CLI csv carries;
* ``multibench cite A,B`` works like ``multibench cite A B``.
"""
import os
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry, resolve
from tests.conftest import full_scan

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


def test_inputs_for_plain_typos_keep_their_messages():
    with pytest.raises(KeyError, match="Did you mean StabMap"):
        mtb.engine.resolve.inputs_for("D52", "cross", "StabMapp")
    with pytest.raises(ValueError, match="unknown category 'cros'; valid"):
        mtb.engine.resolve.inputs_for("D52", "cros", "StabMap")
    # a method id in BOTH slots is a bad category, not a swap
    with pytest.raises(ValueError, match="unknown category 'StabMap'"):
        mtb.engine.resolve.inputs_for("D52", "StabMap", "StabMap")
    # a category token in the method slot with an unknown method is a bad id, not a swap
    with pytest.raises(KeyError) as e:
        mtb.engine.resolve.inputs_for("D52", "cross", "nonsense")
    assert "argument order" not in str(e.value)


# ----------------------------------------------------------------- dataset spelling
def _cite(root, name):
    import h5py
    d = root / name
    d.mkdir()
    for role, nf in (("rna", 30), ("adt", 10)):
        with h5py.File(d / f"{role}.h5", "w") as f:
            g = f.create_group("matrix")
            g.create_dataset("data", data=np.zeros((nf, 40)))
            g.create_dataset("features", data=np.array([f"g{i}" for i in range(nf)], dtype="S8"))
            g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(40)], dtype="S8"))
    pd.DataFrame({"x": ["A", "B"] * 20}).to_csv(d / "cty.csv", index=False)
    return d


def test_canonical_dataset_returns_the_listed_spelling(tmp_path):
    _cite(tmp_path, "MYCITE")
    assert resolve.canonical_dataset(tmp_path, "MYCITE") == "MYCITE"
    with pytest.warns(UserWarning, match="'mycite' is not a folder under .* but 'MYCITE' is"):
        assert resolve.canonical_dataset(tmp_path, "mycite") == "MYCITE"
    assert resolve.canonical_dataset(tmp_path, "nothere") == "nothere"       # caller's error fires
    assert resolve.canonical_dataset(tmp_path / "missing", "x") == "x"


def test_scan_and_inputs_for_carry_the_on_disk_name(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    _cite(tmp_path, "MYCITE")
    with pytest.warns(UserWarning, match="on-disk spelling"):
        df = full_scan("mycite", "vertical", data_path=tmp_path)
    assert (df["files_reason"].str.contains("/mycite")).sum() == 0
    assert len(df) == len(full_scan("MYCITE", "vertical", data_path=tmp_path))
    with pytest.warns(UserWarning, match="on-disk spelling"):
        inp = mtb.engine.resolve.inputs_for("mycite", "vertical", "Matilda", modalities=["rna", "adt"],
                             data_path=tmp_path)
    assert all("/MYCITE/" in p for p in inp.values())
    with pytest.warns(UserWarning, match="on-disk spelling"):
        labs = mtb.engine.resolve.labels_for("mycite", data_path=tmp_path)
    assert "/MYCITE/" in labs["cty"]


def test_run_all_names_out_dir_and_records_after_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(out_dir)

        class R:
            output = np.zeros((40, 5))
        return R()
    monkeypatch.setattr(W, "_run", fake_run)
    _cite(tmp_path, "MYCITE")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = mtb.run(os.path.join(tmp_path, 'mycite'), 'vertical', methods=['Matilda'], out_dir=str(tmp_path / 'runs'), evaluate=False, verbose=False)
    assert sum("on-disk spelling" in str(x.message) for x in w) == 1     # once, not per call
    assert res.dataset == "MYCITE"
    assert calls == [str(tmp_path / "runs" / "Matilda_MYCITE")]
    assert all(r["dataset"] == "MYCITE" for r in res.records)


def test_scan_without_out_dir_uses_the_placeholder():
    plan = full_scan("D11", "vertical", methods=["Matilda"], verbose=False)
    ok = plan[plan["files_ok"]]
    assert ok["command"].str.contains(W.OUT_DIR_PLACEHOLDER + "/Matilda_D11").all()
    assert (plan[~plan["files_ok"]]["command"] == "").all()


