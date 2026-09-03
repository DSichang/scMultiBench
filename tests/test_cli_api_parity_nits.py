"""CLI/API parity nits from the re-test, each with its evidence.

* ``multibench scan D52`` without ``--category`` scans all four, like Python;
* ``inputs_for('D52', 'cross', 'StabMap')`` says the 2nd/3rd arguments are swapped;
* a lower-case dataset id is canonicalised to the on-disk folder name;
* ``run_all(dry_run=True)`` returns the ``command`` column the CLI csv carries;
* ``multibench cite A,B`` works like ``multibench cite A B``.
"""
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli
from multibench import workflow as W
from multibench.engine import envs, registry, resolve

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


# ----------------------------------------------------------------- swapped arguments
def test_inputs_for_category_in_method_slot_names_the_swap():
    with pytest.raises(KeyError) as e:
        mtb.inputs_for("D52", "cross", "StabMap")
    msg = str(e.value)
    assert "unknown method 'cross'" in msg and "category token" in msg
    assert "argument order is (dataset, method, category)" in msg
    assert "unlike scan/plan/run_all(dataset, category, ...)" in msg
    assert "inputs_for('D52', 'StabMap', 'cross')" in msg


def test_inputs_for_method_in_category_slot_names_the_swap():
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("D52", "StabMap", "StabMap")
    msg = str(e.value)
    assert "unknown category 'StabMap'" in msg and "method id" in msg
    assert "inputs_for('D52', 'StabMap', 'StabMap')" in msg
    assert resolve.SWAPPED_ARGS_HINT in msg


def test_inputs_for_plain_typos_keep_their_messages():
    with pytest.raises(KeyError, match="did you mean 'StabMap'"):
        mtb.inputs_for("D52", "Stabmap", "cross")
    with pytest.raises(ValueError, match="unknown category 'cros'; valid"):
        mtb.inputs_for("D52", "StabMap", "cros")
    # a category token in the method slot with an unknown category is NOT a swap
    with pytest.raises(KeyError) as e:
        mtb.inputs_for("D52", "cross", "nonsense")
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
        df = mtb.scan("mycite", "vertical", data_path=tmp_path)
    assert (df["files_reason"].str.contains("/mycite")).sum() == 0
    assert len(df) == len(mtb.scan("MYCITE", "vertical", data_path=tmp_path))
    with pytest.warns(UserWarning, match="on-disk spelling"):
        inp = mtb.inputs_for("mycite", "Matilda", "vertical", modalities=["rna", "adt"],
                             data_path=tmp_path)
    assert all("/MYCITE/" in p for p in inp.values())
    with pytest.warns(UserWarning, match="on-disk spelling"):
        labs = mtb.labels_for("mycite", data_path=tmp_path)
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
        res = mtb.run_all("mycite", "vertical", methods=["Matilda"], data_path=tmp_path,
                          out_dir=str(tmp_path / "runs"), evaluate=False, verbose=False)
    assert sum("on-disk spelling" in str(x.message) for x in w) == 1     # once, not per call
    assert res.dataset == "MYCITE"
    assert calls == [str(tmp_path / "runs" / "Matilda_MYCITE")]
    assert all(r["dataset"] == "MYCITE" for r in res.records)


# ----------------------------------------------------------------- dry-run command column
def test_run_all_dry_run_returns_the_cli_csv_command_column(capsys):
    plan = mtb.run_all("D11", "vertical", out_dir="runs", methods=["Matilda"],
                       dry_run=True, verbose=False)
    assert list(plan.columns)[-1] == "command"
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--methods", "Matilda",
                   "--out", "runs", "--dry-run", "--format", "csv"])
    assert rc == 0
    csv = pd.read_csv(io.StringIO(capsys.readouterr().out)).fillna("")
    assert list(csv["command"]) == list(plan["command"])
    ok = plan[plan["files_ok"]]
    assert len(ok) and ok["command"].str.contains("--save_path").all()
    assert ok["command"].str.contains("runs/Matilda_D11").all()


def test_plan_without_out_dir_uses_the_placeholder():
    plan = mtb.plan("D11", "vertical", methods=["Matilda"])
    ok = plan[plan["files_ok"]]
    assert ok["command"].str.contains(W.OUT_DIR_PLACEHOLDER + "/Matilda_D11").all()
    assert (plan[~plan["files_ok"]]["command"] == "").all()


# ----------------------------------------------------------------- cite delimiters
def test_cli_cite_accepts_commas_and_spaces(capsys):
    assert cli.main(["cite", "Matilda,MOFA2", "--format", "text"]) == 0
    commas = capsys.readouterr().out
    assert cli.main(["cite", "Matilda", "MOFA2", "--format", "text"]) == 0
    spaces = capsys.readouterr().out
    assert commas == spaces == mtb.cite("Matilda", "MOFA2", fmt="text") + "\n"
    assert cli.main(["cite", "Matilda,", "MOFA2", "--format", "text"]) == 0   # stray comma
    assert capsys.readouterr().out == spaces
    assert cli.main(["cite", "Matilda,Matlida"]) == 1
    assert "did you mean 'Matilda'" in capsys.readouterr().err


def test_cli_scan_unknown_method_message_without_category(capsys):
    rc = cli.main(["scan", "D11", "--methods", "Matilda"])   # no --category: fine
    assert rc == 0 and "Matilda" in capsys.readouterr().out
