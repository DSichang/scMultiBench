"""scan() is a TWO-gate preflight: files_ok / env_ok, and the file gate ALWAYS runs.

Before this, the env gate ran first and `continue`d, so on a machine without the
method conda envs (every laptop) the file checks never executed: a dataset with
only rna.h5 + cty.csv, an empty folder and a typo'd dataset name all produced the
same 58 rows of "conda env ... not installed". These tests pin the new contract
with the env probe monkeypatched both ways, so they hold on ANY machine.
"""
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
OLD_COLUMNS = ["method", "category", "modalities", "env", "output_kind", "n_tunable",
               "runtime_tier", "observed_worst_sec", "caveat", "runnable", "reason"]
NEW_COLUMNS = ["files_ok", "files_reason", "env_ok", "env_reason", "needs_labels", "atac"]


def _h5(path, n_feat, n_cells, prefix="g"):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(n_feat, n_cells)).astype(float))
        g.create_dataset("features", data=np.array([f"{prefix}{i}" for i in range(n_feat)], dtype="S12"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(n_cells)], dtype="S12"))


def _rna_only(root, name="ONLYRNA", n=60):
    d = root / name
    d.mkdir(parents=True)
    _h5(d / "rna.h5", 30, n)
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / "cty.csv", index=False)
    return d


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


@pytest.fixture
def all_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)


# --- the file gate runs even when no env is installed ---------------------------

def test_scan_file_gate_runs_without_envs(tmp_path, no_envs):
    _rna_only(tmp_path)
    df = mtb.scan("ONLYRNA", "vertical", data_path=tmp_path)
    rows = df[df["modalities"] == "rna+adt"]
    assert len(rows) > 0
    assert not rows["files_ok"].any()
    assert rows["files_reason"].str.contains("adt.h5").all()
    assert not rows["env_ok"].any()
    assert not rows["runnable"].any()
    # reason joins BOTH parts with '; ' - the file problem is visible even though
    # the env is also missing. The file half is the SHORT form (no exception
    # class, no absolute paths); the env half is verbatim (it carries the
    # install command).
    for _, r in rows.iterrows():
        assert r["reason"] == f"{W._short_reason(r['files_reason'], r['method'], 'ONLYRNA', 'vertical')}; {r['env_reason']}"
        assert r["reason"].endswith("; " + r["env_reason"])
        assert "adt.h5" in r["reason"] and "not installed" in r["reason"]
        assert "FileNotFoundError" in r["files_reason"] and str(tmp_path) in r["files_reason"]
        assert "FileNotFoundError" not in r["reason"] and str(tmp_path) not in r["reason"]


def test_scan_missing_dataset_folder_raises(tmp_path):
    (tmp_path / "HERE").mkdir()
    with pytest.raises(FileNotFoundError) as e:
        mtb.scan("D99", data_path=tmp_path)
    msg = str(e.value)
    assert "D99" in msg and "does not exist" in msg
    assert "HERE" in msg, "must list the folders that DO exist"


def test_scan_empty_folder_flags_data_dir_methods(tmp_path, all_envs):
    (tmp_path / "EMPTY").mkdir()
    df = mtb.scan("EMPTY", "cross", data_path=tmp_path)
    paste = df[df["method"].isin(["PASTE", "PASTE2"])]
    assert len(paste) == 2
    assert not paste["files_ok"].any()
    assert paste["files_reason"].str.contains(".h5ad").all()
    assert not paste["runnable"].any()
    # env gate passed (mocked), so the ONLY reason is the file problem - in
    # its short form: the exception class and the absolute dir are gone
    assert paste["env_ok"].all()
    assert paste["reason"].str.contains(r"needs >=2 \.h5ad slice files; found 0 in EMPTY$").all()
    assert paste["files_reason"].str.startswith("FileNotFoundError: ").all()
    assert not paste["reason"].str.contains("FileNotFoundError").any()


def test_scan_runnable_equals_files_ok_and_env_ok(all_envs):
    df = mtb.scan("D11")
    assert (df["runnable"] == (df["files_ok"] & df["env_ok"])).all()
    assert ((df["reason"].str.len() > 0) == ~df["runnable"]).all()
    assert df["runnable"].any(), "D11 CITE-seq must have runnable vertical rows once envs exist"


def test_scan_env_reason_names_method_and_category(no_envs):
    df = mtb.scan("D11", "cross")
    row = df[df["method"] == "StabMap"].iloc[0]
    assert "--methods StabMap" in row["env_reason"]
    assert "--category cross" in row["env_reason"]
    assert "--packed --run" in row["env_reason"]
    assert f"conda env {row['env']!r} is not installed" in row["env_reason"]
    assert not row["env_ok"]


def test_scan_new_columns_present_and_old_order_kept(all_envs):
    df = mtb.scan("D11")
    assert list(df.columns[:len(OLD_COLUMNS)]) == OLD_COLUMNS
    assert list(df.columns) == W.SCAN_COLUMNS == OLD_COLUMNS + NEW_COLUMNS
    assert set(df["atac"].dropna()) <= {"peak", "gene_activity"}
    assert df["needs_labels"].dtype == bool
    assert df["files_ok"].dtype == bool and df["env_ok"].dtype == bool
    # needs_labels is PER VARIANT (what the runner will demand for that row)
    mat = df[(df["method"] == "Matilda") & (df["modalities"] == "rna+adt")].iloc[0]
    assert bool(mat["needs_labels"]) is True
    # atac is None for variants that take no ATAC input
    assert mat["atac"] is None or pd.isna(mat["atac"])
    glue = df[df["method"] == "GLUE"].iloc[0]
    assert glue["atac"] == "peak"


def test_scan_methods_filter_and_unknown_raises():
    df = mtb.scan("D11", "vertical", methods=["Matilda"])
    assert set(df["method"]) == {"Matilda"} and len(df) >= 1
    with pytest.raises(KeyError, match="unknown method 'Nope'"):
        mtb.scan("D11", "vertical", methods=["Nope"])
    with pytest.raises(KeyError, match="did you mean 'StabMap'"):
        mtb.scan("D11", "cross", methods=["Stabmap"])


def test_scan_modalities_filter_accepts_protein_alias():
    a = mtb.scan("D11", "vertical", modalities=["rna", "protein"])
    b = mtb.scan("D11", "vertical", modalities=["rna", "adt"])
    assert len(a) > 0 and set(a["modalities"]) == {"rna+adt"}
    pd.testing.assert_frame_equal(a, b)


def test_scan_verbose_prints_one_summary_line(capsys, no_envs):
    df = mtb.scan("D11", "vertical", verbose=True)
    out = capsys.readouterr().out
    n = len(df)
    assert f"[scan] files OK for {int(df['files_ok'].sum())}/{n} method rows; 0/{n} envs installed" in out
    # default is silent (run_all calls scan internally)
    mtb.scan("D11", "vertical")
    assert capsys.readouterr().out == ""


def test_scan_label_length_mismatch_is_a_file_problem(tmp_path, all_envs):
    d = tmp_path / "BADCTY"
    d.mkdir()
    _h5(d / "rna.h5", 30, 60, "g")
    _h5(d / "adt.h5", 5, 60, "p")
    pd.DataFrame({"x": ["A"] * 59}).to_csv(d / "cty.csv", index=False)   # one short
    df = mtb.scan("BADCTY", "vertical", data_path=tmp_path)
    mat = df[(df["method"] == "Matilda") & (df["modalities"] == "rna+adt")].iloc[0]
    assert not mat["files_ok"]
    assert "cty.csv has 59 labels" in mat["files_reason"] and "60 cells" in mat["files_reason"]
    # a method that takes no label file is unaffected
    tot = df[(df["method"] == "totalVI") & (df["modalities"] == "rna+adt")].iloc[0]
    assert tot["files_ok"]


def test_scan_atac_gas_peak_caveat(tmp_path, all_envs):
    d = tmp_path / "PEAKY"
    d.mkdir()
    _h5(d / "rna.h5", 30, 50, "g")
    rng = np.random.default_rng(1)
    with h5py.File(d / "atac.h5", "w") as f:       # the atac_gas FALLBACK file ...
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(1.0, size=(40, 45)).astype(float))
        g.create_dataset("features", data=np.array(      # ... holding PEAKS
            [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(40)], dtype="S24"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(45)], dtype="S8"))
    pd.DataFrame({"x": ["A"] * 50}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["A"] * 45}).to_csv(d / "atac_cty.csv", index=False)
    df = mtb.scan("PEAKY", "diagonal", data_path=tmp_path)
    portal = df[df["method"] == "Portal"].iloc[0]          # wants gene activity
    assert portal["files_ok"]
    assert "atac_gas resolved to a PEAK matrix" in portal["caveat"]
    assert "chr:start-end" in portal["caveat"]
    # a method that WANTS peaks behind the atac_gas role gets no caveat
    peak_wanters = df[(df["atac"] == "peak") & df["modalities"].str.contains("atac_gas")]
    assert len(peak_wanters) > 0
    assert not peak_wanters["caveat"].str.contains("PEAK matrix").any()


def test_scan_registration_needs_obsm_spatial(tmp_path, all_envs):
    import anndata as ad
    d = tmp_path / "SLICES"
    d.mkdir()
    for i in range(2):
        a = ad.AnnData(np.ones((10, 4), dtype=float))
        a.write_h5ad(d / f"slice_{i}.h5ad")              # no obsm['spatial']
    df = mtb.scan("SLICES", "cross", data_path=tmp_path)
    paste = df[df["method"] == "PASTE"].iloc[0]
    assert not paste["files_ok"] and "obsm['spatial']" in paste["files_reason"]
    for i in range(2):
        a = ad.AnnData(np.ones((10, 4), dtype=float))
        a.obsm["spatial"] = np.zeros((10, 2))
        a.write_h5ad(d / f"slice_{i}.h5ad")
    df = mtb.scan("SLICES", "cross", data_path=tmp_path)
    paste = df[df["method"] == "PASTE"].iloc[0]
    assert paste["files_ok"], paste["files_reason"]


# --- run_all(dry_run=True) is the scan frame, never silently empty ------------

def test_run_all_dry_run_keeps_blocked_rows_with_reason(no_envs):
    plan = mtb.run_all("D11", "vertical", out_dir="/tmp/unused", methods=["Matilda"],
                       dry_run=True, verbose=False)
    assert len(plan) >= 1
    assert not plan["runnable"].any()
    assert plan["reason"].str.contains("not installed").all()
    assert "files_ok" in plan.columns and "env_ok" in plan.columns


def test_run_all_dry_run_prints_summary(capsys, no_envs):
    mtb.run_all("D11", "vertical", out_dir="/tmp/unused", methods=["Matilda"],
                dry_run=True, verbose=True)
    out = capsys.readouterr().out
    assert "[run_all] dry run: 0 of" in out and "reason column" in out


def test_run_all_dry_run_never_returns_empty():
    with pytest.raises(ValueError, match="no 'diagonal' variant matches"):
        mtb.run_all("D28", "diagonal", out_dir="/tmp/unused", methods=["Matilda"],
                    dry_run=True, verbose=False)


def test_run_all_dry_run_equals_scan_with_methods():
    plan = mtb.run_all("D11", "vertical", out_dir="/tmp/unused",
                       methods=["Matilda", "totalVI"], dry_run=True, verbose=False)
    # the plan is the scan frame plus the `command` column the CLI csv carries
    assert list(plan.columns) == W.SCAN_COLUMNS + ["command"]
    pd.testing.assert_frame_equal(plan.drop(columns="command"),
                                  mtb.scan("D11", "vertical", methods=["Matilda", "totalVI"]))


def test_run_all_dry_run_unknown_method_still_keyerror():
    with pytest.raises(KeyError, match="unknown method"):
        mtb.run_all("D28", "diagonal", out_dir="/tmp/unused", methods=["NotAMethod"],
                    dry_run=True, verbose=False)
    with pytest.raises(KeyError, match="did you mean 'StabMap'"):
        mtb.run_all("D11", "cross", out_dir="/tmp/unused", methods=["Stabmap"],
                    dry_run=True, verbose=False)


def test_run_all_dry_run_missing_dataset_raises_filenotfound():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mtb.run_all("NO_SUCH_DATASET_XYZ", "vertical", out_dir="/tmp/unused",
                    dry_run=True, verbose=False)


def test_plan_is_the_dry_run_frame(no_envs):
    from multibench.workflow import plan
    a = plan("D11", "vertical", methods=["Matilda"])
    b = mtb.run_all("D11", "vertical", out_dir=None, methods=["Matilda"],
                    dry_run=True, verbose=False)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) >= 1 and not a["runnable"].any()
    with pytest.raises(ValueError):
        plan("D28", "diagonal", methods=["Matilda"])
