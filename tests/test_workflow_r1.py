"""Student-study round 1, work package 'workflow': scan / run_all / run / save.

One block per ledger item: L13 (one rule for ``modalities``), L17 (setup
steps in scan and the dry run), L18 (Linux-only wording off Linux), L20
(method scripts not on this machine), L25 (several jobs, one out_dir), L30
(the dry run names the files the run passes), L31 (reasons never cut a
path), L32 (the atac column), L53 (``RunResult.obs_names``), L61 (plain
wording).

Host-independent: the platform is pinned to Linux by conftest; the tests of
the other hosts patch ``envs.host_platform_problem`` themselves.
"""
import inspect
import json
import re
import shlex
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config, workflow as W
from multibench.engine import envs, registry, runner

DARWIN = ("method environments are linux-64 conda envs (packed archives + "
          "lockfiles); this host is darwin/arm64")
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


@pytest.fixture
def off_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)


def _rows(df):
    return set(zip(df["method"], df["modalities"]))


# ----------------------------------------------------------------- L13: modalities
def test_scan_base_atac_token_matches_every_atac_role():
    """'atac' used to be an exact role: scan('D28', 'diagonal', ['rna', 'atac'])
    returned no row although every diagonal method reads RNA + ATAC."""
    with pytest.warns(UserWarning, match="scBridge"):
        df = mtb.scan("D28", "diagonal", modalities=["rna", "atac"], verbose=False)
    full = mtb.scan("D28", "diagonal", verbose=False)
    assert _rows(df) == _rows(full[full["modalities"] != "(data_dir)"])
    assert len(df) >= 13


def test_scan_base_atac_token_keeps_the_atac_gas_rows_of_vertical():
    df = mtb.scan("D11", "vertical", modalities=["rna", "atac"], verbose=False)
    mods = set(df["modalities"])
    assert {"rna+atac", "rna+atac_gas"} <= mods
    assert all(set(m.split("+")) <= {"rna", "atac", "atac_gas", "atac_peak"} for m in mods)
    # the combination is exact: scMoMaT's rna+adt+atac variant is a different one
    assert "rna+adt+atac" not in mods


def test_scan_representation_token_uses_the_methods_atac():
    """'atac_peak' means ATAC + the method's representation, like
    find_methods(atac='peak'); it used to keep only the rna+atac_peak roles."""
    with pytest.warns(UserWarning, match="scBridge"):
        df = mtb.scan("D28", "diagonal", modalities=["rna", "atac_peak"], verbose=False)
    want = set(mtb.find_methods("diagonal", atac="peak")) - {"scBridge"}
    assert set(df["method"]) == want
    assert "MultiMAP" in want and "SCALEX" not in want
    with pytest.warns(UserWarning):
        gas = mtb.scan("D28", "diagonal", modalities=["rna", "gene_activity"], verbose=False)
    assert set(gas["method"]) == set(mtb.find_methods("diagonal", atac="gene_activity")) - {"scBridge"}


def test_scan_modality_order_does_not_matter():
    a = mtb.scan("D11", "vertical", modalities=["adt", "rna"], verbose=False)
    b = mtb.scan("D11", "vertical", modalities=["rna", "protein"], verbose=False)
    assert len(a) > 0 and set(a["modalities"]) == {"rna+adt"}
    pd.testing.assert_frame_equal(a, b)


def test_scan_numbered_and_base_tokens_on_mosaic():
    base = mtb.scan("D46", "mosaic", modalities=["rna", "adt", "atac"], verbose=False)
    assert set(base["method"]) == {"StabMap", "scMoMaT"}
    exact = mtb.scan("D46", "mosaic", modalities=["rna1", "rna2", "rna3", "adt1", "atac2"],
                     verbose=False)
    assert _rows(exact) == _rows(base)


def test_scan_unknown_modality_token_raises_with_the_vocabulary():
    with pytest.raises(ValueError, match="unknown modality 'peaks2x'.*known"):
        mtb.scan("D11", "vertical", modalities=["rna", "peaks2x"], verbose=False)


def test_run_all_dry_run_uses_the_same_rule():
    with pytest.warns(UserWarning):
        plan = mtb.run_all("D28", "diagonal", modalities=["rna", "atac"], dry_run=True,
                           verbose=False)
    assert len(plan) >= 13


@pytest.mark.parametrize("ds,cat", [("D11", "vertical"), ("D28", "diagonal"),
                                    ("D45", "mosaic"), ("D52", "cross")])
def test_scan_rows_without_modalities_are_one_per_variant(ds, cat):
    """No modalities= keeps every variant of the category, exactly once."""
    df = mtb.scan(ds, cat, verbose=False)
    want = {(s.id, "+".join(v.when.get("modalities", [])) or "(data_dir)")
            for s in registry.load() for v in s.variants if v.when.get("category") == cat}
    assert _rows(df) == want and len(df) == len(want)


# ----------------------------------------------------------------- L17: setup steps
def test_glue_setup_hint_is_user_text():
    hint = mtb.method_info("GLUE")["setup_hint"]
    assert "gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz" in hint
    assert "tools_scripts/GLUE/" in hint and "mouse" in hint
    for s in registry.load():                     # no internal registry names
        assert "cwd_at_script" not in s.setup_hint and "engine/drivers" not in s.setup_hint


def test_scan_caveat_carries_the_first_sentence_of_the_setup_hint():
    df = mtb.scan("D28", "diagonal", methods=["GLUE"], verbose=False)
    r = df.iloc[0]
    assert r["files_ok"]
    assert ("setup: GLUE needs the GENCODE v43 human annotation "
            "(gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz) in "
            "<repo_path>/tools_scripts/GLUE/") in r["caveat"]
    assert "mouse" not in r["caveat"]                   # first sentence only
    # a method without a hint gets no setup note
    tv = mtb.scan("D11", "vertical", methods=["totalVI"], verbose=False).iloc[0]
    assert "setup:" not in tv["caveat"]


def test_dry_run_prints_the_setup_hint_to_stderr_once(capsys):
    inp = mtb.inputs_for("D28", "diagonal", "GLUE")
    argv = mtb.run("GLUE", "diagonal", inputs=inp, out_dir="/tmp/unused/GLUE", dry_run=True)
    err = capsys.readouterr().err
    assert err.count("# setup: GLUE needs the GENCODE v43 human annotation") == 1
    assert isinstance(argv, list)
    # scan previews every row without printing
    mtb.scan("D28", "diagonal", methods=["GLUE"], verbose=False)
    assert capsys.readouterr().err == ""


# ----------------------------------------------------------------- L18: off Linux
def test_scan_off_linux_keeps_env_reason_short_and_says_what_works(off_linux, no_envs, capsys):
    df = mtb.scan("D11", "vertical", methods=["totalVI"])
    out = capsys.readouterr().out
    r = df.iloc[0]
    assert r["env_reason"] == (f"Linux-only environment {r['env']} "
                               f"(not installable on this computer)")
    assert "multibench env install" not in r["reason"]
    assert out.count("[scan]") == 1 and W.LINUX_ONLY_SUMMARY in out


def test_scan_on_linux_keeps_the_install_command(no_envs, capsys):
    df = mtb.scan("D11", "vertical", methods=["totalVI"])
    assert "--packed --run" in df.iloc[0]["env_reason"]
    assert "Linux-only" not in capsys.readouterr().out


def test_run_off_linux_starts_with_the_platform(off_linux, tmp_path, monkeypatch):
    for probe in (["base"], []):                    # conda present, or nothing at all
        monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None, p=probe: p)
        with pytest.raises(OSError) as e:
            mtb.run("totalVI", "vertical", inputs=mtb.inputs_for("D11", "vertical", "totalVI"),
                    out_dir=str(tmp_path / "out"))
        msg = str(e.value)
        assert msg.startswith("Methods run only on Linux (this computer is darwin/arm64). "
                              "Run this call on a Linux machine; dry_run=True previews "
                              "the method's command here.")
        # R3-16: off Linux an install refuses, so no install line follows
        assert "is not installed" not in msg and "env install" not in msg
        assert not (tmp_path / "out").exists()


# ----------------------------------------------------------------- L20: scripts
@pytest.fixture
def no_scripts(tmp_path, monkeypatch):
    empty = tmp_path / "cache" / "scMultiBench_ref"
    monkeypatch.setattr(config.DEFAULT, "repo_path", empty)
    monkeypatch.setattr(runner, "_repo_root_no_fetch", lambda: empty)
    monkeypatch.setattr(W._runner, "_repo_root_no_fetch", lambda: empty)
    return empty


def test_dry_run_says_when_the_method_scripts_are_not_here(no_scripts, capsys):
    inp = mtb.inputs_for("D11", "vertical", "totalVI")
    mtb.run("totalVI", "vertical", inputs=inp, out_dir="/tmp/unused/t", dry_run=True)
    err = capsys.readouterr().err
    assert f"# method scripts not found under {no_scripts}: the first real run clones " \
           f"PYangLab/scMultiBench with git" in err
    assert "multibench fetch --scripts" in err
    assert not no_scripts.exists()                   # nothing fetched or created


def test_cli_dry_run_says_it_too(no_scripts, capsys, tmp_path):
    d = Path(config.DEFAULT.data_path) / "D11"
    rc = cli.main(["run", "--method", "totalVI", "--category", "vertical",
                   "--input", f"rna={d / 'rna.h5'}", "--input", f"adt={d / 'adt.h5'}",
                   "--out", str(tmp_path / "o"), "--dry-run"])
    assert rc == 0 and "method scripts not found under" in capsys.readouterr().err


def test_scan_notes_missing_scripts_in_caveat_not_files_ok(no_scripts):
    df = mtb.scan("D11", "vertical", modalities=["rna", "adt"], verbose=False)
    assert df["files_ok"].all()
    assert df["caveat"].str.contains("method scripts not found under").all()


def test_real_run_without_network_names_the_fetch_command(tmp_path, monkeypatch):
    def offline(path=None):
        raise subprocess.CalledProcessError(128, ["git", "clone"])
    monkeypatch.setattr(runner.config, "ensure_repo", offline)
    with pytest.raises(RuntimeError, match="multibench fetch --scripts") as e:
        mtb.run("totalVI", "vertical", inputs=mtb.inputs_for("D11", "vertical", "totalVI"),
                out_dir=str(tmp_path / "out"), cmd_template="{cmd}")
    assert isinstance(e.value.__cause__, subprocess.CalledProcessError)
    assert not (tmp_path / "out").exists()


# ----------------------------------------------------------------- L25: one out_dir
def _res(method, dataset="D11", category="vertical", ari=0.5):
    rec = {"method": method, "category": category, "dataset": dataset,
           "modalities": ["rna", "adt"], "status": "CHAIN_OK", "out_dir": f"x/{method}",
           "metrics": {"ARI": ari, "NMI": ari}}
    return mtb.BatchResult([rec], dataset, category)


def test_save_merges_jobs_that_share_one_out_dir(tmp_path, capsys):
    _res("StabMap").save(tmp_path)
    assert capsys.readouterr().out == ""                    # nothing to merge yet
    _res("scMoMaT").save(tmp_path)
    assert f"# merged with 1 earlier record(s) in {tmp_path} (StabMap)" in capsys.readouterr().out
    back = mtb.load_batch(tmp_path)
    assert sorted(r["method"] for r in back.records) == ["StabMap", "scMoMaT"]
    assert set(pd.read_csv(tmp_path / "summary.csv")["method"]) == {"StabMap", "scMoMaT"}
    assert set(pd.read_csv(tmp_path / "long.csv")["method"]) == {"StabMap", "scMoMaT"}
    # a re-run replaces that method's record and keeps the other one
    _res("StabMap", ari=0.9).save(tmp_path)
    back = mtb.load_batch(tmp_path)
    assert len(back) == 2
    assert back.summary.set_index("method").loc["StabMap", "ARI"] == 0.9


def test_save_refuses_another_dataset_before_writing(tmp_path):
    _res("StabMap").save(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="dataset='D11' category='vertical'.*another folder"):
        _res("scMoMaT", dataset="D52", category="cross").save(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_run_all_refuses_a_mismatched_out_dir_before_any_method(tmp_path, monkeypatch):
    _res("StabMap", dataset="D52", category="cross").save(tmp_path)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(W, "_run", lambda **k: pytest.fail("dispatched"))
    with pytest.raises(ValueError, match="already holds a saved result"):
        mtb.run_all("D11", "vertical", methods=["totalVI"], out_dir=tmp_path, verbose=False)


# ----------------------------------------------------------------- L30: dry-run plan
def test_dry_run_with_anndata_names_the_converted_files(tmp_path):
    import anndata as ad
    a = ad.AnnData(np.ones((5, 3)))
    argv = mtb.run("scMVP", "vertical", inputs={"rna": a, "atac": a},
                   out_dir=str(tmp_path / "o"), dry_run=True)
    line = shlex.join(argv)
    assert "AnnData object" not in line
    assert str(tmp_path / "o" / "inputs" / "rna.h5") in argv
    assert str(tmp_path / "o" / "inputs" / "atac.h5") in argv
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("bad", ["foo.h5mu", "mudata"])
def test_dry_run_refuses_mudata_like_the_real_run(tmp_path, bad):
    if bad == "mudata":
        mudata = pytest.importorskip("mudata")
        import anndata as ad
        bad = mudata.MuData({"rna": ad.AnnData(np.ones((4, 2)))})
    for dry in (True, False):
        with pytest.raises(ValueError, match="MuData"):
            mtb.run("scMVP", "vertical", inputs={"rna": bad, "atac": bad},
                    out_dir=str(tmp_path / "o"), dry_run=dry, cmd_template="{cmd}")
    assert not (tmp_path / "o").exists()


def test_dry_run_names_the_renamed_peak_copy(tmp_path):
    inp = mtb.inputs_for("D28", "diagonal", "Seurat_v3")
    argv = mtb.run("Seurat_v3", "diagonal", inputs=inp, out_dir=str(tmp_path / "o"),
                   dry_run=True)
    assert str(tmp_path / "o" / "inputs" / "atac_peak_normpeaks.h5") in argv
    assert inp["atac_gas"] in argv                         # canonical file: as is


def _fake_popen(n_rows, calls, dims=4):
    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None, start_new_session=False):
            calls["cmd"] = cmd
            self.pid, self.returncode = 4242, 0
            with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
                f.create_dataset("data", data=np.arange(n_rows * dims, dtype=float)
                                 .reshape(n_rows, dims))

        def communicate(self):
            return "", ""

        def kill(self):
            pass

        def wait(self):
            return 0
    return FakePopen


def test_dry_run_argv_is_the_real_command(tmp_path, monkeypatch):
    import anndata as ad
    a = ad.AnnData(np.ones((6, 3)))
    a.obs_names = [f"c{i}" for i in range(6)]
    repo = tmp_path / "repo"
    (repo / "tools_scripts").mkdir(parents=True)
    kw = dict(inputs={"rna": a, "atac": a}, out_dir=str(tmp_path / "o"),
              cmd_template="{cmd}", repo_path=repo)
    preview = mtb.run("scMVP", "vertical", dry_run=True, **kw)
    calls = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen(6, calls))
    res = mtb.run("scMVP", "vertical", **kw)
    assert calls["cmd"] == preview == res.cmd


# ----------------------------------------------------------------- L31: full reasons
def _lung(tmp_path):
    """A peaks-only diagonal folder under a long path (the cut used to land
    inside it)."""
    d = tmp_path / ("a_rather_long_folder_name_" * 4) / "data" / "LUNG"
    d.mkdir(parents=True)
    src = Path(config.DEFAULT.data_path) / "D28"
    for f in ("rna.h5", "atac_peak.h5", "rna_cty.csv", "atac_cty.csv"):
        (d / f).write_bytes((src / f).read_bytes())
    return d.parent


def test_scan_reason_leads_with_the_atac_meaning_and_never_cuts_a_path(tmp_path, no_envs):
    data = _lung(tmp_path)
    df = mtb.scan("LUNG", "diagonal", data_path=data, verbose=False)
    for col in ("reason", "files_reason"):
        assert not df[col].str.contains(r" \.\.\. ", regex=True).any(), col
    scalex = df[df["method"] == "SCALEX"].iloc[0]
    assert scalex["reason"].startswith(
        "needs gene-activity ATAC (atac_gas.h5); folder has peaks (atac_peak.h5)")
    assert str(data / "LUNG" / "atac_gas.h5") in scalex["files_reason"]   # full path kept


def test_cli_json_carries_the_full_reason(tmp_path, no_envs, capsys):
    data = _lung(tmp_path)
    rc = cli.main(["scan", "LUNG", "--category", "diagonal", "--data-path", str(data),
                   "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0
    scalex = next(r for r in rows if r["method"] == "SCALEX")
    assert " ... " not in scalex["reason"] and " ... " not in scalex["files_reason"]
    assert str(data / "LUNG" / "atac_gas.h5") in scalex["files_reason"]


# ----------------------------------------------------------------- L32: atac column
def test_atac_column_is_none_for_a_variant_without_atac():
    df = mtb.scan("D11", "vertical", methods=["Seurat_WNN"], verbose=False)
    by = df.set_index("modalities")["atac"]
    assert by["rna+adt"] is None or pd.isna(by["rna+adt"])
    assert by["rna+atac"] == "peak"


def test_scan_examples_show_modalities():
    assert 'modalities=["rna", "adt"]' in inspect.getdoc(mtb.scan).split("Examples")[1]


# ----------------------------------------------------------------- L53: obs_names
def _barcodes(path):
    with h5py.File(path, "r") as f:
        return [b.decode() for b in f["matrix/barcodes"][:]]


def _fake_run(monkeypatch, tmp_path, method, category, dataset, n):
    """Run ``method`` on a demo dataset with a fake process that writes an
    ``n`` x 4 embedding into out_dir."""
    inp = mtb.inputs_for(dataset, category, method)
    repo = tmp_path / "repo"
    (repo / "tools_scripts").mkdir(parents=True, exist_ok=True)
    calls = {}
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen(n, calls))
    return inp, mtb.run(method, category, inputs=inp, out_dir=str(tmp_path / method),
                        cmd_template="{cmd}", repo_path=repo)


def test_obs_names_vertical_are_the_paired_barcodes(tmp_path, monkeypatch):
    inp = mtb.inputs_for("D11", "vertical", "totalVI")
    bars = _barcodes(inp["rna"])
    _, res = _fake_run(monkeypatch, tmp_path, "totalVI", "vertical", "D11", n=len(bars))
    assert res.obs_names == bars and res.output.shape[0] == len(res.obs_names)


@pytest.mark.parametrize("method,first", [("SCALEX", "rna"), ("uniPort", "atac_gas")])
def test_obs_names_diagonal_follow_the_stacking_order(tmp_path, monkeypatch, method, first):
    inp = mtb.inputs_for("D28", "diagonal", method)
    rna, atac = _barcodes(inp["rna"]), _barcodes(inp["atac_gas"])
    want = rna + atac if first == "rna" else atac + rna
    _, res = _fake_run(monkeypatch, tmp_path, method, "diagonal", "D28", n=len(want))
    assert res.obs_names == want
    # the same order labels_for gives the label files
    order = list(mtb.labels_for("D28", "diagonal", method))
    assert order[0].startswith(first.split("_")[0])


def test_obs_names_mosaic_one_block_per_batch(tmp_path, monkeypatch):
    inp = mtb.inputs_for("D46", "mosaic", "StabMap")
    want = sum((_barcodes(inp[r]) for r in ("rna1", "rna2", "rna3")), [])
    _, res = _fake_run(monkeypatch, tmp_path, "StabMap", "mosaic", "D46", n=len(want))
    assert res.obs_names == want


def test_obs_names_none_with_a_warning_when_counts_differ(tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="obs_names is None"):
        _, res = _fake_run(monkeypatch, tmp_path, "totalVI", "vertical", "D11", n=7)
    assert res.obs_names is None


def test_runresult_documents_obs_names_and_row_order():
    doc = inspect.getdoc(runner.RunResult)
    assert "obs_names" in doc and "labels_for" in doc
    assert 'adata.obsm["X_totalVI"] = res.output' in doc
    assert runner.RunResult("m", Path("."), [], None, {}).obs_names is None   # positional OK


# ----------------------------------------------------------------- L61: plain wording
_ALLOWED_CAPS = {"RNA", "ADT", "ATAC", "CSV", "TSV", "JSON", "GPU", "CPU", "CUDA",
                 "NVIDIA", "KNN", "UMAP", "ARI", "NMI", "ASW", "LISI", "PATH", "GTF",
                 "GENCODE", "HDF5", "CITE", "API", "CLI", "OS", "NB", "MB", "GB",
                 "ID", "UMI", "PCA", "SVD", "HVG", "FDR", "MPLBACKEND", "CONDA_PREFIX",
                 "CONDA_DEFAULT_ENV", "PYTHONNOUSERSITE", "LD_PRELOAD", "MULTIBENCH_RUN_MODE",
                 "MULTIBENCH_ENVS_DIR", "TIMEOUT", "FAIL", "WNN", "MOFA", "NULL", "README"}


def _caps(text):
    text = re.sub(r"``[^`]*``|`[^`]*`|'[^']*'|\"[^\"]*\"", "", text)   # code and quoted names
    words = re.findall(r"\b[A-Z][A-Z_]{2,}\b", text)
    ids = set(registry.list_methods())                      # GLUE, MIRA ... are names
    return sorted({w for w in words if w not in _ALLOWED_CAPS and w not in ids
                   and not w.startswith(("CHAIN_OK", "RUN_OK"))})


@pytest.mark.parametrize("obj", [mtb.scan, mtb.run_all, mtb.run, mtb.sweep, mtb.load_batch,
                                 mtb.describe_layout, mtb.BatchResult, runner.RunResult,
                                 mtb.BatchResult.save, mtb.data.fetch,
                                 mtb.data.fetch_outputs])
def test_docstrings_use_plain_wording(obj):
    doc = inspect.getdoc(obj)
    assert _caps(doc) == [], _caps(doc)
    flat = " ".join(doc.split())
    for banned in (" iff ", " trap", " gate", "tidy frame", "SCAN_COLUMNS",
                   "OUT_DIR_PLACEHOLDER", "plausible", "Replaces a hand-written",
                   "never silently", "stand-in"):
        assert banned not in flat, banned


def test_sweep_summary_line():
    assert inspect.getdoc(mtb.sweep).splitlines()[0] == \
        "Run one method once per value of one hyperparameter, each in its own out_dir."


@pytest.mark.parametrize("cat", [None, "vertical", "diagonal", "mosaic", "cross"])
def test_describe_layout_output_is_plain(cat):
    txt = mtb.describe_layout(cat)
    assert "!!" not in txt and "trap" not in txt and "gate" not in txt
    assert _caps(txt.replace("<DATASET>", "").replace("MYDATA", "")) == []


def test_cli_table_clips_reasons_at_a_word_boundary():
    """The table view may shorten a reason, but never inside a file name (L31)."""
    txt = "needs gene-activity ATAC (atac_gas.h5); folder has peaks (atac_peak.h5); x" * 2
    clipped = cli._truncate(txt, 80)
    assert len(clipped) <= 80 and clipped.endswith("...")
    assert txt.startswith(clipped[:-3])
    assert txt[len(clipped) - 3] in " ;,)"           # cut after a whole word
    assert cli._truncate("short", 80) == "short"


def test_setup_note_is_dropped_once_the_helper_file_is_in_place(tmp_path, monkeypatch):
    """MIRA's hint is about its logger.py; with the file there it is moot."""
    script = tmp_path / "tools_scripts" / "MIRA" / "main_MIRA.py"
    script.parent.mkdir(parents=True)
    script.write_text("from logger import *")
    spec = registry.get("MIRA")
    v = spec.variants[0]
    assert runner.script_notes(spec, v, tmp_path)[0].startswith("setup: MIRA needs a logger.py")
    (script.parent / "logger.py").write_text("")
    assert runner.script_notes(spec, v, tmp_path) == []
