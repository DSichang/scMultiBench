"""Student study round 2, package items M17, M19, M23, M28 and M29.

M17: the diagonal ATAC lines of ``describe_layout`` name each method once.
M19: messages printed by the command line name ``multibench ...`` commands,
not Python calls; JSON output keeps ``/`` unescaped.
M23: a vertical folder holding ``atac_peak.h5`` gets a reason that says how
to fix it.
M28: ``assume_gpu`` / ``--assume-gpu`` let a GPU-less login node check a job
for a GPU node; ``--strict`` counts the GPU test on its own.
M29: one ``--labels`` spelling for a ``.h5mu`` batch, and an error that
names it.
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config, workflow as W
from multibench.engine import envs, ingest

ad = pytest.importorskip("anndata")

ALL_ENVS = frozenset(envs.group_for(m) for m in mtb.list_methods())
CATS = ("vertical", "diagonal", "mosaic", "cross")


# ------------------------------------------------------------------- M17
def test_diagonal_layout_names_each_method_on_one_line():
    txt = mtb.describe_layout("diagonal")
    lines = txt.splitlines()
    assert ("Give atac_peak.h5, atac_gas.h5 or both. Each method needs one of them, "
            "or both:") in lines
    keys = ("need both files:", "need peaks:", "need gene activity:")
    listed = {}
    for key in keys:
        line = next(l for l in lines if l.strip().startswith(key))
        listed[key] = line.split(":", 1)[1].strip().split(", ")
    readers = set(mtb.find_methods("diagonal", atac="peak")) | set(
        mtb.find_methods("diagonal", atac="gene_activity"))
    for m in readers:
        hits = [k for k in keys if m in listed[k]]
        assert len(hits) == 1, (m, hits)
    assert {"MultiMAP", "Seurat_v3"} <= set(listed["need both files:"])
    assert "MultiMAP" not in listed["need peaks:"]


# ------------------------------------------------------------------- M19
@pytest.mark.parametrize("cat", CATS)
def test_cli_layout_names_commands_not_python_calls(cat, capsys):
    assert cli.main(["layout", cat]) == 0
    out = capsys.readouterr().out
    assert "multibench scan" in out and "mtb." not in out
    assert out.rstrip().splitlines()[-1] == (
        f"Next: multibench scan MYDATA --category {cat}; on Linux, "
        f"multibench run-all MYDATA --category {cat} --out-dir out/")
    py = mtb.describe_layout(cat)
    assert py.splitlines()[-1] == (f"Next: mtb.scan('MYDATA', '{cat}'), then, on Linux, "
                                   f"mtb.run_all('MYDATA', '{cat}', out_dir='out/')")


def test_cli_layout_overview_names_commands(capsys):
    assert cli.main(["layout"]) == 0
    out = capsys.readouterr().out
    assert "mtb." not in out and "multibench scan MYDATA --category <category>" in out


def _cite_h5ad(path, n=60):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(n, 20)).astype(float))
    a.var_names = [f"g{i}" for i in range(20)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type"] = rng.choice(["T", "B"], n)
    a.obsm["protein"] = pd.DataFrame(rng.poisson(4.0, size=(n, 5)).astype(float),
                                     index=a.obs_names, columns=[f"CD{i}" for i in range(5)])
    a.write_h5ad(path)
    return a


def test_cli_overwrite_refusal_names_the_flag(tmp_path, capsys):
    src = tmp_path / "cite.h5ad"
    _cite_h5ad(src)
    argv = ["convert", str(src), str(tmp_path / "D"), "--rna", "X", "--adt", "obsm:protein",
            "--labels", "obs:cell_type"]
    assert cli.main(argv) == 0
    capsys.readouterr()
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "pass --overwrite to replace them" in err and "overwrite=True" not in err


def test_python_overwrite_refusal_names_the_keyword(tmp_path):
    a = _cite_h5ad(tmp_path / "cite.h5ad")
    ingest.export_dataset(a, tmp_path / "D", adt="obsm:protein", labels="obs:cell_type")
    with pytest.raises(FileExistsError) as ei:
        ingest.export_dataset(a, tmp_path / "D", adt="obsm:protein", labels="obs:cell_type")
    assert "pass overwrite=True to replace them" in str(ei.value)
    assert "--overwrite" not in str(ei.value)


def _two_datasets():
    rows = []
    for ds, methods in (("D1", ["A", "B", "C"]), ("D2", ["A", "B"])):
        for i, m in enumerate(methods):
            for met, v in (("ARI", 0.3 + 0.1 * i), ("NMI", 0.4 + 0.1 * i)):
                rows.append({"metric": met, "value": v, "method": m, "dataset": ds,
                             "category": "vertical"})
    return pd.DataFrame(rows)


def _plot_warnings(fn, **kw):
    import matplotlib
    matplotlib.use("Agg")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(_two_datasets(), **kw)
    return [str(w.message) for w in rec]


@pytest.mark.parametrize("cli_mode", [False, True])
def test_plot_incomplete_fix_follows_the_caller(cli_mode, monkeypatch):
    monkeypatch.setattr(config, "_CLI", cli_mode)
    bub = [m for m in _plot_warnings(mtb.plot.bubble, aggregate="summary")
           if "C" in m and ("require" in m)]
    bar = [m for m in _plot_warnings(mtb.plot.bar) if "every dataset" in m]
    assert bub and bar
    if cli_mode:
        assert "--require-complete" in bub[0] and "require_complete=" not in bub[0]
        assert "--methods" in bar[0] and "long_df" not in bar[0]
    else:
        assert "require_complete=True" in bub[0] and "--require-complete" not in bub[0]
        assert "long_df" in bar[0] and "--methods" not in bar[0]


def test_json_output_keeps_slashes_unescaped(capsys):
    assert cli.main(["scan", "D11", "--category", "vertical", "--format", "json",
                     "--methods", "Matilda"]) == 0
    out = capsys.readouterr().out
    assert "\\/" not in out
    rows = json.loads(out)
    assert rows and "/" in rows[0]["command"]


def test_run_all_dry_run_note_names_the_cli_doctor(monkeypatch, capsys):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    mtb.run_all("D11", "vertical", dry_run=True, methods=["Matilda"])
    assert "mtb.env.doctor()" in capsys.readouterr().out
    monkeypatch.setattr(config, "_CLI", True)
    mtb.run_all("D11", "vertical", dry_run=True, methods=["Matilda"])
    out = capsys.readouterr().out
    assert "multibench env doctor" in out and "mtb." not in out


# ------------------------------------------------------------------- M23
def _mu_tc(root):
    """A vertical RNA + peaks folder written without category: atac_peak.h5."""
    rng = np.random.default_rng(0)
    n = 60
    cells = [f"c{i}" for i in range(n)]
    rna = ad.AnnData(rng.poisson(1.0, size=(n, 30)).astype(float))
    rna.obs_names, rna.var_names = cells, [f"g{i}" for i in range(30)]
    atac = ad.AnnData(rng.poisson(0.5, size=(n, 40)).astype(float))
    atac.obs_names = cells
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(40)]
    d = root / "MU_TC"
    d.mkdir()
    ingest.to_canonical(rna, d, modality="rna")
    ingest.to_canonical(atac, d, modality="peak")
    pd.DataFrame({"x": rng.choice(["T", "B"], n)}).to_csv(d / "cty.csv", index=False)
    return d


def test_vertical_peak_file_under_the_diagonal_name_says_how_to_fix(tmp_path):
    _mu_tc(tmp_path)
    df = mtb.scan("MU_TC", "vertical", data_path=tmp_path, verbose=False)
    df = df.set_index(["method", "modalities"])
    peak = df.loc[("scMVP", "rna+atac"), "reason"]
    assert ('scMVP reads atac.h5 for vertical. Rename atac_peak.h5 to atac.h5, or write it '
            'with category="vertical".') in peak
    assert "needs peak ATAC" not in peak
    gas = df.loc[("Matilda", "rna+atac"), "reason"]
    assert ("Matilda needs gene-activity ATAC (atac.h5), and the folder has peaks "
            "(atac_peak.h5).") in gas
    assert "ename" not in gas
    # peak methods whose role is atac_gas: the reason and files_reason give one
    # rule, 'vertical reads atac.h5', and never atac_gas.h5 as the rename target
    for m in ("moETM", "scMM", "iPOLNG"):
        row = df.loc[(m, "rna+atac_gas")]
        for col in ("reason", "files_reason"):
            assert "ename atac_peak.h5 to atac.h5" in row[col], (m, col, row[col])
            assert "vertical reads atac_gas.h5" not in row[col], (m, col)
            assert "reads atac_gas.h5 for vertical" not in row[col], (m, col)
            assert "to atac_gas.h5" not in row[col], (m, col)


def test_vertical_peak_fix_names_the_cli_flag(tmp_path, monkeypatch):
    _mu_tc(tmp_path)
    monkeypatch.setattr(config, "_CLI", True)
    df = mtb.scan("MU_TC", "vertical", data_path=tmp_path, methods=["scMVP"], verbose=False)
    assert ("Rename atac_peak.h5 to atac.h5, or write it with --category vertical."
            in df["reason"].iloc[0])


def test_convert_category_help_in_short_sentences():
    p = next(p for n, p in _subparsers() if n == "convert")
    act = next(a for a in p._actions if "--category" in a.option_strings)
    assert act.help.endswith("Sets the file names. vertical writes atac.h5. diagonal "
                             "writes atac_peak.h5 or atac_gas.h5, and the labels as "
                             "rna_cty.csv and atac_cty.csv. mosaic writes atac<i>.h5.")
    assert "cross" not in act.help.split("Sets the file names.")[1]


def _subparsers():
    import argparse
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return list(sub.choices.items())


def test_export_dataset_atac_lines_leave_out_cross():
    doc = ingest.export_dataset.__doc__
    assert "``'diagonal'``: ``atac_peak.h5`` or ``atac_gas.h5``" in doc
    assert "``'cross'``: ``atac_peak" not in doc


# ------------------------------------------------------------------- M28
@pytest.fixture
def login_node(monkeypatch):
    """Every env installed, no GPU on this host."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


def test_strict_assume_gpu_passes_a_gpu_only_method(login_node, capsys):
    rc = cli.main(["scan", "D45", "--category", "mosaic", "--methods", "SMILE",
                   "--strict", "--assume-gpu"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    assert cap.err == ""


def test_strict_counts_the_gpu_test_on_its_own(login_node, capsys):
    rc = cli.main(["scan", "D45", "--category", "mosaic", "--methods", "SMILE,Cobolt",
                   "--strict"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "Rows that need a GPU this host lacks: 1." in err
    assert "environment is not ready" not in err
    assert "--assume-gpu" in err


def test_strict_counts_a_missing_env_and_a_missing_gpu_apart(monkeypatch, capsys):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    rc = cli.main(["scan", "D45", "--category", "mosaic", "--methods", "SMILE,Cobolt",
                   "--strict"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "Rows whose environment is not ready: 2." in err \
        and "Rows that need a GPU this host lacks: 1." in err
    # the flag would not make SMILE runnable here: no pointer to it
    assert "--assume-gpu" not in err


def test_scan_assume_gpu_keeps_the_row_and_says_so(login_node):
    kw = dict(methods=["SMILE"], verbose=False)
    plain = mtb.scan("D45", "mosaic", **kw).iloc[0]
    assert not plain["runnable"] and "NVIDIA GPU" in plain["env_reason"]
    row = mtb.scan("D45", "mosaic", assume_gpu=True, **kw).iloc[0]
    assert row["runnable"] and row["env_ok"] and row["env_reason"] == ""
    assert "assumes the job runs on a GPU node" in row["caveat"]


def test_scan_assume_gpu_previews_the_gpu_node_command(login_node, tmp_path):
    from tests.test_gpu_requirements import _cite
    _cite(tmp_path)
    kw = dict(methods=["scMDC"], data_path=tmp_path, verbose=False)
    cpu = mtb.scan("CITE", "vertical", **kw)
    gpu = mtb.scan("CITE", "vertical", assume_gpu=True, **kw)
    assert "--device cpu" in cpu[cpu.modalities == "rna+adt"]["command"].iloc[0]
    assert "--device" not in gpu[gpu.modalities == "rna+adt"]["command"].iloc[0]


def test_scan_assume_gpu_is_a_no_op_on_a_gpu_host(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    row = mtb.scan("D45", "mosaic", methods=["SMILE"], assume_gpu=True,
                   verbose=False).iloc[0]
    assert row["runnable"] and "GPU node" not in row["caveat"]


def test_run_all_dry_run_takes_assume_gpu(login_node):
    plan = mtb.run_all("D45", "mosaic", methods=["SMILE"], dry_run=True,
                       assume_gpu=True, verbose=False)
    assert bool(plan["runnable"].iloc[0])
    with pytest.raises(ValueError, match="assume_gpu=True applies to a dry run only"):
        mtb.run_all("D45", "mosaic", "out/", methods=["SMILE"], assume_gpu=True)


def test_cli_run_all_dry_run_takes_assume_gpu(login_node, capsys):
    rc = cli.main(["run-all", "D45", "--category", "mosaic", "--out-dir", "out/",
                   "--methods", "SMILE", "--dry-run", "--assume-gpu"])
    cap = capsys.readouterr()
    assert rc == 0 and "1 of 1 row can run" in cap.err
    with pytest.raises(SystemExit) as ei:
        cli.main(["run-all", "D45", "--category", "mosaic", "--out-dir", "out/",
                  "--methods", "SMILE", "--assume-gpu"])
    assert ei.value.code == 2
    assert "--assume-gpu" in capsys.readouterr().err


# ------------------------------------------------------------------- M29
def test_h5mu_labels_spelling_is_one_and_documented():
    lay = mtb.describe_layout("mosaic")
    h5mu = [l for l in lay.splitlines() if ".h5mu" in l and "convert" in l]
    assert h5mu and all("--labels rna:cell_type" in l for l in h5mu)
    epi = [l for l in cli._CONVERT_EPILOG.splitlines() if ".h5mu" in l]
    assert epi and all("--labels rna:cell_type" in l for l in epi)
    assert "mod:rna.obs:cell_type" not in cli._CONVERT_EPILOG
    assert "obs:<col> reads the global obs" in lay


def _mdata_with_labels_in_rna_obs():
    mu = pytest.importorskip("mudata")
    rng = np.random.default_rng(0)
    n = 40
    rna = ad.AnnData(rng.poisson(1.0, size=(n, 20)).astype(float))
    rna.obs_names = [f"c{i}" for i in range(n)]
    rna.var_names = [f"g{i}" for i in range(20)]
    rna.obs["cell_type"] = rng.choice(["T", "B"], n)
    atac = ad.AnnData(rng.poisson(0.5, size=(n, 30)).astype(float))
    atac.obs_names = rna.obs_names
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(30)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mu.MuData({"rna": rna, "atac": atac})


def test_global_obs_miss_names_the_modality_spelling(tmp_path):
    m = _mdata_with_labels_in_rna_obs()
    with pytest.raises(KeyError) as ei:
        ingest.export_dataset(m, tmp_path / "B", rna="rna", atac="atac", atac_kind="peak",
                              labels="obs:cell_type", category="mosaic", batch_index=2)
    msg = str(ei.value)
    assert "mdata['rna'].obs has 'cell_type': use labels='rna:cell_type'" in msg


def test_global_obs_miss_names_the_cli_flag(tmp_path, capsys):
    m = _mdata_with_labels_in_rna_obs()
    src = tmp_path / "B.h5mu"
    m.write(src)
    rc = cli.main(["convert", str(src), str(tmp_path / "LAB"), "--rna", "mod:rna",
                   "--atac", "mod:atac", "--atac-kind", "peak", "--labels", "obs:cell_type",
                   "--category", "mosaic", "--batch-index", "2"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "mdata['rna'].obs has 'cell_type': use --labels rna:cell_type" in err
    # the spelling the error names works
    rc = cli.main(["convert", str(src), str(tmp_path / "LAB"), "--rna", "mod:rna",
                   "--atac", "mod:atac", "--atac-kind", "peak", "--labels", "rna:cell_type",
                   "--category", "mosaic", "--batch-index", "2"])
    assert rc == 0, capsys.readouterr().err
    assert (tmp_path / "LAB" / "cty2.csv").is_file()
