"""CLI parity with the Python API (work package F_cli, proposal P10).

scan / layout / convert / cite / find filters / plot bubble|bar from stored or
own tables / evaluate --only/--method/--dataset + npy/csv / flag aliases /
help strings on every parser and argument / exit codes.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

import multibench
from multibench import cli


# ----------------------------------------------------------------- helpers
def _walk_parsers(parser):
    """Yield (name, parser) for the root and every (nested) subparser."""
    yield parser.prog, parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for name, sub in a.choices.items():
                yield from _walk_parsers(sub)


def _long_df():
    rows = []
    for m, vals in {"A": (0.8, 0.7, 0.6), "B": (0.5, 0.9, 0.4)}.items():
        for met, v in zip(("ARI", "NMI", "ASW"), vals):
            rows.append(dict(metric=met, value=v, method=m, dataset="D1",
                             category="vertical"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- help / exit codes
def test_cli_help_strings_everywhere():
    root = cli.build_parser()
    missing = []
    for name, p in _walk_parsers(root):
        for a in p._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            if isinstance(a, argparse._SubParsersAction):
                for ca in a._choices_actions:
                    if not ca.help:
                        missing.append(f"{name}: subcommand {ca.dest}")
                continue
            if not a.help:
                missing.append(f"{name}: {a.option_strings or a.dest}")
    assert missing == [], f"arguments without help=: {missing}"


def test_cli_top_level_help_lists_new_commands(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("scan", "layout", "convert", "cite", "plot", "run-all", "evaluate"):
        assert cmd in out
    assert "Exit codes" in out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["--version"])
    assert ei.value.code == 0
    assert multibench.__version__ in capsys.readouterr().out


def test_cli_usage_error_exit_2():
    with pytest.raises(SystemExit) as ei:
        cli.main(["nonsense"])
    assert ei.value.code == 2


def test_cli_runtime_error_exit_1_and_debug_reraise(monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("kaboom from the api")
    monkeypatch.setattr(multibench, "scan", boom)
    monkeypatch.delenv("MULTIBENCH_DEBUG", raising=False)
    rc = cli.main(["scan", "D28", "--category", "diagonal"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error: kaboom from the api" in err
    assert "MULTIBENCH_DEBUG=1" in err
    monkeypatch.setenv("MULTIBENCH_DEBUG", "1")
    with pytest.raises(ValueError, match="kaboom"):
        cli.main(["scan", "D28", "--category", "diagonal"])


# ----------------------------------------------------------------- scan / layout
def test_cli_scan_prints_frame(capsys):
    rc = cli.main(["scan", "D28", "--category", "diagonal", "--methods", "SCALEX"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SCALEX" in out
    assert "runnable" in out
    # the table is printed as-is: every column scan() returns appears
    for col in multibench.scan("D28", "diagonal").columns:
        assert col in out
    assert "scBridge" not in out          # --methods filters rows


def test_cli_scan_columns_and_format(capsys):
    rc = cli.main(["scan", "D28", "--category", "diagonal", "--columns", "method,env",
                   "--format", "csv"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0] == "method,env"
    assert "runnable" not in out


def test_cli_scan_unknown_column_and_method_errors(capsys):
    rc = cli.main(["scan", "D28", "--category", "diagonal", "--columns", "nope"])
    err = capsys.readouterr().err
    assert rc == 1 and "unknown column(s) ['nope']" in err and "available:" in err
    rc = cli.main(["scan", "D28", "--category", "diagonal", "--methods", "NotAMethod"])
    err = capsys.readouterr().err
    assert rc == 1 and "not in the scan table" in err


def test_cli_scan_requires_category():
    with pytest.raises(SystemExit) as ei:
        cli.main(["scan", "D28"])
    assert ei.value.code == 2


def test_cli_layout(capsys):
    rc = cli.main(["layout", "vertical"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == multibench.describe_layout("vertical").strip()
    rc = cli.main(["layout"])
    assert rc == 0
    assert "data_path" in capsys.readouterr().out


# ----------------------------------------------------------------- find
def test_cli_find_modalities(capsys):
    rc = cli.main(["find", "--modalities", "rna,adt"])
    out = capsys.readouterr().out.split()
    assert rc == 0
    assert out == multibench.find_methods(modalities=["rna", "adt"])


def test_cli_find_needs_labels_false(capsys):
    rc = cli.main(["find", "--category", "diagonal", "--needs-labels", "false"])
    out = capsys.readouterr().out.split()
    assert rc == 0
    assert "SCALEX" in out and "scBridge" not in out
    assert out == multibench.find_methods(category="diagonal", needs_labels=False)
    # the bare flag still means True (regression of the pre-existing behaviour)
    rc = cli.main(["find", "--category", "diagonal", "--needs-labels"])
    out = capsys.readouterr().out.split()
    assert rc == 0 and "scBridge" in out and "SCALEX" not in out
    # absent = no filter
    rc = cli.main(["find", "--category", "diagonal"])
    out = capsys.readouterr().out.split()
    assert "SCALEX" in out and "scBridge" in out


def test_cli_find_needs_labels_bad_value():
    with pytest.raises(SystemExit) as ei:
        cli.main(["find", "--needs-labels", "maybe"])
    assert ei.value.code == 2


# ----------------------------------------------------------------- cite
def test_cli_cite(capsys, tmp_path):
    rc = cli.main(["cite", "SCALEX", "--format", "text"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == multibench.cite(["SCALEX"], fmt="text").strip()
    assert "scMultiBench" in multibench.cite(["SCALEX"]) and "Xiong" in out
    rc = cli.main(["cite", "--out", str(tmp_path / "refs.bib")])
    assert rc == 0 and (tmp_path / "refs.bib").read_text().startswith("@article")
    with pytest.raises(SystemExit) as ei:
        cli.main(["cite", "SCALEX", "--all"])
    assert ei.value.code == 2


def test_cli_cite_unknown_method_exit_1(capsys):
    rc = cli.main(["cite", "NotAMethod"])
    assert rc == 1
    assert "NotAMethod" in capsys.readouterr().err


# ----------------------------------------------------------------- convert
def test_cli_convert_single_file_and_dataset(tmp_path, capsys):
    ad = pytest.importorskip("anndata")
    import h5py
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(30, 12)).astype(float))
    a.var_names = [f"g{i}" for i in range(12)]
    a.obs_names = [f"c{i}" for i in range(30)]
    a.obsm["protein"] = rng.poisson(2.0, size=(30, 4)).astype(float)
    a.obs["ct"] = pd.Categorical(rng.choice(["T", "B"], size=30))
    src = tmp_path / "cite.h5ad"
    a.write_h5ad(src)

    # mode 1: one canonical file; OUT is a dir + --modality picks the name
    d1 = tmp_path / "one"; d1.mkdir()
    rc = cli.main(["convert", str(src), str(d1), "--modality", "rna"])
    out = capsys.readouterr().out
    assert rc == 0 and f"wrote {d1 / 'rna.h5'}" in out
    with h5py.File(d1 / "rna.h5") as f:
        assert f["matrix/data"].shape == (12, 30)        # features x cells
    # mode 1 with --obsm
    rc = cli.main(["convert", str(src), str(d1 / "adt.h5"), "--modality", "adt",
                   "--obsm", "protein"])
    assert rc == 0
    with h5py.File(d1 / "adt.h5") as f:
        assert f["matrix/data"].shape == (4, 30)

    # mode 2: whole dataset folder via export_dataset
    d2 = tmp_path / "MYCITE"
    rc = cli.main(["convert", str(src), str(d2), "--rna", "X", "--adt", "obsm:protein",
                   "--labels", "obs:ct"])
    out = capsys.readouterr().out
    assert rc == 0 and "wrote dataset folder" in out
    assert sorted(p.name for p in d2.iterdir()) == ["adt.h5", "cty.csv", "rna.h5"]
    assert list(pd.read_csv(d2 / "cty.csv").columns) == ["x"]
    # and the folder scans (data_path = parent)
    rc = cli.main(["scan", "MYCITE", "--category", "vertical", "--data-path", str(tmp_path)])
    assert rc == 0 and "Matilda" in capsys.readouterr().out


def test_cli_convert_mode_clash_is_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["convert", "x.h5ad", str(tmp_path), "--rna", "X", "--modality", "rna"])
    assert ei.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        cli.main(["convert", "x.h5ad", str(tmp_path), "--obsm", "p", "--layer", "l"])
    assert ei.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        cli.main(["convert", "x.h5ad", str(tmp_path), "--batch", "obs:b"])
    assert ei.value.code == 2
    assert "at least one of --rna" in capsys.readouterr().err


def test_cli_convert_missing_src_exit_1(tmp_path, capsys):
    rc = cli.main(["convert", str(tmp_path / "nope.h5ad"), str(tmp_path / "o.h5"),
                   "--modality", "rna"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


# ----------------------------------------------------------------- plot
def test_cli_plot_rejects_unknown_kind(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["plot", "violin", "--category", "diagonal", "--out", "x.png"])
    assert ei.value.code == 2
    assert "invalid choice: 'violin'" in capsys.readouterr().err


def test_cli_plot_needs_category_or_input(tmp_path, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["plot", "bubble", "--out", str(tmp_path / "x.png")])
    assert ei.value.code == 2
    assert "need --category (stored results) or --input" in capsys.readouterr().err


def test_cli_plot_bar_from_input_csv(tmp_path, capsys):
    import matplotlib; matplotlib.use("Agg")
    csv = tmp_path / "long.csv"
    _long_df().to_csv(csv, index=False)
    png = tmp_path / "bar.png"
    rc = cli.main(["plot", "bar", "--input", str(csv), "--methods", "A,B", "--out", str(png)])
    assert rc == 0 and png.exists()
    # stored-results path for bar
    png2 = tmp_path / "bar2.png"
    rc = cli.main(["plot", "bar", "--category", "diagonal", "--dataset", "D27",
                   "--out", str(png2)])
    assert rc == 0 and png2.exists()
    # bubble from the same csv with a title and --overall
    pdf = tmp_path / "bub.pdf"
    rc = cli.main(["plot", "bubble", "--input", str(csv), "--metrics", "ARI,NMI",
                   "--title", "mine", "--overall", "mean_overall", "--out", str(pdf)])
    assert rc == 0 and pdf.exists()


def test_cli_plot_unknown_method_and_bad_input(tmp_path, capsys):
    import matplotlib; matplotlib.use("Agg")
    csv = tmp_path / "long.csv"
    _long_df().to_csv(csv, index=False)
    rc = cli.main(["plot", "bar", "--input", str(csv), "--methods", "Zed",
                   "--out", str(tmp_path / "x.png")])
    err = capsys.readouterr().err
    assert rc == 1 and "unknown method(s) ['Zed']" in err and "methods in the table" in err
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    rc = cli.main(["plot", "bar", "--input", str(bad), "--out", str(tmp_path / "x.png")])
    err = capsys.readouterr().err
    assert rc == 1 and "expected a long results table" in err
    rc = cli.main(["plot", "bar", "--input", str(tmp_path / "missing.csv"),
                   "--out", str(tmp_path / "x.png")])
    assert rc == 1 and "not a file or a run_all output directory" in capsys.readouterr().err


def test_cli_plot_single_method_warning_and_default_title(tmp_path, capsys, monkeypatch):
    import matplotlib; matplotlib.use("Agg")
    from multibench import plot as plot_ns
    seen = {}

    def fake_bubble(df, **kw):
        seen.update(kw); seen["n"] = df["method"].nunique()
    monkeypatch.setattr(plot_ns, "bubble", fake_bubble)
    csv = tmp_path / "one.csv"
    _long_df().query("method == 'A'").to_csv(csv, index=False)
    rc = cli.main(["plot", "bubble", "--input", str(csv), "--dataset", "D1",
                   "--out", str(tmp_path / "x.png")])
    cap = capsys.readouterr()
    assert rc == 0
    assert "only one method" in cap.err
    assert seen["n"] == 1 and seen["title"] == "D1"
    # default title for stored results is '<category> <dataset>'
    monkeypatch.setattr(multibench, "load_results", lambda **k: _long_df())
    rc = cli.main(["plot", "bubble", "--category", "vertical", "--dataset", "D1",
                   "--out", str(tmp_path / "x.png")])
    assert rc == 0 and seen["title"] == "vertical D1"


def test_cli_plot_source_and_methods_forwarded(tmp_path, monkeypatch):
    from multibench import plot as plot_ns
    import multibench.cli as cli_mod
    calls = {}

    def fake_load(**kw):
        calls["load"] = kw
        return _long_df()

    def fake_bubble(df, **kw):
        calls["bubble"] = kw
    monkeypatch.setattr(multibench, "load_results", fake_load)
    monkeypatch.setattr(plot_ns, "bubble", fake_bubble)
    rc = cli_mod.main(["plot", "bubble", "--category", "vertical", "--dataset", "D1",
                       "--source", "rerun", "--methods", "A,B", "--require-complete",
                       "--aggregate", "summary", "--out", str(tmp_path / "x.png")])
    assert rc == 0
    assert calls["load"] == {"category": "vertical", "dataset": ["D1"], "source": "rerun"}
    assert calls["bubble"]["methods"] == ["A", "B"]
    assert calls["bubble"]["require_complete"] is True
    assert calls["bubble"]["aggregate"] == "summary"
    assert "overall" not in calls["bubble"]          # library default kept


def test_cli_plot_input_run_all_dir(tmp_path, monkeypatch):
    """--input may be a run_all output dir: it is reloaded via load_batch()."""
    import matplotlib; matplotlib.use("Agg")
    from multibench import workflow

    class FakeBatch:
        def long(self):
            return _long_df()
    d = tmp_path / "runs"; d.mkdir()
    monkeypatch.setattr(workflow, "load_batch", lambda p: FakeBatch())
    png = tmp_path / "b.png"
    rc = cli.main(["plot", "bar", "--input", str(d), "--out", str(png)])
    assert rc == 0 and png.exists()


# ----------------------------------------------------------------- evaluate
def _wide():
    return pd.DataFrame({"Value": [0.5, 0.6, 0.7]}, index=["ARI", "NMI", "ASW"])


def test_cli_evaluate_npy_only_method_dataset(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_eval(**kw):
        captured.update(kw)
        return _wide()
    monkeypatch.setattr(multibench, "evaluate", fake_eval)
    emb = tmp_path / "e.npy"
    np.save(emb, np.zeros((5, 2)))
    labels = tmp_path / "cty.csv"
    pd.DataFrame({"x": list("ABABA")}).to_csv(labels, index=False)
    out = tmp_path / "long.csv"
    rc = cli.main(["evaluate", "--output", str(emb), "--labels", str(labels),
                   "--category", "vertical", "--only", "ARI,NMI", "--method", "M",
                   "--dataset", "D", "--out", str(out)])
    assert rc == 0
    assert captured["only"] == ["ARI", "NMI"]
    assert captured["output"] == str(emb)
    df = pd.read_csv(out)
    assert list(df.columns) == ["metric", "value", "method", "dataset", "category"]
    assert set(df["method"]) == {"M"} and set(df["dataset"]) == {"D"}
    capsys.readouterr()
    # long mode printed (no --out) has no index column
    rc = cli.main(["evaluate", "--output", str(emb), "--labels", str(labels),
                   "--category", "vertical", "--method", "M", "--dataset", "D"])
    out_txt = capsys.readouterr().out
    assert rc == 0 and out_txt.lstrip().startswith("metric")


def test_cli_evaluate_wide_default_unchanged(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(multibench, "evaluate", lambda **kw: _wide())
    out = tmp_path / "metric.csv"
    rc = cli.main(["evaluate", "--output", "e.h5", "--labels", "l.csv", "--out", str(out)])
    assert rc == 0
    df = pd.read_csv(out, index_col=0)
    assert list(df.columns) == ["Value"] and list(df.index) == ["ARI", "NMI", "ASW"]


def test_cli_evaluate_long_mode_requires_all_three(monkeypatch, capsys):
    monkeypatch.setattr(multibench, "evaluate", lambda **kw: _wide())
    with pytest.raises(SystemExit) as ei:
        cli.main(["evaluate", "--output", "e.h5", "--labels", "l.csv", "--method", "M"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "need all of --method, --dataset, --category" in err
    assert "--dataset, --category" in err


def test_cli_evaluate_clustering_alias(monkeypatch):
    captured = {}

    def fake_eval(**kw):
        captured.clear(); captured.update(kw)
        return _wide()
    monkeypatch.setattr(multibench, "evaluate", fake_eval)
    for flag in ("--clustering", "--cluster"):
        rc = cli.main(["evaluate", "--output", "e.h5", "--labels", "l.csv", flag, "x.h5"])
        assert rc == 0 and captured["clustering"] == "x.h5"
    assert "obsm" not in captured and "column" not in captured
    rc = cli.main(["evaluate", "--output", "e.h5ad", "--labels", "l.csv", "--obsm", "X_pca",
                   "--column", "celltype", "--task", "all", "--batch", "b.csv"])
    assert rc == 0
    assert captured["obsm"] == "X_pca" and captured["column"] == "celltype"
    assert captured["task"] == "all" and captured["batch"] == "b.csv"


def test_cli_evaluate_real_npy_and_csv(tmp_path, capsys):
    """End to end through the real evaluate(): .npy and .csv embeddings, --only,
    --clustering skipping the Leiden sweep."""
    rng = np.random.default_rng(0)
    n = 120
    labels = np.repeat(["A", "B", "C"], n // 3)
    emb = rng.normal(size=(n, 4)) + np.repeat(np.arange(3) * 6.0, n // 3)[:, None]
    np.save(tmp_path / "e.npy", emb)
    pd.DataFrame(emb).to_csv(tmp_path / "e.csv", index=False)
    pd.DataFrame({"x": labels}).to_csv(tmp_path / "cty.csv", index=False)
    pd.DataFrame({"x": labels}).to_csv(tmp_path / "clu.csv", index=False)
    for emb_file in ("e.npy", "e.csv"):
        rc = cli.main(["evaluate", "--output", str(tmp_path / emb_file),
                       "--labels", str(tmp_path / "cty.csv"),
                       "--clustering", str(tmp_path / "clu.csv"),
                       "--only", "ARI,NMI", "--method", "Mine", "--dataset", "T",
                       "--category", "vertical", "--out", str(tmp_path / "long.csv")])
        assert rc == 0, capsys.readouterr().err
        df = pd.read_csv(tmp_path / "long.csv")
        assert sorted(df["metric"]) == ["ARI", "NMI"]
        assert (df["value"] > 0.99).all()          # clustering == labels


def test_cli_evaluate_unknown_only_exit_1(tmp_path, capsys):
    np.save(tmp_path / "e.npy", np.zeros((4, 2)))
    pd.DataFrame({"x": list("ABAB")}).to_csv(tmp_path / "c.csv", index=False)
    rc = cli.main(["evaluate", "--output", str(tmp_path / "e.npy"),
                   "--labels", str(tmp_path / "c.csv"), "--only", "ARI,BOGUS"])
    assert rc == 1
    assert "unknown metric(s) ['BOGUS']" in capsys.readouterr().err


# ----------------------------------------------------------------- run / run-all
def test_cli_run_out_dir_alias(monkeypatch, tmp_path):
    captured = {}

    def fake_run(**kw):
        captured.clear(); captured.update(kw)
        class R: out_dir = tmp_path
        return R()
    monkeypatch.setattr(multibench, "run", fake_run)
    for flag in ("--out-dir", "--out"):
        rc = cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                       "--input", "rna=a.h5", flag, str(tmp_path / "o")])
        assert rc == 0 and captured["out_dir"] == str(tmp_path / "o")


def test_cli_run_bad_input_pair_is_usage_error(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                  "--input", "justapath.h5", "--out", "o"])
    assert ei.value.code == 2
    assert "--input must be role=path" in capsys.readouterr().err


def test_cli_run_all_dry_run_and_summary(monkeypatch, tmp_path, capsys):
    captured = {}
    plan = pd.DataFrame({"method": ["SCALEX"], "runnable": [False], "reason": ["no env"]})

    class FakeRes:
        summary = pd.DataFrame({"method": ["SCALEX"], "status": ["ok"]})

    def fake_run_all(dataset, category, **kw):
        captured.clear(); captured.update(kw, dataset=dataset, category=category)
        return plan if kw["dry_run"] else FakeRes()
    monkeypatch.setattr(multibench, "run_all", fake_run_all)
    rc = cli.main(["run-all", "D27", "--category", "diagonal", "--out-dir", str(tmp_path),
                   "--dry-run", "--methods", "SCALEX,scBridge", "--timeout", "60",
                   "--skip-existing", "--no-evaluate", "--data-path", "d"])
    out = capsys.readouterr().out
    assert rc == 0 and "dry run" in out and "no env" in out
    assert captured["methods"] == ["SCALEX", "scBridge"] and captured["timeout"] == 60.0
    assert captured["skip_existing"] is True and captured["evaluate"] is False
    assert captured["data_path"] == "d" and captured["out_dir"] == str(tmp_path)
    rc = cli.main(["run-all", "D27", "--category", "diagonal", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "status" in out and f"saved under {tmp_path}" in out
    assert captured["dry_run"] is False and captured["evaluate"] is True


# ----------------------------------------------------------------- env
def _fake_env_rows():
    return [{"env": "scmb_torch", "methods": ["SCALEX"], "exists": False, "has_lock": True},
            {"env": "lonely", "methods": ["X"], "exists": False, "has_lock": True},
            {"env": "nolock", "methods": ["Y"], "exists": False, "has_lock": False},
            {"env": "have_it", "methods": ["Z"], "exists": True, "has_lock": True}]


def test_cli_env_install_packed_dry_run_labels(monkeypatch, capsys):
    from multibench.engine import envs
    monkeypatch.setattr(envs, "doctor", lambda **kw: _fake_env_rows())
    monkeypatch.setattr(envs, "create_all", lambda **kw: _fake_env_rows())
    monkeypatch.setattr(envs, "install_packed", lambda env: pytest.fail("must not download"))
    monkeypatch.setattr(cli, "_packed_manifest", lambda: {"scmb_torch": "https://x/y.tar.gz"})
    rc = cli.main(["env", "install", "--packed"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = {l.split()[0]: l for l in out.splitlines() if l and not l.startswith("#")}
    assert "packed archive published" in lines["scmb_torch"]
    assert "no archive - lockfile build" in lines["lonely"]
    assert "no archive - NO-LOCK" in lines["nolock"]
    assert "[have" in lines["have_it"]
    assert "dry-run" in out and "--run" in out
    # without --packed the wording is unchanged
    rc = cli.main(["env", "install"])
    out = capsys.readouterr().out
    assert "build(dry-run)" in out and "packed" not in out.split("#")[0]


def test_cli_env_doctor_next_hint_and_strict(monkeypatch, capsys):
    from multibench.engine import envs
    monkeypatch.setattr(envs, "doctor", lambda **kw: _fake_env_rows())
    rc = cli.main(["env", "doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "# legend:" in out
    assert "multibench env install --methods SCALEX,X,Y --packed --run" in out
    rc = cli.main(["env", "doctor", "--strict"])
    assert rc == 1
    capsys.readouterr()
    monkeypatch.setattr(envs, "doctor", lambda **kw: [_fake_env_rows()[-1]])
    rc = cli.main(["env", "doctor", "--strict"])
    out = capsys.readouterr().out
    assert rc == 0 and "# next:" not in out


def test_packed_manifest_reads_shipped_file():
    mf = cli._packed_manifest()
    assert isinstance(mf, dict) and len(mf) > 0
    assert all(v.startswith("http") for v in mf.values())
