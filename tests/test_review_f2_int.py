"""Review of the round-2 integration (wp/f2_int).

M21: UnitedNet's old input key ``rna_cty`` keeps working (DeprecationWarning),
and labels_for keys an older folder's ``rna_cty.csv`` as ``cty``.
M23: every vertical near-miss hint names ``atac.h5``, whatever the role.
M02: the summary figure names a tied mean rank, not a metric value; the CLI
prints one one-method warning for bubble.
New: ``mtb.run`` applies the same-cell and ATAC-order checks to the files it
is given, in the dry run too.
M33: a reused output keeps the provenance of the run that made it; ``multibench
config`` shows the scripts commit as its own row.
M22, M13, M28 and the category help: message and help wording.
M25, M06: docstring wording. M10: the missing-script fix for a folder that is
not a git checkout. M05, M09, M10, M06, M30: facts on the docs pages.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config, workflow as W
from multibench.engine import envs, registry, resolve, runner
from multibench.plot import bubble as B

ROOT = Path(__file__).resolve().parents[1]
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _h5(path, feats, cells, seed=0):
    """A canonical .h5: matrix/data (features x cells) of whole counts."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data",
                         data=rng.poisson(1.0, size=(len(feats), len(cells))).astype("float32"))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(cells, dtype="S"))


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


# ====================================================================== M21
def _old_unitednet_folder(root):
    """A vertical RNA + gene-activity folder that names its labels rna_cty.csv."""
    d = root / "OLD"
    d.mkdir()
    cells = [f"c{i}" for i in range(30)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(20)], cells)
    _h5(d / "atac.h5", [f"G{i}" for i in range(15)], cells, seed=1)
    pd.DataFrame({"x": ["T", "B"] * 15}).to_csv(d / "rna_cty.csv", index=False)
    return d


def _old_inputs(d):
    return {"rna": str(d / "rna.h5"), "atac_gas": str(d / "atac.h5"),
            "rna_cty": str(d / "rna_cty.csv")}


def test_unitednet_takes_the_old_rna_cty_key_with_a_deprecation_warning(tmp_path):
    d = _old_unitednet_folder(tmp_path)
    with pytest.warns(DeprecationWarning, match=r"inputs key 'rna_cty' of UnitedNet .* "
                                                 r"use 'cty'"):
        argv = mtb.run("UnitedNet", "vertical", inputs=_old_inputs(d),
                       out_dir=str(tmp_path / "o"), dry_run=True)
    i = argv.index("--train_cty_path")
    assert argv[i + 1] == str(d / "rna_cty.csv")
    # the new key: no warning, the same command
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = mtb.run("UnitedNet", "vertical",
                      inputs={**{k: v for k, v in _old_inputs(d).items() if k != "rna_cty"},
                              "cty": str(d / "rna_cty.csv")},
                      out_dir=str(tmp_path / "o"), dry_run=True)
    assert new == argv


def test_cli_run_takes_the_old_rna_cty_key_and_shows_the_warning(tmp_path):
    """A subprocess, so Python's default warning filters apply, not pytest's."""
    d = _old_unitednet_folder(tmp_path)
    argv = [sys.executable, "-m", "multibench", "run", "--method", "UnitedNet",
            "--category", "vertical"]
    for k, v in _old_inputs(d).items():
        argv += ["--input", f"{k}={v}"]
    argv += ["--out-dir", str(tmp_path / "o"), "--dry-run"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("PYTHONWARNINGS", None)
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "--train_cty_path" in proc.stdout
    assert ("warning: inputs key 'rna_cty' of UnitedNet is deprecated and will be "
            "removed in 0.4; use 'cty' (the same file)") in proc.stderr


def test_labels_for_gives_an_older_folder_the_cty_key(tmp_path):
    d = _old_unitednet_folder(tmp_path)
    got = mtb.labels_for("OLD", "vertical", "UnitedNet", data_path=tmp_path)
    assert dict(got) == {"cty": str(d / "rna_cty.csv")}
    inp = mtb.inputs_for("OLD", "vertical", "UnitedNet", data_path=tmp_path)
    assert inp["cty"] == got["cty"]
    # without a method the stem is kept, as before
    assert list(mtb.labels_for("OLD", data_path=tmp_path)) == ["rna_cty"]


# ====================================================================== M02
def _rows(method, dataset, base=0.5):
    return [{"metric": m, "value": base + 0.01 * k, "method": method,
             "dataset": dataset, "category": "mosaic"}
            for k, m in enumerate(("ARI", "NMI", "ASW", "cLISI"))]


def _trade_places():
    """B leads on D1, A on D2: every mean rank ties at 1.5."""
    return pd.DataFrame(_rows("A", "D1") + _rows("B", "D1", 0.6)
                        + _rows("A", "D2", 0.3) + _rows("B", "D2", 0.2))


def _figure_texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts] + [t.get_text()
                                                                  for t in fig.texts]


def test_summary_mode_names_the_tied_mean_rank_not_a_metric_value():
    import matplotlib.pyplot as plt
    tbl, msgs = _messages(mtb.plot.build_table, _trade_places(), aggregate="summary")
    grey = [m for m in msgs if m.endswith("are grey.")]
    assert len(grey) == 1, msgs
    assert grey[0].startswith("All methods have the same mean rank in ")
    assert grey[0].endswith(", so those columns are grey.")
    for metric in ("ARI", "NMI", "ASW", "cLISI"):
        assert f"{metric} (1.5)" in grey[0]
    assert not any("same value" in m or "1.500" in m for m in msgs)
    for overall in ("rank", "mean_overall"):
        fig, msgs = _messages(mtb.plot.bubble, _trade_places(), aggregate="summary",
                              overall=overall)
        notes = [t for t in _figure_texts(fig) if t.startswith("Grey fill")]
        assert len(notes) == 1 and notes[0].startswith(
            "Grey fill: every method has the same mean rank in "), notes
        assert "1.500" not in notes[0] and "(1.5)" in notes[0]
        plt.close(fig)


def test_dataset_mode_keeps_the_metric_value():
    import matplotlib.pyplot as plt
    df = pd.DataFrame(_rows("A", "D1") + _rows("B", "D1", 0.6))
    df.loc[df["metric"] == "cLISI", "value"] = 0.0
    _, msgs = _messages(mtb.plot.build_table, df)
    assert "All methods have the same cLISI (0.000), so that column is grey." in msgs
    fig, _ = _messages(mtb.plot.bubble, df)
    assert "Grey fill: all rows equal in cLISI (0.000)." in _figure_texts(fig)
    plt.close(fig)


def test_cli_plot_bubble_with_one_method_warns_once(tmp_path, capsys):
    path = tmp_path / "mine.csv"
    pd.DataFrame(_rows("MyMethod", "MINE")).to_csv(path, index=False)
    rc = cli.main(["plot", "bubble", "--input", str(path), "--out", str(tmp_path / "b.png")])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert err.count("Only one method") == 1, err
    assert "Only one method, MyMethod, is in this figure" in err
    rc = cli.main(["plot", "bar", "--input", str(path), "--out", str(tmp_path / "r.png")])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "warning: this table has one method, so every rank is the same" in err


# ====================================================== cell checks in run()
def test_run_refuses_seurat_v5_on_files_from_different_cells(tmp_path, capsys):
    inp = mtb.inputs_for("D28", "diagonal", "Seurat_v5")
    want = ("Seurat_v5/D28/diagonal: Seurat_v5 needs RNA and ATAC from the same cells "
            "as its bridge. These files share 0 of 6,408 and 4,606 cells")
    for dry in (True, False):
        with pytest.raises(ValueError) as e:
            mtb.run("Seurat_v5", "diagonal", inputs=inp, out_dir=str(tmp_path / "o"),
                    dry_run=dry)
        assert str(e.value) == want
    assert not (tmp_path / "o").exists()
    rc = cli.main(["run", "--method", "Seurat_v5", "--category", "diagonal",
                   "--input", f"rna={inp['rna']}", "--input", f"atac_peak={inp['atac_peak']}",
                   "--out-dir", str(tmp_path / "o"), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 1 and want in err


def _diagonal(root, name, gas_order):
    """rna.h5, atac_peak.h5 and atac_gas.h5 whose cells follow ``gas_order``."""
    d = root / name
    d.mkdir()
    atac = [f"a{i}" for i in range(24)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(20)], [f"r{i}" for i in range(30)])
    _h5(d / "atac_peak.h5", [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)], atac)
    _h5(d / "atac_gas.h5", [f"g{i}" for i in range(20)], [atac[i] for i in gas_order])
    return d


def test_run_refuses_a_gene_activity_file_in_another_cell_order(tmp_path):
    d = _diagonal(tmp_path, "SHUF", list(range(23, -1, -1)))
    inp = {r: str(d / f"{r}.h5") for r in ("rna", "atac_peak", "atac_gas")}
    with pytest.raises(ValueError, match=r"^MultiMAP/SHUF/diagonal: atac_gas\.h5 lists "
                                         r"the ATAC cells in another order than "
                                         r"atac_peak\.h5"):
        mtb.run("MultiMAP", "diagonal", inputs=inp, out_dir=str(tmp_path / "o"),
                dry_run=True)
    # SCALEX reads rna + atac_gas: the peak file next to it sets the order
    with pytest.raises(ValueError, match="another order"):
        mtb.run("SCALEX", "diagonal", inputs={"rna": inp["rna"], "atac_gas": inp["atac_gas"]},
                out_dir=str(tmp_path / "o"), dry_run=True)
    # other cells
    d2 = tmp_path / "OTHER"
    d2.mkdir()
    for f in ("rna.h5", "atac_peak.h5"):
        shutil.copy(d / f, d2 / f)
    _h5(d2 / "atac_gas.h5", [f"g{i}" for i in range(20)], [f"x{i}" for i in range(24)])
    with pytest.raises(ValueError, match="hold different cells"):
        mtb.run("MultiMAP", "diagonal",
                inputs={r: str(d2 / f"{r}.h5") for r in ("rna", "atac_peak", "atac_gas")},
                out_dir=str(tmp_path / "o"), dry_run=True)


def test_run_passes_files_that_agree(tmp_path):
    d = _diagonal(tmp_path, "OK", list(range(24)))
    inp = {r: str(d / f"{r}.h5") for r in ("rna", "atac_peak", "atac_gas")}
    argv = mtb.run("MultiMAP", "diagonal", inputs=inp, out_dir=str(tmp_path / "o"),
                   dry_run=True)
    assert isinstance(argv, list) and argv
    # Seurat_v5 on RNA and ATAC of the same cells
    p = tmp_path / "PAIRED"
    p.mkdir()
    cells = [f"c{i}" for i in range(25)]
    _h5(p / "rna.h5", [f"g{i}" for i in range(20)], cells)
    _h5(p / "atac_peak.h5", [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(30)],
        cells[::-1])
    argv = mtb.run("Seurat_v5", "diagonal",
                   inputs={"rna": str(p / "rna.h5"), "atac_peak": str(p / "atac_peak.h5")},
                   out_dir=str(tmp_path / "o5"), dry_run=True)
    assert isinstance(argv, list) and argv


# ====================================================================== M33
def _reused_matilda(tmp_path, monkeypatch, earlier=None):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(config, "scripts_commit", lambda repo=None: "e" * 40)
    monkeypatch.setattr(W, "_run", lambda **kw: pytest.fail("reused output was re-run"))
    mdir = tmp_path / "Matilda_D11"
    mdir.mkdir()
    with h5py.File(mdir / "embedding.h5", "w") as f:
        f.create_dataset("data", data=np.zeros((2864, 3)))
    if earlier is not None:
        (tmp_path / "batch_result.json").write_text(json.dumps(
            {"dataset": "D11", "category": "vertical", "records": [earlier]}))
    res = mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir=str(tmp_path),
                      skip_existing=True, evaluate=False, verbose=False)
    return res.results[0]


def test_reused_output_without_an_earlier_record_has_unknown_provenance(tmp_path,
                                                                       monkeypatch):
    rec = _reused_matilda(tmp_path, monkeypatch)
    assert rec["reused"] is True
    assert rec["scripts_commit"] is None
    assert rec["hostname"] == ""
    assert rec["env_flavor"] == "unknown"


def test_reused_output_copies_the_earlier_record(tmp_path, monkeypatch):
    earlier = {"method": "Matilda", "status": "RUN_OK", "scripts_commit": "a" * 40,
               "env_flavor": "gpu", "hostname": "gpu-node-7"}
    rec = _reused_matilda(tmp_path, monkeypatch, earlier)
    assert (rec["scripts_commit"], rec["env_flavor"], rec["hostname"]) == \
        ("a" * 40, "gpu", "gpu-node-7")
    saved = json.loads((tmp_path / "batch_result.json").read_text())["records"]
    assert saved[0]["hostname"] == "gpu-node-7"


def _git_repo(path):
    (path / "tools_scripts").mkdir(parents=True)
    (path / "tools_scripts" / "x.py").write_text("print(1)\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(path), "PATH": "/usr/bin:/bin"}
    for argv in (["git", "init", "-q"], ["git", "add", "."],
                 ["git", "commit", "-q", "-m", "x"]):
        subprocess.run(argv, cwd=path, check=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


@needs_git
def test_config_prints_the_scripts_commit_as_its_own_row(tmp_path, monkeypatch, capsys):
    head = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "repo")
    assert cli.main(["config"]) == 0
    lines = capsys.readouterr().out.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("repo_path"))
    assert "method scripts" not in lines[i + 1] and len(lines[i + 1]) < 110
    assert re.fullmatch(rf"scripts_commit\s+{head}", lines[i + 2])
    assert lines[i + 3].strip() == f"(the method scripts in {tmp_path / 'repo' / 'tools_scripts'})"
    assert cli.main(["config", "--format", "json"]) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["scripts_commit"]["value"] == head
    assert "method scripts" not in rows["repo_path"]["source"]


def test_config_says_when_the_scripts_are_not_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "none")
    row = next(r for r in config._sources() if r["name"] == "scripts_commit")
    assert row["value"] == "not fetched"
    assert "multibench fetch --scripts" in row["source"]
    (tmp_path / "plain" / "tools_scripts").mkdir(parents=True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "plain")
    row = next(r for r in config._sources() if r["name"] == "scripts_commit")
    assert row["value"] == "unknown" and "is not a git checkout" in row["source"]


# ====================================================================== M22
def test_glue_d28_caveat_names_the_rename_call(tmp_path):
    # R3-04: mtb.run rewrites D28's chr_start_end names for GLUE, so the
    # caveat has no peak-name clause and does not name the dataset
    r = mtb.scan("D28", "diagonal", methods=["GLUE"], verbose=False).iloc[0]
    assert r["caveat"].startswith("GLUE needs the GENCODE v43 human annotation")
    assert "peak names" not in r["caveat"] and "D28" not in r["caveat"]
    assert ";" not in r["caveat"] and "D27" not in r["caveat"]
    # the named call writes the spelling GLUE needs
    src = Path(mtb.inputs_for("D28", "diagonal", "GLUE")["atac_peak"])
    out = mtb.io.normalize_peak_names(src, tmp_path / "atac_peak.h5")
    with h5py.File(out, "r") as f:
        first = f["matrix/features"][0].decode()
    assert re.fullmatch(r"chr\w+:\d+-\d+", first), first


def test_caveat_clauses_join_as_sentences():
    # R6-10: the caveat parts are sentences, joined with a space
    assert W._join_clauses(["A sentence.", "", "GLUE needs b"]) == "A sentence. GLUE needs b."
    assert W._join_clauses(["Only one."]) == "Only one."


# ====================================================================== M13
def test_cli_messages_name_commands_not_python_calls(tmp_path, capsys):
    inp = mtb.inputs_for("D28", "diagonal", "SCALEX")
    rc = cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                   "--input", f"rna={inp['rna']}", "--input", f"atac_gas={inp['atac_gas']}",
                   "--out-dir", str(tmp_path / "o"), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 0 and "# Dry run. Nothing was executed. multibench run would execute:" in err
    assert "run()" not in err
    rc = cli.main(["params", "Matilda"])
    out = capsys.readouterr().out
    assert rc == 0 and "(set with --param KEY=VALUE)" in out
    assert "run(params" not in out and "mtb.params_for" not in out
    rc = cli.main(["run-all", "D52", "--category", "cross", "--methods", "UINMF",
                   "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 0
    assert ("The commands below are what multibench run would execute. Rows with "
            "files_ok False have none.") in err
    assert "run()" not in err
    assert "method_info(" not in mtb.env.DIFFICULTY["blocked-script"]


# ====================================================== M28 and the help texts
def _actions():
    import argparse

    def walk(parser, path):
        for a in parser._actions:
            if isinstance(a, argparse._SubParsersAction):
                for n, sp in a.choices.items():
                    yield from walk(sp, path + [n])
            elif a.help:
                yield path, a
    return list(walk(cli.build_parser(), []))


def _help(path, flag):
    return next(a.help for p, a in _actions() if p == path and flag in a.option_strings)


def test_help_texts_put_the_object_first_and_end_in_plain_sentences():
    assert _help(["scan"], "--assume-gpu") == (
        "skip this host's GPU test. Use it on a login node without a GPU to check a "
        "job for a GPU node (mtb.scan(assume_gpu=True))")
    assert _help(["fetch"], "--ref").endswith(
        "Scripts already present must be at that commit or tag.")
    doc = " ".join(mtb.scan.__doc__.split())
    assert ("``True`` = skip this host's GPU test; for a login node without a GPU that "
            "checks a GPU-node job.") in doc
    doubled = [(" ".join(p), a.option_strings or [a.dest], a.help) for p, a in _actions()
               if re.search(r"\)\s*\(", a.help)]
    assert doubled == []
    assert _help(["params"], "--category").endswith(
        "each with RNA and ADT). Only that category's variants.")
    assert _help(["list"], "--task") == (
        "task within the category: clustering, batch or dimension_reduction. "
        "mtb.list_tasks() lists them. The same filter as mtb.find_methods(task=).")


# ====================================================================== M25
def test_docstrings_drop_the_justification_tails():
    from multibench.eval.pipeline import to_long
    texts = {
        "to_long": to_long.__doc__,
        "summary": W.BatchResult.summary.__doc__,
        "plan": mtb.env.plan.__doc__,
        "load_results": mtb.load_results.__doc__,
    }
    flat = {k: " ".join(v.split()) for k, v in texts.items()}
    assert "keep their provenance through a CSV round trip" not in flat["to_long"]
    assert ("The result has the columns of ``mtb.load_results`` and concatenates with "
            "the stored tables.") in flat["to_long"]
    assert "the placeholder the plotting layer uses" not in flat["to_long"]
    assert "bounded above by the ARI itself" not in flat["summary"]
    # R4-14 deleted the ratio sentence; the formula stays
    assert "not a difference, because" not in flat["summary"]
    assert "``(best - runner_up) / best``" in flat["summary"]
    assert "so a whole category needs only a few envs" not in flat["plan"]
    assert "go straight to" not in flat["load_results"]


# ====================================================================== M06
def test_public_docstrings_say_cell_order():
    import inspect
    from multibench.eval.pipeline import evaluate
    for obj in (mtb.labels_for, mtb.inputs_for, evaluate, runner.RunResult, mtb.run):
        doc = inspect.getdoc(obj)
        for old in ("stacking order", "cell-stacking", "stacks its cells", "stacks the cells"):
            assert old not in doc, (obj, old)
    assert inspect.getdoc(mtb.labels_for).splitlines()[0] == (
        "Return a dataset's cell-type label files, in the method's cell order.")


# ====================================================================== M10
def test_missing_script_in_a_folder_that_is_not_a_git_checkout(tmp_path, monkeypatch):
    """A tools_scripts/ made by hand for GLUE's annotation, before any fetch."""
    glue = tmp_path / "tools_scripts" / "GLUE"
    glue.mkdir(parents=True)
    (glue / "gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz").write_bytes(b"")
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path)
    variant = registry.get("GLUE").variants[0]
    why = W._missing_script(variant, method="GLUE")
    assert "missing from the reference checkout" in why
    assert "git pull" not in why
    assert "not a git checkout" in why and "multibench fetch --scripts" in why
    notes = runner.script_notes(registry.get("GLUE"), variant, tmp_path)
    assert any("not a git checkout" in n for n in notes) and not any(
        "git pull" in n for n in notes)


@needs_git
def test_missing_script_in_a_git_checkout_keeps_git_pull(tmp_path, monkeypatch):
    _git_repo(tmp_path / "repo")
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "repo")
    why = W._missing_script(registry.get("GLUE").variants[0], method="GLUE")
    assert "git pull" in why


# ============================================================ docs pages
def _docs_root():
    root = os.environ.get("SCMULTIBENCH_DOCS")
    return Path(root) if root and Path(root).is_dir() else None


needs_docs = pytest.mark.skipif(_docs_root() is None, reason="SCMULTIBENCH_DOCS not set")


def _laptop_tab():
    text = (_docs_root() / "quickstart.md").read_text()
    tab = text.split('=== "CITE-seq on a laptop"', 1)[1].split('\n=== "', 1)[0]
    code = tab.split("```python", 1)[1].split("```", 1)[0]
    return "\n".join(line[4:] if line.startswith("    ") else line.lstrip()
                     for line in code.splitlines())


def _cite_adata(n_cells=400, n_genes=2500, n_adt=14, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.choice(["T", "B", "NK"], n_cells)
    shift = pd.Series(types).map({"T": 0.5, "B": 1.5, "NK": 3.0}).to_numpy()[:, None]
    a = ad.AnnData(rng.poisson(0.3 + shift * rng.random(n_genes), (n_cells, n_genes))
                   .astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["celltype"] = types
    a.obsm["protein"] = rng.poisson(5 + 10 * shift * rng.random(n_adt),
                                    (n_cells, n_adt)).astype("float32")
    return a


@needs_docs
def test_quickstart_laptop_tab_runs_on_a_panel_of_14_adts(tmp_path, monkeypatch, capsys):
    """Pasted alone into a fresh file, the laptop tab imports multibench, reads
    my_citeseq.h5ad itself (R3-20/R3-21), prints the 2 x 6 score table and
    draws the figure."""
    import matplotlib.pyplot as plt
    code = _laptop_tab()
    assert "sc.pp.pca(adt, n_comps=min(20, adt.n_vars - 1))" in code
    assert max(len(line) for line in code.splitlines()) <= 80
    monkeypatch.chdir(tmp_path)
    _cite_adata().write_h5ad(tmp_path / "my_citeseq.h5ad")
    scope = {}                        # nothing pre-defined: no mtb, no adata
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        exec(compile(code, "quickstart-laptop-tab", "exec"), scope)
    assert set(scope["mine"]["method"]) == {"RNA PCA", "ADT PCA"}
    assert scope["adata"].obsm["X_adt"].shape == (400, 13)
    out = capsys.readouterr().out
    table = out[out.index("metric"):].splitlines()
    assert table[0].split() == ["metric", "ARI", "ASW", "NMI", "cLISI", "iASW", "iF1"]
    assert [row.split()[:2] for row in table[2:4]] == [["ADT", "PCA"], ["RNA", "PCA"]]
    msgs = [str(w.message) for w in rec if issubclass(w.category, UserWarning)
            and "multibench" in str(w.filename)]
    assert not any("only one method" in m.lower() for m in msgs), msgs
    plt.close("all")


@needs_docs
def test_run_guide_says_to_fetch_the_scripts_before_adding_setup_files():
    run = (_docs_root() / "tutorials" / "run.md").read_text()
    glue = run.split("- GLUE:", 1)[1].split("\n- ", 1)[0]
    assert glue.index("multibench fetch --scripts") < glue.index("gencode.v43")
    mira = run.split("- MIRA:", 1)[1].split("\n\n", 1)[0]
    assert mira.index("fetch the method scripts") < mira.index("logger.py")


@needs_docs
def test_discover_explains_a_filtered_grand_score():
    text = " ".join((_docs_root() / "tutorials" / "discover.md").read_text().split())
    assert "On each dataset the best method gets 1 and the lowest 0." not in text
    assert ("On each dataset, the best of the category's scored methods gets 1 and the "
            "lowest gets 0.") in text
    assert "a filtered list can start below 1 and end above 0" in text
    # the facts behind it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peak = mtb.recommend("diagonal", atac="peak")["grand_score"]
        full = mtb.recommend("vertical")["grand_score"]
    assert peak.max() < 1 and peak.min() > 0
    assert full.max() == 1 and full.min() == 0


@needs_docs
def test_plot_guide_describes_the_averaged_row_labels():
    import matplotlib.pyplot as plt
    text = " ".join((_docs_root() / "tutorials" / "plot.md").read_text().split())
    assert "row labels carry the dataset ids" not in text
    assert ("Each row label shows the method's dataset, or how many datasets it averages "
            "(`Matilda · 2 ds`).") in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = mtb.plot.bubble(mtb.load_results("vertical", dataset=["D11", "D11s"],
                                               source="rerun"))
    labels = {t.get_text() for ax in fig.axes for t in ax.get_yticklabels()}
    labels |= set(_figure_texts(fig))
    assert "Matilda · 2 ds" in labels
    plt.close(fig)


@needs_docs
def test_api_and_changes_pages_say_cell_order():
    docs = _docs_root()
    for page in ("api.md", "changes.md"):
        text = " ".join((docs / page).read_text().split())
        for old in ("stacks its cells", "stacks the cells", "the order it stacks",
                    "in which Seurat_v5 stacks"):
            assert old not in text, (page, old)
    api = " ".join((docs / "api.md").read_text().split())
    assert "only the files that method reads, in the method's cell order." in api
    assert "in the method's cell order. It is `None` when they could not be matched." in api
    assert "run on the CPU even on a GPU host." in api
    assert "Seurat_v5's cell order." in " ".join((docs / "changes.md").read_text().split())
