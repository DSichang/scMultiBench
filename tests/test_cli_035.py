"""The command line of 0.3.5: --missing-labels, --modalities all, --n-batches.

``multibench evaluate --missing-labels drop``, ``multibench scan`` and
``run-all`` with ``--modalities all`` and their note on stderr for the rows
that are not listed, ``run-all --batch`` given a column name, and
``multibench convert --n-batches``.
"""
import contextlib
import io
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
NOTE = "# D11 holds RNA and ADT. 19 rows that read ATAC are not listed. --modalities all lists them."


def _on_disk(ds):
    if not (config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")


def _cli(argv, capsys):
    """``(exit code, stdout, stderr)`` of one command, warnings as the
    command line shows them."""
    capsys.readouterr()
    rc = cli.main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _one_stream(argv):
    """``(exit code, lines)`` of one command with stdout and stderr in the
    order they were written."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue().splitlines()


def _sub(name):
    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    return action.choices[name]


def _help(name, flag):
    return next(a.help for a in _sub(name)._actions if flag in a.option_strings)


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


@pytest.fixture
def ready(monkeypatch):
    """Every environment, a GPU and the method scripts: rows with their files run."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "scripts_present", lambda cfg=None: True)
    monkeypatch.setattr(W, "_missing_script", lambda variant, method=None: "")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)


# ------------------------------------------------------------------ evaluate
def _embedding(tmp_path, n=300, gaps=30):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, 5))
    emb[n // 2:, 0] += 6.0
    np.save(tmp_path / "emb.npy", emb)
    lab = np.array(["A"] * (n // 2) + ["B"] * (n // 2), dtype=object)
    lab[:gaps] = "NA"
    pd.DataFrame({"x": lab}).to_csv(tmp_path / "labels.csv", index=False)
    return ["evaluate", "--output", str(tmp_path / "emb.npy"), "--labels",
            str(tmp_path / "labels.csv"), "--metrics", "ASW"]


def test_evaluate_missing_labels_flag(tmp_path, capsys):
    base = _embedding(tmp_path)
    told = ("error: --labels (labels.csv): 30 of the 300 cells have no label, for example "
            "'NA'. Pass --missing-labels drop to score the other 270.")
    for argv in (base, base + ["--missing-labels", "error"]):
        rc, out, err = _cli(argv, capsys)
        assert rc == 1 and out == "" and err.splitlines()[0] == told
    rc, out, err = _cli(base + ["--missing-labels", "drop"], capsys)
    assert rc == 0 and out.splitlines()[0].split() == ["Value"] and "ASW" in out
    assert err.splitlines() == ["warning: Scored the 270 labelled cells of 300."]
    # complete labels: the flag changes nothing
    pd.DataFrame({"x": ["A"] * 150 + ["B"] * 150}).to_csv(tmp_path / "labels.csv",
                                                         index=False)
    plain = _cli(base, capsys)
    assert plain[0] == 0 and plain[2] == ""
    assert _cli(base + ["--missing-labels", "drop"], capsys) == plain
    # another value is a usage error
    with pytest.raises(SystemExit) as e:
        cli.main(base + ["--missing-labels", "skip"])
    assert e.value.code == 2
    assert "invalid choice: 'skip' (choose from" in capsys.readouterr().err
    assert _help("evaluate", "--missing-labels") == (
        "what to do when a label is missing. error (default) stops. drop scores the "
        "labelled cells only and says how many")


def test_evaluate_drop_reaches_the_long_table(tmp_path, capsys):
    base = _embedding(tmp_path) + ["--name", "mine", "--dataset", "MYDATA", "--category",
                                   "vertical", "--out", str(tmp_path / "long.csv"),
                                   "--missing-labels", "drop"]
    rc, _, err = _cli(base, capsys)
    assert rc == 0 and err.startswith("warning: Scored the 270 labelled cells of 300.")
    long = pd.read_csv(tmp_path / "long.csv")
    assert set(long["scored_with"].str.rsplit("/", n=1).str[-1]) == {"270 of 300 cells"}


def test_evaluate_column_names_the_word_of_the_sheet(tmp_path, capsys):
    """--column makes a copy of the chosen column; the copy keeps the word
    the sheet writes for a missing label."""
    base = _embedding(tmp_path)
    lab = np.array(["A"] * 150 + ["B"] * 150, dtype=object)
    lab[:4] = "None"
    pd.DataFrame({"celltype": lab, "donor": "d1"},
                 index=[f"c{i}" for i in range(300)]).to_csv(tmp_path / "labels.csv")
    rc, _, err = _cli(base + ["--column", "celltype"], capsys)
    (told,) = [l for l in err.splitlines() if l.startswith("error: ")]
    assert rc == 1 and told == (
        "error: --labels (labels.csv): 4 of the 300 cells have no label, for example "
        "'None'. Pass --missing-labels drop to score the other 296.")
    rc, _, err = _cli(base + ["--column", "celltype", "--missing-labels", "drop"], capsys)
    assert rc == 0 and "warning: Scored the 296 labelled cells of 300." in err.splitlines()


@pytest.mark.parametrize("category,method,files,named", [
    ("vertical", "totalVI", ["rna.h5", "adt.h5"], "cty.csv"),
    ("diagonal", "SCALEX", ["rna.h5", "atac_gas.h5"], "rna_cty.csv and atac_cty.csv"),
    ("cross", "StabMap", ["rna1.h5", "adt1.h5", "rna2.h5", "adt2.h5"],
     "cty1.csv, cty2.csv, ..."),
    ("mosaic", "StabMap", ["rna1.h5", "adt1.h5", "rna2.h5", "adt2.h5"],
     "cty1.csv, cty2.csv, ..."),
])
def test_evaluate_dataset_without_label_files(category, method, files, named, tmp_path,
                                              capsys):
    """A folder written without labels: the error named labels_for and an
    empty dict, under a line '# labels:  from NOLAB'. The 0.3.5 text is the
    same for every category and does not name the label files (``named``)."""
    assert named
    d = tmp_path / "NOLAB"
    d.mkdir()
    for name in files:
        with h5py.File(d / name, "w") as f:
            f["matrix/data"] = np.ones((5, 30))
            f["matrix/features"] = np.array([f"f{i}" for i in range(5)], dtype="S8")
            f["matrix/barcodes"] = np.array([f"{name}{i}" for i in range(30)], dtype="S16")
    cells = 30 if category == "vertical" else 60        # the cells of the folder
    np.save(tmp_path / "emb.npy", np.random.default_rng(0).normal(size=(cells, 5)))
    argv = ["evaluate", "--output", str(tmp_path / "emb.npy"), "--dataset", "NOLAB",
            "--method", method, "--category", category, "--data-path", str(tmp_path),
            "--metrics", "ASW"]
    for extra in ([], ["--missing-labels", "drop"], ["--column", "celltype"]):
        rc, out, err = _cli(argv + extra, capsys)
        assert rc == 1 and out == ""
        assert err.splitlines() == [
            "error: NOLAB has no label file. Pass one with --labels.",
            "(set MULTIBENCH_DEBUG=1 for the full traceback)"]
    # with its label files the folder is read as before
    stems = {"vertical": ["cty"], "diagonal": ["rna_cty", "atac_cty"]}.get(
        category, ["cty1", "cty2"])
    for stem in stems:
        pd.DataFrame({"x": ["A", "B"] * 15}).to_csv(d / f"{stem}.csv", index=False)
    rc, out, err = _cli(argv, capsys)
    assert rc == 0 and err.splitlines()[0].startswith("# labels: ")


# ------------------------------------------------------------------ scan
def test_scan_modalities_all_and_the_stderr_note(no_envs, capsys):
    rc, out, err = _cli(["scan", "D11", "--category", "vertical"], capsys)
    assert rc == 0 and len(out.splitlines()) == 1 + 14
    assert out.splitlines()[0].split() == cli._COMPACT_PLAN_COLUMNS      # no atac column
    assert err.splitlines() == [NOTE]
    rc, lines = _one_stream(["scan", "D11", "--category", "vertical"])
    assert rc == 0 and lines[-1] == NOTE and len(lines) == 16              # after the table
    rc, out, err = _cli(["scan", "D11", "--category", "vertical", "--modalities", "all"],
                        capsys)
    assert rc == 0 and len(out.splitlines()) == 1 + 33 and err == ""
    assert "atac" in out.splitlines()[0].split()
    # machine formats: the table on stdout is the data, the note stays on stderr
    rc, out, err = _cli(["scan", "D11", "--category", "vertical", "--format", "csv"], capsys)
    assert rc == 0 and len(pd.read_csv(io.StringIO(out))) == 14 and err.splitlines() == [NOTE]
    rc, out, err = _cli(["scan", "D11", "--category", "vertical", "--format", "csv",
                         "--modalities", "all"], capsys)
    assert len(pd.read_csv(io.StringIO(out))) == 33 and err == ""
    # a list of tokens selects as before, and names nothing
    rc, out, err = _cli(["scan", "D11", "--category", "vertical", "--modalities", "rna,atac",
                         "--format", "csv"], capsys)
    assert set(pd.read_csv(io.StringIO(out))["modalities"]) == {"rna+atac", "rna+atac_gas"}
    assert err == ""
    # every modality held: nothing to say
    _on_disk("D22mini")
    rc, out, err = _cli(["scan", "D22mini", "--category", "vertical"], capsys)
    assert rc == 0 and len(out.splitlines()) == 1 + 33 and err == ""


def test_all_is_one_token_of_its_own(no_envs, capsys):
    rc, _, err = _cli(["scan", "D11", "--category", "vertical", "--modalities", "all,rna"],
                      capsys)
    assert rc == 1 and err.startswith("error: Unknown modality all. The tokens are rna, ")
    assert cli._modalities_arg(None) is None and cli._modalities_arg("all") == "all"
    assert cli._modalities_arg(" all ") == "all"
    assert cli._modalities_arg("rna, adt") == ["rna", "adt"] and cli._modalities_arg("") == []


def test_the_help_of_modalities():
    end = (" Default: the combinations whose modalities the folder holds. all lists every "
           "combination")
    for name in ("scan", "run-all"):
        text = _help(name, "--modalities")
        assert "atac_peak and atac_gas select by what the method reads" in text
        assert text.endswith(end), name
        assert "`" not in text
    assert _help("convert", "--overwrite") == (
        "mode 2: replace files already in OUT (default: refuse and list them). Files of "
        "other data that the call does not write are refused either way")
    assert _help("convert", "--n-batches") == (
        "mode 2, with --batch-index: how many batches the dataset has. Files of a higher "
        "batch number in OUT are refused")


def test_the_note_follows_the_strict_error(no_envs, capsys):
    rc, lines = _one_stream(["scan", "D11", "--category", "vertical", "--strict"])
    assert rc == 1
    strict = next(i for i, l in enumerate(lines) if l.startswith("error: --strict: "))
    assert lines[strict].startswith("error: --strict: 0 of 14 rows are runnable. ")
    assert lines.index(NOTE) == len(lines) - 1 > strict
    rc, _, err = _cli(["scan", "D11", "--category", "vertical", "--strict"], capsys)
    assert err.startswith("error: --strict: 0 of ") and err.splitlines()[-1] == NOTE
    rc, _, err = _cli(["scan", "D11", "--category", "vertical", "--strict", "--modalities",
                       "all"], capsys)
    assert rc == 1 and err.startswith("error: --strict: 0 of 33 rows are runnable. Rows with "
                                      "missing input files: 19. ")
    assert "not listed" not in err


@pytest.mark.parametrize("folder", ["D11", "D52mini", "EMPTY"])
def test_strict_exit_codes_are_unchanged(folder, tmp_path, monkeypatch, capsys):
    if folder == "EMPTY":
        (tmp_path / "EMPTY").mkdir()
        base = ["scan", "EMPTY", "--category", "vertical", "--data-path", str(tmp_path)]
    else:
        _on_disk(folder)
        base = ["scan", folder, "--category", "cross" if folder == "D52mini" else "vertical"]

    def codes(*extra):
        return [_cli(base + ["--strict", *more, *extra], capsys)[0]
                for more in ([], ["--modalities", "all"])]
    # nothing installed: no row can run, with every row or the listed ones
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    assert codes() == [1, 1]
    # everything in place: the rows with their files run
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "scripts_present", lambda cfg=None: True)
    monkeypatch.setattr(W, "_missing_script", lambda variant, method=None: "")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    want = 1 if folder == "EMPTY" else 0
    assert codes() == [want, want]
    if folder == "D11":
        # a named method without a runnable row fails either way
        assert codes("--methods", "Matilda,MIRA") == [1, 1]
        assert codes("--methods", "Matilda,totalVI") == [0, 0]


# ------------------------------------------------------------------ run-all
def test_run_all_dry_run_header(no_envs, capsys):
    rc, out, err = _cli(["run-all", "D11", "--category", "vertical", "--dry-run"], capsys)
    assert rc == 0 and err.splitlines()[:2] == [
        "# Dry run. Nothing was executed. 0 of 14 methods can run on D11 (vertical). The "
        "commands below are what multibench run would execute. A method whose input files "
        "are missing has no command.", NOTE]
    assert sum(l.startswith("# Commands of the 14 methods") for l in out.splitlines()) == 1
    rc, out, err = _cli(["run-all", "D11", "--category", "vertical", "--dry-run",
                         "--modalities", "all"], capsys)
    assert rc == 0 and "not listed" not in err
    assert err.startswith("# Dry run. Nothing was executed. 0 of 33 rows can run on D11 ")
    assert "# Commands of the 14 rows whose input files are in place." in out
    # the plan is the scan table, in both forms
    for extra, n in (([], 14), (["--modalities", "all"], 33)):
        rc, plan, _ = _cli(["run-all", "D11", "--category", "vertical", "--dry-run",
                            "--format", "csv", *extra], capsys)
        rc, scan, _ = _cli(["scan", "D11", "--category", "vertical", "--format", "csv",
                            *extra], capsys)
        assert plan == scan and len(pd.read_csv(io.StringIO(plan))) == n


def test_run_all_nothing_runnable_names_the_listed_rows(no_envs, tmp_path, capsys):
    base = ["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path / "out")]
    rc, _, err = _cli(base, capsys)
    lines = err.splitlines()
    assert rc == 1 and lines[0] == "error: No method can run on D11 (vertical)."
    assert lines[-3] == NOTE[2:]
    assert lines[-2].startswith("multibench scan D11 --category vertical shows the listed "
                                "rows. ")
    rc, _, err = _cli(base + ["--modalities", "all"], capsys)
    assert rc == 1 and "not listed" not in err
    assert err.splitlines()[-2].startswith(
        "multibench scan D11 --category vertical --modalities all shows every row. ")
    assert not (tmp_path / "out").exists()


def test_run_all_batch_column_name(tmp_path, capsys, no_envs, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = ["run-all", "D11", "--category", "vertical", "--methods", "totalVI"]
    for given, col in (("donor", "donor"), ("obs:donor", "donor")):
        told = (f"error: --batch {given} is not a file. Pass a CSV file, as "
                f"adata.obs[[{col!r}]].to_csv(\"batch.csv\") writes in Python.")
        for extra in (["--dry-run"], ["--out-dir", str(tmp_path / "out")],
                      ["--dry-run", "--batch-column", "donor"]):
            rc, out, err = _cli(base + ["--batch", given, *extra], capsys)
            assert rc == 1 and out == "" and err.splitlines()[0] == told, extra
    assert not (tmp_path / "out").exists()
    # a name that looks like a file keeps the file error
    rc, _, err = _cli(base + ["--batch", "missing.csv", "--dry-run"], capsys)
    assert rc == 1 and err.startswith("error: The --batch file missing.csv does not exist.")
    # the file the message describes is taken
    cells = W._dataset_cell_ids("D11", None)
    pd.DataFrame({"donor": ["d1", "d2"] * (len(cells) // 2)}, index=cells).to_csv("batch.csv")
    rc, _, err = _cli(base + ["--batch", "batch.csv", "--dry-run"], capsys)
    assert rc == 0, err


# ------------------------------------------------------------------ convert
def _adata(n, tag, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n, 30)).astype(float))
    a.obs_names = [f"{tag}{i:04d}-1" for i in range(n)]
    a.var_names = [f"GENE{i}" for i in range(30)]
    a.obs["celltype"] = rng.choice(["T", "B", "NK"], n)
    return a


def test_convert_n_batches(tmp_path, capsys):
    out = tmp_path / "DONORS"
    for b in (1, 2, 3):
        _adata(40, f"EX{b}", seed=b).write_h5ad(tmp_path / f"ex{b}.h5ad")
    _adata(40, "ME", seed=9).write_h5ad(tmp_path / "me.h5ad")

    def convert(src, *extra):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _cli(["convert", str(tmp_path / src), str(out), "--rna", "X", "--labels",
                         "obs:celltype", "--category", "cross", *extra], capsys)
    for b in (1, 2, 3):
        assert convert(f"ex{b}.h5ad", "--batch-index", str(b), "--n-batches", "3")[0] == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert sorted(before) == ["cty1.csv", "cty2.csv", "cty3.csv", "rna1.h5", "rna2.h5",
                              "rna3.h5"]
    # the reader has two batches: batch 3 of the example is named by its first file, and
    # nothing is written
    rc, stdout, err = convert("me.h5ad", "--batch-index", "1", "--n-batches", "2",
                              "--overwrite")
    assert rc == 1 and stdout == "" and err.splitlines()[0] == (
        f"error: {out} holds rna3.h5 and 1 more file of batch 3, above --n-batches 2. "
        "Convert into a new folder.")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    # without --n-batches the call passes, as in 0.3.4
    rc, stdout, _ = convert("me.h5ad", "--batch-index", "1", "--overwrite")
    assert rc == 0 and stdout.startswith(f"wrote dataset folder {out} (files: cty1.csv, "
                                         "rna1.h5)")
    # the option goes with --batch-index
    new = tmp_path / "NEW"
    for extra, told in (
            (["--n-batches", "2"],
             "error: --n-batches goes with --batch-index only. Leave --n-batches out."),
            (["--batch-index", "3", "--n-batches", "2"],
             "error: --batch-index 3 is larger than --n-batches 2."),
            (["--batch-index", "1", "--n-batches", "0"],
             "error: --n-batches must be a positive integer, got 0")):
        rc, _, err = _cli(["convert", str(tmp_path / "me.h5ad"), str(new), "--rna", "X",
                           "--category", "cross", *extra], capsys)
        assert rc == 1 and err.splitlines()[0] == told
    assert not new.exists()
    # it selects the dataset export, and takes a whole number
    with pytest.raises(SystemExit) as e:
        cli.main(["convert", str(tmp_path / "me.h5ad"), str(new), "--n-batches", "2"])
    assert e.value.code == 2
    assert "dataset export needs at least one of --rna" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main(["convert", str(tmp_path / "me.h5ad"), str(new), "--rna", "X",
                  "--n-batches", "two"])
    assert e.value.code == 2
    capsys.readouterr()


def test_convert_help_says_which_file_takes_mod_col():
    """'<mod>:<col>' next to 'the global obs' read as a spelling for an
    .h5ad file, where every string is the name of an obs column."""
    assert _help("convert", "--labels") == (
        "mode 2: cell-type column: obs:<col>, or <mod>:<col> for a modality of a .h5mu "
        "file. Written to cty.csv, for diagonal to rna_cty.csv and atac_cty.csv")
    folded = " ".join(_sub("convert").format_help().split())
    assert "<mod>:<col> for a modality of a .h5mu file" in folded


def test_convert_refuses_a_10x_file(tmp_path, capsys):
    import scipy.sparse as sp
    src = tmp_path / "filtered_feature_bc_matrix.h5"
    m = sp.random(20, 12, density=0.3, format="csc", random_state=0)
    with h5py.File(src, "w") as f:                   # as cellranger writes it
        g = f.create_group("matrix")
        g.create_dataset("data", data=(m.data * 10).astype("int32"))
        g.create_dataset("indices", data=m.indices)
        g.create_dataset("indptr", data=m.indptr)
        g.create_dataset("shape", data=np.array(m.shape, dtype="int32"))
        g.create_dataset("barcodes", data=np.array([f"c{i}-1" for i in range(12)], dtype="S"))
        g.create_group("features").create_dataset(
            "name", data=np.array([f"GENE{i}" for i in range(20)], dtype="S"))
    told = (f'error: {src} is a 10x Genomics file. See the recipe "A 10x .h5 file" in the '
            'docs.')
    out = tmp_path / "OUT"
    for argv in (["convert", str(src), f"{out}/", "--modality", "rna"],
                 ["convert", str(src), str(out / "rna.h5")],
                 ["convert", str(src), str(out), "--rna", "X"]):
        rc, stdout, err = _cli(argv, capsys)
        assert rc == 1 and stdout == "" and err.splitlines()[0] == told, argv
        assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [src.name]
