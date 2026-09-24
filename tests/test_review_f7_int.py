"""Review of round 7 (integration): the findings applied in the package.

- R7-10: on Linux, scan's summary line names the doctor check when an
  environment is missing; the category sentence of the env reason starts
  with 'Use'.
- R7-11: the reason column drops '<method> reads <file> of <dataset>, which'.
- A missing batch file is called a batch file, in sentences.
- describe_layout('diagonal') points to its own list, not method_info.
- The several-datasets warning of the plots lists its datasets in words.
- The no-conda install error, the modalities warning of scan, the missing
  feature-names warning and the several-variants errors are sentences.
- Docstrings: BatchResult.long, method_info, bubble's L badge,
  normalize_peak_names' example.
"""
import inspect
import re
import shutil
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry
from multibench.engine import runner as R
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_study_r7_msg import ALL_ENVS, GENES, _h5, _lung, _quiet

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _doc(obj) -> str:
    return " ".join(inspect.getdoc(obj).split())


# ======================================================================= R7-10
def test_scan_summary_names_doctor_on_linux_when_an_environment_is_missing(monkeypatch,
                                                                            capsys):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: None)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    mtb.scan("D11", "vertical")
    line = capsys.readouterr().out.strip()
    assert line.startswith("[scan] ") and line.endswith(
        " have their environment installed. mtb.env.doctor() checks the environments."), line
    monkeypatch.setattr(config, "_CLI", True)
    mtb.scan("D11", "vertical")
    assert capsys.readouterr().out.strip().endswith(
        "multibench env doctor checks the environments.")
    # every environment installed: nothing to check
    monkeypatch.setattr(config, "_CLI", False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    mtb.scan("D11", "vertical")
    assert "doctor" not in capsys.readouterr().out


def test_env_reason_names_the_category_in_a_sentence(monkeypatch):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: None)
    text = W._env_hint("glue", "GLUE", "diagonal")
    assert text == ("Environment glue is not installed. Run multibench env install "
                    "--methods GLUE --packed --run. Use --category diagonal to install the "
                    "environments of every diagonal method.")
    assert not re.search(r"\. --category", text)


# ======================================================================= R7-11
def test_reason_column_starts_with_the_file_of_a_cell_check(tmp_path):
    d = _lung(tmp_path)
    _h5(d / "atac_gas.h5", GENES, [f"a{i}" for i in reversed(range(20))])
    row = _quiet(mtb.scan, "LUNG", "diagonal", methods=["SCALEX"], data_path=tmp_path,
                 verbose=False).iloc[0]
    assert row["reason"].startswith("atac_gas.h5 lists the ATAC cells in another order "
                                    "than atac_peak.h5."), row["reason"]
    assert "than atac_peak.h5" in row["reason"][:80]
    assert "SCALEX reads" not in row["reason"]
    # files_reason keeps the exception verbatim
    assert row["files_reason"].startswith("ValueError: SCALEX reads atac_gas.h5 of LUNG, "
                                          "which lists")


def test_reason_column_of_the_other_cells_and_orientation_checks():
    other = ("ValueError: SCALEX reads atac_gas.h5 of LUNG, which holds other cells than "
             "atac_peak.h5. The files hold 18 and 20 cells and share 0.")
    assert W._short_reason(other, "SCALEX", "LUNG", "diagonal").startswith(
        "atac_gas.h5 holds other cells than atac_peak.h5.")
    turned = ("ValueError: totalVI reads rna.h5 of TRANS, which stores matrix/data as "
              "cells x features, shape (20, 60).")
    assert W._short_reason(turned, "totalVI", "TRANS", "vertical") == (
        "rna.h5 stores matrix/data as cells x features, shape (20, 60).")
    # another method or dataset is left alone
    assert W._short_reason(other, "GLUE", "LUNG", "diagonal").startswith("SCALEX reads")


# ================================================================ batch file
def test_missing_batch_file_is_named_as_a_batch_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    emb = np.random.default_rng(0).normal(size=(20, 3))
    labels = ["A", "B"] * 10
    with pytest.raises(FileNotFoundError) as e:
        mtb.evaluate(emb, labels=labels, batch="missing.csv", metrics=["iLISI"])
    msg = str(e.value)
    assert msg == (f"The batch file missing.csv does not exist. From the working directory "
                   f"{tmp_path.resolve()}, the path resolves to "
                   f"{(tmp_path / 'missing.csv').resolve()}.")
    assert "labels file" not in msg and ";" not in msg and "(cwd" not in msg


def test_missing_batch_file_on_the_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    src = ROOT / "data" / "outputs" / "D28"
    out = tmp_path / "out"
    shutil.copytree(src / "scalex_D28" if (src / "scalex_D28").is_dir() else
                    next(p for p in src.iterdir() if p.is_dir()), out)
    rc = _quiet(cli.main, ["evaluate", "--output", str(out / "embedding.h5"),
                           "--method", "SCALEX", "--dataset", "D28", "--category",
                           "diagonal", "--batch", "missing.csv", "--metrics", "iLISI"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error: The --batch file missing.csv does not exist." in err, err
    assert "labels file" not in err


# ============================================================ describe_layout
def test_diagonal_layout_points_to_its_own_list(monkeypatch):
    text = mtb.describe_layout("diagonal")
    assert "method_info(m)['atac']" not in text
    assert "mtb.run only warns, so check the list above first." in text
    assert "need both files:       MultiMAP, Seurat_v3" in text
    monkeypatch.setattr(config, "_CLI", True)
    cli_text = mtb.describe_layout("diagonal")
    assert "multibench run only warns, so check the list above first." in cli_text
    assert "multibench info METHOD" not in cli_text
    # vertical has no method that reads two files: the pointer is right there
    monkeypatch.setattr(config, "_CLI", False)
    assert ("mtb.run only warns, so check method_info(m)['atac'] first."
            in mtb.describe_layout("vertical"))


# ====================================================== several-datasets warning
def test_several_datasets_warning_lists_the_datasets_in_words():
    df = mtb.load_results("diagonal", dataset=["D24", "D28"])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mtb.plot.build_table(df, metrics=["ARI", "NMI"])
    msgs = [str(w.message) for w in rec if "averages each method" in str(w.message)]
    assert len(msgs) == 1, [str(w.message) for w in rec]
    assert msgs[0].startswith("This figure averages each method over 2 datasets, D24 and "
                              "D28. Its rows mix datasets. ")
    assert "(D24" not in msgs[0] and ", so" not in msgs[0]


# ============================================================ no conda
@pytest.fixture
def linux_without_conda(monkeypatch):
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    monkeypatch.setattr(envs, "doctor", lambda **kw: [
        {"env": envs.group_for("StabMap"), "exists": False}])
    monkeypatch.setattr(envs, "install_packed",
                        lambda env, **kw: pytest.fail(f"download of {env} started"))


def test_no_conda_error_names_the_flag_on_the_cli(linux_without_conda, capsys):
    env = envs.group_for("StabMap")
    assert env in envs.packed_manifest()
    rc = cli.main(["env", "install", "--methods", "StabMap", "--run"])
    err = capsys.readouterr().err
    assert rc == 1
    assert (f"error: Conda is not installed on this computer. Environment {env} has a "
            f"packed archive. Pass --packed to install it.") in err, err
    assert ";" not in err and "packed=True" not in err


def test_no_conda_error_in_python(linux_without_conda):
    env = envs.group_for("StabMap")
    with pytest.raises(RuntimeError) as e:
        mtb.env.install(["StabMap"], packed=False, dry_run=False)
    assert str(e.value) == (f"Conda is not installed on this computer. Environment {env} "
                            f"has a packed archive. Pass packed=True to install it.")
    assert str(e.value).startswith(envs.NO_CONDA)


# ==================================================== scan's modalities warning
def _lung_data(root):
    d = _lung(root / "data")
    return d


def test_modalities_warning_on_the_cli_is_sentences(tmp_path, monkeypatch, capsys):
    _lung_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["scan", "LUNG", "--category", "diagonal", "--data-path", "data",
                   "--modalities", "rna,atac_gas"])
    err = capsys.readouterr().err
    assert rc == 0
    line = next(l for l in err.splitlines() if "scBridge" in l)
    assert line.endswith("The modalities rna and atac_gas leave out scBridge, which reads "
                         "a folder instead of modality files. Pass --modalities \"\" to "
                         "select it, or leave out --modalities to see every variant."), line
    assert "modalities=[" not in line and ";" not in line and "scan:" not in line


def test_modalities_warning_in_python_is_sentences(tmp_path):
    _lung_data(tmp_path)
    with pytest.warns(UserWarning) as rec:
        mtb.scan("LUNG", "diagonal", data_path=tmp_path / "data",
                 modalities=["rna", "atac_gas"], verbose=False)
    msg = next(str(w.message) for w in rec if "scBridge" in str(w.message))
    assert msg == ("The modalities rna and atac_gas leave out scBridge, which reads a "
                   "folder instead of modality files. Pass modalities=[] to select it, or "
                   "leave out modalities= to see every variant.")


# ==================================================== missing feature names
def _citeseq():
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(30, 40)).astype(float))
    a.obs_names = [f"c{i}" for i in range(30)]
    a.var_names = [f"G{i}" for i in range(40)]
    a.obs["cell_type"] = ["T", "B", "NK"] * 10
    a.obsm["protein"] = rng.poisson(3.0, size=(30, 20)).astype(float)
    return a


def _names_warning(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*args, **kw)
    return [str(w.message) for w in rec if "feature names" in str(w.message)]


@pytest.mark.parametrize("cli_mode", [False, True])
def test_missing_feature_names_warning_is_sentences(tmp_path, monkeypatch, cli_mode):
    monkeypatch.setattr(config, "_CLI", cli_mode)
    msgs = _names_warning(ingest.export_dataset, _citeseq(), tmp_path / "MYCITE",
                          rna="X", adt="obsm:protein", labels="obs:cell_type")
    assert len(msgs) == 1, msgs
    msg = msgs[0]
    assert msg.startswith("The adt matrix obsm['protein'] has no feature names. The file "
                          "names them feature_0 to feature_19. ")
    for bad in (";", " - ", "e.g.", "..feature"):
        assert bad not in msg, bad
    assert msg.endswith(".")
    assert ("adt_names=" in msg) is (not cli_mode)
    assert "adata.uns['protein_names']" in msg


def test_missing_feature_names_of_to_canonical_name_its_argument(tmp_path):
    msgs = _names_warning(ingest.to_canonical, _citeseq(), tmp_path / "adt.h5",
                          modality="adt", obsm="protein")
    assert len(msgs) == 1 and "Pass feature_names=, store obsm['protein'] as a " \
        "DataFrame" in msgs[0], msgs


def test_missing_feature_names_of_a_bare_array(tmp_path):
    a = _citeseq()
    msgs = _names_warning(ingest.export_dataset, a, tmp_path / "ARR", rna="X",
                          adt=np.ones((30, 4)), labels="obs:cell_type")
    assert msgs == ["The adt matrix array has no feature names. The file names them "
                    "feature_0 to feature_3. Pass adt_names=, or give adt= as a DataFrame "
                    "or AnnData with named features."]


# ==================================================== several variants
def test_several_variants_errors_are_sentences(tmp_path, monkeypatch):
    d = tmp_path / "AMB"
    d.mkdir()
    for f in ("rna.h5", "adt.h5", "atac.h5", "cty.csv"):
        (d / f).touch()
    msgs = []
    for cli_mode in (False, True):
        monkeypatch.setattr(config, "_CLI", cli_mode)
        with pytest.raises(mtb.AmbiguousVariantError) as e:
            mtb.inputs_for("AMB", "vertical", "Matilda", data_path=tmp_path)
        msgs.append(str(e.value))
    assert msgs[0] == (f"Matilda has 2 vertical variants, rna+adt and rna+atac. The folder "
                       f"{d} has every input file of both. Pass modalities=['rna', 'adt'] "
                       f"or modalities=['rna', 'atac'].")
    assert msgs[1].endswith("Pass --modalities rna,adt or --modalities rna,atac.")
    monkeypatch.setattr(config, "_CLI", False)
    with pytest.raises(mtb.AmbiguousVariantError) as e:
        mtb.params_for("Matilda")
    msgs.append(str(e.value))
    for msg in msgs:
        for bad in (";", " - ", "e.g.", "[[", "available:"):
            assert bad not in msg, (bad, msg)
        assert msg.endswith(".")


# ==================================================== docstrings
def test_long_names_when_scored_with_is_there():
    res = mtb.load_batch(ROOT / "data" / "outputs" / "D11")
    assert list(res.long.columns) == ["metric", "value", "method", "dataset", "category",
                                      "clustering", "source"]
    doc = _doc(W.BatchResult.long)
    assert "When the scores carry ``scored_with``, so does the table." in doc
    assert ("Neither that dict nor the folders that ``mtb.data.fetch_outputs`` downloads "
            "carry ``scored_with``.") in doc
    assert "for rows ``run_all`` scored" not in doc


def test_method_info_returns_and_raises_read_as_a_card():
    doc = inspect.getdoc(mtb.method_info)
    returns = doc.split("Returns\n-------\n", 1)[1].split("\n\n", 1)[0]
    raises = doc.split("Raises\n------\n", 1)[1].split("\n\n", 1)[0]
    assert "(" not in returns and ";" not in returns
    assert "declared stub" not in raises
    flat = " ".join(doc.split())
    assert "A method listed but not yet runnable raises ``KeyError``." in flat
    assert "``params`` - what ``run(params=...)`` can change" in flat


def test_bubble_badge_note_names_the_needs_labels_column():
    doc = inspect.getdoc(mtb.plot.bubble)
    flat = " ".join(doc.split("**Chips and badge**", 1)[1].split())
    assert "(supervised)" not in flat and "method-level flag" not in flat
    assert "A ``needs_labels`` column decides first." in flat
    assert "scMoMaT has it in mosaic only." in flat


def test_normalize_peak_names_example_follows_to_canonical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    atac = ad.AnnData(np.ones((6, 3)))
    atac.obs_names = [f"c{i}" for i in range(6)]
    atac.var_names = ["chr1_100_200", "chr1_300_400", "chr2_10_20"]
    written = _quiet(mtb.io.to_canonical, atac, "data/MYMULTI/", modality="peak",
                     category="vertical")
    assert str(written) == "data/MYMULTI/atac.h5"
    example = next(l for l in inspect.getdoc(mtb.io.normalize_peak_names).splitlines()
                   if "normalize_peak_names(" in l)
    src, dst = re.findall(r'"([^"]+)"', example)
    assert src == str(written)
    assert mtb.io.normalize_peak_names(src, dst).is_file()


# ==================================================== docs pages
@needs_docs
def test_run_page_names_the_methods_that_read_both_atac_files():
    text = _flat("tutorials/run.md")
    para = text.split("set `atac_kind=`", 1)[1].split("=== ", 1)[0]
    assert "Each method reads one of the two" not in para
    assert ("Most methods read one of the two, and `method_info(m)[\"atac\"]` names it. "
            "MultiMAP and Seurat_v3 read both.") in para
    assert all(m in para for m in ("MultiMAP", "Seurat_v3"))


@needs_docs
def test_installation_heading_quotes_the_no_conda_error():
    text = _flat("installation.md")
    assert (f"`RuntimeError: {envs.NO_CONDA} Environment <env> has a packed archive.`"
            in text)
    assert "no conda/mamba" not in text


@needs_docs
def test_changes_page_quotes_the_live_several_datasets_warning():
    df = mtb.load_results("diagonal", dataset=["D24", "D25", "D28"])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mtb.plot.build_table(df, metrics=["ARI", "NMI"])
    shown = ("This figure averages each method over 3 datasets, D24, D25 and D28. Its rows "
             "mix datasets.")
    assert any(str(w.message).startswith(shown + " ") for w in rec)
    assert f"`{shown}`" in _flat("changes.md")


@needs_docs
def test_changes_page_lists_the_review_changes():
    text = _flat("changes.md")
    for phrase in (
            "- A `bubble` figure of one to three metrics is widened so that its row labels "
            "and key fit. The height is unchanged.",
            "On Linux, it names `mtb.env.doctor()` when an environment is missing.",
            "- the error of `mtb.env.install` when conda is not installed",
            "- the several-variants errors of `inputs_for` and `params_for`",
            "- the missing-file errors of `evaluate`"):
        assert phrase in text, phrase
    assert "`verbose=True` prints a line before each ranking." not in text
    # the height of a narrow figure is that of the full one
    df = mtb.load_results("diagonal", dataset="D28")
    narrow = _quiet(mtb.plot.bubble, df, metrics=["ARI"]).get_size_inches()
    full = _quiet(mtb.plot.bubble, df).get_size_inches()
    assert narrow[1] == full[1] and narrow[0] < full[0]


@needs_docs
def test_plot_page_says_what_require_complete_does_where_it_is_asked_for():
    text = _flat("tutorials/plot.md")
    assert ("A summary compares methods scored on the same datasets. "
            "`require_complete=True` keeps only the methods scored on every dataset.") in text
    assert "Use `require_complete=True`." not in text
    assert ("They cannot be rebuilt. For a summary with your own method, use the full "
            "datasets.") in text
    # said once on the page
    assert text.count("keeps only the methods") == 1
