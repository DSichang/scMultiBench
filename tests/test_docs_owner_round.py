"""The pages after the owner's decisions on 0.3.5: the Changes entries for
the CSV barcodes, the 10x type names, the three new warnings and the help
text, and the donor sentence of the Quickstart tab.

The sentences quoted from messages are compared with what the package says.
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import ingest
from multibench.eval import io
from multibench.eval import scib as escib

_ROOT = os.environ.get("SCMULTIBENCH_DOCS")
DOCS = Path(_ROOT) if _ROOT and Path(_ROOT).is_dir() else None
pytestmark = pytest.mark.skipif(DOCS is None, reason="SCMULTIBENCH_DOCS not set")


def _flat(page):
    return " ".join((DOCS / page).read_text().split())


def _changes(head):
    text = (DOCS / "changes.md").read_text().split("## Behaviour changes in 0.3.5", 1)[1]
    text = text.split("\n## ", 1)[0]
    return " ".join(text.split(f"### {head}\n", 1)[1].split("\n### ", 1)[0].split())


def test_changes_say_how_a_csv_embedding_is_matched(tmp_path, capsys, monkeypatch):
    said = _changes("Cell order in evaluate")
    for sentence in (
            "`mtb.evaluate` and `multibench evaluate` keep the barcodes of an embedding in "
            "a `.csv` or `.tsv` file whose first column holds them.",
            "0.3.4 dropped that column and matched by position, with a warning that the "
            "output has no cell ids.",
            "When the two do not name the same cells, `ValueError` gives the number of "
            "cells of the embedding that the other file lacks, and one of them.",
            "The first column of the embedding holds barcodes when it holds text: at least "
            "one of its values is not a number.",
            "A first column of numbers is read as in 0.3.4. Without a header it is left "
            "out as row numbers, and with a header it is a column of the embedding.",
            "The scores of `run_all` and `rescore` do not change."):
        assert sentence in said, sentence
    # the line the page quotes is the line evaluate prints
    line = re.search(r"such as `(evaluate: [^`]+)`", said).group(1)
    monkeypatch.setattr(escib, "compute", lambda *a, **k: pd.DataFrame(
        {"Value": [1.0]}, index=pd.Index(["ARI"], name="metric")))
    bars = [f"AAAC{i:04d}-1" for i in range(30)]
    types = ["T", "B", "NK"] * 10
    pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)), index=bars).to_csv(
        tmp_path / "emb.csv")
    pd.DataFrame({"celltype": types}, index=bars).iloc[::-1].to_csv(tmp_path / "labels.csv")
    capsys.readouterr()
    mtb.evaluate(str(tmp_path / "emb.csv"), labels=str(tmp_path / "labels.csv"),
                 metrics=["ARI"])
    assert capsys.readouterr().err == line + "\n"


def test_changes_list_the_new_messages():
    said = _changes("Messages")
    names = sorted(set(ingest._TENX_TYPE.values()))
    assert names == ["Antibody Capture", "Gene Expression", "Peaks"]
    assert ("The error for a matrix with several feature types no longer ends with the "
            "advice to delete `var[\"feature_types\"]`. With the 10x names `Gene Expression`, "
            "`Antibody Capture` or `Peaks`, following that advice wrote genes and peaks, or "
            "genes and antibodies, into one file.") in said
    assert ("`mtb.io.export_dataset` and `multibench convert` warn when a matrix has a "
            "feature name more than once, as the genes of a 10x file can. The names are "
            "written as they are, as in 0.3.4, which wrote them without a message.") in said
    assert "[A 10x `.h5` file](tutorials/run.md#recipe-10x-h5)" in said
    # the two signs are those of the rule
    assert ("With 20 cells or more, there are more distinct labels than half of the cells. "
            "Or the labels are numbers and some are not whole numbers.") in said
    assert io._labels_look_measured(np.arange(20)) == ("distinct", 20, 20)
    assert io._labels_look_measured(np.arange(19)) is None
    assert io._labels_look_measured([0] * 10 + list(range(10))) is None      # half
    assert io._labels_look_measured(np.array([0.5, 1.0])) == ("fraction", 2, 2)
    assert ("`export_dataset`, `evaluate` and `run_all` warn when `batch=` is a Series whose "
            "index is the row numbers `0` to `n-1` while the cells have barcodes.") in said
    assert ("`multibench evaluate --help` says that `--labels` and `--batch` can name a "
            "column of the `obs` of an `.h5ad` output, as `celltype` or `obs:celltype`, and "
            "that the barcodes of a `.csv` output are kept.") in said


def test_changes_say_what_the_new_warnings_leave_open(tmp_path):
    said = _changes("Not solved in 0.3.5")
    assert ("`mtb.io.to_canonical`, `multibench convert --modality` and the conversion of "
            "an `.h5ad` input that `mtb.run` does write a feature name that occurs more "
            "than once without a message of their own.") in said
    assert ("A `labels=` Series whose index is the row numbers `0` to `n-1` is matched by "
            "position without a message. Only `batch=` warns.") in said
    ad = pytest.importorskip("anndata")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.AnnData(np.ones((20, 3)))
        a.var_names = ["G1", "G2", "G2"]
    a.obs_names = [f"cell{i}-1" for i in range(20)]
    a.obs["donor"] = ["a", "b"] * 10
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ingest.to_canonical(a, tmp_path / "rna.h5", modality="rna")
        a.var_names_make_unique()
        ingest.export_dataset(a, tmp_path / "OUT", rna="X", category="cross",
                              labels=pd.Series(["T", "B"] * 10), batch="obs:donor")
    assert [str(w.message) for w in rec
            if "feature name" in str(w.message) or "Series" in str(w.message)] == []


def test_the_quickstart_tab_names_batch_for_donors():
    page = _flat("quickstart.md")
    tab = page.split('=== "Linux, environments installed"', 1)[1].split("10x Multiome,", 1)[0]
    assert ("With several donors or samples in the file, add `batch=adata.obs[\"donor\"]` to "
            "`run_all`. The summary then has the batch metrics. The recipe "
            "[Another number of donors](tutorials/run.md#recipe-donors) has the steps.") in tab
    # the recipe the bullet and the tab point at says the same
    recipe = _flat("tutorials/run.md").split("{ #recipe-donors }", 1)[1]
    assert "`batch=` takes the column itself, not its name." in recipe
    assert 'batch=adata.obs["donor"])' in recipe
    assert "The summary then has the batch metrics" in recipe
    assert ("A cross method reads a fixed number of batches: three for RNA + ADT") in recipe
    assert ("With another number of donors, treat the data as a vertical task") in recipe
