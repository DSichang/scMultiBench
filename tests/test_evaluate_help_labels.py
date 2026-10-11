"""``multibench evaluate --help`` says that --labels can name an obs column.

With an ``.h5ad`` --output, ``--labels celltype`` and ``--labels obs:celltype``
read the column of its obs. The help named files only. The forms that do
not work stay errors: a column next to ``--column``, a column given twice,
and a column with an output that has no obs.
"""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench import cli

ad = pytest.importorskip("anndata")

N = 60
TYPES = np.array(["T", "B", "NK"] * (N // 3))
EMB = np.random.default_rng(0).normal(size=(N, 5)) + pd.Categorical(TYPES).codes[:, None] * 4.0


def _help(capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        cli.main(["evaluate", "--help"])
    assert e.value.code == 0
    # argparse wraps at the width of the terminal, and inside --batch-column
    text = re.sub(r"(?<=\w)-\n\s+", "-", capsys.readouterr().out)
    return re.sub(r"\s+", " ", text)


def test_the_help_names_the_obs_column(capsys):
    text = _help(capsys)
    assert ("With an .h5ad --output, --labels can instead name a column of its obs, as "
            "celltype or obs:celltype: give it once, without --column.") in text
    assert ("--batch BATCH per-cell batch labels CSV; a first column of barcodes is "
            "aligned as for --labels. With an .h5ad --output it can instead name a column "
            "of its obs, without --batch-column.") in text
    assert ("A first column of barcodes is aligned to the obs_names of an .h5ad --output, "
            "or to the barcodes in the first column of a .csv --output.") in text
    assert (".csv/.tsv (a first column of barcodes gives the cell ids: labels and batches "
            "with barcodes are matched by them)") in text
    assert "is dropped" not in text


@pytest.fixture
def out(tmp_path):
    a = ad.AnnData(np.ones((N, 3)))
    a.obs_names = [f"cell{i}-1" for i in range(N)]
    a.obsm["X_emb"] = EMB
    a.obs["celltype"] = TYPES
    a.obs["donor"] = ["d1", "d2"] * (N // 2)
    a.write_h5ad(tmp_path / "out.h5ad")
    np.save(tmp_path / "emb.npy", EMB)
    return tmp_path


def _run(argv, capsys):
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(["evaluate", *argv])
    got = capsys.readouterr()
    return rc, got.out, got.err


@pytest.mark.parametrize("labels", ["celltype", "obs:celltype"])
def test_what_the_help_says_works(out, capsys, labels):
    scores = out / "scores.csv"
    rc, _, err = _run(["--output", str(out / "out.h5ad"), "--labels", labels, "--batch",
                       labels.replace("celltype", "donor"), "--metrics", "ARI,NMI,iLISI",
                       "--out", str(scores)], capsys)
    assert rc == 0, err
    got = pd.read_csv(scores, index_col=0)["Value"]
    assert got["ARI"] == got["NMI"] == 1.0 and 0.0 <= got["iLISI"] <= 1.0


def test_the_forms_the_help_rules_out_are_errors(out, capsys):
    h5ad = str(out / "out.h5ad")
    for argv in (["--output", h5ad, "--labels", "celltype", "--column", "x"],
                 ["--output", h5ad, "--labels", "celltype", "--labels", "donor"],
                 ["--output", str(out / "emb.npy"), "--labels", "celltype"],
                 ["--output", h5ad, "--labels", "celltype", "--batch", "donor",
                  "--batch-column", "x"]):
        rc, _, err = _run([*argv, "--metrics", "ARI"], capsys)
        assert rc == 1 and "error: " in err
