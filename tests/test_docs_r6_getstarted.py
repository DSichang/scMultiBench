"""Round 6 of the student study, get started (R6-16).

The Quickstart CLI Details tell a user with their own embedding to give only
``--name`` and one ``--labels`` per file. The test runs that route on a small
diagonal folder.
"""
import numpy as np
import pandas as pd

from multibench import cli
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_f4_cli import _diag_folder


@needs_docs
def test_quickstart_cli_details_give_the_own_method_route(tmp_path):
    text = _flat("quickstart.md").split("### From the command line", 1)[1]
    bullet = ("`evaluate --method` sets the label order. `--name` sets the row name. "
              "For your own method, give only `--name` and repeat `--labels` once per "
              "file, in your embedding's cell order.")
    assert bullet in text
    assert "`--method` is then only the row name" not in text
    # the route itself: --name, no --method, two label files in the embedding's order
    d, emb = _diag_folder(tmp_path / "data")
    np.save(tmp_path / "emb.npy", emb)
    out = tmp_path / "mine.csv"
    argv = ["evaluate", "--output", str(tmp_path / "emb.npy"), "--name", "RNA+ATAC PCA",
            "--labels", str(d / "rna_cty.csv"), "--labels", str(d / "atac_cty.csv"),
            "--dataset", "MYDATA", "--category", "diagonal", "--metrics", "ASW,iLISI",
            "--out", str(out)]
    assert cli.main(argv) == 0
    df = pd.read_csv(out)
    assert set(df["method"]) == {"RNA+ATAC PCA"} and set(df["dataset"]) == {"MYDATA"}
    assert set(df["metric"]) == {"ASW", "iLISI"}
