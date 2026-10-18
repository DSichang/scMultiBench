"""Round 6 of the student study, get started (R6-16).

The Quickstart CLI Details say that ``evaluate --method`` names the rows
(0.4.0: any name; ``--name`` is a hidden deprecated alias). The test runs the
own-method route, ``--method`` with one ``--labels`` per file, on a small
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
    bullet = ("`evaluate --method` names the rows. A package method also reads the "
              "`--dataset` label files in its cell order.")
    assert bullet in text
    assert "`--name`" not in text
    # the route itself: --method with your own name, two label files in the embedding's order
    d, emb = _diag_folder(tmp_path / "data")
    np.save(tmp_path / "emb.npy", emb)
    out = tmp_path / "mine.csv"
    argv = ["evaluate", "--embedding", str(tmp_path / "emb.npy"), "--method", "RNA+ATAC PCA",
            "--labels", str(d / "rna_cty.csv"), "--labels", str(d / "atac_cty.csv"),
            "--dataset", "MYDATA", "--category", "diagonal", "--metrics", "ASW,iLISI",
            "--out", str(out)]
    assert cli.main(argv) == 0
    df = pd.read_csv(out)
    assert set(df["method"]) == {"RNA+ATAC PCA"} and set(df["dataset"]) == {"MYDATA"}
    assert set(df["metric"]) == {"ASW", "iLISI"}
