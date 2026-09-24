"""Round 5 of the student study: what the docs pages now say.

Get started (R5-15): the Quickstart CLI block lets ``multibench evaluate``
read the D28 label files in SCALEX's cell order (``--method SCALEX``) and
names the rows with ``--name SCALEX_rerun``. The block's evaluate line is run
as printed, on a small D28 folder.

Guides (R5-17, R5-02): the comparability sentence on tutorials/evaluate is
plain, and the page says that a CSV whose first column holds the cell ids is
aligned by id.

API overview and Changes (R5-16) and the code font (R5-18): exit code ``3``
follows the CLI rule, the Changes page lists the round-5 changes and quotes
the live stderr line, and code is drawn without ligatures.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it.
"""
import re
import shlex

import h5py
import numpy as np
import pandas as pd

from multibench import cli, config
from tests.test_docs_r4 import _docs_root, _flat, _read, _visible, needs_docs
from tests.test_f4_cli import _diag_folder


# ---- get started (R5-15) -------------------------------------------------

def _cli_block():
    text = _read("quickstart.md").split("### From the command line", 1)[1]
    return text.split('```bash title="terminal"', 1)[1].split("```", 1)[0]


@needs_docs
def test_quickstart_cli_block_reads_the_labels_in_the_method_order(tmp_path, monkeypatch,
                                                                   capsys):
    block = _cli_block()
    assert "--labels" not in block and "<data_path>" not in block
    cmds = [" ".join(c.split()) for c in block.replace("\\\n", " ").splitlines()]
    line = next(c for c in cmds if c.startswith("multibench evaluate "))
    assert "--method SCALEX --name SCALEX_rerun --dataset D28 --category diagonal" in line
    # run the printed line on a small D28 folder (RNA cells first, as SCALEX)
    d, emb = _diag_folder(tmp_path / "data")
    monkeypatch.setattr(config.DEFAULT, "data_path", tmp_path / "data")
    argv = shlex.split(line)[1:]
    out_h5, out_csv = tmp_path / "embedding.h5", tmp_path / "mine.csv"
    with h5py.File(out_h5, "w") as f:
        f["data"] = emb
    argv[argv.index("--output") + 1] = str(out_h5)
    argv[argv.index("--out") + 1] = str(out_csv)
    assert cli.main(argv) == 0
    err = capsys.readouterr().err
    assert "# labels: rna_cty.csv, atac_cty.csv from D28, in SCALEX's cell order" in err
    df = pd.read_csv(out_csv)
    assert set(df["method"]) == {"SCALEX_rerun"} and set(df["dataset"]) == {"D28"}
    assert np.isfinite(df.set_index("metric").loc["ARI", "value"])


# ---- guides (R5-17, R5-02) -----------------------------------------------

@needs_docs
def test_evaluate_page_comparability_sentence_is_plain():
    visible = " ".join(_visible(_read("tutorials/evaluate.md")).split())
    assert "compare ranks, not decimals" not in visible
    assert "Most methods' scores vary slightly between runs. Compare ranks." in visible


@needs_docs
def test_evaluate_page_says_a_barcode_csv_is_aligned_by_id():
    text = _flat("tutorials/evaluate.md")
    assert "A CSV whose first column holds the cell ids is aligned the same way." in text
    assert "Plain arrays, lists and label files are always matched by position." not in text


# ---- API overview, Changes, code font (R5-16, R5-18) ---------------------

@needs_docs
def test_api_page_exit_code_3_follows_the_cli_rule():
    api = _flat("api.md")
    assert "failed or was skipped" not in api
    assert ("`3` when `run-all` finished but a method failed, or a method named in "
            "`--methods` was skipped.") in api
    assert "A skipped method that `--methods` did not name does not set `3`" in api
    # the CLI says the same (R5-03)
    assert "a method named in ``--methods`` was skipped" in " ".join(cli.__doc__.split())


@needs_docs
def test_changes_page_lists_round_5_changes():
    text = _flat("changes.md")
    assert "failed or was skipped" not in text
    for phrase in ("`run-all` exits with `3` after saving when a method failed, or a "
                   "method named in `--methods` was skipped.",
                   "`# 1 failed (Seurat_WNN), 1 skipped (totalVI).`",
                   "`# Not run: ...`",
                   "end with a `caveat` column and a `reason` column.",
                   "`BatchResult.rescore(labels=...)` aligns a Series",
                   "A batch or labels CSV whose first column holds cell ids is aligned",
                   "`mtb.data.fetch` and `fetch_outputs` raise `OSError`",
                   "`multibench evaluate --name NAME`",
                   "`<method> needs an NVIDIA GPU, and this computer has none.`",
                   "`mosaic has no published tables.`"):
        assert phrase in text, phrase
    assert "needs an NVIDIA GPU; this computer has none" not in text


@needs_docs
def test_changes_page_quotes_the_live_stderr_lines():
    bad = pd.DataFrame({"method": ["Seurat_WNN", "totalVI"], "status": ["FAIL", "SKIPPED"]})
    res = type("R", (), {"failures": bad, "records": []})()
    line = cli._failed_line(res, "out/failures.csv")
    shown = "# 1 failed (Seurat_WNN), 1 skipped (totalVI)."
    assert line.startswith(shown) and f"`{shown}`" in _flat("changes.md")


@needs_docs
def test_code_font_draws_no_ligatures():
    css = (_docs_root() / "stylesheets" / "extra.css").read_text()
    rule = re.search(r"\.md-typeset code, \.md-typeset pre, \.md-typeset kbd \{([^}]*)\}", css)
    assert rule, "the code-font rule of extra.css moved"
    body = rule.group(1)
    assert "font-variant-ligatures: none" in body
    assert re.search(r'"calt" 0', body) and re.search(r'"liga" 0', body)
    # no other rule turns them back on
    assert not re.search(r'font-feature-settings:[^;]*"(calt|liga)"(?! 0)', css)
