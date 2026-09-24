"""Changes page, round 7 (R7-15).

The 0.3.2 lists name this round's changes (R7-02, R7-03, R7-04, R7-06, R7-07 and the
reworded messages of R7-08 to R7-11), and the texts the page quotes are what the
package prints.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is skipped without it.
"""
import pytest

import multibench as mtb
from multibench.plot.bubble import CHIP_KEY
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_f6_cli import _DATA, _lung


@needs_docs
def test_changes_page_lists_round_7_changes():
    text = _flat("changes.md")
    for phrase in (
            # R7-07: the Python dry run prints the scripts-ref note only with verbose,
            # as a [run_all] line on stdout
            "`run_all(dry_run=True)` prints the note only with `verbose=True`, as a "
            "`[run_all]` line on stdout after its count line.",
            # R7-06, in its own bullet: it holds only when rescore ranks
            "- `BatchResult.rescore(verbose=True)` prints a line before each label-order "
            "ranking.",
            # R7-02
            "The last line of the `No method can run` error repeats the data path, "
            "modalities and other scan options of the call.",
            "So does the list header of the `run_all` error.",
            # R7-08 to R7-11, folded into one bullet by R8-10 and its review
            "- Reworded: the messages of `export_dataset`, `to_canonical`, `inputs_for`, "
            "`params_for` and `evaluate`. Also the environment messages, file checks, "
            "plot and scan warnings,"):
        assert phrase in text, phrase


@needs_docs
def test_changes_page_quotes_the_live_chip_key():
    shown = "? = a name the package does not know"
    assert shown in CHIP_KEY
    assert f"The chip key of `bubble` reads `{shown}`." in _flat("changes.md")


@needs_docs
@pytest.mark.parametrize("method", ["MultiMAP", "Seurat_v3"])
def test_changes_page_quotes_the_live_both_files_sentence(tmp_path, method):
    d = _lung(tmp_path)
    for f in ("rna_cty.csv", "atac_cty.csv"):
        (d / f).write_bytes((_DATA / "D28" / f).read_bytes())
    with pytest.raises(FileNotFoundError) as exc:
        mtb.inputs_for("LUNG", "diagonal", method, data_path=str(tmp_path / "data"),
                       check=True)
    assert str(exc.value).endswith(f"{method} reads both atac_peak.h5 and atac_gas.h5.")
    assert ("For MultiMAP and Seurat_v3, the text ends "
            "`<method> reads both atac_peak.h5 and atac_gas.h5.`") in _flat("changes.md")
