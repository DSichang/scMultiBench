"""Round 6 of the student study: the Changes page (R6-17).

The 0.3.2 lists name this round's changes (R6-01 to R6-13), the quotes that
went stale are gone, and the texts the page quotes are what the package
prints.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it.
"""
from multibench import cli
from multibench.engine import registry, resolve, schema
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_f6_cli import _lung


@needs_docs
def test_changes_page_lists_round_6_changes():
    text = _flat("changes.md")
    for stale in ("expects gene activity; atac.h5 holds peaks",
                  "reads batches 1-2 of 3; batch 3 is not used",
                  "Count lines say rows,",
                  "`--method` must then be a package method, and it sets the label order."):
        assert stale not in text, stale
    for phrase in ("`multibench evaluate --batch-column NAME` and "
                   "`multibench run-all --batch-column NAME`",
                   "With `--labels`, `--method` can be left out.",
                   "a `needs_labels` column gives the rows the `L` badge of `plot bubble`",
                   "`mtb.load_batch(out_dir, data_path=...)`",
                   "then under `data_root`.",
                   "`batch_<hash>.csv`. The record key `batch_file` names the file.",
                   "`BatchResult.rescore()` reuses that batch, also with `labels=`.",
                   "keeps no metrics, `batch_source` or `n_batches` from an earlier scoring.",
                   "with no ARI, NMI or iF1 keeps the stored label order and runs no "
                   "Leiden sweep.",
                   "no longer warns about the batch it takes from the label files",
                   "`label_order_confidence` stays within 0-1",
                   "`UINMF reads batches 1-2 of 3. Batch 3 is not used.`",
                   "Count lines say methods when each method has one row, and rows otherwise",
                   "and `scan --strict` prints it in full"):
        assert phrase in text, phrase


@needs_docs
def test_changes_page_quotes_the_live_texts():
    text = _flat("changes.md")
    spec = registry.get("SCALEX")
    for shown in (
            registry.unknown_method_message("stabmap"),
            schema.no_category_message("SCALEX", spec.wired_categories, "vertical"),
            resolve.UNUSED_BATCHES_CAVEAT.format(method="UINMF", used="1-2", n=3,
                                                 unused="Batch 3 is not used"),
            resolve.NOT_COUNTS_CAVEAT.format(method="totalVI", file="rna.h5")):
        assert f"`{shown}`" in text, shown


@needs_docs
def test_changes_page_run_dry_run_lines_follow_the_cli(tmp_path, monkeypatch, capsys):
    _lung(tmp_path)
    monkeypatch.chdir(tmp_path)
    missing = "data/LUNG/atac_gas.h5"
    assert cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                     "--input", "rna=data/LUNG/rna.h5", "--input", f"atac_gas={missing}",
                     "--out", "runs/SCALEX", "--dry-run"]) == 0
    err = capsys.readouterr().err.splitlines()
    note = f"SCALEX reads {missing}, which does not exist."
    assert err[0] == "# Dry run. Nothing was executed."
    assert any(line.startswith(f"# {note}") for line in err), err
    assert err[-1] == "# multibench run would execute:"
    text = _flat("changes.md")
    assert f"`{note}`" in text
    assert ("print `# Dry run. Nothing was executed.` first. `run --dry-run` then prints "
            "its notes, `# multibench run would execute:` and the command.") in text
