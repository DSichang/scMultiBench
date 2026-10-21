"""Regression tests for usability fixes found by non-coder reviewers."""
import os
import pytest
import multibench as mtb


def test_describe_layout_names_files_and_flags_the_atac_trap():
    txt = mtb.describe_layout("vertical")
    assert "rna.h5" in txt and "adt.h5" in txt and "cty.csv" in txt
    # vertical reads atac.h5, and the method decides what it must hold (L34);
    # 0.4.0: scan and run_all skip a method whose file holds the other one
    assert "atac.h5" in txt and "mtb.scan and mtb.run_all skip such a method." in txt
    # assert the SUBSTANCE of the warning, not one word of its phrasing: the
    # text must say which representation each method needs, and that feeding
    # the wrong one does not raise.
    low = txt.lower()
    assert "peaks" in low and ("gene activity" in low or "gene-activity" in low)
    assert "wrong" in low, "must warn that the wrong representation is silently wrong"
    # the diagonal text names both representation files and the older name
    diag = mtb.describe_layout("diagonal")
    assert "atac_peak.h5" in diag and "atac_gas.h5" in diag and "peak.h5" in diag


def test_package_docstring_has_import_and_categories():
    doc = mtb.__doc__ or ""
    assert "import multibench as mtb" in doc
    for c in ("vertical", "diagonal", "mosaic", "cross"):
        assert c in doc


def test_skip_existing_with_params_is_refused():
    """Reviewer footgun: reuse is keyed on the output FILE, not on params, so
    changing a hyperparameter and reusing out_dir would silently return the OLD
    result. Refuse rather than mislead."""
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir="/tmp/unused",
                    params={"Matilda": {"epochs": 5}}, skip_existing=True)
    assert "earlier parameters" in str(e.value)
    assert "OLD" not in str(e.value)                  # no emphasis by capitals (L61)


def test_batch_result_attributes_are_documented():
    import inspect
    for attr in ("summary", "long", "failures", "plot"):
        a = getattr(mtb.BatchResult, attr)
        doc = inspect.getdoc(a.fget if isinstance(a, property) else a)
        assert doc and len(doc) > 40, f"BatchResult.{attr} needs a real docstring"
    # records (results until 0.4) is an instance attribute, documented on the class
    assert "records : list[dict]" in inspect.getdoc(mtb.BatchResult)


def test_failures_excludes_successful_but_unscorable_runs():
    """A method that ran fine but emits only a graph is NOT a failure.

    Seurat_WNN and MIRA write a neighbour graph and no embedding, so there is
    nothing to score - listing them under .failures sends users hunting for a
    bug that does not exist.
    """
    from multibench.workflow import BatchResult
    recs = [{"method": "Seurat_WNN", "status": "RUN_OK_NO_EMBEDDING", "_long": None},
            {"method": "MIRA", "status": "RUN_OK_NO_EMBEDDING", "_long": None},
            {"method": "Broken", "status": "FAIL", "error": "boom", "_long": None},
            {"method": "Slow", "status": "TIMEOUT", "error": "too long", "_long": None},
            {"method": "Good", "status": "CHAIN_OK", "_long": None}]
    r = BatchResult(recs, "D11", "vertical")
    assert set(r.failures["method"]) == {"Broken", "Slow"}
    assert repr(r) == "<BatchResult vertical/D11: 1 scored, 2 ran, 1 failed, 1 timeout>"
import multibench as mtb
from multibench.workflow import BatchResult


def test_runresult_is_documented():
    import inspect
    from multibench.engine.runner import RunResult
    doc = inspect.getdoc(RunResult) or ""
    assert len(doc) > 200, "RunResult is the return of run(); it needs real docs"
    for word in ("out_dir", "output", "stderr", "extra"):
        assert word in doc


def test_run_all_raises_when_nothing_can_run(tmp_path, monkeypatch):
    """A per-method failure is recorded, but "not one method could start" means the
    REQUEST is wrong. Returning an empty result would report "0 failed", which reads
    as success and hides the problem - the reviewer's top concern. (A dataset whose
    FOLDER is missing is the even earlier FileNotFoundError below.)"""
    from multibench import workflow as W
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    (tmp_path / "EMPTYDS").mkdir()
    with pytest.raises(ValueError) as e:
        mtb.run_all(os.path.join(tmp_path, 'EMPTYDS'), 'vertical', out_dir=tmp_path / 'out', verbose=False)
    assert str(e.value).startswith("No method can run on EMPTYDS (vertical).\n")
    assert "mtb.run_all" in str(e.value)       # tells you how to diagnose it


def test_run_all_missing_dataset_folder_is_filenotfound():
    """A typo'd dataset name used to surface as 'nothing is runnable'; now the
    folder check fires first and lists the folders that DO exist."""
    with pytest.raises(FileNotFoundError) as e:
        mtb.run_all("NO_SUCH_DATASET_XYZ", "vertical",
                    out_dir="/tmp/mtb_empty_test", verbose=False)
    assert str(e.value).startswith("The folder ") and "NO_SUCH_DATASET_XYZ does not exist. " \
        in str(e.value)
    assert "D11" in str(e.value)


def test_describe_layout_is_category_specific():
    """It used to print the CITE-seq layout whatever category you asked for, so an
    unpaired user was told to write a single cty.csv - which silently mis-scores."""
    diag = mtb.describe_layout("diagonal")
    assert "rna_cty.csv" in diag and "atac_cty.csv" in diag
    assert "\ndiagonal: " in diag
    vert = mtb.describe_layout("vertical")
    assert "\nvertical: " in vert
    # per-category output prints only that category (L33): no role table of
    # the others, so the whole text is checked, not a block of it
    assert "atac_cty.csv" not in vert and "rna_cty.csv" not in vert
    assert "\ndiagonal: " not in vert and "\nvertical: " not in diag


# --- handover: registry-generated describe_layout lists, validated tokens -------

def test_describe_layout_rejects_typo():
    with pytest.raises(ValueError, match="unknown category 'crosss'"):
        mtb.describe_layout("crosss")
    assert mtb.describe_layout(None)            # None still prints everything


def test_read_labels_reads_cty_and_obs_exports(tmp_path):
    from multibench.eval import io
    p = tmp_path / "cty.csv"
    p.write_text("x\nA\nB\nA\n")
    assert list(io.read_labels(p)) == ["A", "B", "A"]
    # the obs-style export (barcode index + label column) now also reads
    q = tmp_path / "obs.csv"
    q.write_text(",celltype\nc1,T\nc2,B\n")
    assert list(io.read_labels(q)) == ["T", "B"]


def test_batch_result_plot_docstring_names_the_kept_keywords():
    # c1 (C05): bubble has no order= since 0.4.0
    import inspect
    doc = inspect.getdoc(mtb.BatchResult.plot)
    assert "order=" not in doc and "``methods=``" in doc
