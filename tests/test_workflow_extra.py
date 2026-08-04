"""Regression tests for usability fixes found by non-coder reviewers."""
import pytest
import multibench as mtb


def test_list_categories_names_all_four():
    """Reviewers' #1 blocker: category is required but its values were undocumented."""
    cats = mtb.list_categories()
    assert set(cats) == {"vertical", "diagonal", "mosaic", "cross"}
    for name, desc in cats.items():
        assert len(desc) > 30, f"{name} needs a real description, not a label"


def test_describe_layout_names_files_and_flags_the_atac_trap():
    txt = mtb.describe_layout("vertical")
    assert "rna.h5" in txt and "adt.h5" in txt and "cty.csv" in txt
    # a peak matrix dropped into atac.h5 runs everything on the wrong representation
    assert "peak.h5" in txt and "atac.h5" in txt
    assert "Careful" in txt


def test_package_docstring_has_import_and_categories():
    doc = mtb.__doc__ or ""
    assert "import multibench as mtb" in doc
    for c in ("vertical", "diagonal", "mosaic", "cross"):
        assert c in doc


def test_find_methods_tunable_filter():
    tunable = set(mtb.find_methods(tunable=True))
    fixed = set(mtb.find_methods(tunable=False))
    assert tunable and fixed
    assert not (tunable & fixed)
    assert tunable | fixed == set(mtb.find_methods())


def test_skip_existing_with_params_is_refused():
    """Reviewer footgun: reuse is keyed on the output FILE, not on params, so
    changing a hyperparameter and reusing out_dir would silently return the OLD
    result. Refuse rather than mislead."""
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir="/tmp/unused",
                    params={"Matilda": {"epochs": 5}}, skip_existing=True)
    assert "OLD parameters" in str(e.value)


def test_batch_result_attributes_are_documented():
    import inspect
    for attr in ("summary", "long", "failures", "results", "plot"):
        a = getattr(mtb.BatchResult, attr)
        doc = inspect.getdoc(a.fget if isinstance(a, property) else a)
        assert doc and len(doc) > 40, f"BatchResult.{attr} needs a real docstring"


def test_failures_excludes_successful_but_unscorable_runs():
    """A method that ran fine but emits coords/graph is NOT a failure.

    The four spatial-registration methods produce aligned coordinates, so there is
    no embedding to score - listing them under .failures sends users hunting for a
    bug that does not exist.
    """
    from multibench.workflow import BatchResult
    recs = [{"method": "PASTE", "status": "RUN_OK_NO_EMBEDDING", "_long": None},
            {"method": "SPIRAL", "status": "RUN_OK_NO_EMBEDDING", "_long": None},
            {"method": "Broken", "status": "FAIL", "error": "boom", "_long": None},
            {"method": "Slow", "status": "TIMEOUT", "error": "too long", "_long": None},
            {"method": "Good", "status": "CHAIN_OK", "_long": None}]
    r = BatchResult(recs, "D63", "cross")
    assert set(r.failures["method"]) == {"Broken", "Slow"}
    assert "ran but not scorable" in repr(r)
import multibench as mtb
from multibench.workflow import BatchResult, _order_confidence


def test_order_confidence_is_scale_free():
    """A plain difference is bounded above by the ARI, so a method scoring 0.3 could
    never look well-separated however unambiguous its ordering. The ratio must."""
    strong_high = _order_confidence([{"ARI": 0.90}, {"ARI": 0.001}])
    strong_low = _order_confidence([{"ARI": 0.30}, {"ARI": 0.001}])
    ambiguous = _order_confidence([{"ARI": 0.30}, {"ARI": 0.28}])
    assert strong_high > 0.99
    assert strong_low > 0.99, "a low-ARI method with an unambiguous order must still read high"
    assert ambiguous < 0.2
    assert _order_confidence([{"ARI": 0.5}]) is None      # nothing to choose between
    assert _order_confidence([]) is None
    # At chance the ratio would compare two noise values (0.0004 vs 0.0002 reads
    # 0.5 while BOTH orderings are garbage), so it must refuse to answer.
    assert _order_confidence([{"ARI": 0.0}, {"ARI": 0.0}]) is None
    assert _order_confidence([{"ARI": 0.0004}, {"ARI": 0.0002}]) is None
    assert _order_confidence([{"ARI": 0.30}, {"ARI": 0.001}]) > 0.99


def test_summary_reports_label_order_and_confidence():
    recs = [{"method": "M", "status": "CHAIN_OK", "metrics": {"ARI": 0.3},
             "labels_used": ["rna_cty.csv", "atac_cty.csv"],
             "label_order_candidates": [{"order": ["rna_cty.csv", "atac_cty.csv"], "ARI": 0.30},
                                        {"order": ["atac_cty.csv", "rna_cty.csv"], "ARI": 0.001}],
             "_long": None}]
    s = BatchResult(recs, "D28", "diagonal").summary
    assert s.loc[0, "label_order"] == "rna_cty.csv+atac_cty.csv"
    assert s.loc[0, "label_order_confidence"] > 0.99
import pytest
import multibench as mtb


def test_sweep_rejects_a_param_the_method_does_not_expose():
    """A sweep over a nonexistent knob would burn hours and change nothing."""
    with pytest.raises(KeyError) as e:
        mtb.sweep("D11", "vertical", "Multigrate", "not_a_real_param", [1, 2],
                  out_dir="/tmp/unused", modalities=["rna", "adt"])
    assert "does not expose" in str(e.value)
    assert "lr" in str(e.value)          # tells you what it DOES accept


def test_runresult_is_documented():
    import inspect
    from multibench.engine.runner import RunResult
    doc = inspect.getdoc(RunResult) or ""
    assert len(doc) > 200, "RunResult is the return of run(); it needs real docs"
    for word in ("out_dir", "output", "stderr", "extra"):
        assert word in doc


def test_sweep_exported():
    assert hasattr(mtb, "sweep")


def test_run_all_raises_when_nothing_can_run():
    """A per-method failure is recorded, but "not one method could start" means the
    REQUEST is wrong. Returning an empty result would report "0 failed", which reads
    as success and hides a typo in the dataset name - the reviewer's top concern."""
    with pytest.raises(ValueError) as e:
        mtb.run_all("NO_SUCH_DATASET_XYZ", "vertical",
                    out_dir="/tmp/mtb_empty_test", verbose=False)
    assert "nothing is runnable" in str(e.value)
    assert "mtb.scan" in str(e.value)          # tells you how to diagnose it


def test_describe_layout_is_category_specific():
    """It used to print the CITE-seq layout whatever category you asked for, so an
    unpaired user was told to write a single cty.csv - which silently mis-scores."""
    diag = mtb.describe_layout("diagonal")
    assert "rna_cty.csv" in diag and "atac_cty.csv" in diag
    assert "LAYOUT FOR DIAGONAL" in diag
    vert = mtb.describe_layout("vertical")
    assert "LAYOUT FOR VERTICAL" in vert
    # the LAYOUT block is category-specific; the role table below it is a general
    # reference and legitimately lists every role, so scope the check to the block
    block = vert.split("LAYOUT FOR VERTICAL")[1].split("Modality roles")[0]
    assert "atac_cty.csv" not in block
    diag_block = diag.split("LAYOUT FOR DIAGONAL")[1].split("Modality roles")[0]
    assert "atac_cty.csv" in diag_block
