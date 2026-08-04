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
from multibench.workflow import BatchResult


def test_summary_surfaces_label_order_and_margin():
    """Priya's blocker: 'I get an ARI and have no way to know if it is meaningful.'

    The chosen label order and its lead over the runner-up must be visible in
    .summary - the table people actually read - not buried in .results.
    """
    recs = [{"method": "M", "status": "CHAIN_OK", "metrics": {"ARI": 0.7},
             "labels_used": ["rna_cty.csv", "atac_cty.csv"],
             "label_order_candidates": [{"order": ["rna_cty.csv", "atac_cty.csv"], "ARI": 0.7005},
                                        {"order": ["atac_cty.csv", "rna_cty.csv"], "ARI": 0.0009}],
             "_long": None}]
    s = BatchResult(recs, "D28", "diagonal").summary
    assert s.loc[0, "label_order"] == "rna_cty.csv+atac_cty.csv"
    assert abs(s.loc[0, "label_order_margin"] - 0.6996) < 1e-6


def test_summary_label_order_margin_none_when_unambiguous():
    recs = [{"method": "M", "status": "CHAIN_OK", "metrics": {"ARI": 0.9},
             "labels_used": ["cty.csv"], "_long": None}]
    s = BatchResult(recs, "D11", "vertical").summary
    assert s.loc[0, "label_order"] == "cty.csv"
    assert s.loc[0, "label_order_margin"] is None
