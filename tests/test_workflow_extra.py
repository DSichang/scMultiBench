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
