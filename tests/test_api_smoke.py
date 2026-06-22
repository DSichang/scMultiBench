def test_package_imports_and_has_version():
    import multibench
    assert isinstance(multibench.__version__, str)
    assert multibench.__version__ != ""


def test_top_level_exposes_data_api():
    import multibench as mtb
    assert hasattr(mtb, "load_results")
    assert hasattr(mtb, "catalog")
    assert hasattr(mtb, "config")


def test_top_level_exposes_spec_symbols():
    import multibench as mtb
    assert hasattr(mtb, "inputs_for")
    assert hasattr(mtb.io, "to_canonical")
    assert "clustering" in mtb.list_tasks()


def test_end_to_end_load_results(result_dir, monkeypatch):
    import multibench as mtb
    df = mtb.load_results(category="diagonal", dataset="D27", result_path=result_dir)
    assert len(df) > 0
    assert "value" in df.columns
