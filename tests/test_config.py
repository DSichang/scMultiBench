from multibench import config


def test_category_token_to_folder():
    assert config.category_folder("vertical") == "vertical integration"
    assert config.category_folder("diagonal") == "diagonal integration"
    assert config.category_folder("mosaic") == "mosaic integration"
    assert config.category_folder("cross") == "cross integration"


def test_category_folder_rejects_unknown():
    import pytest
    with pytest.raises(ValueError) as exc:
        config.category_folder("nope")
    assert "nope" in str(exc.value)
    # error lists valid options
    assert "vertical" in str(exc.value)


def test_the_package_tables_are_under_the_package():
    # the stored results and the dataset and metric lists: read, not settings
    assert config._RESULT_PATH.name == "result" and config._FILES_PATH.name == "files"
    assert config._RESULT_PATH.parent.name == "multibench"


def test_the_repr_shows_the_settings_and_not_the_package_tables():
    # Z32 (r3): result_path and files_path are the package's
    text = repr(config)
    assert "files_path" not in text and "result_path" not in text
    assert text.count("=") == 4
