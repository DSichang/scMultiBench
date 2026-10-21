from multibench import config
from multibench import _settings


def test_category_tokens_are_checked():
    import pytest
    from multibench.engine import registry
    assert [registry.check_category(c) for c in registry.CATEGORIES] == list(registry.CATEGORIES)
    with pytest.raises(ValueError) as exc:
        registry.check_category("nope")
    assert "nope" in str(exc.value) and "vertical" in str(exc.value)


def test_leiden_flavor_is_checked_when_set():
    import pytest
    with pytest.raises(ValueError, match="leiden_flavor takes 'leidenalg' or 'igraph', "
                                         "not 'louvain'"):
        config.leiden_flavor = "louvain"
    config.leiden_flavor = "igraph"
    assert config.leiden_flavor == "igraph"
    config.leiden_flavor = None
    assert config.leiden_flavor == "leidenalg"


def test_config_is_a_plain_object_with_four_settings():
    import pytest
    with pytest.raises(AttributeError, match="has no setting 'ensure_repo'"):
        config.ensure_repo = 3
    assert not hasattr(config, "__dict__")              # __slots__: no stray attribute
    config.data_path = "~/somewhere"
    assert str(config.data_path).startswith(str(__import__("pathlib").Path.home()))


def test_the_package_tables_are_under_the_package():
    # the stored results and the dataset and metric lists: read, not settings
    assert _settings._RESULT_PATH.name == "result" and _settings._FILES_PATH.name == "files"
    assert _settings._RESULT_PATH.parent.name == "multibench"


def test_the_repr_shows_the_settings_and_not_the_package_tables():
    # Z32 (r3): result_path and files_path are the package's
    text = repr(config)
    assert "files_path" not in text and "result_path" not in text
    assert text.count("=") == 4
