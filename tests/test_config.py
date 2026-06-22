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


def test_metric_set_dir():
    assert config.metric_set_dir("scib") == "scib_metric"


def test_metric_set_dir_rejects_unwired():
    # only "scib" is wired in v1; other metric sets are not advertised
    import pytest
    with pytest.raises(ValueError):
        config.metric_set_dir("classification")


def test_defaults_point_at_local_data(tmp_path, monkeypatch):
    cfg = config.Config()
    # default result/files paths resolve under the repo's multibench/ dir
    assert cfg.result_path.name == "result"
    assert cfg.files_path.name == "files"
