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


def test_defaults_point_at_local_data(tmp_path, monkeypatch):
    cfg = config._Config()
    # default result/files paths resolve under the repo's multibench/ dir
    assert cfg.result_path.name == "result"
    assert cfg.files_path.name == "files"


def test_config_signature_and_repr_leave_out_the_package_tables():
    # Z32 (r3): result_path and files_path are the package's, still readable
    import inspect
    params = inspect.signature(config._Config).parameters
    assert "result_path" not in params and "files_path" not in params
    cfg = config._Config()
    assert "result_path" not in repr(cfg) and cfg.result_path.name == "result"
