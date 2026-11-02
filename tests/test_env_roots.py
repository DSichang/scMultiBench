"""An installed env stays found when the default envs_dir moves.

With no writable conda envs folder the envs go to ~/.cache/multibench/envs;
a later `conda create -n x` or `conda run -n x` makes ~/.conda/envs, which
then becomes the default, and the installed envs were no longer found: run
fell back to conda and install downloaded them again.
"""
import multibench as mtb
from multibench import _settings
from multibench.engine import envs


def _installed(root, env="scmb_scjoint"):
    """What install_packed leaves at <root>/<env>."""
    (root / env / "bin").mkdir(parents=True)
    (root / env / envs.FLAVOR_FILE).write_text("cpu\n")
    return root / env


def _no_conda(monkeypatch):
    monkeypatch.setattr(envs, "_find_conda", lambda: None)


def test_an_env_in_a_folder_an_install_used_is_found_after_the_default_moves(
        tmp_path, monkeypatch):
    _no_conda(monkeypatch)
    monkeypatch.setattr(_settings, "_ENVS_FALLBACK", tmp_path / "fallback")
    first = tmp_path / "first"
    prefix = _installed(first)
    envs._remember_root(first)
    monkeypatch.setattr(mtb.config, "envs_dir", tmp_path / "later")
    assert envs.env_prefix("scmb_scjoint") == prefix
    assert "scmb_scjoint" in envs.installed_envs()


def test_the_fallback_folder_is_searched_without_a_record(tmp_path, monkeypatch):
    # envs a version before the record installed there
    _no_conda(monkeypatch)
    monkeypatch.setattr(_settings, "_ENVS_FALLBACK", tmp_path / "fallback")
    prefix = _installed(tmp_path / "fallback")
    monkeypatch.setattr(mtb.config, "envs_dir", tmp_path / "conda_envs")
    assert envs.env_prefix("scmb_scjoint") == prefix


def test_envs_dir_comes_first(tmp_path, monkeypatch):
    _no_conda(monkeypatch)
    monkeypatch.setattr(_settings, "_ENVS_FALLBACK", tmp_path / "fallback")
    _installed(tmp_path / "fallback")
    mine = _installed(tmp_path / "mine")
    monkeypatch.setattr(mtb.config, "envs_dir", tmp_path / "mine")
    assert envs.env_prefix("scmb_scjoint") == mine


def test_a_folder_is_recorded_once(tmp_path):
    for _ in range(3):
        envs._remember_root(tmp_path / "a")
    envs._remember_root(tmp_path / "b")
    assert envs.ROOTS_FILE.read_text().splitlines() == [str(tmp_path / "a"),
                                                         str(tmp_path / "b")]


def test_an_unwritable_record_only_loses_the_record(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(envs, "ROOTS_FILE", blocker / "envs_dirs.txt")
    envs._remember_root(tmp_path / "a")        # no error
    assert envs._known_roots() == [_settings._ENVS_FALLBACK]


def test_a_lockfile_build_records_its_folder(tmp_path, monkeypatch):
    _no_conda(monkeypatch)
    monkeypatch.setattr(mtb.config, "envs_dir", tmp_path / "conda_envs")
    _installed(tmp_path / "conda_envs").joinpath(envs.FLAVOR_FILE).unlink()
    envs._mark_built("scmb_scjoint", None, before=set())
    assert tmp_path / "conda_envs" in envs._known_roots()
