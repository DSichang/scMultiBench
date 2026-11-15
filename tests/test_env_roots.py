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


# ------------------------------------- the default folder holds a user's env
# On Windows and macOS an env is unpacked, never built. When envs_dir was not
# set and resolved to conda's envs folder, and that folder holds a user's own
# env of the archive's name, the archive goes to the package's cache folder.
import pytest


@pytest.fixture
def mac_default(tmp_path, monkeypatch):
    """A Mac with Apple silicon whose default envs_dir is conda's envs folder
    (returned), which holds the user's own ``matilda``; the cache folder is
    ``tmp_path / "cache"``; archives for Matilda and StabMap are published."""
    from multibench.engine import runner
    from tests.test_macos_envs import publish
    monkeypatch.setattr(envs, "host_system", lambda: "macOS")
    monkeypatch.setattr(envs, "_machine", lambda: "arm64")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    _no_conda(monkeypatch)
    monkeypatch.setenv(envs.MACOS_ARCHIVES_VAR, str(tmp_path / "archives"))
    monkeypatch.delenv(runner.RUN_MODE_VAR, raising=False)
    monkeypatch.delenv(_settings.ENVS_DIR_VAR, raising=False)
    monkeypatch.setattr(mtb.config, "_envs_dir", None)
    conda_envs = tmp_path / "conda" / "envs"
    monkeypatch.setattr(_settings, "_conda_envs_dir", lambda: conda_envs)
    monkeypatch.setattr(_settings, "_ENVS_FALLBACK", tmp_path / "cache")
    theirs = conda_envs / "matilda"
    (theirs / "bin").mkdir(parents=True)
    (theirs / "bin" / "python").write_text("theirs\n")
    (theirs / "conda-meta").mkdir()
    (theirs / "conda-meta" / "history").write_text("# cmd: conda create -n matilda python\n")
    publish(tmp_path, "matilda", ["Matilda"])
    publish(tmp_path, "scmb_r", ["StabMap"])
    assert mtb.config.envs_dir == conda_envs
    return conda_envs


def _listing(folder):
    return sorted((str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in folder.rglob("*"))


def test_the_archive_goes_to_the_cache_folder_when_the_default_holds_a_users_env(
        mac_default, tmp_path, monkeypatch, capsys):
    from multibench.engine import runner
    theirs, cache = mac_default / "matilda", tmp_path / "cache"
    before = _listing(theirs)
    plan = envs.install(["Matilda", "StabMap"], dry_run=True)
    assert set(plan["state"]) == {"packed archive published"}
    capsys.readouterr()

    table = envs.install(["Matilda", "StabMap"])
    assert set(table["state"]) == {"PACKED"}
    out = capsys.readouterr().out
    assert out.count(f"[env] {theirs} is another environment, so matilda goes to {cache}\n") == 1
    assert "goes to" not in out.replace(f"so matilda goes to {cache}", "")
    # found afterwards, by a lookup that starts again
    envs._conda_prefixes.cache_clear()
    assert envs.env_prefix("matilda") == cache / "matilda"
    assert envs.env_prefix("scmb_r") == mac_default / "scmb_r"     # no env in its way
    assert envs.serves("matilda", "Matilda")
    assert runner.run_mode("matilda") == ("prefix", cache / "matilda")
    runner._check_env("Matilda", "matilda")
    assert "matilda" in envs.installed_envs()
    # theirs is as it was
    assert _listing(theirs) == before and not (theirs / envs.FLAVOR_FILE).exists()

    # a second install downloads nothing
    monkeypatch.setattr(envs, "download_archive",
                        lambda *a, **k: pytest.fail("downloaded again"))
    assert set(envs.install(["Matilda", "StabMap"])["state"]) == {"have"}
    assert set(envs.install(["Matilda", "StabMap"], dry_run=True)["state"]) == {"have"}
    assert "goes to" not in capsys.readouterr().out
    assert _listing(theirs) == before


@pytest.mark.parametrize("how", ["config", "variable"])
def test_an_envs_dir_the_user_set_is_not_replaced(mac_default, tmp_path, monkeypatch, how):
    theirs = mac_default / "matilda"
    before = _listing(theirs)
    if how == "config":
        monkeypatch.setattr(mtb.config, "envs_dir", mac_default)
    else:
        monkeypatch.setenv(_settings.ENVS_DIR_VAR, str(mac_default))
    monkeypatch.setattr(envs, "download_archive",
                        lambda *a, **k: pytest.fail("downloaded despite the env in the way"))
    assert list(envs.install(["Matilda"], dry_run=True)["state"]) == ["name taken"]
    with pytest.raises(OSError) as e:
        envs.install(["Matilda"])
    assert str(e.value) == (
        f"{theirs} is not an environment multibench made, so matilda is not unpacked "
        f"there. Set mtb.config.envs_dir to another folder and run the install again.")
    assert _listing(theirs) == before and not (tmp_path / "cache").exists()


def test_a_users_folder_in_the_cache_folder_too_is_refused(mac_default, tmp_path, monkeypatch):
    also = tmp_path / "cache" / "matilda"
    (also / "bin").mkdir(parents=True)
    monkeypatch.setattr(envs, "download_archive",
                        lambda *a, **k: pytest.fail("downloaded despite the env in the way"))
    assert list(envs.install(["Matilda"], dry_run=True)["state"]) == ["name taken"]
    with pytest.raises(OSError) as e:
        envs.install(["Matilda"])
    assert str(e.value) == (
        f"{also} is not an environment multibench made, so matilda is not unpacked "
        f"there. Set mtb.config.envs_dir to another folder and run the install again.")
    assert _listing(also) == [("bin", (also / "bin").stat().st_size,
                               (also / "bin").stat().st_mtime_ns)]


def test_linux_refuses_a_users_env_in_the_default_folder_as_before(mac_default, monkeypatch):
    monkeypatch.setattr(envs, "host_system", lambda: None)
    monkeypatch.setattr(envs, "other_env_prefix",
                        lambda env, conda=None: mac_default / env if env == "matilda" else None)
    with pytest.raises(RuntimeError, match="Rename or remove .*matilda, then install again"):
        envs.install(["Matilda"])
