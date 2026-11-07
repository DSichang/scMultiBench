"""``mtb.env.install`` downloads the method scripts after the environments, so
a job on compute nodes without network finds both where the login node put
them; before, only a run downloaded the scripts."""
import pytest

import multibench as mtb
from multibench.engine import envs, scripts


@pytest.fixture
def steps(monkeypatch):
    """The order of the env step and the scripts download; the env step is a stub."""
    seen = []

    def install_rows(*a, **kw):
        seen.append("envs")
        return []
    monkeypatch.setattr(envs, "_install_rows", install_rows)
    fetch = scripts._archive_bytes
    monkeypatch.setattr(scripts, "_archive_bytes",
                        lambda url: seen.append("scripts") or fetch(url))
    return seen


def test_install_downloads_the_scripts_after_the_envs(tmp_path, scripts_download, steps,
                                                      capsys):
    mtb.config.repo_path = tmp_path / "scripts"
    mtb.env.install(["scMM"])
    assert steps == ["envs", "scripts"]
    assert (tmp_path / "scripts" / "tools_scripts" / "scMM" / "main_scMM.py").is_file()
    assert capsys.readouterr().out == ""
    mtb.env.install(["scMM"])                            # present: no second download
    assert steps == ["envs", "scripts", "envs"]


def test_a_dry_run_downloads_no_scripts(tmp_path, scripts_download, steps):
    mtb.config.repo_path = tmp_path / "scripts"
    mtb.env.install(["scMM"], dry_run=True)
    assert steps == ["envs"] and not (tmp_path / "scripts").exists()


def test_without_network_install_says_what_to_copy(tmp_path, steps):
    mtb.config.repo_path = tmp_path / "scripts"
    with pytest.raises(RuntimeError, match="^Could not reach github.com to fetch the method "
                                           "scripts. Copy the scripts folder"):
        mtb.env.install(["scMM"])
    assert steps == ["envs", "scripts"] and not (tmp_path / "scripts").exists()


def test_off_linux_install_refuses_before_the_scripts(tmp_path, scripts_download, monkeypatch):
    monkeypatch.setattr(envs, "host_system", lambda: "macOS")
    monkeypatch.setattr(scripts, "_archive_bytes", lambda url: pytest.fail("downloaded"))
    mtb.config.repo_path = tmp_path / "scripts"
    with pytest.raises(OSError, match="^Methods run only on Linux"):
        mtb.env.install(["scMM"])
    assert not (tmp_path / "scripts").exists()
