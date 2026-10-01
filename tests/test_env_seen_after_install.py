"""An environment installed after a first scan() counts at once.

scan() cached the installed envs for the session, so after
mtb.env.install(...) the next scan()/run_all() still reported the env as
not installed until the kernel restarted.
"""
import multibench as mtb
from multibench.engine import envs


def test_scan_sees_an_env_installed_after_the_first_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(mtb.config.DEFAULT, "envs_dir", tmp_path / "envs")
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    before = mtb.scan("D11", "vertical", methods=["sciPENN"])
    assert not before.env_ok.any()
    (tmp_path / "envs" / "env_sciPENN" / "bin").mkdir(parents=True)   # what install_packed leaves
    after = mtb.scan("D11", "vertical", methods=["sciPENN"])
    assert after.env_ok.all()
