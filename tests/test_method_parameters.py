"""Every method script takes hyperparameters, and the package's records of them
follow the scripts in ``tools_scripts/``."""
import hashlib
import importlib.util
import json

import pytest

from multibench import _settings, config
from multibench.engine import builder, registry, scripts
from multibench.tools import gen_params

from conftest import ROOT


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_params_table_is_the_one_the_scripts_give():
    """A script edit without ``python -m multibench.tools.gen_params`` fails here."""
    assert gen_params.OUT.read_text() == gen_params.render()


def test_recorded_archive_is_the_one_the_scripts_pack_to():
    """A script edit without ``python tools/pack_scripts.py`` (and the upload) fails
    here: a wheel would otherwise download scripts that are not these."""
    tool = _tool("pack_scripts")
    data = tool.pack()
    assert tool.pack() == data                       # the same bytes every time
    assert json.loads(tool.RECORD.read_text()) == tool.record(data)
    assert scripts.archive_id() == hashlib.sha256(data).hexdigest()[:12]


def test_every_variant_has_parameters_and_every_script_exists():
    for spec in registry.load():
        for v in spec.variants:
            assert (ROOT / v.entrypoint).is_file(), (spec.id, v.entrypoint)
            assert v.tunable, (spec.id, v.when)


def test_gpu_evidence_cites_a_cuda_call_in_the_package_scripts():
    for spec in registry.load():
        for cited in filter(None, (c.strip() for c in spec.gpu_evidence.split(","))):
            path, line = cited.rsplit(":", 1)
            text = (ROOT / path).read_text().splitlines()[int(line) - 1]
            assert "cuda" in text.lower(), (spec.id, cited)


def test_a_default_is_the_script_value_or_none():
    def tunable(method):
        return registry.get(method).variants[0].tunable
    tun = tunable("scMoMaT")
    assert (tun["K"]["default"], tun["T"]["default"], tun["lr"]["default"]) == (30, 4000, 0.01)
    assert tunable("iNMF")["k"]["default"] == 20 and tunable("iNMF")["lambda"]["default"] is None
    conos = tunable("Conos")
    assert conos["alpha"]["default"] == 0.5 and conos["n.odgenes"]["default"] == 5000
    # StabMap's options are read from its R driver
    assert tunable("StabMap")["maxFeatures"] == {"default": float("inf"), "type": "float"}


def test_a_numeric_default_is_a_number():
    """sciPENN's script writes its lr default as ``10 ** (-3)``: the table holds 0.001."""
    import yaml
    table = yaml.safe_load(gen_params.OUT.read_text())
    numeric = [(m, k, p["default"]) for m, variants in table.items()
               for params in variants.values() for k, p in params.items()
               if p.get("type") in ("int", "float")]
    assert numeric and all(d is None or type(d) in (int, float) for _, _, d in numeric), \
        [x for x in numeric if not (x[2] is None or type(x[2]) in (int, float))]
    assert {d for m, k, d in numeric if (m, k) == ("sciPENN", "lr")} == {0.001}


def _variant(method, language):
    return next(v for v in registry.get(method).variants if v.language == language)


def _values(v, value):
    return {r: value for a in v.args for r in ([a.role] if a.role else a.roles or [])
            if not str(r).startswith("=")}


def test_builder_writes_r_options_as_name_equals_value():
    v = _variant("iNMF", "R")
    cmd = builder.build_command(v, _values(v, "x.h5"), "out/", params={"k": 10, "seed": 2})
    assert cmd[-2:] == ["--k=10", "--seed=2"]


def test_builder_writes_lists_and_true_false_options():
    v = _variant("MultiMAP", "python")
    cmd = builder.build_command(v, _values(v, "x.h5"), "out/", params={"strengths": [0.3, 0.7]})
    i = cmd.index("--strengths")
    assert cmd[i:i + 3] == ["--strengths", "0.3", "0.7"]
    v = _variant("totalVI", "python")
    cmd = builder.build_command(v, _values(v, ["x.h5"]), "out/", params={"early_stopping": False})
    i = cmd.index("--early_stopping")
    assert cmd[i + 1] == "false"                     # not dropped as an unset flag


def test_download_refuses_an_archive_with_another_sha256(tmp_path, monkeypatch, scripts_download):
    monkeypatch.setattr(scripts, "_archive_bytes", lambda url: b"not the archive")
    with pytest.raises(RuntimeError, match="are not the ones this multibench version names"):
        scripts.ensure_repo(tmp_path / "scripts")
    assert not (tmp_path / "scripts").exists() and not (tmp_path / "scripts.partial").exists()


def test_an_archive_not_published_is_named_with_its_url(tmp_path, monkeypatch, scripts_download):
    """A 404 is not a network problem: the message names the URL and where to report it."""
    import urllib.error

    def missing(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    monkeypatch.setattr(scripts, "_archive_bytes", missing)
    with pytest.raises(RuntimeError) as e:
        scripts.ensure_repo(tmp_path / "scripts")
    msg = str(e.value)
    assert scripts.scripts_archive()["url"] in msg and "(HTTP 404)" in msg
    assert msg.endswith(f"Report it at {scripts.ISSUES_URL}.")
    assert not (tmp_path / "scripts").exists() and not (tmp_path / "scripts.partial").exists()


def test_download_unpacks_the_scripts_and_a_run_record_names_them(tmp_path, monkeypatch,
                                                                  scripts_download):
    monkeypatch.setattr(config, "repo_path", tmp_path / "scripts")
    repo = scripts.ensure_repo()
    assert repo == tmp_path / "scripts"
    assert (repo / "tools_scripts" / "totalVI" / "main_totalVI.py").is_file()
    assert scripts.run_provenance(None)["scripts_commit"] == scripts_download


def test_a_scripts_ref_is_fetched_apart_from_the_package_scripts(tmp_path, monkeypatch,
                                                                 scripts_download):
    """A checkout's own scripts are not looked at for a ref, and downloaded ones
    are never at one."""
    repo = scripts.ensure_repo(tmp_path / "scripts")
    (tmp_path / "pkg" / "tools_scripts").mkdir(parents=True)
    monkeypatch.setattr(_settings, "_ROOT", tmp_path / "pkg")
    monkeypatch.setenv(scripts.SCRIPTS_REF_VAR, "0c68f87")
    assert scripts.folders(tmp_path / "other") == (tmp_path / "other",)
    with pytest.raises(RuntimeError, match=f"are {scripts_download}, not 0c68f87"):
        scripts.ensure_repo(repo)
