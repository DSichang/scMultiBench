"""Archive flavours: the CPU-only ``'<env>-cpu'`` archives next to the CUDA ones.

The benchmark host publishes CPU-only variants of the tutorial torch envs
(3-4x smaller: no CUDA libraries). ``flavor=`` on ``install_packed`` /
``mtb.env.install`` / ``multibench env install --flavor`` picks one;
``'auto'`` asks :func:`host_has_gpu`. Whatever the flavour the prefix is
``<envs_dir>/<env>`` (the runner and the registry never see a flavour) and
``<prefix>/.multibench_flavor`` records which one landed.

Everything runs against a fake manifest / sizes table and the tiny tar.gz of
``test_prefix_mode``, with ``host_has_gpu`` monkeypatched both ways - so the
cases hold on the CPU laptop and the GPU host alike.
"""
from __future__ import annotations

import inspect
import json
import re
import urllib.request
import warnings
from pathlib import Path

import pytest

import multibench as mtb
from multibench import cli
from multibench.engine import envs
from tests.test_prefix_mode import (_tiny_archive, envs_dir, make_prefix,  # noqa: F401 - fixtures
                                    no_conda)

ROOT = Path(__file__).resolve().parents[1]
CPU_ENVS = ["env_sciPENN", "matilda", "scmb_scjoint", "scmb_torch", "scmb_scmm2"]
RELEASE = "https://github.com/DSichang/scMultiBench/releases/download/envs-v1"

# a manifest where matilda's CPU archive IS published (listed + measured),
# scmb_torch's is a placeholder (listed, size still null) and scmb_r has none
MANIFEST = {"matilda": "https://x/matilda.tar.gz",
            "matilda-cpu": "https://x/matilda-cpu.tar.gz",
            "scmb_torch": "https://x/scmb_torch.tar.gz",
            "scmb_torch-cpu": "https://x/scmb_torch-cpu.tar.gz",
            "scmb_r": "https://x/scmb_r.tar.gz"}
SIZES = {"matilda": {"archive_bytes": 3_000_000_000, "unpacked_bytes": 9_000_000_000},
         "matilda-cpu": {"archive_bytes": 800_000_000, "unpacked_bytes": 2_500_000_000},
         "scmb_torch": {"archive_bytes": 4_500_000_000, "unpacked_bytes": None},
         "scmb_torch-cpu": {"archive_bytes": None, "unpacked_bytes": None},
         "scmb_r": {"archive_bytes": 900_000_000, "unpacked_bytes": None}}


@pytest.fixture
def fake_tables(monkeypatch):
    monkeypatch.setattr(envs, "packed_manifest", lambda: dict(MANIFEST))
    monkeypatch.setattr(envs, "packed_sizes", lambda: {k: dict(v) for k, v in SIZES.items()})
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)


@pytest.fixture
def gpu_host(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


@pytest.fixture
def cpu_host(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


# ---------------------------------------------------------------- the flavour vocabulary
def test_flavors_and_check():
    assert envs.FLAVORS == ("auto", "cpu", "gpu") and cli._FLAVORS == envs.FLAVORS
    for f in envs.FLAVORS:
        assert envs.check_flavor(f) == f
    with pytest.raises(ValueError) as e:
        envs.check_flavor("cuda")
    assert str(e.value) == "flavor='cuda': choose one of 'auto', 'cpu', 'gpu'"
    with pytest.raises(ValueError, match="flavor=None: choose one of 'auto', 'cpu', 'gpu'"):
        envs.resolve_flavor(None)


def test_resolve_flavor_both_ways(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    assert envs.resolve_flavor("auto") == "cpu"
    assert envs.resolve_flavor("cpu") == "cpu" and envs.resolve_flavor("gpu") == "gpu"
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    assert envs.resolve_flavor("auto") == "gpu"
    assert envs.resolve_flavor("cpu") == "cpu" and envs.resolve_flavor("gpu") == "gpu"
    assert envs.resolve_flavor() == "gpu"                    # default is auto


def test_archive_key_is_a_pure_name_rule():
    assert envs.archive_key("matilda", "cpu") == "matilda-cpu"
    assert envs.archive_key("matilda", "gpu") == "matilda"
    assert envs.archive_key("matilda", None) == "matilda"


# ---------------------------------------------------------------- host_has_gpu
class _Proc:
    def __init__(self, rc, out):
        self.returncode, self.stdout = rc, out


@pytest.fixture
def probe(monkeypatch, tmp_path):
    """Control the three inputs of ``host_has_gpu``; the cache is cleared
    around every case so each probe runs afresh."""
    state = {"which": None, "proc": None, "calls": []}
    monkeypatch.setattr(envs.shutil, "which", lambda name: state["which"])

    def fake_run(argv, **kw):
        state["calls"].append(argv)
        if isinstance(state["proc"], Exception):
            raise state["proc"]
        return state["proc"]
    monkeypatch.setattr(envs.subprocess, "run", fake_run)
    monkeypatch.setattr(envs, "_NVIDIA_PROC", tmp_path / "nvidia_version")
    envs.host_has_gpu.cache_clear()
    yield state
    envs.host_has_gpu.cache_clear()


def test_host_has_gpu_false_without_tool_or_driver(probe):
    assert envs.host_has_gpu() is False and probe["calls"] == []


def test_host_has_gpu_true_when_nvidia_smi_lists_a_device(probe):
    probe["which"], probe["proc"] = "/usr/bin/nvidia-smi", _Proc(0, "GPU 0: NVIDIA A100 (UUID: x)\n")
    assert envs.host_has_gpu() is True
    assert probe["calls"] == [["/usr/bin/nvidia-smi", "-L"]]
    # cached per process: a second call does not re-run the tool
    assert envs.host_has_gpu() is True and len(probe["calls"]) == 1


@pytest.mark.parametrize("proc", [_Proc(0, "\n  \n"), _Proc(1, "GPU 0: x"),
                                  OSError("no such binary")])
def test_host_has_gpu_false_when_the_tool_answers_nothing(probe, proc):
    probe["which"], probe["proc"] = "/usr/bin/nvidia-smi", proc
    assert envs.host_has_gpu() is False


def test_host_has_gpu_true_from_the_driver_record(probe):
    envs._NVIDIA_PROC.write_text("NVRM version: 550.54\n")
    assert envs.host_has_gpu() is True and probe["calls"] == []
    # the tool failing does not veto the driver record
    envs.host_has_gpu.cache_clear()
    probe["which"], probe["proc"] = "/usr/bin/nvidia-smi", _Proc(2, "")
    assert envs.host_has_gpu() is True


def test_host_has_gpu_importable_but_not_advertised():
    assert callable(envs.host_has_gpu) and callable(mtb.env.host_has_gpu)
    assert "host_has_gpu" not in mtb.env.__all__ and "host_has_gpu" not in dir(mtb.env)
    assert set(mtb.env.__all__) == {"status", "plan", "install", "doctor", "recipe"}


# ---------------------------------------------------------------- archive_for: published, placeholder, none
def test_archive_for_every_flavour(fake_tables, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    assert envs.archive_for("matilda", "cpu") == ("matilda-cpu", "cpu")
    assert envs.archive_for("matilda", "auto") == ("matilda-cpu", "cpu")
    assert envs.archive_for("matilda", "gpu") == ("matilda", "gpu")
    # listed but unmeasured = not published yet -> the GPU build
    assert envs.archive_for("scmb_torch", "cpu") == ("scmb_torch", "gpu")
    # no CPU key at all -> the GPU build
    assert envs.archive_for("scmb_r", "cpu") == ("scmb_r", "gpu")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    assert envs.archive_for("matilda", "auto") == ("matilda", "gpu")
    assert envs.archive_for("matilda", "cpu") == ("matilda-cpu", "cpu")
    # explicit tables win over the shipped ones
    assert envs.archive_for("matilda", "cpu", manifest={}, sizes={}) == ("matilda", "gpu")
    with pytest.raises(ValueError, match="choose one of"):
        envs.archive_for("matilda", "fast")


# ---------------------------------------------------------------- install_packed
def _fetch(monkeypatch, tgz):
    fetched = []
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url: (fetched.append(url), (str(tgz), None))[1])
    return fetched


def test_install_packed_cpu_unpacks_the_cpu_archive_under_the_env_name(
        envs_dir, tmp_path, monkeypatch, fake_tables):
    fetched = _fetch(monkeypatch, _tiny_archive(tmp_path / "a.tar.gz"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # a published CPU archive: no warning
        assert envs.install_packed("matilda", flavor="cpu") is True
    assert fetched == ["https://x/matilda-cpu.tar.gz"]
    prefix = envs_dir / "matilda"                        # the env NAME, never matilda-cpu
    assert (prefix / "bin" / "python").exists() and not (envs_dir / "matilda-cpu").exists()
    assert (prefix / envs.FLAVOR_FILE).read_text() == "cpu\n"
    assert envs.installed_flavor("matilda") == "cpu"
    assert envs.env_prefix("matilda") == prefix and envs.installed_envs() == ["matilda"]


def test_install_packed_gpu_takes_the_cuda_archive(envs_dir, tmp_path, monkeypatch, fake_tables):
    fetched = _fetch(monkeypatch, _tiny_archive(tmp_path / "a.tar.gz"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert envs.install_packed("matilda", flavor="gpu") is True
    assert fetched == ["https://x/matilda.tar.gz"]
    assert (envs_dir / "matilda" / envs.FLAVOR_FILE).read_text() == "gpu\n"
    assert envs.installed_flavor("matilda") == "gpu"


def test_install_packed_auto_follows_the_host(envs_dir, tmp_path, monkeypatch, fake_tables):
    fetched = _fetch(monkeypatch, _tiny_archive(tmp_path / "a.tar.gz"))
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    assert envs.install_packed("matilda") is True         # default flavor="auto"
    assert fetched == ["https://x/matilda-cpu.tar.gz"]
    assert envs.installed_flavor("matilda") == "cpu"
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    assert envs.install_packed("matilda", envs_dir=tmp_path / "other", flavor="auto") is True
    assert fetched[-1] == "https://x/matilda.tar.gz"
    assert (tmp_path / "other" / "matilda" / envs.FLAVOR_FILE).read_text() == "gpu\n"


def test_install_packed_cpu_falls_back_with_one_warning(envs_dir, tmp_path, monkeypatch,
                                                        fake_tables):
    fetched = _fetch(monkeypatch, _tiny_archive(tmp_path / "a.tar.gz"))
    # no CPU key at all
    with pytest.warns(UserWarning) as rec:
        assert envs.install_packed("scmb_r", flavor="cpu") is True
    assert len(rec) == 1
    assert str(rec[0].message) == "no CPU archive for scmb_r; installing the GPU build (0.9 GB)"
    assert fetched == ["https://x/scmb_r.tar.gz"]
    assert (envs_dir / "scmb_r" / envs.FLAVOR_FILE).read_text() == "gpu\n"
    assert envs.installed_flavor("scmb_r") == "gpu"
    # listed, but the size is still null: the placeholder of an archive not
    # uploaded yet - the warning says what is missing and which tool fills it
    with pytest.warns(UserWarning) as rec:
        assert envs.install_packed("scmb_torch", flavor="auto") is True
    assert len(rec) == 1
    assert str(rec[0].message) == (
        "no CPU archive for scmb_torch; installing the GPU build (4.5 GB) - "
        "scmb_torch-cpu is listed in packed_urls.json but has no measured size in "
        "packed_sizes.json (not published yet; tools/packed_sizes.py records it "
        "after the upload)")
    assert fetched[-1] == "https://x/scmb_torch.tar.gz"
    # an unmeasured GPU archive prints '?' for the size, never a guess
    monkeypatch.setattr(envs, "packed_sizes", lambda: {})
    with pytest.warns(UserWarning, match=re.escape("installing the GPU build (?)")):
        envs.install_packed("scmb_r", envs_dir=tmp_path / "o2", flavor="cpu")


def test_install_packed_gpu_never_warns(envs_dir, tmp_path, monkeypatch, fake_tables):
    _fetch(monkeypatch, _tiny_archive(tmp_path / "a.tar.gz"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert envs.install_packed("scmb_r", flavor="gpu") is True
        monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
        assert envs.install_packed("scmb_torch", flavor="auto") is True


def test_install_packed_invalid_flavor_before_anything(envs_dir, monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem",
                        lambda: pytest.fail("platform checked before the flavour"))
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url: pytest.fail("downloaded"))
    with pytest.raises(ValueError) as e:
        envs.install_packed("matilda", flavor="cuda")
    assert str(e.value) == "flavor='cuda': choose one of 'auto', 'cpu', 'gpu'"


def test_install_packed_failed_unpack_leaves_no_flavor_record(envs_dir, tmp_path, monkeypatch,
                                                              fake_tables, capsys):
    _fetch(monkeypatch, _tiny_archive(tmp_path / "bad.tar.gz", unpack_ok=False))
    assert envs.install_packed("matilda", flavor="cpu") is False
    assert not (envs_dir / "matilda").exists()
    assert envs.installed_flavor("matilda") is None
    assert "prebuilt matilda failed" in capsys.readouterr().out


def test_install_packed_existing_prefix_skips_the_download_whatever_the_flavor(
        envs_dir, monkeypatch, fake_tables):
    make_prefix(envs_dir, "matilda")
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url: pytest.fail("downloaded"))
    assert envs.install_packed("matilda", flavor="cpu") is True
    assert envs.installed_flavor("matilda") is None     # built elsewhere: no record


def test_installed_flavor_reads_one_word_only(envs_dir):
    assert envs.installed_flavor("matilda") is None
    prefix = make_prefix(envs_dir, "matilda")
    assert envs.installed_flavor("matilda") is None
    (prefix / envs.FLAVOR_FILE).write_text("cpu\n")
    assert envs.installed_flavor("matilda") == "cpu"
    (prefix / envs.FLAVOR_FILE).write_text("  gpu  \n")
    assert envs.installed_flavor("matilda") == "gpu"
    for junk in ("", "   \n", "tpu\n"):
        (prefix / envs.FLAVOR_FILE).write_text(junk)
        assert envs.installed_flavor("matilda") is None, repr(junk)
    (prefix / envs.FLAVOR_FILE).write_text("cpu gpu\n")   # first word only
    assert envs.installed_flavor("matilda") == "cpu"
    assert envs.FLAVOR_FILE == ".multibench_flavor"


# ---------------------------------------------------------------- mtb.env.install: plan rows per flavour
@pytest.fixture
def plan_rows(monkeypatch, fake_tables):
    rows = [{"env": "matilda", "methods": ["Matilda"], "exists": False, "has_lock": True,
             "cmds": []},
            {"env": "scmb_torch", "methods": ["SCALEX"], "exists": False, "has_lock": True,
             "cmds": []},
            {"env": "scmb_r", "methods": ["UINMF"], "exists": False, "has_lock": True,
             "cmds": []}]
    monkeypatch.setattr(envs, "create_all", lambda **kw: [dict(r) for r in rows])
    monkeypatch.setattr(envs, "install_packed", lambda env, **kw: pytest.fail("no download"))
    return rows


def _by_env(rows):
    return {r["env"]: r for r in rows}


def test_install_dry_run_sizes_follow_the_flavour(plan_rows, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    cpu = _by_env(mtb.env.install(["Matilda", "SCALEX", "UINMF"], flavor="cpu"))
    assert cpu["matilda"]["flavor"] == "cpu"
    assert cpu["matilda"]["archive_bytes"] == 800_000_000
    assert cpu["matilda"]["unpacked_bytes"] == 2_500_000_000
    assert cpu["matilda"]["packed_url"] == "https://x/matilda-cpu.tar.gz"
    # the placeholder and the env without a CPU key fall back to the GPU build
    assert cpu["scmb_torch"]["flavor"] == "gpu"
    assert cpu["scmb_torch"]["archive_bytes"] == 4_500_000_000
    assert cpu["scmb_torch"]["packed_url"] == "https://x/scmb_torch.tar.gz"
    assert cpu["scmb_r"]["flavor"] == "gpu" and cpu["scmb_r"]["archive_bytes"] == 900_000_000
    assert all(r["state"] == "packed archive published" for r in cpu.values())
    auto = _by_env(mtb.env.install(["Matilda", "SCALEX", "UINMF"]))      # auto on a CPU host
    assert {k: (v["flavor"], v["archive_bytes"]) for k, v in auto.items()} == \
        {k: (v["flavor"], v["archive_bytes"]) for k, v in cpu.items()}
    gpu = _by_env(mtb.env.install(["Matilda", "SCALEX", "UINMF"], flavor="gpu"))
    assert gpu["matilda"]["flavor"] == "gpu"
    assert gpu["matilda"]["archive_bytes"] == 3_000_000_000
    assert gpu["matilda"]["unpacked_bytes"] == 9_000_000_000
    assert gpu["matilda"]["packed_url"] == "https://x/matilda.tar.gz"
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    auto = _by_env(mtb.env.install(["Matilda", "SCALEX", "UINMF"]))      # auto on a GPU host
    assert auto["matilda"]["flavor"] == "gpu" and auto["matilda"]["archive_bytes"] == 3_000_000_000


def test_install_dry_run_null_size_is_unknown(plan_rows, monkeypatch):
    """A published CPU archive whose unpacked size is still null reads as
    unknown - exactly like an env missing from the table."""
    monkeypatch.setattr(envs, "packed_sizes", lambda: {
        "matilda-cpu": {"archive_bytes": 800_000_000, "unpacked_bytes": None}})
    row = _by_env(mtb.env.install(["Matilda"], flavor="cpu"))["matilda"]
    assert row["flavor"] == "cpu" and row["archive_bytes"] == 800_000_000
    assert row["unpacked_bytes"] is None
    assert envs._gb(row["unpacked_bytes"]) == "?"


def test_install_packed_false_has_no_flavour(plan_rows, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: pytest.fail("no probe without archives"))
    rows = mtb.env.install(["Matilda"], packed=False, flavor="cpu")
    assert rows[0]["flavor"] is None and rows[0]["state"] == "build(dry-run)"
    assert rows[0]["archive_bytes"] is None and rows[0]["packed_url"] is None


def test_install_reports_the_installed_flavour_for_existing_envs(envs_dir, monkeypatch,
                                                                 fake_tables):
    prefix = make_prefix(envs_dir, "matilda")
    (prefix / envs.FLAVOR_FILE).write_text("cpu\n")
    monkeypatch.setattr(envs, "install_packed", lambda env, **kw: pytest.fail("no download"))
    row = _by_env(mtb.env.install(["Matilda"], flavor="gpu"))["matilda"]
    assert row["state"] == "have" and row["flavor"] == "cpu"


def test_install_invalid_flavor_is_a_value_error(monkeypatch):
    monkeypatch.setattr(envs, "_check_methods", lambda m: pytest.fail("methods before flavour"))
    with pytest.raises(ValueError) as e:
        mtb.env.install(["Matilda"], flavor="fast")
    assert str(e.value) == "flavor='fast': choose one of 'auto', 'cpu', 'gpu'"


def test_install_run_hands_the_flavour_to_install_packed(envs_dir, monkeypatch, fake_tables):
    seen = []

    def fake_unpack(env, **kw):
        seen.append((env, kw.get("flavor")))
        make_prefix(envs_dir, env)
        return True
    monkeypatch.setattr(envs, "install_packed", fake_unpack)
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build started"))
    rows = mtb.env.install(["Matilda"], packed=True, dry_run=False, flavor="cpu")
    assert seen == [("matilda", "cpu")]
    assert rows[0]["state"] == "PACKED"
    # the outcome row carries the recorded flavour (fake_unpack wrote none)
    assert rows[0]["flavor"] is None
    (envs_dir / "matilda" / envs.FLAVOR_FILE).write_text("cpu\n")
    assert _by_env(mtb.env.install(["Matilda"], flavor="gpu"))["matilda"]["flavor"] == "cpu"


def test_install_signature_and_docs():
    sig = inspect.signature(mtb.env.install)
    assert sig.parameters["flavor"].default == "auto"
    assert sig.parameters["flavor"].kind is inspect.Parameter.KEYWORD_ONLY
    sig = inspect.signature(envs.install_packed)
    assert sig.parameters["flavor"].default == "auto"
    for fn in (envs.install, envs.install_packed, envs.host_has_gpu, envs.resolve_flavor,
               envs.archive_for, envs.installed_flavor, envs.check_flavor):
        doc = inspect.getdoc(fn)
        assert doc and "Returns" in doc, fn.__name__
    assert "flavor" in inspect.getdoc(envs.install) and "flavor" in inspect.getdoc(envs.plan)
    assert "flavor" in inspect.getdoc(envs.doctor) and "flavor" in inspect.getdoc(envs.status)


# ---------------------------------------------------------------- plan / doctor / status show the record
def test_env_tables_show_the_installed_flavour(envs_dir):
    torch = envs.group_for("SCALEX")
    prefix = make_prefix(envs_dir, torch)
    (prefix / envs.FLAVOR_FILE).write_text("cpu\n")
    make_prefix(envs_dir, "matilda")                    # installed, no record
    doc = _by_env(envs.doctor(methods=["SCALEX", "Matilda", "UINMF"]))
    assert doc[torch]["exists"] and doc[torch]["flavor"] == "cpu"
    assert doc["matilda"]["exists"] and doc["matilda"]["flavor"] is None
    assert not doc["scmb_r"]["exists"] and doc["scmb_r"]["flavor"] is None
    assert list(doc[torch]) == ["env", "methods", "exists", "has_lock", "flavor"]
    pl = _by_env(envs.plan(methods=["SCALEX", "Matilda", "UINMF"]))
    assert pl[torch]["flavor"] == "cpu" and pl["matilda"]["flavor"] is None
    assert list(pl[torch]) == ["env", "shared", "methods", "availability", "flavor"]
    st = {r["method"]: r for r in envs.status()}
    assert st["SCALEX"]["flavor"] == "cpu" and st["Matilda"]["flavor"] is None
    assert st["UINMF"]["flavor"] is None
    assert list(st["SCALEX"]) == ["method", "env", "group", "own_env", "exists", "has_lock",
                                  "difficulty", "verified_working", "has_recipe", "flavor"]
    for fn, kw in ((envs.plan, {"methods": ["SCALEX"]}), (envs.doctor, {"methods": ["SCALEX"]}),
                   (envs.status, {})):
        df = fn(**kw, as_frame=True)
        assert "flavor" in df.columns


# ---------------------------------------------------------------- CLI
@pytest.fixture
def cli_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)


def test_cli_install_flavor_flag(cli_linux, monkeypatch, capsys):
    seen = {}

    def fake_install(methods, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr(envs, "install", fake_install)
    for argv, want in ((["env", "install", "--packed"], "auto"),
                       (["env", "install", "--packed", "--flavor", "cpu"], "cpu"),
                       (["env", "install", "--flavor", "gpu", "--methods", "Matilda"], "gpu")):
        assert cli.main(argv) == 0
        assert seen["flavor"] == want, argv
        capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        cli.main(["env", "install", "--flavor", "cuda"])
    assert e.value.code == 2
    assert "invalid choice: 'cuda'" in capsys.readouterr().err


def test_cli_install_help_explains_auto(capsys):
    for sub in ("install", "plan"):
        with pytest.raises(SystemExit) as e:
            cli.main(["env", sub, "--help"])
        assert e.value.code == 0
        out = " ".join(capsys.readouterr().out.split())     # argparse wraps lines
        assert "--flavor {auto,cpu,gpu}" in out
        assert "'auto' (default) = 'cpu' when no NVIDIA GPU is visible on this host" in out
        assert "'<env>-cpu' archive" in out and "mtb.env.host_has_gpu" in out
    with pytest.raises(SystemExit):
        cli.main(["env", "install", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--flavor picks the CPU-only or the CUDA archive" in out


def test_cli_install_dry_run_total_names_the_flavour(cli_linux, plan_rows, monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    assert cli.main(["env", "install", "--packed", "--flavor", "cpu"]) == 0
    cap = capsys.readouterr()
    lines = {l.split()[0]: l for l in cap.out.splitlines()}
    assert "0.8 GB dl" in lines["matilda"] and "2.5 GB disk" in lines["matilda"] \
        and "https://x/matilda-cpu.tar.gz" in lines["matilda"]
    assert "4.5 GB dl" in lines["scmb_torch"] and "https://x/scmb_torch.tar.gz" in lines["scmb_torch"]
    total = [l for l in cap.err.splitlines() if l.startswith("# total")][0]
    assert total.startswith("# total at least: 6.2 GB to download, 2.5 GB on disk (3 archives; "
                            "download size unknown for 0, disk size unknown for 2)")
    assert "; summed the cpu archives; 2 of 3 envs have no CPU archive yet, their GPU archive " \
           "is counted; sizes are the shipped snapshot engine/packed_sizes.json" in total
    assert "(auto:" not in total
    # auto on a CPU host says so; gpu sums the CUDA archives and never falls back
    assert cli.main(["env", "install", "--packed"]) == 0
    total = [l for l in capsys.readouterr().err.splitlines() if l.startswith("# total")][0]
    assert "summed the cpu archives (auto: no NVIDIA GPU visible on this host); 2 of 3" in total
    assert cli.main(["env", "install", "--packed", "--flavor", "gpu"]) == 0
    total = [l for l in capsys.readouterr().err.splitlines() if l.startswith("# total")][0]
    assert total.startswith("# total at least: 8.4 GB to download, 9.0 GB on disk (3 archives; ")
    assert "; summed the gpu archives; sizes are the shipped" in total
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    assert cli.main(["env", "install", "--packed"]) == 0
    total = [l for l in capsys.readouterr().err.splitlines() if l.startswith("# total")][0]
    assert "summed the gpu archives (auto: NVIDIA GPU visible on this host); sizes" in total


def test_cli_plan_flavor(cli_linux, fake_tables, monkeypatch, capsys):
    monkeypatch.setattr(envs, "installed_flavor", lambda env, conda=None: None)
    assert cli.main(["env", "plan", "--methods", "Matilda,SCALEX,UINMF", "--flavor", "cpu"]) == 0
    cap = capsys.readouterr()
    lines = {l.split()[0]: l for l in cap.out.splitlines()}
    assert "0.8 GB dl" in lines["matilda"] and "2.5 GB disk" in lines["matilda"]
    assert "4.5 GB dl" in lines["scmb_torch"] and "0.9 GB dl" in lines["scmb_r"]
    assert cap.err.startswith("# total at least: 6.2 GB download, 2.5 GB on disk (3 archives; "
                              "download size unknown for 0, disk size unknown for 2); summed the "
                              "cpu archives; 2 of 3 envs have no CPU archive yet, their GPU "
                              "archive is counted; sizes are")
    assert cli.main(["env", "plan", "--methods", "Matilda", "--flavor", "gpu"]) == 0
    cap = capsys.readouterr()
    assert "3.0 GB dl" in cap.out and "9.0 GB disk" in cap.out
    assert cap.err.startswith("# total: 3.0 GB download, 9.0 GB on disk (1 archive; download size "
                              "unknown for 0, disk size unknown for 0); summed the gpu archives; ")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    assert cli.main(["env", "plan", "--methods", "Matilda"]) == 0        # auto
    cap = capsys.readouterr()
    assert "0.8 GB dl" in cap.out
    assert "summed the cpu archives (auto: no NVIDIA GPU visible on this host); sizes" in cap.err


def test_cli_status_doctor_plan_print_the_record(cli_linux, envs_dir, capsys):
    torch = envs.group_for("SCALEX")
    (make_prefix(envs_dir, torch) / envs.FLAVOR_FILE).write_text("cpu\n")
    make_prefix(envs_dir, "matilda")
    assert cli.main(["env", "status", "--methods", "SCALEX,Matilda,UINMF"]) == 0
    out = capsys.readouterr().out.splitlines()
    by = {l.split()[1]: l for l in out}
    assert by["SCALEX"].startswith("[x]") and by["SCALEX"].endswith(" flavor=cpu")
    assert by["Matilda"].startswith("[x]") and "flavor=" not in by["Matilda"]
    assert by["UINMF"].startswith("[L]") and "flavor=" not in by["UINMF"]
    assert cli.main(["env", "doctor", "--methods", "SCALEX,Matilda,UINMF"]) == 0
    out = capsys.readouterr().out
    torch_line = [l for l in out.splitlines() if l.startswith(f"[x] {torch}")][0]
    assert torch_line.endswith(" flavor=cpu")
    assert [l for l in out.splitlines() if l.startswith("[x] matilda")][0].count("flavor") == 0
    assert cli.main(["env", "plan", "--methods", "SCALEX,Matilda", "--flavor", "gpu"]) == 0
    out = capsys.readouterr().out
    assert [l for l in out.splitlines() if l.startswith(torch)][0].endswith(" flavor=cpu")
    assert "flavor" not in [l for l in out.splitlines() if l.startswith("matilda")][0]
    assert cli.main(["env", "install", "--packed", "--methods", "SCALEX,Matilda"]) == 0
    out = capsys.readouterr().out
    assert [l for l in out.splitlines() if l.startswith(torch)][0].endswith("[have          ] <- SCALEX flavor=cpu")
    assert [l for l in out.splitlines() if l.startswith("matilda")][0].endswith("<- Matilda")


def test_size_total_line_without_flavor_is_unchanged():
    rows = [{"env": "a"}, {"env": "b", "flavor": "cpu"}]
    sizes = {"a": {"archive_bytes": 1_000_000_000, "unpacked_bytes": 2_000_000_000},
             "b": {"archive_bytes": 5_000_000_000, "unpacked_bytes": None},
             "b-cpu": {"archive_bytes": 1_000_000_000, "unpacked_bytes": None}}
    line = cli._size_total_line(rows, sizes)
    assert line == ("# total at least: 2.0 GB download, 2.0 GB on disk (2 archives; download "
                    "size unknown for 0, disk size unknown for 1); sizes are the shipped "
                    "snapshot engine/packed_sizes.json")
    assert cli._flavor_token("cpu") == " flavor=cpu" and cli._flavor_token(None) == ""
    assert cli._flavor_token("tpu") == ""


# ---------------------------------------------------------------- the shipped placeholders
def test_manifests_carry_the_five_cpu_placeholders():
    urls = json.loads((ROOT / "multibench/engine/packed_urls.json").read_text())
    sizes = json.loads((ROOT / "multibench/engine/packed_sizes.json").read_text())
    for env in CPU_ENVS:
        assert env in urls, env
        assert urls[f"{env}-cpu"] == f"{RELEASE}/{env}-cpu.tar.gz"
        assert f"{env}-cpu" in sizes and set(sizes[f"{env}-cpu"]) == {"archive_bytes", "unpacked_bytes"}
    # until the orchestrator fills a size the placeholder is inert: every
    # flavour resolves to the GPU archive, so nothing changes on any host
    # before the archives exist (the runtime reads null as unknown)
    for env in CPU_ENVS:
        if sizes[f"{env}-cpu"]["archive_bytes"] is None:
            assert envs.archive_for(env, "cpu") == (env, "gpu")
        else:
            assert envs.archive_for(env, "cpu") == (f"{env}-cpu", "cpu")
    assert set(envs.packed_manifest()) == set(urls)
    assert set(envs.packed_sizes()) == set(sizes) - {"_meta"}
