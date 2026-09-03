import re
"""Study-2 CLI work (Ben, Chen, Priya): ``multibench params``, ``convert`` on
an already-canonical file + ``--category``, ``plot --input`` next to the
stored table, and the ``env`` platform guard / sizes / legends.

Every new flag has a test here; ``test_cli_parity`` pins that every flag has
help=.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench
from multibench import cli
from multibench.engine import envs

DARWIN = "method environments are linux-64 conda envs (packed archives + lockfiles); this host is darwin/arm64"


def _sub(name):
    p = cli.build_parser()
    return next(a for a in p._actions if a.dest == "command").choices[name]


def _env_sub(name):
    env = _sub("env")
    return next(a for a in env._actions if a.dest == "env_cmd").choices[name]


# ----------------------------------------------------------------- params
def test_cli_params_prints_every_variant_without_a_selector(capsys):
    rc = cli.main(["params", "Matilda"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "# Matilda vertical:rna+adt: 10 tunable" in out
    assert "# Matilda vertical:rna+atac: 10 tunable" in out
    header = next(l for l in out.splitlines() if l.strip().startswith("key"))
    assert header.split() == ["key", "type", "default", "effective"]
    assert "epochs" in out and "z_dim" in out and "--param KEY=VALUE" in out
    tun = multibench.params_for("Matilda", "vertical", ["rna", "adt"])["tunable"]
    for k in tun:
        assert k in out


def test_cli_params_category_and_modalities_select(capsys):
    rc = cli.main(["params", "Matilda", "--category", "vertical", "--modalities", "rna,protein"])
    out = capsys.readouterr().out
    assert rc == 0 and "vertical:rna+adt" in out and "vertical:rna+atac" not in out
    rc = cli.main(["params", "scBridge", "--category", "diagonal"])
    assert rc == 0 and "scBridge diagonal:-" in capsys.readouterr().out


def test_cli_params_fixed_in_script_and_knobs_for_untunable_method(capsys):
    rc = cli.main(["params", "MOFA2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0 tunable" in out and "(none:" in out
    assert "# fixed in the script" in out and "seed = 42" in out and "main_MOFA2.Rmd:37" in out
    assert "# upstream library knobs" in out and "num_factors" in out


def test_cli_params_formats(capsys):
    rc = cli.main(["params", "Matilda", "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0 and len(rows) == 2
    assert set(rows[0]) >= {"method", "variant", "tunable", "effective", "fixed_in_script"}
    rc = cli.main(["params", "Matilda", "--format", "csv"])
    out = capsys.readouterr().out
    assert rc == 0 and out.splitlines()[0] == "variant,key,type,default,effective"
    assert out.count("\n") == 21           # header + 2 variants x 10 keys


def test_cli_params_errors(capsys):
    rc = cli.main(["params", "Matlda"])
    assert rc == 1 and "did you mean 'Matilda'" in capsys.readouterr().err
    rc = cli.main(["params", "Matilda", "--category", "cross"])
    err = capsys.readouterr().err
    assert rc == 1 and "no variant for category='cross'" in err and "vertical:rna+adt" in err
    rc = cli.main(["params", "Matilda", "--category", "vertcal"])
    assert rc == 1 and "vertical" in capsys.readouterr().err


def test_param_help_points_at_the_params_command():
    for name in ("run", "run-all"):
        act = next(a for a in _sub(name)._actions if a.dest == "param")
        assert "multibench params METHOD" in act.help
        assert "mtb.params_for(METHOD) or" not in act.help
    act = next(a for a in _sub("find")._actions if a.dest == "tunable")
    assert "multibench params METHOD" in act.help


# ----------------------------------------------------------------- convert
def _canonical(path, dtype="float64"):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(6, 9)).astype(dtype),
                         compression="gzip", chunks=True)
        g.create_dataset("features", data=np.array([f"g{i}" for i in range(6)], dtype="S4"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(9)], dtype="S4"))
    return path


def test_cli_convert_canonical_src_is_copied_not_claimed_written(tmp_path, capsys):
    src = _canonical(tmp_path / "adt.h5")
    out = tmp_path / "conv" / "adt.h5"
    rc = cli.main(["convert", str(src), str(out)])
    o = capsys.readouterr().out
    assert rc == 0 and out.is_file()
    assert o.startswith(f"copied {src} -> {out}") and "already canonical" in o
    assert "wrote" not in o
    with h5py.File(out) as f:
        assert f["matrix/data"].dtype == np.float64 and f["matrix/data"].shape == (6, 9)


def test_cli_convert_canonical_src_honours_dtype(tmp_path, capsys):
    src = _canonical(tmp_path / "adt.h5")
    out = tmp_path / "adt32.h5"
    rc = cli.main(["convert", str(src), str(out), "--dtype", "float32"])
    o = capsys.readouterr().out
    assert rc == 0 and "re-encoded as float32" in o
    with h5py.File(out) as f, h5py.File(src) as s:
        assert f["matrix/data"].dtype == np.float32
        assert np.allclose(f["matrix/data"][...], s["matrix/data"][...])
        assert list(f["matrix/features"][...]) == list(s["matrix/features"][...])
        assert list(f["matrix/barcodes"][...]) == list(s["matrix/barcodes"][...])


def test_cli_convert_canonical_src_into_directory_and_same_path(tmp_path, capsys):
    src = _canonical(tmp_path / "x.h5")
    d = tmp_path / "MYDATA"; d.mkdir()
    rc = cli.main(["convert", str(src), str(d), "--modality", "protein"])
    o = capsys.readouterr().out
    assert rc == 0 and (d / "adt.h5").is_file() and f"-> {d / 'adt.h5'}" in o
    # --category vertical: an ATAC modality lands in atac.h5
    rc = cli.main(["convert", str(src), str(d), "--modality", "peak", "--category", "vertical"])
    assert rc == 0 and (d / "atac.h5").is_file() and not (d / "atac_peak.h5").exists()
    capsys.readouterr()
    # OUT == SRC: nothing written, said plainly
    rc = cli.main(["convert", str(src), str(src)])
    o = capsys.readouterr().out
    assert rc == 0 and o.startswith("already canonical - nothing written") and "wrote" not in o
    # a directory without --modality is a usage error, not a file named MYDATA
    with pytest.raises(SystemExit) as ei:
        cli.main(["convert", str(src), str(d)])
    assert ei.value.code == 2 and "pass --modality" in capsys.readouterr().err


def test_cli_convert_category_forwarded_to_export_dataset(tmp_path, capsys):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(20, 8)).astype(float))
    a.var_names = [f"g{i}" for i in range(8)]
    a.obs_names = [f"c{i}" for i in range(20)]
    a.obsm["atac"] = rng.poisson(1.0, size=(20, 5)).astype(float)
    a.uns["atac_names"] = [f"chr1:{i * 1000}-{i * 1000 + 300}" for i in range(5)]
    a.obs["ct"] = pd.Categorical(rng.choice(["T", "B"], size=20))
    src = tmp_path / "mo.h5ad"
    a.write_h5ad(src)
    d = tmp_path / "SYNMO"
    rc = cli.main(["convert", str(src), str(d), "--rna", "X", "--atac", "obsm:atac",
                   "--atac-kind", "gene_activity", "--labels", "obs:ct",
                   "--category", "vertical"])
    out = capsys.readouterr().out
    assert rc == 0 and "wrote dataset folder" in out
    names = sorted(p.name for p in d.iterdir())
    assert "atac.h5" in names and "atac_gas.h5" not in names
    act = next(a for a in _sub("convert")._actions if a.dest == "category")
    assert "atac.h5" in act.help


# ----------------------------------------------------------------- plot overlay
def _long(method, n=3):
    return pd.DataFrame({"metric": ["ARI", "NMI", "ASW"][:n], "value": [0.5, 0.6, 0.7][:n],
                         "method": method, "dataset": "D1", "category": "vertical"})


def test_cli_plot_input_with_category_concatenates_onto_stored(tmp_path, monkeypatch, capsys):
    from multibench import plot as plot_ns
    seen = {}
    monkeypatch.setattr(multibench, "load_results",
                        lambda **kw: pd.concat([_long("A"), _long("B")], ignore_index=True))
    monkeypatch.setattr(plot_ns, "bubble", lambda df, **kw: seen.update(df=df, kw=kw))
    mine = tmp_path / "mine.csv"
    _long("Mine").to_csv(mine, index=False)
    rc = cli.main(["plot", "bubble", "--category", "vertical", "--dataset", "D1",
                   "--source", "rerun", "--input", str(mine), "--out", str(tmp_path / "x.png")])
    cap = capsys.readouterr()
    assert rc == 0
    assert sorted(seen["df"]["method"].unique()) == ["A", "B", "Mine"]
    assert "# overlay: 3 row(s) from --input (methods: Mine)" in cap.err
    assert "stored vertical table (6 rows, source=rerun)" in cap.err
    assert "overlay" not in cap.out


def test_cli_plot_input_is_repeatable_and_dataset_filters_inputs(tmp_path, monkeypatch, capsys):
    from multibench import plot as plot_ns
    seen = {}
    monkeypatch.setattr(plot_ns, "bar", lambda df, **kw: seen.update(df=df))
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _long("A").to_csv(a, index=False)
    other = _long("B"); other["dataset"] = "D2"
    pd.concat([_long("B"), other]).to_csv(b, index=False)
    rc = cli.main(["plot", "bar", "--input", str(a), "--input", str(b), "--dataset", "D1",
                   "--out", str(tmp_path / "x.png")])
    assert rc == 0
    df = seen["df"]
    assert sorted(df["method"].unique()) == ["A", "B"] and set(df["dataset"]) == {"D1"}
    assert len(df) == 6
    act = next(x for x in _sub("plot")._actions if x.dest == "input")
    assert "repeatable" in act.help and "concatenated" in act.help
    assert "replaces" not in act.help


# ----------------------------------------------------------------- env: platform / sizes / legend
def test_env_doctor_and_install_warn_off_linux(monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    rc = cli.main(["env", "doctor", "--methods", "Matilda"])
    cap = capsys.readouterr()
    assert rc == 0
    assert cap.out.splitlines()[0].startswith("[L] matilda") and "# legend:" in cap.out
    assert cap.err.startswith("warning: ") and "linux-64" in cap.err and "--force" in cap.err
    assert "warning" not in cap.out
    # --run refuses with error: on stderr, exit 1, nothing on stdout
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    rc = cli.main(["env", "install", "--run", "--methods", "Matilda"])
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == "" and "error: " in cap.err and "linux-64" in cap.err
    # --force skips the guard (build stubbed)
    built = []
    monkeypatch.setattr(envs, "_run_all", lambda cmds: built.append(cmds))
    rc = cli.main(["env", "install", "--run", "--force", "--methods", "Matilda"])
    cap = capsys.readouterr()
    assert rc == 0 and built and "[BUILD" in cap.out


def test_env_commands_silent_on_linux(monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    for argv in (["env", "doctor", "--methods", "Matilda"], ["env", "plan", "--methods", "Matilda"],
                 ["env", "install", "--methods", "Matilda"], ["env", "status", "--methods", "Matilda"]):
        rc = cli.main(argv)
        cap = capsys.readouterr()
        assert rc == 0 and "warning:" not in cap.err, argv


def test_force_flag_everywhere_it_matters():
    for name in ("install", "create", "create-group"):
        act = next(a for a in _env_sub(name)._actions if a.dest == "force")
        assert "linux-64" in act.help


def test_env_install_packed_dry_run_shows_sizes_url_and_total(monkeypatch, capsys):
    rows = [{"env": "scmb_torch", "methods": ["SCALEX"], "exists": False, "has_lock": True},
            {"env": "mystery", "methods": ["X"], "exists": False, "has_lock": True},
            {"env": "have_it", "methods": ["Z"], "exists": True, "has_lock": True}]
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "create_all", lambda **kw: rows)
    monkeypatch.setattr(envs, "install_packed", lambda env, **kw: pytest.fail("no download"))
    monkeypatch.setattr(cli, "_packed_manifest",
                        lambda: {"scmb_torch": "https://x/t.tar.gz", "mystery": "https://x/m.tar.gz"})
    envs.packed_sizes.cache_clear()
    monkeypatch.setattr(envs, "packed_sizes",
                        lambda: {"scmb_torch": {"archive_bytes": 900_000_000,
                                                "unpacked_bytes": 3_100_000_000}})
    rc = cli.main(["env", "install", "--packed"])
    cap = capsys.readouterr()
    assert rc == 0
    lines = {l.split()[0]: l for l in cap.out.splitlines()}
    assert "0.9 GB dl" in lines["scmb_torch"] and "3.1 GB disk" in lines["scmb_torch"]
    assert "https://x/t.tar.gz" in lines["scmb_torch"]
    assert "? dl" in lines["mystery"] and "? disk" in lines["mystery"]
    assert "dl" not in lines["have_it"]
    # unknowns are counted PER COLUMN: the row printing '? disk' is a disk unknown
    assert ("# total at least: 0.9 GB to download, 3.1 GB on disk (2 archives; "
            "download size unknown for 1, disk size unknown for 1)") in cap.err
    assert not any(l.startswith("#") for l in cap.out.splitlines())


def test_env_plan_shows_sizes_and_total(monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    rc = cli.main(["env", "plan", "--methods", "UINMF,sciPENN,Matilda"])
    cap = capsys.readouterr()
    assert rc == 0
    lines = {l.split()[0]: l for l in cap.out.splitlines()}
    assert "0.9 GB dl" in lines["scmb_r"] and "2.1 GB dl" in lines["env_sciPENN"]
    assert re.search(r"(\d+(\.\d+)? [KMG]B|\?) dl", lines["matilda"])   # size when known, ? otherwise
    # "# total: X download" when every size is known, "# total at least: ..."
    # when some are still null in the shipped snapshot
    assert cap.err.startswith("# total") and "GB download" in cap.err
    # counts depend on the shipped snapshot; the disk count must equal the '? disk' rows
    m = re.search(r"download size unknown for (\d+), disk size unknown for (\d+)", cap.err)
    assert m, cap.err
    assert int(m.group(2)) == sum("? disk" in l for l in cap.out.splitlines())
    assert int(m.group(1)) == sum("? dl" in l for l in cap.out.splitlines())
    assert lines["scmb_r"].endswith("<- UINMF")


def test_env_status_legend_explains_every_tag_on_stderr(monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    rc = cli.main(["env", "status", "--methods", "Matilda,MIRA,UINMF"])
    cap = capsys.readouterr()
    assert rc == 0
    assert len(cap.out.strip().splitlines()) == 3            # rows only on stdout
    assert "old-scvi" in cap.out and "blocked-script" in cap.out and " R" in cap.out
    assert cap.err.startswith("# legend:")
    for tag in ("old-scvi", "blocked-script", "R"):
        assert f"{tag} = {envs.DIFFICULTY[tag]}" in cap.err
    assert "verified_working" in cap.err
    desc = _env_sub("status").description
    for tag in envs.DIFFICULTY:
        assert tag in desc
