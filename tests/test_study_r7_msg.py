"""Round 7, work package 'msg': messages read as plain sentences.

- R7-03: the file check of a variant that reads both ATAC files says so,
  instead of pointing to method_info(m)['atac'] (which says 'peak').
- R7-07: the scripts-ref note of a run-all dry run comes after the header.
- R7-08: the dataset-mixing warnings of the plots and to_long's warning for
  scores with no record start with a capital and do not join facts with ', so'.
- R7-09: export_dataset's full-folder error and the raw-counts warning.
- R7-10: the Linux-only refusals and the missing-environment texts.
- R7-11: the cell-check ValueErrors start with the method, not with a
  'method/dataset/category:' prefix.
"""
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry
from multibench.engine import runner as R

GENES = [f"GENE{i}" for i in range(60)]
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
DARWIN = ("method environments are linux-64 conda envs (packed archives + lockfiles); "
          "this host is darwin/arm64")
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _h5(path, feats, bars, value=1.0):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.full((len(feats), len(bars)), value))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))


def _labels(path, n):
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(path, index=False)


def _lung(root, n_rna=20, n_atac=20, atac_prefix="a"):
    """S3's LUNG: diagonal, rna.h5 + atac_peak.h5 and the two label files."""
    d = root / "LUNG"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", GENES, [f"r{i}" for i in range(n_rna)])
    _h5(d / "atac_peak.h5", PEAKS, [f"{atac_prefix}{i}" for i in range(n_atac)])
    _labels(d / "rna_cty.csv", n_rna)
    _labels(d / "atac_cty.csv", n_atac)
    return d


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Every env installed, a GPU, no scripts ref."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)


# ======================================================================= R7-03
@pytest.mark.parametrize("method", ["MultiMAP", "Seurat_v3"])
def test_both_file_variants_say_they_read_both_files(tmp_path, monkeypatch, method, pinned):
    _lung(tmp_path)
    both = f"{method} reads both atac_peak.h5 and atac_gas.h5."
    for cli_mode in (False, True):
        monkeypatch.setattr(config, "_CLI", cli_mode)
        with pytest.raises(FileNotFoundError) as e:
            mtb.inputs_for("LUNG", "diagonal", method, data_path=tmp_path, check=True)
        msg = str(e.value)
        assert "reads both atac_peak.h5 and atac_gas.h5" in msg
        assert msg.endswith(both), msg
        assert "method_info" not in msg and "multibench info" not in msg, msg
    monkeypatch.setattr(config, "_CLI", False)
    row = _quiet(mtb.scan, "LUNG", "diagonal", methods=[method], data_path=tmp_path,
                 verbose=False).iloc[0]
    assert both in row["files_reason"], row["files_reason"]
    assert "method_info" not in row["files_reason"]


def test_single_file_variant_keeps_the_method_info_pointer(tmp_path, pinned):
    _lung(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        mtb.inputs_for("LUNG", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    assert str(e.value).endswith(
        'method_info("SCALEX")["atac"] says which ATAC SCALEX needs.')


# ======================================================================= R7-07
WRONG_REF = "The method scripts are at 0000000, not deadbeef (MULTIBENCH_SCRIPTS_REF)."


@pytest.fixture
def wrong_ref(monkeypatch):
    monkeypatch.setattr(config, "scripts_ref_problem", lambda repo=None: WRONG_REF)
    return WRONG_REF


def test_cli_run_all_dry_run_starts_with_the_header(wrong_ref, capsys):
    rc = _quiet(cli.main, ["run-all", "D11", "--category", "vertical", "--methods",
                           "totalVI", "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 0
    lines = err.splitlines()
    assert lines[0].startswith("# Dry run. Nothing was executed."), err
    assert err.count(WRONG_REF) == 1, err
    assert f"# {WRONG_REF}" in lines[1:], err


def test_python_run_all_dry_run_prints_the_ref_note_only_when_verbose(wrong_ref, capsys):
    df = mtb.run_all("D11", "vertical", methods=["totalVI"], dry_run=True, verbose=False)
    cap = capsys.readouterr()
    assert WRONG_REF not in cap.out and WRONG_REF not in cap.err
    assert df["reason"].str.contains(WRONG_REF[:-1], regex=False).all()
    mtb.run_all("D11", "vertical", methods=["totalVI"], dry_run=True, verbose=True)
    cap = capsys.readouterr()
    assert WRONG_REF not in cap.err
    lines = cap.out.splitlines()
    count = next(i for i, ln in enumerate(lines) if ln.startswith("[run_all] Dry run: "))
    assert lines.count(f"[run_all] {WRONG_REF}") == 1, cap.out
    assert lines.index(f"[run_all] {WRONG_REF}") > count


# ======================================================================= R7-08
def _stored(dataset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results("vertical", dataset=dataset, source="rerun")


def _mine(method="PriyaNet", dataset="MYCITE"):
    return pd.DataFrame([
        {"metric": m, "value": v, "method": method, "dataset": dataset,
         "category": "vertical", "clustering": "default", "source": "user",
         "scored_with": "leidenalg/sweep/0.3.1"}
        for m, v in zip(("ARI", "NMI", "ASW", "cLISI"), (0.351, 0.5, 0.55, 0.9))])


def _warned(fn, *a, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    plt.close("all")
    return [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def test_plot_warnings_on_mixed_datasets_are_capitalized_sentences():
    df = pd.concat([_stored("D11"), _mine()], ignore_index=True)
    seen = []
    for fn, kw in [(mtb.plot.bubble, {}), (mtb.plot.build_table, {}),
                   (mtb.plot.build_table, {"aggregate": "summary"}),
                   (mtb.plot.build_table, {"aggregate": "summary",
                                           "overall": "mean_overall"}),
                   (mtb.plot.bar, {})]:
        msgs = _warned(fn, df, **kw)
        seen += msgs
        for m in msgs:
            assert m[:1].isupper(), (fn, kw, m)
            assert "(D11, MYCITE)" not in m, m
            assert ", so " not in m.split(".")[0], m
    assert ("The rows come from 2 datasets, D11 and MYCITE, that share no method. The "
            "figure ranks unrelated rows against each other. Plot each dataset on its "
            "own, or score your method on D11 and add it to that table.") in seen
    assert ("Dataset MYCITE has only one method, PriyaNet. Its row is ranked against rows "
            "from other datasets. Plot it with methods scored on the same dataset.") in seen
    assert ("Dataset MYCITE has only one method, PriyaNet. Its rank there is always the "
            "lowest. Plot it with methods scored on the same dataset.") in seen
    assert ("Dataset MYCITE has only one method, PriyaNet. Its Overall there is always "
            "1.0. Plot it with methods scored on the same dataset.") in seen


def test_no_overlap_of_own_datasets_keeps_the_same_cells_sentence():
    from multibench.plot import style
    parts = {"MINE_A": pd.DataFrame(index=["X"]), "MINE_B": pd.DataFrame(index=["Y"])}
    assert style._no_overlap_message(parts) == (
        "The rows come from 2 datasets, MINE_A and MINE_B, that share no method. The "
        "figure ranks unrelated rows against each other. Plot each dataset on its own, "
        "or score the same methods on every dataset. If these datasets hold the same "
        "cells, give their rows one dataset name first.")


def test_to_long_warning_for_scores_with_no_record_is_sentences():
    wide = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    msgs = _warned(mtb.to_long, wide, method="Mine", dataset="MYCITE",
                   category="vertical")
    assert msgs == [
        "These scores carry no record of how they were scored. Their scored_with is "
        "\"unknown\". A wide CSV read back loses that record. To keep it, save the "
        "mtb.to_long(...) table instead of the wide one."]


# ======================================================================= R7-09
ad = pytest.importorskip("anndata")


def _cite(n=40, log=False):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(n, 30)).astype(float))
    a.var_names = [f"g{i}" for i in range(30)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type"] = ["T", "B"] * (n // 2)
    a.obsm["protein"] = pd.DataFrame(rng.poisson(4.0, size=(n, 6)).astype(float),
                                     index=a.obs_names, columns=[f"CD{i}" for i in range(6)])
    if log:
        a.layers["counts"] = a.X.copy()
        a.X = np.log1p(a.X)
    return a


def test_full_folder_error_is_sentences_in_python(tmp_path):
    a = _cite()
    out = tmp_path / "MYCITE"
    ingest.export_dataset(a, out, adt="obsm:protein", labels="obs:cell_type")
    with pytest.raises(FileExistsError) as e:
        ingest.export_dataset(a, out, adt="obsm:protein", labels="obs:cell_type")
    msg = str(e.value)
    assert msg == (f"{out} already holds rna.h5, adt.h5 and cty.csv. Pass overwrite=True "
                   f"to replace them.")
    assert "[" not in msg and ";" not in msg


def test_full_folder_error_is_sentences_on_the_cli(tmp_path, capsys):
    a = _cite()
    src = tmp_path / "cite.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "MYCITE"
    argv = ["convert", str(src), str(out), "--rna", "X", "--adt", "obsm:protein",
            "--labels", "obs:cell_type"]
    assert _quiet(cli.main, argv) == 0
    capsys.readouterr()
    rc = _quiet(cli.main, argv)
    err = capsys.readouterr().err
    assert rc == 1
    line = next(ln for ln in err.splitlines() if "already holds" in ln)
    assert line.endswith("already holds rna.h5, adt.h5 and cty.csv. Pass --overwrite to "
                         "replace them."), line
    assert "[" not in line and ";" not in line


def test_one_existing_file_is_named_alone(tmp_path):
    a = _cite()
    out = tmp_path / "ONE"
    ingest.export_dataset(a, out, adt=None, labels=None)
    with pytest.raises(FileExistsError) as e:
        ingest.export_dataset(a, out, adt=None, labels=None)
    assert str(e.value) == f"{out} already holds rna.h5. Pass overwrite=True to replace it."


def _count_warnings(fn, *a, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*a, **kw)
    return [str(w.message) for w in rec if "whole numbers" in str(w.message)]


def test_raw_counts_warning_of_an_anndata_export(tmp_path):
    a = _cite(log=True)
    (msg,) = _count_warnings(ingest.export_dataset, a, tmp_path / "LOG",
                             adt="obsm:protein", labels="obs:cell_type")
    assert msg == ("The rna values are not whole numbers, so they look log-normalised. "
                   "The methods normalise raw counts themselves. Export raw counts, for "
                   "example with rna='layer:counts'.")
    assert "(log-normalised?)" not in msg and "e.g." not in msg and "MuData" not in msg
    (msg,) = _count_warnings(ingest.to_canonical, a, tmp_path / "x.h5", modality="rna")
    assert msg.endswith("Export raw counts, for example with layer='counts'.")


def test_raw_counts_warning_names_the_mudata_form_for_a_mudata_selector(tmp_path):
    mu = pytest.importorskip("mudata")
    a = _cite(log=True)
    rna = ad.AnnData(a.X.copy(), obs=a.obs[[]].copy(), var=a.var.copy())
    adt = ad.AnnData(a.obsm["protein"].to_numpy(), obs=a.obs[[]].copy(),
                     var=pd.DataFrame(index=[f"CD{i}" for i in range(6)]))
    m = mu.MuData({"gex": rna, "adt": adt})
    msgs = _count_warnings(ingest.export_dataset, m, tmp_path / "MU", rna="gex",
                           adt="adt")
    assert msgs == ["The rna values are not whole numbers, so they look log-normalised. "
                    "The methods normalise raw counts themselves. Export raw counts, for "
                    "example with rna='mod:gex.layer:counts'."]


def test_raw_counts_warning_uses_the_cli_spelling(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CLI", True)
    a = _cite(log=True)
    (msg,) = _count_warnings(ingest.export_dataset, a, tmp_path / "LOG",
                             adt="obsm:protein", labels="obs:cell_type")
    assert msg.endswith("Export raw counts, for example with --rna layer:counts.")


# ======================================================================= R7-10
@pytest.mark.parametrize("host,name", [("darwin/arm64", "macOS"), ("win32/AMD64", "Windows"),
                                       ("cygwin/x86_64", "Windows"),
                                       ("freebsd14/amd64", "freebsd14")])
def test_linux_only_sentences_name_the_system(monkeypatch, host, name):
    monkeypatch.setattr(envs, "host_platform_problem",
                        lambda: f"method environments are linux-64 only; this host is {host}")
    assert R.linux_only_sentence() == f"Methods run only on Linux, and this computer runs {name}."
    assert envs.linux_only_text(envs.host_platform_problem()) == (
        f"Method environments run only on Linux, and this computer runs {name}.")


@pytest.fixture
def off_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)


def test_run_refusal_off_linux_has_no_parenthesis_or_semicolon(off_linux, tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(R.envs, "installed_envs", lambda conda=None: ["base"])
    inp = mtb.inputs_for("D11", "vertical", "totalVI")
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=inp, out_dir=str(tmp_path / "out"))
    assert str(e.value) == ("Methods run only on Linux, and this computer runs macOS. Run "
                            "this call on a Linux machine. dry_run=True previews the "
                            "method's command here.")
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=inp, out_dir=str(tmp_path / "out"))
    assert str(e.value) == ("Methods run only on Linux, and this computer runs macOS. Run "
                            "this command on a Linux machine. --dry-run previews the "
                            "method's command here.")


def test_run_all_refusal_off_linux_has_no_parenthesis_or_semicolon(off_linux, tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", methods=["totalVI"], out_dir=tmp_path, verbose=False)
    msg = str(e.value)
    assert "(this computer is" not in msg
    assert "Methods run only on Linux, and this computer runs macOS." in msg
    assert ";" not in msg, msg


def test_install_refusal_off_linux_has_no_parenthesis_or_semicolon(off_linux, monkeypatch,
                                                                   capsys):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    with pytest.raises(RuntimeError) as e:
        envs.create_all(methods=["Matilda"], dry_run=False)
    assert str(e.value) == ("Method environments run only on Linux, and this computer runs "
                            "macOS. Run the install on a Linux machine. force=True tries "
                            "anyway.")
    rc = cli.main(["env", "install", "--methods", "Matilda", "--run"])
    err = capsys.readouterr().err
    assert rc == 1
    assert ("Method environments run only on Linux, and this computer runs macOS. Run the "
            "install on a Linux machine. --force tries anyway.") in err
    assert "(this computer is" not in err and ";" not in err


def test_env_reason_on_linux_leads_with_the_install_command(monkeypatch):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: None)
    text = W._env_hint("scmb_r", "StabMap", "mosaic")
    assert text == ("Environment scmb_r is not installed. Run multibench env install "
                    "--methods StabMap --packed --run. Use --category mosaic to install "
                    "the environments of every mosaic method.")
    assert "multibench env install --methods StabMap --packed --run." in text[:120]
    assert W._env_hint("scmb_r", "StabMap", None) == (
        "Environment scmb_r is not installed. Run multibench env install --methods "
        "StabMap --packed --run.")
    # every method's install command fits in the first 120 characters
    for m in registry.list_methods():
        env = envs.group_for(m)
        cmd = f"multibench env install --methods {m} --packed --run."
        assert cmd in W._env_hint(env, m, "vertical")[:120], m


def test_scan_env_reason_on_linux(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    row = _quiet(mtb.scan, "D11", "vertical", methods=["totalVI"], verbose=False).iloc[0]
    assert row["env_reason"].startswith(f"Environment {row['env']} is not installed. ")
    assert "doctor" not in row["env_reason"]


def test_run_error_on_linux_is_sentences(tmp_path, monkeypatch):
    monkeypatch.setattr(R.envs, "installed_envs", lambda conda=None: ["base"])
    inp = mtb.inputs_for("D11", "vertical", "totalVI")
    env = envs.group_for("totalVI")
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=inp, out_dir=str(tmp_path / "out"))
    assert str(e.value) == (
        f"Environment {env} of totalVI is not installed. Run multibench env install "
        f"--methods totalVI --packed --run. In Python, call "
        f"mtb.env.install(['totalVI'], dry_run=False).")
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=inp, out_dir=str(tmp_path / "out"))
    assert str(e.value) == (
        f"Environment {env} of totalVI is not installed. Run multibench env install "
        f"--methods totalVI --packed --run.")


# ======================================================================= R7-11
def test_same_cells_error_starts_with_the_method(tmp_path, pinned):
    _lung(tmp_path, n_rna=20, n_atac=16)
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("LUNG", "diagonal", "Seurat_v5", data_path=tmp_path, check=True)
    msg = str(e.value)
    assert msg == ("Seurat_v5 needs RNA and ATAC from the same cells as its bridge. In "
                   "LUNG, rna.h5 and atac_peak.h5 share 0 of 20 and 16 cells.")
    row = _quiet(mtb.scan, "LUNG", "diagonal", methods=["Seurat_v5"], data_path=tmp_path,
                 verbose=False).iloc[0]
    assert row["files_reason"] == f"ValueError: {msg}"
    assert row["reason"].startswith(msg)


def _gas(d, bars):
    _h5(d / "atac_gas.h5", GENES, bars)


def test_gas_order_error_is_sentences(tmp_path, pinned):
    d = _lung(tmp_path)
    _gas(d, [f"a{i}" for i in reversed(range(20))])
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("LUNG", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    msg = str(e.value)
    assert msg == ("SCALEX reads atac_gas.h5 of LUNG, which lists the ATAC cells in another "
                   "order than atac_peak.h5. atac_cty.csv follows atac_peak.h5. Write "
                   "atac_gas.h5 again with mtb.io.to_canonical(..., modality='gas').")
    assert ";" not in msg


def test_gas_other_cells_error_is_sentences(tmp_path, pinned):
    d = _lung(tmp_path)
    _gas(d, [f"x{i}" for i in range(18)])
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("LUNG", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    msg = str(e.value)
    assert msg.startswith("SCALEX reads atac_gas.h5 of LUNG, which holds other cells than "
                          "atac_peak.h5.")
    assert msg.endswith(".") and ";" not in msg and " - " not in msg
    assert "LUNG/diagonal" not in msg


def test_orientation_error_starts_with_the_method(tmp_path, pinned):
    d = tmp_path / "TRANS"
    d.mkdir()
    bars = [f"c{i}" for i in range(20)]
    with h5py.File(d / "rna.h5", "w") as f:        # cells x features: the easy mistake
        f.create_dataset("matrix/data", data=np.ones((20, 60)))
        f.create_dataset("matrix/features", data=np.array(GENES, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))
    _h5(d / "adt.h5", [f"p{i}" for i in range(6)], bars)
    _labels(d / "cty.csv", 20)
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("TRANS", "vertical", "totalVI", data_path=tmp_path, check=True)
    msg = str(e.value)
    assert msg == ("totalVI reads rna.h5 of TRANS, which stores matrix/data as cells x "
                   "features, shape (20, 60). The file lists 60 features and 20 cells. Its "
                   "matrix/data needs features x cells, shape (60, 20). Re-export it with "
                   "mtb.io.to_canonical(src, dst), or transpose matrix/data.")
    assert " - " not in msg and "TRANS/vertical" not in msg
