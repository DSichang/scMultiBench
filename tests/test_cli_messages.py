"""The command line: help texts, evaluate, convert, plot, config, cmd templates."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry  # noqa: F401


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _d11():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results("vertical", dataset="D11")


def _mine(base, name="Mine", scored_with="leidenalg/search/0.4.0"):
    rows = base[base.method == "Matilda"].assign(method=name, source="user")
    if scored_with is not None:
        rows = rows.assign(scored_with=scored_with)
    return rows


MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.run(method, category, inputs=inputs, out_dir=str(tmp_path / "o"),
                   params=params, dry_run=True, cmd_template="conda run -n x {cmd}")


from tests.test_evaluate_and_io_messages import _adata  # noqa: E402 - an AnnData of 60 cells

# --- command line: X15, X17-X24, X31, X33, X53, X77, X78, X95, X97, X98 ------

def _cli(argv, capsys):
    from multibench import cli
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


def _help(*cmd):
    from multibench import cli
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        cli.main([*cmd, "--help"])
    return " ".join(buf.getvalue().split())


def test_x15_cli_plot_renames_your_clashing_rows(tmp_path, capsys):
    d11 = _d11()
    rows = d11[d11.method == "scMM"].assign(source="user")
    rows.to_csv(tmp_path / "mine.csv", index=False)
    rc, _, err = _cli(["plot", "bubble", "--input", str(tmp_path / "mine.csv"),
                       "--category", "vertical", "--dataset", "D11",
                       "--out", str(tmp_path / "x.png")], capsys)
    assert rc == 0, err
    assert ("# Your rows of scMM are drawn as scMM (this run): the stored results have "
            "the same method and dataset.") in err


def test_x17_x18_x53_x95_x97_x98_help_texts():
    # c1 (C35): run --help is its deprecated line
    assert "Deprecated: use multibench run-all DATASET --methods M." in _help("run")
    assert ("the dataset folder: its path, or its name under the data root "
            "(multibench config)") in _help("run-all")
    assert "--data-path" not in _help("run-all")
    assert "integration category: vertical (several" not in _help("scan")
    assert "vertical, diagonal, mosaic or cross (multibench layout explains them)" in \
        _help("scan")
    ev = _help("evaluate")
    assert ("--method METHOD fills the method column: a package method id, or your own "
            "name. A package method, also as 'StabMap (this run)', reads --dataset's label "
            "files in its cell order, with --category") in ev
    assert "or with --labels any row name" not in ev
    assert "--leiden-flavor" not in ev                    # hidden in 0.4.0
    assert "--n-batches" not in _help("export")           # hidden in 0.4.0


def test_x19_cli_evaluate_takes_the_embedding_first(tmp_path, capsys):
    import numpy as np
    np.save(tmp_path / "emb.npy", np.random.default_rng(6).normal(size=(30, 3)))
    pd.DataFrame({"x": ["a", "b", "c"] * 10}).to_csv(tmp_path / "l.csv", index=False)
    rc, out, err = _cli(["evaluate", str(tmp_path / "emb.npy"), "--labels",
                         str(tmp_path / "l.csv"), "--metrics", "cLISI"], capsys)
    assert rc == 0 and "cLISI" in out, err
    rc, out2, _ = _cli(["evaluate", "--embedding", str(tmp_path / "emb.npy"), "--labels",
                        str(tmp_path / "l.csv"), "--metrics", "cLISI"], capsys)
    assert rc == 0 and out2 == out
    assert "--embedding" not in _help("evaluate")
    with pytest.raises(SystemExit):
        _cli(["evaluate", "a.npy", "--embedding", "b.npy"], capsys)


def test_x20_x24_cli_run_dry_run(capsys):
    rc, _, err = _cli(["run", "--method", "totalVI", "--category", "vertical",
                       "--input", "rna=a.h5", "--dry-run"], capsys)
    assert rc == 1 and "# Dry run" not in err             # no header before the error
    rc, out, err = _cli(["run", "--method", "totalVI", "--category", "vertical",
                         "--input", "rna=a.h5", "--input", "adt=b.h5", "--dry-run",
                         "--runner", "srun {env_cmd}"], capsys)
    assert rc == 0 and out.startswith("srun ") and "<out_dir>" in out
    assert ("warning: run --runner is deprecated and will be removed in 0.5. Use a job "
            "script that calls multibench run-all." in err)
    assert "--cmd-template" not in _help("run") and "--runner" not in _help("run")


def test_x21_x22_cli_params_tokens_and_int_defaults(capsys):
    rc, out, err = _cli(["params", "Multigrate", "--category", "cross", "--modalities",
                         "rna,adt"], capsys)
    assert rc == 0, err
    assert "cross:rna1+rna2+rna3+adt1+adt2+adt3" in out
    assert "256.000" not in out and " 256\n" in out


def test_x27_x31_x33_cli_env_install(monkeypatch, capsys):
    from multibench.engine import envs
    assert "--dry-run" in _help("env", "install")
    assert "--force" not in _help("env", "install") and "--run" not in _help("env", "install")
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "this host is darwin/arm64")
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    rc, out, _ = _cli(["env", "doctor", "--methods", "StabMap"], capsys)
    assert "# next (on the Linux machine): multibench env install --methods StabMap" in out
    assert "--run" not in out
    rc, _, err = _cli(["env", "install", "--methods", "StabMap", "--run"], capsys)
    assert rc == 1 and "run only on Linux" in err and "deprecated" not in err


def test_x77_x78_cli_convert_atac_from_and_no_rna(tmp_path, capsys):
    rna, atac = _adata(), _adata()
    atac.var_names = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(20)]
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    rc, out, err = _cli(["convert", str(tmp_path / "rna.h5ad"), str(tmp_path / "V"),
                         "--rna", "X", "--atac-from", str(tmp_path / "atac.h5ad"),
                         "--atac-kind", "peak", "--category", "vertical"], capsys)
    assert rc == 0 and "atac.h5" in out, err
    rc, out, err = _cli(["convert", str(tmp_path / "atac.h5ad"), str(tmp_path / "A"),
                         "--atac", "X", "--atac-kind", "peak", "--category", "vertical"],
                        capsys)
    assert rc == 0 and "# No RNA written: pass --rna X (or a layer) to write it." in err


# X43: config --get takes only the settings it prints
def test_config_get(capsys):
    from multibench import cli
    assert cli.main(["config", "--get", "leiden_flavor"]) == 0
    assert capsys.readouterr().out.strip() == str(mtb.config.DEFAULT.leiden_flavor)
    with pytest.raises(SystemExit):
        cli.main(["config", "--get", "result_path"])


# X19: the --output warning names what replaces it
def test_evaluate_output_says_pass_the_embedding_first(tmp_path, capsys):
    import numpy as np
    from multibench import cli
    emb = tmp_path / "e.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(30, 4)))
    lab = tmp_path / "l.csv"
    lab.write_text("x\n" + "\n".join("ab"[i % 2] for i in range(30)) + "\n")
    cli.main(["evaluate", "--output", str(emb), "--labels", str(lab), "--metrics", "cLISI"])
    assert ("--output is deprecated and will be removed in 0.5. Pass the embedding as "
            "the first argument.") in capsys.readouterr().err


# Y09 / X95: --dataset and --category read the label files without --method
def test_cli_evaluate_dataset_without_method_reads_its_labels(tmp_path, capsys):
    import numpy as np
    from multibench import cli
    from multibench.engine.resolve import labels_for      # mtb.labels_for is deprecated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # its default-order note
        files = labels_for("D52mini", "cross")
    n = sum(len(open(p).read().splitlines()) - 1 for p in files.values())
    emb = tmp_path / "e.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(n, 5)))
    rc = cli.main(["evaluate", str(emb), "--dataset", "D52mini", "--category", "cross",
                   "--method", "mine", "--metrics", "cLISI"])
    out = capsys.readouterr()
    assert rc == 0 and "mine" in out.out
    assert "# labels: cty1.csv, cty2.csv, cty3.csv from D52mini, in that order" in out.err


def test_cli_evaluate_help_gives_each_flag_one_job(capsys):
    from multibench import cli
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--name" not in text                           # a hidden alias of --method
    assert ("fills the method column: a package method id, or your own name. A package "
            "method, also as 'StabMap (this run)', reads --dataset's label files in its "
            "cell order, with --category") in text


# Y36 / Y37 / X98: maintainer and multi-dataset flags leave --help; leiden flavor is advanced
def test_help_leaves_out_ref_and_overall(capsys):
    from multibench import cli
    for cmd, flag in ((["fetch"], "--ref"), (["plot", "bubble"], "--overall")):
        with pytest.raises(SystemExit):
            cli.main([*cmd, "--help"])
        assert flag not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--help"])
    out = capsys.readouterr().out
    assert "--leiden-flavor" not in out                   # hidden in 0.4.0
    assert "0.1" not in out


# X76: the command line takes the family names in any letter case
def test_cli_metrics_family_any_case(tmp_path, capsys):
    import numpy as np
    from multibench import cli
    emb = tmp_path / "e.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(40, 4)))
    lab = tmp_path / "l.csv"
    lab.write_text("x\n" + "\n".join("ab"[i % 2] for i in range(40)) + "\n")
    assert cli.main(["evaluate", str(emb), "--labels", str(lab), "--metrics",
                     "Clustering"]) == 0


# Y25: an unknown option shows the usage of the command it was given to
def test_unknown_option_shows_the_command_usage(capsys):
    from multibench import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["scan", "D11", "--foo"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and err.startswith("usage: multibench scan ")
    assert "unrecognized arguments: --foo" in err


# Y27: an ATAC-only batch of a mosaic folder gets no "No RNA written" note
def test_cli_convert_atac_only_mosaic_batch(tmp_path, capsys):
    import numpy as np
    import pandas as pd
    ad = pytest.importorskip("anndata")
    from multibench import cli
    a = ad.AnnData(np.ones((6, 4)), obs=pd.DataFrame({"celltype": list("ababab")},
                                                     index=[f"c{i}" for i in range(6)]))
    a.var_names = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(4)]
    a.write_h5ad(tmp_path / "b3.h5ad")
    rc = cli.main(["convert", str(tmp_path / "b3.h5ad"), str(tmp_path / "MOSX"), "--atac", "X",
                   "--labels", "obs:celltype", "--category", "mosaic", "--batch-index", "3",
                   "--n-batches", "3", "--atac-kind", "peak"])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "No RNA written" not in err
    # Y27 (r3): nor when the folder already holds the RNA
    r = ad.AnnData(np.ones((6, 4)), obs=a.obs.copy())
    r.var_names = [f"g{i}" for i in range(4)]
    r.write_h5ad(tmp_path / "rna.h5ad")
    assert cli.main(["convert", str(tmp_path / "rna.h5ad"), str(tmp_path / "DIAG"), "--rna", "X",
                     "--category", "diagonal"]) == 0
    capsys.readouterr()
    rc = cli.main(["convert", str(tmp_path / "b3.h5ad"), str(tmp_path / "DIAG"), "--atac", "X",
                   "--atac-kind", "peak", "--category", "diagonal"])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "No RNA written" not in err and "already in the folder: rna.h5" in err


# Y03: the cmd-template placeholder is replaced token by token
def test_cmd_template_bare_quoted_and_mid_string():
    from multibench.engine.runner import wrap_command
    cmd = ["python", "x.py", "--a", "1"]
    assert wrap_command(cmd, "srun {cmd}") == ["srun", *cmd]
    assert wrap_command(cmd, "sbatch --wrap '{cmd}'") == ["sbatch", "--wrap",
                                                         "python x.py --a 1"]
    assert wrap_command(cmd, "bash -c '{cmd} > log'") == ["bash", "-c",
                                                         "python x.py --a 1 > log"]


def test_cmd_template_needs_one_placeholder(tmp_path):
    for bad in ("srun", "{env_cmd} {env_cmd}"):
        with pytest.raises(ValueError, match=r"must hold \{env_cmd\} \(or \{cmd\}\) once"):
            mtb.run("Matilda", "vertical", inputs=MATILDA, out_dir=str(tmp_path),
                    dry_run=True, cmd_template=bad)


def test_cmd_template_env_cmd_in_quotes(tmp_path):
    argv = mtb.run("Matilda", "vertical", inputs=MATILDA, out_dir=str(tmp_path),
                   dry_run=True, cmd_template="sbatch --wrap '{env_cmd}'")
    assert argv[:2] == ["sbatch", "--wrap"] and len(argv) == 3
    assert argv[2].startswith("conda run -n ") or " run_matilda.py " in argv[2]
