"""Fix round 6, work package 'cli'.

R6-07: ``multibench evaluate --name`` is the row name in every case: with
``--labels`` it works without ``--method``. The errors for an unknown
``--method`` lead to the call that works; unknown-method and no-category
messages are short sentences without '?;' or 'variant'. Rows written with
``--name`` and a package ``--method`` carry ``needs_labels``, so bubble's
``L`` badge follows the method.
R6-08: bubble's duplicate-rows error says when your rows share a name with
the stored table, and how to rename them.
R6-09: ``run(dry_run=True)`` notes an input file that does not exist; the
CLI dry run prints its header first.
R6-13: plot messages read as sentences: the CLI overlay errors, the igraph
backend warning, the require_complete warning and error.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench.engine import registry
from tests.test_f4_cli import _diag_folder

pytest.importorskip("scib")

_DATA = Path(mtb.__file__).resolve().parent.parent / "data"


@pytest.fixture
def diag(tmp_path, monkeypatch):
    data = tmp_path / "data"
    d, emb = _diag_folder(data)
    monkeypatch.setattr(config.DEFAULT, "data_path", data)
    np.save(tmp_path / "emb.npy", emb)
    return d, tmp_path / "emb.npy"


def _err(capsys, argv):
    rc = cli.main(argv)
    return rc, capsys.readouterr().err


# ====================================================================== R6-07
def test_name_alone_with_labels_writes_rows_with_that_name(diag, tmp_path, capsys):
    d, emb = diag
    out = tmp_path / "mine.csv"
    rc = cli.main(["evaluate", "--output", str(emb), "--name", "RNA+ATAC PCA",
                   "--labels", str(d / "rna_cty.csv"), "--labels", str(d / "atac_cty.csv"),
                   "--dataset", "MYDIAG", "--category", "diagonal", "--metrics", "ASW",
                   "--out", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert set(df["method"]) == {"RNA+ATAC PCA"} and set(df["dataset"]) == {"MYDIAG"}
    assert "needs_labels" not in df.columns          # no package method to ask


def test_name_without_method_or_labels_is_a_usage_error(diag, capsys):
    d, emb = diag
    with pytest.raises(SystemExit) as e:
        cli.main(["evaluate", "--output", str(emb), "--name", "x", "--dataset", "D28",
                  "--category", "diagonal"])
    assert e.value.code == 2
    assert ("Without --method, pass the label files with --labels, once per file, in "
            "your embedding's cell order.") in capsys.readouterr().err


def test_unknown_method_with_name_says_leave_out_method(diag, capsys):
    d, emb = diag
    rc, err = _err(capsys, ["evaluate", "--output", str(emb), "--method", "MyPCA",
                            "--name", "RNA+ADT PCA", "--labels", "cty.csv",
                            "--dataset", "MYCITE", "--category", "vertical"])
    assert rc == 1
    assert ('error: MyPCA is not a package method. For your own embedding, leave out '
            '--method: --name "RNA+ADT PCA" --labels cty.csv.') in err
    # a close name adds the did-you-mean sentence
    rc, err = _err(capsys, ["evaluate", "--output", str(emb), "--method", "Matlida",
                            "--name", "x", "--labels", "cty.csv",
                            "--dataset", "MYCITE", "--category", "vertical"])
    assert rc == 1 and "Matlida is not a package method. Did you mean Matilda? " in err
    assert "leave out --method: --name x --labels cty.csv." in err


def test_unknown_method_without_labels_names_the_rename(diag, capsys):
    d, emb = diag
    rc, err = _err(capsys, ["evaluate", "--output", str(emb), "--method", "uniPort_rerun",
                            "--dataset", "D28", "--category", "diagonal"])
    assert rc == 1
    assert ("error: uniPort_rerun is not a package method. To name your rows "
            "uniPort_rerun, pass --method uniPort --name uniPort_rerun. For your own "
            "method, pass --name and the label files with --labels, once per file, in "
            "your embedding's cell order.") in err
    # a typo gets 'Did you mean', not the rename
    rc, err = _err(capsys, ["evaluate", "--output", str(emb), "--method", "Matlida",
                            "--dataset", "D28", "--category", "diagonal"])
    assert rc == 1 and "Did you mean Matilda?" in err and "--name Matlida" not in err


def test_check_method_is_short_sentences():
    with pytest.raises(KeyError) as e:
        registry.check_method("stabmap")
    assert e.value.args[0] == ("Unknown method stabmap. Did you mean StabMap? "
                               "mtb.list_methods() shows all methods.")
    with pytest.raises(KeyError) as e:
        registry.check_method("zzz")
    assert e.value.args[0] == "Unknown method zzz. mtb.list_methods() shows all methods."


def test_no_category_message_names_the_categories():
    with pytest.raises(KeyError) as e:
        mtb.labels_for("D11", "vertical", "SCALEX", data_path=_DATA)
    assert e.value.args[0] == "SCALEX does not run on vertical data. Its categories: diagonal."


def _messages_of(calls):
    out = []
    for call in calls:
        with pytest.raises((KeyError, ValueError)) as e, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call()
        out.append(e.value.args[0])
    return out


def test_no_unknown_method_or_no_category_message_has_variant_or_qsemicolon(tmp_path,
                                                                            capsys):
    rna, adt = str(_DATA / "D11" / "rna.h5"), str(_DATA / "D11" / "adt.h5")
    msgs = _messages_of([
        lambda: registry.check_method("stabmap"),
        lambda: mtb.method_info("stabmap"),
        lambda: mtb.catalog.canonical_id("Matlida", strict=True),
        lambda: mtb.load_results("vertical", methods=["Matlida"]),
        lambda: mtb.labels_for("D11", "vertical", "SCALEX", data_path=_DATA),
        lambda: mtb.inputs_for("D11", "vertical", "SCALEX", data_path=_DATA),
        lambda: mtb.run("SCALEX", "vertical", inputs={"rna": rna, "adt": adt},
                        out_dir=str(tmp_path / "o"), dry_run=True),
        lambda: mtb.params_for("SCALEX", "vertical"),
    ])
    np.save(tmp_path / "emb.npy", np.zeros((10, 3)))
    for argv in (["info", "stabmap"], ["params", "Matilda", "--category", "cross"],
                 ["evaluate", "--output", str(tmp_path / "emb.npy"), "--method", "SCALEX",
                  "--name", "ADT_PCA", "--dataset", "D11", "--category", "vertical",
                  "--data-path", str(_DATA)],
                 ["evaluate", "--output", str(tmp_path / "emb.npy"), "--method",
                  "uniPort_rerun", "--dataset", "D11", "--category", "vertical"]):
        rc, err = _err(capsys, argv)
        assert rc == 1
        msgs.append(err)
    assert len(msgs) == 12
    for m in msgs:
        assert "?;" not in m and "variant" not in m, m
    assert ("error: SCALEX does not run on vertical data. Its categories: diagonal. For "
            "your own embedding, pass the label files with --labels.") in msgs[10]
    assert "error: Unknown method stabmap. Did you mean StabMap? multibench list shows " \
           "all methods." in msgs[8]


def _d11_emb(tmp_path):
    n = len(pd.read_csv(_DATA / "D11" / "cty.csv"))
    emb = tmp_path / "emb.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(n, 4)).astype("float32"))
    return emb


def test_name_with_package_method_writes_needs_labels_and_bubble_badges(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from multibench.plot import bubble as B
    emb = _d11_emb(tmp_path)
    out = tmp_path / "m.csv"
    assert cli.main(["evaluate", "--output", str(emb), "--method", "Matilda", "--name",
                     "Matilda_rerun", "--dataset", "D11", "--category", "vertical",
                     "--data-path", str(_DATA), "--metrics", "ASW", "--out", str(out)]) == 0
    mine = pd.read_csv(out)
    assert set(mine["method"]) == {"Matilda_rerun"}
    assert mine["needs_labels"].tolist() == [True] * len(mine)
    # the default long table is unchanged: no --name, no column
    out2 = tmp_path / "plain.csv"
    assert cli.main(["evaluate", "--output", str(emb), "--method", "Matilda",
                     "--dataset", "D11", "--category", "vertical", "--data-path",
                     str(_DATA), "--metrics", "ASW", "--out", str(out2)]) == 0
    assert "needs_labels" not in pd.read_csv(out2).columns
    # bubble badges the renamed row
    df = pd.concat([mtb.load_results("vertical", dataset="D11", source="rerun"), mine],
                   ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = B.build_table(df, na="skip")
        fig = B.render(tbl)
    assert tbl.needs_labels.get("Matilda_rerun") is True
    from matplotlib.patches import Circle
    badged = [tbl.methods[len(tbl.methods) - int(p.center[1] + 0.5)]
              for p in fig.axes[0].patches
              if isinstance(p, Circle) and abs(p.center[0] - (-0.38)) < 1e-6]
    assert "Matilda_rerun" in badged and "Matilda" in badged


def test_name_help_says_method_can_be_left_out():
    parser = cli.build_parser()
    ev = next(a for a in parser._actions if a.dest == "command").choices["evaluate"]
    act = next(a for a in ev._actions if a.dest == "name")
    assert act.help == ("row name in the long table, e.g. SCALEX_rerun; with --labels, "
                        "--method can be left out")


# ====================================================================== R6-08
def _user_rows(method="uniPort", dataset="D28", metrics=("ARI", "NMI", "ASW")):
    return pd.DataFrame({"metric": list(metrics), "value": [0.3, 0.5, 0.6][:len(metrics)],
                         "method": method, "dataset": dataset, "category": "diagonal",
                         "clustering": "default", "source": "user"})


def test_cli_overlay_with_a_stored_name_says_rename_with_name(tmp_path, capsys):
    up = tmp_path / "up.csv"
    _user_rows().to_csv(up, index=False)
    rc, err = _err(capsys, ["plot", "bubble", "--input", str(up), "--category", "diagonal",
                            "--dataset", "D28", "--out", str(tmp_path / "f.png")])
    assert rc == 1
    line = [l for l in err.splitlines() if l.startswith("error: ")][0]
    assert line == ("error: Your rows and the stored table both have uniPort on D28. Give "
                    "your rows another name, such as uniPort_rerun, with evaluate --name.")
    for bad in ("(", "{", "variants", "sweep()"):
        assert bad not in line


def test_python_overlay_with_a_stored_name_names_to_long():
    df = pd.concat([mtb.load_results("diagonal", dataset="D28"), _user_rows()],
                   ignore_index=True)
    with pytest.raises(ValueError) as e:
        mtb.plot.build_table(df)
    assert str(e.value) == ("Your rows and the stored table both have uniPort on D28. "
                            "Give your rows another name, such as uniPort_rerun, with "
                            "to_long(method=...).")


def test_repeated_user_rows_give_the_plain_repeat_text():
    rows = pd.concat([_user_rows("Mine"), _user_rows("Mine")], ignore_index=True)
    with pytest.raises(ValueError) as e:
        mtb.plot.build_table(rows)
    assert str(e.value) == ("6 rows repeat the same method, metric and dataset, for "
                            "example Mine, ARI, D28. The figure does not average them. "
                            "Remove the repeats, or give each version its own method name.")


# ====================================================================== R6-09
def _lung(tmp_path):
    d = tmp_path / "data" / "LUNG"
    d.mkdir(parents=True)
    for f in ("rna.h5", "atac_peak.h5"):
        (d / f).write_bytes((_DATA / "D28" / f).read_bytes())
    return d


def test_python_dry_run_notes_a_missing_input(tmp_path, capsys):
    d = _lung(tmp_path)
    inp = {"rna": str(d / "rna.h5"), "atac_gas": str(d / "atac_gas.h5")}
    argv = mtb.run("SCALEX", "diagonal", inputs=inp, out_dir=str(tmp_path / "o"),
                   dry_run=True)
    assert isinstance(argv, list) and str(d / "atac_gas.h5") in argv
    err = capsys.readouterr().err
    assert (f"# SCALEX reads {d / 'atac_gas.h5'}, which does not exist. mtb.scan shows "
            f"what the folder holds.") in err.splitlines()
    assert "rna.h5, which" not in err


def test_cli_dry_run_prints_the_header_first_and_the_path_as_given(tmp_path, capsys,
                                                                  monkeypatch):
    _lung(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                   "--input", "rna=data/LUNG/rna.h5",
                   "--input", "atac_gas=data/LUNG/atac_gas.h5",
                   "--out-dir", "runs/x", "--dry-run"])
    cap = capsys.readouterr()
    assert rc == 0
    lines = cap.err.splitlines()
    assert lines[0] == "# Dry run. Nothing was executed."
    assert ("# SCALEX reads data/LUNG/atac_gas.h5, which does not exist. multibench scan "
            "shows what the folder holds.") in lines
    assert lines[-1] == "# multibench run would execute:"
    assert "main_SCALEX.py" in cap.out


def test_dry_run_with_existing_inputs_has_no_missing_note(tmp_path, capsys):
    inp = mtb.inputs_for("D28", "diagonal", "SCALEX", data_path=_DATA)
    mtb.run("SCALEX", "diagonal", inputs=inp, out_dir=str(tmp_path / "o"), dry_run=True)
    assert "which does not exist" not in capsys.readouterr().err


# ====================================================================== R6-13
def _stored(ds):
    return mtb.load_results("vertical", dataset=ds, source="rerun")


def test_plot_messages_are_sentences(tmp_path, capsys):
    from multibench.plot import style
    own = _stored("D11").query("method == 'Matilda'").assign(
        method="PriyaNet", source="user", scored_with="igraph/sweep/0.3.2")
    msg = style.backend_warning(pd.concat([_stored("D11"), own], ignore_index=True))
    assert msg.startswith("The rows for PriyaNet were clustered with the igraph Leiden "
                          "backend. The stored tables used leidenalg, which can move ARI "
                          "by up to about 0.1. ")
    # require_complete: one sentence per group of methods lacking the same datasets
    rows = []
    for m, dss in (("A", ("D1", "D2")), ("B", ("D1", "D2")), ("C", ("D1",)),
                   ("E", ("D1",))):
        for ds in dss:
            rows += [{"method": m, "metric": k, "value": v, "dataset": ds}
                     for k, v in (("ARI", 0.1 + len(m) / 10), ("NMI", 0.3))]
    with pytest.warns(UserWarning) as rec:
        mtb.plot.build_table(pd.DataFrame(rows), aggregate="summary",
                             require_complete=True)
    (w,) = [str(r.message) for r in rec if "require_complete" in str(r.message)]
    assert w == ("require_complete=True dropped 2 methods that lack a dataset: C and E "
                 "lack D2. Pass require_complete=False to keep them. A missing dataset "
                 "then scores rank 0 under overall='rank'.")
    # A only on D1, C only on D2: no method has both
    part = pd.DataFrame([r for r in rows if r["method"] == "C"]
                        + [dict(r, dataset="D2", method="E") for r in rows
                           if r["method"] == "C"])
    with pytest.raises(ValueError) as e:
        mtb.plot.build_table(part, aggregate="summary", require_complete=True)
    assert str(e.value) == ("No method has results on all 2 datasets (D1, D2). Pass "
                            "require_complete=False.")
    for text in (msg, w, str(e.value)):
        assert ";" not in text.split("overall=")[0] and "(s)" not in text


def test_cli_require_complete_warning_uses_the_flags(tmp_path, capsys):
    rows = []
    for m, dss in (("A", ("D1", "D2")), ("C", ("D1",))):
        for ds in dss:
            rows += [{"method": m, "metric": k, "value": 0.2, "dataset": ds}
                     for k in ("ARI", "NMI")]
    p = tmp_path / "in.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    rc, err = _err(capsys, ["plot", "bubble", "--input", str(p), "--aggregate", "summary",
                            "--require-complete", "--out", str(tmp_path / "f.png")])
    assert rc == 0
    assert ("warning: --require-complete dropped 1 method that lacks a dataset: C lacks "
            "D2. Leave out --require-complete to keep them. A missing dataset then scores "
            "rank 0 under --overall rank.") in err
