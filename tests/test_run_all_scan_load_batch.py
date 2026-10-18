"""run_all, scan and load_batch: plans, hints, saved trees."""
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


# --- run_all, scan, load_batch: X01, X11, X13, X35, X63-X66, X68, X79-X81 ----

def _tree(tmp_path, scored_with="leidenalg/search/0.4.0", version="0.4.0"):
    """A saved run_all folder of one scored method."""
    import json
    rec = {"method": "Matilda", "status": "CHAIN_OK", "metrics": {"ARI": 0.5},
           "multibench_version": version, "out_dir": str(tmp_path / "Matilda_D11")}
    (tmp_path / "batch_result.json").write_text(json.dumps(
        {"dataset": "D11", "category": "vertical", "records": [rec]}))
    long = pd.DataFrame({"metric": ["ARI"], "value": [0.5], "method": ["Matilda"],
                         "dataset": ["D11"], "category": ["vertical"],
                         "clustering": ["default"], "source": ["user"]})
    if scored_with is not None:
        long["scored_with"] = scored_with
    long.to_csv(tmp_path / "long.csv", index=False)
    return tmp_path


@pytest.mark.parametrize("scored_with,warns", [
    (None, True), ("leidenalg/sweep/0.3.9", True), ("leidenalg/search/0.4.0", False)])
def test_x01_load_batch_warns_for_scores_before_040(tmp_path, scored_with, warns):
    _, msgs = _messages(mtb.load_batch, _tree(tmp_path, scored_with))
    hit = "These scores come from multibench 0.3.x. res.rescore() scores them as " \
          "0.4.0 does."
    assert (hit in msgs) is warns, msgs


def test_x35_summary_puts_the_scores_first(tmp_path):
    res, _ = _messages(mtb.load_batch, _tree(tmp_path))
    cols = res.summary.columns.tolist()
    assert cols[:4] == ["method", "status", "ARI", "run_sec"]
    assert cols[-2:] == ["caveat", "reason"]


@pytest.mark.parametrize("value", [2.5, "abc", True, 3.0])
def test_x63_param_values_are_type_checked(value):
    with pytest.raises(TypeError, match=r"^Multigrate: epochs takes a whole number, got "):
        mtb.run_all("D52mini", "cross", methods=["Multigrate"],
                    params={"Multigrate": {"epochs": value}}, dry_run=True, verbose=False)
    plan = mtb.run_all("D52mini", "cross", methods=["Multigrate"],
                       params={"Multigrate": {"epochs": "3"}}, dry_run=True, verbose=False)
    assert "--epochs 3 " in plan.command.iloc[0] + " "


def test_x64_tunable_leaves_out_what_the_package_or_script_sets():
    tun = mtb.params_for("scJoint", "diagonal", ["rna", "atac_gas"])["tunable"]
    for name in ("cty_path1", "cty_path2", "rna_protein_paths", "atac_protein_paths",
                 "checkpoint", "input_size", "number_of_class"):
        assert name not in tun, name
    assert {"batch_size", "epochs_stage1", "lr_stage1"} <= set(tun)
    with pytest.raises(KeyError, match=r"scJoint does not accept \['cty_path1'\]"):
        mtb.run_all("D28", "diagonal", methods=["scJoint"],
                    params={"scJoint": {"cty_path1": "x.csv"}}, dry_run=True, verbose=False)
    assert "r_dim" not in mtb.params_for("scMM", "vertical", ["rna", "atac"])["tunable"]


def test_x65_method_colon_key_gets_the_nested_example():
    with pytest.raises(TypeError, match=r"such as params=\{'Multigrate': \{'epochs': 3\}\}\.$"):
        mtb.run_all("D52mini", "cross", params={"Multigrate:epochs": 3}, dry_run=True)


def test_x66_dry_run_refuses_skip_existing_with_params():
    with pytest.raises(ValueError, match=r"^skip_existing=True with params="):
        mtb.run_all("D52mini", "cross", methods=["Multigrate"],
                    params={"Multigrate": {"epochs": 3}}, skip_existing=True,
                    dry_run=True, verbose=False)


def test_x68_size_error_names_the_label_files_that_fit(tmp_path):
    import numpy as np
    labels = mtb.labels_for("D52mini", "cross", "UINMF")
    n = sum(len(pd.read_csv(labels[k])) for k in ("cty1", "cty2"))
    emb = np.random.default_rng(0).normal(size=(n, 5))
    with pytest.raises(ValueError, match=rf"^The embedding has {n:,} cells = cty1\+cty2\. "
                                         r"Pass label_order=\['cty1', 'cty2'\]\.$"):
        mtb.evaluate(emb, labels=labels, category="cross", metrics=["ARI"])


def test_x68_dry_run_marks_the_nested_variant(capsys):
    # r3: the row keeps its own caveat; the dry run says once which row runs
    plan = mtb.run_all("D52mini", "cross", methods=["UINMF"], dry_run=True)
    row = plan.set_index("modalities").loc["rna1+rna2+adt1+adt2"]
    assert "so this row is not run" not in row.caveat
    assert ("[run_all] UINMF runs on rna1+rna2+rna3+adt1+adt2+adt3, not on "
            "rna1+rna2+adt1+adt2") in capsys.readouterr().out


def test_x80_one_batch_folder_lists_no_batched_variant(capsys):
    plan = mtb.run_all("D27mini", "diagonal", methods=["iNMF", "online_iNMF"],
                       dry_run=True)
    assert set(plan.modalities) == {"rna+atac_gas"}
    assert "Variants for numbered batches are not listed." in capsys.readouterr().out
    every = mtb.run_all("D27mini", "diagonal", methods=["iNMF"], dry_run=True,
                        modalities="all", verbose=False)
    assert len(every) == 2


def test_x81_online_inmf_size_note_only_on_a_small_folder():
    big = mtb.run_all("D28", "diagonal", methods=["online_iNMF"], dry_run=True,
                      verbose=False)
    assert not big.caveat.str.contains("5,000 cells").any()
    from multibench.engine import registry
    v = registry.get("online_iNMF").select("diagonal", {"rna", "atac_gas"})
    assert v.min_cells == 5000 and "5,000 cells" in v.size_hint


def test_x79_scan_names_the_batch_renumbering(tmp_path, capsys, monkeypatch):
    import os
    src = mtb.config.DEFAULT.data_path / "D45mini"
    d = tmp_path / "MYMOSAIC"
    d.mkdir()
    for new, old in (("rna1", "rna2"), ("atac1", "atac2"), ("rna2", "rna1"),
                     ("atac3", "atac3")):
        os.symlink(src / f"{old}.h5", d / f"{new}.h5")
    monkeypatch.chdir(tmp_path)
    mtb.scan("./MYMOSAIC", "mosaic")
    out = capsys.readouterr().out
    assert ("[scan] Renumber the batches to 1 RNA, 2 RNA+ATAC, 3 ATAC (MYMOSAIC has "
            "1 RNA+ATAC, 2 RNA, 3 ATAC).") in out
    assert out.count("Renumber the batches") == 1                  # Z20 (r3): once


def test_x11_a_local_folder_shadowed_by_the_data_path_is_named(tmp_path, monkeypatch):
    from multibench.engine import resolve
    monkeypatch.setattr(resolve, "_SHADOW_NOTED", set())
    (tmp_path / "D11").mkdir()
    monkeypatch.chdir(tmp_path)
    _, msgs = _messages(mtb.inputs_for, "D11", "vertical", "Matilda")
    want = (f"Using {mtb.config.DEFAULT.data_path / 'D11'}; ./D11 also exists: pass "
            "'./D11' for it.")
    assert msgs.count(want) == 1, msgs
    _, msgs = _messages(mtb.inputs_for, "D11", "vertical", "Matilda")
    assert want not in msgs                                  # once a session


def test_x13_vertical_peak_methods_read_the_atac_role():
    for m in ("UnitedNet", "iPOLNG", "moETM", "scMDC", "scMM"):
        got = mtb.inputs_for("D27mini_vertical", "vertical", m)
        assert "atac" in got and "atac_gas" not in got, m
        assert got["atac"].endswith("atac_peak.h5"), m
    plan = mtb.run_all("D27mini_vertical", "vertical", modalities=["rna", "atac"],
                       dry_run=True, verbose=False)
    assert {"scMM", "scMDC", "moETM", "UnitedNet", "iPOLNG"} <= set(plan.method)
    # 0.3's key and saved records still select the variant
    argv = mtb.run("scMM", "vertical", inputs={"rna": "a.h5", "atac_gas": "b.h5"},
                   out_dir="o", dry_run=True)
    assert "--path2" in argv
    from multibench.engine import registry
    assert registry.get("scMM").select("vertical", {"rna", "atac_gas"}).when[
        "modalities"] == ["rna", "atac"]


# X79: the renumber hint is each blocked row's reason, so every view shows it
def test_mosaic_batches_numbered_wrong_say_renumber_in_the_reason(tmp_path, capsys):
    from multibench import cli
    d = tmp_path / "MOSBAD"
    d.mkdir()
    for f in ("rna1.h5", "atac1.h5", "rna2.h5", "atac3.h5", "cty1.csv", "cty2.csv",
              "cty3.csv"):
        (d / f).write_text("")
    # r3: the action first, so a clipped column keeps it
    hint = "Renumber the batches to 1 RNA, 2 RNA+ATAC, 3 ATAC (MOSBAD has 1 RNA+ATAC, 2 RNA, 3 ATAC)."
    plan = mtb.run_all(str(d), "mosaic", methods=["Cobolt"], out_dir=str(tmp_path / "o"),
                       dry_run=True, verbose=False)
    assert plan.loc[0, "reason"].startswith(hint)
    capsys.readouterr()
    cli.main(["scan", str(d), "--category", "mosaic", "--methods", "Cobolt",
              "--format", "csv"])
    assert hint in capsys.readouterr().out
    cli.main(["scan", str(d), "--category", "mosaic", "--methods", "Cobolt"])
    assert "Renumber the batches" in capsys.readouterr().out          # the clipped table


# X23: scan, run_all and the command line say the same, without counts
def test_hidden_rows_one_sentence(capsys):
    from multibench import cli
    said = "D11 holds RNA and ADT. Methods and variants that read ATAC are not listed."
    mtb.scan("D11", "vertical")
    assert said in capsys.readouterr().out
    mtb.run_all("D11", "vertical", out_dir="/o", dry_run=True)
    assert said in capsys.readouterr().out
    cli.main(["scan", "D11", "--category", "vertical"])
    assert said in capsys.readouterr().err


# X68: scan shows the variant run_all runs
def test_scan_shows_the_variant_run_all_runs():
    s = mtb.scan("D52mini", "cross", verbose=False)
    assert s.set_index("method").at["UINMF", "modalities"] == "rna1+rna2+rna3+adt1+adt2+adt3"


# Y21: a bare name that is data/<name> here gets that path
def test_scan_names_data_folder_here(tmp_path, monkeypatch):
    (tmp_path / "data" / "CITE").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match=r"data/CITE exists here: pass 'data/CITE'\."):
        mtb.scan("CITE", "vertical")
    assert "mtb.scan('data/MYDATA', 'vertical')" in mtb.describe_layout("vertical")


# Y50: scan without a category names the one category whose rows have their files
def test_scan_names_the_category_that_fits():
    import pandas as pd
    from multibench import workflow
    df = pd.DataFrame({"category": ["mosaic", "cross", "vertical"],
                       "files_ok": [True, False, False]})
    assert workflow._category_hint(df, None) == 'The files fit mosaic: pass category="mosaic".'
    assert workflow._category_hint(df, "mosaic") == ""
    assert workflow._category_hint(df.assign(files_ok=True), None) == ""


# load_batch reads a relative out_dir from the tree's folder
def test_load_batch_reads_a_relative_out_dir_from_the_tree(tmp_path, monkeypatch):
    import json
    tree = tmp_path / "trees" / "D11"
    _write_tree(tree)
    blob = json.loads((tree / "batch_result.json").read_text())
    for r in blob["records"]:
        r["out_dir"] = f"{r['method']}_D11"
    (tree / "batch_result.json").write_text(json.dumps(blob))
    elsewhere = tmp_path / "cwd"
    (elsewhere / "Matilda_D11").mkdir(parents=True)   # must not be picked
    monkeypatch.chdir(elsewhere)
    with pytest.warns(UserWarning):
        res = mtb.load_batch(tree)
    assert [r["out_dir"] for r in res.records] == [
        str((tree / f"{m}_D11").resolve()) for m in ("Matilda", "MOFA2", "scMoMaT")]


# Y02: a tree records its source URL; a new manifest URL fetches it again
from test_fetch_outputs import F, _write_tree, served  # noqa: E402,F401


def test_a_tree_of_another_url_is_fetched_again(served, monkeypatch, capsys):  # noqa: F811
    data, calls = served
    out = F.fetch_outputs("D11", data_path=data, quiet=True)
    assert (out / ".source").read_text().strip() == F._output_urls()["D11"]
    (out / "mine.txt").write_text("x")
    new = {**F._output_urls(), "D11": "https://example.org/new/outputs-D11.tar.gz"}
    monkeypatch.setattr(F, "_output_urls", lambda: new)
    out = F.fetch_outputs("D11", data_path=data)
    old = data / "outputs" / "D11.old"
    assert calls[-1] == new["D11"] and (old / "mine.txt").is_file()
    assert (out / ".source").read_text().strip() == new["D11"]
    assert f"(the earlier tree moves to {old})" in capsys.readouterr().out


def test_a_tree_without_a_source_is_fetched_again(served):  # noqa: F811
    data, calls = served
    _write_tree(data / "outputs" / "D11")           # a 0.3.x cache: no .source
    F.fetch_outputs("D11", data_path=data, quiet=True)
    assert len(calls) == 1 and (data / "outputs" / "D11.old").is_dir()


def test_a_stored_tree_of_0_3_says_to_rescore(tmp_path):
    # the stored-outputs concept left in 0.4.0: one warning for every 0.3.x tree
    tree = tmp_path / "outputs" / "D11"
    _write_tree(tree)                               # long.csv without scored_with
    with pytest.warns(UserWarning, match=r"^These scores come from multibench 0\.3\.x\. "
                      r"res\.rescore\(\) scores them as 0\.4\.0 does\.$"):
        mtb.load_batch(tree)


def test_an_earlier_old_tree_is_kept(served, capsys):  # noqa: F811
    # Z28: a second refetch moves the tree to the first free name, deleting none
    import shutil
    data, calls = served
    for _ in range(2):
        _write_tree(data / "outputs" / "D11")       # a tree without .source
        F.fetch_outputs("D11", data_path=data)
        shutil.rmtree(data / "outputs" / "D11")
    assert (data / "outputs" / "D11.old").is_dir() and (data / "outputs" / "D11.old-2").is_dir()
    assert f"(the earlier tree moves to {data / 'outputs' / 'D11.old-2'})" in \
        capsys.readouterr().out


# Y40: a second dataset is refused before any download
def test_fetch_outputs_takes_one_dataset(served):  # noqa: F811
    data, calls = served
    with pytest.raises(TypeError, match="fetch_outputs takes one dataset; call it once "
                                        "per dataset."):
        F.fetch_outputs("D11", "D28", data_path=data)
    assert calls == []
