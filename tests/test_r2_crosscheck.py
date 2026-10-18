"""Round 2 of the 0.4.0 cross-check: one test per fix and deprecation."""
from __future__ import annotations

import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry

MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.run(method, category, inputs=inputs, out_dir=str(tmp_path / "o"),
                   params=params, dry_run=True, cmd_template="conda run -n x {cmd}")


# X63: a param is rendered by the type the script declares
def test_a_bool_argument_gets_an_explicit_value(tmp_path, gpu):
    # main_scJoint.py: --use_cuda type=bool; bool('') is False
    argv = " ".join(_dry("scJoint", "diagonal", SCJOINT, {"use_cuda": False}, tmp_path))
    assert "--use_cuda ''" in argv
    argv = " ".join(_dry("scJoint", "diagonal", SCJOINT, {"use_cuda": True}, tmp_path))
    assert "--use_cuda True" in argv


def test_matilda_augmentation_false_reaches_the_driver(tmp_path):
    argv = _dry("Matilda", "vertical", MATILDA, {"augmentation": False}, tmp_path)
    i = argv.index("--augmentation")
    assert argv[i + 1] == ""
    src = (registry._YAML.parent / "drivers" / "run_matilda.py").read_text()
    assert 'a.augmentation not in ("", "False")' in src


def test_a_switch_is_present_only_for_true(tmp_path):
    inputs = {"rna": "/d/rna.h5", "adt": "/d/adt.h5"}
    argv = _dry("scMDC", "vertical", inputs, {"filter1": True}, tmp_path)
    assert argv[-1] == "--filter1"
    argv = _dry("scMDC", "vertical", inputs, {"filter1": False}, tmp_path)
    assert "--filter1" not in argv


def test_a_switch_or_bool_refuses_another_value(tmp_path):
    with pytest.raises(TypeError, match=r"scMDC: filter1 takes True or False, got 1\."):
        _dry("scMDC", "vertical", {"rna": "/d/rna.h5", "adt": "/d/adt.h5"},
             {"filter1": 1}, tmp_path)
    with pytest.raises(TypeError, match=r"use_cuda takes True or False, got 'False'"):
        _dry("scJoint", "diagonal", SCJOINT, {"use_cuda": "False"}, tmp_path)


def test_a_store_false_switch_is_present_for_false():
    decl = {"type": "flag", "default": True}
    assert builder._param_tokens("umap_plot", False, decl) == ["--umap_plot"]
    assert builder._param_tokens("umap_plot", True, decl) == []


# X64: params_for leaves out what the package sets
@pytest.mark.parametrize("method, category, mods, gone", [
    ("scMDC", "vertical", ["rna", "adt"], {"nbatch", "data_file"}),
    ("scMM", "vertical", ["rna", "atac"], {"model", "experiment", "analytics"}),
    ("MIRA", "vertical", ["rna", "atac"], {"save_name", "load_model",
                                           "rna_topic_modal_path"}),
])
def test_tunable_leaves_out_what_the_package_sets(method, category, mods, gone):
    assert not gone & set(mtb.params_for(method, category, mods)["tunable"])


def test_a_package_set_key_is_refused():
    with pytest.raises(KeyError, match=r"scMDC does not accept \['nbatch'\]"):
        mtb.run_all("D52mini", "cross", methods=["scMDC"], params={"scMDC": {"nbatch": 5}},
                    dry_run=True, verbose=False)


# Y38: one default, the one a run uses
def test_tunable_default_is_what_a_run_uses():
    t = mtb.params_for("scMM", "vertical", ["rna", "atac"])["tunable"]["no_cuda"]
    assert t["default"] is True and t["script_default"] is False


# Y02: a tree records its source URL; a new manifest URL fetches it again
from test_fetch_outputs import F, _write_tree, served  # noqa: E402,F401


def test_a_tree_of_another_url_is_fetched_again(served, monkeypatch, capsys):  # noqa: F811
    data, calls = served
    out = mtb.data.fetch_outputs("D11", data_path=data, quiet=True)
    assert (out / ".source").read_text().strip() == F._output_urls()["D11"]
    (out / "mine.txt").write_text("x")
    new = {**F._output_urls(), "D11": "https://example.org/new/outputs-D11.tar.gz"}
    monkeypatch.setattr(F, "_output_urls", lambda: new)
    out = mtb.data.fetch_outputs("D11", data_path=data)
    old = data / "outputs" / "D11.old"
    assert calls[-1] == new["D11"] and (old / "mine.txt").is_file()
    assert (out / ".source").read_text().strip() == new["D11"]
    assert f"(the earlier tree moves to {old})" in capsys.readouterr().out


def test_a_tree_without_a_source_is_fetched_again(served):  # noqa: F811
    data, calls = served
    _write_tree(data / "outputs" / "D11")           # a 0.3.x cache: no .source
    mtb.data.fetch_outputs("D11", data_path=data, quiet=True)
    assert len(calls) == 1 and (data / "outputs" / "D11.old").is_dir()


def test_a_stored_tree_of_0_3_says_to_fetch_again(tmp_path):
    tree = tmp_path / "outputs" / "D11"
    _write_tree(tree)                               # long.csv without scored_with
    with pytest.warns(UserWarning, match=r"These are the 0\.3\.x stored outputs\. "
                      r"mtb\.data\.fetch_outputs\('D11'\) gets the 0\.4\.0 ones\."):
        mtb.load_batch(tree)


# Y40: a second dataset is refused before any download
def test_fetch_outputs_takes_one_dataset(served):  # noqa: F811
    data, calls = served
    with pytest.raises(TypeError, match="fetch_outputs takes one dataset; call it once "
                                        "per dataset."):
        mtb.data.fetch_outputs("D11", "D28", data_path=data)
    assert calls == []


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


# X07: a family counts in the Overall only when every row has it, silently
def test_a_family_two_rows_have_counts_in_no_overall():
    import warnings
    import pandas as pd
    stored = mtb.load_results("diagonal", dataset="D28")
    mine = stored[stored["method"].isin(["scJoint", "iNMF"])].assign(
        method=lambda d: d["method"] + "_rerun", source="user",
        scored_with="leidenalg/search/0.4.0")
    cls = pd.DataFrame([{"metric": c, "value": 0.5, "method": m, "dataset": "D28",
                         "category": "diagonal", "clustering": "default", "source": "user",
                         "scored_with": "leidenalg/search/0.4.0"}
                        for m in ("scJoint_rerun", "iNMF_rerun") for c in ("OCA", "F1")])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = mtb.plot.build_table(pd.concat([stored, mine, cls]))
    order = list(t.methods)
    for m in ("scJoint", "iNMF"):
        assert order.index(m + "_rerun") == order.index(m) + 1      # same scores, side by side
    assert "Classification" in [b.label for b in t.blocks]
    assert not [x for x in w if "Classification" in str(x.message)
                or "has no OCA" in str(x.message)]


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


# X79: the renumber hint is each blocked row's reason, so every view shows it
def test_mosaic_batches_numbered_wrong_say_renumber_in_the_reason(tmp_path, capsys):
    from multibench import cli
    d = tmp_path / "MOSBAD"
    d.mkdir()
    for f in ("rna1.h5", "atac1.h5", "rna2.h5", "atac3.h5", "cty1.csv", "cty2.csv",
              "cty3.csv"):
        (d / f).write_text("")
    hint = ("MOSBAD has batches 1 RNA+ATAC, 2 RNA, 3 ATAC. Methods read 1 RNA, "
            "2 RNA+ATAC, 3 ATAC: renumber the files to match.")
    plan = mtb.run_all(str(d), "mosaic", methods=["Cobolt"], out_dir=str(tmp_path / "o"),
                       dry_run=True, verbose=False)
    assert plan.loc[0, "reason"].startswith(hint)
    capsys.readouterr()
    cli.main(["scan", str(d), "--category", "mosaic", "--methods", "Cobolt",
              "--format", "csv"])
    assert hint in capsys.readouterr().out


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
    n = sum(len(open(p).read().splitlines()) - 1
            for p in mtb.labels_for("D52mini", "cross").values())
    emb = tmp_path / "e.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(n, 5)))
    rc = cli.main(["evaluate", str(emb), "--dataset", "D52mini", "--category", "cross",
                   "--name", "mine", "--metrics", "cLISI"])
    out = capsys.readouterr()
    assert rc == 0 and "mine" in out.out
    assert "# labels: cty1.csv, cty2.csv, cty3.csv from D52mini, in that order" in out.err


def test_cli_evaluate_help_gives_each_flag_one_job(capsys):
    from multibench import cli
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "the row name in the long table (default: --method)" in text
    assert "a package method: read the dataset's label files in its batch order" in text


# X67: labels_for without a method warns once per dataset and category
def test_default_order_warning_once():
    import warnings
    from multibench.engine import resolve
    resolve._ORDER_NOTED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mtb.labels_for("D52mini", "cross")
        mtb.labels_for("D52mini", "cross")
    assert len([x for x in w if "default order" in str(x.message)]) == 1


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


# Y08: the Rank key says best and worst, as BubbleTable.ranks counts n = best
def test_rank_key_says_best_and_worst():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = mtb.plot.bubble(mtb.load_results("mosaic"))
    texts = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "legend"]
    assert "best" in texts and "worst" in texts


# Y11: in rescore, the Leiden search warning names its method
def test_rescore_leiden_warning_names_the_method(monkeypatch):
    import numpy as np
    from multibench import workflow
    from multibench.eval import scib

    class G:
        obs = {}
    monkeypatch.setattr(scib, "_search_n_clusters",
                        lambda g, n, key, flavor: g.obs.update({key: _Ser()}) or False)
    with pytest.warns(UserWarning, match=r"^No Leiden resolution gave 13 clusters, one per "
                                         r"cell type, for scMM; using the 12 clusters"):
        with workflow._named_leiden_warnings("scMM"):
            scib._n_clusters(G(), 13, "leidenalg", per="cell type")
    assert scib.FOR_METHOD.get() is None


class _Ser:
    def nunique(self):
        return 12

    def astype(self, t):
        return ["0"] * 3
