"""Round 1 of the 0.4.0 cross-validation: one test per fixed defect and per
deprecation (ids X.. of rounds040/r1/crosscheck/items.json)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb


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


# --- stored tables: X03, X04, X42, X59, X69 ----------------------------------

def test_x03_conos_rows_left_the_d28_table():
    d28 = mtb.load_results("diagonal", dataset="D28")
    assert "Conos" not in set(d28.method)
    r, msgs = _messages(mtb.recommend, "diagonal")
    assert pd.isna(r.set_index("method").loc["Conos", "grand_score"])
    assert not any("ARI below" in m for m in msgs)


def test_x04_one_stored_table_per_category():
    cov = mtb.results_coverage()
    assert cov.groupby("category").dataset.nunique().to_dict() == {
        "cross": 1, "diagonal": 1, "mosaic": 1, "vertical": 1}
    assert not cov.dataset.str.endswith("s").any()
    r = mtb.recommend("diagonal")
    assert set(r.loc[r.grand_score.notna(), "n_datasets"]) == {1}


def test_x04_bar_of_one_dataset_has_no_whiskers():
    fig = mtb.plot.bar(mtb.load_results("diagonal"), overall="mean_overall")
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "overall — D28"
    assert ax.get_xlabel() == "overall score (D28)"
    from matplotlib.collections import LineCollection
    assert not [c for c in ax.collections if isinstance(c, LineCollection)
                and len(c.get_segments())]


def test_x42_results_coverage_has_three_columns():
    cov = mtb.results_coverage("cross")
    assert list(cov.columns) == ["category", "dataset", "method"]
    assert cov.attrs["rerun_version"] == "0.4.0"


def test_x59_available_datasets_lists_the_stored_datasets():
    with pytest.warns(FutureWarning, match="available_datasets is deprecated"):
        got = mtb.available_datasets()
    assert got == sorted(mtb.results_coverage().dataset.unique())


def test_x69_recommend_cross_names_uinmf_two_batches():
    _, msgs = _messages(mtb.recommend, "cross")
    assert any("UINMF's stored D52 scores cover batches 1-2 of 3" in m for m in msgs)
    _, msgs = _messages(mtb.recommend, "vertical")
    assert not any("batches 1-2" in m for m in msgs)


# --- recommend: X05, X14, X89 ------------------------------------------------

def test_x05_recommend_docstring_names_the_min_max_scaling():
    doc = " ".join(mtb.recommend.__doc__.split())
    assert ("each per-dataset score is the mean rank over the metrics, min-max scaled "
            "within the dataset (1 = best, 0 = worst there)") in doc


@pytest.mark.parametrize("fn", ["recommend", "bar"])
def test_x14_rows_repeating_stored_rows_are_refused(fn):
    d28 = mtb.load_results("diagonal", dataset="D28")
    both = pd.concat([d28, d28[d28.method == "iNMF"].assign(source="user")])
    call = (lambda: mtb.recommend("diagonal", long_df=both)) if fn == "recommend" \
        else (lambda: mtb.plot.bar(both))
    with pytest.raises(ValueError, match=r"Your rows and the stored table both have iNMF "
                                         r"on D28\. Give your rows another name, e\.g\. "
                                         r"rows\.assign\(method=\"iNMF_rerun\"\)\."):
        call()


def test_x16_the_rename_advice_works_on_a_long_frame():
    d11 = _d11()
    rows = d11[d11.method == "scMM"].assign(source="user")
    with pytest.raises(ValueError, match=r'rows\.assign\(method="scMM_rerun"\)') as e:
        mtb.plot.build_table(pd.concat([d11, rows]))
    assert "to_long(method" not in str(e.value)
    # the advice runs as written
    fixed = rows.assign(method="scMM_rerun")
    mtb.plot.build_table(pd.concat([d11, fixed]))


def test_x89_batch_family_on_vertical_says_why():
    with pytest.raises(ValueError, match=r'^The vertical tables have no batch scores '
                                         r'\(D11 has one batch\)\. Use metrics="clustering"\.$'):
        mtb.recommend("vertical", metrics="batch")


# --- figures: X07, X86, X87, X88, X94 ----------------------------------------

def test_x07_a_family_only_your_row_has_does_not_lift_it():
    d11 = _d11()
    mine = _mine(d11)
    cls = pd.DataFrame({"metric": ["OCA", "F1"], "value": [0.9, 0.8], "method": "Mine",
                        "dataset": "D11", "category": "vertical", "clustering": "default",
                        "source": "user", "scored_with": "leidenalg/search/0.4.0"})
    plain, _ = _messages(mtb.plot.build_table, pd.concat([d11, mine]))
    tbl, msgs = _messages(mtb.plot.build_table, pd.concat([d11, mine, cls]))
    assert tbl.overall["Mine"] == pytest.approx(plain.overall["Mine"])
    assert "Classification" in [b.label for b in tbl.blocks]          # still drawn
    assert any("Classification has a value for one method only, so it does not count "
               "in the row Overall." in m for m in msgs), msgs
    # bar: the lone metrics are left out of the ranks, with one line
    _, msgs = _messages(mtb.plot.bar, pd.concat([d11, mine, cls]))
    assert "F1 and OCA have a value for one method only, so they do not count in " \
           "the Overall." in msgs


def test_x86_language_chips_and_summary_scale_have_a_legend():
    d11 = _d11()
    fig, _ = _messages(mtb.plot.bubble, pd.concat([d11, _mine(d11)]))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Py / R: the method's language; ?: not a package method" in texts
    fig, _ = _messages(mtb.plot.bubble, d11)
    assert "Py / R: the method's language" in [t.get_text() for t in fig.axes[0].texts]
    two = pd.concat([d11, d11.assign(dataset="D11b")])
    fig, _ = _messages(mtb.plot.bubble, two, aggregate="summary")
    assert "(mean rank over datasets, scaled per column)" in \
        [t.get_text() for t in fig.axes[0].texts]


def test_x87_clustering_columns_follow_catalog_metrics():
    tbl = mtb.plot.build_table(_d11())
    order = mtb.catalog.metrics().query("family == 'clustering'").metric.tolist()
    assert tbl.blocks[0].raw.columns.tolist() == order


def test_x88_bubble_table_repr_is_short():
    text = repr(mtb.plot.build_table(_d11()))
    assert len(text.splitlines()) == 1 and text.startswith("BubbleTable: 13 methods (")
    assert "families DR and clustering (6)" in text and "datasets D11" in text


@pytest.mark.parametrize("scored_with,expect", [
    ("leidenalg/sweep/0.3.9", "were scored before multibench 0.4.0"),
    (None, "do not record how they were scored"),
    ("leidenalg/search/0.4.0", None),
    ("leidenalg/user/0.4.0", None),
    ("unknown", None)])
@pytest.mark.parametrize("fn", [mtb.plot.build_table, mtb.plot.bar])
def test_x94_rows_scored_before_040_warn_next_to_stored_rows(fn, scored_with, expect):
    d11 = _d11()
    _, msgs = _messages(fn, pd.concat([d11, _mine(d11, scored_with=scored_with)]))
    hits = [m for m in msgs if "multibench 0.4.0" in m]
    if expect is None:
        assert not hits, msgs
    else:
        assert len(hits) == 1 and expect in hits[0] and hits[0].startswith(
            "The rows for Mine "), msgs


def test_x94_recommend_warns_for_old_rows_too():
    d11 = _d11()
    _, msgs = _messages(mtb.recommend, "vertical",
                        long_df=pd.concat([d11, _mine(d11, scored_with="igraph/sweep/0.3.1")]))
    assert any("The rows for Mine were scored before multibench 0.4.0." in m for m in msgs)


# --- deprecations and catalog: X39, X40, X41, X43 ----------------------------

def test_x39_no_tasks_column_or_key():
    assert "tasks" not in mtb.catalog.methods().columns
    assert "tasks" not in mtb.method_info("Matilda", verbose=True)


def test_x40_method_info_default_keys_and_the_verbose_hint():
    info = mtb.method_info("Matilda")
    assert set(info) == {"id", "language", "notes", "categories", "env", "atac",
                         "needs_labels", "supports", "runtime", "gpu", "requires_gpu",
                         "cpu_params", "setup_hint", "repo_url", "reference"}
    with pytest.raises(KeyError, match=r"'params' is in method_info\('Matilda', "
                                       r"verbose=True\)"):
        info["params"]
    full = mtb.method_info("Matilda", verbose=True)
    assert {"params", "scripts_url", "notes_long", "upstream_knobs"} <= set(full)


def test_x41_build_table_na_is_deprecated():
    with pytest.warns(FutureWarning, match=r"^plot\.build_table\(na=\) is deprecated and "
                                           r"will be removed in 0\.5; leave it out"):
        mtb.plot.build_table(_d11(), na="skip")
    import inspect
    assert "na" not in inspect.signature(mtb.plot.build_table).parameters


@pytest.mark.parametrize("fn", ["methods", "datasets", "metrics"])
def test_x43_catalog_files_dir_is_deprecated(fn):
    from multibench import config
    f = getattr(mtb.catalog, fn)
    import inspect
    assert "files_dir" not in inspect.signature(f).parameters
    want = f()
    with pytest.warns(FutureWarning, match=rf"^catalog\.{fn}\(files_dir=\) is deprecated "
                                           r"and will be removed in 0\.5\. Use "
                                           r"config\.DEFAULT\.files_path\.$"):
        by_kw = f(files_dir=config.DEFAULT.files_path)
    with pytest.warns(FutureWarning):
        by_pos = f(config.DEFAULT.files_path)               # 0.3 took it by position
    pd.testing.assert_frame_equal(by_kw, want)
    pd.testing.assert_frame_equal(by_pos, want)


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


def test_x68_dry_run_marks_the_nested_variant():
    plan = mtb.run_all("D52mini", "cross", methods=["UINMF"], dry_run=True, verbose=False)
    row = plan.set_index("modalities").loc["rna1+rna2+adt1+adt2"]
    assert row.caveat.startswith("Not run: UINMF runs on rna1+rna2+rna3+adt1+adt2+adt3 "
                                 "instead.")


def test_x80_one_batch_folder_lists_no_batched_variant(capsys):
    plan = mtb.run_all("D27mini", "diagonal", methods=["iNMF", "online_iNMF"],
                       dry_run=True)
    assert set(plan.modalities) == {"rna+atac_gas"}
    assert "2 rows for numbered batches are not listed." in capsys.readouterr().out
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
    assert ("[scan] MYMOSAIC has batches 1 RNA+ATAC, 2 RNA, 3 ATAC. Methods read 1 RNA, "
            "2 RNA+ATAC, 3 ATAC: renumber the files to match.") in out


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
