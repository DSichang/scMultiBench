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
    assert not any("Classification" in m for m in msgs), msgs   # the n/a cells show it
    # bar: the lone metrics are left out of the ranks, with one line
    _, msgs = _messages(mtb.plot.bar, pd.concat([d11, mine, cls]))
    assert "F1 and OCA have a value for one method only, so they do not count in " \
           "the Overall." in msgs


def test_x86_language_chips_and_summary_scale_have_a_legend():
    d11 = _d11()
    fig, _ = _messages(mtb.plot.bubble, pd.concat([d11, _mine(d11)]))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Py / R: the method's language; ?: your rows" in texts
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
                                           r"and will be removed in 0\.5; leave it out "
                                           r"for the package's tables\.$"):
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
    assert "UINMF runs on rna1+rna2+rna3+adt1+adt2+adt3 instead, so this row is not " \
           "run." in row.caveat


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


# --- evaluate, io, catalog lookups: X09, X10, X12, X36, X37, X47, X49, X50,
#     X58, X61, X62, X67, X71, X72, X74, X76, X83, X90, X91 ----------------

def _adata(n=60, names=True):
    import anndata as ad
    import numpy as np
    a = ad.AnnData(np.random.default_rng(0).poisson(1, (n, 20)).astype(float))
    if names:
        a.obs_names = [f"AAAC{i}-1" for i in range(n)]
    return a


def test_x09_repeated_barcodes_are_refused(tmp_path):
    a = _adata()
    a.obs_names = [f"AAAC{i % 30}-1" for i in range(60)]
    with pytest.raises(ValueError, match=r"^30 barcodes repeat, e\.g\. AAAC0-1\. Make them "
                                         r"unique \(prefix the sample\)\.$"):
        mtb.io.export_dataset(a, tmp_path / "DUP", category="vertical")
    assert not (tmp_path / "DUP" / "rna.h5").exists()


def test_x10_no_shared_barcode_names_both_spellings(tmp_path):
    rna = _adata()
    atac = _adata()
    atac.obs_names = [f"AAAC{i}" for i in range(60)]
    atac.var_names = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(20)]
    with pytest.raises(ValueError, match=r"^RNA and ATAC share no barcode \(RNA e\.g\. "
                                         r"AAAC0-1, ATAC e\.g\. AAAC0\)\. Make the names "
                                         r"match; for different cells pass "
                                         r'category="diagonal"\.$'):
        mtb.io.export_dataset(rna, tmp_path / "V", atac=atac, category="vertical")


def test_x12_a_label_file_without_ids_next_to_barcodes_warns(tmp_path):
    import numpy as np
    a = _adata()
    a.obsm["X_pca"] = np.random.default_rng(1).normal(size=(60, 4))
    lab = tmp_path / "cty.csv"
    pd.DataFrame({"x": ["a", "b", "c"] * 20}).to_csv(lab, index=False)
    _, msgs = _messages(mtb.evaluate, a, labels=str(lab), metrics=["cLISI"])
    assert ("cty.csv has no cell ids, so it is matched by position. Check that the "
            "embedding rows follow the file.") in msgs
    b = _adata(names=False)
    b.obsm["X_pca"] = a.obsm["X_pca"]
    _, msgs = _messages(mtb.evaluate, b, labels=str(lab), metrics=["cLISI"])
    assert not any("matched by position" in m for m in msgs)


def test_x36_to_long_clustering_and_source_are_deprecated():
    wide = pd.DataFrame({"Value": [0.5]}, index=["ARI"])
    with pytest.warns(FutureWarning, match=r"^to_long\(source=\) is deprecated and will "
                                           r"be removed in 0\.5; it has no effect\.$"):
        out = mtb.to_long(wide, method="M", source="rerun", clustering="mine")
    assert out[["clustering", "source"]].iloc[0].tolist() == ["default", "user"]


def test_x37_to_canonical_keywords_are_deprecated(tmp_path):
    import inspect
    sig = inspect.signature(mtb.io.to_canonical).parameters
    assert not {"convert", "compression", "block"} & set(sig)
    a = _adata()
    with pytest.warns(FutureWarning, match=r"^io\.to_canonical\(compression=\) is "
                                           r"deprecated"):
        p = mtb.io.to_canonical(a, tmp_path / "rna.h5", compression=None)
    assert mtb.io.read_canonical(p).shape == (60, 20)
    with pytest.warns(FutureWarning, match=r"^io\.to_canonical\(convert=\) is deprecated"):
        assert mtb.io.to_canonical(p, tmp_path / "x.h5", "rna", False) == p


def test_x38_scan_params_is_deprecated():
    with pytest.warns(FutureWarning, match=r"^scan\(params=\) is deprecated and will be "
                                           r"removed in 0\.5\. Use run_all\(\.\.\., "
                                           r"params=\.\.\., dry_run=True\)\.$"):
        mtb.scan("D11", "vertical", methods=["Matilda"], params={"Matilda": {"epochs": 3}},
                 verbose=False)


def test_x47_text_returns_echo_as_text():
    for text in (mtb.describe_layout("vertical"), mtb.cite(["Matilda"])):
        assert isinstance(text, str) and repr(text) == str(text) and "\n" in text


def test_x49_evaluate_takes_the_only_obsm_key():
    import numpy as np
    a = _adata()
    a.obsm["X_pca"] = np.random.default_rng(2).normal(size=(60, 4))
    got = mtb.evaluate(a, labels=np.repeat(["a", "b", "c"], 20), metrics=["cLISI"])
    want = mtb.evaluate(a, labels=np.repeat(["a", "b", "c"], 20), metrics=["cLISI"],
                        obsm="X_pca")
    pd.testing.assert_frame_equal(got, want)
    a.obsm["X_umap"] = a.obsm["X_pca"][:, :2]
    # round 2 (X49): the message names the keys, not an obsm= the user did not pass
    with pytest.raises(ValueError, match=r"^The AnnData has 2 obsm keys \(X_pca, X_umap\): "
                                         r"pass obsm=, or obsm='X' for \.X\.$"):
        mtb.evaluate(a, labels=np.repeat(["a", "b", "c"], 20), metrics=["cLISI"])


def test_x50_x71_classification_docs_and_left_out_cells():
    doc = " ".join(mtb.classification_metrics.__doc__.split())
    assert ('The ``mean`` row is what ``mtb.evaluate(metrics="classification")`` '
            "returns; this adds one row per reference.") in doc
    import numpy as np
    emb = np.random.default_rng(3).normal(size=(90, 4))
    labels = np.array(["a", "b", "c"] * 15 + ["a", "b"] * 22 + ["a"])
    batch = np.array(["1"] * 45 + ["2"] * 45)
    pred = mtb.classify(emb, labels=labels, batch=batch)
    # batch 2 lacks type c: the 15 c cells of batch 1 are left out with 2 as reference
    assert pred.attrs["left_out"] == {"1": 0, "2": 15}
    assert "lacks is left out" in " ".join(mtb.classify.__doc__.split())


def test_x58_find_methods_task_names_no_replacement():
    with pytest.warns(FutureWarning, match=r"find_methods\(task=\) is deprecated and will "
                                           r"be removed in 0\.5; it has no replacement\."):
        mtb.find_methods(task="clustering")


def test_x61_x62_to_long_reads_a_path_and_checks_the_category(tmp_path):
    wide = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    wide.to_csv(tmp_path / "metric.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mtb.to_long(tmp_path / "metric.csv", method="Old", dataset="D11",
                          category="vertical")
    assert out.metric.tolist() == ["ARI", "NMI"] and out.value.tolist() == [0.5, 0.6]
    with pytest.raises(ValueError, match=r"^unknown category 'verticall'"):
        mtb.to_long(wide, method="M", category="verticall")


def test_x67_labels_for_names_the_dataset_category_once():
    _, msgs = _messages(mtb.labels_for, "D52mini")
    hits = [m for m in msgs if "default order" in m]
    assert len(hits) == 1 and "In cross, StabMap" in hits[0] and "mosaic" not in hits[0]
    _, msgs = _messages(mtb.labels_for, "D52mini")
    assert not [m for m in msgs if "default order" in m]          # once a session


def test_x72_batch_leiden_warning_says_what_it_affects(monkeypatch):
    import numpy as np
    from multibench.eval import scib as S
    monkeypatch.setattr(S, "_search_n_clusters", lambda adata, n, key, flavor: (
        adata.obs.__setitem__(key, pd.Categorical(["a"] * 4 + ["b"] * (adata.n_obs - 4)))
        or False))
    with pytest.warns(UserWarning, match=r"^ARI_batch and NMI_batch use 2 clusters: no "
                                         r"resolution gave 3\. Nothing to do\.$"):
        S._n_clusters(S._clustering_graph(np.random.default_rng(4).normal(size=(40, 5))),
                      3, "leidenalg", "batch")


def test_x74_an_h5mu_path_names_the_fix(tmp_path):
    p = tmp_path / "x.h5mu"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match=r"x\.h5mu is an \.h5mu: pass one modality, e\.g\. "
                                         r"mudata\.read\(path\)\['rna'\]"):
        mtb.evaluate(str(p), labels=["a"] * 3)


def test_x76_tokens_ignore_letter_case():
    import numpy as np
    plan = mtb.run_all("D11", "vertical", modalities=["RNA", "ADT"], dry_run=True,
                       verbose=False)
    assert len(plan)
    emb = np.random.default_rng(5).normal(size=(60, 4))
    got = mtb.evaluate(emb, labels=np.repeat(["a", "b", "c"], 20), metrics="Clustering")
    assert "ARI" in got.index


def test_x83_read_canonical_reads_an_embedding_file(tmp_path):
    import h5py
    import numpy as np
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f["data"] = np.arange(30.0).reshape(10, 3)
    a = mtb.io.read_canonical(p)
    assert a.n_obs == 10 and a.obsm["X_emb"].shape == (10, 3)


def test_x90_x91_hints_for_an_assay_and_a_prefix():
    with pytest.raises(ValueError, match=r"CITE-seq and 10x Multiome are vertical\.$"):
        mtb.find_methods(category="CITE-seq")
    with pytest.raises(KeyError) as e:
        mtb.method_info("Seurat")
    assert all(s in str(e.value) for s in ("Seurat_v3", "Seurat_v5", "Seurat_WNN"))


# --- environments: X25, X26, X27, X29, X30, X32, X33, X34 --------------------

def test_x27_x26_install_acts_by_default_and_returns_a_frame(monkeypatch):
    import inspect
    from multibench.engine import envs
    sig = inspect.signature(mtb.env.install).parameters
    assert sig["dry_run"].default is False and "packed" not in sig and "force" not in sig
    plan = mtb.env.install(["Matilda"], dry_run=True)
    assert isinstance(plan, pd.DataFrame) and list(plan.columns[:3]) == [
        "env", "methods", "state"]
    assert isinstance(plan.to_dict("records"), list)
    doc = mtb.env.doctor("vertical")
    assert isinstance(doc, pd.DataFrame) and list(doc.columns[:3]) == [
        "env", "methods", "exists"]
    # a real install (the default) refuses off Linux before anything is fetched
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "this host is darwin/arm64")
    with pytest.raises(OSError) as e:
        mtb.env.install(["Matilda"])
    assert isinstance(e.value, RuntimeError)                         # X32: both
    assert "force" not in str(e.value)                               # X33: hidden


def test_x25_packed_is_build():
    with pytest.warns(FutureWarning, match=r"^env\.install\(packed=\) is deprecated and "
                                           r"will be removed in 0\.5\. Use build=\.$"):
        old = mtb.env.install(["SCALEX"], packed=False, dry_run=True)
    new = mtb.env.install(["SCALEX"], build=True, dry_run=True)
    pd.testing.assert_frame_equal(old, new)
    assert set(new.state) <= {"build(dry-run)", "NO-LOCK", "have"}


def test_x29_x30_dry_run_rows_say_what_runs():
    plan = mtb.env.install(["Multigrate", "scMM"], dry_run=True).set_index("env")
    published = plan[plan.state == "packed archive published"]
    assert len(published)
    for env, row in published.iterrows():
        assert row["cmds"] == [] and row["fallback_cmds"], env
    from multibench.engine import envs
    for env, row in plan.iterrows():
        if row["flavor"] and envs._single_build(env):
            assert row["flavor"] == "any", env


def test_x34_env_dir_is_its_all():
    assert mtb.env.__all__ == ["install", "doctor"]
    assert [n for n in dir(mtb.env) if not n.startswith("__")] == ["doctor", "install"]
    assert callable(mtb.env.recipe)


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
    assert ("# Your rows of scMM are drawn as scMM_rerun: the stored table has the same "
            "method and dataset.") in err


def test_x17_x18_x53_x95_x97_x98_help_texts():
    assert "as printed by multibench find" in _help("run")
    assert "the dataset folder name under --data-path, or its path" in _help("run-all")
    assert "integration category: vertical (several" not in _help("scan")
    assert "vertical, diagonal, mosaic or cross (multibench layout explains them)" in \
        _help("scan")
    ev = _help("evaluate")
    assert ("--method METHOD a package method: read the dataset's label files in its "
            "batch order") in ev
    assert "or with --labels any row name" not in ev
    assert "leidenalg, the article's (keep it to compare with the stored tables)" in ev
    assert "so a later call cannot leave a stale batch file from an earlier export" in \
        _help("convert")


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
    assert ("--runner is deprecated and will be removed in 0.5. Use --cmd-template."
            in err)
    assert "--cmd-template" in _help("run") and "--runner" not in _help("run")


def test_x21_x22_cli_params_tokens_and_int_defaults(capsys):
    rc, out, err = _cli(["params", "Multigrate", "--category", "cross", "--modalities",
                         "rna,adt"], capsys)
    assert rc == 0, err
    assert "cross:rna1+rna2+rna3+adt1+adt2+adt3" in out
    assert "256.000" not in out and " 256 " in out


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
