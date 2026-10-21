"""Core round c1 (0.4.0): the smaller fixes and simplifications."""
import importlib
import json
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow
from multibench.engine import registry, resolve
from multibench.eval import io as eio, pipeline, scib as escib
from multibench.plot import style
from multibench.plot.bubble import _build_table
from tests.test_c1_decisions import _tree  # noqa: F401

bar_module = importlib.import_module("multibench.plot.bar")


# --------------------------------------------------------------- FIX items
_CUT = ["to_long", "results_coverage", "inputs_for", "labels_for", "to_canonical",
        "sweep", "build_table", "n_tunable", "order=", "requires_gpu"]


def test_kept_docstrings_name_no_cut_name():
    import inspect
    objs = {n: getattr(mtb, n) for n in mtb.__all__}
    for ns in ("catalog", "config", "data", "env", "io", "plot"):
        m = getattr(mtb, ns)
        objs[ns] = m
        objs.update({f"{ns}.{a}": getattr(m, a) for a in dir(m) if not a.startswith("_")})
    for a in ("summary", "long", "plot", "rescore", "save", "embedding"):
        objs[f"BatchResult.{a}"] = getattr(mtb.BatchResult, a)
    hits = {n: [w for w in _CUT if w in (inspect.getdoc(o) or "")] for n, o in objs.items()}
    assert {n: h for n, h in hits.items() if h} == {}


def test_params_for_returns_the_parameters_the_variants_share():
    p = mtb.params_for("Matilda")
    assert "epochs" in p["tunable"] and p["variant"].count("vertical:") == 3
    assert mtb.params_for("iNMF")["tunable"] == {}


def test_batch_metrics_on_one_batch_are_refused():
    emb = np.random.default_rng(0).normal(size=(40, 3))
    labels = ["a", "b"] * 20
    for metrics in ("batch", ["ARI_batch", "NMI_batch"], ["ASW_batch"]):
        with pytest.raises(ValueError, match="needs two or more batches; every cell is in "
                                             "one batch"):
            mtb.evaluate(emb, labels, batch=["b1"] * 40, metrics=metrics, verbose=False)
    # the default leaves them out
    out = mtb.evaluate(emb, labels, batch=["b1"] * 40, verbose=False)
    assert "ASW_batch" not in set(out.metric)


def test_evaluate_without_category_leaves_it_empty():
    out = mtb.evaluate(np.eye(4), ["a", "b", "a", "b"], metrics=["NMI"], verbose=False)
    assert out["category"].isna().all() and out["dataset"].tolist() == ["user"]


def test_method_info_without_repeated_keys():
    info = mtb.method_info("Matilda")
    assert "requires_gpu" not in info
    assert all("n_tunable" not in v for v in info["supports"])


def test_the_dry_run_plan_has_seven_columns():
    plan = mtb.run("D11", "vertical", dry_run=True, verbose=False)
    assert list(plan.columns) == ["method", "modalities", "runnable", "reason", "env",
                                  "caveat", "command"]


def test_bar_says_its_scale():
    import matplotlib.pyplot as plt
    fig = mtb.plot.bar(mtb.load_results("mosaic"))
    assert fig.axes[0].get_xlabel() == "overall score (D45)"
    plt.close(fig)


def test_skip_existing_scores_a_saved_output_off_linux(tmp_path, monkeypatch):
    from multibench.engine import envs
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "Methods run only on Linux.")
    out = _tree(tmp_path, dims=4).parent / "out"
    res = mtb.run("D11", "vertical", methods=["Matilda"], out_dir=out,
                      skip_existing=True, verbose=False)
    assert res.summary.status.tolist() == ["scored"]


def test_a_two_batch_diagonal_folder_gets_one_folder_line(tmp_path, capsys):
    import anndata as ad
    rng = np.random.default_rng(0)
    d = tmp_path / "DIAG2"
    for b in (1, 2):
        rna = ad.AnnData(rng.poisson(1, (30, 20)).astype(float))
        rna.obs_names = [f"r{b}_{i}" for i in range(30)]
        rna.var_names = [f"G{i}" for i in range(20)]
        rna.obs["celltype"] = ["T", "B"] * 15
        atac = rna.copy()
        atac.obs_names = [f"a{b}_{i}" for i in range(30)]
        mtb.io.export_dataset(rna, d, atac=atac, atac_kind="gene_activity",
                              labels="obs:celltype", category="diagonal", batch_index=b)
    capsys.readouterr()
    out = mtb.scan(d, "diagonal")
    log = capsys.readouterr().out
    assert log.count("This folder holds per-batch files") == 1
    assert "also read exactly 3 numbered RNA and ATAC batches" in log
    assert not out["reason"].str.contains("per-batch files").any()


def test_the_batch_gap_is_named(tmp_path, capsys):
    from multibench import workflow as W
    d = tmp_path / "GAP"
    d.mkdir()
    for f in ("rna1.h5", "rna2.h5", "atac2.h5", "atac4.h5"):
        (d / f).touch()
    assert W._renumber_hint(d, "mosaic") == (
        "GAP has batches 1, 2 and 4: export them to a new folder, numbered 1 to 3.")


def test_a_shared_barcode_gets_its_batch_and_one_batch_names_none():
    one = mtb.load_batch(Path(__file__).resolve().parent.parent / "data" / "outputs" / "D11")
    m = one.records[0]["method"]
    assert "batch" not in one.embedding(m).obs                     # c3 (E02)


def test_recommend_gives_this_run_rows_their_method_chips():
    stored = mtb.load_results("vertical")
    mine = stored[stored.method == "Matilda"].assign(method="Matilda (this run)",
                                                     source="user")
    out = mtb.recommend("vertical", long_df=pd.concat([stored, mine])).set_index("method")
    assert out.loc["Matilda (this run)", "env"] == out.loc["Matilda", "env"] is not None


def test_modalities_select_the_variant_spelled_so(capsys):
    plan = mtb.run("D28", "diagonal", methods=["scJoint"], modalities=["rna", "atac_gas"],
                       dry_run=True, verbose=False)
    assert plan["modalities"].tolist() == ["rna+atac_gas"]
    # base tokens still reach the numbered variants of another category
    mosaic = mtb.scan("D45", "mosaic", modalities=["rna", "atac"], verbose=False)
    assert "Multigrate" in set(mosaic["method"])


def test_a_folder_method_shows_the_files_it_reads():
    plan = mtb.run("D28", "diagonal", methods=["scBridge"], dry_run=True, verbose=False)
    assert plan["modalities"].tolist() == ["rna+atac_gas"]


def test_the_shadow_folder_note_is_left_out_for_a_path_and_in_rescore(tmp_path, monkeypatch):
    (tmp_path / "D11").mkdir()
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resolve._note_local_folder(tmp_path / "elsewhere", "D11")   # a path the caller named
        _tree(tmp_path / "t")
        mtb.load_batch(tmp_path / "t" / "out").rescore(metrics=["NMI"], verbose=False)
    assert not [x for x in w if "also exists" in str(x.message)]


def test_a_vertical_gene_activity_export_warns(tmp_path):
    import anndata as ad
    a = ad.AnnData(np.ones((10, 4)))
    a.var_names = ["CD3E", "CD19", "MS4A1", "NKG7"]
    with pytest.warns(UserWarning, match=r"^Vertical methods need peak ATAC, and this ATAC "
                                         r"holds gene activity\. Export the ATAC as peaks\.$"):
        mtb.io.export_dataset(a, tmp_path / "V", atac="X", rna=None,
                              atac_kind="gene_activity", category="vertical")


def test_mosaic_rna_next_to_other_cells_atac_names_the_mosaic_fix(tmp_path):
    import anndata as ad
    rna = ad.AnnData(np.ones((6, 3)))
    rna.obs_names = [f"r{i}" for i in range(6)]
    atac = ad.AnnData(np.ones((6, 3)))
    atac.obs_names = [f"a{i}" for i in range(6)]
    atac.var_names = [f"chr1:{i}-{i + 5}" for i in range(3)]
    with pytest.raises(ValueError, match="for a batch with RNA only, export it without atac="):
        mtb.io.export_dataset(rna, tmp_path / "M", atac=atac, atac_kind="peak",
                              category="mosaic", batch_index=2)


def test_one_text_for_an_unknown_metric():
    for metrics in ("bogus", ["bogus"]):
        with pytest.raises(ValueError, match=r"^Unknown metrics= token bogus\. Pass 'all'"):
            mtb.evaluate(np.eye(4), ["a", "b", "a", "b"], metrics=metrics, verbose=False)


