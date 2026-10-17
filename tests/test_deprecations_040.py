"""The 0.4 deprecations: each old name warns once, still works, and is gone from
``__all__``, ``dir()`` and the command-line help."""
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli


def _one_warning(call):
    """``(result, message)`` of ``call()``, which must emit exactly one FutureWarning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = call()
    future = [str(w.message) for w in rec if issubclass(w.category, FutureWarning)]
    assert len(future) == 1, future
    return out, future[0]


def _msg(old, new=None):
    return f"{old} is deprecated and will be removed in 0.5. " + (
        f"Use {new}." if new else "It has no effect.")


# ------------------------------------------------------------------ discover
def test_list_methods_is_find_methods():
    got, msg = _one_warning(lambda: mtb.list_methods("vertical"))
    assert got == mtb.find_methods("vertical")
    assert msg == _msg("list_methods", "find_methods")


def test_list_categories_still_describes_the_four():
    got, msg = _one_warning(mtb.list_categories)
    assert list(got) == ["vertical", "diagonal", "mosaic", "cross"]
    assert msg == _msg("list_categories", "describe_layout")


def test_list_tasks_still_lists_the_tasks():
    got, msg = _one_warning(mtb.list_tasks)
    assert got == ["batch", "clustering", "dimension_reduction"]
    assert msg == _msg("list_tasks", "category= and metrics=")


def test_find_methods_task_still_filters():
    got, msg = _one_warning(lambda: mtb.find_methods(task="dimension_reduction"))
    assert got == ["Matilda", "sciPENN"]
    assert msg == _msg("find_methods(task=)", "category=")


def test_find_methods_runnable_has_no_effect():
    for value in (True, False):
        got, msg = _one_warning(lambda: mtb.find_methods("vertical", runnable=value))
        assert got == mtb.find_methods("vertical")
        assert msg == _msg("find_methods(runnable=)")


def test_cite_fmt_is_format():
    got, msg = _one_warning(lambda: mtb.cite("Matilda", fmt="bibtex"))
    assert got == mtb.cite("Matilda", format="bibtex") and got.startswith("@article")
    assert msg == _msg("cite(fmt=)", "format=")
    assert mtb.cite("Matilda") == mtb.cite("Matilda", format="text")


# ------------------------------------------------------------------ run
def test_run_task_has_no_effect_and_convert_still_works():
    inp = mtb.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    ref = mtb.run("Matilda", "vertical", inputs=inp, out_dir="o", dry_run=True)
    got, msg = _one_warning(lambda: mtb.run("Matilda", "vertical", inputs=inp, out_dir="o",
                                            dry_run=True, task="clustering"))
    assert got == ref and msg == _msg("run(task=)")
    got, msg = _one_warning(lambda: mtb.run("Matilda", "vertical", inputs=inp, out_dir="o",
                                            dry_run=True, convert=False))
    assert got == ref
    assert msg == _msg("run(convert=)", "canonical .h5 inputs, which run passes through "
                                        "unchanged")


def test_run_repo_path_still_points_at_the_scripts():
    inp = mtb.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    repo = mtb.config.DEFAULT.repo_path
    got, msg = _one_warning(lambda: mtb.run("Matilda", "vertical", inputs=inp, out_dir="o",
                                            dry_run=True, repo_path=repo))
    assert got == mtb.run("Matilda", "vertical", inputs=inp, out_dir="o", dry_run=True)
    assert msg == _msg("run(repo_path=)", "config.DEFAULT.repo_path")


def test_scan_per_method_still_works():
    default = mtb.scan("D11", "vertical", verbose=False)
    assert list(default.columns) == ["method", "modalities", "runnable", "reason"]
    got, msg = _one_warning(lambda: mtb.scan("D11", "vertical", per_method=True, verbose=False))
    pd.testing.assert_frame_equal(got, default)
    assert msg == _msg("scan(per_method=)", "scan(); run_all(dry_run=True) returns one row "
                                            "per variant")
    full, _ = _one_warning(lambda: mtb.scan("D11", "vertical", per_method=False, verbose=False))
    pd.testing.assert_frame_equal(full, mtb.run_all("D11", "vertical", dry_run=True,
                                                    verbose=False))


def test_batch_result_results_is_records():
    res = mtb.BatchResult([{"method": "Matilda", "status": "CHAIN_OK"}], "D11", "vertical")
    got, msg = _one_warning(lambda: res.results)
    assert got is res.records
    assert msg == _msg("BatchResult.results", "BatchResult.records")


# ------------------------------------------------------------------ evaluate
def test_evaluate_output_and_clustering_keywords():
    rng = np.random.default_rng(0)
    emb, labels = rng.normal(size=(60, 4)), np.repeat(["a", "b", "c"], 20)
    clusters = np.repeat([0, 1, 2], 20)
    ref = mtb.evaluate(emb, labels, metrics=["ARI"], clusters=clusters, verbose=False)
    got, msg = _one_warning(lambda: mtb.evaluate(output=emb, labels=labels, metrics=["ARI"],
                                                 clusters=clusters, verbose=False))
    pd.testing.assert_frame_equal(got, ref)
    assert msg == _msg("evaluate(output=)", "embedding=")
    got, msg = _one_warning(lambda: mtb.evaluate(emb, labels, metrics=["ARI"],
                                                 clustering=clusters, verbose=False))
    pd.testing.assert_frame_equal(got, ref)
    assert msg == _msg("evaluate(clustering=)", "clusters=")


# ------------------------------------------------------------------ stored results
def test_source_still_reads_the_published_tables():
    got, msg = _one_warning(lambda: mtb.load_results("vertical", dataset="D11",
                                                     source="published"))
    assert set(got.source) == {"published"}
    assert msg == _msg("load_results(source=)", "the default, the package's re-run tables")
    assert set(mtb.load_results("vertical", dataset="D11").source) == {"rerun"}


def test_result_path_and_clustering_still_work(result_dir):
    got, msg = _one_warning(lambda: mtb.load_results("vertical", result_path=result_dir))
    pd.testing.assert_frame_equal(got, mtb.load_results("vertical"))
    assert msg == _msg("load_results(result_path=)", "config.DEFAULT.result_path")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mtb.load_results("vertical", dataset="D11", source="published", clustering="default")
    assert _msg("load_results(clustering=)", "the default, the package's re-run tables") in \
        [str(w.message) for w in rec]


def test_available_datasets_is_results_coverage():
    got, msg = _one_warning(mtb.available_datasets)
    assert got == sorted(mtb.results_coverage().dataset.unique())
    assert msg == _msg("available_datasets", "results_coverage")


def test_results_coverage_and_recommend_source():
    _, msg = _one_warning(lambda: mtb.results_coverage("vertical", source="rerun"))
    assert msg == _msg("results_coverage(source=)", "the default, the package's re-run tables")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = mtb.recommend("vertical")
        got, msg = _one_warning(lambda: mtb.recommend("vertical", source="rerun"))
    pd.testing.assert_frame_equal(got, ref)
    assert msg == _msg("recommend(source=)", "the default, the package's re-run tables")


# ------------------------------------------------------------------ catalog / io / plot / env
@pytest.mark.parametrize("name, new", [
    ("canonical_id", "the name as written; every function accepts any spelling"),
    ("canonical_metric", "the name as written; every function accepts any spelling"),
    ("known_metrics", "catalog.metrics()"),
    ("PAPER_COLUMNS", None),
])
def test_catalog_names(name, new):
    value, msg = _one_warning(lambda: getattr(mtb.catalog, name))
    assert msg == _msg(f"catalog.{name}", new)
    assert name not in mtb.catalog.__all__ and name not in dir(mtb.catalog)
    if name == "known_metrics":
        assert value() == list(mtb.catalog.metrics().metric)
    if name == "canonical_id":
        assert value("mofa+") == "MOFA2"


def test_catalog_metrics_has_the_families():
    tab = mtb.catalog.metrics()
    fam = tab.groupby("family", sort=False).metric.apply(list).to_dict()
    assert fam == {"clustering": ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"],
                   "batch": ["ASW_batch", "ARI_batch", "NMI_batch", "GC", "iLISI", "kBET"],
                   "classification": ["OCA", "ACA", "Spec", "Sens", "F1"]}


def test_catalog_methods_uses_ids():
    m = mtb.catalog.methods()
    assert {"MOFA2", "Seurat_WNN"} <= set(m.method)
    assert m.set_index("method").loc["MOFA2", "name"] == "MOFA+"
    assert "dataset name" not in mtb.catalog.datasets().columns


def test_io_normalize_peak_names(tmp_path):
    import h5py
    src, dst = tmp_path / "p.h5", tmp_path / "q.h5"
    with h5py.File(src, "w") as h:
        g = h.create_group("matrix")
        g.create_dataset("data", data=np.ones((2, 3)))
        g.create_dataset("features", data=np.array([b"chr1_1_200", b"chr2-5-9"]))
    fn, msg = _one_warning(lambda: mtb.io.normalize_peak_names)
    fn(src, dst)
    with h5py.File(dst) as h:
        assert [x.decode() for x in h["matrix/features"][:]] == ["chr1:1-200", "chr2:5-9"]
    assert msg == _msg("io.normalize_peak_names", "run, which renames the peaks itself")
    assert "normalize_peak_names" not in mtb.io.__all__ + dir(mtb.io)


@pytest.mark.parametrize("name", ["FAMILIES", "CLUSTERING_METRICS", "BATCH_METRICS",
                                  "CLASSIFICATION_METRICS"])
def test_plot_metric_lists(name):
    value, msg = _one_warning(lambda: getattr(mtb.plot, name))
    assert msg == _msg(f"plot.{name}", "catalog.metrics()")
    assert name not in mtb.plot.__all__ and name not in dir(mtb.plot)
    if name == "BATCH_METRICS":
        assert value == ["ASW_batch", "ARI_batch", "NMI_batch", "GC", "iLISI", "kBET"]


def _long():
    return mtb.load_results("vertical", dataset="D11")


def test_bar_group_is_metrics_and_the_styling_keywords_have_no_effect():
    import matplotlib.pyplot as plt
    _, msg = _one_warning(lambda: mtb.plot.bar(_long(), group="clustering"))
    assert msg == _msg("plot.bar(group=)", "metrics=")
    for kw in ({"cmap": "Reds"}, {"show_datasets": False}):
        _, msg = _one_warning(lambda: mtb.plot.bar(_long(), **kw))
        assert msg == _msg(f"plot.bar({next(iter(kw))}=)")
    plt.close("all")


@pytest.mark.parametrize("kw", [{"na": "skip"}, {"cmap": "Reds"}, {"show_language": False}])
def test_bubble_styling_keywords_have_no_effect(kw):
    import matplotlib.pyplot as plt
    _, msg = _one_warning(lambda: mtb.plot.bubble(_long(), **kw))
    assert msg == _msg(f"plot.bubble({next(iter(kw))}=)")
    plt.close("all")


def test_bar_and_bubble_share_the_rank_default():
    import inspect
    for fn in (mtb.plot.bar, mtb.plot.bubble):
        assert inspect.signature(fn).parameters["overall"].default == "rank"


def test_export_dataset_compression_still_works(tmp_path):
    import anndata as ad
    import h5py
    a = ad.AnnData(np.ones((4, 3), dtype=np.float32))
    a.var_names = ["g1", "g2", "g3"]
    out, msg = _one_warning(lambda: mtb.io.export_dataset(a, tmp_path / "X", compression=None))
    with h5py.File(out / "rna.h5") as h:
        assert h["matrix/data"].compression is None
    assert msg == _msg("io.export_dataset(compression=)", "the default, gzip")


def test_export_dataset_reads_the_atac_kind_from_the_names(tmp_path):
    import anndata as ad
    rna = ad.AnnData(np.ones((4, 3), dtype=np.float32))
    rna.var_names = ["g1", "g2", "g3"]
    peaks = ad.AnnData(np.ones((4, 2), dtype=np.float32))
    peaks.var_names = ["chr1:1-100", "chr2:5-90"]
    genes = rna.copy()
    out = mtb.io.export_dataset(None, tmp_path / "P", rna=rna, atac=peaks, category="vertical")
    assert (out / "atac.h5").exists()
    out = mtb.io.export_dataset(None, tmp_path / "G", rna=rna, atac=genes)
    assert (out / "atac_gas.h5").exists()
    odd = peaks.copy()
    odd.var_names = ["x", "y"]
    with pytest.raises(ValueError, match="neither peaks such as chr1:100-200 nor the RNA's "
                                         "gene names. Pass atac_kind='peak' or "
                                         "atac_kind='gene_activity'."):
        mtb.io.export_dataset(None, tmp_path / "Q", rna=rna, atac=odd)


@pytest.mark.parametrize("name, new", [("status", "env.doctor"), ("plan", "env.install")])
def test_env_names(name, new):
    rows, msg = _one_warning(lambda: getattr(mtb.env, name)(**({} if name == "status" else
                                                             {"methods": ["Matilda"]})))
    assert msg == _msg(f"env.{name}", new)
    assert rows and name not in mtb.env.__all__ and name not in dir(mtb.env)


def test_a_dataset_can_be_a_folder_path(tmp_path, monkeypatch):
    root = mtb.config.DEFAULT.data_path
    monkeypatch.chdir(root.parent)
    rel = f"{root.name}/D11"
    assert mtb.inputs_for(rel, "vertical", "Matilda") == mtb.inputs_for("D11", "vertical",
                                                                        "Matilda")
    assert mtb.labels_for(root / "D11") == mtb.labels_for("D11")
    pd.testing.assert_frame_equal(mtb.scan(rel, "vertical", verbose=False),
                                  mtb.scan("D11", "vertical", verbose=False))


# ------------------------------------------------------------------ the surface
def test_names_left_the_top_level():
    for name in ("list_methods", "list_tasks", "list_categories", "available_datasets",
                 "eval"):
        assert name not in mtb.__all__ and name not in dir(mtb)
    assert callable(mtb.eval.evaluate)          # importable, not listed
    assert dir(mtb.eval) == sorted(["evaluate", "to_long"] + [
        n for n in dir(mtb.eval) if n.startswith("__")])
    assert [n for n in dir(mtb.data) if not n.startswith("_")] == sorted(mtb.data.__all__)


def _help(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        cli.main([*argv, "--help"])
    return out.getvalue()


def test_cli_help_hides_the_deprecated_commands_and_flags():
    top = _help()
    assert "\n    list " not in top and "find" in top
    env = _help("env")
    for hidden in ("status", "groups", "plan", "recipe", "yml", "create", "create-group",
                   "freeze"):
        assert f"\n    {hidden} " not in env, hidden
    for cmd, flags in (("find", ("--task", "--runnable")), ("run", ("--task",)),
                       ("evaluate", ("--output", "--clustering", "--cluster ", "--task",
                                     "--only")),
                       ("plot", ("--source", "--result-path", "--na", "--group")),
                       ("env", ()), ):
        text = _help(*cmd.split())
        for flag in flags:
            assert flag not in text, (cmd, flag)
    assert "--embedding" in _help("evaluate") and "--clusters" in _help("evaluate")
    assert "--build" in _help("env", "install") and "--packed" not in _help("env", "install")
    assert "advanced" in _help("run") and "advanced" in _help("run-all")


def _cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_cli_deprecated_commands_warn_once_and_work():
    code, out, err = _cli("list", "--category", "vertical")
    assert code == 0 and out.split() == mtb.find_methods("vertical")
    assert err.count("warning:") == 1
    assert "warning: multibench list is deprecated and will be removed in 0.5. Use multibench " \
           "find." in err
    code, out, err = _cli("find", "--category", "vertical", "--runnable")
    assert out.split() == mtb.find_methods("vertical")
    assert err.strip() == "warning: --runnable is deprecated and will be removed in 0.5. " \
                          "It has no effect."
    code, out, err = _cli("find", "--task", "dimension_reduction")
    assert out.split() == ["Matilda", "sciPENN"]
    assert err.strip() == "warning: --task is deprecated and will be removed in 0.5. " \
                          "Use --category."
    code, out, err = _cli("env", "groups")
    assert code == 0 and "scmb_torch" in out
    assert "multibench env groups is deprecated and will be removed in 0.5. Use multibench " \
           "env doctor." in err


def test_cli_cite_prints_text_by_default():
    _, out, _ = _cli("cite", "Matilda")
    assert out.strip() == mtb.cite("Matilda")
