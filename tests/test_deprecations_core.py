"""The names the 0.4.0 core API cuts or merges: each works in 0.4.x with one
FutureWarning naming its replacement, and is hidden from __all__ and dir()."""
import warnings

import numpy as np
import pytest

import multibench as mtb

D11 = lambda: mtb.config.DEFAULT.data_path / "D11"                      # noqa: E731
EMB = np.random.default_rng(0).normal(size=(40, 3))
LAB = ["T", "B"] * 20


def _wide():
    return mtb.eval.pipeline._scores(EMB, LAB, metrics=["ARI"], verbose=False)


CALLS = {
    "to_long": (lambda: mtb.to_long(_wide(), method="X"), "Use mtb.evaluate(..., method=, dataset=)."),
    "labels_for": (lambda: mtb.labels_for("D11"), "Use mtb.evaluate(..., dataset=, method=)."),
    "inputs_for": (lambda: mtb.inputs_for("D11", "vertical", "Matilda"),
                   "Use mtb.run_all(dataset, category, methods=[method])."),
    "run": (lambda: mtb.run("Matilda", "vertical", out_dir="o", dry_run=True,
                            inputs=mtb.engine.resolve.inputs_for("D11", "vertical", "Matilda")),
            "Use mtb.run_all(dataset, category, methods=[method])."),
    "classify": (lambda: mtb.classify(EMB, labels=LAB, batch=["x"] * 20 + ["y"] * 20),
                 "Use mtb.evaluate(..., metrics='classification')."),
    "results_coverage": (lambda: mtb.results_coverage("vertical"), "Use mtb.load_results(category)."),
    "AmbiguousVariantError": (lambda: mtb.AmbiguousVariantError, "Use except ValueError."),
    "config.Config": (lambda: mtb.config.Config, "Use mtb.config.DEFAULT."),
    "catalog.methods": (lambda: mtb.catalog.methods(),
                        "Use mtb.find_methods() and mtb.method_info(method)."),
    "data.fetchable": (lambda: mtb.data.fetchable(), "Use mtb.catalog.datasets()."),
    "plot.build_table": (lambda: mtb.plot.build_table(mtb.load_results("vertical")),
                         "Use mtb.recommend(category, long_df=...)."),
    "plot.BubbleTable": (lambda: mtb.plot.BubbleTable, "Use mtb.recommend(category, long_df=...)."),
    "env.doctor": (lambda: mtb.env.doctor(methods=["Matilda"]),
                   "Use mtb.env.install(methods, dry_run=True)."),
    "scan(data_path=)": (lambda: mtb.scan("D11", data_path=D11().parent, verbose=False),
                         "Use the dataset folder path or mtb.config.DEFAULT.data_path."),
    "run_all(assume_gpu=)": (lambda: mtb.run_all("D11", "vertical", dry_run=True, verbose=False,
                                                 assume_gpu=True),
                             "Use mtb.scan(..., assume_gpu=True)."),
    "find_methods(tunable=)": (lambda: mtb.find_methods(tunable=True), "Use mtb.params_for."),
    "method_info(verbose=)": (lambda: mtb.method_info("Matilda", verbose=True), "Use mtb.params_for."),
    "params_for(dataset=)": (lambda: mtb.params_for("Matilda", dataset="D11"),
                             "Use params_for(method, category, modalities)."),
    "evaluate(missing_labels=)": (lambda: mtb.evaluate(EMB, LAB, metrics="ARI", verbose=False,
                                                       missing_labels="drop"),
                                  "; keep the cells that have a label"),
    "recommend(min_methods=)": (lambda: mtb.recommend("vertical", min_methods=2), "; it has no effect."),
    "plot.bar(top=)": (lambda: mtb.plot.bar(mtb.load_results("vertical"), top=3), "Use methods=."),
    "io.read_canonical(sparse=)": (lambda: mtb.io.read_canonical(D11() / "adt.h5", sparse=False),
                                   "Use the default."),
    "data.write_example(quiet=)": (None, "Use verbose=."),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_warns_once_and_still_works(name, tmp_path):
    call, tail = CALLS[name]
    if call is None:
        call = lambda: mtb.data.write_example("D11", tmp_path, quiet=True)   # noqa: E731
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        call()
    future = [str(w.message) for w in seen if issubclass(w.category, FutureWarning)]
    assert len(future) == 1, future
    assert "is deprecated and will be removed in 0.5" in future[0] and tail in future[0]


@pytest.mark.parametrize("module, name", [
    (mtb, "to_long"), (mtb, "labels_for"), (mtb, "inputs_for"), (mtb, "run"), (mtb, "sweep"),
    (mtb, "classify"), (mtb, "classification_metrics"), (mtb, "results_coverage"),
    (mtb, "AmbiguousVariantError"), (mtb.config, "Config"), (mtb.catalog, "methods"),
    (mtb.data, "fetchable"), (mtb.io, "to_canonical"), (mtb.plot, "build_table"),
    (mtb.plot, "BubbleTable"), (mtb.env, "doctor")])
def test_hidden_from_all_and_dir(module, name):
    assert name not in module.__all__
    assert name not in dir(module)


def test_the_still_raised_variant_error_is_a_value_error():
    from multibench.engine.schema import AmbiguousVariantError
    assert issubclass(AmbiguousVariantError, ValueError)


@pytest.mark.parametrize("argv, flag", [
    (["find", "--tunable"], "find --tunable"),
    (["config", "--get", "data_path"], "config --get"),
    (["cite", "--all"], "cite --all"),
    (["env", "doctor", "--methods", "Matilda"], "multibench env doctor"),
])
def test_command_line_names_warn_once_with_their_command(argv, flag, capsys):
    from multibench import cli
    assert cli.main(argv) == 0
    err = capsys.readouterr().err
    assert err.count("is deprecated and will be removed in 0.5") == 1, err
    assert f"warning: {flag} is deprecated" in err


def test_convert_is_export_under_its_old_name(tmp_path, capsys):
    from multibench import cli
    mtb.data.write_example("D11", tmp_path, verbose=False)
    assert cli.main(["convert", str(tmp_path / "my_file.h5ad"), str(tmp_path / "A"),
                     "--rna", "X", "--adt", "obsm:protein"]) == 0
    assert ("warning: multibench convert is deprecated and will be removed in 0.5. Use "
            "multibench export.") in capsys.readouterr().err
    assert cli.main(["export", str(tmp_path / "my_file.h5ad"), str(tmp_path / "B"),
                     "--adt", "obsm:protein"]) == 0
    names = lambda p: sorted(x.name for x in p.iterdir())                # noqa: E731
    assert names(tmp_path / "A") == names(tmp_path / "B") == ["adt.h5", "rna.h5"]
    assert "is deprecated" not in capsys.readouterr().err   # export: RNA from X by default
