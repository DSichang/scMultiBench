"""Tests for the params-discovery API (params_for / method_info['params']).

`tunable` is DOC-ONLY metadata describing what each upstream script accepts on
its command line; it must never change the command the runner builds.
"""
import pytest

import multibench as mtb
from multibench.engine import builder, registry


def test_params_for_returns_defaults_and_tunable():
    r = mtb.params_for("Multigrate", "vertical", ["rna", "adt"])
    assert r["method"] == "Multigrate"
    assert r["variant"] == "vertical:rna+adt"
    # the default is what the package actually emits (Y38: no separate 'defaults';
    # 0.4.0: no 'effective' copy either)
    assert r["tunable"]["epochs"] == {"type": "int", "default": 200}
    assert "defaults" not in r and "effective" not in r
    # tunable is documentation harvested from the upstream argparse
    assert set(r["tunable"]) >= {"epochs", "lr"}


def test_params_for_reports_untunable_methods_honestly():
    # totalVI's upstream script exposes only paths -> nothing to tune
    r = mtb.params_for("totalVI", "vertical", ["rna", "adt"])
    assert r["tunable"] == {}


def test_params_for_tunable_carries_default_and_type():
    # scMDC's lr is declared but never read (X64, r3): gamma stands in
    spec = mtb.params_for("scMDC", "vertical", ["rna", "adt"])["tunable"]
    assert spec["gamma"]["type"] == "float" and "lr" not in spec


def test_params_for_lists_choices_and_a_value_outside_them_is_refused():
    # 0.4.0: an argparse choices= list is recorded and checked before a run
    from multibench.engine import runner
    acc = mtb.params_for("MIRA", "vertical", ["rna", "atac"])["tunable"]["acc"]
    assert set(acc) == {"type", "default", "choices"} and acc["default"] in acc["choices"]
    v = registry.get("MIRA").select("vertical", {"rna", "atac"})
    runner.check_param_types("MIRA", v, {"acc": "cpu"})
    with pytest.raises(ValueError, match=r"^MIRA: acc takes cpu, gpu, .* or auto, got 'banana'\.$"):
        runner.check_param_types("MIRA", v, {"acc": "banana"})


def test_params_for_requires_disambiguation_when_multiple_variants():
    with pytest.raises(KeyError):
        mtb.params_for("Multigrate")          # 3 variants -> ambiguous


def test_params_for_single_variant_needs_no_selector():
    # pick a single-variant method dynamically: hardcoding one breaks as soon as
    # someone wires an extra variant for it (which is exactly what happened to
    # totalVI when its cross variant was added).
    single = next(s.id for s in registry.load() if len(s.variants) == 1)
    r = mtb.params_for(single)                # no selector needed
    assert r["method"] == single
    assert ":" in r["variant"]


def test_tunable_is_doc_only_and_never_emitted():
    """The safety property: documenting a param must not add it to the command."""
    spec = registry.get("scMDC")
    v = spec.select("vertical", {"rna", "adt"})
    assert v.tunable, "scMDC should have documented tunable params"
    cmd = builder.build_command(v, {"rna": "r.h5", "adt": "a.h5"}, "/out/")
    # a documented-but-not-defaulted param must not appear
    assert "--sigma1" not in cmd
    assert "sigma1" in v.tunable
    # while a declared default IS emitted
    assert "--nbatch" in cmd


def test_user_params_override_defaults_in_command():
    spec = registry.get("Multigrate")
    v = spec.select("vertical", {"rna", "adt"})
    cmd = builder.build_command(v, {"rna": "r.h5", "adt": "a.h5"}, "/out/",
                                params={"epochs": 5})
    assert "--epochs" in cmd
    assert cmd[cmd.index("--epochs") + 1] == "5"


def test_params_for_category_only_reaches_data_dir_variants():
    """A data_dir variant has NO modalities, so category alone must be enough.

    Regression: params_for used to demand BOTH category and modalities, which made
    scBridge unreachable through this API.
    """
    r = mtb.params_for("scBridge", "diagonal")
    assert r["variant"].startswith("diagonal")
    assert r["tunable"]


def test_params_for_modalities_only():
    r = mtb.params_for("Multigrate", modalities=["rna", "adt"])
    assert r["variant"] == "vertical:rna+adt"


def test_params_for_ambiguous_category_lists_the_options():
    with pytest.raises(KeyError) as e:
        mtb.params_for("Multigrate", "mosaic")     # 4 mosaic variants, other parameters
    assert "Pass the modalities too" in str(e.value)
    assert "Multigrate has 4 mosaic variants, rna1+rna2+adt2+adt3," in str(e.value)


def test_params_for_unknown_category_is_explicit():
    with pytest.raises(KeyError) as e:
        mtb.params_for("totalVI", "mosaic")        # declared, but no such variant
    assert "totalVI does not run on mosaic data. Its categories: vertical, cross." in str(e.value)


# ------------------------------------------------ X64 / Z05 (r3): what tunable lists
def _gen_params():
    import importlib
    import warnings
    with warnings.catch_warnings():          # the generator silences warnings on import
        return importlib.import_module("multibench.tools.gen_params")


def _scripts():
    """``[(method, variant, path, folder text)]`` of every variant whose script is here."""
    gen = _gen_params()
    out = []
    for s in registry.load():
        for v in s.variants:
            path = gen.resolve(v.entrypoint)
            if path and path.endswith(".py"):
                out.append((s, v, path, gen.folder_text(path)))
    if not out:
        pytest.skip("the method scripts are not in this checkout")
    return out


def test_every_tunable_key_is_an_argument_the_script_reads():
    import re
    for s, v, path, every in _scripts():
        txt = open(path, errors="ignore").read()
        for k in v.tunable:
            assert re.search(r"add_argument\(\s*['\"]--%s['\"]" % re.escape(k), txt), (s.id, k)
            assert _gen_params().is_read(k, every), (s.id, k)


def test_the_keys_left_out_are_script_arguments():
    # the table names real arguments, so it fails loudly when a script changes
    import re
    gen = _gen_params()
    declared = {}
    for s, v, path, _ in _scripts():
        txt = open(path, errors="ignore").read()
        declared.setdefault(s.id, set()).update(
            re.findall(r"add_argument\(\s*['\"]--([A-Za-z0-9_\-]+)['\"]", txt))
    for method, keys in gen.NOT_TUNABLE.items():
        assert keys <= declared[method], (method, keys - declared[method])
    # a cpu_params flag the script does not declare would make argparse refuse the run
    for s in registry.load():
        if s.cpu_params:
            assert set(s.cpu_params) <= declared[s.id], s.id


def test_host_switches_and_dead_knobs_are_not_tunable():
    assert "index" not in mtb.params_for("UnitedNet", "vertical", ["rna", "atac"])["tunable"]
    assert not any(v.tunable for v in registry.get("UnitedNet").variants)
    gone = {("scMDC", "vertical", ("rna", "adt")): {"device", "run"},
            ("scMM", "vertical", ("rna", "atac")): {"no_cuda", "print_freq"},
            ("Matilda", "vertical", ("rna", "adt")): {"device"},
            ("scJoint", "diagonal", ("rna", "atac_gas")): {
                "use_cuda", "threads", "epochs_stage3", "lr_stage3", "with_crossentorpy",
                "center_weight"}}
    for (m, cat, mods), keys in gone.items():
        p = mtb.params_for(m, cat, list(mods))
        assert not keys & set(p["tunable"]), m
    # the package still turns CUDA off on a CPU host
    plan = mtb.scan("D52mini", "cross", methods=["scMDC"], verbose=False)
    assert " --device cpu" in plan.command.iloc[0] or envs_has_gpu()


def envs_has_gpu():
    from multibench.engine import envs
    return envs.host_has_gpu()
