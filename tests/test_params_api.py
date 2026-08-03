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
    # defaults are what the package actually emits
    assert r["defaults"]["epochs"] == 200
    # tunable is documentation harvested from the upstream argparse
    assert set(r["tunable"]) >= {"epochs", "lr"}


def test_params_for_reports_untunable_methods_honestly():
    # totalVI's upstream script exposes only paths -> nothing to tune
    r = mtb.params_for("totalVI", "vertical", ["rna", "adt"])
    assert r["tunable"] == {}


def test_params_for_tunable_carries_default_and_type():
    spec = mtb.params_for("scMDC", "vertical", ["rna", "adt"])["tunable"]
    assert spec["lr"]["default"] == 1.0
    assert spec["lr"]["type"] == "float"


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


def test_method_info_exposes_params_per_variant():
    info = mtb.method_info("Multigrate")
    assert "vertical:rna+adt" in info["params"]
    assert info["params"]["vertical:rna+adt"]["defaults"]["bs"] == 256


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
    scBridge and the spatial methods unreachable through this API.
    """
    r = mtb.params_for("scBridge", "diagonal")
    assert r["variant"].startswith("diagonal")
    assert r["tunable"]
    assert mtb.params_for("PASTE", "cross")["variant"].startswith("cross")


def test_params_for_modalities_only():
    r = mtb.params_for("Multigrate", modalities=["rna", "adt"])
    assert r["variant"] == "vertical:rna+adt"


def test_params_for_ambiguous_category_lists_the_options():
    with pytest.raises(KeyError) as e:
        mtb.params_for("Matilda", "vertical")      # 2 vertical variants
    assert "also pass modalities" in str(e.value)
    assert "vertical:rna+adt" in str(e.value)


def test_params_for_unknown_category_is_explicit():
    with pytest.raises(KeyError) as e:
        mtb.params_for("totalVI", "mosaic")        # declared, but no such variant
    assert "no 'mosaic' variant" in str(e.value)
