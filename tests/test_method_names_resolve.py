"""A method can be named by its id in any letter case or by its display name.

``load_results`` and ``recommend`` took ``"totalvi"`` and ``"MOFA+"``; ``run_all``,
``scan``, ``method_info``, ``cite``, ``params_for``, ``inputs_for``, ``labels_for``,
``run`` and the env functions raised "Unknown method". They all resolve the
name through ``registry.check_method`` now; an unknown name still raises with
the did-you-mean.
"""
from __future__ import annotations

import pytest

import multibench as mtb
from multibench import cli
from multibench.engine import registry


@pytest.mark.parametrize("name, mid", [("totalvi", "totalVI"), ("TOTALVI", "totalVI"),
                                       ("MOFA+", "MOFA2"), ("Seurat(WNN)", "Seurat_WNN"),
                                       ("seurat v3", "Seurat_v3"), ("Matilda", "Matilda")])
def test_check_method_returns_the_id(name, mid):
    assert registry.check_method(name) == mid
    assert registry.get(name).id == mid


def test_unknown_names_still_raise_with_the_hint():
    with pytest.raises(KeyError, match="Unknown method Matlida. Did you mean Matilda?"):
        registry.check_method("Matlida")
    with pytest.raises(KeyError, match="Unknown method Seurat v4"):
        registry.check_method("Seurat v4")        # an alias, but no method of the package


def test_discovery_functions():
    assert mtb.method_info("totalvi")["id"] == "totalVI"
    assert mtb.params_for("matilda", "vertical", ["rna", "adt"])["method"] == "Matilda"
    assert mtb.cite("MOFA+") == mtb.cite("MOFA2")
    assert mtb.cite(["seurat(wnn)"]) == mtb.cite("Seurat_WNN")


def test_inputs_and_labels(tmp_path):
    d = tmp_path / "C3"
    d.mkdir()
    for f in [f"{m}{i}.h5" for m in ("rna", "adt") for i in (1, 2, 3)]:
        (d / f).write_text("")
    for i in (1, 2, 3):
        (d / f"cty{i}.csv").write_text("x\nA\n")
    assert list(mtb.labels_for("C3", "cross", "stabmap", data_path=tmp_path)) == \
        ["cty3", "cty1", "cty2"]
    assert mtb.inputs_for("C3", "cross", "STABMAP", data_path=tmp_path) == \
        mtb.inputs_for("C3", "cross", "StabMap", data_path=tmp_path)


def test_run_dry_run_uses_the_methods_env():
    argv = mtb.run("totalvi", "vertical", inputs={"rna": "/a.h5", "adt": "/b.h5"},
                   out_dir="/o", dry_run=True)
    assert argv == mtb.run("totalVI", "vertical", inputs={"rna": "/a.h5", "adt": "/b.h5"},
                           out_dir="/o", dry_run=True)


def test_env_functions():
    assert mtb.env.plan(methods=["Seurat(WNN)"]) == mtb.env.plan(methods=["Seurat_WNN"])
    assert [r["methods"] for r in mtb.env.doctor(methods=["matilda"]).to_dict("records")] == [["Matilda"]]
    assert [r["methods"] for r in mtb.env.install(["scipenn"], dry_run=True).to_dict("records")] == [["sciPENN"]]
    assert mtb.env.recipe("matilda") == mtb.env.recipe("Matilda")


def test_scan_and_run_all_take_any_case(tmp_path):
    d = tmp_path / "V"
    d.mkdir()
    for f in ("rna.h5", "adt.h5", "cty.csv"):
        (d / f).write_text("")
    df = mtb.scan("V", "vertical", methods=["totalvi"], data_path=tmp_path, verbose=False)
    assert list(df.method) == ["totalVI"]
    with pytest.raises(KeyError, match="totalVI does not accept"):
        mtb.run_all("V", "vertical", methods=["TotalVI"], data_path=tmp_path,
                    params={"TOTALVI": {"nonsense": 1}}, dry_run=True, verbose=False)


def test_cli_run_takes_any_case(capsys):
    assert cli.main(["run", "--method", "totalvi", "--category", "vertical",
                     "--out-dir", "/o", "--input", "rna=/a.h5", "--input", "adt=/b.h5",
                     "--dry-run"]) == 0
    assert "scmb_scvi" in capsys.readouterr().out
