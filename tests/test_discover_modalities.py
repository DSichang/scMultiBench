"""find_methods keeps directory-fed methods (P03); supports[i]['labels'];
params_for ambiguity (Rin/Noor/Sam); cite is variadic (Sam)."""
import warnings

import pytest

import multibench as mtb
from multibench import discover
from multibench.engine.schema import AmbiguousVariantError


def test_find_methods_modalities_keeps_scbridge_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "scBridge" in discover.find_methods(category="diagonal", modalities=["rna", "atac"])
        assert "scBridge" in discover.find_methods(category="diagonal", modalities=["rna", "atac_gas"])
        assert "scBridge" not in discover.find_methods(category="diagonal", modalities=["rna", "adt"])
        assert len(discover.find_methods(category="diagonal", modalities=["rna", "atac"])) == \
            len(mtb.list_methods(category="diagonal"))


def test_find_methods_modalities_keeps_data_dir_methods_with_warning():
    with pytest.warns(UserWarning, match="data_dir") as rec:
        ids = discover.find_methods(category="cross", modalities=["rna"])
    assert {"SPIRAL", "GPSA", "PASTE", "PASTE2"} <= set(ids)
    msg = str(rec[0].message)
    for m in ("SPIRAL", "GPSA", "PASTE", "PASTE2"):
        assert m in msg
    assert "could not be filtered by modalities" in msg and "task='registration'" in msg
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        discover.find_methods(category="cross")
        discover.find_methods(task="registration")
    doc = discover.find_methods.__doc__
    assert "KEPT" in doc and "scBridge" in doc


def test_recommend_modalities_keeps_scbridge(result_dir):
    a = mtb.recommend("diagonal", modalities=["rna", "atac"], result_path=result_dir)
    b = mtb.recommend("diagonal", result_path=result_dir)
    assert a.method.tolist() == b.method.tolist()
    assert "scBridge" in a.method.tolist()


def test_method_info_supports_exposes_labels():
    for m in mtb.list_methods():
        for s in mtb.method_info(m)["supports"]:
            assert isinstance(s["labels"], list)
    assert mtb.method_info("UnitedNet")["supports"][0]["labels"] == ["rna_cty"]
    assert mtb.method_info("Matilda")["supports"][0]["labels"] == ["cty"]
    assert mtb.method_info("SCALEX")["supports"][0]["labels"] == []


def test_params_for_ambiguity_is_valueerror_with_the_exact_call():
    with pytest.raises(ValueError) as e:
        mtb.params_for("Matilda")
    assert isinstance(e.value, AmbiguousVariantError) and isinstance(e.value, KeyError)
    msg = str(e.value)
    assert "pass category and modalities, e.g. params_for('Matilda', 'vertical', ['rna', 'adt'])" in msg
    with pytest.raises(ValueError) as e:
        mtb.params_for("Matilda", "vertical")
    assert "also pass modalities, e.g. params_for('Matilda', 'vertical', ['rna', 'adt'])" in str(e.value)
    assert "vertical:rna+adt" in str(e.value)
    # 'atac' selects the atac_gas variant; unknown category stays KeyError
    assert mtb.params_for("scMM", "vertical", ["rna", "atac"])["variant"] == "vertical:rna+atac_gas"
    assert mtb.params_for("scMM", modalities=["rna", "atac"])["variant"] == "vertical:rna+atac_gas"
    with pytest.raises(KeyError, match="no 'mosaic' variant"):
        mtb.params_for("totalVI", "mosaic")


def test_params_for_dataset_disambiguates_by_folder(tmp_path):
    d = tmp_path / "D11"; d.mkdir()
    for n in ("rna.h5", "adt.h5", "cty.csv"):
        (d / n).write_text("")
    r = mtb.params_for("Matilda", dataset="D11", data_path=tmp_path)
    assert r["variant"] == "vertical:rna+adt"
    assert mtb.params_for("Matilda", "vertical", dataset="D11", data_path=tmp_path)["variant"] == "vertical:rna+adt"
    # both variants satisfiable -> still ambiguous
    (d / "atac.h5").write_text("")
    with pytest.raises(ValueError, match="pass category and modalities"):
        mtb.params_for("Matilda", dataset="D11", data_path=tmp_path)
    # missing folder settles nothing
    with pytest.raises(ValueError):
        mtb.params_for("Matilda", dataset="D99", data_path=tmp_path)
    doc = mtb.params_for.__doc__
    assert "dataset" in doc and "Returns" in doc and "Raises" in doc


def test_cite_is_variadic_and_keeps_the_list_form():
    both = mtb.cite("Matilda", "MOFA2", fmt="bibtex")
    assert both == mtb.cite(["Matilda", "MOFA2"], fmt="bibtex")
    assert both == mtb.cite(("Matilda", "MOFA2"), fmt="bibtex")
    assert "Matilda" in both and "MOFA2" in both and both.startswith("@article{scMultiBench")
    assert mtb.cite() == mtb.cite(None)
    assert "https://doi.org/" in mtb.cite() and "@article" not in mtb.cite()   # text is the default
    assert mtb.cite("Matilda") == mtb.cite(["Matilda"], fmt="text")
    assert mtb.cite("all") == mtb.cite(["all"][0]) and mtb.cite("all", fmt="bibtex").count("@") > 30
    with pytest.raises(ValueError, match="unknown fmt 'nope'"):
        mtb.cite("Matilda", fmt="nope")
    with pytest.raises(KeyError, match="did you mean 'Matilda'"):
        mtb.cite("Matlida")
    with pytest.raises(TypeError, match="pass ONE list"):
        mtb.cite(["Matilda"], ["MOFA2"])
    # the 0.2 methods= keyword and the positional fmt are gone: loud
    with pytest.raises(TypeError, match="methods"):
        mtb.cite(methods=["MOFA2"])
    with pytest.raises(TypeError, match="method ids must be strings"):
        mtb.cite(["Matilda"], "text")
