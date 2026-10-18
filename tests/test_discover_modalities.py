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
            len(mtb.find_methods(category="diagonal"))


def test_find_methods_modalities_never_warns():
    """The one directory-fed method (scBridge) names its matrices, so
    ``modalities=`` filters every variant and nothing is kept unfiltered."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for cat in (None, "vertical", "diagonal", "mosaic", "cross"):
            discover.find_methods(category=cat, modalities=["rna"])


def test_recommend_modalities_keeps_scbridge(result_dir):
    a = mtb.recommend("diagonal", modalities=["rna", "atac"])
    b = mtb.recommend("diagonal")
    assert a.method.tolist() == b.method.tolist()
    assert "scBridge" in a.method.tolist()


def test_method_info_supports_exposes_labels():
    for m in mtb.find_methods():
        for s in mtb.method_info(m)["supports"]:
            assert isinstance(s["labels"], list)
    assert mtb.method_info("UnitedNet")["supports"][0]["labels"] == ["cty"]
    assert mtb.method_info("Matilda")["supports"][0]["labels"] == ["cty"]
    assert mtb.method_info("SCALEX")["supports"][0]["labels"] == []


def test_params_for_ambiguity_is_valueerror_with_the_exact_call():
    with pytest.raises(ValueError) as e:
        mtb.params_for("Matilda")
    assert isinstance(e.value, AmbiguousVariantError) and isinstance(e.value, KeyError)
    msg = str(e.value)
    assert ("Pass the category and modalities of one, for example "
            "params_for('Matilda', 'vertical', ['rna', 'adt']).") in msg
    assert "vertical:rna+adt, vertical:rna+atac and vertical:rna+adt+atac" in msg
    with pytest.raises(ValueError) as e:
        mtb.params_for("Matilda", "vertical")
    assert ("Pass the modalities too, for example "
            "params_for('Matilda', 'vertical', ['rna', 'adt']).") in str(e.value)
    assert "Matilda has 3 vertical variants, rna+adt, rna+atac and rna+adt+atac." in str(e.value)
    # scMM's vertical variant reads the atac role since 0.4.0 (X13); unknown
    # category stays KeyError
    assert mtb.params_for("scMM", "vertical", ["rna", "atac"])["variant"] == "vertical:rna+atac"
    assert mtb.params_for("scMM", modalities=["rna", "atac"])["variant"] == "vertical:rna+atac"
    with pytest.raises(KeyError, match="totalVI does not run on mosaic data"):
        mtb.params_for("totalVI", "mosaic")


def test_params_for_dataset_disambiguates_by_folder(tmp_path):
    d = tmp_path / "D11"; d.mkdir()
    for n in ("rna.h5", "adt.h5", "cty.csv"):
        (d / n).write_text("")
    r = mtb.params_for("Matilda", dataset="D11", data_path=tmp_path)
    assert r["variant"] == "vertical:rna+adt"
    assert mtb.params_for("Matilda", "vertical", dataset="D11", data_path=tmp_path)["variant"] == "vertical:rna+adt"
    # every variant satisfiable, and rna+adt and rna+atac are inside
    # rna+adt+atac: the largest one
    (d / "atac.h5").write_text("")
    assert mtb.params_for("Matilda", dataset="D11",
                          data_path=tmp_path)["variant"] == "vertical:rna+adt+atac"
    # two satisfiable variants that are not nested -> still ambiguous
    with pytest.raises(ValueError, match="Pass the category and modalities of one"):
        mtb.params_for("VIMCCA", dataset="D11", data_path=tmp_path)
    # missing folder settles nothing
    with pytest.raises(ValueError):
        mtb.params_for("Matilda", dataset="D99", data_path=tmp_path)
    # dataset= is deprecated in 0.4.0: it keeps its effect and is out of the docs
    doc = mtb.params_for.__doc__
    assert "dataset" not in doc and "Returns" in doc and "Raises" in doc


def test_cite_is_variadic_and_keeps_the_list_form():
    both = mtb.cite("Matilda", "MOFA2", format="bibtex")
    assert both == mtb.cite(["Matilda", "MOFA2"], format="bibtex")
    assert both == mtb.cite(("Matilda", "MOFA2"), format="bibtex")
    assert "Matilda" in both and "MOFA2" in both and both.startswith("@article{scMultiBench")
    assert mtb.cite() == mtb.cite(None)
    assert "https://doi.org/" in mtb.cite() and "@article" not in mtb.cite()   # text is the default
    assert mtb.cite("Matilda") == mtb.cite(["Matilda"], format="text")
    assert mtb.cite("all") == mtb.cite(["all"][0]) and mtb.cite("all", format="bibtex").count("@") > 30
    with pytest.raises(ValueError, match="unknown format 'nope'"):
        mtb.cite("Matilda", format="nope")
    with pytest.raises(KeyError, match="Did you mean Matilda"):
        mtb.cite("Matlida")
    with pytest.raises(TypeError, match="pass one list"):
        mtb.cite(["Matilda"], ["MOFA2"])
    # the 0.2 methods= keyword and the positional fmt are gone: loud
    with pytest.raises(TypeError, match="methods"):
        mtb.cite(methods=["MOFA2"])
    with pytest.raises(TypeError, match="method ids must be strings"):
        mtb.cite(["Matilda"], "text")
