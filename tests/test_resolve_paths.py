"""inputs_for: absolute paths (P01), 'atac' for any ATAC role (P12), the folder
picks the variant (Sam), ValueError for ambiguity (Rin), labels_for validates
its method id (Rin)."""
import os
import warnings

import pytest

import multibench as mtb
from multibench.engine import resolve
from multibench.engine.schema import AmbiguousVariantError


def _touch(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_text("")


def test_inputs_for_returns_absolute_paths_for_relative_data_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "data" / "MYCITE", ["rna.h5", "adt.h5", "cty.csv"])
    got = mtb.inputs_for("MYCITE", "vertical", "Matilda", modalities=["rna", "adt"],
                         data_path="data", check=False)
    assert all(os.path.isabs(p) for p in got.values()), got
    assert got["rna"] == str(tmp_path / "data" / "MYCITE" / "rna.h5")
    # a data_dir method: absolute AND trailing separator
    import anndata as ad
    import numpy as np
    sl = tmp_path / "data" / "MYVISIUM"
    sl.mkdir()
    for i in range(2):
        a = ad.AnnData(np.ones((4, 3))); a.obsm["spatial"] = np.zeros((4, 2))
        a.write_h5ad(sl / f"s{i}.h5ad")
    got = mtb.inputs_for("MYVISIUM", "cross", "PASTE", data_path="data", check=False)
    assert os.path.isabs(got["data_dir"]) and got["data_dir"].endswith(os.sep)
    assert got["data_dir"] == os.path.join(str(sl), "")


def test_inputs_for_docstring_promises_absolute_paths():
    import inspect
    assert "ABSOLUTE" in inspect.getdoc(mtb.inputs_for)


def test_atac_means_any_atac_role(tmp_path):
    _touch(tmp_path / "SYNMULTI", ["rna.h5", "atac.h5", "cty.csv"])
    # scMM's vertical variant is declared [rna, atac_gas]; the file is atac.h5 either way
    a = mtb.inputs_for("SYNMULTI", "vertical", "scMM", modalities=["rna", "atac"],
                       data_path=tmp_path, check=True)
    b = mtb.inputs_for("SYNMULTI", "vertical", "scMM", modalities=["rna", "atac_gas"],
                       data_path=tmp_path, check=True)
    assert a == b and set(a) == {"rna", "atac_gas"} and a["atac_gas"].endswith("/atac.h5")
    # exact tokens still win when they exist; an unrelated set still raises KeyError
    with pytest.raises(KeyError, match="no variant for category='vertical'"):
        mtb.inputs_for("SYNMULTI", "vertical", "scMM", modalities=["adt", "atac"],
                       data_path=tmp_path, check=False)
    # the aliases the layout text uses reach the same variant
    c = mtb.inputs_for("SYNMULTI", "vertical", "scMM", modalities=["rna", "gas"],
                       data_path=tmp_path, check=False)
    assert c == a


def test_folder_disambiguates_multi_variant_method(tmp_path):
    _touch(tmp_path / "D11", ["rna.h5", "adt.h5", "cty.csv"])
    got = mtb.inputs_for("D11", "vertical", "Matilda", data_path=tmp_path, check=True)
    assert set(got) == {"rna", "adt", "cty"}
    # scMM as well: vertical rna+adt vs rna+atac_gas, only atac.h5 present
    _touch(tmp_path / "SYNMULTI", ["rna.h5", "atac.h5", "cty.csv"])
    got = mtb.inputs_for("SYNMULTI", "vertical", "scMM", data_path=tmp_path, check=True)
    assert set(got) == {"rna", "atac_gas"}


def test_ambiguity_is_a_valueerror_listing_the_folder(tmp_path):
    # both Matilda variants satisfiable -> still ambiguous, ValueError (and KeyError for old callers)
    _touch(tmp_path / "D11", ["rna.h5", "adt.h5", "atac.h5", "cty.csv"])
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("D11", "vertical", "Matilda", data_path=tmp_path)
    msg = str(e.value)
    assert isinstance(e.value, AmbiguousVariantError) and isinstance(e.value, KeyError)
    assert "multiple variants" in msg and "pass modalities= to disambiguate" in msg
    assert "2 of them have every input file" in msg and "adt.h5" in msg
    assert "e.g. modalities=" in msg
    assert not msg.startswith('"')          # plain message, no KeyError quoting
    # nothing on disk -> same error, no folder note
    with pytest.raises(ValueError, match="pass modalities= to disambiguate"):
        mtb.inputs_for("D99", "vertical", "Matilda", data_path=tmp_path)


def test_labels_for_validates_method_and_returns_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "data" / "D11", ["rna.h5", "adt.h5", "cty.csv"])
    with pytest.raises(KeyError, match="did you mean 'Matilda'"):
        mtb.labels_for("D11", method="Matlida", data_path="data")
    got = mtb.labels_for("D11", data_path="data")
    assert list(got) == ["cty"] and os.path.isabs(got["cty"])
    # method + category where the folder settles the variant: no error, same set
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mtb.labels_for("D11", "vertical", "Matilda", data_path="data") == got
    doc = mtb.labels_for.__doc__
    assert "validated whenever given" in doc


def test_select_variant_helper_is_shared():
    from multibench.engine import registry
    spec = registry.get("Matilda")
    with pytest.raises(AmbiguousVariantError):
        resolve.select_variant(spec, "vertical", None)
    v = resolve.select_variant(spec, "vertical", ["rna", "atac"])
    assert v.when["modalities"] == ["rna", "atac"]
    with pytest.raises(KeyError, match="has no variant for category='cross'"):
        resolve.select_variant(spec, "cross", None)
