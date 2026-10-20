"""export_dataset says when it writes a feature name more than once.

A 10x file names some genes twice, and scanpy reads them as they are. Every
selector form wrote such names without a word: ``X``, a layer, a feature
filter, an ``obsm`` entry, an object and a MuData modality. The names are
still written as given; the warning gives their number, the first of them
and the line that makes them unique for the form that was used.

A selector reads the object passed as ``data``. Its variable name is not
known to the package, so the warning calls it the object you export; a
modality passed as its own object is named by its keyword.

``var_names_make_unique()`` is named for an object that holds these features
alone. On an object with genes and antibodies it would also rename an
antibody that has the name of a gene: there the warning names the recipe
"A 10x .h5 file", which splits the object first.
"""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench.engine import ingest

ad = pytest.importorskip("anndata")

N = 20
GENES = ["G1", "G2", "G2", "G3", "G3", "G4"]
PROTEINS = ["P1", "P2", "G1", "P3"]                  # an antibody named like a gene
TWO = "has 2 repeated feature names, for example G2."
SPLIT = 'Split the object first, as in the recipe "A 10x .h5 file" in the docs.'
#: the advice for a selector that reads the whole object passed as data
RUN = "Run var_names_make_unique() on the object you export."


def _cite():
    """Genes and antibodies in one X, as sc.read_10x_h5(gex_only=False) gives them."""
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # anndata: names are not unique
        a = ad.AnnData(rng.poisson(1.0, (N, len(GENES) + len(PROTEINS))).astype(float))
        a.var_names = GENES + PROTEINS
    a.obs_names = [f"cell{i}-1" for i in range(N)]
    a.var["feature_types"] = ["Gene Expression"] * len(GENES) \
        + ["Antibody Capture"] * len(PROTEINS)
    a.layers["counts"] = a.X.copy()
    a.obs["celltype"] = ["T", "B"] * (N // 2)
    return a


def _genes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _cite()
        return a[:, a.var["feature_types"] == "Gene Expression"].copy()


def _export(tmp_path, *args, **kw):
    """The folder and the warnings of export_dataset about feature names."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = ingest.export_dataset(*args[:1], tmp_path / "OUT", *args[1:], overwrite=True,
                                    **kw)
    return out, [str(w.message) for w in rec
                 if re.search("(feature|peak) name", str(w.message))]


def test_a_feature_filter(tmp_path):
    out, said = _export(tmp_path, _cite(), rna="X[feature_types=Gene Expression]",
                        adt="X[feature_types=Antibody Capture]")
    assert said == [f"rna='X[feature_types=Gene Expression]' {TWO} {SPLIT}"]
    # the names are written as they are, as before
    assert list(ingest.read_canonical(out / "rna.h5").var_names) == GENES
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == PROTEINS


def test_the_advice_for_a_feature_filter_works(tmp_path):
    """The split of the recipe "A 10x .h5 file", which the warning names."""
    adata = _cite()
    part = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
    part.var_names_make_unique()
    out, said = _export(tmp_path, adata, rna=part, adt="X[feature_types=Antibody Capture]")
    assert said == []
    assert list(ingest.read_canonical(out / "rna.h5").var_names) == [
        "G1", "G2", "G2-1", "G3", "G3-1", "G4"]
    # the antibody G1 keeps its name; on the whole object it would be G1-1
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == PROTEINS
    adata.var_names_make_unique()
    assert list(adata.var_names[len(GENES):]) == ["P1", "P2", "G1-1", "P3"]


def test_a_layer_with_a_feature_filter(tmp_path):
    _, said = _export(tmp_path, _cite(), rna="layer:counts[feature_types=Gene Expression]")
    assert said == [f"rna='layer:counts[feature_types=Gene Expression]' {TWO} {SPLIT}"]


@pytest.mark.parametrize("selector", ["X", "layer:counts"])
def test_x_and_a_layer(tmp_path, selector):
    out, said = _export(tmp_path, _genes(), rna=selector)
    assert said == [f"rna={selector!r} {TWO} {RUN}"]
    assert "adata" not in said[0]                        # no variable name is guessed
    assert list(ingest.read_canonical(out / "rna.h5").var_names) == GENES
    adata = _genes()
    adata.var_names_make_unique()
    assert _export(tmp_path, adata, rna=selector)[1] == []


def test_the_advice_after_a_split_is_for_the_object_that_is_exported(tmp_path):
    """The recipe splits a 10x object into ``rna`` and ``adt`` and exports
    ``rna``. No object is called ``adata`` then, and the call on the object
    before the split changes nothing in ``rna``. Done as the warning says, on
    the object that is exported, the gene names are unique and the antibody
    named like a gene keeps its name."""
    whole = _cite()
    kind = whole.var["feature_types"]
    rna = whole[:, kind == "Gene Expression"].copy()
    adt = whole[:, kind == "Antibody Capture"].copy()
    out, said = _export(tmp_path, rna, rna="X", adt=adt)
    assert said == [f"rna='X' {TWO} {RUN}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole.var_names_make_unique()                    # what 0.3.5 first said: no effect
    assert _export(tmp_path, rna, rna="X", adt=adt)[1] == said
    rna.var_names_make_unique()                          # the object that is exported
    out, said = _export(tmp_path, rna, rna="X", adt=adt)
    assert said == []
    assert list(ingest.read_canonical(out / "rna.h5").var_names) == [
        "G1", "G2", "G2-1", "G3", "G3-1", "G4"]
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == PROTEINS


def test_an_object(tmp_path):
    rna = _genes()
    _, said = _export(tmp_path, None, rna=rna)
    assert said == [f"rna=AnnData {TWO} Run rna.var_names_make_unique() before the "
                    f"export."]
    rna.var_names_make_unique()
    assert _export(tmp_path, None, rna=rna)[1] == []


def test_an_object_next_to_another_modality_keeps_the_names_of_that_one(tmp_path):
    """rna=rna beside the antibodies as data: the line of the warning renames
    genes only."""
    whole = _cite()
    adt = whole[:, whole.var["feature_types"] == "Antibody Capture"].copy()
    rna = _genes()
    _, said = _export(tmp_path, adt, rna=rna, adt="X")
    assert said == [f"rna=AnnData {TWO} Run rna.var_names_make_unique() before the "
                    f"export."]
    rna.var_names_make_unique()
    out, said = _export(tmp_path, adt, rna=rna, adt="X")
    assert said == []
    assert list(ingest.read_canonical(out / "rna.h5").var_names) == [
        "G1", "G2", "G2-1", "G3", "G3-1", "G4"]
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == PROTEINS


def test_a_mudata_modality(tmp_path):
    mudata = pytest.importorskip("mudata")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mudata.MuData({"rna": _genes()})
    for selector in ("rna", "mod:rna"):
        _, said = _export(tmp_path, mdata, rna=selector)
        assert said == [f"rna={selector!r} {TWO} Run var_names_make_unique() on "
                        f"mod['rna'] of the object you export."]
    mdata.mod["rna"].var_names_make_unique()             # as the warning says
    assert _export(tmp_path, mdata, rna="mod:rna")[1] == []


def test_an_obsm_entry_a_dataframe_and_adt_names(tmp_path):
    rna = _genes()
    rna.var_names_make_unique()
    values = np.random.default_rng(1).poisson(1.0, (N, 3)).astype(float)
    one = "has 1 repeated feature name, for example CD3. Give each column its own name."
    rna.obsm["protein"] = pd.DataFrame(values, index=rna.obs_names,
                                       columns=["CD3", "CD3", "CD4"])
    out, said = _export(tmp_path, rna, rna="X", adt="obsm:protein")
    assert said == [f"adt='obsm:protein' {one}"]
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == ["CD3", "CD3", "CD4"]
    frame = pd.DataFrame(values, index=rna.obs_names, columns=["CD3", "CD3", "CD4"])
    del rna.obsm["protein"]
    assert _export(tmp_path, rna, rna="X", adt=frame)[1] == [f"adt=DataFrame {one}"]
    rna.obsm["protein"] = values
    assert _export(tmp_path, rna, rna="X", adt="obsm:protein",
                   adt_names=["CD3", "CD3", "CD4"])[1] == [f"adt='obsm:protein' {one}"]
    assert _export(tmp_path, rna, rna="X", adt="obsm:protein",
                   adt_names=["CD3", "CD4", "CD8"])[1] == []


def test_peaks_are_not_renamed(tmp_path):
    rna = _genes()
    rna.var_names_make_unique()
    peaks = ad.AnnData(np.ones((N, 4)))
    peaks.obs_names = rna.obs_names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks.var_names = ["chr1:100-600", "chr1:100-600", "chr1:1000-1500",
                           "chr2:100-600"]
    _, said = _export(tmp_path, rna, rna="X", atac=peaks, atac_kind="peak")
    # a peak name is a region: the advice does not rename it
    assert said == ["atac=AnnData has 1 repeated peak name, for example chr1:100-600. Keep "
                    "one column for each peak."]


def test_many_names_show_one(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.AnnData(np.ones((N, 4000)))
        a.var_names = [f"GENE{i}" for i in range(2000)] * 2
    a.obs_names = [f"cell{i}-1" for i in range(N)]
    _, said = _export(tmp_path, a, rna="X")
    assert said == [f"rna='X' has 2,000 repeated feature names, for example GENE0. {RUN}"]


def test_unique_names_give_no_message(tmp_path):
    a = _genes()
    a.var_names_make_unique()
    a.obsm["protein"] = pd.DataFrame(np.ones((N, 2)), index=a.obs_names,
                                     columns=["CD3", "CD4"])
    assert _export(tmp_path, a, rna="X", adt="obsm:protein", labels="obs:celltype")[1] == []


