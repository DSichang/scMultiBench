"""export_dataset says when it writes a feature name more than once.

A 10x file names some genes twice, and scanpy reads them as they are. Every
selector form wrote such names without a word: ``X``, a layer, a feature
filter, an ``obsm`` entry, an object and a MuData modality. The names are
still written as given; the warning gives their number, up to three of them
and the line that makes them unique for the form that was used.

``var_names_make_unique()`` is named for an object that holds these features
alone. On an object with genes and antibodies it would also rename an
antibody that has the name of a gene (the recipe "A 10x .h5 file").
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench import cli
from multibench.engine import ingest

ad = pytest.importorskip("anndata")

N = 20
GENES = ["G1", "G2", "G2", "G3", "G3", "G4"]
PROTEINS = ["P1", "P2", "G1", "P3"]                  # an antibody named like a gene
TWO = ("2 feature names occur more than once: G2 and G3. They are written as they are.")
WHOLE = ("On the whole object, var_names_make_unique() also renames a feature of another "
         "type that has the same name.")


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
    return out, [str(w.message) for w in rec if "feature name" in str(w.message)]


def test_a_feature_filter(tmp_path):
    out, said = _export(tmp_path, _cite(), rna="X[feature_types=Gene Expression]",
                        adt="X[feature_types=Antibody Capture]")
    assert said == [
        f"In rna='X[feature_types=Gene Expression]', {TWO} To make them unique, take these "
        f"features into their own object first: part = adata[:, adata.var['feature_types'] "
        f"== 'Gene Expression'].copy(). Then run part.var_names_make_unique() and pass "
        f"rna=part. {WHOLE}"]
    # the names are written as they are, as before
    assert list(ingest.read_canonical(out / "rna.h5").var_names) == GENES
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == PROTEINS


def test_the_advice_for_a_feature_filter_works(tmp_path):
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
    assert said == [
        f"In rna='layer:counts[feature_types=Gene Expression]', {TWO} To make them unique, "
        f"take these features into their own object first: part = adata[:, "
        f"adata.var['feature_types'] == 'Gene Expression'].copy(). Then run "
        f"part.var_names_make_unique(), set part.X = part.layers['counts'] and pass "
        f"rna=part. {WHOLE}"]


@pytest.mark.parametrize("selector", ["X", "layer:counts"])
def test_x_and_a_layer(tmp_path, selector):
    out, said = _export(tmp_path, _genes(), rna=selector)
    assert said == [f"In rna={selector!r}, {TWO} To make them unique, run "
                    f"adata.var_names_make_unique() before the export."]
    assert list(ingest.read_canonical(out / "rna.h5").var_names) == GENES
    adata = _genes()
    adata.var_names_make_unique()
    assert _export(tmp_path, adata, rna=selector)[1] == []


def test_an_object(tmp_path):
    rna = _genes()
    _, said = _export(tmp_path, None, rna=rna)
    assert said == [f"In rna=AnnData, {TWO} To make them unique, run "
                    f"rna.var_names_make_unique() before the export."]
    rna.var_names_make_unique()
    assert _export(tmp_path, None, rna=rna)[1] == []


def test_a_mudata_modality(tmp_path):
    mudata = pytest.importorskip("mudata")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mudata.MuData({"rna": _genes()})
    for selector in ("rna", "mod:rna"):
        _, said = _export(tmp_path, mdata, rna=selector)
        assert said == [f"In rna={selector!r}, {TWO} To make them unique, run "
                        f"mdata.mod['rna'].var_names_make_unique() before the export."]


def test_an_obsm_entry_a_dataframe_and_adt_names(tmp_path):
    rna = _genes()
    rna.var_names_make_unique()
    values = np.random.default_rng(1).poisson(1.0, (N, 3)).astype(float)
    one = ("1 feature name occurs more than once: CD3. It is written as it is. Give each "
           "column its own name before the export, or pass unique names as adt_names=.")
    rna.obsm["protein"] = pd.DataFrame(values, index=rna.obs_names,
                                       columns=["CD3", "CD3", "CD4"])
    out, said = _export(tmp_path, rna, rna="X", adt="obsm:protein")
    assert said == [f"In adt='obsm:protein', {one}"]
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == ["CD3", "CD3", "CD4"]
    frame = pd.DataFrame(values, index=rna.obs_names, columns=["CD3", "CD3", "CD4"])
    del rna.obsm["protein"]
    assert _export(tmp_path, rna, rna="X", adt=frame)[1] == [f"In adt=DataFrame, {one}"]
    rna.obsm["protein"] = values
    assert _export(tmp_path, rna, rna="X", adt="obsm:protein",
                   adt_names=["CD3", "CD3", "CD4"])[1] == [f"In adt='obsm:protein', {one}"]
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
    assert said == [
        "In atac=AnnData, 1 feature name occurs more than once: chr1:100-600. It is "
        "written as it is. A peak name is a region, so do not rename it. Keep one column "
        "for each peak before the export."]


def test_many_names_show_three(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.AnnData(np.ones((N, 4000)))
        a.var_names = [f"GENE{i}" for i in range(2000)] * 2
    a.obs_names = [f"cell{i}-1" for i in range(N)]
    _, said = _export(tmp_path, a, rna="X")
    assert said == [
        "In rna='X', 2,000 feature names occur more than once, for example GENE0, GENE1 "
        "and GENE2. They are written as they are. To make them unique, run "
        "adata.var_names_make_unique() before the export."]


def test_unique_names_give_no_message(tmp_path):
    a = _genes()
    a.var_names_make_unique()
    a.obsm["protein"] = pd.DataFrame(np.ones((N, 2)), index=a.obs_names,
                                     columns=["CD3", "CD4"])
    assert _export(tmp_path, a, rna="X", adt="obsm:protein", labels="obs:celltype")[1] == []


def test_the_command_line(tmp_path, capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _cite().write_h5ad(tmp_path / "cite.h5ad")
        _genes().write_h5ad(tmp_path / "genes.h5ad")
    capsys.readouterr()
    assert cli.main(["convert", str(tmp_path / "cite.h5ad"), str(tmp_path / "A"), "--rna",
                     "X[feature_types=Gene Expression]"]) == 0
    assert (
        f"In --rna X[feature_types=Gene Expression], {TWO} To make them unique, keep these "
        f"features in their own file first. In Python: part = adata[:, "
        f"adata.var['feature_types'] == 'Gene Expression'].copy(), then "
        f'part.var_names_make_unique() and part.write_h5ad("one_type.h5ad"). {WHOLE}'
    ) in capsys.readouterr().err
    assert cli.main(["convert", str(tmp_path / "genes.h5ad"), str(tmp_path / "B"), "--rna",
                     "X"]) == 0
    assert (f"In --rna X, {TWO} To make them unique, run adata.var_names_make_unique() in "
            f"Python and save the file again.") in capsys.readouterr().err
