"""Guards of export_dataset on what the cells carry.

- Data with 0 cells is refused. 0.3.4 wrote files with no cell in them.
- A matrix with 0 features is refused. 0.3.4 wrote a file of barcodes.
- A batch column with a missing value is refused. 0.3.4 left those cells
  out without a word, or made the text ``'nan'`` batch 1.
- A label that a label file reads back as missing (``'NA'``, ``'None'``,
  ...) is refused, by the definition ``mtb.evaluate`` scores with
  (``eval.io.missing_mask``): what passes here is read back as a label.
- ADT taken from ``X`` is refused only when the object holds another matrix
  that could be the protein one. An embedding (``obsm["X_umap"]``) is none.
- A sparse ``X`` left on disk (``backed="r"``) is refused before the folder
  is made. 0.3.4 refused it afterwards and left the folder empty.
"""
import random
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench.engine import ingest
from multibench.eval import io as eio

ad = pytest.importorskip("anndata")

NO_CELLS = ("The data holds 0 cells, so there is nothing to write. Check the filter that "
            "selects the cells.")


def _cite(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n, 20)).astype(float))
    a.var_names = [f"GENE{i}" for i in range(20)]
    a.obs_names = [f"cell{i}-1" for i in range(n)]
    a.obs["ct"] = np.array(["T", "B", "NK"])[np.arange(n) % 3]
    a.obs["donor"] = np.array(["d1", "d2"])[np.arange(n) % 2]
    return a


def _peaks(n=60, seed=2):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(0.5, (n, 30)).astype(float))
    a.var_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(30)]
    a.obs_names = [f"cell{i}-1" for i in range(n)]
    return a


def _value_error(tmp_path, data, **kw):
    out = tmp_path / "OUT"
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(data, out, **kw)
    assert not out.exists(), "a refused call made the folder"
    return str(e.value)


# -------------------------------------------------------------------- 0 cells
def test_zero_cells_are_refused(tmp_path):
    a = _cite()
    none = a[:0].copy()
    assert _value_error(tmp_path, none, rna="X") == NO_CELLS
    assert _value_error(tmp_path, none, rna="X", labels="obs:ct") == NO_CELLS
    # labels alone
    assert _value_error(tmp_path, none, rna=None, labels="obs:ct") == NO_CELLS
    assert _value_error(tmp_path, None, rna=None, labels=[]) == NO_CELLS
    # one batch, a batch column, one side of unpaired data
    assert _value_error(tmp_path, none, rna="X", batch_index=1, category="cross") == NO_CELLS
    assert _value_error(tmp_path, none, rna="X", batch="obs:donor", category="cross") \
        == NO_CELLS
    assert _value_error(tmp_path, a, rna="X", atac=_peaks()[:0].copy(), atac_kind="peak",
                        category="diagonal") == NO_CELLS


def test_the_way_there_is_a_label_table_with_other_barcodes(tmp_path):
    """Labels joined by barcode from a table that spells them without -1:
    no cell gets one, and keeping the labelled cells keeps none. The text is
    the one of any data with 0 cells: it names the filter and no barcode."""
    a = _cite()
    table = pd.Series("T", index=[b[:-2] for b in a.obs_names])
    a.obs["celltype"] = table.reindex(a.obs_names)
    kept = a[a.obs["celltype"].notna()].copy()
    assert kept.n_obs == 0
    assert _value_error(tmp_path, kept, rna="X", labels="obs:celltype") == NO_CELLS
    assert "barcode" not in NO_CELLS and "obs_names" not in NO_CELLS


# ----------------------------------------------------------------- 0 features
NO_FEATURES = "The {what}matrix holds 0 features for 60 cells. "
WHY = "For a 10x file, read it with gex_only=False."
#: the object has var['feature_types']: its features were selected by type
TYPE = ("Check the feature type you selected: var['feature_types'] of the whole object "
        "lists the names.")


def test_the_advice_for_a_wrong_type_name_works(tmp_path):
    """A 10x object with genes and peaks, split with ``'Peak'`` for
    ``'Peaks'``: the reader of 0.3.5 was sent to ``gex_only=False``, which
    the file was read with. The names of the whole object show the type."""
    a = _cite()
    a.var["feature_types"] = ["Gene Expression"] * 12 + ["Peaks"] * 8
    a.var_names = [f"GENE{i}" for i in range(12)] + [f"chr1:{i}00-{i}50" for i in range(8)]
    kind = a.var["feature_types"]
    rna, none = a[:, kind == "Gene Expression"].copy(), a[:, kind == "Peak"].copy()
    kw = dict(rna="X", atac_kind="peak", labels="obs:ct")
    assert _value_error(tmp_path, rna, atac=none, **kw) \
        == NO_FEATURES.format(what="atac ") + TYPE
    names = sorted(set(a.var["feature_types"]))          # of the whole object
    assert names == ["Gene Expression", "Peaks"]
    out = ingest.export_dataset(rna, tmp_path / "OK", atac=a[:, kind == names[1]].copy(), **kw)
    assert ingest.read_canonical(out / "atac.h5").n_vars == 8


def test_an_object_without_feature_types_keeps_the_advice_for_a_10x_file(tmp_path):
    """Nothing says how the features were selected: the text of before."""
    rna = _cite()
    assert "feature_types" not in rna.var.columns
    none = rna[:, :0].copy()
    assert _value_error(tmp_path, rna, rna="X", atac=none, atac_kind="peak") \
        == NO_FEATURES.format(what="atac ") + WHY
    with pytest.raises(ValueError) as e:
        ingest._to_canonical(none, tmp_path / 'OUT' / 'atac_peak.h5')
    assert str(e.value) == NO_FEATURES.format(what="") + WHY
    # a matrix in obsm has no var: the column of the object says nothing about it
    rna.var["feature_types"] = "Gene Expression"
    rna.obsm["protein"] = np.zeros((60, 0))
    assert _value_error(tmp_path, rna, rna="X", adt="obsm:protein") \
        == NO_FEATURES.format(what="adt ") + WHY


# ------------------------------------------------------------- a missing batch
@pytest.mark.parametrize("gap,seen", [
    (np.nan, ""), (None, ""), ("", ""), ("  ", ""), ("nan", ", for example 'nan'"),
    ("NA", ", for example 'NA'"), ("None", ", for example 'None'")],
    ids=["NaN", "None", "empty", "blank", "text nan", "text NA", "text None"])
def test_a_batch_column_with_gaps_is_refused(tmp_path, gap, seen):
    """A text that a label file reads as missing is shown; NaN and blank
    text are not."""
    a = _cite()
    donor = a.obs["donor"].astype(object).to_numpy()
    donor[[3, 10, 41]] = gap
    a.obs["donor"] = donor
    told = f": 3 of the 60 cells have no batch{seen}. Give every cell a batch."
    kw = dict(rna="X", category="cross")
    assert _value_error(tmp_path, a, batch="obs:donor", **kw) == "batch='obs:donor'" + told
    assert _value_error(tmp_path, a, batch="donor", **kw) == "batch='donor'" + told
    assert _value_error(tmp_path, a, batch=a.obs["donor"], **kw) == "batch=Series" + told
    # a list of text and NaN becomes an array of text: its NaN is the text 'nan'
    as_list = told.replace("no batch.", "no batch, for example 'nan'.") \
        if isinstance(gap, float) else told
    assert _value_error(tmp_path, a, batch=list(donor), **kw) == "batch=list" + as_list
    # labels alone, split by the batch column
    assert _value_error(tmp_path, a, rna=None, labels="obs:ct", batch="obs:donor",
                        category="cross") == "batch='obs:donor'" + told


def test_several_spellings_of_a_missing_batch(tmp_path):
    a = _cite(n=2000)
    donor = a.obs["donor"].astype(object).to_numpy()
    donor[:1200:2] = np.nan
    donor[1:1201:2] = "NA"
    a.obs["donor"] = donor
    assert _value_error(tmp_path, a, rna="X", batch="obs:donor", category="cross") == (
        "batch='obs:donor': 1,200 of the 2,000 cells have no batch, for example 'NA'. Give "
        "every cell a batch.")


def test_a_complete_batch_column_writes_every_cell(tmp_path):
    a = _cite()
    out = ingest.export_dataset(a, tmp_path / "OK", rna="X", labels="obs:ct",
                                batch="obs:donor", category="cross")
    cells = [ingest.read_canonical(out / f"rna{b}.h5").n_obs for b in (1, 2)]
    assert cells == [30, 30] and sorted(p.name for p in out.iterdir()) == [
        "cty1.csv", "cty2.csv", "rna1.h5", "rna2.h5"]


# ------------------------------------------------------------- a missing label
WHERE = "labels for the cells (the AnnData's obs) are missing"
#: d1 (u1): the error gives the code that keeps the labelled cells
KEEP = 'keep the labelled cells first: adata = adata['
NOTNA = KEEP + 'adata.obs["ct"].notna()]'
SERIES_ADVICE = "give every cell a label, or keep the labelled cells first"


def _words(*words):
    return (KEEP + '~adata.obs["ct"].astype(str).str.strip().isin('
            + repr(sorted({"", "nan", "None", *words})) + ")]")


@pytest.mark.parametrize("word", ["NA", "None", "nan", "null", "N/A", "NA "])
def test_labels_that_read_back_as_missing_are_refused(tmp_path, word):
    """0.3.4 wrote these. The label file then held a missing value for the cell."""
    a = _cite()
    ct = a.obs["ct"].astype(object).to_numpy()
    ct[[0, 7]] = word
    a.obs["ct"] = ct
    shown = repr(word.strip())
    assert _value_error(tmp_path, a, rna="X", labels="obs:ct") == (
        f"labels='obs:ct': 2 of the 60 {WHERE} (NaN, None, '' or text such as {shown}); "
        f"{_words(word.strip())}")


def test_several_missing_words_show_the_first(tmp_path):
    a = _cite()
    ct = a.obs["ct"].astype(object).to_numpy()
    ct[0], ct[1], ct[2], ct[3], ct[4] = "NA", "None", np.nan, "null", "n/a"
    a.obs["ct"] = ct
    assert _value_error(tmp_path, a, rna="X", labels=a.obs["ct"]) == (
        f"labels=Series: 5 of the 60 labels for the cells (the labels you passed) are "
        f"missing (NaN, None, '' or text such as 'NA'); {SERIES_ADVICE}")


@pytest.mark.parametrize("gap", [np.nan, None, "", "   "], ids=["NaN", "None", "empty", "blank"])
def test_the_older_forms_keep_their_text(tmp_path, gap):
    a = _cite()
    ct = a.obs["ct"].astype(object).to_numpy()
    ct[5] = gap
    a.obs["ct"] = ct
    assert _value_error(tmp_path, a, rna="X", labels="obs:ct") == (
        f"labels='obs:ct': 1 of the 60 {WHERE} (NaN, None or ''); "
        f"{_words() if gap in ('', '   ') else NOTNA}")


ACCEPTED = ["na", "none", "Unknown", "-", "?", "NaT", "0", "T cell", "CD4+, naive",
            'the "other" cells', "two\nlines", "a\rb", "x\r\ny", " padded ", "NAIVE", "N.A.",
            "é漢", "#1"]


def test_accepted_labels_read_back_complete(tmp_path):
    """What the check lets through, the label reader gives back: one label
    per cell, none of them missing, the text unchanged."""
    a = _cite(n=len(ACCEPTED))
    a.obs["ct"] = ACCEPTED
    out = ingest.export_dataset(a, tmp_path / "OK", rna="X", labels="obs:ct")
    back = eio.read_labels(out / "cty.csv")
    assert len(back) == a.n_obs and not eio.missing_mask(back).any()
    assert [str(x) for x in back] == ACCEPTED
    for values in ([1, 2, 3], [True, False, True], [1.5, 2.5, 3.5]):
        out = ingest.export_dataset(None, tmp_path / "NUM", rna=None, labels=values,
                                    overwrite=True)
        back = eio.read_labels(out / "cty.csv")
        assert len(back) == 3 and not eio.missing_mask(back).any()


def test_the_check_and_the_reader_agree_on_random_labels(tmp_path):
    """The export refuses a label vector exactly when the definition calls a
    value of it missing, and a vector it writes is read back row for row
    with no missing value."""
    rng = random.Random(4)
    parts = ["a", "B", " ", ",", '"', "'", "\n", "\r", "\t", "NA", "N/A", "nan", "null",
             "None", "é", "1", ".", "-", "#", "<NA>", ""]
    refused = written = 0
    for _ in range(400):
        vals = ["".join(rng.choice(parts) for _ in range(rng.randint(0, 3)))
                for _ in range(rng.randint(1, 6))]
        missing = bool(eio.missing_mask(vals).any())
        try:
            out = ingest.export_dataset(None, tmp_path / "R", rna=None, labels=vals,
                                        overwrite=True)
        except ValueError as e:
            assert missing and "are missing (NaN, None" in str(e), vals
            refused += 1
            continue
        assert not missing, vals
        back = eio.read_labels(out / "cty.csv")
        assert len(back) == len(vals) and not pd.isna(back).any(), vals
        written += 1
    assert refused > 50 and written > 50


# ------------------------------------------------- a refusal writes nothing
def test_a_refused_call_leaves_a_folder_as_it_was(tmp_path):
    """Every check runs before the first write: a call that is refused for
    what it was given leaves each file of an existing folder byte for byte."""
    a = _cite()
    out = ingest.export_dataset(a, tmp_path / "HERE", rna="X", labels="obs:ct")
    (out / "notes.txt").write_text("mine\n")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    gaps, words, mixed, prot = _cite(), _cite(), _cite(), _cite()
    gaps.obs["donor"] = ["d1", None] * 30
    words.obs["ct"] = ["T", "NA"] * 30
    mixed.var["feature_types"] = ["Gene Expression"] * 10 + ["Peaks"] * 10
    prot.obsm["protein"] = np.ones((60, 3))
    refused = [
        (a[:0].copy(), dict(rna="X")),
        (gaps, dict(rna="X", batch="obs:donor", category="cross")),
        (words, dict(rna="X", labels="obs:ct")),
        (mixed, dict(rna="X")),
        (prot, dict(rna=None, adt="X")),
        (a, dict(rna="X", labels="nope")),
        (a, dict(rna="X", labels="annotation.csv")),
        (a, dict(rna="X", labels="obs:nope")),
        (a, dict(rna="layer:nope")),
    ]
    for data, kw in refused:
        with pytest.raises((ValueError, KeyError)):
            ingest.export_dataset(data, out, overwrite=True, **kw)
        assert {p.name: p.read_bytes() for p in out.iterdir()} == before, kw


def test_adt_in_x_next_to_a_protein_matrix(tmp_path):
    """Another obsm matrix may be the protein one: X is then usually the RNA."""
    a = _cite()
    a.obsm["protein"] = np.ones((a.n_obs, 4))
    a.obsm["X_umap"] = np.zeros((a.n_obs, 2))
    assert _value_error(tmp_path, a, rna=None, adt="X") == (
        "adt='X' takes X, and the object also holds obsm['protein']. If that is the protein "
        "matrix, pass adt='obsm:protein'.")
    a.obsm["totalvi"] = np.ones((a.n_obs, 3))
    assert _value_error(tmp_path, _cite(), rna="X", adt=a) == (
        "adt=AnnData takes X, and the object also holds 2 obsm entries, for example "
        "'protein'. If X is the protein matrix, delete them first, as del adt.obsm['protein'].")
    # both ways out work: the key that the first message names
    ingest.export_dataset(a, tmp_path / "KEY", rna=None, adt="obsm:protein",
                          adt_names=["CD3", "CD4", "CD8", "CD19"])
    # and the line of the second, once for each entry
    adt = a
    del adt.obsm['protein']
    assert _value_error(tmp_path, _cite(), rna="X", adt=adt) == (
        "adt=AnnData takes X, and the object also holds obsm['totalvi']. If X is the protein "
        "matrix, run del adt.obsm['totalvi'] first.")
    del adt.obsm['totalvi']
    own = ingest.export_dataset(_cite(), tmp_path / "OWN", rna="X", adt=adt)
    assert ingest.read_canonical(own / "adt.h5").n_vars == 20
    ingest.export_dataset(a, tmp_path / "X", rna=None, adt="X")
    assert ingest.read_canonical(tmp_path / "KEY" / "adt.h5").n_vars == 4
    assert ingest.read_canonical(tmp_path / "X" / "adt.h5").n_vars == 20


def test_to_canonical_keeps_its_text_and_ignores_embeddings(tmp_path):
    a = _cite()
    a.obsm["X_umap"] = np.zeros((a.n_obs, 2))
    a.obsm["protein"] = np.ones((a.n_obs, 4))
    with pytest.raises(ValueError) as e:
        ingest._to_canonical(a, tmp_path / 'adt.h5', modality='adt')
    assert str(e.value) == (
        "adt requested but obsm= not given; found obsm keys ['X_umap', 'protein'] - pass "
        "obsm='<key>' (or layer=) to select the protein matrix, otherwise adata.X (usually "
        "the RNA) would be written as adt.h5")
    assert not (tmp_path / "adt.h5").exists()


