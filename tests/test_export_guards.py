"""Guards of export_dataset on what the cells carry.

- Data with 0 cells is refused. 0.3.4 wrote files with no cell in them.
- A batch column with a missing value is refused. 0.3.4 left those cells
  out without a word, or made the text ``'nan'`` batch 1.
- A label that a label file reads back as missing (``'NA'``, ``'None'``,
  ...) is refused, by the definition ``mtb.evaluate`` scores with
  (``eval.io.missing_mask``): what passes here is read back as a label.
- ADT taken from ``X`` is refused only when the object holds another matrix
  that could be the protein one. An embedding (``obsm["X_umap"]``) is none.
"""
import random
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench import cli
from multibench.engine import ingest
from multibench.eval import io as eio

ad = pytest.importorskip("anndata")

NO_CELLS = ("The data holds 0 cells, so there is nothing to write. If you kept the cells that "
            "have a label, no cell had one: check that the barcodes of the label table are "
            "spelled as the cell names of the data (obs_names) are, for example with the "
            "same -1 ending.")


def _cite(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n, 20)).astype(float))
    a.var_names = [f"GENE{i}" for i in range(20)]
    a.obs_names = [f"cell{i}-1" for i in range(n)]
    a.obs["ct"] = np.array(["T", "B", "NK"])[np.arange(n) % 3]
    a.obs["donor"] = np.array(["d1", "d2"])[np.arange(n) % 2]
    return a


def _adt(n=60, seed=1):
    """Protein counts in X, as the ADT file of a mosaic or cross tutorial."""
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(4.0, (n, 6)).astype(float))
    a.var_names = [f"CD{i}" for i in range(6)]
    a.obs_names = [f"cell{i}-1" for i in range(n)]
    a.obs["ct"] = np.array(["T", "B", "NK"])[np.arange(n) % 3]
    a.obs["batch"] = np.array(["1", "2"])[np.arange(n) % 2]
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
    no cell gets one, and keeping the labelled cells keeps none."""
    a = _cite()
    table = pd.Series("T", index=[b[:-2] for b in a.obs_names])
    a.obs["celltype"] = table.reindex(a.obs_names)
    kept = a[a.obs["celltype"].notna()].copy()
    assert kept.n_obs == 0
    assert _value_error(tmp_path, kept, rna="X", labels="obs:celltype") == NO_CELLS


# ------------------------------------------------------------- a missing batch
@pytest.mark.parametrize("gap,seen", [
    (np.nan, "NaN"), (None, "NaN"), ("", "''"), ("  ", "''"), ("nan", "'nan'"),
    ("NA", "'NA'"), ("None", "'None'")],
    ids=["NaN", "None", "empty", "blank", "text nan", "text NA", "text None"])
def test_a_batch_column_with_gaps_is_refused(tmp_path, gap, seen):
    a = _cite()
    donor = a.obs["donor"].astype(object).to_numpy()
    donor[[3, 10, 41]] = gap
    a.obs["donor"] = donor
    told = (f": 3 of the 60 cells have no batch. The missing value is written {seen}. Give "
            f"every cell a batch, or leave these cells out first.")
    kw = dict(rna="X", category="cross")
    assert _value_error(tmp_path, a, batch="obs:donor", **kw) == "batch='obs:donor'" + told
    assert _value_error(tmp_path, a, batch="donor", **kw) == "batch='donor'" + told
    assert _value_error(tmp_path, a, batch=a.obs["donor"], **kw) == "batch=Series" + told
    # a list of text and NaN becomes an array of text: its NaN is the text 'nan'
    as_list = told.replace("written NaN", "written 'nan'") if isinstance(gap, float) else told
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
        "batch='obs:donor': 1,200 of the 2,000 cells have no batch. The missing values are "
        "written NaN and 'NA'. Give every cell a batch, or leave these cells out first.")


def test_a_complete_batch_column_writes_every_cell(tmp_path):
    a = _cite()
    out = ingest.export_dataset(a, tmp_path / "OK", rna="X", labels="obs:ct",
                                batch="obs:donor", category="cross")
    cells = [ingest.read_canonical(out / f"rna{b}.h5").n_obs for b in (1, 2)]
    assert cells == [30, 30] and sorted(p.name for p in out.iterdir()) == [
        "cty1.csv", "cty2.csv", "rna1.h5", "rna2.h5"]


def test_a_missing_batch_on_the_command_line(tmp_path, capsys):
    a = _cite()
    a.obs["donor"] = ["d1", None] * 30
    a.write_h5ad(tmp_path / "a.h5ad")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(["convert", str(tmp_path / "a.h5ad"), str(tmp_path / "OUT"), "--rna", "X",
                       "--batch", "obs:donor", "--category", "cross"])
    err = capsys.readouterr().err
    assert rc == 1 and not (tmp_path / "OUT").exists()
    assert next(ln for ln in err.splitlines() if ln.startswith("error: ")) == (
        "error: --batch obs:donor: 30 of the 60 cells have no batch. The missing value is "
        "written NaN. Give every cell a batch, or leave these cells out first.")


# ------------------------------------------------------------- a missing label
WHERE = "labels for the cells (data.obs) are missing"
ADVICE = "give every cell a label, or subset the data to the labelled cells first"


@pytest.mark.parametrize("word", ["NA", "None", "nan", "null", "N/A", "NA "])
def test_labels_that_read_back_as_missing_are_refused(tmp_path, word):
    """0.3.4 wrote these. The label file then held a missing value for the cell."""
    a = _cite()
    ct = a.obs["ct"].astype(object).to_numpy()
    ct[[0, 7]] = word
    a.obs["ct"] = ct
    shown = repr(word.strip())
    assert _value_error(tmp_path, a, rna="X", labels="obs:ct") == (
        f"labels='obs:ct': 2 of the 60 {WHERE} (NaN, None, '' or text that a label file "
        f"reads as missing: {shown}); {ADVICE}. If {shown} is the name of a cell type, "
        f"rename it")


def test_several_missing_words_are_listed(tmp_path):
    a = _cite()
    ct = a.obs["ct"].astype(object).to_numpy()
    ct[0], ct[1], ct[2], ct[3], ct[4] = "NA", "None", np.nan, "null", "n/a"
    a.obs["ct"] = ct
    assert _value_error(tmp_path, a, rna="X", labels=a.obs["ct"]) == (
        f"labels=Series: 5 of the 60 labels for the cells (the labels you passed) are "
        f"missing (NaN, None, '' or text that a label file reads as missing: 'NA', 'None', "
        f"'n/a' and 1 more); {ADVICE}. If one of them is the name of a cell type, rename it")


@pytest.mark.parametrize("gap", [np.nan, None, "", "   "], ids=["NaN", "None", "empty", "blank"])
def test_the_older_forms_keep_their_text(tmp_path, gap):
    a = _cite()
    ct = a.obs["ct"].astype(object).to_numpy()
    ct[5] = gap
    a.obs["ct"] = ct
    assert _value_error(tmp_path, a, rna="X", labels="obs:ct") == (
        f"labels='obs:ct': 1 of the 60 {WHERE} (NaN, None or ''); {ADVICE}")


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


# ------------------------------------------------------------ ADT taken from X
def test_adt_in_x_with_an_embedding(tmp_path):
    """A processed ADT file carries its embeddings. The four tutorials that
    read protein counts from X export it as they export a bare one."""
    rna, plain, peaks = _cite(), _adt(), _peaks()
    rna.obs["batch"] = plain.obs["batch"].to_numpy()
    rich = plain.copy()
    rich.obsm["X_umap"] = np.zeros((rich.n_obs, 2))
    rich.obsm["X_pca"] = np.zeros((rich.n_obs, 5))
    calls = {
        # mosaic: an ADT-only batch, and ADT next to the RNA of the batch
        "adt_only": lambda adt, d: ingest.export_dataset(
            adt, d, rna=None, adt="X", labels="obs:ct", batch_index=3, category="mosaic"),
        "with_rna": lambda adt, d: ingest.export_dataset(
            rna, d, rna="X", adt=adt, labels="obs:ct", batch_index=1, category="mosaic"),
        # cross: ADT and peaks of the same cells, split by the batch column
        "cross": lambda adt, d: ingest.export_dataset(
            adt, d, rna=None, adt="X", atac=peaks, atac_kind="peak", labels="obs:ct",
            batch="obs:batch", category="cross"),
        # and the same without a category
        "plain": lambda adt, d: ingest.export_dataset(adt, d, rna=None, adt="X"),
    }
    for name, call in calls.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a, b = call(plain, tmp_path / f"{name}_plain"), call(rich, tmp_path / f"{name}_rich")
        files = sorted(p.name for p in a.iterdir())
        assert files == sorted(p.name for p in b.iterdir()) and any("adt" in f for f in files)
        for f in files:
            if f.endswith(".h5"):
                x = ingest.read_canonical(a / f, sparse=False)
                y = ingest.read_canonical(b / f, sparse=False)
                assert np.array_equal(x.X, y.X) and list(x.var_names) == list(y.var_names) \
                    and list(x.obs_names) == list(y.obs_names)
            else:
                assert (a / f).read_bytes() == (b / f).read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ingest.to_canonical(rich, tmp_path / "adt.h5", modality="adt").exists()


def test_adt_in_x_next_to_a_protein_matrix(tmp_path):
    """Another obsm matrix may be the protein one: X is then usually the RNA."""
    a = _cite()
    a.obsm["protein"] = np.ones((a.n_obs, 4))
    a.obsm["X_umap"] = np.zeros((a.n_obs, 2))
    assert _value_error(tmp_path, a, rna=None, adt="X") == (
        "adt='X' takes X of this object, and the object also holds the obsm key 'protein'. "
        "If one of them is the protein matrix, pass adt='obsm:<key>'. If X is the protein "
        "matrix, delete it first, for example del adata.obsm['protein'].")
    a.obsm["totalvi"] = np.ones((a.n_obs, 3))
    assert _value_error(tmp_path, _cite(), rna="X", adt=a) == (
        "adt=AnnData takes X of this object, and the object also holds the obsm keys "
        "'protein' and 'totalvi'. If one of them is the protein matrix, pass "
        "adt='obsm:<key>'. If X is the protein matrix, delete them first, for example del "
        "adata.obsm['protein'].")
    # both ways out work
    ingest.export_dataset(a, tmp_path / "KEY", rna=None, adt="obsm:protein",
                          adt_names=["CD3", "CD4", "CD8", "CD19"])
    del a.obsm["protein"], a.obsm["totalvi"]
    ingest.export_dataset(a, tmp_path / "X", rna=None, adt="X")
    assert ingest.read_canonical(tmp_path / "KEY" / "adt.h5").n_vars == 4
    assert ingest.read_canonical(tmp_path / "X" / "adt.h5").n_vars == 20


def test_adt_in_x_on_the_command_line(tmp_path, capsys):
    a = _cite()
    a.obsm["protein"] = np.ones((a.n_obs, 4))
    a.write_h5ad(tmp_path / "a.h5ad")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(["convert", str(tmp_path / "a.h5ad"), str(tmp_path / "OUT"), "--adt", "X"])
    err = capsys.readouterr().err
    assert rc == 1 and not (tmp_path / "OUT").exists()
    assert next(ln for ln in err.splitlines() if ln.startswith("error: ")) == (
        "error: --adt X takes X of this object, and the object also holds the obsm key "
        "'protein'. If one of them is the protein matrix, pass --adt obsm:<key>. If X is the "
        "protein matrix, delete the obsm entries in Python and save the file again.")


def test_to_canonical_keeps_its_text_and_ignores_embeddings(tmp_path):
    a = _cite()
    a.obsm["X_umap"] = np.zeros((a.n_obs, 2))
    a.obsm["protein"] = np.ones((a.n_obs, 4))
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(a, tmp_path / "adt.h5", modality="adt")
    assert str(e.value) == (
        "adt requested but obsm= not given; found obsm keys ['X_umap', 'protein'] - pass "
        "obsm='<key>' (or layer=) to select the protein matrix, otherwise adata.X (usually "
        "the RNA) would be written as adt.h5")
    assert not (tmp_path / "adt.h5").exists()
