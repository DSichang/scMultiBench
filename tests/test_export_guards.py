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

from multibench import cli
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


def _split_without_peaks():
    """``(rna, none)``: a 10x file read without ``gex_only=False`` holds the
    genes alone, so the split by feature type gives peaks of 0 columns."""
    a = _cite()
    a.var["feature_types"] = "Gene Expression"
    kind = a.var["feature_types"]
    rna, none = a[:, kind == "Gene Expression"].copy(), a[:, kind == "Peaks"].copy()
    assert rna.shape == (60, 20) and none.shape == (60, 0)
    return rna, none


def test_a_matrix_without_features_is_refused(tmp_path):
    """0.3.4 wrote such a matrix as a file of barcodes, and said only that 0%
    of the ATAC feature names look like peaks."""
    rna, none = _split_without_peaks()
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # nothing about peak names before the refusal
        assert _value_error(tmp_path, none, rna="X") == NO_FEATURES.format(what="rna ") + TYPE
        assert _value_error(tmp_path, rna, rna="X", adt=none) \
            == NO_FEATURES.format(what="adt ") + TYPE
        for kind in ("peak", "gene_activity"):
            assert _value_error(tmp_path, rna, rna="X", atac=none, atac_kind=kind,
                                labels="obs:ct") == NO_FEATURES.format(what="atac ") + TYPE
        assert _value_error(tmp_path, rna, rna="X", atac=none, atac_kind="peak",
                            labels="obs:ct", category="diagonal") \
            == NO_FEATURES.format(what="atac ") + TYPE
        for modality in ("atac_peak", "rna", None):
            with pytest.raises(ValueError) as e:
                ingest.to_canonical(none, tmp_path / "OUT" / "atac_peak.h5", modality=modality)
            assert str(e.value) == NO_FEATURES.format(what="") + TYPE
        assert not (tmp_path / "OUT").exists()
    # a folder that is there keeps every file as it was
    out = ingest.export_dataset(rna, tmp_path / "HERE", rna="X", labels="obs:ct")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(rna, out, rna="X", atac=none, atac_kind="peak", labels="obs:ct",
                              overwrite=True)
    assert str(e.value) == NO_FEATURES.format(what="atac ") + TYPE
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(none, out, modality="atac_peak")
    assert str(e.value) == NO_FEATURES.format(what="") + TYPE
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    # one feature is a matrix to write
    one = ingest.export_dataset(rna, tmp_path / "ONE", rna="X", adt=rna[:, :1].copy())
    assert ingest.read_canonical(one / "adt.h5").shape == (60, 1)


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
        ingest.to_canonical(none, tmp_path / "OUT" / "atac_peak.h5")
    assert str(e.value) == NO_FEATURES.format(what="") + WHY
    # a matrix in obsm has no var: the column of the object says nothing about it
    rna.var["feature_types"] = "Gene Expression"
    rna.obsm["protein"] = np.zeros((60, 0))
    assert _value_error(tmp_path, rna, rna="X", adt="obsm:protein") \
        == NO_FEATURES.format(what="adt ") + WHY


def test_a_nameless_matrix_without_features_gets_no_line_about_its_names(tmp_path):
    """A bare array and an array in obsm have no feature names, and with 0
    columns there is none to make up. The warning said that the file names
    them feature_0 to feature_-1, and then the call was refused."""
    rna = _cite()
    rna.obsm["protein"] = np.zeros((60, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _value_error(tmp_path, rna, rna="X", adt=np.zeros((60, 0))) \
            == NO_FEATURES.format(what="adt ") + WHY
        assert _value_error(tmp_path, rna, rna="X", adt="obsm:protein") \
            == NO_FEATURES.format(what="adt ") + WHY
        with pytest.raises(ValueError) as e:
            ingest.to_canonical(rna, tmp_path / "OUT" / "adt.h5", modality="adt", obsm="protein")
        assert str(e.value) == NO_FEATURES.format(what="") + WHY
    assert not (tmp_path / "OUT").exists()
    # with columns to name, the warning is the one of before
    rna.obsm["protein"] = np.ones((60, 2))
    for n, adt in enumerate(("obsm:protein", np.ones((60, 2)))):
        with pytest.warns(UserWarning, match=r"The file names them feature_0 to feature_1\. "):
            two = ingest.export_dataset(rna, tmp_path / f"TWO{n}", rna="X", adt=adt)
        assert list(ingest.read_canonical(two / "adt.h5").var_names) == ["feature_0", "feature_1"]


def test_a_matrix_without_features_on_the_command_line(tmp_path, capsys):
    """The command line reads files: it names no argument of export_dataset.
    The file it was given was saved after the split and holds no feature
    type to count, so the text sends the reader to the object before it."""
    rna, none = _split_without_peaks()
    rna.write_h5ad(tmp_path / "rna.h5ad")
    none.write_h5ad(tmp_path / "none.h5ad")
    assert ad.read_h5ad(tmp_path / "none.h5ad").var["feature_types"].value_counts().sum() == 0
    src, out = str(tmp_path / "rna.h5ad"), str(tmp_path / "OUT")
    for argv, what in (
            (["convert", str(tmp_path / "none.h5ad"), out, "--rna", "X"], "rna "),
            (["convert", src, out, "--rna", "X", "--atac-from", str(tmp_path / "none.h5ad"),
              "--atac-kind", "peak", "--category", "diagonal"], "atac "),
            (["convert", str(tmp_path / "none.h5ad"), out + "/", "--modality", "atac_peak"], "")):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rc = cli.main(argv)
        err = capsys.readouterr().err
        assert rc == 1 and not (tmp_path / "OUT").exists()
        assert next(ln for ln in err.splitlines() if ln.startswith("error: ")) \
            == "error: " + NO_FEATURES.format(what=what) + TYPE


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
        "error: --batch obs:donor: 30 of the 60 cells have no batch. Give every cell a "
        "batch.")


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
        f"labels='obs:ct': 2 of the 60 {WHERE} (NaN, None, '' or text such as {shown}); "
        f"{ADVICE}")


def test_several_missing_words_show_the_first(tmp_path):
    a = _cite()
    ct = a.obs["ct"].astype(object).to_numpy()
    ct[0], ct[1], ct[2], ct[3], ct[4] = "NA", "None", np.nan, "null", "n/a"
    a.obs["ct"] = ct
    assert _value_error(tmp_path, a, rna="X", labels=a.obs["ct"]) == (
        f"labels=Series: 5 of the 60 labels for the cells (the labels you passed) are "
        f"missing (NaN, None, '' or text such as 'NA'); {ADVICE}")


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
        (a, dict(rna="X", n_batches=2)),
        (a, dict(rna="X", batch_index=3, n_batches=2, category="cross")),
        (a, dict(rna="layer:nope")),
    ]
    for data, kw in refused:
        with pytest.raises((ValueError, KeyError)):
            ingest.export_dataset(data, out, overwrite=True, **kw)
        assert {p.name: p.read_bytes() for p in out.iterdir()} == before, kw


def test_a_sparse_matrix_left_on_disk_is_refused_before_the_folder_is_made(tmp_path):
    """An object read with ``backed="r"`` keeps a sparse ``X`` on disk, which
    is no matrix to write. 0.3.4 said so after it had made the folder, and
    left it empty."""
    import scipy.sparse as sp
    a = _cite()
    a.X = sp.csr_matrix(a.X)
    a.write_h5ad(tmp_path / "a.h5ad")
    a[::-1].copy().write_h5ad(tmp_path / "reversed.h5ad")
    text = "matrix must be 2-D (cells x features), got shape ()"
    assert _value_error(tmp_path, ad.read_h5ad(tmp_path / "a.h5ad", backed="r"), rna="X",
                        labels="obs:ct") == text
    # a folder that holds these cells in another order: the same refusal
    out = ingest.export_dataset(_adt(), tmp_path / "HERE", rna=None, adt="X")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(ad.read_h5ad(tmp_path / "reversed.h5ad", backed="r"), out,
                              rna="X", labels="obs:ct")
    assert str(e.value) == text
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    # read into memory, it is written, in the order of the folder
    ingest.export_dataset(ad.read_h5ad(tmp_path / "reversed.h5ad"), out, rna="X",
                          labels="obs:ct")
    assert ingest.read_canonical(out / "rna.h5", sparse=False).obs_names.tolist() \
        == list(a.obs_names)


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
        "error: --adt X takes X, and the object also holds obsm['protein']. If that is the "
        "protein matrix, pass --adt obsm:protein.")


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
