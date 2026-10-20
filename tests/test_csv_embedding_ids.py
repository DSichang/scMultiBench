"""A CSV embedding whose first column holds the cell barcodes keeps them.

0.3.4 dropped that column, so labels and batches were matched by position
even when both files named their cells, and a label file in another row
order gave other scores with one warning. The barcodes are now the cell ids
of the embedding, as the ``obs_names`` of an ``.h5ad`` output are.

The rule (``multibench.eval.io._read_csv_embedding``): the first column is
left out of the matrix when the file has two or more columns and that column
has no header or holds text (the rule of 0.3.4), and its values are the cell
ids when it holds text, that is, when at least one value is not a number.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.eval import io, pipeline
from multibench.eval import scib as escib

from .test_workflow_batch import N, cite  # noqa: F401 - the fixture of the run_all tests

CELLS = 60
TYPES = np.array(["T", "B", "NK"] * (CELLS // 3))
BARS = [f"AAAC{i:04d}-1" for i in range(CELLS)]
EMB = np.random.default_rng(0).normal(size=(CELLS, 5)) \
    + pd.Categorical(TYPES).codes[:, None] * 4.0
PERM = np.random.default_rng(1).permutation(CELLS)
MOVED = "evaluate: matched {0} to the rows of emb.csv by barcode.\n"
NO_IDS = ("The first column of {0} looks like cell ids, but the embedding has no cell ids. "
          "The file is matched by position. Check that its rows follow the rows of the "
          "embedding.")
NO_SUFFIX = [b.replace("-1", "") for b in BARS]          # the same cells, written differently


def _emb(tmp_path, name="emb.csv", **kw):
    """The embedding with its barcodes as the first column, as ``to_csv`` writes it."""
    path = tmp_path / name
    pd.DataFrame(EMB, index=BARS, columns=[f"d{i}" for i in range(5)]).to_csv(
        path, sep="\t" if name.endswith(".tsv") else ",", **kw)
    return str(path)


def _labels(tmp_path, name="labels.csv", order=None, values=TYPES, bars=BARS,
            column="celltype"):
    order = np.arange(len(values)) if order is None else order
    path = tmp_path / name
    pd.DataFrame({column: values}, index=bars).iloc[order].to_csv(path)
    return str(path)


@pytest.fixture
def seen(monkeypatch):
    """What reaches the scorer: the embedding, the labels and the batches,
    in the order they are scored. No metric is computed."""
    got = {}

    def compute(emb, ct, cl, ba, **kw):
        got.update(emb=np.asarray(emb), labels=list(ct), batch=list(ba),
                   clustering=None if cl is None else list(cl))
        return pd.DataFrame({"Value": [1.0]}, index=pd.Index(["ARI"], name="metric"))
    monkeypatch.setattr(escib, "compute", compute)
    return got


def _quiet(capsys, *args, **kw):
    """evaluate with no warning and nothing on stderr."""
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mtb.evaluate(*args, **kw)
    assert capsys.readouterr().err == ""
    return out


# ------------------------------------------------------------------ the rule
def test_a_first_column_of_text_gives_the_cell_ids(tmp_path):
    for kw in ({}, {"index_label": "barcode"}):          # no header, and a header
        m, ids = io._read_csv_embedding(_emb(tmp_path, **kw))
        np.testing.assert_allclose(m, EMB)
        assert list(ids) == BARS
    m, ids = io._read_csv_embedding(_emb(tmp_path, "emb.tsv"))
    np.testing.assert_allclose(m, EMB)
    assert list(ids) == BARS


def test_a_first_column_of_numbers_is_read_as_in_0_3_4(tmp_path):
    cols = [f"d{i}" for i in range(5)]
    # numbers under a header are data: all five columns stay
    path = tmp_path / "plain.csv"
    pd.DataFrame(EMB, columns=cols).to_csv(path, index=False)
    m, ids = io._read_csv_embedding(path)
    assert ids is None and m.shape == (CELLS, 5)
    np.testing.assert_allclose(m, EMB)
    # the row numbers to_csv writes have no header: left out, and no ids
    path = tmp_path / "numbered.csv"
    pd.DataFrame(EMB, columns=cols).to_csv(path)
    m, ids = io._read_csv_embedding(path)
    assert ids is None and m.shape == (CELLS, 5)
    np.testing.assert_allclose(m, EMB)
    # row numbers under a header are a column of numbers: data, as before
    path = tmp_path / "named.csv"
    pd.DataFrame(EMB, columns=cols).to_csv(path, index_label="cell")
    m, ids = io._read_csv_embedding(path)
    assert ids is None and m.shape == (CELLS, 6)
    np.testing.assert_allclose(m[:, 0], np.arange(CELLS))
    # one column: nothing is left out
    path = tmp_path / "one.csv"
    pd.DataFrame({"d0": EMB[:, 0]}).to_csv(path, index=False)
    m, ids = io._read_csv_embedding(path)
    assert ids is None and m.shape == (CELLS, 1)


def test_as_matrix_reads_the_same_matrix_as_before(tmp_path):
    """The public reader drops the same column as 0.3.4 and returns the matrix."""
    def before(path):                                    # the reader of 0.3.4
        d = pd.read_csv(path)
        if d.shape[1] > 1 and (str(d.columns[0]).startswith("Unnamed")
                               or d.iloc[:, 0].dtype == object):
            d = d.iloc[:, 1:]
        return d.to_numpy(dtype=float)
    files = [_emb(tmp_path), _emb(tmp_path, "named.csv", index_label="barcode")]
    for name, kw in (("plain.csv", {"index": False}), ("numbered.csv", {}),
                     ("cell.csv", {"index_label": "cell"})):
        pd.DataFrame(EMB, columns=list("abcde")).to_csv(tmp_path / name, **kw)
        files.append(str(tmp_path / name))
    for f in files:
        np.testing.assert_array_equal(io.as_matrix(f), before(f))


# ----------------------------------------------------------- matched by barcode
def test_labels_in_another_order_are_matched_by_barcode(tmp_path, capsys, seen):
    emb = _emb(tmp_path)
    _quiet(capsys, emb, labels=_labels(tmp_path), metrics=["ARI"])     # same order: no line
    assert seen["labels"] == list(TYPES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mtb.evaluate(emb, labels=_labels(tmp_path, "shuffled.csv", PERM), metrics=["ARI"])
    assert capsys.readouterr().err == MOVED.format("shuffled.csv")
    assert seen["labels"] == list(TYPES)
    np.testing.assert_allclose(seen["emb"], EMB)


def test_the_scores_no_longer_follow_the_row_order_of_the_label_file(tmp_path, capsys):
    emb = _emb(tmp_path)
    same = mtb.evaluate(emb, labels=_labels(tmp_path), metrics=["ARI", "NMI"],
                        verbose=False)
    moved = mtb.evaluate(emb, labels=_labels(tmp_path, "shuffled.csv", PERM),
                         metrics=["ARI", "NMI"], verbose=False)
    assert same["value"].tolist() == moved["value"].tolist() == [1.0, 1.0]


def test_batch_and_clustering_files_and_a_series_are_matched_too(tmp_path, capsys, seen):
    emb = _emb(tmp_path)
    donors = np.array(["d1", "d2"] * (CELLS // 2))
    clusters = pd.Categorical(TYPES).codes
    other = PERM[::-1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # clustering= is deprecated since 0.4: its implementation, without the warning
        pipeline._evaluate(emb, labels=pd.Series(TYPES, index=BARS).iloc[PERM],
                           batch=_labels(tmp_path, "batch.csv", other, donors,
                                         column="donor"),
                           clustering=_labels(tmp_path, "clusters.csv", PERM, clusters,
                                              column="leiden"), metrics="all")
    assert capsys.readouterr().err == (
        "evaluate: matched the labels Series and 2 more to the rows of emb.csv by "
        "barcode.\n")
    assert seen["labels"] == list(TYPES) and seen["batch"] == list(donors)
    assert seen["clustering"] == list(clusters)


def test_several_label_files_are_matched_by_their_barcodes(tmp_path, capsys, seen):
    emb = _emb(tmp_path)
    half = CELLS // 2
    first = _labels(tmp_path, "cty1.csv", np.arange(half, CELLS))      # the second half
    second = _labels(tmp_path, "cty2.csv", np.arange(half))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mtb.evaluate(emb, labels=[first, second], metrics=["ARI"])
    assert capsys.readouterr().err == (
        "evaluate: matched cty1.csv and 1 more to the rows of emb.csv by barcode.\n")
    assert seen["labels"] == list(TYPES)


# ------------------------------------------------------------- other cells
def test_other_cells_are_an_error_with_counts_and_examples(tmp_path, seen):
    emb = _emb(tmp_path)
    bars = BARS[:55] + [f"TTTG{i:04d}-1" for i in range(4)]
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=_labels(tmp_path, values=TYPES[:59], bars=bars),
                     metrics=["ARI"])
    assert str(e.value) == (
        "labels: 5 cells of emb.csv are not in labels.csv, for example AAAC0055-1. Write "
        "both for the same cells.")
    # the 4 barcodes that only labels.csv holds are not counted, and the
    # message no longer says to delete the barcode column
    assert "TTTG" not in str(e.value) and "by position" not in str(e.value)
    assert not seen


def test_a_file_with_fewer_cells(tmp_path, seen):
    emb = _emb(tmp_path)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=_labels(tmp_path, values=TYPES[:59], bars=BARS[:59]),
                     metrics=["ARI"])
    assert str(e.value) == (
        "labels: 1 cell of emb.csv is not in labels.csv, for example AAAC0059-1. Write "
        "both for the same cells.")
    assert not seen


# ------------------------------------------------------- no cell id in common
def test_a_file_with_no_id_in_common_is_refused(tmp_path, seen):
    """c3 (E05): the same cells under other names (barcodes without ``-1``):
    the ids decide, so the file is refused, for a CSV and an AnnData output."""
    ad = pytest.importorskip("anndata")
    donors = np.array(["d1", "d2", "d3"] * (CELLS // 3))[PERM]
    for output in (_emb(tmp_path), ad.AnnData(EMB, obs=pd.DataFrame(index=BARS))):
        for name, kw in (
                ("labels.csv", dict(labels=_labels(tmp_path, bars=NO_SUFFIX,
                                                   values=TYPES[PERM]))),
                ("batch.csv", dict(labels=TYPES, batch=_labels(
                    tmp_path, "batch.csv", bars=NO_SUFFIX, values=donors, column="donor")))):
            with pytest.raises(ValueError, match=rf"60 of the 60 ids in the first column of "
                                                 rf"{name} are not cells of the embedding"):
                mtb.evaluate(output, obsm="X", metrics=["ARI", "iLISI"] if "batch" in kw
                             else ["ARI"], **kw)
    with pytest.raises(ValueError, match="60 of the 60 ids in the first column of cty1.csv "
                                         "and cty2.csv are not cells of the embedding"):
        mtb.evaluate(_emb(tmp_path), labels=[
            _labels(tmp_path, "cty1.csv", values=TYPES[:30], bars=NO_SUFFIX[:30]),
            _labels(tmp_path, "cty2.csv", values=TYPES[30:], bars=NO_SUFFIX[30:])],
            metrics=["ARI"])
    # a Series: the ids decide too, and the fix for the suffix is named
    with pytest.raises(ValueError, match=r"They match after adding '-1' to the labels ids"):
        mtb.evaluate(_emb(tmp_path), labels=pd.Series(TYPES[PERM], index=NO_SUFFIX),
                     metrics=["ARI"])
    assert not seen


def test_ids_of_which_only_some_are_cells_stay_an_error(tmp_path, seen):
    """Half of the barcodes written differently: no order to fall back on."""
    emb = _emb(tmp_path)
    half = CELLS // 2
    bars = BARS[:half] + NO_SUFFIX[half:]
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=_labels(tmp_path, bars=bars), metrics=["ARI"])
    assert str(e.value) == (
        "labels: 30 cells of emb.csv are not in labels.csv, for example AAAC0030-1. Write "
        "both for the same cells.")
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=pd.Series(TYPES, index=bars), metrics=["ARI"])
    assert str(e.value) == (
        "labels: 30 cells of emb.csv are not in the labels Series, for example "
        "AAAC0030-1. Pass labels for the cells of emb.csv.")
    assert not seen


def test_following_the_error_for_other_cells(tmp_path, seen):
    """What each error says to do, done: the same cells in the file, and
    labels for the cells of the embedding."""
    emb = _emb(tmp_path)
    full = pd.Series(TYPES, index=BARS)
    with pytest.raises(ValueError, match="Write both for the same cells"):
        mtb.evaluate(emb, labels=_labels(tmp_path, values=TYPES[:59], bars=BARS[:59]),
                     metrics=["ARI"])
    mtb.evaluate(emb, labels=_labels(tmp_path, order=PERM), metrics=["ARI"])
    assert seen["labels"] == list(TYPES)
    extra = pd.concat([full, pd.Series(["T"] * 3, index=["ZZZ0", "ZZZ1", "ZZZ2"])])
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=extra, metrics=["ARI"])
    assert str(e.value) == (
        "labels: 3 cells of the labels Series are not in emb.csv, for example ZZZ0. Pass "
        "labels for the cells of emb.csv.")
    seen.clear()
    mtb.evaluate(emb, labels=extra.loc[BARS], metrics=["ARI"])
    assert seen["labels"] == list(TYPES)


def test_verbose_false_prints_no_line(tmp_path, capsys, seen):
    """The line for labels put in the order of the embedding is output of
    ``verbose``: ``verbose=False`` prints nothing, and matches the same way."""
    emb = _emb(tmp_path)
    kw = dict(labels=_labels(tmp_path, "shuffled.csv", PERM), metrics=["ARI"])
    mtb.evaluate(emb, **kw)
    assert capsys.readouterr().err == MOVED.format("shuffled.csv")
    assert seen["labels"] == list(TYPES)
    seen.clear()
    _quiet(capsys, emb, verbose=False, **kw)
    assert seen["labels"] == list(TYPES)


def test_a_series_with_other_cells(tmp_path, seen):
    emb = _emb(tmp_path)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=pd.Series(TYPES[:58], index=BARS[2:]), metrics=["ARI"])
    assert str(e.value) == (
        "labels: 2 cells of emb.csv are not in the labels Series, for example "
        "AAAC0000-1. Pass labels for the cells of emb.csv.")
    assert "to_numpy" not in str(e.value)
    assert not seen


def test_several_label_files_with_other_cells(tmp_path, seen):
    emb = _emb(tmp_path)
    half = CELLS // 2
    first = _labels(tmp_path, "cty1.csv", np.arange(half))
    second = _labels(tmp_path, "cty2.csv", np.arange(half, CELLS - 1))
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=[first, second], metrics=["ARI"])
    assert str(e.value) == (
        "labels: 1 cell of emb.csv is not in the label files, for example AAAC0059-1. "
        "Write them for the same cells.")
    assert not seen


# --------------------------------------------------- one side without cell ids
def test_labels_without_ids_next_to_cell_ids(tmp_path, capsys, seen):
    """c3 (E05): a file or a 0..n-1 Series without cell ids is refused next to an
    embedding with ids; an array or a list is matched by position."""
    emb = _emb(tmp_path)
    shuffled = TYPES[PERM]
    one = tmp_path / "cty.csv"                           # the benchmark's label file
    pd.DataFrame({"x": shuffled}).to_csv(one, index=False)
    numbered = tmp_path / "numbered.csv"                 # R's row numbers, then the labels
    pd.DataFrame({"x": shuffled}, index=np.arange(1, CELLS + 1)).to_csv(numbered)
    sample = tmp_path / "obs.csv"                        # a sample column: no barcodes
    pd.DataFrame({"sample": ["s1", "s2"] * (CELLS // 2), "x": shuffled}).to_csv(
        sample, index=False)
    for path in (one, numbered, sample):
        with pytest.raises(ValueError, match=rf"^{path.name} has no cell ids, and the "
                                             r"embedding has: put the barcodes in its first "
                                             r"column, or pass labels=pd\.read_csv"):
            mtb.evaluate(emb, labels=str(path), metrics=["ARI"])
    with pytest.raises(ValueError, match=r"^labels is a Series without cell ids, and the "
                                         r"embedding has: index it by the barcodes, or pass "
                                         r"labels\.to_numpy\(\) to match by position\.$"):
        mtb.evaluate(emb, labels=pd.Series(shuffled), metrics=["ARI"])
    for labels in (shuffled, list(shuffled), pd.read_csv(one).iloc[:, 0].to_numpy()):
        seen.clear()
        _quiet(capsys, emb, labels=labels, metrics=["ARI"])
        assert seen["labels"] == list(shuffled)


def test_an_embedding_without_ids_keeps_the_warning(tmp_path, capsys, seen):
    """Numbers in the first column, or no first column: the label file is
    matched by position, with the warning of 0.3.4."""
    labels = _labels(tmp_path, "shuffled.csv", PERM)
    cols = [f"d{i}" for i in range(5)]
    for name, kw in (("plain.csv", {"index": False}), ("numbered.csv", {})):
        pd.DataFrame(EMB, columns=cols).to_csv(tmp_path / name, **kw)
        seen.clear()
        with pytest.warns(UserWarning) as rec:
            mtb.evaluate(str(tmp_path / name), labels=labels, metrics=["ARI"])
        assert [str(w.message) for w in rec] == [NO_IDS.format("shuffled.csv")]
        assert seen["labels"] == list(TYPES[PERM]) and seen["emb"].shape == (CELLS, 5)
    assert capsys.readouterr().err == ""


def test_repeated_barcodes_in_the_embedding_are_not_ids(tmp_path, seen):
    path = tmp_path / "emb.csv"
    bars = [BARS[0]] + BARS[:-1]
    pd.DataFrame(EMB, index=bars).to_csv(path)
    with pytest.warns(UserWarning) as rec:
        mtb.evaluate(str(path), labels=_labels(tmp_path), metrics=["ARI"])
    assert [str(w.message) for w in rec] == [
        "The first column of labels.csv looks like cell ids, but the embedding repeats cell "
        "ids. The file is matched by position. Check that its rows follow the rows of the "
        "embedding."]
    assert seen["labels"] == list(TYPES)


def test_repeated_barcodes_and_a_series_raise_as_for_an_anndata(tmp_path, seen):
    """0.3.4 matched this by position. An AnnData with repeated obs_names
    raises for a Series with barcodes, and the CSV output does the same."""
    ad = pytest.importorskip("anndata")
    path = tmp_path / "emb.csv"
    bars = [BARS[0]] + BARS[:-1]
    pd.DataFrame(EMB, index=bars).to_csv(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # anndata: names are not unique
        adata = ad.AnnData(EMB, obs=pd.DataFrame(index=bars))
    said = []
    for output in (str(path), adata):
        with pytest.raises(ValueError) as e:
            mtb.evaluate(output, labels=pd.Series(TYPES, index=BARS), obsm="X",
                         metrics=["ARI"])
        said.append(str(e.value))
    assert said[0] == said[1] == (
        "output carries 1 duplicated cell id(s) (first: ['AAAC0000-1']); cannot align "
        "labels by cell id - pass labels.to_numpy() to match positionally")
    # what it says to do
    mtb.evaluate(str(path), labels=pd.Series(TYPES, index=BARS).to_numpy(), metrics=["ARI"])
    assert seen["labels"] == list(TYPES)


def test_a_dims_by_cells_table_is_still_transposed(tmp_path, capsys, seen):
    """The first column names the dimensions, not the cells: no ids."""
    path = tmp_path / "wide.csv"
    pd.DataFrame(EMB.T, index=[f"dim{i}" for i in range(5)], columns=BARS).to_csv(path)
    with pytest.warns(UserWarning) as rec:
        mtb.evaluate(str(path), labels=_labels(tmp_path, order=PERM), metrics=["ARI"])
    assert [str(w.message) for w in rec] == [
        NO_IDS.format("labels.csv"),
        "embedding was (dims x cells) (5, 60); transposed to (cells x dims)"]
    assert seen["emb"].shape == (CELLS, 5) and seen["labels"] == list(TYPES[PERM])
    assert capsys.readouterr().err == ""


def test_a_dims_by_cells_table_with_another_number_of_labels(tmp_path, seen):
    """The first column names the dimensions. With a label count that fits
    neither side, the barcodes of the labels are not rows of the table: refused
    without calling a dimension a cell."""
    path = tmp_path / "wide.csv"
    pd.DataFrame(EMB.T, index=[f"dim{i}" for i in range(5)], columns=BARS).to_csv(path)
    with pytest.raises(ValueError, match=r"^labels: 50 of the 50 ids in the first column of "
                                         r"labels\.csv are not cells of the embedding"):
        mtb.evaluate(str(path), labels=_labels(tmp_path, values=TYPES[:50], bars=BARS[:50]),
                     metrics=["ARI"])
    with pytest.raises(ValueError, match=r"^The labels' cell ids do not match the embedding's"):
        mtb.evaluate(str(path), labels=pd.Series(TYPES[:50], index=BARS[:50]),
                     metrics=["ARI"])
    assert not seen


# ------------------------------------------------------------- method outputs
def test_method_runs_are_scored_in_file_order_as_before(cite, tmp_path, monkeypatch,  # noqa: F811
                                                        capsys):
    """run_all scores embedding.h5, which has no cell ids: the label file is
    read in file order, whatever its first column holds, and no CSV
    embedding is read."""
    data, labels = cite
    monkeypatch.setattr(io, "_read_csv_embedding",
                        lambda *a, **k: pytest.fail("a CSV embedding was read"))
    monkeypatch.setattr(pipeline, "_Table",
                        lambda *a, **k: pytest.fail("cell ids from a CSV embedding"))
    kw = dict(methods=["Matilda"], modalities=["rna", "adt"], data_path=data, verbose=False)
    plain = mtb.run_all("MYCITE", "vertical", out_dir=tmp_path / "plain", **kw)
    scores = plain.summary.loc[0, ["ARI", "NMI"]].tolist()
    assert plain.summary.loc[0, "status"] == "scored" and scores == [1.0, 1.0]
    # the scores of the same embedding and labels, passed in memory by position
    (h5,) = (tmp_path / "plain").rglob("embedding.h5")
    with_arrays = mtb.evaluate(io.read_embedding(h5), labels=labels,
                               metrics=["ARI", "NMI"], verbose=False)
    assert with_arrays["value"].round(4).tolist() == scores
    # a barcode column in front of the labels, rows as before: the same scores
    cells = [f"c{i}" for i in range(N)]
    pd.DataFrame({"x": labels}, index=cells).to_csv(data / "MYCITE" / "cty.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        named = mtb.run_all("MYCITE", "vertical", out_dir=tmp_path / "named", **kw)
    assert named.summary.loc[0, ["ARI", "NMI"]].tolist() == scores
    # the rows of that file reversed: scored by position, so the scores change
    pd.DataFrame({"x": labels}, index=cells).iloc[::-1].to_csv(data / "MYCITE" / "cty.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        turned = mtb.run_all("MYCITE", "vertical", out_dir=tmp_path / "turned", **kw)
    assert turned.summary.loc[0, ["ARI", "NMI"]].tolist() == scores    # A|B halves swap
    assert "evaluate:" not in capsys.readouterr().err
