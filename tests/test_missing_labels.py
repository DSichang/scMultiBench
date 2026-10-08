"""evaluate() does not score a missing label as a cell type (0.3.5).

0.3.4 cast the labels to text, so NaN became one more cell type ``"nan"``
and every score moved without a word (ASW 0.7160 where the labelled cells
score 0.7814 on the cells below). Now:

- one definition of "missing" (``eval.io.missing_mask``): NaN, ``None``,
  ``pd.NA``, blank text, and any text that the label reader itself reads back
  as missing (``NA``, ``None``, ``nan``, ...), with or without blanks around it;
- ``missing_labels="error"`` (the default) raises and says how many cells and
  how the missing values are written; ``"drop"`` scores the labelled cells
  only, warns with the two counts, and records them;
- complete labels score exactly as before.
"""
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench.engine import ingest
from multibench.eval import io as eio
from multibench.eval import pipeline

pytest.importorskip("scib")

N = 300
#: the 30 cells without a label: 10 of each cell type, and of each donor
GAPS = np.arange(5, N, 10)
KEEP = ~np.isin(np.arange(N), GAPS)
#: ASW of the cells below as the released multibench-sc 0.3.4 scored it:
#: complete labels, and the 270 labelled cells passed by hand
ASW_034 = 0.7814517904520452
ASW_034_LABELLED = 0.7814125597558947

NAN = "The missing value is written NaN."
RENAME_NA = (" A label file reads 'NA' as a missing value. If it is the name of a cell "
             "type, rename it.")


def _cells():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(N, 8))
    emb[:100] += 3
    emb[100:200] -= 3
    lab = np.repeat(["A", "B", "C"], 100).astype(object)
    return emb, lab


def _donors():
    return np.tile(["d1", "d2", "d3"], 100).astype(object)


def _gaps(values, fill=np.nan, at=GAPS):
    out = np.array(values, dtype=object)
    out[at] = fill
    return out


def _adata(emb, **obs):
    import anndata as ad
    a = ad.AnnData(np.zeros((len(emb), 2)), obs=pd.DataFrame(
        obs, index=[f"cell{i}" for i in range(len(emb))]))
    a.obsm["X_emb"] = emb
    return a


def _e1(src="", seen=NAN, rename="", n=30, total=N):
    return (f"labels{src}: {n} of the {total} cells have no label. {seen} A missing "
            f"label is not a cell type, so evaluate does not score it. Give every cell "
            f'a label, or pass missing_labels="drop" to score the {total - n} labelled '
            f"cells only.{rename}")


def _scored(rec):
    """The messages of the warnings about cells left out."""
    return [str(w.message) for w in rec if str(w.message).startswith("Scored the")]


def _same(got, want):
    """The same metrics with the same values, NaN included."""
    assert got.index.tolist() == want.index.tolist()
    np.testing.assert_array_equal(got["Value"].to_numpy(), want["Value"].to_numpy())


def _drop(*args, **kw):
    """evaluate(missing_labels="drop"), with its warnings about the cells left out."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = mtb.evaluate(*args, missing_labels="drop", verbose=False, **kw)
    return out, _scored(rec)


def _silent(*args, **kw):
    """evaluate(), which must not warn (evaluate warns with UserWarning)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = mtb.evaluate(*args, **kw)
    assert [str(w.message) for w in rec if issubclass(w.category, UserWarning)] == []
    return out


# ============================================================ the definition
MISSING_NOT_A_WORD = [np.nan, None, pd.NA, "", " ", "  \t"]
MISSING_WORDS = ["NA", "NA ", " NA", "N/A", "n/a", "NULL", "null", "NaN", "nan", "-nan",
                 "None", "<NA>", "#N/A", "#NA"]
NOT_MISSING = ["na", "none", "Unknown", "-", "?", "NaT", "0", 0, 1.5, True, False,
               "B, C", 'say "x"', "two\nlines"]


@pytest.mark.parametrize("value", MISSING_NOT_A_WORD, ids=repr)
def test_missing_mask_forms_null_and_blank(value):
    for box in (list, lambda v: np.array(v, dtype=object), pd.Series, pd.Index):
        values = box(["A", value, "B"])
        assert eio.missing_mask(values).tolist() == [False, True, False], type(values)
        assert eio.missing_words(values) == []


@pytest.mark.parametrize("value", MISSING_WORDS, ids=repr)
def test_missing_mask_forms_words(value):
    for box in (list, lambda v: np.array(v, dtype=object), pd.Series, pd.Index,
                pd.Categorical):
        values = box(["A", value, "B"])
        assert eio.missing_mask(values).tolist() == [False, True, False], type(values)
        # the word is listed without the blanks around it
        assert eio.missing_words(values) == [value.strip()]


@pytest.mark.parametrize("value", NOT_MISSING, ids=repr)
def test_missing_mask_forms_values_that_are_labels(value):
    for box in (list, lambda v: np.array(v, dtype=object), pd.Series):
        values = box(["A", value, "B"])
        assert eio.missing_mask(values).tolist() == [False, False, False], type(values)
        assert eio.missing_words(values) == []


def test_missing_mask_forms():
    """The containers of the table that the three lists above do not reach."""
    def mask(v):
        return eio.missing_mask(v).tolist()
    assert mask(pd.Categorical(["A", None, "B"])) == [False, True, False]
    assert mask(pd.Series(["A", np.nan, "B"], dtype="category")) == [False, True, False]
    assert mask(pd.array(["A", pd.NA, "B"], dtype="string")) == [False, True, False]
    assert mask(np.array([1.0, np.nan, 2.0])) == [False, True, False]
    assert mask(np.array([1, 2, 3])) == [False, False, False]
    assert mask(np.array([True, False])) == [False, False]
    assert mask([]) == [] and eio.missing_words([]) == []
    # numpy made the NaN a text before anyone could see it
    text = np.array(["a", np.nan])
    assert text.dtype.kind == "U"
    assert mask(text) == [False, True] and eio.missing_words(text) == ["nan"]
    # several spellings: every one is found, the words come back sorted
    mixed = ["NA", "A", None, "null", "", " NA", "N/A", "none"]
    assert mask(mixed) == [True, False, True, True, True, True, True, False]
    assert eio.missing_words(mixed) == ["N/A", "NA", "null"]
    assert eio.missing_mask(mixed).dtype == bool


def test_missing_words_agree_with_a_real_label_file(tmp_path):
    """The words are what the label reader reads as missing from a file that
    export wrote, not a list kept by hand."""
    texts = [t.strip() for t in MISSING_WORDS] + [str(v) for v in NOT_MISSING] + [
        "A", "B cell", "1.#QNAN", "inf", "TRUE", "x", "N.A.", "nil", "NAN", "Nan"]
    texts = list(dict.fromkeys(texts))
    back = eio.read_labels(ingest._write_labels(texts, tmp_path / "cty.csv"))
    assert len(back) == len(texts)
    read_as_missing = {t for t, v in zip(texts, back) if pd.isna(v)}
    assert read_as_missing == set(eio.missing_words(texts))
    assert {"NA", "None", "nan", "null", "N/A", "<NA>"} <= read_as_missing
    assert not read_as_missing & {"na", "none", "Unknown", "NaT", "A", "0"}
    # and the mask is exactly those rows
    assert eio.missing_mask(texts).tolist() == [t in read_as_missing for t in texts]


def test_the_label_reader_and_the_definition_share_one_pandas_call(tmp_path, monkeypatch):
    calls = []
    real = eio._read_table

    def spy(src, sep):
        calls.append(sep)
        return real(src, sep)
    monkeypatch.setattr(eio, "_read_table", spy)
    path = ingest._write_labels(["A", "B"], tmp_path / "cty.tsv")
    assert eio.read_labels(path).tolist() == ["A", "B"]
    assert calls == ["\t"]
    assert eio.missing_words(["A", "NA"]) == ["NA"]
    assert calls == ["\t", ","]


def test_a_round_trip_that_loses_a_row_fails_loudly(monkeypatch):
    real = eio._read_table
    monkeypatch.setattr(eio, "_read_table", lambda src, sep: real(src, sep).iloc[:-1])
    with pytest.raises(RuntimeError, match="read back, they are 2 rows"):
        eio.missing_mask(["A", "B", "NA"])


def test_seen_sentence_and_rename_sentence():
    assert eio._missing_seen(["A", np.nan]) == NAN
    assert eio._missing_seen(["A", " "]) == "The missing value is written ''."
    assert eio._missing_seen(["A", "NA "]) == "The missing value is written 'NA'."
    assert eio._missing_seen(["A", None, "NA"]) == \
        "The missing values are written NaN and 'NA'."
    assert eio._missing_seen(["A", None, "", "NA"]) == \
        "The missing values are written NaN, '' and 'NA'."
    many = ["NA", "None", "nan", "null", "N/A", "A"]
    assert eio._missing_seen(many) == \
        "The missing values are written 'N/A', 'NA', 'None' and 2 more."
    assert eio._missing_seen(["A", "na", 0]) == ""
    assert eio._missing_rename(["A", np.nan, ""]) == ""
    assert eio._missing_rename(["A", " NA"]) == RENAME_NA
    assert eio._missing_rename(["A", "NA", "None"]) == (
        " A label file reads 'NA' and 'None' as missing values. If one of them is the "
        "name of a cell type, rename it.")


# ============================================================ "error"
def _label_forms(tmp_path):
    """``{form: (output, labels, src)}``: 30 of 300 labels missing, each way
    of passing labels."""
    emb, lab = _cells()
    nan, ids = _gaps(lab), [f"cell{i}" for i in range(N)]
    one = tmp_path / "cty.csv"                      # as export writes it
    pd.DataFrame({"x": _gaps(lab, "NA")}).to_csv(one, index=False)
    two = tmp_path / "obs.csv"                      # obs[["celltype"]].to_csv
    pd.DataFrame({"celltype": nan}, index=ids).sample(frac=1.0, random_state=3).to_csv(two)
    halves = [tmp_path / "cty1.csv", tmp_path / "cty2.csv"]
    for path, part in zip(halves, (nan[:150], nan[150:])):
        pd.DataFrame({"x": part}).to_csv(path, index=False)
    a = _adata(emb, celltype=nan, ct_cat=pd.Categorical(nan))
    return {
        "array": (emb, nan, ""),
        "list": (emb, list(nan), ""),
        "Series": (emb, pd.Series(nan), ""),
        "Series by id": (a, pd.Series(nan, index=ids).sample(frac=1.0, random_state=1), ""),
        "Categorical": (emb, pd.Categorical(nan), ""),
        "string dtype": (emb, pd.Series(_gaps(lab, pd.NA), dtype="string"), ""),
        "obs column": (a, "celltype", ""),
        "obs category column": (a, "ct_cat", ""),
        "one-column file": (emb, str(one), " (cty.csv)"),
        "Path": (emb, one, " (cty.csv)"),
        "id,label file": (a, str(two), " (obs.csv)"),
        "file list": (emb, [str(p) for p in halves], " (cty1.csv, cty2.csv)"),
        "labels_for dict": (emb, {"cty1": str(halves[0]), "cty2": str(halves[1])},
                            " (cty1.csv, cty2.csv)"),
    }


@pytest.mark.parametrize("form", [
    "array", "list", "Series", "Series by id", "Categorical", "string dtype",
    "obs column", "obs category column", "one-column file", "Path", "id,label file",
    "file list", "labels_for dict"])
def test_evaluate_raises_on_missing_labels(form, tmp_path):
    output, labels, src = _label_forms(tmp_path)[form]
    for kw in ({}, {"missing_labels": "error"}):
        with pytest.raises(ValueError) as e:
            mtb.evaluate(output, labels=labels, metrics=["ASW"], **kw)
        assert str(e.value) == _e1(src), form


def test_evaluate_raises_on_missing_labels_whole_message():
    emb, lab = _cells()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=_gaps(lab), metrics=["ASW"])
    assert str(e.value) == (
        "labels: 30 of the 300 cells have no label. The missing value is written NaN. "
        "A missing label is not a cell type, so evaluate does not score it. Give every "
        'cell a label, or pass missing_labels="drop" to score the 270 labelled cells '
        "only.")


@pytest.mark.parametrize("fill, seen, rename", [
    (None, NAN, ""),
    (pd.NA, NAN, ""),
    ("", "The missing value is written ''.", ""),
    ("  ", "The missing value is written ''.", ""),
    ("NA", "The missing value is written 'NA'.", RENAME_NA),
    ("NA ", "The missing value is written 'NA'.", RENAME_NA),
    (" NA", "The missing value is written 'NA'.", RENAME_NA),
    ("None", "The missing value is written 'None'.", RENAME_NA.replace("'NA'", "'None'")),
    ("nan", "The missing value is written 'nan'.", RENAME_NA.replace("'NA'", "'nan'")),
], ids=repr)
def test_every_spelling_of_missing_raises_and_is_named(fill, seen, rename):
    emb, lab = _cells()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=_gaps(lab, fill), metrics=["ASW"])
    assert str(e.value) == _e1(seen=seen, rename=rename)


def test_several_spellings_are_listed_once_each():
    emb, lab = _cells()
    lab = _gaps(lab)                       # 30 NaN
    lab[[0, 1]], lab[2], lab[3] = "NA", "None", " "
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    assert str(e.value) == _e1(
        n=34, seen="The missing values are written NaN, '', 'NA' and 'None'.",
        rename=(" A label file reads 'NA' and 'None' as missing values. If one of them "
                "is the name of a cell type, rename it."))


def test_a_text_numpy_made_of_nan_is_missing_too():
    emb, lab = _cells()
    text = np.array(list(_gaps(lab)))               # dtype <U3: NaN is now 'nan'
    assert text.dtype.kind == "U"
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=text, metrics=["ASW"])
    assert str(e.value) == _e1(seen="The missing value is written 'nan'.",
                               rename=RENAME_NA.replace("'NA'", "'nan'"))


def test_a_file_whose_missing_word_has_a_blank_is_named(tmp_path):
    """pandas reads ``NA`` as missing and keeps ``NA `` as text: both are
    missing here."""
    emb, lab = _cells()
    path = tmp_path / "cty.csv"
    path.write_text("x\n" + "".join(f"{v}\n" for v in _gaps(lab, "NA ")))
    assert set(eio.read_labels(path)) == {"A", "B", "C", "NA "}
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(path), metrics=["ASW"])
    assert str(e.value) == _e1(" (cty.csv)", seen="The missing value is written 'NA'.",
                               rename=RENAME_NA)


def test_counts_carry_thousands_separators():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(1200, 4))
    lab = np.repeat(["A", "B"], 600).astype(object)
    emb[:600] += 4
    lab[np.arange(0, 1200, 12)] = None              # 100 cells
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    assert str(e.value) == (
        "labels: 100 of the 1,200 cells have no label. The missing value is written "
        "NaN. A missing label is not a cell type, so evaluate does not score it. Give "
        'every cell a label, or pass missing_labels="drop" to score the 1,100 labelled '
        "cells only.")
    out, said = _drop(emb, labels=lab, metrics=["ASW"])
    assert said == ["Scored the 1,100 labelled cells of 1,200. 100 cells have no label "
                    "and were left out, so the scores describe the labelled cells only."]
    assert (out.attrs["cells_scored"], out.attrs["cells_total"]) == (1100, 1200)
    long = mtb.to_long(out, method="M")
    assert set(long["scored_with"]) == {f"none/none/{mtb.__version__}/1100 of 1200 cells"}


def test_blank_lines_in_a_one_column_file_keep_the_count_error(tmp_path):
    """A truly empty line is no row for the reader: the file is short, as before."""
    emb, lab = _cells()
    path = tmp_path / "cty.csv"
    path.write_text("x\n" + "".join(f"{v}\n" for v in _gaps(lab, "")))
    assert len(eio.read_labels(path)) == 270
    for kw in ({}, {"missing_labels": "drop"}):
        with pytest.raises(ValueError, match=r"^labels has 270 entries for 300 cells "
                                             r"in the embedding\."):
            mtb.evaluate(emb, labels=str(path), metrics=["ASW"], **kw)


@pytest.mark.parametrize("missing_labels", ["error", "drop"])
def test_evaluate_raises_when_every_label_is_missing(missing_labels, tmp_path):
    emb, _ = _cells()
    kw = dict(metrics=["ASW"], missing_labels=missing_labels)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=np.full(N, np.nan, dtype=object), **kw)
    assert str(e.value) == ("labels: none of the 300 cells has a label, so there is "
                            "nothing to score. The missing value is written NaN.")
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=["NA"] * N, **kw)
    assert str(e.value) == ("labels: none of the 300 cells has a label, so there is "
                            "nothing to score. The missing value is written 'NA'."
                            + RENAME_NA)
    path = tmp_path / "cty.csv"
    pd.DataFrame({"x": ["NA"] * N}).to_csv(path, index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(path), **kw)
    assert str(e.value) == ("labels (cty.csv): none of the 300 cells has a label, so "
                            "there is nothing to score. The missing value is written NaN.")


@pytest.mark.parametrize("value", ["skip", "Drop", "", None, True, 0, ["drop"]], ids=repr)
def test_missing_labels_value_is_checked(value):
    emb, lab = _cells()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, metrics=["ASW"], missing_labels=value)
    assert str(e.value) == f'missing_labels must be "error" or "drop", got {value!r}.'


def test_missing_labels_value_is_checked_before_anything_else():
    # no labels, an unknown category, an output that does not exist: the
    # value of missing_labels is still the first thing said
    with pytest.raises(ValueError) as e:
        mtb.evaluate("nothere.h5", category="nope", missing_labels="keep")
    assert str(e.value) == 'missing_labels must be "error" or "drop", got \'keep\'.'


def test_the_command_line_names_its_flags(tmp_path, monkeypatch):
    emb, lab = _cells()
    path = tmp_path / "cty.csv"
    pd.DataFrame({"x": _gaps(lab)}).to_csv(path, index=False)
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(path), metrics=["ASW"])
    assert str(e.value) == (
        "--labels (cty.csv): 30 of the 300 cells have no label. The missing value is "
        "written NaN. A missing label is not a cell type, so evaluate does not score "
        "it. Give every cell a label, or pass --missing-labels drop to score the 270 "
        "labelled cells only.")
    assert "missing_labels=" not in str(e.value)


# ============================================================ "drop"
def test_drop_equals_scoring_the_labelled_cells(tmp_path):
    emb, lab = _cells()
    bat = _donors()
    nan = _gaps(lab)
    # the plain call, and the value the released 0.3.4 gives for those cells
    got, said = _drop(emb, labels=nan, metrics=["ASW"])
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], metrics=["ASW"]))
    assert got.loc["ASW", "Value"] == pytest.approx(ASW_034_LABELLED, abs=1e-9)
    assert len(said) == 1
    # batch= and clustering= lose the same cells
    kw = dict(metrics=["ARI", "NMI", "ASW", "ASW_batch", "GC"])
    got, _ = _drop(emb, labels=nan, batch=bat, clustering=lab, **kw)
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], batch=bat[KEEP],
                            clustering=lab[KEEP], **kw))
    # an AnnData with obs columns, every argument by name
    a = _adata(emb, celltype=nan, donor=bat, leiden=lab)
    got, _ = _drop(a, labels="celltype", batch="donor", clustering="leiden", **kw)
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], batch=bat[KEEP],
                            clustering=lab[KEEP], **kw))
    # a list of label files: the file of origin is the batch, cut alike
    files = [tmp_path / "cty1.csv", tmp_path / "cty2.csv"]
    for path, part in zip(files, (nan[:150], nan[150:])):
        pd.DataFrame({"x": part}).to_csv(path, index=False)
    origin = np.repeat([1, 2], 150)
    got, _ = _drop(emb, labels=[str(p) for p in files])
    assert {"ARI", "ASW_batch", "iLISI"} <= set(got.index)
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], batch=origin[KEEP],
                            verbose=False))
    # a dims x cells output is turned by the count of all the labels
    with pytest.warns(UserWarning) as rec:
        got = mtb.evaluate(emb.T, labels=nan, metrics=["ASW"], missing_labels="drop")
    assert [str(w.message) for w in rec][0] == \
        "output was (dims x cells) (8, 300); transposed to (cells x dims)"
    assert len(_scored(rec)) == 1
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], metrics=["ASW"]))


def test_drop_scores_the_sweep_on_the_labelled_cells():
    """ARI without clustering=: the Leiden sweep runs on the cells that are
    scored, as it does for a call with those cells only."""
    emb, lab = _cells()
    got, _ = _drop(emb, labels=_gaps(lab, "NA"), metrics=["ARI", "NMI"])
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], metrics=["ARI", "NMI"],
                            verbose=False))
    assert got.attrs["clustering"] == "sweep"


def test_drop_warns_and_records_the_cell_counts():
    emb, lab = _cells()
    with pytest.warns(UserWarning) as rec:
        wide = mtb.evaluate(emb, labels=_gaps(lab), metrics=["ASW"], missing_labels="drop")
    assert _scored(rec) == [
        "Scored the 270 labelled cells of 300. 30 cells have no label and were left "
        "out, so the scores describe the labelled cells only."]
    # the warning points at the line that called evaluate
    assert [Path(w.filename).name for w in rec
            if str(w.message).startswith("Scored the")] == [Path(__file__).name]
    assert wide.attrs["cells_scored"] == 270 and wide.attrs["cells_total"] == 300
    assert type(wide.attrs["cells_scored"]) is int and type(wide.attrs["cells_total"]) is int
    assert list(wide.attrs) == ["leiden_flavor", "clustering", "multibench_version",
                                "scib_version", "cells_scored", "cells_total"]
    assert pipeline._scored_with(wide.attrs) == \
        f"none/none/{mtb.__version__}/270 of 300 cells"
    long = mtb.to_long(wide, method="Mine", dataset="D11", category="vertical")
    assert set(long["scored_with"]) == {f"none/none/{mtb.__version__}/270 of 300 cells"}
    assert all(s.endswith("/270 of 300 cells") for s in long["scored_with"])
    assert long.attrs == wide.attrs
    # the consumer of scored_with reads its first two parts, which did not move
    assert long["scored_with"].str.split("/").str[:2].tolist() == [["none", "none"]]


def test_drop_of_one_cell_reads_in_the_singular():
    emb, lab = _cells()
    lab[7] = None
    out, said = _drop(emb, labels=lab, metrics=["ASW"])
    assert said == ["Scored the 299 labelled cells of 300. 1 cell has no label and was "
                    "left out, so the scores describe the labelled cells only."]
    assert pipeline._scored_with(out.attrs).endswith("/299 of 300 cells")


def test_drop_without_gaps_changes_nothing():
    emb, lab = _cells()
    bat = _donors()
    kw = dict(labels=lab, batch=bat, metrics=["ASW", "ASW_batch"])
    plain = _silent(emb, **kw)
    drop = _silent(emb, missing_labels="drop", **kw)
    error = _silent(emb, missing_labels="error", **kw)
    _same(drop, plain)
    _same(error, plain)
    for out in (plain, drop, error):
        assert list(out.attrs) == ["leiden_flavor", "clustering", "multibench_version",
                                   "scib_version"]
    # the value of the released 0.3.4 for these cells
    assert plain.loc["ASW", "Value"] == pytest.approx(ASW_034, abs=1e-9)
    long = mtb.to_long(drop, method="M")
    assert set(long["scored_with"]) == {f"none/none/{mtb.__version__}"}
    assert list(long.attrs) == list(plain.attrs)


def test_drop_names_a_lost_batch(tmp_path):
    emb, lab = _cells()
    bat = _donors()
    base = ("Scored the 200 labelled cells of 300. 100 cells have no label and were "
            "left out, so the scores describe the labelled cells only.")
    kw = dict(metrics=["ASW", "ASW_batch"])
    # donor d3 has no labelled cell: two of the three batches are left
    _, said = _drop(emb, labels=_gaps(lab, at=bat == "d3"), batch=bat, **kw)
    assert said == [base + " The labelled cells come from 2 of the 3 batches."]
    # only donor d1 is labelled
    _, said = _drop(emb, labels=_gaps(lab, at=bat != "d1"), batch=bat, **kw)
    assert said == [
        "Scored the 100 labelled cells of 300. 200 cells have no label and were left "
        "out, so the scores describe the labelled cells only. The labelled cells are "
        "all in one batch, so the batch metrics do not measure mixing."]
    # every batch keeps a labelled cell: nothing to add
    _, said = _drop(emb, labels=_gaps(lab), batch=bat, **kw)
    assert said == ["Scored the 270 labelled cells of 300. 30 cells have no label and "
                    "were left out, so the scores describe the labelled cells only."]
    # no batch metric is scored: the batches are not counted
    _, said = _drop(emb, labels=_gaps(lab, at=bat == "d3"), metrics=["ASW"])
    assert said == [base]


def test_drop_with_one_of_two_label_files_all_missing(tmp_path):
    emb, lab = _cells()
    files = [tmp_path / "cty1.csv", tmp_path / "cty2.csv"]
    pd.DataFrame({"x": lab[:150]}).to_csv(files[0], index=False)
    pd.DataFrame({"x": ["NA"] * 150}).to_csv(files[1], index=False)
    out, said = _drop(emb, labels=[str(p) for p in files])
    assert said == [
        "Scored the 150 labelled cells of 300. 150 cells have no label and were left "
        "out, so the scores describe the labelled cells only. The labelled cells are "
        "all in one batch, so the batch metrics do not measure mixing."]
    assert (out.attrs["cells_scored"], out.attrs["cells_total"]) == (150, 300)
    _same(out, mtb.evaluate(emb[:150], labels=lab[:150], batch=np.ones(150, dtype=int),
                            metrics="all", verbose=False))
    # without the flag, the same files stop the call
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=[str(p) for p in files])
    assert str(e.value) == _e1(" (cty1.csv, cty2.csv)", n=150)


# ============================================================ the reference
def test_docstring_names_the_argument_and_its_errors():
    doc = inspect.getdoc(mtb.evaluate)
    flat = " ".join(doc.split())
    # the first choice is the default (tests/test_docstring_defaults.py)
    assert '\nmissing_labels : {"error", "drop"}\n' in doc
    assert ("A cell without a label (unless ``missing_labels=\"drop\"``), no cell "
            "with a label, or a cell without a batch in ``batch``.") in flat
    assert "**Missing labels.** A label is missing when it is NaN" in flat
    assert "``cells_scored`` and ``cells_total``" in flat
    assert "cells_scored" in " ".join(mtb.to_long.__doc__.split())
