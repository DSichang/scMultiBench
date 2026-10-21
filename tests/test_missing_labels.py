"""evaluate() does not score a missing label as a cell type (0.3.5).

0.3.4 cast the labels to text, so NaN became one more cell type ``"nan"``
and every score moved without a word (ASW 0.7160 where the labelled cells
score 0.7814 on the cells below). Now:

- one definition of "missing" (``eval.io.missing_mask``): NaN, ``None``,
  ``pd.NA``, blank text, and any text that the label reader itself reads back
  as missing (``NA``, ``None``, ``nan``, ...), with or without blanks around it;
- evaluate raises and says how many cells and one text that reads as missing,
  if any, and to score the labelled cells instead;
- complete labels score exactly as before.

evaluate returns the long table (``metric, value, ...``).
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

#: what the message adds for a text that a label file reads as missing
EX_NA = ", for example 'NA'"
#: the end of the message when no cell has a label
NONE_FIX = " Pass the column or file that holds the cell types."
#: the end of the message when some cells have no label (Python, command line)
KEEP_FIX = " Score the other cells only, as adata[adata.obs[col].notna()]."
KEEP_FIX_CLI = (" Score the other cells only: leave the unlabelled ones out of the "
                "embedding and the label file.")


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


def _e1(src="", example="", n=30, total=N):
    return f"labels{src}: {n} of the {total} cells have no label{example}." + KEEP_FIX


def _value(out, code):
    """The value of ``code`` in evaluate's long table."""
    return out.set_index("metric")["value"][code]


def _same(got, want):
    """The same metrics with the same values, NaN included."""
    assert got["metric"].tolist() == want["metric"].tolist()
    np.testing.assert_array_equal(got["value"].to_numpy(), want["value"].to_numpy())


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
#: labels too. Python's csv writer leaves a lone carriage return bare up to
#: 3.10, and 3.10 refuses a NUL, so the file that export writes of them
#: (DataFrame.to_csv) depends on the Python version; the definition does not
CR_OR_NUL = ["a\rb", "x\r\ny", "C\x00cell"]


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


@pytest.mark.parametrize("value", NOT_MISSING + CR_OR_NUL, ids=repr)
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


def test_texts_with_a_line_break_or_a_nul_follow_the_reader(tmp_path):
    """Whatever the reader makes of such a text in a label file (every field
    in quotes here), the definition says the same, one answer per text."""
    texts = CR_OR_NUL + ["two\nlines", "NA\rNA", "NA\x00x", "\x00cell", "A", "NA", "None",
                         'say "x"', "B, C"]
    path = tmp_path / "cty.csv"
    with open(path, "w", newline="") as f:
        f.write("x\n" + "".join('"' + t.replace('"', '""') + '"\n' for t in texts))
    back = eio.read_labels(path)
    assert len(back) == len(texts)
    read_as_missing = {t for t, v in zip(texts, back) if pd.isna(v)}
    assert read_as_missing == set(eio.missing_words(texts))
    assert {"NA", "None"} <= read_as_missing and not read_as_missing & set(CR_OR_NUL)
    assert eio.missing_mask(texts).tolist() == [t in read_as_missing for t in texts]


def test_the_round_trip_does_not_go_through_the_csv_writer(monkeypatch):
    """Python 3.10 and older write ``a\\rb`` as two rows (and 3.10 raises on
    a NUL), so complete labels with such a name stopped evaluate there. The
    definition writes its own rows, the same on every Python."""
    def to_csv(*args, **kwargs):
        raise AssertionError("missing_words wrote its texts with DataFrame.to_csv")
    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    values = ["A", "a\rb", "C\x00cell", "NA", "two\nlines", 'say "x"', "B, C"]
    assert eio.missing_words(values) == ["NA"]
    assert eio.missing_mask(values).tolist() == [False, False, False, True, False, False,
                                                 False]


def test_one_row_per_text_whatever_the_text(monkeypatch):
    """Every text is one row of the round trip: a comma, a quote, a line
    break or a NUL inside it does not move the answers to other texts."""
    rows = []
    real = eio._read_table

    def spy(src, sep):
        out = real(src, sep)
        rows.append(len(out))
        return out
    monkeypatch.setattr(eio, "_read_table", spy)
    awkward = ["a\rb", "a\nb", "a\r\nb", "a\x00b", "a,b", 'a"b', '"', '""', "'", "a\tb",
               "#a", "a\\"]
    texts = sorted(set(awkward + ["NA", "nan", "A", "B"]))
    assert eio.missing_words(texts) == ["NA", "nan"]
    assert rows == [len(texts)]
    for text in awkward:
        assert eio.missing_mask(["NA", text, "A"]).tolist() == [True, False, False], text
    # blanks around a word are stripped before the round trip, a carriage
    # return among them
    assert eio.missing_mask(["A", "\rNA\r"]).tolist() == [False, True]


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
    """The two sentences are now one example (``_missing_example``): the
    first text that a label file reads as missing, nothing for NaN or a blank."""
    assert eio._missing_example(["A", np.nan]) == ""
    assert eio._missing_example(["A", " "]) == ""
    assert eio._missing_example(["A", "NA "]) == EX_NA
    assert eio._missing_example(["A", None, "NA"]) == EX_NA
    assert eio._missing_example(["A", None, "", "NA"]) == EX_NA
    many = ["NA", "None", "nan", "null", "N/A", "A"]
    assert eio._missing_example(many) == ", for example 'N/A'"
    assert eio._missing_example(["A", "na", 0]) == ""
    assert eio._missing_example(["A", np.nan, ""]) == ""
    assert eio._missing_example(["A", " NA"]) == EX_NA
    assert eio._missing_example(["A", "NA", "None"]) == EX_NA


# ============================================================ "error"
def _label_forms(tmp_path):
    """``{form: (output, labels, src)}``: 30 of 300 labels missing, each way
    of passing labels. A file that writes the missing labels as a word adds
    the example that names one, ``(..., example)``."""
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
    # files whose missing labels are a word, or an empty field
    none = tmp_path / "none" / "cty.csv"
    none.parent.mkdir()
    pd.DataFrame({"x": _gaps(lab, "None")}).to_csv(none, index=False)
    empty = tmp_path / "empty.csv"
    empty.write_text("barcode,celltype\n" + "".join(
        f"{i},{v}\n" for i, v in zip(ids, _gaps(lab, ""))))
    null = tmp_path / "null.tsv"                    # ids in another order than the cells
    pd.DataFrame({"celltype": _gaps(lab, "null")}, index=ids).sample(
        frac=1.0, random_state=5).to_csv(null, sep="\t")
    words = [tmp_path / "w" / "cty1.csv", tmp_path / "w" / "cty2.csv"]
    words[0].parent.mkdir()
    for path, part in zip(words, (_gaps(lab, "NA")[:150], _gaps(lab, "N/A")[150:])):
        pd.DataFrame({"x": part}).to_csv(path, index=False)
    na = (EX_NA,)
    return {
        "array": (emb, nan, ""),
        "list": (emb, list(nan), ""),
        "Series": (emb, pd.Series(nan), ""),
        "Series by id": (a, pd.Series(nan, index=ids).sample(frac=1.0, random_state=1), ""),
        "Categorical": (emb, pd.Categorical(nan), ""),
        "string dtype": (emb, pd.Series(_gaps(lab, pd.NA), dtype="string"), ""),
        "obs column": (a, "celltype", ""),
        "obs category column": (a, "ct_cat", ""),
        "one-column file": (emb, str(one), " (cty.csv)", *na),
        "Path": (emb, one, " (cty.csv)", *na),
        "id,label file": (a, str(two), " (obs.csv)"),
        "file list": (emb, [str(p) for p in halves], " (cty1.csv, cty2.csv)"),
        "labels_for dict": (emb, {"cty1": str(halves[0]), "cty2": str(halves[1])},
                            " (cty1.csv, cty2.csv)"),
        "file of None": (emb, str(none), " (cty.csv)", ", for example 'None'"),
        "file with an empty field": (a, str(empty), " (empty.csv)"),
        "id,label file of null": (a, null, " (null.tsv)", ", for example 'null'"),
        # two words: the first one, in sorted order, is the example
        "file list of words": (
            emb, [str(p) for p in words], " (cty1.csv, cty2.csv)", ", for example 'N/A'"),
    }


@pytest.mark.parametrize("form", [
    "array", "list", "Series", "Series by id", "Categorical", "string dtype",
    "obs column", "obs category column", "one-column file", "Path", "id,label file",
    "file list", "labels_for dict", "file of None", "file with an empty field",
    "id,label file of null", "file list of words"])
def test_evaluate_raises_on_missing_labels(form, tmp_path):
    output, labels, src, *said = _label_forms(tmp_path)[form]
    # NaN, and an empty field of a file, are no word: no example
    (example,) = said or ("",)
    for kw in ({},):
        with pytest.raises(ValueError) as e:
            mtb.evaluate(output, labels=labels, metrics=["ASW"], **kw)
        assert str(e.value) == _e1(src, example), form


def test_a_label_file_names_the_word_it_holds_whole_message(tmp_path):
    """A cell type called None in a CSV: the reader reads NaN, and the message
    names the word in the file as its example. No sentence says to rename it."""
    emb, lab = _cells()
    path = tmp_path / "cty.csv"
    path.write_text("x\n" + "".join(f"{v}\n" for v in np.where(lab == "C", "None", lab)))
    assert pd.isna(eio.read_labels(path)).sum() == 100
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(path), metrics=["ASW"])
    assert str(e.value) == (
        "labels (cty.csv): 100 of the 300 cells have no label, for example 'None'."
        + KEEP_FIX)
    assert "rename" not in str(e.value)


def test_a_label_file_is_read_as_written_for_the_message_only(tmp_path):
    emb, lab = _cells()
    ids = [f"cell{i}" for i in range(N)]
    mixed = _gaps(lab, "NA")
    mixed[0], mixed[1], mixed[2], mixed[3] = "None", "", " ", "NA "
    path = tmp_path / "obs.csv"
    path.write_text("barcode,celltype\n" + "".join(
        f'{i},"{v}"\n' for i, v in zip(ids, mixed)))            # a quoted "NA" is NA too
    plain, first = eio.read_labels_ids(path)
    shown, first_shown = eio.read_labels_ids(path, as_written=True)
    assert first.equals(first_shown)
    # the reader's values did not move: NaN for the words and the empty field
    assert pd.isna(plain).sum() == 32 and set(plain[~pd.isna(plain)]) == {
        "A", "B", "C", " ", "NA "}
    # as written, the words are back and the empty field is still NaN
    assert shown[0] == "None" and pd.isna(shown[1]) and shown[2] == " " and shown[3] == "NA "
    assert set(shown[GAPS]) == {"NA"}
    keep = ~pd.isna(plain)
    assert (shown[keep] == plain[keep]).all()
    assert (eio.missing_mask(shown) == eio.missing_mask(plain)).all()
    assert eio.read_labels(path, as_written=True).tolist() == shown.tolist()
    assert eio.as_vector([path, path], as_written=True).tolist() == shown.tolist() * 2
    with pytest.raises(ValueError) as e:
        mtb.evaluate(_adata(emb), labels=str(path), metrics=["ASW"])
    assert str(e.value) == _e1(" (obs.csv)", EX_NA, n=34)
    # numbers stay numbers where nothing is missing, and a cluster number
    # next to NA is named as the file writes it
    numbers = tmp_path / "clusters.csv"
    numbers.write_text("x\n" + "".join(f"{v}\n" for v in np.tile([1, 2, 3], 100)))
    assert eio.read_labels(numbers, as_written=True).dtype == np.int64
    numbers.write_text("x\n" + "".join(f"{v}\n" for v in _gaps(np.tile([1, 2, 3], 100), "NA")))
    assert eio.read_labels(numbers).dtype == np.float64
    assert eio._missing_example(eio.read_labels(numbers, as_written=True)) == EX_NA


def test_a_file_read_twice_for_the_message_warns_once(tmp_path):
    """The second read, which names the words, repeats no warning of the first."""
    emb, lab = _cells()
    path = tmp_path / "obs.csv"
    pd.DataFrame({"celltype": _gaps(lab, "NA")},
                 index=[f"cell{i}" for i in range(N)]).to_csv(path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as e:
            mtb.evaluate(emb, labels=str(path), metrics=["ASW"])
    assert str(e.value) == _e1(" (obs.csv)", EX_NA)
    assert [str(w.message) for w in rec] == [
        "The first column of obs.csv looks like cell ids, but the embedding has no cell "
        "ids. The file is matched by position. Check that its rows follow the rows of "
        "the embedding."]


def test_evaluate_raises_on_missing_labels_whole_message():
    emb, lab = _cells()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=_gaps(lab), metrics=["ASW"])
    assert str(e.value) == "labels: 30 of the 300 cells have no label." + KEEP_FIX
    assert "missing_labels" not in str(e.value)


@pytest.mark.parametrize("fill, example", [
    (None, ""),
    (pd.NA, ""),
    ("", ""),
    ("  ", ""),
    ("NA", EX_NA),
    ("NA ", EX_NA),
    (" NA", EX_NA),
    ("None", ", for example 'None'"),
    ("nan", ", for example 'nan'"),
], ids=repr)
def test_every_spelling_of_missing_raises_and_is_named(fill, example):
    """A text that reads as missing is named as the example; NaN and a blank
    are not named."""
    emb, lab = _cells()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=_gaps(lab, fill), metrics=["ASW"])
    assert str(e.value) == _e1(example=example)


def test_several_spellings_are_listed_once_each():
    """Of several spellings the message now names one, the first word."""
    emb, lab = _cells()
    lab = _gaps(lab)                       # 30 NaN
    lab[[0, 1]], lab[2], lab[3] = "NA", "None", " "
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    assert str(e.value) == _e1(n=34, example=EX_NA)
    assert "'None'" not in str(e.value)


def test_a_text_numpy_made_of_nan_is_missing_too():
    emb, lab = _cells()
    text = np.array(list(_gaps(lab)))               # dtype <U3: NaN is now 'nan'
    assert text.dtype.kind == "U"
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=text, metrics=["ASW"])
    assert str(e.value) == _e1(example=", for example 'nan'")


def test_a_file_whose_missing_word_has_a_blank_is_named(tmp_path):
    """pandas reads ``NA`` as missing and keeps ``NA `` as text: both are
    missing here."""
    emb, lab = _cells()
    path = tmp_path / "cty.csv"
    path.write_text("x\n" + "".join(f"{v}\n" for v in _gaps(lab, "NA ")))
    assert set(eio.read_labels(path)) == {"A", "B", "C", "NA "}
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(path), metrics=["ASW"])
    assert str(e.value) == _e1(" (cty.csv)", example=EX_NA)


def test_blank_lines_in_a_one_column_file_keep_the_count_error(tmp_path):
    """A truly empty line is no row for the reader: the file is short, as before."""
    emb, lab = _cells()
    path = tmp_path / "cty.csv"
    path.write_text("x\n" + "".join(f"{v}\n" for v in _gaps(lab, "")))
    assert len(eio.read_labels(path)) == 270
    for kw in ({},):
        with pytest.raises(ValueError, match=r"^labels has 270 entries for 300 cells "
                                             r"in the embedding\."):
            mtb.evaluate(emb, labels=str(path), metrics=["ASW"], **kw)


def test_evaluate_raises_when_every_label_is_missing(tmp_path):
    emb, _ = _cells()
    kw = dict(metrics=["ASW"])
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=np.full(N, np.nan, dtype=object), **kw)
    assert str(e.value) == "labels: none of the 300 cells has a label." + NONE_FIX
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=["NA"] * N, **kw)
    assert str(e.value) == ("labels: none of the 300 cells has a label, for example "
                            "'NA'." + NONE_FIX)
    path = tmp_path / "cty.csv"
    pd.DataFrame({"x": ["NA"] * N}).to_csv(path, index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(path), **kw)
    assert str(e.value) == ("labels (cty.csv): none of the 300 cells has a label, for "
                            "example 'NA'." + NONE_FIX)
    path.write_text("x\n" + '""\n' * N)                         # 300 empty fields
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(path), **kw)
    assert str(e.value) == ("labels (cty.csv): none of the 300 cells has a label."
                            + NONE_FIX)


def test_complete_labels_score_as_before():
    emb, lab = _cells()
    bat = _donors()
    plain = _silent(emb, labels=lab, batch=bat, metrics=["ASW", "ASW_batch"])
    assert list(plain.attrs) == ["leiden_flavor", "clustering", "multibench_version",
                                 "scib_version"]
    # the value of the released 0.3.4 for these cells
    assert _value(plain, "ASW") == pytest.approx(ASW_034, abs=1e-9)
    assert set(plain["scored_with"]) == {
        f"{plain.attrs['leiden_flavor']}/search/{mtb.__version__}"}


@pytest.mark.parametrize("name", ["C\rcell", "C\r\ncell", "C\ncell", "C\x00cell"], ids=repr)
def test_a_line_break_or_a_nul_in_a_name_changes_nothing(name):
    """Complete labels and batches score as the released 0.3.4 scored them,
    also on Python 3.10 and older, whose csv writer splits or refuses such a
    name."""
    emb, lab = _cells()
    named = np.where(lab == "C", name, lab).astype(object)
    for kw in ({},):
        got = _silent(emb, labels=named, metrics=["ASW"], **kw)
        assert _value(got, "ASW") == pytest.approx(ASW_034, abs=1e-9)
    bat = _donors()
    kw = dict(labels=lab, metrics=["ASW_batch"])
    _same(_silent(emb, batch=np.where(bat == "d3", name, bat).astype(object), **kw),
          _silent(emb, batch=bat, **kw))


# ============================================================ the reference
def test_docstring_names_the_argument_and_its_errors():
    doc = inspect.getdoc(mtb.evaluate)
    flat = " ".join(doc.split())
    # 0.4.0: missing_labels= is deprecated, so the reference no longer lists it
    assert "missing_labels" not in doc
    assert "missing_labels" not in inspect.signature(mtb.evaluate).parameters
    assert "Missing or misaligned labels or batches" in flat              # Raises

