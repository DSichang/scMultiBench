"""evaluate(batch=) does not count a missing batch as a batch (0.3.5).

0.3.4 cast the batch vector to text, so the cells without a donor became one
more batch ``"nan"`` and the batch metrics moved without a word. A batch that
the caller passes is now checked with the definition the labels use
(``eval.io.missing_mask``) and a missing value raises, whatever
``missing_labels`` says. With ``missing_labels="drop"`` only the cells that
are scored need a batch. The batch that evaluate makes itself (the label file
of each cell, or the placeholder when no batch metric is scored) is not
checked.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench.eval import io as eio
from multibench.eval import pipeline
from tests.test_missing_labels import (GAPS, KEEP, N, _adata, _cells, _donors, _drop,
                                       _gaps, _same, _silent)

pytest.importorskip("scib")

METRICS = ["ASW", "ASW_batch", "GC"]
#: the cells of _cells() with donor d1 moved apart, as the released
#: multibench-sc 0.3.4 scored them
SCORES_034 = {"ASW": 0.7603380351178893, "ASW_batch": 0.9207041314771548, "GC": 1.0}

def _bv1(src="", example="", n=30, total=N):
    return (f"batch{src}: {n} of the {total} cells have no batch{example}. Give every "
            f"cell a batch.")


@pytest.mark.parametrize("fill, seen", [
    (np.nan, ""),
    (None, ""),
    (pd.NA, ""),
    ("", ""),
    ("nan", ", for example 'nan'"),
    ("NA", ", for example 'NA'"),
], ids=repr)
def test_evaluate_raises_on_a_missing_batch(fill, seen, tmp_path):
    emb, lab = _cells()
    bat = _gaps(_donors(), fill)
    a = _adata(emb, celltype=lab, donor=bat)
    path = tmp_path / "batch.csv"
    pd.DataFrame({"x": bat}).to_csv(path, index=False)
    # a file holds NaN, None and '' as an empty field, which the reader gives
    # as NaN and the message does not name; a word that it reads as NaN is
    # named as the file writes it
    in_file = seen if isinstance(fill, str) and fill else ""
    forms = {"array": (emb, bat, "", seen), "list": (emb, list(bat), "", seen),
             "Series": (emb, pd.Series(bat), "", seen),
             "obs column": (a, "donor", "", seen), "obs: column": (a, "obs:donor", "", seen),
             "file": (emb, str(path), " (batch.csv)", in_file)}
    for form, (output, batch, src, said) in forms.items():
        for kw in ({}, {"missing_labels": "drop"}):
            with pytest.raises(ValueError) as e:
                mtb.evaluate(output, labels=lab, batch=batch, metrics=["ASW_batch"], **kw)
            assert str(e.value) == _bv1(src, said), (form, kw)


def test_evaluate_raises_on_a_missing_batch_whole_message():
    emb, lab = _cells()
    donors = np.tile([1.0, 2.0, 3.0], 100)
    donors[GAPS] = np.nan                          # a float column with NaN
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, batch=donors, metrics=["ASW_batch"])
    assert str(e.value) == (
        "batch: 30 of the 300 cells have no batch. Give every cell a batch.")
    # a batch that no metric of the call reads is checked all the same
    with pytest.warns(UserWarning, match="batch= changes nothing here"):
        with pytest.raises(ValueError) as e:
            mtb.evaluate(emb, labels=lab, batch=donors, metrics=["ASW"])
    assert str(e.value) == _bv1()


def test_the_command_line_names_its_flag(tmp_path, monkeypatch):
    emb, lab = _cells()
    path = tmp_path / "donors.csv"
    pd.DataFrame({"x": _gaps(_donors(), "")}).to_csv(path, index=False)
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, batch=str(path), metrics=["ASW_batch"])
    assert str(e.value) == "--" + _bv1(" (donors.csv)")


def test_a_batch_file_names_the_word_it_holds(tmp_path):
    """Donors written NA or None in a file: the reader reads NaN, and the
    message names the word, whichever way the file is matched to the cells."""
    emb, lab = _cells()
    ids = [f"cell{i}" for i in range(N)]
    sheet = tmp_path / "sheet.csv"                 # ids in another order than the cells
    pd.DataFrame({"donor": _gaps(_donors(), "None")}, index=ids).sample(
        frac=1.0, random_state=2).to_csv(sheet)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(_adata(emb), labels=lab, batch=sheet, metrics=["ASW_batch"])
    assert str(e.value) == _bv1(" (sheet.csv)", ", for example 'None'")
    halves = [tmp_path / "b1.csv", tmp_path / "b2.csv"]
    for path, part in zip(halves, (_gaps(_donors(), "NA")[:150], _gaps(_donors())[150:])):
        pd.DataFrame({"x": part}).to_csv(path, index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, batch=[str(p) for p in halves], metrics=["ASW_batch"])
    assert str(e.value) == _bv1(" (b1.csv, b2.csv)", ", for example 'NA'")
    # a dict of one file, the last of the forms that name a file
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb[:150], labels=lab[:150], batch={"batch": halves[0]},
                     metrics=["ASW_batch"])
    assert str(e.value) == (
        "batch (b1.csv): 15 of the 150 cells have no batch, for example 'NA'. Give "
        "every cell a batch.")


def test_the_copy_of_a_batch_column_keeps_the_words_of_the_sheet(tmp_path, monkeypatch):
    """``--batch-column`` hands evaluate a copy of the sheet with one column.
    The copy reads as the column does, and the message names the sheet's word."""
    import contextlib
    emb, lab = _cells()
    donors = _gaps(_donors(), "None")
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"barcode": [f"cell{i}" for i in range(N)], "run": np.arange(N) % 7,
                  "sample": donors}).to_csv(sheet, index=False)
    monkeypatch.setattr(config, "_CLI", True)
    with contextlib.ExitStack() as stack:
        copy = pipeline._one_column_file(sheet, "sample", stack)
        assert pd.read_csv(copy, keep_default_na=False)["x"].tolist() == donors.tolist()
        want, _ = eio.read_labels_ids(sheet, "sample")
        got, first = eio.read_labels_ids(copy)
        assert pd.Series(got).equals(pd.Series(want)) and pd.isna(got).sum() == 30
        assert first.tolist() == [f"cell{i}" for i in range(N)]
        with pytest.raises(ValueError) as e:
            mtb.evaluate(_adata(emb), labels=lab, batch=copy, metrics=["ASW_batch"])
    assert str(e.value) == "--" + _bv1(" (sheet.csv)", ", for example 'None'")


def test_a_complete_batch_changes_nothing(tmp_path):
    emb, lab = _cells()
    bat = _donors()
    emb = emb + np.where(bat == "d1", 0.8, 0.0)[:, None]      # donor d1 sits apart
    a = _adata(emb, celltype=lab, donor=bat)
    path = tmp_path / "batch.csv"
    pd.DataFrame({"x": bat}).to_csv(path, index=False)
    want = _silent(emb, labels=lab, batch=bat, metrics=METRICS)
    for output, batch in ((emb, pd.Series(bat)), (emb, pd.Categorical(bat)),
                          (a, "donor"), (a, "obs:donor"), (emb, str(path))):
        for kw in ({}, {"missing_labels": "drop"}):
            _same(_silent(output, labels=lab, batch=batch, metrics=METRICS, **kw), want)
    assert want["Value"].to_dict() == pytest.approx(SCORES_034, abs=1e-9)
    assert list(want.attrs) == ["leiden_flavor", "clustering", "multibench_version",
                                "scib_version"]
    # batch names that only look like the missing words are batches
    odd = pd.Series(bat).map({"d1": "na", "d2": "none", "d3": "0"}).to_numpy()
    got = mtb.evaluate(emb, labels=lab, batch=odd, metrics=METRICS)
    assert got["Value"].to_dict() == pytest.approx(SCORES_034, abs=1e-9)


def test_docstring_says_every_cell_needs_a_batch():
    doc = " ".join(mtb.evaluate.__doc__.split())
    assert ("needed for the batch metrics unless ``labels`` lists two or more files. "
            "Every cell needs a batch.") in doc
    assert "or a cell without a batch in ``batch``." in doc           # Raises


def test_drop_checks_the_batch_of_the_cells_it_keeps():
    emb, lab = _cells()
    bat = _donors()
    nan = _gaps(lab)
    kw = dict(metrics=["ASW", "ASW_batch"])
    # the cells without a label are the cells without a batch: nothing is left to refuse
    got, said = _drop(emb, labels=nan, batch=_gaps(bat), **kw)
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], batch=bat[KEEP], **kw))
    assert said == ["Scored the 270 labelled cells of 300."]
    # without the flag the labels stop the call first
    with pytest.raises(ValueError, match=r"^labels: 30 of the 300 cells have no label\."):
        mtb.evaluate(emb, labels=nan, batch=_gaps(bat), **kw)
    # four labelled cells have no batch: counted among the 270 that are scored
    labelled = np.flatnonzero(KEEP)[:4]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as e:
            mtb.evaluate(emb, labels=nan, batch=_gaps(bat, None, at=labelled),
                         missing_labels="drop", **kw)
    assert str(e.value) == _bv1(n=4, total=270)
    # nothing was scored, so nothing says "Scored"
    assert [str(w.message) for w in rec] == []


def test_drop_names_the_words_of_the_cells_it_keeps(tmp_path):
    """A batch file that writes NA where the label is missing and None for
    four labelled cells: only the word of the cells that are scored is named."""
    emb, lab = _cells()
    bat = _gaps(_donors(), "NA")
    path = tmp_path / "batch.csv"
    pd.DataFrame({"x": bat}).to_csv(path, index=False)
    kw = dict(labels=_gaps(lab), batch=str(path), metrics=["ASW", "ASW_batch"])
    got, said = _drop(emb, **kw)
    assert len(said) == 1
    _same(got, mtb.evaluate(emb[KEEP], labels=lab[KEEP], batch=_donors()[KEEP],
                            metrics=["ASW", "ASW_batch"]))
    bat[np.flatnonzero(KEEP)[:4]] = "None"
    pd.DataFrame({"x": bat}).to_csv(path, index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, missing_labels="drop", **kw)
    assert str(e.value) == _bv1(" (batch.csv)", ", for example 'None'", n=4, total=270)


def test_the_file_of_origin_is_not_checked(tmp_path, monkeypatch):
    emb, lab = _cells()
    files = [tmp_path / "cty1.csv", tmp_path / "cty2.csv"]
    for path, part in zip(files, (lab[:150], lab[150:])):
        pd.DataFrame({"x": part}).to_csv(path, index=False)
    checked = []
    real = eio.missing_mask

    def spy(values):
        checked.append(np.asarray(values, dtype=object).tolist())
        return real(values)
    monkeypatch.setattr(eio, "missing_mask", spy)
    # the batch is the label file of each cell: evaluate made it, only the labels are checked
    out = mtb.evaluate(emb, labels=[str(p) for p in files], metrics=["ASW", "ASW_batch"])
    assert checked == [lab.tolist()]
    assert out.loc["ASW_batch", "Value"] > 0.9
    # no batch at all: the placeholder is not checked either
    del checked[:]
    mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    assert checked == [lab.tolist()]
    # a batch the caller passes is
    del checked[:]
    mtb.evaluate(emb, labels=lab, batch=_donors(), metrics=["ASW_batch"])
    assert checked == [lab.tolist(), _donors().tolist()]
