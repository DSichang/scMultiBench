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
from tests.test_missing_labels import (GAPS, KEEP, N, _adata, _cells, _donors, _drop,
                                       _gaps, _same, _silent)

pytest.importorskip("scib")

METRICS = ["ASW", "ASW_batch", "GC"]
#: the cells of _cells() with donor d1 moved apart, as the released
#: multibench-sc 0.3.4 scored them
SCORES_034 = {"ASW": 0.7603380351178893, "ASW_batch": 0.9207041314771548, "GC": 1.0}

NAN = "The missing value is written NaN."


def _bv1(src="", seen=NAN, n=30, total=N):
    return (f"batch{src}: {n} of the {total} cells have no batch. {seen} A missing "
            f"batch is not a batch, so the batch metrics are not computed on it. Give "
            f"every cell a batch, or leave these cells out.")


@pytest.mark.parametrize("fill, seen", [
    (np.nan, NAN),
    (None, NAN),
    (pd.NA, NAN),
    ("", "The missing value is written ''."),
    ("nan", "The missing value is written 'nan'."),
    ("NA", "The missing value is written 'NA'."),
], ids=repr)
def test_evaluate_raises_on_a_missing_batch(fill, seen, tmp_path):
    emb, lab = _cells()
    bat = _gaps(_donors(), fill)
    a = _adata(emb, celltype=lab, donor=bat)
    path = tmp_path / "batch.csv"
    pd.DataFrame({"x": bat}).to_csv(path, index=False)
    # a file holds no NaN, None or 'nan': the reader gives NaN for each of them
    forms = {"array": (emb, bat, "", seen), "list": (emb, list(bat), "", seen),
             "Series": (emb, pd.Series(bat), "", seen),
             "obs column": (a, "donor", "", seen), "obs: column": (a, "obs:donor", "", seen),
             "file": (emb, str(path), " (batch.csv)", NAN)}
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
        "batch: 30 of the 300 cells have no batch. The missing value is written NaN. "
        "A missing batch is not a batch, so the batch metrics are not computed on it. "
        "Give every cell a batch, or leave these cells out.")
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
    assert said == ["Scored the 270 labelled cells of 300. 30 cells have no label and "
                    "were left out, so the scores describe the labelled cells only."]
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
