"""run and rescore do not score a missing batch (0.3.5).

``batch=adata.obs["donor"]`` with a donor missing for some cells counted the
missing value as one more batch up to 0.3.4 (iLISI of the same embedding:
0.8076 with every donor, 0.7541 with 90 of them missing, no warning). The
vector is now checked where it is aligned to the cells, so a dry run stops
too.

A synthetic CITE-seq folder and a fake ``_run`` that writes ``embedding.h5``
keep these independent of any conda env.
"""
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
N = 120
BARS = [f"c{i}" for i in range(N)]


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _h5(path, feats):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), N)).astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array(BARS, dtype="S32"))


class _Res:
    def __init__(self, out):
        self.output = out


@pytest.fixture
def cite(tmp_path, monkeypatch):
    """``(data root, calls)``: MYCITE (RNA + ADT, N cells) on a Linux host with
    every environment; the fake ``_run`` records the methods it ran."""
    d = tmp_path / "data" / "MYCITE"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)])
    _h5(d / "adt.h5", [f"p{i}" for i in range(6)])
    pd.DataFrame({"x": ["A"] * (N // 2) + ["B"] * (N // 2)}).to_csv(d / "cty.csv",
                                                                   index=False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(method)
        emb = np.random.default_rng(7).normal(size=(N, 4))
        emb[N // 2:, 0] += 8.0
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    return tmp_path / "data", calls


def _donor(gap=None, n_gaps=5):
    """One donor per cell, by barcode; the first ``n_gaps`` are ``gap``."""
    s = pd.Series(["d1", "d2"] * (N // 2), index=BARS, dtype=object, name="donor")
    if n_gaps:
        s.iloc[:n_gaps] = gap
    return s


def _bv2(n, example, head="batch"):
    return (f"{head}: {n} of the 120 cells of MYCITE have no batch{example}. Give every "
            "cell a batch.")


#: NaN and a blank are not named; a text that reads as missing is the example
NAN = ""
FORMS = [(np.nan, NAN), (None, NAN), (pd.NA, NAN), ("", ""), (" ", ""),
         ("nan", ", for example 'nan'"),
         ("NA", ", for example 'NA'"),
         ("None", ", for example 'None'")]


@pytest.mark.parametrize("gap,seen", FORMS, ids=[repr(g) for g, _ in FORMS])
def test_run_all_stops_on_a_missing_batch(cite, tmp_path, gap, seen):
    data, calls = cite
    series = _donor(gap)
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['totalVI'], verbose=False)
    for batch in (series, series.sample(frac=1, random_state=0), series.to_numpy(),
                  list(series), series.to_frame()):
        with pytest.raises(ValueError) as e:
            _quiet(mtb.run, "MYCITE", "vertical", dry_run=True, batch=batch, **kw)
        assert str(e.value) == _bv2(5, seen)
        with pytest.raises(ValueError) as e:
            _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "out", batch=batch, **kw)
        assert str(e.value) == _bv2(5, seen)
    assert calls == [] and not (tmp_path / "out").exists()


def test_a_float_batch_with_a_gap(cite, tmp_path):
    data, calls = cite
    batch = np.tile([1.0, 2.0], N // 2)
    batch[[3, 4, 50]] = np.nan
    with pytest.raises(ValueError) as e:
        mtb.run(os.path.join(data, 'MYCITE'), 'vertical', tmp_path / 'out', methods=['totalVI'], verbose=False, batch=batch)
    assert str(e.value) == _bv2(3, NAN)
    assert calls == []


def test_a_batch_file_names_the_word_it_holds(cite, tmp_path):
    data, calls = cite
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['totalVI'], verbose=False, dry_run=True)
    sheet = tmp_path / "donor.csv"
    _donor("NA").to_frame().to_csv(sheet)                 # barcodes, then the donor
    for batch in (sheet, str(sheet)):
        with pytest.raises(ValueError) as e:
            mtb.run("MYCITE", "vertical", batch=batch, **kw)
        assert str(e.value) == _bv2(5, ", for example 'NA'")
    _donor(np.nan).to_frame().to_csv(sheet)               # empty fields
    with pytest.raises(ValueError) as e:
        mtb.run("MYCITE", "vertical", batch=sheet, **kw)
    assert str(e.value) == _bv2(5, NAN)
    pd.DataFrame({"x": _donor("None", 2).to_numpy()}).to_csv(sheet, index=False)
    with pytest.raises(ValueError) as e:
        mtb.run("MYCITE", "vertical", batch=sheet, **kw)
    assert str(e.value) == _bv2(2, ", for example 'None'")
    assert calls == []


def test_rescore_stops_on_a_missing_batch(cite, tmp_path):
    data, calls = cite
    mtb.config.DEFAULT.data_path = data
    res = _quiet(mtb.run, 'MYCITE', 'vertical', tmp_path / 'out', methods=['totalVI'], verbose=False)
    assert [r["status"] for r in res.records] == ["CHAIN_OK"]
    for gap, seen in ((np.nan, NAN), ("nan", ", for example 'nan'")):
        series = _donor(gap)
        for batch in (series, series.to_numpy()):
            with pytest.raises(ValueError) as e:
                res.rescore(batch=batch)
            assert str(e.value) == _bv2(5, seen)
    sheet = tmp_path / "donor.csv"
    _donor("NA").to_frame().to_csv(sheet)
    with pytest.raises(ValueError) as e:
        res.rescore(batch=sheet)
    assert str(e.value) == _bv2(5, ", for example 'NA'")
    # with labels in the rows of the embedding as well
    with pytest.raises(ValueError, match="5 of the 120 cells of MYCITE have no batch"):
        res.rescore(labels=np.array(["A"] * 60 + ["B"] * 60), batch=_donor(np.nan))
    assert [r["status"] for r in res.records] == ["CHAIN_OK"] and calls == ["totalVI"]


def test_a_complete_batch_runs(cite, tmp_path):
    data, calls = cite
    series = _donor(n_gaps=0)
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['totalVI'], verbose=False)
    plan = mtb.run("MYCITE", "vertical", dry_run=True, batch=series, **kw)
    assert len(plan) == 1 and plan["runnable"].all()
    res = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "out", batch=series, **kw)
    (rec,) = res.records
    assert rec["status"] == "CHAIN_OK" and rec["batch_source"] == "user"
    assert rec["n_batches"] == 2 and rec["metrics"]["ASW_batch"] is not None
    # names that only look like missing values are batches
    odd = pd.Series(["na", "none"] * (N // 2), index=BARS)
    assert res.rescore(batch=odd).records[0]["n_batches"] == 2
    assert res.rescore(batch=np.tile([0, 1], N // 2)).records[0]["n_batches"] == 2
    # the same scores as 0.3.4 gave for this call: the check reads, it does not change
    again = res.rescore(batch=series.sample(frac=1, random_state=1))
    assert again.records[0]["metrics"] == rec["metrics"]


def test_the_check_names_nothing_for_a_complete_vector():
    assert W._check_batch(np.array(["d1", "d2"]), "X") is None
    assert W._has_gap([1, 2, 3]) is False and W._has_gap(["a", None]) is True
    assert W._has_gap(pd.Series(["a", "NA "])) is True and W._has_gap(["na", "0"]) is False
