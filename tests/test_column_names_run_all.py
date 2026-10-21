"""run and rescore say why they cannot take the name of an obs column (0.3.5).

``export_dataset`` and ``evaluate`` take ``"obs:donor"`` and ``"donor"``,
because they are given the object. ``run`` and ``rescore`` are given a
folder. Up to 0.3.4 ``batch="donor"`` answered ``The batch file donor does
not exist``; the message now says what to pass. A name that looks like a
file keeps the file error.

A synthetic CITE-seq folder and a fake ``_run`` keep these independent of any
conda env.
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
    every environment, with the working directory at ``<tmp>``."""
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
    monkeypatch.chdir(tmp_path)
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


def _c1(what, given, col):
    return (f"{what}={given!r} is not a file. Pass the column itself, as "
            f"{what}=adata.obs[{col!r}].")


@pytest.mark.parametrize("given", ["donor", "obs:donor"])
def test_a_column_name_is_explained(cite, tmp_path, given):
    data, calls = cite
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['totalVI'], verbose=False)
    with pytest.raises(FileNotFoundError) as e:
        mtb.run("MYCITE", "vertical", dry_run=True, batch=given, **kw)
    assert str(e.value) == _c1("batch", given, "donor")
    with pytest.raises(FileNotFoundError) as e:
        mtb.run("MYCITE", "vertical", tmp_path / "out", batch=given, **kw)
    assert str(e.value) == _c1("batch", given, "donor")
    assert calls == [] and not (tmp_path / "out").exists()
    res = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "out", **kw)
    with pytest.raises(FileNotFoundError) as e:
        res.rescore(batch=given)
    assert str(e.value) == _c1("batch", given, "donor")
    with pytest.raises(FileNotFoundError) as e:
        res.rescore(labels=given.replace("donor", "celltype"))
    assert str(e.value) == _c1("labels", given.replace("donor", "celltype"), "celltype")
    assert [r["status"] for r in res.records] == ["CHAIN_OK"]


def test_other_names_that_are_no_file(cite, tmp_path):
    data, _ = cite
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['totalVI'], verbose=False, dry_run=True)
    for given, col in (("sample id", "sample id"), ("rna:donor", "rna:donor"),
                       ("donor.h5ad", "donor.h5ad"), ("obs:", "")):
        with pytest.raises(FileNotFoundError) as e:
            mtb.run("MYCITE", "vertical", batch=given, **kw)
        assert str(e.value) == _c1("batch", given, col)


def test_a_missing_csv_keeps_its_text(cite, tmp_path):
    data, calls = cite
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['totalVI'], verbose=False)
    cwd = Path.cwd()

    def gone(name, what="batch"):
        return (f"The {what} file {Path(name)} does not exist. From the working directory "
                f"{cwd}, the path resolves to {Path(name).resolve()}.")
    for name in ("missing.csv", "donor.tsv", "x.TXT", "sub/donor", "./donor"):
        with pytest.raises(FileNotFoundError) as e:
            mtb.run("MYCITE", "vertical", dry_run=True, batch=name, **kw)
        assert str(e.value) == gone(name), name
        with pytest.raises(FileNotFoundError) as e:
            mtb.run("MYCITE", "vertical", tmp_path / "out", batch=name, **kw)
        assert str(e.value) == gone(name), name
    # a Path is a path, whatever its name
    with pytest.raises(FileNotFoundError) as e:
        mtb.run("MYCITE", "vertical", dry_run=True, batch=Path("donor"), **kw)
    assert str(e.value) == gone("donor")
    res = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "out", **kw)
    with pytest.raises(FileNotFoundError) as e:
        res.rescore(labels="missing.csv")
    assert str(e.value) == gone("missing.csv", "labels")
    with pytest.raises(FileNotFoundError) as e:
        res.rescore(batch="missing.csv")
    assert str(e.value) == gone("missing.csv")
    assert calls == ["totalVI"]


def test_a_file_named_like_a_column_is_read(cite, tmp_path):
    data, _ = cite
    donor = pd.Series(["d1", "d2"] * (N // 2), index=BARS, name="donor")
    donor.to_frame().to_csv(tmp_path / "donor")        # no suffix: it is there, so a file
    plan = mtb.run(os.path.join(data, 'MYCITE'), 'vertical', methods=['totalVI'], verbose=False, dry_run=True, batch='donor')
    assert len(plan) == 1


def test_the_column_itself_works(cite, tmp_path):
    data, calls = cite
    donor = pd.Series(["d1", "d2"] * (N // 2), index=BARS, name="donor")
    obs = pd.DataFrame({"donor": donor, "celltype": ["A"] * 60 + ["B"] * 60}, index=BARS)
    mtb.config.DEFAULT.data_path = data
    kw = dict(methods=['totalVI'], verbose=False)
    # in another row order: matched to the cells by barcode
    shuffled = obs.sample(frac=1, random_state=3)
    plan = mtb.run("MYCITE", "vertical", dry_run=True, batch=shuffled["donor"], **kw)
    assert plan["runnable"].all()
    res = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "out",
                 batch=shuffled["donor"], **kw)
    (rec,) = res.records
    assert rec["status"] == "CHAIN_OK" and rec["batch_source"] == "user"
    assert rec["n_batches"] == 2
    straight = _quiet(mtb.run, "MYCITE", "vertical", tmp_path / "out2",
                      batch=obs["donor"], **kw).records[0]
    assert straight["metrics"] == rec["metrics"]
    again = res.rescore(batch=shuffled["donor"], labels=shuffled["celltype"])
    assert again.records[0]["status"] == "CHAIN_OK"
    assert again.records[0]["metrics"]["ARI"] == rec["metrics"]["ARI"]
    assert calls == ["totalVI", "totalVI"]


def test_the_docstrings_state_the_rule():
    flat = " ".join(W.BatchResult.rescore.__doc__.split())
    assert 'Pass the column itself: ``batch=adata.obs["donor"]``.' in flat
