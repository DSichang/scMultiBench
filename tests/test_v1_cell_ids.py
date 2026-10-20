"""v1 NEW-1: evaluate(dataset=, method=) names the folder's cells by the files the
method read, as res.embedding(method) names them, so an embedding in any cell
order scores as in the method's order. NEW-4/NEW-6: ids that are not all
distinct cells of the folder are refused, not matched by position, with the one
fix that works."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, workflow
from multibench.eval import pipeline

ad = pytest.importorskip("anndata")

#: cLISI, GC and ASW_batch read the kNN graph or the distances: the cell order
#: moves them by ties only
CLOSE = ["cLISI", "GC", "ASW_batch"]


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _values(long) -> dict:
    return long.set_index("metric")["value"].to_dict()


def _embedding(ds, category, method):
    """An embedding of ``ds`` in ``method``'s cell order, its cells named as
    ``res.embedding(method)`` names them: one block per cell type, plus noise."""
    root = mtb.config.DEFAULT.data_path
    files = _quiet(pipeline._dataset_labels, ds, category, method, None)
    files = list(files.values()) if isinstance(files, dict) else list(files)
    n = sum(len(pd.read_csv(f)) for f in files)
    obs = _quiet(workflow._embedding_obs, {"method": method, "modalities": None}, ds,
                 category, root, n)
    codes = pd.factorize(obs["label"])[0]
    rng = np.random.default_rng(0)
    x = np.eye(codes.max() + 1)[codes] * 4 + rng.normal(0, 0.5, (n, codes.max() + 1))
    return ad.AnnData(obs=pd.DataFrame(index=obs.index), obsm={"X_emb": x.astype("float32")})


CASES = [("D11", "vertical", "Matilda", ["cLISI", "ARI"]),
         ("D28", "diagonal", "scJoint", CLOSE),
         ("D45", "mosaic", "Cobolt", CLOSE),
         ("D52", "cross", "StabMap", CLOSE)]


@pytest.mark.parametrize("ds,category,method,metrics", CASES)
def test_a_shuffled_embedding_scores_as_in_order(ds, category, method, metrics):
    e = _embedding(ds, category, method)
    kw = dict(dataset=ds, category=category, method=method, metrics=metrics, verbose=False)
    want = _values(_quiet(mtb.evaluate, e, **kw))
    perm = np.random.default_rng(1).permutation(e.n_obs)
    got = _values(_quiet(mtb.evaluate, e[perm].copy(), **kw))
    for m in metrics:
        # Leiden clusters depend on the cell order (evaluate.md); on this clean
        # embedding of D11 they move by little
        tol = 0.05 if m == "ARI" else 0.005
        assert got[m] == pytest.approx(want[m], abs=tol), (m, got, want)
    if ds == "D28":
        # scJoint read atac_gas.h5: its cells repeat the RNA barcodes, so each is
        # named by its label file
        assert e.obs_names[0].startswith("rna_cty:")


def test_the_stored_d28_scjoint_embedding_scores_shuffled_as_in_order():
    """The verification's case: cLISI 0.82 / GC 0.18 shuffled against 0.99 / 0.87."""
    stored = Path(mtb.config.DEFAULT.data_path) / "outputs" / "D28"
    repo = Path(__file__).resolve().parent.parent / "data" / "outputs" / "D28"
    folder = stored if (stored / "batch_result.json").is_file() else repo
    if not (folder / "batch_result.json").is_file():
        pytest.skip("no stored D28 outputs")
    e = _quiet(mtb.load_batch, folder).embedding("scJoint")
    kw = dict(dataset="D28", category="diagonal", method="scJoint",
              metrics=["cLISI", "GC"], verbose=False)
    want = _values(_quiet(mtb.evaluate, e, **kw))
    perm = np.random.default_rng(0).permutation(e.n_obs)
    got = _values(_quiet(mtb.evaluate, e[perm].copy(), **kw))
    assert got == pytest.approx(want, abs=0.005)


def test_your_own_diagonal_embedding_with_the_atac_cells_first():
    """No method: the folder's default files name the cells (atac_peak.h5 on
    D28); the ATAC cells may come first."""
    files = _quiet(pipeline._dataset_labels, "D28", "diagonal", None, None)
    files = list(files.values()) if isinstance(files, dict) else list(files)
    assert [Path(f).name for f in files] == ["rna_cty.csv", "atac_cty.csv"]
    keys = pipeline._file_barcodes([Path(f) for f in files])
    labels = np.concatenate([pd.read_csv(f).iloc[:, 0].to_numpy() for f in files])
    n_rna = len(pd.read_csv(files[0]))
    codes = pd.factorize(labels)[0]
    x = np.eye(codes.max() + 1)[codes] * 4 + \
        np.random.default_rng(0).normal(0, 0.5, (len(codes), codes.max() + 1))
    e = ad.AnnData(obs=pd.DataFrame(index=keys.to_numpy()), obsm={"X_emb": x.astype("float32")})
    kw = dict(dataset="D28", category="diagonal", metrics=CLOSE, verbose=False)
    want = _values(_quiet(mtb.evaluate, e, **kw))
    atac_first = np.r_[np.arange(n_rna, e.n_obs), np.arange(n_rna)]
    got = _values(_quiet(mtb.evaluate, e[atac_first].copy(), **kw))
    assert got == pytest.approx(want, abs=0.005)
    # your own method's name reads the same files
    own = _values(_quiet(mtb.evaluate, e[atac_first].copy(), method="mine", **kw))
    assert own == pytest.approx(want, abs=0.005)


def test_a_package_methods_cell_names_under_another_name_name_the_method():
    """v1 NEW-6: scJoint's names (atac_gas.h5) are none of the folder's default
    cells; scored as your own method, the error names the methods whose output
    is named so."""
    e = _embedding("D28", "diagonal", "scJoint")
    for kw in ({}, {"method": "mine"}):
        with pytest.raises(ValueError) as err:
            _quiet(mtb.evaluate, e, dataset="D28", category="diagonal", metrics=["cLISI"],
                   verbose=False, **kw)
        msg = str(err.value)
        assert msg.startswith(
            f"None of the embedding's cell ids is a cell of rna_cty.csv and atac_cty.csv, "
            f"such as {e.obs_names[0]!r}. These cells are named as the output of "), msg
        m = msg.split("such as method=", 1)[1].split(" ", 1)[0].strip("'")
        assert msg.endswith(f"pass method= the method that made it, such as "
                            f"method={m!r} (or '{m} (this run)').")
    # the method named works, and so does the method that made it, renamed
    want = _values(_quiet(mtb.evaluate, e, dataset="D28", category="diagonal",
                          method="scJoint", metrics=["cLISI"], verbose=False))
    for name in (m, "scJoint (this run)"):
        got = _values(_quiet(mtb.evaluate, e, dataset="D28", category="diagonal",
                             method=name, metrics=["cLISI"], verbose=False))
        assert got == pytest.approx(want, abs=1e-9)


# ------------------------------------------------- v1 NEW-4: ids not all distinct cells
def _d28_raw():
    """D28's labels in the folder's order (RNA, then ATAC), an embedding of them,
    and the raw barcodes of rna.h5 and atac_gas.h5 (43 of them on both sides)."""
    from multibench.engine import resolve
    folder = Path(mtb.config.DEFAULT.data_path) / "D28"
    labels = np.concatenate([pd.read_csv(folder / f).iloc[:, 0].to_numpy()
                             for f in ("rna_cty.csv", "atac_cty.csv")])
    rna = list(resolve._barcodes_of(folder / "rna.h5"))
    gas = list(resolve._barcodes_of(folder / "atac_gas.h5"))
    assert len(rna) + len(gas) == len(labels) and len(set(rna) & set(gas)) == 43
    codes = pd.factorize(labels)[0]
    x = np.eye(codes.max() + 1)[codes] * 4 + \
        np.random.default_rng(0).normal(0, 0.5, (len(codes), codes.max() + 1))
    return x.astype("float32"), rna, gas


def _named(x, names, order):
    return ad.AnnData(obs=pd.DataFrame(index=[names[i] for i in order]),
                      obsm={"X_emb": x[order]})


KW28 = dict(dataset="D28", category="diagonal", method="mine", metrics=["cLISI"],
            verbose=False)


def test_repeated_or_partly_matching_ids_are_refused_not_matched_by_position():
    x, rna, gas = _d28_raw()
    n = len(rna)
    atac_first = np.r_[np.arange(n, len(x)), np.arange(n)]
    keys = pipeline._file_barcodes([Path(mtb.config.DEFAULT.data_path) / "D28" / f
                                    for f in ("rna_cty.csv", "atac_cty.csv")])
    fix = f"Name the embedding's cells as the folder does, such as {keys.iloc[0]!r}."
    with pytest.raises(ValueError) as err:          # was cLISI 0.8212, not 0.9904
        _quiet(mtb.evaluate, _named(x, rna + gas, atac_first), **KW28)
    assert str(err.value).startswith("The embedding repeats cell ids, such as ")
    assert str(err.value).endswith(fix)
    rna_side = rna + ["atac_" + b for b in gas]     # unique, the RNA side only
    with pytest.raises(ValueError) as err:
        _quiet(mtb.evaluate, _named(x, rna_side, atac_first), **KW28)
    assert str(err.value) == (
        f"4,606 of the embedding's 11,014 cell ids are not cells of rna_cty.csv and "
        f"atac_cty.csv, such as 'atac_{gas[0]}'. {fix}")


def test_cells_in_the_folders_order_are_told_to_pass_an_array():
    x, rna, gas = _d28_raw()
    with pytest.raises(ValueError) as err:
        _quiet(mtb.evaluate, _named(x, rna + gas, np.arange(len(x))), **KW28)
    assert str(err.value).endswith(
        "The cells are in the folder's order: pass the embedding as an array, as "
        "np.asarray(adata.obsm[key]) or adata.X, to match them by position.")
    keys = pipeline._file_barcodes([Path(mtb.config.DEFAULT.data_path) / "D28" / f
                                    for f in ("rna_cty.csv", "atac_cty.csv")])
    by_id = _values(_quiet(mtb.evaluate, _named(x, list(keys), np.arange(len(x))), **KW28))
    as_array = _values(_quiet(mtb.evaluate, x, **KW28))
    assert as_array == pytest.approx(by_id, abs=1e-9) and by_id["cLISI"] > 0.95


def test_the_command_line_refuses_repeated_ids_and_takes_the_npy(tmp_path, capsys):
    x, rna, gas = _d28_raw()
    _named(x, rna + gas, np.arange(len(x))).write_h5ad(tmp_path / "raw.h5ad")
    args = ["--dataset", "D28", "--category", "diagonal", "--method", "mine",
            "--metrics", "cLISI"]
    assert _quiet(cli.main, ["evaluate", str(tmp_path / "raw.h5ad"), *args]) == 1
    assert capsys.readouterr().err.splitlines()[-1].endswith(
        "The cells are in the folder's order: save the embedding as an .npy file to "
        "match them by position.")
    np.save(tmp_path / "raw.npy", x)
    assert _quiet(cli.main, ["evaluate", str(tmp_path / "raw.npy"), *args,
                             "--out", str(tmp_path / "o.csv")]) == 0
    assert _values(pd.read_csv(tmp_path / "o.csv"))["cLISI"] > 0.95


def test_own_names_are_told_the_folders_names():
    x, rna, gas = _d28_raw()
    with pytest.raises(ValueError, match=r"^None of the embedding's cell ids is a cell of "
                                         r"rna_cty\.csv and atac_cty\.csv, such as 'cell0'\. "
                                         r"Name the embedding's cells as the folder does"):
        _quiet(mtb.evaluate, _named(x, [f"cell{i}" for i in range(len(x))],
                                    np.arange(len(x))), **KW28)


def test_the_command_line_scores_a_shuffled_embedding_as_in_order(tmp_path):
    e = _embedding("D28", "diagonal", "scJoint")
    perm = np.random.default_rng(2).permutation(e.n_obs)
    e.write_h5ad(tmp_path / "in_order.h5ad")
    e[perm].copy().write_h5ad(tmp_path / "shuffled.h5ad")
    got = {}
    for name in ("in_order", "shuffled"):
        rc = _quiet(cli.main, ["evaluate", str(tmp_path / f"{name}.h5ad"), "--dataset", "D28",
                               "--category", "diagonal", "--method", "scJoint", "--metrics",
                               "cLISI,GC", "--out", str(tmp_path / f"{name}.csv")])
        assert rc == 0
        got[name] = _values(pd.read_csv(tmp_path / f"{name}.csv"))
    assert got["shuffled"] == pytest.approx(got["in_order"], abs=0.005)
