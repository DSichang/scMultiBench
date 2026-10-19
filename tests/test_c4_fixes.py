"""Round c4 fixes of the 0.4.0 core API."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench.eval import pipeline


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _value(long, metric="ASW"):
    return float(long.set_index("metric").loc[metric, "value"])


# ------------------------------------------------- u2 #1: a package method's order
def _d52mini():
    """StabMap's labels of D52mini (cty3, cty1, cty2), the folder's (cty1, cty2,
    cty3), the folder's cell ids and an embedding in StabMap's order."""
    from multibench.engine import resolve
    st = _quiet(resolve.labels_for, "D52mini", "cross", "StabMap")
    fo = _quiet(resolve.labels_for, "D52mini", "cross", None)
    assert list(st) == ["cty3", "cty1", "cty2"] and list(fo) == ["cty1", "cty2", "cty3"]

    def stack(d):
        return np.concatenate([pd.read_csv(p).iloc[:, 0].to_numpy() for p in d.values()])
    lab = stack(st)
    codes = pd.factorize(lab)[0]
    emb = np.eye(codes.max() + 1)[codes] * 3 + \
        np.random.default_rng(0).normal(0, 1.0, (len(codes), codes.max() + 1))
    keys = pipeline._file_barcodes([Path(p) for p in fo.values()])
    return st, fo, stack(fo), keys, emb


KW = dict(dataset="D52mini", category="cross", method="StabMap", metrics=["ASW"],
          verbose=False)


def test_labels_in_the_folder_order_are_refused_for_a_method_with_its_own_order():
    st, fo, lab_folder, keys, emb = _d52mini()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab_folder, **KW)
    assert str(e.value) == (
        "labels= is matched to the cells by position, and StabMap puts the cells of "
        "D52mini in the order cty3, cty1, cty2. Leave labels= out to read the label "
        "files in that order.")
    # the folder's files in the folder's order are refused the same way
    with pytest.raises(ValueError, match="^labels= is matched"):
        mtb.evaluate(emb, labels=[str(p) for p in fo.values()], **KW)


def test_labels_with_ids_or_in_the_method_order_score_as_the_dataset_route():
    st, fo, lab_folder, keys, emb = _d52mini()
    want = _value(mtb.evaluate(emb, **KW))
    by_id = pd.Series(lab_folder, index=keys.to_numpy())
    assert _value(mtb.evaluate(emb, labels=by_id, **KW)) == pytest.approx(want)
    files = [str(p) for p in st.values()]
    assert _value(mtb.evaluate(emb, labels=files, **KW)) == pytest.approx(want)
    # your own method reads the folder's order: nothing to refuse
    mtb.evaluate(emb, labels=lab_folder, **{**KW, "method": "PCA"})


def test_a_batch_with_ids_follows_the_method_order():
    st, fo, lab_folder, keys, emb = _d52mini()
    sizes = [len(pd.read_csv(p)) for p in fo.values()]
    folder_batch = np.repeat(["b1", "b2", "b3"], sizes)
    by_id = pd.Series(folder_batch, index=keys.to_numpy())
    in_stabmap_order = np.concatenate([folder_batch[sum(sizes[:2]):],
                                       folder_batch[:sum(sizes[:2])]])
    kw = {**KW, "metrics": ["iLISI"]}
    got = _value(mtb.evaluate(emb, batch=by_id, **kw), "iLISI")
    want = _value(mtb.evaluate(emb, batch=in_stabmap_order, **kw), "iLISI")
    assert got == pytest.approx(want)


def test_the_command_line_refuses_labels_in_another_order(tmp_path, capsys):
    st, fo, lab_folder, keys, emb = _d52mini()
    np.save(tmp_path / "e.npy", emb)
    pd.DataFrame({"x": lab_folder}).to_csv(tmp_path / "all.csv", index=False)
    rc = cli.main(["evaluate", str(tmp_path / "e.npy"), "--labels", str(tmp_path / "all.csv"),
                   "--dataset", "D52mini", "--category", "cross", "--method", "StabMap",
                   "--metrics", "ASW"])
    assert rc == 1
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: --labels is matched to the cells by position, and StabMap puts the cells "
        "of D52mini in the order cty3, cty1, cty2. Leave out --labels to read the label "
        "files in that order.")


# --------------------------------------- u3 #1, u1 #1, u2 #2-#4: a subset of the cells
@pytest.fixture
def folder(tmp_path):
    """MINE: 60 cells with barcodes, written by export_dataset; the AnnData."""
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(1)
    a = ad.AnnData(rng.poisson(2, (60, 20)).astype(np.float32))
    a.obs_names = [f"BC{i:03d}-1" for i in range(60)]
    a.var_names = [f"g{i}" for i in range(20)]
    a.obs["celltype"] = np.repeat(["A", "B", "C"], 20)
    a.obsm["X_emb"] = np.eye(3)[np.repeat([0, 1, 2], 20)] * 3 + rng.normal(0, 1, (60, 3))
    _quiet(mtb.io.export_dataset, a, tmp_path / "MINE", rna="X", labels="obs:celltype",
           category="vertical")
    return tmp_path / "MINE", a


def test_a_subset_of_a_folders_cells_is_matched_by_id(folder):
    path, a = folder
    sub = a[np.random.default_rng(2).permutation(60)[:45]].copy()
    want = _value(mtb.evaluate(sub, labels="celltype", metrics=["ASW"], verbose=False))
    got = mtb.evaluate(sub, labels=str(path / "cty.csv"), metrics=["ASW"], verbose=False)
    assert _value(got) == pytest.approx(want)
    got = mtb.evaluate(sub, dataset=str(path), category="vertical", method="mine",
                       metrics=["ASW"], verbose=False)
    assert _value(got) == pytest.approx(want)


def test_a_label_file_without_ids_for_a_subset_names_one_fix(tmp_path, folder):
    path, a = folder
    sub = a[:45].copy()
    pd.DataFrame({"x": np.repeat(["A", "B", "C"], 20)}).to_csv(tmp_path / "own.csv",
                                                               index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(sub, labels=str(tmp_path / "own.csv"), metrics=["ASW"])
    assert str(e.value) == ("own.csv has no cell ids, and the embedding has: put the "
                            "barcodes in its first column.")


def test_your_own_cells_on_a_benchmark_name_are_told_to_take_a_name_of_their_own():
    emb = np.random.default_rng(0).normal(size=(1718, 4))
    lab = np.repeat(["a", "b"], 859)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=lab, dataset="D11", method="PCA", metrics=["ASW"])
    assert str(e.value) == (
        "The label files of D11 (cty.csv) hold 2,864 cells; the embedding has 1,718 rows. "
        "To score other cells, pass their labels and a dataset name of your own, such as "
        "dataset='D11_subset'.")
    got = mtb.evaluate(emb, labels=lab, dataset="D11_subset", category="vertical",
                       method="PCA", metrics=["ASW"], verbose=False)
    assert set(got["dataset"]) == {"D11_subset"}
    # a package method's output keeps the check of the embedding
    with pytest.raises(ValueError, match="Check that the embedding is the one of method= "
                                         "on D11.$"):
        mtb.evaluate(emb, dataset="D11", method="Matilda", metrics=["ASW"])


def test_the_command_line_checks_a_benchmark_dataset_as_python(tmp_path, capsys):
    """u2 #2, #3: the CLI refuses the subset as Python does and fills the category."""
    rng = np.random.default_rng(0)
    np.save(tmp_path / "sub.npy", rng.normal(size=(1718, 4)))
    pd.DataFrame({"x": np.repeat(["a", "b"], 859)}).to_csv(tmp_path / "l.csv", index=False)
    rc = cli.main(["evaluate", str(tmp_path / "sub.npy"), "--labels", str(tmp_path / "l.csv"),
                   "--method", "PCA", "--dataset", "D11", "--metrics", "ASW"])
    assert rc == 1
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: the label files of D11 (cty.csv) hold 2,864 labels for 1,718 cells in the "
        "embedding file. To score other cells, pass their labels with --labels and a "
        "dataset name of your own, such as --dataset D11_subset.")
    np.save(tmp_path / "full.npy", rng.normal(size=(2864, 4)))
    rc = cli.main(["evaluate", str(tmp_path / "full.npy"), "--dataset", "D11", "--method",
                   "PCA", "--metrics", "ASW", "--out", str(tmp_path / "o.csv")])
    assert rc == 0
    assert set(pd.read_csv(tmp_path / "o.csv")["category"]) == {"vertical"}


def test_a_barcoded_batch_next_to_an_npy_follows_the_folder(tmp_path, folder):
    """u3 #5: the folder's barcodes give the cells of an embedding without ids."""
    path, a = folder
    np.save(tmp_path / "e.npy", a.obsm["X_emb"])
    batch = pd.Series(np.tile(["d1", "d2"], 30), index=a.obs_names)
    batch.iloc[::-1].rename("donor").to_frame().to_csv(tmp_path / "donor.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # no "matched by position" warning
        got = mtb.evaluate(str(tmp_path / "e.npy"), dataset=str(path), category="vertical",
                           method="PCA", batch=str(tmp_path / "donor.csv"),
                           metrics=["iLISI"], verbose=False)
    want = mtb.evaluate(str(tmp_path / "e.npy"), dataset=str(path), category="vertical",
                        method="PCA", batch=batch.to_numpy(), metrics=["iLISI"],
                        verbose=False)
    assert _value(got, "iLISI") == pytest.approx(_value(want, "iLISI"))


def test_a_missing_embedding_file_is_named(tmp_path, capsys):
    rc = cli.main(["evaluate", str(tmp_path / "nothere.h5ad"), "--labels", "celltype"])
    assert rc == 1
    assert capsys.readouterr().err.startswith(
        f"error: The embedding {tmp_path / 'nothere.h5ad'} does not exist.")


def test_a_printed_table_leaves_the_category_empty(tmp_path, capsys):
    np.save(tmp_path / "e.npy", np.random.default_rng(0).normal(size=(40, 3)))
    pd.DataFrame({"x": np.repeat(["a", "b"], 20)}).to_csv(tmp_path / "l.csv", index=False)
    assert cli.main(["evaluate", str(tmp_path / "e.npy"), "--labels",
                     str(tmp_path / "l.csv"), "--metrics", "ASW"]) == 0
    assert "None" not in capsys.readouterr().out
