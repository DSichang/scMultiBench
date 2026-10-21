"""Round c4 fixes of the 0.4.0 core API."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
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
    # an obs column follows the embedding's own rows
    ad = pytest.importorskip("anndata")
    lab_st = np.concatenate([pd.read_csv(p).iloc[:, 0].to_numpy() for p in st.values()])
    a = ad.AnnData(obs=pd.DataFrame({"ct": lab_st}, index=[f"c{i}" for i in range(len(emb))]))
    a.obsm["X_emb"] = emb
    assert _value(mtb.evaluate(a, labels="ct", **KW)) == pytest.approx(want)


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
        "dataset='D11_subset', and plot them apart from the stored results.")
    got = mtb.evaluate(emb, labels=lab, dataset="D11_subset", category="vertical",
                       method="PCA", metrics=["ASW"], verbose=False)
    assert set(got["dataset"]) == {"D11_subset"}
    # a package method's output keeps the check of the embedding
    with pytest.raises(ValueError, match="Check that the embedding is the one of method= "
                                         "on D11.$"):
        mtb.evaluate(emb, dataset="D11", method="Matilda", metrics=["ASW"])


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


# ------------------------------------------- u4 #1, u5 #8: the stored rows; u5 #2: snap
from multibench.plot.style import overall  # noqa: E402


def test_a_pandas_copy_of_stored_rows_is_your_row():
    st = mtb.load_results("diagonal")
    copy = st[st["method"] == "scJoint"].assign(method="scJoint (copy)")   # source "stored"
    s, base = overall(pd.concat([st, copy])), overall(st)
    assert np.allclose(s[base.index], base, atol=1e-12)
    assert s["scJoint (copy)"] == pytest.approx(s["scJoint"])


def test_stored_rows_under_a_custom_result_path_stay_stored(tmp_path, monkeypatch):
    rows = [dict(metric=m, value=v, method=meth, dataset="DX", category="vertical",
                 source="stored")
            for meth, v in (("Matilda", 0.9), ("scMM", 0.5), ("sciPENN", 0.2))
            for m in ("ARI", "NMI")]
    (tmp_path / "rerun").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "rerun" / "long_all_x.csv", index=False)
    monkeypatch.setattr(config.DEFAULT, "result_path", tmp_path)
    st = _quiet(mtb.load_results, "vertical")
    assert set(st["dataset"]) == {"DX"}
    mine = st[st["method"] == "scMM"].assign(method="mine", source="user", value=0.7)
    s = overall(pd.concat([st, mine]))
    assert s.to_dict() == pytest.approx({"Matilda": 1.0, "mine": 0.75, "scMM": 0.5,
                                         "sciPENN": 0.0})


def test_noise_snaps_to_the_stored_value_of_the_same_method_only():
    st = mtb.load_results("mosaic")
    base = overall(st)
    for shift in (5e-7, -5e-7):
        near = st[st["method"] == "Cobolt"].assign(
            method="Cobolt (this run)", source="user", value=lambda d: d["value"] + shift)
        s = overall(pd.concat([st, near]))
        assert np.allclose(s[base.index], base, atol=1e-12)
        assert s["Cobolt (this run)"] == pytest.approx(s["Cobolt"])
    # your own method 5e-7 above the best stored value is above it: no snap
    other = st[st["method"] == "MultiVI"].assign(method="mine", source="user",
                                                 value=lambda d: d["value"] + 5e-7)
    s = overall(pd.concat([st, other]), ["NMI_batch"])
    assert s["mine"] > s["MultiVI"]


# ------------------------------------------- minimal #4: scan's per-method "reads" lines
def _diag(tmp_path, batches):
    import anndata as ad
    rng = np.random.default_rng(0)
    d = tmp_path / f"DIAG{batches}"
    for b in range(1, batches + 1):
        rna = ad.AnnData(rng.poisson(1, (30, 20)).astype(float))
        rna.obs_names = [f"r{b}_{i}" for i in range(30)]
        rna.var_names = [f"G{i}" for i in range(20)]
        rna.obs["celltype"] = ["T", "B"] * 15
        atac = rna.copy()
        atac.obs_names = [f"a{b}_{i}" for i in range(30)]
        _quiet(mtb.io.export_dataset, rna, d, atac=atac, atac_kind="gene_activity",
               labels="obs:celltype", category="diagonal", batch_index=b)
    return d


@pytest.mark.parametrize("batches", [2, 3])
def test_scan_prints_no_reads_line_for_a_folder_of_numbered_batches(tmp_path, capsys,
                                                                     batches):
    d = _diag(tmp_path, batches)
    capsys.readouterr()
    reason = mtb.scan(d, "diagonal").set_index("method")["reason"]
    lines = capsys.readouterr().out.splitlines()
    assert not [l for l in lines if " reads rna.h5" in l]
    # the table keeps what each method reads
    assert reason["Conos"].startswith("Conos reads rna.h5 and atac_gas.h5.")


# ------------------------------------------- E18, u2 #5, u3 #11: plot.bar
def test_the_default_bar_runs_no_rank_matrix(monkeypatch):
    from multibench.plot import style

    def boom(*a, **k):
        raise AssertionError("the 0.3 ranks ran")
    monkeypatch.setattr(style, "per_dataset_ranks", boom)
    st = mtb.load_results("vertical")
    mtb.plot.bar(st)
    two = pd.concat([st, st.assign(dataset="D11b")]).assign(source="user")
    _quiet(mtb.plot.bar, two, metrics="clustering")


def test_skip_existing_without_a_saved_run_says_so(tmp_path):
    with pytest.raises(ValueError) as e:
        mtb.run("D11", "vertical", out_dir=str(tmp_path / "outs"), methods=["Matilda"],
                    skip_existing=True, verbose=False)
    assert str(e.value) == (
        f"No saved output of Matilda in {tmp_path / 'outs'} (it would be "
        f"{tmp_path / 'outs' / 'Matilda_D11'}). Pass the out_dir of the earlier run.")


def test_a_renamed_run_keeps_its_language_chip():
    from multibench.plot.bubble import _method_language
    assert _method_language("scJoint (out_D28_cli)") == _method_language("scJoint") == "python"
    assert _method_language("scJoint epochs=10") == "python"
    assert _method_language("MyPCA") is None


# ------------------------------------------- rescore, embedding, export
def test_rescore_of_a_folder_without_outputs_prints_no_status_line(tmp_path, capsys):
    res = mtb.BatchResult([{"method": "Matilda", "status": "RUN_OK", "out_dir":
                            str(tmp_path / "Matilda_D11"), "modalities": ["rna", "adt"]}],
                          "D11", "vertical", out_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="^No output of Matilda"):
        _quiet(res.rescore)
    assert "[rescore] Matilda ->" not in capsys.readouterr().out


def test_export_length_errors_name_the_fix(tmp_path):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((10, 5), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(10)]
    a.var_names = [f"g{i}" for i in range(5)]
    with pytest.raises(ValueError) as e:
        mtb.io.export_dataset(a, tmp_path / "X", rna="X", labels=["A"] * 9)
    assert str(e.value) == "labels has 9 entries for 10 cells: give one label per cell."


def _fake_run(tmp_path, method, mods, n):
    import h5py
    out = tmp_path / f"{method}_D28"
    out.mkdir(parents=True)
    with h5py.File(out / "embedding.h5", "w") as f:
        f.create_dataset("data", data=np.zeros((4, n)))
    return mtb.BatchResult([{"method": method, "status": "RUN_OK", "out_dir": str(out),
                             "modalities": mods}], "D28", "diagonal", out_dir=str(tmp_path))


def test_the_embedding_names_cells_by_the_file_the_method_read(tmp_path):
    """u4 #5: scJoint reads atac_gas.h5 (-1 barcodes), GLUE atac_peak.h5 (-2)."""
    from multibench.engine import resolve
    folder = mtb.config.DEFAULT.data_path / "D28"
    n = sum(len(pd.read_csv(folder / f)) for f in ("rna_cty.csv", "atac_cty.csv"))
    gas = resolve._barcodes_of(folder / "atac_gas.h5")
    peak = resolve._barcodes_of(folder / "atac_peak.h5")
    assert gas[0] != peak[0]
    for method, mods, bars in (("scJoint", ["rna", "atac_gas"], gas),
                               ("GLUE", ["rna", "atac_peak"], peak)):
        obs = _quiet(_fake_run(tmp_path / method, method, mods, n).embedding, method).obs_names
        assert obs[-1].endswith(bars[-1]), (method, obs[-1])


# ------------------------------------------- minimal #2: the labels recipe, 10+ batches
from tests.test_docs_r4 import _read, needs_docs  # noqa: E402


@needs_docs
def test_the_labels_recipe_orders_twelve_batches_by_number(tmp_path):
    import re
    from multibench.engine.resolve import _label_sort_key
    page = _read("tutorials/evaluate.md")
    block = next(b for b in re.findall(r"```python\n(.*?)```", page, re.S) if "order = {" in b)
    code = "\n".join(line for line in block.splitlines() if "fetch(" not in line)
    folder = tmp_path / "SYN12"
    folder.mkdir()
    for i in range(1, 13):
        pd.DataFrame({"x": [f"t{i}"] * 3}).to_csv(folder / f"cty{i}.csv", index=False)
    ns = {"pd": pd, "folder": folder}
    exec(code, ns)
    stems = [f.stem for f in ns["files"]]
    assert stems == sorted(stems, key=_label_sort_key) == [f"cty{i}" for i in range(1, 13)]
    assert list(ns["labels"][::3]) == [f"t{i}" for i in range(1, 13)]
