"""Round d2 of the 0.4.0 review: the cell names RunResult.embedding gives are
accepted back as batch= / labels= ids; a misspelt method in another letter
case gets a did-you-mean hint; messages name the embedding, a missing folder and
an empty table; evaluate of a run's file reads the label files the run read."""
import json
import shutil
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W


def _set_barcodes(path, bars):
    with h5py.File(path, "a") as f:
        del f["matrix/barcodes"]
        f.create_dataset("matrix/barcodes", data=np.asarray(bars, dtype=object),
                         dtype=h5py.string_dtype())


@pytest.fixture
def d28_like(tmp_path):
    """Diagonal folder as D28: atac_peak.h5 and atac_gas.h5 hold the same cells,
    with -2 and -1 suffixes; the -1 barcodes repeat three RNA barcodes."""
    rng = np.random.default_rng(0)
    n_r, n_a = 30, 20
    rna = ad.AnnData(rng.poisson(1, (n_r, 15)).astype(np.float32),
                     obs=pd.DataFrame({"celltype": list("ab") * (n_r // 2)},
                                      index=[f"R{i}-1" for i in range(n_r)]))
    atac = ad.AnnData(rng.poisson(1, (n_a, 15)).astype(np.float32),
                      obs=pd.DataFrame({"celltype": list("ab") * (n_a // 2)},
                                       index=[f"A{i}-1" for i in range(n_a)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mtb.io.export_dataset(rna, tmp_path / "DD", rna="X", atac=atac,
                              atac_kind="gene_activity", labels="obs:celltype",
                              category="diagonal")
    folder = tmp_path / "DD"
    gas = list(atac.obs_names)
    gas[:3] = list(rna.obs_names[:3])
    _set_barcodes(folder / "atac_gas.h5", gas)
    shutil.copy(folder / "atac_gas.h5", folder / "atac_peak.h5")
    _set_barcodes(folder / "atac_peak.h5", [b[:-2] + "-2" for b in gas])
    return folder, list(rna.obs_names), gas


def test_embedding_names_are_accepted_as_batch_ids(d28_like):
    folder, rna_bars, gas = d28_like
    names = ([f"rna_cty:{b}" for b in rna_bars] + [f"atac_cty:{b}" for b in gas])
    assert names in W._embedding_cell_ids("DD", folder.parent)
    v = np.where(np.arange(len(names)) % 3, "x", "y")
    shuffled = pd.Series(v, index=names).sample(frac=1, random_state=0)
    vec, by_cell = W._cell_vector(shuffled, "DD", folder.parent)
    assert by_cell and list(vec) == list(v)


def test_dataset_barcodes_still_match(d28_like):
    folder, rna_bars, gas = d28_like
    ids = rna_bars + [b[:-2] + "-2" for b in gas]
    assert W._dataset_cell_ids("DD", folder.parent) == ids
    v = np.arange(len(ids)).astype(str)
    vec, by_cell = W._cell_vector(pd.Series(v, index=ids)[::-1], "DD", folder.parent)
    assert by_cell and list(vec) == list(v)


def test_repeated_raw_barcodes_are_still_refused(d28_like):
    folder, rna_bars, gas = d28_like
    s = pd.Series("x", index=rna_bars + gas)
    with pytest.raises(ValueError, match=r"the index repeats 3 ids"):
        W._cell_vector(s, "DD", folder.parent)


def test_a_misspelt_method_in_other_letter_case_gets_a_hint():
    with pytest.raises(KeyError, match=r"Unknown method scalexx\. Did you mean SCALEX\?"):
        mtb.cite("scalexx")


def test_load_run_of_a_missing_folder_says_it_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"nothing does not exist\. Pass the out_dir"):
        mtb.load_run(tmp_path / "nothing")


def test_an_empty_table_is_not_blamed_on_filters():
    empty = pd.DataFrame(columns=["method", "metric", "value"])
    with pytest.raises(ValueError, match=r"^The table has no rows\.$"):
        mtb.plot.bubble(empty)


def test_evaluate_of_a_run_file_reads_the_label_files_the_run_read(tmp_path):
    """UINMF on D52 reads two of the three batches; its record names them."""
    data = tmp_path / "data" / "MYX"
    data.mkdir(parents=True)
    rng = np.random.default_rng(0)
    labs = {}
    for i, n in ((1, 40), (2, 30), (3, 50)):
        labs[i] = np.array(list("abc") * n)[:n]
        pd.DataFrame({"x": labs[i]}).to_csv(data / f"cty{i}.csv", index=False)
    lab = np.concatenate([labs[1], labs[2]])
    emb = rng.normal(scale=0.3, size=(len(lab), 5))
    emb[np.arange(len(lab)), pd.factorize(lab)[0]] += 4.0
    out = tmp_path / "out"
    (out / "UINMF_MYX").mkdir(parents=True)
    with h5py.File(out / "UINMF_MYX" / "embedding.h5", "w") as f:
        f.create_dataset("data", data=emb.T)
    (out / "batch_result.json").write_text(json.dumps(
        {"dataset": "MYX", "category": "cross", "records": [
            {"method": "UINMF", "status": "scored", "out_dir": str(out / "UINMF_MYX"),
             "labels_used": ["cty1.csv", "cty2.csv"]}]}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = mtb.evaluate(out / "UINMF_MYX" / "embedding.h5", dataset=str(data),
                           metrics=["ARI"], verbose=False)
        want = mtb.evaluate(emb, labels=lab, metrics=["ARI"], verbose=False)
    assert got.value.tolist() == want.value.tolist()
