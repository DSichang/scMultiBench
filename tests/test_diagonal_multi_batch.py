"""Diagonal data in several batches: ``rna<i>.h5`` with ``atac_gas<i>.h5`` or
``atac_peak<i>.h5``, labelled by ``rna_cty<i>.csv`` and ``atac_cty<i>.csv``.

iNMF, online_iNMF, scJoint (gene activity) and GLUE (peaks) read three RNA
and three ATAC batches and write the RNA batches in number order, then the
ATAC batches. Before these roles were known to the resolver:

* ``labels_for`` fell back to the alphabetical order (``atac_cty1..3,
  rna_cty1..3``), so every score was computed against misordered labels;
* ``labels_for`` and ``inputs_for`` reported the folder as a per-batch
  export of single-batch data;
* ``RunResult.obs_names`` was dropped, because ``rna1, rna2, rna3`` counted
  as one set of cells;
* a truncated ``rna_cty2.csv`` passed the file check.
"""
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import registry, resolve, runner
from multibench.eval import pipeline

GAS_METHODS = ("iNMF", "online_iNMF", "scJoint")
METHODS = GAS_METHODS + ("GLUE",)
ORDER = ["rna_cty1", "rna_cty2", "rna_cty3", "atac_cty1", "atac_cty2", "atac_cty3"]
#: cells per file: three RNA batches, three ATAC batches
RNA_CELLS, ATAC_CELLS = (11, 12, 15), (13, 14, 16)


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


def _h5(path, feats, cells):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), len(cells))).astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S40"))
        g.create_dataset("barcodes", data=np.array(cells, dtype="S40"))


def _labels(path, n):
    pd.DataFrame({"x": (["A", "B"] * n)[:n]}).to_csv(path, index=False)


def _folder(root, name="MB"):
    """Three RNA and three ATAC batches, the ATAC as gene activity and as peaks."""
    d = root / name
    d.mkdir(parents=True)
    genes = [f"g{i}" for i in range(30)]
    peaks = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
    for i, (n_rna, n_atac) in enumerate(zip(RNA_CELLS, ATAC_CELLS), start=1):
        atac = [f"a{i}_{k}" for k in range(n_atac)]
        _h5(d / f"rna{i}.h5", genes, [f"r{i}_{k}" for k in range(n_rna)])
        _h5(d / f"atac_gas{i}.h5", genes, atac)
        _h5(d / f"atac_peak{i}.h5", peaks, atac)
        _labels(d / f"rna_cty{i}.csv", n_rna)
        _labels(d / f"atac_cty{i}.csv", n_atac)
    return d


def _variant(method, d):
    return resolve.select_variant(registry.get(method), "diagonal", None, ds_dir=d)


# ------------------------------------------------------------------ label order
@pytest.mark.parametrize("method", METHODS)
def test_labels_follow_the_rna_batches_then_the_atac_batches(tmp_path, method):
    _folder(tmp_path)
    got = mtb.labels_for("MB", "diagonal", method, data_path=tmp_path)
    assert list(got) == ORDER
    assert [p.rsplit("/", 1)[-1] for p in got.values()] == [f"{st}.csv" for st in ORDER]


def test_the_default_order_is_the_same_and_evaluate_takes_it(tmp_path):
    _folder(tmp_path)
    got = mtb.labels_for("MB", data_path=tmp_path)
    assert list(got) == ORDER
    assert sorted(reversed(ORDER), key=resolve._label_sort_key) == ORDER
    # a plain dict in this order needs no label_order=
    assert pipeline._labels_from_dict(dict(got), None) == list(got.values())
    # numbers sort as numbers within a modality
    assert sorted(["atac_cty2", "rna_cty10", "rna_cty2", "atac_cty1"],
                  key=resolve._label_sort_key) == ["rna_cty2", "rna_cty10",
                                                   "atac_cty1", "atac_cty2"]


def test_single_batch_diagonal_order_is_unchanged():
    key = resolve._label_sort_key
    assert sorted(["atac_cty", "rna_cty", "cty2", "cty", "cty1", "peak_cty"], key=key) == [
        "cty", "cty1", "cty2", "rna_cty", "atac_cty", "peak_cty"]


def test_label_partners_pair_a_numbered_file_with_its_own_batch():
    lp = resolve._label_partners
    roles = ["rna1", "rna2", "atac_gas1", "atac_gas2", "atac_peak2", "rna_cty1"]
    assert lp("rna_cty1", roles) == ["rna1"]
    assert lp("atac_cty2", roles) == ["atac_gas2", "atac_peak2"]
    assert lp("rna_cty", roles) == []                  # the unnumbered file: no numbered role
    assert lp("rna_cty1", ["rna", "atac_gas"]) == []   # and the reverse


def test_a_variant_gets_only_the_label_files_of_its_batches(tmp_path):
    d = _folder(tmp_path)
    _h5(d / "rna4.h5", ["g0", "g1", "g2"], ["x0", "x1"])
    _labels(d / "rna_cty4.csv", 2)
    assert list(mtb.labels_for("MB", "diagonal", "iNMF", data_path=tmp_path)) == ORDER


# ---------------------------------------------------------- batch = each file
def test_run_all_scores_against_this_order_with_one_batch_per_file(tmp_path):
    _folder(tmp_path)
    n = sum(RNA_CELLS) + sum(ATAC_CELLS)
    cands = W._label_candidates("MB", n, tmp_path)
    names = [c[0] for c in cands]
    assert names[0] == [f"{st}.csv" for st in ORDER]
    # the ATAC files first is the one alternative, as for one batch
    assert names[1:] == [[f"{st}.csv" for st in ORDER[3:] + ORDER[:3]]]
    batch = cands[0][2]
    assert len(batch) == n and sorted(set(batch.tolist())) == [1, 2, 3, 4, 5, 6]
    assert batch.tolist()[:RNA_CELLS[0] + 1] == [1] * RNA_CELLS[0] + [2]


def test_cell_ids_follow_the_label_files(tmp_path):
    _folder(tmp_path)
    ids = W._dataset_cell_ids("MB", tmp_path)
    assert len(ids) == sum(RNA_CELLS) + sum(ATAC_CELLS)
    assert ids[0] == "r1_0" and ids[sum(RNA_CELLS)] == "a1_0" and ids[-1] == "a3_15"


# ------------------------------------------------------------ no per-batch hint
def test_no_per_batch_hint_on_such_a_folder(tmp_path):
    d = _folder(tmp_path)
    assert resolve._per_batch_hint(d, "diagonal") is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mtb.labels_for("MB", "diagonal", data_path=tmp_path)
        mtb.labels_for("MB", "diagonal", "iNMF", data_path=tmp_path, check=True)
        for method in METHODS:
            mtb.inputs_for("MB", "diagonal", method, data_path=tmp_path, check=True)
    # a method that reads one rna.h5 is not told to export again
    with pytest.raises(FileNotFoundError) as e:
        mtb.inputs_for("MB", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    assert "per-batch files" not in str(e.value)


def test_the_hint_still_fires_when_no_variant_reads_the_folder(tmp_path):
    d = tmp_path / "SPLIT"
    d.mkdir()
    for i in (1, 2):
        _h5(d / f"rna{i}.h5", ["g0", "g1", "g2"], [f"c{i}_{k}" for k in range(5)])
    _h5(d / "atac_gas.h5", ["g0", "g1", "g2"], [f"a{k}" for k in range(5)])
    assert "per-batch files (rna1.h5, rna2.h5, ...)" in resolve._per_batch_hint(d, "diagonal")
    with pytest.warns(UserWarning, match="This folder holds per-batch files"):
        mtb.labels_for("SPLIT", "diagonal", data_path=tmp_path)
    # vertical is not affected by the diagonal rule
    assert resolve._per_batch_hint(_folder(tmp_path), "vertical") is not None


def test_scan_finds_the_four_rows_and_no_export_advice(tmp_path, no_envs):
    _folder(tmp_path)
    df = mtb.scan("MB", "diagonal", data_path=tmp_path, verbose=False)
    ok = df[df["files_ok"]]
    assert sorted(ok["method"]) == sorted(METHODS)
    assert all(m.startswith("rna1+rna2+rna3+atac_") for m in ok["modalities"])
    assert not df["reason"].str.contains("per-batch files").any()
    assert not df["caveat"].str.contains("not used").any()


# ------------------------------------------------------------------ cell groups
@pytest.mark.parametrize("method", METHODS)
def test_each_file_is_its_own_cell_group(tmp_path, method):
    d = _folder(tmp_path)
    variant = _variant(method, d)
    kind = "atac_peak" if method == "GLUE" else "atac_gas"
    roles = [f"rna{i}" for i in (1, 2, 3)] + [f"{kind}{i}" for i in (1, 2, 3)]
    assert runner._cell_group_roles(variant) == [[r] for r in roles]
    inputs = mtb.inputs_for("MB", "diagonal", method, data_path=tmp_path)
    n = sum(RNA_CELLS) + sum(ATAC_CELLS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        names = runner._obs_names(method, variant, inputs, np.zeros((n, 4)))
    want = [f"r{i}_{k}" for i, c in enumerate(RNA_CELLS, 1) for k in range(c)]
    want += [f"a{i}_{k}" for i, c in enumerate(ATAC_CELLS, 1) for k in range(c)]
    assert names == want


def test_single_batch_diagonal_groups_are_unchanged():
    v = registry.get("MultiMAP").select("diagonal", {"rna", "atac_peak", "atac_gas"})
    groups = runner._cell_group_roles(v)
    assert sorted(map(sorted, groups)) == [["atac_gas", "atac_peak"], ["rna"]]


# ------------------------------------------------------------ label length check
@pytest.mark.parametrize("short", ["rna_cty2", "atac_cty3"])
def test_a_truncated_label_file_fails_the_file_check(tmp_path, no_envs, short):
    d = _folder(tmp_path)
    full = len(pd.read_csv(d / f"{short}.csv"))
    _labels(d / f"{short}.csv", full - 2)
    data = "rna2.h5" if short == "rna_cty2" else "atac_gas3.h5"
    for method in GAS_METHODS:
        with pytest.raises(ValueError) as e:
            mtb.inputs_for("MB", "diagonal", method, data_path=tmp_path, check=True)
        assert f"{short}.csv has {full - 2} labels, but {data} has {full} cells" in str(e.value)
    with pytest.raises(ValueError, match=rf"{short}\.csv has {full - 2} labels"):
        mtb.inputs_for("MB", "diagonal", "GLUE", data_path=tmp_path, check=True)
    df = mtb.scan("MB", "diagonal", data_path=tmp_path, verbose=False)
    rows = df[df["method"].isin(METHODS) & df["modalities"].str.startswith("rna1+")]
    assert len(rows) == 4 and not rows["files_ok"].any()
    assert rows["files_reason"].str.contains(f"{short}.csv has {full - 2} labels").all()


def test_batch_label_file_names_the_diagonal_files():
    blf = resolve._batch_label_file
    assert blf("rna2", "/d/rna2.h5", "diagonal") == ("2", resolve.Path("/d/rna_cty2.csv"))
    assert blf("atac_gas3", "/d/atac_gas3.h5", "diagonal") == (
        "3", resolve.Path("/d/atac_cty3.csv"))
    assert blf("atac_peak1", "/d/atac_peak1.h5", "diagonal") == (
        "1", resolve.Path("/d/atac_cty1.csv"))
    assert blf("rna", "/d/rna.h5", "diagonal") is None
    # cross and mosaic keep cty<i>.csv
    assert blf("rna2", "/d/rna2.h5", "cross") == ("2", resolve.Path("/d/cty2.csv"))
    assert blf("rna2", "/d/rna2.h5") == ("2", resolve.Path("/d/cty2.csv"))


def test_folder_and_variant_batches_count_the_diagonal_label_files(tmp_path):
    d = tmp_path / "L"
    d.mkdir()
    _labels(d / "rna_cty1.csv", 3)
    _labels(d / "atac_cty4.csv", 3)
    _labels(d / "rna_cty.csv", 3)
    assert resolve._folder_batches(d) == {1, 4}
    assert resolve._variant_batches(["rna_cty2", "atac_gas3", "rna", "atac_cty"]) == {2, 3}


# --------------------------------------------------- gene activity follows peaks
def test_a_gene_activity_batch_in_another_cell_order_is_refused(tmp_path):
    d = _folder(tmp_path)
    cells = [f"a2_{k}" for k in range(ATAC_CELLS[1])]
    _h5(d / "atac_gas2.h5", [f"g{i}" for i in range(30)], cells[::-1])
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("MB", "diagonal", "iNMF", data_path=tmp_path, check=True)
    msg = str(e.value)
    assert "atac_gas2.h5" in msg and "another order than atac_peak2.h5" in msg
    assert "atac_cty2.csv follows atac_peak2.h5" in msg
    # GLUE reads the peak files, which the labels follow
    mtb.inputs_for("MB", "diagonal", "GLUE", data_path=tmp_path, check=True)


# ------------------------------------------------------------------------ layout
def test_the_file_table_lists_the_per_batch_label_files():
    text = mtb.describe_layout()
    assert "rna_cty1.csv, atac_cty1.csv ... - the same per batch (diagonal" in text
    assert "rna_cty1.csv" not in mtb.describe_layout("cross")
