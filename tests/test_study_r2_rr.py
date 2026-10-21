"""Dataset-fit and message fixes from the round-2 student study (ledger M15, M16,
M18, M21, M22, M31, M36: resolve.py, ingest.py and two methods.yaml entries).

Each test fails on the code before its fix.
"""
import os
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config, workflow
from multibench.engine import envs, ingest, resolve
from tests.conftest import full_scan

PEAKS = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(40)]
GENES = [f"g{i}" for i in range(30)]


def _h5(path, feats, barcodes, *, data=None):
    """A canonical .h5 (features x cells) with Poisson counts."""
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.poisson(1.0, size=(len(feats), len(barcodes))).astype(float)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=data)
        g.create_dataset("features", data=np.array(feats, dtype="S24"))
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S24"))


def _labels(path, n):
    pd.DataFrame({"x": ["A"] * n}).to_csv(path, index=False)


def _diagonal(root, name, rna_cells, peak_cells, gas_cells=None):
    d = root / name
    d.mkdir()
    _h5(d / "rna.h5", GENES, rna_cells)
    _h5(d / "atac_peak.h5", PEAKS, peak_cells)
    if gas_cells is not None:
        _h5(d / "atac_gas.h5", GENES, gas_cells)
    _labels(d / "rna_cty.csv", len(rna_cells))
    _labels(d / "atac_cty.csv", len(peak_cells))
    return d


def _row(name, category, method, data_path, **kw):
    df = full_scan(name, category, methods=[method], data_path=data_path, verbose=False, **kw)
    return df[df.method == method].iloc[0]


# --- M15: Seurat_v5 needs its bridge files from the same cells ---------------
SEURAT_D28 = ("Seurat_v5 needs RNA and ATAC from the same cells as its bridge. In D28, "
              "rna.h5 and atac_peak.h5 share 0 of 6,408 and 4,606 cells.")


def test_seurat_v5_setup_hint_says_unpaired_data_cannot_run():
    assert mtb.method_info("Seurat_v5")["setup_hint"] == (
        "Seurat_v5 uses your RNA and ATAC files as its paired bridge, so both must hold "
        "the same cells. Unpaired data cannot run through the package.")


def test_gene_activity_of_other_cells_fails_the_file_check(tmp_path):
    atac = [f"a{i}" for i in range(40)]
    _diagonal(tmp_path, "OTHER", [f"c{i}" for i in range(50)], atac,
              [f"b{i}" for i in range(40)])
    other = ("SCALEX reads atac_gas.h5 of OTHER, which holds other cells than "
             "atac_peak.h5. The files hold 40 and 40 cells and share 0. Both files need "
             "the same ATAC cells.")
    with pytest.raises(ValueError) as e:
        mtb.engine.resolve.inputs_for("OTHER", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    assert str(e.value) == other
    row = _row("OTHER", "diagonal", "SCALEX", tmp_path)
    assert other in row["files_reason"]
    assert row["reason"].startswith(other.replace("SCALEX reads atac_gas.h5 of OTHER, which ",
                                                  "atac_gas.h5 "))


def test_a_gem_well_suffix_alone_is_the_same_cells(tmp_path):
    """D28's atac_gas.h5 ends its barcodes in -1 and atac_peak.h5 in -2."""
    atac = [f"AAC{i:03d}" for i in range(40)]
    _diagonal(tmp_path, "SUFFIX", [f"c{i}" for i in range(50)],
              [f"{b}-2" for b in atac], [f"{b}-1" for b in atac])
    mtb.engine.resolve.inputs_for("SUFFIX", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    _diagonal(tmp_path, "SUFFIX_SHUF", [f"c{i}" for i in range(50)],
              [f"{b}-2" for b in atac], [f"{b}-1" for b in reversed(atac)])
    with pytest.raises(ValueError, match="another order"):
        mtb.engine.resolve.inputs_for("SUFFIX_SHUF", "diagonal", "SCALEX", data_path=tmp_path, check=True)


def test_d28_scans_as_before(root):
    df = full_scan("D28", "diagonal", data_path=root / "data", verbose=False)
    # the rows for several RNA and ATAC batches read numbered files D28 lacks:
    # not listed since 0.4.0 (X80), and counted
    several = df.modalities.str.contains(r"\d")
    assert df[several].empty and df.attrs["not_listed_batch"] == 4
    every = full_scan("D28", "diagonal", data_path=root / "data", verbose=False,
                      modalities="all")
    several = every.modalities.str.contains(r"\d")
    assert sorted(every[several].method) == ["GLUE", "iNMF", "online_iNMF", "scJoint"]
    assert not every[several].files_ok.any()
    blocked = df[~df.files_ok & ~several]
    assert blocked.method.tolist() == ["Seurat_v5"], blocked[["method", "files_reason"]]


# --- M18: export_dataset barcode messages and the ADT raw-count hint ---------
def _rna_adata(barcodes, *, obsm=None):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(len(barcodes), len(GENES))).astype(float))
    a.obs_names = list(barcodes)
    a.var_names = GENES
    if obsm is not None:
        a.obsm["protein"] = obsm
    return a


def _peak_adata(barcodes):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(len(barcodes), len(PEAKS))).astype(float))
    a.obs_names = list(barcodes)
    a.var_names = PEAKS
    return a


def test_disjoint_rna_and_atac_lead_with_the_diagonal_hint(tmp_path):
    rna = _rna_adata([f"c{i}" for i in range(30)])
    with pytest.raises(ValueError) as e:
        mtb.io.export_dataset(rna, tmp_path / "X", atac=_peak_adata([f"a{i}" for i in range(20)]),
                              atac_kind="peak")
    msg = str(e.value)
    # 0.4.0 (X10): an example barcode of each, then the two fixes
    assert msg == ("RNA and ATAC share no barcode (RNA e.g. c0, ATAC e.g. a0). Make the "
                   "names match; for different cells pass category=\"diagonal\".")
    assert "subset" not in msg.lower()


def test_partial_overlap_says_which_cells_in_short_sentences(tmp_path):
    rna = _rna_adata([f"c{i}" for i in range(30)])
    atac = _peak_adata([f"c{i}" for i in range(20)] + [f"x{i}" for i in range(5)])
    with pytest.raises(ValueError) as e:
        mtb.io.export_dataset(rna, tmp_path / "X", atac=atac, atac_kind="peak")
    msg = str(e.value)
    # R3-14: the object is named in plain words, not by the parameter name
    assert msg.startswith("atac has 25 cells, and 5 of them are not in the object "
                          "passed as data")
    assert ("All modalities of one dataset must hold the same cells. Subset each modality "
            "to the shared barcodes first.") in msg


def test_adt_hint_follows_the_obsm_selector(tmp_path):
    clr = np.random.default_rng(0).normal(size=(30, 5))
    rna = _rna_adata([f"c{i}" for i in range(30)], obsm=clr)
    with pytest.warns(UserWarning, match=r"for example with adt='obsm:protein_counts'\."):
        mtb.io.export_dataset(rna, tmp_path / "O", adt="obsm:protein",
                              adt_names=[f"p{i}" for i in range(5)])


def test_adt_hint_for_a_layer_stays_layer_counts(tmp_path):
    """For ADT read by a selector. ADT passed as its own object has no
    selector: ``adt='layer:counts'`` would read a layer of the RNA object, so
    the hint names the X of the ADT object (review round 3 of 0.3.5)."""
    rna = _rna_adata([f"c{i}" for i in range(30)])
    prot = ad.AnnData(np.random.default_rng(0).normal(size=(30, 5)))
    prot.obs_names = list(rna.obs_names)
    prot.var_names = [f"p{i}" for i in range(5)]
    with pytest.warns(UserWarning) as rec:
        mtb.io.export_dataset(rna, tmp_path / "L", adt=prot)
    assert [str(w.message) for w in rec if "whole numbers" in str(w.message)] == [
        "The adt values are not whole numbers, so they look log-normalised. Put the raw "
        "counts in adt.X first."]
    with pytest.warns(UserWarning, match=r"for example with adt='layer:counts'\."):
        mtb.io.export_dataset(prot, tmp_path / "S", rna=None, adt="X")


def test_unitednet_still_reads_an_older_rna_cty_file(tmp_path):
    cells = [f"c{i}" for i in range(30)]
    d = tmp_path / "OLD"
    d.mkdir()
    _h5(d / "rna.h5", GENES, cells)
    _h5(d / "atac.h5", GENES, cells)        # the vertical ATAC name (X13)
    _labels(d / "rna_cty.csv", 30)
    got = mtb.engine.resolve.inputs_for("OLD", "vertical", "UnitedNet", data_path=tmp_path, check=True)
    assert got["cty"] == str(d / "rna_cty.csv")


def test_rna_cty_next_to_atac_cty_is_not_paired_labels(root, tmp_path):
    """In a diagonal folder rna_cty.csv labels the RNA cells only."""
    got = mtb.engine.resolve.inputs_for("D28", "vertical", "UnitedNet", data_path=root / "data",
                         check=False)
    assert got["cty"].endswith("/D28/cty.csv")
    row = _row("D28", "vertical", "UnitedNet", root / "data")
    assert not row["files_ok"] and "cty.csv is missing." in row["reason"]


def test_unitednet_reason_names_atac_first_then_cty(tmp_path):
    d = tmp_path / "RNAONLY"
    d.mkdir()
    _h5(d / "rna.h5", GENES, [f"c{i}" for i in range(30)])
    row = _row("RNAONLY", "vertical", "UnitedNet", tmp_path)
    assert row["reason"].startswith("UnitedNet needs peak ATAC (atac.h5), which the "
                                    "folder does not have. cty.csv is missing.")


# --- M22 / R6-10: caveats start with the method, then the problem --------------
def test_every_caveat_leads_with_its_warning():
    heads = {
        resolve.PEAK_IN_GAS_CAVEAT: "needs gene-activity ATAC.",
        resolve.PEAK_FED_TO_GAS_CAVEAT: "needs gene-activity ATAC.",
        resolve.GAS_FED_TO_PEAK_CAVEAT: "needs peak ATAC.",
        resolve.NOT_COUNTS_CAVEAT: "needs raw counts.",
        resolve.DIAGONAL_CTY_CAVEAT: "needs rna_cty.csv",
        resolve.UNUSED_BATCHES_CAVEAT: "reads batches",
    }
    for text, head in heads.items():
        assert text.startswith("{method} " + head), text


# --- M31: a variant that reads fewer batches than the folder holds ------------
TWO_BATCHES = ["rna1", "rna2", "adt1", "adt2"]        # UINMF's two-batch cross variant


def test_labels_for_uinmf_returns_the_batches_it_reads(root):
    """UINMF has a cross variant for two batches and one for three. D52 holds
    three, so the three-batch variant is the one the folder picks."""
    assert list(mtb.engine.resolve.labels_for("D52", "cross", "UINMF", data_path=root / "data")) == \
        ["cty1", "cty2", "cty3"]
    assert list(mtb.engine.resolve.labels_for("D52", "cross", "UINMF", modalities=TWO_BATCHES,
                               data_path=root / "data")) == ["cty1", "cty2"]
    assert list(mtb.engine.resolve.labels_for("D52", "cross", "StabMap", data_path=root / "data")) == \
        ["cty3", "cty1", "cty2"]
    assert list(mtb.engine.resolve.labels_for("D52", data_path=root / "data")) == ["cty1", "cty2", "cty3"]


def test_scan_caveat_names_the_unused_batch(root):
    df = full_scan("D52", "cross", data_path=root / "data", verbose=False)
    df = df[df.files_ok]
    by = df.set_index(["method", "modalities"])["caveat"]
    # r3 (X68): the dry run's one line says the three-batch row runs instead; the
    # row itself keeps only its own caveat
    two = by[("UINMF", "+".join(TWO_BATCHES))]
    assert two.startswith("UINMF reads batches 1-2 of 3. Batch 3 is not used.")
    assert "so this row is not run" not in two
    others = [k for k in by.index if k != ("UINMF", "+".join(TWO_BATCHES))]
    assert ("UINMF", "rna1+rna2+rna3+adt1+adt2+adt3") in others
    assert not any("reads batches" in by[k] for k in others)


def test_dry_run_prints_the_unused_batch(root, capsys):
    mtb.run(os.path.join(root / 'data', 'D52'), 'cross', methods=['UINMF'], modalities=TWO_BATCHES, dry_run=True)
    out = capsys.readouterr().out
    assert "[run] UINMF reads batches 1-2 of 3. Batch 3 is not used.\n" in out
    # without modalities= the three-batch row runs, and the caveat of the
    # two-batch row inside it is not printed
    mtb.run(os.path.join(root / 'data', 'D52'), 'cross', methods=['UINMF', 'StabMap'], dry_run=True)
    out = capsys.readouterr().out
    assert "reads batches" not in out


# --- M36: read_canonical after pip install ------------------------------------
def test_read_canonical_missing_path_names_the_data_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        mtb.io.read_canonical("data/D11/rna.h5")
    assert str(e.value) == (
        f"data/D11/rna.h5 not found. Demo datasets are under mtb.config.DEFAULT.data_path "
        f"({config.DEFAULT.data_path}).")


def test_read_canonical_example_builds_the_path_from_data_path():
    doc = mtb.io.read_canonical.__doc__
    assert 'rna = mtb.io.read_canonical(mtb.config.DEFAULT.data_path / "D11" / "rna.h5")' \
        in doc
    assert "sparse=" not in doc                    # deprecated in 0.4.0
    assert '"data/D11/rna.h5"' not in doc


def test_no_dataset_keyed_caveats_remain():
    # R3-04: GLUE's D28 peak-name caveat became a content check
    assert not hasattr(workflow, "_CAVEATS")
