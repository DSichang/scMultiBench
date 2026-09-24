"""Dataset-fit and message fixes from the round-2 student study (ledger M15, M16,
M18, M21, M22, M31, M36: resolve.py, ingest.py and two methods.yaml entries).

Each test fails on the code before its fix.
"""
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config, workflow
from multibench.engine import envs, resolve

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


def _barcodes(path):
    with h5py.File(path, "r") as f:
        return [x.decode() for x in f["matrix/barcodes"][:]]


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
    df = mtb.scan(name, category, methods=[method], data_path=data_path, verbose=False, **kw)
    return df[df.method == method].iloc[0]


@pytest.fixture
def envs_installed(monkeypatch):
    """Every env counts as installed, so ``runnable`` follows ``files_ok``."""
    every = frozenset(envs.group_for(m) for m in mtb.list_methods())
    monkeypatch.setattr(workflow, "_installed_envs", lambda: every)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


# --- M15: Seurat_v5 needs its bridge files from the same cells ---------------
SEURAT_D28 = ("Seurat_v5 needs RNA and ATAC from the same cells as its bridge; these "
              "files hold different cells (6,408 and 4,606, 0 shared)")


def test_seurat_v5_is_not_file_ready_on_d28(root, envs_installed):
    row = _row("D28", "diagonal", "Seurat_v5", root / "data")
    assert not row["files_ok"] and not row["runnable"]
    assert SEURAT_D28 in row["reason"]
    assert SEURAT_D28 in row["files_reason"]


def test_seurat_v5_unpaired_folder_fails_the_file_check(tmp_path, envs_installed):
    _diagonal(tmp_path, "UNPAIRED", [f"c{i}" for i in range(50)], [f"a{i}" for i in range(45)])
    row = _row("UNPAIRED", "diagonal", "Seurat_v5", tmp_path)
    assert not row["files_ok"] and not row["runnable"]
    assert ("Seurat_v5 needs RNA and ATAC from the same cells as its bridge; these files "
            "hold different cells (50 and 45, 0 shared)") in row["reason"]
    with pytest.raises(ValueError, match="Seurat_v5 needs RNA and ATAC from the same cells"):
        mtb.inputs_for("UNPAIRED", "diagonal", "Seurat_v5", data_path=tmp_path, check=True)
    # check=False keeps returning the paths
    assert set(mtb.inputs_for("UNPAIRED", "diagonal", "Seurat_v5", data_path=tmp_path)) == \
        {"rna", "atac_peak"}


def test_seurat_v5_paired_folder_passes(tmp_path, envs_installed):
    cells = [f"c{i}" for i in range(50)]
    _diagonal(tmp_path, "PAIRED", cells, list(reversed(cells)))
    row = _row("PAIRED", "diagonal", "Seurat_v5", tmp_path)
    assert row["files_ok"], row["files_reason"]
    assert row["runnable"]
    mtb.inputs_for("PAIRED", "diagonal", "Seurat_v5", data_path=tmp_path, check=True)


def test_seurat_v5_setup_hint_says_unpaired_data_cannot_run():
    assert mtb.method_info("Seurat_v5")["setup_hint"] == (
        "Seurat_v5 uses your RNA and ATAC files as its paired bridge, so both must hold "
        "the same cells. Unpaired data cannot run through the package.")


def test_recommend_notes_name_the_bridge_dataset_of_the_stored_scores():
    doc = mtb.recommend.__doc__
    assert ("The stored Seurat_v5 diagonal scores come from runs with a separate paired "
            "bridge dataset") in " ".join(doc.split())


# --- M16: atac_gas.h5 must list the ATAC cells of atac_peak.h5, in its order --
def test_shuffled_gene_activity_fails_the_file_check(tmp_path, envs_installed):
    atac = [f"a{i}" for i in range(40)]
    shuffled = list(np.random.default_rng(1).permutation(atac))
    _diagonal(tmp_path, "SHUF", [f"c{i}" for i in range(50)], atac, shuffled)
    order = ("atac_gas.h5 lists the ATAC cells in another order than atac_peak.h5; "
             "atac_cty.csv follows atac_peak.h5. Write it again with")
    for m in ("SCALEX", "MultiMAP", "Seurat_v3"):
        row = _row("SHUF", "diagonal", m, tmp_path)
        assert not row["files_ok"], m
        assert order in row["reason"], (m, row["reason"])
    with pytest.raises(ValueError, match="another order than atac_peak.h5"):
        mtb.inputs_for("SHUF", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    # a method that reads only the peaks is not affected by atac_gas.h5
    assert _row("SHUF", "diagonal", "GLUE", tmp_path)["files_ok"]


def test_gene_activity_of_other_cells_fails_the_file_check(tmp_path):
    atac = [f"a{i}" for i in range(40)]
    _diagonal(tmp_path, "OTHER", [f"c{i}" for i in range(50)], atac,
              [f"b{i}" for i in range(40)])
    with pytest.raises(ValueError, match="atac_gas.h5 and atac_peak.h5 hold different cells"):
        mtb.inputs_for("OTHER", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    row = _row("OTHER", "diagonal", "SCALEX", tmp_path)
    assert "atac_gas.h5 and atac_peak.h5 hold different cells" in row["reason"]


def test_a_gem_well_suffix_alone_is_the_same_cells(tmp_path):
    """D28's atac_gas.h5 ends its barcodes in -1 and atac_peak.h5 in -2."""
    atac = [f"AAC{i:03d}" for i in range(40)]
    _diagonal(tmp_path, "SUFFIX", [f"c{i}" for i in range(50)],
              [f"{b}-2" for b in atac], [f"{b}-1" for b in atac])
    mtb.inputs_for("SUFFIX", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    _diagonal(tmp_path, "SUFFIX_SHUF", [f"c{i}" for i in range(50)],
              [f"{b}-2" for b in atac], [f"{b}-1" for b in reversed(atac)])
    with pytest.raises(ValueError, match="another order"):
        mtb.inputs_for("SUFFIX_SHUF", "diagonal", "SCALEX", data_path=tmp_path, check=True)


def test_d28_scans_as_before(root):
    df = mtb.scan("D28", "diagonal", data_path=root / "data", verbose=False)
    blocked = df[~df.files_ok]
    assert blocked.method.tolist() == ["Seurat_v5"], blocked[["method", "files_reason"]]


def _gas_adata(barcodes, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(2.0, size=(len(barcodes), len(GENES))).astype(float))
    a.obs_names = list(barcodes)
    a.var_names = GENES
    return a


def test_to_canonical_gas_follows_the_peak_order(tmp_path, capsys):
    atac = [f"a{i}" for i in range(40)]
    d = _diagonal(tmp_path, "TOCAN", [f"c{i}" for i in range(50)], atac)
    shuffled = list(np.random.default_rng(2).permutation(atac))
    gas = _gas_adata(shuffled)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mtb.io.to_canonical(gas, d, modality="gas")
    assert out == d / "atac_gas.h5"
    assert _barcodes(out) == atac
    back = mtb.io.read_canonical(out, sparse=False)
    np.testing.assert_array_equal(back[atac].X, gas[atac].X)
    err = capsys.readouterr().err
    assert "atac_peak.h5" in err and len(err.strip().splitlines()) == 1
    mtb.inputs_for("TOCAN", "diagonal", "SCALEX", data_path=tmp_path, check=True)


def test_to_canonical_gas_of_other_cells_warns(tmp_path):
    d = _diagonal(tmp_path, "TOCAN2", [f"c{i}" for i in range(50)],
                  [f"a{i}" for i in range(40)])
    with pytest.warns(UserWarning, match="different cells"):
        mtb.io.to_canonical(_gas_adata([f"b{i}" for i in range(40)]), d, modality="gas")


def test_to_canonical_gas_in_peak_order_prints_nothing(tmp_path, capsys):
    atac = [f"a{i}" for i in range(40)]
    d = _diagonal(tmp_path, "TOCAN3", [f"c{i}" for i in range(50)], atac)
    mtb.io.to_canonical(_gas_adata(atac), d, modality="gas")
    assert capsys.readouterr().err == ""


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
    assert msg.startswith("RNA and ATAC have no cells in common. For RNA and ATAC from "
                          "different cells (diagonal integration), pass category=\"diagonal\".")
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
    with pytest.warns(UserWarning, match=r"e\.g\. adt='obsm:protein_counts'"):
        mtb.io.export_dataset(rna, tmp_path / "O", adt="obsm:protein",
                              adt_names=[f"p{i}" for i in range(5)])


def test_adt_hint_for_a_layer_stays_layer_counts(tmp_path):
    rna = _rna_adata([f"c{i}" for i in range(30)])
    prot = ad.AnnData(np.random.default_rng(0).normal(size=(30, 5)))
    prot.obs_names = list(rna.obs_names)
    prot.var_names = [f"p{i}" for i in range(5)]
    with pytest.warns(UserWarning, match=r"e\.g\. adt='layer:counts'"):
        mtb.io.export_dataset(rna, tmp_path / "L", adt=prot)


# --- M21: UnitedNet reads the vertical cty.csv --------------------------------
def test_unitednet_runs_on_an_exported_vertical_folder(tmp_path, envs_installed):
    cells = [f"c{i}" for i in range(30)]
    rna = _rna_adata(cells)
    rna.obs["ct"] = ["A", "B"] * 15
    mtb.io.export_dataset(rna, tmp_path / "MYMULTI", atac=_gas_adata(cells),
                          atac_kind="gene_activity", labels="obs:ct", category="vertical")
    assert sorted(p.name for p in (tmp_path / "MYMULTI").iterdir()) == \
        ["atac.h5", "cty.csv", "rna.h5"]
    row = _row("MYMULTI", "vertical", "UnitedNet", tmp_path)
    assert row["files_ok"], row["files_reason"]
    assert list(mtb.labels_for("MYMULTI", "vertical", "UnitedNet", data_path=tmp_path)) == ["cty"]
    got = mtb.inputs_for("MYMULTI", "vertical", "UnitedNet", data_path=tmp_path, check=True)
    assert got["cty"].endswith("cty.csv")
    assert mtb.method_info("UnitedNet")["supports"][0]["labels"] == ["cty"]


def test_unitednet_still_reads_an_older_rna_cty_file(tmp_path):
    cells = [f"c{i}" for i in range(30)]
    d = tmp_path / "OLD"
    d.mkdir()
    _h5(d / "rna.h5", GENES, cells)
    _h5(d / "atac_gas.h5", GENES, cells)
    _labels(d / "rna_cty.csv", 30)
    got = mtb.inputs_for("OLD", "vertical", "UnitedNet", data_path=tmp_path, check=True)
    assert got["cty"] == str(d / "rna_cty.csv")


def test_rna_cty_next_to_atac_cty_is_not_paired_labels(root, tmp_path):
    """In a diagonal folder rna_cty.csv labels the RNA cells only."""
    got = mtb.inputs_for("D28", "vertical", "UnitedNet", data_path=root / "data")
    assert got["cty"].endswith("/D28/cty.csv")
    row = _row("D28", "vertical", "UnitedNet", root / "data")
    assert not row["files_ok"] and "cty.csv is missing." in row["reason"]


def test_unitednet_reason_names_atac_first_then_cty(tmp_path):
    d = tmp_path / "RNAONLY"
    d.mkdir()
    _h5(d / "rna.h5", GENES, [f"c{i}" for i in range(30)])
    row = _row("RNAONLY", "vertical", "UnitedNet", tmp_path)
    assert row["reason"].startswith("UnitedNet needs gene-activity ATAC (atac.h5), which is "
                                    "not in the folder. cty.csv is missing.")


# --- M22: caveats lead with the problem ---------------------------------------
def test_every_caveat_leads_with_its_warning():
    heads = {
        resolve.PEAK_IN_GAS_CAVEAT: "expects gene activity",
        resolve.PEAK_FED_TO_GAS_CAVEAT: "expects gene activity",
        resolve.GAS_FED_TO_PEAK_CAVEAT: "expects peaks",
        resolve.NOT_COUNTS_CAVEAT: "expects raw counts",
        resolve.DIAGONAL_CTY_CAVEAT: "needs rna_cty.csv",
        resolve.UNUSED_BATCHES_CAVEAT: "reads batches",
    }
    for text, head in heads.items():
        assert head in text[:30], text


def test_compact_cli_table_keeps_expects_gene_activity(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())
    cells = [f"c{i}" for i in range(30)]
    d = tmp_path / "MYMULTIOME"
    d.mkdir()
    _h5(d / "rna.h5", GENES, cells)
    _h5(d / "atac.h5", PEAKS, cells)
    _labels(d / "cty.csv", 30)
    rc = cli.main(["scan", "MYMULTIOME", "--category", "vertical", "--data-path", str(tmp_path),
                   "--modalities", "rna,atac"])
    out = capsys.readouterr().out
    assert rc == 0
    line = next(ln for ln in out.splitlines() if ln.strip().startswith("Matilda"))
    assert "expects gene activity" in line
    assert "resolved to a peak matrix" not in out


# --- M31: a variant that reads fewer batches than the folder holds ------------
def test_labels_for_uinmf_returns_the_batches_it_reads(root):
    assert list(mtb.labels_for("D52", "cross", "UINMF", data_path=root / "data")) == \
        ["cty1", "cty2"]
    assert list(mtb.labels_for("D52", "cross", "StabMap", data_path=root / "data")) == \
        ["cty3", "cty1", "cty2"]
    assert list(mtb.labels_for("D52", data_path=root / "data")) == ["cty1", "cty2", "cty3"]


def test_scan_caveat_names_the_unused_batch(root):
    df = mtb.scan("D52", "cross", data_path=root / "data", verbose=False)
    by = df.set_index("method")["caveat"]
    assert "reads batches 1-2 of 3; batch 3 is not used" in by["UINMF"]
    others = [m for m in by.index if m != "UINMF"]
    assert not any("reads batches" in by[m] for m in others)


def test_dry_run_prints_the_unused_batch(root, capsys):
    mtb.run_all("D52", "cross", methods=["UINMF", "StabMap"], data_path=root / "data",
                dry_run=True)
    out = capsys.readouterr().out
    assert "UINMF reads batches 1-2 of 3; batch 3 is not used" in out
    assert "StabMap reads batches" not in out


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
    assert 'd = mtb.config.DEFAULT.data_path / "D11"' in doc
    assert 'rna = mtb.io.read_canonical(d / "rna.h5")' in doc
    assert 'adt = mtb.io.read_canonical(d / "adt.h5", sparse=False)' in doc
    assert '"data/D11/rna.h5"' not in doc


def test_cli_dry_run_prints_the_unused_batch(root, capsys, tmp_path):
    rc = cli.main(["run-all", "D52", "--category", "cross", "--data-path", str(root / "data"),
                   "--dry-run", "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    err = capsys.readouterr().err
    assert "# UINMF reads batches 1-2 of 3; batch 3 is not used" in err
    assert "StabMap reads batches" not in err


def test_run_all_prints_the_unused_batch(root, tmp_path, capsys, monkeypatch):
    """A real sweep says it too; the method run itself is stubbed out."""
    every = frozenset(envs.group_for(m) for m in mtb.list_methods())
    monkeypatch.setattr(workflow, "_installed_envs", lambda: every)

    def fake_run(**kw):
        raise RuntimeError("stub: not run in tests")
    monkeypatch.setattr(workflow, "_run", fake_run)
    mtb.run_all("D52", "cross", tmp_path / "out", methods=["UINMF"],
                data_path=root / "data", evaluate=False)
    assert "[run_all]   UINMF reads batches 1-2 of 3; batch 3 is not used" in capsys.readouterr().out


def test_to_canonical_gas_for_vertical_keeps_its_order(tmp_path, capsys):
    atac = [f"a{i}" for i in range(40)]
    d = _diagonal(tmp_path, "VERT", atac, atac)
    shuffled = list(np.random.default_rng(3).permutation(atac))
    out = mtb.io.to_canonical(_gas_adata(shuffled), d, modality="gas", category="vertical")
    assert out == d / "atac.h5" and _barcodes(out) == shuffled
    assert capsys.readouterr().err == ""


def test_no_dataset_keyed_caveats_remain():
    # R3-04: GLUE's D28 peak-name caveat became a content check
    assert not hasattr(workflow, "_CAVEATS")
