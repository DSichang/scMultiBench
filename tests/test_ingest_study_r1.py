"""export_dataset / convert fixes from student study round 1 (ledger L01, L03-L09).

Each test pins a behaviour a student hit: a log-normalised X accepted silently
(L01), batch= folders that vertical scans read as batch 1 only (L03), unpaired
diagonal data refused with an empty folder left behind (L04), silent overwrites
(L05), mosaic ATAC written under a name no mosaic method reads (L06), no way to
write one file as batch N (L07), genes and peaks in one X (L08) and MuData
labels in the global obs (L09).
"""
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli
from multibench.engine import ingest
from multibench.engine import resolve

ad = pytest.importorskip("anndata")

GENES = [f"g{i}" for i in range(30)]
PEAKS = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(50)]


def _counts(n, n_feat, seed, lam=1.0):
    return np.random.default_rng(seed).poisson(lam, size=(n, n_feat)).astype(float)


def _rna(n=120, prefix="r", seed=0):
    a = ad.AnnData(_counts(n, len(GENES), seed))
    a.var_names = GENES
    a.obs_names = [f"{prefix}{i}" for i in range(n)]
    a.obs["cell_type"] = np.random.default_rng(seed).choice(["T", "B"], n)
    return a


def _atac(n=90, prefix="a", seed=1):
    a = ad.AnnData(_counts(n, len(PEAKS), seed, lam=0.5))
    a.var_names = PEAKS
    a.obs_names = [f"{prefix}{i}" for i in range(n)]
    a.obs["cell_type"] = np.random.default_rng(seed).choice(["T", "B"], n)
    return a


def _cite(n=90, seed=0, batches=3):
    a = _rna(n, "c", seed)
    a.obsm["protein"] = pd.DataFrame(_counts(n, 6, seed + 5, lam=4.0), index=a.obs_names,
                                     columns=[f"CD{i}" for i in range(6)])
    a.obs["batch"] = np.repeat([f"s{i}" for i in range(batches)], n // batches)
    return a


def _n_cells(path):
    with h5py.File(path) as f:
        return f["matrix/barcodes"].shape[0]


# ------------------------------------------------------------------ L01 raw counts
def test_log_normalised_rna_warns_on_export_and_to_canonical(tmp_path):
    a = _cite()
    a.layers["counts"] = a.X.copy()
    a.X = np.log1p(a.X)
    with pytest.warns(UserWarning, match=r"rna values are not whole numbers .*"
                                         r"rna='layer:counts' \(MuData: "
                                         r"rna='mod:rna.layer:counts'\)"):
        ingest.export_dataset(a, tmp_path / "LOG", adt="obsm:protein", labels="obs:cell_type")
    with pytest.warns(UserWarning, match=r"rna values are not whole numbers .*layer='counts'"):
        ingest.to_canonical(a, tmp_path / "x.h5", modality="rna")
    # raw counts: no warning at all
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ingest.export_dataset(a, tmp_path / "RAW", rna="layer:counts", adt="obsm:protein",
                              labels="obs:cell_type")
        ingest.to_canonical(a, tmp_path / "y.h5", modality="rna", layer="counts")


def test_gene_activity_scores_are_not_count_checked(tmp_path):
    g = ad.AnnData(np.random.default_rng(0).random((20, 10)))
    g.var_names = [f"G{i}" for i in range(10)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ingest.export_dataset(g, tmp_path / "G", rna=None, atac="X",
                              atac_kind="gene_activity", category="diagonal")


def test_scan_caveat_names_the_non_integer_file(tmp_path):
    a = _cite()
    a.X = np.log1p(a.X)
    with pytest.warns(UserWarning, match="not whole numbers"):
        ingest.export_dataset(a, tmp_path / "LOG", adt="obsm:protein", labels="obs:cell_type")
    sc = mtb.scan("LOG", "vertical", data_path=tmp_path, modalities=["rna", "adt"])
    assert sc["files_ok"].all()                      # a caveat, not a file failure
    assert sc["caveat"].str.contains(
        "rna.h5 holds non-integer values; methods expect raw counts").all()
    assert not sc["caveat"].str.contains("adt.h5").any()
    b = _cite()
    ingest.export_dataset(b, tmp_path / "INT", adt="obsm:protein", labels="obs:cell_type")
    sc = mtb.scan("INT", "vertical", data_path=tmp_path, modalities=["rna", "adt"])
    assert not sc["caveat"].str.contains("non-integer").any()


@pytest.mark.parametrize("ds,cat", [("D11", "vertical"), ("D28", "diagonal"),
                                    ("D45", "mosaic"), ("D46", "mosaic"), ("D52", "cross")])
def test_demo_folders_hold_raw_counts(root, ds, cat):
    sc = mtb.scan(ds, cat, data_path=root / "data")
    assert not sc["caveat"].str.contains("non-integer").any()


# ------------------------------------------------------------------ L03 per-batch folders
def test_batch_with_vertical_or_diagonal_is_refused_before_writing(tmp_path):
    a = _cite()
    for cat, read in (("vertical", "one rna.h5"), ("diagonal", "one rna.h5 and one ATAC file")):
        with pytest.raises(ValueError, match=f"{cat} methods read {read}: export without "
                                             r"batch= and pass the batch column to "
                                             r"evaluate\(batch=\.\.\.\)") as ei:
            ingest.export_dataset(a, tmp_path / cat, adt="obsm:protein", batch="obs:batch",
                                  category=cat)
        assert not (tmp_path / cat).exists()
        # cross is RNA+ADT: never offered to a diagonal (RNA+ATAC) user
        assert ("category='cross'" in str(ei.value)) == (cat == "vertical")


def test_batch_with_atac_and_no_category_warns(tmp_path):
    m = _rna(60)
    m.obsm["peaks"] = pd.DataFrame(_counts(60, len(PEAKS), 3), index=m.obs_names, columns=PEAKS)
    m.obs["batch"] = np.repeat(["x", "y"], 30)
    with pytest.warns(UserWarning, match="only mosaic methods read numbered ATAC files"):
        ingest.export_dataset(m, tmp_path / "B", atac="obsm:peaks", atac_kind="peak",
                              batch="obs:batch")


def test_mudata_docstring_example_is_vertical_without_batch():
    doc = ingest.export_dataset.__doc__
    assert 'batch="rna:sample"' not in doc
    assert 'mtb.io.export_dataset(mdata, "data/MYMULTIOME", rna="rna", atac="atac",' in doc


# ------------------------------------------------------------------ L04 unpaired diagonal
def test_diagonal_export_writes_unpaired_rna_and_atac(tmp_path):
    rna, atac = _rna(120), _atac(90)
    d = ingest.export_dataset(rna, tmp_path / "LUNG", atac=atac, atac_kind="peak",
                              labels="obs:cell_type", category="diagonal")
    assert sorted(os.listdir(d)) == ["atac_cty.csv", "atac_peak.h5", "rna.h5", "rna_cty.csv"]
    assert _n_cells(d / "rna.h5") == 120 and _n_cells(d / "atac_peak.h5") == 90
    assert pd.read_csv(d / "rna_cty.csv")["x"].tolist() == rna.obs["cell_type"].tolist()
    assert pd.read_csv(d / "atac_cty.csv")["x"].tolist() == atac.obs["cell_type"].tolist()
    sc = mtb.scan("LUNG", "diagonal", data_path=tmp_path)
    assert sc.loc[sc["method"] == "GLUE", "files_ok"].item()
    # gene activity goes to atac_gas.h5
    g = _atac(90)
    g.var_names = [f"G{i}" for i in range(len(PEAKS))]
    d2 = ingest.export_dataset(rna, tmp_path / "GAS", atac=g, atac_kind="gene_activity",
                               labels="obs:cell_type", category="diagonal")
    assert "atac_gas.h5" in os.listdir(d2)


def test_diagonal_label_column_missing_names_the_object(tmp_path):
    rna, atac = _rna(), _atac()
    del atac.obs["cell_type"]
    with pytest.raises(KeyError, match="not in obs of the ATAC object"):
        ingest.export_dataset(rna, tmp_path / "L", atac=atac, atac_kind="peak",
                              labels="obs:cell_type", category="diagonal")
    assert not (tmp_path / "L").exists()


def test_disjoint_barcodes_point_at_diagonal_and_leave_no_folder(tmp_path):
    with pytest.raises(ValueError, match="RNA and ATAC from different cells is diagonal "
                                         "integration: pass category='diagonal'"):
        ingest.export_dataset(_rna(), tmp_path / "X", atac=_atac(), atac_kind="peak",
                              labels="obs:cell_type")
    assert not (tmp_path / "X").exists()
    # a failed selector leaves nothing on disk either
    with pytest.raises(KeyError):
        ingest.export_dataset(_rna(), tmp_path / "Y", adt="obsm:nope")
    assert not (tmp_path / "Y").exists()


# ------------------------------------------------------------------ L05 no silent overwrite
def test_diagonal_label_file_is_named_after_the_modality(tmp_path):
    ingest.export_dataset(None, tmp_path / "LC", atac=_atac(), atac_kind="peak",
                          labels="obs:cell_type", category="diagonal")
    assert sorted(os.listdir(tmp_path / "LC")) == ["atac_cty.csv", "atac_peak.h5"]
    ingest.export_dataset(_rna(), tmp_path / "LC", labels="obs:cell_type", category="diagonal")
    assert sorted(os.listdir(tmp_path / "LC")) == ["atac_cty.csv", "atac_peak.h5", "rna.h5",
                                                   "rna_cty.csv"]
    assert len(pd.read_csv(tmp_path / "LC" / "atac_cty.csv")) == 90


def test_existing_files_raise_unless_overwrite(tmp_path):
    a = _cite()
    ingest.export_dataset(a, tmp_path / "D", adt="obsm:protein", labels="obs:cell_type")
    before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "D").iterdir()}
    with pytest.raises(FileExistsError, match=r"pass overwrite=True to "
                                              r"replace them") as ei:
        ingest.export_dataset(a, tmp_path / "D", adt="obsm:protein", labels="obs:cell_type")
    assert "rna.h5" in str(ei.value) and "cty.csv" in str(ei.value)
    assert {p.name: p.stat().st_mtime_ns for p in (tmp_path / "D").iterdir()} == before
    ingest.export_dataset(a, tmp_path / "D", adt="obsm:protein", labels="obs:cell_type",
                          overwrite=True)


def test_scan_checks_diagonal_label_rows_and_flags_a_lone_cty(tmp_path):
    d = ingest.export_dataset(_rna(), tmp_path / "LC", atac=_atac(), atac_kind="peak",
                              labels="obs:cell_type", category="diagonal")
    pd.DataFrame({"x": ["T"] * 10}).to_csv(d / "atac_cty.csv", index=False)
    sc = mtb.scan("LC", "diagonal", data_path=tmp_path)
    glue = sc[sc["method"] == "GLUE"].iloc[0]
    assert not glue["files_ok"]
    assert "atac_cty.csv has 10 labels but atac_peak.h5 has 90 cells" in glue["files_reason"]
    # a diagonal folder whose only label file is cty.csv: caveat
    (d / "atac_cty.csv").unlink()
    (d / "rna_cty.csv").rename(d / "cty.csv")
    sc = mtb.scan("LC", "diagonal", data_path=tmp_path)
    glue = sc[sc["method"] == "GLUE"].iloc[0]
    assert glue["files_ok"]
    assert ("label files: cty.csv found; diagonal needs rna_cty.csv and atac_cty.csv"
            in glue["caveat"])


def test_cli_success_line_lists_only_what_this_call_wrote(tmp_path, capsys):
    rna, atac = _rna(), _atac()
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    out = tmp_path / "data" / "LUNG"
    rc = cli.main(["convert", str(tmp_path / "rna.h5ad"), str(out), "--rna", "X",
                   "--atac-from", str(tmp_path / "atac.h5ad"), "--atac-kind", "peak",
                   "--labels", "obs:cell_type", "--category", "diagonal"])
    cap = capsys.readouterr()
    assert rc == 0
    assert "(files: atac_cty.csv, atac_peak.h5, rna.h5, rna_cty.csv)" in cap.out
    # a second call refuses, --overwrite replaces only its own files
    rc = cli.main(["convert", str(tmp_path / "rna.h5ad"), str(out), "--rna", "X",
                   "--labels", "obs:cell_type", "--category", "diagonal"])
    assert rc == 1 and "--overwrite" in capsys.readouterr().err
    rc = cli.main(["convert", str(tmp_path / "rna.h5ad"), str(out), "--rna", "X",
                   "--labels", "obs:cell_type", "--category", "diagonal", "--overwrite"])
    cap = capsys.readouterr()
    assert rc == 0 and "(files: rna.h5, rna_cty.csv)" in cap.out
    assert ("# already in the folder: atac_cty.csv, atac_peak.h5 (not written by this call)"
            in cap.err)
    with pytest.raises(SystemExit) as ei:
        cli.main(["convert", str(tmp_path / "rna.h5ad"), str(tmp_path / "Z"), "--rna", "X",
                  "--atac-from", str(tmp_path / "atac.h5ad"), "--atac-kind", "peak"])
    assert ei.value.code == 2 and "--atac-from is for --category diagonal" in \
        capsys.readouterr().err


# ------------------------------------------------------------------ L06 / L07 mosaic
def _mosaic_batches():
    cite = _cite(90, seed=0, batches=1)
    mo = _rna(80, "m", seed=3)
    mo.obsm["atac"] = pd.DataFrame(_counts(80, len(PEAKS), 4, 0.5), index=mo.obs_names,
                                   columns=PEAKS)
    return cite, mo, _rna(60, "o", seed=5)


def test_batch_index_writes_the_d46_pattern_and_scan_finds_it(tmp_path):
    cite, mo, ro = _mosaic_batches()
    kw = dict(labels="obs:cell_type", category="mosaic")
    ingest.export_dataset(cite, tmp_path / "LAB", adt="obsm:protein", batch_index=1, **kw)
    ingest.export_dataset(mo, tmp_path / "LAB", atac="obsm:atac", atac_kind="peak",
                          batch_index=2, **kw)
    ingest.export_dataset(ro, tmp_path / "LAB", batch_index=3, **kw)
    assert sorted(os.listdir(tmp_path / "LAB")) == [
        "adt1.h5", "atac2.h5", "cty1.csv", "cty2.csv", "cty3.csv",
        "rna1.h5", "rna2.h5", "rna3.h5"]
    assert _n_cells(tmp_path / "LAB" / "rna2.h5") == 80
    sc = mtb.scan("LAB", "mosaic", data_path=tmp_path)
    ok = sc[sc["method"].isin(["StabMap", "scMoMaT"])]
    assert len(ok) == 2 and ok["files_ok"].all(), ok["files_reason"].tolist()
    # a loop cannot mix batches: the same index again refuses
    with pytest.raises(FileExistsError, match="rna3.h5"):
        ingest.export_dataset(ro, tmp_path / "LAB", batch_index=3, **kw)


def test_batch_index_argument_checks(tmp_path):
    ro = _rna(20)
    with pytest.raises(ValueError, match="pass batch= or batch_index=, not both"):
        ingest.export_dataset(ro, tmp_path / "B", batch="obs:cell_type", batch_index=1,
                              category="mosaic")
    with pytest.raises(ValueError, match="only mosaic and cross methods read"):
        ingest.export_dataset(ro, tmp_path / "B", batch_index=1)
    with pytest.raises(ValueError, match="positive integer"):
        ingest.export_dataset(ro, tmp_path / "B", batch_index=0, category="cross")
    assert not (tmp_path / "B").exists()


def test_mosaic_batch_split_writes_atac_i(tmp_path, capsys):
    m = _rna(90)
    m.obsm["peaks"] = pd.DataFrame(_counts(90, len(PEAKS), 7, 0.3), index=m.obs_names,
                                   columns=PEAKS)
    m.obsm["protein"] = pd.DataFrame(_counts(90, 6, 8, 4.0), index=m.obs_names,
                                     columns=[f"CD{i}" for i in range(6)])
    m.obs["batch"] = np.repeat(["x", "y", "z"], 30)
    m.write_h5ad(tmp_path / "mos.h5ad")
    rc = cli.main(["convert", str(tmp_path / "mos.h5ad"), str(tmp_path / "MOS"), "--rna", "X",
                   "--adt", "obsm:protein", "--atac", "obsm:peaks", "--atac-kind", "peak",
                   "--labels", "obs:cell_type", "--batch", "obs:batch", "--category", "mosaic"])
    assert rc == 0
    names = sorted(os.listdir(tmp_path / "MOS"))
    assert {"atac1.h5", "atac2.h5", "atac3.h5"} <= set(names)
    assert not any(n.startswith("atac_peak") for n in names)
    # the folder holds the D46 pattern (rna1-3, adt1, atac2): scan finds it
    sc = mtb.scan("MOS", "mosaic", data_path=tmp_path)
    ok = sc[sc["method"].isin(["StabMap", "scMoMaT"])]
    assert len(ok) == 2 and ok["files_ok"].all(), ok["files_reason"].tolist()
    with pytest.warns(UserWarning, match="every mosaic method reads peaks"):
        ingest.export_dataset(m, tmp_path / "G", rna=None, atac="obsm:peaks",
                              atac_kind="gene_activity", batch="obs:batch", category="mosaic")


# ------------------------------------------------------------------ L08 var filter
def _arc(n=50):
    rng = np.random.default_rng(9)
    a = ad.AnnData(np.hstack([rng.poisson(1, (n, len(GENES))),
                              rng.poisson(0.3, (n, len(PEAKS)))]).astype(float))
    a.var_names = GENES + PEAKS
    a.var["feature_types"] = ["Gene Expression"] * len(GENES) + ["Peaks"] * len(PEAKS)
    a.obs["ct"] = rng.choice(["T", "B"], n)
    return a


def test_var_filter_splits_genes_and_peaks(tmp_path):
    d = ingest.export_dataset(_arc(), tmp_path / "ARC", rna="X[feature_types=Gene Expression]",
                              atac="X[feature_types=Peaks]", atac_kind="peak",
                              labels="obs:ct", category="vertical")
    assert sorted(os.listdir(d)) == ["atac.h5", "cty.csv", "rna.h5"]
    assert list(ingest.read_canonical(d / "rna.h5").var_names) == GENES
    assert list(ingest.read_canonical(d / "atac.h5").var_names) == PEAKS
    with pytest.raises(ValueError, match=r"no feature has var\['feature_types'\] == 'Genes'; "
                                         r"the values of var\['feature_types'\] are "
                                         r"\['Gene Expression', 'Peaks'\]"):
        ingest.export_dataset(_arc(), tmp_path / "E", rna="X[feature_types=Genes]")
    with pytest.raises(KeyError, match="var column 'kind' not found"):
        ingest.export_dataset(_arc(), tmp_path / "E", rna="X[kind=Genes]")
    assert not (tmp_path / "E").exists()


def test_var_filter_on_the_command_line(tmp_path):
    _arc().write_h5ad(tmp_path / "arc.h5ad")
    rc = cli.main(["convert", str(tmp_path / "arc.h5ad"), str(tmp_path / "ARC"),
                   "--rna", "X[feature_types=Gene Expression]",
                   "--atac", "X[feature_types=Peaks]", "--atac-kind", "peak",
                   "--labels", "obs:ct", "--category", "vertical"])
    assert rc == 0
    assert ingest.read_canonical(tmp_path / "ARC" / "atac.h5").shape == (50, len(PEAKS))


# ------------------------------------------------------------------ L09 MuData labels
def test_mudata_global_obs_labels(tmp_path):
    mu = pytest.importorskip("mudata")
    r = _rna(40, "c")
    del r.obs["cell_type"]
    t = _atac(40, "c")
    del t.obs["cell_type"]
    m = mu.MuData({"rna": r, "atac": t})
    m.obs["celltype"] = np.random.default_rng(0).choice(["T", "B"], 40)
    with pytest.raises(KeyError) as ei:
        ingest.export_dataset(m, tmp_path / "MU1", rna="rna", atac="atac", atac_kind="peak",
                              labels="rna:celltype", category="vertical")
    msg = str(ei.value)
    assert "searched mdata['rna'].obs" in msg and "pass labels='obs:celltype'" in msg
    assert not (tmp_path / "MU1").exists()
    d = ingest.export_dataset(m, tmp_path / "MU1", rna="rna", atac="atac", atac_kind="peak",
                              labels="obs:celltype", category="vertical")
    assert pd.read_csv(d / "cty.csv")["x"].tolist() == m.obs["celltype"].tolist()
    # muon's prefixed copy in the global obs is found for '<mod>:<col>' too
    m.obs["rna:ct2"] = m.obs["celltype"].values
    d = ingest.export_dataset(m, tmp_path / "MU2", rna="rna", labels="rna:ct2")
    assert (d / "cty.csv").exists()


# ------------------------------------------------------------------ review of round 1
def _unpaired_mudata():
    mu = pytest.importorskip("mudata")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mu.MuData({"rna": _rna(60, "r"), "atac": _atac(50, "a")})
    # the global obs as muon fills it: prefixed copies, NaN for the other side's cells
    for mod in ("rna", "atac"):
        m.obs[f"{mod}:cell_type"] = m[mod].obs["cell_type"].reindex(m.obs_names).values
    return m


@pytest.mark.parametrize("labels", ["rna:cell_type", "atac:cell_type", "cell_type",
                                    "obs:cell_type"])
def test_diagonal_mudata_reads_each_modality_obs(tmp_path, labels):
    m = _unpaired_mudata()
    assert "cell_type" not in m.obs.columns
    d = ingest.export_dataset(m, tmp_path / "MU", rna="mod:rna", atac="mod:atac",
                              atac_kind="peak", labels=labels, category="diagonal")
    assert pd.read_csv(d / "rna_cty.csv")["x"].tolist() == \
        m["rna"].obs["cell_type"].tolist()
    assert pd.read_csv(d / "atac_cty.csv")["x"].tolist() == \
        m["atac"].obs["cell_type"].tolist()


def test_diagonal_mudata_global_obs_labels_with_gaps_are_refused(tmp_path):
    # the global 'rna:cell_type' column is NaN for every ATAC cell
    m = _unpaired_mudata()
    with pytest.raises(ValueError) as ei:
        ingest.export_dataset(m, tmp_path / "MU", rna="mod:rna", atac="mod:atac",
                              atac_kind="peak", labels="obs:rna:cell_type",
                              category="diagonal")
    msg = str(ei.value)
    assert "50 of the 50 labels for the ATAC cells (mdata['atac']) are missing" in msg
    assert "labels='cell_type' reads that column from each modality's own obs" in msg
    assert "labels='obs:<col>'" not in msg
    assert not (tmp_path / "MU").exists()


@pytest.mark.parametrize("category", [None, "vertical", "diagonal", "cross", "mosaic"])
@pytest.mark.parametrize("gap", [np.nan, None, ""])
def test_labels_with_missing_values_are_refused_in_every_category(tmp_path, category, gap):
    a = _cite(60)
    lab = a.obs["cell_type"].astype(object).copy()
    lab.iloc[[3, 7]] = gap
    a.obs["ct"] = lab
    kw = {"batch_index": 1} if category in ("cross", "mosaic") else {}
    with pytest.raises(ValueError, match=r"labels='obs:ct': 2 of the 60 labels for the "
                                         r"(RNA )?cells \((data.obs|data)\) are missing"):
        ingest.export_dataset(a, tmp_path / "GAP", adt="obsm:protein", labels="obs:ct",
                              category=category, **kw)
    assert not (tmp_path / "GAP").exists()
