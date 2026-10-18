"""What scan and the dry run say about a folder: one action per problem."""
import numpy as np
import pandas as pd
import pytest

import multibench as mtb

ad = pytest.importorskip("anndata")


def test_peaks_under_the_vertical_name_are_told_the_diagonal_name(tmp_path, capsys):
    # Z35 (r3): "needs peak ATAC (atac_peak.h5), and the folder has peaks (atac.h5)"
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"ct": ["A", "B"] * 15}, index=[f"c{i}" for i in range(30)])
    rna = ad.AnnData(rng.poisson(1, (30, 20)).astype("float32"), obs=obs,
                     var=pd.DataFrame(index=[f"g{i}" for i in range(20)]))
    atac = ad.AnnData(rng.poisson(1, (30, 25)).astype("float32"), obs=obs.copy(),
                      var=pd.DataFrame(index=[f"chr1:{i * 100}-{i * 100 + 50}"
                                              for i in range(25)]))
    mtb.io.export_dataset(rna, tmp_path / "MULTI", atac=atac, labels="obs:ct",
                          category="vertical")
    capsys.readouterr()
    mtb.scan(str(tmp_path / "MULTI"), "diagonal")
    out = capsys.readouterr().out
    assert "[scan] GLUE: GLUE reads atac_peak.h5: rename atac.h5 to atac_peak.h5." in out
    assert "the folder has peaks" not in out


def _mosaic(root, n):
    from tests.test_study_r4_wf import _h5
    d = root / f"MOS{n}"
    d.mkdir(parents=True)
    for f, b in (("rna1", 1), ("rna2", 2), ("atac2", 2), ("atac3", 3)):
        feats = ([f"g{i}" for i in range(5)] if f.startswith("rna")
                 else [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(5)])
        _h5(d / f"{f}.h5", feats, [f"b{b}c{i}" for i in range(n)])
    for b in (1, 2, 3):
        pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / f"cty{b}.csv", index=False)
    return d


@pytest.mark.parametrize("n, shown", [(40, True), (1300, False)])
def test_a_size_hint_shows_only_for_data_below_its_size(tmp_path, n, shown):
    # Z11 (r3): Multigrate's per-batch note is checked against the folder
    d = _mosaic(tmp_path, n)
    plan = mtb.run_all(str(d), "mosaic", methods=["Multigrate"], dry_run=True, verbose=False)
    assert ("needs about 1,300 cells in every batch" in plan["caveat"].iloc[0]) is shown


def test_every_size_hint_is_checked_on_the_data():
    from multibench.engine import registry
    for s in registry.load():
        for v in s.variants:
            if v.size_hint:
                assert v.min_cells or v.min_cells_per_batch or v.min_genes, s.id
    assert not any(v.size_hint for v in registry.get("scMVP").variants)


def test_a_variant_inside_a_larger_one_is_named_once():
    # X68 (r3): the dry run's one line says which row runs; no row repeats it.
    # 0.4.0: the nested row itself is left out, as a run leaves it out
    plan = mtb.run_all("D52mini", "cross", methods=["UINMF"], dry_run=True, verbose=False)
    assert "rna1+rna2+adt1+adt2" not in set(plan["modalities"])
    assert not plan["caveat"].str.contains("so this row is not run").any()
