"""recommend says which datasets each score comes from (study round 1, L16).

Before the fix ``recommend('vertical', modalities=['rna', 'atac'])`` ranked
moETM, scMM and scMoMaT at coverage 1.0 on D11 alone, a CITE-seq (RNA+ADT)
dataset, and nothing said so.
"""
import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.data import results


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(x.message) for x in w]


def test_warns_when_no_ranked_dataset_measured_the_modality():
    with pytest.warns(UserWarning, match=(
            r"stored vertical scores come from D11, D11s \(rna\+adt\); none measured "
            r"atac, so this ranking does not describe RNA\+ATAC data")):
        mtb.recommend("vertical", modalities=["rna", "atac"], source="rerun")
    # atac= names the modality as well
    with pytest.warns(UserWarning, match=r"come from D11 \(rna\+adt\); none measured atac"):
        mtb.recommend("vertical", atac="peak")


def test_no_modality_warning_when_the_data_measured_it():
    for call in (lambda: mtb.recommend("vertical", modalities=["rna", "adt"], source="rerun"),
                 lambda: mtb.recommend("diagonal", modalities=["rna", "atac"], source="rerun"),
                 lambda: mtb.recommend("vertical", source="rerun")):
        _, msgs = _quiet(call)
        assert not any("none measured" in m for m in msgs), msgs


def test_datasets_column_names_where_each_score_comes_from():
    r, _ = _quiet(mtb.recommend, "vertical", source="rerun")
    assert list(r.columns[:5]) == ["method", "grand_score", "n_datasets",
                                   "n_datasets_total", "coverage"]
    assert "datasets" in r.columns
    scored = r[r.grand_score.notna()]
    assert len(scored) > 0
    for row in scored.itertuples():
        ids = row.datasets.split(", ")
        assert set(ids) <= {"D11", "D11s"} and len(ids) == row.n_datasets, row
    assert (r[r.grand_score.isna()].datasets == "").all()


def test_every_stored_dataset_has_its_measured_modalities():
    stored = set(mtb.available_datasets(source="both"))
    assert stored == set(results._STORED_DATASET_MODALITIES)


def test_a_dataset_of_unknown_modalities_is_not_judged():
    rows = []
    for m, v in (("totalVI", 0.9), ("Matilda", 0.7), ("sciPENN", 0.5)):
        for metric in ("ARI", "NMI"):
            rows.append({"metric": metric, "value": v, "method": m, "dataset": "MINE"})
    _, msgs = _quiet(mtb.recommend, "vertical", modalities=["rna", "atac"],
                     long_df=pd.DataFrame(rows), metrics=["ARI", "NMI"])
    assert not any("none measured" in m for m in msgs), msgs
