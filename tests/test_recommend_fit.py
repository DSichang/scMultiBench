"""recommend says which datasets each score comes from (study round 1, L16).

Before the fix ``recommend('vertical', modalities=['rna', 'atac'])`` ranked
moETM, scMM and scMoMaT at coverage 1.0 on D11 alone, a CITE-seq (RNA+ADT)
dataset, and nothing said so. 0.3.x warned; since 0.4 it raises.
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


def test_no_modality_warning_when_the_data_measured_it():
    for call in (lambda: mtb.recommend("vertical", modalities=["rna", "adt"]),
                 lambda: mtb.recommend("diagonal", modalities=["rna", "atac"]),
                 lambda: mtb.recommend("vertical")):
        _, msgs = _quiet(call)
        assert not any("datasets measured" in m for m in msgs), msgs


def test_recommend_has_the_six_columns():
    # 0.4.0 dropped n_datasets, n_datasets_total, coverage and datasets: with one
    # stored dataset per category they were constant (the error above names it)
    r, _ = _quiet(mtb.recommend, "vertical")
    assert list(r.columns) == ["method", "grand_score", "needs_labels", "runtime_tier",
                               "worst_sec", "env"]
    assert r.grand_score.notna().any()


def test_a_dataset_of_unknown_modalities_is_not_judged():
    rows = []
    for m, v in (("totalVI", 0.9), ("Matilda", 0.7), ("sciPENN", 0.5)):
        for metric in ("ARI", "NMI"):
            rows.append({"metric": metric, "value": v, "method": m, "dataset": "MINE"})
    _, msgs = _quiet(mtb.recommend, "vertical", modalities=["rna", "atac"],
                     long_df=pd.DataFrame(rows), metrics=["ARI", "NMI"])
    assert not any("datasets measured" in m for m in msgs), msgs
