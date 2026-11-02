"""Review round 4 of 0.3.5: the raw-count warning for every selector, and wording.

- The warning for values that are not whole numbers named ``'layer:counts'``
  for a matrix read from ``obsm`` and for a feature filter. A layer belongs
  to ``X``, so the suggested call wrote the main matrix of the object (the
  RNA) as the ATAC file, or the features of every modality. The warning now
  names no selector.
- The refusal of one matrix for two modalities.
- ``adt_names`` given a string.
- On the command line the diagonal label error began with ``labels='...'``.
- The Changes page said that ``adt='layer:counts'`` is refused.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench import config
from multibench.engine import ingest

from tests.test_review3_fixes import (CELLS, COUNTS, DEBUG, LOOKS, N, OWN, SAME, _adt, _atac,
                                      _dense, _refused, _rna, _two_cohorts, _warned)

ad = pytest.importorskip("anndata")


def _table(a, *, counts=False):
    return pd.DataFrame(np.asarray(a.layers["counts"] if counts else a.X),
                        index=a.obs_names, columns=a.var_names)


def _cite():
    """RNA in X with its counts layer; ADT and peaks in obsm, each normalised
    and as counts."""
    a, adt, atac = _rna(), _adt(), _atac()
    a.obsm["protein"], a.obsm["protein_counts"] = _table(adt), _table(adt, counts=True)
    a.obsm["peaks"], a.obsm["peak_counts"] = _table(atac), _table(atac, counts=True)
    return a


def _tenx(col="feature_types"):
    """One X with the features of three modalities, as a 10x file has them."""
    parts = [_rna(), _adt(), _atac()]
    a = ad.AnnData(np.hstack([np.asarray(p.X) for p in parts]))
    a.obs_names = CELLS
    a.var_names = [v for p in parts for v in p.var_names]
    a.var[col] = ["Gene Expression"] * 40 + ["Antibody Capture"] * 8 + ["Peaks"] * 30
    a.layers["counts"] = np.hstack([p.layers["counts"] for p in parts])
    a.obs["celltype"] = parts[0].obs["celltype"].to_numpy()
    return a


def _in_obsm(what):
    """The whole warning, for any selector."""
    return COUNTS.format(what=what)


# --------------------------------------------- 1: a matrix read from obsm
def test_atac_in_obsm_is_not_told_to_read_a_layer(tmp_path):
    """0.3.4 and the first 0.3.5 candidate named ``atac='layer:counts'``,
    the counts layer of the RNA."""
    cite = _cite()
    kw = dict(rna="layer:counts", atac_kind="peak", labels="obs:celltype")
    msgs = _warned(ingest.export_dataset, cite, tmp_path / "A", atac="obsm:peaks", **kw)
    assert msgs == [_in_obsm("atac")]
    # the entry that holds the counts, named: the ATAC counts are written
    assert _warned(ingest.export_dataset, cite, tmp_path / "B", atac="obsm:peak_counts",
                   **kw) == []
    assert np.array_equal(_dense(tmp_path / "B" / "atac.h5"), _atac().layers["counts"])
    # what the old hint wrote, for the record: the layer of the RNA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = ingest.export_dataset(cite, tmp_path / "C", rna=None, atac="layer:counts",
                                    atac_kind="peak")
    assert np.array_equal(_dense(old / "atac.h5"), _rna().layers["counts"])


def test_rna_in_obsm_is_not_told_to_read_a_layer(tmp_path):
    a = _adt(log=False)
    a.obsm["rna"] = _table(_rna())
    msgs = _warned(ingest.export_dataset, a, tmp_path / "A", rna="obsm:rna")
    assert msgs == [_in_obsm("rna")]


def test_adt_in_obsm_keeps_its_hint_and_the_hint_works(tmp_path):
    cite = _cite()
    msgs = _warned(ingest.export_dataset, cite, tmp_path / "A", rna="layer:counts",
                   adt="obsm:protein")
    assert msgs == [_in_obsm("adt")]
    assert _warned(ingest.export_dataset, cite, tmp_path / "B", rna="layer:counts",
                   adt="obsm:protein_counts") == []
    assert np.array_equal(_dense(tmp_path / "B" / "adt.h5"), _adt().layers["counts"])


def test_the_entry_the_adt_hint_names_is_not_named_to_itself(tmp_path):
    """``adt='obsm:protein_counts'`` with normalised values in that entry."""
    cite = _cite()
    cite.obsm["protein_counts"] = cite.obsm["protein"]
    msgs = _warned(ingest.export_dataset, cite, tmp_path / "A", rna="layer:counts",
                   adt="obsm:protein_counts")
    assert msgs == [_in_obsm("adt")]


# ------------------------------------------------- 1: a feature filter
@pytest.mark.parametrize("col", ["feature_types", "kind"])
@pytest.mark.parametrize("what, value, part", [("rna", "Gene Expression", _rna),
                                               ("adt", "Antibody Capture", _adt),
                                               ("atac", "Peaks", _atac)])
def test_the_hint_keeps_the_feature_filter(tmp_path, col, what, value, part):
    """Without the filter the layer holds the features of every modality:
    0.3.4 wrote all 78 of them as the file of one."""
    tenx = _tenx(col)
    kw = dict(atac_kind="peak") if what == "atac" else {}
    kw.update(rna=None) if what != "rna" else None
    msgs = _warned(ingest.export_dataset, tenx, tmp_path / "A", **{what: f"X[{col}={value}]"},
                   **kw)
    hinted = f"layer:counts[{col}={value}]"
    assert msgs == [_in_obsm(what)]
    assert _warned(ingest.export_dataset, tenx, tmp_path / "B", **{what: hinted}, **kw) == []
    name = {"rna": "rna.h5", "adt": "adt.h5", "atac": "atac.h5"}[what]
    assert np.array_equal(_dense(tmp_path / "B" / name), part().layers["counts"])


# ------------------------------------------------------ 1: a MuData
def test_the_hints_of_a_mudata_keep_the_modality(tmp_path):
    md = pytest.importorskip("mudata")
    rna = _cite()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = md.MuData({"rna": rna, "all": _tenx()})
    kw = dict(rna="mod:rna.layer:counts")
    msgs = _warned(ingest.export_dataset, m, tmp_path / "A", adt="mod:rna.obsm:protein", **kw)
    assert msgs == [_in_obsm("adt")]
    assert _warned(ingest.export_dataset, m, tmp_path / "B",
                   adt="mod:rna.obsm:protein_counts", **kw) == []
    assert np.array_equal(_dense(tmp_path / "B" / "adt.h5"), _adt().layers["counts"])
    msgs = _warned(ingest.export_dataset, m, tmp_path / "C", atac="mod:rna.obsm:peaks",
                   atac_kind="peak", **kw)
    assert msgs == [_in_obsm("atac")]
    msgs = _warned(ingest.export_dataset, m, tmp_path / "D",
                   adt="mod:all[feature_types=Antibody Capture]", **kw)
    hinted = "mod:all.layer:counts[feature_types=Antibody Capture]"
    assert msgs == [_in_obsm("adt")]
    assert _warned(ingest.export_dataset, m, tmp_path / "E", adt=hinted, **kw) == []
    assert np.array_equal(_dense(tmp_path / "E" / "adt.h5"), _adt().layers["counts"])


# ------------------------------------------- 3(a): data passed once more
def test_data_passed_as_the_second_modality_is_named_as_such(tmp_path):
    """The text names the two arguments as they were written."""
    rna = _rna()
    assert _refused(tmp_path, rna, rna="X", atac=rna, atac_kind="gene_activity") == (
        "rna='X' and atac=" + SAME + OWN)
    assert _refused(tmp_path, rna, rna=rna, adt=rna) == "rna= and adt=" + SAME + OWN
    assert _refused(tmp_path, rna, rna=rna, atac="X", atac_kind="gene_activity") == (
        "rna= and atac='X'" + SAME + OWN)


# ------------------------------------------------ 2: what is refused and what is not
def test_a_layer_of_the_rna_object_passes_as_another_modality(tmp_path):
    # not solved in 0.3.5: the layer is written as that modality, without a message
    rna = _rna(log=False)
    rna.layers["counts"] = rna.layers["counts"] + 1      # told apart from X
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ingest.export_dataset(rna, tmp_path / "OUT", rna="X", adt="layer:counts")
        gas = ingest.export_dataset(rna, tmp_path / "GAS", rna="X", atac="layer:counts",
                                    atac_kind="gene_activity")
    assert np.array_equal(_dense(out / "adt.h5"), rna.layers["counts"])
    assert np.array_equal(_dense(gas / "atac_gas.h5"), rna.layers["counts"])
    assert list(ingest.read(out / "adt.h5").var_names) == list(rna.var_names)
    # the two selectors that get the hint
    assert _warned(ingest.export_dataset, _cite(), tmp_path / "A", rna=None,
                   atac="obsm:peaks", atac_kind="peak") == [_in_obsm("atac")]
    assert _warned(ingest.export_dataset, _tenx(), tmp_path / "B", rna=None,
                   adt="X[feature_types=Antibody Capture]") == [_in_obsm("adt")]
