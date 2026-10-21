"""Fix round 6, integration: behaviour that needs two work packages at once.

- eval + wf: ``run-all --batch SHEET --batch-column COL`` saves the column it
  scored with (``batch_<hash>.csv``, the same file a one-column barcode CSV
  gives), and ``rescore()`` reuses it.
- eval + cli: ``evaluate --name`` without ``--method`` takes the batch from a
  sample sheet with ``--batch-column``.
- cli + msg: the dry run of ``multibench run`` prints its header, then the
  missing-input note, then the setup note as a sentence, then the command.
"""
import re
import warnings

import pandas as pd
import pytest

import multibench as mtb
from tests.test_study_r4_wf import _quiet
from tests.test_study_r5_wf import (BARS, CELLTYPE, SAMPLES, _adata,  # noqa: F401
                                    _barcode_csv)

pytest.importorskip("scib")


# --------------------------------------------------------------- integration repairs
# wf R6-02 x R6-05: rescore reuses the batch run saved. That batch is not
# one the caller gave, so a metrics= without a batch metric must not warn
# 'batch= changes nothing here' (the first Example of RunResult.rescore).
from tests.test_study_r4_wf import N, _batch_metrics, cite  # noqa: E402,F401
from multibench.eval import pipeline  # noqa: E402

_KW = dict(methods=["Matilda"], modalities=["rna", "adt"], verbose=False)


def test_rescore_with_the_saved_batch_and_no_batch_metric_does_not_warn(cite, tmp_path):
    data, batch, shuffled = cite
    mtb.config.data_path = data
    res = _quiet(mtb.run, 'MYCITE', 'vertical', tmp_path / 'out', batch=shuffled, **_KW)
    back = mtb.load_run(tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # the toy cells give no resolution with one cluster per cell type
        warnings.filterwarnings("ignore", message="No Leiden resolution gave")
        warnings.filterwarnings("ignore", message="ARI_batch and NMI_batch use")
        new = back.rescore(metrics=["ARI", "NMI"])
    rec = new.records[0]
    assert rec["status"] == "scored"
    # c1 (C10): ARI and NMI are scored again, the other saved scores stay
    assert set(rec["metrics"]) == set(back.records[0]["metrics"])
    assert round(float(new.summary["ARI"].iloc[0]), 4) == round(
        float(res.summary["ARI"].iloc[0]), 4)
    # the record still says which batch it has, and a later batch metric finds it
    assert (rec["batch_source"], rec["n_batches"]) == ("user", 2)
    assert rec["batch_file"] == res.records[0]["batch_file"]
    new.save(tmp_path / "ari_only")
    again = _quiet(mtb.load_run(tmp_path / "ari_only").rescore)
    assert _batch_metrics(again) == _batch_metrics(res)
    # a batch the caller gives still warns, as R6-05 keeps
    with pytest.warns(UserWarning, match=r"^batch= changes nothing here, because "
                                         r"metrics=\['ARI'\] has no batch metric"):
        back.rescore(batch=batch, metrics=["ARI"])


def test_the_batch_label_errors_name_only_metrics_the_family_computes():
    import numpy as np
    rng = np.random.default_rng(0)
    emb, ct = rng.normal(size=(80, 4)), np.array(["A", "B"] * 40)
    bat = np.array(["x"] * 40 + ["y"] * 40)
    for fam in ("all", "batch"):
        with pytest.raises(ValueError) as exc:
            pipeline.evaluate(emb, labels=ct, metrics=fam)
        named = re.search(r"needs batch labels for (.*?)\. ", str(exc.value)).group(1)
        named = set(re.split(r", | and ", named))
        # 0.4.0: evaluate returns the long table, one row per metric
        computed = set(_quiet(pipeline.evaluate, emb, labels=ct, batch=bat,
                              metrics=fam, verbose=False)["metric"])
        assert named and named <= computed, (fam, named, computed)
