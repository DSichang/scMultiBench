"""Three messages of the API review (D17).

- run_all on a single-batch diagonal folder listed the multi-batch rows too
  (about 1,700 characters); it lists the rows the folder can feed and counts
  the rest.
- rescore with a batch that has one value among a method's cells said "Pass
  batch=" although a batch was passed; it names the cause.
- bubble(methods=[...]) with package methods absent from the table said "Did
  you mean Matilda and totalVI?" for Matilda and totalVI.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow


def test_requested_rows_the_folder_cannot_feed_are_counted(tmp_path):
    d = tmp_path / "DG"
    d.mkdir()
    for f in ("rna.h5", "atac_gas.h5", "rna_cty.csv", "atac_cty.csv"):
        (d / f).write_text("")
    with pytest.raises(ValueError) as e:
        mtb.run_all("DG", "diagonal", methods=["iNMF"], out_dir=tmp_path / "o",
                    data_path=tmp_path, verbose=False)
    text = str(e.value)
    assert "rna1+rna2+rna3" not in text
    assert "\n1 other row needs files that DG does not have.\n" in text
    assert text.count("\n  iNMF (") == 1


def test_one_batch_value_names_the_cause():
    batch = np.array(["A"] * 50)
    cands = [(["cty"], np.repeat(["x", "y"], 25), batch)]
    emb = np.random.default_rng(0).normal(size=(50, 3))
    names, val, spread = workflow._evaluate_best_order(
        emb, "cross", cands, metrics=["iLISI"], user_batch=True, batch_given=True)
    assert val is None
    assert spread[0]["error"] == (
        "ValueError: The batch has one value among the 50 cells this method embedded, "
        "so the batch metrics cannot be computed. Pass a batch that splits these cells.")


def test_bubble_does_not_suggest_the_name_it_was_given():
    df = pd.DataFrame([{"method": m, "metric": "ARI", "value": v}
                       for m, v in (("scMM", 0.5), ("Multigrate", 0.6), ("moETM", 0.4))])
    with pytest.raises(ValueError) as e:
        mtb.plot.bubble(df, methods=["Matilda", "totalVI"])
    assert str(e.value) == ("Matilda and totalVI are not in this table. The table has "
                            "Multigrate, moETM and scMM.")
    with pytest.raises(ValueError, match=r"^Unknown method Multigrat\. Did you mean "
                                         r"Multigrate\?"):
        mtb.plot.bubble(df, methods=["Multigrat"])
