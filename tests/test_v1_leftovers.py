"""v1 leftovers: an unreadable .h5ad in one sentence, "1 cell", and recommend on
rows of another category."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------------------------------------- NEW-2
def test_an_unreadable_h5ad_is_named_in_one_sentence(tmp_path, capsys):
    bad = tmp_path / "bad.h5ad"
    bad.write_text("not hdf5\n")
    said = "bad.h5ad cannot be read: it is not an .h5ad file, or it is damaged."
    with pytest.raises(ValueError) as e:
        mtb.evaluate(str(bad), labels="celltype")
    assert str(e.value) == said
    for args in (["evaluate", str(bad), "--labels", "celltype"],
                 ["export", str(bad), str(tmp_path / "OUT"), "--category", "vertical"]):
        assert cli.main(args) == 1
        assert capsys.readouterr().err.splitlines() == [f"error: {said}"]


# ---------------------------------------------------------------- NEW-3
def test_one_cell_is_singular(tmp_path, capsys):
    (tmp_path / "one.csv").write_text("a,b\n1,2\n")
    pd.DataFrame({"x": ["T", "B", "T"]}).to_csv(tmp_path / "lab.csv", index=False)
    assert _quiet(cli.main, ["evaluate", str(tmp_path / "one.csv"), "--labels",
                             str(tmp_path / "lab.csv"), "--metrics", "ASW"]) == 1
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: --labels gave 3 labels (lab.csv) for 1 cell in the embedding file: give "
        "one label per cell.")
    with pytest.raises(ValueError, match=r"^labels has 1 entry for 3 embedding rows: "):
        _quiet(mtb.evaluate, np.zeros((3, 2)), labels=np.array(["T"]), metrics=["ASW"])


# ---------------------------------------------------------------- u5 #6
def test_recommend_refuses_rows_of_another_category():
    vertical = _quiet(mtb.load_results, "vertical")
    cross = _quiet(mtb.load_results, "cross")
    with pytest.raises(ValueError) as e:
        _quiet(mtb.recommend, "cross", long_df=vertical)
    assert str(e.value) == ("long_df holds vertical rows: pass recommend('vertical', "
                            "long_df=...).")
    both = pd.concat([cross, vertical])
    with pytest.raises(ValueError) as e:
        _quiet(mtb.recommend, "cross", long_df=both)
    assert str(e.value) == (
        "long_df holds vertical rows next to cross ones: rank one category at a time, as "
        "recommend('cross', long_df=long_df[long_df['category'] == 'cross']).")
    with pytest.raises(ValueError, match=r"^long_df holds cross and vertical rows: rank one "
                                         r"category at a time, as recommend\('cross', "):
        _quiet(mtb.recommend, "mosaic", long_df=both)
    # rows of your own without a category rank next to the stored ones
    mine = cross[cross["method"] == "StabMap"].assign(method="mine", category=None)
    out = _quiet(mtb.recommend, "cross", long_df=pd.concat([cross, mine]))
    assert "mine" in set(out["method"])
