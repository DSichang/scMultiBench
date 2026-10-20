"""v1 leftovers: an unreadable .h5ad in one sentence, "1 cell", recommend on
rows of another category, and the quickstart's lines for counts in a layer."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from tests.test_docs_r4 import _cite_adata, _laptop_tab, _read, needs_docs


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


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


# ---------------------------------------------------------------- u1 #5
@needs_docs
def test_the_quickstart_lines_for_counts_in_a_layer(tmp_path, monkeypatch, capsys):
    """The two lines the laptop tab names for counts in a layer give the PCA of
    counts in X: X is logged once."""
    import matplotlib.pyplot as plt
    tab = " ".join(_read("quickstart.md").split('=== "CITE-seq on a laptop"', 1)[1]
                   .split('\n=== "', 1)[0].split())
    export, start = 'rna="layer:counts"', 'pc = sc.AnnData(adata.layers["counts"].copy())'
    assert f"pass `{export}` to `export_dataset`" in tab and f"`{start}`" in tab
    code = _laptop_tab()
    layered = code.replace('rna="X"', export).replace("pc = adata.copy()", start)
    assert layered.count(export) == 1 and layered.count(start) == 1
    pcs = []
    for name, src in (("in_x", code), ("layer", layered)):
        (tmp_path / name).mkdir()
        monkeypatch.chdir(tmp_path / name)
        a = _cite_adata()
        if name == "layer":
            import scanpy as sc
            a.layers["counts"] = a.X.copy()
            sc.pp.normalize_total(a)
            sc.pp.log1p(a)
        a.write_h5ad("my_citeseq.h5ad")
        scope = {}
        _quiet(exec, compile(src, f"quickstart-{name}", "exec"), scope)
        pcs.append(np.abs(scope["adata"].obsm["X_pca"]))
        plt.close("all")
    capsys.readouterr()
    np.testing.assert_allclose(pcs[1], pcs[0], rtol=1e-3, atol=1e-3)
