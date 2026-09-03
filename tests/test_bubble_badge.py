"""The supervised 'L' badge must follow the plotted category's variants."""
import importlib

import multibench as mtb

B = importlib.import_module("multibench.plot.bubble")


def test_table_records_the_frame_category():
    df = mtb.load_results("vertical", dataset="D11", source="rerun")
    tbl = B.build_table(df)
    assert tbl.category == "vertical"
    assert B._single_category(tbl) == "vertical"


def test_scmomat_is_unsupervised_on_a_vertical_figure_but_supervised_in_mosaic():
    # scMoMaT consumes labels only in its mosaic variant
    assert B._method_needs_labels("scMoMaT", "vertical") is False
    assert B._method_needs_labels("scMoMaT", "mosaic") is True
    assert B._method_needs_labels("scMoMaT") is True          # any-variant fallback


def test_mixed_category_frame_has_no_single_category():
    import pandas as pd
    a = mtb.load_results("vertical", dataset="D11", source="rerun")
    b = mtb.load_results("cross", dataset="D52", source="rerun")
    tbl = B.build_table(pd.concat([a, b]))
    assert tbl.category is None


# --- P07: the badge follows the frame's category; a needs_labels column ----
# overrides the registry (the only way to badge an unregistered method)
import warnings

import pandas as pd
import pytest


def _d11():
    return mtb.load_results("vertical", dataset="D11", source="rerun")


def _render_texts(tbl):
    import matplotlib; matplotlib.use("Agg")
    fig = B.render(tbl)
    return fig, [t.get_text() for t in fig.axes[0].texts]


def test_d11_vertical_figure_badges_matilda_not_scmomat():
    from matplotlib.patches import Circle
    tbl = B.build_table(_d11(), na="skip")
    assert tbl.needs_labels == {}                 # no column -> behaviour unchanged
    fig, texts = _render_texts(tbl)
    assert texts.count("L") == 1
    badge = [p for p in fig.axes[0].patches if isinstance(p, Circle)
             and abs(p.center[0] - (-0.38)) < 1e-6]
    assert len(badge) == 1
    row = len(tbl.methods) - int(badge[0].center[1] + 0.5)
    assert tbl.methods[row] == "Matilda"


def test_needs_labels_column_overrides_registry():
    d11 = _d11()
    mine = mtb.to_long(pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"]),
                       "MySupervised", "D11", "vertical", needs_labels=True)
    tbl = B.build_table(pd.concat([d11, mine]), na="skip")
    assert tbl.needs_labels == {"MySupervised": True}
    _, texts = _render_texts(tbl)
    assert texts.count("L") == 2
    # the override beats the registry in the other direction too: scMoMaT is
    # supervised in mosaic, but an explicit False removes its badge
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    mosaic = pd.concat([mtb.to_long(w, "scMoMaT", "D45", "mosaic"),
                        mtb.to_long(w * 0.5, "Cobolt", "D45", "mosaic")])
    assert B._method_needs_labels("scMoMaT", "mosaic") is True
    _, texts_ref = _render_texts(B.build_table(mosaic, na="skip"))
    assert texts_ref.count("L") == 1
    off = mosaic.assign(needs_labels=mosaic.method.map({"scMoMaT": False}))
    tbl2 = B.build_table(off, na="skip")
    assert tbl2.needs_labels == {"scMoMaT": False}
    _, texts2 = _render_texts(tbl2)
    assert texts2.count("L") == 0


def test_needs_labels_column_inconsistent_raises():
    d11 = _d11()
    bad = d11.assign(needs_labels=[True if i == 0 else (False if i == 1 else None)
                                   for i in range(len(d11))])
    m = bad.method.iloc[0]
    if bad.method.iloc[1] != m:
        bad.loc[bad.index[1], "needs_labels"] = None
        bad.loc[bad.index[bad.method == m][1], "needs_labels"] = False
    with pytest.raises(ValueError, match=f"needs_labels differs between rows of method {m!r}"):
        B.build_table(bad, na="skip")
    # NaN rows are "no override"; a methods= filter removes the offender first
    ok = d11.assign(needs_labels=None)
    assert B.build_table(ok, na="skip").needs_labels == {}
    keep = [x for x in d11.method.unique() if x != m][:2]
    assert B.build_table(bad, methods=keep, na="skip").needs_labels == {}


def test_frame_without_column_unchanged():
    tbl = B.build_table(_d11(), na="skip")
    assert tbl.needs_labels == {} and tbl.category == "vertical"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        B.build_table(_d11(), na="skip")
