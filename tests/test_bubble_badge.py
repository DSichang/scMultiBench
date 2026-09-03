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
