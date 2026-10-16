"""``plot.bubble`` / ``bar`` / ``build_table`` and ``plot --metrics`` take the
family words ``evaluate`` takes: clustering, batch, classification, all."""
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli


def _long():
    rows = []
    for m, v in (("A", 0.9), ("B", 0.5), ("C", 0.2)):
        for code in ("ARI", "NMI", "ASW", "iLISI", "GC"):
            rows.append({"method": m, "metric": code, "value": v, "dataset": "D1"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("word, codes", [("clustering", ["ARI", "NMI", "ASW"]),
                                         ("batch", ["GC", "iLISI"]),
                                         (["Clustering"], ["ARI", "NMI", "ASW"])])
def test_build_table_takes_a_family(word, codes):
    got = mtb.plot.build_table(_long(), metrics=word)
    assert sorted(c for b in got.blocks for c in b.raw.columns) == sorted(codes)


def test_all_is_every_metric():
    a = mtb.plot.build_table(_long(), metrics="all")
    b = mtb.plot.build_table(_long())
    assert [list(x.raw.columns) for x in a.blocks] == [list(x.raw.columns) for x in b.blocks]


def test_a_family_the_table_lacks_says_so():
    with pytest.raises(ValueError) as e:
        mtb.plot.build_table(_long(), metrics="classification")
    assert str(e.value) == ("The table has no classification metrics. It has ARI, ASW, GC, "
                            "NMI and iLISI.")


def test_bubble_and_bar_take_it(tmp_path):
    mtb.plot.bubble(_long(), metrics="batch")
    mtb.plot.bar(_long(), metrics="clustering")


def test_cli_plot_metrics_family(tmp_path):
    src = tmp_path / "long.csv"
    _long().to_csv(src, index=False)
    assert cli.main(["plot", "bubble", "--input", str(src), "--metrics", "clustering",
                     "--out", str(tmp_path / "b.png")]) == 0
