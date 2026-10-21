"""``plot.bubble`` / ``bar`` / ``build_table`` and ``plot --metrics`` take the
family words ``evaluate`` takes: clustering, batch, classification, all."""
import pandas as pd
import pytest

import multibench as mtb


def _long():
    rows = []
    for m, v in (("A", 0.9), ("B", 0.5), ("C", 0.2)):
        for code in ("ARI", "NMI", "ASW", "iLISI", "GC"):
            rows.append({"method": m, "metric": code, "value": v, "dataset": "D1"})
    return pd.DataFrame(rows)


def test_bubble_and_bar_take_it(tmp_path):
    mtb.plot.bubble(_long(), metrics="batch")
    mtb.plot.bar(_long(), metrics="clustering")


