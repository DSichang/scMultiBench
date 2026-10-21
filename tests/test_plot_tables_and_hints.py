"""bubble, bar and build_table: the Overall, the Rank key, legends, repeated rows."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb
# the table behind a bubble figure: mtb.plot.build_table is deprecated in 0.4.0
from multibench.plot.bubble import _build_table  # noqa: E402
from multibench.engine import builder, envs, registry  # noqa: F401


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _d11():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results("vertical", dataset="D11")


def _mine(base, name="Mine", scored_with="leidenalg/search/0.4.0"):
    rows = base[base.method == "Matilda"].assign(method=name, source="user")
    if scored_with is not None:
        rows = rows.assign(scored_with=scored_with)
    else:      # the stored rows carry scored_with since c2 (C15): yours have none
        rows = rows.drop(columns="scored_with", errors="ignore")
    return rows


MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.engine.runner.preview(method, category, inputs=inputs, out_dir=str(tmp_path / "o"), params=params)[0]


# --- figures: X07, X86, X87, X88, X94 ----------------------------------------

def test_x07_a_family_only_your_row_has_does_not_lift_it():
    d11 = _d11()
    mine = _mine(d11)
    cls = pd.DataFrame({"metric": ["OCA", "F1"], "value": [0.9, 0.8], "method": "Mine",
                        "dataset": "D11", "category": "vertical", "clustering": "default",
                        "source": "user", "scored_with": "leidenalg/search/0.4.0"})
    plain, _ = _messages(_build_table, pd.concat([d11, mine]))
    tbl, msgs = _messages(_build_table, pd.concat([d11, mine, cls]))
    assert tbl.overall["Mine"] == pytest.approx(plain.overall["Mine"])
    assert "Classification" in [b.label for b in tbl.blocks]          # still drawn
    assert not any("Classification" in m for m in msgs), msgs   # the n/a cells show it
    # bar: the lone metrics are left out of the ranks, with one line
    _, msgs = _messages(mtb.plot.bar, pd.concat([d11, mine, cls]))
    assert "F1 and OCA have a value for one method only, so they do not count in " \
           "the Overall." in msgs


def test_x87_clustering_columns_follow_catalog_metrics():
    tbl = _build_table(_d11())
    order = mtb.catalog.metrics().query("family == 'clustering'").metric.tolist()
    assert tbl.blocks[0].raw.columns.tolist() == order


@pytest.mark.parametrize("scored_with,expect", [
    ("leidenalg/sweep/0.3.9", "were scored before multibench 0.4.0"),
    (None, "do not record how they were scored"),
    ("leidenalg/search/0.4.0", None),
    ("leidenalg/user/0.4.0", None),
    ("unknown", None)])
@pytest.mark.parametrize("fn", [_build_table, mtb.plot.bar])
def test_x94_rows_scored_before_040_warn_next_to_stored_rows(fn, scored_with, expect):
    d11 = _d11()
    _, msgs = _messages(fn, pd.concat([d11, _mine(d11, scored_with=scored_with)]))
    hits = [m for m in msgs if "multibench 0.4.0" in m]
    if expect is None:
        assert not hits, msgs
    else:
        assert len(hits) == 1 and expect in hits[0] and hits[0].startswith(
            "The rows for Mine "), msgs


def test_x94_recommend_warns_for_old_rows_too():
    d11 = _d11()
    _, msgs = _messages(mtb.recommend, "vertical",
                        long_df=pd.concat([d11, _mine(d11, scored_with="igraph/sweep/0.3.1")]))
    assert any("The rows for Mine were scored before multibench 0.4.0." in m for m in msgs)


# X07: a family counts in the Overall only when every row has it, silently
def test_a_family_two_rows_have_counts_in_no_overall():
    import warnings
    import pandas as pd
    stored = mtb.load_results("diagonal", dataset="D28")
    # a family only two rows have: the stored classification rows left out
    stored = stored[~stored["metric"].isin(["OCA", "ACA", "Spec", "Sens", "F1"])]
    mine = stored[stored["method"].isin(["scJoint", "iNMF"])].assign(
        method=lambda d: d["method"] + "_rerun", source="user",
        scored_with="leidenalg/search/0.4.0")
    cls = pd.DataFrame([{"metric": c, "value": 0.5, "method": m, "dataset": "D28",
                         "category": "diagonal", "clustering": "default", "source": "user",
                         "scored_with": "leidenalg/search/0.4.0"}
                        for m in ("scJoint_rerun", "iNMF_rerun") for c in ("OCA", "F1")])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = _build_table(pd.concat([stored, mine, cls]))
    order = list(t.methods)
    for m in ("scJoint", "iNMF"):
        # same scores, side by side (c1: placed by their rank against the stored rows)
        assert abs(order.index(m + "_rerun") - order.index(m)) == 1
    assert "Classification" in [b.label for b in t.blocks]
    assert not [x for x in w if "Classification" in str(x.message)
                or "has no OCA" in str(x.message)]


# Y08: the Rank key says best and worst, as BubbleTable.ranks counts n = best
def test_rank_key_says_best_and_worst():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = mtb.plot.bubble(mtb.load_results("mosaic"))
    texts = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "legend"]
    assert "best" in texts and "worst" in texts


