"""Core round c2 (0.4.0): one overall score, style.overall, for recommend, plot.bar and
plot.bubble (D01), its refusal of your rows on another dataset (C18), and the
classification-only family of your rows (D02).

The proof of D01 runs 36 cases: the four stored tables x {every metric, clustering,
batch, classification} x {+run, +copy, +copy stored}, where they have rows."""
import importlib
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench._data.catalog import FAMILY_METRICS
from multibench.plot import style
from multibench.plot.bubble import _build_table

matplotlib.use("Agg")
bar_module = importlib.import_module("multibench.plot.bar")
CATEGORIES = ("vertical", "diagonal", "mosaic", "cross")
FAMILIES = {"all": None, **{k: list(v) for k, v in FAMILY_METRICS.items()}}
CLASSIFICATION = list(FAMILY_METRICS["classification"])


def _stored(category):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results(category)


def _run_rows(stored, category):
    """Your rows: three stored methods' rows (equal values) named '<m> (this run)',
    with classification scores on the multi-batch categories, as a run has."""
    picked = sorted(stored["method"].unique())[:3]
    run = stored[stored["method"].isin(picked)].assign(
        method=lambda d: d["method"] + " (this run)", source="user")
    if category != "vertical":
        rng = np.random.default_rng(0)
        first = run.drop_duplicates("method")
        cls = pd.concat([first.assign(metric=c) for c in CLASSIFICATION], ignore_index=True)
        run = pd.concat([run, cls.assign(value=rng.random(len(cls)))], ignore_index=True)
    return run


def _tables(category):
    st = _stored(category)
    copy = st.assign(method=st["method"] + " copy", source="user")
    return st, {"+run": pd.concat([st, _run_rows(st, category)], ignore_index=True),
                "+copy": pd.concat([st, copy], ignore_index=True),
                "+copy stored": pd.concat([st, copy.assign(source="rerun")],
                                          ignore_index=True)}


def _cases():
    for category in CATEGORIES:
        st, tables = _tables(category)
        for family, metrics in FAMILIES.items():
            for name, df in tables.items():
                if metrics is not None and not df["metric"].isin(metrics).any():
                    continue
                yield category, family, name


CASES = list(_cases())


def test_the_proof_has_36_cases():
    assert len(CASES) == 36


@pytest.mark.parametrize("category", CATEGORIES)
def test_recommend_bar_and_bubble_use_the_one_score(category):
    st, tables = _tables(category)
    for df in (st, tables["+run"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = style.overall(df)
            rec = mtb.recommend(category, long_df=df)
            fig = bar_module.bar(df)
            tbl = _build_table(df)
        rec = rec[rec["grand_score"].notna()]
        assert list(rec["method"]) == list(want.index)
        assert np.allclose(rec["grand_score"], want.to_numpy())
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_yticklabels()][::-1] == list(want.index)
        assert np.allclose([p.get_width() for p in ax.patches][::-1], want.to_numpy())
        matplotlib.pyplot.close(fig)
        assert tbl.methods[:len(want)] == list(want.index)
        for b in tbl.blocks:
            fam = style.overall(df, list(b.raw.columns))
            assert np.allclose(b.overall.reindex(fam.index), fam)


def test_the_family_bars_of_stored_methods_do_not_move():
    st, tables = _tables("vertical")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = _build_table(st).blocks[0].overall
        mixed = _build_table(tables["+run"]).blocks[0].overall
    assert np.allclose(mixed[alone.index], alone)
    assert round(alone["Multigrate"], 3) == 0.909


def test_a_row_better_than_every_stored_row_scores_above_one():
    st = _stored("vertical")
    top = st[st["method"] == "Matilda"].assign(method="Mine", source="user",
                                               value=lambda d: d["value"] + 1)
    s = style.overall(pd.concat([st, top]))
    assert s.index[0] == "Mine" and s["Mine"] > 1 and s["Matilda"] == 1


# --------------------------------------------------------------- C18
def _mine(dataset):
    st = _stored("vertical")
    return st, st[st["method"] == "scMM"].assign(method="Mine", source="user",
                                                dataset=dataset)


@pytest.mark.parametrize("dataset", ["MYCITE", "user"])
def test_your_rows_on_another_dataset_are_refused(dataset):
    st, mine = _mine(dataset)
    both = pd.concat([st, mine])
    text = (f"Your rows are on {dataset}, the stored results on D11: plot them apart, "
            "or score your embedding with dataset='D11'." if dataset != "user" else
            "Your rows were scored without a dataset: score your embedding with "
            "dataset='D11' to compare it with the stored results.")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for call in (lambda: mtb.recommend("vertical", long_df=both),
                     lambda: mtb.plot.bubble(both), lambda: mtb.plot.bar(both)):
            with pytest.raises(ValueError) as e:
                call()
            assert str(e.value) == text
    # alone, your rows plot
    matplotlib.pyplot.close(mtb.plot.bubble(pd.concat([mine, mine.assign(method="Two")])))


# --------------------------------------------------------------- D02
def _with_classification():
    st, tables = _tables("diagonal")
    return tables["+run"]


@pytest.mark.parametrize("metrics", ["all", ["ARI", "OCA"]])
def test_bubble_draws_a_family_only_your_rows_have(metrics):
    both = _with_classification()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = _build_table(both, metrics=metrics)
        matplotlib.pyplot.close(mtb.plot.bubble(both, metrics=metrics))
    cls = [b for b in tbl.blocks if b.label == "Classification"][0]
    # ranked among your rows; it counts in no Overall
    assert set(cls.overall.dropna().index) == {m for m in both.method.unique()
                                               if m.endswith(" (this run)")}
    assert tbl.methods[:len(style.overall(both, ["ARI"]))] == list(
        style.overall(both, [c for b in tbl.blocks for c in b.raw.columns]).index)


def test_bar_of_a_family_only_your_rows_have():
    both = _with_classification()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = mtb.plot.bar(both, metrics="classification")
    names = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    matplotlib.pyplot.close(fig)
    assert names and all(n.endswith(" (this run)") for n in names)


# ---- restored from before the 0.4 cut: kept behaviour
def _article(st, metrics):
    """The article's score of a stored-only table, from the per-dataset max-ranks:
    mean over the datasets of minmax(mean rank), best first, ties by name."""
    parts = style.per_dataset_ranks(st, metrics)
    def per_ds(mat):            # minmax of the mean per-column max-rank
        mean_rank = mat.apply(lambda col: style.rank_max(col.to_numpy()), axis=0).mean(axis=1)
        return pd.Series(style.minmax(mean_rank.to_numpy()), index=mat.index)
    s = pd.DataFrame({ds: per_ds(p) for ds, p in parts.items()}).mean(axis=1)
    s = s.dropna()
    return s.iloc[np.lexsort([s.index.to_numpy(dtype=str), -s.to_numpy()])]


@pytest.mark.parametrize("category, family, name", CASES)
def test_overall(category, family, name):
    metrics = FAMILIES[family]
    st, tables = _tables(category)
    df = tables[name]
    s = style.overall(df, metrics)
    # sorted best first, ties by method name
    assert list(s.index) == sorted(s.index, key=lambda m: (-s[m], m))
    # identical rows score the same
    suffix = " (this run)" if name == "+run" else " copy"
    pairs = [(m[:-len(suffix)], m) for m in s.index
             if m.endswith(suffix) and m[:-len(suffix)] in s.index]
    for a, b in pairs:
        assert s[a] == s[b], (a, b)
        # and the stored row sits right above its copy
        assert list(s.index).index(b) == list(s.index).index(a) + 1
    if st["metric"].isin(metrics or st["metric"]).any():
        alone = style.overall(st, metrics)
        # stored-only tables reproduce today's numbers: the article's rule
        pd.testing.assert_series_equal(alone, _article(st, metrics), check_names=False)
        if name != "+copy stored":
            # stored scores and order are unchanged by what is added
            kept = s[s.index.isin(alone.index)]
            assert list(kept.index) == list(alone.index)
            assert np.allclose(kept.to_numpy(), alone.to_numpy(), rtol=0, atol=0)
    else:
        # a family only your rows have is ranked among your rows
        assert set(s.index) == {m for m in df["method"].unique() if m.endswith(suffix)}


def test_your_rows_alone_rank_among_themselves():
    st = _stored("vertical").assign(source="user")
    assert np.allclose(style.overall(st), _article(st, None))
    one = st[st["method"] == "Matilda"]
    assert style.overall(one).to_dict() == {"Matilda": 1.0}

