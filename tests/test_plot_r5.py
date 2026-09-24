"""Plot and evaluate fixes after the fifth student study (fix round 5).

R5-08: a bubble figure of few metrics clipped its Rank legend (and, with one
metric, its Score ramp) at the right edge of the axes.
R5-11: the plot warnings, the CLI overlay line and the label-order error read
as plain sentences; the na='raise' error no longer describes a figure it does
not draw.
R5-14: the evaluate Notes say how LISI turns per-cell scores into one number.
"""
import importlib
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config

matplotlib.use("Agg")

B = importlib.import_module("multibench.plot.bubble")
P = importlib.import_module("multibench.eval.pipeline")

QUICKSTART = ["ARI", "NMI", "cLISI"]


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def _stored(dataset, source="published", category="diagonal"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results(category, dataset=dataset, source=source)


def _bubble(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.plot.bubble(df, **kw)


# --- R5-08: the legends stay inside the axes and the figure ------------------

def _legend_outside(fig):
    """Legend artists whose drawn extent leaves the axes or the figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    ax_box, fig_box = ax.get_window_extent(renderer), fig.bbox
    legend = [a for a in list(ax.patches) + list(ax.texts) if a.get_gid() == "legend"]
    assert legend, "no artist carries gid='legend'"
    bad = []
    for a in legend:
        bb = a.get_window_extent(renderer)
        for box, where in ((ax_box, "axes"), (fig_box, "figure")):
            if (bb.x0 < box.x0 - 0.5 or bb.x1 > box.x1 + 0.5
                    or bb.y0 < box.y0 - 0.5 or bb.y1 > box.y1 + 0.5):
                bad.append((type(a).__name__, getattr(a, "get_text", lambda: "")(), where))
    return bad


def test_quickstart_figure_keeps_its_rank_legend_whole():
    df = _stored("D28")
    fig = _bubble(df, metrics=QUICKSTART)
    assert _legend_outside(fig) == []
    # all five Rank circles are drawn, the largest (rank 1) last
    circles = [p for p in fig.axes[0].patches if p.get_gid() == "legend"
               and type(p).__name__ == "Circle"]
    assert len(circles) == 5


@pytest.mark.parametrize("kw", [
    {"metrics": ["ARI"]},                                   # one metric
    {"metrics": QUICKSTART, "methods": ["Conos", "GLUE"]},  # two rows
    {"metrics": ["ARI"], "methods": ["Conos"]},               # one row, one metric
    {"metrics": ["ARI", "iLISI"]},                           # two families, one metric each
    {"metrics": QUICKSTART, "show_language": False},
])
def test_narrow_figures_keep_their_legends_inside(kw):
    fig = _bubble(_stored("D28"), **kw)
    assert _legend_outside(fig) == [], kw


def test_narrow_summary_figure_keeps_its_score_legend_inside():
    df = _stored(["D24", "D25", "D28"])
    fig = _bubble(df, metrics=["ARI"], aggregate="summary")
    assert _legend_outside(fig) == []


def _old_width(tbl):
    """The figure width before R5-08: 0.6 * table width + 2.8 inches."""
    total_w = sum(1.5 + 1.1 * b.raw.shape[1] + 0.5 for b in tbl.blocks) - 0.5
    return 0.6 * total_w + 2.8


@pytest.mark.parametrize("metrics", [
    ["cLISI", "ARI", "ASW", "iASW", "NMI"],            # five, one family
    ["ARI", "NMI", "cLISI", "ASW", "iLISI"],           # five, two families
    None,                                              # every metric
])
def test_figures_of_five_or_more_metrics_keep_their_size(metrics):
    df = _stored("D28")
    tbl = B.build_table(df, metrics=metrics, na="skip")
    fig = B.render(tbl)
    w, h = fig.get_size_inches()
    assert w == pytest.approx(_old_width(tbl))
    assert h == pytest.approx(0.42 * len(tbl.methods) + 2.9)
    assert _legend_outside(fig) == []


# --- R5-11 (a): the several-datasets warning -----------------------------------

def test_several_datasets_warning_is_plain():
    df = pd.concat([_stored("D11", "rerun", "vertical"),
                    _stored("D11s", "rerun", "vertical")], ignore_index=True)
    _, msgs = _messages(mtb.plot.build_table, df)
    assert msgs == [
        "This figure averages each method over 2 datasets, D11 and D11s. Its rows "
        "mix datasets. Pass aggregate='summary' for the rank-averaged summary panel, "
        "or filter to one dataset."]


def test_several_datasets_warning_with_a_lone_dataset():
    diag = _stored(["D24", "D25", "D28"])
    own = pd.DataFrame([{"metric": m, "value": v, "method": "PCA_standin",
                         "dataset": "LUNG", "category": "diagonal", "source": "user"}
                        for m, v in (("ARI", 0.3), ("NMI", 0.4), ("cLISI", 0.9))])
    _, msgs = _messages(mtb.plot.build_table, pd.concat([diag, own], ignore_index=True))
    assert ("This figure averages each method over 4 datasets, D24, D25, D28 and LUNG. "
            "Its rows mix datasets. Plot each dataset on its own.") in msgs
    assert not any("paper's" in m for m in msgs), msgs


def test_several_datasets_warning_cli_spelling(monkeypatch):
    monkeypatch.setattr(config, "_CLI", True)
    df = pd.concat([_stored("D11", "rerun", "vertical"),
                    _stored("D11s", "rerun", "vertical")], ignore_index=True)
    _, msgs = _messages(mtb.plot.build_table, df)
    assert msgs == [
        "This figure averages each method over 2 datasets, D11 and D11s. Its rows "
        "mix datasets. Pass --aggregate summary for the rank-averaged summary panel, "
        "or filter with --dataset."]


# --- R5-11 (b): one method, equal columns ------------------------------------

def _rows(method, values):
    return [{"method": method, "metric": m, "value": v}
            for m, v in zip(("ARI", "NMI", "cLISI"), values)]


def test_one_method_warning_is_plain():
    _, msgs = _messages(mtb.plot.build_table,
                        pd.DataFrame(_rows("PCA_LSI_stack", (0.3, 0.4, 0.9))))
    assert msgs == [
        "Only one method, PCA_LSI_stack, is in this figure. With nothing to rank it "
        "against, the fills are grey. Plot it with methods scored on the same dataset."]


def test_equal_column_warning_is_plain():
    df = pd.DataFrame(_rows("A", (0.3, 0.4, 1.0)) + _rows("B", (0.5, 0.2, 1.0)))
    _, msgs = _messages(mtb.plot.build_table, df)
    assert msgs == ["cLISI is 1.000 for every method. That column is grey."]
    df = pd.DataFrame(_rows("A", (0.3, 0.4, 1.0)) + _rows("B", (0.3, 0.2, 1.0)))
    _, msgs = _messages(mtb.plot.build_table, df)
    # columns in figure order: cLISI leads its family
    assert msgs == ["cLISI is 1.000 and ARI is 0.300 for every method. Those "
                    "columns are grey."]


def test_equal_mean_rank_warning_is_plain():
    rows = []
    for ds, (a, b) in (("D1", (0.9, 0.1)), ("D2", (0.1, 0.9))):
        for m, v in (("A", a), ("B", b)):
            rows += [{"method": m, "dataset": ds, "metric": k, "value": v}
                     for k in ("ARI", "NMI")]
    _, msgs = _messages(mtb.plot.build_table, pd.DataFrame(rows), aggregate="summary")
    grey = [m for m in msgs if m.startswith("Every method has mean rank")]
    assert grey == ["Every method has mean rank 1.5 in ARI and NMI. Those columns "
                    "are grey."], msgs


# --- R5-11 (c): the CLI overlay line -----------------------------------------

def test_cli_overlay_line_is_plain(tmp_path, monkeypatch, capsys):
    from multibench import plot as plot_ns

    def _long(method):
        return pd.DataFrame({"metric": ["ARI", "NMI", "ASW"], "value": [0.5, 0.6, 0.7],
                             "method": method, "dataset": "D1", "category": "diagonal"})
    monkeypatch.setattr(mtb, "load_results",
                        lambda **kw: pd.concat([_long("A"), _long("B")], ignore_index=True))
    monkeypatch.setattr(plot_ns, "bubble", lambda df, **kw: None)
    mine = tmp_path / "mine.csv"
    _long("PCA_LSI_stack").to_csv(mine, index=False)
    rc = cli.main(["plot", "bubble", "--category", "diagonal", "--dataset", "D1",
                   "--input", str(mine), "--out", str(tmp_path / "x.png")])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert ("# Added your 3 rows (PCA_LSI_stack) to the stored diagonal table, "
            "which has 6 rows (source published).") in err.splitlines()
    assert "# overlay" not in err


# --- R5-11 (d): na='raise' does not describe the figure it does not draw -------

def _with_gap():
    df = pd.DataFrame(_rows("A", (0.3, 0.4, 0.9)) + _rows("B", (0.5, 0.2, 0.8))
                      + _rows("C", (0.1, 0.6, 0.7)))
    return df[~((df.method == "B") & (df.metric == "NMI"))]


def test_na_raise_leaves_out_the_overall_sentence():
    with pytest.raises(ValueError) as e:
        mtb.plot.build_table(_with_gap(), na="raise")
    assert str(e.value) == ("Row B has no NMI. Pass na='warn' to draw the figure "
                            "with these gaps.")
    _, msgs = _messages(mtb.plot.build_table, _with_gap())
    assert ("Row B has no NMI. Its Overall uses the metrics it has. Pass na='skip' "
            "to hide this message.") in msgs


# --- R5-11 (e): the label-order error names a method only when one was used ---

def _cty(tmp_path):
    p1, p2 = tmp_path / "cty1.csv", tmp_path / "cty2.csv"
    pd.DataFrame({"x": ["a", "b"] * 5}).to_csv(p1, index=False)
    pd.DataFrame({"x": ["a", "b"] * 5}).to_csv(p2, index=False)
    return str(p1), str(p2)


def test_reordered_plain_dict_error_names_only_the_default_order(tmp_path):
    p1, p2 = _cty(tmp_path)
    with pytest.raises(ValueError) as e:
        P._labels_from_dict({"cty2": p2, "cty1": p1}, None)
    assert str(e.value) == (
        "The label keys cty2 and cty1 are not in the default order. Pass the "
        "dict from mtb.labels_for(dataset, category, method) unchanged, a list of "
        "paths in cell order, or label_order=[...]. The default order is cty1, "
        "cty2, ... by number, with rna before adt before atac. It is not "
        "alphabetical.")


def test_reordered_labels_for_dict_names_the_method_order(tmp_path):
    from multibench.engine.resolve import LabelFiles
    p1, p2 = _cty(tmp_path)
    # labels_for with a method whose cell order puts cty2 first
    d = LabelFiles({"cty2": p2, "cty1": p1})
    assert P._labels_from_dict(d, None) == [p2, p1]
    d["cty2"] = d.pop("cty2")                     # now cty1, cty2: the default order
    assert P._labels_from_dict(d, None) == [p1, p2]
    three = LabelFiles({"cty2": p2, "cty1": p1, "cty3": p1})
    three["cty2"] = three.pop("cty2")             # cty1, cty3, cty2: neither order
    with pytest.raises(ValueError) as e:
        P._labels_from_dict(three, None)
    assert str(e.value).startswith(
        "The label keys cty1, cty3 and cty2 are in neither the method's cell "
        "order nor the default order. Pass the dict from mtb.labels_for(")
    # labels_for without a method returns the default order: no method to name
    plain = LabelFiles({"cty1": p1, "cty2": p2})
    plain["cty1"] = plain.pop("cty1")
    with pytest.raises(ValueError) as e:
        P._labels_from_dict(plain, None)
    assert "method's cell order" not in str(e.value)
    assert "are not in the default order" in str(e.value)


# --- R5-14: the LISI median and scaling are written down ----------------------

def test_evaluate_notes_state_the_lisi_median_and_scaling():
    doc = mtb.evaluate.__doc__
    sentence = ("cLISI and iLISI take the median m of the per-cell scores and scale "
                "it: cLISI = (L - m)/(L - 1), iLISI = (m - 1)/(B - 1), with L cell "
                "types and B batches.")
    assert sentence in " ".join(doc.split())
    # after the scib-call table, before the re-run paragraph
    flat = " ".join(doc.split())
    assert flat.index("computed only when named in metrics=[...]") < flat.index(sentence)
    assert flat.index(sentence) < flat.index("The re-run tables were scored")


def test_lisi_formula_matches_scib():
    """The documented formula is what scib computes from the per-cell scores."""
    scib_lisi = pytest.importorskip("scib.metrics.lisi")
    import inspect
    src = inspect.getsource(scib_lisi)
    assert "np.nanmedian" in src
    assert "(ilisi - 1) / (nbatches - 1)" in src
    assert "(nlabs - clisi) / (nlabs - 1)" in src
