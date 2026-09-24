"""Fixes after the eighth student study (fix round 8, package 'other').

R8-03: a family of one metric in a bubble figure had a header wider than its
coloured pill. The pill now holds its header, and two widened pills keep a
gap; figures whose headers already fit are drawn as before.
R8-07: the last messages in the old style (grey columns, the one-method
table, the label-order errors, a failed scripts fetch) are short capitalised
sentences, and the fetch error fits run_all's FAIL line whole.
R8-08: the help of ``multibench scan`` and ``multibench convert`` is short
sentences without ';', nested parentheses or slash lists.
"""
import importlib
import re
import subprocess
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench.eval import pipeline as P

matplotlib.use("Agg")

B = importlib.import_module("multibench.plot.bubble")
R = importlib.import_module("multibench.data.results")
W = importlib.import_module("multibench.workflow")

HEADERS = {label for label, _, _ in B.FAMILIES} | {"Other"}


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _d52():
    return _quiet(mtb.load_results, "cross", dataset="D52", source="rerun")


# ============================================================ R8-03 pills
def _pills_and_headers(fig):
    from matplotlib.patches import FancyBboxPatch
    fig.draw_without_rendering()
    ax = fig.axes[0]
    pills = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    heads = [t for t in ax.texts if t.get_text() in HEADERS]
    assert len(pills) == len(heads) >= 1
    return ax, pills, heads


@pytest.mark.parametrize("metrics", [["ARI", "iLISI"], ["cLISI"],
                                     ["ARI", "NMI", "ASW_batch"]])
def test_each_family_header_lies_inside_its_pill(metrics):
    fig = _quiet(mtb.plot.bubble, _d52(), metrics=metrics)
    _, pills, heads = _pills_and_headers(fig)
    pt = fig.dpi / 72.0
    for pill, head in zip(pills, heads):
        p, t = pill.get_window_extent(), head.get_window_extent()
        assert p.x0 <= t.x0 - 2.5 * pt and t.x1 + 2.5 * pt <= p.x1, (
            metrics, head.get_text(), (t.x0, t.x1), (p.x0, p.x1))
        assert p.y0 <= t.y0 and t.y1 <= p.y1, (metrics, head.get_text())
    boxes = sorted((p.get_window_extent() for p in pills), key=lambda b: b.x0)
    for a, b in zip(boxes, boxes[1:]):
        assert a.x1 < b.x0, (metrics, a, b)


@pytest.mark.parametrize("n_rows,title", [(1, None), (1, "D52"), (2, "D52")])
def test_widened_pills_of_a_small_figure_keep_a_gap(n_rows, title):
    df = _d52()
    methods = sorted(df["method"].unique())[:n_rows]
    fig = _quiet(mtb.plot.bubble, df, metrics=["ARI", "iLISI"], methods=methods,
                 title=title)
    _, pills, heads = _pills_and_headers(fig)
    pt = fig.dpi / 72.0
    for pill, head in zip(pills, heads):
        p, t = pill.get_window_extent(), head.get_window_extent()
        assert p.x0 <= t.x0 and t.x1 <= p.x1, (n_rows, head.get_text())
    a, b = (p.get_window_extent() for p in pills)
    assert b.x0 - a.x1 >= 1.9 * pt, (n_rows, title, a.x1, b.x0)
    # the widened pills stay inside the figure
    assert a.x0 >= fig.bbox.x0 and b.x1 <= fig.bbox.x1


def test_pills_that_hold_their_header_keep_their_width():
    fig = _quiet(mtb.plot.bubble, _d52(), metrics=["ARI", "NMI", "iLISI", "GC"])
    _, pills, _ = _pills_and_headers(fig)
    # [Overall 1.5][2 metrics x 1.1] minus 0.35: the width before round 8
    assert [round(p.get_width(), 6) for p in pills] == [3.35, 3.35]
    xs = [round(p.get_x(), 6) for p in pills]
    assert xs == [0.05, 3.7 + 0.5 + 0.05]


# ============================================================ R8-07 text
def _plain(msg, *code, lead=None):
    """A capitalised sentence text: no ', so', no ';' and no '[' outside code.

    ``lead`` is a code name the text may start with instead of a capital.
    """
    body = msg
    for c in code:
        body = body.replace(c, "")
    if lead:
        assert body.startswith(lead), msg
    else:
        assert body[:1].isupper(), msg
    assert ", so" not in body and ";" not in body and "[" not in body, msg
    return msg


def _warned(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _two(values):
    """Two methods on one dataset with the given (cLISI, iLISI) per method."""
    rows = []
    for m, (cl, il, ari) in zip(("A", "B"), values):
        rows += [{"method": m, "dataset": "D1", "metric": k, "value": v}
                 for k, v in (("cLISI", cl), ("iLISI", il), ("ARI", ari))]
    return pd.DataFrame(rows)


def test_grey_column_warnings_are_plain_sentences():
    _, msgs = _warned(mtb.plot.build_table, _two([(1.0, 0.0, 0.3), (1.0, 0.0, 0.6)]))
    assert msgs == [_plain("cLISI is 1.000 and iLISI is 0.000 for every method. "
                           "Those columns are grey.", lead="cLISI")]
    _, msgs = _warned(mtb.plot.build_table, _two([(1.0, 0.1, 0.3), (1.0, 0.0, 0.6)]))
    assert msgs == [_plain("cLISI is 1.000 for every method. That column is grey.",
                           lead="cLISI")]
    rows = []
    for ds, (a, b) in (("D1", (0.9, 0.1)), ("D2", (0.1, 0.9))):
        rows += [{"method": "A", "dataset": ds, "metric": "ARI", "value": a},
                 {"method": "B", "dataset": ds, "metric": "ARI", "value": b},
                 {"method": "A", "dataset": ds, "metric": "NMI", "value": 0.5},
                 {"method": "B", "dataset": ds, "metric": "NMI", "value": 0.4}]
    _, msgs = _warned(mtb.plot.build_table, pd.DataFrame(rows), aggregate="summary")
    grey = [m for m in msgs if m.endswith("grey.")]
    assert grey == [_plain("Every method has mean rank 1.5 in ARI. That column is grey.")]


def test_one_method_table_warning_is_plain_sentences():
    _, msgs = _warned(mtb.load_results, "cross", dataset="D52")
    one = [m for m in msgs if re.match(R._ONE_METHOD_WARNING, m)]
    assert len(one) == 1, msgs
    assert one[0].startswith("The published table for cross/D52 has one method, "
                             "scMoMaT. Every rank in it is the same. The re-run "
                             "tables have 8 methods for cross/D52: ")
    _plain(one[0])


def test_one_method_warning_joins_several_datasets_with_and(monkeypatch):
    monkeypatch.setattr(R, "_other_source_methods", lambda *a, **k: {"A", "B"})
    out = pd.DataFrame({"method": ["A"], "category": ["cross"]})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        R._warn_single_method(out, "published", "cross", ["D52", "D53"], "leiden",
                              R.Path("."))
    msg = str(rec[0].message)
    assert re.match(R._ONE_METHOD_WARNING, msg)
    assert msg.startswith("The published table for cross/D52 and D53 has one method, A. ")
    _plain(msg)


def test_cli_one_method_bar_line_is_a_sentence(tmp_path, capsys):
    path = tmp_path / "one.csv"
    pd.DataFrame({"method": ["M"] * 2, "dataset": ["D1"] * 2, "metric": ["ARI", "NMI"],
                  "value": [0.3, 0.4]}).to_csv(path, index=False)
    rc = cli.main(["plot", "bar", "--input", str(path), "--out", str(tmp_path / "b.png")])
    err = capsys.readouterr().err
    assert rc == 0, err
    line = next(l for l in err.splitlines() if "one method" in l)
    assert line == "warning: The table has one method. Every rank is the same."
    _plain(line[len("warning: "):])


def test_label_order_errors_are_plain_sentences():
    d = {"atac_cty": "a.csv", "rna_cty": "b.csv"}
    with pytest.raises(ValueError) as e:
        P._labels_from_dict(d, None)
    assert str(e.value).startswith("The label keys atac_cty and rna_cty are not in the "
                                   "default order. Pass the dict from mtb.labels_for(")
    _plain(str(e.value), "label_order=[...]")
    cases = {
        "rna_cty": (TypeError, "label_order= takes a list of keys of the labels dict, "
                    "such as label_order=['atac_cty', 'rna_cty']. It got a str."),
        (): (ValueError, "label_order= is empty. List the keys of the labels dict in "
             "the method's cell order, such as ['atac_cty', 'rna_cty']."),
        ("x",): (ValueError, "label_order names x, which is not a key of the labels "
                 "dict. Its keys are atac_cty and rna_cty."),
        ("rna_cty", "rna_cty"): (ValueError, "label_order names rna_cty twice."),
    }
    for order, (exc, want) in cases.items():
        with pytest.raises(exc) as e:
            P._labels_from_dict(d, order if isinstance(order, str) else list(order))
        assert str(e.value) == want
        _plain(want, "['atac_cty', 'rna_cty']", lead="label_order")


def _fetch_error(monkeypatch, tmp_path, ref=None, missing_git=False):
    def fail(argv, **kw):
        if missing_git:
            raise FileNotFoundError("git")
        raise subprocess.CalledProcessError(128, argv)
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    with pytest.raises(RuntimeError) as e:
        if ref:
            config.ensure_repo(tmp_path / "scripts", ref=ref)
        else:
            config.ensure_repo(tmp_path / "scripts")
    return e.value


def test_scripts_fetch_errors_are_plain_and_fit_the_fail_line(monkeypatch, tmp_path, capsys):
    tail = (" On a host without network, run multibench fetch --scripts on a connected "
            "machine, copy the folder and set MULTIBENCH_REPO_PATH.")
    e = _fetch_error(monkeypatch, tmp_path)
    assert str(e) == "Could not reach github.com to fetch the method scripts." + tail
    record = f"{type(e).__name__}: {e}"
    assert len(record) <= 200
    # run_all's FAIL line shows the whole text, not a clipped '...' tail
    assert W._error_tail(record) == record
    e = _fetch_error(monkeypatch, tmp_path, ref="abc123")
    assert str(e) == ("Could not fetch the method scripts at 'abc123' from github.com. "
                      "Either there is no network, or github.com has no such commit "
                      "or tag." + tail)
    e2 = _fetch_error(monkeypatch, tmp_path, missing_git=True)
    assert str(e2) == "Git is not installed. It is needed to fetch the method scripts." + tail
    for err in (e, e2):
        _plain(str(err))
    capsys.readouterr()


# ============================================================ R8-08 help
def _subparser(name):
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    return sub.choices[name]


def _help_texts():
    scan, conv = _subparser("scan"), _subparser("convert")
    texts = {"scan": scan.description, "convert": conv.description}
    for a in conv._actions:
        if a.dest in ("src", "out"):
            texts[a.dest] = a.help
    return texts


def _depth(text):
    depth = deepest = 0
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        deepest = max(deepest, depth)
    return deepest


def test_scan_and_convert_help_are_short_sentences():
    texts = _help_texts()
    assert set(texts) == {"scan", "convert", "src", "out"}
    for name, text in texts.items():
        assert ";" not in text, (name, text)
        assert _depth(text) <= 1, (name, text)
        # no slash lists such as .csv/.tsv or --rna/--adt
        assert not re.search(r"[\w.]/[-.\w]", text), (name, text)
    assert "preflight" not in texts["scan"]
    assert f"{cli._TRUNCATE_WIDTH} characters" in texts["scan"]
    # the modes the other convert options refer to are still named
    assert "Mode 1" in texts["convert"] and "Mode 2" in texts["convert"]
    assert "Give raw counts." in texts["convert"]
