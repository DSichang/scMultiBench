"""Round 7 of the virtual-student study, package work package 'wf' (workflow.py).

R7-01  run_all's Notes quoted the "nothing runnable" message with bare
       ``<dataset>`` / ``<category>`` placeholders. Markdown reads them as HTML
       elements and the browser drops them, so the reference page showed empty
       parentheses. The bullet now quotes concrete messages in code spans, and a
       check walks every rendered docstring (and the docs pages) for a bare tag.
R7-02  the "No method can run" error named a ``mtb.scan`` call without the
       caller's ``data_path`` / ``modalities`` / ... (it failed or showed other
       rows), and its list header said "rows" when each method has one row.
R7-05  ``BatchResult.long`` did not name the ``scored_with`` column.
R7-06  ``rescore(verbose=True)`` also prints a line before each label-order
       ranking; the parameter row said one line per method.
R7-12  template-like passages in workflow.py docstrings, reworded.
"""
import inspect
import os
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from tests.test_docs_consistency import REFERENCE, _resolve


def _flat(obj) -> str:
    return " ".join((inspect.getdoc(obj) or "").split())


@pytest.fixture
def no_envs(monkeypatch):
    """No method environment is installed (the laptop situation)."""
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


# ============================================================ R7-01
#: HTML and SVG element names a docs page may use on purpose
_ELEMENTS = set("""
a abbr address area article aside audio b base bdi bdo blockquote body br button
canvas caption cite code col colgroup data datalist dd del details dfn dialog div dl
dt em embed fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 head header hr
html i iframe img input ins kbd label legend li link main map mark menu meta meter
nav noscript object ol optgroup option output p param picture pre progress q rp rt
ruby s samp script section select slot small source span strong style sub summary
sup table tbody td template textarea tfoot th thead time title tr track u ul var
video wbr center font
svg g path rect circle ellipse line polyline polygon text tspan defs use symbol
lineargradient radialgradient stop clippath mask pattern marker filter foreignobject
""".split())

#: what Python-Markdown's inline-HTML pattern takes for a tag: ``<name ...>`` or
#: ``</name>``, the name without spaces or ``@``; ``<https://...>`` is an autolink
_TAG = re.compile(r"<(/?)([A-Za-z][^<>@\s/]*)[^<>]*>")


def _tags(text):
    """The ``(tag, name)`` of each HTML tag Markdown would read in ``text``."""
    return [(m.group(0), m.group(2).lower()) for m in _TAG.finditer(text)
            if ":" not in m.group(2)]


def _docstring_prose(doc: str) -> str:
    """``doc`` without what mkdocstrings renders as code: fenced blocks, the
    indented block after a ``::`` line, doctest blocks (a ``>>>`` line and the
    lines up to the next blank one) and ````...```` / ```...``` spans."""
    lines, out, i = doc.splitlines(), [], 0
    while i < len(lines):
        line, s = lines[i], lines[i].strip()
        if s.startswith(("```", "~~~")):
            fence, i = s[:3], i + 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                i += 1
            i += 1
            continue
        if s.startswith(">>>"):
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        out.append(line)
        i += 1
        if s.endswith("::"):
            indent = len(line) - len(line.lstrip())
            while i < len(lines) and (not lines[i].strip()
                                      or len(lines[i]) - len(lines[i].lstrip()) > indent):
                i += 1
    text = re.sub(r"``.*?``", "", "\n".join(out), flags=re.S)
    return re.sub(r"`[^`]*`", "", text)


def _rendered_objects():
    """Every object a reference page renders: the ``:::`` entries and the
    public methods and properties of the classes among them."""
    for name, (_page, path) in REFERENCE.items():
        obj = _resolve(path)
        yield name, obj
        if inspect.isclass(obj):
            for attr, val in vars(obj).items():
                if attr.startswith("_"):
                    continue
                if isinstance(val, property):
                    val = val.fget
                elif isinstance(val, (staticmethod, classmethod)):
                    val = val.__func__
                if inspect.isfunction(val):
                    yield f"{name}.{attr}", val


def test_no_docstring_holds_a_bare_tag():
    """A ``<name>`` outside code is an HTML tag to Markdown: an unknown name
    disappears from the page and a known one (``<i>``) formats the rest. No
    docstring uses HTML on purpose, so any tag outside code fails."""
    bad = [f"{name}: {tag}" for name, obj in _rendered_objects()
           for tag, _ in _tags(_docstring_prose(inspect.getdoc(obj) or ""))]
    assert bad == []


def test_the_prose_filter_keeps_bare_placeholders_and_drops_code():
    doc = ("Nothing runnable: the \"No method can run on <dataset> (<category>).\"\n"
           "``atac<i>.h5`` and `<out_dir>` are code.\n\n"
           "Table::\n\n    <method>_<dataset>\n\n>>> run('<x>')\n<Figure>\n")
    assert [t for t, _ in _tags(_docstring_prose(doc))] == ["<dataset>", "<category>"]


def _page_prose(text: str) -> str:
    """A docs page without HTML comments, fenced code and inline code."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"^([ \t]*)(```|~~~).*?^\1\2[^\n]*$", "", text, flags=re.S | re.M)
    text = re.sub(r"``.*?``", "", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def test_no_docs_page_holds_a_bare_placeholder_tag():
    root = os.environ.get("SCMULTIBENCH_DOCS")
    if not root or not Path(root).is_dir():
        pytest.skip("SCMULTIBENCH_DOCS not set")
    pages = sorted(Path(root).rglob("*.md"))
    assert pages
    bad = [f"{p.relative_to(root)}: {tag}" for p in pages
           for tag, name in _tags(_page_prose(p.read_text())) if name not in _ELEMENTS]
    assert bad == []


def test_run_all_notes_quote_the_live_first_lines(no_envs, tmp_path):
    doc = _flat(mtb.run_all)
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir=tmp_path, verbose=False)
    first = str(e.value).splitlines()[0]
    assert first == "No method can run on D11 (vertical)."
    assert f"Nothing runnable: ``ValueError``. Its first line is ``{first}``" in doc
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir=tmp_path, methods=["Matilda", "totalVI"],
                    verbose=False)
    first = str(e.value).splitlines()[0]
    assert f"With ``methods=``, it starts ``{first}``" in doc
    # the other facts of the bullet are kept, as sentences
    assert "Without ``methods``, it gives the first 3 of N." in doc
    assert "never lists the reasons of methods you did not ask for" in doc
    assert "its second line says that methods run only on Linux" in doc


# ============================================================ R7-05
def test_long_names_every_column_it_returns():
    wide = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        attached = mtb.to_long(wide, method="Matilda", dataset="D11", category="vertical")
    res = W.BatchResult([{"method": "Matilda", "status": "CHAIN_OK", "_long": attached,
                          "metrics": {"ARI": 0.5, "NMI": 0.6}}], "D11", "vertical")
    doc = inspect.getdoc(W.BatchResult.long)
    assert "scored_with" in res.long.columns
    for col in res.long.columns:
        assert col in doc, col
    # rebuilt from the metrics dict, or empty: the first seven columns
    rebuilt = W.BatchResult([{"method": "Matilda", "status": "CHAIN_OK",
                              "metrics": {"ARI": 0.5}}], "D11", "vertical").long
    empty = W.BatchResult([], "D11", "vertical").long
    assert list(rebuilt.columns) == list(empty.columns) == list(attached.columns)[:7]
    summary = doc.splitlines()[0]
    assert summary == "The scores as a long table, for plotting."
    assert "Empty, with the first seven columns, when no method produced metrics." \
        in " ".join(doc.split())


# ============================================================ R7-06
def test_rescore_verbose_names_the_ranking_line():
    doc = _flat(W.BatchResult.rescore)
    assert ("verbose : bool Print one line per method, and one before each "
            "label-order ranking.") in doc


# ============================================================ R7-12
def test_workflow_passages_read_as_plain_sentences():
    cls = inspect.getdoc(W.BatchResult).splitlines()[0]
    assert cls == "The result of ``mtb.run_all``: its summary table, long table and figure."
    load = _flat(mtb.load_batch)
    assert ("data_path : path-like | None Folder that holds the dataset folder; "
            "``None`` = the path each record saved.") in load
    summary = _flat(W.BatchResult.summary)
    assert ("**Optimistic bias.** When more than one ordering is possible, the reported "
            "metrics are those of the ordering with the highest ARI. So they are slightly "
            "optimistic. ``label_order_confidence`` shows how far ahead the chosen order "
            "was.") in summary
    rescore = _flat(W.BatchResult.rescore)
    assert ("**Other hosts.** When the data folder has moved, give its new location to "
            "``mtb.load_batch(data_path=)``. When the dataset folder is not found, ") \
        in rescore
    for old in ("Outcome of", "Data root for re-scoring", "finds moved folders",
                "clear-cut", "small optimistic bias"):
        assert old not in " ".join([cls, load, summary, rescore]), old
