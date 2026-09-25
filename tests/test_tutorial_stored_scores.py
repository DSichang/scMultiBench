"""The stored scores the tutorials draw, checked against the live package.

Section 6 of each category tutorial reads the package's stored re-run scores;
``notebooks/results`` holds a second copy of the same tables. These tests pin
the counts and the error the section states, the ``source=`` every
``load_results`` call names, and the agreement of the two copies.
"""
import ast
import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "notebooks" / "results"


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
CATS = list(GEN.SCEN)
NOTEBOOKS = [f"tutorial_{c}" for c in CATS] + ["tutorial_end_to_end", "colab_quickstart"]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _tree(src):
    ipy = pytest.importorskip("IPython.core.inputtransformer2")
    return ast.parse(ipy.TransformerManager().transform_cell(src))


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_load_results_call_names_its_source(name):
    """The default source raises for mosaic, so every call names one."""
    n = 0
    for kind, src in _cells(name):
        if kind != "code":
            continue
        for node in ast.walk(_tree(src)):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "mtb.load_results":
                n += 1
                assert any(k.arg == "source" for k in node.keywords), \
                    f"{name}: load_results without source=: {ast.unparse(node)[:80]}"
    assert n >= 1, name


@pytest.mark.parametrize("cat", CATS)
def test_stored_scores_section_states_the_measured_counts(cat):
    """The method counts of section 6 are those load_results gives, and the
    sentence on source="published" holds on the live package."""
    s = GEN.SCEN[cat]
    ds = s.get("stored_ds", s["ds"])
    md = next(src for kind, src in _cells(f"tutorial_{cat}")
              if kind == "markdown" and src.startswith("## 6. Stored scores"))
    md = " ".join(md.split())
    cov = mtb.results_coverage(cat)
    if ds != s["ds"]:
        assert cov[cov.dataset == s["ds"]].empty
        assert f"There are no stored scores for `{s['ds']}`." in md
    cov = cov[cov.dataset == ds]
    rerun = _quiet(mtb.load_results, cat, dataset=ds, source="rerun")
    assert f"stored scores for {rerun.method.nunique()} methods on `{ds}`" in md
    n_pub = cov[cov.source == "published"].method.nunique()
    if n_pub:
        published = _quiet(mtb.load_results, cat, dataset=ds)
        assert published.method.nunique() == n_pub
        assert re.search(rf"which hold {n_pub} methods? for `{ds}`", md), md
    else:
        with pytest.raises(FileNotFoundError):
            _quiet(mtb.load_results, cat, dataset=ds)
        assert '`source="published"`, the default, raises `FileNotFoundError`' in md


@pytest.mark.parametrize("ds", ["D11", "D28", "D45", "D52"])
def test_the_package_rerun_tables_match_the_results_folder(ds):
    """notebooks/results and the package's stored re-run tables are two copies
    of the same scores."""
    long_pkg = _quiet(mtb.load_results, dataset=ds, source="rerun")
    long_here = pd.read_csv(RESULTS / f"long_all_{ds}.csv")
    both = long_here.merge(long_pkg, on=["method", "metric"], how="outer", indicator=True)
    assert (both._merge == "both").all() and np.allclose(both.value_x, both.value_y), ds
    summary_pkg = (long_pkg.pivot_table(index="method", columns="metric", values="value")
                   .rename_axis(columns=None).reset_index())
    summary_here = pd.read_csv(RESULTS / f"summary_{ds}.csv")
    both = summary_here.merge(summary_pkg, on="method", how="outer", indicator=True)
    assert (both._merge == "both").all(), ds
    for m in ("ARI", "NMI", "ASW"):
        assert np.allclose(both[f"{m}_x"], both[f"{m}_y"], atol=1e-4), (ds, m)


def test_stored_summaries_record_the_order_labels_for_returns():
    """The label_order column of the stored summaries agrees with
    labels_for(dataset, category, method), also for a method that reads only
    some batches."""
    def stored(ds, method):
        df = pd.read_csv(RESULTS / f"summary_{ds}.csv")
        return [Path(f).stem for f in df.set_index("method").at[method, "label_order"].split("+")]

    assert list(mtb.labels_for("D52", "cross", "StabMap")) == stored("D52", "StabMap") \
        == ["cty3", "cty1", "cty2"]
    assert list(mtb.labels_for("D28", "diagonal", "uniPort")) == stored("D28", "uniPort") \
        == ["atac_cty", "rna_cty"]
    assert list(mtb.labels_for("D52", "cross", "UINMF")) == stored("D52", "UINMF") \
        == ["cty1", "cty2"]
