"""The stored scores the tutorials draw, checked against the live package.

Section 6 of a task tutorial whose task has stored scores reads the package's
stored re-run scores;
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
KEYS = list(GEN.TASKS)
# the tasks whose tutorial has a stored-scores section
STORED = [k for k in KEYS if GEN.TASKS[k].get("stored_ds")]
GENERATED = [f"tutorial_{k}{suffix}" for k in KEYS for suffix in ("", "_all")
             if not suffix or GEN.has_all(k)]
NOTEBOOKS = GENERATED + ["tutorial_end_to_end", "colab_quickstart"]
# the notebooks that draw stored scores
DRAWING = [f"tutorial_{k}" for k in STORED] + ["tutorial_end_to_end", "colab_quickstart"]


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
    assert (n >= 1) == (name in DRAWING), name


def test_the_stored_scores_section_follows_the_task():
    """A tutorial has the section only when its task names a stored dataset,
    and the four benchmark datasets with stored scores each have one."""
    for key in KEYS:
        heads = [src.splitlines()[0] for kind, src in _cells(f"tutorial_{key}")
                 if kind == "markdown" and "Stored scores" in src.splitlines()[0]]
        assert heads == (["## 6. Stored scores"] if key in STORED else []), key
    assert sorted(GEN.TASKS[k]["stored_ds"] for k in STORED) == ["D11", "D28", "D45", "D52"]


@pytest.mark.parametrize("key", STORED)
def test_stored_scores_section_states_the_measured_counts(key):
    """The method counts of section 6 are those load_results gives, and the
    sentence on source="published" holds on the live package."""
    t = GEN.TASKS[key]
    cat, ds, shown = t["cat"], t["stored_ds"], t["ds"]
    cells = _cells(f"tutorial_{key}")
    md = next(src for kind, src in cells
              if kind == "markdown" and src.startswith("## 6. Stored scores"))
    md = " ".join(md.split())
    code = "\n".join(src for kind, src in cells if kind == "code")
    assert f'mtb.load_results("{cat}", dataset="{ds}", source="rerun")' in code
    cov = mtb.results_coverage(cat)
    if ds != shown:
        # a tutorial-size subset: its own scores are never stored
        assert cov[cov.dataset == shown].empty
        if ds in shown:
            assert (f"The stored scores are for the full `{ds}`, the benchmark dataset "
                    f"`{shown}` is drawn from.") in md
        else:
            assert (f"There are no stored scores for `{shown}`. `{ds}` is another "
                    f"benchmark dataset of the same task.") in md
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
        assert (f"There is no published scIB table for {cat}, so `source=\"published\"`, "
                f"the default, raises `FileNotFoundError`.") in md


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
    # the stored UINMF run is its two-batch variant; the variant the folder
    # rule picks on D52 today reads all three, and the tutorial says so
    assert list(mtb.labels_for("D52", "cross", "UINMF",
                               modalities=["rna1", "rna2", "adt1", "adt2"])) \
        == stored("D52", "UINMF") == ["cty1", "cty2"]
    assert list(mtb.labels_for("D52", "cross", "UINMF")) == ["cty1", "cty2", "cty3"]
    nb = json.loads((ROOT / "notebooks" / "tutorial_cross_rna_adt.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert ("The stored scores of UINMF are from its two-batch variant, which read batches "
            "1 and 2 of `D52`. `run_all` on `D52mini` runs its three-batch variant.") in text
