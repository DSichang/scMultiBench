"""The tutorials after fix round 2 of the student study.

The package changed underneath the notebooks: 0.3.2 is the first release with
every call they make; a one-method bubble table is grey and warns, and
igraph-scored rows next to the stored tables warn (M02, M03); to_long of a
frame without a scoring record warns (M35); labels_for returns only the
label files of the batches a variant reads, and scan says so in ``caveat``
(M31); Seurat_v5 needs RNA and ATAC from the same cells (M15); a diagonal
atac_gas.h5 must follow atac_peak.h5's cell order (M16); method_info gains
``gpu`` and scan gains ``assume_gpu`` (M28, M30). These tests pin what the
notebooks say and do about each, on the committed notebooks (the generated
four plus the quickstart, and the hand-maintained end-to-end tutorial).
"""
import ast
import importlib.util
import inspect
import json
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
CATS = list(GEN.SCEN)
E2E = "tutorial_end_to_end"


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _code(name):
    return [src for kind, src in _cells(name) if kind == "code"]


def _markdown(name, start):
    return next(src for kind, src in _cells(name) if kind == "markdown" and src.startswith(start))


def _visible(md):
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


# ------------------------------------------------------------ install floor
def test_install_cell_asks_for_the_first_release_with_every_call_the_notebooks_make():
    """0.3.1 on PyPI lacks export_dataset(batch_index=, overwrite=), which the
    mosaic demo calls: the pip line asks for 0.3.2 or later, and the package
    that runs the tests is at least that release."""
    from packaging.version import Version
    import multibench as mtb
    pip = [line for line in GEN.INSTALL_CELLS[0].splitlines() if "pip -q install" in line]
    floor = re.search(r'"multibench-sc>=([0-9.]+)"', pip[0])
    assert floor and floor.group(1) == "0.3.2", pip[0]
    assert Version(mtb.__version__) >= Version(floor.group(1))
    params = inspect.signature(mtb.io.export_dataset).parameters
    assert {"batch_index", "overwrite"} <= set(params)


# ------------------------------------------------------- generated tutorials
@pytest.mark.parametrize("cat", CATS)
def test_plot_details_say_constant_columns_are_grey(cat):
    """M02: a column whose rows all hold one value is drawn grey and named in
    the footnote; the Plot section says so, in the words the figure uses."""
    md = _markdown(f"tutorial_{cat}", "### Plot")
    assert "A column whose rows all hold the same value is drawn grey" in md
    assert "A figure of one method is grey everywhere" in md
    bubble = importlib.import_module("multibench.plot.bubble")      # the module, not the function
    assert bubble._constant_note(["A"], {}).startswith("Grey fill")
    assert bubble._constant_note(["A", "B"], {"iLISI": 0.0}).startswith("Grey fill: all rows equal in iLISI")


@pytest.mark.parametrize("cat", CATS)
def test_reference_scan_table_shows_the_caveat_column(cat):
    """scan's caveat column carries the notes a run needs (GLUE's peak names
    and setup on D28, UINMF's unused batch on D52); section 5 shows it and
    says what it is, with env_reason moved into the details."""
    code = "\n".join(_code(f"tutorial_{cat}"))
    assert '"runtime_tier", "caveat"]]' in code
    md = _markdown(f"tutorial_{cat}", "## 5. Reference")
    assert "`caveat` says what to check before a run" in _visible(md)
    assert "`env_reason` gives the install command" in md


def test_cross_tutorial_says_which_batches_uinmf_reads():
    """M31: UINMF's cross variant reads batches 1-2 of D52's three. The run
    details and the label-order details say so, from labels_for, and scan's
    caveat on D52 says the same."""
    import multibench as mtb
    assert list(mtb.labels_for("D52", "cross", "UINMF")) == ["cty1", "cty2"]
    sc = mtb.scan("D52", "cross", methods=["UINMF"], verbose=False)
    assert sc.caveat.iloc[0] == "UINMF reads batches 1-2 of 3. Batch 3 is not used."
    run = _markdown("tutorial_cross", "### Run the methods")
    assert ("UINMF reads only batches 1 and 2 of 3, and `scan` says so in its `caveat` "
            "column.") in run
    labels = next(src for kind, src in _cells("tutorial_cross")
                  if kind == "markdown" and "Details: label order" in src)
    assert "A method that reads only some batches gets only their label files: `cty1, cty2` for UINMF." in labels
    assert GEN.partial_label_methods("cross", "D52") == {"UINMF": ["cty1", "cty2"]}
    for cat in ("vertical", "diagonal", "mosaic"):
        assert "reads only batches" not in "\n".join(src for _, src in _cells(f"tutorial_{cat}"))


def test_diagonal_tutorial_states_seurat_v5_cells_and_the_gas_order():
    """M15: Seurat_v5 needs RNA and ATAC from the same cells, a visible fact
    next to the ATAC forms. M16: with both ATAC files, atac_gas.h5 follows
    atac_peak.h5's cell order; the export details say so."""
    import multibench as mtb
    title = _visible(_cells("tutorial_diagonal")[0][1])
    assert "Seurat_v5 also needs RNA and ATAC from the same cells." in title
    assert "same cells" in mtb.method_info("Seurat_v5")["setup_hint"]
    export = next(src for kind, src in _cells("tutorial_diagonal")
                  if kind == "markdown" and "Details: export" in src)
    assert ("With both ATAC files, `atac_gas.h5` must list the cells of `atac_peak.h5` "
            "in the same order") in export
    assert '`mtb.io.to_canonical(..., modality="gas")`' in export


@pytest.mark.parametrize("cat", CATS)
def test_method_record_shows_gpu_and_troubleshooting_names_assume_gpu(cat):
    """M30 and M28: the record cell shows method_info's gpu field and the
    details list its values; the troubleshooting table points a computer
    without a GPU at scan(..., assume_gpu=True)."""
    import multibench as mtb
    cell = next(src for src in _code(f"tutorial_{cat}") if "mtb.method_info(" in src and "verbose=True" in src)
    keys = ast.literal_eval(re.search(r"for k in (\(.*?\))", cell).group(1))
    info = mtb.method_info(GEN.SCEN[cat]["live"][0], verbose=True)
    assert "gpu" in keys and set(keys) <= set(info)
    values = sorted({mtb.method_info(m)["gpu"] for m in mtb.list_methods()})
    md = _markdown(f"tutorial_{cat}", "### A method's record")
    for v in values:
        assert f"`{v}`" in md, v
    trouble = _markdown(f"tutorial_{cat}", "## Troubleshooting")
    assert "`mtb.scan(..., assume_gpu=True)`" in trouble
    assert "assume_gpu" in inspect.signature(mtb.scan).parameters


# ------------------------------------------------------------- end-to-end
def _e2e_cell(token):
    found = [src for src in _code(E2E) if token in src]
    assert len(found) == 1, token
    return found[0]


class _Recorder:
    """Enough of ``mtb`` for the evaluate cell: evaluate records the Leiden
    backend set when it is called."""

    def __init__(self, mtb, seen):
        self._mtb, self._seen = mtb, seen

    def __getattr__(self, name):
        return getattr(self._mtb, name)

    def evaluate(self, *a, **kw):
        import pandas as pd
        self._seen.append(self._mtb.config.DEFAULT.leiden_flavor)
        return pd.DataFrame({"Value": [0.5]}, index=["ARI"])


def test_end_to_end_evaluates_with_the_backend_of_the_stored_tables(monkeypatch):
    """Section 6 plots this run next to the stored D11 table, so section 5
    sets the stored tables' Leiden backend before evaluate (M03: igraph rows
    next to stored rows warn)."""
    import numpy as np
    import pandas as pd
    import multibench as mtb
    monkeypatch.setattr(mtb.config.DEFAULT, "leiden_flavor", "igraph")
    seen = []
    ns = {"mtb": _Recorder(mtb, seen), "pd": pd, "emb": np.zeros((3, 2)),
          "DATASET": "D11", "CATEGORY": "vertical"}
    exec(_e2e_cell("mtb.evaluate("), ns)
    assert seen == ["leidenalg"]
    md = _visible(_markdown(E2E, "## 5. Evaluate"))
    assert "Leiden backend of the stored tables" in md


def _scores_like_evaluate(flavor):
    """Matilda's stored D11 scores in evaluate's shape, with the scoring
    record evaluate attaches."""
    import pandas as pd
    import multibench as mtb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stored = mtb.load_results("vertical", dataset="D11", source="rerun", methods=["Matilda"])
    df = pd.DataFrame({"Value": dict(zip(stored.metric, stored.value))})
    df.attrs = {"leiden_flavor": flavor, "clustering": "sweep",
                "multibench_version": mtb.__version__, "scib_version": "1.1.7"}
    return df


def _run_plot_cell(emb, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import multibench as mtb
    ns = {"mtb": mtb, "pd": pd, "emb": emb, "scores": scores,
          "DATASET": "D11", "CATEGORY": "vertical", "display": lambda fig: None}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        exec(_e2e_cell("mtb.plot.bubble(long)"), ns)
    plt.close("all")
    ours = [str(w.message) for w in seen if issubclass(w.category, UserWarning)]
    return ns, ours


def test_end_to_end_plot_compares_this_run_with_the_stored_table():
    """M02 greys a one-method figure and warns: section 6 plots this run as
    its own row next to the stored D11 results, with no multibench warning
    when section 5 used the stored tables' backend. With igraph scores the
    M03 warning fires, which is why section 5 sets leidenalg."""
    import numpy as np
    import multibench as mtb
    ns, ours = _run_plot_cell(np.zeros((3, 2)), _scores_like_evaluate("leidenalg"))
    long = ns["long"]
    assert "Matilda (this run)" in set(long.method) and "Matilda" in set(long.method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_stored = mtb.load_results("vertical", dataset="D11", source="rerun").method.nunique()
    assert long.method.nunique() == n_stored + 1
    assert ours == [], ours
    _, ours = _run_plot_cell(np.zeros((3, 2)), _scores_like_evaluate("igraph"))
    assert any("igraph Leiden backend" in m for m in ours), ours


def test_end_to_end_plot_without_an_embedding_draws_the_stored_table_alone():
    """Offline, section 5 shows Matilda's stored scores (a frame with no
    scoring record); section 6 plots the stored table without passing them
    through to_long, so M35's 'no record of how they were scored' warning
    does not appear."""
    import pandas as pd
    ns, ours = _run_plot_cell(None, pd.DataFrame({"Value": {"ARI": 0.9}}))
    assert "Matilda (this run)" not in set(ns["long"].method)
    assert ours == [], ours


def test_end_to_end_draws_the_stored_d11_table_once():
    """Section 6 already shows the stored D11 results next to this run; section
    7 keeps the summary table and no second D11 figure."""
    code = _code(E2E)
    assert not any('stored_table("long_all", "D11")' in src for src in code)
    section7 = _markdown(E2E, "## 7. Comparing all 14 methods")
    assert "section 6" in section7


def test_end_to_end_run_times_match_method_info():
    """The Run time table lists observed times; each one is the value
    method_info(m)['runtime'] gives, and the note on GPUs names the gpu field
    instead of saying every training method is slower on a CPU."""
    import multibench as mtb
    md = _markdown(E2E, "### Run time")
    rows = re.findall(r"^\| (\w+) \| (D\d+) \(([\d,]+)\) \| ([\d.]+) (s|h) \|$", md, re.M)
    assert len(rows) >= 5
    for method, ds, cells, value, unit in rows:
        obs = {o["dataset"]: o for o in mtb.method_info(method)["runtime"]["observed"]}[ds]
        assert obs["cells"] == int(cells.replace(",", "")), (method, ds)
        sec = float(value) * (3600 if unit == "h" else 1)
        tol = 0.05 * 3600 if unit == "h" else 0.5
        assert abs(obs["sec"] - sec) <= tol, (method, ds, obs["sec"], value, unit)
    assert '`mtb.method_info(m)["gpu"]`' in md
    assert "Without a GPU, training methods take much longer" not in md
