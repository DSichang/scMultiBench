"""The tutorials after fix round 2 of the student study.

The package changed underneath the notebooks: a one-method bubble table is
grey and warns, and igraph-scored rows next to the stored tables warn (M02,
M03); labels_for returns only the label files of the batches a variant
reads, and scan says so in ``caveat`` (M31); Seurat_v5 needs RNA and ATAC
from the same cells (M15). These tests pin what the notebooks say and do
about each, on the committed notebooks (the generated four and the
hand-maintained end-to-end tutorial).
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


# ------------------------------------------------------ export keywords
def test_export_dataset_takes_every_keyword_the_notebooks_pass():
    """The install cell asks for no minimum release, so the installed package
    must accept every keyword the export cells pass (batch_index=, overwrite=)."""
    from IPython.core.inputtransformer2 import TransformerManager
    import multibench as mtb
    params = set(inspect.signature(mtb.io.export_dataset).parameters)
    passed = set()
    for name in [f"tutorial_{c}" for c in CATS] + [E2E, "colab_quickstart"]:
        for src in _code(name):
            tree = ast.parse(TransformerManager().transform_cell(src))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "mtb.io.export_dataset":
                    passed |= {k.arg for k in node.keywords}
    assert {"batch_index", "overwrite"} <= passed
    assert passed <= params, passed - params


# ------------------------------------------------------- generated tutorials
@pytest.mark.parametrize("cat", CATS)
def test_plot_details_say_constant_columns_are_grey(cat):
    """M02: a column whose rows all hold one value is drawn grey and named in
    the footnote; the Plot section says so, in the words the figure uses."""
    md = _markdown(f"tutorial_{cat}", "## 4. Plot")
    assert "A column whose rows all hold the same value is drawn grey" in md
    bubble = importlib.import_module("multibench.plot.bubble")      # the module, not the function
    assert bubble._constant_note(["A"], {}).startswith("Grey fill")
    assert bubble._constant_note(["A", "B"], {"iLISI": 0.0}).startswith("Grey fill: all rows equal in iLISI")


def test_uinmf_reads_two_of_d52s_three_batches():
    """M31: UINMF's cross variant reads batches 1-2 of D52's three;
    labels_for and scan's caveat say so."""
    import multibench as mtb
    assert list(mtb.labels_for("D52", "cross", "UINMF")) == ["cty1", "cty2"]
    sc = mtb.scan("D52", "cross", methods=["UINMF"], verbose=False)
    assert sc.caveat.iloc[0] == "UINMF reads batches 1-2 of 3. Batch 3 is not used."


def test_diagonal_tutorial_states_seurat_v5_cells():
    """M15: Seurat_v5 needs RNA and ATAC from the same cells, a visible fact
    next to the ATAC forms."""
    import multibench as mtb
    title = _visible(_cells("tutorial_diagonal")[0][1])
    assert "Seurat_v5 also needs RNA and ATAC from the same cells." in title
    assert "same cells" in mtb.method_info("Seurat_v5")["setup_hint"]


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
    """Section 8 plots this run next to the stored D11 scores, so section 7
    sets the stored scores' Leiden backend before evaluate (M03: igraph rows
    next to stored rows warn)."""
    import numpy as np
    import pandas as pd
    import multibench as mtb
    monkeypatch.setattr(mtb.config.DEFAULT, "leiden_flavor", "igraph")
    seen = []
    ns = {"mtb": _Recorder(mtb, seen), "pd": pd, "emb": np.zeros((3, 2)),
          "labels": mtb.labels_for("D11")}
    exec(_e2e_cell("mtb.evaluate("), ns)
    assert seen == ["leidenalg"]
    md = _visible(_markdown(E2E, "## 7. Score"))
    assert "Leiden backend of the stored scores" in md


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


def _run_plot_cell(scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import multibench as mtb
    ns = {"mtb": mtb, "pd": pd, "scores": scores}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        exec(_e2e_cell("mtb.plot.bubble("), ns)
    plt.close("all")
    ours = [str(w.message) for w in seen if issubclass(w.category, UserWarning)]
    return ns, ours


def test_end_to_end_plot_compares_this_run_with_the_stored_table():
    """M02 greys a one-method figure and warns: section 8 plots this run as
    its own row next to the stored D11 results, with no multibench warning
    when section 7 used the stored scores' backend. With igraph scores the
    M03 warning fires, which is why section 7 sets leidenalg."""
    import pandas as pd
    ns, ours = _run_plot_cell(_scores_like_evaluate("leidenalg"))
    long = pd.concat([ns["stored"], ns["mine"]], ignore_index=True)
    assert "Matilda (this run)" in set(long.method) and "Matilda" in set(long.method)
    assert long.method.nunique() == ns["stored"].method.nunique() + 1
    assert ours == [], ours
    _, ours = _run_plot_cell(_scores_like_evaluate("igraph"))
    assert any("igraph Leiden backend" in m for m in ours), ours
