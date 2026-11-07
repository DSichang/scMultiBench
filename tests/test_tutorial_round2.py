"""The tutorials after fix round 2 of the student study.

The package changed underneath the notebooks: a one-method bubble table is
grey and warns, and igraph-scored rows next to the stored tables warn (M02,
M03); labels_for returns only the label files of the batches a variant
reads, and scan says so in ``caveat`` (M31); Seurat_v5 needs RNA and ATAC
from the same cells (M15). These tests pin what the notebooks say and do
about each, on the committed notebooks (the generated task tutorials).
"""
import ast
import importlib.util
import inspect
import json
import re
from pathlib import Path

import pytest
from tests.conftest import full_scan

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
KEYS = list(GEN.TASKS)
TUTORIALS = [f"tutorial_{k}" for k in KEYS]
EVERY = [f"tutorial_{k}_all" for k in KEYS if GEN.has_all(k)]        # every method of a task


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
    for name in TUTORIALS + EVERY + ["colab_quickstart"]:
        for src in _code(name):
            tree = ast.parse(TransformerManager().transform_cell(src))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "mtb.io.export_dataset":
                    passed |= {k.arg for k in node.keywords}
    assert {"batch_index", "overwrite"} <= passed
    assert passed <= params, passed - params


# ------------------------------------------------------- generated tutorials
@pytest.mark.parametrize("name", TUTORIALS)
def test_plot_details_say_constant_columns_are_grey(name):
    """M02: a column with fewer than two different values is drawn grey; the
    Plot section says so. bubble warns only where two or more rows hold the one
    value, so the text promises no warning."""
    md = _markdown(name, "## 4. Plot")
    assert "A column with fewer than two different values is drawn grey." in md
    assert "a warning names it" not in md


def test_uinmf_reads_every_batch_of_d52_and_its_two_batch_variant_says_what_it_skips():
    """M31: UINMF has a three-batch cross variant, which the folder rule
    picks on D52's three batches. Its two-batch variant reads batches 1-2;
    labels_for and the caveat of that scan row say so."""
    import multibench as mtb
    two = ["rna1", "rna2", "adt1", "adt2"]
    assert list(mtb.engine.resolve.labels_for("D52", "cross", "UINMF")) == ["cty1", "cty2", "cty3"]
    assert list(mtb.engine.resolve.labels_for("D52", "cross", "UINMF", modalities=two)) == ["cty1", "cty2"]
    sc = full_scan("D52", "cross", methods=["UINMF"], verbose=False)
    sc = sc[sc.files_ok].set_index("modalities")
    # the dry run's one line says the three-batch row runs instead (X68, r3)
    assert sc.at["+".join(two), "caveat"] == (
        "UINMF reads batches 1-2 of 3. Batch 3 is not used.")
    assert not sc.at["rna1+rna2+rna3+adt1+adt2+adt3", "caveat"]
