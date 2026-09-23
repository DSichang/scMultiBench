"""User-visible text of the evaluation and plotting API (student study, round 1).

- The bubble legend says the fill is scaled per column (S1-03): with two rows,
  an ARI of 0.537 was drawn white and read as zero.
- evaluate's reference shows the call an AnnData or MuData user makes
  (S1-07, S2-11).
- No emphasis by capitals in docstrings or messages (L61).
"""
import ast
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.eval import pipeline

matplotlib.use("Agg")

PKG = Path(mtb.__file__).parent


def _two_rows():
    return pd.DataFrame(
        [{"method": m, "metric": k, "value": v, "dataset": "D1"}
         for m, vals in (("RNA_PCA", (0.537, 0.60)), ("Other", (0.564, 0.55)))
         for k, v in zip(("ARI", "NMI"), vals)])


def test_score_legend_says_the_fill_is_scaled_per_column():
    import importlib
    bmod = importlib.import_module("multibench.plot.bubble")
    fig = mtb.plot.bubble(_two_rows())
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Score" in texts and "(scaled per column)" in texts
    assert "Low" in texts and "High" in texts
    assert bmod.SCORE_SCALE_NOTE == "(scaled per column)"
    # the caption sits under the title, clear of the ramp's "Low" label
    ax = fig.axes[0]
    by = {t.get_text(): t.get_position() for t in ax.texts}
    assert by["(scaled per column)"][0] == by["Score"][0]
    assert by["(scaled per column)"][1] < by["Score"][1] - 0.4


def test_bubble_notes_say_the_lightest_fill_is_not_zero():
    doc = " ".join(mtb.plot.bubble.__doc__.split())
    assert "The lightest fill is the lowest value of that column in this figure, not zero." in doc
    assert '"(scaled per column)"' in doc


def test_evaluate_examples_show_the_anndata_and_mudata_calls():
    doc = pipeline.evaluate.__doc__
    assert '>>> mtb.evaluate(adata, labels="celltype", obsm="X_pca")' in doc
    assert ('>>> mtb.evaluate(mdata, labels="celltype", batch="sample", obsm="X_joint",'
            in doc)
    flat = " ".join(doc.split())
    assert "AnnData or MuData (``.obsm[obsm]``; ``labels``/``batch`` may name ``.obs`` columns)" in flat
    assert ("``mtb.to_long`` makes the long frame (lowercase ``value``) that "
            "``load_results`` returns") in flat


def test_evaluate_takes_a_mudata_with_obs_keys():
    ad = pytest.importorskip("anndata")
    md = pytest.importorskip("mudata")
    pytest.importorskip("scib")
    rng = np.random.default_rng(0)
    n = 90
    ct = np.repeat(["a", "b", "c"], n // 3)
    rna = ad.AnnData(rng.normal(size=(n, 4)))
    rna.obs_names = [f"c{i}" for i in range(n)]
    m = md.MuData({"rna": rna})
    m.obs["celltype"] = ct
    m.obs["sample"] = np.tile(["s1", "s2"], n // 2)
    m.obsm["X_joint"] = rng.normal(size=(n, 5)) + pd.Categorical(ct).codes[:, None] * 4
    out = mtb.evaluate(m, labels="celltype", batch="sample", obsm="X_joint",
                       metrics=["ASW", "ASW_batch"])
    assert list(out.index) == ["ASW", "ASW_batch"]


# words written in capitals to shout; acronyms and code names are not flagged
_ACRONYMS = {"ARI", "NMI", "ASW", "GC", "LISI", "CSV", "TSV", "SD", "DR", "API",
             "PCR", "PATH", "README", "PEP", "OK", "RNA", "ADT", "ATAC", "UMAP",
             "PCA", "HVG", "GPU", "CPU", "ID", "IDS", "URL", "R", "DDDDDD", "NA"}


def _user_visible_strings(path: Path):
    """Docstrings and string literals (messages) of one module."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and not node.value.isidentifier():        # names in __all__ etc.
            yield node.lineno, node.value


@pytest.mark.parametrize("path", sorted((PKG / "eval").glob("*.py"))
                         + sorted((PKG / "plot").glob("*.py")), ids=lambda p: p.name)
def test_no_emphasis_by_capitals(path):
    shouted = []
    for lineno, text in _user_visible_strings(path):
        for word in re.findall(r"(?<![\w`.#/-])[A-Z]{2,}(?![\w`])", text):
            if word not in _ACRONYMS:
                shouted.append(f"{path.name}:{lineno}: {word}")
    assert shouted == []
