"""Round 4 of the student study: what the docs pages now say (get-started
R4-15 to R4-17, run guide R4-18, API overview and Changes R4-19), checked
against the live package where the page quotes a number or runs code.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it. The round-3 assertions that R4-16 replaced were re-pointed
in place (tests/test_docs_r3.py, tests/test_review_f3_int.py,
tests/test_docs_consistency.py).
"""
import os
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import multibench as mtb


def _docs_root():
    root = os.environ.get("SCMULTIBENCH_DOCS")
    return Path(root) if root and Path(root).is_dir() else None


needs_docs = pytest.mark.skipif(_docs_root() is None, reason="SCMULTIBENCH_DOCS not set")


def _read(page):
    return (_docs_root() / page).read_text()


def _flat(page):
    return " ".join(_read(page).split())


def _visible(text):
    """The page without its collapsed ``??? `` Details blocks."""
    out, skip = [], False
    for line in text.splitlines():
        if line.startswith("??? "):
            skip = True
            continue
        if skip and line.strip() and not line.startswith("    "):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)


def _step1_visible():
    return _flat("quickstart.md").split("## Step 1:", 1)[1].split("```", 1)[0]


# ---- R4-15: the laptop tab prints the dry-run command ----------------------
def _laptop_tab():
    section = _read("quickstart.md").split("### Your own data", 1)[1].split("## Step 4", 1)[0]
    return section.split("```python", 1)[1].split("```", 1)[0]


def _cite_adata(n_cells=400, n_genes=2500, n_adt=14, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.choice(["T", "B", "NK"], n_cells)
    shift = pd.Series(types).map({"T": 0.5, "B": 1.5, "NK": 3.0}).to_numpy()[:, None]
    a = ad.AnnData(rng.poisson(0.3 + shift * rng.random(n_genes), (n_cells, n_genes))
                   .astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["celltype"] = types
    a.obsm["protein"] = rng.poisson(5 + 10 * shift * rng.random(n_adt),
                                    (n_cells, n_adt)).astype("float32")
    return a


@needs_docs
def test_laptop_tab_prints_the_totalvi_command_before_the_table(tmp_path, monkeypatch, capsys):
    import matplotlib.pyplot as plt
    code = _laptop_tab()
    # the default modalities are left out since 0.4.0 (X48)
    assert 'mtb.scan("data/MYCITE", "vertical")' in code
    assert max(len(line) for line in code.splitlines()) <= 80
    monkeypatch.chdir(tmp_path)
    _cite_adata().write_h5ad(tmp_path / "my_citeseq.h5ad")
    scope = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "quickstart-laptop-tab", "exec"), scope)
    out = capsys.readouterr().out
    lines = out.splitlines()
    scan_line = [i for i, l in enumerate(lines) if l.startswith("[scan] ")]
    table = [i for i, l in enumerate(lines) if l.split()[:1] == ["metric"]]
    assert scan_line and table and scan_line[0] < table[0], out
    plt.close("all")


# ---- R4-16, then the 0.5.1 docs review: one line on donors, the recipe has the rest --------
@needs_docs
def test_step1_donor_advice_is_one_line_that_links_the_recipe():
    step1 = _step1_visible()
    # a pointer only: whether vertical or cross fits depends on the donor count (round 2)
    assert ("Several samples or donors: see "
            "[Another number of donors](tutorials/run.md#recipe-donors).") in step1
    assert "use `vertical`" not in step1 and "cross" not in step1
    recipe = " ".join(_read("tutorials/run.md").split("{ #recipe-donors }", 1)[1].split())
    assert ("A cross method reads a fixed number of batches: three for RNA + ADT and for "
            "RNA + ATAC, and two for ADT + ATAC and for RNA + ADT + ATAC.") in recipe
    assert "`batch=` takes the column itself, not its name." in recipe


# ---- R4-18: the ATAC override; one Multiome folder serves every method ------
@needs_docs
def test_run_guide_skips_an_atac_mismatch_and_needs_one_multiome_folder():
    """The layer-1 sentence stays true after R4-01 (a named method is skipped
    too). The override allow_atac_mismatch= is deprecated in 0.4.0 and left
    off the page. The recipe that drew the results of a peak folder and a
    gene-activity folder as one dataset is gone: Matilda, scMDC and UnitedNet
    read peaks, so one Multiome folder serves every vertical RNA + ATAC
    method."""
    run = _read("tutorials/run.md")
    assert "also when `methods=` names it" in _flat("tutorials/run.md")
    assert "also when `methods=` names it" in " ".join(_visible(run).split())
    assert "allow_atac_mismatch" not in run and "--allow-atac-mismatch" not in run
    assert mtb.find_methods("vertical", modalities=["atac_gas"]) == []
    assert "Every vertical method that reads ATAC reads peaks." in _flat("tutorials/run.md")
    assert "gene-activity folder" not in run and "second folder" not in _flat("tutorials/run.md")
    assert '.assign(dataset="MYMULTIOME")' not in run


# ---- R4-19: Changes page (the API-page exit-code sentence: tests/test_docs_r5.py) --
