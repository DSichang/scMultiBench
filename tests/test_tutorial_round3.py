"""The tutorials after fix round 3 of the student study.

R3-02 changed what ``runnable`` means. In a category with ATAC methods,
``scan`` marks a variant whose ATAC file holds the other form not runnable,
even when ``files_ok`` and ``env_ok`` pass, and ``run_all`` skips it. Since
R4-01 this holds also when ``methods=`` names the method
(tests/test_tutorial_round4.py checks the named-method calls). Section 5 of
each generated tutorial says which ATAC form the reader's data needs: these
tests pin that sentence on the committed notebooks and check the rule on the
live package, on folders that hold the other ATAC form.
"""
import importlib.util
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
KEYS = list(GEN.TASKS)
OWN_DATA = "## 5. Your own data"
# "peak": the three-modality tutorial says "an ATAC peak matrix", the others "peaks"
FORM_WORDS = {"gene_activity": "gene-activity", "peak": "peak"}


def _markdown(name, start):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    hits = ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "markdown" and "".join(c["source"]).startswith(start)]
    assert len(hits) == 1, f"{name}: one {start!r} cell"
    return hits[0]


def _visible(md):
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


def _reads_atac(key):
    """Whether the task's default methods read an ATAC file in this task."""
    import multibench as mtb
    t = GEN.TASKS[key]
    want = [set(v) for v in t["variants"]]
    return any(v["category"] == t["cat"] and set(v["modalities"]) in want
               and any(mod.startswith("atac") for mod in v["modalities"])
               for m in t["methods"] for v in mtb.method_info(m)["supports"])


@pytest.mark.parametrize("key", KEYS)
def test_section_5_states_the_atac_form_where_the_task_reads_atac(key):
    """Visible, above the own-data demo: the ATAC form the tutorial's methods
    read. A task without ATAC does not bring it up."""
    import multibench as mtb
    md = _visible(_markdown(f"tutorial_{key}", OWN_DATA))
    if not _reads_atac(key):
        assert "ATAC" not in md, md
        return
    forms = {mtb.method_info(m)["atac"] for m in GEN.TASKS[key]["methods"]}
    assert len(forms) == 1, (key, forms)
    assert "ATAC" in md and FORM_WORDS[forms.pop()] in md, md


def test_the_tasks_that_read_atac_are_the_ones_labelled_so():
    """The article names two mosaic tasks by their pattern ("Mixed, with /
    without shared modality"); both hold an RNA + ATAC batch. Every other
    task that reads ATAC says so in its label."""
    mixed = {k for k, t in GEN.TASKS.items() if t["label"].startswith("Mixed, ")}
    assert mixed == {"mosaic_shared", "mosaic_unshared"}
    assert {k for k in KEYS if _reads_atac(k)} == \
        {k for k, t in GEN.TASKS.items() if "ATAC" in t["label"]} | mixed


def test_every_category_has_methods_that_read_atac():
    """Since the article's tasks, cross has RNA + ATAC and ADT + ATAC tasks,
    so all four categories read ATAC (before: all but cross)."""
    import multibench as mtb
    cats = {t["cat"] for t in GEN.TASKS.values()}
    assert {c for c in cats if mtb.find_methods(c, modalities=["atac"])} == \
        {"vertical", "diagonal", "mosaic", "cross"}


@pytest.fixture
def every_env_and_a_gpu(monkeypatch):
    """Pin the env probe, so ``env_ok`` passes on any host."""
    from multibench import workflow as W
    from multibench.engine import envs, registry
    every = frozenset(envs.group_for(m) for m in registry.list_methods())
    monkeypatch.setattr(W, "_installed_envs", lambda: every)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _batch(rng, n, feats):
    import anndata as ad
    a = ad.AnnData(X=rng.poisson(1.0, size=(n, len(feats))).astype(float))
    a.var_names = feats
    a.obs["celltype"] = rng.choice(["T", "B", "NK"], n)
    return a


def _scan(name, cat, root, modalities):
    """The rows of one modality set (a method has one row per variant)."""
    import multibench as mtb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = mtb.scan(name, cat, data_path=root, verbose=False)
    df = df[df.modalities == "+".join(modalities)].set_index("method")
    assert df.index.is_unique
    return df


def test_the_sentence_holds_for_a_mosaic_folder_with_gene_activity(tmp_path, every_env_and_a_gpu):
    """The mosaic demo's layout (D46's pattern), with gene activity where the
    methods read peaks: StabMap and scMoMaT pass both checks and are not
    runnable, and ``reason`` says why."""
    import numpy as np
    import multibench as mtb
    rng = np.random.default_rng(0)
    genes = [f"gene{i}" for i in range(40)]
    b1, b2, b3 = _batch(rng, 100, genes), _batch(rng, 80, genes), _batch(rng, 60, genes)
    b1.obsm["protein"] = rng.poisson(3.0, size=(100, 12)).astype(float)
    b1.uns["protein_names"] = [f"CD{i}" for i in range(12)]
    b2.obsm["gas"] = rng.poisson(0.3, size=(80, 50)).astype(float)
    b2.uns["gas_names"] = [f"GENE{i}" for i in range(50)]
    folder = tmp_path / "MYMOSAIC"
    kw = dict(labels="obs:celltype", category="mosaic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mtb.io.export_dataset(b1, folder, adt="obsm:protein", batch_index=1, **kw)
        mtb.io.export_dataset(b2, folder, atac="obsm:gas", atac_kind="gene_activity",
                              batch_index=2, **kw)
        mtb.io.export_dataset(b3, folder, batch_index=3, **kw)
    df = _scan("MYMOSAIC", "mosaic", tmp_path, GEN.TASKS["mosaic_shared"]["variants"][0])
    for m in ("StabMap", "scMoMaT"):
        r = df.loc[m]
        assert r.files_ok and r.env_ok and not r.runnable, m
        assert r.reason.startswith(f"{m} needs peak ATAC, and atac2.h5 holds gene "
                                   "activity"), r.reason


def test_the_sentence_holds_for_a_diagonal_folder_with_peaks_as_gene_activity(
        tmp_path, every_env_and_a_gpu):
    """The diagonal demo's layout, with a peak matrix in ``atac_gas.h5``: the
    gene-activity methods of the tutorial's run pass both checks and are not
    runnable."""
    import numpy as np
    import multibench as mtb
    rng = np.random.default_rng(0)
    rna = _batch(rng, 120, [f"gene{i}" for i in range(40)])
    atac = _batch(rng, 90, [f"chr1:{100 * i}-{100 * i + 50}" for i in range(40)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mtb.io.export_dataset(rna, tmp_path / "MYDIAG", atac=atac, atac_kind="gene_activity",
                              labels="obs:celltype", category="diagonal")
    df = _scan("MYDIAG", "diagonal", tmp_path, ["rna", "atac_gas"])
    for m in GEN.TASKS["diagonal_rna_atac"]["methods"]:
        r = df.loc[m]
        assert r.files_ok and r.env_ok and not r.runnable, m
        assert r.reason.startswith(f"{m} needs gene-activity ATAC, and atac_gas.h5 holds "
                                   "peaks"), r.reason
