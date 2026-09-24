"""The tutorials after fix round 3 of the student study.

R3-02 changed what ``runnable`` means. In a category with ATAC methods,
``scan`` marks a variant whose ATAC file holds the other form not runnable,
even when ``files_ok`` and ``env_ok`` pass, and ``run_all`` skips it. Since
R4-01 this holds also when ``methods=`` names the method
(tests/test_tutorial_round4.py checks the named-method calls). Section 3 of each generated tutorial states the rule ``runnable``
follows: these tests pin that sentence on the committed notebooks and check it
against the live package on folders that hold the other ATAC form.
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
CATS = list(GEN.SCEN)
SCAN_PARAGRAPH = "`scan` checks each method variant."
ATAC_RULE = ("A variant is `runnable` only when both pass and its ATAC file holds the "
             "form the method reads. `reason` says what failed.")


def _markdown(name, start):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    hits = ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "markdown" and "".join(c["source"]).startswith(start)]
    assert len(hits) == 1, f"{name}: one section-3 scan paragraph"
    return hits[0]


def _visible(md):
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


@pytest.mark.parametrize("cat", CATS)
def test_section_3_states_the_atac_form_rule_where_the_category_reads_atac(cat):
    """Visible, in the paragraph above the scan of the user's folder: a row
    whose ATAC file holds the other form is not runnable although both checks
    pass. Cross reads no ATAC, so its sentence names the two checks only."""
    import multibench as mtb
    md = _visible(_markdown(f"tutorial_{cat}", SCAN_PARAGRAPH))
    if mtb.find_methods(cat, modalities=["atac"]):
        assert ATAC_RULE in md, md
        assert "only when both pass, and" not in md
    else:
        assert "ATAC" not in md, md
        assert "A variant is `runnable` only when both pass, and `reason` says what failed." in md
    assert GEN.runnable_sentence(cat) in md


def test_the_atac_categories_are_the_three_that_read_atac():
    import multibench as mtb
    assert {c for c in CATS if mtb.find_methods(c, modalities=["atac"])} == \
        {"vertical", "diagonal", "mosaic"}


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


def _scan(name, cat, root):
    import multibench as mtb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.scan(name, cat, data_path=root, verbose=False).set_index("method")


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
    df = _scan("MYMOSAIC", "mosaic", tmp_path)
    for m in ("StabMap", "scMoMaT"):
        r = df.loc[m]
        assert r.files_ok and r.env_ok and not r.runnable, m
        assert r.reason.startswith("needs peak ATAC; atac2.h5 holds gene activity"), r.reason


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
    df = _scan("MYDIAG", "diagonal", tmp_path)
    for m in GEN.SCEN["diagonal"]["own_trio"]:
        r = df.loc[m]
        assert r.files_ok and r.env_ok and not r.runnable, m
        assert r.reason.startswith("needs gene-activity ATAC; atac_gas.h5 holds peaks"), r.reason
