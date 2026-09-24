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
from multibench import cli


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


def _step1_all():
    return _flat("quickstart.md").split("## Step 1:", 1)[1].split("## Step 2:", 1)[0]


def _counts():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_v = len(mtb.find_methods("vertical", modalities=["rna", "adt"]))
        n_c = len(mtb.find_methods("cross", modalities=["rna", "adt"]))
        assert "D11" in set(mtb.results_coverage("vertical").dataset)
        assert "D52" in set(mtb.results_coverage("cross").dataset)
    return n_v, n_c


# ---- R4-15: the laptop tab prints the dry-run command ----------------------
def _laptop_tab():
    tab = _read("quickstart.md").split('=== "CITE-seq on a laptop"', 1)[1].split('\n=== "', 1)[0]
    code = tab.split("```python", 1)[1].split("```", 1)[0]
    return "\n".join(line[4:] if line.startswith("    ") else line.lstrip()
                     for line in code.splitlines())


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
    assert 'cmd = mtb.run("totalVI", "vertical", inputs=inputs, out_dir="out/totalVI",' in code
    assert 'print(" ".join(cmd))' in code
    assert max(len(line) for line in code.splitlines()) <= 80
    monkeypatch.chdir(tmp_path)
    _cite_adata().write_h5ad(tmp_path / "my_citeseq.h5ad")
    scope = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "quickstart-laptop-tab", "exec"), scope)
    out = capsys.readouterr().out
    lines = out.splitlines()
    cmd_line = [i for i, l in enumerate(lines) if "main_totalVI.py" in l]
    table = [i for i, l in enumerate(lines) if l.split()[:1] == ["metric"]]
    assert cmd_line and table and cmd_line[0] < table[0], out
    assert lines[cmd_line[0]] == " ".join(scope["cmd"])
    plt.close("all")


# ---- R4-16: two plain bullets visible, the counts in Step 1 Details ---------
@needs_docs
def test_step1_donor_block_is_two_plain_bullets():
    step1 = _step1_visible()
    assert ("- 10x Multiome samples: use `vertical`. Keep all cells in one folder and "
            "pass each cell's sample as `batch=` to `run_all` or `evaluate`.") in step1
    assert ("- CITE-seq donors: use `vertical` in the same way. The `cross` category "
            "also fits three donors, one file each: see the "
            "[One file per batch](tutorials/run.md#your-own-data-as-a-dataset-folder) tab.") in step1
    block = step1.split("For several samples or donors:", 1)[1]
    assert "fit both" not in block and "(" not in block.replace("](", "").replace(".md#", "")
    assert "stored table" not in block and "UINMF" not in block
    run = _read("tutorials/run.md")
    assert "### Your own data as a dataset folder" in run
    section = run.split("### Your own data as a dataset folder", 1)[1].split("\n## ", 1)[0]
    assert '=== "One file per batch"' in section and 'category="cross"' in section


@needs_docs
def test_step1_details_carry_the_donor_counts():
    n_v, n_c = _counts()
    details = _step1_all().split("```", 1)[1]           # after the first code block
    assert (f"For CITE-seq donors, the `vertical` category has {n_v} RNA + ADT methods "
            "and the stored table D11.") in details
    assert (f"The `cross` category has {n_c} methods, which integrate the donors as "
            "batches, and the stored table D52.") in details
    assert "The cross methods read batches 1-3. UINMF reads only the first two." in details


# ---- R4-17: job.sh says what exit 3 means ---------------------------------
@needs_docs
def test_job_script_names_exit_code_3_above_run_all(capsys):
    text = _read("installation.md")
    job = text.split('```bash title="job.sh (Slurm)"', 1)[1].split("```", 1)[0].splitlines()
    i = next(k for k, l in enumerate(job) if l.startswith("multibench run-all"))
    line = "# exits 3 when the method fails, so Slurm marks the job as failed"
    assert job[i - 1] == line and len(line) <= 80
    # the package side of the same fact
    with pytest.raises(SystemExit):
        cli.main(["run-all", "--help"])
    assert "Exit code 3" in " ".join(capsys.readouterr().out.split())


# ---- R4-18: the ATAC override and one figure for two Multiome folders ------
@needs_docs
def test_run_guide_names_the_atac_override_and_one_dataset_name():
    """The layer-1 sentence stays true after R4-01 (a named method is skipped
    too); the Details give the override and the one-figure recipe."""
    run = _read("tutorials/run.md")
    assert "also when `methods=` names it" in _flat("tutorials/run.md")
    assert "also when `methods=` names it" in " ".join(_visible(run).split())
    assert "allow_atac_mismatch=True" in run and "--allow-atac-mismatch" in run
    assert "To run a method that is skipped for its ATAC file, pass" in run
    assert '.assign(dataset="MYMULTIOME")' in run
    # the page's override is a real keyword of both calls
    import inspect
    for fn in (mtb.scan, mtb.run_all):
        param = inspect.signature(fn).parameters["allow_atac_mismatch"]
        assert param.default is False, fn.__name__


# ---- R4-19: API overview and Changes page ----------------------------------
@needs_docs
def test_api_page_names_exit_code_3_and_the_atac_override():
    api = _flat("api.md")
    assert ("`2` on a usage error, and `3` when `run-all` finished but a method is "
            "listed in `failures.csv`.") in api
    assert "`--allow-atac-mismatch` on `scan` and `run-all` counts a method" in api
    assert "It also exits with `1` while the method scripts are not fetched." in api


@needs_docs
def test_changes_page_lists_round_4_and_drops_the_named_method_override():
    text = _flat("changes.md")
    assert "to run it anyway" not in text
    assert "Name the method" not in text
    for phrase in ("`multibench run-all --allow-atac-mismatch`",
                   "`mtb.run_all(..., allow_atac_mismatch=True)`",
                   "is not runnable in `scan`, and `run_all` skips it, also when "
                   "`methods=` names the method.",
                   "`run-all` exits with `3` after saving",
                   "status `SKIPPED`",
                   "indexed by barcode is matched by barcode",
                   "fills an existing empty `repo_path`",
                   "It also exits with `1` while the method scripts are not fetched.",
                   "an order that contradicts the method's cell order exits with `1`.",
                   "`summary` shows NaN there.",
                   "a single build, the same archive for CPU and GPU hosts.",
                   # integration of round 4 (tests/test_int_f4.py)
                   "`BatchResult.rescore` keeps `FAIL` and `TIMEOUT` records as they are."):
        assert phrase in text, phrase
