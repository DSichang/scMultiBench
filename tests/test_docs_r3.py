"""Round 3 of the student study: what the docs pages now say (get-started,
guides, API and Changes pages), checked against the live package where the
page quotes a number.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it.
"""
import os
import warnings
from pathlib import Path

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


# ---- get started: quickstart, installation, FAQ (R3-20 to R3-26) ---------
@needs_docs
def test_linux_tab_reads_the_file_itself():
    tab = _read("quickstart.md").split('=== "Linux, environments installed"', 1)[1]
    code = tab.split("```python", 1)[1].split("```", 1)[0]
    assert code.index("import multibench as mtb") < code.index('ad.read_h5ad("my_citeseq.h5ad")') \
        < code.index("mtb.io.export_dataset(adata")


@needs_docs
def test_dry_run_line_is_on_the_quickstart_only():
    """R3-25: the CLI dry-run example stays on the Quickstart; Installation
    no longer repeats it (tutorials/run carries the dry-run step)."""
    quick, install = _read("quickstart.md"), _read("installation.md")
    for text in (quick, install):
        assert "mtb.describe_layout(" in text
    assert "multibench run --method SCALEX --category diagonal" in quick and "--dry-run" in quick
    assert "**Preview a run.**" not in install and "**Custom launch.**" not in install


@needs_docs
def test_intro_sends_each_data_shape_to_its_recipe():
    """R3-22: one line per data shape, the non-CITE-seq ones straight to tutorials/run."""
    intro = _read("quickstart.md").split("??? note", 1)[0]
    assert "- CITE-seq: see [Your own data](#your-own-data) below." in intro
    assert "(tutorials/run.md#your-own-data-as-a-dataset-folder)" in intro


@needs_docs
def test_step1_names_both_routes_for_citeseq_donors():
    """R3-23: the counts on the page are the registry's."""
    step1 = _flat("quickstart.md").split("## Step 1:", 1)[1].split("```", 1)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_v = len(mtb.find_methods("vertical", modalities=["rna", "adt"]))
        n_c = len(mtb.find_methods("cross", modalities=["rna", "adt"]))
        assert "D11" in set(mtb.results_coverage("vertical").dataset)
        assert "D52" in set(mtb.results_coverage("cross").dataset)
    assert "CITE-seq donors fit both `vertical` and `cross`." in step1
    assert f"({n_v} methods, stored table D11)" in step1
    assert f"({n_c} methods that integrate the donors as batches, stored table D52)" in step1


@needs_docs
def test_quickstart_batch_lines_name_run_all_as_the_guides_do():
    """Integration of round 3: run_all without batch= scores a two-sample
    folder as one batch, so the Quickstart names batch= on run_all as well as
    on evaluate, as tutorials/run and tutorials/discover do."""
    step1 = _flat("quickstart.md").split("## Step 1:", 1)[1].split("```", 1)[0]
    assert "`evaluate(batch=...)`" not in step1
    assert "pass each cell's sample as `batch=` to `run_all` or `evaluate`." in step1
    assert "scores donor mixing with `batch=` on `run_all` or `evaluate`." in step1
    assert "pass each cell's sample as `batch=` to `run_all` or `evaluate`." \
        in _flat("tutorials/discover.md")


@needs_docs
def test_cluster_recipe_pins_scripts_and_reads_strict():
    """R3-26: both blocks can pin the scripts; the scan line says what exit 1 means."""
    install = _read("installation.md")
    cluster = install.split("## On a cluster with offline compute nodes", 1)[1].split("\n## ", 1)[0]
    line = "export MULTIBENCH_SCRIPTS_REF=<commit>   # optional: same scripts for every job"
    assert cluster.count(line) == 2 and len(line) <= 80
    login = cluster.split('```bash title="login node"', 1)[1].split("```", 1)[0]
    assert login.index(line) < login.index("multibench fetch --scripts")
    assert login.index("# --strict: exit 1 when any method in --methods has no runnable row") \
        < login.index("multibench scan LAB")
    assert "Check it with `du` after the first install." in cluster


@needs_docs
def test_method_environment_details_have_no_bold_labels():
    """R3-25: a plain list led by code terms; the scripts go in their own block."""
    envs = _read("installation.md").split("## Method environments", 1)[1] \
        .split("### Google Colab", 1)[0]
    assert "**" not in envs
    assert '??? note "Details: method scripts"' in envs
    faq = _flat("faq.md")
    assert "The method environments run only on Linux, so the install refuses here." in faq
    assert "score embeddings and plot" not in faq


# ---- guides: run and discover (R3-03, R3-19) -------------------------------
@needs_docs
def test_discover_keeps_the_full_ranking_pointer_visible():
    """R3-19: the pointer to the full benchmark ranking is in the visible
    text again, not only in a Details block."""
    visible = " ".join(_visible(_read("tutorials/discover.md")).split())
    assert "The full benchmark ranking is in the" in visible


@needs_docs
def test_run_guide_shows_the_peak_token_and_both_batch_spellings():
    """R3-03 / R3-24: the Multiome tab passes atac_peak and each cell's sample
    to run_all; the one-file cross export scores donors via obs:donor."""
    run = _read("tutorials/run.md")
    assert 'modalities=["rna", "atac_peak"]' in run
    assert 'batch=mdata.obs["sample"]' in run
    assert 'batch="obs:donor"' in run


# ---- API overview and Changes (R3-28) --------------------------------------
@needs_docs
def test_strict_pages_name_both_exit_rules():
    for page in ("api.md", "changes.md"):
        text = _flat(page)
        assert "when nothing you asked for can run" not in text, page
        assert ("exits with `1` when no requested row is runnable. With `--methods`, it "
                "exits with `1` when any named method has no runnable row.") in text, page


@needs_docs
def test_api_page_run_all_batch_and_off_linux_refusal():
    api = _flat("api.md")
    assert "`--batch CSV` on `evaluate` and `run-all` gives one batch id per cell." in api
    assert "The message names the install command. On macOS and Windows it first says" not in api
    assert ("On Linux, a missing environment gets the install command. On macOS and "
            "Windows, the message says that methods run only on Linux.") in api


@needs_docs
def test_changes_page_lists_round_3_behaviour():
    text = _flat("changes.md")
    for phrase in ("`multibench run-all --batch CSV`",
                   "`\"atac_gas\"` no longer selects moETM, scMM and iPOLNG",
                   "is not runnable in `scan`, and `run_all` skips it",
                   "end with a `caveat` column",
                   "GLUE reads a copy of `atac_peak.h5`",
                   "a `scripts_ref` row says whether it matches",
                   "`labels has 60 entries for 90 cells in the embedding.`"):
        assert phrase in text, phrase
