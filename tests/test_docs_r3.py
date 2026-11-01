"""Round 3 of the student study: what the docs pages now say (get-started,
guides, API and Changes pages), checked against the live package where the
page quotes a number.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it.
"""
import os
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
def test_dry_run_line_is_on_the_quickstart_only():
    """R3-25: the dry-run step stays on the Quickstart; Installation no longer
    repeats it (tutorials/run carries the dry-run step)."""
    quick, install = _read("quickstart.md"), _read("installation.md")
    for text in (quick, install):
        assert "(tutorials/run.md#folder-layout)" in text
    assert 'mtb.run("D28", "diagonal", methods=["SCALEX"]' in quick
    assert "dry_run=True" in quick
    assert "**Preview a run.**" not in install and "**Custom launch.**" not in install


@needs_docs
def test_intro_sends_each_data_shape_to_its_recipe():
    """R3-22: one line per data shape; CITE-seq goes to its Colab tutorial (0.5.1 review),
    the others straight to tutorials/run."""
    intro = _read("quickstart.md").split("??? note", 1)[0]
    assert ("- CITE-seq: the [RNA + ADT tutorial](tutorials/vertical_rna_adt.ipynb), which runs "
            "in Colab.") in intro
    assert "(tutorials/run.md#your-own-data-as-a-dataset-folder)" in intro


@needs_docs
def test_step1_sends_donors_to_the_recipe():
    """R3-23, then the 0.5.1 docs review: Step 1 names the vertical route in one line and
    links the donors recipe, which names the cross route and its batch count."""
    step1 = _flat("quickstart.md").split("## Step 1:", 1)[1].split("```", 1)[0]
    assert "(tutorials/run.md#recipe-donors)" in step1
    recipe = " ".join(_read("tutorials/run.md").split("{ #recipe-donors }", 1)[1].split())
    assert "three for RNA + ADT" in recipe and "treat the data as a vertical task" in recipe


@needs_docs
def test_quickstart_batch_lines_name_run_all_as_the_guides_do():
    """Integration of round 3: run without batch= scores a two-sample
    folder as one batch, so the Quickstart names batch= on run as well as
    on evaluate, as tutorials/run and tutorials/discover do."""
    step1 = _flat("quickstart.md").split("## Step 1:", 1)[1].split("```", 1)[0]
    assert "`evaluate(batch=...)`" not in step1
    # Step 1 points at the donors recipe; the discover guide names batch= on both calls
    assert "(tutorials/run.md#recipe-donors)" in step1
    assert "pass each cell's sample as `batch=` to `run` or `evaluate`." \
        in _flat("tutorials/discover.md")


@needs_docs
def test_installation_has_no_cluster_recipe():
    """The owner dropped the offline-cluster recipe (2026-11-01); no page links it."""
    assert "On a cluster" not in _read("installation.md")
    assert "on-a-cluster" not in _read("tutorials/run.md")


@needs_docs
def test_method_environment_details_have_no_bold_labels():
    """R3-25: a plain list led by code terms; the scripts go in their own block."""
    envs = _read("installation.md").split("## Method environments", 1)[1] \
        .split("### Google Colab", 1)[0]
    assert "**" not in envs
    assert '??? note "Details: method scripts"' in envs
    # 0.5.1: the FAQ repeated these entries and is gone (redirected to Troubleshooting)
    assert not (_docs_root() / "faq.md").exists()


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
    for page in ("changes.md",):
        text = _flat(page)
        assert "when nothing you asked for can run" not in text, page
        assert ("exits with `1` when no requested row is runnable. With `--methods`, it "
                "exits with `1` when any named method has no runnable row.") in text, page


@needs_docs
def test_api_page_run_all_batch_and_off_linux_refusal():
    api = _flat("api.md")
    assert "The message names the install command. On macOS and Windows it first says" not in api
    assert "Methods run only on Linux: elsewhere `mtb.run` raises `OSError`." not in api
    assert "## 2. Run (Linux)" in _read("api.md")


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
