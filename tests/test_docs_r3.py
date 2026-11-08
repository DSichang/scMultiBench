"""Round 3 of the student study: what the docs pages now say (get-started,
guides, API and Changes pages), checked against the live package where the
page quotes a number.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is
skipped without it.
"""
import os
from pathlib import Path

import pytest


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
def test_dry_run_line_is_on_the_run_guide():
    """R3-25, then O36 (owner, 2026-11-09): the Quickstart runs SCALEX and links
    the guides, tutorials/run carries the dry-run step, and Installation does
    not repeat it."""
    quick, install = _read("quickstart.md"), _read("installation.md")
    assert "(tutorials/run.md#folder-layout)" in install
    assert 'mtb.run("D28", "diagonal", methods=["SCALEX"]' in quick
    assert "(tutorials/run.md)" in quick
    assert "dry_run=True" in _read("tutorials/run.md")
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
def test_the_donors_recipe_names_both_routes():
    """R3-23, then the 0.5.1 docs review: the donors recipe names the vertical
    route and the cross route with its batch count."""
    recipe = " ".join(_read("tutorials/run.md").split("{ #recipe-donors }", 1)[1].split())
    assert "three for RNA + ADT" in recipe and "treat the data as a vertical task" in recipe


@needs_docs
def test_the_discover_guide_names_batch_on_run_and_evaluate():
    """Integration of round 3: run without batch= scores a two-sample
    folder as one batch, so the discover guide names batch= on run as well as
    on evaluate."""
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
    to run; the one-file cross export scores donors via obs:donor."""
    run = _read("tutorials/run.md")
    assert 'modalities=["rna", "atac_peak"]' in run
    assert 'batch=mdata.obs["sample"]' in run
    assert 'batch="obs:donor"' in run


# ---- API overview (R3-28) --------------------------------------------------
@needs_docs
def test_api_page_run_all_batch_and_off_linux_refusal():
    api = _flat("api.md")
    assert "The message names the install command. On macOS and Windows it first says" not in api
    assert "Methods run only on Linux: elsewhere `mtb.run` raises `OSError`." not in api
    assert "## 2. Run (Linux)" in _read("api.md")


