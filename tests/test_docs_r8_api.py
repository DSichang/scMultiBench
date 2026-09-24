"""Changes page, round 8 (R8-10).

The 0.3.2 lists name this round's changes (R8-01 to R8-06), the reworded messages of
rounds 7 and 8 are one sentence instead of a nested list, the id checks of the
three alignment entries are stated once, and the texts the page quotes are what the
package prints.

Every test needs the docs source (SCMULTIBENCH_DOCS=<docs dir>) and is skipped without it.
"""
import re
import warnings

import pytest

import multibench as mtb
from multibench import workflow as W
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_study_r8_wf import labmos, lung, no_envs  # noqa: F401 - fixtures


@needs_docs
def test_changes_page_lists_round_8_changes():
    text = _flat("changes.md")
    for phrase in (
            # R8-01
            "- The result lines of `run_all` and `rescore` name the ARI, such as "
            "`ARI 0.629`.",
            # R8-04
            "- `scan` and `run_all`, dry run included, raise `ValueError` before anything "
            "runs when `methods=` names a method with no variant in the category or "
            "modalities. 0.3.1 dropped that method. The commands exit with `1`.",
            # R8-06
            "The column is numeric, also when every row is blank.",
            # R8-03
            "widened so that its row labels, family headers and key fit.",
            # R8-05
            "On macOS and Windows, the error lists only the rows that something else "
            "also blocks.",
            # R8-02
            '- `scan` with `modalities=["rna", "atac_peak"]` no longer warns that '
            "scBridge is left out. scBridge reads gene activity."):
        assert phrase in text, phrase


@needs_docs
def test_changes_page_folds_the_reworded_list_and_states_each_fact_once():
    text = _flat("changes.md")
    assert "- Reworded: - " not in text
    (sentence,) = re.findall(r"- Reworded: the messages of [^.]*\.", text)
    assert len(sentence.split()) <= 35, sentence
    # the id checks of run_all(batch=), rescore(labels=) and the CSV entry
    assert text.count("repeated or not cells of the dataset raise `ValueError`") == 1
    assert "as for `batch=`" not in text and "when an id repeats" not in text
    # the FAIL/TIMEOUT tail sits beside the result lines
    assert text.count("the last line of the error that names a cause") == 1


@needs_docs
def test_changes_page_quotes_the_live_ari_tail():
    shown = W._ari_tail({"metrics": {"ARI": 0.62894}})
    assert shown == "ARI 0.629"
    assert f"`{shown}`" in _flat("changes.md")


@needs_docs
def test_changes_page_named_method_entry_follows_the_package(no_envs, labmos):
    with pytest.raises(ValueError, match=r"^totalVI does not run on mosaic data\."):
        mtb.run_all("LABMOS", "mosaic", data_path="data",
                    methods=["StabMap", "totalVI"], dry_run=True)


@needs_docs
def test_changes_page_scbridge_entry_follows_the_package(lung):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        mtb.scan("LUNG", "diagonal", data_path="data", verbose=False,
                 modalities=["rna", "atac_peak"])
    assert not [w for w in seen if "scBridge" in str(w.message)]
    assert mtb.method_info("scBridge")["atac"] == "gene_activity"
