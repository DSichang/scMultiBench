"""Round 7 of the student study, guides (R7-14): tutorials/plot.

The chip sentence in 'Reading the figure' uses the wording of the figure key
('?: not a package method'). Since 0.4.0 a renamed run of a package method,
'Matilda (this run)', carries that method's chip.

Needs the docs source (SCMULTIBENCH_DOCS=<docs dir>); skipped without it.
"""
from tests.test_docs_r4 import _flat, needs_docs


@needs_docs
def test_plot_page_chip_sentence_matches_the_key():
    text = _flat("tutorials/plot.md")
    assert "The chips give each method's language, `Py` or `R`" in text
    assert "`Matilda (this run)` has Matilda's." in text
    assert "`?` marks a name the package does not know, such as your own method." in text
    assert "renamed re-run" not in text
    assert "`?` marks a method the package does not know" not in text
