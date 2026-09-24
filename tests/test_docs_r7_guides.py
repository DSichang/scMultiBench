"""Round 7 of the student study, guides (R7-14): tutorials/plot.

The chip sentence in 'Reading the figure' uses the wording of the figure key,
which covers a renamed re-run as well as your own method.

Needs the docs source (SCMULTIBENCH_DOCS=<docs dir>); skipped without it.
"""
from multibench.plot.bubble import CHIP_KEY
from tests.test_docs_r4 import _flat, needs_docs


@needs_docs
def test_plot_page_chip_sentence_matches_the_key():
    text = _flat("tutorials/plot.md")
    assert "? = a name the package does not know" in CHIP_KEY
    assert ("The chips give each method's language, `Py` or `R`. `?` marks a name "
            "the package does not know, such as your own method or a renamed re-run.") in text
    assert "`?` marks a method the package does not know" not in text
    assert ("`L` marks a method that uses cell-type labels. Its scores are not "
            "comparable") in text
