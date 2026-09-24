"""Round 6 of the student study, guides (R6-15): tutorials/evaluate.

The 'Inputs' Details names what is matched by position for a bare-array
output (the Series and the CSV, not "both"), and the metric section says what
mtb.catalog.metrics() holds (one line per metric) and links evaluate's Notes
for the exact computation.

Needs the docs source (SCMULTIBENCH_DOCS=<docs dir>); skipped without it.
"""
import multibench as mtb
from tests.test_docs_r4 import _flat, _read, _visible, needs_docs


@needs_docs
def test_evaluate_page_names_what_a_bare_array_output_matches_by_position():
    text = _flat("tutorials/evaluate.md")
    assert "both are matched by position" not in text
    assert ("When the output is a bare array, the Series and the CSV are matched "
            "by position, with a warning.") in text


@needs_docs
def test_evaluate_page_points_to_evaluate_notes_for_the_exact_computation():
    visible = " ".join(_visible(_read("tutorials/evaluate.md")).split())
    section = visible.split("### What each metric measures", 1)[1].split("## ", 1)[0]
    assert "gives each definition in full" not in section
    assert "describes each metric in one line." in section
    assert ("[evaluate's Notes](../reference/score.md#multibench.eval.pipeline.evaluate) "
            "give the exact computation.") in section
    # the catalog rows are indeed one line each, and evaluate's Notes hold the median
    rows = mtb.catalog.metrics().set_index("metric")["description"]
    assert all("\n" not in d for d in rows)
    assert "take the median m of the per-cell scores" in " ".join(mtb.evaluate.__doc__.split())
