"""How the stored tables were scored is stated where a user compares with them
(study round 1, L37 and L40).

The sentences name multibench 0.2.1 as the scorer of the re-run rows; the
last test fails when shipped re-run tables carry another version stamp, so
the docstrings cannot drift from the files silently.
"""
import inspect
import warnings

import multibench as mtb
from multibench import config
from multibench.data import results
from multibench.eval import pipeline


def _doc(obj):
    return " ".join(inspect.getdoc(obj).split())


def test_load_results_says_how_rerun_rows_were_scored():
    doc = _doc(results.load_results)
    assert ("Re-run rows were scored by multibench 0.2.1's ``evaluate`` with the "
            "leidenalg backend and every cell type counted as isolated") in doc
    assert 'mtb.config.DEFAULT.leiden_flavor = "leidenalg"' in doc


def test_leiden_flavor_names_the_backend_of_the_stored_sources():
    doc = _doc(config.Config)
    assert '``"leidenalg"`` (the backend of both stored sources)' in doc


def test_evaluate_lists_the_metric_definitions():
    doc = _doc(pipeline.evaluate)
    assert "**Metric definitions.**" in doc
    assert "iso_threshold = number of batches + 1" in doc
    assert "scanpy's default neighbour graph (15 neighbours)" in doc
    for fn in ("isolated_labels_asw", "clisi_graph", "silhouette_batch",
               "graph_connectivity", "ilisi_graph"):
        assert fn in doc, fn


def test_the_shipped_rerun_tables_are_the_version_the_docs_name():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rr = mtb.load_results(source="rerun")
    assert rr.attrs["rerun_version"] == "0.2.1"
