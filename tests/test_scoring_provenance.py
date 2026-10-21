"""How the stored tables were scored is stated where a user compares with them
(study round 1, L37 and L40).

The re-run rows are stamped with the package version that scored them
(0.4.0, the article's protocol); the last test fails when shipped re-run tables carry another version stamp, so
the docstrings cannot drift from the files silently.
"""
import inspect
import warnings

import multibench as mtb
from multibench import config


def test_the_shipped_rerun_tables_are_the_version_the_docs_name():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rr = mtb.load_results()
    assert rr.attrs["rerun_version"] == "0.4.0"
