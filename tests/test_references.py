"""method_info provenance (repo_url / reference / notes / driver / variants) and mtb.cite.

DOIs in engine/references.yaml were each resolved against Crossref when added;
the tests here pin shape and reachability, not truth - do not add a DOI from
memory (see the header of references.yaml).
"""
import re
from pathlib import Path

import pytest
import yaml

import multibench as mtb
from multibench.engine import registry

# Methods with no Crossref-verified DOI yet. Shrink this list as references are
# verified; a method appearing here WITH a reference, or missing from here
# WITHOUT one, fails loudly.
UNREFERENCED = {"VIMCCA", "VIPCCA", "scMSI"}

DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


def test_cite_bibtex_and_text():
    from multibench.discover import cite
    s = cite(["StabMap", "GLUE"], format="bibtex")
    assert s.count("@article{") == 3
    assert "10.1038/s41592-025-02856-3" in s
    assert "10.1038/s41587-023-01766-z" in s and "10.1038/s41587-022-01284-4" in s
    t = cite(format="text")
    assert t.count("\n") == 0 and "https://doi.org/10.1038/s41592-025-02856-3" in t
    with pytest.raises(ValueError, match="unknown format 'ris'"):
        cite(format="ris")
    with pytest.raises(KeyError, match="Unknown method NoSuchMethod"):
        cite(["NoSuchMethod"])
    # an unverified method is flagged, not silently dropped
    u = cite(["VIMCCA"], format="bibtex")
    assert "% VIMCCA: no verified reference" in u and "scbean" in u
    assert "VIMCCA: no verified reference" in cite("VIMCCA") and "%" not in cite("VIMCCA")
    allb = cite("all", format="bibtex")
    assert allb.count("@article{") == 1 + len(registry.list_methods()) - len(UNREFERENCED)
    assert cite("StabMap") == cite(["StabMap"])
