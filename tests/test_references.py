"""method_info provenance: the reference of each method.

DOIs in engine/references.yaml were each resolved against Crossref when added;
the tests here pin shape and reachability, not truth - do not add a DOI from
memory (see the header of references.yaml).
"""
import re

import multibench as mtb
from multibench.engine import registry

# Methods with no Crossref-verified DOI yet. Shrink this list as references are
# verified; a method appearing here WITH a reference, or missing from here
# WITHOUT one, fails loudly.
UNREFERENCED = {"VIMCCA", "VIPCCA", "scMSI"}

DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


def test_every_verified_method_has_a_reference():
    for m in registry.list_methods():
        ref = mtb.method_info(m)["reference"]
        if m in UNREFERENCED:
            assert ref is None, f"{m} now has a reference - drop it from UNREFERENCED"
        else:
            assert ref and DOI_RE.fullmatch(ref["doi"]), (m, ref)
            for k in ("title", "authors", "journal", "year"):
                assert ref.get(k), (m, k)
    assert mtb.method_info("StabMap")["reference"]["doi"] == "10.1038/s41587-023-01766-z"
    assert mtb.method_info("GLUE")["reference"]["doi"] == "10.1038/s41587-022-01284-4"
