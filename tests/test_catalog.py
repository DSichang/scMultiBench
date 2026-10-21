import warnings

import pytest

from multibench.data import catalog
from multibench.engine import registry

# canonical_id / canonical_metric / known_metrics / PAPER_COLUMNS are deprecated
# (0.4) but still work until 0.5; these tests exercise them on purpose.
pytestmark = pytest.mark.filterwarnings(r"ignore:catalog\.\w+ is deprecated:FutureWarning")


# --- Rin: canonical_id folds case and can insist; canonical_metric can insist

# --- re-test round 3: output column agrees with the registry, dir() hygiene ---

def test_catalog_dir_lists_public_api_not_leaked_imports():
    """mtb.catalog.annotations() was a TypeError for a re-tester: dir() listed
    the __future__ feature object and the module's imports. Tab completion now
    shows the documented API (__all__) and the dunders only; every attribute is
    still there."""
    import multibench as mtb
    listed = dir(mtb.catalog)
    assert set(catalog.__all__) <= set(listed)
    for leaked in ("annotations", "pd", "Path", "re", "config"):
        assert leaked not in listed, leaked
        assert hasattr(mtb.catalog, leaked)          # nothing removed
    assert "_registry_ids" not in listed and "__all__" in listed
    assert [n for n in listed if not n.startswith("__")] == sorted(catalog.__all__)
    assert listed == sorted(listed)
