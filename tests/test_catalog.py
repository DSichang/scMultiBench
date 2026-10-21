import warnings

import pytest

from multibench._data import catalog
from multibench.engine import registry

# canonical_id / canonical_metric / known_metrics / PAPER_COLUMNS are deprecated
# (0.4) but still work until 0.5; these tests exercise them on purpose.
pytestmark = pytest.mark.filterwarnings(r"ignore:catalog\.\w+ is deprecated:FutureWarning")


# --- Rin: canonical_id folds case and can insist; canonical_metric can insist

# --- re-test round 3: output column agrees with the registry, dir() hygiene ---

def test_catalog_dir_lists_public_api_not_leaked_imports():
    """mtb.catalog.annotations() was a TypeError for a re-tester: dir() listed
    the __future__ feature object and the module's imports. mtb.catalog is a
    two-name facade (d1): its internals are not reachable through it."""
    import multibench as mtb
    listed = dir(mtb.catalog)
    assert [n for n in listed if not n.startswith("__")] == ["datasets", "metrics"]
    for leaked in ("annotations", "pd", "Path", "re", "config", "_registry_ids"):
        assert not hasattr(mtb.catalog, leaked), leaked
    assert mtb.catalog.datasets is catalog.datasets