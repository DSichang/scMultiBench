"""The wheel must ship every non-.py file the package reads at runtime.

engine/params.yaml and engine/runtimes.yaml were silently missing from the
published wheel once: pip installs reported zero tunables for every method
and 'unknown' runtime tiers, with no error anywhere. This pins the
package-data globs to the runtime-read inventory so it cannot recur.
"""
from pathlib import Path
import fnmatch
import re

ROOT = Path(__file__).resolve().parents[1]

RUNTIME_READ = [
    "engine/methods.yaml",
    "engine/params.yaml",
    "engine/upstream_knobs.yaml",
    "engine/references.yaml",
    "engine/drivers/run_spiral.py",
    "engine/drivers/run_gpsa.py",
    "engine/drivers/spiral_mclust_shim.Rprofile",
    "engine/drivers/spiral_support/spiral/main.py",
    "result/rerun/long_all_D11.csv",
    "result/scib_metric/vertical integration/D11/scMoMaT/metric.csv",
    "engine/runtimes.yaml",
    "engine/env_specs.yaml",
    "engine/env_groups.yaml",
    "engine/packed_urls.json",
    "engine/drivers/run_matilda.py",
    "engine/drivers/run_stabmap.R",
    "files/dataset.csv",
    "files/method.csv",
]


def _package_data_globs():
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r"multibench\s*=\s*\[(.*?)\]", text, re.S)
    assert m, "package-data stanza not found"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_every_runtime_read_file_is_packaged():
    globs = _package_data_globs()
    for rel in RUNTIME_READ:
        assert (ROOT / "multibench" / rel).is_file(), f"missing on disk: {rel}"
        assert any(fnmatch.fnmatch(rel, g) for g in globs), (
            f"{rel} is read at runtime but matches no package-data glob {globs}")


def test_lockfiles_are_packaged():
    globs = _package_data_globs()
    locks = list((ROOT / "multibench/engine/env_locks").glob("*.yml"))
    assert len(locks) >= 29
    for p in locks:
        rel = f"engine/env_locks/{p.name}"
        assert any(fnmatch.fnmatch(rel, g) for g in globs), rel
