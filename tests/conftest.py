from pathlib import Path
import pytest

# Repo root = two levels up from this file (tests/ -> <ROOT>)
ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture
def root():
    return ROOT

@pytest.fixture
def files_dir(root):
    return root / "multibench" / "files"

@pytest.fixture
def result_dir(root):
    return root / "multibench" / "result"


# ---- host-agnostic run mode -------------------------------------------------
# Most tests assert the classic ``conda run -n <env>`` command line. On a host
# where the envs exist as prefixes (the benchmark host, or a laptop that
# unpacked one) the runner would pick prefix mode and those assertions would
# fail for a reason unrelated to what they test. Pin conda mode everywhere
# except in the prefix-mode tests themselves, which set the mode explicitly.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _pin_conda_run_mode(request, monkeypatch):
    if request.node.fspath.basename in ("test_prefix_mode.py", "test_env_flavor.py"):
        return
    monkeypatch.setenv("MULTIBENCH_RUN_MODE", "conda")
