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
