"""Round-3 follow-ups that cross the worktree boundaries (orchestrator)."""
import subprocess
import sys
import warnings

import pytest

import multibench as mtb


def test_io_dir_hides_leaked_imports():
    names = dir(mtb.io)
    assert "to_canonical" not in names      # deprecated: hidden from dir()
    assert "export_dataset" in names and "read" in names
    for leaked in ("annotations", "np", "os", "re", "Path"):
        assert leaked not in names
    assert not hasattr(mtb.io, "annotations")  # d1: a facade, no module internals


def test_load_results_dataset_ids_are_case_sensitive():
    with pytest.raises(FileNotFoundError, match=r"^The stored results of cross hold .*, not d52\."):
        mtb.load_results("cross", dataset="d52")


def test_load_results_unknown_dataset_still_fails_loudly():
    with pytest.raises(FileNotFoundError):
        mtb.load_results("cross", dataset="D99")


