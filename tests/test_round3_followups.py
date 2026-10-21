"""Round-3 follow-ups that cross the worktree boundaries (orchestrator)."""
import subprocess
import sys
import warnings

import pytest

import multibench as mtb


def test_io_dir_hides_leaked_imports():
    names = dir(mtb.io)
    assert "to_canonical" not in names      # deprecated: hidden from dir()
    assert "export_dataset" in names and "read_canonical" in names
    for leaked in ("annotations", "np", "os", "re", "Path"):
        assert leaked not in names
    assert mtb.io.annotations is not None      # attribute itself untouched


def test_load_results_canonicalises_dataset_case():
    with pytest.warns(UserWarning, match="dataset 'd52' is not in the stored results but 'D52' is"):
        df = mtb.load_results("cross", dataset="d52")
    assert set(df["dataset"].astype(str)) == {"D52"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = mtb.load_results("cross", dataset="D52")
    assert len(ok) == len(df)


def test_load_results_unknown_dataset_still_fails_loudly():
    with pytest.raises(FileNotFoundError):
        mtb.load_results("cross", dataset="D99")


