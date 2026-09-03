"""Round-3 follow-ups that cross the worktree boundaries (orchestrator)."""
import subprocess
import sys
import warnings

import pytest

import multibench as mtb


def test_fetchable_is_reexported_under_data():
    ids = mtb.data.fetchable()
    assert ids == mtb.data.results.fetchable()
    assert "D11" in ids and len(ids) >= 5


def test_io_dir_hides_leaked_imports():
    names = dir(mtb.io)
    assert "to_canonical" in names and "export_dataset" in names
    for leaked in ("annotations", "np", "os", "re", "Path"):
        assert leaked not in names
    assert mtb.io.annotations is not None      # attribute itself untouched


def test_load_results_canonicalises_dataset_case():
    with pytest.warns(UserWarning, match="'d52' is not a stored table id but 'D52' is"):
        df = mtb.load_results("cross", dataset="d52", source="rerun")
    assert set(df["dataset"].astype(str)) == {"D52"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = mtb.load_results("cross", dataset="D52", source="rerun")
    assert len(ok) == len(df)


def test_load_results_unknown_dataset_still_fails_loudly():
    with pytest.raises(FileNotFoundError):
        mtb.load_results("cross", dataset="D99", source="rerun")


def test_cli_plot_category_relies_on_api_warning(tmp_path):
    out = tmp_path / "b.png"
    r = subprocess.run([sys.executable, "-m", "multibench", "plot", "bubble",
                        "--category", "cross", "--dataset", "D52", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("only one method") == 1, r.stderr   # the API warning, once
