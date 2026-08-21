import pytest

import multibench as mtb
from multibench.engine import resolve


def test_available_datasets_lists_diagonal(result_dir):
    ds = mtb.available_datasets("diagonal", result_path=result_dir)
    assert isinstance(ds, list) and "D27" in ds
    # mosaic has no published results -> empty list (not an error)
    assert mtb.available_datasets("mosaic", result_path=result_dir) == []


def test_available_datasets_rejects_unwired_metric_set(result_dir):
    with pytest.raises(NotImplementedError):
        mtb.available_datasets("vertical", metric_set="classification", result_path=result_dir)


def test_inputs_for_check_raises_on_missing(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")  # peak.h5 deliberately missing
    with pytest.raises(FileNotFoundError):
        resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path, check=True)


def test_inputs_for_check_passes_when_present(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    for n in ["rna.h5", "peak.h5"]:
        (d / n).write_text("")
    got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path, check=True)
    assert "atac_peak" in got


def test_to_long_exposed_top_level():
    import pandas as pd
    wide = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    long = mtb.to_long(wide, method="M", dataset="D", category="vertical")
    assert list(long.columns) == ["metric", "value", "method", "dataset", "category"]


def test_namespace_all_hygiene():
    # env/config/io expose a curated __all__ (no leaked stdlib imports)
    assert "recipe" in mtb.env.__all__ and "subprocess" not in mtb.env.__all__
    assert "category_folder" in mtb.config.__all__ and "Path" not in mtb.config.__all__
    assert "to_canonical" in mtb.io.__all__


# --- P12: inputs_for's default warns about phantom paths ---------------------

def test_inputs_for_default_warns_on_missing(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")   # peak.h5 deliberately missing
    with pytest.warns(UserWarning, match="atac_peak") as rec:
        got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path)
    assert got["atac_peak"].endswith("/D27/atac_peak.h5")     # fallback path still returned
    msg = str(rec[0].message)
    assert "1 resolved input path(s) do not exist" in msg and "check=True to raise" in msg


def test_inputs_for_check_false_is_silent(tmp_path):
    import warnings
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path,
                                 check=False)
    assert "atac_peak" in got


def test_inputs_for_default_no_warning_when_present(tmp_path):
    import warnings
    d = tmp_path / "D27"; d.mkdir()
    for n in ["rna.h5", "peak.h5"]:
        (d / n).write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path)
    assert got["atac_peak"].endswith("/D27/peak.h5")
