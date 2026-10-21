"""export_dataset and run(batch=) say how a Series lines up with the cells: by
position with a warning, or the barcode-suffix fix that evaluate names (core040 S-3, S-5)."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow


def _cells():
    return ad.AnnData(np.ones((3, 2)), obs=pd.DataFrame(index=["x-1", "y-1", "z-1"]))


def test_a_labels_series_without_barcodes_is_matched_by_position_with_a_warning(tmp_path):
    with pytest.warns(UserWarning, match="labels is a Series without barcodes and is "
                                         "matched to the cells by position"):
        mtb.io.export_dataset(_cells(), tmp_path / "A", labels=pd.Series(["t", "u", "v"]))


def test_export_names_the_barcode_suffix_fix(tmp_path):
    with pytest.raises(ValueError, match=r"They match after adding '-1' to the labels ids: "
                                         r"pass labels\.set_axis\(labels\.index \+ '-1'\)"):
        mtb.io.export_dataset(_cells(), tmp_path / "B",
                              labels=pd.Series(["t", "u", "v"], index=["x", "y", "z"]))


def test_run_all_batch_names_the_barcode_suffix_fix():
    ids = workflow._dataset_cell_ids("D11", None)
    batch = pd.Series(["a"] * len(ids), index=[i + "-9" for i in ids])
    with pytest.raises(ValueError, match="They match after removing '-9' from the batch ids"):
        workflow._cell_vector(batch, "D11", None)
