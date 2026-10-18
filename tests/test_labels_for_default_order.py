"""``labels_for`` without a method warns when a method whose files the folder
holds reads the label files in another order.

``evaluate(stabmap_D52_embedding, labels=labels_for("D52"))`` scored ASW 0.463
instead of 0.557 without a word: StabMap puts its reference batch first
(cty3, cty1, cty2). The warning names those methods, per category when the
call names none; a call with the method, a one-file folder, a folder whose
methods keep the default order, and package code that means the default
order stay silent.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import multibench as mtb
from multibench import workflow


def _notes(*args, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mtb.labels_for(*args, **kw)
    return [str(x.message) for x in w if "default order" in str(x.message)]


def test_d52_names_its_own_category_without_a_category():
    # the dataset's category only since 0.4.0 (X67): D52's batches hold the same
    # modalities, so it reads as cross
    assert _notes("D52") == [
        "labels_for('D52') returns the files in the default order cty1, cty2, cty3. In "
        "cross, StabMap reads them in another order: pass labels_for('D52', 'cross', "
        "'StabMap') for its output."]


def test_with_the_category_the_call_is_named():
    assert _notes("D52", "cross") == [
        "labels_for('D52', 'cross') returns the files in the default order cty1, cty2, "
        "cty3. StabMap reads them in another order: pass labels_for('D52', 'cross', "
        "'StabMap') for its output."]


def test_only_methods_whose_files_pass_the_check_are_named():
    # Seurat_v5 also puts ATAC first, but D28's RNA and ATAC are different cells
    assert _notes("D28") == [
        "labels_for('D28') returns the files in the default order rna_cty, atac_cty. In "
        "diagonal, uniPort reads them in another order: pass labels_for('D28', "
        "'diagonal', 'uniPort') for its output."]


def test_silent_calls():
    assert _notes("D52", "cross", "StabMap") == []
    assert list(mtb.labels_for("D52", "cross", "StabMap")) == ["cty3", "cty1", "cty2"]
    assert _notes("D11") == []                 # one label file
    assert _notes("D46") == []                 # StabMap keeps the default order there


def test_package_code_that_means_the_default_order_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sizes = workflow._label_file_sizes("D52")
        assert [n for n, _ in sizes] == ["cty1.csv", "cty2.csv", "cty3.csv"]
        batch = np.concatenate([np.full(k, i) for i, (_, k) in enumerate(sizes)])
        assert set(workflow._batch_segments("D52", None, batch)) == \
            {"cty1.csv", "cty2.csv", "cty3.csv"}
        workflow._dataset_cell_ids("D52", None)
