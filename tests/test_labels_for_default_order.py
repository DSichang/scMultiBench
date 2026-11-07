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

import multibench as mtb
from multibench import workflow


def _notes(*args, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mtb.engine.resolve.labels_for(*args, **kw)
    return [str(x.message) for x in w if "default order" in str(x.message)]


def test_d52_names_its_own_category_without_a_category():
    assert _notes("D52") == [
        "labels_for('D52') returns the files in the default order cty1, cty2, cty3. StabMap "
        "(cross, mosaic) reads them in another order: pass category= and method= for their "
        "output."]


def test_with_the_category_the_call_is_named():
    assert _notes("D52", "cross") == [
        "labels_for('D52', 'cross') returns the files in the default order cty1, cty2, "
        "cty3. StabMap (cross) reads them in another order: pass category= and method= for "
        "their output."]


def test_the_methods_whose_files_the_folder_holds_are_named():
    assert _notes("D28") == [
        "labels_for('D28') returns the files in the default order rna_cty, atac_cty. "
        "Seurat_v5 (diagonal) and uniPort (diagonal) read them in another order: pass category= and method= for "
        "their output."]


def test_silent_calls():
    assert _notes("D52", "cross", "StabMap") == []
    assert list(mtb.engine.resolve.labels_for("D52", "cross", "StabMap")) == ["cty3", "cty1", "cty2"]
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


