"""``labels_for`` without a method warns when a method stacks the cells in
another order.

``evaluate(stabmap_D52_embedding, labels=labels_for("D52"))`` scored ASW 0.463
instead of 0.557 without a word: StabMap puts its reference batch first
(cty3, cty1, cty2). The call without a method now names the methods that can
run on the folder and put these cells in another order; a call with the method,
or a folder whose methods all use the default order, stays silent.
"""
from __future__ import annotations

import warnings

import pytest

import multibench as mtb


def _folder(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text("x\nA\n" if f.endswith(".csv") else "")
    return tmp_path


def _notes(*args, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mtb.labels_for(*args, **kw)
    return [str(x.message) for x in w if "has no method" in str(x.message)]


@pytest.fixture
def cross3(tmp_path):
    """A three-batch rna+adt folder, as D52."""
    return _folder(tmp_path, "C3", [f"{m}{i}.h5" for m in ("rna", "adt") for i in (1, 2, 3)]
                   + [f"cty{i}.csv" for i in (1, 2, 3)])


def test_no_method_on_a_folder_stabmap_reorders_warns(cross3):
    assert _notes("C3", data_path=cross3) == [
        "labels_for('C3') has no method: the files are in the default order cty1, cty2, "
        "cty3, and StabMap puts its cells in another order. For a method's output, "
        "pass labels_for('C3', 'cross', 'StabMap')."]


def test_the_category_alone_also_warns(cross3):
    assert len(_notes("C3", "cross", data_path=cross3)) == 1


def test_with_the_method_no_warning(cross3):
    assert _notes("C3", "cross", "StabMap", data_path=cross3) == []
    assert list(mtb.labels_for("C3", "cross", "StabMap", data_path=cross3)) == [
        "cty3", "cty1", "cty2"]


def test_one_label_file_no_warning(tmp_path):
    root = _folder(tmp_path, "V", ["rna.h5", "adt.h5", "cty.csv"])
    assert _notes("V", data_path=root) == []


def test_a_folder_whose_methods_keep_the_default_order_no_warning(tmp_path):
    # D46's files: StabMap's mosaic variant reads them in the default order
    root = _folder(tmp_path, "M", ["rna1.h5", "rna2.h5", "rna3.h5", "adt1.h5", "atac2.h5",
                                   "cty1.csv", "cty2.csv", "cty3.csv"])
    assert _notes("M", data_path=root) == []
    assert list(mtb.labels_for("M", "mosaic", "StabMap", data_path=root)) == [
        "cty1", "cty2", "cty3"]


def test_diagonal_names_the_methods_that_put_atac_first(tmp_path):
    root = _folder(tmp_path, "DG", ["rna.h5", "atac_peak.h5", "atac_gas.h5",
                                    "rna_cty.csv", "atac_cty.csv"])
    notes = _notes("DG", data_path=root)
    assert len(notes) == 1 and "Seurat_v5 and uniPort put their cells" in notes[0]
