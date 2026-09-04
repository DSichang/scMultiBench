"""A transposed modality matrix must be rejected at preflight.

Storing matrix/data as cells x features is the easy mistake - it is the
scanpy/AnnData convention, and describe_layout only says "the matrix under
matrix/data" without stating an orientation. It does NOT silently produce a
wrong number (measured: the methods fail), but without this check it fails late,
after conda-env startup and potentially hours of compute, with a third-party
error that never mentions orientation.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from multibench.engine import resolve as _resolve


def _dataset(root, name, transpose, n_cells=120, n_genes=40, n_adt=8,
             with_dimnames=True, square=False):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if square:
        n_genes = n_cells
    for fname, n_feat, prefix in (("rna.h5", n_genes, "gene"), ("adt.h5", n_adt, "prot")):
        cells_by_feat = rng.poisson(3.0, size=(n_cells, n_feat)).astype(float)
        m = cells_by_feat.T if transpose else cells_by_feat
        with h5py.File(d / fname, "w") as f:
            g = f.create_group("matrix")
            g.create_dataset("data", data=m)
            if with_dimnames:
                g.create_dataset("barcodes", data=np.array(
                    [f"c{i}" for i in range(n_cells)], dtype="S8"))
                g.create_dataset("features", data=np.array(
                    [f"{prefix}{i}" for i in range(n_feat)], dtype="S8"))
    pd.DataFrame({"x": ["A"] * n_cells}).to_csv(d / "cty.csv", index=False)
    return d


def _resolve_it(root, name):
    return _resolve.inputs_for(name, "vertical", "Matilda",
                               modalities=["rna", "adt"],
                               data_path=str(root), check=True)


def test_correct_orientation_passes(tmp_path):
    _dataset(tmp_path, "GOOD", transpose=True)
    got = _resolve_it(tmp_path, "GOOD")
    assert got["rna"].endswith("rna.h5")


def test_transposed_matrix_is_rejected_with_an_actionable_message(tmp_path):
    _dataset(tmp_path, "BAD", transpose=False)
    with pytest.raises(ValueError) as e:
        _resolve_it(tmp_path, "BAD")
    msg = str(e.value)
    assert "cells x features" in msg, msg
    assert "features x cells" in msg, msg
    assert "to_canonical" in msg, "the message must say how to fix it"


def test_square_matrix_is_left_alone(tmp_path):
    """features == cells is genuinely ambiguous; guessing would be worse."""
    _dataset(tmp_path, "SQUARE", transpose=False, square=True, n_adt=8)
    # rna.h5 is square so it cannot be judged; adt stays non-square and correct
    _dataset(tmp_path, "SQUARE2", transpose=True, square=True, n_adt=8)
    _resolve_it(tmp_path, "SQUARE2")   # must not raise


def test_file_without_dimnames_is_not_rejected(tmp_path):
    """No features/barcodes means nothing to compare - do not invent a verdict."""
    _dataset(tmp_path, "NODIMS", transpose=False, with_dimnames=False)
    _resolve_it(tmp_path, "NODIMS")    # must not raise
