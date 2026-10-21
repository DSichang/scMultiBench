"""One modality rule for find_methods and recommend (study round 1, L13).

A base token (``rna``, ``adt``/``protein``, ``atac``) matches every role of that
type. A representation token (``atac_peak``/``peak``, ``atac_gas``/``gas``)
means base ``atac`` plus the ATAC representation the method reads, the same as
``atac='peak'`` / ``atac='gene_activity'``. Before the fix
``find_methods('diagonal', modalities=['rna', 'atac_peak'])`` returned SCALEX,
which reads gene activity, ``recommend(atac=...)`` was a TypeError and
``recommend(modalities=['rna', 'peak'])`` ranked all 14 diagonal methods.
"""
import warnings

import pytest

import multibench as mtb
from multibench import discover


def test_unknown_token_raises_with_the_vocabulary():
    with pytest.raises(ValueError, match=r"Unknown modality peaks2\.") as e:
        discover.find_methods(modalities=["rna", "peaks2"])
    assert "atac_peak" in str(e.value) and "protein" in str(e.value)


