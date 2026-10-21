"""Fixes from the independent review of the 0.4.0 behaviour changes (WP1)."""
from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import envs



def test_off_linux_the_requested_methods_error_counts_the_rows_without_files(
        tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem",
                        lambda: "method environments are linux-64 conda envs (packed "
                                "archives + lockfiles); this host is darwin/arm64")
    d = tmp_path / "DG"
    d.mkdir()
    for f in ("rna.h5", "atac_gas.h5", "rna_cty.csv", "atac_cty.csv"):
        (d / f).write_text("")
    with pytest.raises(ValueError) as e:
        mtb.run_all(os.path.join(tmp_path, 'DG'), 'diagonal', methods=['iNMF'], out_dir=tmp_path / 'o', verbose=False)
    assert "rna1+rna2+rna3" not in str(e.value)
    # the row for numbered batches of this one-batch folder is not listed (X80)
    assert "other row" not in str(e.value)


