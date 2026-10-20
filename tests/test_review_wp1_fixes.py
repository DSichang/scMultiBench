"""Fixes from the independent review of the 0.4.0 behaviour changes (WP1)."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import envs

from test_fetch_outputs import F, served  # noqa: F401 - the local outputs archive


def test_fetch_outputs_takes_method_names_in_any_case(served):  # noqa: F811
    data, _ = served
    out = F.fetch_outputs("D11", methods=["matilda", "MOFA+"], data_path=data,
                                 quiet=True)
    assert (out / "batch_result.json").is_file()
    with pytest.raises(KeyError, match="no shipped output for"):
        F.fetch_outputs("D11", methods=["totalVI"], data_path=data, quiet=True)


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
        mtb.run_all("DG", "diagonal", methods=["iNMF"], out_dir=tmp_path / "o",
                    data_path=tmp_path, verbose=False)
    assert "rna1+rna2+rna3" not in str(e.value)
    # the row for numbered batches of this one-batch folder is not listed (X80)
    assert "other row" not in str(e.value)


def test_params_for_takes_every_token_set_find_methods_takes():
    for cat in ("vertical", "diagonal", "mosaic", "cross"):
        for mods in (["rna", "adt"], ["rna", "atac"], ["adt", "atac"],
                     ["rna", "adt", "atac"]):
            for m in mtb.find_methods(cat, modalities=mods):
                try:
                    mtb.params_for(m, cat, mods)
                except mtb.AmbiguousVariantError:
                    pass                      # several variants: the message names one
    assert mtb.params_for("UINMF", "mosaic", ["rna", "adt"])["variant"].startswith("mosaic:")
    assert mtb.params_for("Matilda", "vertical", ["adt", "atac"])["variant"] == \
        "vertical:rna+adt+atac"
