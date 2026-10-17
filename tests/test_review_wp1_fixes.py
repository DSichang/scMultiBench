"""Fixes from the independent review of the 0.4.0 behaviour changes (WP1)."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli
from multibench.engine import envs

from test_fetch_outputs import served  # noqa: F401 - the local outputs archive


def test_fetch_outputs_takes_method_names_in_any_case(served):  # noqa: F811
    data, _ = served
    out = mtb.data.fetch_outputs("D11", methods=["matilda", "MOFA+"], data_path=data,
                                 quiet=True)
    assert (out / "batch_result.json").is_file()
    with pytest.raises(KeyError, match="no shipped output for"):
        mtb.data.fetch_outputs("D11", methods=["totalVI"], data_path=data, quiet=True)


def test_cli_dry_run_refuses_a_bad_param_before_its_header(capsys):
    rc = cli.main(["run", "--method", "Matilda", "--category", "vertical", "--out-dir", "/o",
                   "--input", "rna=/d/rna.h5", "--input", "adt=/d/adt.h5",
                   "--input", "cty=/d/cty.csv", "--param", "epochz=1", "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 1 and "Dry run" not in err
    assert err.startswith("error: Matilda does not accept ['epochz']")


@pytest.fixture
def h5mu(tmp_path):
    md = pytest.importorskip("mudata")
    obs = pd.DataFrame({"celltype": ["A", "B"] * 10}, index=[f"c{i}" for i in range(20)])
    rna = ad.AnnData(np.ones((20, 5), dtype="float32"), obs=obs)
    rna.var_names = [f"g{i}" for i in range(5)]
    adt = ad.AnnData(np.ones((20, 3), dtype="float32"), obs=pd.DataFrame(index=obs.index))
    adt.var_names = [f"p{i}" for i in range(3)]
    path = tmp_path / "one.h5mu"
    md.MuData({"rna": rna, "adt": adt}).write(path)
    return path


def test_convert_labels_alone_on_a_mudata_names_its_selector(h5mu, tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["convert", str(h5mu), str(tmp_path / "CV"), "--category", "vertical",
                  "--labels", "rna:celltype"])
    assert ("error: --labels alone writes only the label file. Add --rna mod:rna to "
            "export the matrix too.") in capsys.readouterr().err


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
    assert "\n1 other row needs files that DG does not have.\n" in str(e.value)


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
