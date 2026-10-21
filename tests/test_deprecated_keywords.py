"""Deprecated keywords, flags and names: one warning, still working, out of the signature and help."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry  # noqa: F401


def _mine(base, name="Mine", scored_with="leidenalg/search/0.4.0"):
    rows = base[base.method == "Matilda"].assign(method=name, source="user")
    if scored_with is not None:
        rows = rows.assign(scored_with=scored_with)
    return rows


MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.engine.runner.preview(method, category, inputs=inputs, out_dir=str(tmp_path / "o"), params=params)[0]


# --- deprecations and catalog: X39, X40, X41, X43 ----------------------------

def test_x40_method_info_default_keys_and_the_verbose_hint():
    info = mtb.method_info("Matilda")
    assert set(info) == {"id", "language", "notes", "categories", "env", "atac",
                         "needs_labels", "supports", "runtime", "gpu",
                         "cpu_params", "setup_hint", "repo_url", "reference"}
    # the hint names params_for
    with pytest.raises(KeyError, match=r"'params' is in mtb\.params_for\('Matilda', "
                                       r"category, modalities\)"):
        info["params"]


