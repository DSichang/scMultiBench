"""The tutorials after fix round 7 of the student study.

Round 7 reworded package messages that the notebooks quote or point to:

- R7-03: for MultiMAP and Seurat_v3, which read atac_peak.h5 and atac_gas.h5,
  the file check no longer points to ``method_info(m)["atac"]``, which lists
  them under peak. The diagonal tutorial's first cell names these two as
  needing both files and then pointed to the same ``method_info`` field for
  "which form a method reads". It now points to the run guide's folder
  layout, which lists each method under the ATAC files it needs
  (tests/test_docs_folder_layout.py pins that list to the registry).

The other round-7 changes (R7-09 export texts, R7-10 environment texts) keep
the notebooks' sentences true; the guards below check those sentences on the
live package, so a later change that makes one untrue fails here.
"""
import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import full_scan

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
KEYS = list(GEN.TASKS)
TUTORIALS = [f"tutorial_{k}" for k in KEYS]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _markdown(name, start):
    return next(src for kind, src in _cells(name) if kind == "markdown" and src.startswith(start))


def _visible(md):
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------- R7-03: which ATAC files a method needs
def test_diagonal_title_points_to_the_page_that_lists_the_files_each_method_needs():
    import multibench as mtb
    title = _visible(_cells("tutorial_diagonal_rna_atac")[0][1])
    _, _, both = GEN.atac_forms("diagonal")
    assert f"{GEN.and_list(both)} need both." in title
    # method_info names one form and lists the both-file methods under peak,
    # so it cannot be the pointer next to "need both"
    assert {mtb.method_info(m)["atac"] for m in both} == {"peak"}
    assert 'method_info(m)["atac"]' not in title
    assert f"[The folder layout]({GEN.LAYOUT}) lists each method's ATAC files." in title


# ---------------------- R7-09: export texts the tutorials describe (guards)
def _demo(n=60, counts=True):
    import anndata as ad
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(n, 20)).astype(float)
    a = ad.AnnData(X=x if counts else x + 0.5)
    a.var_names = [f"gene{i}" for i in range(20)]
    a.obs_names = [f"cell{i}" for i in range(n)]
    a.obsm["protein"] = rng.poisson(3.0, size=(n, 6)).astype(float)
    a.uns["protein_names"] = [f"CD{i}" for i in range(6)]
    a.obs["celltype"] = rng.choice(["T", "B"], n)
    return a


def test_the_overwrite_error_says_what_to_do(tmp_path):
    """Without ``overwrite=True`` a second export raises FileExistsError and
    says to pass it; with it, the export replaces the files. The tutorials
    leave this to the error."""
    import multibench as mtb
    kw = dict(rna="X", adt="obsm:protein", labels="obs:celltype")
    folder = mtb.io.export_dataset(_demo(), tmp_path / "MYCITE", **kw)
    with pytest.raises(FileExistsError) as e:
        mtb.io.export_dataset(_demo(), folder, **kw)
    msg = str(e.value)
    assert "already holds" in msg and msg.endswith("Pass overwrite=True to replace them.")
    mtb.io.export_dataset(_demo(), folder, overwrite=True, **kw)


# ---------------------------- R7-10: the environment reason (guard)
@pytest.mark.parametrize("name", TUTORIALS)
def test_env_reason_on_linux_holds_the_install_command_the_tutorials_name(name, monkeypatch):
    """Section 2 names the terminal command that installs the tutorial's
    environments; ``env_reason`` gives the same command for a missing one."""
    import multibench as mtb
    from multibench import workflow as W
    from multibench.engine import runner as R
    t = GEN.TASKS[name.removeprefix("tutorial_")]
    cat, methods = t["cat"], t["methods"]
    md = _markdown(name, "## 2. Download the data and the environments")
    # installing is the default since 0.4.0 (X27)
    assert "mtb.env.install" in md
    monkeypatch.setattr("multibench.engine.envs.host_system", lambda: None)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    ds = t["ds"]
    if not (mtb.config.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    for m in methods:
        row = _quiet(full_scan, ds, cat, methods=[m], verbose=False).iloc[0]
        assert not row["env_ok"]
        assert f"mtb.env.install([{m!r}])." in row["env_reason"]
