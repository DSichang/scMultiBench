"""The tutorials after fix round 7 of the student study.

Round 7 reworded package messages that the notebooks quote or point to:

- R7-11: the matrix-orientation ValueError no longer says "which is cells x
  features". It reads "<method> reads rna.h5 of <dataset>, which stores
  matrix/data as cells x features, shape (...)". The Troubleshooting table of
  each category tutorial quoted the old words as the symptom to look for; it
  now quotes "matrix/data as cells x features".
- R7-03: for MultiMAP and Seurat_v3, which read atac_peak.h5 and atac_gas.h5,
  the file check no longer points to ``method_info(m)["atac"]``, which lists
  them under peak. The diagonal tutorial's first cell names these two as
  needing both files and then pointed to the same ``method_info`` field for
  "which form a method reads". It now points to ``describe_layout``, which
  lists each method under the ATAC files it needs.

The other round-7 changes (R7-09 export texts, R7-10 environment texts) keep
the notebooks' sentences true; the guards below check those sentences on the
live package, so a later change that makes one untrue fails here.
"""
import importlib.util
import json
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
CATS = list(GEN.SCEN)
TUTORIALS = [f"tutorial_{c}" for c in CATS]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _markdown(name, start):
    return next(src for kind, src in _cells(name) if kind == "markdown" and src.startswith(start))


def _visible(md):
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


def _trouble_row(name, fix_word):
    """The Troubleshooting row whose fix column holds ``fix_word``, as
    ``(symptom, fix)``."""
    md = _markdown(name, "## Troubleshooting")
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and fix_word in ln]
    assert len(rows) == 1, rows
    symptom, fix = (c.strip() for c in rows[0].strip("|").split("|"))
    return symptom, fix


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ------------------------------------------ R7-11: the orientation symptom
def _transposed_folder(root, name="MYCITE", n_cells=120):
    """A vertical RNA + ADT folder whose .h5 files store cells x features:
    the AnnData orientation, the easy mistake."""
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for fn, n_feat, prefix in (("rna.h5", 40, "gene"), ("adt.h5", 8, "prot")):
        with h5py.File(d / fn, "w") as f:
            f.create_dataset("matrix/data",
                             data=rng.poisson(3.0, size=(n_cells, n_feat)).astype(float))
            f.create_dataset("matrix/barcodes",
                             data=np.array([f"c{i}" for i in range(n_cells)], dtype="S8"))
            f.create_dataset("matrix/features",
                             data=np.array([f"{prefix}{i}" for i in range(n_feat)], dtype="S8"))
    pd.DataFrame({"x": ["A"] * n_cells}).to_csv(d / "cty.csv", index=False)
    return d


@pytest.mark.parametrize("name", TUTORIALS)
def test_troubleshooting_quotes_the_live_orientation_error(name, tmp_path):
    """The symptom a reader searches for is in the error that inputs_for,
    scan's reason column and run raise for a transposed matrix."""
    import multibench as mtb
    symptom, fix = _trouble_row(name, "transposed")
    quoted = re.fullmatch(r"`(?:\.\.\. )?(.+?)`", symptom)
    assert quoted, symptom
    words = quoted.group(1)
    _transposed_folder(tmp_path)
    with pytest.raises(ValueError) as e:
        mtb.inputs_for("MYCITE", "vertical", "Matilda", modalities=["rna", "adt"],
                       data_path=tmp_path, check=True)
    assert words in str(e.value), (words, str(e.value))
    row = _quiet(mtb.scan, "MYCITE", "vertical", methods=["Matilda"], data_path=tmp_path,
                 modalities=["rna", "adt"], verbose=False).iloc[0]
    assert not row["files_ok"]
    assert words in row["reason"], row["reason"]
    # the fix column names the calls the error itself offers
    assert "`mtb.io.to_canonical`" in fix and "to_canonical" in str(e.value)


# ---------------------------------- R7-03: which ATAC files a method needs
def test_diagonal_title_points_to_the_call_that_lists_the_files_each_method_needs():
    import multibench as mtb
    title = _visible(_cells("tutorial_diagonal")[0][1])
    _, _, both = GEN.atac_forms("diagonal")
    assert f"{GEN.and_list(both)} need both." in title
    # method_info names one form and lists the both-file methods under peak,
    # so it cannot be the pointer next to "need both"
    assert {mtb.method_info(m)["atac"] for m in both} == {"peak"}
    assert 'method_info(m)["atac"]' not in title
    assert "`mtb.describe_layout(\"diagonal\")` lists each method's ATAC files." in title
    lines = [ln.strip() for ln in mtb.describe_layout("diagonal").splitlines()]
    need_both = next(ln for ln in lines if ln.startswith("need both files:"))
    assert set(need_both.split(":", 1)[1].strip().split(", ")) == set(both)


def test_diagonal_section_3_prints_the_layout_the_title_points_to():
    code = [src for kind, src in _cells("tutorial_diagonal") if kind == "code"]
    assert "print(mtb.describe_layout(CATEGORY))" in code
    assert '"diagonal"' in next(src for src in code if src.startswith("%matplotlib"))


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


@pytest.mark.parametrize("name", TUTORIALS)
def test_the_overwrite_sentences_match_the_live_error(name, tmp_path):
    """Troubleshooting and the export details say a second export raises
    FileExistsError and ``overwrite=True`` replaces the files."""
    import multibench as mtb
    symptom, fix = _trouble_row(name, "overwrite=True")
    assert symptom == "`FileExistsError` from `export_dataset`"
    assert "already holds" in fix
    label = GEN.EXPORT_DETAIL_LABEL[name.removeprefix("tutorial_")]
    export = [src for kind, src in _cells(name) if kind == "markdown"
              and f"<summary>{label}</summary>" in src]
    assert len(export) == 1 and GEN.OVERWRITE_NOTE in export[0]
    kw = dict(rna="X", adt="obsm:protein", labels="obs:celltype")
    folder = mtb.io.export_dataset(_demo(), tmp_path / "MYCITE", **kw)
    with pytest.raises(FileExistsError) as e:
        mtb.io.export_dataset(_demo(), folder, **kw)
    msg = str(e.value)
    assert "already holds" in msg and msg.endswith("Pass overwrite=True to replace them.")
    mtb.io.export_dataset(_demo(), folder, overwrite=True, **kw)


@pytest.mark.parametrize("name", TUTORIALS)
def test_the_raw_counts_row_matches_the_live_warning(name, tmp_path):
    import multibench as mtb
    symptom, fix = _trouble_row(name, "raw counts")
    assert symptom == "a warning that values are not whole numbers"
    assert fix == 'export raw counts, for example with `rna="layer:counts"`'
    with pytest.warns(UserWarning) as rec:
        mtb.io.export_dataset(_demo(counts=False), tmp_path / "LOGNORM", rna="X",
                              adt="obsm:protein", labels="obs:celltype")
    msgs = [str(w.message) for w in rec if "whole numbers" in str(w.message)]
    assert len(msgs) == 1, msgs
    assert "not whole numbers" in msgs[0]
    assert "Export raw counts, for example with rna='layer:counts'." in msgs[0]


# ---------------------------- R7-10: the environment reason (guard)
@pytest.mark.parametrize("name", TUTORIALS)
def test_env_reason_on_linux_holds_the_install_command_the_tutorials_name(name, monkeypatch):
    """Troubleshooting: "run the `multibench env install ...` command in
    `env_reason`"; section 5: "`env_reason` gives the install command"."""
    import multibench as mtb
    from multibench import workflow as W
    from multibench.engine import runner as R
    symptom, fix = _trouble_row(name, "`env_reason`")
    assert symptom == "`env_ok` False on Linux"
    assert "`multibench env install ...`" in fix
    ref = _markdown(name, "## 5. Reference")
    assert "`env_reason` gives the install command" in ref
    monkeypatch.setattr(R, "linux_only_sentence", lambda: None)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    cat = name.removeprefix("tutorial_")
    m = GEN.SCEN[cat]["own_trio"][0]
    ds = GEN.SCEN[cat]["live_ds"] or GEN.SCEN[cat]["ds"]
    if not (mtb.config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    row = _quiet(mtb.scan, ds, cat, methods=[m], verbose=False).iloc[0]
    assert not row["env_ok"]
    assert f"multibench env install --methods {m} --packed --run" in row["env_reason"]
