"""GPSA reads ``obs['Ground_Truth']`` from every slice; scan says so up front.

``engine/drivers/run_gpsa.py`` indexes ``slice_i.obs['Ground_Truth']`` at load,
so a slice folder without that column passed scan (``files_ok=True``, empty
caveat) and died with a KeyError right after the 5 GB env build. The
requirement is declared on the variant (``slice_obs: [Ground_Truth]`` in
methods.yaml) and checked where scan already opens each slice for
``obsm['spatial']``; PASTE / PASTE2 declare nothing and are unaffected.
"""
import numpy as np
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import registry, resolve


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


def _slice(path, n=30, *, ground_truth=True, spatial=True):
    import anndata as ad
    a = ad.AnnData(X=np.random.default_rng(0).poisson(1.0, (n, 12)).astype(np.float32))
    if spatial:
        a.obsm["spatial"] = np.random.default_rng(1).normal(size=(n, 2))
    if ground_truth:
        a.obs["Ground_Truth"] = ["L1", "L2"] * (n // 2)
    a.write_h5ad(path)


def _folder(root, name, **kw):
    d = root / name / "processed"
    d.mkdir(parents=True)
    for i in range(3):
        _slice(d / f"slice_{i}.h5ad", **kw)
    return d


def test_gpsa_declares_the_column_and_paste_does_not():
    assert registry.get("GPSA").variants[0].slice_obs == ["Ground_Truth"]
    for m in ("PASTE", "PASTE2"):
        assert registry.get(m).variants[0].slice_obs == []


def test_slices_without_ground_truth_block_gpsa_only(tmp_path, no_envs):
    _folder(tmp_path, "NOGT", ground_truth=False)
    df = mtb.scan("NOGT", "cross", data_path=tmp_path, modalities=[]).set_index("method")
    assert not df.loc["GPSA", "files_ok"]
    why = df.loc["GPSA", "files_reason"]
    assert "slice_" in why and ".h5ad has no obs['Ground_Truth'] column" in why
    assert "EVERY slice" in why and "describe_layout('cross')" in why
    assert "obs['Ground_Truth']" in df.loc["GPSA", "reason"]
    assert df.loc["PASTE", "files_ok"] and df.loc["PASTE2", "files_ok"]


def test_slices_with_ground_truth_pass_all_three(tmp_path, no_envs):
    _folder(tmp_path, "WITHGT", ground_truth=True)
    df = mtb.scan("WITHGT", "cross", data_path=tmp_path, modalities=[]).set_index("method")
    for m in ("GPSA", "PASTE", "PASTE2"):
        assert df.loc[m, "files_ok"], df.loc[m, "files_reason"]


def test_one_bad_slice_is_named(tmp_path, no_envs):
    d = _folder(tmp_path, "ONEBAD", ground_truth=True)
    _slice(d / "slice_1.h5ad", ground_truth=False)
    df = mtb.scan("ONEBAD", "cross", data_path=tmp_path, modalities=[]).set_index("method")
    assert "slice_1.h5ad has no obs['Ground_Truth']" in df.loc["GPSA", "files_reason"]


def test_spatial_check_still_precedes_the_column_check(tmp_path, no_envs):
    _folder(tmp_path, "NOSPATIAL", ground_truth=False, spatial=False)
    df = mtb.scan("NOSPATIAL", "cross", data_path=tmp_path, modalities=[]).set_index("method")
    assert "obsm['spatial']" in df.loc["GPSA", "files_reason"]
    assert "obsm['spatial']" in df.loc["PASTE", "files_reason"]


def test_check_data_dir_reads_slice_obs_from_the_variant(tmp_path):
    d = _folder(tmp_path, "X", ground_truth=False)
    ok, why = resolve._check_data_dir(registry.get("GPSA").variants[0], str(d))
    assert not ok and "obs['Ground_Truth']" in why
    ok, why = resolve._check_data_dir(registry.get("PASTE").variants[0], str(d))
    assert ok and why == ""


def test_describe_layout_states_the_contracts():
    txt = mtb.describe_layout("cross")
    assert "GPSA additionally" in txt and "obs['Ground_Truth']" in txt
    assert "EVERY slice" in txt
    assert "PASTE writes its slices WITHOUT any obs column" in txt
    assert "PASTE2 rewrites .X" in txt and "log1p" in txt
    assert "slices_manifest.json" in txt
    assert "every slice carries obs['Ground_Truth']" in txt
    # the cross-only block is also in the full listing
    assert "GPSA additionally" in mtb.describe_layout()
    assert "GPSA additionally" not in mtb.describe_layout("vertical")
