"""``mtb.data.fetch_outputs``: the benchmark host's ``run_all`` tree as a stand-in.

On a host without the methods' conda envs (Colab, a laptop) the tutorial
loads real outputs instead of running: ``mtb.load_batch(
mtb.data.fetch_outputs("D11"), methods=trio)``. These tests serve a local
tarball through the module's single network seam (``fetch._download``) so
download bookkeeping, idempotence, the traversal guard, the methods filter
and the ``load_batch(methods=)`` filter run without the network.
"""
import fnmatch
import importlib
import json
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb

# `multibench.data.fetch` the attribute is the fetch() FUNCTION; this is the module
F = importlib.import_module("multibench.data.fetch")

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Matilda", "MOFA2", "scMoMaT"]
DATASET = "D11"


def _write_tree(d: Path, dataset=DATASET, methods=METHODS) -> None:
    """A run_all output tree exactly as BatchResult.save + run() lay it out."""
    d.mkdir(parents=True)
    recs, rows = [], []
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        mdir = d / f"{m}_{dataset}"                     # run_all's per-method folder
        mdir.mkdir()
        with h5py.File(mdir / "embedding.h5", "w") as f:
            f.create_dataset("data", data=rng.normal(size=(4, 50)))
        metrics = {"ARI": 0.5 + i / 10, "NMI": 0.6 + i / 10}
        recs.append({"method": m, "category": "vertical", "dataset": dataset,
                     "modalities": ["rna", "adt"], "output_kind": "embedding",
                     "env": "x", "n_tunable": 1, "status": "CHAIN_OK",
                     "run_sec": 1.0, "emb_shape": [50, 4], "metrics": metrics,
                     "out_dir": f"/benchmark/host/out/{m}_{dataset}"})
        for k, v in metrics.items():
            rows.append({"metric": k, "value": v, "method": m, "dataset": dataset,
                         "category": "vertical", "clustering": "default",
                         "source": "user"})
    pd.DataFrame(rows).to_csv(d / "long.csv", index=False)
    pd.DataFrame([{"method": r["method"], "status": r["status"]} for r in recs]
                 ).to_csv(d / "summary.csv", index=False)
    with open(d / "batch_result.json", "w") as fh:
        json.dump({"dataset": dataset, "category": "vertical", "records": recs}, fh)


def _tar_of(src: Path, tgz: Path, arcroot: str | None) -> Path:
    """Tar ``src`` rooted at ``arcroot/`` (``None``: the tree's files at top level)."""
    with tarfile.open(tgz, "w:gz") as t:
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src).as_posix()
            t.add(p, arcname=f"{arcroot}/{rel}" if arcroot else rel, recursive=False)
    return tgz


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A local ``outputs-D11.tar.gz`` behind the monkeypatched downloader.

    Returns ``(data_path, calls)``; ``calls`` collects every URL requested.
    """
    src = tmp_path / "src" / f"outputs-{DATASET}"
    _write_tree(src)
    tgz = _tar_of(src, tmp_path / "outputs-D11.tar.gz", f"outputs-{DATASET}")
    calls = []

    def fake_download(url):
        calls.append(url)
        return tgz

    monkeypatch.setattr(F, "_download", fake_download)
    return tmp_path / "data", calls


# ------------------------------------------------------------------ manifest
def test_manifest_is_valid_json_with_the_four_tutorial_datasets():
    mf = ROOT / "multibench" / "engine" / "output_urls.json"
    urls = json.loads(mf.read_text())
    assert sorted(urls) == ["D11", "D28", "D46", "D52"]
    for ds, url in urls.items():
        assert url == ("https://github.com/DSichang/scMultiBench/releases/download/"
                       f"data-v1/outputs-{ds}.tar.gz")
    assert F.OUTPUT_MANIFEST == mf.resolve()
    assert F._output_urls() == urls


def test_manifest_ships_in_the_wheel():
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r"multibench\s*=\s*\[(.*?)\]", text, re.S)
    globs = re.findall(r'"([^"]+)"', m.group(1))
    assert any(fnmatch.fnmatch("engine/output_urls.json", g) for g in globs)


def test_fetch_outputs_is_public_under_data():
    assert "fetch_outputs" in mtb.data.__all__
    assert mtb.data.fetch_outputs is F.fetch_outputs


# ------------------------------------------------------------------ download
def test_downloads_extracts_and_returns_the_dataset_folder(served, capsys):
    data, calls = served
    out = mtb.data.fetch_outputs(DATASET, data_path=data)
    assert out == data / "outputs" / DATASET
    assert calls == [F._output_urls()[DATASET]]
    assert (out / "batch_result.json").is_file()
    assert (out / "long.csv").is_file() and (out / "summary.csv").is_file()
    for m in METHODS:
        assert (out / f"{m}_{DATASET}" / "embedding.h5").is_file()
    # the scratch extraction dir is gone
    assert [p.name for p in (data / "outputs").iterdir()] == [DATASET]
    printed = capsys.readouterr().out
    assert f"downloading run_all outputs for {DATASET} from {calls[0]}" in printed


def test_idempotent_second_call_downloads_nothing(served, capsys):
    data, calls = served
    first = mtb.data.fetch_outputs(DATASET, data_path=data)
    capsys.readouterr()
    again = mtb.data.fetch_outputs(DATASET, data_path=data)
    assert again == first and len(calls) == 1
    assert capsys.readouterr().out == ""


def test_quiet_prints_nothing(served, capsys):
    data, _ = served
    mtb.data.fetch_outputs(DATASET, data_path=data, quiet=True)
    assert capsys.readouterr().out == ""


def test_default_data_path_is_the_configured_one(served, monkeypatch, tmp_path):
    data, _ = served
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path / "cfg")
    out = mtb.data.fetch_outputs(DATASET, quiet=True)
    assert out == tmp_path / "cfg" / "outputs" / DATASET
    assert (out / "batch_result.json").is_file()


def test_archive_rooted_at_the_tree_itself_is_accepted(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _write_tree(src)
    tgz = _tar_of(src, tmp_path / "flat.tar.gz", None)
    monkeypatch.setattr(F, "_download", lambda url: tgz)
    out = mtb.data.fetch_outputs(DATASET, data_path=tmp_path / "data", quiet=True)
    assert (out / "batch_result.json").is_file()
    assert (out / f"{METHODS[0]}_{DATASET}" / "embedding.h5").is_file()


def test_archive_without_batch_result_raises_and_leaves_no_folder(tmp_path, monkeypatch):
    src = tmp_path / "src" / "outputs-D11"
    src.mkdir(parents=True)
    (src / "summary.csv").write_text("method\n")
    tgz = _tar_of(src.parent, tmp_path / "bad.tar.gz", None)
    monkeypatch.setattr(F, "_download", lambda url: tgz)
    with pytest.raises(RuntimeError, match="did not contain batch_result.json"):
        mtb.data.fetch_outputs(DATASET, data_path=tmp_path / "data", quiet=True)
    assert not (tmp_path / "data" / "outputs" / DATASET).exists()
    assert list((tmp_path / "data" / "outputs").iterdir()) == []


def test_traversal_safe(tmp_path, monkeypatch):
    tgz = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("z")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(payload, arcname="outputs-D11/../../escape.txt")
    monkeypatch.setattr(F, "_download", lambda url: tgz)
    with pytest.raises(RuntimeError, match="escapes target dir"):
        mtb.data.fetch_outputs(DATASET, data_path=tmp_path / "data", quiet=True)
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "data" / "escape.txt").exists()


def test_foreign_nonempty_folder_without_batch_result_is_refused(served):
    data, calls = served
    dest = data / "outputs" / DATASET
    dest.mkdir(parents=True)
    (dest / "notes.txt").write_text("mine")
    with pytest.raises(RuntimeError, match="exists without batch_result.json"):
        mtb.data.fetch_outputs(DATASET, data_path=data, quiet=True)
    assert (dest / "notes.txt").read_text() == "mine" and calls == []


def test_empty_leftover_folder_is_replaced(served):
    data, _ = served
    (data / "outputs" / DATASET).mkdir(parents=True)
    out = mtb.data.fetch_outputs(DATASET, data_path=data, quiet=True)
    assert (out / "batch_result.json").is_file()


# ------------------------------------------------------------------ ids / filter
def test_unknown_dataset_raises_listing_the_manifest_ids(served):
    data, calls = served
    with pytest.raises(ValueError) as e:
        mtb.data.fetch_outputs("D99", data_path=data)
    assert str(e.value) == ("'D99' has no shipped run_all outputs; available: "
                            "D11, D28, D46, D52")
    assert calls == []                      # validated before any download


def test_methods_filter_validates_against_the_tree(served):
    data, _ = served
    out = mtb.data.fetch_outputs(DATASET, ["Matilda", "scMoMaT"], data_path=data, quiet=True)
    assert out == data / "outputs" / DATASET       # the folder either way
    with pytest.raises(KeyError) as e:
        mtb.data.fetch_outputs(DATASET, ["Matilda", "nope"], data_path=data, quiet=True)
    msg = str(e.value)
    assert "no shipped output for ['nope'] in D11" in msg
    assert "methods in the tree: ['Matilda', 'MOFA2', 'scMoMaT']" in msg


def test_methods_filter_checks_after_the_download_on_a_fresh_tree(served):
    data, calls = served
    with pytest.raises(KeyError):
        mtb.data.fetch_outputs(DATASET, ["nope"], data_path=data, quiet=True)
    assert len(calls) == 1                  # the tree is on disk for next time
    assert (data / "outputs" / DATASET / "batch_result.json").is_file()


# ------------------------------------------------------------------ load_batch
def test_load_batch_methods_filter_is_the_tutorial_path(served):
    data, _ = served
    trio = ["scMoMaT", "Matilda"]
    res = mtb.load_batch(mtb.data.fetch_outputs(DATASET, trio, data_path=data, quiet=True),
                         methods=trio)
    assert [r["method"] for r in res.records] == ["Matilda", "scMoMaT"]   # tree order
    assert sorted(res.summary["method"]) == ["Matilda", "scMoMaT"]
    assert set(res.long["method"]) == {"Matilda", "scMoMaT"}
    assert float(res.summary.set_index("method").loc["scMoMaT", "ARI"]) == pytest.approx(0.7)
    assert res.dataset == DATASET and res.category == "vertical"


def test_load_batch_without_methods_is_unchanged(served):
    data, _ = served
    res = mtb.load_batch(mtb.data.fetch_outputs(DATASET, data_path=data, quiet=True))
    assert [r["method"] for r in res.records] == METHODS
    assert len(res.long) == 2 * len(METHODS)


def test_load_batch_unknown_method_raises_listing_the_tree(served):
    data, _ = served
    out = mtb.data.fetch_outputs(DATASET, data_path=data, quiet=True)
    with pytest.raises(KeyError) as e:
        mtb.load_batch(out, methods=["Matilda", "ghost"])
    msg = str(e.value)
    assert "no record for ['ghost'] in" in msg
    assert "methods in the tree: ['Matilda', 'MOFA2', 'scMoMaT']" in msg


def test_load_batch_methods_is_keyword_only():
    import inspect
    p = inspect.signature(mtb.load_batch).parameters["methods"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


# ------------------------------------------------------------------ seam
def test_fetch_of_reference_datasets_goes_through_the_same_downloader(tmp_path, monkeypatch):
    """The refactor that added ``_download`` must not change ``fetch()``."""
    src = tmp_path / "src" / "D11"
    src.mkdir(parents=True)
    (src / "rna.h5").write_text("x")
    tgz = _tar_of(src.parent, tmp_path / "D11.tar.gz", None)
    calls = []

    def fake(url):
        calls.append(url)
        return tgz

    monkeypatch.setattr(F, "_download", fake)
    root = mtb.data.fetch("D11", data_path=tmp_path / "data", quiet=True)
    assert calls == [f"{F.RELEASE_URL}/D11.tar.gz"]
    assert (root / "D11" / "rna.h5").read_text() == "x"
