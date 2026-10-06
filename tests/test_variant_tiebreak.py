"""The folder tie-break between nested variants of one method.

scMoMaT, Multigrate, MOFA2 and Matilda have ``rna+adt``, ``rna+atac`` and
``rna+adt+atac`` under vertical. A folder with all three files has every
input of all three, so ``inputs_for`` without ``modalities=`` raised
``AmbiguousVariantError`` and ``run_all`` ran the method once per row, each
into the same output folder. The rule: of the variants the folder satisfies,
one whose modality set is a strict subset of another's is dropped. What
remains, when it is one variant, is used. ``modalities=`` still selects any.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry, resolve

DEMO = Path(os.environ.get("MULTIBENCH_DATA_PATH", ""))
NESTED = ("scMoMaT", "Multigrate", "MOFA2")       # these read peak ATAC


@pytest.fixture
def all_envs(monkeypatch):
    """Every env installed, so only the file check decides (test_run_all_dispatch)."""
    every = frozenset(envs.group_for(m) for m in registry.list_methods())
    monkeypatch.setattr(W, "_installed_envs", lambda: every)


def _h5(path, feats, cells):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), len(cells))).astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S40"))
        g.create_dataset("barcodes", data=np.array(cells, dtype="S40"))


def _tri(root, name="TRI", n=24):
    """rna.h5 + adt.h5 + atac.h5 (peaks) + cty.csv, the same cells."""
    d = root / name
    d.mkdir(parents=True)
    cells = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], cells)
    _h5(d / "adt.h5", [f"p{i}" for i in range(8)], cells)
    _h5(d / "atac.h5", [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)], cells)
    pd.DataFrame({"x": (["A", "B"] * n)[:n]}).to_csv(d / "cty.csv", index=False)
    return d


def _v(category, *mods):
    return SimpleNamespace(when={"category": category, "modalities": list(mods)})


def _mods(variants):
    return ["+".join(v.when["modalities"]) for v in variants]


# -------------------------------------------------------------------- the rule
def test_drop_nested_keeps_the_largest_of_nested_sets():
    small, other, big = _v("vertical", "rna", "adt"), _v("vertical", "rna", "atac"), \
        _v("vertical", "rna", "adt", "atac")
    assert _mods(resolve._drop_nested([small, big, other])) == ["rna+adt+atac"]
    assert _mods(resolve._drop_nested([small, big])) == ["rna+adt+atac"]
    # not nested: both stay
    assert _mods(resolve._drop_nested([small, other])) == ["rna+adt", "rna+atac"]
    # a chain keeps its top only, a second top stays
    two = _v("cross", "adt1", "adt2", "atac1", "atac2")
    three = _v("cross", "rna1", "rna2", "adt1", "adt2", "atac1", "atac2")
    lone = _v("cross", "rna1", "rna2", "rna3", "adt1", "adt2", "adt3")
    assert resolve._drop_nested([two, three, lone]) == [three, lone]
    # another category's variant is never the larger one
    assert resolve._drop_nested([small, _v("cross", "rna", "adt", "atac")])[0] is small
    assert resolve._drop_nested([]) == [] and resolve._drop_nested([small]) == [small]


# ------------------------------------------------------- inputs_for / labels_for
@pytest.mark.parametrize("method", NESTED + ("Matilda",))
def test_the_folder_picks_the_largest_nested_variant(tmp_path, method):
    _tri(tmp_path)
    got = mtb.inputs_for("TRI", "vertical", method, data_path=tmp_path)
    assert {"rna", "adt", "atac"} <= set(got)
    assert Path(got["atac"]).name == "atac.h5" and Path(got["adt"]).name == "adt.h5"
    # labels_for selects the same variant and does not raise
    assert list(mtb.labels_for("TRI", "vertical", method, data_path=tmp_path)) == ["cty"]


@pytest.mark.parametrize("mods", [["rna", "adt"], ["rna", "atac"], ["rna", "adt", "atac"]])
def test_modalities_still_select_any_variant(tmp_path, mods):
    _tri(tmp_path)
    got = mtb.inputs_for("TRI", "vertical", "scMoMaT", modalities=mods, data_path=tmp_path)
    assert set(got) == set(mods)


def test_a_smaller_folder_still_picks_its_own_variant(tmp_path):
    d = _tri(tmp_path)
    (d / "atac.h5").unlink()
    assert set(mtb.inputs_for("TRI", "vertical", "scMoMaT", data_path=tmp_path)) == {"rna", "adt"}


@pytest.mark.parametrize("method", ["VIMCCA", "Seurat_WNN"])
def test_variants_that_are_not_nested_stay_ambiguous(tmp_path, method):
    """rna+adt and rna+atac, with no rna+adt+atac variant above them."""
    d = _tri(tmp_path)
    with pytest.raises(mtb.AmbiguousVariantError) as e:
        mtb.inputs_for("TRI", "vertical", method, data_path=tmp_path)
    msg = str(e.value)
    assert f"{method} has 2 vertical variants, rna+adt and rna+atac." in msg
    assert f"The folder {d} has every input file of both." in msg
    assert "modalities=['rna', 'adt']" in msg
    with pytest.raises(mtb.AmbiguousVariantError):
        mtb.params_for(method, "vertical", dataset="TRI", data_path=tmp_path)
    got = mtb.inputs_for("TRI", "vertical", method, modalities=["rna", "atac"],
                         data_path=tmp_path)
    assert set(got) == {"rna", "atac"}


def test_no_file_of_any_variant_is_still_ambiguous(tmp_path):
    (tmp_path / "EMPTY").mkdir()
    with pytest.raises(mtb.AmbiguousVariantError, match="None of them has every input file"):
        mtb.inputs_for("EMPTY", "vertical", "scMoMaT", data_path=tmp_path)


# -------------------------------------------------------------------- params_for
def test_params_for_uses_the_same_tie_break(tmp_path):
    _tri(tmp_path)
    for method in NESTED:
        p = mtb.params_for(method, "vertical", dataset="TRI", data_path=tmp_path)
        assert p["variant"] == "vertical:rna+adt+atac"
    # with no category the folder holds the files of the vertical variants only
    assert mtb.params_for("MOFA2", dataset="TRI",
                          data_path=tmp_path)["variant"] == "vertical:rna+adt+atac"
    assert mtb.params_for("scMoMaT", "vertical", ["rna", "adt"], dataset="TRI",
                          data_path=tmp_path)["variant"] == "vertical:rna+adt"


# ------------------------------------------------------------- scan and run_all
class _Res:
    def __init__(self, out):
        self.output = out


def _fake_run(calls, n_cells=24):
    def _inner(method, category, inputs, out_dir, params=None):
        calls.append((method, sorted(inputs)))
        return _Res(np.zeros((n_cells, 5)))
    return _inner


def test_scan_keeps_every_row(tmp_path, all_envs):
    _tri(tmp_path)
    df = mtb.scan("TRI", "vertical", methods=["scMoMaT"], data_path=tmp_path, verbose=False)
    assert sorted(df["modalities"]) == ["rna+adt", "rna+adt+atac", "rna+atac"]
    assert df["runnable"].all()


def test_run_all_runs_the_largest_variant_once(tmp_path, all_envs, monkeypatch, capsys):
    _tri(tmp_path)
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    res = mtb.run_all("TRI", "vertical", methods=list(NESTED), data_path=tmp_path,
                      out_dir=str(tmp_path / "out"), evaluate=False)
    assert sorted(calls) == [(m, ["adt", "atac", "rna"]) for m in sorted(NESTED)]
    assert sorted(res.summary["method"]) == sorted(NESTED)
    out = capsys.readouterr().out
    assert ("[run_all] scMoMaT runs on rna+adt+atac, not on rna+adt and rna+atac, whose "
            "files are part of it. Pass modalities= to run a smaller combination.") in out


def test_run_all_with_modalities_runs_the_smaller_variant(tmp_path, all_envs, monkeypatch):
    _tri(tmp_path)
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    mtb.run_all("TRI", "vertical", methods=["scMoMaT"], modalities=["rna", "adt"],
                data_path=tmp_path, out_dir=str(tmp_path / "out"), evaluate=False,
                verbose=False)
    assert calls == [("scMoMaT", ["adt", "rna"])]


def test_dry_run_keeps_the_frame_and_names_the_row_that_runs(tmp_path, all_envs, capsys):
    _tri(tmp_path)
    plan = mtb.run_all("TRI", "vertical", methods=["scMoMaT", "VIMCCA"], data_path=tmp_path,
                       dry_run=True)
    assert len(plan) == 5 and plan["runnable"].all()
    out = capsys.readouterr().out
    assert "[run_all] scMoMaT runs on rna+adt+atac, not on rna+adt and rna+atac" in out
    assert "[run_all] VIMCCA runs on" not in out       # its two rows are not nested


def test_a_blocked_larger_row_leaves_the_smaller_one_running(tmp_path, all_envs, monkeypatch,
                                                             gas_matilda):
    """A method that reads gene activity (``gas_matilda``; the real Matilda
    reads peaks): on peaks its ATAC rows are blocked, and the sweep runs the
    largest row that can run."""
    _tri(tmp_path)
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    mtb.run_all("TRI", "vertical", methods=["Matilda"], data_path=tmp_path,
                out_dir=str(tmp_path / "out"), evaluate=False, verbose=False)
    assert calls == [("Matilda", ["adt", "cty", "rna"])]


def test_nested_rows_leaves_unrelated_rows_alone():
    plan = pd.DataFrame({"method": ["A", "A", "A", "B", "B", "C"],
                         "modalities": ["rna+adt", "rna+adt+atac", "rna+atac",
                                        "rna+adt", "rna+atac", "(data_dir)"]})
    drop, notes = W._nested_rows(plan)
    assert drop == [0, 2] and len(notes) == 1 and notes[0].startswith("A runs on rna+adt+atac")


# ------------------------------------------------------------------ cross batches
def _cross(root, name="B3", batches=3):
    d = root / name
    d.mkdir(parents=True)
    for i in range(1, batches + 1):
        cells = [f"b{i}_{k}" for k in range(10 + i)]
        _h5(d / f"rna{i}.h5", [f"g{k}" for k in range(20)], cells)
        _h5(d / f"adt{i}.h5", [f"p{k}" for k in range(6)], cells)
        pd.DataFrame({"x": ["A"] * len(cells)}).to_csv(d / f"cty{i}.csv", index=False)
    return d


def test_uinmf_resolves_to_the_variant_of_the_folders_batch_count(tmp_path):
    """UINMF has a cross variant for two batches and one for three. The
    two-batch files are part of the three-batch ones, so a folder with three
    batches picks the three-batch variant."""
    _cross(tmp_path)
    got = mtb.inputs_for("B3", "cross", "UINMF", data_path=tmp_path)
    assert list(got) == ["rna1", "rna2", "rna3", "adt1", "adt2", "adt3"]
    assert list(mtb.labels_for("B3", "cross", "UINMF", data_path=tmp_path)) == [
        "cty1", "cty2", "cty3"]
    two = ["rna1", "rna2", "adt1", "adt2"]
    assert list(mtb.inputs_for("B3", "cross", "UINMF", modalities=two,
                               data_path=tmp_path)) == two
    _cross(tmp_path, "B2", batches=2)
    assert list(mtb.inputs_for("B2", "cross", "UINMF", data_path=tmp_path)) == two
    assert list(mtb.labels_for("B2", "cross", "UINMF", data_path=tmp_path)) == ["cty1", "cty2"]


def test_run_all_runs_uinmf_once_on_three_batches(tmp_path, all_envs, monkeypatch, capsys):
    _cross(tmp_path)
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls, n_cells=36))
    res = mtb.run_all("B3", "cross", methods=["UINMF"], data_path=tmp_path,
                      out_dir=str(tmp_path / "out"), evaluate=False)
    assert calls == [("UINMF", ["adt1", "adt2", "adt3", "rna1", "rna2", "rna3"])]
    assert list(res.summary["method"]) == ["UINMF"]
    assert ("[run_all] UINMF runs on rna1+rna2+rna3+adt1+adt2+adt3, not on "
            "rna1+rna2+adt1+adt2, whose files are part of it.") in capsys.readouterr().out


# -------------------------------------------------------------------- demo folders
def _demo(name):
    if not (DEMO / name).is_dir():
        pytest.skip(f"demo folder {name} is not in MULTIBENCH_DATA_PATH")
    return DEMO


def test_d22mini_resolves_its_three_modality_variants():
    data = _demo("D22mini")
    for method in NESTED + ("Matilda",):
        assert set(mtb.inputs_for("D22mini", "vertical", method, data_path=data)) >= {
            "rna", "adt", "atac"}
    with pytest.raises(mtb.AmbiguousVariantError):
        mtb.inputs_for("D22mini", "vertical", "VIMCCA", data_path=data)


def test_d52mini_uinmf_reads_every_batch():
    data = _demo("D52mini")
    assert list(mtb.inputs_for("D52mini", "cross", "UINMF", data_path=data)) == [
        "rna1", "rna2", "rna3", "adt1", "adt2", "adt3"]


@pytest.mark.parametrize("dataset, category", [
    ("D11", "vertical"), ("D27mini", "diagonal"), ("D27mini_vertical", "vertical"),
    ("D38mini", "mosaic"), ("D45mini", "mosaic"), ("D46mini", "mosaic"),
    ("D52mini", "cross")])
def test_the_rule_changes_nothing_where_one_variant_fits(dataset, category):
    """A method with one satisfiable variant among several gets that variant,
    as before the rule."""
    data = _demo(dataset)
    for spec in registry.load():
        cands = [v for v in spec.variants if v.when.get("category") == category]
        ok = [v for v in cands if resolve._variant_satisfiable(v, data / dataset, spec.id)]
        if len(ok) == 1 and len(cands) > 1:
            assert resolve.select_variant(spec, category, None,
                                          ds_dir=data / dataset) is ok[0]
