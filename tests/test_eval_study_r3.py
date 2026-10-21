"""Scoring and run messages after the third student study (fix round 3).

R3-14: a label count that differs from the embedding's cell count names the
argument the user passed (the CLI flag on the command line) and, for one
label file of several batches, the fix. The export error names the object
passed as ``data`` in plain words.
R3-16: the sweep note is short sentences (one since 0.4.0); off Linux, run's
missing-env error is the Linux sentence alone.
"""
import re

import numpy as np
import pandas as pd
import pytest

import multibench as mtb

pytest.importorskip("scib")


def _blobs(n_per=30, n_labels=3, dims=5, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 8, size=(n_labels, dims))
    emb = np.vstack([centres[i] + rng.normal(0, 1, size=(n_per, dims))
                     for i in range(n_labels)])
    lab = np.array([f"t{i}" for i in range(n_labels) for _ in range(n_per)])
    return emb, lab


def _batch_folder(tmp_path):
    """An embedding that stacks three batches, and one label file per batch."""
    emb, lab = _blobs()
    np.save(tmp_path / "emb.npy", emb)
    folder = tmp_path / "LABMOS"
    folder.mkdir()
    for i in range(3):
        pd.DataFrame({"x": lab[30 * i:30 * (i + 1)]}).to_csv(folder / f"cty{i + 1}.csv",
                                                             index=False)
    return emb, lab, folder


# --- R3-14 (a) ---------------------------------------------------------------

def test_python_label_count_error_names_dataset_category_and_method(tmp_path):
    # 0.4.0: the hint names evaluate's own keywords, not the deprecated labels_for
    emb, _, folder = _batch_folder(tmp_path)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(folder / "cty1.csv"), metrics=["ASW"])
    assert str(e.value) == (
        "labels has 30 entries for 90 cells in the embedding. For a folder with one "
        "label file per batch, pass dataset=, category= and method=.")


def test_more_labels_than_cells_gives_no_per_batch_advice():
    emb, lab = _blobs()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb[:60], labels=lab, metrics=["ASW"])
    assert str(e.value) == ("labels has 90 entries for 60 cells in the embedding: give one "
                            "label per cell.")


# --- the review of 0.3.5: label files of one diagonal side, and none at all ---

def _folder(root, name, cells, labels):
    """A dataset folder ``root/name``: ``cells`` gives the cells of each data
    file, ``labels`` the labels of each label file."""
    import h5py
    d = root / name
    d.mkdir(parents=True)
    for file, n in cells.items():
        with h5py.File(d / file, "w") as f:
            g = f.create_group("matrix")
            g.create_dataset("data", data=np.ones((4, n)))
            g.create_dataset("features", data=np.array(
                [f"GENE{i}" for i in range(4)], dtype="S32"))
            g.create_dataset("barcodes", data=np.array(
                [f"{file[:-3]}_{i}" for i in range(n)], dtype="S32"))
    for file, n in labels.items():
        pd.DataFrame({"x": ["T", "B"] * (n // 2)}).to_csv(d / file, index=False)
    return d


def test_an_empty_labels_dict_has_nothing_to_score(tmp_path):
    """mtb.labels_for returns an empty dict for a folder without label files.
    evaluate says that the dict is empty, not that the labels have 0 entries."""
    emb, _ = _blobs()
    text = "labels is an empty dict. Pass the cell types, as labels=adata.obs['celltype']."
    data = tmp_path / "data"
    _folder(data, "NOLAB", {"rna.h5": 50, "atac_gas.h5": 40}, {})
    _folder(data, "X_NOLAB", {f"{m}{i}.h5": 45 for m in ("rna", "adt") for i in (1, 2)},
            {})
    for labels in ({}, mtb.engine.resolve.labels_for("NOLAB", "diagonal", "SCALEX", data_path=data),
                   mtb.engine.resolve.labels_for("X_NOLAB", "cross", data_path=data)):
        assert labels == {}
        with pytest.raises(ValueError) as e:
            mtb.evaluate(emb, labels=labels, metrics=["ASW"])
        assert str(e.value) == text


# --- R3-14 (b) ---------------------------------------------------------------

def test_export_partial_overlap_names_the_object_passed_as_data(tmp_path):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    rna = ad.AnnData(rng.poisson(1.0, size=(30, 4)).astype(float))
    rna.obs_names = [f"c{i}" for i in range(30)]
    rna.var_names = [f"g{i}" for i in range(4)]
    atac = ad.AnnData(rng.poisson(1.0, size=(25, 3)).astype(float))
    atac.obs_names = [f"c{i}" for i in range(20)] + [f"x{i}" for i in range(5)]
    atac.var_names = ["chr1:1-100", "chr1:200-300", "chr2:1-100"]
    with pytest.raises(ValueError) as e:
        mtb.io.export_dataset(rna, tmp_path / "X", atac=atac, atac_kind="peak")
    msg = str(e.value)
    assert " in data," not in msg
    assert msg.startswith("atac has 25 cells, and 5 of them are not in the object "
                          "passed as data, for example")


# --- R3-16 (a) ---------------------------------------------------------------

def test_sweep_note_is_short_sentences(capsys, monkeypatch):
    from multibench.eval import scib as mscib
    monkeypatch.setattr(mscib, "_SWEEP_NOTICE_CELLS", 0)
    emb, lab = _blobs()
    mtb.evaluate(emb, labels=lab)
    err = capsys.readouterr().err.strip()
    # 0.4.0: one sentence, without "To skip it ..."
    assert re.fullmatch(
        r"Clustering 90 cells for ARI, NMI, ASW, iASW and iF1 with the "
        r"(igraph|leidenalg) Leiden backend\.", err), err
    assert " - " not in err and ";" not in err and "per 3,000 cells" not in err


# --- R3-16 (b) ---------------------------------------------------------------

DARWIN = ("method environments are linux-64 only; this host is darwin/arm64")


def test_run_off_linux_is_the_linux_sentence_alone(tmp_path, monkeypatch):
    from multibench.engine import envs, runner
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])
    with pytest.raises(OSError) as e:
        mtb.engine.runner.run("totalVI", "vertical", inputs=mtb.engine.resolve.inputs_for("D11", "vertical", "totalVI"),
                out_dir=str(tmp_path / "out"))
    msg = str(e.value)
    assert msg == ("Methods run only on Linux, and this computer runs macOS. Run this "
                   "call on a Linux machine. mtb.scan shows the method's command here.")
    assert "env install" not in msg and "is not installed" not in msg


def test_run_on_linux_keeps_the_install_line(tmp_path, monkeypatch):
    from multibench.engine import runner
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])
    with pytest.raises(OSError) as e:
        mtb.engine.runner.run("totalVI", "vertical", inputs=mtb.engine.resolve.inputs_for("D11", "vertical", "totalVI"),
                out_dir=str(tmp_path / "out"))
    msg = str(e.value)
    assert "is not installed" in msg and "mtb.env.install(['totalVI'])" in msg
    assert "Methods run only on Linux" not in msg


def test_label_files_of_one_diagonal_side_are_named(tmp_path, capsys):
    """rna_cty.csv without atac_cty.csv: the 70 ATAC cells of the embedding
    have no entry. The message names the side that is labelled and a recipe.
    It does not blame the embedding or name the call that was made, and
    missing_labels="drop" does not change it."""
    emb = np.random.default_rng(0).normal(size=(150, 5))
    np.save(tmp_path / 'emb.npy', emb)
    data = tmp_path / 'data'
    d = _folder(data, 'BRAIN', {'rna.h5': 80, 'atac_gas.h5': 70}, {'rna_cty.csv': 80})
    found = mtb.engine.resolve.labels_for('BRAIN', 'diagonal', 'SCALEX', data_path=data)
    assert list(found) == ['rna_cty']
    out = ['--embedding', str(tmp_path / 'emb.npy'), '--metrics', 'ASW']
    dataset = ['--dataset', 'BRAIN', '--method', 'SCALEX', '--category', 'diagonal', '--data-path', str(data)]
    d = _folder(data, 'ATAC', {}, {'atac_cty.csv': 70})
    d = _folder(data, 'RNA2', {}, {'rna_cty1.csv': 40, 'rna_cty2.csv': 40})


def test_other_short_label_files_keep_their_advice(tmp_path, capsys):
    """rna_cty.csv is also the older name of a paired folder's label file, and
    a diagonal folder may hold atac_cty.csv next to it: those keep their text."""
    emb = np.random.default_rng(0).normal(size=(150, 5))
    np.save(tmp_path / 'emb.npy', emb)
    np.save(tmp_path / 'emb80.npy', emb[:80])
    data = tmp_path / 'data'
    out = ['--embedding', str(tmp_path / 'emb.npy'), '--metrics', 'ASW']
    _folder(data, 'OLDPAIR', {'rna.h5': 120, 'adt.h5': 120}, {'rna_cty.csv': 120})
    d = _folder(data, 'BOTH', {'rna.h5': 80, 'atac_gas.h5': 70}, {'rna_cty.csv': 80, 'atac_cty.csv': 70})
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(d / 'rna_cty.csv'), metrics=['ASW'])
    assert str(e.value) == 'labels has 80 entries for 150 cells in the embedding. For a folder with one label file per batch, pass dataset=, category= and method=.'
    d = _folder(data, 'RNA2', {}, {'rna_cty1.csv': 40, 'rna_cty2.csv': 40})
