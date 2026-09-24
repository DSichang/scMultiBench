"""Scoring and run messages after the third student study (fix round 3).

R3-14: a label count that differs from the embedding's cell count names the
argument the user passed (the CLI flag on the command line) and, for one
label file of several batches, the fix. The export error names the object
passed as ``data`` in plain words.
R3-16: the sweep note is two or three short sentences; off Linux, run's
missing-env error is the Linux sentence alone.
"""
import re

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli

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

def test_cli_one_label_file_of_three_names_the_flags_and_the_fix(tmp_path, capsys):
    _, _, folder = _batch_folder(tmp_path)
    code = cli.main(["evaluate", "--output", str(tmp_path / "emb.npy"),
                     "--labels", str(folder / "cty1.csv"), "--metrics", "ASW"])
    err = capsys.readouterr().err
    assert code == 1
    assert ("error: --labels gave 30 labels (cty1.csv) for 90 cells in --output. "
            "For one label file per batch, repeat --labels in batch order (cty1.csv, "
            "cty2.csv, cty3.csv), or pass --dataset, --category and --method.") in err
    assert "celltype" not in err and "emb has" not in err


def test_python_label_count_error_names_labels_for(tmp_path):
    emb, _, folder = _batch_folder(tmp_path)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb, labels=str(folder / "cty1.csv"), metrics=["ASW"])
    assert str(e.value) == (
        "labels has 30 entries for 90 cells in the embedding. For a folder with one "
        "label file per batch, pass mtb.labels_for(dataset, category, method).")


def test_batch_and_clustering_counts_follow_the_same_pattern(tmp_path, capsys):
    emb, lab = _blobs()
    with pytest.raises(ValueError, match=r"^batch has 30 entries for 90 cells in the "
                                         r"embedding\.$"):
        mtb.evaluate(emb, labels=lab, batch=lab[:30], metrics="all")
    with pytest.raises(ValueError, match=r"^clustering has 30 entries for 90 cells in "
                                         r"the embedding\.$"):
        mtb.evaluate(emb, labels=lab, clustering=lab[:30], metrics=["ARI"])
    np.save(tmp_path / "emb.npy", emb)
    pd.DataFrame({"x": lab}).to_csv(tmp_path / "cty.csv", index=False)
    pd.DataFrame({"x": lab[:30]}).to_csv(tmp_path / "batch.csv", index=False)
    code = cli.main(["evaluate", "--output", str(tmp_path / "emb.npy"), "--labels",
                     str(tmp_path / "cty.csv"), "--batch", str(tmp_path / "batch.csv"),
                     "--metrics", "all"])
    err = capsys.readouterr().err
    assert code == 1
    assert "error: --batch gave 30 labels (batch.csv) for 90 cells in --output." in err


def test_more_labels_than_cells_gives_no_per_batch_advice():
    emb, lab = _blobs()
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb[:60], labels=lab, metrics=["ASW"])
    assert str(e.value) == "labels has 90 entries for 60 cells in the embedding."


def test_cli_one_label_file_without_siblings_gives_the_general_fix(tmp_path, capsys):
    emb, lab = _blobs()
    np.save(tmp_path / "emb.npy", emb)
    pd.DataFrame({"x": lab[:30]}).to_csv(tmp_path / "cells.csv", index=False)
    assert cli.main(["evaluate", "--output", str(tmp_path / "emb.npy"),
                     "--labels", str(tmp_path / "cells.csv"), "--metrics", "ASW"]) == 1
    err = capsys.readouterr().err
    assert ("--labels gave 30 labels (cells.csv) for 90 cells in --output. For one "
            "label file per batch, repeat --labels in batch order, or pass --dataset, "
            "--category and --method.") in err


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
    assert re.fullmatch(
        r"Clustering 90 cells at 10 resolutions for ARI, NMI and iF1, with the "
        r"(igraph|leidenalg) Leiden backend\. This takes from seconds to a few "
        r"minutes\. To skip it, pass metrics= without ARI, NMI and iF1\.", err), err
    assert " - " not in err and ";" not in err and "per 3,000 cells" not in err


def test_sweep_note_cli_spelling(tmp_path, capsys, monkeypatch):
    from multibench.eval import scib as mscib
    monkeypatch.setattr(mscib, "_SWEEP_NOTICE_CELLS", 0)
    emb, lab = _blobs()
    np.save(tmp_path / "emb.npy", emb)
    pd.DataFrame({"x": lab}).to_csv(tmp_path / "cty.csv", index=False)
    assert cli.main(["evaluate", "--output", str(tmp_path / "emb.npy"), "--labels",
                     str(tmp_path / "cty.csv"), "--metrics", "ARI,NMI"]) == 0
    err = capsys.readouterr().err
    assert ("for ARI and NMI, with the") in err
    assert "To skip it, pass --clustering, or --metrics without ARI and NMI." in err


# --- R3-16 (b) ---------------------------------------------------------------

DARWIN = ("method environments are linux-64 only; this host is darwin/arm64")


def test_run_off_linux_is_the_linux_sentence_alone(tmp_path, monkeypatch):
    from multibench.engine import envs, runner
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=mtb.inputs_for("D11", "vertical", "totalVI"),
                out_dir=str(tmp_path / "out"))
    msg = str(e.value)
    assert msg == ("Methods run only on Linux (this computer is darwin/arm64). Run this "
                   "call on a Linux machine; dry_run=True previews the method's command "
                   "here.")
    assert "env install" not in msg and "is not installed" not in msg


def test_run_on_linux_keeps_the_install_line(tmp_path, monkeypatch):
    from multibench.engine import runner
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=mtb.inputs_for("D11", "vertical", "totalVI"),
                out_dir=str(tmp_path / "out"))
    msg = str(e.value)
    assert "is not installed" in msg and "multibench env install --methods totalVI" in msg
    assert "Methods run only on Linux" not in msg


def test_cli_column_keeps_the_label_file_name_in_the_error(tmp_path, capsys):
    emb, lab = _blobs()
    np.save(tmp_path / "emb.npy", emb)
    pd.DataFrame({"barcode": range(30), "celltype": lab[:30]}).to_csv(
        tmp_path / "meta.csv", index=False)
    assert cli.main(["evaluate", "--output", str(tmp_path / "emb.npy"), "--labels",
                     str(tmp_path / "meta.csv"), "--column", "celltype",
                     "--metrics", "ASW"]) == 1
    assert "--labels gave 30 labels (meta.csv) for 90 cells" in capsys.readouterr().err
