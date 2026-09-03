"""evaluate()'s vocabulary and file errors (Rin / Noor findings).

A typo in the family selector must be a ValueError listing the valid values,
a metric_set typo the same message config gives, `only=` must fold case like
every other metric argument, and a wrong or missing output path must name
the path, the working directory and - for a canonical INPUT matrix handed in
as an output - what the file actually is.
"""
import re

import h5py
import numpy as np
import pandas as pd
import pytest

from multibench.eval import evaluate, io as eio, pipeline


def _blobs(n_per=40, n_labels=3, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 8, size=(n_labels, dims))
    emb = np.vstack([centres[i] + rng.normal(0, 1, size=(n_per, dims)) for i in range(n_labels)])
    lab = np.array([f"t{i}" for i in range(n_labels) for _ in range(n_per)])
    return emb, lab


def test_family_typo_is_a_value_error_listing_valid_values():
    emb, lab = _blobs()
    with pytest.raises(ValueError, match=r"unknown family 'clusterin' \(given as task=\); valid: \['all', 'batch', 'clustering', 'dimension_reduction'\]"):
        evaluate(emb, labels=lab, task="clusterin")
    with pytest.raises(ValueError, match=r"unknown family 'clusterin' \(given as family=\)"):
        evaluate(emb, labels=lab, family="clusterin")
    # a declared benchmark task evaluate() cannot score is the one case that
    # stays NotImplementedError - and it says so without the word "v1"
    with pytest.raises(NotImplementedError, match="metrics for the 'imputation' task are not wired") as e:
        evaluate(emb, labels=lab, task="imputation")
    assert "v1" not in str(e.value)


def test_family_keyword_is_an_alias_of_task():
    emb, lab = _blobs()
    a = evaluate(emb, labels=lab, only={"ASW"})
    b = evaluate(emb, labels=lab, family="clustering", only={"ASW"})
    assert a.loc["ASW", "Value"] == pytest.approx(b.loc["ASW", "Value"])
    # family wins over the alias when both are given
    with pytest.raises(ValueError, match="batch labels required"):
        evaluate(emb, labels=lab, task="clustering", family="batch", only={"GC"})


def test_metric_set_typo_is_a_value_error_with_the_valid_list():
    emb, lab = _blobs()
    with pytest.raises(ValueError, match=r"unknown metric_set 'scibb'; valid: \['scib'\]"):
        evaluate(emb, labels=lab, metric_set="scibb")


def test_only_folds_case_like_the_other_metric_arguments():
    emb, lab = _blobs()
    out = evaluate(emb, labels=lab, only=["asw"])
    assert out.index.tolist() == ["ASW"]
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['nope'\]; choose from \['ARI'"):
        evaluate(emb, labels=lab, only=["asw", "nope"])


def test_missing_output_file_names_path_and_cwd(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"output nothere\.h5 does not exist \(cwd ") as e:
        evaluate("nothere.h5", labels=["a", "b"])
    assert "resolved" in str(e.value)
    with pytest.raises(FileNotFoundError, match="is a directory, not a file"):
        evaluate(str(tmp_path), labels=["a", "b"])
    with pytest.raises(FileNotFoundError, match=r"labels file .*nothere\.csv does not exist \(cwd"):
        eio.read_labels(tmp_path / "nothere.csv")
    with pytest.raises(FileNotFoundError, match="clustering file .* does not exist"):
        eio.read_clustering(tmp_path / "nothere.h5")


def test_input_matrix_passed_as_output_is_explained(tmp_path):
    f = tmp_path / "rna.h5"
    with h5py.File(f, "w") as h:
        g = h.create_group("matrix")
        g.create_dataset("data", data=np.zeros((3, 4)))
        g.create_dataset("barcodes", data=np.array([b"a", b"b", b"c"]))
        g.create_dataset("features", data=np.array([b"g1", b"g2", b"g3", b"g4"]))
    with pytest.raises(ValueError, match=r"has no dataset 'data'; found keys \['matrix'\]") as e:
        evaluate(str(f), labels=["a", "b", "c"])
    msg = str(e.value)
    assert "looks like a canonical INPUT matrix (matrix/data" in msg
    assert "not an embedding" in msg and "out/<method>/embedding.h5" in msg
    # any other h5 without 'data' lists its keys, without the input hint
    g2 = tmp_path / "other.h5"
    with h5py.File(g2, "w") as h:
        h.create_dataset("X", data=np.zeros((3, 4)))
    with pytest.raises(ValueError, match=r"found keys \['X'\]") as e2:
        eio.read_embedding(g2)
    assert "INPUT matrix" not in str(e2.value)
    with pytest.raises(ValueError, match="/obs/cluster_leiden"):
        eio.read_clustering(g2)


def test_verbose_sweep_notice(capsys):
    emb, lab = _blobs()
    evaluate(emb, labels=lab, only={"ARI"}, verbose=True)
    err = capsys.readouterr().err
    assert re.search(r"Leiden resolution sweep \(10 resolutions\) over 120 cells for ARI", err)
    assert "pass clustering= or only=" in err
    evaluate(emb, labels=lab, only={"ARI"})            # 120 cells: auto-quiet
    assert "Leiden" not in capsys.readouterr().err
    evaluate(emb, labels=lab, only={"ARI"}, verbose=False)
    assert "Leiden" not in capsys.readouterr().err
    evaluate(emb, labels=lab, only={"ASW"}, verbose=True)     # no sweep needed
    assert "Leiden" not in capsys.readouterr().err
    from multibench.eval import scib as mscib
    assert mscib._SWEEP_NOTICE_CELLS == 2000


def test_docstrings_match_the_code():
    doc = " ".join(evaluate.__doc__.split())
    assert "goes in AS IS when its insertion order is the method's stacking order" in doc
    assert "a dict in ANY OTHER order needs ``label_order=``" in doc
    assert "Leiden sweep" in doc and "minutes for ~10^4" in doc
    assert "v1" not in doc
    assert "verbose" in doc and "2,000 cells" in doc
    import multibench.plot as mplot
    assert "BatchResult.long()" not in mplot.bubble.__doc__
    assert ":meth:`BatchResult.long`" not in mplot.bar.__doc__
    assert "``BatchResult.long`` property" in mplot.bubble.__doc__
    from multibench.data import results
    assert "v1" not in results.__doc__ and "v1" not in results.load_results.__doc__
    assert "Parameters" in pipeline.to_long.__doc__
    assert pipeline.to_long.__doc__.strip().splitlines()[0] == \
        "Reshape :func:`evaluate`'s wide frame into the tidy long results frame."
