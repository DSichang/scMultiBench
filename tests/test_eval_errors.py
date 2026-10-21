"""evaluate()'s vocabulary and file errors (Rin / Noor findings).

A typo in the metrics= token must be a ValueError listing the valid values,
a list of codes must fold case like every other metric argument, and a wrong or missing output path must name
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


def test_metrics_token_typo_is_a_value_error_listing_valid_values():
    emb, lab = _blobs()
    with pytest.raises(ValueError, match=r"Unknown metrics= token clusterin\. Pass 'all', "
                                         r"'clustering', 'batch' or 'classification'"):
        evaluate(emb, labels=lab, metrics="clusterin")
    with pytest.raises(ValueError, match=r"Unknown metrics= token dimension_reduction\.") as e:
        evaluate(emb, labels=lab, metrics="dimension_reduction")
    assert "v1" not in str(e.value)


def test_metrics_family_token_and_code_list_agree():
    emb, lab = _blobs()
    a = evaluate(emb, labels=lab, metrics=["ASW"]).set_index("metric")["value"]
    b = evaluate(emb, labels=lab, metrics="clustering").set_index("metric")["value"]
    assert a["ASW"] == pytest.approx(b["ASW"])
    # the batch family needs batch labels
    with pytest.raises(ValueError, match="metrics='batch' needs batch labels"):
        evaluate(emb, labels=lab, metrics="batch")


def test_metrics_folds_case_like_the_other_metric_arguments():
    emb, lab = _blobs()
    out = evaluate(emb, labels=lab, metrics=["asw"])
    assert out["metric"].tolist() == ["ASW"]
    # c1 (C36): one text for an unknown token, alone or in a list
    with pytest.raises(ValueError, match=r"^Unknown metrics= token nope\. Pass 'all'"):
        evaluate(emb, labels=lab, metrics=["asw", "nope"])


def test_missing_output_file_names_path_and_cwd(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"^The output nothere\.h5 does not exist\. "
                       r"From the working directory ") as e:
        evaluate("nothere.h5", labels=["a", "b"])
    assert "the path resolves to " in str(e.value)
    with pytest.raises(FileNotFoundError, match="is a folder, not a file"):
        evaluate(str(tmp_path), labels=["a", "b"])
    # an absolute path needs no working directory
    with pytest.raises(FileNotFoundError, match=r"^The labels file .*nothere\.csv does not exist\.$"):
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
    assert "looks like a canonical input matrix (matrix/data" in msg
    assert "not an embedding" in msg and "out/<method>/embedding.h5" in msg
    # any other h5 without 'data' lists its keys, without the input hint
    g2 = tmp_path / "other.h5"
    with h5py.File(g2, "w") as h:
        h.create_dataset("X", data=np.zeros((3, 4)))
    with pytest.raises(ValueError, match=r"found keys \['X'\]") as e2:
        eio.read_embedding(g2)
    assert "input matrix" not in str(e2.value)
    with pytest.raises(ValueError, match="/obs/cluster_leiden"):
        eio.read_clustering(g2)


def test_docstrings_match_the_code():
    doc = " ".join(evaluate.__doc__.split())
    assert "v1" not in doc
    assert "verbose" in doc and "2,000 cells" in doc
    import multibench.plot as mplot
    assert "BatchResult.long()" not in mplot.bubble.__doc__
    assert ":meth:`BatchResult.long`" not in mplot.bar.__doc__
    # 0.4.0 docstring: "as mtb.load_results, mtb.evaluate and BatchResult.long return"
    assert "``mtb.evaluate`` and ``BatchResult.long`` return" in \
        " ".join(mplot.bubble.__doc__.split())
    from multibench.data import results
    assert "v1" not in results.__doc__ and "v1" not in results.load_results.__doc__
    # to_long is deprecated since 0.4: evaluate returns the long table
    assert pipeline.to_long.__doc__.strip() == \
        "Deprecated: ``mtb.evaluate(..., method=, dataset=)`` returns the long table."


def test_a_wrong_batch_column_gets_its_error_without_the_unused_batch_warning(tmp_path):
    # Z23 (r3): the batch is read before the "changes nothing" warning
    import warnings
    import anndata as ad
    import numpy as np
    import pandas as pd
    import multibench as mtb
    rng = np.random.default_rng(0)
    a = ad.AnnData(np.zeros((40, 3)), obs=pd.DataFrame({"celltype": ["A", "B"] * 20},
                                                      index=[f"c{i}" for i in range(40)]))
    a.obsm["X_emb"] = rng.normal(size=(40, 4))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="batch='Donor' is neither an obs column"):
            mtb.evaluate(a, labels="celltype", batch="Donor", metrics=["ARI"])
    assert not [w for w in rec if "changes nothing" in str(w.message)]
