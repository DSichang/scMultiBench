import numpy as np
import pandas as pd
import h5py
from multibench.eval import io


def test_read_embedding_autoorients_to_cells_by_dims(tmp_path):
    # store as dims x cells (4 x 10) -> reader returns 10 x 4.
    # Use distinct values so we verify the data is correctly transposed,
    # not merely reshaped: stored[d, c] must land at result[c, d].
    p = tmp_path / "embedding.h5"
    stored = np.arange(40, dtype=float).reshape(4, 10)  # (dims=4, cells=10)
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=stored)
    X = io.read_embedding(p)
    assert X.shape == (10, 4)
    np.testing.assert_array_equal(X, stored.T)
    # cell 0 should hold the dim-0..dim-3 values from column 0 of the source
    np.testing.assert_array_equal(X[0], stored[:, 0])
    np.testing.assert_array_equal(X[9], stored[:, 9])


def test_read_embedding_keeps_cells_by_dims(tmp_path):
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((100, 8)))
    assert io.read_embedding(p).shape == (100, 8)


def test_read_labels_skips_header_and_codes(tmp_path):
    # header-aware reader: the first row is the header, the values come back
    # as written (raw strings) - every consumer casts to str anyway
    p = tmp_path / "cty.csv"
    pd.DataFrame(["celltype", "B", "T", "B", "NK"]).to_csv(p, index=False, header=False)
    labels = io.read_labels(p)
    assert len(labels) == 4              # header row dropped
    assert labels[0] == labels[2]        # both "B"
    assert labels[1] != labels[0]
    assert list(labels) == ["B", "T", "B", "NK"]


def test_read_clustering_decodes_bytes(tmp_path):
    p = tmp_path / "cluster.h5"
    with h5py.File(p, "w") as f:
        grp = f.create_group("obs")
        grp.create_dataset("cluster_leiden", data=np.array([b"0", b"1", b"0"]))
    c = io.read_clustering(p)
    assert list(c) == [0, 1, 0]


def test_evaluate_rejects_non_scib():
    import pytest
    from multibench.eval import pipeline
    with pytest.raises(ValueError) as exc:
        pipeline.evaluate(embedding="x.h5", category="mosaic", metrics="dimension_reduction",
                          labels="c.csv")
    assert "Unknown metrics= token dimension_reduction" in str(exc.value)


def test_evaluate_requires_labels_and_clustering(tmp_path):
    import pytest
    from multibench.eval import pipeline
    with pytest.raises(ValueError):
        pipeline.evaluate(embedding="x.h5", category="vertical", metrics="clustering",
                          labels=None)


def test_evaluate_requires_batch_for_batch_task():
    import pytest
    from multibench.eval import pipeline
    emb = np.zeros((6, 3))
    cl = np.array([0, 0, 1, 1, 2, 2])
    ct = cl.copy()
    with pytest.raises(ValueError) as exc:
        pipeline.evaluate(embedding=emb, category="vertical", metrics="batch",
                          labels=ct, batch=None)
    assert "metrics='batch' needs batch labels" in str(exc.value)


def test_compute_rejects_length_mismatch():
    import pytest
    anndata = pytest.importorskip("anndata")
    from multibench.eval import scib as escib
    emb = np.zeros((6, 3))
    ct = np.array([0, 0, 1, 1, 2])      # one short -> mismatch
    cl = np.array([0, 0, 1, 1, 2, 2])
    ba = np.array([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError) as exc:
        escib.compute(emb, ct, cl, ba, group="clustering")
    # R3-14: named after evaluate's argument, with the two counts
    assert str(exc.value) == "labels has 5 entries for 6 cells in the embedding."


def test_compute_rejects_unknown_group():
    import pytest
    anndata = pytest.importorskip("anndata")
    from multibench.eval import scib as escib
    emb = np.zeros((4, 2))
    lab = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError) as exc:
        escib.compute(emb, lab, lab, lab, group="bogus")
    assert "unknown group" in str(exc.value)


def test_top_level_exposes_evaluate():
    import multibench as mtb
    assert hasattr(mtb, "evaluate")


# ---------------------------------------------------------------- P03: coercion
# evaluate() used to accept exactly an ndarray or an h5/CSV path; a pandas
# Series, a Categorical, labels_for()'s dict, a DataFrame embedding, an .npy
# file ... all failed deep inside pandas/h5py with messages that named neither
# the argument nor the fix, and an obs-style CSV (barcode index + label column)
# silently scored ARI 0.0. These pin the coercion layer and its error messages.
import warnings
from pathlib import Path

import pytest


def _toy(n_per=45, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.1, size=(n_per, 5))
    b = rng.normal(5, 0.1, size=(n_per, 5))
    emb = np.vstack([a, b])
    ct = np.array(["B"] * n_per + ["T"] * n_per)
    bat = np.array(["s1", "s2"] * n_per)
    return emb, ct, bat


def _value(df, metric):
    """One value of evaluate's long table."""
    return float(df.set_index("metric").loc[metric, "value"])


def _ari(output, **kw):
    from multibench.eval.pipeline import evaluate
    return _value(evaluate(output, metrics=["ARI"], **kw), "ARI")


def test_evaluate_accepts_series_categorical_list_dataframe_labels():
    pytest.importorskip("scib")
    emb, ct, _ = _toy()
    base = _ari(output=emb, labels=ct)
    assert base == pytest.approx(1.0)
    for lab in (pd.Series(ct), pd.Categorical(ct), list(ct), pd.Index(ct),
                pd.DataFrame({"celltype": ct}), ct.reshape(-1, 1),
                pd.Series(pd.Categorical(ct), index=[f"c{i}" for i in range(len(ct))])):
        assert _ari(output=emb, labels=lab) == base, type(lab).__name__


def test_as_vector_rejects_unsupported_types_and_wide_frames():
    with pytest.raises(TypeError) as exc:
        io.as_vector(3.5, what="labels")
    assert "labels must be a CSV path, list of paths, or 1-D array-like" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        io.as_vector(np.zeros((4, 3)), what="batch")
    assert "batch must be 1-D" in str(exc.value)
    df = pd.DataFrame({"celltype": ["a", "b"], "batch": ["x", "y"]})
    with pytest.raises(ValueError) as exc:
        io.as_vector(df, what="labels")
    assert "2 columns" in str(exc.value) and "column=" in str(exc.value)
    assert list(io.as_vector(df, what="labels", column="batch")) == ["x", "y"]


def test_evaluate_accepts_dataframe_anndata_npy_csv_h5ad_output(tmp_path):
    pytest.importorskip("scib")
    ad = pytest.importorskip("anndata")
    emb, ct, _ = _toy()
    base = _ari(output=emb, labels=ct)
    # in-memory forms
    assert _ari(output=pd.DataFrame(emb), labels=ct) == base
    adata = ad.AnnData(np.zeros((len(ct), 2), dtype=np.float32))
    adata.obsm["X_emb"] = emb
    adata.obsm["X_pca"] = emb.copy()
    assert _ari(output=adata, labels=ct) == base
    assert _ari(output=adata, labels=ct, obsm="X_pca") == base
    adata_x = ad.AnnData(emb.copy())
    assert _ari(output=adata_x, labels=ct, obsm="X") == base
    # files
    npy = tmp_path / "emb.npy"; np.save(npy, emb)
    assert _ari(output=npy, labels=ct) == base
    assert _ari(output=str(npy), labels=ct) == base
    csv = tmp_path / "emb.csv"; pd.DataFrame(emb).to_csv(csv)            # with index
    assert _ari(output=csv, labels=ct) == base
    csv2 = tmp_path / "emb2.csv"; pd.DataFrame(emb).to_csv(csv2, index=False)
    assert _ari(output=csv2, labels=ct) == base
    csv3 = tmp_path / "emb3.csv"
    pd.DataFrame(emb, index=[f"AAAC-{i}" for i in range(len(ct))]).to_csv(csv3)
    assert _ari(output=csv3, labels=ct) == base
    h5ad = tmp_path / "emb.h5ad"; adata.write_h5ad(h5ad)
    assert _ari(output=h5ad, labels=ct) == base
    h5 = tmp_path / "embedding.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("data", data=emb.T)                            # dims x cells
    assert _ari(output=h5, labels=ct) == base


def test_as_matrix_errors_name_the_accepted_forms(tmp_path):
    with pytest.raises(TypeError) as exc:
        io.as_matrix(object())
    assert "output must be a (cells x dims) ndarray/DataFrame/AnnData or a path" in str(exc.value)
    bad = tmp_path / "emb.xyz"; bad.write_text("1,2\n3,4\n")
    with pytest.raises(ValueError) as exc:
        io.as_matrix(bad)
    assert "unrecognised suffix" in str(exc.value) and ".npy" in str(exc.value)


def test_as_matrix_anndata_missing_obsm_lists_keys():
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.zeros((3, 2)))
    a.obsm["X_pca"] = np.zeros((3, 2))
    with pytest.raises(ValueError) as exc:
        io.as_matrix(a, obsm="X_emb")
    assert "has no obsm['X_emb']" in str(exc.value) and "X_pca" in str(exc.value)


def test_evaluate_anndata_labels_as_obs_key():
    pytest.importorskip("scib")
    ad = pytest.importorskip("anndata")
    from multibench.eval.pipeline import evaluate
    emb, ct, bat = _toy()
    a = ad.AnnData(np.zeros((len(ct), 2)))
    a.obsm["X_emb"] = emb
    a.obs["celltype"] = pd.Categorical(ct)
    a.obs["sample"] = bat
    df = evaluate(a, labels="celltype", batch="sample", metrics=["ARI", "GC"])
    assert _value(df, "ARI") == pytest.approx(1.0)
    assert _value(df, "GC") == pytest.approx(1.0)
    with pytest.raises(ValueError) as exc:
        evaluate(a, labels="cell_type", metrics=["ARI"])
    assert "neither an obs column" in str(exc.value) and "celltype" in str(exc.value)


def _write_cty(path, values):
    pd.DataFrame({"x": values}).to_csv(path, index=False)


def test_evaluate_list_of_label_paths_concatenates_in_order_and_builds_batch(tmp_path):
    pytest.importorskip("scib")
    from multibench.eval.pipeline import evaluate
    emb, ct, _ = _toy()
    # split the cells into two "batches" (files) of unequal size
    p1, p2 = tmp_path / "cty1.csv", tmp_path / "cty2.csv"
    _write_cty(p1, ct[:30]); _write_cty(p2, ct[30:])
    got = io.as_vector([p1, p2], what="labels")
    assert list(got) == list(ct)
    assert list(io.as_vector([p2, p1], what="labels")) == list(ct[30:]) + list(ct[:30])
    # batch=None with several files -> batch = file of origin (1, 2), like run_all
    df = evaluate(emb, labels=[p1, str(p2)], metrics=["ARI", "GC", "ASW_batch"])
    ref = evaluate(emb, labels=ct, batch=np.array([1] * 30 + [2] * 60),
                   metrics=["ARI", "GC", "ASW_batch"])
    pd.testing.assert_frame_equal(df.sort_index(), ref.sort_index())
    # a single file in a list still needs an explicit batch for metrics='all'
    with pytest.raises(ValueError) as exc:
        evaluate(emb, metrics="all", labels=[tmp_path / "all.csv"])
    assert "metrics='all' needs batch labels" in str(exc.value)


def test_evaluate_dict_with_several_label_files_raises_listing_keys(tmp_path):
    pytest.importorskip("scib")
    from multibench.eval.pipeline import evaluate
    emb, ct, _ = _toy()
    p = tmp_path / "cty.csv"; _write_cty(p, ct)
    # single-entry dict (labels_for() on a one-label dataset) plugs straight in
    assert _ari(output=emb, labels={"cty": str(p)}) == pytest.approx(1.0)
    # a dict whose insertion order IS the stacking order (what labels_for
    # returns) is accepted as-is: the two files are concatenated in that order
    # (here both hold the same 90 cells, so the only complaint is the length)
    # the first files that hold the embedding's cells are named (0.4.0, X68), by
    # their file names
    p1, p2 = tmp_path / "cty1.csv", tmp_path / "cty2.csv"
    _write_cty(p1, ct); _write_cty(p2, ct)
    with pytest.raises(ValueError, match=r"The embedding has 90 cells = cty1\. Pass the "
                                         r"label files cty1\.csv only\.$"):
        evaluate(emb, labels={"cty1": str(p1), "cty2": str(p2)}, metrics=["ARI"])
    # any other order must be explicit
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels={"cty2": str(p), "cty1": str(p)}, metrics=["ARI"])
    msg = str(exc.value)
    assert "The label keys cty2 and cty1 " in msg
    assert "list of paths in cell order" in msg
    # 0.3.x: the hint must no longer recommend the alphabetical order; it names
    # the cell-order rule and (0.4.0) dataset=, category= and method=, not the
    # deprecated label_order=
    assert "dataset=, category= and method=" in msg and "are not in the default order" in msg
    assert "label_order=" not in msg
    assert "not alphabetical" in msg
    assert "list(d.values())" not in msg


def test_read_labels_obs_style_csv_uses_label_column_not_barcode(tmp_path):
    pytest.importorskip("scib")
    emb, ct, _ = _toy()
    p = tmp_path / "obs.csv"
    barcodes = [f"AAACCTG-{i}" for i in range(len(ct))]
    pd.DataFrame({"celltype": ct}, index=barcodes).to_csv(p)
    assert list(io.read_labels(p)) == list(ct)
    assert _ari(output=emb, labels=p) == pytest.approx(1.0)      # was 0.0
    # a written RangeIndex ("Unnamed: 0") is index-like too
    p2 = tmp_path / "obs2.csv"
    pd.Series(ct, name="celltype").to_csv(p2)
    assert list(io.read_labels(p2)) == list(ct)
    # R's write.csv(x) layout: row numbers + "x"
    p3 = tmp_path / "r.csv"
    pd.DataFrame({"": range(1, len(ct) + 1), "x": ct}).to_csv(p3, index=False)
    assert list(io.read_labels(p3)) == list(ct)


def test_read_labels_ambiguous_multicolumn_raises_unless_column_given(tmp_path):
    ct = ["B", "T", "B", "NK"]
    p = tmp_path / "meta.csv"
    pd.DataFrame({"celltype": ct, "batch": ["a", "a", "b", "b"]}).to_csv(p, index=False)
    with pytest.raises(ValueError) as exc:
        io.read_labels(p)
    msg = str(exc.value)
    assert "several columns: celltype, batch" in msg and "Pass column=<name>" in msg
    assert list(io.read_labels(p, column="celltype")) == ct
    assert list(io.read_labels(p, column="batch")) == ["a", "a", "b", "b"]
    # barcode index + two columns: ambiguous, and the message lists the
    # columns after the barcodes, without pandas' 'Unnamed: 0'
    p2 = tmp_path / "meta2.csv"
    pd.DataFrame({"celltype": ct, "batch": ["a", "a", "b", "b"]},
                 index=[f"bc{i}" for i in range(4)]).to_csv(p2)
    with pytest.raises(ValueError) as exc:
        io.read_labels(p2)
    assert "after the cell ids: celltype, batch" in str(exc.value)
    assert "Unnamed" not in str(exc.value)
    assert list(io.read_labels(p2, column="celltype")) == ct
    with pytest.raises(ValueError) as exc:
        io.read_labels(p2, column="nope")
    assert "no column named 'nope'" in str(exc.value) and "celltype" in str(exc.value)
    # evaluate() takes the column itself (its column= keyword is gone)
    pytest.importorskip("scib")
    emb, ct90, _ = _toy()
    p3 = tmp_path / "meta3.csv"
    pd.DataFrame({"celltype": ct90, "other": ct90[::-1]}).to_csv(p3, index=False)
    assert _ari(output=emb, labels=pd.read_csv(p3)["celltype"]) == pytest.approx(1.0)


def test_read_labels_tsv_and_numeric_labels(tmp_path):
    p = tmp_path / "cty.tsv"
    pd.DataFrame({"x": [1, 2, 1]}).to_csv(p, sep="\t", index=False)
    assert list(io.read_labels(p)) == [1, 2, 1]


def test_evaluate_metrics_rejects_unknown_metric_and_bare_code_string():
    from multibench.eval.pipeline import evaluate
    emb, ct, _ = _toy()
    with pytest.raises(ValueError) as exc:
        evaluate(emb, labels=ct, metrics=["ARI", "XYZ"])
    msg = str(exc.value)
    # c1 (C36): one text for an unknown token, alone or in a list
    assert msg.startswith("Unknown metrics= token XYZ. Pass 'all', 'clustering'")
    # 0.4.0: one code is accepted as it is, as in a list
    pd.testing.assert_frame_equal(evaluate(emb, labels=ct, metrics="ARI"),
                                  evaluate(emb, labels=ct, metrics=["ARI"]))


def test_evaluate_rejects_unknown_category_and_allows_none():
    pytest.importorskip("scib")
    from multibench.eval.pipeline import evaluate
    emb, ct, _ = _toy()
    with pytest.raises(ValueError) as exc:
        evaluate(emb, category="bogus", labels=ct, metrics=["ARI"])
    msg = str(exc.value)
    assert "unknown category 'bogus'" in msg and "valid:" in msg and "vertical" in msg
    assert _value(evaluate(emb, labels=ct, metrics=["ARI"]), "ARI") == pytest.approx(1.0)
    assert _value(evaluate(emb, ct, category="vertical", metrics=["ARI"]), "ARI") == \
        pytest.approx(1.0)


def test_evaluate_warns_on_transpose():
    pytest.importorskip("scib")
    from multibench.eval.pipeline import evaluate
    emb, ct, _ = _toy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        evaluate(emb.T, labels=ct, metrics=["ASW"])
    hits = [str(x.message) for x in w if "transposed" in str(x.message)]
    assert hits and "dims x cells" in hits[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        evaluate(emb, labels=ct, metrics=["ASW"])
    assert not [x for x in w if "transposed" in str(x.message)]


def test_read_labels_reads_benchmark_style_files(tmp_path):
    """eval.io.read_labels is the label reader of run_all and evaluate() alike; it
    reads the benchmark's cty layout (single column, header "x")."""
    p = tmp_path / "cty.csv"
    vals = ["mDC.Lung", "T.CD.EM", "B", "T.CD.EM"]
    _write_cty(p, vals)
    assert list(io.read_labels(p)) == vals
    # and the shipped reference datasets, when present
    root = Path(__file__).resolve().parent.parent / "data"
    for f in sorted(root.glob("D*/*cty*.csv")) if root.is_dir() else []:
        if "scjoint" in f.name.lower():
            continue
        io.read_labels(f)


def test_align_vector_matches_as_evaluate_does():
    """The docstring says evaluate() matches the same way: a Series by its ids."""
    from multibench.eval import pipeline
    ids = pd.Index([f"c{i}" for i in range(6)])
    x = pd.Series([f"t{i}" for i in range(6)], index=ids).iloc[[3, 0, 5, 1, 4, 2]]
    mine = io.align_vector(x, ids)
    assert list(mine) == [f"t{i}" for i in range(6)]
    assert list(pipeline._obs_or_vector(x, None, what="labels", ids=ids)) == list(mine)
    assert "calls this" not in io.align_vector.__doc__
