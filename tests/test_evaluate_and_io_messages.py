"""evaluate, export_dataset and the label files: alignment, columns, warnings."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry  # noqa: F401


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _d11():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results("vertical", dataset="D11")


def _mine(base, name="Mine", scored_with="leidenalg/search/0.4.0"):
    rows = base[base.method == "Matilda"].assign(method=name, source="user")
    if scored_with is not None:
        rows = rows.assign(scored_with=scored_with)
    return rows


MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.engine.runner.run(method, category, inputs=inputs, out_dir=str(tmp_path / "o"),
                   params=params, dry_run=True, cmd_template="conda run -n x {cmd}")


# --- evaluate, io, catalog lookups: X09, X10, X12, X36, X37, X47, X49, X50,
#     X58, X61, X62, X67, X71, X72, X74, X76, X83, X90, X91 ----------------

def _adata(n=60, names=True):
    import anndata as ad
    import numpy as np
    a = ad.AnnData(np.random.default_rng(0).poisson(1, (n, 20)).astype(float))
    if names:
        a.obs_names = [f"AAAC{i}-1" for i in range(n)]
    return a


def test_x09_repeated_barcodes_are_refused(tmp_path):
    a = _adata()
    a.obs_names = [f"AAAC{i % 30}-1" for i in range(60)]
    with pytest.raises(ValueError, match=r"^30 barcodes repeat, e\.g\. AAAC0-1\. Make them "
                                         r"unique \(prefix the sample\)\.$"):
        mtb.io.export_dataset(a, tmp_path / "DUP", category="vertical")
    assert not (tmp_path / "DUP" / "rna.h5").exists()


def test_x10_no_shared_barcode_names_both_spellings(tmp_path):
    rna = _adata()
    atac = _adata()
    atac.obs_names = [f"AAAC{i}" for i in range(60)]
    atac.var_names = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(20)]
    # round 2 (X10): names that match after removing a suffix say so
    with pytest.raises(ValueError, match=r"^RNA and ATAC share no barcode \(RNA e\.g\. "
                                         r"AAAC0-1, ATAC e\.g\. AAAC0\)\. They match after "
                                         r"removing '-1' from the RNA names\.$"):
        mtb.io.export_dataset(rna, tmp_path / "V", atac=atac, category="vertical")


def test_x12_a_label_file_without_ids_is_matched_by_position_silently(tmp_path):
    # c2 (D10): the documented rule, with nothing to do: no warning; c3 (E05): only
    # where the embedding has no cell ids (next to barcodes it is refused)
    import numpy as np
    a = _adata()
    a.obsm["X_pca"] = np.random.default_rng(1).normal(size=(60, 4))
    lab = tmp_path / "cty.csv"
    pd.DataFrame({"x": ["a", "b", "c"] * 20}).to_csv(lab, index=False)
    with pytest.raises(ValueError, match="^cty.csv has no cell ids, and the embedding has"):
        mtb.evaluate(a, labels=str(lab), metrics=["cLISI"], obsm="X_pca")
    b = _adata(names=False)
    b.obsm["X_pca"] = a.obsm["X_pca"]
    _, msgs = _messages(mtb.evaluate, b, labels=str(lab), metrics=["cLISI"], obsm="X_pca")
    assert not any("matched by position" in m for m in msgs)


def test_x47_text_returns_echo_as_text():
    for text in (mtb.describe_layout("vertical"), mtb.cite(["Matilda"])):
        assert isinstance(text, str) and repr(text) == str(text) and "\n" in text


def test_x49_obsm_none_reads_x_emb_only():
    # c1 (C02): never "the only key"; the message names the keys
    import numpy as np
    a = _adata()
    a.obsm["X_pca"] = np.random.default_rng(2).normal(size=(60, 4))
    # c2 (D08): one key is named; X only when the AnnData has one
    for keys, pick in (("X_pca", "obsm='X_pca'"), ("X_pca, X_umap", "obsm= with one of them")):
        with pytest.raises(ValueError, match=rf"^The AnnData has no obsm\['X_emb'\]; its obsm "
                                             rf"keys are {keys}: pass {pick} or obsm='X' for \.X\.$"):
            mtb.evaluate(a, labels=np.repeat(["a", "b", "c"], 20), metrics=["cLISI"])
        a.obsm["X_umap"] = a.obsm["X_pca"][:, :2]


def test_x50_x71_classification_docs_and_left_out_cells():
    # 0.4.0: mtb.classify and mtb.classification_metrics are deprecated wrappers
    # ("Deprecated: use ..."); the implementation keeps these docs
    from multibench.eval import classify as C
    doc = " ".join(C.classification_metrics.__doc__.split())
    assert ('The ``mean`` row is what ``mtb.evaluate(metrics="classification")`` '
            "returns; this adds one row per reference.") in doc
    import numpy as np
    emb = np.random.default_rng(3).normal(size=(90, 4))
    labels = np.array(["a", "b", "c"] * 15 + ["a", "b"] * 22 + ["a"])
    batch = np.array(["1"] * 45 + ["2"] * 45)
    pred = C.classify(emb, labels=labels, batch=batch)
    # batch 2 lacks type c: the 15 c cells of batch 1 are left out with 2 as reference
    assert pred.attrs["left_out"] == {"1": 0, "2": 15}
    assert "lacks is left out" in " ".join(C.classify.__doc__.split())


def test_x67_labels_for_names_the_dataset_category_once():
    _, msgs = _messages(mtb.engine.resolve.labels_for, "D52mini")
    hits = [m for m in msgs if "default order" in m]
    assert len(hits) == 1 and "In cross, StabMap" in hits[0] and "mosaic" not in hits[0]
    _, msgs = _messages(mtb.engine.resolve.labels_for, "D52mini")
    assert not [m for m in msgs if "default order" in m]          # once a session


def test_x72_batch_leiden_warning_says_what_it_affects(monkeypatch):
    import numpy as np
    from multibench.eval import scib as S
    monkeypatch.setattr(S, "_search_n_clusters", lambda adata, n, key, flavor: (
        adata.obs.__setitem__(key, pd.Categorical(["a"] * 4 + ["b"] * (adata.n_obs - 4)))
        or False))
    # Y11 (r3): the warning went (nothing to do); the last step's clusters are used
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        got = S._n_clusters(S._clustering_graph(np.random.default_rng(4).normal(size=(40, 5))),
                            3, "leidenalg", "batch")
    assert sorted(set(got)) == ["a", "b"]


def test_x74_an_h5mu_path_names_the_fix(tmp_path):
    p = tmp_path / "x.h5mu"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match=r"x\.h5mu is an \.h5mu: pass one modality, e\.g\. "
                                         r"mudata\.read\(path\)\['rna'\]"):
        mtb.evaluate(str(p), labels=["a"] * 3)


def test_x76_tokens_ignore_letter_case():
    import numpy as np
    plan = mtb.run("D11", "vertical", modalities=["RNA", "ADT"], dry_run=True,
                       verbose=False)
    assert len(plan)
    emb = np.random.default_rng(5).normal(size=(60, 4))
    got = mtb.evaluate(emb, labels=np.repeat(["a", "b", "c"], 20), metrics="Clustering")
    assert "ARI" in set(got["metric"])


def test_x83_read_canonical_reads_an_embedding_file(tmp_path):
    import h5py
    import numpy as np
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f["data"] = np.arange(30.0).reshape(10, 3)
    a = mtb.io.read_canonical(p)
    assert a.n_obs == 10 and a.obsm["X_emb"].shape == (10, 3)


def test_x90_x91_hints_for_an_assay_and_a_prefix():
    with pytest.raises(ValueError, match=r"CITE-seq and 10x Multiome are vertical\.$"):
        mtb.find_methods(category="CITE-seq")
    with pytest.raises(KeyError) as e:
        mtb.method_info("Seurat")
    assert all(s in str(e.value) for s in ("Seurat_v3", "Seurat_v5", "Seurat_WNN"))


# X67: labels_for without a method warns once per dataset and category
def test_default_order_warning_once():
    import warnings
    from multibench.engine import resolve
    resolve._ORDER_NOTED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mtb.engine.resolve.labels_for("D52mini", "cross")
        mtb.engine.resolve.labels_for("D52mini", "cross")
    assert len([x for x in w if "default order" in str(x.message)]) == 1


# X10: barcodes that match after removing a suffix say so; vertical drops the diagonal hint
def test_export_barcodes_that_differ_by_a_suffix():
    import numpy as np
    ad = pytest.importorskip("anndata")
    from test_study_r2_rr import _peak_adata, _rna_adata
    rna = _rna_adata([f"AAAC{i:04d}-1" for i in range(30)])
    atac = _peak_adata([f"AAAC{i:04d}" for i in range(30)])
    with pytest.raises(ValueError) as e:
        mtb.io.export_dataset(rna, "/nonexistent/SUF", atac=atac, atac_kind="peak",
                              category="vertical")
    assert str(e.value) == ("RNA and ATAC share no barcode (RNA e.g. AAAC0000-1, ATAC e.g. "
                            "AAAC0000). They match after removing '-1' from the RNA names.")
    with pytest.raises(ValueError, match=r"\. Make the names match\.$"):
        mtb.io.export_dataset(rna, "/nonexistent/SUF", atac=_peak_adata(
            [f"x{i}" for i in range(30)]), atac_kind="peak", category="vertical")


# Y17: export_dataset given a path says to read the file first
def test_export_dataset_given_a_path():
    with pytest.raises(TypeError, match=r"^export_dataset takes an AnnData or MuData "
                                        r"object, not a path: read the file first"):
        mtb.io.export_dataset("own/citeseq.h5ad", "/nonexistent/CITEP", adt="obsm:protein")


# Y18: one file "has", once per file
# Y19: a label column that differs in letter case gets a did-you-mean
def test_label_column_case_did_you_mean(tmp_path):
    import numpy as np
    import pandas as pd
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((4, 3)), obs=pd.DataFrame({"celltype": list("abab")},
                                                     index=[f"c{i}" for i in range(4)]))
    with pytest.raises(KeyError, match=r"Did you mean 'celltype'\?"):
        mtb.io.export_dataset(a, tmp_path / "A", labels="obs:Celltype")
    with pytest.raises(ValueError, match=r"Did you mean 'celltype'\?$"):
        mtb.io.export_dataset(a, tmp_path / "B", labels="Celltype")


# Y11: in rescore, the Leiden search warning names its method
def test_rescore_leiden_warning_names_the_method(monkeypatch):
    import numpy as np
    from multibench import workflow
    from multibench.eval import scib

    class G:
        obs = {}
    monkeypatch.setattr(scib, "_search_n_clusters",
                        lambda g, n, key, flavor: g.obs.update({key: _Ser()}) or False)
    # Y11 (r3): no warning at all, so nothing names a method
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        scib._n_clusters(G(), 13, "leidenalg", per="cell type")
    assert not hasattr(workflow, "_named_leiden_warnings") and not hasattr(scib, "FOR_METHOD")


class _Ser:
    def nunique(self):
        return 12

    def astype(self, t):
        return ["0"] * 3
