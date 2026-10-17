import warnings

import pytest

from multibench.data import catalog
from multibench.engine import registry

# canonical_id / canonical_metric / known_metrics / PAPER_COLUMNS are deprecated
# (0.4) but still work until 0.5; these tests exercise them on purpose.
pytestmark = pytest.mark.filterwarnings(r"ignore:catalog\.\w+ is deprecated:FutureWarning")


def _paper_columns():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return catalog.PAPER_COLUMNS


def test_methods_table_matches_registry(files_dir):
    df = catalog.methods(files_dir)
    # catalog and registry are 1:1 (every registry method has a catalog row)
    assert len(df) == len(registry.load())
    assert set(df["method"]) == {s.id for s in registry.load()}
    # normalized column names (no embedded newlines/spaces)
    assert "method" in df.columns
    assert "language" in df.columns
    assert "categories" in df.columns  # list-valued
    assert "tasks" in df.columns       # list-valued


def test_methods_default_files_dir_matches_registry():
    # called with NO argument -> config.DEFAULT.files_path
    df = catalog.methods()
    assert len(df) == len(registry.load())


def test_categories_are_split_into_lists(files_dir):
    df = catalog.methods(files_dir)
    row = df[df["name"] == "MOFA+"].iloc[0]
    assert isinstance(row["categories"], list)
    # MOFA+ is multi-category; every entry is a clean lowercase token
    assert all(c in {"vertical", "diagonal", "mosaic", "cross"} for c in row["categories"])


def test_language_normalized(files_dir):
    df = catalog.methods(files_dir)
    langs = set(df["language"])
    assert langs <= {"python", "r"}


def test_canonical_id_and_alias_roundtrip():
    # display/result-dir spellings all resolve to one canonical id
    assert catalog.canonical_id("Seurat v3") == catalog.canonical_id("Seurat_v3")
    assert catalog.canonical_id("Seurat.v3") == catalog.canonical_id("Seurat_v3")


def test_metric_crosswalk_canonicalizes_codes():
    # kBET/KBET and iFI/iF1 collapse; blank dropped
    assert catalog.canonical_metric("KBET") == catalog.canonical_metric("kBET")
    assert catalog.canonical_metric("iFI") == catalog.canonical_metric("iF1")


def test_canonical_metric_blank_and_none_return_none():
    assert catalog.canonical_metric("") is None
    assert catalog.canonical_metric("nan") is None
    assert catalog.canonical_metric(None) is None  # real None, not the string "None"


def test_datasets_table_has_simulated_flag(files_dir):
    df = catalog.datasets(files_dir)
    assert "simulated" in df.columns
    assert df["simulated"].dtype == bool


def test_datasets_table_columns_and_registry_derived_category(files_dir):
    import multibench as mtb
    df = catalog.datasets(files_dir)
    for c in ["dataset", "simulated", "category", "has_results"]:
        assert c in df.columns
    assert df["simulated"].dtype == bool and df["has_results"].dtype == bool
    # every dataset with stored results gets its category from the result tree
    with_res = df[df.has_results]
    assert with_res.category.notna().all()
    assert "D11" in set(with_res.dataset) and with_res.set_index("dataset").loc["D11", "category"] == "vertical"
    for ds in set(mtb.results_coverage("diagonal").dataset):
        if ds in set(df.dataset):
            assert "diagonal" in df.set_index("dataset").loc[ds, "category"]
    # the paper-derived columns are empty in the shipped CSV, so none is returned
    # (study round 1, L56; tests/test_catalog_text.py covers a CSV that fills one)
    assert not set(_paper_columns()) & set(df.columns)
    # category filter; D45s (the re-run subsample) ships mosaic results too
    assert set(catalog.datasets(files_dir, category="mosaic").dataset) == {"D45", "D45s"}


def test_catalog_columns_match_registry(files_dir):
    """needs_labels / atac / categories / tasks are OVERLAID from the registry
    (the CSV hand columns had diverged from it)."""
    df = catalog.methods(files_dir).set_index("method")
    specs = {s.id: s for s in registry.load()}
    for cid, spec in specs.items():
        row = df.loc[cid]
        assert bool(row["needs_labels"]) == spec.needs_labels, cid
        assert row["atac"] == spec.atac or (row["atac"] is None and spec.atac is None), cid
        assert list(row["categories"]) == list(spec.categories), cid
        assert list(row["tasks"]) == list(spec.tasks), cid
    assert df.loc["Matilda", "needs_labels"] == True  # noqa: E712 - derived from its cty role
    assert df.loc["moETM", "atac"] == "peak"
    assert set(df["atac"].dropna()) <= {"peak", "gene_activity"}
    # CSV-only columns survive
    assert df["deep_learning"].notna().all() and df["output"].notna().all()


# --- Rin: canonical_id folds case and can insist; canonical_metric can insist

def test_canonical_id_case_folds_against_the_registry():
    import pytest
    assert catalog.canonical_id("totalvi") == "totalVI"
    assert catalog.canonical_id("SCMOMAT") == "scMoMaT"
    assert catalog.canonical_id("Seurat WNN") == "Seurat_WNN"
    # a name the registry does not know is kept (result-dir tokens, user methods)
    assert catalog.canonical_id("Concerto_louvain") == "Concerto_louvain"
    assert catalog.canonical_id("MyMethod") == "MyMethod"
    with pytest.raises(KeyError, match="Unknown method Matlida. Did you mean Matilda"):
        catalog.canonical_id("Matlida", strict=True)
    with pytest.raises(KeyError, match=r"mtb.find_methods\(\) shows all methods"):
        catalog.canonical_id("zzz", strict=True)
    assert catalog.canonical_id("matilda", strict=True) == "Matilda"


def test_canonical_metric_strict_and_known_metrics():
    import pytest
    assert catalog.canonical_metric("nope") == "nope"          # lenient by default
    with pytest.raises(ValueError, match=r"unknown metric 'nope'; valid: \['ARI', 'NMI'"):
        catalog.canonical_metric("nope", strict=True)
    assert catalog.canonical_metric("ari", strict=True) == "ARI"
    km = catalog.known_metrics()
    assert km[:12] == ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI", "ASW_batch",
                       "ARI_batch", "NMI_batch", "GC", "iLISI", "kBET"] and "PCR" not in km


def test_datasets_lists_every_id_with_results(files_dir):
    import multibench as mtb
    df = catalog.datasets(files_dir)
    assert set(mtb.results_coverage().dataset) <= set(df.dataset)
    assert {"D11s", "D28s", "D45s", "D52s"} <= set(df.dataset)
    assert df[df.dataset == "D52s"].category.iloc[0] == "cross"
    assert not df.dataset.duplicated().any()


# --- re-test round 3: output column agrees with the registry, dir() hygiene ---

def test_methods_output_agrees_with_method_info_output_kind(files_dir):
    """method.csv once disagreed with method_info(...)['supports'][*]['output_kind'].
    The CSV must agree with the registry for EVERY method."""
    import multibench as mtb
    df = catalog.methods(files_dir).set_index("method")
    for cid in df.index:
        kinds = {s["output_kind"] for s in mtb.method_info(cid)["supports"]}
        assert kinds == {df.loc[cid, "output"]}, (cid, kinds, df.loc[cid, "output"])
    assert df.loc["scMoMaT", "output"] == "graph"


def test_catalog_dir_lists_public_api_not_leaked_imports():
    """mtb.catalog.annotations() was a TypeError for a re-tester: dir() listed
    the __future__ feature object and the module's imports. Tab completion now
    shows the documented API (__all__) and the dunders only; every attribute is
    still there."""
    import multibench as mtb
    listed = dir(mtb.catalog)
    assert set(catalog.__all__) <= set(listed)
    for leaked in ("annotations", "pd", "Path", "re", "config"):
        assert leaked not in listed, leaked
        assert hasattr(mtb.catalog, leaked)          # nothing removed
    assert "_registry_ids" not in listed and "__all__" in listed
    assert [n for n in listed if not n.startswith("__")] == sorted(catalog.__all__)
    assert listed == sorted(listed)
