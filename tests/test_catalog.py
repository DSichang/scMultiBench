from multibench.data import catalog
from multibench.engine import registry


def test_methods_table_matches_registry(files_dir):
    df = catalog.methods(files_dir)
    # catalog and registry are 1:1 (every registry method has a catalog row)
    assert len(df) == len(registry.load())
    assert set(df["canonical_id"]) == {s.id for s in registry.load()}
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
    row = df[df["method"] == "MOFA+"].iloc[0]
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
    for c in ["dataset", "dataset name", "simulated", "category", "has_results"]:
        assert c in df.columns
    assert (df["dataset"] == df["dataset name"]).all()
    assert df["simulated"].dtype == bool and df["has_results"].dtype == bool
    # every dataset with stored results gets its category from the result tree
    with_res = df[df.has_results]
    assert with_res.category.notna().all()
    assert "D11" in set(with_res.dataset) and with_res.set_index("dataset").loc["D11", "category"] == "vertical"
    for ds in mtb.available_datasets("diagonal"):
        if ds in set(df.dataset):
            assert "diagonal" in df.set_index("dataset").loc[ds, "category"]
    # paper-derived columns exist and are nullable (empty until transcribed)
    for c in catalog.PAPER_COLUMNS:
        assert c in df.columns
    assert df[catalog.PAPER_COLUMNS].isna().all().all()
    # category filter
    assert set(catalog.datasets(files_dir, category="mosaic").dataset) == {"D45"}


def test_catalog_columns_match_registry(files_dir):
    """needs_labels / atac / categories / tasks are OVERLAID from the registry
    (the CSV hand columns had drifted on 37 of 40 rows)."""
    df = catalog.methods(files_dir).set_index("canonical_id")
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
