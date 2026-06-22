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
