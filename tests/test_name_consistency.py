from multibench.data import catalog
from multibench.run import registry


def test_every_catalog_method_canonical_id_in_registry(files_dir):
    reg_ids = {s.id for s in registry.load()}
    df = catalog.methods(files_dir)
    for _, row in df.iterrows():
        cid = row["canonical_id"]
        assert cid in reg_ids, (
            f"catalog method {row['method']!r} canonicalizes to {cid!r} "
            f"which is not a registry id"
        )
