from multibench.engine import schema


def test_argspec_positional_vs_flag():
    a = schema.ArgSpec(role="rna")
    b = schema.ArgSpec(role="rna", flag="--path1")
    assert a.is_positional and not b.is_positional


def test_variant_matches_category_and_modalities():
    v = schema.Variant(
        when={"category": "mosaic", "modalities": ["rna", "adt"]},
        entrypoint="x.py", language="python",
        args=[schema.ArgSpec(role="rna", flag="--path1")],
        output=schema.OutputSpec(kind="embedding", file="embedding.h5", dataset="data"),
    )
    assert v.matches("mosaic", {"rna", "adt"})
    assert not v.matches("mosaic", {"rna", "atac"})
    assert not v.matches("vertical", {"rna", "adt"})


def test_methodspec_select_variant():
    spec = schema.MethodSpec(
        id="M", language="python", categories=["mosaic"], tasks=["clustering"],
        variants=[
            schema.Variant(when={"category": "mosaic", "modalities": ["rna", "adt"]},
                           entrypoint="adt.py", language="python", args=[],
                           output=schema.OutputSpec(kind="embedding", file="embedding.h5", dataset="data")),
            schema.Variant(when={"category": "mosaic", "modalities": ["rna", "atac"]},
                           entrypoint="atac.py", language="python", args=[],
                           output=schema.OutputSpec(kind="embedding", file="embedding.h5", dataset="data")),
        ],
    )
    assert spec.select("mosaic", {"rna", "atac"}).entrypoint == "atac.py"
