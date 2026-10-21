"""The plan of scan leaves out the rows a real run leaves out: UINMF's two-batch row next
to its three-batch row (core040 R-1)."""
import multibench as mtb


def test_uinmf_has_one_row_on_three_batches():
    plan = mtb.scan("D52mini", "cross", methods=["UINMF"], verbose=False)
    assert plan["modalities"].tolist() == ["rna1+rna2+rna3+adt1+adt2+adt3"]


def test_named_modalities_keep_the_smaller_row():
    plan = mtb.scan("D52mini", "cross", methods=["UINMF"], modalities=["rna1", "rna2", "adt1", "adt2"], verbose=False)
    assert plan["modalities"].tolist() == ["rna1+rna2+adt1+adt2"]
