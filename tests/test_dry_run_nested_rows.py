"""The dry-run plan leaves out the rows a real run leaves out: UINMF's two-batch row next
to its three-batch row (core040 R-1)."""
import multibench as mtb


def test_uinmf_has_one_row_on_three_batches(capsys):
    plan = mtb.run("D52mini", "cross", methods=["UINMF"], dry_run=True)
    assert plan["modalities"].tolist() == ["rna1+rna2+rna3+adt1+adt2+adt3"]
    out = capsys.readouterr().out
    assert "0 of 1 requested method" in out or "1 of 1 requested method" in out
    assert "UINMF runs on rna1+rna2+rna3+adt1+adt2+adt3, not on rna1+rna2+adt1+adt2" in out


def test_named_modalities_keep_the_smaller_row():
    plan = mtb.run("D52mini", "cross", methods=["UINMF"],
                       modalities=["rna1", "rna2", "adt1", "adt2"], dry_run=True, verbose=False)
    assert plan["modalities"].tolist() == ["rna1+rna2+adt1+adt2"]
