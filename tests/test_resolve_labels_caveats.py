"""labels_for stacking order; ATAC representation caveats; near-miss filenames.

Workshop findings (Marcus, Chen, Tomás): labels_for('D28') came back
alphabetical ({'atac_cty', 'rna_cty'}) while the methods emit RNA cells first,
so the documented list(labels_for(ds).values()) recipe fed evaluate()
mislabelled cells; scan's caveat caught peak-in-gas only; a vertical variant
asking for atac.h5 next to an atac_peak.h5 said just "not found".
"""
import inspect

import h5py
import numpy as np
import pytest

import multibench as mtb
from multibench.engine import resolve


def _h5(path, n_feat, n_cell, feats=None):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.zeros((n_feat, n_cell)))
        f.create_dataset("matrix/features",
                         data=np.array(feats or [f"g{i}" for i in range(n_feat)], dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array([f"c{i}" for i in range(n_cell)], dtype="S"))


# ----------------------------------------------------------------- H2 labels_for
def test_labels_for_d28_is_rna_then_atac(root):
    lab = mtb.labels_for("D28", data_path=root / "data")
    assert list(lab) == ["rna_cty", "atac_cty"]          # not alphabetical
    assert list(mtb.labels_for("D28", "Portal", "diagonal", data_path=root / "data")) == ["rna_cty", "atac_cty"]
    assert list(mtb.labels_for("D28", "Seurat_v3", "diagonal", data_path=root / "data")) == ["rna_cty", "atac_cty"]


def test_labels_for_d52_numbered_batches(root):
    assert list(mtb.labels_for("D52", data_path=root / "data")) == ["cty1", "cty2", "cty3"]
    assert list(mtb.labels_for("D52", "scMoMaT", "cross", data_path=root / "data")) == ["cty1", "cty2", "cty3"]
    assert list(mtb.labels_for("D11", data_path=root / "data")) == ["cty"]


def test_labels_for_synthetic_folder_orders_numerically_and_by_modality(tmp_path):
    d = tmp_path / "SYN"; d.mkdir()
    for n in ["atac_cty.csv", "cty10.csv", "cty2.csv", "cty1.csv", "rna_cty.csv",
              "peak_cty.csv", "adt_cty.csv", "cty.csv", "source_cty.csv", "zz_cty_scjoint.csv"]:
        (d / n).write_text("x\na\n")
    got = list(mtb.labels_for("SYN", data_path=tmp_path))
    assert got == ["cty", "cty1", "cty2", "cty10", "rna_cty", "adt_cty", "atac_cty", "peak_cty",
                   "source_cty"]
    # plain cty first, numbered numerically (cty10 after cty2), rna < adt < atac (peak = atac)
    assert got.index("cty2") < got.index("cty10")
    assert got.index("rna_cty") < got.index("atac_cty") and got.index("rna_cty") < got.index("peak_cty")
    assert "zz_cty_scjoint" not in got


def test_labels_for_follows_the_variants_modality_order_when_method_and_category_given(tmp_path, monkeypatch):
    """A (synthetic) variant that takes the ATAC file before the RNA file
    stacks ATAC cells first - labels_for(ds, method, category) follows it,
    while the canonical order stays rna-first."""
    from multibench.engine import registry
    from multibench.engine.schema import ArgSpec, MethodSpec, OutputSpec, Variant
    d = tmp_path / "SYN"; d.mkdir()
    for n in ["rna_cty.csv", "atac_cty.csv"]:
        (d / n).write_text("x\na\n")
    v = Variant(when={"category": "diagonal", "modalities": ["atac_gas", "rna"]},
                entrypoint="tools_scripts/X/main.py", language="python",
                args=[ArgSpec(role="atac_gas", flag="--a"), ArgSpec(role="rna", flag="--r"),
                      ArgSpec(role="atac_cty", flag="--ac"), ArgSpec(role="rna_cty", flag="--rc")],
                output=OutputSpec(kind="embedding", file="e.h5"))
    spec = MethodSpec(id="AtacFirst", language="python", categories=["diagonal"],
                      tasks=["clustering"], atac="gene_activity", variants=[v])
    monkeypatch.setattr(registry, "load", lambda: [spec])
    assert list(mtb.labels_for("SYN", data_path=tmp_path)) == ["rna_cty", "atac_cty"]
    assert list(mtb.labels_for("SYN", "AtacFirst", "diagonal", data_path=tmp_path)) == ["atac_cty", "rna_cty"]
    # method alone (no category) does not change the order; the SET never changes
    assert list(mtb.labels_for("SYN", "AtacFirst", data_path=tmp_path)) == ["rna_cty", "atac_cty"]


def test_labels_for_docstring_documents_the_order():
    doc = inspect.getdoc(resolve.labels_for)
    assert "NOT alphabetical" in doc and "rna, adt, atac" in doc
    assert "NUMERICALLY" in doc and "list(labels_for(ds).values())" in doc
    assert "evaluate(labels=" in doc and "modalities" in doc


# ----------------------------------------------------------------- H3 caveats
PEAKS = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(40)]


def test_gas_fed_to_peak_method_is_flagged(tmp_path):
    d = tmp_path / "GAS"; d.mkdir()
    _h5(d / "rna.h5", 30, 50)
    _h5(d / "atac_gas.h5", 40, 50)                     # gene names, not peaks
    # moETM wants peaks behind its atac_gas role
    got = resolve.inputs_for("GAS", "moETM", "vertical", modalities=["rna", "atac_gas"],
                             data_path=tmp_path, check=True)
    assert got["atac_gas"].endswith("atac_gas.h5")
    cav = resolve._preflight_caveats(got, atac="peak")
    assert cav == ["atac_gas resolved to a matrix whose features do not look like peaks "
                   "(chr:start-end); this method expects PEAKS"]
    # legacy call (no atac=): today's single check, nothing for this case
    assert resolve._preflight_caveats(got) == []
    # a peak file satisfies a peak method
    _h5(d / "atac_gas.h5", 40, 50, feats=PEAKS)
    assert resolve._preflight_caveats(got, atac="peak") == []


def test_peak_fed_to_gas_method_is_flagged_on_the_plain_atac_role(tmp_path):
    d = tmp_path / "PK"; d.mkdir()
    _h5(d / "rna.h5", 30, 50)
    _h5(d / "atac.h5", 40, 50, feats=PEAKS)
    (d / "cty.csv").write_text("x\n" + "\n".join(["a"] * 50) + "\n")
    # Matilda rna+atac wants gene activity but reads the plain atac role
    got = resolve.inputs_for("PK", "Matilda", "vertical", modalities=["rna", "atac"],
                             data_path=tmp_path, check=True)
    cav = resolve._preflight_caveats(got, atac="gene_activity")
    assert cav == ["atac resolved to a PEAK matrix (features look like chr:start-end); "
                   "this method expects GENE ACTIVITY"]
    # the legacy atac_gas-role check is unchanged
    _h5(d / "atac.h5", 40, 50, feats=PEAKS)
    got2 = resolve.inputs_for("PK", "Portal", "diagonal", data_path=tmp_path)
    assert resolve._preflight_caveats(got2) == [resolve.PEAK_IN_GAS_CAVEAT]
    assert resolve._preflight_caveats(got2, atac="gene_activity") == [
        resolve.PEAK_FED_TO_GAS_CAVEAT.format(role="atac_gas")]
    # mixed names (10-90 %) -> no verdict either way
    _h5(d / "atac.h5", 40, 50, feats=PEAKS[:20] + [f"g{i}" for i in range(20)])
    got3 = resolve.inputs_for("PK", "Matilda", "vertical", modalities=["rna", "atac"],
                              data_path=tmp_path)
    assert resolve._preflight_caveats(got3, atac="gene_activity") == []
    assert resolve._preflight_caveats(got3, atac="peak") == []


def test_near_miss_vertical_atac_names_the_peak_file(tmp_path):
    d = tmp_path / "MM"; d.mkdir()
    _h5(d / "rna.h5", 30, 50)
    _h5(d / "atac_peak.h5", 40, 50, feats=PEAKS)
    (d / "cty.csv").write_text("x\n" + "\n".join(["a"] * 50) + "\n")
    with pytest.raises(FileNotFoundError) as ei:
        resolve.inputs_for("MM", "Matilda", "vertical", modalities=["rna", "atac"],
                           data_path=tmp_path, check=True)
    msg = str(ei.value)
    assert ("atac.h5 not found; found atac_peak.h5 - vertical methods read atac.h5 "
            "(pass the representation this method wants: see method_info(m)['atac'])") in msg
    # the warn-only default carries the same hint
    with pytest.warns(UserWarning, match="found atac_peak.h5"):
        resolve.inputs_for("MM", "Matilda", "vertical", modalities=["rna", "atac"],
                           data_path=tmp_path)


def test_near_miss_diagonal_gas_names_the_peak_file_and_vice_versa(tmp_path):
    d = tmp_path / "DG"; d.mkdir()
    _h5(d / "rna.h5", 30, 50)
    _h5(d / "atac_peak.h5", 40, 45, feats=PEAKS)
    (d / "rna_cty.csv").write_text("x\n" + "\n".join(["a"] * 50) + "\n")
    with pytest.raises(FileNotFoundError) as ei:
        resolve.inputs_for("DG", "Portal", "diagonal", data_path=tmp_path, check=True)
    assert ("atac_gas.h5 not found; found atac_peak.h5 - diagonal methods read "
            "atac_gas.h5 or atac.h5") in str(ei.value)
    (d / "atac_peak.h5").unlink()
    _h5(d / "atac_gas.h5", 40, 45)
    with pytest.raises(FileNotFoundError) as ei:
        resolve.inputs_for("DG", "Seurat_v3", "diagonal", data_path=tmp_path, check=True)
    assert ("atac_peak.h5 not found; found atac_gas.h5 - diagonal methods read "
            "atac_peak.h5 or peak.h5") in str(ei.value)
    # no sibling at all -> no near-miss clause, the plain message stands
    (d / "atac_gas.h5").unlink()
    with pytest.raises(FileNotFoundError) as ei:
        resolve.inputs_for("DG", "Portal", "diagonal", data_path=tmp_path, check=True)
    assert "not found; found" not in str(ei.value)
