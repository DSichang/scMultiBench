"""Review round 4 of 0.3.5: the raw-count hint for every selector, and wording.

- The warning for values that are not whole numbers named ``'layer:counts'``
  for a matrix read from ``obsm`` and for a feature filter. A layer belongs
  to ``X``, so the suggested call wrote the main matrix of the object (the
  RNA) as the ATAC file, or the features of every modality. The hint now
  names the raw counts of the matrix that the selector reads.
- The refusal of one matrix for two modalities told a caller who had passed
  ``data`` again as the second modality to pass an object.
- ``adt_names`` given a string named ``uns['protein_names']`` for an object
  without that key.
- On the command line the diagonal label error began with ``labels='...'``.
- The Changes page said that ``adt='layer:counts'`` is refused.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench import cli, config
from multibench.engine import ingest

from tests.test_docs_r4 import needs_docs
from tests.test_review3_fixes import (CELLS, DEBUG, LOOKS, N, NOT_COUNTS, SAME, _adt, _atac,
                                      _changes, _dense, _refused, _rna, _two_cohorts, _warned)

ad = pytest.importorskip("anndata")


def _table(a, *, counts=False):
    return pd.DataFrame(np.asarray(a.layers["counts"] if counts else a.X),
                        index=a.obs_names, columns=a.var_names)


def _cite():
    """RNA in X with its counts layer; ADT and peaks in obsm, each normalised
    and as counts."""
    a, adt, atac = _rna(), _adt(), _atac()
    a.obsm["protein"], a.obsm["protein_counts"] = _table(adt), _table(adt, counts=True)
    a.obsm["peaks"], a.obsm["peak_counts"] = _table(atac), _table(atac, counts=True)
    return a


def _tenx(col="feature_types"):
    """One X with the features of three modalities, as a 10x file has them."""
    parts = [_rna(), _adt(), _atac()]
    a = ad.AnnData(np.hstack([np.asarray(p.X) for p in parts]))
    a.obs_names = CELLS
    a.var_names = [v for p in parts for v in p.var_names]
    a.var[col] = ["Gene Expression"] * 40 + ["Antibody Capture"] * 8 + ["Peaks"] * 30
    a.layers["counts"] = np.hstack([p.layers["counts"] for p in parts])
    a.obs["celltype"] = parts[0].obs["celltype"].to_numpy()
    return a


def _in_obsm(what, example=None):
    """The whole warning for an obsm entry that has no counts entry to name."""
    example = example or f"{what}='obsm:<key>'"
    return (LOOKS.format(what=what)
            + f"Name the obsm entry that holds the raw counts, as {example}.")


# --------------------------------------------- 1: a matrix read from obsm
def test_atac_in_obsm_is_not_told_to_read_a_layer(tmp_path):
    """0.3.4 and the first 0.3.5 candidate named ``atac='layer:counts'``,
    the counts layer of the RNA."""
    cite = _cite()
    kw = dict(rna="layer:counts", atac_kind="peak", labels="obs:celltype")
    msgs = _warned(ingest.export_dataset, cite, tmp_path / "A", atac="obsm:peaks", **kw)
    assert msgs == [_in_obsm("atac")]
    # the entry that holds the counts, named: the ATAC counts are written
    assert _warned(ingest.export_dataset, cite, tmp_path / "B", atac="obsm:peak_counts",
                   **kw) == []
    assert np.array_equal(_dense(tmp_path / "B" / "atac.h5"), _atac().layers["counts"])
    # what the old hint wrote, for the record: the layer of the RNA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = ingest.export_dataset(cite, tmp_path / "C", rna=None, atac="layer:counts",
                                    atac_kind="peak")
    assert np.array_equal(_dense(old / "atac.h5"), _rna().layers["counts"])


def test_rna_in_obsm_is_not_told_to_read_a_layer(tmp_path):
    a = _adt(log=False)
    a.obsm["rna"] = _table(_rna())
    msgs = _warned(ingest.export_dataset, a, tmp_path / "A", rna="obsm:rna")
    assert msgs == [_in_obsm("rna")]


def test_adt_in_obsm_keeps_its_hint_and_the_hint_works(tmp_path):
    cite = _cite()
    msgs = _warned(ingest.export_dataset, cite, tmp_path / "A", rna="layer:counts",
                   adt="obsm:protein")
    assert msgs == [NOT_COUNTS.format(what="adt") + "Export raw counts, for example with "
                                                    "adt='obsm:protein_counts'."]
    assert _warned(ingest.export_dataset, cite, tmp_path / "B", rna="layer:counts",
                   adt="obsm:protein_counts") == []
    assert np.array_equal(_dense(tmp_path / "B" / "adt.h5"), _adt().layers["counts"])


def test_the_entry_the_adt_hint_names_is_not_named_to_itself(tmp_path):
    """``adt='obsm:protein_counts'`` with normalised values in that entry."""
    cite = _cite()
    cite.obsm["protein_counts"] = cite.obsm["protein"]
    msgs = _warned(ingest.export_dataset, cite, tmp_path / "A", rna="layer:counts",
                   adt="obsm:protein_counts")
    assert msgs == [_in_obsm("adt")]


def test_one_file_from_obsm(tmp_path, monkeypatch):
    cite = _cite()
    msgs = _warned(ingest.to_canonical, cite, tmp_path / "a.h5", modality="atac_peak",
                   obsm="peaks")
    assert msgs == [_in_obsm("atac_peak", "obsm='<key>'")]
    msgs = _warned(ingest.to_canonical, cite, tmp_path / "b.h5", modality="adt", obsm="protein")
    assert msgs == [NOT_COUNTS.format(what="adt") + "Export raw counts, for example with "
                                                    "obsm='protein_counts'."]
    msgs = _warned(ingest.to_canonical, cite, tmp_path / "c.h5", modality="rna")
    assert msgs == [NOT_COUNTS.format(what="rna") + "Export raw counts, for example with "
                                                    "layer='counts'."]
    monkeypatch.setattr(config, "_CLI", True)
    msgs = _warned(ingest.to_canonical, cite, tmp_path / "d.h5", modality="atac_peak",
                   obsm="peaks")
    assert msgs == [_in_obsm("atac_peak", "--obsm <key>")]


# ------------------------------------------------- 1: a feature filter
@pytest.mark.parametrize("col", ["feature_types", "kind"])
@pytest.mark.parametrize("what, value, part", [("rna", "Gene Expression", _rna),
                                               ("adt", "Antibody Capture", _adt),
                                               ("atac", "Peaks", _atac)])
def test_the_hint_keeps_the_feature_filter(tmp_path, col, what, value, part):
    """Without the filter the layer holds the features of every modality:
    0.3.4 wrote all 78 of them as the file of one."""
    tenx = _tenx(col)
    kw = dict(atac_kind="peak") if what == "atac" else {}
    kw.update(rna=None) if what != "rna" else None
    msgs = _warned(ingest.export_dataset, tenx, tmp_path / "A", **{what: f"X[{col}={value}]"},
                   **kw)
    hinted = f"layer:counts[{col}={value}]"
    assert msgs == [NOT_COUNTS.format(what=what) + "Export raw counts, for example with "
                                                   f"{what}='{hinted}'."]
    assert _warned(ingest.export_dataset, tenx, tmp_path / "B", **{what: hinted}, **kw) == []
    name = {"rna": "rna.h5", "adt": "adt.h5", "atac": "atac.h5"}[what]
    assert np.array_equal(_dense(tmp_path / "B" / name), part().layers["counts"])


# ------------------------------------------------------ 1: a MuData
def test_the_hints_of_a_mudata_keep_the_modality(tmp_path):
    md = pytest.importorskip("mudata")
    rna = _cite()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = md.MuData({"rna": rna, "all": _tenx()})
    kw = dict(rna="mod:rna.layer:counts")
    msgs = _warned(ingest.export_dataset, m, tmp_path / "A", adt="mod:rna.obsm:protein", **kw)
    assert msgs == [NOT_COUNTS.format(what="adt") + "Export raw counts, for example with "
                                                    "adt='mod:rna.obsm:protein_counts'."]
    assert _warned(ingest.export_dataset, m, tmp_path / "B",
                   adt="mod:rna.obsm:protein_counts", **kw) == []
    assert np.array_equal(_dense(tmp_path / "B" / "adt.h5"), _adt().layers["counts"])
    msgs = _warned(ingest.export_dataset, m, tmp_path / "C", atac="mod:rna.obsm:peaks",
                   atac_kind="peak", **kw)
    assert msgs == [_in_obsm("atac", "atac='mod:rna.obsm:<key>'")]
    msgs = _warned(ingest.export_dataset, m, tmp_path / "D",
                   adt="mod:all[feature_types=Antibody Capture]", **kw)
    hinted = "mod:all.layer:counts[feature_types=Antibody Capture]"
    assert msgs == [NOT_COUNTS.format(what="adt") + "Export raw counts, for example with "
                                                    f"adt='{hinted}'."]
    assert _warned(ingest.export_dataset, m, tmp_path / "E", adt=hinted, **kw) == []
    assert np.array_equal(_dense(tmp_path / "E" / "adt.h5"), _adt().layers["counts"])


# ------------------------------------------------ 1: the command line
def _export(capsys, *argv):
    assert cli.main(["export", *map(str, argv)]) == 0
    return [line for line in capsys.readouterr().err.splitlines() if "whole numbers" in line]


def test_export_names_the_obsm_entry_and_quotes_the_filter(tmp_path, capsys):
    _cite().write_h5ad(tmp_path / "cite.h5ad")
    _tenx("kind").write_h5ad(tmp_path / "tenx.h5ad")
    lines = _export(capsys, tmp_path / "cite.h5ad", tmp_path / "A", "--rna", "layer:counts",
                     "--atac", "obsm:peaks", "--atac-kind", "peak")
    assert lines == ["warning: " + _in_obsm("atac", "--atac obsm:<key>")]
    lines = _export(capsys, tmp_path / "tenx.h5ad", tmp_path / "B", "--adt",
                     "X[kind=Antibody Capture]")
    assert len(lines) == 1 and lines[0].endswith(
        NOT_COUNTS.format(what="adt") + "Export raw counts, for example with "
        "--adt \"layer:counts[kind=Antibody Capture]\".")
    # export reads the RNA from X unless --adt reads X: name the counts for it too
    assert _export(capsys, tmp_path / "tenx.h5ad", tmp_path / "C", "--adt",
                    "layer:counts[kind=Antibody Capture]", "--rna",
                    "layer:counts[kind=Gene Expression]") == []
    assert np.array_equal(_dense(tmp_path / "C" / "adt.h5"), _adt().layers["counts"])


def test_export_follows_the_selector_of_the_atac_file(tmp_path, capsys):
    """``--atac`` is the selector for the ``--atac-from`` file: the hint
    follows it as it does for SRC."""
    other = [f"o{i}" for i in range(N)]
    holder = _rna(cells=other)
    holder.obsm["peaks"] = _table(_atac(cells=other))
    holder.obsm["peak_counts"] = _table(_atac(cells=other), counts=True)
    holder.write_h5ad(tmp_path / "holder.h5ad")
    tenx = _tenx("kind")
    tenx.obs_names = other
    tenx.write_h5ad(tmp_path / "tenx.h5ad")
    _atac(cells=other).write_h5ad(tmp_path / "atac.h5ad")
    _rna(log=False).write_h5ad(tmp_path / "rna.h5ad")
    base = [tmp_path / "rna.h5ad", None, "--rna", "X", "--atac-kind", "peak", "--category",
            "diagonal", "--atac-from"]

    def run(name, *rest):
        return _export(capsys, *[tmp_path / name if a is None else a for a in base], *rest)

    lines = run("A", tmp_path / "holder.h5ad", "--atac", "obsm:peaks")
    assert lines == ["warning: " + _in_obsm("atac", "--atac obsm:<key>")]
    assert run("B", tmp_path / "holder.h5ad", "--atac", "obsm:peak_counts") == []
    assert np.array_equal(_dense(tmp_path / "B" / "atac_peak.h5"), _atac().layers["counts"])
    lines = run("C", tmp_path / "tenx.h5ad", "--atac", "X[kind=Peaks]")
    assert len(lines) == 1 and lines[0].endswith(
        "Export raw counts, for example with --atac \"layer:counts[kind=Peaks]\".")
    assert run("D", tmp_path / "tenx.h5ad", "--atac", "layer:counts[kind=Peaks]") == []
    assert np.array_equal(_dense(tmp_path / "D" / "atac_peak.h5"), _atac().layers["counts"])
    # no --atac, and X: the layer of that file, as before
    for n, rest in (("E", ()), ("F", ("--atac", "X"))):
        lines = run(n, tmp_path / "atac.h5ad", *rest)
        assert len(lines) == 1 and lines[0].endswith(
            "Export raw counts, for example with --atac layer:counts.")
    assert run("G", tmp_path / "atac.h5ad", "--atac", "layer:counts") == []
    assert np.array_equal(_dense(tmp_path / "G" / "atac_peak.h5"), _atac().layers["counts"])


# ------------------------------------------- 3(a): data passed once more
def test_data_passed_as_the_second_modality_is_named_as_such(tmp_path):
    """The 0.3.5 text names the two arguments and no longer says that they
    are the object passed as ``data``."""
    rna = _rna()
    assert _refused(tmp_path, rna, rna="X", atac=rna, atac_kind="gene_activity") == (
        "rna='X' and atac=" + SAME + "Pass the ATAC as its own object, as atac=<AnnData>.")
    assert _refused(tmp_path, rna, rna=rna, adt=rna) == (
        "rna= and adt=" + SAME + "Name where the ADT is, as adt='obsm:<key>'.")
    # the second modality came by a selector: an object is still the advice
    assert _refused(tmp_path, rna, rna=rna, atac="X", atac_kind="gene_activity") == (
        "rna= and atac='X'" + SAME + "Pass the ATAC as its own object, as atac=<AnnData>.")


# ------------------------------------------- 3(b): the uns key of adt_names
def test_adt_names_names_a_uns_key_only_when_the_string_is_one(tmp_path):
    cite = _rna(log=False)
    cite.obsm["protein"] = _adt(log=False).X
    cite.uns["mynames"] = [f"CD{j}" for j in range(8)]
    head = "adt_names={!r} is one string. Pass a list of names, as adt_names=list(adata.uns[{}])."
    for names, shown in (("mynames", "'mynames'"), ("protein_names", "'<key>'"),
                         ("CD3,CD4", "'<key>'")):
        assert _refused(tmp_path, cite, adt="obsm:protein", adt_names=names) == (
            head.format(names, shown))


# -------------------------------- 3(c): the diagonal label error of export
def test_export_names_the_labels_flag(tmp_path, capsys):
    rna, atac = _two_cohorts()
    atac.obs = atac.obs[[]]
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    assert cli.main(["export", str(tmp_path / "rna.h5ad"), str(tmp_path / "OUT"), "--rna",
                     "X", "--atac-from", str(tmp_path / "atac.h5ad"), "--atac-kind",
                     "gene_activity", "--labels", "obs:cell_type", "--category",
                     "diagonal"]) == 1
    assert capsys.readouterr().err == (
        "error: --labels obs:cell_type: column 'cell_type' not in obs of the ATAC object "
        "(atac); available: []. Without cell types for the ATAC cells, leave --labels "
        "out.\n" + DEBUG)


# ------------------------------------------------ 2: the Changes page
@needs_docs
def test_the_changes_page_says_what_is_refused_and_what_is_not(tmp_path):
    messages, open_ = _changes("Messages"), _changes("Not solved in 0.3.5")
    assert "It is now refused" not in messages
    assert ("0.3.5 no longer gives that advice for an object, an `obsm` entry or a feature "
            "filter. The call itself is refused only when another modality names the same "
            "matrix, as described under *Dataset folders and exports*. Otherwise it still "
            "passes: see *Not solved in 0.3.5*.") in messages
    assert ("- A selector for `adt=` or `atac=` that names a layer or the `X` of the object "
            "that holds the RNA writes RNA values as that modality. "
            "`export_dataset(rna, out, rna=\"X\", adt=\"layer:counts\")` writes the layer "
            "of the RNA object as `adt.h5`, and no message says so. For the ATAC, only "
            "`atac_kind=\"peak\"` warns, because gene names do not look like peaks.") in open_
    assert "`export_dataset` does not compare the features of two modalities." in open_
    # the call of the page, as the page describes it
    rna = _rna(log=False)
    rna.layers["counts"] = rna.layers["counts"] + 1      # told apart from X
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ingest.export_dataset(rna, tmp_path / "OUT", rna="X", adt="layer:counts")
        gas = ingest.export_dataset(rna, tmp_path / "GAS", rna="X", atac="layer:counts",
                                    atac_kind="gene_activity")
    assert np.array_equal(_dense(out / "adt.h5"), rna.layers["counts"])
    assert np.array_equal(_dense(gas / "atac_gas.h5"), rna.layers["counts"])
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == list(rna.var_names)
    with pytest.warns(UserWarning, match="look like peaks"):
        ingest.export_dataset(rna, tmp_path / "PEAK", rna="X", atac="layer:counts",
                              atac_kind="peak")
    # the two selectors the page names for the hint
    assert _warned(ingest.export_dataset, _cite(), tmp_path / "A", rna=None,
                   atac="obsm:peaks", atac_kind="peak") == [_in_obsm("atac")]
    assert _warned(ingest.export_dataset, _tenx(), tmp_path / "B", rna=None,
                   adt="X[feature_types=Antibody Capture]")[0].endswith(
        "adt='layer:counts[feature_types=Antibody Capture]'.")
