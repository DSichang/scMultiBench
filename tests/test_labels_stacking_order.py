"""labels_for follows each method's cell order; evaluate takes that dict as is.

Two defects, both reproduced on the demo data before the fix:

* ``labels_for("D52", "cross", "StabMap")`` returned ``cty1, cty2, cty3``,
  but StabMap stacks its reference batch first (``reference=data3``; its
  ``stabMap()`` rbinds the reference's own embedding before the others).
  Scored against that dict, ``data/outputs/D52/StabMap_D52/embedding.h5``
  gave ARI 0.0005; against ``cty3, cty1, cty2`` it gave 0.700. uniPort is the
  same case on D28: it takes RNA as ``--path1`` but concatenates the ATAC
  cells first (``main_uniPort.py:48``).
* ``evaluate(labels=labels_for(ds, category, method))`` raised the
  multi-entry dict ``ValueError`` - whose hint recommends that very call -
  whenever the method's order was not the default one: SMILE
  (``cty2, cty1, cty3``) and MultiVI (``cty1, cty3, cty2``) on D45.
"""
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import registry, resolve


def _write_cty(path, values):
    pd.DataFrame({"x": values}).to_csv(path, index=False)


# ------------------------------------------------ the order labels_for returns
def test_stabmap_puts_its_reference_batch_first(root):
    data = root / "data"
    assert list(mtb.labels_for("D52", "cross", "StabMap", data_path=data)) == ["cty3", "cty1", "cty2"]
    # the mosaic variant's reference is data1, so the default order is right there
    assert list(mtb.labels_for("D46", "mosaic", "StabMap", data_path=data)) == ["cty1", "cty2", "cty3"]
    # without a method nothing changes
    assert list(mtb.labels_for("D52", data_path=data)) == ["cty1", "cty2", "cty3"]


def test_uniport_puts_its_atac_cells_first(root):
    data = root / "data"
    assert list(mtb.labels_for("D28", "diagonal", "uniPort", data_path=data)) == ["atac_cty", "rna_cty"]
    assert list(mtb.labels_for("D28", "diagonal", "SCALEX", data_path=data)) == ["rna_cty", "atac_cty"]


def test_smile_and_multivi_follow_their_upstream_concatenation(root):
    """main_SMILE.py: (paired rna2+atac2), query rna1, query atac3.
    main_MultiVI.py: rna1, atac3, then the paired cells (D45: equal batch sizes)."""
    data = root / "data"
    assert list(mtb.labels_for("D45", "mosaic", "SMILE", data_path=data)) == ["cty2", "cty1", "cty3"]
    assert list(mtb.labels_for("D45", "mosaic", "MultiVI", data_path=data)) == ["cty1", "cty3", "cty2"]


# ------------------------------------------------ evaluate takes the dict as is
_CASES = [
    # method, category, label files, the order that method stacks them
    ("SMILE", "mosaic", ["cty1", "cty2", "cty3"], ["cty2", "cty1", "cty3"]),
    ("MultiVI", "mosaic", ["cty1", "cty2", "cty3"], ["cty1", "cty3", "cty2"]),
    ("StabMap", "cross", ["cty1", "cty2", "cty3"], ["cty3", "cty1", "cty2"]),
    ("uniPort", "diagonal", ["rna_cty", "atac_cty"], ["atac_cty", "rna_cty"]),
]


def _stacked_dataset(tmp_path, files, order):
    """A folder whose label files only line up with ``emb`` in ``order``:
    block k of the embedding (10*(k+1) cells, one tight cluster) is labelled
    by the k-th file of ``order``, one cell type per file. Blocks of distinct
    sizes make every other file order a different partition of the cells."""
    d = tmp_path / "SYN"
    d.mkdir()
    rng = np.random.default_rng(0)
    blocks = []
    for k, stem in enumerate(order):
        n = 10 * (k + 1)
        _write_cty(d / f"{stem}.csv", [f"type{k}"] * n)
        blocks.append(rng.normal(10.0 * k, 0.1, size=(n, 5)))
    assert sorted(files) == sorted(order)
    return np.vstack(blocks)


@pytest.mark.parametrize("method,category,files,order", _CASES)
def test_evaluate_takes_the_labels_for_dict_as_is(tmp_path, method, category, files, order):
    pytest.importorskip("scib")
    emb = _stacked_dataset(tmp_path, files, order)
    d = mtb.labels_for("SYN", category, method, data_path=tmp_path)
    assert list(d) == order
    got = mtb.evaluate(emb, labels=d, metrics=["ASW"], verbose=False)
    ref = mtb.evaluate(emb, labels=[d[k] for k in order], metrics=["ASW"], verbose=False)
    pd.testing.assert_frame_equal(got, ref)
    assert float(got.loc["ASW", "Value"]) > 0.9
    # the fixture can tell the orders apart: the default order scores worse
    default = mtb.labels_for("SYN", data_path=tmp_path)
    assert list(default) != order
    worse = mtb.evaluate(emb, labels=list(default.values()), metrics=["ASW"], verbose=False)
    assert float(worse.loc["ASW", "Value"]) < float(got.loc["ASW", "Value"]) - 0.1


def test_a_copied_or_reordered_dict_is_checked_like_any_dict(tmp_path):
    """Only the dict labels_for returned, in the order it returned it, goes in
    as is; a copy or an edit is held to the default order like any dict."""
    pytest.importorskip("scib")
    emb = _stacked_dataset(tmp_path, *_CASES[0][2:])
    d = mtb.labels_for("SYN", "mosaic", "SMILE", data_path=tmp_path)
    with pytest.raises(ValueError) as exc:
        mtb.evaluate(emb, labels=dict(d), metrics=["ASW"], verbose=False)
    msg = str(exc.value)
    assert "mtb.labels_for(dataset, method=<method>, category=<category>)" in msg
    assert "unchanged" in msg
    moved = mtb.labels_for("SYN", "mosaic", "SMILE", data_path=tmp_path)
    moved["cty2"] = moved.pop("cty2")          # same keys, cty2 moved to the end
    with pytest.raises(ValueError, match="default cell order"):
        mtb.evaluate(emb, labels=moved, metrics=["ASW"], verbose=False)
    # label_order= still overrides the recorded order
    got = mtb.evaluate(emb, labels=d, label_order=["cty2", "cty1", "cty3"],
                       metrics=["ASW"], verbose=False)
    assert float(got.loc["ASW", "Value"]) > 0.9


def test_the_error_hint_is_valid_for_every_wired_variant(root):
    """The ValueError for a mis-ordered dict recommends
    labels_for(dataset, method=..., category=...): that dict must never raise
    it, for any method wired for a demo dataset's category."""
    from multibench.eval.pipeline import _labels_from_dict
    data = root / "data"
    demo = {"D11": "vertical", "D28": "diagonal", "D45": "mosaic",
            "D46": "mosaic", "D52": "cross"}
    checked = 0
    for ds, cat in demo.items():
        for spec in registry.load():
            if not any(v.when.get("category") == cat for v in spec.variants):
                continue
            d = mtb.labels_for(ds, cat, spec.id, data_path=data)
            assert _labels_from_dict(d, None) == list(d.values()), (ds, spec.id)
            checked += 1
    assert checked > 20


# ------------------------------------------------ the registry declaration
def _variant(args, cell_order, category="cross"):
    mods = [a["role"] for a in args if a.get("role") and a["role"] not in ("out_dir", "reference")]
    return {"when": {"category": category, "modalities": mods},
            "entrypoint": "tools_scripts/X/main.R", "language": "R", "args": args,
            "output": {"kind": "embedding", "file": "embedding.h5", "dataset": "data",
                       "cell_order": cell_order}}


_STABMAP_ARGS = [{"role": "rna1", "flag": "--rna"}, {"role": "rna2", "flag": "--rna"},
                 {"role": "adt1", "flag": "--adt"}, {"roles": ["=None"], "flag": "--adt"},
                 {"role": "out_dir", "flag": "--save_path"}]


def test_reference_first_is_derived_from_the_reference_const():
    args = _STABMAP_ARGS + [{"role": "reference", "flag": "--reference", "const": "data2"}]
    v = registry._parse_variant(_variant(args, "reference_first"), "X")
    assert v.stacked_roles()[:1] == ["rna2"]
    assert v.stacked_roles()[1:3] == ["rna1", "adt1"]


@pytest.mark.parametrize("args,cell_order,match", [
    (_STABMAP_ARGS, "reference_first", "no `reference` arg"),
    (_STABMAP_ARGS + [{"role": "reference", "flag": "--reference", "const": "ref"}],
     "reference_first", "data<N>"),
    (_STABMAP_ARGS + [{"role": "reference", "flag": "--reference", "const": "data3"}],
     "reference_first", "batch 3"),
    # slot k of a repeated flag must hold batch k: util.R names the slots data1..dataN
    ([{"role": "rna2", "flag": "--rna"}, {"role": "rna1", "flag": "--rna"},
      {"role": "reference", "flag": "--reference", "const": "data1"}],
     "reference_first", "slot 1 of --rna"),
    (_STABMAP_ARGS, "atac_first", "cell_order"),
    (_STABMAP_ARGS, ["rna2", "atac9"], "atac9"),
    (_STABMAP_ARGS, ["rna2", "rna2"], "twice"),
])
def test_a_cell_order_the_variant_cannot_honour_fails_at_load(args, cell_order, match):
    with pytest.raises(ValueError, match=match):
        registry._parse_variant(_variant(args, cell_order), "X")


def test_every_declared_cell_order_parses(root):
    declared = {(s.id, v.when["category"]): v.output.cell_order
                for s in registry.load() for v in s.variants if v.output.cell_order}
    assert declared == {("StabMap", "cross"): "reference_first",
                        ("StabMap", "mosaic"): "reference_first",
                        ("uniPort", "diagonal"): ["atac_gas", "rna"],
                        ("Seurat_v5", "diagonal"): ["atac_peak", "rna"]}
    assert type(mtb.labels_for("D11", data_path=root / "data")) is resolve.LabelFiles
