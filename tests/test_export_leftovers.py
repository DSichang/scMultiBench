"""export_dataset refuses a folder that holds files of other data.

A method reads every file of the dataset layout in the folder. 0.3.4 wrote
its own files and left the others, so a folder that was used for an example
and then for the reader's data gave a run on a mix of both. A call now
raises ``FileExistsError`` and writes nothing when the folder holds layout
files of other cells or of other batches that it does not write.

Every refusal is checked on the whole message and on the folder: each file
is byte for byte what it was.
"""
import numpy as np
import pandas as pd
import pytest

from multibench import cli
from multibench.engine import ingest
from multibench.engine import resolve

ad = pytest.importorskip("anndata")

PEAKS = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(40)]
ADVICE = ("A method would read them as part of this dataset. Export to a new folder, and "
          "use that folder name in scan and run_all too. To keep this folder, delete these "
          "files and run the export again.")
CITE = dict(rna="X", adt="obsm:protein", labels="obs:celltype")


def _cells(n, tag, donors=3, seed=0):
    """``(adata, atac)``: RNA with protein, cell types and donors, and peaks of
    the same cells. ``tag`` starts every barcode, so two tags are two sets
    of cells."""
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n, 30)).astype(float))
    a.obs_names = [f"{tag}{i:04d}-1" for i in range(n)]
    a.var_names = [f"GENE{i}" for i in range(30)]
    a.obsm["protein"] = pd.DataFrame(rng.poisson(3.0, (n, 5)).astype(float),
                                     index=a.obs_names, columns=[f"P{i}" for i in range(5)])
    a.obs["celltype"] = rng.choice(["T", "B", "NK"], n)
    a.obs["donor"] = [f"d{i % donors}" for i in range(n)]
    atac = ad.AnnData(rng.poisson(0.5, (n, len(PEAKS))).astype(float))
    atac.obs_names, atac.var_names = a.obs_names, PEAKS
    atac.obs["celltype"] = a.obs["celltype"].to_numpy()
    return a, atac


def _genes(atac):
    """The ATAC object as gene-activity scores: gene names, not peaks."""
    g = atac[:, :30].copy()
    g.var_names = [f"GENE{i}" for i in range(30)]
    return g


def _snapshot(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())} if folder.is_dir() else None


def _refused(data, folder, **kw):
    """The message of a call that must be refused, having checked that no
    file of the folder changed."""
    before = _snapshot(folder)
    with pytest.raises(FileExistsError) as e:
        ingest.export_dataset(data, folder, **{"overwrite": True, **kw})
    assert _snapshot(folder) == before, "a refused call changed the folder"
    return str(e.value)


def _write(data, folder, **kw):
    return ingest.export_dataset(data, folder, **{"overwrite": True, **kw})


def _names(folder):
    return sorted(p.name for p in folder.iterdir())


# ------------------------------------------------------------ calls that pass
@pytest.mark.parametrize("how", [{}, {"batch": "obs:donor", "category": "mosaic"},
                                 {"batch_index": 2, "category": "mosaic"}],
                         ids=["unnumbered", "batch", "batch_index"])
def test_a_folder_built_in_steps_passes(tmp_path, how):
    """RNA with labels, then ADT, then ATAC, then the labels again: four
    calls for the same cells, each writing what the others did not."""
    me, atac = _cells(120, "ME")
    f = tmp_path / "STEPS"
    _write(me, f, rna="X", labels="obs:celltype", **how)
    _write(me, f, rna=None, adt="obsm:protein", **how)
    _write(me, f, rna=None, atac=atac, atac_kind="peak", **how)
    _write(me, f, rna=None, labels="obs:celltype", **how)
    sufs = {"unnumbered": [""], "batch": ["1", "2", "3"], "batch_index": ["2"]}
    key = "batch" if "batch" in how else "batch_index" if how else "unnumbered"
    expected = [f"{stem}{s}{ext}" for s in sufs[key]
                for stem, ext in (("adt", ".h5"), ("atac", ".h5"), ("cty", ".csv"), ("rna", ".h5"))]
    if not how:                             # no category: the peaks under both names
        expected.append("atac_peak.h5")
    assert _names(f) == sorted(expected)


def test_the_same_call_again_passes(tmp_path):
    me, _ = _cells(120, "ME")
    f = tmp_path / "AGAIN"
    for _ in range(2):
        _write(me, f, **CITE)
        _write(me, f, **CITE, batch="obs:donor", category="cross")
    assert len(_names(f)) == 3 + 9


def test_more_donors_after_fewer_pass(tmp_path):
    five, _ = _cells(200, "ME", donors=5)
    three = five[five.obs["donor"].isin(["d0", "d1", "d2"])].copy()
    f = tmp_path / "MORE"
    _write(three, f, **CITE, batch="obs:donor", category="cross")
    _write(five, f, **CITE, batch="obs:donor", category="cross")
    assert len(_names(f)) == 15


def test_two_layouts_of_the_same_cells_pass(tmp_path):
    """All cells in one set of files and the same cells per batch: both are
    this data, in either order."""
    me, _ = _cells(120, "ME")
    for first, second in (({}, {"batch": "obs:donor", "category": "cross"}),
                          ({"batch": "obs:donor", "category": "cross"}, {})):
        f = tmp_path / ("A" if not first else "B")
        _write(me, f, **CITE, **first)
        _write(me, f, **CITE, **second)
        _write(me, f, **CITE, **first)
        assert len(_names(f)) == 12


def test_diagonal_sides_in_two_calls_pass(tmp_path):
    """The RNA cells and the ATAC cells of unpaired data are other cells by
    design, and each side can be written on its own."""
    rna, _ = _cells(120, "RNA")
    _, atac = _cells(90, "ATAC", seed=1)
    f = tmp_path / "DIAG"
    kw = dict(labels="obs:celltype", category="diagonal")
    _write(rna, f, rna="X", **kw)
    _write(None, f, rna=None, atac=_genes(atac), atac_kind="gene_activity", **kw)
    _write(rna, f, rna="X", **kw)
    _write(rna, f, rna="X", atac=_genes(atac), atac_kind="gene_activity", **kw)
    assert _names(f) == ["atac_cty.csv", "atac_gas.h5", "rna.h5", "rna_cty.csv"]


# -------------------------------------------------------- (a) other batches
def test_fewer_donors_after_more_are_refused(tmp_path):
    five, _ = _cells(200, "ME", donors=5)
    three = five[five.obs["donor"].isin(["d0", "d1", "d2"])].copy()
    two = five[five.obs["donor"].isin(["d0", "d1"])].copy()
    f = tmp_path / "FEWER"
    kw = dict(**CITE, batch="obs:donor", category="cross")
    _write(five, f, **kw)
    assert _refused(three, f, **kw) == (
        f"{f} holds files of other data that this call does not write. rna4.h5, adt4.h5, "
        f"cty4.csv, rna5.h5, adt5.h5 and cty5.csv belong to batches 4-5, and this call "
        f"writes batches 1-3. {ADVICE}")
    # more than six files are counted, not all named
    assert _refused(two, f, **kw) == (
        f"{f} holds files of other data that this call does not write. rna3.h5, adt3.h5, "
        f"cty3.csv, rna4.h5, adt4.h5, cty4.csv and 3 more belong to batches 3-5, and this "
        f"call writes batches 1-2. {ADVICE}")
    g = tmp_path / "THREE"
    _write(three, g, **kw)
    assert _refused(two, g, **kw) == (
        f"{g} holds files of other data that this call does not write. rna3.h5, adt3.h5 "
        f"and cty3.csv belong to batch 3, and this call writes batches 1-2. {ADVICE}")


def test_batch_index_checks_other_batch_numbers_only_with_n_batches(tmp_path):
    """Three example batches, then the reader's batch 1 of two. A call with
    batch_index= cannot know how many batches the data has: n_batches= says
    it."""
    f = tmp_path / "INDEX"
    kw = dict(rna="X", labels="obs:celltype", category="cross")
    for b in (1, 2, 3):
        _write(_cells(100, f"EX{b}", seed=b)[0], f, batch_index=b, **kw)
    mine, _ = _cells(90, "MINE", seed=9)
    assert _refused(mine, f, batch_index=1, n_batches=2, **kw) == (
        f"{f} holds files of other data that this call does not write. rna3.h5 and "
        f"cty3.csv belong to batch 3, and n_batches=2 says that the data has 2 batches. "
        f"{ADVICE}")
    # n_batches=3: batch 3 is part of the data
    third = {n: (f / n).read_bytes() for n in ("rna3.h5", "cty3.csv")}
    _write(mine, f, batch_index=1, n_batches=3, **kw)
    # without n_batches, as in 0.3.4: the call passes and batch 3 stays
    _write(mine, f, batch_index=1, **kw)
    assert {n: (f / n).read_bytes() for n in third} == third
    assert resolve._barcodes_of(f / "rna1.h5")[0] == "MINE0000-1"


def test_n_batches_of_one_is_singular(tmp_path):
    f = tmp_path / "ONE"
    kw = dict(rna="X", labels="obs:celltype", category="cross")
    for b in (1, 2):
        _write(_cells(60, f"EX{b}", seed=b)[0], f, batch_index=b, **kw)
    assert _refused(_cells(60, "EX1", seed=1)[0], f, batch_index=1, n_batches=1, **kw) == (
        f"{f} holds files of other data that this call does not write. rna2.h5 and "
        f"cty2.csv belong to batch 2, and n_batches=1 says that the data has 1 batch. "
        f"{ADVICE}")


def test_the_diagonal_loop_with_fewer_batches_is_refused(tmp_path):
    """The tutorial's loop over two lists: with one batch taken out of both,
    n_batches is 2 and the third batch of the example is named."""
    f = tmp_path / "LOOP"
    kw = dict(rna="X", atac_kind="gene_activity", labels="obs:celltype", category="diagonal")
    rna = [_cells(80, f"R{b}", seed=b)[0] for b in (1, 2, 3)]
    atac = [_genes(_cells(70, f"A{b}", seed=10 + b)[1]) for b in (1, 2, 3)]
    for b in range(3):
        _write(rna[b], f, atac=atac[b], batch_index=b + 1, n_batches=3, **kw)
    assert len(_names(f)) == 12
    assert _refused(rna[0], f, atac=atac[0], batch_index=1, n_batches=2, **kw) == (
        f"{f} holds files of other data that this call does not write. rna3.h5, "
        f"atac_gas3.h5, rna_cty3.csv and atac_cty3.csv belong to batch 3, and n_batches=2 "
        f"says that the data has 2 batches. {ADVICE}")


# ----------------------------------------------------------- (b) other cells
def test_another_dataset_into_the_same_mosaic_folder(tmp_path):
    """Batch 1 has the same file names, so it is replaced. Batch 2 leaves the
    ADT of the first dataset next to the RNA and ATAC of the second."""
    f = tmp_path / "MOSAIC"
    kw = dict(labels="obs:celltype", category="mosaic")
    one, one_atac = _cells(100, "ONE")
    _write(one, f, rna="X", batch_index=1, **kw)
    _write(one, f, rna="X", adt="obsm:protein", batch_index=2, **kw)
    _write(one, f, rna=None, adt="obsm:protein", batch_index=3, **kw)
    two, two_atac = _cells(100, "TWO", seed=1)
    _write(two, f, rna="X", batch_index=1, **kw)
    assert _refused(two, f, rna="X", atac=two_atac, atac_kind="peak", batch_index=2, **kw) == (
        f"{f} holds files of other data that this call does not write. adt2.h5 holds other "
        f"cells than this call writes. {ADVICE}")


@pytest.mark.parametrize("how,files", [
    ({}, "rna.h5 holds"),
    ({"batch_index": 1, "category": "cross"}, "rna1.h5 holds"),
    ({"batch": "obs:donor", "category": "cross"}, "rna1.h5, rna2.h5 and rna3.h5 hold"),
], ids=["unnumbered", "batch_index", "batch"])
def test_labels_alone_for_other_cells_are_refused(tmp_path, how, files):
    f = tmp_path / "LABELS"
    _write(_cells(100, "EX")[0], f, rna="X", **how)
    other, _ = _cells(100, "ME", seed=1)
    assert _refused(other, f, rna=None, labels="obs:celltype", **how) == (
        f"{f} holds files of other data that this call does not write. {files} other cells "
        f"than this call writes. {ADVICE}")


def test_labels_without_barcodes_are_judged_by_their_number(tmp_path):
    f = tmp_path / "PLAIN"
    ex, _ = _cells(100, "EX")
    _write(ex, f, rna="X")
    assert _refused(None, f, rna=None, labels=["T"] * 90) == (
        f"{f} holds files of other data that this call does not write. rna.h5 holds other "
        f"cells than this call writes. {ADVICE}")
    _write(None, f, rna=None, labels=list(ex.obs["celltype"]))
    assert pd.read_csv(f / "cty.csv")["x"].tolist() == list(ex.obs["celltype"])


def test_another_atac_kind_for_other_cells_is_refused(tmp_path):
    """Peaks without a category are atac_peak.h5 and atac.h5. Gene activity
    for other cells would leave both next to the new atac_gas.h5."""
    ex, ex_atac = _cells(100, "EX")
    me, me_atac = _cells(100, "ME", seed=1)
    f = tmp_path / "KIND"
    _write(ex, f, rna="X", atac=ex_atac, atac_kind="peak", labels="obs:celltype")
    assert _refused(me, f, rna="X", atac=_genes(me_atac), atac_kind="gene_activity",
                    labels="obs:celltype") == (
        f"{f} holds files of other data that this call does not write. atac.h5 and "
        f"atac_peak.h5 hold other cells than this call writes. {ADVICE}")
    # the same cells: the folder holds both forms of their ATAC
    _write(ex, f, rna="X", atac=_genes(ex_atac), atac_kind="gene_activity",
           labels="obs:celltype")
    assert _names(f) == ["atac.h5", "atac_gas.h5", "atac_peak.h5", "cty.csv", "rna.h5"]


# ----------------------------------------------------- (c) other label counts
def test_labels_left_out_for_other_cells(tmp_path):
    f = tmp_path / "NOLABELS"
    _write(_cells(100, "EX")[0], f, **CITE)
    assert _refused(_cells(90, "ME", seed=1)[0], f, rna="X", adt="obsm:protein") == (
        f"{f} holds files of other data that this call does not write. cty.csv has another "
        f"number of labels than this call has cells. {ADVICE}")
    # as many cells as the old file has labels: the call cannot tell (a label
    # file has no barcodes), and the old labels stay
    old = (f / "cty.csv").read_bytes()
    _write(_cells(100, "ME", seed=1)[0], f, rna="X", adt="obsm:protein")
    assert (f / "cty.csv").read_bytes() == old


# -------------------------------------------- (d), (e) files of another layout
def test_other_cells_by_batch_column_after_unnumbered_files(tmp_path):
    f = tmp_path / "LAYOUT"
    _write(_cells(100, "EX")[0], f, **CITE)
    assert _refused(_cells(90, "ME", seed=1)[0], f, **CITE, batch="obs:donor") == (
        f"{f} holds files of other data that this call does not write. rna.h5 and adt.h5 "
        f"hold other cells than this call writes. cty.csv has another number of labels "
        f"than this call has cells. {ADVICE}")


def test_unnumbered_diagonal_files_into_a_numbered_diagonal_folder(tmp_path):
    f = tmp_path / "DIAG"
    kw = dict(rna="X", atac_kind="gene_activity", labels="obs:celltype", category="diagonal")
    for b in (1, 2, 3):
        _write(_cells(80, f"R{b}", seed=b)[0], f, batch_index=b,
               atac=_genes(_cells(70, f"A{b}", seed=10 + b)[1]), **kw)
    assert _refused(_cells(80, "ME", seed=5)[0], f,
                    atac=_genes(_cells(70, "MA", seed=6)[1]), **kw) == (
        f"{f} holds files of other data that this call does not write. rna1.h5, "
        f"atac_gas1.h5, rna2.h5, atac_gas2.h5, rna3.h5 and atac_gas3.h5 hold other cells "
        f"than this call writes. {ADVICE}")


def test_a_trailing_gem_well_suffix_is_the_same_cell(tmp_path):
    """Barcodes that differ only in the -<n> ending are the same cells here,
    as they are for the resolver."""
    me, _ = _cells(100, "ME")
    f = tmp_path / "SUFFIX"
    _write(me, f, rna="X", batch="obs:donor", category="cross")
    two = me.copy()
    two.obs_names = [b[:-2] + "-2" for b in me.obs_names]
    _write(two, f, rna=None, adt="obsm:protein")            # (d): a subset, by suffix
    _write(two, f, rna=None, adt="obsm:protein", batch="obs:donor", category="cross")   # (b)
    assert len(_names(f)) == 7


# ------------------------------------------------------------------ overwrite
def test_overwrite_does_not_remove_leftovers(tmp_path):
    """overwrite=True replaces the files of the call. It is no permission to
    leave the others, and nothing is deleted."""
    f = tmp_path / "OVER"
    _write(_cells(100, "EX")[0], f, rna="X", adt="obsm:protein", labels="obs:celltype")
    me, _ = _cells(90, "ME", seed=1)
    head = (f"{f} holds files of other data that this call does not write. adt.h5 holds "
            f"other cells than this call writes. cty.csv has another number of labels than "
            f"this call has cells. {ADVICE}")
    assert _refused(me, f, rna="X", overwrite=True) == head
    assert _refused(me, f, rna="X", overwrite=False) == (
        head + " The files this call writes are there too, so also pass overwrite=True.")
    assert _names(f) == ["adt.h5", "cty.csv", "rna.h5"]


def test_an_empty_list_leaves_the_old_check_and_text(tmp_path):
    me, _ = _cells(60, "ME")
    f = tmp_path / "OLD"
    _write(me, f, **CITE)
    with pytest.raises(FileExistsError) as e:
        ingest.export_dataset(me, f, **CITE)
    assert str(e.value) == (f"{f} already holds rna.h5, adt.h5 and cty.csv. Pass "
                            f"overwrite=True to replace them.")


# ------------------------------------------------------------------ n_batches
def test_n_batches_arguments(tmp_path):
    me, _ = _cells(60, "ME")
    f = tmp_path / "ARGS"
    kw = dict(rna="X", category="cross")
    for extra in ({}, {"batch": "obs:donor"}):
        with pytest.raises(ValueError) as e:
            ingest.export_dataset(me, f, n_batches=3, **kw, **extra)
        assert str(e.value) == (
            "n_batches= goes with batch_index=. It says how many batches the calls with "
            "batch_index= write in total. With batch=, the column gives the number.")
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(me, f, batch_index=3, n_batches=2, **kw)
    assert str(e.value) == "batch_index=3 is larger than n_batches=2."
    for bad in (0, -1, 2.0, "3", True):
        with pytest.raises(ValueError) as e:
            ingest.export_dataset(me, f, batch_index=1, n_batches=bad, **kw)
        assert str(e.value) == f"n_batches= must be a positive integer, got {bad!r}"
    assert not f.exists()
    ingest.export_dataset(me, f, batch_index=2, n_batches=2, **kw)
    ingest.export_dataset(me, f, batch_index=1, n_batches=np.int64(2), **kw)
    assert _names(f) == ["rna1.h5", "rna2.h5"]


# ------------------------------------------------------------- command line
def _quiet(fn, *a, **kw):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _error_line(err):
    return next(ln for ln in err.splitlines() if ln.startswith("error: "))


def test_cli_wording(tmp_path, capsys):
    """On the command line the message names the command and its options,
    with or without --overwrite."""
    ex, me = _cells(100, "EX")[0], _cells(90, "ME", seed=1)[0]
    for name, a in (("ex", ex), ("me", me)):
        a.obsm["protein"] = a.obsm["protein"].to_numpy()
        a.write_h5ad(tmp_path / f"{name}.h5ad")
    out = tmp_path / "OUT"
    assert _quiet(cli.main, ["convert", str(tmp_path / "ex.h5ad"), str(out), "--rna", "X",
                             "--labels", "obs:celltype"]) == 0
    before = _snapshot(out)
    capsys.readouterr()
    argv = ["convert", str(tmp_path / "me.h5ad"), str(out), "--rna", "X"]
    told = ("A method would read them as part of this dataset. Convert into a new folder, "
            "and use that folder name in scan and run-all too. To keep this folder, delete "
            "these files and run convert again.")
    head = (f"error: {out} holds files of other data that this command does not write. "
            f"cty.csv has another number of labels than this command has cells. {told}")
    assert _quiet(cli.main, argv + ["--overwrite"]) == 1
    err = capsys.readouterr().err
    assert _error_line(err) == head
    assert _quiet(cli.main, argv) == 1
    err = capsys.readouterr().err
    assert _error_line(err) == (
        head + " The files this command writes are there too, so also pass --overwrite.")
    for word in ("--overwrite", "this command", "scan and run-all"):
        assert word in err
    assert "overwrite=True" not in err and "this call" not in err and "run_all" not in err
    assert _snapshot(out) == before


def test_cli_wording_of_n_batches(tmp_path, monkeypatch):
    """The option --n-batches comes with the command line's other new options.
    What export_dataset says while a command runs is fixed here."""
    from multibench import config
    f = tmp_path / "INDEX"
    kw = dict(rna="X", labels="obs:celltype", category="cross")
    for b in (1, 2, 3):
        _write(_cells(60, f"EX{b}", seed=b)[0], f, batch_index=b, **kw)
    me, _ = _cells(60, "ME", seed=9)
    monkeypatch.setattr(config, "_CLI", True)
    assert _refused(me, f, batch_index=1, n_batches=2, **kw) == (
        f"{f} holds files of other data that this command does not write. rna3.h5 and "
        f"cty3.csv belong to batch 3, and --n-batches 2 says that the data has 2 batches. A "
        f"method would read them as part of this dataset. Convert into a new folder, and use "
        f"that folder name in scan and run-all too. To keep this folder, delete these files "
        f"and run convert again.")
    g = tmp_path / "ARGS"
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(me, g, n_batches=2, rna="X", category="cross")
    assert str(e.value) == (
        "--n-batches goes with --batch-index. It says how many batches the commands with "
        "--batch-index write in total. With --batch, the column gives the number.")
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(me, g, batch_index=3, n_batches=2, rna="X", category="cross")
    assert str(e.value) == "--batch-index 3 is larger than --n-batches 2."
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(me, g, batch_index=1, n_batches=0, rna="X", category="cross")
    assert str(e.value) == "--n-batches must be a positive integer, got 0"
    assert not g.exists()


# ------------------------------------------------------------------ the names
def test_the_rule_uses_the_resolver_names():
    """Every name export_dataset writes is a layout name, and for a name with
    a batch number the layout pattern and the resolver's batch pattern agree."""
    roles = ("rna", "adt", "atac_peak", "atac_gas")
    names = {ingest._export_filename(role, suf, cat) for role in roles
             for suf in ("", "1", "2", "12") for cat in (None, *ingest._CATEGORIES)}
    names |= {f"{stem}{suf}" for stem in ("atac",) for suf in (".h5", "3.h5")}      # the link
    names |= {f"{side}cty{suf}.csv" for side in ("", "rna_", "atac_") for suf in ("", "1", "12")}
    for name in sorted(names):
        m = resolve._LAYOUT_FILE_RE.match(name)
        assert m, name
        digits = m.group(1) or m.group(2)
        b = resolve._BATCH_FILE_RE.match(name)
        assert bool(b) == bool(digits), name
        if b:
            assert (b.group(1) or b.group(2)) == digits, name
    assert sorted({ingest._layout_order(n)[1] for n in names}) == list(range(8))
    for other in ("notes.txt", "peak.h5", "rna.h5ad", "my_rna.h5", "cty.csv.bak",
                  "adt_cty.csv", "rna_.h5", "RNA.h5", "scMM_data"):
        assert resolve._LAYOUT_FILE_RE.match(other) is None, other
        assert ingest._layout_unit(other, False) is None


def test_files_outside_the_layout_are_never_judged(tmp_path):
    """What is not a file of the layout is not read, judged or touched: notes,
    a folder a method made, the older name peak.h5, a folder with a layout
    name, a layout file without barcodes."""
    ex, ex_atac = _cells(100, "EX")
    f = tmp_path / "OTHER"
    _write(ex, f, rna="X")
    (f / "notes.txt").write_text("cells of another experiment\n")
    (f / "scMM_data").mkdir()
    (f / "scMM_data" / "rna.h5").write_bytes(b"not a file of this folder")
    ingest.to_canonical(ex_atac, f / "peak.h5")
    (f / "rna7.h5").mkdir()
    (f / "adt.h5").write_bytes(b"not an HDF5 file")
    (f / "atac_cty.csv").write_bytes(b"\x00\xff not a table")
    before = _snapshot(f / "scMM_data"), {p.name: p.read_bytes() for p in f.iterdir()
                                         if p.is_file() and p.name != "rna.h5"}
    me, _ = _cells(90, "ME", seed=1)
    _write(me, f, rna="X")
    _write(me, f, rna="X", batch="obs:donor", category="cross")
    assert (_snapshot(f / "scMM_data"), {p.name: p.read_bytes() for p in f.iterdir()
                                         if p.is_file() and p.name in before[1]}) == before
    assert resolve._barcodes_of(f / "rna.h5")[0] == "ME0000-1"
