"""evaluate() takes an obs column by its name, with or without ``obs:`` (0.3.5).

``export_dataset`` wrote a column as ``labels="obs:celltype"`` and ``evaluate``
took only ``labels="celltype"``, so the spelling learnt from one call failed
in the other. Both spellings now name the same column in ``labels``, ``batch``
and ``clustering``. A column whose own name starts with ``obs:`` or carries a
MuData prefix is still found under that name. A column name passed with an
array has no obs to look it up in, and the error says so in place of the
missing-file text.
"""
from pathlib import Path

import numpy as np
import pytest

import multibench as mtb
from multibench import config
from multibench.eval import io as eio
from multibench.eval import pipeline
from tests.test_missing_labels import _adata, _cells, _donors, _same, _silent

pytest.importorskip("scib")

METRICS = ["ASW", "ASW_batch"]


def _value(df, metric):
    """One value of evaluate's long table."""
    return df.set_index("metric").loc[metric, "value"]


def _scrambled(values, seed=2):
    return np.random.default_rng(seed).permutation(values)


def test_obs_prefix_names_the_same_column():
    emb, lab = _cells()
    a = _adata(emb, celltype=lab, donor=_donors())
    bare = _silent(a, labels="celltype", batch="donor", metrics=METRICS)
    both = _silent(a, labels="obs:celltype", batch="obs:donor", metrics=METRICS)
    mixed = _silent(a, labels="obs:celltype", batch="donor", metrics=METRICS)
    _same(both, bare)
    _same(mixed, bare)
    # the column itself, which is what the name stands for
    _same(bare, mtb.evaluate(a, labels=a.obs["celltype"], batch=a.obs["donor"],
                             metrics=METRICS))
    assert _value(bare, "ASW") > 0.7
    for name in ("celltype", "obs:celltype"):
        assert pipeline._obs_column_name(name, a) == "celltype"
    assert pipeline._obs_column_name("obs:nope", a) is None
    assert pipeline._obs_column_name("nope", a) is None


def test_obs_prefix_works_for_an_h5ad_path(tmp_path):
    emb, lab = _cells()
    a = _adata(emb, celltype=lab, donor=_donors())
    a.write_h5ad(tmp_path / "mine.h5ad")
    want = mtb.evaluate(a, labels="celltype", batch="donor", metrics=METRICS)
    _same(mtb.evaluate(str(tmp_path / "mine.h5ad"), labels="obs:celltype",
                       batch="obs:donor", metrics=METRICS), want)


def test_exact_column_name_wins():
    emb, lab = _cells()
    wrong = _scrambled(lab)
    # a column that is itself called obs:x, next to a column x
    a = _adata(emb, **{"x": wrong, "obs:x": lab})
    assert pipeline._obs_column_name("obs:x", a) == "obs:x"
    assert pipeline._obs_column_name("x", a) == "x"
    good = mtb.evaluate(a, labels="obs:x", metrics=["ARI"])
    _same(good, mtb.evaluate(emb, labels=lab, metrics=["ARI"]))
    bad = mtb.evaluate(a, labels="x", metrics=["ARI"])
    _same(bad, mtb.evaluate(emb, labels=wrong, metrics=["ARI"]))
    assert _value(good, "ARI") > _value(bad, "ARI") + 0.2
    # obs:obs:x reaches it too, through the prefix
    _same(mtb.evaluate(a, labels="obs:obs:x", metrics=["ARI"]), good)
    # the obs of a MuData names its columns <modality>:<column>
    m = _adata(emb, **{"rna:celltype": lab, "celltype": wrong, "rna:sample": _donors()})
    got = mtb.evaluate(m, labels="rna:celltype", batch="rna:sample", metrics=METRICS)
    _same(got, mtb.evaluate(emb, labels=lab, batch=_donors(), metrics=METRICS))
    _same(mtb.evaluate(m, labels="obs:rna:celltype", batch="obs:rna:sample",
                       metrics=METRICS), got)


def test_clustering_takes_the_obs_prefix():
    # clustering= is deprecated since 0.4: its implementation, without the warning
    emb, lab = _cells()
    a = _adata(emb, celltype=lab, leiden=lab, other=_scrambled(lab))
    kw = dict(labels="celltype", metrics=["ARI", "NMI"])
    bare = pipeline._evaluate(a, clustering="leiden", **kw)
    # before, "obs:leiden" was taken for the path of an h5 file
    prefixed = pipeline._evaluate(a, clustering="obs:leiden", **kw)
    _same(prefixed, bare)
    assert bare["value"].tolist() == [1.0, 1.0] and bare.attrs["clustering"] == "user"
    assert _value(pipeline._evaluate(a, clustering="obs:other", **kw), "ARI") < 0.1
    # a column called obs:leiden wins over the prefix reading
    a.obs["obs:leiden"] = a.obs["other"]
    assert _value(pipeline._evaluate(a, clustering="obs:leiden", **kw), "ARI") < 0.1


def test_unknown_column_keeps_its_error():
    emb, lab = _cells()
    a = _adata(emb, celltype=lab, donor=_donors())
    for what, kw in (("labels", dict(labels="cell_type", batch="donor")),
                     ("labels", dict(labels="obs:cell_type", batch="obs:donor")),
                     ("batch", dict(labels="celltype", batch="obs:sample")),
                     ("batch", dict(labels="celltype", batch="sample"))):
        name = kw[what]
        with pytest.raises(ValueError) as e:
            mtb.evaluate(a, metrics=METRICS, **kw)
        # Y19 (r3): cell_type gets the did-you-mean
        hint = " Did you mean 'celltype'?" if "cell_type" in name else ""
        assert str(e.value) == (
            f"{what}={name!r} is neither an obs column of the AnnData (obs columns: "
            f"['celltype', 'donor']) nor an existing file.{hint}")


def test_a_file_named_like_a_prefixed_column_is_still_read(tmp_path, monkeypatch):
    """Not a column, but an existing file: read as a file, as before."""
    emb, lab = _cells()
    a = _adata(emb, donor=_donors())
    monkeypatch.chdir(tmp_path)
    Path("obs:celltype").write_text("cell,x\n" + "".join(
        f"{c},{v}\n" for c, v in zip(a.obs_names, lab)))
    _same(mtb.evaluate(a, labels="obs:celltype", metrics=["ASW"]),
          mtb.evaluate(emb, labels=lab, metrics=["ASW"]))


def test_array_output_with_a_column_name(tmp_path, monkeypatch):
    emb, lab = _cells()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        mtb.evaluate(emb, labels="celltype", metrics=["ASW"])
    assert str(e.value) == (
        "labels='celltype' is not a file, and the embedding has no obs. Pass the values, as "
        "labels=adata.obs['celltype'].")
    # the example names the column without its prefix
    with pytest.raises(FileNotFoundError) as e:
        mtb.evaluate(emb, labels=lab, batch="obs:donor", metrics=METRICS)
    assert str(e.value) == (
        "batch='obs:donor' is not a file, and the embedding has no obs. Pass the values, as "
        "batch=adata.obs['donor'].")
    # clustering= is deprecated since 0.4: its implementation, without the warning
    with pytest.raises(FileNotFoundError) as e:
        pipeline._evaluate(emb, labels=lab, clustering="obs:leiden", metrics=["ARI"])
    assert str(e.value) == (
        "clusters='obs:leiden' is not a file, and the embedding has no obs. Pass the "
        "values, as clusters=adata.obs['leiden'].")
    # a name that reads as a file keeps the text of a missing file
    cwd = Path.cwd()
    with pytest.raises(FileNotFoundError) as e:
        mtb.evaluate(emb, labels="missing.csv", metrics=["ASW"])
    assert str(e.value) == (
        f"The labels file missing.csv does not exist. From the working directory "
        f"{cwd}, the path resolves to {Path('missing.csv').resolve()}.")
    with pytest.raises(FileNotFoundError) as e:
        mtb.evaluate(emb, labels=lab, batch="sub/donor", metrics=METRICS)
    assert str(e.value) == (
        f"The batch file sub/donor does not exist. From the working directory "
        f"{cwd}, the path resolves to {Path('sub/donor').resolve()}.")
    with pytest.raises(FileNotFoundError) as e:
        pipeline._evaluate(emb, labels=lab, clustering="clusters.h5", metrics=["ARI"])
    assert str(e.value) == (
        f"The clustering file clusters.h5 does not exist. From the working directory "
        f"{cwd}, the path resolves to {Path('clusters.h5').resolve()}.")
    # a Path is a path
    with pytest.raises(FileNotFoundError, match=r"^The labels file celltype does not exist\."):
        mtb.evaluate(emb, labels=Path("celltype"), metrics=["ASW"])
    # an existing file without a suffix is read
    Path("celltype").write_text("x\n" + "".join(f"{v}\n" for v in lab))
    _same(mtb.evaluate(emb, labels="celltype", metrics=["ASW"]),
          mtb.evaluate(emb, labels=lab, metrics=["ASW"]))


def test_the_command_line_keeps_the_missing_file_text(tmp_path, monkeypatch):
    emb, lab = _cells()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(FileNotFoundError) as e:
        mtb.evaluate(emb, labels="celltype", metrics=["ASW"])
    assert str(e.value) == (
        f"The --labels file celltype does not exist. From the working directory "
        f"{Path.cwd()}, the path resolves to {Path('celltype').resolve()}.")


def test_names_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("donor", "obs:donor", "cell type", "leiden_r0.5", "rna:celltype"):
        assert eio._names_no_file(name), name
    for name in ("missing.csv", "labels.tsv", "x.TXT", "sub/donor", "./donor",
                 str(tmp_path / "donor")):
        assert not eio._names_no_file(name), name
    for other in (None, 3, Path("donor"), ["donor"], np.array(["donor"])):
        assert not eio._names_no_file(other), other
    Path("donor").write_text("x\nd1\n")
    assert not eio._names_no_file("donor")             # now it is a file


def test_docstring_states_the_rule():
    doc = " ".join(mtb.evaluate.__doc__.split())
    # 0.4.0 docstring: the obs: spelling still works but is no longer listed
    assert ("an ``.obs`` column name (``\"celltype\"``, "
            "``\"rna:celltype\"`` for a MuData modality)") in doc


def test_a_10x_suffix_on_one_side_names_its_fix():
    # Z25 (r3): the '-1' difference and the one call that fixes it
    import numpy as np
    import pandas as pd
    from multibench.eval import io
    ids = [f"AAAC{i}-1" for i in range(4)]
    lab = pd.Series(list("abab"), index=[f"AAAC{i}" for i in range(4)])
    with pytest.raises(ValueError, match=r"They match after adding '-1' to the labels ids: "
                                         r"pass labels\.set_axis\(labels\.index \+ '-1'\)\.$"):
        io._align_vector(lab, ids, what="labels", column=None, cells=None)
    fixed = lab.set_axis(lab.index + "-1")
    assert list(io._align_vector(fixed, ids, what="labels", column=None, cells=None)) == \
        list("abab")
    back = pd.Series(list("abab"), index=ids)
    with pytest.raises(ValueError, match=r"removesuffix\('-1'\)"):
        io._align_vector(back, [i[:-2] for i in ids], what="labels", column=None, cells=None)


def test_a_label_column_spelled_another_way_gets_a_did_you_mean():
    # Y19 (r3): letter case and underscores, in evaluate as in export_dataset
    import anndata as ad
    import numpy as np
    import pandas as pd
    a = ad.AnnData(np.zeros((6, 2)), obs=pd.DataFrame({"celltype": list("ababab")},
                                                     index=[f"c{i}" for i in range(6)]))
    a.obsm["X_emb"] = np.random.default_rng(0).normal(size=(6, 3))
    for wrong in ("Celltype", "cell_type"):
        with pytest.raises(ValueError, match=r"Did you mean 'celltype'\?$"):
            mtb.evaluate(a, labels=wrong, metrics=["ARI"])
        with pytest.raises(ValueError, match=r"Did you mean 'celltype'\?$"):
            mtb.io.export_dataset(a, "unused", labels=wrong)
