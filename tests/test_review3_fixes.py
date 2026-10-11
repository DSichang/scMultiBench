"""Review round 3 of 0.3.5: advice that wrote the wrong matrix, and messages.

- The raw-count warning told a caller who passed a modality as its own
  object (``adt=adt``) to write ``adt='layer:counts'``. A selector reads the
  object passed as ``data``, so the advice wrote a layer of the RNA object as
  ``adt.h5``, without a message. The warning now names the way that works
  for an object, and two modalities that name one matrix are refused.
- The check of a cross folder written with gene-activity ATAC said that the
  folder holds RNA and that ADT files are missing.
- A diagonal export whose two objects name the cell types differently was
  only told to give up the scores.
- The line about the cell order named a file as following itself, the
  message for 0 cells spoke of a label table only, and ``adt_names`` given a
  string was counted by character.

The env probe is pinned; the platform is pinned by conftest.
"""
import hashlib
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry

from tests.test_docs_r4 import _read, needs_docs

ad = pytest.importorskip("anndata")

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
N = 60
CELLS = [f"c{i:03d}" for i in range(N)]
PEAKS = [f"chr1:{j * 1000}-{j * 1000 + 500}" for j in range(30)]
DEBUG = "(set MULTIBENCH_DEBUG=1 for the full traceback)\n"
NOT_COUNTS = ("The {what} values are not whole numbers, so they look log-normalised. The "
              "methods normalise raw counts themselves. ")


def _obj(names, seed, *, log=True, cells=CELLS):
    """An AnnData with the counts in ``layers['counts']`` and, with ``log``,
    normalised values in ``X``."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, (len(cells), len(names))).astype(np.float32)
    a = ad.AnnData(np.log1p(counts) if log else counts.copy())
    a.obs_names, a.var_names = list(cells), list(names)
    a.layers["counts"] = counts
    a.obs["celltype"] = [("T", "B", "NK")[i % 3] for i in range(len(cells))]
    return a


def _rna(**kw):
    return _obj([f"GENE{j}" for j in range(40)], 0, **kw)


def _adt(**kw):
    return _obj([f"CD{j}" for j in range(8)], 1, **kw)


def _atac(**kw):
    return _obj(PEAKS, 2, **kw)


def _warned(fn, *a, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*a, **kw)
    return [str(w.message) for w in rec if "whole numbers" in str(w.message)]


def _dense(path):
    return np.asarray(ingest.read_canonical(path, sparse=False).X)


def _bytes(folder):
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def _refused(tmp_path, data, **kw):
    out = tmp_path / "OUT"
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(data, out, **kw)
    assert not out.exists(), "a refused call made the folder"
    return str(e.value)


# ------------------------------------------- 1(a): the advice for an object
def test_the_warning_for_an_object_names_its_own_x(tmp_path):
    """``adt=adt`` and ``atac=atac`` with normalised values in X: the advice
    is about that object, under the keyword the caller wrote. The RNA came
    by a selector and keeps the selector advice."""
    msgs = _warned(ingest.export_dataset, _rna(), tmp_path / "A", rna="X", adt=_adt(),
                   atac=_atac(), atac_kind="peak", labels="obs:celltype")
    assert msgs == [
        NOT_COUNTS.format(what="rna") + "Export raw counts, for example with "
                                        "rna='layer:counts'.",
        NOT_COUNTS.format(what="adt") + "adt= is an object, and its X is written. Put the "
        "raw counts of that object in its X first, for example adt.X = "
        "adt.layers[\"counts\"], then export again.",
        NOT_COUNTS.format(what="atac") + "atac= is an object, and its X is written. Put the "
        "raw counts of that object in its X first, for example atac.X = "
        "atac.layers[\"counts\"], then export again."]


def test_the_warning_for_a_table_has_no_layer_to_name(tmp_path):
    a = _adt()
    table = pd.DataFrame(np.asarray(a.X), index=a.obs_names, columns=a.var_names)
    msgs = _warned(ingest.export_dataset, _rna(log=False), tmp_path / "A", rna="X", adt=table)
    assert msgs == [NOT_COUNTS.format(what="adt") + "Pass the raw counts of these cells as "
                                                    "adt=."]


def test_the_command_line_keeps_its_selector_for_the_atac_file(tmp_path, monkeypatch):
    """``--atac-from`` hands the ATAC over as an object, and ``--atac`` is
    the selector for that file: ``--atac layer:counts`` reads its layer."""
    monkeypatch.setattr(config, "_CLI", True)
    other = _atac(cells=[f"o{i}" for i in range(N)])
    msgs = _warned(ingest.export_dataset, _rna(log=False), tmp_path / "A", rna="X",
                   atac=other, atac_kind="peak", category="diagonal")
    assert msgs == [NOT_COUNTS.format(what="atac") + "Export raw counts, for example with "
                                                     "--atac layer:counts."]


def test_following_the_advice_writes_the_counts_of_that_object(tmp_path):
    rna, adt, atac = _rna(), _adt(), _atac()
    adt.X = adt.layers["counts"]
    atac.X = atac.layers["counts"]
    msgs = _warned(ingest.export_dataset, rna, tmp_path / "A", rna="layer:counts", adt=adt,
                   atac=atac, atac_kind="peak", labels="obs:celltype")
    assert msgs == []
    out = tmp_path / "A"
    assert np.array_equal(_dense(out / "rna.h5"), rna.layers["counts"])
    assert np.array_equal(_dense(out / "adt.h5"), _adt().layers["counts"])
    assert np.array_equal(_dense(out / "atac.h5"), _atac().layers["counts"])
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == list(adt.var_names)


# --------------------------------------- 1(b): one matrix for two modalities
SAME = " name the same matrix, and one file holds one modality. "
OWN_ADT = ("Pass the ADT as its own object (adt=<AnnData>), or name its own place in the "
           "object, such as adt='obsm:<key>' or a feature filter "
           "(adt='X[feature_types=Antibody Capture]').")
OWN_ATAC = ("Pass the ATAC as its own object (atac=<AnnData>), or name its own place in the "
            "object, such as atac='obsm:<key>' or a feature filter "
            "(atac='X[feature_types=Peaks]').")


def test_two_modalities_from_one_matrix_are_refused(tmp_path):
    """What 0.3.4 wrote for these calls: the RNA matrix a second time, as
    ``adt.h5`` or as the ATAC file, without a message."""
    rna = _rna()
    assert _refused(tmp_path, rna, rna="X", adt="X") == "rna='X' and adt='X'" + SAME + OWN_ADT
    assert _refused(tmp_path, rna, rna="layer:counts", adt="layer:counts",
                    labels="obs:celltype") == (
        "rna='layer:counts' and adt='layer:counts'" + SAME + OWN_ADT)
    assert _refused(tmp_path, rna, rna="layer:counts", atac="layer:counts",
                    atac_kind="peak") == (
        "rna='layer:counts' and atac='layer:counts'" + SAME + OWN_ATAC)
    assert _refused(tmp_path, rna, rna="X", atac="X", atac_kind="gene_activity",
                    category="diagonal") == "rna='X' and atac='X'" + SAME + OWN_ATAC
    assert _refused(tmp_path, rna, rna=None, adt="obsm:m", atac="obsm:m",
                    atac_kind="peak") == "adt='obsm:m' and atac='obsm:m'" + SAME + OWN_ATAC
    # the object itself, twice, and the object passed as data once more
    assert _refused(tmp_path, None, rna=rna, adt=rna) == (
        "rna= and adt= are given one object, so they" + SAME + OWN_ADT)
    assert _refused(tmp_path, rna, rna="X", adt=rna) == (
        "rna='X' and adt=AnnData" + SAME + OWN_ADT)
    # one filter twice is one matrix
    rna.var["feature_types"] = "Gene Expression"
    assert _refused(tmp_path, rna, rna="X[feature_types=Gene Expression]",
                    adt="X[feature_types=Gene Expression]") == (
        "rna='X[feature_types=Gene Expression]' and adt='X[feature_types=Gene Expression]'"
        + SAME + OWN_ADT)


def test_the_refusal_on_the_command_line(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CLI", True)
    assert _refused(tmp_path, _rna(), rna="X", adt="X") == (
        "--rna X and --adt X name the same matrix, and one file holds one modality. Name "
        "the place of the ADT in SRC, such as --adt obsm:<key> or a feature filter "
        "(--adt \"X[feature_types=Antibody Capture]\").")
    assert _refused(tmp_path, _rna(), rna="layer:counts", atac="layer:counts",
                    atac_kind="peak") == (
        "--rna layer:counts and --atac layer:counts name the same matrix, and one file "
        "holds one modality. Name the place of the ATAC in SRC, such as --atac obsm:<key> "
        "or a feature filter (--atac \"X[feature_types=Peaks]\"). With --category diagonal, "
        "--atac-from names the file of the ATAC cells.")


def test_convert_refuses_one_matrix_twice_and_writes_nothing(tmp_path, capsys):
    src = tmp_path / "rna.h5ad"
    _rna(log=False).write_h5ad(src)
    out = tmp_path / "OUT"
    assert cli.main(["convert", str(src), str(out), "--rna", "X", "--adt", "X"]) == 1
    assert capsys.readouterr().err == (
        "error: --rna X and --adt X name the same matrix, and one file holds one modality. "
        "Name the place of the ADT in SRC, such as --adt obsm:<key> or a feature filter "
        "(--adt \"X[feature_types=Antibody Capture]\").\n" + DEBUG)
    assert not out.exists()


def test_a_refused_call_leaves_the_folder_as_it_was(tmp_path):
    rna = _rna(log=False)
    out = ingest.export_dataset(rna, tmp_path / "KEEP", rna="X", labels="obs:celltype")
    before = _bytes(out)
    with pytest.raises(ValueError, match="name the same matrix"):
        ingest.export_dataset(rna, out, rna="X", adt="X", labels="obs:celltype",
                              overwrite=True)
    assert _bytes(out) == before


def test_adt_in_x_next_to_an_obsm_matrix_keeps_its_message(tmp_path):
    """The object holds what is probably the protein matrix: the message
    that names the key comes first, as before."""
    rna = _rna(log=False)
    rna.obsm["protein"] = _adt(log=False).X
    assert _refused(tmp_path, rna, rna="X", adt="X") == (
        "adt='X' takes X of this object, and the object also holds the obsm key 'protein'. "
        "If one of them is the protein matrix, pass adt='obsm:<key>'. If X is the protein "
        "matrix, delete it first, for example del adata.obsm['protein'].")


def test_two_matrices_of_one_object_pass(tmp_path):
    """The neighbours of the refusal: two filters on one X (a 10x object),
    X next to an obsm matrix, a layer next to X, and one object each."""
    rna, adt, atac = _rna(log=False), _adt(log=False), _atac(log=False)
    tenx = ad.concat([rna, adt], axis=1)
    tenx.obs["celltype"] = rna.obs["celltype"]
    tenx.var["feature_types"] = ["Gene Expression"] * rna.n_vars \
        + ["Antibody Capture"] * adt.n_vars
    out = ingest.export_dataset(tenx, tmp_path / "TENX", rna="X[feature_types=Gene Expression]",
                                adt="X[feature_types=Antibody Capture]", labels="obs:celltype")
    assert np.array_equal(_dense(out / "rna.h5"), rna.X)
    assert np.array_equal(_dense(out / "adt.h5"), adt.X)

    cite = rna.copy()
    cite.obsm["protein"] = pd.DataFrame(adt.X, index=cite.obs_names, columns=adt.var_names)
    out = ingest.export_dataset(cite, tmp_path / "CITE", rna="X", adt="obsm:protein")
    assert np.array_equal(_dense(out / "adt.h5"), adt.X)

    out = ingest.export_dataset(rna, tmp_path / "LAYER", rna="layer:counts", adt=adt,
                                atac=atac, atac_kind="peak")
    assert np.array_equal(_dense(out / "adt.h5"), adt.X)
    assert np.array_equal(_dense(out / "atac.h5"), atac.X)

    out = ingest.export_dataset(None, tmp_path / "EACH", rna=rna, adt=adt)
    assert sorted(p.name for p in out.iterdir()) == ["adt.h5", "rna.h5"]

    # an object that holds the same values is another object
    out = ingest.export_dataset(rna, tmp_path / "COPY", rna="X", adt=rna.copy())
    assert sorted(p.name for p in out.iterdir()) == ["adt.h5", "rna.h5"]


def test_two_modalities_of_a_mudata(tmp_path):
    mu = pytest.importorskip("mudata")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mu.MuData({"rna": _rna(log=False), "adt": _adt(log=False)})
        out = ingest.export_dataset(m, tmp_path / "MU", rna="rna", adt="adt")
    assert sorted(p.name for p in out.iterdir()) == ["adt.h5", "rna.h5"]
    assert _refused(tmp_path, m, rna="rna", adt="mod:rna") == (
        "rna='rna' and adt='mod:rna'" + SAME + OWN_ADT)


# ------------------------------------------ 2: a cross folder of gene activity
def _h5(path, feats, bars, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), len(bars)))
                         .astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array(bars, dtype="S32"))


def _cross(root, name, atac_stem, batches=3, n=40):
    """A cross folder of RNA and of ATAC whose features are gene names."""
    d = root / name
    d.mkdir(parents=True)
    genes = [f"GENE{i}" for i in range(30)]
    for b in range(1, batches + 1):
        bars = [f"d{b}c{i}" for i in range(n)]
        _h5(d / f"rna{b}.h5", genes, bars)
        _h5(d / f"{atac_stem}{b}.h5", genes, bars, seed=1)
        pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / f"cty{b}.csv", index=False)
    return d


@pytest.fixture
def linux(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)


def _check(capsys, *a, **kw):
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = mtb.scan(*a, per_method=True, **kw)
    return pm, capsys.readouterr().out.splitlines()


GAS_REASON = ("{m} needs peak ATAC (atac1.h5, atac2.h5, atac3.h5), and the folder has gene "
              "activity (atac_gas1.h5, atac_gas2.h5, atac_gas3.h5).")


def test_a_cross_folder_of_gene_activity_holds_atac(linux, tmp_path, capsys):
    """``export_dataset(category="cross", atac_kind="gene_activity")`` writes
    ``atac_gas<N>.h5``, which no cross method reads. The folder holds ATAC
    all the same, and the reason is about its form, not about ADT files."""
    _cross(tmp_path, "GA", "atac_gas")
    pm, lines = _check(capsys, "GA", "cross", methods=["StabMap", "scMM"], data_path=tmp_path)
    assert lines == [
        "[scan] 0 of 2 methods have their input files in GA. 0 of 2 can run on this "
        "computer. GA holds RNA and ATAC, in batches 1-3.",
        "[scan] Cells per file: rna1.h5 40, atac_gas1.h5 40, rna2.h5 40, atac_gas2.h5 40, "
        "rna3.h5 40, atac_gas3.h5 40.",
        "[scan] StabMap: " + GAS_REASON.format(m="StabMap"),
        "[scan] scMM: " + GAS_REASON.format(m="scMM")]
    assert list(pm["reason"]) == [GAS_REASON.format(m="StabMap"), GAS_REASON.format(m="scMM")]
    assert pm.attrs["held"] == ["rna", "atac"] and not pm["runnable"].any()


def test_every_combination_still_gives_the_atac_reason(linux, tmp_path, capsys):
    """With ``modalities="all"`` the ADT rows are listed too, and lack as
    many files as the ATAC rows: the row of the families that the folder
    holds speaks for the method. A method that reads ADT alone keeps its
    own reason."""
    _cross(tmp_path, "GA", "atac_gas")
    pm, lines = _check(capsys, "GA", "cross", methods=["StabMap", "totalVI"],
                       data_path=tmp_path, modalities="all")
    assert lines[2:] == ["[scan] StabMap: " + GAS_REASON.format(m="StabMap"),
                         "[scan] totalVI: adt1.h5, adt2.h5, adt3.h5 are missing."]


def test_the_rows_listed_for_a_gene_activity_folder(linux, tmp_path, capsys):
    """The rows that read RNA and ATAC are listed, as for a folder of peak
    files. None has its files, so a run attempts the same rows as before:
    none."""
    _cross(tmp_path, "GA", "atac_gas")
    _cross(tmp_path, "PK", "atac_peak")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gas = mtb.scan("GA", "cross", data_path=tmp_path, verbose=False)
        every = mtb.scan("GA", "cross", data_path=tmp_path, verbose=False, modalities="all")
        peak = mtb.scan("PK", "cross", data_path=tmp_path, verbose=False)
    assert sorted(zip(gas["method"], gas["modalities"])) \
        == sorted(zip(peak["method"], peak["modalities"]))
    assert gas.attrs["held"] == ["rna", "atac"] and gas.attrs["not_held"] == ["adt"]
    assert not gas["files_ok"].any() and not every["files_ok"].any()
    assert not every["runnable"].any()


def test_gene_activity_under_the_peak_name_keeps_its_reason(linux, tmp_path, capsys):
    _cross(tmp_path, "PK", "atac_peak")
    _, lines = _check(capsys, "PK", "cross", methods=["StabMap"], data_path=tmp_path)
    assert lines[0] == ("[scan] 1 of 1 method has its input files in PK. 0 of 1 can run on "
                        "this computer. PK holds RNA and ATAC, in batches 1-3.")
    assert lines[2] == ("[scan] StabMap: StabMap needs peak ATAC, and atac_peak1.h5 holds "
                        "gene activity. Export the ATAC as peaks, or pass "
                        "allow_atac_mismatch=True to run StabMap anyway.")


def test_one_missing_atac_file_reads_as_before(linux, tmp_path, capsys):
    """One ATAC role, one sentence: the text of 0.3.4."""
    d = tmp_path / "ONE"
    d.mkdir()
    bars = [f"c{i}" for i in range(40)]
    _h5(d / "rna.h5", [f"GENE{i}" for i in range(30)], bars)
    _h5(d / "atac_gas.h5", [f"GENE{i}" for i in range(30)], bars, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = mtb.scan("ONE", "vertical", methods=["MOFA2"], modalities=["rna", "atac"],
                      data_path=tmp_path, verbose=False)
    assert list(df["reason"]) == ["MOFA2 needs peak ATAC (atac.h5), and the folder has gene "
                                  "activity (atac_gas.h5)."]


# ------------------------------- 3: two names for the cell types (diagonal)
def _two_cohorts():
    rna = _rna(log=False)
    rna.obs = rna.obs.rename(columns={"celltype": "cell_type"})
    atac = _obj([f"GENE{j}" for j in range(40)], 3, log=False,
                cells=[f"a{i:03d}" for i in range(50)])
    atac.obs = atac.obs.rename(columns={"celltype": "predicted.id"})
    atac.obs["prediction.score"] = 0.9
    return rna, atac


HEAD = ("labels='obs:cell_type': column 'cell_type' not in obs of the ATAC object (atac); "
        "available: ['predicted.id', 'prediction.score']. If one of them holds the cell "
        "types, ")


def test_the_other_column_name_is_the_first_advice(tmp_path):
    rna, atac = _two_cohorts()
    kw = dict(rna="X", atac=atac, atac_kind="gene_activity", category="diagonal")
    out = tmp_path / "DIAG"
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, out, labels="obs:cell_type", **kw)
    assert e.value.args[0] == (
        HEAD + "give the object a column named 'cell_type' first, for example "
        "atac.obs['cell_type'] = atac.obs['<its column>'], then export again. Without cell "
        "types for the ATAC cells, leave labels= out and run with evaluate=False.")
    assert not out.exists()
    # the advice, followed: each side gets the labels of its own cells
    atac.obs["cell_type"] = atac.obs["predicted.id"]
    ingest.export_dataset(rna, out, labels="obs:cell_type", **kw)
    assert sorted(p.name for p in out.iterdir()) == ["atac_cty.csv", "atac_gas.h5", "rna.h5",
                                                     "rna_cty.csv"]
    assert pd.read_csv(out / "rna_cty.csv")["x"].tolist() == rna.obs["cell_type"].tolist()
    assert pd.read_csv(out / "atac_cty.csv")["x"].tolist() == atac.obs["predicted.id"].tolist()
    assert list(ingest.read_canonical(out / "atac_gas.h5").obs_names) == list(atac.obs_names)


def test_the_rna_side_without_the_column(tmp_path):
    rna, atac = _two_cohorts()
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, tmp_path / "DIAG", rna="X", atac=atac, atac_kind="peak",
                              labels="obs:predicted.id", category="diagonal")
    assert e.value.args[0] == (
        "labels='obs:predicted.id': column 'predicted.id' not in obs of the RNA object "
        "(data); available: ['cell_type']. If one of them holds the cell types, give the "
        "object a column named 'predicted.id' first, for example rna.obs['predicted.id'] = "
        "rna.obs['<its column>'], then export again. Without cell types for the RNA cells, "
        "leave labels= out and run with evaluate=False.")


def test_a_side_without_any_column_keeps_the_one_sentence(tmp_path):
    rna, atac = _two_cohorts()
    atac.obs = atac.obs[[]]
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, tmp_path / "DIAG", rna="X", atac=atac,
                              atac_kind="gene_activity", labels="obs:cell_type",
                              category="diagonal")
    assert e.value.args[0] == (
        "labels='obs:cell_type': column 'cell_type' not in obs of the ATAC object (atac); "
        "available: []. Without cell types for the ATAC cells, leave labels= out and run "
        "with evaluate=False.")


def test_the_command_line_converts_each_side_with_its_own_labels(tmp_path, capsys):
    """The message of ``multibench convert`` names two commands. Run as
    written, with the file names and the column in place, they write the
    folder with the labels of each side on its own cells."""
    rna, atac = _two_cohorts()
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    out = tmp_path / "DIAG"
    assert cli.main(["convert", str(tmp_path / "rna.h5ad"), str(out), "--rna", "X",
                     "--atac-from", str(tmp_path / "atac.h5ad"), "--atac-kind",
                     "gene_activity", "--labels", "obs:cell_type", "--category",
                     "diagonal"]) == 1
    first = ("multibench convert RNA_FILE OUT --rna X --labels obs:cell_type --category "
             "diagonal")
    second = ("multibench convert ATAC_FILE OUT --atac X --atac-kind gene_activity --labels "
              "obs:<its column> --category diagonal --overwrite")
    assert capsys.readouterr().err == (
        "error: " + HEAD + "convert each side with its own command and its own --labels, "
        f"for example {first}, then {second}. Without cell types for the ATAC cells, leave "
        "--labels out and run with --no-evaluate.\n" + DEBUG)
    assert not out.exists()
    for command, src, col in ((first, "rna.h5ad", None), (second, "atac.h5ad", "predicted.id")):
        argv = command.replace("<its column>", col or "").split()[1:]
        argv[1:3] = [str(tmp_path / src), str(out)]
        assert cli.main(argv) == 0
    capsys.readouterr()
    assert sorted(p.name for p in out.iterdir()) == ["atac_cty.csv", "atac_gas.h5", "rna.h5",
                                                     "rna_cty.csv"]
    assert pd.read_csv(out / "rna_cty.csv")["x"].tolist() == rna.obs["cell_type"].tolist()
    assert pd.read_csv(out / "atac_cty.csv")["x"].tolist() == atac.obs["predicted.id"].tolist()
    assert list(ingest.read_canonical(out / "rna.h5").obs_names) == list(rna.obs_names)
    assert list(ingest.read_canonical(out / "atac_gas.h5").obs_names) == list(atac.obs_names)
    assert np.array_equal(_dense(out / "atac_gas.h5"), atac.X)


def test_the_command_line_names_the_kind_and_the_side(tmp_path, monkeypatch):
    """Peaks, and the RNA file is the one without the column."""
    monkeypatch.setattr(config, "_CLI", True)
    rna, atac = _two_cohorts()
    atac.var_names = PEAKS + [f"chr2:{j}-{j + 9}" for j in range(10)]
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, tmp_path / "DIAG", rna="X", atac=atac, atac_kind="peak",
                              labels="obs:predicted.id", category="diagonal")
    assert ("multibench convert RNA_FILE OUT --rna X --labels obs:<its column> --category "
            "diagonal, then multibench convert ATAC_FILE OUT --atac X --atac-kind peak "
            "--labels obs:predicted.id --category diagonal --overwrite. ") in e.value.args[0]


# ------------------------------------------------------- 6(a): the order line
def test_a_file_written_again_is_not_said_to_follow_itself(tmp_path, capsys):
    """The peak file is written again from an object in another row order.
    It keeps the row order of the file that it replaces, and the line says
    that, where it named ``atac_peak.h5`` as following ``atac_peak.h5``.
    With another matrix file of the same cells in the folder, the line
    names that file."""
    rna, atac = _rna(log=False), _atac(log=False, cells=[f"a{i:03d}" for i in range(N)])
    out = ingest.export_dataset(rna, tmp_path / "D", rna="X", atac=atac, atac_kind="peak",
                                labels="obs:celltype", category="diagonal")
    order = list(ingest.read_canonical(out / "atac_peak.h5").obs_names)
    labels = (out / "atac_cty.csv").read_bytes()
    shuffled = atac[np.random.default_rng(1).permutation(N)].copy()
    capsys.readouterr()
    ingest.export_dataset(shuffled, out, rna=None, atac="X", atac_kind="peak",
                          category="diagonal", overwrite=True)
    assert capsys.readouterr().err == (
        f"export_dataset: wrote the rows of atac_peak.h5 in the cell order that "
        f"atac_peak.h5 had in {out} before this call.\n")
    assert list(ingest.read_canonical(out / "atac_peak.h5").obs_names) == order
    assert np.array_equal(_dense(out / "atac_peak.h5"), atac.X)
    ingest.export_dataset(shuffled, out, rna=None, atac="X", atac_kind="peak",
                          labels="obs:celltype", category="diagonal", overwrite=True)
    assert capsys.readouterr().err == (
        f"export_dataset: wrote the rows of atac_peak.h5 and atac_cty.csv in the cell order "
        f"that atac_peak.h5 had in {out} before this call.\n")
    assert (out / "atac_cty.csv").read_bytes() == labels
    gas = shuffled.copy()
    gas.var_names = [f"GENE{j}" for j in range(gas.n_vars)]
    follows = (f"export_dataset: wrote the rows of atac_gas.h5 in the cell order of "
               f"atac_peak.h5, which the files of {out} follow.\n")
    ingest.export_dataset(gas, out, rna=None, atac="X", atac_kind="gene_activity",
                          category="diagonal")
    assert capsys.readouterr().err == follows
    # written again, the gene-activity file has the peak file to name
    ingest.export_dataset(gas, out, rna=None, atac="X", atac_kind="gene_activity",
                          category="diagonal", overwrite=True)
    assert capsys.readouterr().err == follows
    assert list(ingest.read_canonical(out / "atac_gas.h5").obs_names) == order


# ---------------------------------------------------------- 6(b): no cell left
def test_the_message_for_0_cells_names_both_causes(tmp_path):
    rna = _rna(log=False)
    assert _refused(tmp_path, rna[rna.obs["celltype"] == "none"].copy(), rna="X") == (
        "The data holds 0 cells, so there is nothing to write. A filter kept no cell, or "
        "you kept the cells that have a label and no cell had one. For a filter, check "
        "that its values are spelled as the data has them. For labels, check that the "
        "barcodes of the label table are spelled as the cell names of the data (obs_names) "
        "are, for example with the same -1 ending.") == ingest._NO_CELLS


# ------------------------------------------------- 6(c): adt_names as a string
def test_adt_names_as_a_string_is_refused(tmp_path):
    """0.3.4 took each character as a name: 13 characters named 13 proteins,
    and another length raised ``13 feature names for 8 features``."""
    cite = _rna(log=False)
    cite.obsm["protein"] = _adt(log=False).X
    cite.uns["protein_ids"] = [f"CD{j}" for j in range(8)]
    for names, key in (("protein_ids", "protein_ids"), ("CD3", "protein_names"),
                       ("12345678", "protein_names")):
        assert _refused(tmp_path, cite, adt="obsm:protein", adt_names=names) == (
            f"adt_names={names!r} is one string, and adt_names= takes a list of the protein "
            f"names, one for each column of the ADT matrix. For names that the object keeps, "
            f"pass the list, for example adt_names=list(adata.uns[{key!r}]).")
    # the example of the message, and a list as before
    out = ingest.export_dataset(cite, tmp_path / "LIST", adt="obsm:protein",
                                adt_names=list(cite.uns["protein_ids"]))
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == cite.uns["protein_ids"]
    out = ingest.export_dataset(cite, tmp_path / "TUPLE", adt="obsm:protein",
                                adt_names=tuple("ABCDEFGH"))
    assert list(ingest.read_canonical(out / "adt.h5").var_names) == list("ABCDEFGH")


# ------------------------------------- 5: the same features in every batch
SAME_FEATURES = {
    "mosaic_rna_adt": "Every RNA file needs the same gene names and every ADT file the same "
                      "protein names, in the same order.",
    "mosaic_rna_atac": "Every RNA file needs the same gene names and every ATAC file the "
                       "same peaks, in the same order.",
    "mosaic_shared": "Every RNA file needs the same gene names, in the same order.",
    "mosaic_unshared": "Every RNA file needs the same gene names and every ADT file the "
                       "same protein names, in the same order.",
}


@pytest.mark.parametrize("key", sorted(SAME_FEATURES))
def test_a_mosaic_tutorial_says_that_the_batches_need_the_same_features(key):
    """Visible in section 5, for the modalities that the task has in more
    than one batch, with the reason: one method takes the features by their
    position, and the check does not compare the names across batches."""
    from tests.test_tutorial_round3 import GEN, OWN_DATA, _markdown, _visible
    text = " ".join(_visible(_markdown(f"tutorial_{key}", OWN_DATA)).split())
    assert (SAME_FEATURES[key] + " scMoMaT matches the features of two batches by position, "
            "and the check does not compare their names across batches.") in text
    roles = GEN.TASKS[key]["variants"][0]
    for fam, word in (("rna", "RNA file"), ("adt", "ADT file"), ("atac", "ATAC file")):
        several = sum(r.rstrip("0123456789") == fam for r in roles) > 1
        assert (f"very {word}" in SAME_FEATURES[key]) == several, (key, fam)


def test_scmomat_reads_no_feature_names():
    """What the sentence of the mosaic tutorials rests on: the script of
    scMoMaT reads ``matrix/data`` and names the features by their number."""
    from multibench.engine import runner
    repo = runner._repo_root_no_fetch()
    script = repo / "tools_scripts" / "scMoMaT" if repo else None
    if script is None or not (script / "util.py").is_file():
        pytest.skip("the method scripts are not on this machine")
    util = (script / "util.py").read_text()
    assert "data['matrix/data']" in util and "matrix/features" not in util
    assert '"rna" + str(i)' in (script / "main_scMoMaT.py").read_text()


# ------------------------------------------------- 7: the Changes page
def _changes(heading):
    """One section of the 0.3.5 part of the Changes page, on one line."""
    part = _read("changes.md").split("## New in 0.3.4", 1)[0]
    return " ".join(part.split(f"### {heading}\n", 1)[1].split("\n### ", 1)[0].split())


@needs_docs
def test_the_changes_page_names_the_new_refusals_and_texts(linux, tmp_path, capsys):
    exports, checks = _changes("Dataset folders and exports"), _changes("scan and dry runs")
    messages, open_ = _changes("Messages"), _changes("Not solved in 0.3.5")
    # the refusal, with the two calls that the page names
    assert ("`mtb.io.export_dataset` and `multibench convert` raise `ValueError` when two "
            "modalities name the same matrix of the same object, such as `rna=\"X\", "
            "adt=\"X\"` or `rna=\"layer:counts\", atac=\"layer:counts\"`, or one object "
            "passed for two modalities. The call writes nothing.") in exports
    rna = _rna()
    for kw in (dict(rna="X", adt="X"),
               dict(rna="layer:counts", atac="layer:counts", atac_kind="peak")):
        assert "name the same matrix" in _refused(tmp_path, rna, **kw)
    assert ("`export_dataset` raises `ValueError` when `adt_names=` is a string.") in exports
    assert "adt_names=list(adata.uns[\"protein_names\"])" in exports
    # the reason that the page quotes is the one the check prints
    _cross(tmp_path, "GA", "atac_gas")
    pm, _ = _check(capsys, "GA", "cross", methods=["StabMap"], data_path=tmp_path)
    assert f"`{pm['reason'][0]}` 0.3.4 wrote one sentence per file." in messages
    assert ("A cross folder written with `atac_kind=\"gene_activity\"` holds `atac_gas1.h5`, "
            "`atac_gas2.h5` and so on") in checks
    assert "for example `adt.X = adt.layers[\"counts\"]`." in messages
    assert "On the command line, convert each side with its own command and its own " \
           "`--labels`, into the same folder." in messages
    assert open_.startswith("- The check does not compare the feature names of two batches.")
