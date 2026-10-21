"""Findings of the review of the round-4 integration (wp/f4_int).

- A peak file whose names are neither chr:start-end nor the folder's gene
  names (peak_0) is not gene activity: MultiMAP, which never reads the names,
  stays runnable; GLUE and Seurat_v3 stay blocked by the peak-name check.
  Methods of either kind get a caveat that does not block.
- The ATAC override ends the reason, after the env and GPU sentences.
- run with allow_atac_mismatch=True reports the caveat once, not again
  as a UserWarning; a direct mtb.run still warns.
- RunResult's repr counts a named SKIPPED method once; summary keeps
  n_batches whole next to SKIPPED rows.
- Batch Series errors: row-number indexes, missing ids and foreign ids each
  get their own fix.
- --strict: the per-method line reads the row whose files are on disk.
- run-all's failure line says when the saved folder kept an earlier record.
- An env with a single build reports flavor 'single'.
- No user-visible text says 'registry'; the cite example cites only the
  methods that finished.
"""
import inspect
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import scripts
from multibench import _settings
from multibench import config, discover
from multibench import workflow as W
from multibench.engine import envs, registry
from multibench.engine import resolve as RS
from multibench.engine import runner as R
from tests.conftest import full_scan

#: the caveat of a gene-activity method given peaks in atac.h5 (after the method)
GAS_CAV = ("needs gene-activity ATAC. The features of atac.h5 look like chr:start-end, "
           "so it holds peaks.")

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
GENES = [f"GENE{i}" for i in range(60)]


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Every env installed, a GPU, no method scripts on this machine, no ref."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(_settings, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(scripts.SCRIPTS_REF_VAR, raising=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _h5(path, feats, bars):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((len(feats), len(bars))))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))


def _labels(path, n):
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(path, index=False)


def _diagonal(root, name, peaks, n=20, rna_genes=GENES):
    d = root / name
    d.mkdir(parents=True)
    rna, atac = [f"r{i}" for i in range(n)], [f"a{i}" for i in range(n)]
    _h5(d / "rna.h5", rna_genes, rna)
    _h5(d / "atac_peak.h5", peaks, atac)
    _h5(d / "atac_gas.h5", rna_genes, atac)
    _labels(d / "rna_cty.csv", n)
    _labels(d / "atac_cty.csv", n)
    return root


def _vertical(root, name, atac_feats, n=60, adt=False):
    d = root / name
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", GENES[:30], bars)
    _h5(d / "atac.h5", atac_feats, bars)
    if adt:
        _h5(d / "adt.h5", [f"p{i}" for i in range(6)], bars)
    _labels(d / "cty.csv", n)
    return root


def test_peak_ids_in_a_vertical_folder_give_both_kinds_a_caveat(tmp_path, pinned,
                                                                gas_matilda):
    """atac.h5 names that are neither chr:start-end nor the RNA's genes: the
    peak methods run with a caveat, and so does a gene-activity method
    (``gas_matilda``: the real Matilda reads peaks)."""
    root = _vertical(tmp_path, "MYMO_ids", [f"peak_{i}" for i in range(40)])
    sc = _quiet(full_scan, "MYMO_ids", "vertical", methods=["scMVP", "Matilda"],
                modalities=["rna", "atac"], data_path=root, verbose=False).set_index("method")
    assert sc["runnable"].all(), sc["reason"].tolist()
    assert sc.loc["scMVP", "caveat"].startswith(
        "scMVP needs peak ATAC. atac.h5 holds names such as peak_0, not chr:start-end.")
    assert sc.loc["Matilda", "caveat"].startswith(
        "Matilda needs gene-activity ATAC. atac.h5 holds names such as peak_0, not the "
        "RNA's genes.")


def test_gene_names_in_the_peak_file_still_block_multimap(tmp_path, pinned):
    """Names that are the folder's genes are gene activity, whatever their case."""
    root = _diagonal(tmp_path, "LUNG_ga", [g.lower() for g in GENES])
    row = _quiet(full_scan, "LUNG_ga", "diagonal", methods=["MultiMAP"], data_path=root,
                 verbose=False).iloc[0]
    assert not row["runnable"]
    assert row["reason"].startswith("MultiMAP needs peak ATAC, and atac_peak.h5 holds gene "
                                    "activity. ")


def test_names_are_genes_needs_a_file_to_compare_with(tmp_path):
    d = tmp_path / "X"
    d.mkdir()
    _h5(d / "atac.h5", [f"peak_{i}" for i in range(50)], ["c0"])
    # nothing to compare with: unknown, and the old check applies
    assert RS._names_are_genes(d / "atac.h5") is None
    _h5(d / "rna.h5", GENES, ["c0"])
    assert RS._names_are_genes(d / "atac.h5") is False
    _h5(d / "atac.h5", GENES[:40], ["c0"])
    assert RS._names_are_genes(d / "atac.h5") is True
    _h5(d / "atac.h5", [f"ENSG{i:011d}" for i in range(50)], ["c0"])
    assert RS._names_are_genes(d / "atac.h5") is True
    # an RNA named by Ensembl id cannot tell symbols from peak ids
    _h5(d / "rna.h5", [f"ENSG{i:011d}" for i in range(50)], ["c0"])
    _h5(d / "atac.h5", [f"peak_{i}" for i in range(50)], ["c0"])
    assert RS._names_are_genes(d / "atac.h5") is None


# ------------------------------- a GPU-blocked row names no override (0.4.0)
def test_a_gpu_blocked_row_ends_with_the_export_fix_and_no_override(
        tmp_path, pinned, monkeypatch, as_gene_activity):
    # a method that needs a GPU and is given the other ATAC form. UnitedNet
    # reads peaks; it is set to gene activity here to meet both conditions
    as_gene_activity("UnitedNet")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    row = _quiet(full_scan, "MU_PEAK", "vertical", methods=["UnitedNet"],
                 modalities=["rna", "atac"], data_path=root, verbose=False).iloc[0]
    assert not row["runnable"]
    assert "needs an NVIDIA GPU" in row["reason"]
    assert row["reason"].endswith("Export the ATAC as gene activity.")
    assert "allow_atac_mismatch" not in row["reason"] and "anyway" not in row["reason"]


# ------------------------------------------------------ RunResult counts, dtypes
def _rec(method, status, **kw):
    return {"method": method, "category": "vertical", "dataset": "MYMULTIOME",
            "status": status, "n_tunable": 3, **kw}


def test_repr_counts_a_named_skipped_method_once():
    recs = [_rec("Matilda", "FAIL", error="x")] + [
        _rec(m, "SKIPPED", error="blocked", requested=True)
        for m in ("scMDC", "UnitedNet", "scMVP", "MIRA")]
    res = W.RunResult(recs, "MYMULTIOME", "vertical")
    assert (res.summary.status != "scored").sum() == 5
    # c1 (C38): one count per status
    assert repr(res) == "<RunResult vertical/MYMULTIOME: 1 failed, 4 skipped>"
    unnamed = W.RunResult([_rec("Matilda", "FAIL", error="x"),
                             _rec("scMDC", "SKIPPED", error="blocked")],
                            "MYMULTIOME", "vertical")
    assert repr(unnamed) == "<RunResult vertical/MYMULTIOME: 1 failed, 1 skipped>"


# -------------------------------------------------------- batch Series messages
@pytest.fixture
def mycite(tmp_path):
    d = tmp_path / "MYCITE"
    d.mkdir()
    bars = [f"c{i}" for i in range(20)]
    _h5(d / "rna.h5", GENES[:10], bars)
    _h5(d / "adt.h5", [f"p{i}" for i in range(4)], bars)
    _labels(d / "cty.csv", 20)
    return tmp_path, bars


def test_an_integer_index_is_named_as_row_numbers(mycite):
    root, bars = mycite
    meta = pd.DataFrame({"batch": [1, 2] * 10, "keep": [True, False] * 10})
    sub = meta.loc[meta.keep, "batch"]            # row numbers 0, 2, 4, ...: not a RangeIndex
    with pytest.raises(ValueError) as e:
        W._batch_vector(sub, "MYCITE", root)
    msg = str(e.value)
    assert msg.startswith("batch: the index holds row numbers (first: [0, 2, 4]), not "
                          "cell barcodes. "), msg
    assert msg.endswith("only when batch is already in the order of the label files of "
                        "MYCITE."), msg      # 0.4: labels_for is deprecated


def test_missing_and_foreign_ids_get_their_own_fix(mycite):
    root, bars = mycite
    short = pd.Series([1] * 5, index=bars[:5])
    with pytest.raises(ValueError) as e:
        W._batch_vector(short, "MYCITE", root)
    assert str(e.value).endswith("Give a batch id for every cell.")
    assert "to_numpy" not in str(e.value)
    foreign = pd.Series([1] * 20, index=[f"x{b}" for b in bars])
    with pytest.raises(ValueError) as e:
        W._batch_vector(foreign, "MYCITE", root)
    assert str(e.value).startswith("batch: 20 ids are not cells of MYCITE")
    assert "Rename the index to the dataset's barcodes, or pass batch.to_numpy() only " \
           "when batch is already in the order of" in str(e.value)
    one = pd.Series([1] * 20, index=bars[:19] + ["zz"])
    with pytest.raises(ValueError, match=r"^batch: 1 id is not a cell of MYCITE"):
        W._batch_vector(one, "MYCITE", root)
    assert "(s)" not in str(e.value)


def test_run_all_raises_names_the_batch_series_error():
    doc = inspect.getdoc(mtb.run)
    raises = doc.split("Raises\n------\n")[1].split("\n\n")[0]
    assert "a ``batch`` that does not fit the cells" in " ".join(raises.split())


# -------------------------------------------------------------- single builds
def test_a_single_build_env_reports_single(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "envs_dir", tmp_path / "envs")
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    for env, word in (("scmb_r", "gpu"), ("matilda", "gpu")):
        prefix = tmp_path / "envs" / env
        (prefix / "bin").mkdir(parents=True)
        (prefix / envs.FLAVOR_FILE).write_text(word + "\n")
    envs._conda_prefixes.cache_clear()
    assert envs._single_build("scmb_r") and not envs._single_build("matilda")
    assert envs.installed_flavor("scmb_r") == "single"      # a record saying gpu
    assert envs.installed_flavor("matilda") == "gpu"        # a real GPU build
    assert scripts.run_provenance("scmb_r")["env_flavor"] == "single"


def test_cite_example_cites_only_the_methods_that_finished():
    ex = inspect.getdoc(mtb.cite).split("Examples\n--------\n")[1].split("\n\n")[0]
    # 0.4: cite takes the RunResult and cites the methods that ran
    assert 'mtb.cite(mtb.load_run("out/"), format="bibtex")' in ex
    assert "list(res.summary.method)" not in ex
