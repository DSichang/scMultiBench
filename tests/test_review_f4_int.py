"""Findings of the review of the round-4 integration (wp/f4_int).

- A peak file whose names are neither chr:start-end nor the folder's gene
  names (peak_0) is not gene activity: MultiMAP, which never reads the names,
  stays runnable; GLUE and Seurat_v3 stay blocked by the peak-name check.
  Methods of either kind get a caveat that does not block.
- The ATAC override ends the reason, after the env and GPU sentences.
- run_all with allow_atac_mismatch=True reports the caveat once, not again
  as a UserWarning; a direct mtb.run still warns.
- BatchResult's repr counts a named SKIPPED method once; summary keeps
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
from multibench import cli, config, discover
from multibench import workflow as W
from multibench.engine import envs, registry
from multibench.engine import resolve as RS
from multibench.engine import runner as R

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
GENES = [f"GENE{i}" for i in range(60)]


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Every env installed, a GPU, no method scripts on this machine, no ref."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)


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


# ------------------------------------------- peak_0 names are not gene activity
def test_peak_ids_leave_multimap_runnable_and_block_glue_and_seurat_v3(tmp_path, pinned):
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    sc = _quiet(mtb.scan, "LUNG_ids", "diagonal", data_path=root, verbose=False,
                methods=discover.find_methods("diagonal", atac="peak")).set_index("method")
    assert sc.loc["MultiMAP", "runnable"], sc.loc["MultiMAP", "reason"]
    assert sc.loc["MultiMAP", "reason"] == ""
    assert sc.loc["MultiMAP", "caveat"].startswith(
        "expects peaks; atac_peak.h5 holds names that are not chr:start-end (e.g. peak_0)")
    assert "gene activity" not in sc.loc["MultiMAP", "caveat"]
    for m in ("GLUE", "Seurat_v3"):
        assert not sc.loc[m, "runnable"], m
        assert sc.loc[m, "reason"].startswith("reads peak names such as chr1:100-200; "
                                              "atac_peak.h5 holds other names (e.g. "
                                              "peak_0)"), sc.loc[m, "reason"]


def test_peak_ids_in_a_vertical_folder_give_both_kinds_a_caveat(tmp_path, pinned):
    """atac.h5 names that are neither chr:start-end nor the RNA's genes: the
    peak methods run with a caveat, and so do the gene-activity methods."""
    root = _vertical(tmp_path, "MYMO_ids", [f"peak_{i}" for i in range(40)])
    sc = _quiet(mtb.scan, "MYMO_ids", "vertical", methods=["scMVP", "Matilda"],
                modalities=["rna", "atac"], data_path=root, verbose=False).set_index("method")
    assert sc["runnable"].all(), sc["reason"].tolist()
    assert sc.loc["scMVP", "caveat"].startswith(
        "expects peaks; atac.h5 holds names that are not chr:start-end (e.g. peak_0)")
    assert sc.loc["Matilda", "caveat"].startswith(
        "expects gene activity; atac.h5 holds names that are not the RNA's genes (e.g. "
        "peak_0)")


def test_gene_names_in_the_peak_file_still_block_multimap(tmp_path, pinned):
    """Names that are the folder's genes are gene activity, whatever their case."""
    root = _diagonal(tmp_path, "LUNG_ga", [g.lower() for g in GENES])
    row = _quiet(mtb.scan, "LUNG_ga", "diagonal", methods=["MultiMAP"], data_path=root,
                 verbose=False).iloc[0]
    assert not row["runnable"]
    assert row["reason"].startswith("needs peak ATAC; atac_peak.h5 holds gene activity. ")


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


# ------------------------------------------------- the override ends the reason
def test_the_override_is_the_last_clause_of_a_gpu_blocked_row(tmp_path, pinned,
                                                              monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    row = _quiet(mtb.scan, "MU_PEAK", "vertical", methods=["UnitedNet"],
                 modalities=["rna", "atac"], data_path=root, verbose=False).iloc[0]
    assert not row["runnable"]
    assert "needs an NVIDIA GPU" in row["reason"]
    assert row["reason"].endswith("or pass allow_atac_mismatch=True to run UnitedNet anyway.")


# ------------------------------------------------ one report of an allowed caveat
class _Stop(RuntimeError):
    pass


def test_run_all_with_the_override_does_not_warn_again(tmp_path, pinned, monkeypatch,
                                                      capsys):
    """The caveat is in the log and the record; the runner's warning is not repeated."""
    def stop(spec, params):
        raise _Stop("stand-in env: stop before the method starts")
    monkeypatch.setattr(R, "cpu_params_for", stop)
    root = _vertical(tmp_path, "MU_PEAK", PEAKS)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = mtb.run_all("MU_PEAK", "vertical", tmp_path / "out", methods=["Matilda"],
                          modalities=["rna", "atac"], data_path=root,
                          allow_atac_mismatch=True)
    assert not [w for w in rec if "expects gene activity" in str(w.message)], \
        [str(w.message) for w in rec]
    assert "[run_all]   Matilda expects gene activity; atac.h5 holds peaks" in \
        capsys.readouterr().out
    assert res.records[0]["status"] == "FAIL" and "stand-in env" in res.records[0]["error"]
    # a direct mtb.run still warns, once
    inp = mtb.inputs_for("MU_PEAK", "vertical", "Matilda", modalities=["rna", "atac"],
                         data_path=root)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(_Stop):
            mtb.run("Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"))
    hits = [w for w in rec if "expects gene activity" in str(w.message)]
    assert len(hits) == 1 and issubclass(hits[0].category, UserWarning)


# ------------------------------------------------------ BatchResult counts, dtypes
def _rec(method, status, **kw):
    return {"method": method, "category": "vertical", "dataset": "MYMULTIOME",
            "status": status, "n_tunable": 3, **kw}


def test_repr_counts_a_named_skipped_method_once():
    recs = [_rec("Matilda", "FAIL", error="x")] + [
        _rec(m, "SKIPPED", error="blocked", requested=True)
        for m in ("scMDC", "UnitedNet", "scMVP", "MIRA")]
    res = W.BatchResult(recs, "MYMULTIOME", "vertical")
    assert len(res.failures) == 5
    assert repr(res) == ("<BatchResult vertical/MYMULTIOME: 0/5 with metrics, "
                         "4 skipped (4 named), 1 failed>")
    unnamed = W.BatchResult([_rec("Matilda", "FAIL", error="x"),
                             _rec("scMDC", "SKIPPED", error="blocked")],
                            "MYMULTIOME", "vertical")
    assert repr(unnamed) == ("<BatchResult vertical/MYMULTIOME: 0/2 with metrics, "
                             "1 skipped, 1 failed>")


def test_summary_keeps_n_batches_whole_next_to_skipped_rows(tmp_path):
    ok = _rec("Matilda", "CHAIN_OK", n_batches=1, labels_used=["cty.csv"],
              metrics={"ARI": 0.5}, run_sec=0.2, output_kind="embedding",
              emb_shape=[120, 4])
    res = W.BatchResult([ok, _rec("scMDC", "SKIPPED", error="blocked")],
                        "MYMULTIOME", "vertical")
    sm = res.summary.set_index("method")
    assert str(sm["n_batches"].dtype) == "Int64" and sm.loc["Matilda", "n_batches"] == 1
    res.save(tmp_path / "out")
    line = next(l for l in (tmp_path / "out" / "summary.csv").read_text().splitlines()
                if l.startswith("Matilda,"))
    assert ",1," in line and ",1.0," not in line, line


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
    assert "only when batch is already in the order of mtb.labels_for('MYCITE')" in msg


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
    doc = inspect.getdoc(mtb.run_all)
    raises = doc.split("Raises\n------\n")[1].split("\n\n")[0]
    assert "A ``batch`` Series holds ids that are not cells of the dataset." in raises


# ------------------------------------------- --strict reads the row on the disk
def test_strict_line_of_a_named_method_reads_the_row_with_its_files(tmp_path, pinned,
                                                                    capsys):
    root = _vertical(tmp_path, "MYMO2", PEAKS)          # Multiome: no adt.h5
    rc = _quiet(cli.main, ["scan", "MYMO2", "--category", "vertical", "--data-path",
                           str(root), "--methods", "Matilda", "--strict"])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("error: --strict: 0 of 2 rows are runnable. Input files are "
                          "missing in 1. The ATAC kind is wrong in 1."), err
    assert re.search(r"^  Matilda: needs gene-activity ATAC; atac\.h5 holds peaks\. "
                     r"Export the ATAC as gene activity, or pass --allow-atac-mismatch "
                     r"to run Matilda anyway\.$", err, re.M), err
    assert "row(s)" not in err and ";" not in err.splitlines()[0]


# -------------------------------------------------- run-all: earlier record kept
def test_failed_line_says_when_the_folder_kept_an_earlier_record(tmp_path):
    ok = _rec("StabMap", "CHAIN_OK", n_batches=1, labels_used=["cty.csv"],
              metrics={"ARI": 0.5})
    out = tmp_path / "runs"
    W.BatchResult([ok], "MYMULTIOME", "vertical").save(out)
    again = W.BatchResult([_rec("StabMap", "SKIPPED", error="blocked", requested=True)],
                          "MYMULTIOME", "vertical")
    again.save(out)
    line = cli._failed_line(again, out / "failures.csv")
    assert line == "# 1 of 1 method failed: StabMap (SKIPPED, earlier record kept)."
    fail = W.BatchResult([_rec("Matilda", "FAIL", error="boom")], "MYMULTIOME", "vertical")
    fail.save(out)
    assert cli._failed_line(fail, out / "failures.csv") == (
        f"# 1 of 1 method failed: Matilda (FAIL). See {out / 'failures.csv'}.")


# -------------------------------------------------------------- single builds
def test_a_single_build_env_reports_single(tmp_path, monkeypatch):
    monkeypatch.setattr(config.DEFAULT, "envs_dir", tmp_path / "envs")
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    for env, word in (("scmb_r", "gpu"), ("matilda", "gpu")):
        prefix = tmp_path / "envs" / env
        (prefix / "bin").mkdir(parents=True)
        (prefix / envs.FLAVOR_FILE).write_text(word + "\n")
    envs._conda_prefixes.cache_clear()
    assert envs._single_build("scmb_r") and not envs._single_build("matilda")
    assert envs.installed_flavor("scmb_r") == "single"      # a record saying gpu
    assert envs.installed_flavor("matilda") == "gpu"        # a real GPU build
    assert config.run_provenance("scmb_r")["env_flavor"] == "single"


# ------------------------------------------------------------ plain vocabulary
def test_no_user_visible_text_says_registry(monkeypatch, capsys):
    for argv in (["list", "--help"], ["cite", "--help"], ["evaluate", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "registry" not in capsys.readouterr().out.lower(), argv
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "macOS")
    cli._platform_note()
    err = capsys.readouterr().err
    assert "The method list, stored results" in err and "registry" not in err
    assert inspect.getdoc(mtb.list_methods).startswith("Return the method ids")
    assert "the package does not know, such as your own" in " ".join(
        inspect.getdoc(mtb.to_long).split())
    for fn in (mtb.run, mtb.inputs_for, mtb.labels_for, mtb.method_info, mtb.params_for,
               mtb.sweep, mtb.env.recipe, mtb.to_long, mtb.list_methods, mtb.recommend,
               mtb.load_results, mtb.catalog.methods, mtb.catalog.canonical_id,
               mtb.env.status, mtb.cite):
        assert "registry" not in inspect.getdoc(fn).lower(), fn.__name__


def test_cite_example_cites_only_the_methods_that_finished():
    ex = inspect.getdoc(mtb.cite).split("Examples\n--------\n")[1].split("\n\n")[0]
    assert "status not in ['SKIPPED', 'FAIL', 'TIMEOUT']" in ex
    assert "list(res.summary.method)" not in ex
