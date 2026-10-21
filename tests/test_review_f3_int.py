"""Findings of the review of the round-3 integration (wp/f3_int).

- The wrong-ATAC rule also blocks GLUE and Seurat_v3 when their peak file
  holds names the rewrite cannot turn into chr:start-end; the peak-name
  caveat has no subject, like the others (R3-02 / R3-04).
- A method named in methods= keeps the row its roles spell, as scan and
  method_info show them; a representation token that drops a named method
  says why (R3-01).
- Dry runs print the caveat of each row the sweep would run, without the
  notes on starting a method, and the scripts note once (R3-02).
- A scripts-ref mismatch is its own blocker: files_ok stays True, --strict
  counts it apart, the dry run keeps its commands (R3-10).
- run-all --dry-run --batch checks the length; a batch in the dataset's cell
  order is put in each method's cell order (R3-09).
- The runnable sets of the demo datasets are pinned, D28's vertical change
  included (R3-02).
- Smaller: the empty summary's columns, the evaluate count messages per
  flag, the scan error's list of methods, the unpacked-size check, the
  export_dataset wording, load_results on a missing file.

The env probe is pinned (every env installed, a GPU present) where a verdict
depends on it.
"""
import inspect
import json
import re
import shutil
import subprocess
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry
from tests.conftest import full_scan

#: the caveat of a gene-activity method given peaks in atac.h5 (after the method)
GAS_CAV = ("needs gene-activity ATAC. The features of atac.h5 look like chr:start-end, "
           "so it holds peaks.")

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Every env installed, a GPU, no method scripts on this machine, no ref."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "repo_path", tmp_path / "no_scripts")
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


def _diagonal(root, name, peaks, n=20):
    d = root / name
    d.mkdir(parents=True)
    rna, atac = [f"r{i}" for i in range(n)], [f"a{i}" for i in range(n)]
    genes = [f"GENE{i}" for i in range(60)]
    _h5(d / "rna.h5", genes, rna)
    _h5(d / "atac_peak.h5", peaks, atac)
    _h5(d / "atac_gas.h5", genes, atac)
    pd.DataFrame({"x": ["A"] * n}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["A"] * n}).to_csv(d / "atac_cty.csv", index=False)
    return root


def _vertical_peaks(root, name="MU_PEAK", n=60):
    d = root / name
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], bars)
    _h5(d / "atac.h5", [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)], bars)
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / "cty.csv", index=False)
    return root


# ---------------------------------------------------------- wrong peak names block
def test_gene_names_in_the_peak_file_block_glue_seurat_v3_and_multimap(tmp_path, pinned):
    root = _diagonal(tmp_path, "LUNG_ga", [f"GENE{i}" for i in range(60)])
    sc = _quiet(full_scan, "LUNG_ga", "diagonal", data_path=root, verbose=False)
    # the rows whose files the folder holds: GLUE's row for several batches has none
    sc = sc[sc["files_ok"]].set_index("method")
    assert sc.index.is_unique
    for m in ("GLUE", "Seurat_v3", "MultiMAP"):
        assert not sc.loc[m, "runnable"] and sc.loc[m, "files_ok"], m
        assert "allow_atac_mismatch" not in sc.loc[m, "reason"], m
    for m in ("GLUE", "Seurat_v3"):
        assert sc.loc[m, "reason"] == (
            f"{m} reads peak names such as chr1:100-200. atac_peak.h5 holds other names, "
            f"for example GENE0. They are genes: export the ATAC as gene activity."), m
    assert sc.loc["MultiMAP", "reason"] == (
        "MultiMAP needs peak ATAC, and atac_peak.h5 holds gene activity. Export the ATAC "
        "as peaks.")
    # R4-01: naming the method keeps it blocked
    for m in ("GLUE", "Seurat_v3", "MultiMAP"):
        named = _quiet(full_scan, "LUNG_ga", "diagonal", methods=[m], data_path=root,
                       verbose=False)
        assert not named["runnable"].any(), m
        assert named["files_ok"].sum() == 1, m


# ---------------------------------------------------------- scripts ref
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _cty(path, labels):
    pd.DataFrame({"x": labels}).to_csv(path, index=False)


def test_a_dataset_order_batch_follows_each_methods_cell_order(tmp_path, monkeypatch):
    """uniPort puts its ATAC cells first: the batch ids go with them."""
    d = tmp_path / "DIAG"
    d.mkdir()
    _cty(d / "rna_cty.csv", ["A", "B"] * 5)
    _cty(d / "atac_cty.csv", ["A", "B"] * 4)
    batch = np.array(["rna_s"] * 10 + ["atac_s"] * 8)          # labels_for order
    seen = {}

    def fake(emb, category, cands, batch=None, metrics=None, **kw):
        names, lab, bat = next(c for c in cands if c[0][0] == "atac_cty.csv")
        seen["bat"] = np.asarray(batch if batch is not None else bat)
        val = pd.DataFrame({"Value": {"ARI": 0.5}})
        return names, val, [{"order": names, "ARI": 0.5}]
    monkeypatch.setattr(W, "_evaluate_best_order", fake)
    rec = {"method": "uniPort"}
    W._score_record(rec, np.zeros((18, 3)), "DIAG", "diagonal", tmp_path,
                    registry.get("uniPort").variants[0], batch=batch)
    assert list(seen["bat"]) == ["atac_s"] * 8 + ["rna_s"] * 10
    assert rec["batch_source"] == "user" and rec["n_batches"] == 2


def test_a_dataset_order_batch_is_cut_to_the_batches_a_method_reads(tmp_path, monkeypatch):
    """UINMF reads batches 1-2 of 3: it gets the ids of those cells."""
    d = tmp_path / "CROSS"
    d.mkdir()
    for i, n in ((1, 4), (2, 6), (3, 5)):
        _cty(d / f"cty{i}.csv", ["A", "B"] * (n // 2) + ["A"] * (n % 2))
    batch = np.array(["d1"] * 4 + ["d2"] * 6 + ["d3"] * 5)
    seen = {}

    def fake(emb, category, cands, batch=None, metrics=None, **kw):
        names, lab, bat = next(c for c in cands if c[0] == ["cty1.csv", "cty2.csv"])
        seen["bat"] = np.asarray(batch if batch is not None else bat)
        return names, pd.DataFrame({"Value": {"ARI": 0.5}}), [{"order": names, "ARI": 0.5}]
    monkeypatch.setattr(W, "_evaluate_best_order", fake)
    rec = {"method": "UINMF"}
    W._score_record(rec, np.zeros((10, 3)), "CROSS", "cross", tmp_path,
                    registry.get("UINMF").variants[0], batch=batch)
    assert list(seen["bat"]) == ["d1"] * 4 + ["d2"] * 6
    with pytest.raises(ValueError, match="batch has 7 entries, embedding has 10 cells"):
        W._score_record({"method": "UINMF"}, np.zeros((10, 3)), "CROSS", "cross",
                        tmp_path, registry.get("UINMF").variants[0], batch=batch[:7])


def test_batch_docs_name_the_label_file_order():
    p = " ".join(inspect.getdoc(mtb.run).split("Parameters", 1)[1]
                 .split("Returns", 1)[0].split())
    # 0.4: labels_for is deprecated, so the doc no longer names it
    assert ("One batch per cell; a Series or a barcode-indexed CSV is aligned by barcode. "
            "``None`` = each cell's label file.") in p
    assert "labels_for" not in p


# ---------------------------------------------------------- pinned demo sets
PINNED = {
    "D11": ["Concerto|vertical|rna+adt", "MOFA2|vertical|rna+adt",
            "Matilda|vertical|rna+adt", "Multigrate|vertical|rna+adt",
            "Seurat_WNN|vertical|rna+adt", "UINMF|vertical|rna+adt",
            "VIMCCA|vertical|rna+adt", "moETM|vertical|rna+adt", "scMDC|vertical|rna+adt",
            "scMM|vertical|rna+adt", "scMSI|vertical|rna+adt", "scMoMaT|vertical|rna+adt",
            "sciPENN|vertical|rna+adt", "totalVI|vertical|rna+adt"],
    # R3-02: moETM, scMM and iPOLNG read peaks; D28's atac_gas.h5 holds gene
    # activity, so their vertical rna+atac_gas rows are no longer runnable
    # scMDC reads peaks too (as the benchmark ran it), so its row joined them
    "D28": ["Conos|diagonal|rna+atac_gas", "GLUE|diagonal|rna+atac_peak",
            "MultiMAP|diagonal|rna+atac_peak+atac_gas", "Portal|diagonal|rna+atac_gas",
            "SCALEX|diagonal|rna+atac_gas", "Seurat_v3|diagonal|rna+atac_peak+atac_gas",
            "VIPCCA|diagonal|rna+atac_gas", "iNMF|diagonal|rna+atac_gas",
            "online_iNMF|diagonal|rna+atac_gas", "scBridge|diagonal|(data_dir)",
            "scJoint|diagonal|rna+atac_gas",
            "sciCAN|diagonal|rna+atac_gas", "uniPort|diagonal|rna+atac_gas"],
    # the variants of the article's 13 tasks: every method of the task of
    # each dataset, in mosaic and cross
    "D45": ["Cobolt|mosaic|rna1+rna2+atac2+atac3", "MultiVI|mosaic|rna1+atac3+rna2+atac2",
            "Multigrate|mosaic|rna1+rna2+atac2+atac3", "SMILE|mosaic|rna2+atac2+rna1+atac3",
            "StabMap|mosaic|rna1+rna2+atac2+atac3", "scMoMaT|mosaic|rna1+rna2+atac2+atac3"],
    "D46": ["Multigrate|mosaic|rna1+rna2+rna3+adt1+atac2",
            "StabMap|mosaic|rna1+rna2+rna3+adt1+atac2",
            "UINMF|mosaic|rna1+rna2+rna3+adt1+atac2",
            "scMoMaT|mosaic|rna1+rna2+rna3+adt1+atac2"],
    # D52's files also fit the mosaic pattern RNA | RNA + ADT | ADT, and both
    # of UINMF's cross variants (two and three batches)
    "D52": ["Concerto|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "MOFA2|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "Multigrate|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "Multigrate|mosaic|rna1+rna2+adt2+adt3",
            "StabMap|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "StabMap|mosaic|rna1+rna2+adt2+adt3",
            "UINMF|cross|rna1+rna2+adt1+adt2",
            "UINMF|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "scMoMaT|mosaic|rna1+rna2+adt2+adt3",
            "scMDC|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "scMM|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "scMoMaT|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "sciPENN|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "totalVI|cross|rna1+rna2+rna3+adt1+adt2+adt3"],
}


@pytest.mark.parametrize("dataset", sorted(PINNED))
def test_demo_datasets_keep_their_pinned_runnable_rows(dataset, pinned):
    df = _quiet(full_scan, dataset, verbose=False)
    got = sorted(f"{r.method}|{r.category}|{r.modalities}" for r in df[df.runnable].itertuples())
    assert got == sorted(PINNED[dataset])
    blocked = df[W._is_wrong_atac(df["reason"])]
    # the vertical peak methods read the atac role since 0.4.0 (X13): on D28 they no
    # longer take atac_gas.h5 for their peaks, so none is blocked for the wrong ATAC
    want = set()
    assert set(blocked["method"]) == want
    assert set(blocked["category"]) <= {"vertical"}


# ---------------------------------------------------------- smaller findings
def test_the_empty_summary_ends_like_a_full_one():
    empty = W.BatchResult([], "D11", "vertical").summary
    full = W.BatchResult([{"method": "Matilda", "status": "RUN_OK"}], "D11",
                         "vertical").summary
    # R5-03: reason is the last column (the diagnostics before caveat since 0.4.0, X35)
    assert list(empty.columns)[-2:] == ["caveat", "reason"]
    assert list(full.columns)[-2:] == ["caveat", "reason"]
    assert list(empty.columns) == list(full.columns)


def test_an_unpacked_size_below_the_archive_reads_unknown(tmp_path, monkeypatch):
    shipped = envs._SIZES_JSON
    table = tmp_path / "packed_sizes.json"
    table.write_text(json.dumps({"a": {"archive_bytes": 5, "unpacked_bytes": 2},
                                 "b": {"archive_bytes": 5, "unpacked_bytes": 9}}))
    monkeypatch.setattr(envs, "_SIZES_JSON", table)
    envs.packed_sizes.cache_clear()
    try:
        sizes = envs.packed_sizes()
    finally:
        envs.packed_sizes.cache_clear()
    assert sizes["a"]["unpacked_bytes"] is None and sizes["b"]["unpacked_bytes"] == 9
    # the shipped table: no new entry of that kind (scmb_scmm2-cpu awaits a
    # measurement on the Linux host)
    raw = json.loads(shipped.read_text())
    bad = {k for k, v in raw.items() if isinstance(v, dict)
           and isinstance(v.get("unpacked_bytes"), int)
           and v["unpacked_bytes"] < (v.get("archive_bytes") or 0)}
    assert bad <= {"scmb_scmm2-cpu"}


def test_export_notes_say_scan_skips_the_other_representation():
    notes = " ".join(inspect.getdoc(ingest.export_dataset).split())
    assert "reports the mismatch in its ``caveat`` column" not in notes


# ---------------------------------------------------------- docs pages
def _docs_root():
    import os
    from pathlib import Path
    root = os.environ.get("SCMULTIBENCH_DOCS")
    return Path(root) if root and Path(root).is_dir() else None


needs_docs = pytest.mark.skipif(_docs_root() is None, reason="SCMULTIBENCH_DOCS not set")


def _flat(page):
    return " ".join((_docs_root() / page).read_text().split())


@needs_docs
def test_pages_state_the_donor_count_cross_reads():
    """The Quickstart and the run guide say how many batches the cross
    patterns of the registry read."""
    from multibench import workflow as W
    from multibench.engine.schema import _batch_of
    reads = {}
    for roles, methods in W.batch_patterns("cross"):
        if any("atac" in r for r in roles):   # the page is about RNA + ADT donors
            continue
        top = max(_batch_of(r) for r in roles)
        for m in methods:
            reads.setdefault(m, set()).add(top)
    # UINMF has a variant for two batches and one for three
    assert reads.pop("UINMF") == {2, 3} and set(map(frozenset, reads.values())) == {
        frozenset({3})}
    # R4-16 moved the counts from the visible bullet to the Step 1 Details block
    step1 = _flat("quickstart.md").split("## Step 1:", 1)[1].split("## Step 2:", 1)[0]
    assert (f"The `cross` category has {len(reads) + 1} RNA + ADT methods, which integrate "
            f"the donors") in step1
    assert "These methods read batches 1-3." in step1
    assert "UINMF reads only the first two" not in step1
    assert ("The `cross` category integrates the donors as batches and reads a fixed number "
            "of them, three for RNA + ADT: see the") in step1
    assert ("With another number of donors, use `vertical`, as in the recipe "
            "[Another number of donors](tutorials/run.md#recipe-donors).") in step1
    run = _flat("tutorials/run.md")
    assert ('category="cross") ``` The cross methods for RNA + ADT read batches 1-3. '
            '[Dataset folder layout](#folder-layout) lists every batch pattern.') in run


@needs_docs
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_job_script_is_valid_bash(tmp_path):
    text = (_docs_root() / "installation.md").read_text()
    job = text.split('```bash title="job.sh (Slurm)"', 1)[1].split("```", 1)[0]
    script = tmp_path / "job.sh"
    script.write_text(job)
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert 'python run_one.py "$1"' in job


@needs_docs
def test_changes_page_follows_the_review_fixes():
    text = _flat("changes.md")
    for phrase in ("wrong ATAC kind, unreadable peak names and scripts not at "
                   "`MULTIBENCH_SCRIPTS_REF`",
                   "With `--dry-run`, a file of the wrong length exits with `1`.",
                   "iPOLNG, which read peaks, unless `methods=` names them.",
                   "GLUE and Seurat_v3 are not runnable in `scan` when the peak file",
                   "takes the ids in the cell order of `labels_for(dataset)`",
                   "the refusal of `mtb.run` starts with `Methods run only on Linux` and "
                   "names no install command.",
                   "`load_results` with a `result_path` that does not exist raises"):
        assert phrase in text, phrase
    assert "is one sentence that starts with" not in text
    assert 'With `"atac_gas"`, `scan(..., methods=["moETM"])` raises' not in text


@needs_docs
def test_discover_details_say_a_representation_token_keeps_the_readers():
    text = _flat("tutorials/discover.md")
    assert ("A representation token keeps every row whose method reads it, so "
            '`"atac_peak"` also keeps MultiMAP and Seurat_v3.') in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = set(full_scan("D28", "diagonal", modalities=["rna", "atac_peak"],
                           verbose=False)["method"])
    assert {"MultiMAP", "Seurat_v3"} <= got


@needs_docs
def test_deploy_gate_refuses_while_the_data_release_is_not_public(tmp_path, monkeypatch):
    """mtb.data.fetch reads RELEASE_URL; a private release answers 404."""
    import importlib
    import importlib.util
    fetch = importlib.import_module("multibench.data.fetch")     # the module, not mtb.data.fetch
    spec = importlib.util.spec_from_file_location(
        "deploy_gate", _docs_root().parent / "hooks" / "deploy_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    from pathlib import Path
    root = Path(mtb.__file__).resolve().parent.parent
    assert gate.checkout_release_url(root) == fetch.RELEASE_URL
    asked = []
    monkeypatch.setattr(gate, "asset_status", lambda url, timeout=15: asked.append(url) or 404)
    (msg,) = gate.data_problems(root)
    assert asked == [f"{fetch.RELEASE_URL}/D11.tar.gz"]
    assert msg.endswith("answers HTTP 404 to an anonymous request: make the data release "
                        "public before the deploy, or mtb.data.fetch and the site's "
                        "download steps fail")
    monkeypatch.setattr(gate, "asset_status", lambda url, timeout=15: 302)
    assert gate.data_problems(root) == []                    # GitHub redirects a public asset

    def offline(url, timeout=15):
        raise OSError("no network")
    monkeypatch.setattr(gate, "asset_status", offline)
    (msg,) = gate.data_problems(root)
    assert msg.startswith("could not reach ") and "cannot be checked" in msg
    assert gate.data_problems(tmp_path) == [
        f"no RELEASE_URL in {tmp_path / 'multibench' / 'data' / 'fetch.py'}"]
    assert "data_problems(checkout)" in (_docs_root().parent / "hooks" /
                                         "deploy_gate.py").read_text()


# ---- the Python part of tests that also ran the command line
def test_a_named_method_keeps_the_row_its_roles_spell(pinned, capsys):
    """moETM's own spelling, as scan and method_info show it (rna+atac since
    0.4.0, X13: the role names the peaks it reads)."""
    assert ['rna', 'atac'] in [s['modalities'] for s in mtb.method_info('moETM')['supports']]
    for call in (full_scan, lambda *a, **k: mtb.run(*a, dry_run=True, **k)):
        df = _quiet(call, 'D28', 'vertical', methods=['moETM'], modalities=['rna', 'atac'], verbose=False)
        assert list(df['modalities']) == ['rna+atac']
    df = _quiet(full_scan, 'D28', 'vertical', modalities=['rna', 'atac_gas'], verbose=False)
    assert 'moETM' not in set(df['method'])


def test_a_representation_token_that_drops_a_named_method_says_why(pinned, capsys):
    for call in (full_scan, lambda *a, **k: mtb.run(*a, dry_run=True, **k)):
        with pytest.raises(ValueError) as e:
            _quiet(call, 'D28', 'vertical', methods=['moETM'], modalities=['rna', 'gene_activity'], verbose=False)
        assert str(e.value) == "moETM does not read rna+gene_activity in vertical data. moETM reads peaks. Pass modalities=['rna', 'atac_peak'] or ['rna', 'atac'].", str(e.value)
        assert "method_info(m)['supports']" not in str(e.value)
    with pytest.raises(ValueError, match="Matilda reads peaks\\. Pass modalities=\\['rna', 'atac_peak'\\] or \\['rna', 'atac'\\]\\.$"):
        _quiet(full_scan, 'D11', 'vertical', methods=['Matilda'], modalities=['rna', 'atac_gas'], verbose=False)
    with pytest.raises(ValueError, match="SCALEX reads gene activity\\. Pass modalities=\\['rna', 'atac_gas'\\] or \\['rna', 'atac'\\]\\.$"):
        _quiet(full_scan, 'D28', 'diagonal', methods=['SCALEX'], modalities=['rna', 'atac_peak'], verbose=False)


def test_dry_run_prints_the_scripts_note_once_and_no_start_notes(pinned, capsys):
    _quiet(mtb.run, 'D11', 'vertical', dry_run=True)
    out = capsys.readouterr().out
    assert out.count('The method scripts are not in') == 1, out
    assert '[run] The method scripts are not in' in out
    _quiet(mtb.run, 'D28', 'diagonal', dry_run=True)
    out = capsys.readouterr().out
    assert 'the command reads' not in out and 'start the method with' not in out
    assert out.count('The method scripts are not in') == 1


def test_run_all_dry_run_checks_the_batch_length(tmp_path, pinned, capsys):
    short = tmp_path / 'short.csv'
    pd.DataFrame({'b': ['s1', 's2', 's1']}).to_csv(short, index=False)
    argv = ['run-all', 'D11', '--category', 'vertical', '--methods', 'totalVI', '--dry-run', '--batch']
    full = tmp_path / 'full.csv'
    pd.DataFrame({'b': np.tile(['s1', 's2'], 1432)}).to_csv(full, index=False)
    with pytest.raises(ValueError, match='^batch has 3 entries for the 2,864 cells of D11'):
        _quiet(mtb.run, 'D11', 'vertical', methods=['totalVI'], dry_run=True, batch=str(short))
