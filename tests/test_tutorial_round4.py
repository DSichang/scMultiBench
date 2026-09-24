"""The tutorials after fix round 4 of the student study.

Round 4 changed what the notebooks tell a reader in five places:

- R4-01: naming a method in ``methods=`` no longer runs it on the other ATAC
  representation, so the section-3 rule on ``runnable`` now holds for the
  named-method calls the run cells make.
- R4-04: ``run_all`` records a blocked method as ``SKIPPED``; the reason is in
  ``res.failures`` for a method that ``methods=`` named, which every tutorial
  run does.
- R4-08: an environment without a CPU archive has a single build, so a CPU
  runtime does not get a smaller build of every environment.
- R4-13: a dry run's printed command alone fails in a job script when the
  method reads files that ``mtb.run`` writes first; and the batch advice
  names ``run_all(batch=...)`` next to ``evaluate(batch=...)``.

Each prose test reads the committed notebooks and checks the same fact on the
live package, so a later package change that makes the sentence untrue fails
here too.
"""
import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
CATS = list(GEN.SCEN)
TUTORIALS = [f"tutorial_{c}" for c in CATS]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _markdown(name):
    return "\n".join(src for kind, src in _cells(name) if kind == "markdown")


def _code(name):
    return "\n".join(src for kind, src in _cells(name) if kind == "code")


def _visible(md):
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ----------------------------------------------------------- R4-04: SKIPPED
SKIP_SENTENCE = ("`res.failures` says why a run failed or a method you named was "
                 "skipped.")


@pytest.mark.parametrize("name", TUTORIALS)
def test_troubleshooting_says_where_the_reason_of_a_skipped_method_is(name):
    """Visible under the Troubleshooting heading: a method the run cell names
    and ``run_all`` skips shows ``SKIPPED`` in ``res.summary`` with no reason
    column there; ``res.failures`` holds the reason."""
    md = next(src for kind, src in _cells(name)
              if kind == "markdown" and src.startswith("## Troubleshooting"))
    assert SKIP_SENTENCE in _visible(md), md
    assert "`scan`'s `reason` column says why a method is not runnable." in _visible(md)
    assert "When a run fails, `res.failures` holds the error." not in md


def test_a_named_method_without_its_environment_is_in_failures_with_the_reason(
        tmp_path, monkeypatch):
    """The sentence on the live package: the vertical tutorial's run of its
    three methods on D11, with one environment missing, lists that method as
    SKIPPED in ``res.failures`` and gives the install command as the reason."""
    import multibench as mtb
    from multibench import config
    from multibench import workflow as W
    from multibench.engine import envs, registry
    if not (config.DEFAULT.data_path / "D11").is_dir():
        pytest.skip("D11 is not on disk")
    trio = GEN.SCEN["vertical"]["own_trio"]
    missing = envs.group_for(trio[-1])
    every = frozenset(envs.group_for(m) for m in registry.list_methods())
    monkeypatch.setattr(W, "_installed_envs", lambda: every - {missing})
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)

    class _Res:
        output = np.zeros((2864, 5))
    monkeypatch.setattr(W, "_run", lambda *a, **k: _Res())
    res = _quiet(mtb.run_all, "D11", "vertical", tmp_path / "out", methods=trio,
                 evaluate=False, verbose=False)
    status = res.summary.set_index("method")["status"]
    assert status[trio[-1]] == "SKIPPED"
    fails = res.failures.set_index("method")
    assert list(fails.index) == [trio[-1]]
    assert fails.loc[trio[-1], "status"] == "SKIPPED"
    assert f"multibench env install --methods {trio[-1]}" in fails.loc[trio[-1], "error"]


# ------------------------------------------------- R4-08: single-build envs
CPU_RUNTIME = "On a CPU runtime, training methods are much slower."


@pytest.mark.parametrize("name", TUTORIALS)
def test_the_colab_note_does_not_promise_a_cpu_build_of_every_environment(name):
    """The flag cell states the download size on a CPU host and on a GPU
    host; the Colab note keeps only the run-time consequence."""
    md = _markdown(name)
    assert CPU_RUNTIME in md
    assert "CPU builds are installed" not in md
    flag = next(src for kind, src in _cells(name)
                if kind == "code" and src.lstrip().startswith("# False:"))
    assert "to download on a CPU host" in flag and "on a GPU host" in flag


def test_some_tutorial_environment_has_a_single_build():
    """Why the note no longer says a CPU runtime gets smaller CPU builds: a
    tutorial installs an environment whose CPU plan takes the one archive it
    has (scmb_r)."""
    import multibench as mtb
    single = set()
    for cat, s in GEN.SCEN.items():
        rows = mtb.env.install(s["own_trio"], category=cat, flavor="cpu")
        single |= {r["env"] for r in rows if r["flavor"] != "cpu"}
    assert single, "every tutorial env has a CPU build: the plain sentence would do"


# --------------------------------------------- R4-13: the dry-run command
@pytest.mark.parametrize("name", TUTORIALS)
def test_troubleshooting_does_not_hand_out_the_dry_run_command_to_run_alone(name):
    md = _markdown(name)
    assert "prints the command to run there" not in md
    row = next(line for line in md.splitlines()
               if line.startswith("| `env_ok` False on macOS or Windows |"))
    assert row.endswith("| methods run only on Linux: run the same calls there |"), row


def test_end_to_end_does_not_say_the_printed_command_runs_on_its_own():
    md = _markdown("tutorial_end_to_end")
    assert "prepare the command on a laptop and run it on Linux" not in md
    assert ("It works on any computer, so you can check a call on a laptop before "
            "you run it on Linux.") in md


def test_a_dry_run_command_alone_can_fail():
    """The package fact behind both rewrites: for a method that reads a file
    ``mtb.run`` writes first, the dry-run note says the command alone fails."""
    from multibench.engine import runner as R
    plan = {"rna": {"value": "/o/inputs/rna.h5", "convert": True, "normpeaks_from": None}}
    note = R._prepared_note(plan, "/o", "X")
    assert note.startswith("the command reads inputs/rna.h5.")
    assert "The printed command alone fails in a job script." in note


# ---------------------------------------------- R4-13 / R4-02: batch advice
BATCH_ADVICE = ("Keep several samples in one folder, without `batch=`. To score the "
                "batch mixing, pass the batch column to `run_all(batch=...)` or "
                "`evaluate(batch=...)`.")


def test_vertical_export_note_names_both_batch_calls():
    assert BATCH_ADVICE in _markdown("tutorial_vertical")


def test_the_package_gives_the_same_batch_advice(tmp_path):
    """export_dataset(batch=) on a vertical dataset names the same two calls."""
    import anndata as ad
    import multibench as mtb
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.poisson(1.0, size=(20, 10)).astype(float))
    a.obs["celltype"] = rng.choice(["T", "B"], 20)
    a.obs["batch"] = rng.choice(["s1", "s2"], 20)
    a.obsm["protein"] = rng.poisson(3.0, size=(20, 4)).astype(float)
    with pytest.raises(ValueError) as e:
        _quiet(mtb.io.export_dataset, a, tmp_path / "V", rna="X", adt="obsm:protein",
               labels="obs:celltype", batch="obs:batch", category="vertical")
    assert "run_all(batch=...)" in str(e.value) and "evaluate(batch=...)" in str(e.value)


# ------------------------------------------------ R4-01: named methods
def test_runnable_sentence_holds_for_the_named_method_calls_of_the_run_cells(
        tmp_path, monkeypatch):
    """The run cells pass ``methods=`` to scan and run_all. On a D46-like
    folder whose ATAC holds gene activity, the mosaic methods stay not
    runnable when named, and run_all's plan agrees."""
    import anndata as ad
    import multibench as mtb
    from multibench import workflow as W
    from multibench.engine import envs, registry
    every = frozenset(envs.group_for(m) for m in registry.list_methods())
    monkeypatch.setattr(W, "_installed_envs", lambda: every)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    rng = np.random.default_rng(0)

    def batch(n):
        a = ad.AnnData(X=rng.poisson(1.0, size=(n, 40)).astype(float))
        a.var_names = [f"gene{i}" for i in range(40)]
        a.obs["celltype"] = rng.choice(["T", "B", "NK"], n)
        return a
    b1, b2, b3 = batch(100), batch(80), batch(60)
    b1.obsm["protein"] = rng.poisson(3.0, size=(100, 12)).astype(float)
    b1.uns["protein_names"] = [f"CD{i}" for i in range(12)]
    b2.obsm["gas"] = rng.poisson(0.3, size=(80, 50)).astype(float)
    b2.uns["gas_names"] = [f"GENE{i}" for i in range(50)]
    folder = tmp_path / "MYMOSAIC"
    kw = dict(labels="obs:celltype", category="mosaic")
    _quiet(mtb.io.export_dataset, b1, folder, adt="obsm:protein", batch_index=1, **kw)
    _quiet(mtb.io.export_dataset, b2, folder, atac="obsm:gas", atac_kind="gene_activity",
           batch_index=2, **kw)
    _quiet(mtb.io.export_dataset, b3, folder, batch_index=3, **kw)
    trio = GEN.SCEN["mosaic"]["own_trio"]
    sc = _quiet(mtb.scan, "MYMOSAIC", "mosaic", methods=trio, data_path=tmp_path,
                verbose=False).set_index("method")
    plan = _quiet(mtb.run_all, "MYMOSAIC", "mosaic", tmp_path / "out", methods=trio,
                  data_path=tmp_path, dry_run=True, verbose=False).set_index("method")
    for df in (sc, plan):
        for m in trio:
            r = df.loc[m]
            assert r.files_ok and r.env_ok and not r.runnable, (m, r.reason)
            assert r.reason.startswith("needs peak ATAC; atac2.h5 holds gene activity"), r.reason
    assert GEN.runnable_sentence("mosaic") in _visible(_markdown("tutorial_mosaic"))


# ------------------------------------------------ no internal names
@pytest.mark.parametrize("name", TUTORIALS)
def test_the_coverage_cell_names_no_registry(name):
    code = _code(name)
    assert "registry" not in code
    assert 'print(f"  {m}: not in this package")' in code
