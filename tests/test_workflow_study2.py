"""Study-2 workflow polish: readable ``reason`` (Noor), str-vs-list guard and
missing-dataset signal (P17), the data_dir note on scan(modalities=) (P03),
``plan_commands`` without an ``out_dir`` and its full-path citation (P16), and
the platform note in the "nothing is runnable" error (P05).

Host-independent: the env probe is pinned where a verdict depends on it.
"""
import inspect
import re
import warnings

import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


@pytest.fixture
def all_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)


# ----------------------------------------------------------------- Noor: short reason
def test_short_reason_strips_exception_prefix_row_prefix_and_absolute_paths():
    raw = ("FileNotFoundError: UnitedNet/D11/vertical: input files not found on disk: "
           "{'atac_gas': '/home/wen/data/D11/atac_gas.h5', 'rna_cty': "
           "'/home/wen/data/D11/rna_cty.csv'}. Available files in /home/wen/data/D11: "
           "['adt.h5', 'cty.csv', 'rna.h5']")
    got = W._short_reason(raw, "UnitedNet", "D11", "vertical")
    assert got == ("input files not found on disk: {'atac_gas': 'atac_gas.h5', "
                   "'rna_cty': 'rna_cty.csv'}. Available files in D11: "
                   "['adt.h5', 'cty.csv', 'rna.h5']")
    assert "/home" not in got and "FileNotFoundError" not in got
    assert W._short_reason("", "M", "D", "c") == ""


def test_short_reason_collapses_benchmark_host_only_to_one_sentence():
    raw = ("benchmark-host-only: script not published: method script not found at "
           "/media/disk2/x/main_SPIRAL_ori.py - this entrypoint is an absolute path on "
           "the benchmark host; the script is not part of the public scMultiBench "
           "repository, so it cannot be fetched (method_info(m)['availability']); "
           "FileNotFoundError: SPIRAL/D11/cross: spatial registration needs >=2 .h5ad "
           "slice files; found 0 in /media/disk2/data/D11")
    got = W._short_reason(raw, "SPIRAL", "D11", "cross")
    assert got == ("benchmark-host-only: script not published (see "
                   "method_info(m)['availability']); spatial registration needs >=2 "
                   ".h5ad slice files; found 0 in D11")
    assert "/media" not in got and len(got) < len(raw) / 2


def test_scan_reason_is_short_but_files_reason_is_verbatim(no_envs):
    from multibench import config
    df = mtb.scan("D11", "vertical")
    blocked = df[~df["files_ok"]]
    assert len(blocked) > 0
    root = str(config.DEFAULT.data_path)
    for _, r in blocked.iterrows():
        # the file half is verbatim: the exception class and the absolute
        # dataset path survive (a script-gate part, e.g. MIRA's missing
        # logger.py helper, may precede it - the two halves are joined by '; ')
        parts = r["files_reason"].split("; ")
        assert any(p.startswith("FileNotFoundError: ") for p in parts) and root in r["files_reason"]
        assert "FileNotFoundError" not in r["reason"] and root not in r["reason"]
        assert r["reason"].endswith("; " + r["env_reason"])      # env half verbatim
        assert "--packed --run" in r["reason"]                    # install command kept
    row = df[(df["method"] == "UnitedNet")].iloc[0]
    assert "{'atac_gas': 'atac_gas.h5', 'rna_cty': 'rna_cty.csv'}" in row["reason"]
    # the nothing-runnable error (built from `reason`) is shorter too
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "cross", methods=["SPIRAL"], out_dir="/tmp/unused", verbose=False)
    # On the benchmark host SPIRAL's absolute script EXISTS, so the block is
    # about the dataset, not the script, and naming the path is right there;
    # everywhere else the collapsed host-only reason must replace the path.
    from pathlib import Path as _P
    from multibench.engine import registry as _reg
    _script = _P(_reg.get("SPIRAL").variants[0].entrypoint)
    if not _script.exists():
        assert "/media/disk2" not in str(e.value) and "benchmark-host-only" in str(e.value)
    assert "SPIRAL" in str(e.value)


def test_scan_and_plan_docs_name_the_four_columns():
    for fn in (mtb.scan, mtb.plan):
        doc = inspect.getdoc(fn)
        assert '["method", "modalities", "runnable", "reason"]' in doc, fn.__name__
        assert "files_reason" in doc and "env_reason" in doc


# ----------------------------------------------------------------- P17: str is not a list
@pytest.mark.parametrize("call", [
    lambda: mtb.scan("D11", "vertical", methods="StabMap"),
    lambda: mtb.plan("D11", "vertical", methods="StabMap"),
    lambda: mtb.run_all("D11", "vertical", methods="StabMap", out_dir="/tmp/unused",
                        verbose=False),
    lambda: mtb.run_all("D11", "vertical", methods="StabMap", out_dir=None, dry_run=True,
                        verbose=False),
])
def test_methods_as_a_string_raises_pass_a_list(call):
    with pytest.raises(TypeError) as e:
        call()
    msg = str(e.value)
    assert "must be a list of ids" in msg and "methods=['StabMap']" in msg
    assert "unknown method 'S'" not in msg


def test_modalities_as_a_string_raises_pass_a_list():
    with pytest.raises(TypeError, match=r"modalities=\['rna'\]"):
        mtb.scan("D11", "vertical", modalities="rna")
    with pytest.raises(TypeError, match="must be a list"):
        mtb.run_all("D11", "vertical", modalities="rna", out_dir="/tmp/unused", verbose=False)


def test_run_all_real_run_raises_on_missing_dataset_before_any_dispatch(monkeypatch):
    """The dataset gate fires in scan(), before the per-method loop where
    inputs_for's fabricated-path warning would otherwise be the only signal."""
    monkeypatch.setattr(W, "_run", lambda **k: pytest.fail("dispatched on a missing dataset"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # no warning may stand in for the error
        with pytest.raises(FileNotFoundError, match="does not exist"):
            mtb.run_all("NO_SUCH_DATASET_XYZ", "vertical", out_dir="/tmp/unused",
                        dry_run=False, verbose=False)


# ----------------------------------------------------------------- P03: data_dir note
def test_scan_modalities_warns_about_excluded_data_dir_methods(all_envs):
    with pytest.warns(UserWarning) as rec:
        df = mtb.scan("D11", "cross", modalities=["rna", "adt"])
    msgs = [str(w.message) for w in rec if "directory-input" in str(w.message)]
    assert len(msgs) == 1, msgs
    for m in ("PASTE", "PASTE2", "GPSA", "SPIRAL"):
        assert m in msgs[0]
    assert "modalities=[]" in msgs[0]
    assert "(data_dir)" not in set(df["modalities"])      # rows unchanged: exact selector
    # modalities=[] selects exactly them, and no note is emitted
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d2 = mtb.scan("D11", "cross", modalities=[])
        assert set(d2["modalities"]) == {"(data_dir)"}
        assert {"PASTE", "PASTE2", "GPSA", "SPIRAL"} <= set(d2["method"])
        # a category without directory variants stays silent
        mtb.scan("D11", "vertical", modalities=["rna", "adt"])


# ----------------------------------------------------------------- P16: plan_commands
def test_plan_commands_works_without_out_dir(no_envs):
    df = W.plan_commands("D11", "vertical", methods=["Matilda"])
    ok = df[df["files_ok"]]
    assert len(ok) == 1
    cmd = ok["command"].iloc[0]
    assert "<out_dir>/Matilda_D11/" in cmd and "conda run -n matilda" in cmd
    assert W.OUT_DIR_PLACEHOLDER == "<out_dir>"
    # positional/keyword compatibility: the explicit form is unchanged
    df2 = W.plan_commands("D11", "vertical", out_dir="runs", methods=["Matilda"])
    assert "runs/Matilda_D11/" in df2[df2["files_ok"]]["command"].iloc[0]


def test_plan_and_run_all_docs_cite_plan_commands_full_path():
    for fn in (mtb.plan, mtb.run_all):
        flat = " ".join(inspect.getdoc(fn).split())          # docstrings wrap
        assert "mtb.workflow.plan_commands(dataset, category, data_path=..., out_dir=...)" \
            in flat, fn.__name__
        assert "'<out_dir>'" in flat
    doc = inspect.getdoc(W.plan_commands)
    assert "Parameters" in doc and "Returns" in doc and "'<out_dir>'" in doc


# ----------------------------------------------------------------- P05: platform note
def test_nothing_runnable_message_names_the_platform_off_linux(no_envs, monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem",
                        lambda: "method environments are linux-64 conda envs; this host is darwin/arm64")
    for kw in ({"methods": ["Matilda"]}, {}):
        with pytest.raises(ValueError) as e:
            mtb.run_all("D11", "vertical", out_dir="/tmp/unused", verbose=False, **kw)
        msg = str(e.value)
        assert "\nNote: method environments are linux-64" in msg
        assert "run methods on a Linux host" in msg
        assert not re.search(r"\n  Note", msg)          # never looks like a variant line
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir="/tmp/unused", verbose=False)
    assert "Note: method environments" not in str(e.value)
