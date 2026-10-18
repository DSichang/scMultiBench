"""scan lists the variants whose modalities the folder holds (0.3.5).

Up to 0.3.4 ``mtb.scan(dataset, category)`` listed every variant of the
category: on a CITE-seq folder, 19 of 33 rows said that ``atac.h5`` is
missing. By default it now lists the variants whose modalities the folder
holds; ``modalities="all"`` gives the table of 0.3.4. A row whose input files
are all there is always listed, so the rows a run tries are the same.

The env probe is pinned where a verdict depends on it; ``_run`` is faked in
the real-run tests.
"""
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, registry
from multibench.engine import runner as R
from tests.conftest import full_scan

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
GENES = [f"GENE{i}" for i in range(40)]

#: spec 2.6: (dataset, category) -> (rows of 0.3.4, rows listed, of them with their files)
LISTED = {
    ("D11", "vertical"): (33, 14, 14),
    ("D27mini_vertical", "vertical"): (33, 14, 14),
    ("D22mini", "vertical"): (33, 33, 33),
    # 0.4.0 (X80): rows for numbered batches a one-batch folder lacks are not listed
    ("D27mini", "diagonal"): (18, 14, 14),
    ("D37mini", "diagonal"): (18, 18, 4),
    ("D38mini", "mosaic"): (16, 3, 3),
    ("D45mini", "mosaic"): (16, 6, 6),
    ("D46mini", "mosaic"): (16, 16, 4),
    ("D49mini", "mosaic"): (16, 16, 3),
    ("D52mini", "cross"): (30, 11, 11),
    ("D56mini", "cross"): (30, 9, 9),
    ("D58mini", "cross"): (30, 5, 5),
    ("D59mini", "cross"): (30, 30, 12),
    ("D11", None): (97, 15, 14),
}


def _core(attrs):
    """The attrs these tests pin; 0.4.0 added not_listed_batch and not_listed_methods."""
    return {k: attrs[k] for k in ("held", "not_listed", "not_held")}


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _on_disk(ds):
    if not (config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")


def _rows(df):
    return set(zip(df["method"], df["category"], df["modalities"]))


def _h5(path, feats, bars, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(len(feats), len(bars)))
                         .astype(float))
        g.create_dataset("features", data=np.array(feats, dtype="S32"))
        g.create_dataset("barcodes", data=np.array(bars, dtype="S32"))


def _labels(path, n):
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(path, index=False)


def _donors(root, name, donors, n=40):
    """A cross folder of RNA + ADT with one batch per donor."""
    d = root / name
    d.mkdir(parents=True)
    for b in range(1, donors + 1):
        bars = [f"d{b}c{i}" for i in range(n)]
        _h5(d / f"rna{b}.h5", GENES[:30], bars)
        _h5(d / f"adt{b}.h5", [f"p{i}" for i in range(8)], bars)
        _labels(d / f"cty{b}.csv", n)
    return root


@pytest.fixture
def no_envs(monkeypatch):
    """A Linux host without any method environment."""
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


@pytest.fixture
def linux(monkeypatch, tmp_path):
    """A Linux host with every environment and no GPU; no method scripts on
    this machine, no ref. ``_run`` is faked and records what it was given."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    calls = []

    class _Res:
        output = np.zeros((60, 5))

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append((method, "+".join(r for r in inputs if "cty" not in r)))
        return _Res()
    monkeypatch.setattr(W, "_run", fake_run)
    return calls


# ------------------------------------------------------------------ the row set
@pytest.mark.parametrize("ds,cat", list(LISTED), ids=lambda v: str(v))
def test_default_rows_are_a_subset_with_the_same_files_ok_rows(ds, cat):
    _on_disk(ds)
    every = _quiet(full_scan, ds, cat, modalities="all", verbose=False)
    listed = _quiet(full_scan, ds, cat, verbose=False)
    n_all, n_listed, n_files = LISTED[ds, cat]
    assert (len(every), len(listed), int(listed["files_ok"].sum())) == \
        (n_all, n_listed, n_files)
    assert len(_rows(listed)) == len(listed) and _rows(listed) <= _rows(every)
    key = ["method", "category", "modalities"]
    gone = every.merge(listed[key], on=key, how="left", indicator=True)
    gone = gone[gone["_merge"] == "left_only"]
    assert len(gone) == n_all - n_listed and not gone["files_ok"].any()
    # the rows with their files, and the rows that can run, are the same
    for flag in ("files_ok", "runnable"):
        assert _rows(listed[listed[flag]]) == _rows(every[every[flag]])
    # what is listed is listed as before: every value, in the same order
    same = every.merge(listed[key], on=key, how="inner")
    pd.testing.assert_frame_equal(same.reset_index(drop=True), listed, check_like=False)
    assert listed.attrs["not_listed"] + listed.attrs["not_listed_batch"] == n_all - n_listed


@pytest.mark.parametrize("ds,cat", [("D11", "vertical"), ("D28", "diagonal"),
                                    ("D45", "mosaic"), ("D52", "cross"), ("D11", None)])
def test_all_is_every_variant(ds, cat):
    df = _quiet(full_scan, ds, cat, modalities="all", verbose=False)
    want = {(s.id, c, "+".join(mods) or "(data_dir)") for s, _v, c, mods in W._variant_rows(cat)}
    assert _rows(df) == want and len(df) == len(want)
    assert df.attrs["not_listed"] == 0 and df.attrs["not_held"] == []


def test_all_is_the_one_string_modalities_takes():
    for call in (lambda m: full_scan("D11", "vertical", modalities=m, verbose=False),
                 lambda m: mtb.run_all("D11", "vertical", modalities=m, dry_run=True,
                                       verbose=False)):
        assert len(call("all")) == 33
        for other in ("rna", "ALL", "all "):
            with pytest.raises(TypeError) as e:
                call(other)
            assert str(e.value) == (f"modalities must be a list of ids, got the string "
                                    f"{other!r} - did you mean modalities=[{other!r}]?")
        # a list is the selection it was, with nothing left out
        df = call(["rna", "adt"])
        assert set(df["modalities"]) == {"rna+adt"} and df.attrs["not_listed"] == 0
        with pytest.raises(ValueError, match="all"):
            call(["all"])                       # a token list holds modality tokens


@pytest.mark.parametrize("mods", [None, "all"])
@pytest.mark.parametrize("ds,cat", [("D11", "vertical"), ("D52", "cross"),
                                    ("D27mini", "diagonal")])
def test_scan_and_dry_run_are_one_table(ds, cat, mods, capsys):
    _on_disk(ds)
    scan = _quiet(full_scan, ds, cat, modalities=mods, verbose=False)
    plan = _quiet(mtb.run_all, ds, cat, modalities=mods, dry_run=True, verbose=False)
    pd.testing.assert_frame_equal(scan, plan)
    assert scan.attrs == plan.attrs
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("ds,cat", [("D11", "vertical"), ("D52mini", "cross"),
                                    ("D27mini", "diagonal")])
def test_a_real_run_tries_the_same_rows(ds, cat, linux, tmp_path):
    _on_disk(ds)
    runs = {}
    for mods in (None, "all"):
        linux.clear()
        res = _quiet(mtb.run_all, ds, cat, tmp_path / f"out_{mods}", modalities=mods,
                     evaluate=False, verbose=False)
        runs[mods] = (sorted(linux), {r["method"]: r["status"] for r in res.records})
    assert runs[None] == runs["all"] and runs[None][0]
    # the rows tried are the rows scan calls runnable, less the nested ones
    plan = _quiet(full_scan, ds, cat, verbose=False)
    plan = plan[plan["runnable"]]
    tried = plan.drop(index=W._nested_rows(plan)[0])
    assert runs[None][0] == sorted(zip(tried["method"], tried["modalities"]))


def test_skipped_records_of_named_methods_are_the_same(linux, tmp_path):
    # no GPU: moETM has its files on D11 and is skipped; MIRA lacks atac.h5
    named = ["Matilda", "moETM", "MIRA"]
    records = {}
    for mods in (None, "all"):
        res = _quiet(mtb.run_all, "D11", "vertical", tmp_path / f"out_{mods}",
                     methods=named, modalities=mods, evaluate=False, verbose=False)
        records[mods] = {r["method"]: (r["status"], r["modalities"], r.get("error"))
                         for r in res.records}
    assert {m: s for m, (s, _, _) in records[None].items()} == \
        {"Matilda": "RUN_OK", "moETM": "SKIPPED", "MIRA": "SKIPPED"}
    assert {m: s for m, (s, _, _) in records["all"].items()} == \
        {m: s for m, (s, _, _) in records[None].items()}
    # a named method with a row whose files are there: the same record
    assert records[None]["moETM"] == records["all"]["moETM"]
    assert records[None]["moETM"][1] == ["rna", "adt"]
    # MIRA has no such row: it keeps its rows, so its record is the same here too
    assert records[None]["MIRA"] == records["all"]["MIRA"]


def test_the_log_of_a_real_run_names_the_listed_variants_only(linux, tmp_path, capsys):
    # Matilda runs on rna+adt. Its two rows that read ATAC, and moETM's one,
    # are not listed
    kw = dict(methods=["Matilda", "moETM"], evaluate=False)
    gpu = ('moETM needs an NVIDIA GPU, and this computer has none. See '
           'method_info("moETM")["requires_gpu"].')
    _quiet(mtb.run_all, "D11", "vertical", tmp_path / "out", **kw)
    log = capsys.readouterr().out
    assert [l for l in log.splitlines() if "skipping" in l] == [
        f"[run_all] skipping moETM: {gpu}"]
    assert "more row" not in log
    _quiet(mtb.run_all, "D11", "vertical", tmp_path / "out_all", modalities="all", **kw)
    log = capsys.readouterr().out.splitlines()
    assert log[:3] == [
        f"[run_all] skipping moETM (rna+adt): {gpu}",
        "[run_all] skipping moETM (rna+atac): moETM needs peak ATAC (atac.h5), which "
        f"is not in the folder. {gpu}",
        "[run_all] 2 more rows need files this folder does not have. mtb.scan shows them."]


# ------------------------------------------------------------------ no filter
def test_nothing_left_means_no_filter(tmp_path, no_envs):
    d = tmp_path / "RNA_ONLY"
    d.mkdir()
    _h5(d / "rna.h5", GENES[:30], [f"c{i}" for i in range(20)])
    df = full_scan("RNA_ONLY", "vertical", data_path=tmp_path, verbose=False)
    # no vertical variant reads RNA alone: every one is listed, with its reason
    assert len(df) == 33 and not df["files_ok"].any()
    assert _core(df.attrs) == {"held": ["rna"], "not_listed": 0, "not_held": []}
    row = df[(df["method"] == "totalVI")].iloc[0]
    assert row["reason"].startswith("adt.h5 is missing.")


def test_nothing_held_means_no_filter(tmp_path, no_envs, capsys):
    (tmp_path / "EMPTY").mkdir()
    df = full_scan("EMPTY", "vertical", data_path=tmp_path)
    assert len(df) == 33 and _core(df.attrs) == {"held": [], "not_listed": 0, "not_held": []}
    assert capsys.readouterr().out == (
        "[scan] 0 of 33 rows have their input files. 0 of 33 have their environment "
        "installed. mtb.env.doctor() checks the environments.\n")
    # five batches scanned as vertical: rna.h5 is not there, rna1.h5 ... are
    _donors(tmp_path, "FIVE", 5)
    df = _quiet(full_scan, "FIVE", "vertical", data_path=tmp_path, verbose=False)
    assert len(df) == 33 and df.attrs["held"] == [] and df.attrs["not_listed"] == 0
    assert "This folder holds per-batch files" in df["reason"].iloc[0]


def test_a_row_that_reads_a_folder_is_always_listed(tmp_path, no_envs):
    # scBridge reads the folder; D11 holds no ATAC file
    df = full_scan("D11", "diagonal", verbose=False)
    assert "(data_dir)" in set(df["modalities"])
    assert list(df["method"]) == ["scBridge"] and df.attrs["not_held"] == ["atac"]
    assert df.attrs["not_listed"] == len(list(W._variant_rows("diagonal"))) - 1


def test_a_named_method_keeps_its_rows(no_envs):
    df = full_scan("D11", "vertical", methods=["MIRA", "sciPENN"], verbose=False)
    assert sorted(zip(df["method"], df["modalities"])) == [("MIRA", "rna+atac"),
                                                           ("sciPENN", "rna+adt")]
    mira = df.set_index("method").loc["MIRA"]
    assert mira["reason"].startswith("MIRA needs peak ATAC (atac.h5), which is not in "
                                     "the folder.")
    assert _core(df.attrs) == {"held": ["rna", "adt"], "not_listed": 0, "not_held": []}
    # a named method with a listed row is listed by that row
    df = full_scan("D11", "vertical", methods=["Matilda", "MIRA"], verbose=False)
    assert sorted(zip(df["method"], df["modalities"])) == [("MIRA", "rna+atac"),
                                                           ("Matilda", "rna+adt")]
    assert df.attrs["not_listed"] == 2 and df.attrs["not_held"] == ["atac"]
    every = full_scan("D11", "vertical", methods=["Matilda", "MIRA"], modalities="all",
                     verbose=False)
    assert len(every) == 4


def test_params_keys_are_checked_before_the_filter(monkeypatch):
    seen = []
    real = W._check_param_keys

    def check(plan_df, params):
        seen.append(len(plan_df))
        return real(plan_df, params)
    monkeypatch.setattr(W, "_check_param_keys", check)
    df = full_scan("D11", "vertical", params={"Matilda": {"epochs": 1}}, verbose=False)
    assert seen == [33] and len(df) == 14          # every variant was checked
    with pytest.raises(KeyError, match="Matilda does not accept"):
        full_scan("D11", "vertical", params={"Matilda": {"no_such_key": 1}}, verbose=False)


# ------------------------------------------------------------------ the count line
def test_count_line_names_what_is_not_listed(no_envs, capsys, monkeypatch):
    full_scan("D11", "vertical")
    assert capsys.readouterr().out == (
        "[scan] 14 of 14 methods have their input files. 0 of 14 have their environment "
        "installed. D11 holds RNA and ADT. 19 rows that read ATAC are not listed. "
        'modalities="all" lists them. mtb.env.doctor() checks the environments.\n')
    # every modality held: the line of 0.3.4, word for word
    _on_disk("D22mini")
    full_scan("D22mini", "vertical")
    assert capsys.readouterr().out == (
        "[scan] 33 of 33 rows have their input files. 0 of 33 have their environment "
        "installed. mtb.env.doctor() checks the environments.\n")
    full_scan("D11", "vertical", modalities="all")
    assert capsys.readouterr().out == (
        "[scan] 14 of 33 rows have their input files. 0 of 33 have their environment "
        "installed. mtb.env.doctor() checks the environments.\n")
    # one row, and the command-line spelling
    monkeypatch.setattr(config, "_CLI", True)
    full_scan("D11", "vertical", methods=["totalVI", "Seurat_WNN"])
    assert capsys.readouterr().out == (
        "[scan] 2 of 2 methods have their input files. 0 of 2 have their environment "
        "installed. D11 holds RNA and ADT. 1 row that reads ATAC is not listed. "
        "--modalities all lists it. multibench env doctor checks the environments.\n")


def test_count_line_off_linux_and_with_batches(monkeypatch, capsys):
    monkeypatch.setattr(R, "linux_only_sentence",
                        lambda: "Methods run only on Linux, and this computer runs macOS.")
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    _on_disk("D52mini")
    full_scan("D52mini", "cross")
    assert capsys.readouterr().out == (
        "[scan] 11 of 11 rows have their input files. 0 of 11 have their environment "
        "installed. D52mini holds RNA and ADT, in batches 1-3. 19 rows that read ATAC are "
        f'not listed. modalities="all" lists them. {W.LINUX_ONLY_SUMMARY}\n')


def test_two_families_can_be_lacking(tmp_path, no_envs, capsys):
    # RNA alone, every category: the one row left reads the folder itself
    d = tmp_path / "RNA_ONLY"
    d.mkdir()
    _h5(d / "rna.h5", GENES[:30], [f"c{i}" for i in range(20)])
    df = full_scan("RNA_ONLY", data_path=tmp_path)
    line = capsys.readouterr().out
    assert list(df["modalities"]) == ["(data_dir)"]
    assert _core(df.attrs) == {"held": ["rna"], "not_listed": 96, "not_held": ["adt", "atac"]}
    assert line == (
        "[scan] 0 of 1 method has its input files. 0 of 1 has its environment installed. "
        "RNA_ONLY holds RNA. 96 rows that read ADT or ATAC are not listed. "
        'modalities="all" lists them. mtb.env.doctor() checks the environments.\n')


def test_attrs():
    df = full_scan("D11", "vertical", verbose=False)
    assert _core(df.attrs) == {"held": ["rna", "adt"], "not_listed": 19, "not_held": ["atac"]}
    assert _core(full_scan("D11", "vertical", modalities="all", verbose=False).attrs) == \
        {"held": ["rna", "adt"], "not_listed": 0, "not_held": []}
    # what the folder holds does not follow the selection
    assert _core(full_scan("D11", "vertical", modalities=["rna", "atac"], verbose=False).attrs) == \
        {"held": ["rna", "adt"], "not_listed": 0, "not_held": []}
    assert _core(full_scan("D11", "vertical", methods=["MIRA"], verbose=False).attrs) == \
        {"held": ["rna", "adt"], "not_listed": 0, "not_held": []}
    _on_disk("D22mini")
    assert _core(full_scan("D22mini", "vertical", verbose=False).attrs) == \
        {"held": ["rna", "adt", "atac"], "not_listed": 0, "not_held": []}
    # the dry run carries them, and a selection of rows keeps them
    plan = mtb.run_all("D11", "vertical", dry_run=True, verbose=False)
    assert plan.attrs == df.attrs and plan[plan["files_ok"]].attrs == df.attrs
    assert W._held_families(config.DEFAULT.data_path / "D11", "vertical") == ["rna", "adt"]
    assert W._held_families(config.DEFAULT.data_path / "D11", None) == ["rna", "adt"]
    # a numbered folder holds no file of an unnumbered role, and the other way
    assert W._held_families(config.DEFAULT.data_path / "D52", "vertical") == []
    assert W._held_families(config.DEFAULT.data_path / "D52", "cross") == ["rna", "adt"]


def test_an_older_file_name_counts_as_held(tmp_path):
    d = tmp_path / "OLD"
    d.mkdir()
    _h5(d / "rna.h5", GENES[:30], [f"r{i}" for i in range(20)])
    _h5(d / "peak.h5", [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)],
        [f"a{i}" for i in range(20)])
    df = _quiet(full_scan, "OLD", "diagonal", data_path=tmp_path, verbose=False)
    assert df.attrs["held"] == ["rna", "atac"] and df.attrs["not_listed"] == 0


def test_verbose_false_prints_nothing_and_sets_attrs(capsys):
    df = full_scan("D11", "vertical", verbose=False)
    assert capsys.readouterr().out == ""
    assert len(df) == 14 and df.attrs["not_listed"] == 19 and df.attrs["not_held"] == ["atac"]
    assert type(df) is pd.DataFrame and list(df.columns) == W.SCAN_COLUMNS


def test_two_donor_folder_keeps_the_missing_batch_reason(tmp_path, linux):
    _donors(tmp_path, "TWO", 2)
    df = full_scan("TWO", "cross", data_path=tmp_path, verbose=False)
    assert len(df) == 11 and int(df["files_ok"].sum()) == 1
    assert _core(df.attrs) == {"held": ["rna", "adt"], "not_listed": 19, "not_held": ["atac"]}
    assert list(df[df["files_ok"]]["modalities"]) == ["rna1+rna2+adt1+adt2"]
    rest = df[~df["files_ok"]]
    assert set(rest["reason"]) == {"rna3.h5, adt3.h5 are missing."}
    assert {"StabMap", "sciPENN"} <= set(rest["method"])


# ------------------------------------------------------------------ the hint
def _nothing_runnable(**kw):
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir="unused_out", verbose=False, **kw)
    return str(e.value).splitlines()


def test_the_hint_carries_all(no_envs, capsys):
    lines = _nothing_runnable(modalities="all")
    call = "mtb.run_all('D11', 'vertical', modalities=\"all\", dry_run=True)"
    assert lines[-1] == (f"{call} shows every row. Its files_ok and env_ok columns say "
                         f"which check failed. mtb.env.doctor() checks the environments.")
    n = int(re.fullmatch(r"The first 3 of (\d+) blocked rows:", lines[1]).group(1))
    shown = eval(call, {"mtb": mtb})                      # noqa: S307 - our own hint
    assert len(shown) == n == 33
    assert "not listed" not in "\n".join(lines)
    # the default call: the rows it lists, the sentence on the others, the same count
    lines = _nothing_runnable()
    call = "mtb.run_all('D11', 'vertical', dry_run=True)"
    assert lines[-1] == (f"{call} shows the listed rows. Its files_ok and env_ok columns "
                         f"say which check failed. mtb.env.doctor() checks the environments.")
    assert lines[-2] == ("D11 holds RNA and ADT. 19 rows that read ATAC are not listed. "
                         'modalities="all" lists them.')
    n = int(re.fullmatch(r"The first 3 of (\d+) blocked methods:", lines[1]).group(1))
    shown = eval(call, {"mtb": mtb})                      # noqa: S307 - our own hint
    assert len(shown) == n == 14
    capsys.readouterr()


def test_the_hint_of_the_command_line_carries_all(no_envs, monkeypatch):
    monkeypatch.setattr(config, "_CLI", True)
    lines = _nothing_runnable(modalities="all")
    assert lines[-1].startswith("multibench scan D11 --category vertical --modalities all "
                                "shows every row. ")
    lines = _nothing_runnable()
    assert lines[-1].startswith("multibench scan D11 --category vertical shows the listed "
                                "rows. ")
    assert lines[-2] == ("D11 holds RNA and ADT. 19 rows that read ATAC are not listed. "
                         "--modalities all lists them.")


def test_named_methods_keep_the_error_of_0_3_4(no_envs):
    # with methods=, and when every row is listed, the text is unchanged
    lines = _nothing_runnable(methods=["Matilda"])
    assert lines[-1].startswith("mtb.run_all('D11', 'vertical', methods=['Matilda'], dry_run=True) shows "
                                "these rows. ")
    assert "not listed" not in "\n".join(lines)
    _on_disk("D22mini")
    with pytest.raises(ValueError) as e:
        mtb.run_all("D22mini", "vertical", out_dir="unused_out", verbose=False)
    lines = str(e.value).splitlines()
    assert lines[-1].startswith("mtb.run_all('D22mini', 'vertical', dry_run=True) shows every row. ")
    assert "not listed" not in str(e.value)


def test_dry_run_prints_the_note_after_its_count_line(no_envs, capsys):
    mtb.run_all("D11", "vertical", dry_run=True)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[run_all] Dry run: 0 of 14 requested methods can run on D11 "
                             "(vertical). 14 are blocked. ")
    assert out[1] == ("[run_all] D11 holds RNA and ADT. 19 rows that read ATAC are not "
                      'listed. modalities="all" lists them.')
    mtb.run_all("D11", "vertical", dry_run=True, modalities="all")
    out = capsys.readouterr().out
    assert out.startswith("[run_all] Dry run: 0 of 33 requested rows can run on D11 ")
    assert "not listed" not in out
