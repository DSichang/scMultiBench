"""Findings of the review of the round-1 integration (wp/f1_int).

- A scan row's ``modalities`` split on ``+`` selects that row again, in
  scan, run_all and the CLI (since R3-01 a lone representation token must match
  method_info(m)['atac']); ``atac_peak`` with ``atac_gas`` means "reads both
  files" (MultiMAP, Seurat_v3) instead of an error (L13 kept 0.3.1's calls).
- inputs_for applies the representation rule of scan and find_methods when
  the tokens match a variant only loosely (L13).
- scan keeps one exact combination; find_methods keeps the variants that read
  at least the named modalities; both docstrings say so (L13).
"""
import re
import warnings

import pytest

import multibench as mtb
from multibench import cli
from multibench.engine import registry

SHIPPED = [("D11", "vertical"), ("D28", "diagonal"), ("D45", "mosaic"),
           ("D46", "mosaic"), ("D52", "cross")]
BOTH = ["rna", "atac_peak", "atac_gas"]


@pytest.fixture
def data(root):
    return root / "data"


# ------------------------------------------------------------------ L13 round trip
@pytest.mark.parametrize("ds,cat", SHIPPED)
def test_a_scan_rows_modalities_select_that_row_again(data, ds, cat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan(ds, cat, data_path=data, verbose=False)
        rows = sc[sc["modalities"] != "(data_dir)"]
        assert len(rows)
        for m, mods in zip(rows["method"], rows["modalities"]):
            toks = mods.split("+")
            # R3-01: moETM, scMM and iPOLNG read peaks through a role named
            # atac_gas; the representation token selects them, not the role name
            if mtb.method_info(m)["atac"] == "peak" and toks.count("atac_gas") == 1 \
                    and "atac_peak" not in toks:
                toks = ["atac_peak" if t == "atac_gas" else t for t in toks]
            back = mtb.scan(ds, cat, methods=[m], modalities=toks,
                            data_path=data, verbose=False)
            assert ((back["method"] == m) & (back["modalities"] == mods)).any(), (m, mods)


def test_two_representations_select_the_methods_that_read_both_files(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan("D28", "diagonal", modalities=BOTH, data_path=data, verbose=False)
        plan = mtb.run_all("D28", "diagonal", modalities=BOTH, data_path=data, dry_run=True)
    want = {"MultiMAP", "Seurat_v3"}
    assert set(sc["method"]) == want and set(plan["method"]) == want
    assert set(sc["modalities"]) == {"rna+atac_peak+atac_gas"}
    assert set(mtb.find_methods("diagonal", modalities=BOTH)) == want
    assert set(mtb.find_methods("diagonal", modalities=["peak", "gas"])) == want
    # inputs_for agrees (it did before, and still does)
    inp = mtb.inputs_for("D28", "diagonal", "MultiMAP", modalities=BOTH, data_path=data)
    assert {"atac_peak", "atac_gas"} <= set(inp)


def test_the_cli_round_trip_of_a_two_file_row(data, tmp_path, capsys):
    rc = cli.main(["scan", "D28", "--category", "diagonal", "--data-path", str(data),
                   "--modalities", "rna,atac_peak,atac_gas", "--format", "csv",
                   "--columns", "method,modalities"])
    out = capsys.readouterr().out
    assert rc == 0 and "MultiMAP,rna+atac_peak+atac_gas" in out
    rc = cli.main(["run-all", "D28", "--category", "diagonal", "--data-path", str(data),
                   "--modalities", "rna,atac_peak,atac_gas", "--dry-run",
                   "--out-dir", str(tmp_path / "out"), "--format", "csv",
                   "--columns", "method,modalities"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "MultiMAP,rna+atac_peak+atac_gas" in out and "Seurat_v3" in out


def test_a_two_file_method_gets_no_representation_caveat(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan("D28", "diagonal", data_path=data, verbose=False)
    for m in ("MultiMAP", "Seurat_v3"):
        assert "look like peaks" not in sc.loc[sc["method"] == m, "caveat"].item()


# ------------------------------------------------------------------ L13 inputs_for
@pytest.mark.parametrize("ds,cat,method,mods,msg", [
    ("D28", "diagonal", "GLUE", ["rna", "gas"], "GLUE reads peaks; modalities name gene activity"),
    ("D28", "diagonal", "SCALEX", ["rna", "peak"], "SCALEX reads gene activity; modalities name peaks"),
    ("D11", "vertical", "Matilda", ["rna", "peak"], "Matilda reads gene activity; modalities name peaks"),
    ("D28", "diagonal", "GLUE", BOTH, "GLUE reads peaks; modalities name peaks and gene activity"),
])
def test_inputs_for_applies_the_representation_rule(data, ds, cat, method, mods, msg):
    with pytest.raises(KeyError, match=re.escape(msg)):
        mtb.inputs_for(ds, cat, method, modalities=mods, data_path=data)
    # scan leaves the same method out for the same tokens
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan(ds, cat, modalities=mods, data_path=data, verbose=False)
    assert method not in set(sc["method"])


def test_inputs_for_keeps_a_variants_own_roles_and_matching_tokens(data):
    # moETM's role is named atac_gas although it reads peaks: its own roles work
    assert "atac_gas" in mtb.inputs_for("D11", "vertical", "moETM",
                                        modalities=["rna", "atac_gas"], data_path=data)
    assert "atac_gas" in mtb.inputs_for("D11", "vertical", "moETM",
                                        modalities=["rna", "peak"], data_path=data)
    assert "atac" in mtb.inputs_for("D11", "vertical", "Matilda",
                                    modalities=["rna", "gas"], data_path=data)
    assert "atac_gas" in mtb.inputs_for("D28", "diagonal", "SCALEX",
                                        modalities=["rna", "atac"], data_path=data)


def test_the_vocabulary_message_calls_peak_and_gas_representation_tokens():
    with pytest.raises(ValueError) as ei:
        registry.normalize_modalities(["rna", "bogus"])
    msg = str(ei.value)
    assert "aliases: peak" not in msg
    assert "the representation tokens atac_peak (also peak, peaks) and atac_gas " \
           "(also gas, gene_activity)" in msg
    notes = mtb.inputs_for.__doc__
    assert "``peak`` / ``peaks`` for ``atac_peak``" not in notes
    assert "the representation tokens ``atac_peak`` (also ``peak``, ``peaks``)" in notes


# ------------------------------------------------------------------ L13 the two rules
def test_scan_keeps_one_combination_and_find_methods_at_least(data):
    fm = set(mtb.find_methods("mosaic", modalities=["rna", "adt"]))
    assert {"Multigrate", "StabMap", "scMoMaT"} <= fm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan("D46", "mosaic", modalities=["rna", "adt"], data_path=data,
                      verbose=False)
    # every kept row reads exactly RNA and ADT: no ATAC role
    assert len(sc) and not sc["modalities"].str.contains("atac").any()
    assert set(sc["method"]) < fm
    scan_doc, fm_doc = mtb.scan.__doc__, mtb.find_methods.__doc__
    assert "the rule is the one ``mtb.find_methods`` uses" not in scan_doc
    assert "A row is kept when its modalities are exactly that combination." in scan_doc
    assert "A method matches when one variant reads at least" in fm_doc
    assert "a row's modalities must be exactly the named combination" in fm_doc


# ------------------------------------------------------------------ L61 method_info text
@pytest.mark.parametrize("method", mtb.list_methods())
def test_method_info_notes_and_setup_hint_hold_no_internal_names(method):
    info = mtb.method_info(method)
    for key in ("notes", "setup_hint"):
        text = str(info.get(key) or "")
        for bad in ("engine/", "source()d", ".yaml"):
            assert bad not in text, (method, key, bad)
    if method == "StabMap":
        assert info["notes"].endswith("using shared and unshared features; R.")


# ------------------------------------------------------------------ L30 prepared inputs
def test_a_command_that_reads_a_prepared_file_says_so(data, tmp_path, capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan("D28", "diagonal", data_path=data, verbose=False)
    row = sc.set_index("method").loc["Seurat_v3"]
    assert "/inputs/atac_peak_normpeaks.h5" in row["command"]
    assert ("the command reads inputs/atac_peak_normpeaks.h5, which mtb.run writes "
            "first: start the method with mtb.run or mtb.run_all") in row["caveat"]
    # a command that reads only the dataset's own files has no such note
    assert "writes first" not in sc.set_index("method").loc["GLUE", "caveat"]
    # the dry run prints the same note
    inp = mtb.inputs_for("D28", "diagonal", "Seurat_v3", data_path=data)
    mtb.run("Seurat_v3", "diagonal", inputs=inp, out_dir=tmp_path / "o", dry_run=True)
    assert "# the command reads inputs/atac_peak_normpeaks.h5" in capsys.readouterr().err
    # the CLI marks the line in its '# commands' block
    rc = cli.main(["run-all", "D28", "--category", "diagonal", "--data-path", str(data),
                   "--methods", "Seurat_v3,GLUE", "--dry-run", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    lines = {ln.split(" (")[0]: ln for ln in out.splitlines() if ": conda run" in ln}
    assert "[use multibench run]" in lines["Seurat_v3"]
    assert "[use multibench run]" not in lines["GLUE"]


# ------------------------------------------------------------------ L39/L41 CLI help
@pytest.mark.parametrize("cmd", ["evaluate", "run-all"])
def test_leiden_flavor_help_names_both_stored_tables(cmd, capsys):
    with pytest.raises(SystemExit):
        cli.main([cmd, "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "leidenalg matches both stored tables (published and re-run)" in text
    assert "backend of the published tables" not in text


# ------------------------------------------------------------------ L11 cross is RNA+ADT
def test_cross_is_described_as_rna_and_adt(capsys):
    text = mtb.list_categories()["cross"]
    assert "each measured with RNA and ADT" in text and "all modalities" not in text
    with pytest.raises(SystemExit):
        cli.main(["scan", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "cross (several batches, each with RNA and ADT)" in help_text
    assert "several batches with all modalities" not in help_text


# ------------------------------------------------------------------ L48 BatchResult.plot
def test_batchresult_plot_reads_the_fill_as_scaled_per_column():
    doc = " ".join(mtb.BatchResult.plot.__doc__.split())
    assert "darker is higher" not in doc and "rank 1 is the largest" not in doc
    assert "Circle size shows the rank within a column (bigger is better)." in doc
    assert "the lightest is the lowest in this figure, not zero." in doc


# ------------------------------------------------------------------ .loom extra
def test_the_loom_hint_names_the_distribution(tmp_path, monkeypatch):
    import builtins
    import sys
    from multibench.engine import ingest

    real_import = builtins.__import__

    def no_loompy(name, *a, **k):
        if name == "loompy":
            raise ModuleNotFoundError("No module named 'loompy'")
        return real_import(name, *a, **k)

    monkeypatch.setitem(sys.modules, "loompy", None)
    monkeypatch.setattr(builtins, "__import__", no_loompy)
    (tmp_path / "x.loom").write_bytes(b"")
    with pytest.raises(ImportError) as ei:
        ingest.to_canonical(tmp_path / "x.loom", tmp_path / "x.h5", modality="rna")
    assert "pip install 'multibench-sc[loom]'" in str(ei.value)
    assert "'multibench[loom]'" not in str(ei.value)
    from multibench.engine import runner
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    with pytest.raises(ImportError, match=r"pip install 'multibench-sc\[loom\]'"):
        runner._check_input("rna", str(tmp_path / "x.loom"), real=False)


# ------------------------------------------------------------------ n/a hint on the CLI
def test_the_na_hint_uses_the_spelling_of_the_caller(tmp_path, capsys):
    import matplotlib
    matplotlib.use("Agg")
    args = ["plot", "bubble", "--category", "diagonal", "--dataset", "D28",
            "--metrics", "ARI,NMI,ASW", "--out", str(tmp_path / "f.png")]
    assert cli.main(args) == 0
    err = capsys.readouterr().err
    assert "Pass --na skip to silence this" in err and "na='skip'" not in err
    assert cli.main(args + ["--na", "skip"]) == 0
    assert "n/a cells" not in capsys.readouterr().err
    with pytest.warns(UserWarning, match="Pass na='skip' to silence this"):
        mtb.plot.bubble(mtb.load_results("diagonal", dataset="D28"),
                        metrics=["ARI", "NMI", "ASW"])


# ------------------------------------------------------------------ plain vocabulary
def _contract_objects():
    """The reference objects of the docs (the docstring contract), with the
    public methods of the classes among them."""
    import importlib
    import inspect
    from tests.test_docs_consistency import REFERENCE

    out = []
    for name, (_page, path) in sorted(REFERENCE.items()):
        mod, _, attr = path.rpartition(".")
        obj = getattr(importlib.import_module(mod), attr)
        out.append((name, obj))
        if inspect.isclass(obj):
            for k, v in vars(obj).items():
                if not k.startswith("_") and (inspect.isfunction(v) or isinstance(v, property)):
                    out.append((f"{name}.{k}", v))
    from multibench.engine import envs
    out += [("mtb.env.env_prefix", envs.env_prefix),
            ("mtb.env.host_has_gpu", envs.host_has_gpu)]
    return out


@pytest.mark.parametrize("name,obj", _contract_objects(), ids=lambda x: x if isinstance(x, str) else "")
def test_reference_docstrings_use_plain_words(name, obj):
    import inspect
    doc = inspect.getdoc(obj) or ""
    assert not re.search(r"(?i)\btidy\b", doc), name
    flat = " ".join(doc.split())
    for banned in ("stand-in", "mtb.plot.render", "mtb.env.group_for",
                   "mtb.env.default_env_name", "host_platform_problem", "failed silently",
                   "OUT_DIR_PLACEHOLDER", "output_urls.json"):
        assert banned not in flat, (name, banned)


def test_signatures_render_without_internal_names():
    import inspect
    assert str(inspect.signature(mtb.list_methods)) == "(category: 'str | None' = None) -> 'list[str]'"
    assert inspect.signature(mtb.scan).parameters["out_dir"].default == "<out_dir>"
    with pytest.raises(TypeError, match="use find_methods"):
        mtb.list_methods(task="clustering")
    with pytest.raises(TypeError, match="unexpected keyword argument 'bogus'"):
        mtb.list_methods(bogus=1)
    assert mtb.catalog.metrics.__doc__.splitlines()[0] == \
        "Table describing the scIB metrics: one row per metric code."


def test_module_help_is_written_for_users():
    for mod in (mtb.env, mtb.config, mtb.catalog):
        doc = mod.__doc__
        for internal in (".yaml", "install_packed", "create_env", "files/", "space-named",
                         "scib_metric", "env_spec"):
            assert internal not in doc, (mod.__name__, internal)
