"""'Run all' safety and the Colab budget of the generated category tutorials.

The re-test after polish round 2 found the linux-gated env-install cell
starting a 3 GB packed-env download on every Colab attendee (Colab is linux),
the run cells executing for real on a CPU runtime, and section 5's
``avail[avail.runnable]`` table empty on every host without environments.
The Colab speed round then removed condacolab (the packed environments run
without a conda binary), pinned numpy / pandas to the host's versions in the
install cell, and made a host without environments stand in the benchmark
host's real ``run_all`` outputs (``mtb.data.fetch_outputs``) before the stored
metric table. These tests pin all of it: one ``INSTALL_ENVS = False`` flag
cell that the one environment download (``mtb.env.install(...,
dry_run=False)``) sits behind, no ``condacolab`` token anywhere, run cells
gated on ``scan().env_ok`` with the two-level stand-in otherwise, the
evaluate cell scoring a real embedding when one is on disk, library-specific
warning filters, and the file-gate table with the env columns next to it.
"""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
CATS = list(GEN.SCEN)
NOTEBOOKS = [f"tutorial_{cat}" for cat in CATS] + ["colab_quickstart"]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _code(name):
    return [src for kind, src in _cells(name) if kind == "code"]


def _markdown(name):
    return "\n".join(src for kind, src in _cells(name) if kind == "markdown")


def _tree(src):
    ipy = pytest.importorskip("IPython.core.inputtransformer2")
    tree = ast.parse(ipy.TransformerManager().transform_cell(src))
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node
    return tree


def _guarded_by(node, token: str) -> bool:
    """True when an enclosing ``if`` tests an expression mentioning ``token``
    (an ``elif`` chain is nested ``If`` nodes, so it is covered too)."""
    while getattr(node, "_parent", None) is not None:
        node = node._parent
        if isinstance(node, ast.If) and token in ast.unparse(node.test):
            return True
    return False


def _shell_calls(tree):
    """``!cmd`` lines become ``get_ipython().system('cmd')`` after transform."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "system" and node.args
                and isinstance(node.args[0], ast.Constant)):
            yield node, node.args[0].value


def _env_install_calls(tree):
    """``mtb.env.install(...)`` calls, with their keyword dict."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("env.install"):
            yield node, {k.arg: k.value for k in node.keywords if k.arg}


# ---------------------------------------------------------------- the flag
@pytest.mark.parametrize("cat", CATS)
def test_flag_cell_present_default_false_and_before_any_download(cat):
    code = _code(f"tutorial_{cat}")
    flag_cells = []
    for i, src in enumerate(code):
        for node in ast.walk(_tree(src)):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "INSTALL_ENVS"):
                assert isinstance(node.value, ast.Constant) and node.value.value is False, \
                    f"tutorial_{cat}: INSTALL_ENVS must default to False"
                flag_cells.append(i)
    assert len(flag_cells) == 1, f"tutorial_{cat}: exactly one INSTALL_ENVS flag cell"
    downloads = [i for i, src in enumerate(code) if "dry_run=False" in src]
    assert downloads, f"tutorial_{cat}: the opt-in environment install cell is missing"
    assert flag_cells[0] < min(downloads), f"tutorial_{cat}: the flag cell precedes every download"
    assert "GB to download" in code[flag_cells[0]], "the flag comment states the measured size"
    assert "no conda needed" in code[flag_cells[0]]


@pytest.mark.parametrize("cat", CATS)
def test_no_code_cell_downloads_environments_unconditionally(cat):
    n_real = 0
    for src in _code(f"tutorial_{cat}"):
        tree = _tree(src)
        for node, cmd in _shell_calls(tree):
            assert "env install" not in cmd and "condacolab" not in cmd, \
                f"tutorial_{cat}: environments install through mtb.env.install, not a shell line: {cmd!r}"
        for node, kw in _env_install_calls(tree):
            dry = kw.get("dry_run")
            if isinstance(dry, ast.Constant) and dry.value is False:
                n_real += 1
                assert _guarded_by(node, "INSTALL_ENVS"), \
                    f"tutorial_{cat}: environment download not behind INSTALL_ENVS: {ast.unparse(node)[:80]}"
                assert _guarded_by(node, "sys.platform"), \
                    f"tutorial_{cat}: the linux-64 archives need the platform check"
                packed = kw.get("packed")
                assert isinstance(packed, ast.Constant) and packed.value is True, \
                    f"tutorial_{cat}: the opt-in install must take the packed archives (no conda on Colab)"
                assert "category" in kw
    assert n_real == 1, f"tutorial_{cat}: exactly one real env install, behind the flag"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_condacolab_token_anywhere(name):
    """The packed archives run without a conda binary, so nothing provisions
    conda on Colab any more - no cell, code or markdown, names condacolab."""
    for kind, src in _cells(name):
        assert "condacolab" not in src.lower(), f"{name}: {kind} cell still mentions condacolab"
    assert "condacolab" not in (ROOT / "tools" / "gen_tut.py").read_text().split('"""', 2)[2].lower(), \
        "gen_tut.py: no condacolab cell survives (the docstring may say why it is gone)"


@pytest.mark.parametrize("cat", CATS)
def test_env_install_cell_prints_the_measured_size_from_the_dry_run(cat):
    cells = [src for src in _code(f"tutorial_{cat}") if "dry_run=False" in src]
    assert len(cells) == 1
    src = cells[0]
    plans = [kw for _, kw in _env_install_calls(_tree(src)) if "dry_run" not in kw]
    assert plans, f"tutorial_{cat}: the dry-run plan (the default) is printed before the download"
    assert "archive_bytes" in src and "GB" in src
    assert src.index("archive_bytes") < src.index("dry_run=False"), "size first, download second"


# ------------------------------------------------------------- the run cells
@pytest.mark.parametrize("cat", CATS)
def test_run_cells_execute_only_when_scan_finds_an_environment(cat):
    n_runs = 0
    for src in _code(f"tutorial_{cat}"):
        tree = _tree(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "mtb.run_all":
                n_runs += 1
                assert _guarded_by(node, "env_ok"), \
                    f"tutorial_{cat}: run_all not gated on scan().env_ok: {ast.unparse(node)[:80]}"
                assert GEN.SKIP_LINE in src, \
                    f"tutorial_{cat}: a skipped run prints the one line"
                assert "stand_in(" in src or "stored_sweep(" in src, \
                    f"tutorial_{cat}: a skipped run stands in a result computed elsewhere"
    assert n_runs == 2, f"tutorial_{cat}: section 2 and section 3 each run once"
    assert GEN.SKIP_LINE == ("no method environment on this host - the run is skipped; "
                             "a stand-in computed elsewhere covers it")


@pytest.mark.parametrize("cat", CATS)
def test_section_2_stand_in_uses_fetch_outputs_then_the_stored_table(cat):
    """The stand-in path: the benchmark host's real run_all outputs
    (``mtb.data.fetch_outputs`` + ``load_batch(..., methods=)``) first, the
    stored metric table only when that raises, one printed line each."""
    cells = [src for src in _code(f"tutorial_{cat}") if "mtb.run_all(" in src]
    src = cells[0]                                   # section 2, "One call"
    assert GEN.STAND_IN_FN in src and GEN.STORED_SWEEP_FN in src
    tree = _tree(src)
    fetch = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "mtb.data.fetch_outputs"]
    assert len(fetch) == 1
    load = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "mtb.load_batch"]
    assert len(load) == 1 and any(k.arg == "methods" for k in load[0].keywords), \
        "load_batch(path, methods=trio) - the keyword-only filter"
    assert any(isinstance(n, ast.Try) for n in ast.walk(tree)), "fetch_outputs failure is caught"
    # the stand-in is called with the live dataset and the trio the run cell names
    live = GEN.SCEN[cat]["live_ds"] or GEN.SCEN[cat]["ds"]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "stand_in"]
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[0]) == live
    assert ast.literal_eval(calls[0].args[1]) == GEN.SCEN[cat]["own_trio"]
    assert calls[0].keywords[0].arg == "stored"
    assert "stand-in:" in src


@pytest.mark.parametrize("cat", CATS)
def test_stored_sweep_stand_in_renders_summary_and_figure(cat):
    """The last fallback the run cells take on a laptop: the same object run_all
    returns, built from the stored sweep, so .summary and .plot() work."""
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import multibench as mtb
    ns = {"mtb": mtb, "CATEGORY": cat}
    exec(GEN.STORED_SWEEP_FN, ns)
    ds, methods = GEN.stand_in(cat, GEN.SCEN[cat]["ds"], GEN.SCEN[cat]["own_trio"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ns["stored_sweep"](ds, methods)
    assert isinstance(res, mtb.BatchResult)
    assert set(res.summary.status) == {"STORED"}
    assert res.out_dir is None                       # nothing on disk: the evaluate cell says so
    if methods:
        assert sorted(res.summary.method) == sorted(methods)
    else:
        assert len(res.summary) >= 2
    assert {"ARI", "NMI"} <= set(res.summary.columns)
    fig = res.plot()
    assert fig.axes
    plt.close(fig)


class _StubData:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def fetch_outputs(self, dataset):
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


class _StubMtb:
    """Just enough of ``mtb`` for the notebook's ``stand_in`` helper."""

    def __init__(self, data, loaded="HOST"):
        self.data = data
        self.loaded = loaded
        self.load_batch_calls = []

    def load_batch(self, path, *, methods=None):
        self.load_batch_calls.append((path, methods))
        return self.loaded


def _stand_in_ns(mtb):
    ns = {"mtb": mtb, "CATEGORY": "vertical",
          "stored_sweep": lambda *a: ("STORED", *a)}
    exec(GEN.STAND_IN_FN, ns)
    return ns["stand_in"]


def test_stand_in_takes_the_benchmark_host_outputs_when_fetch_outputs_works(capsys):
    mtb = _StubMtb(_StubData(Path("/x/outputs/D11")))
    res = _stand_in_ns(mtb)("D11", ["Matilda"], stored=("D11", ["Matilda"]))
    assert res == "HOST"
    assert mtb.load_batch_calls == [(Path("/x/outputs/D11"), ["Matilda"])]
    out = capsys.readouterr().out
    assert "stand-in: the benchmark host's run_all outputs for D11 (fetch_outputs)" in out
    assert "stored metric table" not in out


@pytest.mark.parametrize("exc", [OSError("offline"), ValueError("unknown dataset"),
                                 AttributeError("module has no attribute fetch_outputs")],
                         ids=["offline", "unknown-id", "package-without-fetch_outputs"])
def test_stand_in_falls_back_to_the_stored_table_when_fetch_outputs_raises(exc, capsys):
    """Offline, an unknown id, or a package that has no fetch_outputs yet: the
    stored metric table stands in, with the exception named in the one line."""
    mtb = _StubMtb(_StubData(exc))
    res = _stand_in_ns(mtb)("D46", ["StabMap"], stored=("D45",))
    assert res == ("STORED", "D45")
    assert mtb.load_batch_calls == []
    out = capsys.readouterr().out
    assert f"stand-in: the package's stored metric table ({type(exc).__name__} from fetch_outputs" in out
    assert "benchmark host's run_all outputs" not in out


def test_stand_in_against_the_live_package_today(capsys):
    """Whatever the installed package has (fetch_outputs landed or not, network
    or not), the helper never raises: it returns a BatchResult and prints
    which path it took."""
    import warnings
    import multibench as mtb
    ns = {"mtb": mtb, "CATEGORY": "vertical"}
    exec(GEN.STORED_SWEEP_FN, ns)
    exec(GEN.STAND_IN_FN, ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ns["stand_in"]("D11", ["Matilda", "sciPENN", "scMM"],
                             stored=("D11", ["Matilda", "sciPENN", "scMM"]))
    assert isinstance(res, mtb.BatchResult)
    assert set(res.summary.method) == {"Matilda", "sciPENN", "scMM"}
    out = capsys.readouterr().out
    assert out.count("stand-in:") == 1


# ------------------------------------------------------------ the evaluate cell
def _evaluate_cell_ns(res, calls):
    import multibench as mtb
    import pandas as pd

    class _Mtb:
        labels_for = staticmethod(mtb.labels_for)

        @staticmethod
        def evaluate(output, labels=None, **kw):
            calls.append((output, labels, kw))
            return pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])

    return {"mtb": _Mtb, "Path": Path, "res": res}


def test_evaluate_cell_scores_the_method_embedding_with_its_label_order(tmp_path):
    """run_all's and fetch_outputs' trees share <out_dir>/<method>_<dataset>/
    embedding.h5; the cell scores it against labels_for(dataset) in the order
    the record's labels_used gives (call shape verified against the real
    evaluate on a synthetic embedding: label_order=['cty'] on a one-file dict
    and ['cty1', 'cty2', 'cty3'] on D46 both score)."""
    import multibench as mtb
    d = tmp_path / "Matilda_D11"
    d.mkdir()
    (d / "embedding.h5").write_bytes(b"")
    recs = [{"method": "Matilda", "status": "CHAIN_OK", "labels_used": ["cty.csv"],
             "metrics": {"ARI": 0.9}}]
    res = mtb.BatchResult(recs, "D11", "vertical", out_dir=tmp_path)
    calls = []
    ns = _evaluate_cell_ns(res, calls)
    exec(GEN.EVALUATE_CELL_TEMPLATE.format(method="Matilda"), ns)
    assert len(calls) == 1
    output, labels, kw = calls[0]
    assert output == d / "embedding.h5"
    assert labels == mtb.labels_for("D11")
    assert kw["label_order"] == ["cty"] and kw["verbose"] is False
    assert ns["scores"] is not None


def test_evaluate_cell_skips_the_stored_table_stand_in(capsys):
    """A BatchResult from the stored metric table has no out_dir and no file:
    the cell prints why and sets ``scores = None``, so the notebook runs through."""
    import multibench as mtb
    res = mtb.BatchResult([{"method": "Matilda", "status": "STORED"}], "D11", "vertical")
    calls = []
    ns = _evaluate_cell_ns(res, calls)
    exec(GEN.EVALUATE_CELL_TEMPLATE.format(method="Matilda"), ns)
    assert calls == [] and ns["scores"] is None
    assert "no embedding on this host" in capsys.readouterr().out


@pytest.mark.parametrize("cat", CATS)
def test_evaluate_cell_names_the_live_method(cat):
    code = _code(f"tutorial_{cat}")
    cells = [src for src in code if "mtb.evaluate(" in src]
    assert len(cells) == 1
    assert cells[0] == GEN.EVALUATE_CELL_TEMPLATE.format(method=GEN.SCEN[cat]["live"][0])
    i_run = next(i for i, src in enumerate(code) if "mtb.run_all(" in src)
    i_plot = next(i for i, src in enumerate(code) if src.strip() == "res.plot()")
    assert i_run < code.index(cells[0]) < i_plot, "run, then evaluate, then plot"


# ------------------------------------------------------------ the install cell
def test_install_cell_pins_the_host_stack():
    """The install cell (shared by every notebook) pins numpy / pandas to the
    running interpreter's versions on both pip lines."""
    src = GEN.INSTALL_CELLS[0]
    assert ('pins = [f"{p}=={importlib.metadata.version(p)}" for p in ("numpy", "pandas") '
            'if importlib.util.find_spec(p)]') in src
    pip_lines = [l for l in src.splitlines() if "pip -q install" in l]
    assert len(pip_lines) == 2 and all('{" ".join(pins)}' in l for l in pip_lines)
    assert '"multibench-sc>=0.3"' in pip_lines[0] and "git+https://github.com/DSichang/scMultiBench.git" in pip_lines[1]
    assert "import importlib.metadata" in src


# ------------------------------------------------------ the rest of the polish
@pytest.mark.parametrize("cat", CATS)
def test_warning_filters_are_library_specific(cat):
    code = _code(f"tutorial_{cat}")
    for src in code:
        assert 'filterwarnings("ignore")' not in src and "filterwarnings('ignore')" not in src, \
            f"tutorial_{cat}: a blanket ignore hides multibench's own warnings (recommend coverage, DegenerateRerunWarning)"
    setup = [src for src in code if "filterwarnings" in src]
    assert len(setup) == 1
    for name in ("FutureWarning", "DeprecationWarning", "PerformanceWarning",
                 "ImplicitModificationWarning"):
        assert name in setup[0], f"tutorial_{cat}: {name} filter missing"
    assert "category=_w" in setup[0]


@pytest.mark.parametrize("cat", CATS)
def test_reference_scan_table_shows_files_ok_rows_with_env_columns(cat):
    code = "\n".join(_code(f"tutorial_{cat}"))
    assert "avail[avail.files_ok]" in code and '"env_ok", "env_reason"' in code
    assert "avail[avail.runnable]" not in code, "empty on every host without environments"
    assert "avail[~avail.files_ok]" in code and '"files_reason"' in code


@pytest.mark.parametrize("cat", CATS)
def test_batch_metrics_prose_names_the_metrics_knob(cat):
    """The one selector is metrics= (0.3.0): the prose shows the family form
    and says what the default None computes; the 0.2 task= never returns."""
    md = _markdown(f"tutorial_{cat}")
    assert '`"clustering"`, `"batch"` or `"all"`: a family' in md
    assert "- `None` (the default): every applicable metric" in md
    assert 'task="all"' not in md and 'task="clustering"' not in md


@pytest.mark.parametrize("cat", [c for c in CATS if c != "vertical"])
def test_labels_for_prose_warns_that_a_wrong_order_scores_silently(cat):
    """labels_for(dataset) returns the default order, which is not every
    method's stacking order (StabMap on D52 stacks cty3, cty1, cty2): a
    multi-file tutorial must say a wrong order scores silently, name
    labels_for(DATASET, CATEGORY, method) as the remedy, list the methods whose
    order the live package gives differently, and show one of them in code."""
    md = " ".join(_markdown(f"tutorial_{cat}").split())
    assert "`labels_for(DATASET, CATEGORY, method)` returns the files in that method's order" in md
    assert "A wrong order gives wrong scores without an error" in md
    assert "scores a multi-file dataset directly" not in md
    assert "in the order `evaluate` expects" not in md
    ds = GEN.SCEN[cat]["ds"]
    others = GEN.reordering_methods(cat, ds)
    assert others, f"{cat}: no method with another order on {ds}"
    for m, order in others.items():
        assert f"{m} (`{', '.join(order)}`)" in md
    m0 = next(iter(others))
    assert any(f'mtb.labels_for(DATASET, CATEGORY, "{m0}")' in src
               for src in _code(f"tutorial_{cat}"))


@pytest.mark.parametrize("cat", CATS)
def test_prose_says_no_conda_is_needed_and_names_the_knobs(cat):
    md = " ".join(_markdown(f"tutorial_{cat}").split()).replace("*", "")   # one line, no emphasis marks
    assert "no conda needed" in md
    assert "mtb.config.DEFAULT.envs_dir" in md
    assert "mtb.data.fetch_outputs" in md and "mtb.load_batch" in md
    assert "restarts the kernel" not in md and "provisions conda" not in md


@pytest.mark.parametrize("cat", CATS)
def test_coverage_cell_derives_from_the_registry(cat):
    code = "\n".join(_code(f"tutorial_{cat}"))
    assert "IMPUTATION_ONLY" not in code
    assert "method_info(m)['categories']" in code and "['tasks']" not in code
    assert "not in the registry" in code


@pytest.mark.parametrize("cat", CATS)
def test_every_load_results_call_names_its_source(cat):
    n = 0
    for src in _code(f"tutorial_{cat}"):
        for node in ast.walk(_tree(src)):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "mtb.load_results":
                n += 1
                assert any(k.arg == "source" for k in node.keywords), \
                    f"tutorial_{cat}: load_results without source=: {ast.unparse(node)[:80]}"
    assert n >= 3            # stand-in helper, section 4 single, section 4 pair
    md = _markdown(f"tutorial_{cat}")
    assert 'source="published"' in md and 'load_results(source="rerun")' in md


def test_published_note_counts_are_measured():
    import multibench as mtb
    note = GEN.published_note("cross", "D52")
    cov = mtb.results_coverage("cross")
    n_pub = cov[(cov.dataset == "D52") & (cov.source == "published")].method.nunique()
    assert f"holds {n_pub} method" in note and 'source="published"' in note
    assert "no scIB tables for mosaic" in GEN.published_note("mosaic", "D45")


def test_stand_in_keeps_methods_only_when_the_sweep_has_them():
    assert GEN.stand_in("cross", "D52", ["UINMF", "sciPENN", "StabMap"]) == \
        ("D52", ["UINMF", "sciPENN", "StabMap"])
    assert GEN.stand_in("mosaic", "D45", ["StabMap", "scMoMaT"]) == ("D45", None)


# ------------------------------------- the end-to-end tutorial (hand-maintained)
# tutorial_end_to_end.ipynb has no generator, so its Run-all safety is pinned
# here directly: the category tutorials' install cell, flag cell and skip line,
# the one environment download behind INSTALL_ENVS and the linux check, the
# mtb.run cell gated on scan().env_ok with a stand-in (fetch_outputs +
# load_batch, else the stored scores), stored tables that work without the
# repository's results/ folder (Colab opens the notebook alone), and section
# 8's counts restricted to methods with a variant for each dataset's files.
E2E = "tutorial_end_to_end"


def _e2e_function(name):
    """Source of the one ``def name`` in the end-to-end notebook, to exec."""
    found = [ast.unparse(node) for src in _code(E2E) for node in ast.walk(_tree(src))
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, f"{E2E}: exactly one def {name}"
    return found[0]


def test_end_to_end_installs_the_package_and_sets_the_flag_before_any_download():
    code = _code(E2E)
    assert code[0] == GEN.INSTALL_CELLS[0], "the shared install cell comes first"
    size = GEN.env_size_text(methods=["Matilda"])
    assert code[1] == GEN.FLAG_CELL_TEMPLATE.format(size=size[0].upper() + size[1:]), \
        "the category tutorials' flag cell, stating Matilda's measured download size"
    flags = [i for i, src in enumerate(code) for node in ast.walk(_tree(src))
             if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "INSTALL_ENVS" for t in node.targets)]
    assert flags == [1], f"{E2E}: exactly one INSTALL_ENVS assignment, the flag cell"
    assert min(i for i, src in enumerate(code) if "import multibench as mtb" in src) > 1


def test_end_to_end_environment_download_sits_behind_the_flag_and_the_platform_check():
    downloads = []
    for i, src in enumerate(_code(E2E)):
        tree = _tree(src)
        for node, cmd in _shell_calls(tree):
            assert "env install" not in cmd, f"{E2E}: environments install through mtb.env.install: {cmd!r}"
        for node, kw in _env_install_calls(tree):
            dry = kw.get("dry_run")
            if isinstance(dry, ast.Constant) and dry.value is False:
                downloads.append(i)
                assert _guarded_by(node, "INSTALL_ENVS") and _guarded_by(node, "sys.platform")
                assert isinstance(kw.get("packed"), ast.Constant) and kw["packed"].value is True
                assert ast.literal_eval(node.args[0]) == ["Matilda"] and "category" in kw
                assert src.index("archive_bytes") < src.index("dry_run=False"), "size first, download second"
    assert len(downloads) == 1 and downloads[0] > 1, f"{E2E}: one download, after the flag cell"


def test_end_to_end_run_cell_is_gated_on_env_ok_and_stands_in_otherwise():
    runs = [(src, node) for src in _code(E2E) for node in ast.walk(_tree(src))
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("mtb.run", "mtb.run_all")]
    assert len(runs) == 1, f"{E2E}: section 4 runs Matilda once"
    src, call = runs[0]
    assert _guarded_by(call, "env_ok"), f"{E2E}: mtb.run not gated on scan().env_ok"
    assert "mtb.scan(" in src and GEN.SKIP_LINE in src
    gate = next(n for n in ast.walk(_tree(src)) if isinstance(n, ast.If) and "env_ok" in ast.unparse(n.test))
    assert "stand_in(DATASET" in ast.unparse(ast.Module(body=gate.orelse, type_ignores=[])), \
        "a skipped run stands in the benchmark host's output"
    helper = ast.parse(_e2e_function("stand_in"))
    names = {ast.unparse(n.func) for n in ast.walk(helper) if isinstance(n, ast.Call)}
    assert {"mtb.data.fetch_outputs", "mtb.load_batch"} <= names
    assert any(isinstance(n, ast.Try) for n in ast.walk(helper)), "a failed download is caught"


def test_end_to_end_stand_in_reads_the_host_embedding_or_returns_none(tmp_path, capsys):
    import h5py
    import numpy as np
    root = tmp_path / "outputs" / "D11"
    (root / "Matilda_D11").mkdir(parents=True)
    with h5py.File(root / "Matilda_D11" / "embedding.h5", "w") as f:
        f["data"] = np.arange(6.0).reshape(3, 2)

    class _Batch:
        out_dir = root

    mtb = _StubMtb(_StubData(root), loaded=_Batch())
    ns = {"mtb": mtb, "Path": Path, "h5py": h5py}
    exec(_e2e_function("stand_in"), ns)
    assert ns["stand_in"]("D11", "Matilda").shape == (3, 2)
    assert mtb.load_batch_calls == [(root, ["Matilda"])]
    assert "stand-in: Matilda's embedding from the benchmark host's run_all outputs for D11" \
        in capsys.readouterr().out

    ns = {"mtb": _StubMtb(_StubData(OSError("offline"))), "Path": Path, "h5py": h5py}
    exec(_e2e_function("stand_in"), ns)
    assert ns["stand_in"]("D11", "Matilda") is None
    assert "(OSError from fetch_outputs: offline)" in capsys.readouterr().out


def test_end_to_end_evaluate_cell_without_an_embedding_shows_the_stored_scores(capsys):
    """The offline stand-in leaves ``emb = None``: section 5 then shows
    Matilda's stored scores in evaluate's shape, so section 6 still plots."""
    import warnings
    import multibench as mtb
    import pandas as pd

    class _Mtb:
        def __getattr__(self, name):
            return getattr(mtb, name)

        @staticmethod
        def evaluate(*a, **kw):
            raise AssertionError("nothing to evaluate without an embedding")

    src = next(s for s in _code(E2E) if "mtb.evaluate(" in s)
    ns = {"mtb": _Mtb(), "pd": pd, "emb": None, "DATASET": "D11", "CATEGORY": "vertical"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(src, ns)
        stored = mtb.load_results("vertical", dataset="D11", source="rerun", methods=["Matilda"])
    assert ns["scores"]["Value"].to_dict() == stored.set_index("metric")["value"].to_dict()
    long = mtb.to_long(ns["scores"], method="Matilda", dataset="D11", category="vertical")
    assert sorted(long.metric) == sorted(stored.metric)
    assert "no embedding on this host for Matilda" in capsys.readouterr().out


def test_end_to_end_stored_tables_do_not_need_the_results_folder(tmp_path):
    """Sections 7 and 8 read results/ next to the notebook; without it (Colab)
    the package's stored rerun tables give the same metric values."""
    import warnings
    import numpy as np
    import multibench as mtb
    import pandas as pd
    for src in _code(E2E):
        if "RESULTS /" in src:
            assert "def stored_table(" in src, f"{E2E}: every results/ read goes through stored_table"
    ns = {"mtb": mtb, "pd": pd}
    exec(_e2e_function("stored_table"), ns)
    for ds in ("D11", "D28", "D45", "D52"):
        ns["RESULTS"] = ROOT / "notebooks" / "results"
        long_here, summary_here = ns["stored_table"]("long_all", ds), ns["stored_table"]("summary", ds)
        ns["RESULTS"] = tmp_path / "results"                 # absent, as on Colab
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            long_pkg, summary_pkg = ns["stored_table"]("long_all", ds), ns["stored_table"]("summary", ds)
        both = long_here.merge(long_pkg, on=["method", "metric"], how="outer", indicator=True)
        assert (both._merge == "both").all() and np.allclose(both.value_x, both.value_y), ds
        both = summary_here.merge(summary_pkg, on="method", how="outer", indicator=True)
        assert (both._merge == "both").all(), ds
        for m in ("ARI", "NMI", "ASW"):
            assert np.allclose(both[f"{m}_x"], both[f"{m}_y"], atol=1e-4), (ds, m)


def test_end_to_end_scenario_counts_only_methods_with_a_variant_for_the_dataset(
        tmp_path, monkeypatch, capsys):
    """find_methods checks modality types only: on D45 it also returns StabMap
    and scMoMaT, whose mosaic variants need files D45 does not have. The
    printed count keeps the methods with a variant whose files resolve on the
    dataset; a dataset that is not on disk (Colab fetches only D11) gets a
    line instead of a wrong count."""
    import re
    import warnings
    import multibench as mtb
    src = next(s for s in _code(E2E) if "SCENARIOS = {" in s)
    ns = {"mtb": mtb}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(src, ns)
        out = capsys.readouterr().out
        shrunk = set()
        for cat, s in ns["SCENARIOS"].items():
            got = mtb.find_methods(category=cat, modalities=[m.rstrip("123") for m in s["modalities"]])
            sc = mtb.scan(s["dataset"], cat, verbose=False)
            n = len(set(got) & set(sc[sc.files_ok].method))
            assert re.search(rf"^{cat}\s+{s['dataset']}\s+->\s+{n} methods$", out, re.M), (cat, n, out)
            if n < len(got):
                shrunk.add(cat)
    assert "mosaic" in shrunk, "mosaic D45 counts fewer methods than find_methods returns"

    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path)
    exec(src, ns)
    out = capsys.readouterr().out
    assert out.count("not on disk") == len(ns["SCENARIOS"]) and "methods" not in out.replace("its methods", "")


def test_end_to_end_label_order_claims_match_labels_for_and_the_stored_summaries():
    """Section 8's Label order cell says what labels_for(ds, category, method)
    returns and what the stored summaries recorded. Each claim is checked
    against both, so the Concerto caveat fails here - and gets reworded - as
    soon as labels_for returns Concerto's recorded order."""
    import pandas as pd
    import multibench as mtb
    md = " ".join(next(src for kind, src in _cells(E2E)
                       if kind == "markdown" and src.startswith("### Label order")).split())
    assert "`labels_for(dataset, category, method)` returns the label files in that order" in md
    assert 'mtb.labels_for("D52", "cross", "StabMap")) # cty3, cty1, cty2' in md

    def stored(ds, method):
        df = pd.read_csv(ROOT / "notebooks" / "results" / f"summary_{ds}.csv")
        return [Path(f).stem for f in df.set_index("method").at[method, "label_order"].split("+")]

    assert list(mtb.labels_for("D52", "cross", "StabMap")) == stored("D52", "StabMap") == ["cty3", "cty1", "cty2"]
    assert list(mtb.labels_for("D28", "diagonal", "uniPort")) == stored("D28", "uniPort") == ["atac_cty", "rna_cty"]
    assert "StabMap on D52 puts batch 3 first (`cty3+cty1+cty2`)" in md
    assert "uniPort on D28 puts ATAC before RNA" in md
    # the caveat: Concerto's stored run is not in the order labels_for returns,
    # which is the order of a run through mtb.run (its driver lists the shards)
    assert '`labels_for("D52", "cross", "Concerto")` returns `cty1, cty2, cty3`' in md
    assert list(mtb.labels_for("D52", "cross", "Concerto")) == ["cty1", "cty2", "cty3"]
    assert stored("D52", "Concerto") == ["cty3", "cty1", "cty2"]
    assert "`mtb.run` has them listed in batch order" in md
    from multibench.engine import registry
    assert all(v.driver == "engine/drivers/run_concerto.py"
               for v in registry.get("Concerto").variants if v.when["category"] == "cross")
    assert 'UINMF on D52 uses `label_order=["cty1", "cty2"]`' in md
    assert stored("D52", "UINMF") == ["cty1", "cty2"]
