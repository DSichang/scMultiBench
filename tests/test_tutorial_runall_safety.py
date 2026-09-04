"""'Run all' safety of the generated category tutorials.

The re-test after polish round 2 found the linux-gated env-install cell
starting a 3 GB packed-env download on every Colab attendee (Colab is linux),
the run cells executing for real on a CPU runtime, and section 5's
``avail[avail.runnable]`` table empty on every host without environments.
These tests pin the fix: one ``INSTALL_ENVS = False`` flag cell that every
environment download sits behind, run cells gated on ``scan().env_ok`` with a
stored-sweep stand-in otherwise, library-specific warning filters, and the
file-gate table with the env columns next to it.
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


def _cells(cat):
    nb = json.loads((ROOT / "notebooks" / f"tutorial_{cat}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _code(cat):
    return [src for kind, src in _cells(cat) if kind == "code"]


def _markdown(cat):
    return "\n".join(src for kind, src in _cells(cat) if kind == "markdown")


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


@pytest.mark.parametrize("cat", CATS)
def test_flag_cell_present_default_false_and_before_any_download(cat):
    code = _code(cat)
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
    first_download = min(i for i, src in enumerate(code)
                         if "condacolab" in src or "env install" in src)
    assert flag_cells[0] < first_download, f"tutorial_{cat}: the flag cell precedes every download"
    assert "GB to download" in code[flag_cells[0]], "the flag comment states the measured size"


@pytest.mark.parametrize("cat", CATS)
def test_no_code_cell_downloads_environments_unconditionally(cat):
    for src in _code(cat):
        tree = _tree(src)
        for node, cmd in _shell_calls(tree):
            if "env install" in cmd or "condacolab" in cmd:
                assert _guarded_by(node, "INSTALL_ENVS"), \
                    f"tutorial_{cat}: environment download not behind INSTALL_ENVS: {cmd!r}"
        # condacolab.install() restarts the kernel - it must sit behind the flag too
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and "condacolab.install" in ast.unparse(node.func):
                assert _guarded_by(node, "INSTALL_ENVS"), f"tutorial_{cat}: condacolab.install() unguarded"


@pytest.mark.parametrize("cat", CATS)
def test_run_cells_execute_only_when_scan_finds_an_environment(cat):
    n_runs = 0
    for src in _code(cat):
        tree = _tree(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "mtb.run_all":
                n_runs += 1
                assert _guarded_by(node, "env_ok"), \
                    f"tutorial_{cat}: run_all not gated on scan().env_ok: {ast.unparse(node)[:80]}"
                assert GEN.SKIP_LINE in src and "stored_sweep(" in src, \
                    f"tutorial_{cat}: a skipped run prints the one line and stands in the stored sweep"
    assert n_runs == 2, f"tutorial_{cat}: section 2 and section 3 each run once"
    assert GEN.SKIP_LINE == ("no method environment on this host - the run is skipped; "
                             "the stored results below cover it")


@pytest.mark.parametrize("cat", CATS)
def test_stored_sweep_stand_in_renders_summary_and_figure(cat):
    """The fallback the run cells take on a laptop: the same object run_all
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
    if methods:
        assert sorted(res.summary.method) == sorted(methods)
    else:
        assert len(res.summary) >= 2
    assert {"ARI", "NMI"} <= set(res.summary.columns)
    fig = res.plot()
    assert fig.axes
    plt.close(fig)


@pytest.mark.parametrize("cat", CATS)
def test_warning_filters_are_library_specific(cat):
    code = _code(cat)
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
    code = "\n".join(_code(cat))
    assert "avail[avail.files_ok]" in code and '"env_ok", "env_reason"' in code
    assert "avail[avail.runnable]" not in code, "empty on every host without environments"
    assert "avail[~avail.files_ok]" in code and '"files_reason"' in code


@pytest.mark.parametrize("cat", CATS)
def test_batch_metrics_prose_names_the_metrics_knob(cat):
    """The one selector is metrics= (0.3.0): the prose shows the family form
    and says what the default None computes; the 0.2 task= never returns."""
    md = _markdown(cat)
    assert 'labels=mtb.labels_for(DATASET), metrics="all")' in md
    assert "`None` (the default) computes every\napplicable metric" in md \
        or "`None` (the default) computes every applicable metric" in md
    assert 'task="all"' not in md and 'task="clustering"' not in md


@pytest.mark.parametrize("cat", CATS)
def test_coverage_cell_derives_from_the_registry(cat):
    code = "\n".join(_code(cat))
    assert "IMPUTATION_ONLY" not in code
    assert "info['tasks']" in code and "info['categories']" in code
    assert "not in the registry" in code


@pytest.mark.parametrize("cat", CATS)
def test_every_load_results_call_names_its_source(cat):
    n = 0
    for src in _code(cat):
        for node in ast.walk(_tree(src)):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "mtb.load_results":
                n += 1
                assert any(k.arg == "source" for k in node.keywords), \
                    f"tutorial_{cat}: load_results without source=: {ast.unparse(node)[:80]}"
    assert n >= 3            # stand-in helper, section 4 single, section 4 pair
    md = _markdown(cat)
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
