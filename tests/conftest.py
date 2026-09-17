from pathlib import Path
import pytest

# Repo root = two levels up from this file (tests/ -> <ROOT>)
ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture
def root():
    return ROOT

@pytest.fixture
def files_dir(root):
    return root / "multibench" / "files"

@pytest.fixture
def result_dir(root):
    return root / "multibench" / "result"


# ---- host-agnostic run mode -------------------------------------------------
# Most tests assert the classic ``conda run -n <env>`` command line. On a host
# where the envs exist as prefixes (the benchmark host, or a laptop that
# unpacked one) the runner would pick prefix mode and those assertions would
# fail for a reason unrelated to what they test. Pin conda mode everywhere
# except in the prefix-mode tests themselves, which set the mode explicitly.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _pin_conda_run_mode(request, monkeypatch):
    if request.node.fspath.basename in ("test_prefix_mode.py", "test_env_flavor.py"):
        return
    monkeypatch.setenv("MULTIBENCH_RUN_MODE", "conda")


# ---- a results root with every published layout the loader reads ------------
def _metric_csv(path: Path, **values: float) -> None:
    """Write a published-style metric table (unnamed index column + Value)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",Value\n" + "".join(f"{k},{v}\n" for k, v in values.items()))


def _kbet_dir(method_dir: Path) -> None:
    """A raw kBET output folder - not a metric table, must be skipped."""
    (method_dir / "kbet").mkdir(parents=True, exist_ok=True)
    (method_dir / "kbet" / "benchmark_results111.csv").write_text("a,b\n1,2\n")


def _clustering(v: float, clisi: float = 0.9) -> dict:
    return dict(ARI=v, NMI=v, ASW=v, iASW=v, iF1=v, cLISI=clisi)


#: cross/D53 of the layout tree: six methods with a table, four of them
#: listed by list_methods('cross'); sciPENN best on every metric
_D53 = {
    "sciPENN": _clustering(0.90, 0.99),
    "scMDC": _clustering(0.80, 0.98),
    "scMM": _clustering(0.70, 0.97),
    "Concerto": _clustering(0.60, 0.96),
    "MOFA2": _clustering(0.50, 0.95),          # not a cross method of the package
    "Multigrate": _clustering(0.40, 0.94),     # not a cross method of the package
}


@pytest.fixture
def layout_tree(tmp_path):
    """A results root exercising every published layout ``load_results``
    reads: single-method datasets (cross D52/D54/D58/D59), a table kept one
    level down in a run-configuration folder (D53 ``MOFA2/8000HVG``, D56
    ``MOFA2/filtered3`` + ``MOFA2/kmeans``, D57 ``MOFA2/filtered5``), raw
    ``kbet/`` folders,
    method folders without a metric table (D53 StabMap/totalVI), methods the
    registry does not list for cross (MOFA2, Multigrate), a
    ``<method>_louvain`` variant directory (vertical D3 ``Concerto_louvain``)
    and two simulated ids whose natural and lexicographic orders differ
    (diagonal SD7, SD10). No ``rerun/`` sweeps."""
    root = tmp_path / "results"
    cross = root / "scib_metric" / "cross integration"
    for ds in ("D52", "D58", "D59"):
        _metric_csv(cross / ds / "scMoMaT" / "metric.csv", **_clustering(0.5))
    for meth, vals in _D53.items():
        sub = cross / "D53" / meth
        _metric_csv(sub / ("8000HVG/metric.csv" if meth == "MOFA2" else "metric.csv"), **vals)
    _kbet_dir(cross / "D53" / "MOFA2")
    for meth in ("StabMap", "totalVI"):
        (cross / "D53" / meth).mkdir(parents=True)
        (cross / "D53" / meth / "benchmark_results111.csv").write_text("a,b\n1,2\n")
    _metric_csv(cross / "D54" / "Concerto" / "metric.csv", **_clustering(0.4))
    _kbet_dir(cross / "D54" / "Concerto")
    d56 = cross / "D56" / "MOFA2"
    _metric_csv(d56 / "filtered3" / "metric.csv", ARI=0.285746391962993, NMI=0.55,
                ASW=0.47, iASW=0.5, iF1=0.5, cLISI=0.9)
    _metric_csv(d56 / "filtered3" / "metric_ari_nmi_batch.csv", ARI=0.99, NMI=0.99)
    _metric_csv(d56 / "filtered3" / "metric_louvain.csv", ARI=0.30, NMI=0.56, ASW=0.48)
    _metric_csv(d56 / "kmeans" / "metric_kmeans.csv", ARI=0.25, NMI=0.50, ASW=0.45)
    _kbet_dir(d56)
    _metric_csv(cross / "D57" / "MOFA2" / "filtered5" / "metric.csv", **_clustering(0.3))
    _kbet_dir(cross / "D57" / "MOFA2")
    _metric_csv(cross / "D57" / "UINMF" / "metric.csv", **_clustering(0.35))
    vert = root / "scib_metric" / "vertical integration" / "D3"
    _metric_csv(vert / "Concerto_louvain" / "metric.csv", ARI=0.7, NMI=0.7, ASW=0.7,
                iASW=0.7, iF1=0.7)
    for meth, v in (("Multigrate", 0.6), ("moETM", 0.5), ("scMM", 0.55), ("scMSI", 0.5),
                    ("scMoMaT", 0.45), ("sciPENN", 0.65)):
        _metric_csv(vert / meth / "metric.csv", ARI=v, NMI=v, ASW=v, iASW=v, iF1=v)
    for ds in ("SD7", "SD10"):
        _metric_csv(root / "scib_metric" / "diagonal integration" / ds / "GLUE" / "metric.csv",
                    **_clustering(0.5))
    return root
