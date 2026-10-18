"""Smaller items of the API review (D25)."""
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np

import multibench as mtb
from multibench.engine.runner import RunResult

ROOT = Path(__file__).resolve().parent.parent


def test_io_catalog_env_import_as_submodules():
    code = ("import multibench.io, multibench.catalog, multibench.env\n"
            "from multibench.io import export_dataset\n"
            "from multibench.env import plan\n"
            "import multibench as mtb\n"
            "assert multibench.io is mtb.io and multibench.env is mtb.env\n"
            "assert multibench.catalog is mtb.catalog and export_dataset is mtb.io.export_dataset\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT,
                   env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})


def test_runresult_repr_is_short():
    r = RunResult(method="totalVI", out_dir="out/totalVI", cmd=["x"],
                  output=np.zeros((5000, 10)), extra={"predict.csv": "x" * 1000},
                  stdout="y" * 50000, obs_names=[f"c{i}" for i in range(5000)])
    assert repr(r) == ("RunResult(method='totalVI', out_dir='out/totalVI', output=ndarray "
                       "(5000, 10), extra=['predict.csv'], obs_names=5,000 cells)")


def test_rescore_prints_by_default():
    assert inspect.signature(mtb.BatchResult.rescore).parameters["verbose"].default is True


def test_data_has_a_module_docstring():
    # 0.4.0 calls them demo datasets (R10)
    assert mtb.data.__doc__.startswith("Demo data: ``fetch`` downloads demo datasets")


def test_fetchable_example_is_the_call():
    shown = inspect.getdoc(mtb.data.fetchable).split(">>> mtb.data.fetchable()\n", 1)[1]
    shown = shown.split("\n\n", 1)[0]
    assert eval(shown) == mtb.data.fetchable()
