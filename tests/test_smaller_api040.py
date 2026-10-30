"""Smaller items of the API review (D25)."""
import importlib
import inspect
import subprocess
import sys
from pathlib import Path


import multibench as mtb

ROOT = Path(__file__).resolve().parent.parent


def test_io_catalog_env_import_as_submodules():
    code = ("import multibench.io, multibench.catalog, multibench.env\n"
            "from multibench.io import export_dataset\n"
            "from multibench.env import install\n"
            "import multibench as mtb\n"
            "assert multibench.io is mtb.io and multibench.env is mtb.env\n"
            "assert multibench.catalog is mtb.catalog and export_dataset is mtb.io.export_dataset\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT,
                   env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})


def test_rescore_prints_by_default():
    assert inspect.signature(mtb.RunResult.rescore).parameters["verbose"].default is True


def test_data_has_a_module_docstring():
    # 0.4.0 calls them demo datasets (R10)
    assert mtb.data.__doc__.startswith("Demo data: ``fetch`` downloads demo datasets")


