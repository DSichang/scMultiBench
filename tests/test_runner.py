from pathlib import Path

import numpy as np
import h5py
from multibench.engine import io as rio
from multibench.engine.schema import OutputSpec


def test_load_embedding_output(tmp_path):
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((10, 5)))
    spec = OutputSpec(kind="embedding", file="embedding.h5", dataset="data")
    arr = rio.load_output(tmp_path, spec)
    assert arr.shape == (10, 5)


def test_load_graph_output(tmp_path):
    p = tmp_path / "knn_indices.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.arange(12).reshape(4, 3))
    spec = OutputSpec(kind="graph", file="knn_indices.h5", dataset="data")
    arr = rio.load_output(tmp_path, spec)
    assert arr.shape == (4, 3)
    assert int(arr[0, 0]) == 0 and int(arr[3, 2]) == 11


def test_load_labels_output(tmp_path):
    p = tmp_path / "Prediction.csv"
    p.write_text("B\nT\nNK\n")
    spec = OutputSpec(kind="labels", file="Prediction.csv")
    labels = rio.load_output(tmp_path, spec)
    assert list(labels) == ["B", "T", "NK"]


def test_wrap_cmd_template():
    from multibench.engine import runner
    wrapped = runner.wrap_command(["python", "x.py", "--a", "1"],
                                  cmd_template="conda run -n scalex {cmd}")
    assert wrapped[:4] == ["conda", "run", "-n", "scalex"]
    assert wrapped[-1] == "1"


def test_run_invokes_subprocess_and_loads_output(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    import numpy as np, h5py

    calls = {}

    def fake_run(cmd, cwd, capture_output, text):
        # simulate the method writing embedding.h5 into the out_dir (cwd)
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=np.zeros((6, 3)))
        class R: returncode = 0; stdout = ""; stderr = ""
        return R()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    res = mtb.run(method="SCALEX", category="diagonal",
                  inputs={"rna": str(tmp_path/"a.h5"), "atac_gas": str(tmp_path/"b.h5")},
                  out_dir=str(tmp_path/"out"), convert=False,
                  cmd_template="conda run -n scalex {cmd}")
    assert "conda" in calls["cmd"][0]
    assert res.output.shape == (6, 3)


def test_run_surfaces_stdout_and_stderr_on_failure(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    import pytest

    def fake_run(cmd, cwd, capture_output, text):
        class R: returncode = 1; stdout = "boom-out"; stderr = "boom-err"
        return R()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        mtb.run(method="SCALEX", category="diagonal",
                inputs={"rna": str(tmp_path/"a.h5"), "atac_gas": str(tmp_path/"b.h5")},
                out_dir=str(tmp_path/"out"), convert=False,
                cmd_template="conda run -n scalex {cmd}")
    msg = str(excinfo.value)
    assert "boom-err" in msg
    assert "boom-out" in msg
