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

    class FakePopen:
        # the runner spawns via Popen(start_new_session=True) so a timeout
        # can kill the whole process tree; the fake mirrors that interface
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            calls["cmd"] = cmd
            calls["cwd"] = cwd
            self.pid = 4242
            self.returncode = 0
            with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
                f.create_dataset("data", data=np.zeros((6, 3)))

        def communicate(self):
            return "", ""

        def kill(self):
            pass

        def wait(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)

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

    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            self.pid = 4242
            self.returncode = 1

        def communicate(self):
            return "boom-out", "boom-err"

        def kill(self):
            pass

        def wait(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)

    with pytest.raises(RuntimeError) as excinfo:
        mtb.run(method="SCALEX", category="diagonal",
                inputs={"rna": str(tmp_path/"a.h5"), "atac_gas": str(tmp_path/"b.h5")},
                out_dir=str(tmp_path/"out"), convert=False,
                cmd_template="conda run -n scalex {cmd}")
    msg = str(excinfo.value)
    assert "boom-err" in msg
    assert "boom-out" in msg


# --- P12: run() preflights the conda env before touching the disk -----------

def _fake_popen_factory(calls):
    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            calls["cmd"] = cmd
            self.pid = 4242
            self.returncode = 0
            with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
                f.create_dataset("data", data=np.zeros((6, 3)))

        def communicate(self):
            return "", ""

        def kill(self):
            pass

        def wait(self):
            return self.returncode
    return FakePopen


def test_run_preflight_raises_when_env_missing(tmp_path, monkeypatch):
    import pytest
    import multibench as mtb
    from multibench.engine import runner

    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])

    def must_not_spawn(*a, **k):
        raise AssertionError("Popen must not be reached when the env is missing")
    monkeypatch.setattr(runner.subprocess, "Popen", must_not_spawn)

    out = tmp_path / "out"
    with pytest.raises(EnvironmentError) as e:
        mtb.run(method="SCALEX", category="diagonal",
                inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                out_dir=str(out), convert=False)
    msg = str(e.value)
    assert "is not installed" in msg
    assert "multibench env install --methods SCALEX" in msg
    # R7-10: the Python call follows the shell command (the doctor pointer is gone)
    assert "In Python, call mtb.env.install(['SCALEX'], dry_run=False)." in msg
    assert not out.exists(), "preflight must fail before anything is written"


def test_run_preflight_skipped_with_cmd_template(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    calls = {}
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(calls))
    res = mtb.run(method="SCALEX", category="diagonal",
                  inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                  out_dir=str(tmp_path / "out"), convert=False,
                  cmd_template="conda run -n scalex {cmd}")
    assert calls["cmd"][:4] == ["conda", "run", "-n", "scalex"]
    assert res.output.shape == (6, 3)


def test_run_preflight_skipped_when_probe_empty(tmp_path, monkeypatch):
    import multibench as mtb
    from multibench.engine import runner
    calls = {}
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen_factory(calls))
    res = mtb.run(method="SCALEX", category="diagonal",
                  inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                  out_dir=str(tmp_path / "out"), convert=False)
    # default template still targets the method's env; no false block
    assert "run" in calls["cmd"] and runner.envs.group_for("SCALEX") in calls["cmd"]
    assert res.output.shape == (6, 3)


# --- the dry run applies the conversion's feature-type check -----------------

def test_the_dry_run_refuses_two_feature_types(tmp_path, monkeypatch):
    """A 10x object that was not split: the conversion of a real run refuses
    it, and the dry run returned a command for it."""
    import shutil

    import pandas as pd
    import pytest
    import multibench as mtb
    from multibench.engine import envs, runner
    ad = pytest.importorskip("anndata")

    y = ad.AnnData(np.random.default_rng(0).poisson(1.0, (20, 120)).astype(float))
    y.obs_names = [f"cell{i}-1" for i in range(20)]
    y.var_names = [f"GENE{i}" for i in range(50)] \
        + [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(70)]
    y.var["feature_types"] = pd.Categorical(["Gene Expression"] * 50 + ["Peaks"] * 70)
    y.layers["counts"] = y.X.copy()
    kind = y.var["feature_types"]
    rna, atac = y[:, kind == "Gene Expression"].copy(), y[:, kind == "Peaks"].copy()
    files = {}
    for name, obj in (("unsplit", y), ("rna", rna), ("atac", atac)):
        files[name] = str(tmp_path / f"{name}.h5ad")
        obj.write_h5ad(files[name])
    head = ("The matrix holds features of 2 types, 'Peaks' and 'Gene Expression'. Pass one "
            "type, as adata[:, adata.var['feature_types'] == ")
    texts = {"rna": head + "'Gene Expression'].",
             "atac": head + "'Peaks'].",
             "atac_gas": head + "'<type>']."}

    out = tmp_path / "out"
    (tmp_path / "repo" / "tools_scripts").mkdir(parents=True)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(runner, "provide_script_files", lambda *a, **k: None)
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("the method was started"))
    real_kw = dict(out_dir=str(out), cmd_template="{cmd}", repo_path=str(tmp_path / "repo"))
    for method, category, roles in (
            ("VIMCCA", "vertical", {"rna": "unsplit", "atac": "unsplit"}),
            ("VIMCCA", "vertical", {"rna": "rna", "atac": "unsplit"}),
            ("SCALEX", "diagonal", {"rna": "rna", "atac_gas": "unsplit"})):
        role = next(r for r, name in roles.items() if name == "unsplit")
        for inputs in ({r: {"unsplit": y, "rna": rna}[n] for r, n in roles.items()},
                       {r: files[n] for r, n in roles.items()}):
            # what the conversion of the real run says, without the check before it
            with monkeypatch.context() as m:
                m.setattr(runner, "_check_feature_types", lambda role, var: None)
                with pytest.raises(ValueError) as conversion:
                    mtb.run(method, category, inputs=inputs, **real_kw)
            assert str(conversion.value) == texts[role]
            shutil.rmtree(out)                  # the conversion had made inputs/
            # the dry run says the same, and so does the real run, before it writes
            with pytest.raises(ValueError) as dry:
                mtb.run(method, category, inputs=inputs, out_dir=str(out), dry_run=True)
            with pytest.raises(ValueError) as real:
                mtb.run(method, category, inputs=inputs, **real_kw)
            assert str(dry.value) == str(real.value) == texts[role]
            assert not out.exists()

    # of an .h5ad file the dry run reads var and nothing else
    with monkeypatch.context() as m:
        m.setattr(ad, "read_h5ad", lambda *a, **k: pytest.fail("the file was read"))
        with pytest.raises(ValueError) as dry:
            mtb.run("VIMCCA", "vertical", out_dir=str(out), dry_run=True,
                    inputs={"rna": files["unsplit"], "atac": files["unsplit"]})
        assert str(dry.value) == texts["rna"]
        # the split objects and their files return the command
        for inputs in ({"rna": rna, "atac": atac},
                       {"rna": files["rna"], "atac": files["atac"]}):
            argv = mtb.run("VIMCCA", "vertical", inputs=inputs, out_dir=str(out),
                           dry_run=True)
            assert argv[-6:] == ["--path1", f"{out}/inputs/rna.h5", "--path2",
                                 f"{out}/inputs/atac.h5", "--save_path", f"{out}/"]
        # a file that is no .h5ad inside is left to the conversion, as before
        (tmp_path / "broken.h5ad").write_text("not hdf5")
        argv = mtb.run("VIMCCA", "vertical", out_dir=str(out), dry_run=True,
                       inputs={"rna": str(tmp_path / "broken.h5ad"), "atac": atac})
        assert argv[-5] == f"{out}/inputs/rna.h5"
    assert not out.exists()
