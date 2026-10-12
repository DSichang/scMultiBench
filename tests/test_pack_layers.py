"""tools/pack_layers.py: the layer holds what every archive holds alike."""
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pack_layers", ROOT / "tools/pack_layers.py")
PL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PL)

SP = "lib/python3.9/site-packages"


def _archive(path, env, files, prefix=None):
    prefix = prefix or f"/build/envs/{env}"
    records = "".join(f"('{p}', '{prefix}', 'text'),\n" for p in files)
    unpack = f"#!python\n_prefix_records = [\n{records}]\nprint('unpack')\n"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in {**files, PL.UNPACK: unpack.encode()}.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755 if name == PL.UNPACK else 0o644
            tar.addfile(info, io.BytesIO(data))
    return path


def _members(path):
    with tarfile.open(path) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isreg()}


def _two(tmp_path):
    common = {f"{SP}/torch/lib/libtorch.so": b"T" * 100, f"{SP}/nvidia/cublas/lib.so": b"N" * 50}
    a = _archive(tmp_path / "env_a.tar.gz", "env_a", {
        **common, f"{SP}/torch/__pycache__/x.pyc": b"/build/envs/env_a", "bin/python": b"py",
        f"{SP}/numpy/core.so": b"same"})
    b = _archive(tmp_path / "env_b.tar.gz", "env_b", {
        **common, f"{SP}/torch/__pycache__/x.pyc": b"/build/envs/env_b", "bin/python": b"py",
        f"{SP}/numpy/core.so": b"same", f"{SP}/nvidia/cudnn/extra.so": b"only b"})
    return a, b, common


def test_layer_and_slim_archives(tmp_path):
    a, b, common = _two(tmp_path)
    out = tmp_path / "out"
    assert PL.main(["--name", "L", "--out", str(out), str(a), str(b)]) == 0
    layer = _members(out / "L.tar.gz")
    info = json.loads(layer.pop(PL.LAYER_INFO))
    assert layer == common and info["files"] == 2
    for env, own in (("env_a", set()), ("env_b", {f"{SP}/nvidia/cudnn/extra.so"})):
        slim = _members(out / f"{env}.tar.gz")
        # a file that differs, one outside the roots and one of a single env stay
        assert {f"{SP}/torch/__pycache__/x.pyc", f"{SP}/numpy/core.so", "bin/python"} | own \
            <= slim.keys()
        assert not common.keys() & slim.keys()
        assert json.loads(slim[PL.MARKER]) == {"layers": {"L": info}}
        unpack = slim[PL.UNPACK].decode()
        assert "libtorch.so" not in unpack and "x.pyc" in unpack and "print('unpack')" in unpack
    entry = json.loads((out / "L.json").read_text())["L"]
    assert entry["envs"] == ["env_a", "env_b"] and entry["digest"] == info["digest"]
    assert entry["archive_bytes"] == (out / "L.tar.gz").stat().st_size


def test_a_shared_file_that_holds_its_prefix_is_refused(tmp_path):
    files = {f"{SP}/torch/config.py": b"home = '/build/envs/one'", "bin/python": b"py"}
    a = _archive(tmp_path / "env_a.tar.gz", "env_a", files, prefix="/build/envs/one")
    b = _archive(tmp_path / "env_b.tar.gz", "env_b", files, prefix="/build/envs/one")
    with pytest.raises(SystemExit, match="holds the prefix /build/envs/one"):
        PL.main(["--name", "L", "--out", str(tmp_path / "out"), str(a), str(b)])
