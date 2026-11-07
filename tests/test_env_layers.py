"""A shared layer is downloaded once and hard-linked into each env."""
import importlib.util
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from multibench.engine import envs

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pack_layers", ROOT / "tools/pack_layers.py")
PL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PL)

SP = "lib/python3.9/site-packages"
LIB = f"{SP}/torch/lib/libtorch.so"
# conda-unpack stand-in: fails unless the layer is linked before it runs
UNPACK = ('#!/bin/sh\ntest -f "$(dirname "$0")/../%s" || exit 1\nexit 0\n'
          "_prefix_records = [\n%%s]\n" % LIB)


def _full(path, env):
    files = {LIB: b"T" * 100, f"{SP}/nvidia/cublas/lib.so": b"N" * 50,
             f"{SP}/torch/x.pyc": env.encode(), "bin/python": b"py"}
    records = "".join(f"('{p}', '/build/{env}', 'text'),\n" for p in files)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in {**files, PL.UNPACK: (UNPACK % records).encode()}.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755 if name.startswith("bin/") else 0o644
            tar.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def published(tmp_path, monkeypatch):
    """Two slim archives and their layer behind file:// URLs."""
    src, out = tmp_path / "src", tmp_path / "pub"
    src.mkdir()
    PL.main(["--name", "L", "--out", str(out),
             str(_full(src / "env_a.tar.gz", "env_a")), str(_full(src / "env_b.tar.gz", "env_b"))])
    layers = json.loads((out / "L.json").read_text())
    layers["L"]["urls"] = [(out / "L.tar.gz").as_uri()]
    layers["L"]["unpacked_bytes"] = 150
    monkeypatch.setattr(envs, "packed_layers", lambda: layers)
    monkeypatch.setattr(envs, "packed_manifest", lambda: {
        e: (out / f"{e}.tar.gz").as_uri() for e in ("env_a", "env_b")})
    monkeypatch.setattr(envs, "packed_sizes", lambda: {})
    return out, layers


def test_marker_names_match_the_tool():
    assert (PL.MARKER, PL.LAYER_INFO) == (envs.LAYER_MARKER, envs.LAYER_INFO)


def test_two_envs_share_one_copy(tmp_path, published):
    out, _ = published
    root = tmp_path / "envs"
    assert envs.install_packed("env_a", envs_dir=root)
    (out / "L.tar.gz").unlink()            # the second env must not download it again
    assert envs.install_packed("env_b", envs_dir=root)
    a, b = root / "env_a" / LIB, root / "env_b" / LIB
    assert a.read_bytes() == b"T" * 100
    assert a.stat().st_ino == b.stat().st_ino == (root / ".layers/L" / LIB).stat().st_ino
    assert (root / "env_a" / SP / "torch/x.pyc").read_bytes() == b"env_a"
    assert not (root / "env_a" / envs.LAYER_INFO).exists()
    # removing one env, or the layer folder, leaves the other env whole
    shutil.rmtree(root / "env_a")
    shutil.rmtree(root / ".layers")
    assert b.read_bytes() == b"T" * 100


def test_a_layer_with_another_digest_installs_nothing(tmp_path, published, capsys):
    _, layers = published
    root = tmp_path / "envs"
    stale = root / ".layers" / "L"
    stale.mkdir(parents=True)
    (stale / envs.LAYER_INFO).write_text(json.dumps({"name": "L", "digest": "other"}))
    assert envs.install_packed("env_a", envs_dir=root) is False
    assert not (root / "env_a").exists()
    assert "not the one this archive was packed with" in capsys.readouterr().out


def test_an_unlisted_layer_installs_nothing(tmp_path, published, monkeypatch):
    monkeypatch.setattr(envs, "packed_layers", lambda: {})
    assert envs.install_packed("env_a", envs_dir=tmp_path / "envs") is False
    assert not (tmp_path / "envs" / "env_a").exists()


def test_two_processes_use_their_own_temporary_names():
    """Two installs into one shared envs folder download and unpack into
    their own files."""
    import subprocess
    import sys
    code = ("from pathlib import Path; from multibench.engine import envs; "
            "print(envs._own_name(Path('/e/.L.tar.gz.download')))")
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT)}) for _ in range(2)]
    names = [p.communicate()[0].strip() for p in procs]
    assert all(n.startswith("/e/.L.tar.gz.download.") for n in names)
    assert names[0] != names[1]


def _finished_meanwhile(monkeypatch, label, finish):
    """Run ``finish()`` (another install completing) right after the download
    of ``label``."""
    real = envs.download_archive

    def download(urls, target, **kw):
        n = real(urls, target, **kw)
        if kw["label"] == label:
            finish()
        return n
    monkeypatch.setattr(envs, "download_archive", download)


def test_a_layer_another_install_unpacked_meanwhile_is_kept(tmp_path, published, monkeypatch):
    out, _ = published
    root = tmp_path / "envs"
    layer = root / ".layers" / "L"
    theirs = {}

    def finish():
        with tarfile.open(out / "L.tar.gz") as t:
            t.extractall(layer, filter="data")
        theirs["ino"] = (layer / envs.LAYER_INFO).stat().st_ino
    _finished_meanwhile(monkeypatch, "L (shared layer)", finish)
    assert envs.install_packed("env_a", envs_dir=root)
    assert (layer / envs.LAYER_INFO).stat().st_ino == theirs["ino"]
    assert (root / "env_a" / LIB).stat().st_ino == (layer / LIB).stat().st_ino
    assert sorted(p.name for p in (root / ".layers").iterdir()) == ["L"]


def test_an_env_another_install_finished_meanwhile_is_kept(tmp_path, published, monkeypatch):
    root = tmp_path / "envs"
    theirs = root / "env_a"

    def finish():
        (theirs / "bin").mkdir(parents=True)
        (theirs / envs.FLAVOR_FILE).write_text("single\n")
    _finished_meanwhile(monkeypatch, "env_a", finish)
    envs.install_packed("env_a", envs_dir=root)
    assert (theirs / envs.FLAVOR_FILE).read_text() == "single\n"
    assert sorted(p.name for p in root.iterdir()) == [".layers", "env_a"]


def test_layers_for_counts_a_layer_once_and_skips_one_on_disk(tmp_path, published):
    _, layers = published
    assert list(envs.layers_for(["env_a", "env_b"], envs_dir=tmp_path)) == ["L"]
    assert envs.layers_for(["other"], envs_dir=tmp_path) == {}
    (tmp_path / ".layers/L").mkdir(parents=True)
    (tmp_path / ".layers/L" / envs.LAYER_INFO).write_text("{}")
    assert envs.layers_for(["env_a"], envs_dir=tmp_path) == {}
    assert list(envs.layers_for(["env_a"], envs_dir=tmp_path, missing_only=False)) == ["L"]


def test_total_line_adds_the_layer_once(published):
    _, layers = published
    sizes = {e: {"archive_bytes": 2_000_000_000, "unpacked_bytes": 3_000_000_000}
             for e in ("env_a", "env_b")}
    layers["L"].update(archive_bytes=4_000_000_000, unpacked_bytes=6_000_000_000)
    totals, text = envs.plan_totals([{"env": "env_a"}, {"env": "env_b"}], sizes, layers=layers)
    assert text == ("2 envs: 8.0 GB to download, 12.0 GB on disk "
                    "(with the shared layer L, 4.0 GB, downloaded once)")
    assert totals["shared_layers"] == {"L": {"archive_bytes": 4_000_000_000,
                                             "unpacked_bytes": 6_000_000_000,
                                             "envs": ["env_a", "env_b"]}}
    assert envs.plan_totals([{"env": "env_a"}], sizes, layers={})[1] == \
        "1 env: 2.0 GB to download, 3.0 GB on disk"


def test_size_tool_names_a_layer_not_published_at_its_size():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import packed_sizes as tool

    class _Resp:
        headers = {"Content-Length": "60"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req):
            if "missing" in req.full_url:
                raise OSError("404")
            return _Resp()
    layers = {"ok": {"urls": ["https://x/a.001", "https://x/a.002"], "archive_bytes": 120},
              "short": {"urls": ["https://x/b.001"], "archive_bytes": 120},
              "gone": {"urls": ["https://x/missing"], "archive_bytes": 60}}
    assert tool.stale_layers(layers, opener=_Opener()) == ["gone", "short"]
