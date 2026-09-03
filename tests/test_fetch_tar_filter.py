"""``tar.extractall`` passes ``filter='data'`` where the interpreter has it.

The bare call raises ``DeprecationWarning`` on Python 3.12/3.13 (and changes
behaviour on 3.14); the shipped tarballs (datasets, packed envs) are plain data,
so the ``data`` filter is right. The traversal guard stays in front of it.
"""
import importlib
import tarfile
import warnings

import pytest

# `multibench.data.fetch` the attribute is the fetch() FUNCTION; this is the module
fetch = importlib.import_module("multibench.data.fetch")


def _tar(tmp_path, entries):
    tgz = tmp_path / "a.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        for name, text in entries:
            p = tmp_path / "payload"
            p.write_text(text)
            t.add(p, arcname=name)
    return tgz


def test_safe_extract_emits_no_deprecation_warning(tmp_path):
    tgz = _tar(tmp_path, [("D99/rna.h5", "x"), ("D99/cty.csv", "y")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with tarfile.open(tgz) as t:
            fetch.safe_extract(t, tmp_path / "out")
    assert (tmp_path / "out" / "D99" / "rna.h5").read_text() == "x"
    assert (tmp_path / "out" / "D99" / "cty.csv").read_text() == "y"


@pytest.mark.skipif(not hasattr(tarfile, "data_filter"), reason="tarfile has no extraction filters")
def test_safe_extract_passes_the_data_filter(tmp_path, monkeypatch):
    seen = {}
    tgz = _tar(tmp_path, [("D99/a.txt", "x")])
    with tarfile.open(tgz) as t:
        real = t.extractall

        def spy(path, *a, **kw):
            seen.update(kw)
            return real(path, *a, **kw)
        monkeypatch.setattr(t, "extractall", spy)
        fetch.safe_extract(t, tmp_path / "out")
    assert seen.get("filter") == "data"


def test_traversal_guard_still_fires_before_extraction(tmp_path):
    tgz = tmp_path / "evil.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        p = tmp_path / "payload"
        p.write_text("z")
        t.add(p, arcname="../escape.txt")
    with tarfile.open(tgz) as t:
        with pytest.raises(RuntimeError, match="escapes target dir"):
            fetch.safe_extract(t, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_install_packed_goes_through_safe_extract():
    import inspect
    from multibench.engine import envs
    src = inspect.getsource(envs.install_packed)
    assert "safe_extract(t, part)" in src and "extractall" not in src
