"""Fakes for the archive downloads of ``envs.download_archive``."""
from pathlib import Path


class _Served:
    """A urlopen response over a local file, as ``download_archive`` reads it."""
    status = 200

    def __init__(self, data: bytes):
        import io
        self._buf = io.BytesIO(data)

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve_archives(monkeypatch, fetch):
    """Answer archive downloads with ``fetch(url)``, which returns
    ``(local_path, None)`` like ``urlretrieve`` or raises (e.g. HTTPError)."""
    import urllib.request

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        path, _ = fetch(url)
        return _Served(Path(path).read_bytes())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
