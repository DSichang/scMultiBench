"""Archive downloads: parts in order, progress lines, resume after a dropped
connection, and no leftover file.

Archives over 2 GiB are published as parts (GitHub release assets must be
under 2 GiB); a download that printed nothing for minutes looked frozen on
Colab, and a stalled connection waited forever.
"""
import io
import tarfile
import urllib.error
import urllib.request

import pytest

from multibench.engine import envs


class _Resp:
    def __init__(self, data, status=200, fail_after=None):
        self._buf, self.status, self._fail_after = io.BytesIO(data), status, fail_after
        self._sent = 0

    def read(self, n=-1):
        if self._fail_after is not None and self._sent >= self._fail_after:
            raise TimeoutError("stalled")
        chunk = self._buf.read(n if self._fail_after is None else min(n, self._fail_after - self._sent) or n)
        self._sent += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, files, *, drop_first=None):
    """``files``: {url: bytes}. ``drop_first``: url whose first answer stalls
    after that many bytes. Records every (url, Range header)."""
    calls = []

    def urlopen(req, timeout=None):
        url, rng = req.full_url, req.get_header("Range")
        calls.append((url, rng))
        data = files[url]
        if data is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return _Resp(data[start:], status=206)
        if drop_first == url and sum(u == url for u, _ in calls) == 1:
            return _Resp(data, fail_after=len(data) // 2)
        return _Resp(data)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls


def test_parts_are_joined_in_order(monkeypatch, tmp_path, capsys):
    parts = {"https://u/a.001": b"A" * 3000, "https://u/a.002": b"B" * 2000}
    _serve(monkeypatch, parts)
    n = envs.download_archive(list(parts), tmp_path / "x", label="demo", total=5000)
    assert n == 5000
    assert (tmp_path / "x").read_bytes() == b"A" * 3000 + b"B" * 2000
    out = capsys.readouterr().out
    assert out.startswith("[env] downloading demo 0 MB ...\n")


def test_progress_is_printed_per_tenth(monkeypatch, tmp_path, capsys):
    data = b"x" * (10 << 20)
    _serve(monkeypatch, {"https://u/big": data})
    envs.download_archive(["https://u/big"], tmp_path / "x", label="big", total=len(data))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if " of " in ln]
    assert len(lines) == 10
    assert lines[-1].startswith("[env]   10 MB of 10 MB (")


def test_a_dropped_connection_resumes_with_a_range(monkeypatch, tmp_path, capsys):
    data = bytes(range(256)) * 40000
    calls = _serve(monkeypatch, {"https://u/p": data}, drop_first="https://u/p")
    envs.download_archive(["https://u/p"], tmp_path / "x", label="p", total=len(data))
    assert (tmp_path / "x").read_bytes() == data
    assert calls[0] == ("https://u/p", None) and calls[1][1].startswith("bytes=")
    assert "connection lost (TimeoutError); resuming" in capsys.readouterr().out


def test_a_404_is_raised_at_once(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, {"https://u/gone": None})
    with pytest.raises(urllib.error.HTTPError):
        envs.download_archive(["https://u/gone"], tmp_path / "x", label="g")
    assert len(calls) == 1


def test_a_server_error_is_retried(monkeypatch, tmp_path):
    """GitHub answered 500 for the second part of a new release asset, twice."""
    calls, data = [], b"x" * 100

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 500, "Internal Server Error", {}, None)
        return _Resp(data)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert envs.download_archive(["https://u/p"], tmp_path / "x", label="p") == 100
    assert len(calls) == 2 and (tmp_path / "x").read_bytes() == data


def test_archive_urls_takes_one_url_or_a_list():
    assert envs.archive_urls("a", {"a": "https://h/a.tar.gz"}) == ["https://h/a.tar.gz"]
    assert envs.archive_urls("b", {"b": ["h/b.001", "h/b.002"]}) == ["h/b.001", "h/b.002"]
    assert envs.archive_urls("c", {}) == [f"{envs.PACKED_URL}/c.tar.gz"]


def _tgz(path):
    src = path.parent / "src"
    (src / "bin").mkdir(parents=True)
    (src / "bin" / "python").write_text("#!/bin/sh\n")
    with tarfile.open(path, "w:gz") as t:
        t.add(src / "bin", arcname="bin")
    return path.read_bytes()


def test_install_packed_joins_parts_and_leaves_no_download(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "host_system", lambda: None)
    data = _tgz(tmp_path / "a.tar.gz")
    half = len(data) // 2
    monkeypatch.setattr(envs, "packed_manifest",
                        lambda: {"scmb_r": ["https://h/scmb_r.001", "https://h/scmb_r.002"]})
    _serve(monkeypatch, {"https://h/scmb_r.001": data[:half], "https://h/scmb_r.002": data[half:]})
    root = tmp_path / "envs"
    assert envs.install_packed("scmb_r", envs_dir=root) is True
    assert (root / "scmb_r" / "bin" / "python").exists()
    assert sorted(p.name for p in root.iterdir()) == ["scmb_r"]


def test_the_plan_shows_the_first_part(monkeypatch):
    monkeypatch.setattr(envs, "packed_manifest",
                        lambda: {"matilda": ["h/m.001", "h/m.002"]})
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    row = envs._install_rows(["Matilda"], dry_run=True, flavor="gpu")[0]
    assert row["packed_url"] == "h/m.001"


def test_a_tiny_total_does_not_hang(monkeypatch, tmp_path):
    _serve(monkeypatch, {"https://u/t": b"abcd"})
    assert envs.download_archive(["https://u/t"], tmp_path / "x", label="t", total=4) == 4


def test_a_small_file_reads_in_megabytes(monkeypatch, tmp_path, capsys):
    data = b"x" * 53_000_000
    _serve(monkeypatch, {"https://u/s": data})
    envs.download_archive(["https://u/s"], tmp_path / "x", label="s", total=len(data))
    out = capsys.readouterr().out
    assert out.startswith("[env] downloading s 53 MB ...") and "of 53 MB" in out


# ------------------------------------------- a body shorter than its Content-Length
@pytest.fixture
def http_body():
    """``start(body, cut)``: a local HTTP server for ``body`` that honours Range
    and whose first ``cut`` answers stop halfway, after announcing the whole
    length; returns ``(url, ranges)``, the Range header of each request."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    servers = []

    def start(body, cut):
        ranges = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                rng = self.headers.get("Range")
                ranges.append(rng)
                rest = body[int(rng.split("=")[1].rstrip("-")):] if rng else body
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(rest)))
                self.end_headers()
                self.wfile.write(rest[:len(rest) // 2] if len(ranges) <= cut else rest)

            def log_message(self, *args):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/a.tar.gz", ranges

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_a_body_cut_short_resumes_with_a_range(http_body, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("time.sleep", lambda s: None)
    data = bytes(range(256)) * 400
    url, ranges = http_body(data, cut=1)
    assert envs.download_archive([url], tmp_path / "x", label="p", total=len(data)) == len(data)
    assert (tmp_path / "x").read_bytes() == data
    assert ranges == [None, f"bytes={len(data) // 2}-"]
    assert "connection lost (ConnectionError); resuming" in capsys.readouterr().out


def test_a_body_cut_short_every_time_raises(http_body, monkeypatch, tmp_path):
    monkeypatch.setattr("time.sleep", lambda s: None)
    url, ranges = http_body(b"x" * 1000, cut=99)
    with pytest.raises(ConnectionError):
        envs.download_archive([url], tmp_path / "x", label="p", total=1000, attempts=2)
    assert len(ranges) == 2


def test_a_whole_body_is_read_once(http_body, tmp_path):
    url, ranges = http_body(b"x" * 1000, cut=0)
    assert envs.download_archive([url], tmp_path / "x", label="p", total=1000) == 1000
    assert ranges == [None]
