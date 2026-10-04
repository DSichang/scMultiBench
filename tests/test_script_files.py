"""Files an upstream script reads from its own folder are put in place before a run.

MIRA's script imports a logger.py the public repository lacks, and GLUE reads
a GENCODE annotation it does not ship; both used to stop the method with a
note asking the user to supply the file by hand.
"""
import io
import urllib.request
from pathlib import Path

from multibench.engine import registry, runner


def _variant(method):
    spec = registry.get(method)
    return spec, spec.variants[0]


def test_mira_logger_is_copied_and_a_users_copy_is_kept(tmp_path):
    spec, v = _variant("MIRA")
    folder = tmp_path / Path(v.entrypoint).parent
    folder.mkdir(parents=True)
    runner.provide_script_files(spec, v, tmp_path)
    assert (folder / "logger.py").read_text() == runner.shipped_helper("MIRA", "logger.py").read_text()
    (folder / "logger.py").write_text("# mine\n")
    runner.provide_script_files(spec, v, tmp_path)
    assert (folder / "logger.py").read_text() == "# mine\n"


def test_glue_annotation_is_downloaded_once(tmp_path, monkeypatch):
    spec, v = _variant("GLUE")
    folder = tmp_path / Path(v.entrypoint).parent
    folder.mkdir(parents=True)
    calls = []

    class R(io.BytesIO):
        status = 200
        headers = {"Content-Length": "4"}
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(req, timeout=None):
        calls.append(req.get_method())
        return R(b"" if req.get_method() == "HEAD" else b"gtf!")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    runner.provide_script_files(spec, v, tmp_path)
    name = v.downloads[0]["file"]
    assert (folder / name).read_bytes() == b"gtf!"
    assert not (folder / (name + ".partial")).exists()
    n = len(calls)
    runner.provide_script_files(spec, v, tmp_path)
    assert len(calls) == n


def test_scan_does_not_block_mira_for_its_helper():
    from multibench import workflow
    spec, v = _variant("MIRA")
    assert "logger.py" not in workflow._missing_script(v, method="MIRA")


def test_size_hints_start_with_the_method_and_reach_the_notes():
    """A method that fails on a small dataset says what it needs in scan's caveat."""
    seen = 0
    for m in registry.list_methods():
        spec = registry.get(m)
        for v in spec.variants:
            if v.size_hint:
                seen += 1
                assert v.size_hint.startswith(f"{m} "), v.size_hint
                assert v.size_hint in runner.script_notes(spec, v, Path("/nonexistent"))
    assert seen >= 6
