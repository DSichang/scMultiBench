"""The deprecated fetch_outputs uses an output tree in place as it is: it never moves,
replaces or deletes one (core040 DEFECTS)."""
import importlib
import json

import pytest

F = importlib.import_module("multibench.data.fetch")


def test_a_tree_from_another_url_is_used_as_it_is(tmp_path, monkeypatch):
    tree = tmp_path / "data" / "outputs" / "D11"
    tree.mkdir(parents=True)
    (tree / "batch_result.json").write_text(json.dumps(
        {"dataset": "D11", "category": "vertical", "records": []}))
    (tree / ".source").write_text("https://example.org/an-older-archive.tar.gz\n")
    (tree / "mine.txt").write_text("kept")

    def no_download(url):
        raise AssertionError(f"downloaded {url}")
    monkeypatch.setattr(F, "_download", no_download)
    assert F.fetch_outputs("D11", data_path=tmp_path / "data") == tree
    assert (tree / "mine.txt").read_text() == "kept"
    assert sorted(p.name for p in tree.parent.iterdir()) == ["D11"]      # no D11.old


def test_a_folder_without_a_tree_is_left_alone(tmp_path, monkeypatch):
    folder = tmp_path / "data" / "outputs" / "D11"
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("mine")
    monkeypatch.setattr(F, "_download", lambda url: pytest.fail("downloaded"))
    with pytest.raises(RuntimeError, match="fetch_outputs leaves it alone"):
        F.fetch_outputs("D11", data_path=tmp_path / "data")
    assert (folder / "notes.txt").read_text() == "mine"
