"""Round 7 of the student study, get started (R7-13).

A Troubleshooting heading on the Installation page quotes the first words of
an error, so that a reader who searches the page for the message finds the
entry. R7-10 reworded the run refusals; the tests raise each quoted error with
this package and check that its heading still matches the message.
"""
import re

import pytest

import multibench as mtb
from multibench.engine import envs
from multibench.engine import runner as R
from tests.test_docs_r4 import _read, needs_docs

DARWIN = ("method environments are linux-64 conda envs (packed archives + lockfiles); "
          "this host is darwin/arm64")


def _heading(needle):
    """The quoted error of the Troubleshooting heading that contains ``needle``."""
    text = _read("installation.md").split("## Troubleshooting", 1)[1]
    quoted = re.findall(r'^\?\?\? note "`([^`]+)`', text, flags=re.M)
    return next(q for q in quoted if needle in q)


def _matches(heading, exc):
    """``<name>`` in the heading stands for one word, a final ``...`` for the rest."""
    name, _, words = heading.partition(": ")
    rest = words.endswith(" ...")
    words = words[:-4] if rest else words
    pattern = r"\S+".join(re.escape(p) for p in re.split(r"<[a-z_]+>", words))
    return type(exc).__name__ == name and re.match(pattern, str(exc)) is not None


def _run_totalvi(tmp_path):
    inp = mtb.inputs_for("D11", "vertical", "totalVI")
    with pytest.raises(OSError) as e:
        mtb.run("totalVI", "vertical", inputs=inp, out_dir=str(tmp_path / "out"))
    return e.value


@needs_docs
def test_missing_env_heading_quotes_the_run_error(tmp_path, monkeypatch):
    heading = _heading("is not installed")
    monkeypatch.setattr(R.envs, "installed_envs", lambda conda=None: ["base"])
    err = _run_totalvi(tmp_path)
    assert _matches(heading, err), (heading, str(err))


@needs_docs
def test_linux_only_heading_quotes_the_run_refusal(tmp_path, monkeypatch):
    heading = _heading("Linux")
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)
    monkeypatch.setattr(R.envs, "installed_envs", lambda conda=None: ["base"])
    err = _run_totalvi(tmp_path)
    assert _matches(heading, err), (heading, str(err))
