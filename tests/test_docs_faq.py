"""The FAQ page (Resources) and the Changes page that is gone.

The owner (2026-11-02): the FAQ says how to reuse the installed environments;
the Changes page is not shown while the package has few users.
"""
import re

import multibench as mtb
from multibench import _settings
from tests.test_docs_r4 import _docs_root, _read, needs_docs


@needs_docs
def test_resources_has_the_faq_and_no_changes_page():
    nav = (_docs_root().parent / "mkdocs.yml").read_text().split("\nnav:\n", 1)[1]
    resources = nav.split("  - Resources:\n", 1)[1].split("\n\n", 1)[0]
    assert "      - FAQ: faq.md" in resources.splitlines()
    assert "changes.md" not in resources
    assert not (_docs_root() / "changes.md").exists()
    for page in _docs_root().rglob("*.md"):
        assert "changes.md" not in page.read_text(), page


@needs_docs
def test_the_faq_names_what_the_package_has():
    faq = _read("faq.md")
    assert f"export {_settings.ENVS_DIR_VAR}=" in faq
    assert "mtb.config.envs_dir" in faq and hasattr(mtb.config, "envs_dir")
    assert 'mtb.env.install(["scJoint"], dry_run=True)' in faq
    assert mtb.method_info("scJoint")["env"] == "scmb_scjoint"
    assert re.search(r"conda activate /\S+/scmb_scjoint", faq)
    assert "(installation.md#troubleshooting)" in faq
