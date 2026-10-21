from pathlib import Path
import pytest

# Repo root = two levels up from this file (tests/ -> <ROOT>)
ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture
def root():
    return ROOT

@pytest.fixture
def files_dir(root):
    return root / "multibench" / "files"

@pytest.fixture
def result_dir(root):
    return root / "multibench" / "result"


# ---- host-agnostic run mode -------------------------------------------------
# Most tests assert the classic ``conda run -n <env>`` command line. On a host
# where the envs exist as prefixes (the benchmark host, or a laptop that
# unpacked one) the runner would pick prefix mode and those assertions would
# fail for a reason unrelated to what they test. Pin conda mode everywhere
# except in the prefix-mode tests themselves, which set the mode explicitly.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _pin_conda_run_mode(request, monkeypatch):
    if request.node.fspath.basename in ("test_prefix_mode.py", "test_env_flavor.py"):
        return
    monkeypatch.setenv("MULTIBENCH_RUN_MODE", "conda")


# ---- host-agnostic platform -------------------------------------------------
# Method environments are Linux-only. Off Linux, scan / run / run word their
# environment messages for that ("Linux-only environment ..."), and run()
# refuses a missing env even when the env probe finds nothing. Most tests
# assert the Linux behaviour, so pin the platform to Linux everywhere except in
# the tests of the platform check itself. A test that needs another host
# monkeypatches envs.host_platform_problem in its body, which overrides this.
@_pytest.fixture(autouse=True)
def _pin_linux_platform(request, monkeypatch):
    if request.node.fspath.basename in ("test_envs_platform.py",):
        return
    from multibench.engine import envs
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)


@pytest.fixture
def as_gene_activity(monkeypatch):
    """``as_gene_activity(method)`` registers ``method`` as reading gene
    activity for the test (see ``gas_matilda``) and returns its name."""
    from multibench.engine import registry

    def set_form(method):
        monkeypatch.setattr(registry.get(method), "atac", "gene_activity")
        return method
    return set_form


@pytest.fixture
def gas_matilda(monkeypatch):
    """Matilda with its ATAC form set back to gene activity, for the test.

    Matilda, scMDC and UnitedNet are registered as reading peaks, so no
    method outside diagonal reads gene activity any more. The package still
    checks the ATAC form of every method against the file it is given, in
    both directions. The tests of the gene-activity direction in a vertical
    folder keep running on this stand-in; their messages are not what the
    package says about the real Matilda."""
    from multibench.engine import registry
    monkeypatch.setattr(registry.get("Matilda"), "atac", "gene_activity")
    return "Matilda"


# tests/_serve.py (archive download fakes) is imported by several test files
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---- host-agnostic conda envs -----------------------------------------------
# An env multibench did not install is reported by name and path ("Environment
# matilda at ... was not installed by this version of multibench"). A developer's own conda can
# hold such an env (a `matilda` env made by hand), which would change the
# messages the tests assert. Pin "no such env" everywhere except in the tests of
# env identity, which build their own conda root.
@_pytest.fixture(autouse=True)
def _no_stray_envs(request, monkeypatch):
    if request.node.fspath.basename in ("test_env_identity.py",):
        return
    from multibench.engine import envs
    monkeypatch.setattr(envs, "other_env_prefix", lambda env, conda=None: None)


# ---- a run that reaches its subprocess --------------------------------------
@_pytest.fixture
def linux_with_envs(monkeypatch):
    """``runner.run`` as on Linux with every environment installed (conda mode),
    for tests that fake the method's subprocess."""
    from multibench.engine import envs, registry, runner
    every = {envs.group_for(m) for m in registry.list_methods()}
    monkeypatch.setattr(runner, "linux_only_sentence", lambda: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: every)
    monkeypatch.setattr(runner, "run_mode", lambda env: ("conda", None))
    return every


# ---- the full plan behind scan ----------------------------------------------
def full_scan(*args, **kwargs):
    """The plan behind ``scan``, one row per method variant with every check
    column (``workflow._scan``); ``mtb.scan`` returns one row per method."""
    from multibench import workflow
    return workflow._scan(*args, **kwargs)[0]


# ---- no test downloads an environment archive -----------------------------------
# env install acts by default since 0.4.0: a test that forgets dry_run=True /
# --dry-run must fail here, not fetch gigabytes from the envs releases. A test
# that serves its own archives patches urlopen itself, over this guard.
@_pytest.fixture(autouse=True)
def _no_archive_download(monkeypatch):
    import urllib.request
    real = urllib.request.urlopen

    def guarded(req, *a, **kw):
        url = str(getattr(req, "full_url", req))
        if "/releases/download/envs-" in url:
            raise RuntimeError(f"a test tried to download {url}")
        return real(req, *a, **kw)
    monkeypatch.setattr(urllib.request, "urlopen", guarded)


# ---- once-a-session notes start fresh in every test -------------------------------
@_pytest.fixture(autouse=True)
def _fresh_session_notes(monkeypatch):
    from multibench.engine import resolve
    monkeypatch.setattr(resolve, "_ORDER_NOTED", set())


# ---- the settings a test changes are put back --------------------------------
# A dataset folder outside the data root is given by its path, or by setting
# mtb.config.data_path in the test; the settings are restored after it.
@_pytest.fixture(autouse=True)
def _restore_config():
    from multibench import config
    saved = {slot: object.__getattribute__(config, slot) for slot in config.__slots__}
    yield
    for slot, value in saved.items():
        object.__setattr__(config, slot, value)
