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
# monkeypatches envs.host_system in its body, which overrides this.
@_pytest.fixture(autouse=True)
def _pin_linux_platform(request, monkeypatch):
    if request.node.fspath.basename in ("test_envs_platform.py",):
        return
    from multibench.engine import envs
    monkeypatch.setattr(envs, "host_system", lambda: None)


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


# The envs folders earlier installs used live in the user's cache, and the
# fallback folder may hold real envs: keep both out of every test but the
# tests of that record, and keep installs from writing it.
@_pytest.fixture(autouse=True)
def _no_recorded_env_roots(request, tmp_path_factory, monkeypatch):
    from multibench.engine import envs
    monkeypatch.setattr(envs, "ROOTS_FILE",
                        tmp_path_factory.mktemp("roots") / "envs_dirs.txt")
    if request.node.fspath.basename != "test_env_roots.py":
        monkeypatch.setattr(envs, "_known_roots", lambda: [])


# ---- a run that reaches its subprocess --------------------------------------
@_pytest.fixture
def linux_with_envs(monkeypatch):
    """``runner.run`` as on Linux with every environment installed (conda mode),
    for tests that fake the method's subprocess."""
    from multibench.engine import envs, registry, runner
    every = {envs.group_for(m) for m in registry.list_methods()}
    monkeypatch.setattr("multibench.engine.envs.host_system", lambda: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: every)
    monkeypatch.setattr(runner, "run_mode", lambda env: ("conda", None))
    return every


# ---- the full plan behind scan ----------------------------------------------
def full_scan(*args, **kwargs):
    """The plan behind ``scan``, one row per method variant with every check
    column (``workflow._scan``); ``mtb.scan`` returns one row per method."""
    from multibench import workflow
    kwargs.pop("verbose", None)          # the plan prints nothing
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
    monkeypatch.setattr(resolve, "_NOTED", set())


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


# ---- the method scripts, as a wheel install finds them ------------------------
# A wheel carries no tools_scripts/: the first run downloads them into
# repo_path. Here the package root holds none, repo_path is this checkout (the
# scripts "already downloaded"), and no test reaches the network for them. A
# test of a machine without the scripts points repo_path at its own folder; a
# test of the download serves this checkout's archive with ``scripts_download``.
@_pytest.fixture(autouse=True)
def _scripts_as_installed(monkeypatch):
    from multibench import _settings, config
    from multibench.engine import scripts

    def unreachable(url):
        raise OSError("tests do not reach the network")
    monkeypatch.setattr(scripts, "_archive_bytes", unreachable)
    monkeypatch.setattr(_settings, "_ROOT", ROOT / "no-scripts-here")
    monkeypatch.setattr(config, "_repo_path", ROOT)


@_pytest.fixture(scope="session")
def _packed_scripts():
    """``(archive bytes, record)`` of this checkout's ``tools_scripts/``."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pack_scripts",
                                                  ROOT / "tools" / "pack_scripts.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    data = tool.pack()
    return data, tool.record(data)


@_pytest.fixture
def scripts_download(monkeypatch, _packed_scripts):
    """Serve the archive of this checkout's ``tools_scripts/`` as the download;
    returns the ``tools_scripts-<id>`` name a run record gives it."""
    from multibench.engine import scripts
    data, rec = _packed_scripts
    monkeypatch.setattr(scripts, "scripts_archive", lambda: rec)
    monkeypatch.setattr(scripts, "_archive_bytes", lambda url: data)
    return f"tools_scripts-{rec['sha256'][:12]}"
