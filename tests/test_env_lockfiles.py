"""Every environment a method needs must be buildable from a committed lockfile.

``mtb.env.install(build=True)`` builds from them on a new machine. An env with
no lockfile is silently unbuildable there.

One env was in exactly that state - it had no lockfile - and nothing in the
suite would have caught it.
"""

from multibench.engine import envs, registry


def _required():
    return {envs.group_for(m) for m in registry.list_methods()}


def test_every_required_env_has_a_lockfile():
    required = _required()
    assert required, "no envs reported as required - the registry looks empty"
    missing = sorted(e for e in required if envs.lockfile(e) is None)
    assert not missing, (
        f"{len(missing)} of {len(required)} required envs have no committed "
        f"lockfile, so mtb.env.install(build=True) cannot build them: {missing}. "
        f"Capture each with envs.freeze(<env>).")


def test_lockfiles_are_substantive():
    """A truncated or empty lockfile fails at install time, not at freeze time."""
    thin = []
    for env in sorted(_required()):
        lock = envs.lockfile(env)
        if lock is None:
            continue
        text = lock.read_text(encoding="utf-8")
        if "dependencies:" not in text or len(text.splitlines()) < 10:
            thin.append((env, len(text.splitlines())))
    assert not thin, f"lockfiles too thin to be a real env export: {thin}"
