"""Every environment a method needs must be buildable from a committed lockfile.

`multibench env install --run` is the whole story for someone setting this up on
a new machine. An env with no lockfile is silently unbuildable there: env doctor
shows it as "[!] missing, no lockfile" and the reviewer has no route forward.

scmb_gpsa2 (GPSA) was in exactly that state - 28 of 29 covered - and nothing in
the suite would have caught it.
"""

from multibench.engine import envs


def test_every_required_env_has_a_lockfile():
    required = set(envs.required_envs())
    assert required, "no envs reported as required - the registry looks empty"
    missing = sorted(e for e in required if envs.lockfile(e) is None)
    assert not missing, (
        f"{len(missing)} of {len(required)} required envs have no committed "
        f"lockfile, so `multibench env install --run` cannot build them: {missing}. "
        f"Capture each with `multibench env freeze <env>`.")


def test_lockfiles_are_substantive():
    """A truncated or empty lockfile fails at install time, not at freeze time."""
    thin = []
    for env in sorted(set(envs.required_envs())):
        lock = envs.lockfile(env)
        if lock is None:
            continue
        text = lock.read_text(encoding="utf-8")
        if "dependencies:" not in text or len(text.splitlines()) < 10:
            thin.append((env, len(text.splitlines())))
    assert not thin, f"lockfiles too thin to be a real env export: {thin}"
