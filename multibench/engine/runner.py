"""Run a method variant: build its command, enter its env, exec in a workdir, load output."""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from .. import _text

from .._settings import config
from . import builder, io, ingest, registry, envs, scripts
from .schema import _NON_MODALITY_ROLES, is_label_role


#: The environment variable that forces a run mode (``conda`` | ``prefix``).
RUN_MODE_VAR = "MULTIBENCH_RUN_MODE"
RUN_MODES = ("conda", "prefix")


def run_mode(env: str) -> tuple[str, Path | None]:
    """How the runner enters ``env``: ``("prefix", <prefix>)`` when
    ``envs.env_prefix(env)`` finds it on disk, else ``("conda", None)``.

    ``MULTIBENCH_RUN_MODE`` forces either; ``prefix`` with no prefix on disk
    raises ``OSError``, any other value ``ValueError``.
    """
    forced = os.environ.get(RUN_MODE_VAR, "").strip().lower() or None
    if forced is not None and forced not in RUN_MODES:
        raise ValueError(
            f"{RUN_MODE_VAR}={os.environ.get(RUN_MODE_VAR)!r}: expected one of "
            f"{RUN_MODES}")
    prefix = envs.env_prefix(env)
    if forced == "prefix" and prefix is None:
        raise OSError(f"{RUN_MODE_VAR}=prefix, but {env} is not in {config.envs_dir}. "
                      f"Run mtb.env.install([<method>]).")
    if forced == "conda":
        return "conda", None
    return ("prefix", prefix) if prefix is not None else ("conda", None)


def prefix_activation(env: str, prefix: Path | str) -> str:
    """The bash snippet that activates a conda prefix without conda.

    Sets ``CONDA_PREFIX`` and ``CONDA_DEFAULT_ENV`` first (the
    ``activate.d`` scripts read them), puts ``<prefix>/bin`` first on
    ``PATH``, then sources ``<prefix>/etc/conda/activate.d/*.sh``.
    """
    quote = shlex.quote
    root = str(prefix)
    return (f"export CONDA_PREFIX={quote(root)}; export CONDA_DEFAULT_ENV={quote(env)}; "
            f"export PATH={quote(root + '/bin')}:\"$PATH\"; "
            f"for _f in {quote(root + '/etc/conda/activate.d')}/*.sh; do "
            f"[ -e \"$_f\" ] && . \"$_f\"; done; unset _f")


def wrap_prefix(cmd: list[str], env: str, prefix: Path | str) -> list[str]:
    """``bash -c '<activate>; exec "$@"' -- <cmd>``: the prefix-mode argv."""
    return ["bash", "-c", f"{prefix_activation(env, prefix)}; exec \"$@\"", "--", *cmd]


def windows_prefix_argv(cmd: list[str], prefix: Path | str) -> list[str]:
    """The prefix-mode argv on Windows: ``cmd`` with its interpreter
    (``python``, ``Rscript``) replaced by the one in the env at ``prefix``.
    There is no shell wrapper: :func:`windows_activation` is the activation."""
    exe = envs.windows_interpreter(prefix, cmd[0])
    return [str(exe), *cmd[1:]] if exe is not None else list(cmd)


def windows_activation(env: str, prefix: Path | str) -> dict:
    """The variables that activate the Windows env at ``prefix``, as ``conda
    activate`` sets them: ``CONDA_PREFIX``, ``CONDA_DEFAULT_ENV`` and the env's
    folders first on ``PATH``. ``activate.d`` scripts are not run."""
    root = Path(prefix)
    folders = [root, root / "Library" / "mingw-w64" / "bin", root / "Library" / "usr" / "bin",
               root / "Library" / "bin", root / "Scripts", root / "bin"]
    out = {"CONDA_PREFIX": str(root), "CONDA_DEFAULT_ENV": env,
           "PATH": os.pathsep.join([*map(str, folders), os.environ.get("PATH", "")])}
    # an R method that calls Python through reticulate (MOFA2) must get the env's
    # own Python: on Linux the env's activate.d script names it, here this does
    python = envs.windows_interpreter(root, "python")
    if python is not None:
        out["RETICULATE_PYTHON"] = str(python)
    # a Python method that calls R through rpy2 (Portal) must get the env's own
    # R, not one that is registered on the computer
    if (root / "Lib" / "R" / "bin").is_dir():
        out["R_HOME"] = str(root / "Lib" / "R")
    return out


#: The ``out_dir`` of scan's ``command`` column.
OUT_DIR_PLACEHOLDER = "<out_dir>"


def normalize_paths(inputs: dict, out_dir) -> tuple[dict, str]:
    """``(inputs, out_dir)`` with every path absolute; ``out_dir``, ``data_dir``
    and any existing directory get a trailing separator.

    The method runs in another folder, and many scripts concatenate a folder
    and a file name (``data_path + "rna.h5"``). ``os.path.abspath`` keeps
    symlinked data roots as written. Values that are not paths pass through.
    """
    def absolute(path):
        # a preview's placeholder stays as written
        path = os.fspath(path)
        return path if path.startswith(OUT_DIR_PLACEHOLDER) else os.path.abspath(path)
    out = os.path.join(absolute(out_dir), "")
    vals: dict = {}
    for role, v in inputs.items():
        if isinstance(v, (str, os.PathLike)):
            p = absolute(v)
            if role == "data_dir" or os.path.isdir(p):
                p = os.path.join(p, "")
            vals[role] = p
        else:
            vals[role] = v
    return vals, out


def _is_modality_role(role: str) -> bool:
    """A role that selects the variant and must be a canonical ``.h5``."""
    return role not in _NON_MODALITY_ROLES and not is_label_role(role)


def _modality_roles(inputs: dict) -> set[str]:
    return {k for k in inputs if _is_modality_role(k)}


def _repo_root_no_fetch() -> Path:
    """Where :func:`run` finds ``tools_scripts/``, in ``scripts.ensure_repo``'s
    order, without fetching; else the configured path, where a real run puts them."""
    return scripts._scripts_checkout() or Path(config.repo_path)



#: What to do about a method script missing from the scripts folder.
MISSING_SCRIPT_FIX = "delete or move that folder so the next run fetches the scripts"


#: Files the package ships for scripts that import them from their own folder.
HELPERS_DIR = Path(__file__).resolve().parent / "helpers"


def shipped_helper(method: str, name: str) -> Path | None:
    """The package's copy of helper ``name`` for ``method``, or ``None``."""
    p = HELPERS_DIR / method / name
    return p if p.is_file() else None


def provide_script_files(spec, variant, repo: Path) -> None:
    """Copy the shipped helpers (MIRA's ``logger.py``) and download the files
    (GLUE's annotation) a script reads from its own folder, when missing.
    A file already there is left alone."""
    import shutil

    folder = repo / Path(variant.entrypoint).parent
    for name in variant.helpers:
        src = shipped_helper(spec.id, name)
        if src is not None and not (folder / name).exists():
            shutil.copyfile(src, folder / name)
    for item in variant.downloads:
        dest = folder / item["file"]
        if dest.exists():
            continue
        part = dest.with_name(dest.name + ".partial")
        try:
            envs.download_archive([item["url"]], part, label=f"{item['file']} for {spec.id}")
            part.rename(dest)
        finally:
            part.unlink(missing_ok=True)


def script_notes(spec, variant, repo: Path) -> list[str]:
    """Setup facts a scan shows before the real run fails on them: the
    method's ``setup_hint`` (unless its helper files are all in place) and its
    ``size_hint``. Scripts not fetched yet get no note: ``run`` fetches them
    and says so."""
    notes = []
    helpers = variant.helpers
    folder = (repo / variant.entrypoint).parent
    done = bool(helpers) and all((folder / h).exists() or shipped_helper(spec.id, h)
                                 for h in helpers)
    if spec.setup_hint and not done:
        notes.append(spec.setup_hint)
    if variant.size_hint:
        notes.append(variant.size_hint)
    return notes


def _check_input(val) -> None:
    """Refuse a modality input that is not a canonical ``.h5`` file; a path
    that is not on this host (a preview's) passes."""
    if Path(val).exists():
        ingest._check_canonical(val)


def _plan_inputs(variant, inputs: dict, inputs_dir: Path) -> dict:
    """``{role: {"value", "normpeaks_from", "link_from"}}``: the file each role
    hands the script, for the run and scan. A ``normalize_peaks`` role reads
    ``inputs/<role>_normpeaks.h5``; every other input file of a variant that
    ``writes_beside_inputs`` reads a link ``inputs/<role><suffix>`` to it.
    Nothing is written; the format checks run.
    """
    plan: dict = {}
    for role, val in inputs.items():
        step = {"value": val, "normpeaks_from": None, "link_from": None}
        if _is_modality_role(role):
            _check_input(val)
            step["value"] = str(val)
        plan[role] = step
    for role in variant.normalize_peaks:
        if role in plan:
            plan[role]["normpeaks_from"] = plan[role]["value"]
            plan[role]["value"] = str(inputs_dir / f"{role}_normpeaks.h5")
    if variant.writes_beside_inputs:
        for role, step in plan.items():
            if (role not in _NON_MODALITY_ROLES and step["normpeaks_from"] is None
                    and isinstance(step["value"], str)):
                step["link_from"] = step["value"]
                step["value"] = str(inputs_dir / f"{role}{Path(step['value']).suffix}")
    return plan


def _link_input(src, dest) -> None:
    """Make ``dest`` a link to ``src`` (absolute), replacing a link an earlier
    run left there."""
    if envs.on_windows():
        _link_input_windows(src, dest)
        return
    if os.path.islink(dest):
        os.unlink(dest)
    os.symlink(os.path.abspath(src), dest)


def _link_input_windows(src, dest) -> None:
    """:func:`_link_input` on Windows, where a symbolic link needs a privilege
    an ordinary user does not have: a hard link, else (another drive) a copy."""
    import shutil

    if os.path.lexists(dest):
        os.unlink(dest)
    try:
        os.link(os.path.abspath(src), dest)
    except OSError:
        shutil.copyfile(src, dest)


def _host_run_env(run_env: dict | None) -> dict:
    """The variant's ``run_env`` for this machine: a value made of absolute
    paths (``LD_PRELOAD``) keeps the paths that exist here, and is left out
    when none does."""
    out = {}
    for key, value in (run_env or {}).items():
        value = str(value)
        paths = value.replace(":", " ").split()
        if paths and all(p.startswith("/") for p in paths):
            kept = [p for p in paths if os.path.exists(p)]
            if not kept:
                continue
            if len(kept) < len(paths):
                value = (":" if ":" in value else " ").join(kept)
        out[key] = value
    return out


def _argv(variant, method: str, values: dict, out_str: str, repo: Path,
          params: dict | None) -> list[str]:
    """The exact argv a method run executes (shared by the run and scan).

    The variant's command, with its package-side ``driver`` in place of the
    script when it has one, the pseudo-tty wrap when declared, then the env
    wrap of the mode :func:`run_mode` picks.
    """
    cmd = builder.build_command(variant, values=values, out_dir=out_str, params=params)
    # a driver runs the script from its folder (--script_dir)
    if variant.driver:
        pkg_root = Path(__file__).resolve().parents[1]      # .../multibench
        driver_abs = pkg_root / variant.driver
        script_dir = (repo / variant.entrypoint).parent
        cmd = [cmd[0], str(driver_abs), "--script_dir", str(script_dir)] + cmd[2:]
    else:
        cmd[1] = str(repo / cmd[1])
    env_name = envs.group_for(method)
    mode, prefix = run_mode(env_name)
    # `script` gives the method a pty (scJoint reads the terminal size) and
    # passes its exit code on (-e); it must be innermost, as conda run
    # redirects stdio
    if variant.pty:
        cmd = ["script", "-q", "-e", "-c",
               " ".join(shlex.quote(c) for c in cmd), "/dev/null"]
    if mode == "prefix" and envs.on_windows():
        return windows_prefix_argv(cmd, prefix)
    if mode == "prefix":
        return wrap_prefix(cmd, env_name, prefix)
    # CONDA_EXE: bare `conda` may not be on the subprocess's PATH
    return [os.environ.get("CONDA_EXE", "conda"), "run", "-n", env_name, *cmd]


def unknown_params_error(method: str, unknown: list, allowed: set) -> KeyError:
    """The error for ``params`` keys a variant does not accept (``run`` and
    ``mtb.run`` say the same)."""
    return KeyError(f"{method} does not accept {unknown}; it accepts "
                    f"{sorted(allowed) or 'no parameters'}.")


def check_params(method: str, variant, params: dict | None) -> None:
    """Raise :func:`unknown_params_error` for a ``params`` key the variant does
    not accept, and :func:`check_param_types`' error for a value of the wrong
    type, before anything runs."""
    allowed = set(variant.tunable)
    unknown = [k for k in (params or {}) if k not in allowed]
    if unknown:
        raise unknown_params_error(method, unknown, allowed)
    check_param_types(method, variant, params)


#: the script's declared type -> what a value must be
_TYPE_WORDS = {"int": "a whole number", "float": "a number", "str": "a text value",
               "bool": "True or False", "boolean": "True or False", "flag": "True or False"}


def _fits(value, kind: str) -> bool:
    """Whether ``value`` can stand on the command line for an argument of
    argparse ``type=kind``: a bool never stands for a number or text."""
    import numbers
    if kind in ("bool", "boolean", "flag"):
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if kind == "str":
        return True
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, numbers.Real):
        return kind == "float"           # 3.0 reaches argparse's int() as "3.0"
    if isinstance(value, str):
        try:
            int(value) if kind == "int" else float(value)
            return True
        except ValueError:
            return False
    return False


def check_param_types(method: str, variant, params: dict | None) -> None:
    """``TypeError`` for a ``params`` value that does not fit the type the
    script declares for it (``int``, ``float``, ``str``, ``bool``, ``boolean``, ``flag``):
    ``2.5`` or ``True`` for ``epochs`` would reach the script as ``--epochs
    2.5`` or a bare ``--epochs``, and ``"False"`` for a ``bool`` is true."""
    for key, value in (params or {}).items():
        decl = variant.tunable.get(key) or {}
        kind = decl.get("type")
        if kind in _TYPE_WORDS and value is not None and not _fits(value, kind):
            raise TypeError(f"{method}: {key} takes {_TYPE_WORDS[kind]}, got {value!r}. "
                            f"Its default is {variant.params.get(key, decl.get('default'))!r}.")
        choices = decl.get("choices")
        if choices and value is not None and str(value) not in map(str, choices):
            raise ValueError(f"{method}: {key} takes {_text._and_list(map(str, choices), 'or')}, "
                             f"got {value!r}.")


def cpu_params_for(spec, params: dict | None, *,
                   gpu: bool | None = None) -> tuple[dict | None, dict]:
    """``(params, applied)`` after the CPU switch.

    Without a GPU (``gpu``, or :func:`envs.host_has_gpu` when ``None``) the
    method's ``cpu_params`` that the caller did not set are merged into
    ``params``; ``applied`` is that subset. ``params`` is returned unchanged
    when nothing applies.
    """
    has_gpu = envs.host_has_gpu() if gpu is None else gpu
    if has_gpu and envs.on_windows() and envs.installed_flavor(envs.group_for(spec.id)) == "cpu":
        # the CPU build of an env cannot use the GPU of the host
        has_gpu = False
    if not spec.cpu_params or has_gpu:
        return params, {}
    given = dict(params or {})
    applied = {k: v for k, v in spec.cpu_params.items() if k not in given}
    if not applied:
        return params, {}
    return {**applied, **given}, applied


def check_gpu_requirement(spec) -> None:
    """Raise ``OSError`` (``spec.requires_gpu_reason``) for a method that
    requires a GPU on a host without one."""
    if spec.requires_gpu and not envs.host_has_gpu():
        raise OSError(spec.requires_gpu_reason)


def run(method: str, category: str, *, inputs: dict, out_dir: str,
        params: dict | None = None):
    """Run one method and return its primary output, loaded (internal:
    ``mtb.run`` calls it after ``inputs_for(check=True)`` and
    :func:`check_params`). ``inputs`` is ``{role: path}``; its modality roles
    select the variant. ``OSError`` when the method needs a GPU this host
    lacks, its env is not installed, or this host is not Linux;
    ``RuntimeError`` when the method fails or does not write a declared extra
    output."""
    spec = registry.get(method)
    method = spec.id
    variant = spec.select(category, _modality_roles(inputs))
    params, applied = cpu_params_for(spec, params)
    check_gpu_requirement(spec)
    if applied:
        # on Windows also with a GPU, when the installed env is its CPU build
        why = ("no GPU on this host" if not envs.host_has_gpu()
               else f"{envs.group_for(method)} is the CPU build")
        print(f"[run] {why}: applying {method} cpu_params {applied}",
              file=sys.stderr, flush=True)
    env_name = envs.group_for(method)
    _check_env(method, env_name)

    inputs, out_str = normalize_paths(inputs, out_dir)
    out = Path(out_str)
    inputs_dir = out / "inputs"
    # every input-format check before anything is written or fetched
    plan = _plan_inputs(variant, inputs, inputs_dir)
    repo = scripts.ensure_repo()
    provide_script_files(spec, variant, repo)
    out.mkdir(parents=True, exist_ok=True)
    if _modality_roles(inputs):
        inputs_dir.mkdir(parents=True, exist_ok=True)

    values: dict = {}
    for role, step in plan.items():
        if step["normpeaks_from"] is not None:
            ingest._normalize_peak_names(step["normpeaks_from"], step["value"])
        elif step["link_from"] is not None:
            _link_input(step["link_from"], step["value"])
        values[role] = step["value"]
    cmd = _argv(variant, method, values, out_str, repo, params)
    cwd = (repo / variant.entrypoint).parent if variant.cwd_at_script else out
    if envs.on_windows():
        _remove_old_outputs_windows(out)
        # no shell wrapper on Windows: the activation is the subprocess's environment
        prefix = run_mode(env_name)[1]
        _execute(method, cmd, str(cwd), variant.run_env,
                 activation=prefix and windows_activation(env_name, prefix))
    else:
        _execute(method, cmd, str(cwd), variant.run_env)
    missing = [o.file for o in variant.extra_outputs if not (out / o.file).is_file()]
    if missing:
        raise RuntimeError(f"{method} did not write {_text._and_list(missing)} in {out}.")
    return io.load_output(out, variant.output)


def _check_env(method: str, env_name: str) -> None:
    """The env check of scan, before anything is written: off Linux, or when
    :func:`envs.env_prefix` does not find ``env_name``."""
    linux_only = envs.linux_only_text()
    if linux_only:
        raise OSError(
            f"{linux_only} Run this call on a Linux machine, such as Colab.")
    if envs.on_windows():
        # no env can be installed on Windows: one built there for the method, or none
        if envs.serves(env_name, method):
            return
        raise EnvironmentError(envs.no_windows_env_text([method]))
    if envs.env_prefix(env_name) is not None:
        return
    # conda's env list is cached per process: probe once more
    envs._conda_prefixes.cache_clear()
    if envs.env_prefix(env_name) is not None:
        return
    other = envs.other_env_prefix(env_name)
    if other is not None:
        raise EnvironmentError(envs.other_env_text(
            env_name, other,
            f"call mtb.env.install([{method!r}])."))
    raise EnvironmentError(f"Environment {env_name} of {method} is not installed. Run "
                           f"mtb.env.install([{method!r}]).")


def _execute(method: str, cmd: list[str], cwd: str, run_env: dict | None,
             activation: dict | None = None) -> None:
    """Run the method's ``cmd`` in ``cwd``; a non-zero exit raises
    ``RuntimeError`` with the tails of stdout and stderr. ``activation``:
    the variables of :func:`windows_activation`, on Windows."""
    # user site-packages (~/.local) can shadow the env's packages, and so can
    # the user's personal R library: our env holds every package it needs
    env = {**envs.without_user_r_library(os.environ), **(activation or {}),
           "PYTHONNOUSERSITE": "1", **_host_run_env(run_env)}
    # a Jupyter kernel's MPLBACKEND (matplotlib_inline) does not exist in the env
    if "MPLBACKEND" not in (run_env or {}):
        env["MPLBACKEND"] = "Agg"
    # on Windows a method's output need not be in the system code page (R writes
    # UTF-8): a byte that does not decode must not lose the output or the error
    decoding = {"errors": "replace"} if envs.on_windows() else {}
    # the method may be a grandchild (under conda run): its own session lets
    # one killpg end the whole tree
    popen = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env,
                             start_new_session=True, **decoding)
    try:
        stdout, stderr = popen.communicate()
    except BaseException:            # TimeoutError from the deadline included
        if envs.on_windows():
            _kill_tree_windows(popen)
        else:
            try:
                os.killpg(popen.pid, signal.SIGKILL)
            except OSError:
                popen.kill()
        popen.wait()
        raise
    if popen.returncode != 0:
        # stderr last: callers cut this message from the left
        raise RuntimeError(
            f"{method} failed (exit {popen.returncode}).\n"
            f"stdout tail:\n{stdout[-1500:]}\n"
            f"stderr tail:\n{stderr[-2500:]}"
        )


def _remove_old_outputs_windows(out: Path) -> None:
    """Remove the ``.h5`` files an earlier run left in the method's output
    folder. The R method scripts remove an existing output file with ``rm``
    before they write it; Windows has no ``rm``, and the write then fails
    ("Object with name 'data' already exists"). Input links are in ``inputs/``
    and are not touched."""
    for old in out.glob("*.h5"):
        try:
            old.unlink()
        except OSError:
            pass


def _kill_tree_windows(popen) -> None:
    """End the method's process and its children on Windows, which has no
    process groups to signal: ``taskkill /T``, else the process alone."""
    try:
        done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(popen.pid)],
                              capture_output=True)
    except OSError:
        done = None
    if done is None or done.returncode != 0:
        popen.kill()


def preview(method: str, category: str, *, inputs: dict, out_dir, params=None,
            gpu: bool | None = None) -> tuple[list[str], str | None]:
    """``(argv, prepared)`` for ``mtb.scan``: the argv a run would execute and
    :func:`_prepared_note`; nothing is written, fetched or launched. ``gpu``
    goes to :func:`cpu_params_for`."""
    spec = registry.get(method)
    method = spec.id
    variant = spec.select(category, _modality_roles(inputs))
    check_params(method, variant, params)            # as mtb.run checks them
    params, _ = cpu_params_for(spec, params, gpu=gpu)
    values, out_str = normalize_paths(inputs, out_dir)
    plan = _plan_inputs(variant, values, Path(out_str) / "inputs")
    values = {role: step["value"] for role, step in plan.items()}
    argv = _argv(variant, method, values, out_str, _repo_root_no_fetch(), params)
    return argv, _prepared_note(plan, out_str, spec.id)


#: what :func:`_prepared_note` says after the method name; :func:`_prepared_at`
#: finds it in a scan caveat
_PREPARED_PREFIX = " reads inputs"
_PREPARED_RE = re.compile(r"(?:^|(?<=[.!?] ))[^\s;]+" + re.escape(_PREPARED_PREFIX)
                          + r"[\\/]")


def _prepared_at(text) -> int:
    """Where the :func:`_prepared_note` in ``text`` starts, or ``-1``."""
    m = _PREPARED_RE.search(str(text or ""))
    return m.start() if m else -1


def _prepared_note(plan: dict, out_str: str, method: str) -> str | None:
    """The note for a command that reads files :func:`run` writes first
    (renamed peak files, links to the input files), or ``None``."""
    files = [os.path.relpath(step["value"], out_str) for step in plan.values()
             if step["normpeaks_from"] or step["link_from"]]
    if not files:
        return None
    that = "that file" if len(files) == 1 else "those files"
    return (f"{method} reads {', '.join(files)}. mtb.run writes {that} first, so "
            f"start {method} with mtb.run.")
