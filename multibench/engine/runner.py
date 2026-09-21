"""Run a method variant: build cmd, wrap via cmd_template, exec in a workdir, load output."""
from __future__ import annotations

import glob
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import config
from . import builder, io, ingest, registry, envs


@dataclass
class RunResult:
    """What :func:`run` returns for a single method.

    Attributes
    ----------
    method : the method id that was run.
    out_dir : directory holding everything the method wrote. The primary file is
        named by the variant's ``output.file`` (usually ``embedding.h5``).
    cmd : the exact argv that was executed - useful for reproducing a run by hand.
    output : the primary output, already loaded. For an embedding method this is a
        numpy array; pass it straight to :func:`multibench.evaluate`. Note it may be
        dims x cells rather than cells x dims - ``evaluate`` re-orients a raw array
        against the label count for you.
    extra : ``{filename: loaded object}`` for any ``extra_outputs`` the variant
        declares (e.g. scMoMaT's UMAP embedding alongside its KNN graph); for
        a registration method the ``slices_manifest.json`` dict as well
        (:func:`stage_slices`).
    stdout, stderr : captured output of the method process. ``stderr`` is where a
        method's own diagnostics go and is the first place to look when a run
        produced a file but the numbers look wrong.

    Not every method returns an embedding: check
    ``registry.get(method).select(...).output.kind`` first (``embedding`` /
    ``graph`` / ``coords``). :func:`multibench.run_all` does this for you, which is
    why it is the recommended entry point.
    """

    method: str
    out_dir: Path
    cmd: list[str]
    output: object                  # primary loaded output (e.g. embedding ndarray)
    extra: dict                     # role/file -> loaded extra outputs
    stdout: str = ""
    stderr: str = ""


def wrap_command(cmd: list[str], cmd_template: str | None) -> list[str]:
    """Wrap argv with a cmd_template like 'conda run -n env {cmd}'."""
    if not cmd_template or cmd_template.strip() == "{cmd}":
        return cmd
    prefix = cmd_template.replace("{cmd}", "").strip()
    return shlex.split(prefix) + cmd


#: The environment variable that forces a run mode (``conda`` | ``prefix``).
RUN_MODE_VAR = "MULTIBENCH_RUN_MODE"
RUN_MODES = ("conda", "prefix")


def run_mode(env: str) -> tuple[str, Path | None]:
    """Which way the runner enters ``env`` for this call, and its prefix.

    Parameters
    ----------
    env : str
        The real env name (``mtb.env.group_for(method)``).

    Returns
    -------
    tuple[str, Path or None]
        ``("prefix", <prefix>)`` whenever ``mtb.env.env_prefix(env)`` finds
        the env on disk - the default, needing no conda binary; else
        ``("conda", None)`` (``conda run -n <env>``). The
        ``MULTIBENCH_RUN_MODE`` environment variable forces either.

    Raises
    ------
    OSError
        ``MULTIBENCH_RUN_MODE=prefix`` with no prefix on disk; the message
        names ``envs_dir`` and ``mtb.env.install``.
    ValueError
        ``MULTIBENCH_RUN_MODE`` set to anything but ``conda`` / ``prefix``.
    """
    forced = os.environ.get(RUN_MODE_VAR, "").strip().lower() or None
    if forced is not None and forced not in RUN_MODES:
        raise ValueError(
            f"{RUN_MODE_VAR}={os.environ.get(RUN_MODE_VAR)!r}: expected one of "
            f"{RUN_MODES}")
    prefix = envs.env_prefix(env)
    if forced == "prefix" and prefix is None:
        raise OSError(
            f"{RUN_MODE_VAR}=prefix but env {env!r} has no prefix under "
            f"envs_dir {config.DEFAULT.envs_dir} (and conda reports none) - "
            f"run mtb.env.install([<method>], dry_run=False) to unpack it, or "
            f"point MULTIBENCH_ENVS_DIR / mtb.config.DEFAULT.envs_dir at it")
    if forced == "conda":
        return "conda", None
    return ("prefix", prefix) if prefix is not None else ("conda", None)


def prefix_activation(env: str, prefix: Path | str) -> str:
    """The shell snippet that activates a conda prefix without conda.

    Sets ``CONDA_PREFIX`` and ``CONDA_DEFAULT_ENV``, puts ``<prefix>/bin``
    first on ``PATH`` and sources every ``<prefix>/etc/conda/activate.d/*.sh``
    (the scripts assume ``CONDA_PREFIX`` is already set, hence the order).
    ``python`` / ``Rscript`` in the wrapped argv then resolve inside the env.

    Parameters
    ----------
    env : str
        The env name (``CONDA_DEFAULT_ENV``).
    prefix : path
        The env prefix (``CONDA_PREFIX``).

    Returns
    -------
    str
        One ``;``-separated bash snippet, every path quoted.
    """
    q = shlex.quote
    p = str(prefix)
    return (f"export CONDA_PREFIX={q(p)}; export CONDA_DEFAULT_ENV={q(env)}; "
            f"export PATH={q(p + '/bin')}:\"$PATH\"; "
            f"for _f in {q(p + '/etc/conda/activate.d')}/*.sh; do "
            f"[ -e \"$_f\" ] && . \"$_f\"; done; unset _f")


def wrap_prefix(cmd: list[str], env: str, prefix: Path | str) -> list[str]:
    """``bash -c '<activate>; exec "$@"' -- <cmd>`` - the prefix-mode argv.

    Parameters
    ----------
    cmd : list of str
        The method argv (``python script.py ...`` / ``Rscript ...``).
    env, prefix
        As for :func:`prefix_activation`.

    Returns
    -------
    list of str
        The wrapped argv; ``bash`` execs the method, so the method is the
        child process (the process-group kill on a timeout is unchanged).
    """
    return ["bash", "-c", f"{prefix_activation(env, prefix)}; exec \"$@\"", "--", *cmd]


# Auxiliary roles are passed through verbatim (never converted to canonical .h5)
# and are excluded from the modality set used for variant selection.
_AUX_ROLES = {"data_dir", "source_data", "target_data", "cty", "source_cty", "target_cty",
              "out_dir"}


#: name of the per-run file mapping ``aligned_slice_<i>`` back to its source slice
SLICES_MANIFEST = "slices_manifest.json"


def stage_slices(data_dir, staged_dir) -> dict:
    """Stage a directory of ``*.h5ad`` slices as sorted, zero-padded symlinks.

    The registration scripts (PASTE, PASTE2, GPSA's driver, SPIRAL) load
    ``glob.glob(data_dir + "*.h5ad")`` and write ``aligned_slice_<i>.h5ad``
    for the ``i``-th file the glob returned - filesystem order, which the
    scripts never sort - and PASTE strips every ``obs`` column at load, so
    nothing in an output slice says which input it came from. Staging gives
    the scripts a directory whose names sort as ``00_<name>.h5ad,
    01_<name>.h5ad, ...`` (the source names in ``sorted()`` order) and records
    the order the glob actually returns there - the order the script loads -
    so the manifest holds even on a filesystem whose listing is not sorted.
    The ``NN_`` prefix is unique per slice, which also satisfies SPIRAL's
    unique-leading-token rule.

    Parameters
    ----------
    data_dir : path
        The user's slice directory (only ``*.h5ad`` regular files are staged).
    staged_dir : path
        Where the symlinks go (created; stale ``*.h5ad`` symlinks from an
        earlier run in the same directory are removed first).

    Returns
    -------
    dict
        The manifest: ``data_dir``, ``staged_dir`` (both with a trailing
        separator), ``n_slices``, ``order`` (how the list was ordered) and
        ``slices`` - one ``{index, output, staged, source}`` per slice in
        load order, ``output`` being ``aligned_slice_<index>.h5ad``.
    """
    src = Path(os.path.abspath(os.fspath(data_dir)))
    dst = Path(os.path.abspath(os.fspath(staged_dir)))
    files = sorted(p for p in src.glob("*.h5ad") if p.is_file())
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.h5ad"):
        if old.is_symlink():
            old.unlink()
    width = max(2, len(str(max(len(files) - 1, 0))))
    source_of: dict[str, str] = {}
    for i, f in enumerate(files):
        link = dst / f"{i:0{width}d}_{f.name}"
        os.symlink(str(f), str(link))
        source_of[link.name] = str(f)
    # the same glob call the upstream scripts make, in the same directory: the
    # result is in os.scandir order, which depends on the filesystem
    seen = [Path(p).name for p in glob.glob(os.path.join(str(dst), "") + "*.h5ad")]
    slices = [{"index": i, "output": f"aligned_slice_{i}.h5ad", "staged": name,
               "source": source_of[name]} for i, name in enumerate(seen)]
    return {
        "data_dir": os.path.join(str(src), ""),
        "staged_dir": os.path.join(str(dst), ""),
        "n_slices": len(slices),
        "order": ("the order glob.glob(data_dir + '*.h5ad') returned in staged_dir - "
                  "the order the script loads the slices, so aligned_slice_<index>.h5ad "
                  "is the registration of 'source'"
                  + ("" if seen == sorted(seen) else
                     " (NOTE: this filesystem does not list the staged names in sorted "
                     "order; trust this list, not the NN_ prefixes)")),
        "slices": slices,
    }


def normalize_paths(inputs: dict, out_dir) -> tuple[dict, str]:
    """Absolutize every path-valued input and ``out_dir``; directory roles get
    a trailing separator.

    The method runs with ``cwd=out_dir`` (or the script's own directory), so a
    relative ``data/MYCITE/rna.h5`` would be looked up in the wrong place and
    a relative ``--save_path out/x/`` would write ``out/x/out/x/embedding.h5``.
    Many upstream scripts also string-concatenate ``data_dir + "*.h5ad"``,
    hence the separator on directory values. ``os.path.abspath`` (not
    ``Path.resolve``) keeps symlinked data roots as the user wrote them.

    Parameters
    ----------
    inputs : ``{role: path-or-object}``. Strings / ``os.PathLike`` values are
        absolutized; ``data_dir`` and any value that is an existing directory
        get a trailing ``os.sep``; anything else (in-memory AnnData / MuData)
        is passed through untouched for ``to_canonical`` to convert.
    out_dir : the output directory (str or path-like).

    Returns
    -------
    tuple[dict, str]
        ``(inputs_with_absolute_paths, absolute_out_dir_with_trailing_sep)``.
    """
    out = os.path.join(os.path.abspath(os.fspath(out_dir)), "")
    vals: dict = {}
    for role, v in inputs.items():
        if isinstance(v, (str, os.PathLike)):
            p = os.path.abspath(os.fspath(v))
            if role == "data_dir" or os.path.isdir(p):
                p = os.path.join(p, "")
            vals[role] = p
        else:
            vals[role] = v
    return vals, out


def _modality_roles(inputs: dict) -> set[str]:
    """The roles of ``inputs`` that select a variant: everything but the
    auxiliary roles and the label roles (anything containing ``cty`` /
    ``label``, e.g. ``rna_cty`` / ``atac_cty``)."""
    return {k for k in inputs if k not in _AUX_ROLES and "cty" not in k and "label" not in k}


def _repo_root_no_fetch() -> Path:
    """Where :func:`run` looks for ``tools_scripts/`` - without cloning it.

    Mirrors ``config.ensure_repo``'s lookup order (configured ``repo_path``,
    then the package root) but never fetches: a dry run must not touch the
    network. When neither holds a checkout the configured path is returned,
    which is where a real run will put the scripts on first use.
    """
    p = Path(config.DEFAULT.repo_path)
    if (p / "tools_scripts").is_dir():
        return p
    root = Path(config.__file__).resolve().parent.parent
    if (root / "tools_scripts").is_dir():
        return root
    return p


def _argv(variant, method: str, values: dict, out_str: str, repo: Path,
          params: dict | None, cmd_template: str | None) -> list[str]:
    """The exact argv a method run executes (shared by the run and the dry run).

    Builds the command from the variant's argument spec, swaps in the
    package-side ``driver`` wrapper when the variant declares one, applies
    the opt-in pseudo-tty wrap and finally the env wrap: the ``cmd_template``
    when given, else the mode :func:`run_mode` picks for the env
    :func:`multibench.env.group_for` would provision - the ``bash -c``
    prefix activation (:func:`wrap_prefix`) when the prefix is on disk,
    ``conda run -n <env>`` otherwise.
    """
    # Pass out_dir with a trailing separator: many method scripts build their
    # output path by string-concatenation (R paste0(save_path,"embedding.h5"),
    # etc.), so a missing slash writes a sibling file instead of into out_dir.
    cmd = builder.build_command(variant, values=values, out_dir=out_str, params=params)
    # entrypoint is relative to the reference repo. A variant may declare a
    # package-side `driver` that source()s/imports the unmodified upstream
    # entrypoint and calls its function: run the driver instead and pass the
    # upstream script's directory via --script_dir, so it is sourced in place.
    if getattr(variant, "driver", None):
        pkg_root = Path(__file__).resolve().parents[1]      # .../multibench
        driver_abs = pkg_root / variant.driver
        script_dir = (repo / variant.entrypoint).parent
        cmd = [cmd[0], str(driver_abs), "--script_dir", str(script_dir)] + cmd[2:]
    else:
        cmd[1] = str(repo / cmd[1])
    activate = None
    if cmd_template is None:
        # Resolve the env through the group system provisioning uses
        # (mtb.env.plan/create/create_group), so the env that is provisioned
        # is the env that runs: group_for() returns a method's shared group
        # env, or its own scmb_<method> env if it is not grouped.
        env_name = envs.group_for(method)
        mode, prefix = run_mode(env_name)
        if mode == "prefix":
            # the prefix is on disk: activate it directly, no conda needed
            # (the bash wrapper is applied last, outside the pty wrap)
            activate = (env_name, prefix)
            cmd_template = "{cmd}"
        else:
            # Resolve conda's full path when available (CONDA_EXE is set by an
            # initialized conda) so the default works even when bare `conda` is
            # not on the spawned subprocess's PATH; fall back to `conda`.
            conda = os.environ.get("CONDA_EXE", "conda")
            cmd_template = f"{conda} run -n {env_name} {{cmd}}"
    # Opt-in pseudo-tty: some upstream scripts read the terminal size
    # (os.popen('stty size')) to draw a progress bar and crash without a tty
    # (scJoint's util/utils.py). `script` allocates a pty, forwards the child's
    # output to captured stdout and (-e) propagates its exit code, so the
    # returncode check in run() still fires. It must be innermost - inside the
    # conda-run wrap, hence before wrap_command - so the method's own stdin is
    # the pty; conda run redirects stdio otherwise.
    if getattr(variant, "pty", False):
        cmd = ["script", "-q", "-e", "-c",
               " ".join(shlex.quote(c) for c in cmd), "/dev/null"]
    if activate is not None:
        return wrap_prefix(cmd, *activate)
    return wrap_command(cmd, cmd_template)


def cpu_params_for(spec, params: dict | None) -> tuple[dict | None, dict]:
    """The params a run on this host should emit, after the CPU switch.

    On a host without an NVIDIA GPU (:func:`multibench.engine.envs.host_has_gpu`
    is False) a method whose upstream script has CUDA on by default is
    given its registry ``cpu_params`` - the command-line values that turn
    CUDA off (scJoint's ``--use_cuda ""``, scMDC's ``--device cpu``) - so
    the script does not die with "Torch not compiled with CUDA enabled".
    A key the caller passed explicitly is never overridden: ``params``
    wins, whatever the host.

    Parameters
    ----------
    spec : MethodSpec
        The method (``spec.cpu_params`` is the registry declaration).
    params : dict or None
        The caller's overrides.

    Returns
    -------
    tuple[dict or None, dict]
        ``(merged, applied)`` - ``merged`` is ``params`` with the applied
        keys added (``params`` itself, untouched, when nothing applies);
        ``applied`` is exactly the subset of ``cpu_params`` that was merged
        (empty on a GPU host, for a method without ``cpu_params``, or when
        the caller set every key).
    """
    if not spec.cpu_params or envs.host_has_gpu():
        return params, {}
    given = dict(params or {})
    applied = {k: v for k, v in spec.cpu_params.items() if k not in given}
    if not applied:
        return params, {}
    return {**applied, **given}, applied


def check_gpu_requirement(spec) -> None:
    """Refuse, before anything is launched, a method that cannot run here.

    Parameters
    ----------
    spec : MethodSpec
        The method about to run.

    Raises
    ------
    OSError
        ``spec.requires_gpu`` is True (the upstream script calls CUDA
        unconditionally - no flag, no ``torch.cuda.is_available()``
        fallback) and :func:`multibench.engine.envs.host_has_gpu` is
        False. The message is ``spec.requires_gpu_reason`` - the same
        sentence ``scan`` puts in ``env_reason``.
    """
    if spec.requires_gpu and not envs.host_has_gpu():
        raise OSError(spec.requires_gpu_reason)


def run(method: str, category: str, *, inputs: dict, out_dir: str,
        params: dict | None = None, task: str | None = None, convert: bool = True,
        cmd_template: str | None = None, repo_path: Path | None = None,
        dry_run: bool = False):
    """Run one method on explicit inputs and load its output.

    Use it for full control over one run; ``mtb.run_all`` runs and scores
    every method of a category on a dataset.

    Parameters
    ----------
    method : str
        Registry method id, e.g. ``"Matilda"``.
    category : str
        Integration category of the variant: ``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``.
    inputs : dict, keyword-only
        ``{role: path or AnnData}``, usually from ``mtb.inputs_for``; its
        modality roles select the variant.
    out_dir : str, keyword-only
        Directory the method writes into; created if missing.
    params : dict | None, keyword-only
        Hyperparameter overrides merged over the variant's defaults;
        ``None`` = the defaults.
    task : str | None, keyword-only
        Accepted for forward compatibility; currently ignored.
    convert : bool, keyword-only
        Convert modality inputs to the canonical ``.h5`` layout before the run.
    cmd_template : str | None, keyword-only
        Command wrapper such as ``"conda run -n myenv {cmd}"``; ``None`` =
        enter the method's own env.
    repo_path : Path | None, keyword-only
        Checkout holding ``tools_scripts/``; ``None`` = the configured path,
        cloned on first use.
    dry_run : bool, keyword-only
        ``True`` = return the command without running it or writing anything.

    Returns
    -------
    RunResult or list[str]
        ``RunResult`` - read ``output`` (the primary output, loaded),
        ``out_dir`` and ``stderr``. With ``dry_run=True``, the argv list.

    Raises
    ------
    KeyError
        Unknown method, or no variant fits ``category`` and the input roles.
    OSError
        The method needs a GPU this host lacks, or its env is not installed.
    RuntimeError
        The method exited with a non-zero status.

    Examples
    --------
    >>> import multibench as mtb
    >>> inp = mtb.inputs_for("D11", "vertical", "Matilda")
    >>> mtb.run("Matilda", "vertical", inputs=inp, out_dir="out/Matilda_D11", dry_run=True)
    >>> res = mtb.run("Matilda", "vertical", inputs=inp, out_dir="out/Matilda_D11",
    ...               params={"epochs": 20})
    >>> mtb.evaluate(res.output, labels=mtb.labels_for("D11"))

    Notes
    -----
    **Result.** ``RunResult`` carries ``output`` (the primary output, loaded),
    ``extra`` (``{file: loaded object}`` for the variant's extra outputs),
    ``cmd`` (the argv that ran), ``stdout``, ``stderr``, ``out_dir`` (an
    absolute path) and ``method``.

    **Dry run.** ``dry_run=True`` returns the argv the call would execute,
    built from the same pieces as a real run: variant selection, the command
    builder, the ``driver`` / ``pty`` wrapping and the real env wrap (the
    prefix activation or ``conda run -n <env>``). It creates nothing: no
    ``out_dir``, no ``inputs/`` copies, no env preflight, no fetch of the
    script checkout. ``shlex.join`` it for a shell line.

    The preview shows the inputs absolute, as the run passes them. A real run
    first copies non-canonical inputs to ``<out_dir>/inputs/<role>.h5``;
    canonical ``.h5`` files pass through unchanged, so for a laid-out dataset
    the preview is the real command. The one exception is a variant that first
    rewrites peak names into ``inputs/<role>_normpeaks.h5``.

    **Variant selection.** Only ``category`` and the modality roles of
    ``inputs`` select the variant. The modality roles are every key except the
    auxiliary roles (``data_dir``, ``source_data``, ``target_data``) and the
    label roles (any key containing ``cty`` or ``label``). A ``data_dir``
    variant declares no modalities, so ``category`` alone selects it: pass no
    modality roles with it. ``method_info(m)['supports']`` lists every variant.

    **GPU and CPU.** On a host without an NVIDIA GPU
    (``mtb.env.host_has_gpu()`` is False), the registry ``cpu_params`` - the
    flags that turn CUDA off in a script that has it on by default,
    ``method_info(m)['cpu_params']`` - are merged into ``params`` first, so a
    key you pass always wins.

    The dry run shows these flags too; a real run prints ``[run] no GPU on
    this host: applying <method> cpu_params {...}`` to stderr.

    A script that calls CUDA unconditionally, with no switch
    (``method_info(m)['requires_gpu']``), is refused on such a host with
    ``OSError`` before anything is written or launched. The message is the
    sentence ``mtb.scan`` reports as that row's ``env_reason``. A dry run
    still returns the argv.

    **Environment.** With ``cmd_template=None`` the method runs in the env
    ``mtb.env.group_for(method)`` names, entered in one of two modes, picked
    per call:

    - ``prefix`` whenever ``mtb.env.env_prefix(env)`` finds the env on disk: a
      ``bash -c`` wrapper sets ``CONDA_PREFIX`` / ``CONDA_DEFAULT_ENV``, puts
      ``<prefix>/bin`` first on ``PATH`` and sources the env's ``activate.d``
      scripts. No conda binary is needed.
    - ``conda`` otherwise: ``conda run -n <env>``.

    ``MULTIBENCH_RUN_MODE=conda|prefix`` forces one mode. ``prefix`` with no
    prefix on disk raises ``OSError`` naming ``envs_dir`` and
    ``mtb.env.install``; any other value raises ``ValueError``.

    **Env preflight.** Before any file is written, the env is looked up with
    the probe ``mtb.scan`` uses. If envs are found on this machine and the
    method's env is not among them, ``EnvironmentError`` (Python's alias of
    ``OSError``) is raised, naming the install command. If the probe finds no
    envs at all, the subprocess reports the failure. A ``cmd_template`` takes
    over env control and skips the preflight.

    The method process gets ``PYTHONNOUSERSITE=1``, so user site-packages
    cannot shadow the env, and ``MPLBACKEND=Agg`` unless the variant sets its
    own backend.

    **Paths.** Relative paths in ``inputs`` and ``out_dir`` are made absolute
    before the argv is built, and ``data_dir`` (like any existing directory)
    gets a trailing separator, because the method runs with ``cwd=out_dir``.
    A few variants run in their script's directory instead; the output path
    is passed on the command line either way.

    **What out_dir holds.** Besides the method's own files:

    - for a method fed modality files, ``inputs/`` with their canonical
      ``.h5`` copies (``convert=True``);
    - for a registration method fed a ``data_dir`` of slices, ``inputs/``
      holds the sorted, zero-padded symlinks the script reads, and
      ``slices_manifest.json`` beside it maps each ``aligned_slice_<i>.h5ad``
      back to its source slice (also in ``RunResult.extra``);
    - a ``data_dir`` method that stages nothing gets no ``inputs/`` at all.

    Auxiliary roles (scBridge's ``data_dir`` / ``source_data`` /
    ``target_data`` / ``source_cty`` / ``target_cty``) and label files are
    passed through verbatim, never converted to the canonical ``.h5``.

    **Failures.** A non-zero exit raises ``RuntimeError`` with the tail of the
    method's stdout, then of its stderr (last, so a truncated message keeps
    it). If the call is interrupted - Ctrl-C, or a ``run_all`` timeout - the
    method's whole process tree is killed before the exception propagates.

    **Method scripts.** The upstream scripts are never modified. A variant
    with a package-side ``driver`` runs the driver, which loads the unmodified
    script from its own directory.

    See Also
    --------
    mtb.inputs_for : builds the ``inputs`` dict from a laid-out dataset folder.

    mtb.run_all : every runnable method on a dataset, scored, with failures recorded.

    mtb.scan : previews the same command per method, with the file and env gates.

    mtb.evaluate : scores ``RunResult.output``.
    """
    spec = registry.get(method)
    variant = spec.select(category, _modality_roles(inputs))
    # The CPU switch of a CUDA-by-default script, on a host without a GPU
    # (a caller's explicit key wins). Applied to the preview too, so
    # dry_run / scan()['command'] show the flags the real run emits.
    params, applied = cpu_params_for(spec, params)

    if dry_run:
        # the preview: absolute paths as the run passes them, the checkout
        # located but never fetched, nothing written
        values, out_str = normalize_paths(inputs, out_dir)
        repo = Path(repo_path) if repo_path else _repo_root_no_fetch()
        return _argv(variant, method, values, out_str, repo, params, cmd_template)

    # A script that calls CUDA unconditionally cannot finish on a GPU-less
    # host: refuse here, with the file:line, instead of leaving the user a
    # "Torch not compiled with CUDA enabled" traceback minutes later.
    check_gpu_requirement(spec)
    if applied:
        print(f"[run] no GPU on this host: applying {method} cpu_params {applied}",
              file=sys.stderr, flush=True)

    # Env preflight: without it a missing env only surfaces after inputs were
    # converted and the subprocess spawned, in a stderr tail with no install
    # hint. Same probe scan() uses (prefixes under envs_dir plus what conda
    # lists). Skipped when the caller controls the env via cmd_template, and
    # when the probe finds nothing (no prefixes, conda absent or broken): the
    # subprocess then reports the failure.
    if cmd_template is None:
        env_name = envs.group_for(method)
        have = envs.installed_envs()
        if have and env_name not in have:
            # conda's env list is cached per process; an env created outside
            # this process since then is only missing from the cache, so
            # re-probe once before refusing
            envs._conda_prefixes.cache_clear()
            have = envs.installed_envs()
        if have and env_name not in have:
            raise EnvironmentError(
                f"conda env {env_name!r} ({method}) is not installed - run "
                f"`multibench env install --methods {method} --packed --run` "
                f"(or mtb.env.install([{method!r}], dry_run=False)); see mtb.env.doctor()")

    # Absolute paths + trailing separator on directory roles before conversion,
    # so canonical passthrough files are absolute too; converted copies live
    # under the (absolute) inputs_dir and come out absolute by construction.
    inputs, out_str = normalize_paths(inputs, out_dir)
    out = Path(out_str)
    workdir = out
    workdir.mkdir(parents=True, exist_ok=True)
    inputs_dir = workdir / "inputs"
    # inputs/ holds the canonical copies of a file-role method or the staged
    # slice links of a registration method; a data_dir method that stages
    # nothing (scBridge) gets none.
    if _modality_roles(inputs):
        inputs_dir.mkdir(parents=True, exist_ok=True)
    manifest = None
    if "data_dir" in inputs and variant.output.kind == "coords":
        manifest = stage_slices(inputs["data_dir"], inputs_dir)
        (workdir / SLICES_MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        inputs = {**inputs, "data_dir": manifest["staged_dir"]}

    # normalize modality inputs to canonical .h5 inside a dedicated inputs dir.
    # For the two ATAC-representation roles the role token is passed as the
    # modality hint so the conversion emits ingest's peak-name warning (an
    # atac_gas role fed a chr:start-end matrix); other roles carry no hint -
    # `modality='adt'` would refuse an AnnData that has obsm keys, and in-memory
    # AnnData inputs must keep working unchanged.
    values: dict = {}
    for role, val in inputs.items():
        if convert and role not in _AUX_ROLES and "cty" not in role and "label" not in role:
            hint = role.rstrip("0123456789")
            kw = {"modality": hint} if hint in ("atac_gas", "atac_peak") else {}
            values[role] = str(ingest.to_canonical(val, out=inputs_dir / f"{role}.h5", **kw))
        else:
            values[role] = val

    # Some methods (Seurat_v3 etc.) require ATAC peak ids in chr:start-end form;
    # normalize the declared peak roles into per-run copies (originals untouched).
    for role in (getattr(variant, "normalize_peaks", None) or []):
        if role in values:
            npath = inputs_dir / f"{role}_normpeaks.h5"
            values[role] = str(ingest.normalize_peak_names(values[role], npath))
    repo = Path(config.ensure_repo(repo_path))
    cmd = _argv(variant, method, values, out_str, repo, params, cmd_template)

    # Isolate the method env from user site-packages (~/.local): a broken or
    # mismatched ~/.local can shadow the conda env (e.g. a libcublas-less torch
    # egg breaking anndata imports). PYTHONNOUSERSITE=1 makes the env hermetic.
    run_env = {**os.environ, "PYTHONNOUSERSITE": "1", **{k: str(v) for k, v in (variant.run_env or {}).items()}}
    # A Jupyter kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline,
    # which leaks through `conda run` into the method's env, where matplotlib_inline
    # does not exist - so any method that imports matplotlib dies at import when
    # run_all is called from a notebook. Agg is the safe headless backend for a
    # subprocess that at most saves figures.
    if "MPLBACKEND" not in (variant.run_env or {}):
        run_env["MPLBACKEND"] = "Agg"
    # Some scripts source/import local files relative to the entrypoint dir,
    # so let variants opt into running with cwd=script's parent rather than
    # cwd=out_dir. The out_dir is still passed via the --save_path arg, so
    # outputs land in the correct place regardless.
    exec_cwd = str((repo / variant.entrypoint).parent) if variant.cwd_at_script else str(workdir)
    # The child is `conda run` (or the prefix-mode bash, which execs the
    # method) and the actual method may be its grandchild; killing only the
    # direct child on a timeout would leave the method running. Own session
    # -> one killpg reaps the whole tree.
    popen = subprocess.Popen(cmd, cwd=exec_cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=run_env,
                             start_new_session=True)
    try:
        _stdout, _stderr = popen.communicate()
    except BaseException:            # TimeoutError from the deadline included
        import os as _os
        import signal as _sig
        try:
            _os.killpg(popen.pid, _sig.SIGKILL)
        except Exception:
            popen.kill()
        popen.wait()
        raise
    proc = subprocess.CompletedProcess(cmd, popen.returncode, _stdout, _stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{method} failed (exit {proc.returncode}).\n"
            # stdout first, stderr last: callers truncate this message from the
            # left, so the most diagnostic part - stderr - has to sit at the end
            # (the other order keeps conda's "see above for error" summary and
            # cuts the error it refers to).
            f"stdout tail:\n{proc.stdout[-1500:]}\n"
            f"stderr tail:\n{proc.stderr[-2500:]}"
        )

    primary = io.load_output(out, variant.output)
    extra = {o.file: io.load_output(out, o) for o in variant.extra_outputs}
    if manifest is not None:
        extra[SLICES_MANIFEST] = manifest
    return RunResult(method=method, out_dir=out, cmd=cmd, output=primary, extra=extra,
                     stdout=proc.stdout, stderr=proc.stderr)
