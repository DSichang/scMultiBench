"""Run a method variant: build cmd, wrap via cmd_template, exec in a workdir, load output."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from .. import config
from . import builder, io, ingest, registry, envs
from .schema import _batch_of, base_modality


@dataclass
class RunResult:
    """What ``mtb.run`` returns for one method.

    Attributes
    ----------
    method : str
        The method id that ran.
    out_dir : Path
        Folder holding everything the method wrote.
    cmd : list[str]
        The argv that ran; ``shlex.join(res.cmd)`` repeats the run by hand.
    output : object
        The primary output, loaded; an embedding array for most methods.
    extra : dict
        ``{filename: loaded object}`` for the variant's extra outputs.
    stdout : str
        Captured standard output of the method.
    stderr : str
        Captured standard error; read it first when a result looks wrong.
    obs_names : list[str] | None
        Cell barcodes of the output, in the method's cell order; ``None`` when
        they could not be matched.
    scripts_commit : str | None
        Commit of the method scripts that ran; ``None`` when they are not a
        git checkout.
    env_flavor : str
        ``'cpu'`` or ``'gpu'`` build of the method's environment, else
        ``'unknown'``.
    hostname : str
        Name of the computer that ran the method.

    Examples
    --------
    >>> import multibench as mtb
    >>> inp = mtb.inputs_for("D11", "vertical", "totalVI")
    >>> res = mtb.run("totalVI", "vertical", inputs=inp, out_dir="out/totalVI_D11")
    >>> adata.obsm["X_totalVI"] = res.output      # rows match res.obs_names
    >>> assert list(adata.obs_names) == res.obs_names

    Notes
    -----
    **Row order.** Rows follow the method's cell order, which is the order
    ``mtb.labels_for(dataset, category, method)`` uses. ``obs_names`` holds
    the barcodes from the input files in that order. It is ``None``, with a
    ``UserWarning``, when their number differs from the output's cells or the
    inputs are a folder (scBridge).

    **Orientation.** Most methods write cells x dims. When
    ``res.output.shape[0] != len(res.obs_names)`` the output is dims x cells:
    store ``res.output.T``. ``mtb.evaluate`` orients a raw array itself.

    **Output kind.** Not every method returns an embedding: Seurat_WNN
    writes only a neighbour graph. ``mtb.run_all`` reads the kind of each
    method and scores only embeddings.
    """

    method: str
    out_dir: Path
    cmd: list[str]
    output: object                  # primary loaded output (e.g. embedding ndarray)
    extra: dict                     # role/file -> loaded extra outputs
    stdout: str = ""
    stderr: str = ""
    obs_names: list | None = None   # input barcodes in the output's cell order
    # provenance (config.run_provenance): what ran where
    scripts_commit: str | None = None
    env_flavor: str = "unknown"
    hostname: str = ""


def wrap_command(cmd: list[str], cmd_template: str | None) -> list[str]:
    """Wrap argv with a cmd_template like 'conda run -n env {cmd}'."""
    if not cmd_template or cmd_template.strip() == "{cmd}":
        return cmd
    prefix = cmd_template.replace("{cmd}", "").strip()
    return shlex.split(prefix) + cmd


#: The ``cmd_template`` placeholder for the command with the env activation
#: the default path uses (``{cmd}`` is the bare command).
ENV_CMD = "{env_cmd}"


def _check_template(cmd_template: str | None) -> None:
    """``ValueError`` for a template that uses both placeholders."""
    if cmd_template and ENV_CMD in cmd_template and "{cmd}" in cmd_template:
        raise ValueError(
            f"cmd_template {cmd_template!r} uses both {{cmd}} and {ENV_CMD}; use "
            f"{ENV_CMD} to run inside the method env, or {{cmd}} for the bare command")


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


def normalize_paths(inputs: dict, out_dir) -> tuple[dict, str]:
    """Absolutize every path-valued input and ``out_dir``; directory roles get
    a trailing separator.

    The method runs with ``cwd=out_dir`` (or the script's own directory), so a
    relative ``data/MYCITE/rna.h5`` would be looked up in the wrong place and
    a relative ``--save_path out/x/`` would write ``out/x/out/x/embedding.h5``.
    Many upstream scripts also string-concatenate a directory and a file name
    (``data_path + "rna.h5"``), hence the separator on directory values.
    ``os.path.abspath`` (not ``Path.resolve``) keeps symlinked data roots as
    the user wrote them.

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


#: input keys a method's variant renamed, ``{method: {old key: new key}}``.
#: UnitedNet's label role was ``rna_cty`` up to 0.3.1 and is ``cty`` now.
_RENAMED_ROLES = {"UnitedNet": {"rna_cty": "cty"}}


def _rename_old_roles(method: str, inputs: dict) -> dict:
    """``inputs`` with a renamed key (:data:`_RENAMED_ROLES`) spelled the new
    way, with a ``DeprecationWarning`` naming the new key. A dict that
    already holds the new key is returned unchanged."""
    renamed = _RENAMED_ROLES.get(method) or {}
    old = [k for k in inputs if k in renamed and renamed[k] not in inputs]
    if not old:
        return inputs
    for k in old:
        warnings.warn(f"inputs key {k!r} of {method} is deprecated and will be removed "
                      f"in 0.4; use {renamed[k]!r} (the same file)",
                      DeprecationWarning, stacklevel=3)
    return {renamed.get(k, k) if k in old else k: v for k, v in inputs.items()}


def _check_input_cells(method: str, category: str, inputs: dict) -> None:
    """The cell checks of ``inputs_for(check=True)`` on explicit input paths.

    Seurat_v5's ``rna`` and ``atac_peak`` must hold the same cells, and a
    diagonal ``atac_gas`` file must list the cells of its ``atac_peak`` file
    (the one given, else the one next to it) in the same order. Raises the
    ``ValueError`` of the file check. Values that are not readable canonical
    ``.h5`` paths (an AnnData, an ``.h5ad``) are not checked.
    """
    from . import resolve
    paths = {r: v for r, v in inputs.items() if isinstance(v, (str, os.PathLike))}
    if not paths:
        return
    dataset = Path(next(iter(paths.values()))).parent.name or "inputs"
    resolve._check_same_cells(method, dataset, category, paths)
    if category == "diagonal":
        resolve._check_atac_gas_cells(method, dataset, paths)


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


#: The shell command that fetches the method scripts ahead of a run.
FETCH_SCRIPTS_CMD = "multibench fetch --scripts"


def linux_only_sentence() -> str | None:
    """``"Methods run only on Linux (this computer is darwin/arm64)."``, or
    ``None`` on Linux (where ``envs.host_platform_problem()`` is ``None``).

    The platform part comes from ``host_platform_problem()``'s own sentence
    ("... this host is <os>/<arch>"), so a patched value in a test and the
    real one read alike.
    """
    problem = envs.host_platform_problem()
    if not problem:
        return None
    marker = "this host is "
    host = problem.rsplit(marker, 1)[1].strip() if marker in problem else sys.platform
    return f"Methods run only on Linux (this computer is {host})."


def missing_script_fix(repo) -> str:
    """What to do about a method script missing from the checkout at ``repo``.

    A git checkout is updated or fetched again. A folder that is not a git
    checkout is most often one made by hand before the scripts were fetched
    (to hold GLUE's annotation): ``git pull`` does not apply there, and the
    folder blocks the fetch until it is moved away.
    """
    if config.scripts_commit(repo) is not None:
        return ("update it with git pull, or delete it so the next run fetches a "
                "fresh copy")
    return (f"that folder is not a git checkout. If you made it by hand, move out "
            f"the files you added, delete it, run `{FETCH_SCRIPTS_CMD}` and put the "
            f"files back; otherwise copy a full scripts checkout there")


def script_notes(spec, variant, repo: Path) -> list[str]:
    """Setup facts a preview must show before the real run fails on them.

    - the method's ``setup_hint`` (a file the user must supply, such as
      GLUE's GENCODE annotation), when it has one. A variant whose declared
      helper files (MIRA's ``logger.py``) are all in place needs no hint;
    - a method script that is not on disk: without any checkout, the first
      real run clones the scripts with ``git``, which an offline compute
      node cannot do.

    Returns plain sentences, without a comment marker; :func:`run` prints
    them to stderr on a dry run and ``mtb.scan`` adds them to ``caveat``.
    The setup note always comes first and starts with ``"setup: "``.
    """
    notes = []
    ep = Path(variant.entrypoint)
    helpers = getattr(variant, "helpers", None) or []
    done = bool(helpers) and all((repo / ep).parent.joinpath(h).exists() for h in helpers)
    if spec.setup_hint and not done:
        notes.append(f"setup: {spec.setup_hint}")
    if not (repo / ep).exists():
        if (repo / "tools_scripts").is_dir():
            notes.append(f"method script {ep} not found in the checkout at {repo}: "
                         + missing_script_fix(repo))
        else:
            notes.append(f"method scripts not found under {repo}: the first real run "
                         f"clones PYangLab/scMultiBench with git; on a host without "
                         f"network, fetch them first ({FETCH_SCRIPTS_CMD})")
    return notes


def _is_modality_role(role: str) -> bool:
    """A role whose value ``run`` converts to a canonical ``.h5`` (not an
    auxiliary role, not a label file)."""
    return role not in _AUX_ROLES and "cty" not in role and "label" not in role


def _check_input(role: str, val, *, real: bool) -> str:
    """Validate one modality input without reading or writing it.

    Returns ``"pass"`` when the value is a canonical ``.h5``, else
    ``"convert"``. Raises the error the conversion would raise, so a dry run
    fails exactly where the real run would. A path that does not exist
    raises ``FileNotFoundError`` in a real run (``real=True``); a preview
    may name files of another host, so it passes an ``.h5`` path through.
    """
    if not isinstance(val, (str, os.PathLike)):
        if ingest._is_mudata(val):
            mods = list(getattr(val, "mod", {}) or {})
            raise ValueError(
                f"inputs[{role!r}] is a MuData; pass one modality, e.g. "
                f"inputs={{{role!r}: mdata.mod[{(mods or [role])[0]!r}]}} (modalities: "
                f"{mods}), or write a dataset folder with mtb.io.export_dataset")
        if hasattr(val, "X") and hasattr(val, "obs"):
            return "convert"
        raise TypeError(f"inputs[{role!r}] must be a file path or an AnnData, "
                        f"got {type(val).__name__}")
    p = Path(val)
    suf = p.suffix.lower()
    if suf == ".h5mu":
        raise ValueError(
            f"inputs[{role!r}] is a MuData file ({p.name}); pass one modality as an "
            f".h5ad or AnnData (mdata.mod[{role!r}]), or write a dataset folder with "
            f"mtb.io.export_dataset")
    if not p.exists():
        if real:
            raise FileNotFoundError(f"input file does not exist: {p} (cwd {os.getcwd()})")
        return "pass" if suf == ".h5" else "convert"
    if suf == ".h5":
        if ingest._is_canonical_h5(p):
            return "pass"
        ingest._to_anndata(p)          # raises the "no dataset 'matrix/data'" error
    if suf in (".h5ad", ".csv", ".tsv"):
        return "convert"
    if suf == ".loom":
        import importlib.util
        if importlib.util.find_spec("loompy") is None:
            raise ImportError(
                "reading .loom requires the optional 'loompy' package "
                "(pip install 'multibench-sc[loom]' or pip install loompy); "
                "alternatively convert the input to .h5ad/.csv first.")
        return "convert"
    raise ValueError(f"unsupported input format: {p.name}")


def _plan_inputs(variant, inputs: dict, inputs_dir: Path, *, convert: bool,
                 real: bool) -> dict:
    """The file each role hands the script, and how it gets there.

    ``{role: {"value": <what the argv gets>, "src": <input>, "convert": bool,
    "normpeaks_from": <path or None>}}``. Shared by the real run and the dry
    run, so the preview names the files the run will pass: a canonical
    ``.h5`` as is, anything else ``<out_dir>/inputs/<role>.h5``, and a
    ``normalize_peaks`` role ``<out_dir>/inputs/<role>_normpeaks.h5``.
    Nothing is written; every format check the conversion applies runs here.
    """
    plan: dict = {}
    for role, val in inputs.items():
        step = {"value": val, "src": val, "convert": False, "normpeaks_from": None}
        if _is_modality_role(role):
            if convert:
                if _check_input(role, val, real=real) == "convert":
                    step["value"], step["convert"] = str(inputs_dir / f"{role}.h5"), True
                else:
                    step["value"] = str(val)
            elif not isinstance(val, (str, os.PathLike)):
                raise ValueError(f"convert=False needs file paths; inputs[{role!r}] is "
                                 f"a {type(val).__name__}")
        plan[role] = step
    for role in (getattr(variant, "normalize_peaks", None) or []):
        if role in plan:
            plan[role]["normpeaks_from"] = plan[role]["value"]
            plan[role]["value"] = str(inputs_dir / f"{role}_normpeaks.h5")
    return plan


def _cell_group_roles(variant) -> list[list[str]]:
    """The variant's input roles grouped by the cells they hold, in the
    order the output stacks the groups (``Variant.stacked_roles``).

    Vertical inputs share one set of cells. Diagonal inputs hold one set per
    base modality (``atac_peak`` and ``atac_gas`` are the same ATAC cells).
    Mosaic and cross inputs hold one set per batch (``rna1`` + ``adt1``).
    """
    category = variant.when.get("category")
    groups: dict = {}
    for r in variant.stacked_roles():
        if category == "vertical":
            key = "cells"
        elif category == "diagonal":
            key = base_modality(r)
        else:
            key = _batch_of(r) if _batch_of(r) is not None else r
        groups.setdefault(key, []).append(r)
    return list(groups.values())


def _read_barcodes(path) -> list[str]:
    import h5py
    with h5py.File(path, "r") as f:
        return [b.decode() if isinstance(b, bytes) else str(b)
                for b in f["matrix/barcodes"][:]]


def _obs_names(method: str, variant, values: dict, output) -> list[str] | None:
    """The input barcodes in the output's cell order, or ``None`` with a warning.

    Each cell group (:func:`_cell_group_roles`) contributes the barcodes of
    its first role with a readable canonical file. The list is kept only
    when its length equals one axis of ``output`` (the cell axis, as
    ``evaluate`` orients a raw array).
    """
    why = None
    names: list[str] = []
    # a const argument (Seurat_WNN's atac 'NULL') is not an input: only roles
    # the call was given can hold cells
    groups = [[r for r in g if r in values] for g in _cell_group_roles(variant)]
    groups = [g for g in groups if g]
    if not groups:
        why = "the method reads a folder, not modality files"
    for roles in groups:
        got = None
        for r in roles:
            v = values[r]
            if isinstance(v, (str, os.PathLike)) and str(v).endswith(".h5"):
                try:
                    got = _read_barcodes(v)
                    break
                except (OSError, KeyError):
                    continue
        if got is None:
            why = f"no barcodes readable for input role(s) {roles}"
            break
        names += got
    if why is None:
        shape = tuple(getattr(output, "shape", ()) or ())
        if not shape and isinstance(output, (list, tuple)):
            shape = (len(output),)
        if len(names) not in shape[:2]:
            why = (f"the inputs hold {len(names)} cells but the output has shape "
                   f"{shape}")
    if why is None:
        return names
    warnings.warn(f"{method}: RunResult.obs_names is None ({why}); the rows follow "
                  f"the order of mtb.labels_for(dataset, category, {method!r})",
                  UserWarning, stacklevel=3)
    return None


def _host_run_env(run_env: dict | None) -> dict:
    """The variant's ``run_env`` as :func:`run` applies it on this machine.

    A value made of absolute paths (``LD_PRELOAD`` may list several,
    ``:``- or space-separated) names files on the host the registry was
    written on: only the paths that exist here are applied, and when none
    does the key is left unset, so the caller's own value or the tool's
    default discovery stays in force. Where every path exists the value is
    passed unchanged; any other value always is.
    """
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
    outer = None
    if cmd_template is not None and ENV_CMD in cmd_template:
        # {env_cmd}: the default env wrap below, then the caller's launcher
        outer, cmd_template = cmd_template.replace(ENV_CMD, "{cmd}"), None
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
        cmd = wrap_prefix(cmd, *activate)
    else:
        cmd = wrap_command(cmd, cmd_template)
    return wrap_command(cmd, outer) if outer is not None else cmd


def cpu_params_for(spec, params: dict | None, *,
                   gpu: bool | None = None) -> tuple[dict | None, dict]:
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
    gpu : bool or None
        Whether the run has a GPU; ``None`` = probe this host.

    Returns
    -------
    tuple[dict or None, dict]
        ``(merged, applied)`` - ``merged`` is ``params`` with the applied
        keys added (``params`` itself, untouched, when nothing applies);
        ``applied`` is exactly the subset of ``cpu_params`` that was merged
        (empty on a GPU host, for a method without ``cpu_params``, or when
        the caller set every key).
    """
    if not spec.cpu_params or (envs.host_has_gpu() if gpu is None else gpu):
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
    inputs : dict
        ``{role: path or AnnData}``, usually from ``mtb.inputs_for``; its
        modality roles select the variant.
    out_dir : str
        Folder the method writes into; created if missing.
    params : dict | None
        Hyperparameter overrides merged over the variant's defaults;
        ``None`` = the defaults.
    task : str | None
        Accepted for forward compatibility; currently ignored.
    convert : bool
        Convert modality inputs to the canonical ``.h5`` layout before the run.
    cmd_template : str | None
        Launcher template: ``{cmd}`` = the bare command, ``{env_cmd}`` = the
        command inside the method env; ``None`` = enter the method env.
    repo_path : Path | None
        Checkout holding ``tools_scripts/``; ``None`` =
        ``mtb.config.DEFAULT.repo_path`` if it has one, else the package root
        if it has one, else a clone into the former.
    dry_run : bool
        ``True`` = return the command without running it or writing anything.

    Returns
    -------
    RunResult or list[str]
        ``RunResult`` - read ``output`` (the primary output, loaded),
        ``obs_names`` (its cell barcodes) and ``stderr``. With
        ``dry_run=True``, the argv list.

    Raises
    ------
    KeyError
        Unknown method, or no variant fits ``category`` and the input roles.
    ValueError
        An input the run cannot convert, such as an ``.h5mu`` file or a MuData.
    ValueError
        Input files that must hold the same cells, in one order, do not.
    OSError
        The method needs a GPU this host lacks, or its env is not installed.
    RuntimeError
        The method exited with a non-zero status, or its scripts could not be fetched.

    Warns
    -----
    UserWarning
        The input barcodes do not match the output's cells; ``obs_names`` is ``None``.
    UserWarning
        An ATAC input holds the other representation or peak names the method cannot read.
    DeprecationWarning
        UnitedNet's labels passed under the old key ``rna_cty``; the key is ``cty``.

    Examples
    --------
    >>> import multibench as mtb
    >>> inp = mtb.inputs_for("D11", "vertical", "Matilda")
    >>> mtb.run("Matilda", "vertical", inputs=inp, out_dir="out/Matilda_D11",
    ...         dry_run=True)
    >>> res = mtb.run("Matilda", "vertical", inputs=inp, out_dir="out/Matilda_D11",
    ...               params={"epochs": 20})
    >>> mtb.evaluate(res.output, labels=mtb.labels_for("D11"))
    >>> adata.obsm["X_Matilda"] = res.output      # rows match res.obs_names

    Notes
    -----
    **Result.** ``RunResult`` carries ``output`` (the primary output, loaded),
    ``obs_names`` (the input barcodes in the output's row order),
    ``extra`` (``{file: loaded object}`` for the variant's extra outputs),
    ``cmd`` (the argv that ran), ``stdout``, ``stderr``, ``out_dir`` (an
    absolute path) and ``method``.

    **Dry run.** ``dry_run=True`` returns the argv the real run would
    execute. It uses the same variant selection, input plan, command builder
    and env wrap as a real run. It creates nothing: no ``out_dir``, no
    ``inputs/`` copies, no env check, no fetch of the method scripts.
    ``shlex.join`` it for a shell line.

    The preview names the files the run passes. A canonical ``.h5`` passes
    through. An AnnData or any other file becomes
    ``<out_dir>/inputs/<role>.h5``, and a peak role the method renames
    becomes ``<out_dir>/inputs/<role>_normpeaks.h5``. Input-format errors,
    such as an ``.h5mu`` file, are raised by the dry run too.

    The dry run prints to stderr what the real run would need first: the
    method's ``setup_hint`` (``method_info(m)['setup_hint']``), a note when
    the method scripts are not on this machine yet or not at
    ``MULTIBENCH_SCRIPTS_REF``, and a note when the command reads a file
    under ``inputs/`` that the run writes first.

    An ATAC file that ``mtb.scan`` would block gets a note too: it holds the
    other representation, or peak names the method cannot read. The real
    run warns and still runs.

    **Variant selection.** Only ``category`` and the modality roles of
    ``inputs`` select the variant. The modality roles are every key except the
    auxiliary roles (``data_dir``, ``source_data``, ``target_data``,
    ``out_dir``) and the label roles (any key containing ``cty`` or
    ``label``). A ``data_dir`` variant declares no modalities, so ``category``
    alone selects it: pass no modality roles with it.

    ``method_info(m)['supports']`` lists every variant. A misspelt method id
    raises ``KeyError`` naming the closest method id; a category and role
    set with no variant raises ``KeyError`` listing the declared
    ``(category, modalities)`` pairs.

    Auxiliary roles (scBridge's ``data_dir`` / ``source_data`` /
    ``target_data`` / ``source_cty`` / ``target_cty``) and label files are
    never converted to the canonical ``.h5``.

    **Inputs.** A MuData, in memory or as an ``.h5mu`` file, raises
    ``ValueError``: pass one modality per role (``mdata.mod["rna"]``), or
    write the folder with ``mtb.io.export_dataset``. With ``convert=False``
    every modality input must already be a file path.

    **Cell checks.** Before anything runs, the dry run included, canonical
    ``.h5`` inputs get the cell checks of ``mtb.inputs_for(check=True)``.
    Seurat_v5's ``rna`` and ``atac_peak`` must hold the same cells. A
    diagonal ``atac_gas`` file must list the cells of its ``atac_peak``
    file, the one given or the one next to it, in the same order.

    UnitedNet's label input is ``cty``. The older key ``rna_cty`` still
    works, with a ``DeprecationWarning``.

    **GPU and CPU.** On a host without an NVIDIA GPU
    (``mtb.env.host_has_gpu()`` is False), the method's ``cpu_params`` - the
    flags that turn CUDA off in a script that has it on by default,
    ``method_info(m)['cpu_params']`` - are merged into ``params`` first. A
    key you pass always wins.

    The dry run shows these flags too; a real run prints ``[run] no GPU on
    this host: applying <method> cpu_params {...}`` to stderr.

    A script that calls CUDA unconditionally, with no switch
    (``method_info(m)['requires_gpu']``), is refused on such a host with
    ``OSError`` before anything is written or launched. The message is the
    sentence ``mtb.scan`` reports as that row's ``env_reason``. A dry run
    still returns the argv.

    **Environment.** With ``cmd_template=None`` the method runs in the env
    ``mtb.method_info(method)['env']`` names, entered in one of two modes,
    picked per call:

    - ``prefix`` whenever ``mtb.env.env_prefix(env)`` finds the env on disk: a
      ``bash -c`` wrapper sets ``CONDA_PREFIX`` / ``CONDA_DEFAULT_ENV``, puts
      ``<prefix>/bin`` first on ``PATH`` and sources the env's ``activate.d``
      scripts. No conda binary is needed.
    - ``conda`` otherwise: ``conda run -n <env>``.

    ``MULTIBENCH_RUN_MODE=conda|prefix`` forces one mode. ``prefix`` with no
    prefix on disk raises ``OSError`` naming ``envs_dir`` and
    ``mtb.env.install``; any other value raises ``ValueError``.

    The method process gets ``PYTHONNOUSERSITE=1``, so user site-packages
    cannot shadow the env, and ``MPLBACKEND=Agg`` unless the method sets its
    own backend.

    Environment variables the method itself needs are set over yours. A
    value made of absolute paths, such as ``LD_PRELOAD``, is applied only
    for the paths that exist on this machine. When none does, the variable
    is left as you set it, or unset, so the tool's default lookup applies.

    **Env check.** Before any file is written, the env is looked up with
    the probe ``mtb.scan`` uses. If envs are found on this machine and the
    method's env is not among them, ``EnvironmentError`` (Python's alias of
    ``OSError``) is raised, naming the install command. On Linux, if the
    probe finds no envs at all, the subprocess reports the failure. A
    ``cmd_template`` with ``{cmd}`` takes over env control and skips the
    check.

    **Other systems.** Method environments are Linux-only. On macOS or
    Windows a missing env always raises ``OSError``, and the message starts
    with that fact: run the call on a Linux machine. ``dry_run=True``
    previews the command here; the command holds this computer's paths.

    **Launcher templates.** ``cmd_template`` wraps the command in your own
    launcher. ``{env_cmd}`` is the command with the env activation above.
    ``{cmd}`` is the bare command: the template must then enter an env
    itself, for example ``"conda run -n myenv {cmd}"``.

    A Slurm job step:

    ```python
    mtb.run("scMoMaT", "mosaic", inputs=mtb.inputs_for("D46", "mosaic", "scMoMaT"),
            out_dir="runs/scMoMaT", cmd_template="srun --gres=gpu:1 {env_cmd}")
    ```

    Request a GPU only for a method that uses one;
    ``mtb.method_info(m)['gpu']`` says which.

    **Paths.** Relative paths in ``inputs`` and ``out_dir`` are made absolute
    before the argv is built, and ``data_dir`` (like any existing directory)
    gets a trailing separator, because the method runs with ``cwd=out_dir``.
    A few variants run in their script's directory instead; the output path
    is passed on the command line either way.

    **What out_dir holds.** Besides the method's own files:

    - for a method fed modality files, ``inputs/`` with their canonical
      ``.h5`` copies (``convert=True``);
    - a ``data_dir`` method (scBridge) gets no ``inputs/`` at all.

    **Failures.** A non-zero exit raises ``RuntimeError`` with the tail of the
    method's stdout, then of its stderr (last, so a truncated message keeps
    it). If the call is interrupted - Ctrl-C, or a ``run_all`` timeout - the
    method's whole process tree is killed before the exception propagates.

    **Method scripts.** The upstream scripts are never modified. A method
    with a package-side driver runs the driver, which loads the unmodified
    script from its own directory. The first run on a machine clones them
    with ``git``. On a host without network, fetch them first with
    ``multibench fetch --scripts``; a failed clone raises ``RuntimeError``
    naming that command.

    See Also
    --------
    mtb.inputs_for : builds the ``inputs`` dict from a laid-out dataset folder.

    mtb.run_all : every runnable method on a dataset, scored, with failures recorded.

    mtb.scan : previews the same command per method, with the file and env checks.

    mtb.evaluate : scores ``RunResult.output``.
    """
    _check_template(cmd_template)
    inputs = _rename_old_roles(method, inputs)
    # the cell checks scan and inputs_for(check=True) apply, before anything runs
    _check_input_cells(method, category, inputs)
    # the ATAC checks of mtb.scan: the other representation, or peak names
    # the script cannot read (workflow imports this module, hence here)
    from ..workflow import _atac_mismatch_caveats
    mismatch = [f"{method} {c}" for c in _atac_mismatch_caveats(method, category, inputs)]
    if dry_run:
        argv, notes = preview(method, category, inputs=inputs, out_dir=out_dir,
                              params=params, convert=convert,
                              cmd_template=cmd_template, repo_path=repo_path)
        for note in notes + mismatch:
            print(f"# {note}", file=sys.stderr, flush=True)
        return argv
    if mismatch:
        warnings.warn("; ".join(mismatch), UserWarning, stacklevel=2)

    spec = registry.get(method)
    variant = spec.select(category, _modality_roles(inputs))
    # The CPU switch of a CUDA-by-default script, on a host without a GPU
    # (a caller's explicit key wins). Applied to the preview too, so
    # dry_run / scan()['command'] show the flags the real run emits.
    params, applied = cpu_params_for(spec, params)

    # A script that calls CUDA unconditionally cannot finish on a GPU-less
    # host: refuse here, pointing at method_info, instead of leaving the
    # user a "Torch not compiled with CUDA enabled" traceback minutes later.
    check_gpu_requirement(spec)
    if applied:
        print(f"[run] no GPU on this host: applying {method} cpu_params {applied}",
              file=sys.stderr, flush=True)

    # Env preflight: without it a missing env only surfaces after inputs were
    # converted and the subprocess spawned, in a stderr tail with no install
    # hint. Same probe scan() uses (prefixes under envs_dir plus what conda
    # lists). Skipped when the caller controls the env via a cmd_template
    # with {cmd}. On Linux it is also skipped when the probe finds nothing (no
    # prefixes, conda absent or broken): the subprocess then reports the
    # failure. Off Linux a missing env cannot be installed at all, so it
    # always refuses.
    if cmd_template is None or ENV_CMD in cmd_template:
        env_name = envs.group_for(method)
        linux_only = linux_only_sentence()
        have = envs.installed_envs()
        if (have or linux_only) and env_name not in have:
            # conda's env list is cached per process; an env created outside
            # this process since then is only missing from the cache, so
            # re-probe once before refusing
            envs._conda_prefixes.cache_clear()
            have = envs.installed_envs()
        if (have or linux_only) and env_name not in have:
            if linux_only:
                # an install refuses off Linux: the platform sentence alone
                this = config.hint("this call", "this command")
                preview_with = config.hint("dry_run=True", "--dry-run")
                raise EnvironmentError(
                    f"{linux_only} Run {this} on a Linux machine; {preview_with} "
                    f"previews the method's command here.")
            py = f" (or mtb.env.install([{method!r}], dry_run=False)); see mtb.env.doctor()"
            raise EnvironmentError(
                f"conda env {env_name!r} ({method}) is not installed - run "
                f"`multibench env install --methods {method} --packed --run`"
                + config.hint(py, "; see `multibench env doctor`"))

    # Absolute paths + trailing separator on directory roles before conversion,
    # so canonical passthrough files are absolute too; converted copies live
    # under the (absolute) inputs_dir and come out absolute by construction.
    inputs, out_str = normalize_paths(inputs, out_dir)
    out = Path(out_str)
    workdir = out
    inputs_dir = workdir / "inputs"
    # every input-format check before anything is written or fetched
    plan = _plan_inputs(variant, inputs, inputs_dir, convert=convert, real=True)
    repo = _fetch_scripts(repo_path)
    workdir.mkdir(parents=True, exist_ok=True)
    # inputs/ holds the canonical copies of a file-role method; a data_dir
    # method (scBridge) gets none.
    if _modality_roles(inputs):
        inputs_dir.mkdir(parents=True, exist_ok=True)

    # normalize modality inputs to canonical .h5 inside a dedicated inputs dir.
    # For the two ATAC-representation roles the role token is passed as the
    # modality hint so the conversion emits ingest's peak-name warning (an
    # atac_gas role fed a chr:start-end matrix); other roles carry no hint -
    # `modality='adt'` would refuse an AnnData that has obsm keys, and in-memory
    # AnnData inputs must keep working unchanged.
    values: dict = {}
    for role, step in plan.items():
        if step["convert"]:
            hint = role.rstrip("0123456789")
            kw = {"modality": hint} if hint in ("atac_gas", "atac_peak") else {}
            ingest.to_canonical(step["src"], out=inputs_dir / f"{role}.h5", **kw)
        # Some methods (Seurat_v3 etc.) require ATAC peak ids in chr:start-end
        # form: the declared peak roles get per-run copies (originals untouched).
        if step["normpeaks_from"] is not None:
            ingest.normalize_peak_names(step["normpeaks_from"], step["value"])
        values[role] = step["value"]
    cmd = _argv(variant, method, values, out_str, repo, params, cmd_template)
    # a {cmd} template runs the method in the caller's env, whose build is not ours
    ours = cmd_template is None or ENV_CMD in cmd_template
    provenance = config.run_provenance(envs.group_for(method) if ours else None, repo)

    # Isolate the method env from user site-packages (~/.local): a broken or
    # mismatched ~/.local can shadow the conda env (e.g. a libcublas-less torch
    # egg breaking anndata imports). PYTHONNOUSERSITE=1 makes the env hermetic.
    run_env = {**os.environ, "PYTHONNOUSERSITE": "1", **_host_run_env(variant.run_env)}
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
    return RunResult(method=method, out_dir=out, cmd=cmd, output=primary, extra=extra,
                     stdout=proc.stdout, stderr=proc.stderr,
                     obs_names=_obs_names(method, variant, values, primary),
                     **provenance)


def preview(method: str, category: str, *, inputs: dict, out_dir, params=None,
            convert: bool = True, cmd_template: str | None = None,
            repo_path=None, gpu: bool | None = None) -> tuple[list[str], list[str]]:
    """The argv a run would execute, plus the setup notes to show with it.

    What ``run(..., dry_run=True)`` computes before it prints the notes to
    stderr; ``mtb.scan`` calls it for the ``command`` column and puts the
    notes in ``caveat``. Nothing is written, fetched or launched. ``gpu``
    is passed to :func:`cpu_params_for` (``True`` previews a GPU node's line).

    Returns
    -------
    tuple[list[str], list[str]]
        ``(argv, notes)``: the scripts-ref mismatch
        (``config.scripts_ref_problem``) first when there is one, then
        :func:`script_notes`, then the prepared-files note.
    """
    spec = registry.get(method)
    variant = spec.select(category, _modality_roles(inputs))
    params, _ = cpu_params_for(spec, params, gpu=gpu)
    # absolute paths as the run passes them, the input plan the run follows,
    # the checkout located but never fetched
    values, out_str = normalize_paths(inputs, out_dir)
    plan = _plan_inputs(variant, values, Path(out_str) / "inputs", convert=convert,
                        real=False)
    values = {role: step["value"] for role, step in plan.items()}
    repo = Path(repo_path) if repo_path else _repo_root_no_fetch()
    argv = _argv(variant, method, values, out_str, repo, params, cmd_template)
    # the scripts at another commit than $MULTIBENCH_SCRIPTS_REF: the real run
    # refuses (config.ensure_repo); mtb.scan reports it as the row's reason
    ref = config.scripts_ref_problem(repo)
    notes = ([ref] if ref else []) + script_notes(spec, variant, repo)
    prepared = _prepared_note(plan, out_str)
    return argv, notes + ([prepared] if prepared else [])


#: how :func:`_prepared_note` starts, so ``mtb.scan`` can pick it out of the notes
_PREPARED_PREFIX = "the command reads "


def _prepared_note(plan: dict, out_str: str) -> str | None:
    """The note for a command that reads files the run writes first, or ``None``.

    A converted input (``inputs/<role>.h5``) or a renamed peak file
    (``inputs/<role>_normpeaks.h5``) exists only after :func:`run` wrote it,
    so the shell line alone fails on a fresh ``out_dir``.
    """
    files = [os.path.relpath(step["value"], out_str) for step in plan.values()
             if step["convert"] or step["normpeaks_from"]]
    if not files:
        return None
    return f"{_PREPARED_PREFIX}{', '.join(files)}, " + config.hint(
        "which mtb.run writes first: start the method with mtb.run or mtb.run_all, "
        "not as a shell line",
        "which `multibench run` writes first: start the method with `multibench run` "
        "or `multibench run-all`, not as a shell line")


def _fetch_scripts(repo_path) -> Path:
    """``config.ensure_repo`` with an error that names the offline route.

    The first run on a machine clones the method scripts with ``git``; on a
    host without network that clone fails with git's own message only.
    """
    try:
        return Path(config.ensure_repo(repo_path))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        target = Path(repo_path) if repo_path else Path(config.DEFAULT.repo_path)
        raise RuntimeError(
            f"could not clone the method scripts (PYangLab/scMultiBench) into "
            f"{target}: {e}. On a host without network, fetch them first on a "
            f"connected machine with `{FETCH_SCRIPTS_CMD}` and copy the folder, or "
            f"point repo_path at an existing checkout") from e
