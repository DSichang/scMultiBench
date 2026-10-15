"""Run a method variant: build cmd, wrap via cmd_template, exec in a workdir, load output."""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from .. import config
from . import builder, io, ingest, registry, envs
from .schema import _batch_of, base_modality, is_label_role


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
        ``'cpu'`` or ``'gpu'`` build of the method's environment, ``'single'``
        when it has one build for both, else ``'unknown'``.
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
    **Row order.** Rows follow the order
    ``mtb.labels_for(dataset, category, method)`` uses. ``obs_names`` is
    ``None``, with a ``UserWarning``, when the input barcodes do not match
    the output's cells or the inputs are a folder (scBridge).

    **Orientation.** When ``res.output.shape[0] != len(res.obs_names)`` the
    output is dims x cells: store ``res.output.T``. ``mtb.evaluate`` orients
    a raw array itself.

    **Output kind.** Not every method returns an embedding: Seurat_WNN
    writes only a neighbour graph, which ``mtb.run_all`` does not score.
    """

    method: str
    out_dir: Path
    cmd: list[str]
    output: object
    extra: dict
    stdout: str = ""
    stderr: str = ""
    obs_names: list | None = None
    scripts_commit: str | None = None
    env_flavor: str = "unknown"
    hostname: str = ""


def wrap_command(cmd: list[str], cmd_template: str | None) -> list[str]:
    """Wrap argv with a cmd_template like 'conda run -n env {cmd}'."""
    if not cmd_template or cmd_template.strip() == "{cmd}":
        return cmd
    prefix = cmd_template.replace("{cmd}", "").strip()
    return shlex.split(prefix) + cmd


#: The ``cmd_template`` placeholder for the command inside the method env.
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
        raise OSError(
            f"{RUN_MODE_VAR}=prefix but env {env!r} has no prefix under "
            f"envs_dir {config.DEFAULT.envs_dir} (and conda reports none) - "
            f"run mtb.env.install([<method>], dry_run=False) to unpack it, or "
            f"point MULTIBENCH_ENVS_DIR / mtb.config.DEFAULT.envs_dir at it")
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


# roles passed through as given: never converted, never select the variant
_AUX_ROLES = {"data_dir", "source_data", "target_data", "cty", "source_cty", "target_cty",
              "out_dir"}


def normalize_paths(inputs: dict, out_dir) -> tuple[dict, str]:
    """``(inputs, out_dir)`` with every path absolute; ``out_dir``, ``data_dir``
    and any existing directory get a trailing separator.

    The method runs in another folder, and many scripts concatenate a folder
    and a file name (``data_path + "rna.h5"``). ``os.path.abspath`` keeps
    symlinked data roots as written. Values that are not paths pass through.
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


def _is_modality_role(role: str) -> bool:
    """A role that selects the variant and is converted to a canonical ``.h5``."""
    return role not in _AUX_ROLES and not is_label_role(role)


def _modality_roles(inputs: dict) -> set[str]:
    return {k for k in inputs if _is_modality_role(k)}


#: input keys a variant renamed, ``{method: {old key: new key}}``
_RENAMED_ROLES = {"UnitedNet": {"rna_cty": "cty"}}


def _rename_old_roles(method: str, inputs: dict) -> dict:
    """``inputs`` with an old key of :data:`_RENAMED_ROLES` renamed, with a
    ``DeprecationWarning``; unchanged when the new key is there."""
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
    """The cell checks of ``inputs_for(check=True)`` on the input paths;
    raises their ``ValueError``."""
    from . import resolve
    paths = {r: v for r, v in inputs.items() if isinstance(v, (str, os.PathLike))}
    if not paths:
        return
    dataset = Path(next(iter(paths.values()))).parent.name or "inputs"
    resolve._check_same_cells(method, dataset, category, paths)
    if category == "diagonal":
        resolve._check_atac_gas_cells(method, dataset, paths)


def _repo_root_no_fetch() -> Path:
    """Where :func:`run` finds ``tools_scripts/``, in ``config.ensure_repo``'s
    order, without fetching; else the configured path, where a real run puts them."""
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
    """``"Methods run only on Linux, and this computer runs macOS."``, or
    ``None`` on Linux."""
    problem = envs.host_platform_problem()
    if not problem:
        return None
    return (f"Methods run only on Linux, and this computer runs "
            f"{envs._host_system_name(problem)}.")


def missing_script_fix(repo) -> str:
    """What to do about a method script missing from the checkout at ``repo``.

    A folder that is not a git checkout (often one made by hand for GLUE's
    annotation) blocks the fetch until it is moved away.
    """
    if config.scripts_commit(repo) is not None:
        return ("update it with git pull, or delete it so the next run fetches a "
                "fresh copy")
    return (f"that folder is not a git checkout. If you made it by hand, move out "
            f"the files you added, delete it, run {FETCH_SCRIPTS_CMD} and put the "
            f"files back; otherwise copy a full scripts checkout there")


#: How the note that no method scripts are here begins; ``mtb.scan`` finds it by this text.
SCRIPTS_NOT_HERE = "The method scripts are not in "


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
    import urllib.request

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
            size = None
            try:
                with urllib.request.urlopen(urllib.request.Request(item["url"], method="HEAD"),
                                            timeout=60) as r:
                    size = int(r.headers.get("Content-Length") or 0) or None
            except OSError:
                pass
            envs.download_archive([item["url"]], part,
                                  label=f"{item['file']} for {spec.id}", total=size)
            part.rename(dest)
        finally:
            part.unlink(missing_ok=True)


def script_notes(spec, variant, repo: Path) -> list[str]:
    """Setup facts a preview shows before the real run fails on them.

    The method's ``setup_hint`` first (unless its helper files are all in
    place), then its ``size_hint``, then a note for a script that is not on
    disk. Plain sentences: :func:`run` prints them on a dry run and
    ``mtb.scan`` adds them to ``caveat``.
    """
    notes = []
    entrypoint = Path(variant.entrypoint)
    helpers = variant.helpers
    done = bool(helpers) and all((repo / entrypoint).parent.joinpath(h).exists()
                                 or shipped_helper(spec.id, h) for h in helpers)
    if spec.setup_hint and not done:
        notes.append(spec.setup_hint)
    if variant.size_hint:
        notes.append(variant.size_hint)
    # a shipped script is copied in before the run
    shipped = entrypoint.name in helpers and shipped_helper(spec.id, entrypoint.name)
    if not (repo / entrypoint).exists():
        if (repo / "tools_scripts").is_dir() and shipped:
            pass
        elif (repo / "tools_scripts").is_dir():
            fix = missing_script_fix(repo)
            notes.append(f"{spec.id}'s script {entrypoint} is not in the checkout at {repo}. "
                         f"{fix[:1].upper()}{fix[1:]}.")
        elif not config.scripts_folder_problem(repo):      # else preview() names it
            notes.append(f"{SCRIPTS_NOT_HERE}{repo}. The first real run clones "
                         f"PYangLab/scMultiBench with git. On a host without network, "
                         f"run {FETCH_SCRIPTS_CMD} first.")
    return notes


def _convert_kw(role: str) -> dict:
    """The keywords the conversion of ``role`` passes ``ingest.to_canonical``.

    Only the ATAC roles pass a modality hint (for the peak-name warning):
    ``modality='adt'`` would refuse an AnnData that has obsm keys.
    """
    hint = role.rstrip("0123456789")
    return {"modality": hint} if hint in ("atac_gas", "atac_peak") else {}


def _h5ad_var(path):
    """``var`` of an ``.h5ad`` file alone (``read_h5ad`` loads every layer,
    even backed), or ``None``; the conversion then says what is wrong."""
    import h5py
    try:
        try:
            from anndata.io import read_elem
        except ImportError:                 # anndata < 0.11
            from anndata.experimental import read_elem
        with h5py.File(path, "r") as f:
            var = read_elem(f["var"])
    except (ImportError, OSError, KeyError):
        return None
    return var if hasattr(var, "columns") else None


def _check_feature_types(role: str, var) -> None:
    """The conversion's one-feature-type check, on the ``var`` table alone."""
    from types import SimpleNamespace
    ingest._check_one_feature_type(
        SimpleNamespace(var=var), arg=None, spec=None,
        role=_convert_kw(role).get("modality") or ingest._layout_family(f"{role}.h5"))


def _check_input(role: str, val, *, real: bool) -> str:
    """``"pass"`` for a canonical ``.h5``, else ``"convert"``; raises the error
    the conversion would raise, without reading the matrix.

    A missing path raises ``FileNotFoundError`` when ``real``; a preview may
    name the files of another host.
    """
    if not isinstance(val, (str, os.PathLike)):
        if ingest._is_mudata(val):
            mods = list(getattr(val, "mod", {}) or {})
            raise ValueError(
                f"inputs[{role!r}] is a MuData; pass one modality, e.g. "
                f"inputs={{{role!r}: mdata.mod[{(mods or [role])[0]!r}]}} (modalities: "
                f"{mods}), or write a dataset folder with mtb.io.export_dataset")
        if hasattr(val, "X") and hasattr(val, "obs"):
            _check_feature_types(role, getattr(val, "var", None))
            return "convert"
        raise TypeError(f"inputs[{role!r}] must be a file path or an AnnData, "
                        f"got {type(val).__name__}")
    path = Path(val)
    suffix = path.suffix.lower()
    if suffix == ".h5mu":
        raise ValueError(
            f"inputs[{role!r}] is a MuData file ({path.name}); pass one modality as an "
            f".h5ad or AnnData (mdata.mod[{role!r}]), or write a dataset folder with "
            f"mtb.io.export_dataset")
    if not path.exists():
        if real:
            raise FileNotFoundError(f"input file does not exist: {path} (cwd {os.getcwd()})")
        return "pass" if suffix == ".h5" else "convert"
    if suffix == ".h5":
        if ingest._is_canonical_h5(path):
            return "pass"
        ingest._to_anndata(path)          # raises the "no dataset 'matrix/data'" error
    if suffix == ".h5ad":
        _check_feature_types(role, _h5ad_var(path))
    if suffix in (".h5ad", ".csv", ".tsv"):
        return "convert"
    if suffix == ".loom":
        import importlib.util
        if importlib.util.find_spec("loompy") is None:
            raise ImportError(
                "reading .loom requires the optional 'loompy' package "
                "(pip install 'multibench-sc[loom]' or pip install loompy); "
                "alternatively convert the input to .h5ad/.csv first.")
        return "convert"
    raise ValueError(f"unsupported input format: {path.name}")


def _missing_input_notes(method: str, inputs: dict) -> list[str]:
    """One note per input path that does not exist, for ``run(dry_run=True)``."""
    scan = config.hint("mtb.scan", "multibench scan")
    return [f"{method} reads {val}, which does not exist. {scan} shows what the "
            f"folder holds."
            for val in inputs.values()
            if isinstance(val, (str, os.PathLike)) and not Path(val).exists()]


def _plan_inputs(variant, inputs: dict, inputs_dir: Path, *, convert: bool,
                 real: bool) -> dict:
    """``{role: {"value", "src", "convert", "normpeaks_from"}}``: the file each
    role hands the script and how it gets there, for the run and the dry run.

    A canonical ``.h5`` passes as is, anything else becomes
    ``inputs/<role>.h5``, a ``normalize_peaks`` role
    ``inputs/<role>_normpeaks.h5``. Nothing is written; the format checks run.
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
    for role in variant.normalize_peaks:
        if role in plan:
            plan[role]["normpeaks_from"] = plan[role]["value"]
            plan[role]["value"] = str(inputs_dir / f"{role}_normpeaks.h5")
    return plan


def _cell_group_roles(variant) -> list[list[str]]:
    """The input roles grouped by the cells they hold, in stacking order.

    Vertical: one group. Diagonal: one per base modality and batch number
    (``atac_peak`` and ``atac_gas`` share cells). Mosaic and cross: one per batch.
    """
    category = variant.when.get("category")
    groups: dict = {}
    for r in variant.stacked_roles():
        if category == "vertical":
            key = "cells"
        elif category == "diagonal":
            key = (base_modality(r), _batch_of(r))
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

    Each cell group gives the barcodes of its first readable canonical file;
    the list is kept when its length matches one axis of ``output``.
    """
    why = None
    names: list[str] = []
    # only roles the call was given hold cells (a const such as 'NULL' does not)
    groups = [[r for r in g if r in values] for g in _cell_group_roles(variant)]
    groups = [g for g in groups if g]
    if not groups:
        why = "the method reads a folder, not modality files"
    for roles in groups:
        barcodes = None
        for r in roles:
            v = values[r]
            if isinstance(v, (str, os.PathLike)) and str(v).endswith(".h5"):
                try:
                    barcodes = _read_barcodes(v)
                    break
                except (OSError, KeyError):
                    continue
        if barcodes is None:
            why = f"no barcodes readable for input role(s) {roles}"
            break
        names += barcodes
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
          params: dict | None, cmd_template: str | None) -> list[str]:
    """The exact argv a method run executes (shared by the run and the dry run).

    The variant's command, with its package-side ``driver`` in place of the
    script when it has one, the pseudo-tty wrap when declared, then the env
    wrap: ``cmd_template``, else the mode :func:`run_mode` picks.
    """
    cmd = builder.build_command(variant, values=values, out_dir=out_str, params=params)
    # a driver runs the unmodified script from its folder (--script_dir)
    if variant.driver:
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
        env_name = envs.group_for(method)
        mode, prefix = run_mode(env_name)
        if mode == "prefix":
            activate = (env_name, prefix)      # applied last, outside the pty wrap
        else:
            # CONDA_EXE: bare `conda` may not be on the subprocess's PATH
            conda = os.environ.get("CONDA_EXE", "conda")
            cmd_template = f"{conda} run -n {env_name} {{cmd}}"
    # `script` gives the method a pty (scJoint reads the terminal size) and
    # passes its exit code on (-e); it must be innermost, as conda run
    # redirects stdio
    if variant.pty:
        cmd = ["script", "-q", "-e", "-c",
               " ".join(shlex.quote(c) for c in cmd), "/dev/null"]
    if activate is not None:
        cmd = wrap_prefix(cmd, *activate)
    else:
        cmd = wrap_command(cmd, cmd_template)
    return wrap_command(cmd, outer) if outer is not None else cmd


def cpu_params_for(spec, params: dict | None, *,
                   gpu: bool | None = None) -> tuple[dict | None, dict]:
    """``(params, applied)`` after the CPU switch.

    Without a GPU (``gpu``, or :func:`envs.host_has_gpu` when ``None``) the
    method's ``cpu_params`` that the caller did not set are merged into
    ``params``; ``applied`` is that subset. ``params`` is returned unchanged
    when nothing applies.
    """
    if not spec.cpu_params or (envs.host_has_gpu() if gpu is None else gpu):
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
        params: dict | None = None, task: str | None = None, convert: bool = True,
        cmd_template: str | None = None, repo_path: Path | None = None,
        dry_run: bool = False):
    """Run one method on explicit inputs and load its output.

    Use it for full control over one run; ``mtb.run_all`` runs and scores
    every method of a category on a dataset.

    Parameters
    ----------
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.list_methods()``.
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
    ValueError
        ``MULTIBENCH_RUN_MODE`` is set to anything but ``conda`` or ``prefix``.
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
    **Dry run.** ``dry_run=True`` returns the argv the real run would execute
    (``shlex.join`` it for a shell line), from the same variant selection,
    input plan and env wrap. It creates nothing: no ``out_dir``, no
    ``inputs/`` copies, no env check, no fetch of the method scripts. It
    raises the same input errors and prints to stderr what the real run needs
    first: the method's ``setup_hint``, method scripts missing here or not at
    ``MULTIBENCH_SCRIPTS_REF``, files under ``inputs/`` the run writes first,
    ATAC files ``mtb.scan`` would block (the real run warns and still runs),
    and a note for an input path that does not exist.

    **Inputs.** Only ``category`` and the modality roles of ``inputs`` select
    the variant: every key except ``data_dir``, ``source_data``,
    ``target_data``, ``out_dir`` and the label keys (containing ``cty`` or
    ``label``), which are passed as given. A ``data_dir`` variant (scBridge)
    is selected by ``category`` alone. A canonical ``.h5`` passes through;
    an AnnData or any other file becomes ``<out_dir>/inputs/<role>.h5``, and
    a peak file the method needs renamed ``<out_dir>/inputs/<role>_normpeaks.h5``.
    A MuData raises ``ValueError``: pass one modality per role
    (``mdata.mod["rna"]``) or write the folder with ``mtb.io.export_dataset``.
    With ``convert=False`` every modality input must be a file path.
    Before anything runs, canonical ``.h5`` inputs get the cell checks of
    ``mtb.inputs_for(check=True)``: Seurat_v5's ``rna`` and ``atac_peak`` must
    hold the same cells, and a diagonal ``atac_gas`` file must list the cells
    of its ``atac_peak`` file (the one given, else the one next to it) in the
    same order.
    Relative paths are made absolute, because the method runs in ``out_dir``
    or in its script's folder.

    **GPU and CPU.** ``cpu_params`` are the flags that turn CUDA off in a
    script that uses it by default (``method_info(m)['cpu_params']``). On a
    host without an NVIDIA GPU (``mtb.env.host_has_gpu()`` is False) they are
    merged into ``params``; a key you pass wins, and the run prints them to
    stderr. A method whose script needs
    a GPU (``method_info(m)['requires_gpu']``) raises ``OSError`` there before
    anything is written; a dry run still returns the argv.

    **Environment.** With ``cmd_template=None`` the method runs in the env
    ``mtb.method_info(method)['env']`` names: through its prefix when
    ``mtb.env.env_prefix(env)`` finds it (no conda needed), else with
    ``conda run -n <env>``. ``MULTIBENCH_RUN_MODE=conda|prefix`` forces one.
    Method environments are Linux-only. Before anything is written, a missing
    env raises ``OSError`` naming the install command (on Linux only when
    some env is found at all); a ``{cmd}`` template skips this check. The
    method gets ``PYTHONNOUSERSITE=1``, and ``MPLBACKEND=Agg`` unless it sets
    its own backend; environment variables the method needs are set over yours.
    A variable made of absolute paths, such as ``LD_PRELOAD``, keeps only the
    paths that exist on this machine; when none does, it is left as you set
    it, or unset.

    **Launcher templates.** A ``{cmd}`` template must enter an env itself,
    for example ``"conda run -n myenv {cmd}"``. A Slurm job step:

    ```python
    mtb.run("scMoMaT", "mosaic", inputs=mtb.inputs_for("D46", "mosaic", "scMoMaT"),
            out_dir="runs/scMoMaT", cmd_template="srun --gres=gpu:1 {env_cmd}")
    ```

    Request a GPU only for a method that uses one;
    ``mtb.method_info(m)['gpu']`` says which.

    **Failures.** A non-zero exit raises ``RuntimeError`` with the tail of the
    method's stdout, then of its stderr. An interrupt, such as Ctrl-C or a
    ``run_all`` timeout, kills the method's whole process tree.

    **Method scripts.** The upstream scripts are never modified. The first
    run clones them with ``git``; on a host without network, run
    ``multibench fetch --scripts`` first.

    See Also
    --------
    mtb.inputs_for : builds the ``inputs`` dict from a laid-out dataset folder.

    mtb.run_all : every runnable method on a dataset, scored, with failures recorded.

    mtb.scan : previews the same command per method, with the file and env checks.

    mtb.evaluate : scores ``RunResult.output``.
    """
    _check_template(cmd_template)
    inputs = _rename_old_roles(method, inputs)
    _check_input_cells(method, category, inputs)
    from ..workflow import _atac_mismatch_caveats     # workflow imports this module
    mismatch = _atac_mismatch_caveats(method, category, inputs)
    if dry_run:
        argv, notes = preview(method, category, inputs=inputs, out_dir=out_dir,
                              params=params, convert=convert,
                              cmd_template=cmd_template, repo_path=repo_path)
        notes = _missing_input_notes(method, inputs) + notes
        for note in notes + mismatch:
            print(f"# {note}", file=sys.stderr, flush=True)
        return argv
    if mismatch:
        # one sentence per file: the peak-name caveat already ends with its fix
        warnings.warn(" ".join(m if m.endswith(".") else m + "." for m in mismatch),
                      UserWarning, stacklevel=2)

    spec = registry.get(method)
    variant = spec.select(category, _modality_roles(inputs))
    params, applied = cpu_params_for(spec, params)
    check_gpu_requirement(spec)
    if applied:
        print(f"[run] no GPU on this host: applying {method} cpu_params {applied}",
              file=sys.stderr, flush=True)

    # the env check of scan, before anything is written; on Linux only when
    # the probe finds some env (else the subprocess reports the failure)
    if cmd_template is None or ENV_CMD in cmd_template:
        env_name = envs.group_for(method)
        linux_only = linux_only_sentence()
        have = envs.installed_envs()
        if (have or linux_only) and env_name not in have:
            # conda's env list is cached per process: probe once more
            envs._conda_prefixes.cache_clear()
            have = envs.installed_envs()
        if (have or linux_only) and env_name not in have:
            if linux_only:
                this = config.hint("this call", "this command")
                preview_with = config.hint("dry_run=True", "--dry-run")
                raise EnvironmentError(
                    f"{linux_only} Run {this} on a Linux machine. {preview_with} "
                    f"previews the method's command here.")
            py = f" In Python, call mtb.env.install([{method!r}], dry_run=False)."
            raise EnvironmentError(
                f"Environment {env_name} of {method} is not installed. Run "
                f"multibench env install --methods {method} --packed --run."
                + config.hint(py, ""))

    inputs, out_str = normalize_paths(inputs, out_dir)
    out = Path(out_str)
    inputs_dir = out / "inputs"
    # every input-format check before anything is written or fetched
    plan = _plan_inputs(variant, inputs, inputs_dir, convert=convert, real=True)
    repo = _fetch_scripts(repo_path)
    provide_script_files(spec, variant, repo)
    out.mkdir(parents=True, exist_ok=True)
    if _modality_roles(inputs):
        inputs_dir.mkdir(parents=True, exist_ok=True)

    values: dict = {}
    for role, step in plan.items():
        if step["convert"]:
            ingest.to_canonical(step["src"], out=inputs_dir / f"{role}.h5",
                                **_convert_kw(role))
        if step["normpeaks_from"] is not None:
            ingest.normalize_peak_names(step["normpeaks_from"], step["value"])
        values[role] = step["value"]
    cmd = _argv(variant, method, values, out_str, repo, params, cmd_template)
    # a {cmd} template runs the method in the caller's env, whose build is not ours
    ours = cmd_template is None or ENV_CMD in cmd_template
    provenance = config.run_provenance(envs.group_for(method) if ours else None, repo)

    # user site-packages (~/.local) can shadow the env's packages
    run_env = {**os.environ, "PYTHONNOUSERSITE": "1", **_host_run_env(variant.run_env)}
    # a Jupyter kernel's MPLBACKEND (matplotlib_inline) does not exist in the env
    if "MPLBACKEND" not in (variant.run_env or {}):
        run_env["MPLBACKEND"] = "Agg"
    exec_cwd = str((repo / variant.entrypoint).parent) if variant.cwd_at_script else str(out)
    # the method may be a grandchild (under conda run): its own session lets
    # one killpg end the whole tree
    popen = subprocess.Popen(cmd, cwd=exec_cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=run_env,
                             start_new_session=True)
    try:
        stdout, stderr = popen.communicate()
    except BaseException:            # TimeoutError from the deadline included
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

    primary = io.load_output(out, variant.output)
    extra = {o.file: io.load_output(out, o) for o in variant.extra_outputs}
    return RunResult(method=method, out_dir=out, cmd=cmd, output=primary, extra=extra,
                     stdout=stdout, stderr=stderr,
                     obs_names=_obs_names(method, variant, values, primary),
                     **provenance)


def preview(method: str, category: str, *, inputs: dict, out_dir, params=None,
            convert: bool = True, cmd_template: str | None = None,
            repo_path=None, gpu: bool | None = None) -> tuple[list[str], list[str]]:
    """``(argv, notes)`` a run would execute and show, for ``run(dry_run=True)``
    and ``mtb.scan``; nothing is written, fetched or launched.

    The notes: a scripts checkout problem first, then :func:`script_notes`,
    then the prepared-files note. ``gpu`` goes to :func:`cpu_params_for`.
    """
    spec = registry.get(method)
    variant = spec.select(category, _modality_roles(inputs))
    params, _ = cpu_params_for(spec, params, gpu=gpu)
    values, out_str = normalize_paths(inputs, out_dir)
    plan = _plan_inputs(variant, values, Path(out_str) / "inputs", convert=convert,
                        real=False)
    values = {role: step["value"] for role, step in plan.items()}
    repo = Path(repo_path) if repo_path else _repo_root_no_fetch()
    argv = _argv(variant, method, values, out_str, repo, params, cmd_template)
    # the real run refuses these (config.ensure_repo)
    ref = config.scripts_ref_problem(repo) or config.scripts_folder_problem(repo)
    notes = ([ref] if ref else []) + script_notes(spec, variant, repo)
    prepared = _prepared_note(plan, out_str, spec.id)
    return argv, notes + ([prepared] if prepared else [])


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
    (converted inputs, renamed peak files), or ``None``."""
    files = [os.path.relpath(step["value"], out_str) for step in plan.values()
             if step["convert"] or step["normpeaks_from"]]
    if not files:
        return None
    that = "that file" if len(files) == 1 else "those files"
    return f"{method} reads {', '.join(files)}. " + config.hint(
        f"mtb.run writes {that} first, so start {method} with mtb.run or "
        f"mtb.run_all. The printed command alone fails in a job script.",
        f"multibench run writes {that} first, so start {method} with "
        f"multibench run or multibench run-all. The printed command alone "
        f"fails in a job script.")


def _fetch_scripts(repo_path) -> Path:
    """``config.ensure_repo``; a failed clone raises ``RuntimeError`` naming
    the offline route."""
    try:
        return Path(config.ensure_repo(repo_path))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        target = Path(repo_path) if repo_path else Path(config.DEFAULT.repo_path)
        raise RuntimeError(
            f"could not clone the method scripts (PYangLab/scMultiBench) into "
            f"{target}: {e}. On a host without network, fetch them first on a "
            f"connected machine with {FETCH_SCRIPTS_CMD} and copy the folder, or "
            f"point repo_path at an existing checkout") from e
