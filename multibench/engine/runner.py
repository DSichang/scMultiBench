"""Run a method variant: build cmd, wrap via cmd_template, exec in a workdir, load output."""
from __future__ import annotations

import os
import shlex
import subprocess
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
    output : the primary output ALREADY LOADED. For an embedding method this is a
        numpy array; pass it straight to :func:`multibench.evaluate`. Note it may be
        dims x cells rather than cells x dims - ``evaluate`` re-orients a raw array
        against the label count for you.
    extra : ``{filename: loaded object}`` for any ``extra_outputs`` the variant
        declares (e.g. scMoMaT's UMAP embedding alongside its KNN graph).
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


# Auxiliary roles are passed through verbatim (never converted to canonical .h5)
# and are excluded from the modality set used for variant selection.
_AUX_ROLES = {"data_dir", "source_data", "target_data", "cty", "source_cty", "target_cty",
              "out_dir"}


def run(method: str, category: str, task: str = "clustering", *, inputs: dict,
        out_dir: str, params: dict | None = None, convert: bool = True,
        cmd_template: str | None = None, repo_path: Path | None = None) -> RunResult:
    """Run a method and load its output. Method scripts are never modified.

    The ``task`` parameter is reserved for future per-task dispatch; variant
    selection in v1 depends only on ``category`` and the supplied modalities.

    Parameters
    ----------
    method : registry id (``KeyError`` with a did-you-mean hint otherwise).
    category : integration category of the variant to run.
    task : reserved (see above).
    inputs : ``{role: path-or-AnnData}``; the non-auxiliary, non-label roles
        select the variant. See ``inputs_for`` / ``method_info(m)['supports']``.
    out_dir : directory the method writes into (created; ``inputs/`` holds the
        canonical .h5 copies when ``convert=True``).
    params : overrides merged over the variant's default hyperparameters.
    convert : convert modality inputs to the canonical .h5 layout (default True).
    cmd_template : wrapper for the argv, e.g. ``"conda run -n myenv {cmd}"``.
        Default: ``conda run -n <env>`` with the env ``mtb.env.group_for(method)``
        would provision. When left as None the env is PREFLIGHTED: if conda lists
        envs and the method's env is not among them, ``EnvironmentError`` is
        raised before any file is written, naming the install command. Pass a
        cmd_template to take over env control (no preflight).
    repo_path : checkout holding ``tools_scripts/`` (default: auto-provisioned).

    Note on auxiliary-role coupling: for methods whose args reference auxiliary
    roles (e.g. scBridge's ``data_dir``/``source_data``/``target_data``/
    ``source_cty``/``target_cty``), the caller must ALSO pass the modality roles
    used for variant selection (e.g. ``rna``, ``atac_gas``). Aux-role inputs are
    passed through verbatim and are NOT converted to the canonical .h5.
    """
    spec = registry.get(method)
    # Label roles (anything containing "cty"/"label", e.g. rna_cty/atac_cty)
    # are auxiliary too: they are method inputs but not modalities for variant
    # selection (consistent with resolve.py treating them as .csv labels).
    modalities = {k for k in inputs
                  if k not in _AUX_ROLES and "cty" not in k and "label" not in k}
    variant = spec.select(category, modalities)

    # Env preflight: the default cmd_template is `conda run -n <env> ...`, and
    # conda's own "EnvironmentLocationNotFound" only arrives AFTER inputs were
    # converted and the subprocess spawned, buried in a stderr tail with no
    # install hint. Check up front (same probe scan() uses), before anything is
    # written. Skipped when the caller controls the env via cmd_template, and
    # when the probe returns nothing (conda absent/broken -> cannot evidence,
    # let the subprocess report as before).
    env_name = None
    if cmd_template is None:
        env_name = envs.group_for(method)
        have = envs.installed_envs()
        if have and env_name not in have:
            raise EnvironmentError(
                f"conda env {env_name!r} ({method}) is not installed - run "
                f"`multibench env install --methods {method} --packed --run` "
                f"(or mtb.env.create_env({env_name!r})); see mtb.env.doctor()")

    out = Path(out_dir)
    workdir = out
    workdir.mkdir(parents=True, exist_ok=True)
    inputs_dir = workdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # normalize modality inputs to canonical .h5 inside a dedicated inputs dir
    values: dict = {}
    for role, val in inputs.items():
        if convert and role not in _AUX_ROLES and "cty" not in role and "label" not in role:
            values[role] = str(ingest.to_canonical(val, out=inputs_dir / f"{role}.h5"))
        else:
            values[role] = val

    # Some methods (Seurat_v3 etc.) require ATAC peak ids in chr:start-end form;
    # normalize the declared peak roles into per-run copies (originals untouched).
    for role in (getattr(variant, "normalize_peaks", None) or []):
        if role in values:
            npath = inputs_dir / f"{role}_normpeaks.h5"
            values[role] = str(ingest.normalize_peak_names(values[role], npath))
    repo = Path(config.ensure_repo(repo_path))
    # Pass out_dir with a trailing separator: many method scripts build their
    # output path by string-concatenation (R paste0(save_path,"embedding.h5"),
    # etc.), so a missing slash writes a SIBLING file instead of into out_dir.
    cmd = builder.build_command(variant, values=values,
                                out_dir=os.path.join(str(out), ""), params=params)
    # entrypoint is relative to the reference repo. A variant may declare a
    # package-side `driver`: a wrapper script (shipped with the package) that
    # source()s the UNMODIFIED upstream entrypoint and calls its function. When
    # set, run the driver instead and hand it the upstream script's dir via
    # --script_dir (so the driver can source it in place; the method script
    # stays byte-identical to upstream).
    if getattr(variant, "driver", None):
        pkg_root = Path(__file__).resolve().parents[1]      # .../multibench
        driver_abs = pkg_root / variant.driver
        script_dir = (repo / variant.entrypoint).parent
        cmd = [cmd[0], str(driver_abs), "--script_dir", str(script_dir)] + cmd[2:]
    else:
        cmd[1] = str(repo / cmd[1])
    if cmd_template is None:
        # Resolve conda's full path when available (CONDA_EXE is set by an
        # initialized conda) so the default works even when bare `conda` is not
        # on the spawned subprocess's PATH; fall back to `conda`.
        conda = os.environ.get("CONDA_EXE", "conda")
        # Resolve the env via the same group system that provisioning builds
        # (mtb.env.plan/create/create_group), so "the env you provision is the
        # env you run". group_for() returns a method's shared group env, or its
        # own scmb_<method> env if it is not grouped. (The legacy per-method
        # methods.yaml `env:` field is no longer consulted here.)
        env_name = env_name or envs.group_for(method)
        cmd_template = f"{conda} run -n {env_name} {{cmd}}"
    # Opt-in pseudo-tty: some upstream scripts read the terminal size
    # (os.popen('stty size')) to draw a progress bar and crash without a tty
    # (scJoint's util/utils.py). Wrap the method command in `script`, which
    # allocates a pty, forwards the child's output to captured stdout, and (-e)
    # propagates the child's exit code so the returncode check below still fires.
    # This must be INNERMOST -- inside the conda-run wrap -- so the method
    # process's own stdin is the pty; conda run otherwise redirects stdio and
    # `stty size` still fails. Hence wrap BEFORE wrap_command. Default off ->
    # no other method affected.
    if getattr(variant, "pty", False):
        cmd = ["script", "-q", "-e", "-c",
               " ".join(shlex.quote(c) for c in cmd), "/dev/null"]
    cmd = wrap_command(cmd, cmd_template)

    # Isolate the method env from user site-packages (~/.local): a broken or
    # mismatched ~/.local can shadow the conda env (e.g. a libcublas-less torch
    # egg breaking anndata imports). PYTHONNOUSERSITE=1 makes the env hermetic.
    run_env = {**os.environ, "PYTHONNOUSERSITE": "1", **{k: str(v) for k, v in (variant.run_env or {}).items()}}
    # A Jupyter kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline,
    # which leaks through `conda run` into the METHOD's env, where matplotlib_inline
    # does not exist - so any method that imports matplotlib dies at import when
    # run_all is called from a notebook (Portal was the first to hit it). Agg is
    # the safe headless backend for a subprocess that at most saves figures.
    if "MPLBACKEND" not in (variant.run_env or {}):
        run_env["MPLBACKEND"] = "Agg"
    # Some scripts source/import local files relative to the entrypoint dir,
    # so let variants opt into running with cwd=script's parent rather than
    # cwd=out_dir. The out_dir is still passed via the --save_path arg, so
    # outputs land in the correct place regardless.
    exec_cwd = str((repo / variant.entrypoint).parent) if variant.cwd_at_script else str(workdir)
    # The child is `conda run` and the actual method is its GRANDchild;
    # killing only the direct child on a timeout left the method computing
    # for hours. Own session -> one killpg reaps the whole tree.
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
            # stdout FIRST, stderr LAST. Callers truncate this message from the
            # left (a traceback's payload is its tail), so whatever is most
            # diagnostic has to sit at the end - and that is stderr. Ordering it
            # the other way meant a left-truncated message kept conda's
            # "see above for error" summary and discarded the error it referred to.
            f"stdout tail:\n{proc.stdout[-1500:]}\n"
            f"stderr tail:\n{proc.stderr[-2500:]}"
        )

    primary = io.load_output(out, variant.output)
    extra = {o.file: io.load_output(out, o) for o in variant.extra_outputs}
    return RunResult(method=method, out_dir=out, cmd=cmd, output=primary, extra=extra,
                     stdout=proc.stdout, stderr=proc.stderr)
