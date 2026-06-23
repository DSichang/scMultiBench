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
_AUX_ROLES = {"data_dir", "source_data", "target_data", "source_cty", "target_cty",
              "out_dir"}


def run(method: str, category: str, task: str = "clustering", *, inputs: dict,
        out_dir: str, params: dict | None = None, convert: bool = True,
        cmd_template: str | None = None, repo_path: Path | None = None) -> RunResult:
    """Run a method and load its output. Method scripts are never modified.

    The ``task`` parameter is reserved for future per-task dispatch; variant
    selection in v1 depends only on ``category`` and the supplied modalities.

    Note on auxiliary-role coupling: for methods whose args reference auxiliary
    roles (e.g. scBridge's ``data_dir``/``source_data``/``target_data``/
    ``source_cty``/``target_cty``), the caller must ALSO pass the modality roles
    used for variant selection (e.g. ``rna``, ``atac_gas``). Aux-role inputs are
    passed through verbatim and are NOT converted to the canonical .h5.
    """
    spec = registry.get(method)
    modalities = {k for k in inputs if k not in _AUX_ROLES}
    variant = spec.select(category, modalities)

    out = Path(out_dir)
    workdir = out
    workdir.mkdir(parents=True, exist_ok=True)
    inputs_dir = workdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # normalize modality inputs to canonical .h5 inside a dedicated inputs dir
    values: dict = {}
    for role, val in inputs.items():
        if convert and role not in _AUX_ROLES:
            values[role] = str(ingest.to_canonical(val, out=inputs_dir / f"{role}.h5"))
        else:
            values[role] = val

    repo = Path(repo_path) if repo_path else config.DEFAULT.repo_path
    cmd = builder.build_command(variant, values=values, out_dir=str(out), params=params)
    # entrypoint is relative to the reference repo
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
        env_name = envs.group_for(method)
        cmd_template = f"{conda} run -n {env_name} {{cmd}}"
    cmd = wrap_command(cmd, cmd_template)

    # Isolate the method env from user site-packages (~/.local): a broken or
    # mismatched ~/.local can shadow the conda env (e.g. a libcublas-less torch
    # egg breaking anndata imports). PYTHONNOUSERSITE=1 makes the env hermetic.
    run_env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True,
                          env=run_env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{method} failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-2000:]}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}"
        )

    primary = io.load_output(out, variant.output)
    extra = {o.file: io.load_output(out, o) for o in variant.extra_outputs}
    return RunResult(method=method, out_dir=out, cmd=cmd, output=primary, extra=extra,
                     stdout=proc.stdout, stderr=proc.stderr)
