"""Build the exact argv for a method variant from resolved input values."""
from __future__ import annotations

from .schema import Variant

_INTERP = {"python": "python", "R": "Rscript"}


def build_command(variant: Variant, values: dict, out_dir: str,
                  params: dict | None = None) -> list[str]:
    """Return argv: [interpreter, entrypoint, *args, *params]."""
    cmd: list[str] = [_INTERP[variant.language], variant.entrypoint]
    vals = dict(values)
    vals["out_dir"] = out_dir

    for a in variant.args:
        if a.const is not None:
            items = [a.const]
        elif a.roles:
            # An entry written "=VALUE" is a LITERAL, not an input role. Upstream
            # scripts that take one slot per batch need a placeholder for batches
            # lacking that modality - scMoMaT documents `None` for exactly this -
            # and without it a mosaic layout cannot be expressed.
            missing = [r for r in a.roles
                       if not str(r).startswith("=") and r not in vals]
            if missing:
                raise KeyError(f"missing input roles {missing} for {variant.entrypoint}")
            items = [str(r)[1:] if str(r).startswith("=") else vals[r]
                     for r in a.roles]
        else:
            if a.role not in vals:
                raise KeyError(f"missing input role {a.role!r} for {variant.entrypoint}")
            val = vals[a.role]
            items = val if (a.repeat and isinstance(val, (list, tuple))) else [val]
        items = [str(x) for x in items]
        if a.is_positional:
            cmd.extend(items)
        elif a.eq:
            cmd.extend([f"{a.flag}={x}" for x in items])
        else:
            cmd.append(a.flag)
            cmd.extend(items)

    merged = {**variant.params, **(params or {})}
    for k, v in merged.items():
        if v is True:            # bare flag, e.g. --no_cuda
            cmd.append(f"--{k}")
        elif v is False or v is None:
            continue
        else:
            cmd.extend([f"--{k}", str(v)])
    return cmd
