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
        cmd.extend([f"--{k}", str(v)])
    return cmd
