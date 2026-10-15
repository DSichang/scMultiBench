"""Build the exact argv for a method variant from resolved input values."""
from __future__ import annotations

from .schema import Variant

_INTERPRETER = {"python": "python", "R": "Rscript"}


def build_command(variant: Variant, values: dict, out_dir: str,
                  params: dict | None = None) -> list[str]:
    """Return argv: [interpreter, entrypoint, *args, *params]."""
    cmd: list[str] = [_INTERPRETER[variant.language], variant.entrypoint]
    by_role = dict(values)
    by_role["out_dir"] = out_dir

    for arg in variant.args:
        if arg.const is not None:
            items = [arg.const]
        elif arg.roles:
            # "=VALUE" is a literal: a placeholder for a batch that lacks the
            # modality (scMoMaT takes `None` there)
            missing = [r for r in arg.roles
                       if not str(r).startswith("=") and r not in by_role]
            if missing:
                raise KeyError(f"missing input roles {missing} for {variant.entrypoint}")
            items = [str(r)[1:] if str(r).startswith("=") else by_role[r]
                     for r in arg.roles]
        else:
            if arg.role not in by_role:
                raise KeyError(f"missing input role {arg.role!r} for {variant.entrypoint}")
            val = by_role[arg.role]
            items = val if (arg.repeat and isinstance(val, (list, tuple))) else [val]
        items = [str(x) for x in items]
        if arg.is_positional:
            cmd.extend(items)
        elif arg.eq:
            cmd.extend([f"{arg.flag}={x}" for x in items])
        else:
            cmd.append(arg.flag)
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
