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
        cmd.extend(_param_tokens(k, v, (variant.tunable.get(k) or {})))
    return cmd


def _param_tokens(key: str, value, decl: dict) -> list[str]:
    """The tokens of one param, by the type the script declares: a
    ``type=bool`` argument gets ``True`` or ``''`` (argparse's ``bool('')``
    is ``False``); a switch (``flag``) is present when the value differs
    from the one the script gets without it. Otherwise ``True`` is a bare
    flag, ``False``/``None`` nothing, anything else its text."""
    kind = decl.get("type")
    if kind == "bool" and isinstance(value, bool):
        return [f"--{key}", "True" if value else ""]
    if kind == "flag" and isinstance(value, bool):
        return [f"--{key}"] if value != bool(decl.get("default")) else []
    if value is True:
        return [f"--{key}"]
    if value is False or value is None:
        return []
    return [f"--{key}", str(value)]
