"""Thin CLI over the multibench Python API."""
from __future__ import annotations

import argparse

from . import discover, load_results
from . import plot as plot_ns


def _cmd_list(args) -> int:
    from .engine import registry
    for m in registry.list_methods(category=args.category, task=args.task,
                                   runnable=args.runnable or None):
        print(m)
    return 0


def _cmd_find(args) -> int:
    needs = True if args.needs_labels else None
    for m in discover.find_methods(category=args.category, task=args.task,
                                   needs_labels=needs, atac=args.atac,
                                   runnable=args.runnable or None):
        print(m)
    return 0


def _cmd_plot(args) -> int:
    if args.kind != "bubble":
        raise SystemExit(f"unknown plot kind {args.kind!r}")
    df = load_results(category=args.category, dataset=args.dataset)
    metrics = args.metrics.split(",") if args.metrics else None
    plot_ns.bubble(df, metrics=metrics, aggregate=args.aggregate, save=args.out)
    print(f"wrote {args.out}")
    return 0


def _parse_inputs(pairs) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--input must be role=path, got {p!r}")
        role, path = p.split("=", 1)
        if role.strip() == "" or path == "":
            raise SystemExit(f"--input must be role=path with non-empty parts, got {p!r}")
        out[role] = path
    return out


def _cmd_run(args) -> int:
    import multibench
    res = multibench.run(method=args.method, category=args.category, task=args.task,
                       inputs=_parse_inputs(args.input), out_dir=args.out,
                       cmd_template=args.runner)
    print(f"ran {args.method} -> {res.out_dir}")
    return 0


def _cmd_evaluate(args) -> int:
    import multibench
    df = multibench.evaluate(output=args.output, category=args.category, task=args.task,
                           labels=args.labels, clustering=args.cluster)
    if args.out:
        df.to_csv(args.out)
        print(f"wrote {args.out}")
    else:
        print(df.to_string())
    return 0


def _cmd_env(args) -> int:
    from .engine import envs
    cmd = args.env_cmd
    if cmd == "status":
        for r in envs.status():
            mark = "x" if r["exists"] else " "
            tag = r["difficulty"] + ("*" if r["verified_working"] else "")
            print(f"[{mark}] {r['method']:16} {r['group']:16} {tag}")
        return 0
    if cmd == "groups":
        for name, spec in envs.groups().items():
            if spec.get("shared"):
                print(f"{name:16} ({len(spec['members']):2}): {', '.join(spec['members'])}")
        return 0
    if cmd == "plan":
        for p in envs.plan(category=getattr(args, "category", None)):
            tag = "shared" if p["shared"] else "own"
            print(f"{p['env']:16} [{tag:6}] <- {', '.join(p['methods'])}")
        return 0
    if cmd == "doctor":
        rows = envs.doctor(category=getattr(args, "category", None))
        for r in rows:
            mark = "x" if r["exists"] else ("L" if r["has_lock"] else "!")
            print(f"[{mark}] {r['env']:18} ({len(r['methods']):2}) <- {', '.join(r['methods'])}")
        missing = [r for r in rows if not r["exists"]]
        nolock = [r["env"] for r in missing if not r["has_lock"]]
        print(f"# {len(rows)} envs needed, {len(missing)} missing"
              + (f"; NO lockfile for: {', '.join(nolock)}" if nolock else ""))
        print("# legend: [x]=installed  [L]=missing, lockfile ready (run `multibench env install --run`)  [!]=missing, no lockfile")
        return 0
    if cmd == "install":
        rows = envs.create_all(category=getattr(args, "category", None),
                               dry_run=not getattr(args, "run", False))
        for r in rows:
            state = ("have" if r["exists"]
                     else ("BUILD" if r["has_lock"] and getattr(args, "run", False)
                           else "build(dry-run)" if r["has_lock"] else "NO-LOCK"))
            print(f"{r['env']:18} [{state:14}] <- {', '.join(r['methods'])}")
        if not getattr(args, "run", False):
            print("# dry-run — add --run to create the missing envs from their lockfiles")
        return 0
    if cmd == "freeze":
        if getattr(args, "all", False):
            for env in envs.required_envs(category=getattr(args, "category", None)):
                try:
                    print(f"froze {env} -> {envs.freeze(env)}")
                except Exception as e:  # noqa: BLE001 - report per-env, keep going
                    print(f"SKIP {env}: {str(e)[:120]}")
        else:
            print(f"froze {args.env} -> {envs.freeze(args.env)}")
        return 0
    if cmd == "create-group":
        cmds = envs.create_group(args.group, dry_run=not getattr(args, "run", False))
        if not getattr(args, "run", False):
            print("# dry-run — add --run to execute:")
            for c in cmds:
                print(" ".join(c))
        else:
            print(f"created group env {args.group}")
        return 0
    method = getattr(args, "method", None)
    name = getattr(args, "name", None)
    if cmd == "recipe":
        for c in envs.create_commands(method, env_name=name):
            print(" ".join(c))
        return 0
    if cmd == "yml":
        y = envs.environment_yml(method, env_name=name)
        out = getattr(args, "out", None)
        if out:
            with open(out, "w") as f:
                f.write(y)
            print(f"wrote {out}")
        else:
            print(y, end="")
        return 0
    if cmd == "create":
        cmds = envs.create(method, env_name=name, dry_run=not getattr(args, "run", False))
        if not getattr(args, "run", False):
            print("# dry-run — add --run to execute:")
            for c in cmds:
                print(" ".join(c))
        else:
            print(f"created environment for {method}")
        return 0
    raise SystemExit(
        "usage: multibench env {status|groups|plan|doctor|install|freeze|recipe|yml|create|create-group}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multibench")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list"); pl.add_argument("--category"); pl.add_argument("--task")
    pl.add_argument("--runnable", action="store_true",
                    help="only methods with a declared variant (usable by run)")
    pl.set_defaults(func=_cmd_list)

    pf = sub.add_parser("find")
    pf.add_argument("--category"); pf.add_argument("--task")
    pf.add_argument("--needs-labels", action="store_true")
    pf.add_argument("--atac", choices=["peak", "gene_activity"],
                    help="filter by ATAC representation the method consumes")
    pf.add_argument("--runnable", action="store_true",
                    help="only methods with a declared variant (usable by run)")
    pf.set_defaults(func=_cmd_find)

    pp = sub.add_parser("plot")
    pp.add_argument("kind")
    pp.add_argument("--category", required=True); pp.add_argument("--dataset")
    pp.add_argument("--metrics"); pp.add_argument("--aggregate", default="dataset")
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=_cmd_plot)

    pr = sub.add_parser("run")
    pr.add_argument("--method", required=True); pr.add_argument("--category", required=True)
    pr.add_argument("--task", default="clustering")
    pr.add_argument("--input", action="append", help="role=path (repeatable)")
    pr.add_argument("--out", required=True); pr.add_argument("--runner")
    pr.set_defaults(func=_cmd_run)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--output", required=True)
    pe.add_argument("--category", help="reserved; not used by v1 metrics")
    pe.add_argument("--task", default="clustering")
    pe.add_argument("--labels"); pe.add_argument("--cluster"); pe.add_argument("--out")
    pe.set_defaults(func=_cmd_evaluate)

    pv = sub.add_parser("env", help="per-method environment recipes")
    ev = pv.add_subparsers(dest="env_cmd", required=True)
    ev.add_parser("status").set_defaults(func=_cmd_env)
    ev.add_parser("groups").set_defaults(func=_cmd_env)
    er = ev.add_parser("recipe"); er.add_argument("method"); er.add_argument("--name"); er.set_defaults(func=_cmd_env)
    ey = ev.add_parser("yml"); ey.add_argument("method"); ey.add_argument("--name"); ey.add_argument("--out"); ey.set_defaults(func=_cmd_env)
    ec = ev.add_parser("create"); ec.add_argument("method"); ec.add_argument("--name"); ec.add_argument("--run", action="store_true"); ec.set_defaults(func=_cmd_env)
    ep = ev.add_parser("plan"); ep.add_argument("--category"); ep.set_defaults(func=_cmd_env)
    eg = ev.add_parser("create-group"); eg.add_argument("group"); eg.add_argument("--run", action="store_true"); eg.set_defaults(func=_cmd_env)
    edoc = ev.add_parser("doctor", help="preflight: which envs are present / need building"); edoc.add_argument("--category"); edoc.set_defaults(func=_cmd_env)
    ei = ev.add_parser("install", help="build every needed env from its lockfile"); ei.add_argument("--category"); ei.add_argument("--run", action="store_true"); ei.set_defaults(func=_cmd_env)
    ef = ev.add_parser("freeze", help="capture an env (or --all) to a committed lockfile"); ef.add_argument("env", nargs="?"); ef.add_argument("--all", action="store_true"); ef.add_argument("--category"); ef.set_defaults(func=_cmd_env)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
