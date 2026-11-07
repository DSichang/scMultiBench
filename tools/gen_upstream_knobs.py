"""Generate multibench/engine/upstream_knobs.yaml from upstream_knobs_audit.json.

`params_for` reports what a method's script accepts on its command line. This
file adds, per method, what the script does not take:

  fixed_in_script - the values the script still fixes, each with the file:line
                    that fixes it
  upstream_knobs  - settings the wrapped library documents and the script has
                    no parameter for

Regenerate with:  python tools/gen_upstream_knobs.py [folder holding tools_scripts]
(default: this repository).

The audit was written against the scripts of PYangLab/scMultiBench. Every
fixed_in_script entry is checked against the scripts given: an entry whose
code moved is cited at its new line, and an entry whose code is gone (the
value became a parameter) is dropped, as is a library setting that is now a
parameter. So a change shows up as a missing fact, not a wrong one. Each drop
is named on stderr.
"""
import json
import pathlib
import re
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
AUDIT = HERE / "upstream_knobs_audit.json"
OUT = HERE.parent / "multibench" / "engine" / "upstream_knobs.yaml"

HEADER = """\
# AUTO-GENERATED - do not hand-edit; regenerate with tools/gen_upstream_knobs.py
#
# What a method's script does not take as a parameter. `params_for` lists the
# parameters; these are the values the script still fixes in its source (with
# the file:line that fixes them), and the settings the wrapped library documents
# that the script has no parameter for.
#
# Every fixed_in_script entry was verified against the shipped script at
# generation time.
"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def is_verified(h: dict, clone: pathlib.Path) -> bool:
    """Whether the line `h["source"]` cites still holds the code `h["evidence"]` quotes.

    Evidence may be the line with a trailing comment cut, or the line plus its
    continuation, so containment is accepted either way round, but only when
    the shorter string is at least half the longer one. Without that bound a
    blank line or a lone ")" is contained in every evidence string, and a
    citation of the wrong file at the right line number would verify.
    """
    rel, _, ln = h.get("source", "").rpartition(":")
    f = clone / rel
    if not (f.is_file() and ln.isdigit()):
        return False
    lines = f.read_text(errors="ignore").splitlines()
    i = int(ln)
    if not 1 <= i <= len(lines):
        return False
    short, long_ = sorted((norm(lines[i - 1]), norm(h.get("evidence", ""))),
                          key=len)
    return bool(short) and short in long_ and 2 * len(short) >= len(long_)


def relocate(h: dict, clone: pathlib.Path) -> dict | None:
    """``h`` cited where its evidence is now, or ``None`` when the code is gone."""
    if is_verified(h, clone):
        return h
    rel = h.get("source", "").rpartition(":")[0]
    f = clone / rel
    if not f.is_file():
        return None
    for i in range(1, len(f.read_text(errors="ignore").splitlines()) + 1):
        moved = dict(h, source=f"{rel}:{i}")
        if is_verified(moved, clone):
            return moved
    return None


def knob_names(name: str) -> set:
    """The setting names a knob entry gives: ``"n_neighbors (calc_post_graph)"``
    names ``n_neighbors``, ``"seed / random_state"`` names both."""
    return {n.strip() for n in name.split(" (")[0].split(" / ")}


def tunable_names() -> dict:
    """``{method: {parameter names of any of its variants}}`` from the registry."""
    sys.path.insert(0, str(HERE.parent))
    from multibench.engine import registry
    return {s.id: {k for v in s.variants for k in v.tunable} for s in registry.load()}


def main(clone: pathlib.Path) -> int:
    audit = json.loads(AUDIT.read_text())
    params = tunable_names()
    out: dict[str, dict] = {}
    kept = dropped = 0
    for m in sorted(audit, key=lambda x: x["method"]):
        fixed = []
        for cited in m.get("hardcoded", []):
            h = relocate(cited, clone)
            if h is None:
                dropped += 1
                print(f"dropped {m['method']}: {cited.get('name')} "
                      f"@ {cited.get('source')}", file=sys.stderr)
                continue
            kept += 1
            fixed.append({"name": h["name"], "value": str(h["value"]),
                          "source": h["source"]})
        knobs = [{"name": k["name"],
                  "default": str(k.get("library_default") or "(undocumented)"),
                  "effect": k["effect"]}
                 for k in (m.get("library_knobs") or [])
                 if not knob_names(k["name"]) & params.get(m["method"], set())]
        entry = {"upstream_url": m.get("upstream_url", ""),
                 "fixed_in_script": fixed,
                 "upstream_knobs": knobs}
        out[m["method"]] = entry

    OUT.write_text(HEADER + yaml.safe_dump(out, sort_keys=True,
                                           default_flow_style=False,
                                           allow_unicode=True, width=88))
    print(f"wrote {OUT.relative_to(HERE.parent)}: {len(out)} methods, "
          f"{kept} fixed settings verified, {dropped} dropped (gone or now a parameter), "
          f"{sum(len(v['upstream_knobs']) for v in out.values())} upstream knobs")
    return 0


if __name__ == "__main__":
    default = HERE.parent
    clone = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not (clone / "tools_scripts").is_dir():
        sys.exit(f"no tools_scripts folder in {clone}\n"
                 f"usage: python {sys.argv[0]} [folder holding tools_scripts]")
    raise SystemExit(main(clone))
