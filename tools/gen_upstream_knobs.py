"""Generate multibench/engine/upstream_knobs.yaml from the audited source.

`params_for` reports what a method's script accepts on its COMMAND LINE, and
for most methods that is nothing: the benchmark scripts fix their
hyperparameters in the source, and hard rule #1 forbids editing them. Reporting
an empty `tunable` is honest but leaves the user with the wrong conclusion -
that the method has no hyperparameters at all. This file records the other two
halves of the truth, per method:

  fixed_in_script - the hyperparameter values the script pins, each with the
                    file:line that pins it, so a reader can check the claim
  upstream_knobs  - what the wrapped library documents, which is what the user
                    was expecting to see, marked as unreachable from the CLI

Regenerate with:  python tools/gen_upstream_knobs.py [path/to/scMultiBench]

Every fixed_in_script entry is re-verified against the checked-out upstream
script before it is written: an entry whose cited line no longer contains the
cited code is DROPPED, not silently carried forward, so upstream drift shows up
as missing facts rather than wrong ones.
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
# Why a method reports zero tunable parameters. `params_for` lists what the
# upstream script accepts on its command line; these are the hyperparameters it
# FIXES in its source (with the file:line that fixes them), plus the knobs the
# wrapped library documents but the script never exposes. Changing the latter
# requires editing tools_scripts/, which this package never does.
#
# Every fixed_in_script entry was verified against the upstream file at
# generation time.
"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def main(clone: pathlib.Path) -> int:
    audit = json.loads(AUDIT.read_text())
    out: dict[str, dict] = {}
    kept = dropped = 0
    for m in sorted(audit, key=lambda x: x["method"]):
        fixed = []
        for h in m.get("hardcoded", []):
            src = h.get("source", "")
            rel, _, ln = src.rpartition(":")
            f = clone / rel
            if not (f.is_file() and ln.isdigit()):
                dropped += 1
                continue
            lines = f.read_text(errors="ignore").splitlines()
            i = int(ln)
            actual = norm(lines[i - 1]) if 1 <= i <= len(lines) else ""
            ev = norm(h.get("evidence", ""))
            if not ev or not (ev in actual or actual in ev):
                dropped += 1
                continue
            kept += 1
            fixed.append({"name": h["name"], "value": str(h["value"]),
                          "source": src})
        knobs = [{"name": k["name"],
                  "default": str(k.get("library_default") or "(undocumented)"),
                  "effect": k["effect"]}
                 for k in (m.get("library_knobs") or [])]
        entry = {"upstream_url": m.get("upstream_url", ""),
                 "fixed_in_script": fixed,
                 "upstream_knobs": knobs}
        if m.get("notes"):
            entry["notes"] = " ".join(m["notes"].split())
        out[m["method"]] = entry

    OUT.write_text(HEADER + yaml.safe_dump(out, sort_keys=True,
                                           default_flow_style=False,
                                           allow_unicode=True, width=88))
    print(f"wrote {OUT.relative_to(HERE.parent)}: {len(out)} methods, "
          f"{kept} fixed settings verified, {dropped} dropped as unverifiable, "
          f"{sum(len(v['upstream_knobs']) for v in out.values())} upstream knobs")
    return 0


if __name__ == "__main__":
    default = pathlib.Path.home() / "scMultiBench"
    clone = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not (clone / "tools_scripts").is_dir():
        sys.exit(f"not a scMultiBench checkout: {clone}\n"
                 f"usage: python {sys.argv[0]} /path/to/scMultiBench")
    raise SystemExit(main(clone))
