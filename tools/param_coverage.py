"""Goal 2b: which upstream hyperparameters can a user actually tune through our API?

The upstream audit recorded, per method, the knobs a benchmark user would
realistically want to change. This checks them against what params_for() exposes.

An upstream knob is only tunable here if the METHOD SCRIPT accepts it on the
command line - hard rule #1 forbids editing tools_scripts/ - so a low number is
often an upstream limitation rather than a wrapper gap. The point is to say
which, per method, instead of leaving it vague.
"""
import json
import re
import sys

import multibench as mtb
from multibench.engine import registry

UP = json.load(open("tools/upstream_params.json"))


def norm(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", "", s)           # drop "(train)" style qualifiers
    s = re.sub(r"^.*?:", "", s)             # drop "buildGraph:" prefixes
    s = re.sub(r"super_parameters\['(.*?)'\]", r"\1", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


rows = []
for m in sorted(mtb.list_methods()):
    spec = registry.get(m)
    ours = set()
    for v in spec.variants:
        ours |= set(getattr(v, "tunable", {}) or {})
        ours |= set(getattr(v, "params", {}) or {})
    up = UP.get(m, [])
    up_n = {norm(x): x for x in up if x}
    ours_n = {norm(x) for x in ours}
    matched = sorted(orig for n, orig in up_n.items()
                     if n and any(n == o or n in o or o in n for o in ours_n))
    rows.append((m, len(up), len(ours), len(matched), sorted(ours)[:6],
                 [u for n, u in up_n.items() if u not in matched][:5]))

print(f"{'method':<13}{'upstream':>9}{'ours':>6}{'matched':>8}   our knobs (first 6)")
print("-" * 104)
zero = []
for m, nu, no, nm, ours, miss in sorted(rows, key=lambda r: (r[2], r[0])):
    print(f"{m:<13}{nu:>9}{no:>6}{nm:>8}   {', '.join(ours) if ours else '(none)'}")
    if no == 0:
        zero.append(m)

print(f"\n{len(rows)} methods | {sum(1 for r in rows if r[2] > 0)} expose at least one "
      f"tunable parameter through our API | {len(zero)} expose none")
print("\nexpose NO tunable parameter (upstream hardcodes them; hard rule #1 forbids")
print("editing tools_scripts/, so these cannot be tuned without upstream changes):")
print("  " + ", ".join(zero))
tot_up = sum(r[1] for r in rows)
tot_m = sum(r[3] for r in rows)
print(f"\nupstream knobs recorded: {tot_up} | reachable through our API: {tot_m}")
print("PARAMCOV_DONE")
