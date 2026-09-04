"""Goal 2, rigorous version: classify the ATAC file each method ACTUALLY receives.

Take 1 judged by role name and produced 5 false positives: it assumed `atac` and
`atac_gas` mean gene activity because describe_layout says so, but D12/D13's
atac.h5 holds PEAKS (`chrX_143482906_143483206`). Role names and file contents
disagree, so the only sound test is to resolve the real input path per method and
look at the feature names in it.
"""
import re
from collections import defaultdict

import h5py

import multibench as mtb
from multibench.engine import resolve as _resolve

NEEDS_GENE_ACTIVITY = {"Matilda", "scMDC", "scJoint", "Portal", "iNMF", "online_iNMF",
                       "SCALEX", "sciCAN", "VIPCCA", "Conos", "MultiMAP"}
NEEDS_PEAKS = {"MultiVI", "scMSI", "scMM", "StabMap", "VIMCCA", "moETM", "Multigrate",
               "MOFA2", "scMoMaT", "SMILE", "Seurat_WNN", "MIRA", "iPOLNG", "scMVP",
               "Cobolt", "GLUE", "Seurat_v5"}
MODE_DEPENDENT = {"Seurat_v3", "uniPort", "UINMF"}

PEAK_RE = re.compile(r"^(chr)?[0-9XYMT]+[:_-]\d+[-_]\d+", re.I)


def classify(path):
    try:
        with h5py.File(path, "r") as f:
            if "matrix/features" not in f:
                return "unknown(no features)"
            feats = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in f["matrix/features"][:60]]
    except Exception as e:
        return f"unreadable({type(e).__name__})"
    if not feats:
        return "unknown(empty)"
    hits = sum(bool(PEAK_RE.match(x)) for x in feats)
    return "peaks" if hits >= len(feats) * 0.5 else "gene_activity"


CATS = ["vertical", "diagonal", "mosaic", "cross"]
datasets = sorted(mtb.available_datasets())

seen = {}
for m in sorted(mtb.list_methods()):
    want = ("gene_activity" if m in NEEDS_GENE_ACTIVITY
            else "peaks" if m in NEEDS_PEAKS
            else "mode" if m in MODE_DEPENDENT else "?")
    if want == "?":
        continue
    for ds in datasets:
        for cat in CATS:
            try:
                sc = mtb.scan(ds, category=cat)
            except Exception:
                continue
            row = sc[(sc.method == m) & sc.runnable]
            if not len(row):
                continue
            try:
                inp = _resolve.inputs_for(ds, cat, m,
                                          modalities=(str(row.iloc[0]["modalities"]).split("+")
                                                      if row.iloc[0]["modalities"] != "(data_dir)" else None),
                                          check=True)
            except Exception:
                continue
            atac_paths = {r: p for r, p in inp.items()
                          if "atac" in r and str(p).endswith(".h5")}
            if not atac_paths:
                continue
            got = {r: classify(p) for r, p in atac_paths.items()}
            seen[m] = (want, ds, cat, got)
            break
        if m in seen:
            break

print(f"{'method':<13}{'upstream wants':<15}{'actually gets':<40}{'verdict':<12}where")
print("-" * 104)
bad = []
for m, (want, ds, cat, got) in sorted(seen.items()):
    kinds = set(got.values())
    if want == "mode":
        verdict = "n/a"
    elif kinds == {want}:
        verdict = "ok"
    elif "unknown" in " ".join(kinds) or "unreadable" in " ".join(kinds):
        verdict = "undetermined"
    else:
        verdict = "MISMATCH"
        bad.append((m, want, got, ds, cat))
    desc = ", ".join(f"{r}={k}" for r, k in sorted(got.items()))
    print(f"{m:<13}{want:<15}{desc:<40}{verdict:<12}{ds}/{cat}")

print(f"\n{len(bad)} MISMATCH out of {len(seen)} checked")
for m, want, got, ds, cat in bad:
    print(f"  {m}: upstream needs {want}, receives {got} (on {ds}/{cat})")
print("ATACV2_DONE")
