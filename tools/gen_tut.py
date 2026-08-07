"""Generate the four integration tutorials (vertical / diagonal / mosaic / cross).

Style reference: the M3 / Matilda docs notebooks - short purposeful code cells,
markdown narration between them, tables for structured facts, an Install section
up front, and a RUNNABLE "your own dataset" walkthrough rather than prose.

Every numeric claim in the text (runtimes, coverage counts, install cost) is a
measured value from the verification runs, not an estimate.
"""
import nbformat as nbf
import os

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)

# Method sets benchmarked per category in the paper (Nature Methods 22:2449-2460
# and the PYangLab/scMultiBench README), so each tutorial states its own
# coverage instead of letting the reader assume parity.
PAPER_METHODS = {
 "vertical": ["totalVI","sciPENN","Concerto","scMSI","Matilda","MOFA2","Multigrate",
              "UINMF","scMoMaT","Seurat_WNN","scMM","scMDC","moETM","VIMCCA",
              "iPOLNG","MIRA","UnitedNet","scMVP"],
 "diagonal": ["scBridge","Portal","SCALEX","VIPCCA","Seurat_v3","MultiMAP","Seurat_v5",
              "sciCAN","Conos","iNMF","online_iNMF","scJoint","GLUE","uniPort"],
 "mosaic":   ["MultiVI","scMoMaT","StabMap","Cobolt","UINMF","Multigrate","SMILE",
              "scMM","moETM","UnitedNet","totalVI","sciPENN"],
 "cross":    ["totalVI","scMoMaT","UnitedNet","sciPENN","Concerto","scMDC","StabMap",
              "UINMF","scMM","MOFA2","Multigrate","PASTE","PASTE2","SPIRAL","GPSA"],
}
MOSAIC_IMPUTATION_ONLY = ["scMM","moETM","UnitedNet","totalVI","sciPENN"]

SCEN = {
 "vertical": dict(
   ds="D11", cells="2,864",
   blurb=("**Vertical integration** fuses several modalities measured **in the same "
          "cells** (here CITE-seq: RNA + surface protein). Cells are already matched, "
          "so the task is combining modalities, not aligning cells."),
   live=("Matilda", '{"epochs": 5}', "Matilda takes well under a minute at 5 epochs"),
   live_ds=None,
   own_src="D11", own_fast="Matilda",
   own_note="Matilda scored ARI 0.95 on this subsample when we ran it - your exact "
            "number will differ slightly, the point is that it is high and computed "
            "end-to-end on data the package has never seen.",
 ),
 "diagonal": dict(
   ds="D28", cells="6,408 RNA + 4,606 ATAC",
   blurb=("**Diagonal integration** is the hard case: RNA and ATAC come from "
          "**different cells**, with no pairing and no shared cell ids. A method must "
          "align two populations using only a shared feature space (for RNA + ATAC, "
          "gene-activity scores)."),
   live=("online_iNMF", "None", "online_iNMF finishes in about 80 s"),
   live_ds=None,
   own_src="D28", own_fast="Portal",
   own_note="Portal reached CHAIN_OK with all nine metrics on this subsample in "
            "about 40 s when we ran it.",
 ),
 "mosaic": dict(
   ds="D45", cells="32,151",
   blurb=("**Mosaic integration** has several batches where only **some** share a "
          "modality - an RNA-only batch, an ATAC-only batch, and a paired batch that "
          "bridges them. Mosaic methods are the most layout-sensitive of the four "
          "categories: each supports specific per-batch modality patterns, so "
          "`scan()` per dataset is the source of truth for what applies."),
   live=("StabMap", "None", "StabMap finishes in about 75 s on D46"),
   live_ds="D46",
   own_src="D46", own_fast="StabMap",
   own_note="StabMap reached CHAIN_OK on this subsample in about a minute when we "
            "ran it.",
 ),
 "cross": dict(
   ds="D52", cells="23,478",
   blurb=("**Cross integration** has several batches in which **all** modalities are "
          "present; the task is removing batch effects while keeping biological "
          "structure. Spatial registration (PASTE, PASTE2, SPIRAL, GPSA) also lives "
          "under `cross` - see the note at the end."),
   live=("StabMap", "None", "StabMap finishes in about 70 s"),
   live_ds=None,
   own_src="D52", own_fast="StabMap",
   own_note="StabMap reached CHAIN_OK on this subsample in under 30 s when we ran it.",
 ),
}

SUBSAMPLE_FN = '''import os, shutil
import h5py
import numpy as np
import pandas as pd

def subsample_dataset(src_dir, dst_dir, frac=0.6, seed=0):
    """Copy a dataset to a new name, keeping a random fraction of the cells.

    Files sharing a cell count get the SAME kept-cell index, so modality files
    and their label CSVs stay aligned - which is exactly the property your own
    export pipeline must preserve. The output is the canonical layout:
    matrix/data as features x cells, plus matrix/features and matrix/barcodes.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(dst_dir, exist_ok=True)
    counts, keep = {}, {}
    for fn in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, fn)
        if fn.endswith(".h5"):
            with h5py.File(p) as f:
                if "matrix/data" in f:
                    counts[fn] = f["matrix/data"].shape[1]   # features x cells
        elif fn.endswith(".csv"):
            counts[fn] = len(pd.read_csv(p))
    for n in set(counts.values()):
        k = max(50, int(n * frac))
        keep[n] = np.sort(rng.choice(n, size=k, replace=False))
    for fn, n in counts.items():
        sp, dp = os.path.join(src_dir, fn), os.path.join(dst_dir, fn)
        idx = keep[n]
        if fn.endswith(".csv"):
            pd.read_csv(sp).iloc[idx].to_csv(dp, index=False)
        else:
            with h5py.File(sp) as f, h5py.File(dp, "w") as g:
                grp = g.create_group("matrix")
                grp.create_dataset("data", data=np.asarray(f["matrix/data"])[:, idx])
                if "matrix/features" in f:
                    grp.create_dataset("features", data=np.asarray(f["matrix/features"]))
                if "matrix/barcodes" in f:
                    grp.create_dataset("barcodes", data=np.asarray(f["matrix/barcodes"])[idx])
    return dst_dir'''


for cat, s in SCEN.items():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    fastm = s["live"][0]
    live_ds = s["live_ds"] or s["ds"]

    # ------------------------------------------------------------------ title
    md(f"""# {cat.capitalize()} integration with `multibench`

{s['blurb']}

This tutorial covers, end to end:

- installing the package and the per-method environments
- the on-disk data layout this category expects
- seeing what runs on a dataset (`scan`) and what each method exposes for tuning
- running one method live, then a whole benchmark sweep with metrics
- reading the two metric families and drawing the standard figures
- **running the same pipeline on your own dataset**, demonstrated for real

**Reference dataset:** `{s['ds']}` ({s['cells']} cells). The stored results shipped
with these notebooks were produced on it, so every table below reproduces.""")

    # ---------------------------------------------------------------- install
    md("""## Install

From the repository directory:

```bash
pip install -e .                # the multibench package + CLI
multibench env doctor           # which method environments exist / are missing
multibench env install --run    # build them all from the committed lockfiles
```

Each method runs in its **own conda environment** (they need mutually
incompatible framework versions), so the wrapper can run torch 1.x, torch 2.x,
TensorFlow and R methods in one sweep. `env install` is a dry run until you add
`--run`.

Measured on a clean machine: **29 environments, ~50 min build, 175 GB** (plus a
52 GB package cache you can drop afterwards with `conda clean -a`). Details and
the smallest end-to-end check live in `SETUP.md`.""")

    code("""import warnings; warnings.filterwarnings("ignore")
%matplotlib inline
from pathlib import Path
import pandas as pd
pd.set_option("display.max_colwidth", None)   # never truncate a `reason`
pd.set_option("display.max_columns", None)    # never hide a metric column
pd.set_option("display.width", 200)
import multibench as mtb

RESULTS = Path("results")     # stored sweep results, so comparisons reproduce
print("multibench", mtb.__version__)""")

    code(f'''DATASET  = "{s['ds']}"
CATEGORY = "{cat}"
FAST_METHOD = "{fastm}"       # used for the quick live demos below''')

    # ------------------------------------------------------------------ layout
    md(f"""## 1. The data layout

A dataset is a **folder of flat files**; the folder name is the dataset name.
`describe_layout` prints the exact filenames for each category:""")
    code("""print(mtb.describe_layout(CATEGORY))""")
    md("""The modality files are HDF5 with three required datasets:

| dataset | contents | shape |
|---|---|---|
| `matrix/data` | the matrix, **features x cells** | `(n_features, n_cells)` |
| `matrix/features` | one name per feature | `(n_features,)` |
| `matrix/barcodes` | one id per cell | `(n_cells,)` |

Note this is the **transpose** of the scanpy/AnnData convention (`AnnData.X` is
cells x genes). Two safety nets exist: `mtb.io.to_canonical(src, dst)` converts
an `.h5ad` correctly, and `scan()` rejects a transposed file at preflight instead
of letting a method fail half an hour in.

> **ATAC caution.** Methods disagree about the ATAC representation - some need
> **gene-activity scores**, others need **peaks** - and feeding the wrong one
> runs to completion and returns a plausible but wrong embedding, with no error.
> `describe_layout` above states which file resolves where; check what your
> files actually contain before trusting a result.""")

    # ------------------------------------------------------------------- scan
    md("""## 2. What can I run on this dataset?

`scan` inspects the folder and reports every method that can run - and, for the
rest, exactly why not (missing file, missing environment, wrong layout). Nothing
executes, so this is instant and safe.""")
    code("""avail = mtb.scan(DATASET, category=CATEGORY)
avail[avail.runnable][["method", "modalities", "env", "output_kind",
                       "n_tunable", "runtime_tier"]]""")
    md("""Methods that are *not* runnable come with a reason rather than a silent absence:""")
    code("""not_ok = avail[~avail.runnable][["method", "modalities", "reason"]]
not_ok.head(5) if len(not_ok) else "(everything in this category runs here)"
""")

    # -------------------------------------------------------- paper coverage
    md(f"""## 2b. How much of the paper does this cover?

`scan()` answers "what runs on THIS dataset". A different question: how many of
the methods the paper benchmarks for **{cat}** does this package wire at all?
Stated explicitly so you never mistake a dataset limitation for full coverage.""")
    code(f"""from multibench.engine import registry

PAPER = {PAPER_METHODS!r}
IMPUTATION_ONLY = {MOSAIC_IMPUTATION_ONLY!r}

paper = PAPER[CATEGORY]
wired = sorted({{m for m in mtb.list_methods()
                if any(v.when.get("category") == CATEGORY
                       for v in registry.get(m).variants)}})
missing = [m for m in paper if m not in wired]
print(f"paper benchmarks {{len(paper)}} methods for {{CATEGORY}}; this package wires {{len(wired)}}")
if missing:
    print("not wired here:", ", ".join(missing))
    imp = [m for m in missing if m in IMPUTATION_ONLY]
    if imp:
        print("  the paper evaluates these only via IMPUTATION, which is not wired:",
              ", ".join(imp))
else:
    print("full parity with the paper for this category")""")

    # ------------------------------------------------------------------ params
    md("""## 3. What can I tune?

`params_for` reports each method's defaults and, where the upstream script
exposes any, the tunable hyperparameters. **An empty `tunable` is honest**: many
upstream scripts hardcode their hyperparameters, and this package never edits
upstream code, so it reports rather than pretends.""")
    code("""rows = []
for m in avail[avail.runnable]["method"]:
    try:
        p = mtb.params_for(m, CATEGORY)
    except Exception:                      # multi-variant: needs modalities
        mods = avail[avail.method == m].iloc[0]["modalities"].split("+")
        p = mtb.params_for(m, CATEGORY, mods)
    rows.append({"method": m, "n_tunable": len(p.get("tunable") or {}),
                 "tunable": ", ".join(sorted((p.get("tunable") or {}))[:6])})
pd.DataFrame(rows).sort_values("n_tunable", ascending=False).reset_index(drop=True)""")

    # ------------------------------------------------------------- run one
    live_p = s["live"][1]
    pstr = (f', params={{"{fastm}": {live_p}}}' if live_p != "None" else "")
    extra = (f'\n\n(This demo runs on `{live_ds}`, whose layout fits {fastm}; '
             f'mosaic layouts vary per dataset - see the note above.)'
             if live_ds != s["ds"] else "")
    md(f"""## 4. Run one method

`run_all` runs methods end to end: resolve inputs -> run in the method's own
conda env -> load the output -> compute metrics -> keep everything in a
`BatchResult`. {s['live'][2]}.{extra}""")
    code(f'''res = mtb.run_all("{live_ds}", CATEGORY,
                  methods=["{fastm}"]{pstr},
                  out_dir="/tmp/tutorial_{cat}")
res.summary''')
    md("""The metrics are already computed - `run_all` picked the right label files,
resolved the label order (see `label_order` in the summary), and scored the
embedding. One figure:""")
    code("""res.plot()""")

    # ------------------------------------------------------------ run all
    md(f"""## 5. Run the whole benchmark

The same call without `methods=` runs everything runnable. On `{s['ds']}` that is
hours of compute, so this cell is shown but not executed here - the stored
results it produced are loaded right below.

```python
res = mtb.run_all(DATASET, CATEGORY, out_dir="results/{cat}_all",
                  timeout=4*3600,       # a hung method is recorded, not fatal
                  skip_existing=True)   # resume instead of repeating hours
print(res.failures)                     # ALWAYS check: failures are recorded, not raised
```

Three behaviours worth knowing before a long sweep:

- **a method that fails is a row, not an exception** - the sweep continues and
  `res.failures` carries the error text;
- **`timeout=` bounds each method's whole step** including metric computation;
- **`skip_existing=True` resumes** a killed sweep, but refuses to combine with
  `params=` (it cannot know the old output used your new parameters).""")
    code(f'''summary = pd.read_csv(RESULTS / "summary_{s['ds']}.csv")
cols = [c for c in ["method","status","run_sec",
                    "ARI","NMI","ASW","iASW","iF1","cLISI",      # clustering
                    "ASW_batch","GC","iLISI"                     # batch correction
                    ] if c in summary.columns]
summary[cols]''')

    # ------------------------------------------------------------- metrics
    md("""## 6. Reading the metrics

Two families, matching the paper's grouping. All are **higher = better**, on
[0, 1] except ARI (can be slightly negative at chance level).

| family | metrics | what they measure |
|---|---|---|
| clustering / bio-conservation | `ARI`, `NMI`, `ASW`, `iASW`, `iF1`, `cLISI` | does the embedding separate the annotated cell types? |
| batch correction | `ASW_batch`, `GC`, `iLISI` (+ opt-in `kBET`) | are the batches mixed within each cell type? |

Notes that save confusion later:

- `iASW`/`iF1` are **isolated-label** scores; this benchmark scores *every*
  label, so they exist even on a single-batch dataset.
- batch metrics appear only when the dataset has real batches - their absence on
  a single-batch dataset is correct, not missing data.
- `kBET` is opt-in (`mtb.evaluate(..., slow_metrics=True)`): it spawns R per
  method and costs hours per dataset.
- graph-output methods (e.g. Seurat_WNN) legitimately have **no** embedding
  metrics; their status says so rather than scoring garbage.""")

    # ------------------------------------------------------------- figures
    md("""## 7. The figures

**Per-dataset bubble chart** - radius encodes rank (rank 1 = largest bubble),
colour the value:""")
    code(f'''long = pd.read_csv(RESULTS / "long_all_{s['ds']}.csv")
fig = mtb.plot.bubble(long, title="{cat} integration - {s['ds']}")
fig.set_dpi(110)
fig''')
    md("""**Across datasets** - when you have `long` tables from several datasets,
concatenate them and `mtb.plot.bar` summarises each method across all of them
(only methods present in more than one dataset are truly comparative):

```python
allx = pd.concat([long_d1, long_d2, ...], ignore_index=True)
mtb.plot.bar(allx, group="clustering")   # or group="batch", or all metrics
```""")

    # ------------------------------------------------------------ own data
    md(f"""## 8. Your own dataset - for real

Everything above used shipped data. This section does what you will actually do:
put files in a folder, point the package at it, and get scored results - executed
here on a dataset the package has never seen (a {int(0.6*100)}% cell subsample of
`{s['own_src']}` under a new name, built with ordinary h5py/pandas code you can
adapt to your own export pipeline).""")
    code(SUBSAMPLE_FN)
    code(f'''DATA_ROOT = "/tmp/mydata"
src = mtb.config.DEFAULT.data_path / "{s['own_src']}"
subsample_dataset(src, f"{{DATA_ROOT}}/MYDATA_{cat}", frac=0.6)

sc = mtb.scan(f"MYDATA_{cat}", category=CATEGORY, data_path=DATA_ROOT)
print(f"{{int(sc.runnable.sum())}} of {{len(sc)}} methods can run on MYDATA_{cat}")''')
    code(f'''mine = mtb.run_all(f"MYDATA_{cat}", CATEGORY,
                   methods=["{s['own_fast']}"],
                   out_dir=f"{{DATA_ROOT}}/out_{cat}",
                   data_path=DATA_ROOT)
mine.summary''')
    code("""mine.plot()""")
    md(f"""{s['own_note']}

For your real data the only work is producing the canonical files: export each
modality with `mtb.io.to_canonical` (from `.h5ad`) or the h5py pattern above,
write one label CSV per the layout in section 1, and the same three calls -
`scan`, `run_all`, `plot` - do the rest.""")

    # -------------------------------------------------------- troubleshooting
    trouble_extra = ""
    if cat == "cross":
        trouble_extra = """
### Note - spatial registration

`PASTE`, `PASTE2`, `SPIRAL` and `GPSA` are cross-integration methods whose output
is **aligned spatial coordinates**, not an embedding - their status reports
`RUN_OK_NO_EMBEDDING` and clustering metrics genuinely do not apply. Point them
at a directory of spatial slices (see `mtb.scan("D63", category="cross")`)."""
    if cat == "mosaic":
        trouble_extra = """
### Note - why is UINMF not in mosaic?

Two verified blockers, recorded in the registry: its upstream script derives the
second batch's unshared features from the FIRST batch's object (harmless when
both unshared blocks are the same modality, fatal otherwise), and its two-batch
shape fits no mosaic dataset shipped here. A method `scan()` reports as runnable
must actually run, so it is deliberately not offered."""
    md(f"""## Troubleshooting

| symptom | meaning | fix |
|---|---|---|
| `scan` says not runnable: input files not found | a required file is absent | the reason names the exact file and lists what IS in the folder |
| `scan` says env missing | that method's conda env is not built | `multibench env install --run` |
| `... looks like cells x features` | matrix stored transposed | re-export with `mtb.io.to_canonical` |
| a method FAILs in seconds | wrong input representation or layout | read `res.failures.iloc[0]["error"]` - the full command line and stderr tail are there |
| a method TIMEOUTs | slow, not broken | raise `timeout=`; runtime tiers in `scan` are measured, not guessed |
| `label_order_confidence` low | several label files fit the cell count | check `label_order_candidates` in the record |
{trouble_extra}

## Next steps

- the other three tutorials: **vertical, diagonal, mosaic, cross** each have one
- `SETUP.md` - measured install cost and the smallest end-to-end check
- `mtb.method_info(name)` - everything the registry knows about one method
- `mtb.sweep(...)` - one method over a range of one hyperparameter""")

    nb = nbf.v4.new_notebook(cells=C, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    })
    path = os.path.join(OUT, f"tutorial_{cat}.ipynb")
    nbf.write(nb, path)
    print(f"wrote {path} {len(C)} cells")
