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
   live=("Matilda", '{"epochs": 5}', "Matilda at 5 epochs is quick - `run_sec` below is the measured time on our host"),
   live_ds=None,
   own_src="D11", own_fast="Matilda",
   own_note="When we ran this, Matilda scored ARI ~0.95 on the subsample - your "
            "exact number will differ slightly; the point is that it is high and "
            "computed end-to-end on data the package has never seen.",
 ),
 "diagonal": dict(
   ds="D28", cells="6,408 RNA + 4,606 ATAC",
   blurb=("**Diagonal integration** is the hard case: RNA and ATAC come from "
          "**different cells**, with no pairing and no shared cell ids. A method must "
          "align two populations using only a shared feature space (for RNA + ATAC, "
          "gene-activity scores). If your RNA and ATAC come from the SAME cells "
          "(10x multiome), that is vertical integration - use that tutorial instead. "
          "Gene-activity scores are computed from peaks upstream of this package "
          "(e.g. Signac's GeneActivity or ArchR's gene score matrix); section 2 "
          "shows how to check which representation a file holds."),
   live=("online_iNMF", "None", "online_iNMF is among the fastest methods in this category - the `run_sec` column below is the measured time on our host"),
   live_ds=None,
   own_src="D28", own_fast="Portal",
   own_note="Portal reached CHAIN_OK with all nine metrics on this subsample when "
            "we ran it; `run_sec` above is the measured time.",
 ),
 "mosaic": dict(
   ds="D45", cells="32,151",
   blurb=("**Mosaic integration** has several batches where only **some** share a "
          "modality - an RNA-only batch, an ATAC-only batch, and a paired batch that "
          "bridges them. Mosaic methods are the most layout-sensitive of the four "
          "categories: each supports specific per-batch modality patterns, so "
          "`scan()` per dataset is the source of truth for what applies."),
   live=("StabMap", "None", "StabMap runs in a few minutes on D46 - `run_sec` below is the measured time on our host"),
   live_ds="D46",
   own_src="D46", own_fast="StabMap",
   own_note="StabMap reached CHAIN_OK on this subsample when we ran it; `run_sec` "
            "above is the measured time.",
 ),
 "cross": dict(
   ds="D52", cells="23,478",
   blurb=("**Cross integration** has several batches in which **all** modalities are "
          "present; the task is removing batch effects while keeping biological "
          "structure. Spatial registration (PASTE, PASTE2, SPIRAL, GPSA) also lives "
          "under `cross` - see the note at the end."),
   live=("StabMap", "None", "StabMap is among the fastest cross methods - `run_sec` below is the measured time on our host"),
   live_ds=None,
   own_src="D52", own_fast="StabMap",
   own_note="StabMap reached CHAIN_OK on this subsample when we ran it; `run_sec` "
            "above is the measured time.",
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
with these notebooks were produced on it, so every table below reproduces.

**Contents** - 1 Install · 2 Data layout · 3 What can I run · 4 Paper coverage ·
5 What can I tune · 6 Run one method · 7 Run the benchmark · 8 Reading the
metrics · 9 Figures · 10 Your own dataset · Troubleshooting""")

    # ---------------------------------------------------------------- install
    md("""## 1. Install

Prerequisites: Linux, `conda` (mamba recommended) and ~230 GB free disk
during the build. `multibench` is not on PyPI yet - install from the repository:

```bash
git clone https://github.com/PYangLab/scMultiBench.git
cd scMultiBench
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
CATEGORY = "{cat}"''')

    # ------------------------------------------------------------------ layout
    md(f"""## 2. The data layout

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

**The label CSV** is the one file you author by hand, so its schema in full:
one row per cell, in **the same order as `matrix/barcodes`** of the matching
modality file(s); the cell-type label is the last column (or a column named
`x`); a header row is expected. Where a category uses several label files
(`cty1.csv`, `rna_cty.csv`, ...), each aligns with its own batch or modality.
The next cell prints the head of a shipped one - this is the whole format:""")
    code("""cty = sorted((mtb.config.DEFAULT.data_path / DATASET).glob("*cty*.csv"))[0]
print(cty.name)
print(*open(cty).read().splitlines()[:4], sep="\\n")""")
    md("""> **ATAC caution.** Methods disagree about the ATAC representation - some need
> **gene-activity scores**, others need **peaks** - and feeding the wrong one
> runs to completion and returns a plausible but wrong embedding, with no error.
> `describe_layout` above states which file resolves where; check what your
> files actually contain before trusting a result.

**From AnnData to canonical, executed.** Most real data starts as `.h5ad`;
`mtb.io.to_canonical` writes the layout above correctly (including the
transpose). Converting a small demo object end to end:""")
    code("""import anndata as ad, numpy as np, h5py, tempfile, os
tmp = tempfile.mkdtemp()
demo = ad.AnnData(X=np.random.poisson(2.0, size=(120, 40)).astype(float))
demo.obs_names = [f"cell{i}" for i in range(120)]
demo.var_names = [f"gene{i}" for i in range(40)]
src = os.path.join(tmp, "demo.h5ad"); demo.write_h5ad(src)

dst = os.path.join(tmp, "rna.h5")
mtb.io.to_canonical(src, dst)
with h5py.File(dst) as f:
    print("keys :", sorted(f["matrix"].keys()))
    print("shape:", f["matrix/data"].shape, "(features x cells - transposed for you)")""")

    # ------------------------------------------------------------------- scan
    md("""## 3. What can I run on this dataset?

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
    md(f"""## 4. How much of the paper does this cover?

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
    md("""## 5. What can I tune?

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
    md(f"""## 6. Run one method

`run_all` runs methods end to end: resolve inputs -> run in the method's own
conda env -> load the output -> compute metrics -> keep everything in a
`BatchResult`. {s['live'][2]}.{extra}""")
    code(f'''res = mtb.run_all("{live_ds}", CATEGORY,
                  methods=["{fastm}"]{pstr},
                  out_dir="/tmp/tutorial_{cat}")
res.summary''')
    md("""(The lines scrolling above the table are scIB's own progress chatter from
the metric computation - harmless; the result is the summary row below.)

The metrics are already computed - `run_all` picked the right label files,
resolved the label order (see `label_order` in the summary), and scored the
embedding. One figure:""")
    code("""res.plot()""")

    # ------------------------------------------------------------ run all
    md(f"""## 7. Run the whole benchmark

The same call without `methods=` runs everything runnable. On `{s['ds']}` that is
hours of compute, so this cell is shown but not executed here - the stored
results it produced are loaded right below.

```python
res = mtb.run_all(DATASET, CATEGORY, out_dir="results/{cat}_all",
                  timeout=4*3600,       # a hung method is recorded, not fatal
                  skip_existing=True)   # resume instead of repeating hours
print(res.failures)                     # ALWAYS check: failures are recorded, not raised
```

`run_all` writes `summary.csv`, `long.csv` and `failures.csv` into `out_dir` -
the stored files under `results/` loaded below are exactly those, kept so the
notebook reproduces without re-running the sweep. (The live demo above used
reduced settings where noted; the stored sweep ran defaults, so its `run_sec`
differs.)

Three behaviours worth knowing before a long sweep:

- **a method that fails is a row, not an exception** - the sweep continues and
  `res.failures` carries the error text;
- **`timeout=` bounds each method's whole step** including metric computation;
- **`skip_existing=True` resumes** a killed sweep, but refuses to combine with
  `params=` (it cannot know the old output used your new parameters).""")
    code(f'''summary = pd.read_csv(RESULTS / "summary_{s['ds']}.csv")
clu = [c for c in ["ARI","NMI","ASW","iASW","iF1","cLISI"] if c in summary.columns]
bat = [c for c in ["ASW_batch","GC","iLISI"] if c in summary.columns]
cols = ["method","status","run_sec"] + clu + bat
(summary[cols].sort_values(clu[0], ascending=False)
 .style.background_gradient(subset=clu, cmap="Blues", vmin=0, vmax=1)
       .background_gradient(subset=bat, cmap="Greens", vmin=0, vmax=1)
       .format({{c: "{{:.3f}}" for c in clu + bat}}, na_rep="")
       .format({{"run_sec": "{{:.0f}}"}})
       .hide(axis="index"))''')

    # ------------------------------------------------------------- metrics
    md("""## 8. Reading the metrics

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
- `label_order` in the summary records which label files scored the embedding,
  in which order - when several candidate orders fit the cell counts, each is
  screened and the best kept; a low `label_order_confidence` means the ranking
  was close and `label_order_candidates` in the record shows the alternatives.
- `kBET` is opt-in (`mtb.evaluate(..., slow_metrics=True)`): it spawns R per
  method and costs hours per dataset.
- graph-output methods come in two kinds: scMoMaT also ships a secondary UMAP
  embedding, which is what gets scored (status `CHAIN_OK_GRAPH_METHOD`), while
  Seurat_WNN emits only the graph - it legitimately has **no** embedding metrics
  and its status says so rather than scoring garbage.""")

    # ------------------------------------------------------------- figures
    md("""## 9. The figures

**Per-dataset panel**, in the paper's layout: methods as rows (best first),
metrics as columns grouped by task family - blues for DR & clustering, greens
for batch correction - each family led by an **Overall** rank column, and the
top three per column carry their rank number. A missing marker means the metric
does not apply to that method.""")
    code(f'''long = pd.read_csv(RESULTS / "long_all_{s['ds']}.csv")
fig = mtb.plot.bubble(long, title="{cat} integration - {s['ds']}")
fig.set_dpi(110)
fig''')
    md("""**Across datasets, executed.** The results folder ships one `long` table per
reference dataset, so the summary API can be demonstrated for real: every marker
becomes a **bar whose length and colour encode the metric's rank averaged across
datasets**, with an **SD whisker over the datasets the method actually ran in**
(>= 2 needed; single-dataset methods get no whisker). The bar's LENGTH follows
the paper's convention of counting absence as rank 0, while the whisker measures
variability only where the method ran. `Overall` carries no whisker by design -
each dataset's overall is a within-dataset relative score over that dataset's
own method pool, so a cross-dataset SD of it would not compare like with like.

**Read this demonstration with its scope in mind.** It pools the four reference
datasets - **one per integration category** - so rows are the union of every
method that ran in any of them, each ranked only within its own dataset. That is
an API demo, not a scientific comparison: the paper's own summary panels pool
many datasets of ONE data type and category (e.g. "Summary of 12 datasets for
[RNA, ATAC]"), which is what you should do with your own data - concatenate
`long` tables from datasets of the same category and call this one function. It
also explains why whiskers cluster on the clustering side here: batch metrics
exist only in the three multi-batch reference datasets, whose method sets barely
overlap, so almost no method has the >= 2 batch datasets a whisker needs.""")
    code("""import glob
longs = [pd.read_csv(p).assign(dataset=Path(p).stem.replace("long_all_", ""))
         for p in sorted(RESULTS.glob("long_all_D*.csv"))]
allx = pd.concat(longs, ignore_index=True)
print(f"{allx.dataset.nunique()} datasets, {allx.method.nunique()} methods")
mtb.plot.bubble(allx, aggregate="summary",
                title=f"Pooled demo across {allx.dataset.nunique()} reference datasets (one per category)")""")


    # ------------------------------------------------------------ own data
    md(f"""## 10. Your own dataset - for real

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
write one label CSV per the layout in section 2, and the same three calls -
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
    siblings = ", ".join(f"**{c}**" for c in SCEN if c != cat)
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

- the other three tutorials: {siblings}
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
