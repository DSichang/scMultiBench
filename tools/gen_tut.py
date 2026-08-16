"""Generate the four integration tutorials (vertical / diagonal / mosaic / cross).

Order follows what a reader actually does: install, run the analysis and get a
figure, run the same three calls on their own data, and only then read stored
results back for the published figures. Reference material (scan, tunables,
metric definitions, coverage) sits at the end, where it is looked up rather than
read through.

Prose earns its place or it is cut: structured facts go in tables, every numeric
claim is a measured value from the verification runs, and nothing is explained
twice.
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

CAT_GB = {"vertical": "~101 GB (18 envs)", "diagonal": "~58 GB (9 envs)",
          "mosaic": "~45 GB (7 envs)", "cross": "~71 GB (11 envs)"}
CAT_DATA = {"vertical": ["D11"], "diagonal": ["D28"],
            "mosaic": ["D45", "D46"], "cross": ["D52"]}
DS_MB = {"D11": "11 MB", "D28": "137 MB", "D45": "290 MB", "D46": "97 MB",
         "D52": "179 MB"}

SCEN = {
 "vertical": dict(
   ds="D11", cells="2,864",
   blurb=("**Vertical integration** fuses several modalities measured **in the same "
          "cells** (here CITE-seq: RNA + surface protein). Cells are already matched, "
          "so the task is combining modalities, not aligning cells."),
   live=("Matilda", '{"epochs": 5}', "Matilda at 5 epochs is quick - `run_sec` in the summary is the measured time on our host"),
   live_ds=None,
   own_src="D11", own_trio=["Matilda", "sciPENN", "scMM"],
   own_note="When we ran this, Matilda scored ARI ~0.95 on the subsample - your "
            "numbers will differ slightly; the point is that they are computed "
            "end to end on data the package has never seen.",
 ),
 "diagonal": dict(
   ds="D28", cells="6,408 RNA + 4,606 ATAC",
   blurb=("**Diagonal integration** is the hard case: RNA and ATAC come from "
          "**different cells**, with no pairing and no shared cell ids, so a method "
          "must align two populations through a shared feature space (for RNA + ATAC, "
          "gene-activity scores, computed upstream of this package by e.g. Signac or "
          "ArchR). If your RNA and ATAC come from the SAME cells (10x multiome), that "
          "is vertical integration - use that tutorial instead."),
   live=("online_iNMF", "None", "online_iNMF is among the fastest methods here - `run_sec` in the summary is the measured time on our host"),
   live_ds=None,
   own_src="D28", own_trio=["online_iNMF", "iNMF", "scJoint"],
   own_note="All three reached CHAIN_OK with all nine metrics on this subsample "
            "when we ran them.",
 ),
 "mosaic": dict(
   ds="D45", cells="32,151",
   blurb=("**Mosaic integration** has several batches where only **some** share a "
          "modality - an RNA-only batch, an ATAC-only batch, and a paired batch that "
          "bridges them. Which methods apply depends on the exact per-batch pattern, "
          "so `scan()` is the source of truth for each dataset."),
   live=("StabMap", "None", "Both run in minutes to tens of minutes on D46 - `run_sec` in the summary is the measured time on our host"),
   live_ds="D46",
   own_src="D46", own_trio=["StabMap", "scMoMaT"],
   own_note="Both reached CHAIN_OK on this subsample when we ran them. Two methods "
            "rather than three because that is every wired mosaic method D46's "
            "layout admits - the four-method D45 figures below are where this "
            "category's methods are compared side by side.",
 ),
 "cross": dict(
   ds="D52", cells="23,478",
   blurb=("**Cross integration** has several batches in which **all** modalities are "
          "present; the task is removing batch effects while keeping biological "
          "structure. Spatial registration (PASTE, PASTE2, SPIRAL, GPSA) also lives "
          "under `cross` - see the note at the end."),
   live=("StabMap", "None", "StabMap is among the fastest cross methods - `run_sec` in the summary is the measured time on our host"),
   live_ds=None,
   own_src="D52", own_trio=["UINMF", "sciPENN", "StabMap"],
   own_note="All three reached CHAIN_OK on this subsample when we ran them.",
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
    ds, ds2 = s["ds"], s["ds"] + "s"

    # ------------------------------------------------------------------ title
    md(f"""# {cat.capitalize()} integration

{s['blurb']}

**Reference dataset:** `{s['ds']}` ({s['cells']} cells). The stored results
shipped with these notebooks were produced on it, so every table here
reproduces.""")

    # ---------------------------------------------------------------- install
    live_methods = sorted({fastm, *s["own_trio"]})
    mlist = ",".join(live_methods)
    md("""## 1. Install

Two layers: the `multibench` package (~2 MB) and the conda environments of the
methods you run. On **Colab**, run the first cell and let the kernel restart
once - then keep running from the next cell.""")
    code("""# Colab ships without conda; this provisions it (the kernel restarts ONCE).
# On a machine that already has conda, this cell does nothing.
import importlib.util, shutil

def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False

if shutil.which("conda") or shutil.which("mamba"):
    print("conda available - nothing to do")
elif _has("google.colab"):
    !pip -q install condacolab
    import condacolab
    condacolab.install()   # restarts the kernel; afterwards, continue below
else:
    print("no conda found - install it first (mamba recommended); see the installation guide")""")
    code("""import importlib.util, os
if importlib.util.find_spec("multibench") is None:
    !git clone --depth 1 https://github.com/DSichang/scMultiBench.git
    %cd scMultiBench
    !pip -q install -e .
elif os.path.isdir("/content/scMultiBench"):
    # reused Colab runtime: refresh the editable install to the latest code,
    # then drop the already-imported modules so the NEXT import sees it -
    # a live kernel never re-reads changed files on its own
    %cd /content/scMultiBench
    !git pull -q
    !pip -q install -e .
    import importlib, sys
    for _m in [m for m in list(sys.modules) if m == "multibench" or m.startswith("multibench.")]:
        del sys.modules[_m]
    importlib.invalidate_caches()
    print("multibench refreshed to the latest repository state")
else:
    print("multibench already installed")""")
    md(f"""Now the environments for the methods this tutorial runs
({', '.join(live_methods)}). `--packed` downloads a prebuilt archive instead of
solving one from scratch, and `env install` skips anything already present.
Other tiers are one flag away: `--category {cat}` ({CAT_GB[cat]}), or no flag
for the whole benchmark (29 envs, ~167 GB).""")
    code(f"""import sys
!{{sys.executable}} -m multibench env install --methods {mlist} --packed --run""")

    code(f'''import warnings; warnings.filterwarnings("ignore")
%matplotlib inline
from pathlib import Path
import pandas as pd
pd.set_option("display.max_colwidth", None)   # never truncate a `reason`
pd.set_option("display.max_columns", None)    # never hide a metric column
pd.set_option("display.width", 200)
import multibench as mtb

DATASET  = "{s['ds']}"
CATEGORY = "{cat}"
RESULTS = Path("results") if Path("results").exists() else Path("notebooks/results")
mtb.data.fetch({', '.join(repr(d) for d in CAT_DATA[cat])})   # reference data ({', '.join(DS_MB[d] for d in CAT_DATA[cat])}); no-op when present
print("multibench", mtb.__version__)''')

    # ------------------------------------------------------------- run + plot
    trio = s["own_trio"]
    params_note = f', params={{"{fastm}": {s["live"][1]}}}' if s["live"][1] != "None" else ""
    extra = (f' (on `{live_ds}`, whose layout fits these methods; mosaic layouts '
             f'vary per dataset)' if live_ds != s["ds"] else "")
    md(f"""## 2. Run the analysis

One call is the whole pipeline: resolve each method's inputs, run it in its own
conda env, load the embeddings, score them with scIB metrics. Here
{', '.join(trio)}{extra}. {s['live'][2]}.""")
    code(f'''res = mtb.run_all("{live_ds}", CATEGORY,
                  methods={trio!r}{params_note},
                  out_dir="/tmp/tutorial_{cat}")
res.summary''')
    md("""The result object plots itself in the paper's layout:""")
    code("""res.plot()""")
    md("""Each circle carries two encodings: its **size is the method's rank** in
that column (largest = rank 1) and its **colour is the metric's value**, min-max
scaled within the column (darker = higher). Columns are grouped by task family -
blues for DR & clustering, greens for batch correction - and each family is led
by an **Overall** rank column, where bar length and colour carry the same two
encodings.""")

    # ------------------------------------------------------------- own data
    md(f"""## 3. Your own data

The same three calls - `scan`, `run_all`, `plot` - on a folder the package has
never seen. A dataset is a folder of flat files named by category:""")
    code("""print(mtb.describe_layout(CATEGORY))""")
    md("""Each modality file is HDF5 with three datasets:

| dataset | contents | shape |
|---|---|---|
| `matrix/data` | the matrix, **features x cells** | `(n_features, n_cells)` |
| `matrix/features` | one name per feature | `(n_features,)` |
| `matrix/barcodes` | one id per cell | `(n_cells,)` |

That is the **transpose** of the AnnData convention (`AnnData.X` is cells x
genes); `mtb.io.to_canonical` converts an `.h5ad` correctly, and `scan()` rejects
a transposed file at preflight rather than letting a method fail half an hour in.
The label CSV is a **single column** with one header line (typically `x`) and one
label per cell, in the same order as `matrix/barcodes` of the matching modality
file. For ATAC, check which representation a method wants - gene-activity scores
or peaks - because the wrong one runs to completion and returns a plausible but
wrong embedding. A shipped label file, in full:""")
    code("""cty = sorted((mtb.config.DEFAULT.data_path / DATASET).glob("*cty*.csv"))[0]
print(cty.name)
print(*open(cty).read().splitlines()[:4], sep="\\n")""")
    md("""Converting an AnnData object to that layout, executed:""")
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
    md(f"""Now a real dataset the package has never seen: a 60% cell subsample of
`{s['own_src']}` under a new name, built with ordinary h5py/pandas code you can
adapt to your own export pipeline.""")
    code(SUBSAMPLE_FN)
    code(f'''DATA_ROOT = "/tmp/mydata"
src = mtb.config.DEFAULT.data_path / "{s['own_src']}"
subsample_dataset(src, f"{{DATA_ROOT}}/MYDATA_{cat}", frac=0.6)

sc = mtb.scan(f"MYDATA_{cat}", category=CATEGORY, data_path=DATA_ROOT)
print(f"{{int(sc.runnable.sum())}} of {{len(sc)}} methods can run on MYDATA_{cat}")''')
    code(f'''mine = mtb.run_all(f"MYDATA_{cat}", CATEGORY,
                   methods={s['own_trio']!r},
                   out_dir=f"{{DATA_ROOT}}/out_{cat}",
                   data_path=DATA_ROOT)
mine.summary''')
    code("""mine.plot()""")
    md(f"""{s['own_note']} For your real data the only work is producing the
canonical files - `mtb.io.to_canonical` from `.h5ad`, or the h5py pattern above,
plus one label CSV - and these same three calls do the rest.""")

    # ------------------------------------------------------------- figures
    batch_note = (
        "Both datasets are single-batch, so this category's summary shows the "
        "clustering family only - the diagonal, mosaic and cross tutorials show "
        "the batch-correction family alongside it."
        if cat == "vertical" else
        "Both metric families appear because both datasets are multi-batch.")
    n_word = {1: "one method", 2: "two methods", 3: "three methods"}.get(
        len(trio), f"{len(trio)} methods")
    md(f"""## 4. Reading stored results

Section 2 ran {n_word}; the repository ships the **full sweep** for `{ds}` -
every wired method at default settings, hours of compute - so the paper's
figures reproduce from stored results in seconds. Reading them back is a plain
CSV read, and `mtb.plot.bubble` takes it from there:""")
    code(f'''long = pd.read_csv(RESULTS / "long_all_{ds}.csv")
fig = mtb.plot.bubble(long)
fig.set_dpi(110)
fig''')
    md(f"""`run_all(DATASET, CATEGORY, out_dir=...)` without `methods=` writes the
same `summary.csv` and `long.csv` for your own data.

**Across datasets.** A summary needs every method to have results on every
dataset it averages over, or absence and performance blur into the same bar. Two
{cat} datasets therefore ship swept identically - `{ds}` and `{ds2}` (a 60% cell
subsample of `{ds}` under a new name) - and the code below keeps their method
**intersection**, so the matrix is complete by construction. Each bar is the
**grand rank**: the min-max scaled mean rank across the datasets, with length and
colour both carrying it, and `Overall` is the same statistic over the grand ranks.
{batch_note}""")
    code(f'''a = pd.read_csv(RESULTS / "long_all_{ds}.csv").assign(dataset="{ds}")
b = pd.read_csv(RESULTS / "long_all_{ds2}.csv").assign(dataset="{ds2}")
both = sorted(set(a.method) & set(b.method))   # complete matrix, by construction
pair = pd.concat([a, b], ignore_index=True)
pair = pair[pair.method.isin(both)]
print(f"{{len(both)}} methods with results on both datasets")
mtb.plot.bubble(pair, aggregate="summary",
                title=f"Summary of 2 {cat} datasets, {{len(both)}} methods")''')

    # ----------------------------------------------------------- reference
    md("""## 5. Reference

### What runs on a dataset, and why not

`scan` inspects a folder and reports every runnable method - and for the rest,
the exact reason (missing file, missing environment, wrong layout). Nothing
executes.""")
    code("""avail = mtb.scan(DATASET, category=CATEGORY)
avail[avail.runnable][["method", "modalities", "env", "output_kind",
                       "n_tunable", "runtime_tier"]]""")
    code("""not_ok = avail[~avail.runnable][["method", "modalities", "reason"]]
not_ok.head(5) if len(not_ok) else "(everything in this category runs here)"
""")
    md("""### What each method exposes for tuning

An empty `tunable` is honest: many upstream scripts hardcode their
hyperparameters, and this package never edits upstream code.""")
    code("""from multibench.engine import registry

rows = []
for m in sorted(mtb.list_methods(category=CATEGORY)):
    p = None
    try:
        p = mtb.params_for(m, CATEGORY)
    except Exception:                      # multi-variant: try each variant's modalities
        for v in registry.get(m).variants:
            if v.when.get("category") != CATEGORY:
                continue
            try:
                p = mtb.params_for(m, CATEGORY, v.when.get("modalities"))
                break
            except Exception:
                continue
    if p is None:                          # variant selection needs a concrete dataset
        rows.append({"method": m, "n_tunable": 0, "tunable": "(see scan() on your dataset)"})
        continue
    rows.append({"method": m, "n_tunable": len(p.get("tunable") or {}),
                 "tunable": ", ".join(sorted((p.get("tunable") or {}))[:6])})
pd.DataFrame(rows).sort_values("n_tunable", ascending=False).reset_index(drop=True)""")
    md("""### The metrics

Two families, matching the paper's grouping. All are **higher = better**, on
[0, 1] except ARI (slightly negative at chance level).

| family | metrics | what they measure |
|---|---|---|
| clustering / bio-conservation | `ARI`, `NMI`, `ASW`, `iASW`, `iF1`, `cLISI` | does the embedding separate the annotated cell types? |
| batch correction | `ASW_batch`, `GC`, `iLISI` (+ opt-in `kBET`) | are the batches mixed within each cell type? |

Batch metrics appear only when the dataset has real batches - their absence on a
single-batch dataset is correct, not missing data. `kBET` is opt-in
(`mtb.evaluate(..., slow_metrics=True)`) because it is much slower than the rest.""")
    md(f"""### Coverage of the paper

`scan()` answers "what runs on this dataset"; this answers how many of the
methods the paper benchmarks for **{cat}** the package wires at all.""")
    code(f"""PAPER = {PAPER_METHODS!r}
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

    # -------------------------------------------------------- troubleshooting
    trouble_extra = ""
    if cat == "cross":
        trouble_extra = """
### Spatial registration

`PASTE`, `PASTE2`, `SPIRAL` and `GPSA` are cross-integration methods whose output
is **aligned spatial coordinates**, not an embedding - their status reports
`RUN_OK_NO_EMBEDDING` and clustering metrics genuinely do not apply. Point them
at a directory of spatial slices (see `mtb.scan("D63", category="cross")`)."""
    if cat == "mosaic":
        trouble_extra = """
### Why is UINMF not in mosaic?

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
- the hosted interactive explorer: <https://shiny.maths.usyd.edu.au/scMultiBench/> -
  the full benchmark's rankings, browsable without installing anything
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
