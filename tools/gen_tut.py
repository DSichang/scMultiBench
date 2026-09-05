"""Generate the four integration tutorials (vertical / diagonal / mosaic / cross)
and the Colab quickstart.

Order follows what a reader actually does: install, run the analysis and get a
figure, run the same three calls on their own data, and only then read stored
results back for the published figures. Reference material (scan, tunables,
metric definitions, citations, coverage) sits at the end, where it is looked up
rather than read through.

Prose earns its place or it is cut: structured facts go in tables, every numeric
claim is a measured value from the verification runs, and nothing is explained
twice. The notebooks are regenerated from this file - never hand-edited - and
executed on the benchmark host afterwards.

Colab / laptop budget: the install cell pins numpy and pandas to what the
interpreter already has (pip never upgrades a host's stack), no cell
provisions conda on Colab (the packed method environments run without a
conda binary, so there is no kernel restart), a host without environments stands in the benchmark host's real ``run_all``
outputs (``mtb.data.fetch_outputs``) before falling back to the stored metric
table, and every stage calls ``tick(label)`` so the last cell can print where
the time went.
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


def stand_in(cat, dataset, methods):
    """Which stored sweep a run cell falls back to on a host without method
    environments: ``(dataset, methods)`` for ``stored_sweep`` in the notebook.
    Read from the live tables at generation time: ``methods`` is kept only when
    every requested method is in that dataset's sweep (vertical / diagonal /
    cross), else ``None`` selects the whole sweep (mosaic runs StabMap and
    scMoMaT on D46, whose layout the D45 sweep does not share)."""
    import warnings
    import multibench as mtb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stored = set(mtb.load_results(cat, dataset=dataset, source="rerun").method)
    return dataset, (list(methods) if set(methods) <= stored else None)


def published_note(cat, dataset):
    """One measured sentence on why every ``load_results`` call in a tutorial
    names its ``source``: the paper's table for the reference dataset vs the
    package's own sweep, counted from ``results_coverage`` at generation time."""
    import multibench as mtb
    cov = mtb.results_coverage(cat)
    cov = cov[cov.dataset == dataset]
    n_pub = cov[cov.source == "published"].method.nunique()
    n_rerun = cov[cov.source.str.startswith("rerun")].method.nunique()
    if n_pub == 0:
        return (f"The paper has no scIB tables for {cat} (`source=\"published\"` "
                f"raises `FileNotFoundError` pointing at `\"rerun\"`), so the "
                f"package's sweep ({n_rerun} methods on `{dataset}`) is this "
                f"category's only stored source.")
    return (f"The paper's table for `{dataset}` holds {n_pub} method"
            f"{'s' if n_pub != 1 else ''} against {n_rerun} in the package's "
            f"sweep, so the default `source=\"published\"` would show "
            f"{n_pub} - every call in this notebook names its source.")


def env_size_text(category=None, methods=None, short=False):
    """One phrase describing the conda envs a category / method set needs,
    derived from the dry-run plan ``mtb.env.install(methods, category=...)``
    returns at generation time (env count, the download total of the archives
    measured so far, and the CLI line that prints the per-env sizes) - never a
    hand-written GB figure. ``short=True`` keeps only the count and the total,
    for a code comment."""
    import multibench as mtb
    rows = mtb.env.install(methods, category=category)      # dry_run=True: the plan, nothing built
    known = [r["archive_bytes"] for r in rows if r["archive_bytes"]]
    n = len(rows)
    flag = f"--category {category}" if category else f"--methods {','.join(methods)}"
    s = f"{n} env{'s' if n != 1 else ''}"
    if known:
        s += f", at least {sum(known) / 1e9:.1f} GB to download"
        if not short:
            s += f" ({len(known)} of {n} archives measured)"
    if short:
        return s
    return s + f"; `multibench env plan {flag}` prints the per-env sizes"

CAT_DATA = {"vertical": ["D11"], "diagonal": ["D28"],
            "mosaic": ["D45", "D46"], "cross": ["D52"]}
DS_MB = {"D11": "11 MB", "D28": "137 MB", "D45": "290 MB", "D46": "97 MB",
         "D52": "179 MB"}

# The ONE install cell every notebook shares (tests/test_docs_consistency.py
# pins it): the package WITH its dependencies - the wheel ships the registry,
# the stored result tables, the env lockfiles and the reference metadata, so no
# clone is needed; evaluate() needs scib/scanpy, which pip brings in. The
# find_spec guard keeps the cell idempotent and leaves a developer's editable
# install alone. numpy and pandas are pinned to the versions the running
# interpreter already has, so pip never upgrades a host's stack (Colab pins
# pandas itself; a mismatched upgrade there costs minutes and breaks imports).
# The cell is also the notebook's setup: it defines the two-line tick(label)
# recorder every later stage calls, so the last cell can print where the time
# went - the install cell must record its own time, so the recorder lives here.
INSTALL_CELLS = [
"""import importlib.metadata, importlib.util, sys, time
_t, TIMES = [time.time()], []                    # "where the time went": tick(label) records the seconds since the previous tick
def tick(label): TIMES.append((label, round(time.time() - _t[0], 1))); _t[0] = time.time()
if importlib.util.find_spec("multibench") is None:
    # pin numpy / pandas to what this interpreter already has, so pip never upgrades the host's stack (Colab pins pandas itself)
    pins = [f"{p}=={importlib.metadata.version(p)}" for p in ("numpy", "pandas") if importlib.util.find_spec(p)]
    !{sys.executable} -m pip -q install "multibench-sc>=0.3" {" ".join(pins)}   # registry, stored tables, env lockfiles and references ship in the wheel - no clone needed
    importlib.invalidate_caches()
    if importlib.util.find_spec("multibench") is None:        # PyPI behind the docs? take the same code straight from GitHub
        !{sys.executable} -m pip -q install "git+https://github.com/DSichang/scMultiBench.git" {" ".join(pins)}
else:
    print("multibench already installed")
tick("install")""",
]

# The "Run all" switch (tests/test_tutorial_runall_safety.py pins it): the
# one download of method environments - `mtb.env.install(..., dry_run=False)`
# - sits behind INSTALL_ENVS, so the default is safe for thirty people on
# Colab at once (the packed envs for three methods are a multi-GB download).
# The size in the comment is measured at generation time, never hand-written.
FLAG_CELL_TEMPLATE = """# "Run all" switch. False (the default): nothing is downloaded and no method executes -
# every section runs on any machine, and the run cells stand in the benchmark host's results.
# True: on Colab / any Linux host, download the prebuilt environments for the methods this
# notebook runs ({size}; no conda needed) and run them here for real.
INSTALL_ENVS = False"""

# The one line a run cell prints on a host without method environments
# (tests pin the phrase), before standing in a result computed elsewhere.
SKIP_LINE = ("no method environment on this host - the run is skipped; "
             "a stand-in computed elsewhere covers it")

# Defined next to its first use: the LAST fallback the run cells take when
# scan() finds no environment for the requested methods. A BatchResult built
# from the stored sweep's rows has the same .summary / .plot() as run_all's,
# so every later cell renders unchanged; status='STORED' says nothing ran and
# nothing is on disk (no embedding to evaluate).
STORED_SWEEP_FN = '''def stored_sweep(dataset, methods=None):
    """The package's own sweep of `dataset` (`load_results(source="rerun")`) as the object
    `run_all` returns, so `.summary` / `.plot()` work on a host that cannot run the methods."""
    long = mtb.load_results(CATEGORY, dataset=dataset, source="rerun", methods=methods)
    recs = [{"method": m, "status": "STORED", "metrics": g.set_index("metric")["value"].to_dict()}
            for m, g in long.groupby("method")]
    return mtb.BatchResult(recs, dataset, CATEGORY)'''

# The FIRST fallback: the benchmark host's real run_all output tree for the
# dataset (batch_result.json, long.csv, one <method>_<dataset>/ folder with
# the embedding), downloaded by mtb.data.fetch_outputs and reloaded by
# load_batch - so .summary carries the host's statuses and run times and the
# evaluate cell scores a real embedding. It raises offline, or before the
# assets are published (or on a package without fetch_outputs yet), and then
# the stored metric table stands in; one printed line says which path was taken.
STAND_IN_FN = '''def stand_in(dataset, methods, stored):
    """What a host without the method environments shows instead of `run_all`'s result: the benchmark
    host's real `run_all` outputs for `dataset` (`mtb.data.fetch_outputs` - embeddings included), else
    the package's stored metric table (`stored_sweep(*stored)`). One line says which."""
    try:
        res = mtb.load_batch(mtb.data.fetch_outputs(dataset), methods=methods)
        print(f"stand-in: the benchmark host's run_all outputs for {dataset} (fetch_outputs) - real embeddings and run times")
        return res
    except Exception as e:                                   # offline, or the outputs are not published yet
        print(f"stand-in: the package's stored metric table ({type(e).__name__} from fetch_outputs: {e})")
        return stored_sweep(*stored)'''

# The scoring step on its own, on the embedding one method wrote - run_all's
# tree and fetch_outputs' tree share the layout <out_dir>/<method>_<dataset>/
# <output file>, and the record says which label files the method's cells
# follow. Nothing to score on the stored-table stand-in (no file on disk).
EVALUATE_CELL_TEMPLATE = '''m, emb = "{method}", None
if res.out_dir is not None:
    emb = Path(res.out_dir) / f"{{m}}_{{res.dataset}}" / "embedding.h5"    # what run_all (and the benchmark host) wrote for it
if emb is None or not emb.is_file():
    print("no embedding on this host (metric-table stand-in) - evaluate runs when a method ran here or fetch_outputs supplied the outputs")
    scores = None
else:
    rec = next(r for r in res.results if r["method"] == m)
    order = [Path(f).stem for f in rec.get("labels_used") or []] or None    # the label files in the order the method stacked its cells
    scores = mtb.evaluate(emb, labels=mtb.labels_for(res.dataset), label_order=order, verbose=False)
tick("evaluate")
scores.T if scores is not None else None'''

# The last cell of every notebook: the compact "where the time went" table
# from the ticks - a Colab attendee sees at a glance what the budget went on.
TIMING_CELL = '''pd.DataFrame(TIMES, columns=["stage", "seconds"])'''

SCEN = {
 "vertical": dict(
   ds="D11", cells="2,864",
   blurb=("**Vertical integration** fuses several modalities measured **in the same "
          "cells** (here CITE-seq: RNA + surface protein). Cells are already matched, "
          "so the task is combining modalities, not aligning cells."),
   live=("Matilda", '{"epochs": 5}', "Matilda at 5 epochs is quick - `run_sec` in the summary is the measured time on our host"),
   live_ds=None, summary_note="",
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
   live_ds=None, summary_note="",
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
   live_ds="D46", summary_note="",
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
   summary_note=("UINMF's cross variant consumes batches 1-2 only (its registry "
                 "modalities are `rna1+rna2+adt1+adt2`), so its `emb_shape` counts "
                 "fewer cells than the three-batch methods and its metrics are "
                 "computed on that subset."),
   own_src="D52", own_trio=["UINMF", "sciPENN", "StabMap"],
   own_note="All three reached CHAIN_OK on this subsample when we ran them.",
 ),
}

# ---------------------------------------------------------------- own data
# One executed demo per category: an in-memory AnnData (or several) becomes a
# dataset folder in the layout describe_layout(CATEGORY) prints. export_dataset
# covers the one-AnnData layouts (vertical; cross via batch=); the two layouts
# whose batches hold DIFFERENT cells or DIFFERENT modality sets (diagonal,
# mosaic) are written file by file: to_canonical per matrix, and the label CSV
# (one header line `x`, one label per cell) with pandas - the public label
# writer was retired in 0.3.0 (export_dataset writes label files itself).
EXPORT_DEMO = {
 "vertical": """import anndata as ad, numpy as np, scipy.sparse as sp, tempfile, os
rng = np.random.default_rng(0)
demo = ad.AnnData(X=sp.random(120, 40, density=0.2, random_state=0, format="csr"))  # RNA, cells x genes (sparse is fine)
demo.obsm["protein"] = rng.poisson(3.0, size=(120, 12)).astype(float)             # ADT, cells x proteins
demo.uns["protein_names"] = [f"CD{i}" for i in range(12)]                        # names travel with the obsm key
demo.obs["celltype"] = rng.choice(["T", "B", "NK"], 120)
demo.obs_names = [f"cell{i}" for i in range(120)]; demo.var_names = [f"gene{i}" for i in range(40)]

tmp = tempfile.mkdtemp()
folder = mtb.io.export_dataset(demo, os.path.join(tmp, "MYCITE"),
                               rna="X", adt="obsm:protein", labels="obs:celltype")
print(sorted(os.listdir(folder)))
# 10x multiome held as MuData (peaks in mdata["atac"]) - category="vertical" writes the plain atac.h5
# the vertical roles read (whatever the ATAC representation; check method_info(m)["atac"] for what each wants):
#   mtb.io.export_dataset(mdata, ".../MYMULTIOME", rna="rna", atac="atac", atac_kind="peak",
#                         labels="rna:celltype", category="vertical")
sc = mtb.scan("MYCITE", CATEGORY, data_path=tmp)
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file gate (the rest want an ATAC matrix too)")""",
 "diagonal": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
genes = [f"gene{i}" for i in range(40)]
rna  = ad.AnnData(X=rng.poisson(1.0, size=(120, 40)).astype(float)); rna.var_names = genes
atac = ad.AnnData(X=rng.poisson(0.5, size=(90, 40)).astype(float));  atac.var_names = genes   # gene-activity scores, DIFFERENT cells
rna.obs["celltype"]  = rng.choice(["T", "B", "NK"], 120)
atac.obs["celltype"] = rng.choice(["T", "B", "NK"], 90)

def write_cty(labels, path):                                  # the label file: one header line ("x"), one label per cell
    pd.Series(np.asarray(labels), name="x").to_csv(path, index=False)

folder = os.path.join(tempfile.mkdtemp(), "MYDIAG"); os.makedirs(folder)
mtb.io.to_canonical(rna,  folder, modality="rna")        # -> rna.h5
mtb.io.to_canonical(atac, folder, modality="atac_gas")   # -> atac_gas.h5 (never a plain atac.h5, which methods read as PEAKS)
write_cty(rna.obs["celltype"],  os.path.join(folder, "rna_cty.csv"))
write_cty(atac.obs["celltype"], os.path.join(folder, "atac_cty.csv"))
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYDIAG", CATEGORY, data_path=os.path.dirname(folder))
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file gate (the rest want a peak matrix too)")""",
 "mosaic": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
def batch(n):
    a = ad.AnnData(X=rng.poisson(1.0, size=(n, 40)).astype(float)); a.var_names = [f"gene{i}" for i in range(40)]
    a.obs["celltype"] = rng.choice(["T", "B", "NK"], n); return a
b1, b2, b3 = batch(100), batch(80), batch(60)                                  # three batches, D46's pattern:
b1.obsm["protein"] = rng.poisson(3.0, size=(100, 12)).astype(float)           #   batch 1 = RNA + ADT
b1.uns["protein_names"] = [f"CD{i}" for i in range(12)]
b2.obsm["peaks"]   = rng.poisson(0.3, size=(80, 50)).astype(float)            #   batch 2 = RNA + ATAC, batch 3 = RNA only
b2.uns["peaks_names"] = [f"chr1:{100 * i}-{100 * i + 50}" for i in range(50)]

def write_cty(labels, path):                                                   # the label file: one header line ("x"), one label per cell
    pd.Series(np.asarray(labels), name="x").to_csv(path, index=False)

folder = os.path.join(tempfile.mkdtemp(), "MYMOSAIC"); os.makedirs(folder)
for i, a in enumerate([b1, b2, b3], start=1):
    mtb.io.to_canonical(a, os.path.join(folder, f"rna{i}.h5"))
    write_cty(a.obs["celltype"], os.path.join(folder, f"cty{i}.csv"))
mtb.io.to_canonical(b1, os.path.join(folder, "adt1.h5"),  modality="adt",  obsm="protein")
mtb.io.to_canonical(b2, os.path.join(folder, "atac2.h5"), modality="atac", obsm="peaks")
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYMOSAIC", CATEGORY, data_path=os.path.dirname(folder))
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file gate - the per-batch pattern decides which")""",
 "cross": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
demo = ad.AnnData(X=rng.poisson(1.0, size=(150, 40)).astype(float))            # RNA, cells x genes
demo.var_names = [f"gene{i}" for i in range(40)]
demo.obsm["protein"] = rng.poisson(3.0, size=(150, 12)).astype(float)          # ADT
demo.uns["protein_names"] = [f"CD{i}" for i in range(12)]
demo.obs["celltype"] = rng.choice(["T", "B", "NK"], 150)
demo.obs["batch"]    = rng.choice(["donor1", "donor2", "donor3"], 150)

folder = mtb.io.export_dataset(demo, os.path.join(tempfile.mkdtemp(), "MYCROSS"),
                               rna="X", adt="obsm:protein", labels="obs:celltype",
                               batch="obs:batch")       # batch= -> one numbered file set per batch value
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYCROSS", CATEGORY, data_path=folder.parent)
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file gate (the rest are spatial-registration methods)")""",
}

SUBSAMPLE_FN = '''import os, shutil
import h5py
import numpy as np
import pandas as pd

def subsample_dataset(src_dir, dst_dir, frac=0.6, seed=0, max_cells=2000, max_features=5000):
    """Copy a dataset to a new name, keeping a random slice of it.

    Cells: a random ``frac`` of each file's cells, capped at ``max_cells``.
    Files sharing a cell count get the SAME kept-cell index, so modality files
    and their label CSVs stay aligned - which is exactly the property your own
    export pipeline must preserve. Features: files with more than
    ``max_features`` rows (peak matrices) keep a random subset, so the copy
    stays small and fast to write. The output is the canonical layout:
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
        k = min(max(50, int(n * frac)), max_cells)
        keep[n] = np.sort(rng.choice(n, size=k, replace=False))
    for fn, n in counts.items():
        sp, dp = os.path.join(src_dir, fn), os.path.join(dst_dir, fn)
        idx = keep[n]
        if fn.endswith(".csv"):
            pd.read_csv(sp).iloc[idx].to_csv(dp, index=False)
        else:
            with h5py.File(sp) as f, h5py.File(dp, "w") as g:
                grp = g.create_group("matrix")
                n_feat = f["matrix/data"].shape[0]
                fidx = np.arange(n_feat) if n_feat <= max_features else np.sort(rng.choice(n_feat, size=max_features, replace=False))
                block = f["matrix/data"][fidx, :] if n_feat > max_features else f["matrix/data"][()]   # one fancy index per h5py read
                grp.create_dataset("data", data=np.asarray(block)[:, idx])
                if "matrix/features" in f:
                    grp.create_dataset("features", data=np.asarray(f["matrix/features"])[fidx])
                if "matrix/barcodes" in f:
                    grp.create_dataset("barcodes", data=np.asarray(f["matrix/barcodes"])[idx])
    return dst_dir'''


def _notebook(cells):
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    })


def build_tutorial(cat, s):
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
    trio = s["own_trio"]
    live_methods = sorted({fastm, *trio})
    md("""## 1. Install

`pip install multibench-sc` is the whole install: the 1.3 MB wheel ships the
method registry, the stored result tables, the env lockfiles and the reference
metadata. The package alone runs this section, section 3 (`scan`) and sections
4-5. Running methods (section 2, and `run_all` in section 3) additionally
needs their environments - prebuilt linux-64 archives that need no conda
binary - on a **Linux** host. The flag cell decides: with `INSTALL_ENVS =
False` (the default) **Run all** is safe anywhere - nothing is downloaded,
and a run cell on a host without the environments prints one line and
stands in the benchmark host's own results; set it `True` on Colab or a
Linux machine to download the environments in section 2 and run the methods
for real. The install cell pins `numpy` / `pandas` to what the interpreter
already has, so pip never upgrades a host's stack, and defines `tick`, the
recorder behind the timing table at the end of the notebook.""")
    for cell in INSTALL_CELLS:
        code(cell)
    code(FLAG_CELL_TEMPLATE.format(size=env_size_text(methods=live_methods, short=True)))
    code(f'''%matplotlib inline
import warnings
from pathlib import Path
import anndata
import pandas as pd
from tqdm import TqdmWarning
for _w in (FutureWarning, DeprecationWarning, pd.errors.PerformanceWarning,
           anndata.ImplicitModificationWarning, TqdmWarning):   # library noise only - multibench's own warnings stay visible
    warnings.filterwarnings("ignore", category=_w)
pd.set_option("display.max_colwidth", None)   # never truncate a `reason`
pd.set_option("display.max_columns", None)    # never hide a metric column
pd.set_option("display.width", 200)
import multibench as mtb

DATASET  = "{s['ds']}"
CATEGORY = "{cat}"
mtb.data.fetch({', '.join(repr(d) for d in CAT_DATA[cat])})   # reference data ({', '.join(DS_MB[d] for d in CAT_DATA[cat])}) -> mtb.config.DEFAULT.data_path; no-op when present
print("multibench", mtb.__version__)
tick("data fetch")''')

    # ------------------------------------------------------------ environments
    md(f"""## 2. Run the analysis

### Environments - only if you will run methods in this session

Linux; {env_size_text(methods=live_methods)}. The cell below does nothing
unless `INSTALL_ENVS = True`; then it prints the download the dry run
measures and installs the prebuilt archives for the methods this tutorial
runs ({', '.join(live_methods)}) under `mtb.config.DEFAULT.envs_dir`
(`~/.cache/multibench/envs` on a host without conda - the archives carry
their own interpreters, and `run` activates a prefix directly, so **no
conda binary is needed**, on Colab included; with conda present the prefix
goes to its envs dir and `conda run` is the fallback: `MULTIBENCH_RUN_MODE`
in the API reference). `env.install` skips anything already present. Other
tiers are one flag away on the command line: `multibench env install
--category {cat} --packed --run` ({env_size_text(category=cat)}), or no
flag for the whole benchmark (29 envs; `multibench env plan` totals them).
The archives are linux-64: on macOS / Windows a real install refuses
(`force=True` / `--force` overrides), which is why the cell checks the
platform too. Without the environments, section 3's `scan` and sections 4-5
work as they are.""")
    code(f"""import sys
if not INSTALL_ENVS:
    print("INSTALL_ENVS is False - no environment is downloaded (sections 3-5 need none)")
elif sys.platform != "linux":
    print("method environments are linux-64 archives - skipped on", sys.platform)
else:
    plan = mtb.env.install({trio!r}, category=CATEGORY)              # dry_run=True: the plan, nothing downloaded
    todo = [r for r in plan if not r["exists"]]
    print(f"{{len(todo)}} of {{len(plan)}} envs to download, {{sum(r['archive_bytes'] or 0 for r in todo) / 1e9:.1f}} GB (measured archive sizes)")
    for r in mtb.env.install({trio!r}, category=CATEGORY, packed=True, dry_run=False):   # no conda binary needed
        print(f"{{r['env']:20s}} {{r['state']}}")
tick("environments")""")

    # ------------------------------------------------------------- run + plot
    params_note = f', params={{"{fastm}": {s["live"][1]}}}' if s["live"][1] != "None" else ""
    extra = (f' (on `{live_ds}`, whose layout fits these methods; mosaic layouts '
             f'vary per dataset)' if live_ds != s["ds"] else "")
    si_ds, si_methods = stand_in(cat, ds, trio)
    si_args = f'("{si_ds}",' + (f" {si_methods!r})" if si_methods else ")")
    si_what = (f"the same methods from the package's own sweep of `{si_ds}`"
               if si_methods else f"the package's own sweep of `{si_ds}`")
    md(f"""### One call

One call is the whole pipeline: resolve each method's inputs, run it in its own
environment, load the embeddings, score them with scIB metrics. Here
{', '.join(trio)}{extra}. {s['live'][2]}. The cell first asks `scan` whether
any of these methods has its environment here; where none does (a laptop,
Colab with the flag off) it prints one line and stands in the benchmark
host's own `run_all` outputs for `{live_ds}` - `mtb.data.fetch_outputs`
downloads the tree `run_all` wrote there (summary, long table and one
embedding per method) and `mtb.load_batch(..., methods=)` reloads it, so the
statuses, run times and the evaluate cell below are real. Offline, or before
those outputs are published, a second fallback stands in {si_what}
(`load_results(source="rerun")`, the package's re-execution of every wired
method - the paper's own tables are `source="published"`); either way one
printed line says which path was taken and every cell below still renders.""")
    code(f'''{STORED_SWEEP_FN}

{STAND_IN_FN}

check = mtb.scan("{live_ds}", CATEGORY, methods={trio!r})
if check.env_ok.any():
    res = mtb.run_all("{live_ds}", CATEGORY,
                      methods={trio!r}{params_note},
                      out_dir="/tmp/tutorial_{cat}")
else:
    print("{SKIP_LINE}")
    res = stand_in("{live_ds}", {trio!r}, stored={si_args})
tick("run-or-fetch")
res.summary''')
    md(f"""`summary` is sorted by method name, whatever order `methods=` listed;
`emb_shape` is the embedding each method produced and `batch_source` says which
batch vector the batch metrics used; a stand-in from the benchmark host carries
that host's `status` and `run_sec`, one from the stored metric table says
`STORED` in `status` and leaves the run columns empty. {s['summary_note']}
The scoring step on its own - what `run_all` did per method - is one
`mtb.evaluate` call on the embedding file a method wrote, with the label files
in the order the method stacked its cells (`labels_for` returns them in the
benchmark's stacking order; the record's `labels_used` is what `run_all`
matched). ARI, NMI and iF1 need scIB's Leiden resolution sweep, the slow
part; the sweep's flavor is the `mtb.config.DEFAULT.leiden_flavor` knob
(`"igraph"`, the fast default, or `"leidenalg"`):""")
    code(EVALUATE_CELL_TEMPLATE.format(method=fastm))
    md("""The result object plots itself in the paper's layout:""")
    code("""fig = res.plot()
tick("plot")
fig""")
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
genes); the `mtb.io` writers below handle it, and `scan()` rejects a transposed
file at preflight rather than letting a method fail half an hour in. The label
CSV is a **single column** with one header line (typically `x`) and one label
per cell, in the same order as `matrix/barcodes` of the matching modality file.
For ATAC, check which representation a method wants - gene-activity scores or
peaks - because the wrong one runs to completion and returns a plausible but
wrong embedding. The shipped label files, as `evaluate` will read them - the
dict comes back in the benchmark's **cell-stacking order** (`cty`; `cty1, cty2,
...` numerically; `rna_cty` before `atac_cty`), so
`mtb.evaluate(embedding, labels=mtb.labels_for(DATASET), metrics="all")` scores a
multi-file dataset directly, with each cell's file of origin as its batch.
`metrics=` is the one selection knob: `None` (the default) computes every
applicable metric - the clustering family, plus the batch family whenever a
batch vector exists; `"clustering"` / `"batch"` / `"all"` name a family, and a
list of codes (`["ARI", "NMI"]`) names exactly those (batch metrics need
`batch=` or a multi-file `labels`). A dict you built in any other order must
say so with `label_order=[...]`:""")
    code("""labels = mtb.labels_for(DATASET)            # {stem: path} in cell-stacking order - what the metrics are scored against
print({k: Path(v).name for k, v in labels.items()})
print(*Path(next(iter(labels.values()))).read_text().splitlines()[:4], sep="\\n")""")
    md("""Writing that layout from AnnData objects, executed here on a synthetic
example - sparse matrices stream without densifying, and the folder passes the
same file gate `scan` applies to the shipped datasets:""")
    code(EXPORT_DEMO[cat])
    md(f"""Now a real dataset the package has never seen: a small slice of
`{s['own_src']}` under a new name - a random 60% of the cells capped at 2,000,
and at most 5,000 features per matrix - built with ordinary h5py/pandas code so
the per-file rule (matching files keep the same cell index) is visible. The cap
keeps the copy small enough to write in seconds on Colab (the full peak matrix
of a mosaic dataset is several GB dense).""")
    code(SUBSAMPLE_FN)
    md("""`scan` checks two independent gates per method: `files_ok` - the folder
itself (files present, features x cells orientation, one label per cell) - and
`env_ok` - that method's conda environment exists on this machine. `runnable`
is both; `reason` says which failed and how to fix it. The file gate runs
anywhere, so a laptop without a single environment still tells you whether your
layout is right.""")
    code(f'''DATA_ROOT = "/tmp/mydata"
src = mtb.config.DEFAULT.data_path / "{s['own_src']}"
subsample_dataset(src, f"{{DATA_ROOT}}/MYDATA_{cat}", frac=0.6)

sc = mtb.scan(f"MYDATA_{cat}", category=CATEGORY, data_path=DATA_ROOT)
print(f"files_ok {{int(sc.files_ok.sum())}}, env_ok {{int(sc.env_ok.sum())}}, runnable {{int(sc.runnable.sum())}} of {{len(sc)}} method variants")
sc[["method", "modalities", "files_ok", "env_ok", "runnable", "reason"]].head(6)''')
    own_ds, own_methods = stand_in(cat, ds2, s["own_trio"])
    own_args = f'"{own_ds}"' + (f", {own_methods!r}" if own_methods else "")
    code(f'''if sc[sc.method.isin({s['own_trio']!r})].env_ok.any():
    mine = mtb.run_all(f"MYDATA_{cat}", CATEGORY,
                       methods={s['own_trio']!r},
                       out_dir=f"{{DATA_ROOT}}/out_{cat}",
                       data_path=DATA_ROOT)
else:
    print("{SKIP_LINE}")
    mine = stored_sweep({own_args})   # {own_ds}: the package's own 60% subsample of {own_ds[:-1]} (the same construction, uncapped)
tick("own data")
mine.summary''')
    code("""mine.plot()""")
    md(f"""{s['own_note']} For your real data the only work is producing the
canonical files - `mtb.io.export_dataset` from an AnnData / MuData, or
`to_canonical` per matrix plus a one-column `x` CSV per label set, as above -
and these same three calls do the rest.""")

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

Section 2 ran {n_word}; the package ships the **full sweep** for `{ds}` -
every wired method at default settings, hours of compute - so the paper's
figures reproduce from stored results in seconds. `load_results` reads them
back as the tidy frame `mtb.plot.bubble` takes, and every row says in its
`source` column which sweep it came from. {published_note(cat, ds)}""")
    code(f'''long = mtb.load_results(CATEGORY, dataset=DATASET, source="rerun")
print(long.method.nunique(), "methods,", long.source.unique())
fig = mtb.plot.bubble(long)
fig.set_dpi(110)
tick("stored results")
fig''')
    md(f"""`run_all(DATASET, CATEGORY, out_dir=...)` without `methods=` writes the
same `summary.csv` and `long.csv` for your own data; `mtb.load_batch(out_dir)`
reads them back (`methods=` keeps a subset, as the stand-in above did). A
single `mtb.evaluate` frame becomes the same seven-column
long frame with `mtb.to_long(metrics, method=..., dataset=..., category=...)`
(`source="user"`), so `pd.concat([long, mine])` plots next to the stored
sweep.

**Across datasets.** A summary needs every method to have results on every
dataset it averages over, or absence and performance blur into the same bar. Two
{cat} datasets therefore ship swept identically - `{ds}` and `{ds2}` (a 60% cell
subsample of `{ds}` under a new name) - and `require_complete=True` keeps only
the methods present on both, so the matrix is complete by construction. Each
bar is the **grand rank**: the min-max scaled mean rank across the datasets,
with length and colour both carrying it, and `Overall` is the same statistic
over the grand ranks. {batch_note}""")
    code(f'''pair = mtb.load_results(CATEGORY, dataset=[DATASET, DATASET + "s"], source="rerun")
print(pair.groupby("dataset").method.nunique().to_dict())
mtb.plot.bubble(pair, aggregate="summary", require_complete=True,
                title=f"Summary of 2 {cat} datasets")''')
    md(f"""**Published vs re-run.** The paper's own tables (`source="published"`)
and the package's re-runs can differ for a method - methods are stochastic and
the published sweep ran on other hardware - so cite the published numbers and
use the re-runs to check reproducibility; compare ranks, not decimals.
`results_coverage` says what exists for this dataset and where it came from:""")
    code('''cov = mtb.results_coverage(CATEGORY)
cov[cov.dataset == DATASET].groupby("source").method.nunique()''')

    # ----------------------------------------------------------- reference
    md("""## 5. Reference

### What runs on a dataset, and why not

`scan` inspects a folder and reports every runnable method - and for the rest,
the exact reason (missing file, missing environment, wrong layout). Nothing
executes. From the shell, `multibench scan DATASET --category CATEGORY` prints
the compact table (method, modalities, runnable, files_ok, env_ok,
runtime_tier, reason); `--columns all` or `--format csv|tsv|json` gives every
column - including `command`, the exact shell line `run()` would execute for
every variant whose inputs resolve (the line to paste into a scheduler job;
`multibench run-all ... --dry-run` prints the same table). Below, the rows the
**data** admits (`files_ok`), with the
environment gate next to them: `env_ok` is what this machine has, `env_reason`
the install line for what it lacks, and `runnable` (both gates) stays empty
until the environments exist.""")
    code("""avail = mtb.scan(DATASET, category=CATEGORY)
print(f"files_ok {int(avail.files_ok.sum())}, env_ok {int(avail.env_ok.sum())}, runnable {int(avail.runnable.sum())} of {len(avail)} method variants")
avail[avail.files_ok][["method", "modalities", "env", "env_ok", "env_reason",
                       "output_kind", "needs_labels", "runtime_tier"]]""")
    code("""not_ok = avail[~avail.files_ok][["method", "modalities", "files_reason"]]
not_ok.head(5) if len(not_ok) else "(every method's inputs resolve on this dataset)"
""")
    md("""### What each method exposes for tuning

`method_info(m)["supports"]` lists a method's variants with how many
hyperparameters each exposes on its command line; `mtb.params_for(m, CATEGORY,
modalities)` names them, `params={"Method": {"key": value}}` sets them in
`run_all` (`--param METHOD:KEY=VALUE` on the command line; `multibench params
METHOD` prints the same table for every variant). An empty `tunable` is honest: many upstream scripts hardcode their hyperparameters, and this
package never edits upstream code.""")
    code("""rows = [{"method": m, "modalities": "+".join(v["modalities"]) or "(data_dir)",
         "n_tunable": v["n_tunable"], "needs_labels": v["needs_labels"],
         "output_kind": v["output_kind"]}
        for m in sorted(mtb.list_methods(category=CATEGORY))
        for v in mtb.method_info(m)["supports"] if v["category"] == CATEGORY]
pd.DataFrame(rows).sort_values(["n_tunable", "method"], ascending=[False, True]).reset_index(drop=True)""")
    md(f"""### What the registry knows about a method, and how to cite it

`method_info` carries the upstream reference, repository and version next to
the run metadata - `availability` says whether a public install can run it
(`'public'`, or `'benchmark-host-only'` for SPIRAL, the one method whose
script is not published), `needs_labels` is the any-variant flag (`supports[i]` has it
per variant), and `verbose=True` adds the long audit notes plus
`verification`, the recorded end-to-end run(s) behind `status='verified'`
(`{{dataset, category, status, wall_s, ARI, baseline, verdict, note}}`);
`mtb.cite` emits the benchmark entry plus one per method you ran.""")
    code(f'''info = mtb.method_info("{fastm}", verbose=True)
{{k: info[k] for k in ("id", "env", "availability", "needs_labels", "atac", "notes", "repo_url", "version", "reference", "verification")}}''')
    code(f'''print(mtb.cite({trio!r}))   # one line per entry; fmt="bibtex" for the .bib entries''')
    md("""### The metrics

Two families, matching the paper's grouping. All are **higher = better**, on
[0, 1] except ARI (slightly negative at chance level).

| family | metrics | what they measure |
|---|---|---|
| clustering / bio-conservation | `ARI`, `NMI`, `ASW`, `iASW`, `iF1`, `cLISI` | does the embedding separate the annotated cell types? |
| batch correction | `ASW_batch`, `GC`, `iLISI` (+ opt-in `kBET`) | are the batches mixed within each cell type? |

Batch metrics appear only when the dataset has real batches - their absence on a
single-batch dataset is correct, not missing data. `kBET` is computed only when
named (`mtb.evaluate(..., metrics=["ASW_batch", "GC", "iLISI", "kBET"])`)
because it is much slower than the rest.""")
    md(f"""### Coverage of the paper

`scan()` answers "what runs on this dataset"; this answers how many of the
methods the paper benchmarks for **{cat}** the package wires at all. For each
one it does not, the registry's own record is printed - the categories it
**is** wired for and the tasks (`mtb.list_tasks()` vocabulary) its variants
are registered under; no variant for this category means the paper scored it
here through a task the package does not run.""")
    code(f"""PAPER = {PAPER_METHODS!r}

paper = PAPER[CATEGORY]
registry = set(mtb.list_methods())
wired = sorted(m for m in registry
               if any(v["category"] == CATEGORY for v in mtb.method_info(m)["supports"]))
missing = [m for m in paper if m not in wired]
print(f"paper benchmarks {{len(paper)}} methods for {{CATEGORY}}; this package wires {{len(wired)}}")
for m in missing:
    if m in registry:
        info = mtb.method_info(m)
        print(f"  {{m}}: wired for {{', '.join(info['categories'])}} only (registered tasks: {{', '.join(info['tasks'])}})")
    else:
        print(f"  {{m}}: not in the registry")
if not missing:
    print("full parity with the paper for this category")""")

    # -------------------------------------------------------- troubleshooting
    trouble_extra = ""
    if cat == "cross":
        trouble_extra = """
### Spatial registration

`PASTE`, `PASTE2`, `SPIRAL` and `GPSA` are cross-integration methods whose output
is **aligned spatial coordinates**, not an embedding - their status reports
`RUN_OK_NO_EMBEDDING` and clustering metrics genuinely do not apply. They take
a directory of per-slice `.h5ad` files (`.X` + `obsm['spatial']`); the SPATIAL
REGISTRATION block of `describe_layout("cross")` above is the contract, and
`scan` checks the directory for it. `PASTE`, `PASTE2` and `GPSA` are public
(GPSA through the package driver `engine/drivers/run_gpsa.py`); `SPIRAL` is
`availability="benchmark-host-only"` - its script is not published, so `scan`
reports it `benchmark-host-only: script not published` and
`mtb.find_methods(task="registration", available=True)` omits it. GPSA
additionally reads `obs['Ground_Truth']` (a region / layer label per spot)
from every slice: it drives GPSA's own PAA / LTARI / SCS scores, written to
`<out_dir>/GPSA_aligned_slices/<data_dir_name>_metrics.csv`. `mtb.run` stages
the slices as sorted, zero-padded symlinks under `<out_dir>/inputs/` and writes
`<out_dir>/slices_manifest.json` mapping each `aligned_slice_<i>` to its source
file (the scripts glob without sorting; the manifest is the authority)."""
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
| `files_ok` False: input files not found | a required file is absent | the reason names the exact file and lists what IS in the folder |
| `env_ok` False | that method's conda env is not built | the reason carries the one-method `multibench env install ...` command |
| `... looks like cells x features` | matrix stored transposed | re-export with `mtb.io.to_canonical` / `export_dataset` |
| a method FAILs in seconds | wrong input representation or layout | read `res.failures.iloc[0]["error"]` - the full command line and stderr tail are there |
| a method TIMEOUTs | slow, not broken | raise `timeout=`; runtime tiers in `scan` are measured, not guessed |
| `label_order_confidence` low | several label files fit the cell count | check `label_order_candidates` in the record |
| batch metrics scored against the wrong batches | `batch_source="file_of_origin"` is the label-file rule | `res.rescore(batch=my_vector)` re-scores the stored outputs without re-running |
{trouble_extra}

## Next steps

- the other three tutorials: {siblings}
- the hosted interactive explorer: <https://shiny.maths.usyd.edu.au/scMultiBench/> -
  the full benchmark's rankings, browsable without installing anything
- `mtb.recommend(CATEGORY, modalities=[...])` - stored-result ranking with coverage made explicit; methods wired for the category but absent from the chosen `source` appear with `grand_score` NaN rather than vanishing
- `mtb.sweep(...)` - one method over a range of one hyperparameter

## Where the time went

Seconds per stage, as `tick` recorded them (the install line is ~0 on a
kernel that already had the package):""")
    code(TIMING_CELL)
    return C


def build_colab_quickstart():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    md("""# scMultiBench API quickstart (Colab)

This notebook runs **entirely in Colab**: it installs the `multibench` API,
explores the method registry, loads the shipped benchmark results, and draws
the standard figures - `pip install multibench-sc` is the whole install (the
wheel ships the registry, the stored tables and the references; the cell
pins `numpy` / `pandas` to the versions Colab already has, so nothing is
upgraded, and defines `tick`, the recorder behind the timing table at the
end).

One scope note: *executing* an integration method needs that method's
environment (a prebuilt linux-64 archive - no conda binary needed;
`multibench env plan` prints the sizes) and reference data, which this
notebook does not download - for that, open a category tutorial and set its
`INSTALL_ENVS` flag. Everything below runs here, now.""")
    for cell in INSTALL_CELLS:
        code(cell)
    code("""%matplotlib inline
import multibench as mtb
import pandas as pd
tick("import")

print(len(mtb.list_methods()), "methods in the registry")
mtb.list_methods(category="vertical")""")
    md("""## Inspect a method

`method_info` returns everything the registry knows - language, environment,
availability, label needs, the upstream reference and the entry-point variants;
`find_methods` filters by what your data has (the filters hold per variant: a
method matches when one of its variants satisfies category, modalities,
`needs_labels` and `atac` together); `cite` emits the benchmark entry plus one
per method.""")
    code("""info = mtb.method_info("Matilda")
{k: info[k] for k in ("id", "language", "env", "availability", "needs_labels", "notes", "repo_url", "reference", "supports")}""")
    code("""mtb.find_methods(category="vertical", modalities=["rna", "adt"], needs_labels=False)""")
    code("""print(mtb.cite(["Matilda"]))   # one line per entry; fmt="bibtex" for the .bib entries""")
    md("""## Load shipped results and draw the standard figures

The package ships the paper's tables (`source="published"`) and its own
re-run sweeps (`source="rerun"`), one long table per dataset - so the figures
reproduce here without running anything.""")
    code("""long = mtb.load_results("vertical", dataset="D11", source="rerun")
fig = mtb.plot.bubble(long)
fig.set_dpi(110)
tick("stored results")
fig""")
    code("""pair = mtb.load_results("diagonal", dataset=["D28", "D28s"], source="rerun")
fig = mtb.plot.bubble(pair, aggregate="summary", require_complete=True,
                      title="Summary of 2 diagonal datasets")
tick("plot")
fig""")
    md("""## Next steps

- the four integration tutorials (vertical / diagonal / mosaic / cross) in the
  docs walk the full pipeline, including your own dataset
- the hosted [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/)
  has the complete published rankings

## Where the time went

Seconds per stage, as `tick` recorded them:""")
    code(TIMING_CELL)
    return C


if __name__ == "__main__":
    for cat, s in SCEN.items():
        C = build_tutorial(cat, s)
        path = os.path.join(OUT, f"tutorial_{cat}.ipynb")
        nbf.write(_notebook(C), path)
        print(f"wrote {path} {len(C)} cells")
    C = build_colab_quickstart()
    path = os.path.join(OUT, "colab_quickstart.ipynb")
    nbf.write(_notebook(C), path)
    print(f"wrote {path} {len(C)} cells")
