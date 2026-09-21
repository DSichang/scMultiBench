"""Generate the four integration tutorials (vertical / diagonal / mosaic / cross)
and the Colab quickstart.

Order follows what a reader does: install, run the analysis and get a figure,
run the same calls on their own data, read stored results back. Reference
material (scan, tunables, metrics, coverage) sits at the end.

Prose has two layers. Visible: 1-3 short sentences per section - what the
step does and what the reader must do or decide - then the code. Collapsed
(``details()``, a ``<details>`` block): options, caveats, platform notes, as
short labelled paragraphs or lists. Internal mechanics, provenance and
anything the API reference already states are left out; nothing is said
twice on a page. The notebooks are regenerated from this file - never
hand-edited - and executed on the benchmark host afterwards.

"Run all" is safe on any host: the method-environment download sits behind
``INSTALL_ENVS`` (default False) and a Linux check; the run cells call
``run_all`` only when ``scan`` finds an environment, else stand in the
benchmark host's real ``run_all`` outputs (``mtb.data.fetch_outputs``), else
the stored metric table. The install cell pins numpy and pandas to what the
interpreter already has, so pip never upgrades a host's stack (Colab pins
pandas itself), and nothing provisions conda: the packed method environments
run without a conda binary.
"""
import nbformat as nbf
import os

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)

# Method sets benchmarked per category in the paper (Nature Methods 22:2449-2460
# and the PYangLab/scMultiBench README), for the tasks this package covers, so
# each tutorial states its own coverage instead of letting the reader assume
# parity.
PAPER_METHODS = {
 "vertical": ["totalVI","sciPENN","Concerto","scMSI","Matilda","MOFA2","Multigrate",
              "UINMF","scMoMaT","Seurat_WNN","scMM","scMDC","moETM","VIMCCA",
              "iPOLNG","MIRA","UnitedNet","scMVP"],
 "diagonal": ["scBridge","Portal","SCALEX","VIPCCA","Seurat_v3","MultiMAP","Seurat_v5",
              "sciCAN","Conos","iNMF","online_iNMF","scJoint","GLUE","uniPort"],
 "mosaic":   ["MultiVI","scMoMaT","StabMap","Cobolt","UINMF","Multigrate","SMILE",
              "scMM","moETM","UnitedNet","totalVI","sciPENN"],
 "cross":    ["totalVI","scMoMaT","UnitedNet","sciPENN","Concerto","scMDC","StabMap",
              "UINMF","scMM","MOFA2","Multigrate"],
}


def details(*paras, label="Details"):
    """A collapsed block of markdown paragraphs. The blank line after
    ``<summary>`` and before ``</details>`` is what makes Jupyter, Colab and
    mkdocs-jupyter render the inside as markdown instead of literal text."""
    body = "\n\n".join(p.strip() for p in paras if p and p.strip())
    return f"<details>\n<summary>{label}</summary>\n\n{body}\n\n</details>"


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


def stored_method_counts(cat, dataset):
    """``(n_published, n_rerun)``: methods each stored source holds for the
    dataset, counted from ``results_coverage`` at generation time."""
    import multibench as mtb
    cov = mtb.results_coverage(cat)
    cov = cov[cov.dataset == dataset]
    return (cov[cov.source == "published"].method.nunique(),
            cov[cov.source.str.startswith("rerun")].method.nunique())


def other_datasets(cat, used):
    """Dataset ids with a stored table for the category beyond the ones the
    tutorial uses, read from ``available_datasets`` at generation time."""
    import multibench as mtb
    return [d for d in mtb.available_datasets(cat, source="both") if d not in used]


def published_note(cat, dataset):
    """Why every ``load_results`` call in a tutorial names its ``source``,
    counted from ``results_coverage`` at generation time."""
    n_pub, n_rerun = stored_method_counts(cat, dataset)
    if n_pub == 0:
        return (f"**One source.** There are no scIB tables for {cat} under "
                f"`source=\"published\"`, the default, which raises "
                f"`FileNotFoundError`; `source=\"rerun\"` is the only stored source.")
    return (f"**Two sources.** For `{dataset}`, the `published` table holds {n_pub} "
            f"method{'s' if n_pub != 1 else ''} and the package's own runs "
            f"(`\"rerun\"`) hold {n_rerun}. `load_results` defaults to "
            f"`source=\"published\"`, so every call here names its source. Where "
            f"both hold a method, the values can differ.")


def env_size_text(category=None, methods=None):
    """``"<n> envs, <x> GB to download on a CPU host, <y> GB on a GPU host"``
    for a category / method set, from the dry-run plans ``mtb.env.install(...)``
    returns at generation time for each archive flavour - never a hand-written
    GB figure. "at least" when an archive size is not measured."""
    import multibench as mtb
    sizes = {}
    for flavor in ("cpu", "gpu"):
        rows = mtb.env.install(methods, category=category, flavor=flavor)   # dry_run=True: nothing built
        known = [r["archive_bytes"] for r in rows if r["archive_bytes"]]
        sizes[flavor] = (("" if len(known) == len(rows) else "at least ")
                         + f"{sum(known) / 1e9:.1f} GB")
    n = len(rows)
    return (f"{n} env{'s' if n != 1 else ''}, {sizes['cpu']} to download on a CPU host, "
            f"{sizes['gpu']} on a GPU host")


def download_size(datasets):
    """The reference-data download size, from the package's own table."""
    from multibench.data.fetch import AVAILABLE
    return " + ".join(AVAILABLE[d] for d in datasets)


def and_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def reordering_methods(cat, dataset):
    """``{method: [stem, ...]}`` for the methods whose
    ``labels_for(dataset, cat, method)`` differs from the default
    ``labels_for(dataset)``, read from the live package at generation time.
    Needs the dataset on disk (``config.DEFAULT.data_path``); a missing one
    raises instead of dropping the list from the tutorial."""
    import multibench as mtb
    default = list(mtb.labels_for(dataset))
    orders = {m: list(mtb.labels_for(dataset, cat, m)) for m in sorted(mtb.list_methods(cat))}
    return {m: o for m, o in orders.items() if o != default}


CAT_DATA = {"vertical": ["D11"], "diagonal": ["D28"],
            "mosaic": ["D45", "D46"], "cross": ["D52"]}

# The ONE install cell every notebook shares (tests/test_docs_consistency.py
# pins it): the package WITH its dependencies (evaluate() needs scib/scanpy).
# The find_spec guard keeps the cell idempotent and leaves a developer's
# editable install alone; the numpy / pandas pins keep pip from upgrading the
# host's stack; the GitHub line covers a PyPI release that lags the docs.
INSTALL_CELLS = [
"""import importlib.metadata, importlib.util, sys
if importlib.util.find_spec("multibench") is None:
    # keep the numpy / pandas this interpreter already has
    pins = [f"{p}=={importlib.metadata.version(p)}" for p in ("numpy", "pandas") if importlib.util.find_spec(p)]
    !{sys.executable} -m pip -q install "multibench-sc>=0.3" {" ".join(pins)}
    importlib.invalidate_caches()
    if importlib.util.find_spec("multibench") is None:        # not on PyPI: install from GitHub
        !{sys.executable} -m pip -q install "git+https://github.com/DSichang/scMultiBench.git" {" ".join(pins)}
else:
    print("multibench already installed")""",
]

# The "Run all" switch (tests/test_tutorial_runall_safety.py pins it): the
# one download of method environments - `mtb.env.install(..., dry_run=False)`
# - sits behind INSTALL_ENVS. The size is measured at generation time.
FLAG_CELL_TEMPLATE = """# False: no environment is downloaded; method cells use stand-in results.
# True (Linux or Colab): run the methods here; no conda needed.
# {size}.
INSTALL_ENVS = False"""

# The one line a run cell prints on a host without method environments
# (tests pin the phrase), before standing in a result computed elsewhere.
SKIP_LINE = ("no method environment on this host - the run is skipped; "
             "a stand-in computed elsewhere covers it")

# The LAST fallback of the run cells: a BatchResult built from the stored
# sweep's rows has the same .summary / .plot() as run_all's, so every later
# cell renders; status='STORED' says nothing ran and nothing is on disk.
STORED_SWEEP_FN = '''def stored_sweep(dataset, methods=None):
    """The stored results for `dataset`, as the object `run_all` returns."""
    long = mtb.load_results(CATEGORY, dataset=dataset, source="rerun", methods=methods)
    recs = [{"method": m, "status": "STORED", "metrics": g.set_index("metric")["value"].to_dict()}
            for m, g in long.groupby("method")]
    return mtb.BatchResult(recs, dataset, CATEGORY)'''

# The FIRST fallback: the benchmark host's real run_all output tree for the
# dataset, downloaded by mtb.data.fetch_outputs and reloaded by load_batch,
# so .summary carries real statuses and run times and the evaluate cell
# scores a real embedding. Offline (or before the assets are published) the
# stored metric table stands in; one printed line says which path was taken.
STAND_IN_FN = '''def stand_in(dataset, methods, stored):
    """The benchmark host's run_all outputs for `dataset`; the stored results if that download fails."""
    try:
        res = mtb.load_batch(mtb.data.fetch_outputs(dataset), methods=methods)
        print(f"stand-in: the benchmark host's run_all outputs for {dataset} (fetch_outputs) - real embeddings and run times")
        return res
    except Exception as e:                                   # offline, or the outputs are not published yet
        print(f"stand-in: the package's stored metric table ({type(e).__name__} from fetch_outputs: {e})")
        return stored_sweep(*stored)'''

# The scoring step on its own, on the embedding one method wrote - run_all's
# tree and fetch_outputs' tree share the layout <out_dir>/<method>_<dataset>/
# embedding.h5, and the record says which label files the method's cells
# follow. Nothing to score on the stored-table stand-in (no file on disk).
EVALUATE_CELL_TEMPLATE = '''m, emb = "{method}", None
if res.out_dir is not None:
    emb = Path(res.out_dir) / f"{{m}}_{{res.dataset}}" / "embedding.h5"
if emb is None or not emb.is_file():
    print(f"no embedding on this host for {{m}} - nothing to score")
    scores = None
else:
    rec = next(r for r in res.results if r["method"] == m)
    order = [Path(f).stem for f in rec.get("labels_used") or []] or None    # label files in the method's cell order
    scores = mtb.evaluate(emb, labels=mtb.labels_for(res.dataset), label_order=order, verbose=False)
scores.T if scores is not None else None'''

SCEN = {
 "vertical": dict(
   ds="D11", cells="2,864",
   blurb=("**Vertical integration** combines modalities measured in the **same "
          "cells** (here CITE-seq: RNA + surface protein)."),
   blurb_detail=None,
   live=("Matilda", '{"epochs": 5}'), live_modalities=["rna", "adt"],
   live_ds=None, live_note=None, summary_note=None,
   own_src="D11", own_trio=["Matilda", "sciPENN", "scMM"],
 ),
 "diagonal": dict(
   ds="D28", cells="6,408 RNA + 4,606 ATAC",
   blurb=("**Diagonal integration** combines RNA and ATAC measured in **different "
          "cells**, with no pairing between them. If your RNA and ATAC come from the "
          "same cells (10x multiome), use the vertical tutorial."),
   blurb_detail=("**ATAC input.** Most diagonal methods take ATAC as gene-activity "
                 "scores, computed beforehand with a tool such as Signac or ArchR. A few "
                 "take the peak matrix, alone or alongside the scores."),
   live=("online_iNMF", "None"), live_modalities=None,
   live_ds=None, live_note=None, summary_note=None,
   own_src="D28", own_trio=["online_iNMF", "iNMF", "scJoint"],
 ),
 "mosaic": dict(
   ds="D45", cells="32,151",
   blurb=("**Mosaic integration** combines batches that share only **some** "
          "modalities, for example an RNA-only batch, an ATAC-only batch and a paired "
          "batch that links them. Which methods apply depends on the batch pattern; "
          "`scan` reports it."),
   blurb_detail=None,
   live=("StabMap", "None"), live_modalities=None,
   live_ds="D46",
   live_note=("**Dataset.** These methods run on `D46`, whose batch pattern they "
              "accept. The stored-results fallback shows the `D45` results instead."),
   summary_note=None,
   own_src="D46", own_trio=["StabMap", "scMoMaT"],
 ),
 "cross": dict(
   ds="D52", cells="23,478",
   blurb=("**Cross integration** combines batches that all measure the **same** "
          "modalities; the task is removing batch effects while keeping the "
          "biological structure."),
   blurb_detail=None,
   live=("StabMap", "None"), live_modalities=None,
   live_ds=None, live_note=None,
   summary_note=("**UINMF** uses batches 1 and 2 only, so its `emb_shape` counts "
                 "fewer cells and its metrics cover those cells."),
   own_src="D52", own_trio=["UINMF", "sciPENN", "StabMap"],
 ),
}

# ---------------------------------------------------------------- own data
# One executed demo per category: an in-memory AnnData (or several) becomes a
# dataset folder in the layout describe_layout(CATEGORY) prints. export_dataset
# covers the one-AnnData layouts (vertical; cross via batch=); the two layouts
# whose batches hold DIFFERENT cells or DIFFERENT modality sets (diagonal,
# mosaic) are written file by file: to_canonical per matrix, and the label CSV
# (one header line `x`, one label per cell) with pandas.
EXPORT_INTRO = {
 "vertical": "`mtb.io.export_dataset` writes the layout from an AnnData, here a synthetic one:",
 "diagonal": ("RNA and ATAC hold different cells, so each file is written on its own: "
              "`mtb.io.to_canonical` per matrix and one label CSV per modality. Here on "
              "synthetic data:"),
 "mosaic": ("Each batch has its own set of modalities, so each file is written on its "
            "own: `mtb.io.to_canonical` per matrix and one label CSV per batch. Here on "
            "synthetic data with `D46`'s pattern:"),
 "cross": ("`mtb.io.export_dataset` with `batch=` writes one numbered set of files per "
           "batch, here from a synthetic AnnData:"),
}
EXPORT_DETAIL = {
 "vertical": ("**10x multiome (MuData).** `mtb.io.export_dataset(mdata, path, rna=\"rna\", "
              "atac=\"atac\", atac_kind=\"peak\", labels=\"rna:celltype\", "
              "category=\"vertical\")` writes `rna.h5`, `atac.h5` and `cty.csv`."),
 "diagonal": None, "mosaic": None, "cross": None,
}
EXPORT_DEMO = {
 "vertical": """import anndata as ad, numpy as np, scipy.sparse as sp, tempfile, os
rng = np.random.default_rng(0)
demo = ad.AnnData(X=sp.random(120, 40, density=0.2, random_state=0, format="csr"))  # RNA, cells x genes
demo.obsm["protein"] = rng.poisson(3.0, size=(120, 12)).astype(float)             # ADT, cells x proteins
demo.uns["protein_names"] = [f"CD{i}" for i in range(12)]
demo.obs["celltype"] = rng.choice(["T", "B", "NK"], 120)
demo.obs_names = [f"cell{i}" for i in range(120)]; demo.var_names = [f"gene{i}" for i in range(40)]

tmp = tempfile.mkdtemp()
folder = mtb.io.export_dataset(demo, os.path.join(tmp, "MYCITE"),
                               rna="X", adt="obsm:protein", labels="obs:celltype")
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYCITE", CATEGORY, data_path=tmp)
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file check (the rest want an ATAC matrix too)")""",
 "diagonal": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
genes = [f"gene{i}" for i in range(40)]
rna  = ad.AnnData(X=rng.poisson(1.0, size=(120, 40)).astype(float)); rna.var_names = genes
atac = ad.AnnData(X=rng.poisson(0.5, size=(90, 40)).astype(float));  atac.var_names = genes   # gene-activity scores, other cells
rna.obs["celltype"]  = rng.choice(["T", "B", "NK"], 120)
atac.obs["celltype"] = rng.choice(["T", "B", "NK"], 90)

def write_cty(labels, path):                                  # header line "x", then one label per cell
    pd.Series(np.asarray(labels), name="x").to_csv(path, index=False)

folder = os.path.join(tempfile.mkdtemp(), "MYDIAG"); os.makedirs(folder)
mtb.io.to_canonical(rna,  folder, modality="rna")        # -> rna.h5
mtb.io.to_canonical(atac, folder, modality="atac_gas")   # -> atac_gas.h5
write_cty(rna.obs["celltype"],  os.path.join(folder, "rna_cty.csv"))
write_cty(atac.obs["celltype"], os.path.join(folder, "atac_cty.csv"))
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYDIAG", CATEGORY, data_path=os.path.dirname(folder))
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file check (the rest need a peak matrix)")""",
 "mosaic": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
def batch(n):
    a = ad.AnnData(X=rng.poisson(1.0, size=(n, 40)).astype(float)); a.var_names = [f"gene{i}" for i in range(40)]
    a.obs["celltype"] = rng.choice(["T", "B", "NK"], n); return a
b1, b2, b3 = batch(100), batch(80), batch(60)                                  # batch 1 RNA + ADT, 2 RNA + ATAC, 3 RNA
b1.obsm["protein"] = rng.poisson(3.0, size=(100, 12)).astype(float)
b1.uns["protein_names"] = [f"CD{i}" for i in range(12)]
b2.obsm["peaks"]   = rng.poisson(0.3, size=(80, 50)).astype(float)
b2.uns["peaks_names"] = [f"chr1:{100 * i}-{100 * i + 50}" for i in range(50)]

def write_cty(labels, path):                                                   # header line "x", then one label per cell
    pd.Series(np.asarray(labels), name="x").to_csv(path, index=False)

folder = os.path.join(tempfile.mkdtemp(), "MYMOSAIC"); os.makedirs(folder)
for i, a in enumerate([b1, b2, b3], start=1):
    mtb.io.to_canonical(a, os.path.join(folder, f"rna{i}.h5"))
    write_cty(a.obs["celltype"], os.path.join(folder, f"cty{i}.csv"))
mtb.io.to_canonical(b1, os.path.join(folder, "adt1.h5"),  modality="adt",  obsm="protein")
mtb.io.to_canonical(b2, os.path.join(folder, "atac2.h5"), modality="atac", obsm="peaks")
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYMOSAIC", CATEGORY, data_path=os.path.dirname(folder))
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file check - the batch pattern decides which")""",
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
                               batch="obs:batch")
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYCROSS", CATEGORY, data_path=folder.parent)
print(f"{int(sc.files_ok.sum())} of {len(sc)} method variants pass the file check")""",
}

SUBSAMPLE_FN = '''import os
import h5py
import numpy as np
import pandas as pd

def subsample_dataset(src_dir, dst_dir, frac=0.6, seed=0, max_cells=2000, max_features=5000):
    """Copy a dataset under a new name, keeping a random `frac` of its cells; files with the same cell count keep the same cells."""
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
                block = f["matrix/data"][fidx, :] if n_feat > max_features else f["matrix/data"][()]
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


def live_param_note(method, category, params, modalities):
    """``**Parameters.** ...`` for a run cell that overrides a method's
    defaults, with the defaults read from ``params_for`` at generation time."""
    import ast
    import multibench as mtb
    tunable = mtb.params_for(method, category, modalities)["tunable"]
    parts = [f"{k} {v} instead of its default {tunable[k]['default']}"
             for k, v in ast.literal_eval(params).items()]
    return (f"**Parameters.** `params={{\"{method}\": {params}}}` runs {method} with "
            f"{and_list(parts)}.")


def build_tutorial(cat, s):
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    fastm = s["live"][0]
    live_ds = s["live_ds"] or s["ds"]
    ds, ds2 = s["ds"], s["ds"] + "s"
    trio = s["own_trio"]
    trio_text = and_list(trio)

    # ------------------------------------------------------------------ title
    title = (f"# {cat.capitalize()} integration\n\n{s['blurb']} Reference dataset: "
             f"`{ds}` ({s['cells']} cells).")
    if s["blurb_detail"]:
        title += "\n\n" + details(s["blurb_detail"])
    md(title)

    # ---------------------------------------------------------------- install
    md("## 1. Install\n\n"
       "`pip install multibench-sc` is all this notebook needs. Methods run in their "
       "own environments, downloaded only when you set `INSTALL_ENVS = True` (Linux "
       "or Colab).\n\n"
       + details(
           "**Default.** With `INSTALL_ENVS = False` no environment is downloaded, only "
           "the reference data and stand-in outputs. Every cell runs; a method cell uses "
           "those outputs instead.",
           "**Running methods.** Set `INSTALL_ENVS = True` on Colab or a Linux machine "
           "to download the prebuilt environments (no conda needed) and run the methods.",
           "**On Colab**, choose a GPU runtime first (Runtime -> Change runtime type -> "
           "T4 GPU). On a CPU runtime the smaller CPU builds are installed and training "
           "methods are much slower."))
    for cell in INSTALL_CELLS:
        code(cell)
    size = env_size_text(methods=trio)
    code(FLAG_CELL_TEMPLATE.format(size=size[0].upper() + size[1:]))
    code(f'''%matplotlib inline
import warnings
from pathlib import Path
import anndata
import pandas as pd
from tqdm import TqdmWarning
for _w in (FutureWarning, DeprecationWarning, pd.errors.PerformanceWarning,
           anndata.ImplicitModificationWarning, TqdmWarning):   # library warnings only; multibench's own stay visible
    warnings.filterwarnings("ignore", category=_w)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
import multibench as mtb

DATASET  = "{ds}"
CATEGORY = "{cat}"
mtb.data.fetch({', '.join(repr(d) for d in CAT_DATA[cat])})   # the reference data ({download_size(CAT_DATA[cat])}), downloaded once
print("multibench", mtb.__version__)''')

    # ------------------------------------------------------------ environments
    md(f"""## 2. Run the analysis

### Environments - only if you will run methods

Installs the environments for {trio_text} ({env_size_text(methods=trio)}). The cell skips the download unless `INSTALL_ENVS = True` on Linux.

""" + details(
        "**Location.** Environments are unpacked under `mtb.config.DEFAULT.envs_dir` "
        "(`~/.cache/multibench/envs` on a host without conda); set it before this cell "
        "to use another disk. Environments already there are skipped.",
        f"**All {cat} methods.** `multibench env install --category {cat} --packed --run` "
        f"installs every {cat} environment ({env_size_text(category=cat)}); "
        f"`multibench env plan --category {cat}` lists the size of each."))
    code(f"""import sys
if not INSTALL_ENVS:
    print("INSTALL_ENVS is False - no environment is downloaded")
elif sys.platform != "linux":
    print("method environments are linux-64 archives - skipped on", sys.platform)
else:
    plan = mtb.env.install({trio!r}, category=CATEGORY)              # dry run: sizes only
    todo = [r for r in plan if not r["exists"]]
    print(f"{{len(todo)}} of {{len(plan)}} envs to download, {{sum(r['archive_bytes'] or 0 for r in todo) / 1e9:.1f}} GB")
    for r in mtb.env.install({trio!r}, category=CATEGORY, packed=True, dry_run=False):
        print(f"{{r['env']:20s}} {{r['state']}}")""")

    # ------------------------------------------------------------- run + plot
    params_note = f', params={{"{fastm}": {s["live"][1]}}}' if s["live"][1] != "None" else ""
    si_ds, si_methods = stand_in(cat, ds, trio)
    si_args = f'("{si_ds}",' + (f" {si_methods!r})" if si_methods else ")")
    run_notes = [
        "**Stand-in.** When none of these environments is installed, the cell downloads "
        f"the benchmark host's `run_all` outputs for `{live_ds}` with `mtb.data.fetch_outputs` (embeddings "
        f"and run times included) and reloads them with `mtb.load_batch`.",
        "**Offline fallback.** If that download fails, the cell uses the stored results "
        "(`load_results(source=\"rerun\")`): `status` reads `STORED`, and there is no "
        "embedding to score.",
    ]
    if s["live"][1] != "None":
        run_notes.append(live_param_note(fastm, cat, s["live"][1], s["live_modalities"]))
    run_notes += [s["live_note"], s["summary_note"]]
    md(f"""### Run the methods

`run_all` runs {trio_text} on `{live_ds}`, each in its own environment, and scores each embedding with scIB metrics. Without environments, the cell prints one line and loads stand-in results.

""" + details(*run_notes))
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
res.summary''')
    md("""### Score one embedding

`run_all` has already scored every method. To score one embedding yourself, pass the file the method wrote and the dataset's label files to `mtb.evaluate`:

""" + details(
        "**Label order.** `label_order=` passes the order in which the method stacked "
        "its cells, taken from the run record's `labels_used`.",
        "**Choosing metrics.** `metrics=` accepts:\n\n"
        "- `None` (the default): every applicable metric\n"
        "- `\"clustering\"`, `\"batch\"` or `\"all\"`: a family\n"
        "- a list such as `[\"ARI\", \"NMI\"]`: those metrics",
        "Batch metrics need `batch=` or several label files."))
    code(EVALUATE_CELL_TEMPLATE.format(method=fastm))
    md("""### Plot

`res.plot()` draws a bubble table. Circle size is the method's rank in each column (largest = best); colour is the metric value, scaled within the column (darker = higher).

""" + details(
        "**Columns.** Metrics are grouped by family: blue for dimension reduction and "
        "clustering, green for batch correction. Each family starts with an **Overall** "
        "bar; its length and colour both show the family score."))
    code("""res.plot()""")

    # ------------------------------------------------------------- own data
    md(f"""## 3. Your own data

The same calls on a dataset folder the package has not seen. `describe_layout` prints the files a {cat} dataset needs and their format:""")
    code("""print(mtb.describe_layout(CATEGORY))""")
    labels_code = """labels = mtb.labels_for(DATASET)            # {file stem: path}
print({k: Path(v).name for k, v in labels.items()})
print(*Path(next(iter(labels.values()))).read_text().splitlines()[:4], sep="\\n")"""
    if cat == "vertical":
        md("`labels_for` returns a dataset's label files; a vertical dataset has one, `cty`:")
    else:
        files = "`rna_cty` then `atac_cty`" if cat == "diagonal" else "`cty1`, `cty2`, ... in that order"
        others = reordering_methods(cat, ds)
        if not others:
            raise SystemExit(f"no {cat} method stacks {ds}'s cells in another order: reword section 3")
        md(f"""`labels_for` returns a dataset's label files: {files}. Some methods stack their cells in another order; `labels_for(DATASET, CATEGORY, method)` returns the files in that method's order. A wrong order gives wrong scores without an error.

""" + details(
            f"**On `{ds}`**, `labels_for` returns another order for "
            + and_list(f"{m} (`{', '.join(o)}`)" for m, o in others.items()) + ".",
            "**With `evaluate`.** Pass the dict `labels_for` returns as is. A dict you "
            "build or reorder yourself goes in as is only in the default order; name "
            "any other order with `label_order=`.",
            "**Check.** `run_all` scores every order that fits the cell count and keeps "
            "the one with the highest ARI; the `label_order` column of `res.summary` "
            "shows it.", label="Details: label order"))
        m0 = next(iter(others))
        labels_code += f'\nprint("{m0}:", list(mtb.labels_for(DATASET, CATEGORY, "{m0}")))'
    code(labels_code)
    md(EXPORT_INTRO[cat] + ("\n\n" + details(EXPORT_DETAIL[cat], label="Details: MuData")
                            if EXPORT_DETAIL[cat] else ""))
    code(EXPORT_DEMO[cat])
    md(f"""A real dataset under a new name: a random 60% of `{s['own_src']}`'s cells, capped at 2,000 cells and 5,000 features per file.

""" + details(
        "**Alignment.** Files with the same number of cells keep the same cells in the "
        "same order, so each modality file stays aligned with its label file. An export "
        "of your own data must keep this alignment.", label="Details: cell alignment"))
    code(SUBSAMPLE_FN)
    md("""`scan` checks each method against the folder (`files_ok`) and against this machine's environments (`env_ok`); `runnable` needs both, and `reason` says what failed. The folder check works on any machine.""")
    code(f'''DATA_ROOT = "/tmp/mydata"
src = mtb.config.DEFAULT.data_path / "{s['own_src']}"
subsample_dataset(src, f"{{DATA_ROOT}}/MYDATA_{cat}", frac=0.6)

sc = mtb.scan(f"MYDATA_{cat}", category=CATEGORY, data_path=DATA_ROOT)
print(f"files_ok {{int(sc.files_ok.sum())}}, env_ok {{int(sc.env_ok.sum())}}, runnable {{int(sc.runnable.sum())}} of {{len(sc)}} method variants")
sc[["method", "modalities", "files_ok", "env_ok", "runnable", "reason"]].head(6)''')
    own_ds, own_methods = stand_in(cat, ds2, trio)
    own_args = f'"{own_ds}"' + (f", {own_methods!r}" if own_methods else "")
    code(f'''if sc[sc.method.isin({trio!r})].env_ok.any():
    mine = mtb.run_all(f"MYDATA_{cat}", CATEGORY,
                       methods={trio!r},
                       out_dir=f"{{DATA_ROOT}}/out_{cat}",
                       data_path=DATA_ROOT)
else:
    print("{SKIP_LINE}")
    mine = stored_sweep({own_args})   # stored results for {own_ds}, a 60% subsample of {own_ds[:-1]}
mine.summary''')
    code("""mine.plot()""")

    # ------------------------------------------------------------- figures
    n_rerun = stored_method_counts(cat, ds)[1]
    extra = other_datasets(cat, {ds, ds2})
    stored_notes = [published_note(cat, ds)]
    if extra:
        stored_notes.append(
            f"**Other datasets.** Stored results also exist for "
            f"{and_list(f'`{d}`' for d in extra)} (`mtb.available_datasets(CATEGORY, "
            f"source=\"both\")`).")
    stored_notes.append(
        "**Your own runs.** `run_all` saves its results in `out_dir`; "
        "`mtb.load_batch(out_dir)` reloads them later without running anything.")
    md(f"""## 4. Stored results

The package ships stored results for {n_rerun} methods on `{ds}` (`source="rerun"`). `load_results` reads them as a long table and `mtb.plot.bubble` draws it; nothing is run.

""" + details(*stored_notes, label="Details: sources"))
    code('''long = mtb.load_results(CATEGORY, dataset=DATASET, source="rerun")
print(long.method.nunique(), "methods,", long.source.unique())
fig = mtb.plot.bubble(long)
fig.set_dpi(110)
fig''')
    pair_notes = [
        "**Bars.** Each metric bar is the method's rank averaged over the datasets, "
        "min-max scaled; `Overall` summarises the family's metric ranks. Bar length and "
        "colour both show the value.",
        "**Missing methods.** Without `require_complete=True`, a method absent from a "
        "dataset gets the lowest rank there, which pulls its bars down.",
    ]
    md(f"""**Across datasets.** `aggregate="summary"` ranks the methods over `{ds}` and `{ds2}`, a 60% cell subsample of `{ds}`; `require_complete=True` keeps only the methods with results on both.

""" + details(*pair_notes, label="Details: summary bars"))
    code(f'''pair = mtb.load_results(CATEGORY, dataset=[DATASET, DATASET + "s"], source="rerun")
print(pair.groupby("dataset").method.nunique().to_dict())
mtb.plot.bubble(pair, aggregate="summary", require_complete=True,
                title=f"Summary of 2 {cat} datasets")''')
    if stored_method_counts(cat, ds)[0]:
        md("""`results_coverage` counts the methods each stored source holds for this dataset:""")
    else:       # one stored source (mosaic)
        md("""`results_coverage` counts the methods stored for this dataset:""")
    code('''cov = mtb.results_coverage(CATEGORY)
cov[cov.dataset == DATASET].groupby("source").method.nunique()''')

    # ----------------------------------------------------------- reference
    md(f"""## 5. Reference

### What runs on a dataset, and why not

`scan` on `{ds}`, nothing run: the first table lists the variants whose data fits, with the install command for any missing environment (`env_reason`); the second, why the other variants do not fit.

""" + details(
        f"**From the shell.** `multibench scan {ds} --category {cat}` prints the scan for "
        "every variant; `--columns all` adds every column, including `command`, the exact "
        "command `run` would execute."))
    code("""avail = mtb.scan(DATASET, category=CATEGORY)
print(f"files_ok {int(avail.files_ok.sum())}, env_ok {int(avail.env_ok.sum())}, runnable {int(avail.runnable.sum())} of {len(avail)} method variants")
avail[avail.files_ok][["method", "modalities", "env", "env_ok", "env_reason",
                       "output_kind", "needs_labels", "runtime_tier"]]""")
    code("""not_ok = avail[~avail.files_ok][["method", "modalities", "files_reason"]]
not_ok.head(5) if len(not_ok) else "(every method's inputs resolve on this dataset)"
""")
    md("""### Tuning

The table counts the hyperparameters each variant exposes. `mtb.params_for(method, CATEGORY, modalities)` lists them; `run_all(..., params={"Method": {"key": value}})` sets them.

""" + details(
        "**None exposed.** Many upstream scripts set their hyperparameters in code; "
        "their variants accept no `params`.",
        "**From the shell.** `multibench params METHOD` prints the table; "
        "`multibench run-all ... --param METHOD:KEY=VALUE` sets a value."))
    code("""rows = [{"method": m, "modalities": "+".join(v["modalities"]) or "(data_dir)",
         "n_tunable": v["n_tunable"], "needs_labels": v["needs_labels"],
         "output_kind": v["output_kind"]}
        for m in sorted(mtb.list_methods(category=CATEGORY))
        for v in mtb.method_info(m)["supports"] if v["category"] == CATEGORY]
pd.DataFrame(rows).sort_values(["n_tunable", "method"], ascending=[False, True]).reset_index(drop=True)""")
    md("""### A method's record and citation

`method_info` returns what the registry holds about a method, including its reference and repository; `mtb.cite` returns the citations for the benchmark and the methods you ran.

""" + details(
        "**needs_labels** is True when any variant needs cell-type labels; each entry of "
        "`supports` gives it per variant.",
        "**verbose=True** adds the long notes."))
    code(f'''info = mtb.method_info("{fastm}", verbose=True)
{{k: info[k] for k in ("id", "env", "needs_labels", "atac", "notes", "repo_url", "version", "reference")}}''')
    code(f'''print(mtb.cite({trio!r}))   # fmt="bibtex" for BibTeX entries''')
    md("""### The metrics

Two families; higher is better for every metric.

| family | metrics | measures |
|---|---|---|
| clustering / bio-conservation | `ARI`, `NMI`, `ASW`, `iASW`, `iF1`, `cLISI` | whether the embedding separates the annotated cell types |
| batch correction | `ASW_batch`, `GC`, `iLISI` (+ opt-in `kBET`) | whether the batches mix within each cell type |

""" + details(
        "**Range.** All lie in [0, 1] except ARI, which can be slightly negative.",
        "**Batch metrics** appear only when the dataset has more than one batch.",
        "**kBET** is computed only when named (`metrics=[\"ASW_batch\", \"GC\", "
        "\"iLISI\", \"kBET\"]`); it is much slower than the others."))
    coverage_notes = []
    if cat == "mosaic":
        coverage_notes.append(
            "**UINMF** has no mosaic variant: its script takes the second batch's "
            "unshared features from the first batch, which fails when the two are "
            "different modalities, and it accepts exactly two batches, a pattern that "
            "fits no mosaic dataset here.")
    md(f"""### Methods from the benchmark study

The cell compares the methods the scMultiBench study benchmarked for {cat} integration with the methods this package has a {cat} variant for, and prints each missing method with the categories it has variants for.""" + ("\n\n" + details(*coverage_notes) if coverage_notes else ""))
    code(f"""paper = {PAPER_METHODS[cat]!r}   # benchmarked for {cat} on the tasks this package covers
registry = set(mtb.list_methods())
wired = sorted(m for m in registry
               if any(v["category"] == CATEGORY for v in mtb.method_info(m)["supports"]))
missing = [m for m in paper if m not in wired]
print(f"the study benchmarks {{len(paper)}} methods for {{CATEGORY}} on the tasks this package covers; this package has a variant for {{len(wired)}}")
for m in missing:
    if m in registry:
        info = mtb.method_info(m)
        print(f"  {{m}}: variants for {{', '.join(info['categories'])}} only (tasks: {{', '.join(info['tasks'])}})")
    else:
        print(f"  {{m}}: not in the registry")
if not missing:
    print("every benchmarked method has a variant for this category")""")

    # -------------------------------------------------------- troubleshooting
    siblings = ", ".join(f"**{c}**" for c in SCEN if c != cat)
    md("""## Troubleshooting

When a method is not runnable, `scan`'s `reason` column says why; when a run fails, `res.failures` holds the error.

""" + details(
        """| symptom | fix |
|---|---|
| `files_ok` False: input files not found | the reason names the missing file and lists what the folder holds |
| `env_ok` False | run the `multibench env install ...` command in the reason |
| `... which is cells x features` | the matrix is transposed: re-export with `mtb.io.to_canonical` or `export_dataset` |
| a method fails | `res.failures.iloc[0]["error"]` ends with the method's stderr |
| a method times out | raise `timeout=` in `run_all` |
| low `label_order_confidence` | several label files fit the cell count: check `label_order_candidates` in `res.results` |
| batch metrics use the wrong batches | `res.rescore(batch=my_vector)` re-scores without re-running |"""))
    md(f"""## Next steps

- the other tutorials: {siblings}
- the [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/): the full benchmark's rankings, no install needed
- `mtb.recommend(CATEGORY, modalities=[...])`: a ranking of methods from the stored results
- `mtb.sweep(...)`: one method over a range of values of one hyperparameter""")
    return C


def build_colab_quickstart():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    md("""# scMultiBench API quickstart (Colab)

Installs `multibench-sc`, looks up methods in the registry and draws the stored benchmark results. No method is run, so nothing else is downloaded.

""" + details(
        "**Running methods** needs each method's environment and the reference data. "
        "The category tutorials download the data, and the environments when you set "
        "`INSTALL_ENVS = True`."))
    for cell in INSTALL_CELLS:
        code(cell)
    code("""%matplotlib inline
import multibench as mtb

print(len(mtb.list_methods()), "methods in the registry")
mtb.list_methods(category="vertical")""")
    md("""## Inspect a method

`method_info` returns what the registry holds about a method, `find_methods` filters methods by what your data has, and `cite` returns the citations.

""" + details(
        "**Filters.** A method matches when one of its variants meets every filter "
        "(category, modalities, `needs_labels`, `atac`)."))
    code("""info = mtb.method_info("Matilda")
{k: info[k] for k in ("id", "language", "env", "needs_labels", "notes", "repo_url", "reference", "supports")}""")
    code("""mtb.find_methods(category="vertical", modalities=["rna", "adt"], needs_labels=False)""")
    code("""print(mtb.cite(["Matilda"]))   # fmt="bibtex" for BibTeX entries""")
    md("""## Draw the stored results

The package ships stored results, so these figures draw without running anything.

""" + details(
        "**Sources.** `source=\"published\"` (the default) reads the published scIB "
        "tables; `source=\"rerun\"` reads the package's own runs of the methods.",
        "**Two datasets.** `aggregate=\"summary\"` ranks the methods over both datasets; "
        "`require_complete=True` keeps only the methods with results on both."))
    code("""long = mtb.load_results("vertical", dataset="D11", source="rerun")
fig = mtb.plot.bubble(long)
fig.set_dpi(110)
fig""")
    code("""pair = mtb.load_results("diagonal", dataset=["D28", "D28s"], source="rerun")
mtb.plot.bubble(pair, aggregate="summary", require_complete=True,
                title="Summary of 2 diagonal datasets")""")
    md("""## Next steps

- the four integration tutorials (vertical, diagonal, mosaic, cross) run the full pipeline, including on your own data
- the [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/) has the full benchmark's rankings""")
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
