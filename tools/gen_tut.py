"""Generate the four integration tutorials (vertical / diagonal / mosaic / cross)
and the Colab quickstart.

Order follows what a reader does: install, run the analysis and get a figure,
run the same calls on their own data, read stored results back. Reference
material (scan, tunables, metrics, coverage) sits at the end.

Prose has two layers. Visible: 1-3 short sentences per section - what the
step does and what the reader must do or decide - then the code. A fact
without which a reader gets a silently wrong result (raw counts, which
category fits, what runs off Linux, which ATAC form a method reads) is
always visible. Collapsed (``details()``, a ``<details>`` block): options,
caveats, platform notes, as short plain paragraphs whose first sentence
names the subject, or lists. No bold lead-in labels in a block of 1-3
paragraphs, no capitals for emphasis, no internal names, short sentences.
The notebooks are regenerated from this file - never hand-edited - and
executed on the benchmark host afterwards.

"Run all" is safe on any host: the method-environment download sits behind
``INSTALL_ENVS`` (default False) and a Linux check; the run cells call
``run_all`` only when ``scan`` finds an environment, else load a replacement:
the benchmark host's real ``run_all`` outputs (``mtb.data.fetch_outputs``),
else the stored metric table. The install cell pins numpy and pandas to what
the interpreter already has, so pip never upgrades a host's stack (Colab pins
pandas itself), and nothing provisions conda: the packed method environments
run without a conda binary.
"""
import hashlib
import os

import nbformat as nbf

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)

# Method sets the paper (Nature Methods 22:2449-2460 and the PYangLab/scMultiBench
# README) benchmarks per category on the tasks named in PAPER_TASKS - the ones
# this package scores - so each tutorial states its own coverage instead of
# letting the reader assume parity. A README list for any other task stays out,
# and so does every method that is on such a list only.
PAPER_METHODS = {
 "vertical": ["totalVI","sciPENN","Concerto","scMSI","Matilda","MOFA2","Multigrate",
              "UINMF","scMoMaT","Seurat_WNN","scMM","scMDC","moETM","VIMCCA",
              "iPOLNG","MIRA","UnitedNet","scMVP"],
 "diagonal": ["scBridge","Portal","SCALEX","VIPCCA","Seurat_v3","MultiMAP","Seurat_v5",
              "sciCAN","Conos","iNMF","online_iNMF","scJoint","GLUE","uniPort"],
 "mosaic":   ["MultiVI","scMoMaT","StabMap","Cobolt","UINMF","Multigrate","SMILE"],
 "cross":    ["totalVI","scMoMaT","UnitedNet","sciPENN","Concerto","scMDC","StabMap",
              "UINMF","scMM","MOFA2","Multigrate"],
}
# The README heading each PAPER_METHODS list comes from, cut to the tasks this
# package scores; the coverage cell prints it with the count.
PAPER_TASKS = {
 "vertical": "dimension reduction and clustering",
 "diagonal": "dimension reduction, batch correction and clustering",
 "mosaic":   "dimension reduction, batch correction and clustering",
 "cross":    "dimension reduction, batch correction and clustering",
}


def details(*paras, label="Details"):
    """A collapsed block of markdown paragraphs. The blank line after
    ``<summary>`` and before ``</details>`` is what makes Jupyter, Colab and
    mkdocs-jupyter render the inside as markdown instead of literal text."""
    body = "\n\n".join(p.strip() for p in paras if p and p.strip())
    return f"<details>\n<summary>{label}</summary>\n\n{body}\n\n</details>"


def fallback_sweep(cat, dataset, methods):
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
        return (f"There are no scIB tables for {cat} under `source=\"published\"`, "
                f"the default of `load_results`. That call raises `FileNotFoundError`, "
                f"so every call here names `source=\"rerun\"`, the only stored source.")
    return (f"For `{dataset}`, the published table holds {n_pub} "
            f"method{'s' if n_pub != 1 else ''} and the package's own runs hold "
            f"{n_rerun}. `load_results` defaults to `source=\"published\"`, so every "
            f"call here names its source. Where both hold a method, the values can differ.")


def env_size_text(category=None, methods=None):
    """``"<n> envs, <x> GB to download on a CPU host, <y> GB on a GPU host"``
    for a category / method set, from the dry-run plans ``mtb.env.install(...)``
    returns at generation time for each archive flavour - never a hand-written
    GB figure. "at least" when an archive size is not measured."""
    import multibench as mtb
    sizes = {}
    for flavor in ("cpu", "gpu"):
        # dry_run=True (the default): nothing is built
        rows = mtb.env.install(methods, category=category, flavor=flavor)
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
    ``labels_for(dataset, cat, method)`` puts the files in another order than
    the default ``labels_for(dataset)``, read from the live package at
    generation time. A method that reads only some batches (UINMF: cty1, cty2)
    keeps the default order of its files and is not listed. Needs the dataset
    on disk (``config.DEFAULT.data_path``); a missing one raises instead of
    dropping the list from the tutorial."""
    import multibench as mtb
    default = list(mtb.labels_for(dataset))
    orders = {m: list(mtb.labels_for(dataset, cat, m)) for m in sorted(mtb.list_methods(cat))}
    return {m: o for m, o in orders.items() if o != [s for s in default if s in o]}


def partial_label_methods(cat, dataset, methods=None):
    """``{method: [stem, ...]}`` for the methods whose
    ``labels_for(dataset, cat, method)`` holds fewer label files than
    ``labels_for(dataset)``: the variant reads only some batches (UINMF on
    D52: cty1, cty2). Read from the live package at generation time."""
    import multibench as mtb
    default = list(mtb.labels_for(dataset))
    out = {}
    for m in sorted(methods or mtb.list_methods(cat)):
        stems = list(mtb.labels_for(dataset, cat, m))
        if len(stems) < len(default):
            out[m] = stems
    return out


def partial_batch_note(cat, dataset, methods):
    """The run-section sentence for each method of ``methods`` that reads only
    some of ``dataset``'s batches, or ``None``."""
    import multibench as mtb
    n = len(mtb.labels_for(dataset))
    notes = []
    for m, stems in partial_label_methods(cat, dataset, methods).items():
        batches = and_list(s.removeprefix("cty") for s in stems)
        notes.append(f"{m} reads only batches {batches} of {n}, and `scan` says so in its "
                     f"`caveat` column. Its `emb_shape` counts fewer cells, and its "
                     f"metrics cover only those cells.")
    return " ".join(notes) or None


def atac_forms(cat):
    """``(gene_activity, peak_only, both)``: the methods of a category by the
    ATAC form they read, from ``find_methods(cat, atac=...)`` and each
    variant's modalities at generation time. ``both`` read peaks and gene
    activity together (``atac_peak`` and ``atac_gas`` in one variant)."""
    import multibench as mtb
    gas = sorted(mtb.find_methods(cat, atac="gene_activity"))
    peak = sorted(mtb.find_methods(cat, atac="peak"))
    both = [m for m in peak
            if any(v["category"] == cat and {"atac_peak", "atac_gas"} <= set(v["modalities"])
                   for v in mtb.method_info(m)["supports"])]
    return gas, [m for m in peak if m not in both], both


def diagonal_atac_sentence():
    """The visible ATAC-form sentence of the diagonal tutorial's first cell."""
    gas, peak_only, both = atac_forms("diagonal")
    if not (peak_only and both and len(gas) > len(peak_only) + len(both)):
        raise SystemExit("diagonal ATAC forms changed: reword the diagonal title cell")
    import multibench as mtb
    if "same cells" not in mtb.method_info("Seurat_v5")["setup_hint"]:
        raise SystemExit("Seurat_v5's setup hint changed: reword the diagonal title cell")
    return (f"Most diagonal methods read ATAC as gene-activity scores, made beforehand "
            f"with a tool such as Signac or ArchR. {and_list(peak_only)} read the peak "
            f"matrix, and {and_list(both)} need both. Seurat_v5 also needs RNA and ATAC "
            f"from the same cells. `mtb.method_info(m)[\"atac\"]` says which form a "
            f"method reads.")


def runnable_sentence(cat):
    """The section-3 sentence on what ``runnable`` needs. In a category with
    ATAC methods, ``scan`` (without ``methods=``) also marks a variant whose
    ATAC file holds the other form not runnable, and ``run_all`` skips it;
    read from ``find_methods`` at generation time."""
    import multibench as mtb
    if mtb.find_methods(cat, modalities=["atac"]):
        return ("A variant is `runnable` only when both pass and its ATAC file holds "
                "the form the method reads. `reason` says what failed.")
    return "A variant is `runnable` only when both pass, and `reason` says what failed."


CAT_DATA = {"vertical": ["D11"], "diagonal": ["D28"],
            "mosaic": ["D45", "D46"], "cross": ["D52"]}

# The one install cell every notebook shares (tests/test_docs_consistency.py
# pins it): the package with its dependencies (evaluate() needs scib/scanpy).
# The find_spec guard keeps the cell idempotent and leaves a developer's
# editable install alone; the numpy / pandas pins keep pip from upgrading the
# host's stack; the GitHub line covers a PyPI release that lags the docs.
# The floor is the first release with every call the notebooks make
# (export_dataset(batch_index=, overwrite=) and the round-2 checks are not in
# 0.3.1), so an older PyPI release fails at install instead of mid-notebook.
INSTALL_CELLS = [
"""import importlib.metadata, importlib.util, sys
if importlib.util.find_spec("multibench") is None:
    # keep the numpy / pandas this interpreter already has
    pins = [f"{p}=={importlib.metadata.version(p)}" for p in ("numpy", "pandas") if importlib.util.find_spec(p)]
    !{sys.executable} -m pip -q install "multibench-sc>=0.3.2" {" ".join(pins)}
    importlib.invalidate_caches()
    # not on PyPI yet: install from GitHub
    if importlib.util.find_spec("multibench") is None:
        !{sys.executable} -m pip -q install "git+https://github.com/DSichang/scMultiBench.git" {" ".join(pins)}
else:
    print("multibench already installed")""",
]

# The "Run all" switch (tests/test_tutorial_runall_safety.py pins it): the
# one download of method environments - `mtb.env.install(..., dry_run=False)`
# - sits behind INSTALL_ENVS. The size is measured at generation time.
FLAG_CELL_TEMPLATE = """# False: no environment is downloaded; method cells use stored outputs.
# True (Linux or Colab): run the methods here; no conda needed.
# {size}.
INSTALL_ENVS = False"""

# The one line a run cell prints on a host without method environments
# (tests pin the phrase), before it loads a replacement computed elsewhere.
SKIP_LINE = ("no method environment on this computer: the run is skipped, "
             "and outputs computed elsewhere replace it")

# The last fallback of the run cells: a BatchResult built from the stored
# sweep's rows has the same .summary / .plot() as run_all's, so every later
# cell renders; status='STORED' says nothing ran and nothing is on disk.
STORED_SWEEP_FN = '''def stored_sweep(dataset, methods=None):
    """The stored results for `dataset`, as the object `run_all` returns."""
    long = mtb.load_results(CATEGORY, dataset=dataset, source="rerun", methods=methods)
    recs = [{"method": m, "status": "STORED", "metrics": g.set_index("metric")["value"].to_dict()}
            for m, g in long.groupby("method")]
    return mtb.BatchResult(recs, dataset, CATEGORY)'''

# The first fallback: the benchmark host's real run_all output tree for the
# dataset, downloaded by mtb.data.fetch_outputs and reloaded by load_batch,
# so .summary carries real statuses and run times and the evaluate cell
# scores a real embedding. Offline (or before the assets are published) the
# stored metric table replaces it; one printed line says which path was taken.
REPLACEMENT_FN = '''def replacement(dataset, methods, stored):
    """The run_all outputs for `dataset` from the benchmark's Linux machine; the stored results if that download fails."""
    try:
        res = mtb.load_batch(mtb.data.fetch_outputs(dataset), methods=methods)
        print(f"replacement: the run_all outputs for {dataset} from the benchmark's Linux machine, with embeddings and run times")
        return res
    # offline, or the outputs are not published yet
    except Exception as e:
        print(f"replacement: the package's stored metric table ({type(e).__name__} from fetch_outputs: {e})")
        return stored_sweep(*stored)'''

# The scoring step on its own, on the embedding one method wrote - run_all's
# tree and fetch_outputs' tree share the layout <out_dir>/<method>_<dataset>/
# embedding.h5, and the record says which label files the method's cells
# follow. Nothing to score on the stored-table replacement (no file on disk).
EVALUATE_CELL_TEMPLATE = '''m, emb = "{method}", None
if res.out_dir is not None:
    emb = Path(res.out_dir) / f"{{m}}_{{res.dataset}}" / "embedding.h5"
if emb is None or not emb.is_file():
    print(f"no embedding on this computer for {{m}}: nothing to score")
    scores = None
else:
    rec = next(r for r in res.results if r["method"] == m)
    # the label files in the order the method stacked its cells
    order = [Path(f).stem for f in rec.get("labels_used") or []] or None
    scores = mtb.evaluate(emb, labels=mtb.labels_for(res.dataset), label_order=order, verbose=False)
scores.T if scores is not None else None'''

SCEN = {
 "vertical": dict(
   ds="D11",
   blurb=("Vertical integration combines modalities measured in the same cells. "
          "Examples are RNA and surface protein from CITE-seq, or RNA and ATAC from "
          "10x Multiome. This tutorial uses `D11`, a CITE-seq dataset of 2,864 cells."),
   live=("Matilda", '{"epochs": 5}'), live_modalities=["rna", "adt"],
   live_ds=None, live_note=None, summary_note=None,
   own_src="D11", own_trio=["Matilda", "sciPENN", "scMM"],
 ),
 "diagonal": dict(
   ds="D28",
   blurb=("Diagonal integration combines RNA and ATAC measured in different cells, "
          "with no pairing between them. If your RNA and ATAC come from the same "
          "cells, as in 10x Multiome, use the vertical tutorial. This tutorial uses "
          "`D28`, with 6,408 RNA cells and 4,606 ATAC cells."),
   live=("online_iNMF", "None"), live_modalities=None,
   live_ds=None, live_note=None, summary_note=None,
   own_src="D28", own_trio=["online_iNMF", "iNMF", "scJoint"],
 ),
 "mosaic": dict(
   ds="D45",
   blurb=("Mosaic integration combines batches that share only some modalities. "
          "For example, a paired RNA + ATAC batch can link an RNA-only batch and an "
          "ATAC-only batch. Each method accepts one batch pattern, and every mosaic "
          "method reads ATAC as peaks. This tutorial uses `D45`, with 32,151 cells "
          "in three batches."),
   live=("StabMap", "None"), live_modalities=None,
   live_ds="D46",
   live_note=("These methods run on `D46`, whose batch pattern they accept. The "
              "stored metric table has no `D46` results, so it shows `D45` instead."),
   summary_note=None,
   own_src="D46", own_trio=["StabMap", "scMoMaT"],
 ),
 "cross": dict(
   ds="D52",
   blurb=("Cross integration combines batches that all measure the same "
          "modalities. The task is to remove batch effects and keep the biological "
          "structure. Every cross method here reads RNA and ADT; for several 10x "
          "Multiome samples, use the vertical tutorial. This tutorial uses `D52`, "
          "with 23,478 cells in three batches."),
   live=("StabMap", "None"), live_modalities=None,
   live_ds=None, live_note=None,
   summary_note=None,
   own_src="D52", own_trio=["UINMF", "sciPENN", "StabMap"],
 ),
}

# ---------------------------------------------------------------- own data
# One executed demo per category: an in-memory AnnData (or several) becomes a
# dataset folder in the layout describe_layout(CATEGORY) prints, written by
# export_dataset: one call for vertical, diagonal (category="diagonal" pairs
# no cells) and cross (batch=), one call per batch for mosaic (batch_index=).
# Each demo writes into a fresh temporary folder, so a re-run never meets the
# files of an earlier one (export_dataset refuses to replace them).
OVERWRITE_NOTE = ("A call that would replace a file already in the folder raises "
                  "`FileExistsError`. Pass `overwrite=True` to replace it.")
EXPORT_INTRO = {
 "vertical": ("`mtb.io.export_dataset` writes the folder from an AnnData, here a "
              "synthetic one. `scan` with `modalities=` then lists only the methods for "
              "RNA + ADT. For RNA + ATAC, each method reads either peaks or gene "
              "activity: `mtb.method_info(m)[\"atac\"]` says which."),
 "diagonal": ("`mtb.io.export_dataset` with `category=\"diagonal\"` writes RNA and ATAC "
              "from different cells, each with its own label file. Here on synthetic "
              "data, with ATAC as gene-activity scores:"),
 "mosaic": ("A mosaic project often arrives as one file per batch. "
            "`mtb.io.export_dataset` with `batch_index=` writes one batch per call; "
            "number the batches to match a pattern that `describe_layout` lists. The "
            "ATAC matrix must hold peaks. Here on synthetic data with `D46`'s pattern:"),
 "cross": ("`mtb.io.export_dataset` with `batch=` splits one AnnData into numbered "
           "files, one set per batch. Here on a synthetic AnnData:"),
}
EXPORT_DETAIL = {
 "vertical": [
     "A 10x Multiome MuData goes in with one call. Here the labels are in "
     "`mdata.obs`; for labels in `mdata[\"rna\"].obs`, write `labels=\"rna:celltype\"`.\n\n"
     "```python\n"
     "mtb.io.export_dataset(mdata, \"data/MYMULTIOME\", rna=\"rna\", atac=\"atac\",\n"
     "                      atac_kind=\"peak\", labels=\"obs:celltype\",\n"
     "                      category=\"vertical\")\n"
     "```",
     "A cellranger-arc AnnData read with `gex_only=False` holds genes and peaks in "
     "one `X`. A feature filter splits them: `rna=\"X[feature_types=Gene Expression]\"` "
     "and `atac=\"X[feature_types=Peaks]\"`.",
     "Keep several samples in one folder, without `batch=`. Score the batch mixing "
     "later with `mtb.evaluate(..., batch=...)`.",
     OVERWRITE_NOTE,
 ],
 "diagonal": [
     "The command line writes the same folder from two .h5ad files:\n\n"
     "```\n"
     "multibench convert rna.h5ad data/MYDIAG --rna X --atac-from atac.h5ad \\\n"
     "    --atac-kind gene_activity --labels obs:celltype --category diagonal\n"
     "```",
     # {peak_only}: filled in from the registry by build_tutorial
     "For ATAC as a peak matrix only, pass `atac_kind=\"peak\"`. {peak_only} read "
     "peaks. Seurat_v5 also needs RNA and ATAC from the same cells, because it uses "
     "them as its paired bridge. `scan` shows a method's setup note in its `caveat` "
     "column.",
     "With both ATAC files, `atac_gas.h5` must list the cells of `atac_peak.h5` in "
     "the same order, because `atac_cty.csv` follows `atac_peak.h5`. `scan` checks "
     "this. `mtb.io.to_canonical(..., modality=\"gas\")` into a folder with "
     "`atac_peak.h5` writes the rows in that order.",
     OVERWRITE_NOTE,
 ],
 "mosaic": [
     "The command line writes one batch per call with `multibench convert ... "
     "--category mosaic --batch-index N`. `describe_layout` above prints the commands "
     "for `D46`'s pattern.",
     OVERWRITE_NOTE,
 ],
 "cross": [
     "For one file per batch, call `export_dataset` once per file with "
     "`batch_index=N` instead of `batch=`. `describe_layout` above prints the command.",
     OVERWRITE_NOTE,
 ],
}
EXPORT_DETAIL_LABEL = {"vertical": "Details: 10x Multiome", "diagonal": "Details: export",
                       "mosaic": "Details: export", "cross": "Details: export"}
EXPORT_DEMO = {
 "vertical": """import anndata as ad, numpy as np, scipy.sparse as sp, tempfile, os
rng = np.random.default_rng(0)
# RNA as raw counts, cells x genes
demo = ad.AnnData(X=sp.csr_matrix(rng.poisson(0.5, size=(120, 40)).astype(float)))
# ADT as raw counts, cells x proteins
demo.obsm["protein"] = rng.poisson(3.0, size=(120, 12)).astype(float)
demo.uns["protein_names"] = [f"CD{i}" for i in range(12)]
demo.obs["celltype"] = rng.choice(["T", "B", "NK"], 120)
demo.obs_names = [f"cell{i}" for i in range(120)]; demo.var_names = [f"gene{i}" for i in range(40)]

tmp = tempfile.mkdtemp()
folder = mtb.io.export_dataset(demo, os.path.join(tmp, "MYCITE"),
                               rna="X", adt="obsm:protein", labels="obs:celltype")
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYCITE", CATEGORY, data_path=tmp, modalities=["rna", "adt"])
sc[["method", "modalities", "files_ok"]]""",
 "diagonal": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
genes = [f"gene{i}" for i in range(40)]
rna = ad.AnnData(X=rng.poisson(1.0, size=(120, 40)).astype(float))
rna.var_names = genes
# gene-activity scores of 90 other cells
atac = ad.AnnData(X=rng.poisson(0.5, size=(90, 40)).astype(float))
atac.var_names = genes
rna.obs["celltype"] = rng.choice(["T", "B", "NK"], 120)
atac.obs["celltype"] = rng.choice(["T", "B", "NK"], 90)

folder = mtb.io.export_dataset(rna, os.path.join(tempfile.mkdtemp(), "MYDIAG"),
                               atac=atac, atac_kind="gene_activity",
                               labels="obs:celltype", category="diagonal")
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYDIAG", CATEGORY, data_path=folder.parent)
# the methods that read peaks fail the file check
sc[["method", "modalities", "files_ok"]]""",
 "mosaic": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
def batch(n):
    a = ad.AnnData(X=rng.poisson(1.0, size=(n, 40)).astype(float))
    a.var_names = [f"gene{i}" for i in range(40)]
    a.obs["celltype"] = rng.choice(["T", "B", "NK"], n)
    return a
# batch 1: RNA + ADT, batch 2: RNA + ATAC peaks, batch 3: RNA only
b1, b2, b3 = batch(100), batch(80), batch(60)
b1.obsm["protein"] = rng.poisson(3.0, size=(100, 12)).astype(float)
b1.uns["protein_names"] = [f"CD{i}" for i in range(12)]
b2.obsm["peaks"] = rng.poisson(0.3, size=(80, 50)).astype(float)
b2.uns["peaks_names"] = [f"chr1:{100 * i}-{100 * i + 50}" for i in range(50)]

folder = os.path.join(tempfile.mkdtemp(), "MYMOSAIC")
kw = dict(labels="obs:celltype", category="mosaic")
mtb.io.export_dataset(b1, folder, adt="obsm:protein", batch_index=1, **kw)
mtb.io.export_dataset(b2, folder, atac="obsm:peaks", atac_kind="peak", batch_index=2, **kw)
mtb.io.export_dataset(b3, folder, batch_index=3, **kw)
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYMOSAIC", CATEGORY, data_path=os.path.dirname(folder))
# the batch pattern decides which methods fit
sc[["method", "modalities", "files_ok"]]""",
 "cross": """import anndata as ad, numpy as np, tempfile, os
rng = np.random.default_rng(0)
# RNA as raw counts, cells x genes
demo = ad.AnnData(X=rng.poisson(1.0, size=(150, 40)).astype(float))
demo.var_names = [f"gene{i}" for i in range(40)]
demo.obsm["protein"] = rng.poisson(3.0, size=(150, 12)).astype(float)
demo.uns["protein_names"] = [f"CD{i}" for i in range(12)]
demo.obs["celltype"] = rng.choice(["T", "B", "NK"], 150)
demo.obs["batch"] = rng.choice(["donor1", "donor2", "donor3"], 150)

folder = mtb.io.export_dataset(demo, os.path.join(tempfile.mkdtemp(), "MYCROSS"),
                               rna="X", adt="obsm:protein", labels="obs:celltype",
                               batch="obs:batch", category="cross")
print(sorted(os.listdir(folder)))
sc = mtb.scan("MYCROSS", CATEGORY, data_path=folder.parent)
sc[["method", "modalities", "files_ok"]]""",
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
                    # matrix/data is features x cells
                    counts[fn] = f["matrix/data"].shape[1]
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


def _notebook(cells, name):
    """The notebook, with cell ids derived from its name and cell position,
    so a regeneration that changes only text leaves the ids alone."""
    for i, cell in enumerate(cells):
        cell["id"] = hashlib.sha1(f"{name}:{i}".encode()).hexdigest()[:8]
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    })


def live_param_note(method, category, params, modalities):
    """The sentence for a run cell that overrides a method's defaults, with
    the defaults read from ``params_for`` at generation time."""
    import ast
    import multibench as mtb
    tunable = mtb.params_for(method, category, modalities)["tunable"]
    parts = [f"{k} {v} instead of its default {tunable[k]['default']}"
             for k, v in ast.literal_eval(params).items()]
    return (f"`params={{\"{method}\": {params}}}` runs {method} with "
            f"{and_list(parts)}, so the demo run is short.")


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
    title = f"# {cat.capitalize()} integration\n\n{s['blurb']}"
    if cat == "diagonal":
        title += "\n\n" + diagonal_atac_sentence()
    md(title)

    # ---------------------------------------------------------------- install
    md("## 1. Install\n\n"
       "The first cell installs `multibench-sc`. Methods run only on Linux, each in "
       "its own environment. Set `INSTALL_ENVS = True` on Linux or Colab to download "
       "the environments. On macOS and Windows every cell still runs and uses stored "
       "outputs instead.\n\n"
       + details(
           "With `INSTALL_ENVS = False`, the notebook downloads only the reference data "
           "and the stored outputs. A method cell then prints one line and loads those "
           "outputs.",
           "With `INSTALL_ENVS = True`, the cells download prebuilt environments and "
           "run the methods, with no conda needed.",
           "On Colab, choose a GPU runtime first: Runtime -> Change runtime type -> T4 "
           "GPU. On a CPU runtime the smaller CPU builds are installed, and training "
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
# hide library warnings; the warnings of multibench stay visible
for _w in (FutureWarning, DeprecationWarning, pd.errors.PerformanceWarning,
           anndata.ImplicitModificationWarning, TqdmWarning):
    warnings.filterwarnings("ignore", category=_w)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
import multibench as mtb

DATASET  = "{ds}"
CATEGORY = "{cat}"
# the reference data ({download_size(CAT_DATA[cat])}), downloaded once
mtb.data.fetch({', '.join(repr(d) for d in CAT_DATA[cat])})
print("multibench", mtb.__version__)''')

    # ------------------------------------------------------------ environments
    md(f"""## 2. Run the analysis

### Environments: only if you will run methods

The next cell installs the environments for {trio_text}: {env_size_text(methods=trio)}. It downloads nothing unless `INSTALL_ENVS = True` and the computer runs Linux.

""" + details(
        "Environments are unpacked under `mtb.config.DEFAULT.envs_dir`. Set it before "
        "this cell to use another disk. Without conda on the computer, the default is "
        "`~/.cache/multibench/envs`. An environment that is already there is not "
        "downloaded again.",
        f"To install every {cat} environment from a terminal, run `multibench env "
        f"install --category {cat} --packed --run`. That is "
        f"{env_size_text(category=cat)}; `multibench env plan --category {cat}` lists "
        f"the size of each."))
    code(f"""import sys
if not INSTALL_ENVS:
    print("INSTALL_ENVS is False: no environment is downloaded")
elif sys.platform != "linux":
    print("method environments run only on Linux: skipped on", sys.platform)
else:
    # dry run: sizes only
    plan = mtb.env.install({trio!r}, category=CATEGORY)
    todo = [r for r in plan if not r["exists"]]
    print(f"{{len(todo)}} of {{len(plan)}} envs to download, {{sum(r['archive_bytes'] or 0 for r in todo) / 1e9:.1f}} GB")
    for r in mtb.env.install({trio!r}, category=CATEGORY, packed=True, dry_run=False):
        print(f"{{r['env']:20s}} {{r['state']}}")""")

    # ------------------------------------------------------------- run + plot
    params_line = (f'\n                      params={{"{fastm}": {s["live"][1]}}},'
                   if s["live"][1] != "None" else "")
    si_ds, si_methods = fallback_sweep(cat, ds, trio)
    si_args = f'("{si_ds}",' + (f" {si_methods!r})" if si_methods else ")")
    run_notes = [
        f"The stored outputs are the `run_all` outputs for `{live_ds}` from the "
        "benchmark's Linux machine. `mtb.data.fetch_outputs` downloads them and "
        "`mtb.load_batch` reloads them, so the embeddings and run times are real.",
        "If that download fails, the cell uses the stored metric table "
        "(`load_results(source=\"rerun\")`) instead. The `status` column then reads "
        "`STORED`, and there is no embedding to score.",
    ]
    if s["live"][1] != "None":
        run_notes.append(live_param_note(fastm, cat, s["live"][1], s["live_modalities"]))
    run_notes += [s["live_note"], s["summary_note"],
                  partial_batch_note(cat, live_ds, trio)]
    md(f"""### Run the methods

`run_all` runs {trio_text} on `{live_ds}`, each in its own environment. Each method writes an embedding. An embedding is a table of numbers with one row per cell. `run_all` scores each embedding with the scIB metrics. Without environments, the cell prints one line and loads stored outputs instead.

""" + details(*run_notes))
    code(f'''{STORED_SWEEP_FN}

{REPLACEMENT_FN}

check = mtb.scan("{live_ds}", CATEGORY, methods={trio!r})
if check.env_ok.any():
    res = mtb.run_all("{live_ds}", CATEGORY,
                      methods={trio!r},{params_line}
                      out_dir="/tmp/tutorial_{cat}")
else:
    print("{SKIP_LINE}")
    res = replacement("{live_ds}", {trio!r}, stored={si_args})
res.summary''')
    score_notes = [
        "`label_order=` gives the order in which the method stacked its cells. The "
        "cell takes it from the run record's `labels_used`.",
        "`metrics=` selects what is computed:\n\n"
        "- `None` (the default): every applicable metric\n"
        "- `\"clustering\"`, `\"batch\"` or `\"all\"`: a family\n"
        "- a list such as `[\"ARI\", \"NMI\"]`: those metrics\n\n"
        "Batch metrics need `batch=` or several label files."
        + (" Here each label file is one modality, so the batch metrics show how "
           "well the RNA and ATAC cells mix." if cat == "diagonal" else ""),
        "The default Leiden backend is igraph. To compare your scores with the stored "
        "tables, set `mtb.config.DEFAULT.leiden_flavor = \"leidenalg\"` before "
        "`evaluate`. The two backends can move ARI by up to about 0.1.",
    ]
    md("""### Score one embedding

`run_all` has already scored every method. To score one embedding yourself, pass the file the method wrote and the dataset's label files to `mtb.evaluate`:

""" + details(*score_notes))
    code(EVALUATE_CELL_TEMPLATE.format(method=fastm))
    md("""### Plot

`res.plot()` draws a bubble table. Circle size shows the rank within a column; bigger is better. The fill compares the value with the other rows in the same column: the lightest fill is the lowest value in this figure, not zero.

""" + details(
        "Metrics are grouped by family: blue for dimension reduction and clustering, "
        "green for batch correction. Each family starts with an `Overall` bar. Its "
        "length and colour both show the family score.",
        "A column whose rows all hold the same value is drawn grey, and the note "
        "under the figure names it. A figure of one method is grey everywhere."))
    code("""res.plot()""")

    # ------------------------------------------------------------- own data
    md(f"""## 3. Your own data

The same calls work on a folder of your own data. Give raw counts for every modality, as in the demo data; the methods normalise the data themselves. `describe_layout` prints the files a {cat} dataset needs:""")
    code("""print(mtb.describe_layout(CATEGORY))""")
    labels_code = """labels = mtb.labels_for(DATASET)            # {file stem: path}
print({k: Path(v).name for k, v in labels.items()})
print(*Path(next(iter(labels.values()))).read_text().splitlines()[:4], sep="\\n")"""
    if cat == "vertical":
        md("`labels_for` returns a dataset's label files. A vertical dataset has one, `cty`:")
    else:
        files = "`rna_cty` then `atac_cty`" if cat == "diagonal" else "`cty1`, `cty2`, ... in that order"
        others = reordering_methods(cat, ds)
        if not others:
            raise SystemExit(f"no {cat} method stacks {ds}'s cells in another order: reword section 3")
        partial = partial_label_methods(cat, ds)
        md(f"""`labels_for` returns a dataset's label files: {files}. Some methods stack their cells in another order. `labels_for(DATASET, CATEGORY, method)` returns the files in that method's order. A wrong order gives wrong scores without an error.

""" + details(
            f"On `{ds}`, `labels_for` returns another order for "
            + and_list(f"{m} (`{', '.join(o)}`)" for m, o in others.items()) + ".",
            ("A method that reads only some batches gets only their label files: "
             + and_list(f"`{', '.join(o)}` for {m}" for m, o in partial.items()) + ".")
            if partial else None,
            "Pass the dict that `labels_for` returns to `evaluate` unchanged. A dict "
            "you build or reorder yourself is read in the default order. For any other "
            "order, name the keys with `label_order=`.",
            "`run_all` scores every order that fits the cell count and keeps the one "
            "with the highest ARI. The `label_order` column of `res.summary` shows the "
            "order it kept.", label="Details: label order"))
        # the printed example is a method that fits the dataset (files_ok, no
        # scan caveat: Seurat_v5 reorders D28 too, but needs paired files D28
        # does not have)
        import multibench as mtb
        fit = mtb.scan(ds, cat, methods=list(others), verbose=False)
        fit = set(fit.loc[fit["files_ok"] & (fit["caveat"] == ""), "method"])
        m0 = next((m for m in others if m in fit), next(iter(others)))
        labels_code += f'\nprint("{m0}:", list(mtb.labels_for(DATASET, CATEGORY, "{m0}")))'
    code(labels_code)
    export_notes = EXPORT_DETAIL[cat]
    if cat == "diagonal":
        export_notes = [p.replace("{peak_only}", and_list(atac_forms(cat)[1]))
                        for p in export_notes]
    md(EXPORT_INTRO[cat] + "\n\n" + details(*export_notes, label=EXPORT_DETAIL_LABEL[cat]))
    code(EXPORT_DEMO[cat])
    md(f"""Next, a real dataset under a new name: a random 60% of `{s['own_src']}`'s cells, with at most 2,000 cells and 5,000 features per file.

""" + details(
        "Files with the same number of cells keep the same cells in the same order. "
        "Each modality file then stays aligned with its label file. An export of your "
        "own data must keep this alignment too.", label="Details: cell alignment"))
    code(SUBSAMPLE_FN)
    md("""`scan` checks each method variant. A variant is one set of input files that a method accepts. `files_ok` checks the folder and works on any computer. `env_ok` checks the environment. """
       + runnable_sentence(cat))
    code(f'''DATA_ROOT = "/tmp/mydata"
src = mtb.config.DEFAULT.data_path / "{s['own_src']}"
subsample_dataset(src, f"{{DATA_ROOT}}/MYDATA_{cat}", frac=0.6)

sc = mtb.scan(f"MYDATA_{cat}", category=CATEGORY, data_path=DATA_ROOT)
print(f"files_ok {{int(sc.files_ok.sum())}}, env_ok {{int(sc.env_ok.sum())}}, runnable {{int(sc.runnable.sum())}} of {{len(sc)}} method variants")
sc[["method", "modalities", "files_ok", "env_ok", "runnable", "reason"]].head(6)''')
    own_ds, own_methods = fallback_sweep(cat, ds2, trio)
    own_args = f'"{own_ds}"' + (f", {own_methods!r}" if own_methods else "")
    code(f'''if sc[sc.method.isin({trio!r})].env_ok.any():
    mine = mtb.run_all(f"MYDATA_{cat}", CATEGORY,
                       methods={trio!r},
                       out_dir=f"{{DATA_ROOT}}/out_{cat}",
                       data_path=DATA_ROOT)
else:
    print("{SKIP_LINE}")
    # the stored results for {own_ds}, a 60% subsample of {own_ds[:-1]}
    mine = stored_sweep({own_args})
mine.summary''')
    code("""mine.plot()""")

    # ------------------------------------------------------------- figures
    n_rerun = stored_method_counts(cat, ds)[1]
    extra = other_datasets(cat, {ds, ds2})
    stored_notes = [published_note(cat, ds)]
    if extra:
        stored_notes.append(
            f"Stored results also exist for {and_list(f'`{d}`' for d in extra)}. "
            f"`mtb.available_datasets(CATEGORY, source=\"both\")` lists them.")
    stored_notes.append(
        "`run_all` saves its results in `out_dir`, and `mtb.load_batch(out_dir)` "
        "reloads them later without running anything. Use one `out_dir` per dataset "
        "and category: a second `run_all` into the same folder adds its methods to "
        "the saved results, and one for another dataset raises `ValueError`.")
    md(f"""## 4. Stored results

The package ships stored results for {n_rerun} methods on `{ds}`. `load_results(..., source="rerun")` reads them as a long table, and `mtb.plot.bubble` draws it. A long table has one row per method and metric. Nothing is run.

""" + details(*stored_notes, label="Details: sources"))
    code('''long = mtb.load_results(CATEGORY, dataset=DATASET, source="rerun")
print(long.method.nunique(), "methods,", long.source.unique())
fig = mtb.plot.bubble(long)
fig.set_dpi(110)
fig''')
    pair_notes = [
        "Each metric bar is the method's rank averaged over the datasets, then "
        "min-max scaled. `Overall` summarises the ranks of the family's metrics. Bar "
        "length and colour both show the value.",
        "Without `require_complete=True`, a method that is missing from one dataset "
        "gets the lowest rank there, which pulls its bars down.",
        f"`{ds2}` is a random subsample used for the package's second run of the "
        f"methods. It cannot be downloaded or rebuilt, so a summary that includes your "
        f"own method uses the full datasets only.",
    ]
    md(f"""A summary compares methods scored on the same datasets. `aggregate="summary"` ranks the methods over `{ds}` and `{ds2}`, a random 60% subsample of `{ds}`'s cells. `require_complete=True` keeps only the methods with results on both.

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

`scan` on `{ds}` runs nothing. The first table lists the variants whose input files are in place, and `caveat` says what to check before a run. The second table says why the other variants do not fit.

""" + details(
        "For a missing environment, `env_reason` gives the install command, or says "
        "that the environment runs only on Linux.",
        f"From a terminal, `multibench scan {ds} --category {cat}` prints the same "
        "scan. `--columns all` adds every column, including `command`: the exact "
        "command `run` would execute."))
    code("""avail = mtb.scan(DATASET, category=CATEGORY)
print(f"files_ok {int(avail.files_ok.sum())}, env_ok {int(avail.env_ok.sum())}, runnable {int(avail.runnable.sum())} of {len(avail)} method variants")
avail[avail.files_ok][["method", "modalities", "env", "env_ok", "env_reason",
                       "output_kind", "needs_labels", "runtime_tier", "caveat"]]""")
    code("""not_ok = avail[~avail.files_ok][["method", "modalities", "files_reason"]]
not_ok.head(5) if len(not_ok) else "(every method's inputs resolve on this dataset)"
""")
    md("""### Tuning

The table counts the parameters each variant exposes. `mtb.params_for(method, CATEGORY, modalities)` lists them, and `run_all(..., params={"Method": {"key": value}})` sets them.

""" + details(
        "Many upstream scripts set their parameters in code. Their variants accept no "
        "`params`.",
        "From a terminal, `multibench params METHOD` prints the table, and "
        "`multibench run-all ... --param METHOD:KEY=VALUE` sets a value."))
    code("""rows = [{"method": m, "modalities": "+".join(v["modalities"]) or "(data_dir)",
         "n_tunable": v["n_tunable"], "needs_labels": v["needs_labels"],
         "output_kind": v["output_kind"]}
        for m in sorted(mtb.list_methods(category=CATEGORY))
        for v in mtb.method_info(m)["supports"] if v["category"] == CATEGORY]
pd.DataFrame(rows).sort_values(["n_tunable", "method"], ascending=[False, True]).reset_index(drop=True)""")
    md(f"""### A method's record and citation

`method_info` returns what the package knows about a method, including its reference and repository. `mtb.cite` returns the citations for the benchmark and for the methods you pass. The cell cites {fastm}; for your own work, pass every method you ran.

""" + details(
        "`needs_labels` is True when any variant of the method needs cell-type labels. "
        "Each entry of `supports` gives it per variant, together with the modalities "
        "and the output kind.",
        "`gpu` says how the method uses a GPU: `required`, `used when present`, "
        "`not used` or `unknown`.",
        "`verbose=True` adds the long notes."))
    code(f'''info = mtb.method_info("{fastm}", verbose=True)
{{k: info[k] for k in ("id", "env", "needs_labels", "atac", "gpu", "notes", "repo_url", "version", "reference")}}''')
    code(f'''print(mtb.cite(["{fastm}"]))   # fmt="bibtex" for BibTeX entries''')
    md("""### The metrics

There are two families, and higher is better for every metric.

| family | metrics | measures |
|---|---|---|
| clustering / bio-conservation | `ARI`, `NMI`, `ASW`, `iASW`, `iF1`, `cLISI` | whether the embedding separates the annotated cell types |
| batch correction | `ASW_batch`, `GC`, `iLISI` (+ opt-in `kBET`) | whether the batches mix within each cell type |

""" + details(
        "ARI can fall slightly below 0; about 0 means a random clustering. Every "
        "other metric lies between 0 and 1. `mtb.catalog.metrics()` describes each one.",
        "Batch metrics appear only when the dataset has more than one batch.",
        "kBET is computed only when named, as in `metrics=[\"ASW_batch\", \"GC\", "
        "\"iLISI\", \"kBET\"]`. It is much slower than the others."))
    coverage_notes = []
    if cat == "mosaic":
        coverage_notes.append(
            "UINMF has no mosaic variant. Its script takes the second batch's unshared "
            "features from the first batch, which fails when the two are different "
            "modalities. It also accepts exactly two batches, a pattern that fits no "
            "mosaic dataset here.")
    md(f"""### Methods from the benchmark study

The cell compares the methods the scMultiBench study benchmarked for {cat} integration with the methods this package has a {cat} variant for. It prints each missing method with the categories it has variants for.""" + ("\n\n" + details(*coverage_notes) if coverage_notes else ""))
    code(f"""# benchmarked for {cat} on {PAPER_TASKS[cat]}
paper = {PAPER_METHODS[cat]!r}
registry = set(mtb.list_methods())
wired = sorted(m for m in registry
               if any(v["category"] == CATEGORY for v in mtb.method_info(m)["supports"]))
missing = [m for m in paper if m not in wired]
print(f"the study benchmarks {{len(paper)}} {{CATEGORY}} methods on {PAPER_TASKS[cat]}; this package has a {{CATEGORY}} variant for {{len(wired)}}")
for m in missing:
    if m in registry:
        print(f"  {{m}}: variants for {{', '.join(mtb.method_info(m)['categories'])}} only")
    else:
        print(f"  {{m}}: not in the registry")
if not missing:
    print("every benchmarked method has a variant for this category")""")

    # -------------------------------------------------------- troubleshooting
    siblings = ", ".join(c for c in SCEN if c != cat)
    md("""## Troubleshooting

When a method is not runnable, the `reason` column of `scan` says why. When a run fails, `res.failures` holds the error.

""" + details(
        """| symptom | fix |
|---|---|
| `files_ok` False: input files not found | `reason` names the missing file |
| `env_ok` False on Linux | run the `multibench env install ...` command in `env_reason` |
| `env_ok` False on macOS or Windows | methods run only on Linux; `mtb.run(..., dry_run=True)` prints the command to run there |
| `env_ok` False: the method needs an NVIDIA GPU | run it on a GPU machine; `mtb.scan(..., assume_gpu=True)` checks everything else on a computer without one |
| `FileExistsError` from `export_dataset` | the folder already holds the file: pass `overwrite=True` to replace it |
| a warning that values are not whole numbers | export raw counts, for example with `rna="layer:counts"` |
| `... which is cells x features` | the matrix is transposed: export it again with `mtb.io.export_dataset` or `mtb.io.to_canonical` |
| a method fails | `res.failures.iloc[0]["error"]` ends with the method's stderr |
| a method times out | raise `timeout=` in `run_all` |
| low `label_order_confidence` | several label files fit the cell count: check `label_order_candidates` in `res.results` |
| batch metrics use the wrong batches | `res.rescore(batch=my_vector)` scores again without running the methods |"""))
    md(f"""## Next steps

- the other tutorials: {siblings}
- the [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/): the full benchmark's rankings, no install needed
- `mtb.recommend(CATEGORY, modalities=[...])`: a ranking of methods from the stored results, to read as a hint
- `mtb.sweep(...)`: one method over a range of values of one parameter""")
    return C


def build_colab_quickstart():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    md("""# scMultiBench API quickstart (Colab)

This notebook installs `multibench-sc`, looks up methods and draws the stored benchmark results. It runs no method, so it downloads nothing else.

""" + details(
        "Running a method needs its environment and the reference data, and methods "
        "run only on Linux or Colab. The category tutorials download the data. They "
        "download the environments when you set `INSTALL_ENVS = True`."))
    for cell in INSTALL_CELLS:
        code(cell)
    code("""%matplotlib inline
import multibench as mtb

print(len(mtb.list_methods()), "methods in the package")
mtb.list_methods(category="vertical")""")
    md("""## Inspect a method

`method_info` returns what the package knows about a method. `find_methods` filters methods by what your data has, and `cite` returns the citations.

""" + details(
        "A method matches when one of its variants meets every filter: category, "
        "modalities, `needs_labels` and `atac`. A variant is one set of input files "
        "that a method accepts.",
        "The modality `\"atac\"` matches every method that reads ATAC. "
        "`\"atac_peak\"` keeps the methods that read peaks, and `\"atac_gas\"` the "
        "methods that read gene activity."))
    code("""info = mtb.method_info("Matilda")
{k: info[k] for k in ("id", "language", "env", "needs_labels", "notes", "repo_url", "reference", "supports")}""")
    code("""mtb.find_methods(category="vertical", modalities=["rna", "adt"], needs_labels=False)""")
    code("""print(mtb.cite(["Matilda"]))   # fmt="bibtex" for BibTeX entries""")
    md("""## Draw the stored results

The package ships stored results, so these figures draw without running anything.

""" + details(
        "`source=\"published\"`, the default, reads the published scIB tables. "
        "`source=\"rerun\"` reads the package's own runs of the methods.",
        "`aggregate=\"summary\"` ranks the methods over both datasets. `D28s` is a "
        "random 60% subsample of `D28`'s cells. `require_complete=True` keeps only "
        "the methods with results on both."))
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
        name = f"tutorial_{cat}"
        path = os.path.join(OUT, f"{name}.ipynb")
        nbf.write(_notebook(C, name), path)
        print(f"wrote {path} {len(C)} cells")
    C = build_colab_quickstart()
    path = os.path.join(OUT, "colab_quickstart.ipynb")
    nbf.write(_notebook(C, "colab_quickstart"), path)
    print(f"wrote {path} {len(C)} cells")
