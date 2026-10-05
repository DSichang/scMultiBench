"""Generate one tutorial per integration task (``TASKS``), the notebook that
runs every method of each task, and the Colab quickstart.

Each tutorial is one straight path, run top to bottom on Linux or Colab:
install the package, download the data and the method environments, run the
task's default methods with ``run_all``, plot, run the same calls on data in
the reader's own format, and draw the benchmark's stored scores. There are no flags, fallbacks
or helper functions. A step that cannot work on the reader's computer fails
with the package's own error: the environment install refuses off Linux.

Every method of a task runs on the task's one dataset. The defaults are the
fastest methods of the task with small environments. Every size quoted
comes from the dry-run plans at generation time, never a hand-written figure.

Prose has two layers. Visible: 1-3 short sentences per section - what the
step does and what the reader must do or decide - then the code. A fact
without which a reader gets a silently wrong result (raw counts, which
category fits, which ATAC form a method reads) is always visible. Collapsed
(``details()``, a ``<details>`` block): options, caveats, platform notes, as
short plain paragraphs or lists. No bold lead-in labels in a block of 1-3
paragraphs, no capitals for emphasis, no internal names, short sentences.
The notebooks are regenerated from this file - never hand-edited - and
executed on a Linux GPU host afterwards.

The install cell pins numpy and pandas to the versions already installed.
Colab's preinstalled stack is older than the newest scanpy and anndata need;
without the pins pip upgrades numpy and pandas there, and the session has to
restart. With the pins pip picks the scanpy and anndata releases that fit.
"""
import hashlib
import json
import os

import nbformat as nbf

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)

COLAB = "https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/"
SITE = "https://dsichang.github.io/scMultiBench/"


def details(*paras, label="Details"):
    """A collapsed block of markdown paragraphs. The blank line after
    ``<summary>`` and before ``</details>`` is what makes Jupyter, Colab and
    mkdocs-jupyter render the inside as markdown instead of literal text."""
    body = "\n\n".join(p.strip() for p in paras if p and p.strip())
    return f"<details>\n<summary>{label}</summary>\n\n{body}\n\n</details>"


def stored_method_counts(cat, dataset):
    """``(n_published, n_rerun)``: methods each stored source holds for the
    dataset, counted from ``results_coverage`` at generation time."""
    import multibench as mtb
    cov = mtb.results_coverage(cat)
    cov = cov[cov.dataset == dataset]
    return (cov[cov.source == "published"].method.nunique(),
            cov[cov.source.str.startswith("rerun")].method.nunique())


def _gb(rows):
    known = [r["archive_bytes"] for r in rows if r["archive_bytes"]]
    return ("" if len(known) == len(rows) else "at least ") + f"{sum(known) / 1e9:.1f} GB"


def env_size_text(category=None, methods=None):
    """``"<n> envs, <x> GB to download on a CPU host, <y> GB on a GPU host"``
    for a category / method set, from the dry-run plans ``mtb.env.install(...)``
    returns at generation time for each archive flavour."""
    import multibench as mtb
    # dry_run=True (the default): nothing is built
    rows = {f: mtb.env.install(methods, category=category, flavor=f) for f in ("cpu", "gpu")}
    n = len(rows["cpu"])
    return (f"{n} env{'s' if n != 1 else ''}, {_gb(rows['cpu'])} to download on a CPU "
            f"host, {_gb(rows['gpu'])} on a GPU host")


def env_download_sentence(methods):
    """The visible download-size sentence of section 2. The GPU builds carry
    the CUDA libraries, so the two flavours differ unless every env has one
    archive only."""
    import multibench as mtb
    cpu = _gb(mtb.env.install(methods, flavor="cpu"))
    gpu = _gb(mtb.env.install(methods, flavor="gpu"))
    if cpu == gpu:
        return f"The download is {cpu}."
    return f"The download is {cpu} on a computer without a GPU and {gpu} on one with a GPU."


def download_size(datasets):
    """The reference-data download size, from the package's own table."""
    from multibench.data.fetch import AVAILABLE
    return " + ".join(AVAILABLE[d] for d in datasets)


def and_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


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
    """The visible ATAC-form sentence of the diagonal tutorial's first cell.

    It points to ``describe_layout``, which lists each method under the ATAC
    files it needs. ``method_info(m)["atac"]`` names one form, and lists the
    methods that read both files under peak."""
    gas, peak_only, both = atac_forms("diagonal")
    if not (peak_only and both and len(gas) > len(peak_only) + len(both)):
        raise SystemExit("diagonal ATAC forms changed: reword the diagonal title cell")
    import multibench as mtb
    if "same cells" not in mtb.method_info("Seurat_v5")["setup_hint"]:
        raise SystemExit("Seurat_v5's setup hint changed: reword the diagonal title cell")
    layout = [ln.strip() for ln in mtb.describe_layout("diagonal").splitlines()]
    if not any(ln.startswith("need both files:") and all(m in ln for m in both)
               for ln in layout):
        raise SystemExit("describe_layout no longer lists the methods that read both "
                         "ATAC files: reword the diagonal title cell")
    return (f"Most diagonal methods read ATAC as gene-activity scores, made beforehand "
            f"with a tool such as Signac or ArchR. {and_list(peak_only)} read the peak "
            f"matrix, and {and_list(both)} need both. Seurat_v5 also needs RNA and ATAC "
            f"from the same cells. `mtb.describe_layout(\"diagonal\")` lists each "
            f"method's ATAC files.")


# The one install cell every notebook shares; the module docstring says why
# numpy and pandas are pinned. IPython expands the {...} in a %pip line.
INSTALL_CELL = """import numpy, pandas
%pip install -q multibench-sc numpy=={numpy.__version__} pandas=={pandas.__version__}"""

COLAB_GPU_NOTE = ("On Colab, choose a GPU runtime before you run the notebook: Runtime -> "
                  "Change runtime type -> T4 GPU. On a CPU runtime, an environment that has a "
                  "smaller CPU build gets that build, and training methods run slower.")

# One notebook per integration task, and one dataset per task: every method of
# the task runs on it (tools/make_tutorial_data.py). `variants` are the
# modality sets of the task's method variants; `methods` are the two the
# notebook runs by default (one when the task has a single method).
TASKS = {
 "vertical_rna_adt": dict(
   cat="vertical", label="RNA + ADT", ds="D11", stored_ds="D11",
   variants=[["rna", "adt"]], methods=["Matilda", "sciPENN"],
   blurb=("Vertical integration combines modalities measured in the same cells. "
          "This tutorial is for RNA and surface protein, as in CITE-seq. It runs "
          "{methods} on `D11`, a CITE-seq dataset of 2,864 cells, and then on data "
          "in your own format."),
 ),
 "vertical_rna_atac": dict(
   cat="vertical", label="RNA + ATAC", ds="D27mini_vertical",
   variants=[["rna", "atac"], ["rna", "atac_gas"]], methods=["scMM", "VIMCCA"],
   blurb=("Vertical integration combines modalities measured in the same cells. "
          "This tutorial is for RNA and ATAC, as in 10x Multiome. It runs {methods} "
          "on `D27mini_vertical`, 5,000 cells of the benchmark dataset `D27`, and "
          "then on data in your own format. Each method reads ATAC as peaks or as "
          "gene-activity scores: `mtb.method_info(m)[\"atac\"]` says which. The "
          "dataset holds both."),
 ),
 "vertical_rna_adt_atac": dict(
   cat="vertical", label="RNA + ADT + ATAC", ds="D22mini",
   variants=[["rna", "adt", "atac"]], methods=["scMoMaT"],
   blurb=("Vertical integration combines modalities measured in the same cells. "
          "This tutorial is for RNA, surface protein and ATAC from the same cells. "
          "It runs {methods} on `D22mini`, 3,000 cells of the benchmark dataset "
          "`D22`, and then on data in your own format."),
 ),
 "diagonal_rna_atac": dict(
   cat="diagonal", label="RNA + ATAC", ds="D27mini", stored_ds="D28",
   variants=None, methods=["iNMF", "online_iNMF"],
   blurb=("Diagonal integration combines RNA and ATAC measured in different cells, "
          "with no pairing between them. If your RNA and ATAC come from the same "
          "cells, as in 10x Multiome, use the vertical tutorial. This tutorial runs "
          "{methods} on `D27mini`, 3,000 RNA and 3,000 ATAC profiles of the "
          "benchmark dataset `D27`, then on data in your own format."),
 ),
 "mosaic_rna_atac": dict(
   cat="mosaic", label="RNA + ATAC", ds="D45mini", stored_ds="D45",
   variants=[["rna1", "rna2", "atac2", "atac3"]], methods=["Cobolt", "SMILE"],
   blurb=("Mosaic integration combines batches that share only some modalities. "
          "Each method accepts one batch pattern, and every mosaic method reads ATAC "
          "as peaks. This tutorial is for an RNA batch, an RNA + ATAC batch and an "
          "ATAC batch. It runs {methods} on `D45mini`, 6,000 cells of the benchmark "
          "dataset `D45`, and then on data in your own format."),
 ),
 "mosaic_rna_adt_atac": dict(
   cat="mosaic", label="RNA + ADT + ATAC", ds="D46mini",
   variants=[["rna1", "rna2", "rna3", "adt1", "atac2"]], methods=["StabMap", "scMoMaT"],
   blurb=("Mosaic integration combines batches that share only some modalities. "
          "Each method accepts one batch pattern, and every mosaic method reads ATAC "
          "as peaks. This tutorial is for an RNA + ADT batch, an RNA + ATAC batch and "
          "an RNA batch. It runs {methods} on `D46mini`, 1,800 cells of the benchmark "
          "dataset `D46`, and then on data in your own format."),
 ),
 "mosaic_rna_adt": dict(
   cat="mosaic", label="RNA + ADT", ds="D38mini",
   variants=[["rna1", "rna2", "adt2", "adt3"]], methods=["Multigrate"],
   blurb=("Mosaic integration combines batches that share only some modalities. "
          "Each method accepts one batch pattern. This tutorial is for an RNA batch, "
          "an RNA + ADT batch and an ADT batch. It runs {methods} on `D38mini`, "
          "6,000 cells of the benchmark dataset `D38`, and then on data in your own "
          "format."),
 ),
 "cross_rna_adt": dict(
   cat="cross", label="RNA + ADT", ds="D52mini", stored_ds="D52",
   variants=None, methods=["StabMap", "sciPENN"],
   blurb=("Cross integration combines batches that all measure the same "
          "modalities. The task is to remove batch effects and keep the biological "
          "structure. Every cross method here reads RNA and ADT. This tutorial runs "
          "{methods} on `D52mini`, 3,000 cells in three batches of the benchmark "
          "dataset `D52`, and then on data in your own format."),
 ),
}


def task_methods(key):
    """Every method with a variant of the task, in name order."""
    from multibench.engine import registry
    t = TASKS[key]
    want = None if t["variants"] is None else [set(v) for v in t["variants"]]
    out = []
    for m in registry.list_methods():
        for v in registry.get(m).variants:
            if v.when.get("category") != t["cat"]:
                continue
            if want is None or set(v.when.get("modalities", [])) in want:
                out.append(m)
                break
    return sorted(out)


def run_kwargs(key):
    """Extra ``run_all`` arguments of the task: the modalities, where a method
    has another variant the dataset also satisfies (scMoMaT on D22mini)."""
    t = TASKS[key]
    return f", modalities={json.dumps(t['variants'][0])}" if key == "vertical_rna_adt_atac" else ""


def check_task(key):
    """Stop when a default method reads only some batches of the dataset."""
    import multibench as mtb
    t = TASKS[key]
    n = len(mtb.labels_for(t["ds"]))
    for m in t["methods"]:
        if len(mtb.labels_for(t["ds"], t["cat"], m)) < n:
            raise SystemExit(f"{m} reads only some batches of {t['ds']}: pick another method")
        if m not in task_methods(key):
            raise SystemExit(f"{m} has no variant of {key}")


# ---------------------------------------------------------------- own data
# Section 5 per task: the shape the reader's data usually has, built from a
# part of the tutorial's dataset, then the export call and the same run_all on
# the new folder. overwrite=True lets the notebook run twice.
OVERWRITE_NOTE = ("`overwrite=True` replaces the files of an earlier run of this cell. "
                  "Without it, `export_dataset` raises `FileExistsError` rather than "
                  "replace a file.")
OWN = {
 "vertical_rna_adt": dict(
   name="MYCITE",
   intro=("Your data needs raw counts for both modalities and a cell type for each "
          "cell. `mtb.io.export_dataset` writes an AnnData as a dataset folder, and "
          "`run_all` runs on that folder."),
   standin="Here an AnnData made from 60% of `D11`'s cells takes the place of your data:",
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D11"
adata = mtb.io.read_canonical(d / "rna.h5")
adata.obsm["protein"] = mtb.io.read_canonical(d / "adt.h5").to_df()
adata.obs["celltype"] = pd.read_csv(d / "cty.csv")["x"].values
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
adata''',
   export='''mtb.io.export_dataset(adata, "mydata/MYCITE", rna="X", adt="obsm:protein",
                      labels="obs:celltype", overwrite=True)''',
   notes=["Keep several samples in one folder, without `batch=`. To score the batch "
          "mixing, pass the batch column to `run_all(batch=...)`."],
 ),
 "vertical_rna_atac": dict(
   name="MYMULTIOME",
   intro=("Your data needs raw RNA counts, an ATAC matrix of the same cells and a "
          "cell type for each cell. `mtb.io.export_dataset` writes them as a dataset "
          "folder, and `run_all` runs on that folder. scMM and VIMCCA read ATAC as "
          "peaks, so the folder holds peaks."),
   standin=("Here 60% of `D27mini_vertical`'s cells, as an RNA and a peak AnnData, "
            "take the place of your data:"),
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D27mini_vertical"
rna = mtb.io.read_canonical(d / "rna.h5")
rna.obs["celltype"] = pd.read_csv(d / "cty.csv")["x"].values
atac = mtb.io.read_canonical(d / "atac_peak.h5")
rna = sc.pp.subsample(rna, fraction=0.6, random_state=0, copy=True)
atac = atac[rna.obs_names].copy()
rna, atac''',
   export='''mtb.io.export_dataset(rna, "mydata/MYMULTIOME", atac=atac, atac_kind="peak",
                      labels="obs:celltype", overwrite=True)''',
   notes=["A 10x Multiome MuData goes in with one call. Here the labels are in "
          "`mdata.obs`. For labels in `mdata[\"rna\"].obs`, write `labels=\"rna:celltype\"`.\n\n"
          "```python\n"
          "mtb.io.export_dataset(mdata, \"mydata/MYMULTIOME\", rna=\"rna\", atac=\"atac\",\n"
          "                      atac_kind=\"peak\", labels=\"obs:celltype\",\n"
          "                      category=\"vertical\")\n"
          "```",
          "A cellranger-arc AnnData read with `gex_only=False` holds genes and peaks in "
          "one `X`. A feature filter splits them: `rna=\"X[feature_types=Gene Expression]\"` "
          "and `atac=\"X[feature_types=Peaks]\"`.",
          "For a method that reads gene activity, export that matrix with "
          "`atac_kind=\"gene_activity\"`. A folder with both `atac_peak.h5` and "
          "`atac_gas.h5` serves every RNA + ATAC method."],
 ),
 "vertical_rna_adt_atac": dict(
   name="MYTRI",
   intro=("Your data needs raw counts for RNA and protein, an ATAC peak matrix of the "
          "same cells and a cell type for each cell. `mtb.io.export_dataset` writes "
          "them as a dataset folder, and `run_all` runs on that folder."),
   standin="Here 60% of `D22mini`'s cells take the place of your data:",
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D22mini"
adata = mtb.io.read_canonical(d / "rna.h5")
adata.obsm["protein"] = mtb.io.read_canonical(d / "adt.h5").to_df()
adata.obs["celltype"] = pd.read_csv(d / "cty.csv")["x"].values
atac = mtb.io.read_canonical(d / "atac.h5")
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
atac = atac[adata.obs_names].copy()
adata, atac''',
   export='''mtb.io.export_dataset(adata, "mydata/MYTRI", rna="X", adt="obsm:protein", atac=atac,
                      atac_kind="peak", labels="obs:celltype", overwrite=True)''',
   notes=[],
 ),
 "diagonal_rna_atac": dict(
   name="MYDIAG",
   intro=("Your data needs raw RNA counts and ATAC as two AnnData objects, with a "
          "cell type for each cell. `mtb.io.export_dataset` writes them as a dataset "
          "folder, and `run_all` runs on that folder. iNMF and online_iNMF read ATAC "
          "as gene-activity scores, and online_iNMF needs about 5,000 cells in total."),
   standin=("Here 90% of `D27mini`'s RNA profiles and 90% of its ATAC profiles take "
            "the place of your data:"),
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D27mini"
rna = mtb.io.read_canonical(d / "rna.h5")
rna.obs["celltype"] = pd.read_csv(d / "rna_cty.csv")["x"].values
atac = mtb.io.read_canonical(d / "atac_gas.h5")
atac.obs["celltype"] = pd.read_csv(d / "atac_cty.csv")["x"].values
rna = sc.pp.subsample(rna, fraction=0.9, random_state=0, copy=True)
atac = sc.pp.subsample(atac, fraction=0.9, random_state=1, copy=True)
rna, atac''',
   export='''mtb.io.export_dataset(rna, "mydata/MYDIAG", atac=atac, atac_kind="gene_activity",
                      labels="obs:celltype", category="diagonal", overwrite=True)''',
   notes=["The command line writes the same folder from two .h5ad files:\n\n"
          "```\n"
          "multibench convert rna.h5ad mydata/MYDIAG --rna X --atac-from atac.h5ad \\\n"
          "    --atac-kind gene_activity --labels obs:celltype --category diagonal\n"
          "```",
          "For ATAC as a peak matrix, pass `atac_kind=\"peak\"`. {peak_only} read "
          "peaks. `mtb.describe_layout(\"diagonal\")` lists the files each method needs."],
 ),
 "mosaic_rna_atac": dict(
   name="MYMOSAIC",
   intro=("Your data needs raw counts, with one AnnData per batch and a cell type for "
          "each cell. `mtb.io.export_dataset` with `batch_index=` writes one batch per "
          "call. Number the batches as below: RNA, then RNA + ATAC, then ATAC. Give "
          "ATAC as peaks, with the same peaks in both ATAC batches."),
   standin="Here the first 1,500 cells of each `D45mini` batch take the place of your data:",
   data='''d = mtb.config.DEFAULT.data_path / "D45mini"
n = 1500
rna1 = mtb.io.read_canonical(d / "rna1.h5")[:n]
rna2 = mtb.io.read_canonical(d / "rna2.h5")[:n]
atac2 = mtb.io.read_canonical(d / "atac2.h5")[:n]
atac3 = mtb.io.read_canonical(d / "atac3.h5")[:n]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA only
mtb.io.export_dataset(rna1, out, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna2, out, atac=atac2, atac_kind="peak", labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ATAC peaks only
mtb.io.export_dataset(atac3, out, rna=None, atac="X", atac_kind="peak", labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=["`mtb.describe_layout(\"mosaic\")` lists the batch patterns and the methods "
          "that accept each one. The command line writes one batch per call with "
          "`multibench convert ... --category mosaic --batch-index N`."],
 ),
 "mosaic_rna_adt_atac": dict(
   name="MYMOSAIC",
   intro=("Your data needs raw counts, with one AnnData per batch and a cell type for "
          "each cell. `mtb.io.export_dataset` with `batch_index=` writes one batch per "
          "call. Number the batches as below: RNA + ADT, then RNA + ATAC, then RNA. "
          "Give ATAC as peaks."),
   standin="Here the first 300 cells of each `D46mini` batch take the place of your data:",
   data='''d = mtb.config.DEFAULT.data_path / "D46mini"
n = 300
rna = [mtb.io.read_canonical(d / f"rna{b}.h5")[:n] for b in (1, 2, 3)]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]
adt1 = mtb.io.read_canonical(d / "adt1.h5")[:n]
atac2 = mtb.io.read_canonical(d / "atac2.h5")[:n]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA + ADT
mtb.io.export_dataset(rna[0], out, adt=adt1, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna[1], out, atac=atac2, atac_kind="peak", labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: RNA only
mtb.io.export_dataset(rna[2], out, labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=["`mtb.describe_layout(\"mosaic\")` lists the batch patterns and the methods "
          "that accept each one. The command line writes one batch per call with "
          "`multibench convert ... --category mosaic --batch-index N`."],
 ),
 "mosaic_rna_adt": dict(
   name="MYMOSAIC",
   intro=("Your data needs raw counts, with one AnnData per batch and a cell type for "
          "each cell. `mtb.io.export_dataset` with `batch_index=` writes one batch per "
          "call. Number the batches as below: RNA, then RNA + ADT, then ADT. "
          "Multigrate needs several thousand cells for a mosaic run."),
   standin="Here every cell of `D38mini` takes the place of your data:",
   data='''d = mtb.config.DEFAULT.data_path / "D38mini"
rna1 = mtb.io.read_canonical(d / "rna1.h5")
rna2 = mtb.io.read_canonical(d / "rna2.h5")
adt2 = mtb.io.read_canonical(d / "adt2.h5")
adt3 = mtb.io.read_canonical(d / "adt3.h5")
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values for b in (1, 2, 3)]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA only
mtb.io.export_dataset(rna1, out, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ADT
mtb.io.export_dataset(rna2, out, adt=adt2, labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ADT only
mtb.io.export_dataset(adt3, out, rna=None, adt="X", labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=["`mtb.describe_layout(\"mosaic\")` lists the batch patterns and the methods "
          "that accept each one."],
 ),
 "cross_rna_adt": dict(
   name="MYCROSS",
   intro=("Your data needs raw RNA and ADT counts in one AnnData, with a cell type and "
          "a batch for each cell. `mtb.io.export_dataset` with `batch=` writes one set "
          "of files per batch, and `run_all` runs on that folder."),
   standin=("Here one AnnData with 60% of `D52mini`'s cells and a batch column takes "
            "the place of your data:"),
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D52mini"
batches = []
for b in (1, 2, 3):
    a = mtb.io.read_canonical(d / f"rna{b}.h5")
    a.obsm["protein"] = mtb.io.read_canonical(d / f"adt{b}.h5").to_df()
    a.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    batches.append(a)
adata = ad.concat(batches, label="batch", keys=["1", "2", "3"], index_unique="-")
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
adata''',
   export='''mtb.io.export_dataset(adata, "mydata/MYCROSS", rna="X", adt="obsm:protein",
                      labels="obs:celltype", batch="obs:batch", category="cross",
                      overwrite=True)''',
   notes=["For one AnnData per batch, call `export_dataset` once per batch with "
          "`batch_index=N` instead of `batch=`."],
 ),
}


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


def _title(key, suffix=""):
    t = TASKS[key]
    return f"# {t['cat'].capitalize()} integration: {t['label']}{suffix}"


def _install(md, code):
    md("## 1. Install\n\n"
       "This cell installs `multibench-sc`. It keeps the numpy and pandas that are "
       "already installed, so Colab needs no restart.\n\n" + details(COLAB_GPU_NOTE))
    code(INSTALL_CELL)


PLOT_TEXT = ("`res.plot()` draws the scores as a bubble table. Circle size shows the rank "
             "within a column, and bigger is better. The fill compares a value with the "
             "other rows in the same column.")
PLOT_DETAILS = (
    "The lightest fill is the lowest value in this figure, not zero.",
    "Metrics are grouped by family: blue for dimension reduction and clustering, "
    "green for batch correction. Each family starts with an `Overall` bar. Its "
    "length and colour both show the family score.",
    "A column whose rows all hold the same value is drawn grey, and the note under "
    "the figure names it. A figure of one method is grey everywhere.")


def build_tutorial(key):
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    check_task(key)
    t, own = TASKS[key], OWN[key]
    cat, ds, methods = t["cat"], t["ds"], t["methods"]
    names = and_list(methods)
    everyone = task_methods(key)
    extra = run_kwargs(key)

    # ------------------------------------------------------------------ title
    title = _title(key) + "\n\n" + t["blurb"].format(methods=names)
    if cat == "diagonal":
        title += "\n\n" + diagonal_atac_sentence()
    title += (f"\n\nMethods run on Linux. On macOS or Windows, "
              f"[open this notebook in Colab]({COLAB}tutorial_{key}.ipynb).")
    md(title)
    _install(md, code)

    # ------------------------------------------------------ data and envs
    md(f"""## 2. Download the data and the environments

`mtb.data.fetch` downloads `{ds}` ({download_size([ds])}) once. Each method runs in its own environment. `mtb.env.install` downloads the environment{'s' if len(methods) > 1 else ''} for {names}, with no conda needed. {env_download_sentence(methods)}

""" + details(
        "`state` is `PACKED` for an environment downloaded now and `have` for one that "
        "was already there. Without `dry_run=False`, `mtb.env.install` downloads "
        "nothing and returns the plan with its sizes.",
        "Environments go to `mtb.config.DEFAULT.envs_dir`. To use another disk, set it "
        "before this cell.",
        f"From a terminal, `multibench env install --methods {','.join(methods)} --packed "
        f"--run` does the same."))
    code(f'''import pandas as pd
import multibench as mtb

mtb.data.fetch("{ds}")''')
    code(f'''METHODS = {json.dumps(methods)}
envs = mtb.env.install(METHODS, dry_run=False)
pd.DataFrame(envs)[["env", "methods", "state"]]''')

    # -------------------------------------------------------------------- run
    run_notes = [
        f"Each method writes an embedding: a table of numbers with one row per cell. "
        f"`run_all` scores it against the cell-type labels of `{ds}`.",
        "The clustering metrics `ARI`, `NMI`, `ASW`, `iASW`, `iF1` and `cLISI` measure "
        "how well the embedding separates the cell types. The batch metrics "
        "`ASW_batch`, `GC` and `iLISI` measure how well the batches mix. They appear "
        "only when the data has several batches.",
    ]
    if "scMoMaT" in methods:
        run_notes.append("scMoMaT writes a graph instead of an embedding. `run_all` scores "
                         "its UMAP, and its status reads `CHAIN_OK_GRAPH_METHOD`.")
    if extra:
        run_notes.append("`modalities=` names the files the methods read. scMoMaT also has "
                         "an RNA + ADT variant, and the folder holds the files of both.")
    run_notes += [
        "`res.failures` says why a method failed. `params={\"Method\": {\"key\": value}}` "
        "sets a method's parameters, and `mtb.params_for` lists them. Many methods "
        "take none.",
        f"`mtb.load_batch(\"out/{ds}\")` reloads these results later without running "
        f"anything.",
    ]
    md(f"""## 3. Run the method{'s' if len(methods) > 1 else ''}

`run_all` runs each method on `{ds}` and scores its output with the scIB metrics. `res.summary` has one row per method with its status, run time and scores. Higher is better for every metric.

""" + details(*run_notes))
    code(f'''res = mtb.run_all("{ds}", "{cat}", methods=METHODS,
                  out_dir="out/{ds}"{extra})
res.summary''')

    # ------------------------------------------------------------------- plot
    md("## 4. Plot\n\n" + PLOT_TEXT + "\n\n" + details(*PLOT_DETAILS))
    code("res.plot()")

    # --------------------------------------------------------------- own data
    peak_only = and_list(atac_forms("diagonal")[1])
    notes = [n.replace("{peak_only}", peak_only) for n in own["notes"]]
    notes += [
        f"The folder name, `{own['name']}`, is the dataset name for `run_all`, and "
        f"`data_path` is the folder that holds it.",
        f"`mtb.scan(\"{own['name']}\", \"{cat}\", data_path=\"mydata\")` checks the folder "
        f"and the environments without running anything. Its `reason` column says what "
        f"is missing.",
        OVERWRITE_NOTE,
    ]
    md(f"## 5. Your own data\n\n{own['intro']}\n\n" + details(*notes, label="Details: export"))
    md(own["standin"])
    code(own["data"])
    md("Write the folder, then run the same method on it:" if len(methods) == 1 else
       "Write the folder, then run the same methods on it:")
    code(own["export"])
    code(f'''mine = mtb.run_all("{own['name']}", "{cat}", methods=METHODS, data_path="mydata",
                   out_dir="out/{own['name']}"{extra})
mine.summary''')
    code("mine.plot()")

    # ------------------------------------------------------------ stored scores
    n_sec = 6
    sds = t.get("stored_ds")
    if sds:
        n_pub, n_rerun = stored_method_counts(cat, sds)
        stored_notes = []
        if sds != ds and sds in ds:
            stored_notes.append(f"The stored scores are for the full `{sds}`, the benchmark "
                                f"dataset `{ds}` is drawn from.")
        elif sds != ds:
            stored_notes.append(f"There are no stored scores for `{ds}`. `{sds}` is another "
                                f"benchmark dataset of the same task.")
        stored_notes.append(
            f"`source=\"rerun\"` reads the package's own runs of the methods. "
            + (f"`source=\"published\"`, the default, reads the published scIB tables, which "
               f"hold {n_pub} method{'s' if n_pub != 1 else ''} for `{sds}`."
               if n_pub else
               f"There is no published scIB table for {cat}, so `source=\"published\"`, "
               f"the default, raises `FileNotFoundError`."))
        stored_notes.append(
            "The stored scores used the `leidenalg` backend for Leiden clustering, and "
            "`run_all` uses `igraph` by default. The two backends can move ARI by up to "
            "about 0.1. To compare your runs with these scores, set "
            "`mtb.config.DEFAULT.leiden_flavor = \"leidenalg\"` before `run_all`.")
        md(f"""## {n_sec}. Stored scores

The package ships stored scores for {n_rerun} methods on `{sds}`. `load_results` reads them and `mtb.plot.bubble` draws them, without running anything.

""" + details(*stored_notes))
        code(f'''long = mtb.load_results("{cat}", dataset="{sds}", source="rerun")
mtb.plot.bubble(long)''')
        n_sec += 1

    # ------------------------------------------------------------ every method
    if has_all(key):
        md(f"""## {n_sec}. Every method of this task

{len(everyone)} method{'s have' if len(everyone) > 1 else ' has'} a variant for {t['cat']} {t['label']}: {and_list(everyone)}. Each runs on `{ds}`. To run them all, set `METHODS` to that list in section 2 and run the notebook again. That downloads {env_size_text(methods=everyone)}.

[Every method on `{ds}`]({SITE}tutorials/{key}_all/) shows that run.""")
    else:
        md(f"""## {n_sec}. Every method of this task

{and_list(everyone)} {'are the only methods' if len(everyone) > 1 else 'is the only method'} with a variant for {t['cat']} {t['label']}, so this tutorial runs every method of the task.""")

    # -------------------------------------------------------- troubleshooting
    md("""## Troubleshooting

`res.failures` lists each method that failed or was skipped. Its `error` column ends with the method's error output.

""" + details(
        """| symptom | fix |
|---|---|
| `env.install` refuses on macOS or Windows | methods run only on Linux: use Colab or a Linux machine |
| a method needs an NVIDIA GPU | choose a GPU runtime on Colab, or a machine with a GPU |
| a method fails on a small dataset | `mtb.scan` names what the method needs from the data's size in its `caveat` column |
| a warning that values are not whole numbers | export raw counts, for example with `rna="layer:counts"` |
| `... matrix/data as cells x features` | the matrix is transposed: export it again with `mtb.io.export_dataset` |
| a method times out | raise `timeout=` in `run_all` |
| low `label_order_confidence` | several label files fit the cell count: check `label_order_candidates` in `res.results` |
| batch metrics use the wrong batches | `res.rescore(batch=my_vector)` scores again without running the methods |"""))

    md(f"""## Next steps

- `mtb.cite(METHODS)` returns the citations for the benchmark and the methods you ran.
- The other tutorials: {", ".join(f"[{TASKS[k]['cat']} {TASKS[k]['label']}]({SITE}tutorials/{k}/)" for k in TASKS if k != key)}.
- The guides: [run]({SITE}tutorials/run/), [evaluate]({SITE}tutorials/evaluate/), [plot]({SITE}tutorials/plot/) and [discover methods]({SITE}tutorials/discover/).
- The [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/) has the full benchmark's rankings, with no install needed.""")
    return C


def has_all(key):
    """Whether the task has methods beyond the tutorial's defaults, and so a
    notebook that runs every method."""
    return set(task_methods(key)) != set(TASKS[key]["methods"])


def graph_methods(key):
    """The methods of a task that return a neighbour graph and no embedding
    (scMoMaT writes a UMAP next to its graph, and that is scored)."""
    from multibench.engine import registry
    t = TASKS[key]
    want = None if t["variants"] is None else [set(v) for v in t["variants"]]
    out = []
    for m in task_methods(key):
        for v in registry.get(m).variants:
            if v.when.get("category") != t["cat"]:
                continue
            if want is None or set(v.when.get("modalities", [])) in want:
                kinds = [v.output.kind] + [o.kind for o in v.extra_outputs]
                if "embedding" not in kinds:
                    out.append(m)
                break
    return out


def build_all_methods(key):
    """The run of every method of a task on its dataset: install, run, plot."""
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    t = TASKS[key]
    cat, ds = t["cat"], t["ds"]
    everyone = task_methods(key)
    extra = run_kwargs(key)
    md(_title(key, ", every method") + f"""

This notebook runs every method that has a variant for {cat} {t['label']} on `{ds}`: {and_list(everyone)}. The [tutorial]({SITE}tutorials/{key}/) explains each step and runs {and_list(t['methods'])} only.

Methods run on Linux. The environments are {env_size_text(methods=everyone)}.""")
    _install(md, code)
    md("## 2. Download the data and the environments")
    code(f'''import pandas as pd
import multibench as mtb

mtb.data.fetch("{ds}")''')
    lines, cur = [], "METHODS = ["
    for m in everyone:
        piece = json.dumps(m) + ", "
        if len(cur) + len(piece) > 88:
            lines.append(cur.rstrip())
            cur = "           "
        cur += piece
    lines.append(cur.rstrip(", ") + "]")
    code("\n".join(lines) + '''
envs = mtb.env.install(METHODS, dry_run=False)
pd.DataFrame(envs)[["env", "methods", "state"]]''')
    graphs = graph_methods(key)
    note = ""
    if graphs:
        note = (f"\n\n{and_list(graphs)} return{'s' if len(graphs) == 1 else ''} a neighbour graph, "
                "not an embedding. The package scores embeddings, so "
                f"{'its row has' if len(graphs) == 1 else 'their rows have'} the status "
                "`RUN_OK_NO_EMBEDDING` and no scores, and the figure leaves "
                f"{'it' if len(graphs) == 1 else 'them'} out.")
    md(f"## 3. Run the method{'s' if len(everyone) > 1 else ''}" + note)
    code(f'''res = mtb.run_all("{ds}", "{cat}", methods=METHODS,
                  out_dir="out/{ds}_all"{extra})
res.summary''')
    md("## 4. Plot\n\n" + PLOT_TEXT)
    code("res.plot()")
    return C


def build_colab_quickstart():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    md("""# scMultiBench API quickstart (Colab)

This notebook installs `multibench-sc`, looks up methods and draws the stored benchmark results. It runs no method, so it downloads nothing else. The integration tutorials install the method environments and run the methods.""")
    code(INSTALL_CELL)
    code("""import multibench as mtb

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
keys = ("id", "language", "env", "needs_labels", "notes", "repo_url", "reference",
        "supports")
{k: info[k] for k in keys}""")
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
mtb.plot.bubble(long)""")
    code("""pair = mtb.load_results("diagonal", dataset=["D28", "D28s"], source="rerun")
mtb.plot.bubble(pair, aggregate="summary", require_complete=True,
                title="Summary of 2 diagonal datasets")""")
    md("## Next steps\n\n- The integration tutorials install the environments, run methods "
       "and repeat the run on your own data: "
       + ", ".join(f"[{t['cat']} {t['label']}]({COLAB}tutorial_{k}.ipynb)" for k, t in TASKS.items())
       + ".\n- The [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/) has "
         "the full benchmark's rankings.")
    return C


if __name__ == "__main__":
    for key in TASKS:
        books = [(f"tutorial_{key}", build_tutorial(key))]
        if has_all(key):
            books.append((f"tutorial_{key}_all", build_all_methods(key)))
        for name, cells in books:
            path = os.path.join(OUT, f"{name}.ipynb")
            nbf.write(_notebook(cells, name), path)
            print(f"wrote {path} {len(cells)} cells")
    C = build_colab_quickstart()
    path = os.path.join(OUT, "colab_quickstart.ipynb")
    nbf.write(_notebook(C, "colab_quickstart"), path)
    print(f"wrote {path} {len(C)} cells")
