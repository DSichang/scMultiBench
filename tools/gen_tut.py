"""Generate one tutorial per integration task (``TASKS``), the notebook that
runs every method of each task, and the Colab quickstart.

The tasks are the 13 of the benchmark article: three vertical, two diagonal,
four mosaic and four cross. A task is named by its modality sets
(``variants``), and a method belongs to the task when the registry gives it a
variant with exactly one of those sets. Every variant of the four categories
belongs to one task. Where a method has two variants in a task (UINMF on two
and on three batches), ``run_all`` runs the larger one on the task's dataset.
``not_wrapped`` names the methods the article evaluates on a task and the
package does not run.

Each tutorial is one straight path, run top to bottom on Linux or Colab:
install the package, download the data and the method environments, run the
task's default methods with ``run_all``, plot, run the same calls on data in
the reader's own format, and draw the benchmark's stored scores where the
package ships them. There are no flags or fallbacks. No notebook defines a
function. The example files of section 5 come from ``mtb.data.write_example``.
A step that cannot work on the reader's computer fails with the package's own
error: the environment install refuses off Linux.

Every method of a task runs on the task's one dataset. The defaults are two
methods of the task (one where it has one) that are fast and need no GPU,
with the smallest environments. ``run_all`` gets the dataset, the category,
the methods and the output folder, and no other argument: where the dataset
also holds the files of a smaller variant of a method, ``run_all`` runs the
variant with the most modalities. ``check_task`` and ``check_own`` stop the
generation when a notebook would not run as written. Every size quoted comes
from the dry-run plans at generation time, never a hand-written figure.

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
The cell asks for the package version the notebooks were generated with or a
later one: section 5 calls what an earlier release does not have.
"""
import hashlib
import json
import os
import re
import warnings

import nbformat as nbf

import multibench as mtb
from multibench.data.examples import EXAMPLES

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


def stored_method_count(cat, dataset):
    """The methods the stored tables hold for the dataset, counted from
    ``results_coverage`` at generation time."""
    cov = mtb.results_coverage(cat)
    return cov[cov.dataset == dataset].method.nunique()


def stored_on_fewer_batches(dataset):
    """The methods whose stored scores for ``dataset`` come from a run on
    fewer batches than the others', from the ``label_order`` column of the
    stored summary (UINMF on ``D52``: its two-batch variant)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(OUT, "results", f"summary_{dataset}.csv"))
    n = df.label_order.str.count(r"\+") + 1
    return sorted(df.method[n < n.max()]), int(n.min()), int(n.max())


def _gb(rows):
    rows = rows.to_dict("records") if hasattr(rows, "to_dict") else rows
    known = [r["archive_bytes"] for r in rows if r["archive_bytes"]]
    # a layer several of the envs share is downloaded once
    shared = mtb.env.layers_for([mtb.env.archive_key(r["env"], r["flavor"]) for r in rows],
                                missing_only=False)
    total = sum(known) + sum(e["archive_bytes"] for e in shared.values())
    return ("" if len(known) == len(rows) else "at least ") + f"{total / 1e9:.1f} GB"


def env_size_text(category=None, methods=None):
    """``"<n> envs, <x> GB to download on a CPU host, <y> GB on a GPU host"``
    for a category / method set, from the dry-run plans ``mtb.env.install(...)``
    returns at generation time for each archive flavour."""
    # dry_run=True: nothing is built
    rows = {f: mtb.env.install(methods, category=category, flavor=f, dry_run=True)
            for f in ("cpu", "gpu")}
    n = len(rows["cpu"])
    return (f"{n} env{'s' if n != 1 else ''}, {_gb(rows['cpu'])} to download on a CPU "
            f"host, {_gb(rows['gpu'])} on a GPU host")


def env_download_sentence(methods):
    """The visible download-size sentence of section 2. The GPU builds carry
    the CUDA libraries, so the two flavours differ unless every env has one
    archive only."""
    cpu = _gb(mtb.env.install(methods, flavor="cpu", dry_run=True))
    gpu = _gb(mtb.env.install(methods, flavor="gpu", dry_run=True))
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
    ATAC form they read, from ``find_methods(cat, modalities=...)`` and each
    variant's modalities at generation time. ``both`` read peaks and gene
    activity together (``atac_peak`` and ``atac_gas`` in one variant)."""
    gas = sorted(mtb.find_methods(cat, modalities=["atac_gas"]))
    peak = sorted(mtb.find_methods(cat, modalities=["atac_peak"]))
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


# The install cell of every notebook (the module docstring says why it pins numpy
# and pandas). The requirement is quoted: in the shell a bare > redirects output.
INSTALL_CELL = ('import numpy, pandas\n'
                'pins = f"numpy=={numpy.__version__} pandas=={pandas.__version__}"\n'
                f'%pip install -q "multibench-sc>={mtb.__version__}" {{pins}}')

COLAB_GPU_NOTE = ("On Colab, choose a GPU runtime before you run the notebook: Runtime -> "
                  "Change runtime type -> T4 GPU. On a CPU runtime, an environment that has a "
                  "smaller CPU build gets that build, and training methods run slower.")

# One notebook per integration task, and one dataset per task: every method of
# the task runs on it (tools/make_tutorial_data.py). The keys, their order and
# the labels follow the benchmark article. `variants` are the modality sets of
# the task's method variants (the cross tasks with three batches also hold
# UINMF's two-batch sets, which belong to no other task); `methods` are the
# two the notebook runs by default (one when the task has a single method);
# `stored_ds` is the dataset whose stored scores the last section draws;
# `not_wrapped` are the methods the article evaluates on the task and the
# package does not run.
def _three(*roles):
    """The roles of three batches: ``rna1, rna2, rna3, adt1, ...``."""
    return [f"{r}{i}" for r in roles for i in (1, 2, 3)]


_MOSAIC = ("Mosaic integration combines batches that share only some modalities. "
           "Each method accepts one batch pattern")
_CROSS = ("Cross integration combines batches that all measure the same "
          "modalities. The task is to remove batch effects and keep the biological "
          "structure. ")
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
   variants=[["rna", "atac"]], methods=["scMM", "VIMCCA"],
   blurb=("Vertical integration combines modalities measured in the same cells. "
          "This tutorial is for RNA and ATAC, as in 10x Multiome. It runs {methods} "
          "on `D27mini_vertical`, 5,000 cells of the benchmark dataset `D27`, and "
          "then on data in your own format. Every method of this task reads ATAC as "
          "a peak matrix."),
 ),
 "vertical_rna_adt_atac": dict(
   cat="vertical", label="RNA + ADT + ATAC", ds="D22mini",
   variants=[["rna", "adt", "atac"]], methods=["Matilda", "UINMF"],
   blurb=("Vertical integration combines modalities measured in the same cells. "
          "This tutorial is for RNA, surface protein and ATAC from the same cells. "
          "It runs {methods} on `D22mini`, 3,000 cells of the benchmark dataset "
          "`D22`, and then on data in your own format."),
 ),
 "diagonal_rna_atac": dict(
   cat="diagonal", label="[RNA, ATAC]", ds="D27mini", stored_ds="D28",
   # one RNA and one ATAC batch; scBridge's variant names no modalities
   variants=[["rna", "atac_gas"], ["rna", "atac_peak"],
             ["rna", "atac_peak", "atac_gas"], []],
   methods=["iNMF", "online_iNMF"],
   blurb=("Diagonal integration combines RNA and ATAC measured in different cells, "
          "with no pairing between them. If your RNA and ATAC come from the same "
          "cells, as in 10x Multiome, use the vertical tutorial. This tutorial runs "
          "{methods} on `D27mini`, 3,000 RNA and 3,000 ATAC profiles of the "
          "benchmark dataset `D27`, then on data in your own format."),
 ),
 "diagonal_multi": dict(
   cat="diagonal", label="[multiple RNA, multiple ATAC]", ds="D37mini",
   variants=[_three("rna", "atac_gas"), _three("rna", "atac_peak")],
   methods=["iNMF", "online_iNMF"], not_wrapped=["Conos"],
   blurb=("Diagonal integration combines RNA and ATAC measured in different cells, "
          "with no pairing between them. This tutorial is for data in several "
          "batches: three RNA batches and three ATAC batches. It runs {methods} on "
          "`D37mini`, 3,000 RNA and 3,000 ATAC profiles of the benchmark dataset "
          "`D37`, then on data in your own format."),
 ),
 "mosaic_rna_adt": dict(
   cat="mosaic", label="[RNA, RNA + ADT, ADT]", ds="D38mini",
   variants=[["rna1", "rna2", "adt2", "adt3"]], methods=["StabMap", "scMoMaT"],
   blurb=(_MOSAIC + ". This tutorial is for an RNA batch, an RNA + ADT batch and "
          "an ADT batch. It runs {methods} on `D38mini`, 6,000 cells of the "
          "benchmark dataset `D38`, and then on data in your own format."),
 ),
 "mosaic_rna_atac": dict(
   cat="mosaic", label="[RNA, RNA + ATAC, ATAC]", ds="D45mini", stored_ds="D45",
   variants=[["rna1", "rna2", "atac2", "atac3"]], methods=["StabMap", "scMoMaT"],
   blurb=(_MOSAIC + ", and every mosaic method reads ATAC as peaks. This tutorial "
          "is for an RNA batch, an RNA + ATAC batch and an ATAC batch. It runs "
          "{methods} on `D45mini`, 6,000 cells of the benchmark dataset `D45`, and "
          "then on data in your own format."),
 ),
 "mosaic_shared": dict(
   cat="mosaic", label="Mixed, with shared modality", ds="D46mini",
   variants=[["rna1", "rna2", "rna3", "adt1", "atac2"]], methods=["StabMap", "UINMF"],
   blurb=(_MOSAIC + ", and every mosaic method reads ATAC as peaks. This tutorial "
          "is for an RNA + ADT batch, an RNA + ATAC batch and an RNA batch, so "
          "every batch holds RNA. It runs {methods} on `D46mini`, 4,500 cells of "
          "the benchmark dataset `D46`, and then on data in your own format."),
 ),
 "mosaic_unshared": dict(
   cat="mosaic", label="Mixed, without shared modality", ds="D49mini",
   variants=[["rna1", "rna2", "adt1", "adt3", "atac2"]], methods=["StabMap", "scMoMaT"],
   blurb=(_MOSAIC + ", and every mosaic method reads ATAC as peaks. This tutorial "
          "is for an RNA + ADT batch, an RNA + ATAC batch and an ADT batch, so no "
          "modality is in every batch. It runs {methods} on `D49mini`, 4,500 cells "
          "of the benchmark dataset `D49`, and then on data in your own format."),
 ),
 "cross_rna_adt": dict(
   cat="cross", label="Multiple RNA + ADT", ds="D52mini", stored_ds="D52",
   variants=[_three("rna", "adt"), ["rna1", "rna2", "adt1", "adt2"]],
   methods=["StabMap", "sciPENN"],
   blurb=(_CROSS + "This tutorial is for batches that each hold RNA and ADT. It "
          "runs {methods} on `D52mini`, 3,000 cells in three batches of the "
          "benchmark dataset `D52`, and then on data in your own format."),
 ),
 "cross_rna_atac": dict(
   cat="cross", label="Multiple RNA + ATAC", ds="D56mini",
   variants=[_three("rna", "atac"), ["rna1", "rna2", "atac1", "atac2"]],
   methods=["StabMap", "scMM"],
   blurb=(_CROSS + "This tutorial is for batches that each hold RNA and ATAC "
          "peaks. It runs {methods} on `D56mini`, 4,500 cells in three batches, "
          "cut from the first three batches of the benchmark dataset `D56`, and "
          "then on data in your own format."),
 ),
 "cross_adt_atac": dict(
   cat="cross", label="Multiple ADT + ATAC", ds="D58mini",
   variants=[["adt1", "adt2", "atac1", "atac2"]], methods=["StabMap", "UINMF"],
   blurb=(_CROSS + "This tutorial is for batches that each hold ADT and ATAC "
          "peaks, with no RNA. It runs {methods} on `D58mini`, 3,000 cells in two "
          "batches of the benchmark dataset `D58`, and then on data in your own "
          "format."),
 ),
 "cross_rna_adt_atac": dict(
   cat="cross", label="Multiple RNA + ADT + ATAC", ds="D59mini",
   variants=[["rna1", "rna2", "adt1", "adt2", "atac1", "atac2"]],
   methods=["StabMap", "UINMF"],
   blurb=(_CROSS + "This tutorial is for batches that each hold RNA, ADT and "
          "ATAC peaks. It runs {methods} on `D59mini`, 3,000 cells in two batches "
          "of the benchmark dataset `D59`, and then on data in your own format."),
 ),
}

def task_variants(key):
    """``(method, variant)`` for every registry variant of the task: a variant
    of the task's category whose modality set is one of ``variants``."""
    from multibench.engine import registry
    t = TASKS[key]
    want = [set(v) for v in t["variants"]]
    return [(m, v) for m in registry.list_methods() for v in registry.get(m).variants
            if v.when.get("category") == t["cat"]
            and set(v.when.get("modalities", [])) in want]


def task_methods(key):
    """Every method with a variant of the task, in name order."""
    return sorted({m for m, _ in task_variants(key)})


def run_variants(key):
    """``{method: modalities}`` of the variant each method of the task runs on
    the task's dataset, as the ``modalities`` column of ``scan`` spells it. A
    method with two variants in the task runs the one with more files."""
    best = {}
    for m, v in task_variants(key):
        mods = v.when.get("modalities") or []
        if m not in best or len(mods) > len(best[m]):
            best[m] = mods
    return {m: "+".join(mods) or "(data_dir)" for m, mods in best.items()}


def would_run(key, methods, dataset=None, **where):
    """The rows ``run_all`` runs for ``methods`` with the notebook's
    arguments, on a computer that has every environment: the rows with their
    input files that nothing but the environment blocks, less the rows inside
    a larger one of the same method."""
    import warnings
    from multibench import workflow
    t = TASKS[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = mtb.run_all(dataset or t["ds"], t["cat"], methods=methods, dry_run=True,
                           verbose=False, assume_gpu=True, **where)
    other = [str(r).replace(str(e), "").strip() for r, e in zip(plan.reason, plan.env_reason)]
    rows = plan[[bool(f) and o == "" for f, o in zip(plan.files_ok, other)]]
    return rows.drop(index=workflow._nested_rows(rows)[0])


def check_task(key):
    """Stop when the notebooks' ``run_all`` call would not run every method
    of the task once, on the task's variant, when a method reads only some
    batches of the dataset, or when section 5 does not work as it stands
    (``check_own``)."""
    t = TASKS[key]
    everyone = task_methods(key)
    for m in t["methods"]:
        if m not in everyone:
            raise SystemExit(f"{m} has no variant of {key}")
    rows = would_run(key, everyone)
    ran = sorted(zip(rows.method, rows.modalities))
    if ran != sorted(run_variants(key).items()):
        raise SystemExit(f"{key}: run_all on {t['ds']} would run {ran}, not one row per "
                         f"method of the task: change the dataset or name the modalities")
    with warnings.catch_warnings():          # a count of the files: no order needed
        warnings.simplefilter("ignore", UserWarning)
        n = len(mtb.labels_for(t["ds"]))
    for m in everyone:
        if len(mtb.labels_for(t["ds"], t["cat"], m)) < n:
            raise SystemExit(f"{m} reads only some batches of {t['ds']}: reword the notebooks")
    # the vertical, mosaic and cross texts say that ATAC is a peak matrix
    gas = mtb.find_methods(t["cat"], modalities=["atac_gas"])
    if t["cat"] != "diagonal" and gas:
        raise SystemExit(f"{and_list(gas)} read gene activity in {t['cat']}: reword the "
                         f"{t['cat']} blurbs and own-data sections")
    check_own(key)


def labels_for_ds(key, method="UINMF"):
    """The label files ``method`` reads on the task's dataset."""
    return list(mtb.labels_for(TASKS[key]["ds"], TASKS[key]["cat"], method))


def not_wrapped_sentence(key):
    """One sentence on the method the article evaluates on the task and the
    package does not run there, or "" when there is none. The wording is for
    Conos, whose script the package runs on one RNA and one ATAC file."""
    names = TASKS[key].get("not_wrapped", [])
    if not names:
        return ""
    if names != ["Conos"]:
        raise SystemExit(f"{key}: not_wrapped is {names}: reword not_wrapped_sentence")
    return ("The article also evaluates Conos on this task. Its published script reads a "
            "third input file in another format, so the package does not run it on more "
            "than two files.")


def diagonal_multi_sentence(key):
    """The visible ATAC-form sentence of the several-batch diagonal tutorial,
    from the modalities of the task's variants."""
    gas = sorted({m for m, v in task_variants(key)
                  if any(r.startswith("atac_gas") for r in v.when["modalities"])})
    peak = sorted({m for m, v in task_variants(key)
                   if any(r.startswith("atac_peak") for r in v.when["modalities"])})
    if not (gas and peak) or set(gas) & set(peak):
        raise SystemExit(f"{key}: ATAC forms changed: reword the title cell")
    return (f"{and_list(gas)} read ATAC as gene-activity scores, made beforehand with a "
            f"tool such as Signac or ArchR. {and_list(peak)} "
            f"read{'s' if len(peak) == 1 else ''} the peak matrix. The dataset holds both.")


# ---------------------------------------------------------------- own data
# Section 5 per task: the files a reader's data usually comes in, the export
# call, a check of the folder and the same run_all on it.
# mtb.data.write_example writes example files under the names the read cell
# reads, so the notebook runs before the reader has put a file in place. A
# name the reader changes is marked "# <-". The folder name is the variable
# NAME, which the check and the run use. overwrite=True lets the notebook run
# twice.
#
# The paragraphs of the collapsed block, in the order they appear.
WHERE = ("The names in quotes say where your file holds each part. `\"X\"` is the main "
         "matrix, `\"obsm:protein\"` the matrix `obsm[\"protein\"]`, `\"layer:counts\"` the "
         "layer `layers[\"counts\"]`, and `\"obs:celltype\"` the column `obs[\"celltype\"]`, "
         "which `\"celltype\"` alone names too. A wrong name after `obsm:`, `layer:` or "
         "`obs:` raises an error that lists the names the file has.")
PROTEIN_NAMES = ("The example keeps the protein names in `adata.uns[\"protein_names\"]`. A "
                 "DataFrame in `adata.obsm[\"protein\"]` brings its own names in its columns.")
ADT_IN_X = ("When the ADT file also holds a matrix in `obsm` under a name that does not "
            "start with `X_`, `export_dataset` asks which one is the protein matrix. "
            "`X_umap` and `X_pca` are left alone.")
TENX_NOTE = ("A 10x Multiome file read with scanpy and `gex_only=False` holds genes and "
             "peaks in one `X`. The recipe for a 10x `.h5` file splits it into the two "
             "objects of this section. Without a split or a filter, `export_dataset` raises "
             "rather than write genes and peaks into one file.")
MUDATA_NOTE = ("A 10x Multiome MuData goes in with one call. Here the labels are in "
               "`mdata.obs`. For labels in `mdata[\"rna\"].obs`, write "
               "`labels=\"rna:celltype\"`.\n\n"
               "```python\n"
               "mtb.io.export_dataset(mdata, \"mydata/\" + NAME, rna=\"rna\", atac=\"atac\",\n"
               "                      atac_kind=\"peak\", labels=\"obs:celltype\",\n"
               "                      category=\"vertical\")\n"
               "```")
DIAG_CLI = ("The command line writes the same folder from the two files:\n\n"
            "```\n"
            "multibench convert my_rna.h5ad mydata/{name} --rna X --atac-from my_atac.h5ad \\\n"
            "    --atac-kind gene_activity --labels obs:celltype --category diagonal\n"
            "```")
DIAG_ONE_SIDE = ("`labels=` needs the column in both files. With cell types for the RNA "
                 "cells only, leave `labels=` out and follow the recipe for no labels.")
DIAG_PEAK = ("For ATAC as a peak matrix, pass `atac_kind=\"peak\"`. {peak_only} read "
             "peaks. `mtb.describe_layout(\"diagonal\")` lists the files each method needs.")
MULTI_NUMBERING = ("The RNA batches and the ATAC batches are numbered separately. RNA batch 1 "
                   "and ATAC batch 1 need not come from the same sample. The loop takes as "
                   "many batches as the list `rna` holds.")
MULTI_PEAK = ("For ATAC as a peak matrix, pass `atac_kind=\"peak\"`. {multi_peak} "
              "the peak matrix.")
MOSAIC_LAYOUT = ("`mtb.describe_layout(\"mosaic\")` lists the batch patterns and the methods "
                 "that accept each one. The command line writes one batch per call with "
                 "`multibench convert ... --category mosaic --batch-index N`.")
MOSAIC_ONE_FILE = ("A batch with RNA and ADT in one file goes in with a selector, for "
                   "example `adt=\"obsm:protein\"`.")
CROSS_ORDER = ("The batches are numbered in the sorted order of their names, read as text: "
               "donors `A`, `B` and `C` become batches 1, 2 and 3, and `10` comes before `2`.")
CROSS_INDEX = ("For one AnnData per batch, call `export_dataset` once per batch with "
               "`batch_index=N` instead of `batch=`, and pass `n_batches=` with the number of "
               "batches in every call.")
FOLDER_NOTE = ("`\"mydata/\" + NAME` is the dataset folder that `scan` and `run_all` read. "
               "With a new value of `NAME`, the folder holds the files of your data only.")
CHECK_NOTE = ("The check lists the methods you name. Without `methods=`, it lists every "
              "method that reads the modalities in the folder. `mtb.run_all(..., "
              "dry_run=True)` has one row per method variant and every column, such as the "
              "command of each method.")
LEFTOVER = ("`export_dataset` raises `FileExistsError` when the folder holds files of other "
            "cells that the call does not write. `overwrite=True` does not remove them. Give "
            "`NAME` a new value.")
LEFTOVER_BATCH = ("After `batch=`, the files of batch numbers that the call does not write "
                  "count as well.")
LEFTOVER_INDEX = ("A call with `batch_index=` checks the files of its own batch number. Files "
                  "of other batch numbers stay, unless `n_batches=` says that the data has "
                  "fewer batches.")
OVERWRITE_NOTE = ("`overwrite=True` replaces the files of an earlier run of this cell. "
                  "Without it, `export_dataset` raises `FileExistsError` rather than "
                  "replace a file.")


def batch_advice(obj):
    """What a vertical tutorial says on samples or donors, as ``(visible,
    collapsed)``: the sentence of the first cell with the call, and the
    paragraph of the collapsed block. ``obj`` is the variable of the read
    cell whose ``obs`` holds the column."""
    return (("Several donors or samples in the file? Keep them in one folder, without "
             "`batch=` in `export_dataset`, and give `run_all` the column itself: "
             f"`mtb.run_all(..., batch={obj}.obs[\"donor\"])`. The summary then has the "
             "scores for how well they mix."),
            ("The donors that `run_all` gets as `batch=` are matched to the cells by "
             "barcode, and every cell needs a donor. A column name alone is not enough "
             "there, because `run_all` reads the folder and the folder holds no `obs` "
             "columns."))


# Words the visible sentences share.
BATCH_WORD = "A batch is a donor, a sample or an experiment."
ADT_WORD = "ADT means the protein counts."
GAS_WORD = ("Gene-activity scores are an ATAC matrix with one column per gene, as Signac or "
            "ArchR compute it.")
PER_BATCH = "`mtb.io.export_dataset` with `batch=` writes one set of files per batch."
MOSAIC_NEEDS = ("Your data needs raw counts, with one `.h5ad` file per batch and modality and "
                f"a cell type for each cell. {BATCH_WORD} `mtb.io.export_dataset` with "
                "`batch_index=` writes one batch per call. Number the batches as below:")
NUMBER = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# The recipes of the run guide, for data that does not come as the files of
# this section, and the ones the tutorials of each category link. A vertical
# tutorial links the donors recipe as the one for a donor or sample column,
# a cross tutorial as the one for another number of batches than its methods
# read.
RECIPES = {
    "10x": f"[a 10x `.h5` file]({SITE}tutorials/run/#recipe-10x-h5)",
    "csv": f"[labels in a CSV file]({SITE}tutorials/run/#recipe-labels-csv)",
    "no_labels": f"[no labels]({SITE}tutorials/run/#recipe-no-labels)",
    "donors": f"[another number of donors]({SITE}tutorials/run/#recipe-donors)",
    "donors_vertical": f"[several donors or samples]({SITE}tutorials/run/#recipe-donors)",
}
RECIPES_OF = {"vertical": ["10x", "csv", "no_labels", "donors_vertical"],
              "diagonal": ["csv", "no_labels"],
              "mosaic": ["10x", "csv", "no_labels"],
              "cross": ["10x", "csv", "no_labels", "donors"]}

# `name` is the folder the export cell writes, a name of its own per tutorial
# and none that a 0.3.4 tutorial wrote another layout under; `needs` is the
# visible sentence; `example` says what the example cell writes and `n_files`
# how many files that is; `read` and `export` are the two cells a reader
# edits; `notes` are the task's paragraphs of the collapsed block; `donors`
# is the variable of the read cell that `batch_advice` names.
OWN = {
 "vertical_rna_adt": dict(
   name="MYCITE",
   needs=("Your data needs raw counts for both modalities and a cell type for each "
          f"cell, in one `.h5ad` file. {ADT_WORD}"),
   example="a small example file, `my_file.h5ad`, from 60% of the cells of `D11`",
   n_files=1,
   read='''import anndata as ad

adata = ad.read_h5ad("my_file.h5ad")   # <- your file
adata''',
   export='''NAME = "MYCITE"   # <- a new name for your own data
mtb.io.export_dataset(adata, "mydata/" + NAME,
                      rna="X",                # <- where the RNA counts are
                      adt="obsm:protein",     # <- where the proteins (ADT) are
                      labels="obs:celltype",  # <- the cell-type column
                      overwrite=True)''',
   notes=[PROTEIN_NAMES], donors="adata",
 ),
 "vertical_rna_atac": dict(
   name="MYMULTIOME",
   needs=("Your data needs raw RNA counts, an ATAC matrix of the same cells and a cell "
          "type for each cell, as two `.h5ad` files. Every method of this task reads "
          "ATAC as peaks, so the folder holds peaks."),
   example=("two small example files, `my_rna.h5ad` and `my_atac.h5ad`, from 60% of the "
            "cells of `D27mini_vertical`"),
   n_files=2,
   read='''import anndata as ad

rna = ad.read_h5ad("my_rna.h5ad")     # <- your RNA file
atac = ad.read_h5ad("my_atac.h5ad")   # <- your ATAC file, the same cells
print(rna)
print(atac)''',
   export='''NAME = "MYMULTIOME"   # <- a new name for your own data
mtb.io.export_dataset(rna, "mydata/" + NAME,
                      rna="X",                # <- where the RNA counts are
                      atac=atac,              # the ATAC file, peak counts in X
                      atac_kind="peak",       # the methods here read peaks
                      labels="obs:celltype",  # <- the cell-type column
                      overwrite=True)''',
   notes=[TENX_NOTE, MUDATA_NOTE], donors="rna",
 ),
 "vertical_rna_adt_atac": dict(
   name="MYTRI",
   needs=("Your data needs raw counts for RNA and protein (ADT) in one `.h5ad` file, an "
          "ATAC peak matrix of the same cells in another, and a cell type for each cell."),
   example=("two small example files, `my_file.h5ad` and `my_atac.h5ad`, from 60% of the "
            "cells of `D22mini`"),
   n_files=2,
   read='''import anndata as ad

adata = ad.read_h5ad("my_file.h5ad")   # <- your RNA and protein file
atac = ad.read_h5ad("my_atac.h5ad")    # <- your ATAC file, the same cells
print(adata)
print(atac)''',
   export='''NAME = "MYTRI"   # <- a new name for your own data
mtb.io.export_dataset(adata, "mydata/" + NAME,
                      rna="X",                # <- where the RNA counts are
                      adt="obsm:protein",     # <- where the proteins (ADT) are
                      atac=atac,              # the ATAC file, peak counts in X
                      atac_kind="peak",       # the methods here read peaks
                      labels="obs:celltype",  # <- the cell-type column
                      overwrite=True)''',
   notes=[PROTEIN_NAMES], donors="adata",
 ),
 "diagonal_rna_atac": dict(
   name="MYDIAG_RNA_ATAC",
   needs=("Your data needs raw RNA counts in one `.h5ad` file and ATAC in another, with a "
          "cell type for each cell in both. iNMF and online_iNMF read ATAC as "
          "gene-activity scores, and online_iNMF needs about 5,000 cells in total. "
          f"{GAS_WORD}"),
   example=("two small example files, `my_rna.h5ad` and `my_atac.h5ad`, from 90% of the "
            "RNA profiles and 90% of the ATAC profiles of `D27mini`"),
   n_files=2,
   read='''import anndata as ad

rna = ad.read_h5ad("my_rna.h5ad")     # <- your RNA file
atac = ad.read_h5ad("my_atac.h5ad")   # <- your ATAC file, other cells
print(rna)
print(atac)''',
   export='''NAME = "MYDIAG_RNA_ATAC"   # <- a new name for your own data
mtb.io.export_dataset(rna, "mydata/" + NAME,
                      rna="X",                    # <- where the RNA counts are
                      atac=atac,                  # the ATAC file, values in X
                      atac_kind="gene_activity",  # <- or "peak"
                      labels="obs:celltype",      # <- the column in both files
                      category="diagonal", overwrite=True)''',
   notes=[DIAG_CLI, DIAG_ONE_SIDE, DIAG_PEAK],
 ),
 "diagonal_multi": dict(
   name="MYDIAG_BATCHES",
   needs=("Your data needs raw RNA counts and ATAC as one `.h5ad` file per batch, with a "
          f"cell type for each cell. {BATCH_WORD} `mtb.io.export_dataset` with "
          "`batch_index=` writes one RNA batch and one ATAC batch per call. iNMF and "
          "online_iNMF read ATAC as gene-activity scores, and online_iNMF needs about "
          f"5,000 cells in total. {GAS_WORD}"),
   example=("six small example files, `my_rna1.h5ad` to `my_rna3.h5ad` and "
            "`my_atac1.h5ad` to `my_atac3.h5ad`, from the first 900 RNA profiles and the "
            "first 900 ATAC profiles of each batch of `D37mini`"),
   n_files=6,
   read='''import anndata as ad

rna1 = ad.read_h5ad("my_rna1.h5ad")     # <- RNA batch 1: your file
rna2 = ad.read_h5ad("my_rna2.h5ad")     # <- RNA batch 2: your file
rna3 = ad.read_h5ad("my_rna3.h5ad")     # <- RNA batch 3: your file
atac1 = ad.read_h5ad("my_atac1.h5ad")   # <- ATAC batch 1: your file
atac2 = ad.read_h5ad("my_atac2.h5ad")   # <- ATAC batch 2: your file
atac3 = ad.read_h5ad("my_atac3.h5ad")   # <- ATAC batch 3: your file
print(rna1)
print(atac1)''',
   export='''NAME = "MYDIAG_BATCHES"        # <- a new name for your own data
rna = [rna1, rna2, rna3]       # <- your RNA batches
atac = [atac1, atac2, atac3]   # <- your ATAC batches
for b in range(len(rna)):      # one call per batch
    mtb.io.export_dataset(rna[b], "mydata/" + NAME,
                          rna="X",                    # <- the RNA counts
                          atac=atac[b],               # the ATAC batch
                          atac_kind="gene_activity",  # <- or "peak"
                          labels="obs:celltype",      # <- the cell-type column
                          category="diagonal", batch_index=b + 1,
                          n_batches=len(rna), overwrite=True)''',
   notes=[MULTI_NUMBERING, MULTI_PEAK],
 ),
 "mosaic_rna_adt": dict(
   name="MYMOSAIC_RNA_ADT",
   needs=f"{MOSAIC_NEEDS} RNA, then RNA + ADT, then ADT. {ADT_WORD}",
   example=("four small example files, one per batch and modality, from the first 1,500 "
            "cells of each batch of `D38mini`"),
   n_files=4,
   read='''import anndata as ad

rna1 = ad.read_h5ad("my_rna1.h5ad")   # <- batch 1: your RNA file
rna2 = ad.read_h5ad("my_rna2.h5ad")   # <- batch 2: your RNA file
adt2 = ad.read_h5ad("my_adt2.h5ad")   # <- batch 2: proteins (ADT), same cells
adt3 = ad.read_h5ad("my_adt3.h5ad")   # <- batch 3: your protein (ADT) file
print(rna2)
print(adt2)''',
   export='''NAME = "MYMOSAIC_RNA_ADT"   # <- a new name for your own data
out = "mydata/" + NAME
celltype = "obs:celltype"   # <- the cell-type column of every file
# batch 1: RNA only
mtb.io.export_dataset(rna1, out, rna="X", labels=celltype,
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ADT
mtb.io.export_dataset(rna2, out, rna="X", adt=adt2, labels=celltype,
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ADT only
mtb.io.export_dataset(adt3, out, rna=None, adt="X", labels=celltype,
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=[MOSAIC_LAYOUT, MOSAIC_ONE_FILE, ADT_IN_X],
 ),
 "mosaic_rna_atac": dict(
   name="MYMOSAIC_RNA_ATAC",
   needs=f"{MOSAIC_NEEDS} RNA, then RNA + ATAC, then ATAC. Give ATAC as peaks.",
   example=("four small example files, one per batch and modality, from the first 1,500 "
            "cells of each batch of `D45mini`"),
   n_files=4,
   read='''import anndata as ad

rna1 = ad.read_h5ad("my_rna1.h5ad")     # <- batch 1: your RNA file
rna2 = ad.read_h5ad("my_rna2.h5ad")     # <- batch 2: your RNA file
atac2 = ad.read_h5ad("my_atac2.h5ad")   # <- batch 2: peaks of the same cells
atac3 = ad.read_h5ad("my_atac3.h5ad")   # <- batch 3: your peak file
print(rna2)
print(atac2)''',
   export='''NAME = "MYMOSAIC_RNA_ATAC"   # <- a new name for your own data
out = "mydata/" + NAME
celltype = "obs:celltype"   # <- the cell-type column of every file
# batch 1: RNA only
mtb.io.export_dataset(rna1, out, rna="X", labels=celltype,
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna2, out, rna="X", atac=atac2, atac_kind="peak",
                      labels=celltype,
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ATAC peaks only
mtb.io.export_dataset(atac3, out, rna=None, atac="X", atac_kind="peak",
                      labels=celltype,
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=[MOSAIC_LAYOUT],
 ),
 "mosaic_shared": dict(
   name="MYMOSAIC_SHARED",
   needs=(f"{MOSAIC_NEEDS} RNA + ADT, then RNA + ATAC, then RNA. Give ATAC as peaks. "
          f"{ADT_WORD}"),
   example=("five small example files, one per batch and modality, from the first 1,000 "
            "cells of each batch of `D46mini`"),
   n_files=5,
   read='''import anndata as ad

rna1 = ad.read_h5ad("my_rna1.h5ad")     # <- batch 1: your RNA file
adt1 = ad.read_h5ad("my_adt1.h5ad")     # <- batch 1: proteins (ADT), same cells
rna2 = ad.read_h5ad("my_rna2.h5ad")     # <- batch 2: your RNA file
atac2 = ad.read_h5ad("my_atac2.h5ad")   # <- batch 2: peaks of the same cells
rna3 = ad.read_h5ad("my_rna3.h5ad")     # <- batch 3: your RNA file
print(rna1)
print(adt1)''',
   export='''NAME = "MYMOSAIC_SHARED"   # <- a new name for your own data
out = "mydata/" + NAME
celltype = "obs:celltype"   # <- the cell-type column of every file
# batch 1: RNA + ADT
mtb.io.export_dataset(rna1, out, rna="X", adt=adt1, labels=celltype,
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna2, out, rna="X", atac=atac2, atac_kind="peak",
                      labels=celltype,
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: RNA only
mtb.io.export_dataset(rna3, out, rna="X", labels=celltype,
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=[MOSAIC_LAYOUT, MOSAIC_ONE_FILE, ADT_IN_X],
 ),
 "mosaic_unshared": dict(
   name="MYMOSAIC_UNSHARED",
   needs=(f"{MOSAIC_NEEDS} RNA + ADT, then RNA + ATAC, then ADT. Give ATAC as peaks. "
          f"{ADT_WORD}"),
   example=("five small example files, one per batch and modality, from the first 1,000 "
            "cells of each batch of `D49mini`"),
   n_files=5,
   read='''import anndata as ad

rna1 = ad.read_h5ad("my_rna1.h5ad")     # <- batch 1: your RNA file
adt1 = ad.read_h5ad("my_adt1.h5ad")     # <- batch 1: proteins (ADT), same cells
rna2 = ad.read_h5ad("my_rna2.h5ad")     # <- batch 2: your RNA file
atac2 = ad.read_h5ad("my_atac2.h5ad")   # <- batch 2: peaks of the same cells
adt3 = ad.read_h5ad("my_adt3.h5ad")     # <- batch 3: your protein (ADT) file
print(rna1)
print(adt1)''',
   export='''NAME = "MYMOSAIC_UNSHARED"   # <- a new name for your own data
out = "mydata/" + NAME
celltype = "obs:celltype"   # <- the cell-type column of every file
# batch 1: RNA + ADT
mtb.io.export_dataset(rna1, out, rna="X", adt=adt1, labels=celltype,
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna2, out, rna="X", atac=atac2, atac_kind="peak",
                      labels=celltype,
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ADT only
mtb.io.export_dataset(adt3, out, rna=None, adt="X", labels=celltype,
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=[MOSAIC_LAYOUT, MOSAIC_ONE_FILE, ADT_IN_X],
 ),
 "cross_rna_adt": dict(
   name="MYCROSS_RNA_ADT",
   needs=("Your data needs raw RNA and ADT counts in one `.h5ad` file, with a cell type "
          f"and a batch for each cell. {ADT_WORD} {BATCH_WORD} {PER_BATCH}"),
   example=("a small example file, `my_file.h5ad`, from 60% of the cells of `D52mini`, "
            "with a batch column"),
   n_files=1,
   read='''import anndata as ad

adata = ad.read_h5ad("my_file.h5ad")   # <- your file, every batch in it
adata''',
   export='''NAME = "MYCROSS_RNA_ADT"   # <- a new name for your own data
mtb.io.export_dataset(adata, "mydata/" + NAME,
                      rna="X",                # <- where the RNA counts are
                      adt="obsm:protein",     # <- where the proteins (ADT) are
                      labels="obs:celltype",  # <- the cell-type column
                      batch="obs:batch",      # <- the batch or donor column
                      category="cross", overwrite=True)''',
   notes=[PROTEIN_NAMES],
 ),
 "cross_rna_atac": dict(
   name="MYCROSS_RNA_ATAC",
   needs=("Your data needs raw RNA counts in one `.h5ad` file and an ATAC peak matrix of "
          "the same cells in another, with a cell type and a batch for each cell. "
          f"{BATCH_WORD} {PER_BATCH} Give the same peaks in every batch."),
   example=("two small example files, `my_rna.h5ad` and `my_atac.h5ad`, from 60% of the "
            "cells of `D56mini`, with a batch column"),
   n_files=2,
   read='''import anndata as ad

rna = ad.read_h5ad("my_rna.h5ad")     # <- your RNA file, every batch in it
atac = ad.read_h5ad("my_atac.h5ad")   # <- your peak file, the same cells
print(rna)
print(atac)''',
   export='''NAME = "MYCROSS_RNA_ATAC"   # <- a new name for your own data
mtb.io.export_dataset(rna, "mydata/" + NAME,
                      rna="X",                # <- where the RNA counts are
                      atac=atac,              # the ATAC file, peak counts in X
                      atac_kind="peak",       # the methods here read peaks
                      labels="obs:celltype",  # <- the cell-type column
                      batch="obs:batch",      # <- the batch or donor column
                      category="cross", overwrite=True)''',
   notes=[],
 ),
 "cross_adt_atac": dict(
   name="MYCROSS_ADT_ATAC",
   needs=("Your data needs raw ADT counts in one `.h5ad` file and an ATAC peak matrix of "
          "the same cells in another, with a cell type and a batch for each cell. "
          f"{ADT_WORD} {BATCH_WORD} {PER_BATCH} `rna=None` says that the data has no RNA."),
   example=("two small example files, `my_adt.h5ad` and `my_atac.h5ad`, from 60% of the "
            "cells of `D58mini`, with a batch column"),
   n_files=2,
   read='''import anndata as ad

adt = ad.read_h5ad("my_adt.h5ad")     # <- your protein (ADT) file, every batch
atac = ad.read_h5ad("my_atac.h5ad")   # <- your peak file, the same cells
print(adt)
print(atac)''',
   export='''NAME = "MYCROSS_ADT_ATAC"   # <- a new name for your own data
mtb.io.export_dataset(adt, "mydata/" + NAME,
                      rna=None,               # this data has no RNA
                      adt="X",                # <- where the proteins (ADT) are
                      atac=atac,              # the ATAC file, peak counts in X
                      atac_kind="peak",       # the methods here read peaks
                      labels="obs:celltype",  # <- the cell-type column
                      batch="obs:batch",      # <- the batch or donor column
                      category="cross", overwrite=True)''',
   notes=[ADT_IN_X],
 ),
 "cross_rna_adt_atac": dict(
   name="MYCROSS_TRI",
   needs=("Your data needs raw RNA and ADT counts in one `.h5ad` file and an ATAC peak "
          "matrix of the same cells in another, with a cell type and a batch for each "
          f"cell. {ADT_WORD} {BATCH_WORD} {PER_BATCH}"),
   example=("two small example files, `my_file.h5ad` and `my_atac.h5ad`, from 60% of the "
            "cells of `D59mini`, with a batch column"),
   n_files=2,
   read='''import anndata as ad

adata = ad.read_h5ad("my_file.h5ad")   # <- your RNA and protein file
atac = ad.read_h5ad("my_atac.h5ad")    # <- your peak file, the same cells
print(adata)
print(atac)''',
   export='''NAME = "MYCROSS_TRI"   # <- a new name for your own data
mtb.io.export_dataset(adata, "mydata/" + NAME,
                      rna="X",                # <- where the RNA counts are
                      adt="obsm:protein",     # <- where the proteins (ADT) are
                      atac=atac,              # the ATAC file, peak counts in X
                      atac_kind="peak",       # the methods here read peaks
                      labels="obs:celltype",  # <- the cell-type column
                      batch="obs:batch",      # <- the batch or donor column
                      category="cross", overwrite=True)''',
   notes=[PROTEIN_NAMES],
 ),
}


def mosaic_features_sentence(key):
    """What the files of two batches must share, for the modalities that the
    task has in more than one batch. scMoMaT reads ``matrix/data`` alone
    (tools_scripts/scMoMaT/util.py) and takes feature ``i`` of one batch as
    feature ``i`` of the next; StabMap and UINMF match the names. The folder
    check compares no feature names across batches."""
    roles = TASKS[key]["variants"][0]
    words = {"rna": "RNA file the same gene names", "adt": "ADT file the same protein names",
             "atac": "ATAC file the same peaks"}
    several = [w for fam, w in words.items()
               if sum(r.rstrip("0123456789") == fam for r in roles) > 1]
    several[0] = several[0].replace(" file ", " file needs ")
    return ("Every " + " and every ".join(several) + ", in the same order. scMoMaT matches "
            "the features of two batches by position, and the check does not compare their "
            "names across batches.")


def cross_batches_sentence(key):
    """How many batches the methods of a cross task read, for the visible
    text of section 5's first cell, from the modalities of the task's
    variants: every method the same number, or every method one number and
    some of them a smaller one as well (UINMF's two-batch variants)."""
    reads = {}
    for m, v in task_variants(key):
        numbers = [int(n) for r in v.when["modalities"] for n in re.findall(r"\d+$", r)]
        if not numbers or len(numbers) != len(v.when["modalities"]):
            raise SystemExit(f"{key}: {m} reads {v.when['modalities']}, not numbered "
                             f"batches: reword cross_batches_sentence")
        reads.setdefault(max(numbers), set()).add(m)
    most, *fewer = sorted(reads, reverse=True)
    if reads[most] != set(task_methods(key)) or len(fewer) > 1 or not set(reads) <= set(NUMBER):
        shape = {n: sorted(ms) for n, ms in sorted(reads.items())}
        raise SystemExit(f"{key}: the batches each method reads are {shape}: reword "
                         f"cross_batches_sentence")
    text = f"The methods of this task read {NUMBER[most]} batches"
    if fewer:
        also = sorted(reads[fewer[0]])
        text += (f", and {and_list(also)} also read{'s' if len(also) == 1 else ''} "
                 f"{NUMBER[fewer[0]]}")
    return text + ". The recipe for another number of donors covers the other cases."


def check_own(key):
    """Stop when section 5 of a task does not work as it stands. The example
    cell has to write the files the read cell reads, and the export cell has
    to name its folder in ``NAME``. Run in an empty folder, the three cells
    must not warn, and the folder they write has to hold the input files of
    the notebook's methods."""
    import contextlib
    import io
    import tempfile
    import warnings
    import pandas as pd
    t, own = TASKS[key], OWN[key]
    files = EXAMPLES[t["ds"]]
    read = re.findall(r'"(my_\w+\.h5ad)"', own["read"])
    if sorted(read) != sorted(files) or own["n_files"] != len(files):
        raise SystemExit(f"{key}: the read cell reads {read} and n_files is "
                         f"{own['n_files']}, and write_example writes {list(files)} for "
                         f"{t['ds']}")
    count = "a" if len(files) == 1 else NUMBER[len(files)]
    if not own["example"].startswith(f"{count} small example file"):
        raise SystemExit(f"{key}: write_example writes {len(files)} files for {t['ds']}: "
                         f"reword the example sentence")
    if not re.match(rf'NAME = "{own["name"]}" +# <- a new name for your own data\n',
                    own["export"]):
        raise SystemExit(f"{key}: the export cell does not start with NAME = "
                         f"\"{own['name']}\" and its mark")
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            ns = {"mtb": mtb, "pd": pd, "METHODS": list(t["methods"])}
            with warnings.catch_warnings(record=True) as seen, \
                    contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                mtb.data.write_example(t["ds"], quiet=True)
                exec(compile(own["read"], f"tutorial_{key}: read cell", "exec"), ns)
                exec(compile(own["export"], f"tutorial_{key}: export cell", "exec"), ns)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = mtb.run_all("mydata/" + own["name"], t["cat"],
                                    dry_run=True, verbose=False)
        finally:
            os.chdir(here)
    warned = [str(w.message) for w in seen if issubclass(w.category, UserWarning)]
    if warned:
        raise SystemExit(f"{key}: the cells of section 5 warn: {warned}")
    lacking = sorted(set(t["methods"]) - set(frame[frame.files_ok].method))
    if lacking:
        raise SystemExit(f"{key}: the folder section 5 writes lacks the input files of "
                         f"{and_list(lacking)}")


def own_cells(key):
    """Section 5 of a task tutorial as ``[(kind, source), ...]``: the heading
    with its collapsed block, then a sentence and a code cell each for the
    example files, the read, the export, the check and the run, and the
    plot."""
    t, own = TASKS[key], OWN[key]
    cat, methods, export = t["cat"], t["methods"], own["export"]
    one = own["n_files"] == 1
    same = "the same method" if len(methods) == 1 else "the same methods"
    peak_only = and_list(atac_forms("diagonal")[1])
    multi_peak = [m for m, v in task_variants(key) if "atac_peak1" in v.when["modalities"]]
    multi_peak = and_list(multi_peak) + (" reads" if len(multi_peak) == 1 else " read") \
        if multi_peak else ""
    notes = [WHERE] + [n.replace("{peak_only}", peak_only).replace("{multi_peak}", multi_peak)
                       .replace("{name}", own["name"]) for n in own["notes"]]
    # a paragraph of its own between the needs and the recipes: what to do
    # with a donor column, or how many batches the methods read
    also = ""
    if cat == "vertical":
        also, donors_note = batch_advice(own["donors"])
        notes.append(donors_note)
    elif cat == "cross":
        also = cross_batches_sentence(key)
        notes += [CROSS_ORDER, CROSS_INDEX]
    elif cat == "mosaic":
        also = mosaic_features_sentence(key)
    leftover = LEFTOVER
    if cat == "cross":
        leftover += " " + LEFTOVER_BATCH
    elif "batch_index=" in export:
        leftover += " " + LEFTOVER_INDEX
    notes += [FOLDER_NOTE, CHECK_NOTE, leftover, OVERWRITE_NOTE]
    recipes = and_list([RECIPES[r] for r in RECIPES_OF[cat]])
    tenx = (" The cell reads `.h5ad` files. A 10x `.h5` file is read another way: see the "
            "recipe for a 10x `.h5` file, linked at the top of this section."
            if "10x" in RECIPES_OF[cat] else "")
    # the read cell shows one object where it ends in a name, else one per print
    shown = len(re.findall(r"^print\(", own["read"], re.M)) or 1
    printed = ("your file" if one else "your files" if shown == own["n_files"] else
               f"{NUMBER[shown]} of your files. The other files need the same names")
    write = ("Write the dataset folder. Change each name in quotes marked `# <-`: give "
             "`NAME` a new value, and write the names that your "
             f"{'file uses' if one else 'files use'} for the other parts. The printout "
             f"above lists the names in {printed}. Write a name listed after `obs:` with "
             "that prefix, as `\"obs:celltype\"`"
             + (", and one after `obsm:` as `\"obsm:protein\"`" if '"obsm:' in export else "")
             + ".")
    if cat == "mosaic":
        write += " Each call writes one batch."
    elif "n_batches=" in export:
        write += (" The loop writes one RNA batch and one ATAC batch per call, and "
                  "`n_batches` says how many batches the data has.")
    elif cat == "cross":
        write += (" When the folder holds batches that your data does not have, "
                  "`export_dataset` stops and names the files. A new value of `NAME` "
                  "avoids that.")
    check = ("Check the folder before you run anything. `runnable` says whether a method "
             "can run on your data on this computer, and `reason` says why not. The first "
             "two lines above the table say what the folder holds and how many cells each "
             "file has. Compare them with your data.")
    if 'atac_kind="peak"' in export and "category=" not in export:
        # without a category, the export writes the peaks under both names
        check += (" `atac.h5` and `atac_peak.h5` are one matrix under two names. Diagonal "
                  "methods read the second.")
    check += (" A further line is about one method: why it cannot run, or what it does not "
              "read or needs from your data.")
    if cat == "cross":
        check += (" With fewer batches than the methods read, such a line names the files "
                  "that are not there. With more, it says which batches are not used.")
    return [
        ("markdown", f"## 5. Your own data\n\n{own['needs']}\n\n"
                     + (f"{also}\n\n" if also else "")
                     + f"Starting from something else? See the recipes for {recipes}.\n\n"
                     + details(*notes, label="Details: export")),
        ("markdown", f"The cells below read your data, write it as a dataset folder, check "
                     f"the folder and run {same} on it. So that the notebook runs as it "
                     f"is, the next cell writes {own['example']}. It leaves a file of your "
                     f"own with {'that name' if one else 'one of those names'} as it is."),
        ("code", f'mtb.data.write_example("{t["ds"]}")'),
        ("markdown", f"Read {'your file' if one else 'your files'}. For your own data, "
                     f"change {'the file name' if one else 'every file name'} marked `# <-`."
                     + ("" if one else " A name that you leave is read from the example.")
                     + f" On Colab, upload {'the file' if one else 'the files'} first: click "
                       f"the folder icon on the left, then the upload icon." + tenx),
        ("code", own["read"]),
        ("markdown", write),
        ("code", export),
        ("markdown", check),
        ("code", f'mtb.scan("mydata/" + NAME, "{cat}", methods=METHODS)'),
        ("markdown", f"Run {same} on the folder:"),
        ("code", f'mine = mtb.run_all("mydata/" + NAME, "{cat}", methods=METHODS,\n'
                 f'                   out_dir="out/" + NAME)\n'
                 f'mine.summary'),
        ("code", "mine.plot()"),
    ]


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


#: what run_all scores besides the scIB metrics on data in several batches
SCORED_TOO = " and the classification scores (OCA, ACA, Spec, Sens, F1)"


def no_classification_sentence(key, methods):
    """The sentence after ``SCORED_TOO`` for a tutorial that runs a method
    whose output is a graph: run_all scores no classification for it."""
    graphs = sorted({m for m, v in task_variants(key)
                     if m in methods and v.output.kind == "graph"})
    if not graphs:
        return ""
    one = len(graphs) == 1
    return (f" {and_list(graphs)} write{'s' if one else ''} a graph, so "
            f"{'it gets' if one else 'they get'} no classification scores.")

PLOT_TEXT = ("`res.plot()` draws the scores as a bubble table. Circle size shows the rank "
             "within a column, and bigger is better. The fill compares a value with the "
             "other rows in the same column.")


def plot_details(cat):
    """The collapsed paragraphs under the plot of a ``cat`` tutorial: data in
    several batches (every category but vertical) has the classification family."""
    pink = ", pink for classification" if cat != "vertical" else ""
    return (
        "The lightest fill is the lowest value in this figure, not zero.",
        "Metrics are grouped by family: blue for dimension reduction and clustering, "
        f"green for batch correction{pink}. Each family starts with an `Overall` bar. "
        "Its length and colour both show the family score.",
        "A column whose rows all hold the same value is drawn grey, and a warning "
        "names it. A figure of one method is grey everywhere.")


def build_tutorial(key):
    cells = []
    md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: cells.append(nbf.v4.new_code_cell(t))
    check_task(key)
    t = TASKS[key]
    cat, ds, methods = t["cat"], t["ds"], t["methods"]
    names = and_list(methods)
    everyone = task_methods(key)

    title = _title(key) + "\n\n" + t["blurb"].format(methods=names)
    if key == "diagonal_rna_atac":
        title += "\n\n" + diagonal_atac_sentence()
    elif cat == "diagonal":
        title += "\n\n" + diagonal_multi_sentence(key)
    title += (f"\n\nMethods run on Linux. On macOS or Windows, "
              f"[open this notebook in Colab]({COLAB}tutorial_{key}.ipynb).")
    md(title)
    _install(md, code)

    md(f"""## 2. Download the data and the environments

`mtb.data.fetch` downloads `{ds}` ({download_size([ds])}) once. Each method runs in its own environment. `mtb.env.install` downloads the environment{'s' if len(methods) > 1 else ''} for {names}, with no conda needed. {env_download_sentence(methods)}

""" + details(
        "`state` is `PACKED` for an environment downloaded now and `have` for one that "
        "was already there. With `dry_run=True`, `mtb.env.install` downloads "
        "nothing and returns the plan with its sizes.",
        "Environments go to `mtb.config.DEFAULT.envs_dir`. To use another disk, set it "
        "before this cell.",
        f"From a terminal, `multibench env install --methods {','.join(methods)}` "
        f"does the same."))
    code(f'''import pandas as pd
import multibench as mtb

mtb.data.fetch("{ds}")''')
    code(f'''METHODS = {json.dumps(methods)}
envs = mtb.env.install(METHODS)
envs[["env", "methods", "state"]]''')

    run_notes = [
        f"Each method writes an embedding: a table of numbers with one row per cell. "
        f"`run_all` scores it against the cell-type labels of `{ds}`.",
        "The clustering metrics `ARI`, `NMI`, `ASW`, `iASW`, `iF1` and `cLISI` measure "
        "how well the embedding separates the cell types. The batch metrics "
        "`ASW_batch`, `ARI_batch`, `NMI_batch`, `GC` and `iLISI` measure how well the "
        "batches mix. They appear "
        "only when the data has several batches.",
    ]
    if "scMoMaT" in methods:
        run_notes.append("scMoMaT writes a graph instead of an embedding. `run_all` scores "
                         "its UMAP, and its status reads `CHAIN_OK_GRAPH_METHOD`.")
    run_notes += [
        "`res.failures` says why a method failed. `params={\"Method\": {\"key\": value}}` "
        "sets a method's parameters, and `mtb.params_for` lists them. Many methods "
        "take none.",
        f"`mtb.load_batch(\"out/{ds}\")` reloads these results later without running "
        f"anything.",
    ]
    md(f"""## 3. Run the method{'s' if len(methods) > 1 else ''}

`run_all` runs each method on `{ds}` and scores its output with the scIB metrics{SCORED_TOO + "." + no_classification_sentence(key, methods) if cat != "vertical" else "."} `res.summary` has one row per method with its status, run time and scores. Higher is better for every metric.

""" + details(*run_notes))
    code(f'''res = mtb.run_all("{ds}", "{cat}", methods=METHODS,
                  out_dir="out/{ds}")
res.summary''')

    md("## 4. Plot\n\n" + PLOT_TEXT + "\n\n" + details(*plot_details(cat)))
    code("res.plot()")

    for kind, src in own_cells(key):
        (md if kind == "markdown" else code)(src)

    section = 6
    stored_ds = t.get("stored_ds")
    if stored_ds:
        n_stored = stored_method_count(cat, stored_ds)
        stored_notes = []
        if stored_ds != ds and stored_ds in ds:
            stored_notes.append(f"The stored scores are for the full `{stored_ds}`, the benchmark "
                                f"dataset `{ds}` is drawn from.")
        elif stored_ds != ds:
            stored_notes.append(f"There are no stored scores for `{ds}`. `{stored_ds}` is another "
                                f"benchmark dataset of the same task.")
        stored_notes.append("The stored scores are the package's own runs of the methods.")
        fewer, n_few, n_all = stored_on_fewer_batches(stored_ds)
        if fewer:
            if fewer != ["UINMF"] or (n_few, n_all) != (2, 3) or len(labels_for_ds(key)) != 3:
                raise SystemExit(f"{key}: stored scores of {fewer} read {n_few} of {n_all} "
                                 f"batches: reword the stored-scores note")
            stored_notes.append(
                f"The stored scores of UINMF are from its two-batch variant, which read "
                f"batches 1 and 2 of `{stored_ds}`. `run_all` on `{ds}` runs its three-batch "
                f"variant.")
        stored_notes.append(
            "Scores of your runs differ a little from the stored ones: most methods give "
            "a slightly different embedding on each run.")
        md(f"""## {section}. Stored scores

The package ships stored scores for {n_stored} methods on `{stored_ds}`. `load_results` reads them and `mtb.plot.bubble` draws them, without running anything.

""" + details(*stored_notes))
        code(f'''long = mtb.load_results("{cat}", dataset="{stored_ds}")
mtb.plot.bubble(long)''')
        section += 1

    more = not_wrapped_sentence(key)
    more = "\n\n" + more if more else ""
    colab_all = (f" [This notebook]({COLAB}tutorial_{key}_all.ipynb) does the same in Colab."
                 if has_all(key) else "")
    md(f"""## {section}. Every method of this task

{len(everyone)} method{'s have' if len(everyone) > 1 else ' has'} a variant for {t['cat']} {t['label']}. Each ran on `{ds}` in the package's own check, on a server with a GPU:

{runs_table(key)}

To run them all, set `METHODS` to these names in section 2 and run the notebook again. That downloads {env_size_text(methods=everyone)}.{colab_all}{more}""")

    md("""## Troubleshooting

`res.failures` lists each method that failed or was skipped. Its `error` column ends with the method's error output.

""" + details(
        """| symptom | fix |
|---|---|
| `env.install` refuses on macOS or Windows | methods run only on Linux: use Colab or a Linux machine |
| a method needs an NVIDIA GPU | choose a GPU runtime on Colab, or a machine with a GPU |
| a method fails on a small dataset | `mtb.run_all(..., dry_run=True)` names what the method needs from the data's size in its `caveat` column |
| a warning that values are not whole numbers | export raw counts, for example with `rna="layer:counts"` |
| `... matrix/data as cells x features` | the matrix is transposed: export it again with `mtb.io.export_dataset` |
| a method times out | raise `timeout=` in `run_all` |
| low `label_order_confidence` | several label files fit the cell count: check `label_order_candidates` in `res.records` |
| batch metrics use the wrong batches | `res.rescore(batch=my_vector)` scores again without running the methods |"""))

    md(f"""## Next steps

- `mtb.cite(METHODS)` returns the citations for the benchmark and the methods you ran.
- The other tutorials: {", ".join(f"[{TASKS[k]['cat']} {TASKS[k]['label']}]({SITE}tutorials/{k}/)" for k in TASKS if k != key)}.
- The guides: [run]({SITE}tutorials/run/), [evaluate]({SITE}tutorials/evaluate/), [plot]({SITE}tutorials/plot/) and [discover methods]({SITE}tutorials/discover/).
- The [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/) has the full benchmark's rankings, with no install needed.""")
    return cells


def has_all(key):
    """Whether the task has methods beyond the tutorial's defaults, and so a
    notebook that runs every method."""
    return set(task_methods(key)) != set(TASKS[key]["methods"])


#: what a run status means to a reader
RUN_WORDS = {"CHAIN_OK": "scored",
             "CHAIN_OK_GRAPH_METHOD": "scored on the UMAP it writes",
             "RUN_OK_NO_EMBEDDING": "ran, graph output not scored"}


def runs_table(key):
    """The run of every method of the task on its dataset, as a table: status
    and seconds, read from ``tools/all_method_runs.json`` (the package's check
    of the every-method notebooks on the benchmark host). No scores."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "all_method_runs.json")) as f:
        runs = json.load(f)[key]
    everyone = task_methods(key)
    if sorted(runs) != everyone:
        raise SystemExit(f"all_method_runs.json[{key}] lists {sorted(runs)}, the task has {everyone}")
    rows = ["| Method | Result | Seconds |", "|---|---|---|"]
    for m in everyone:
        status, sec = runs[m]
        rows.append(f"| {m} | {RUN_WORDS[status]} | {sec} |")
    return "\n".join(rows)


def graph_methods(key):
    """The methods of a task that return a neighbour graph and no embedding
    (scMoMaT writes a UMAP next to its graph, and that is scored)."""
    return sorted({m for m, v in task_variants(key)
                   if "embedding" not in [v.output.kind] + [o.kind for o in v.extra_outputs]})


def build_all_methods(key):
    """The run of every method of a task on its dataset: install, run, plot."""
    cells = []
    md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: cells.append(nbf.v4.new_code_cell(t))
    t = TASKS[key]
    cat, ds = t["cat"], t["ds"]
    everyone = task_methods(key)
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
envs = mtb.env.install(METHODS)
envs[["env", "methods", "state"]]''')
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
                  out_dir="out/{ds}_all")
res.summary''')
    md("## 4. Plot\n\n" + PLOT_TEXT)
    code("res.plot()")
    return cells


def build_colab_quickstart():
    cells = []
    md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: cells.append(nbf.v4.new_code_cell(t))
    md("""# scMultiBench API quickstart (Colab)

This notebook installs `multibench-sc`, looks up methods and draws the stored benchmark results. It runs no method, so it downloads nothing else. The integration tutorials install the method environments and run the methods.""")
    code(INSTALL_CELL)
    code("""import multibench as mtb

print(len(mtb.find_methods()), "methods in the package")
mtb.find_methods(category="vertical")""")
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
    code("""print(mtb.cite(["Matilda"]))   # format="bibtex" for BibTeX entries""")
    md("""## Draw the stored results

The package ships stored results, so these figures draw without running anything.

""" + details(
        "The stored results are the package's own runs of the methods, one dataset "
        "per category: D11, D28, D45 and D52.",
        "`mtb.plot.bar` draws each method's overall score on the dataset as one bar, "
        "best on top."))
    code("""long = mtb.load_results("vertical", dataset="D11")
mtb.plot.bubble(long)""")
    code("""mtb.plot.bar(mtb.load_results("diagonal", dataset="D28"))""")
    md("## Next steps\n\n- The integration tutorials install the environments, run methods "
       "and repeat the run on your own data: "
       + ", ".join(f"[{t['cat']} {t['label']}]({COLAB}tutorial_{k}.ipynb)" for k, t in TASKS.items())
       + ".\n- The [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/) has "
         "the full benchmark's rankings.")
    return cells


if __name__ == "__main__":
    # every notebook is built, and so checked, before one is written
    books = []
    for key in TASKS:
        books.append((f"tutorial_{key}", build_tutorial(key)))
        if has_all(key):
            books.append((f"tutorial_{key}_all", build_all_methods(key)))
    books.append(("colab_quickstart", build_colab_quickstart()))
    for name, cells in books:
        path = os.path.join(OUT, f"{name}.ipynb")
        nbf.write(_notebook(cells, name), path)
        print(f"wrote {path} {len(cells)} cells")
