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
package ships them. There are no flags, fallbacks or helper functions. A step
that cannot work on the reader's computer fails with the package's own error:
the environment install refuses off Linux.

Every method of a task runs on the task's one dataset. The defaults are two
methods of the task (one where it has one) that are fast and need no GPU,
with the smallest environments. ``run_all`` gets the dataset, the category,
the methods and the output folder, and no other argument: where the dataset
also holds the files of a smaller variant of a method, ``run_all`` runs the
variant with the most modalities. ``check_task`` stops the generation when
that call would not run every method of the task once, on the task's variant
and on every batch. Every size quoted comes from the dry-run plans at
generation time, never a hand-written figure.

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


def stored_on_fewer_batches(dataset):
    """The methods whose stored scores for ``dataset`` come from a run on
    fewer batches than the others', from the ``label_order`` column of the
    stored summary (UINMF on ``D52``: its two-batch variant)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(OUT, "results", f"summary_{dataset}.csv"))
    n = df.label_order.str.count(r"\+") + 1
    return sorted(df.method[n < n.max()]), int(n.min()), int(n.max())


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
   variants=[["rna", "atac"], ["rna", "atac_gas"]], methods=["scMM", "VIMCCA"],
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
    import multibench as mtb
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
    of the task once, on the task's variant, or when a method reads only some
    batches of the dataset."""
    import multibench as mtb
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
    n = len(mtb.labels_for(t["ds"]))
    for m in everyone:
        if len(mtb.labels_for(t["ds"], t["cat"], m)) < n:
            raise SystemExit(f"{m} reads only some batches of {t['ds']}: reword the notebooks")
    # the vertical, mosaic and cross texts say that ATAC is a peak matrix
    gas = mtb.find_methods(t["cat"], atac="gene_activity")
    if t["cat"] != "diagonal" and gas:
        raise SystemExit(f"{and_list(gas)} read gene activity in {t['cat']}: reword the "
                         f"{t['cat']} blurbs and own-data sections")


def labels_for_ds(key, method="UINMF"):
    """The label files ``method`` reads on the task's dataset."""
    import multibench as mtb
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
          "folder, and `run_all` runs on that folder. Every method of this task reads "
          "ATAC as peaks, so the folder holds peaks."),
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
          "and `atac=\"X[feature_types=Peaks]\"`."],
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
 "diagonal_multi": dict(
   name="MYDIAG",
   intro=("Your data needs raw RNA counts and ATAC as one AnnData per batch, with a "
          "cell type for each cell. `mtb.io.export_dataset` with `batch_index=` writes "
          "one RNA batch and one ATAC batch per call. iNMF and online_iNMF read ATAC as "
          "gene-activity scores, and online_iNMF needs about 5,000 cells in total."),
   standin=("Here the first 900 RNA profiles and the first 900 ATAC profiles of each "
            "`D37mini` batch take the place of your data:"),
   data='''d = mtb.config.DEFAULT.data_path / "D37mini"
n = 900
rna, atac = [], []
for b in (1, 2, 3):
    r = mtb.io.read_canonical(d / f"rna{b}.h5")
    r.obs["celltype"] = pd.read_csv(d / f"rna_cty{b}.csv")["x"].values
    a = mtb.io.read_canonical(d / f"atac_gas{b}.h5")
    a.obs["celltype"] = pd.read_csv(d / f"atac_cty{b}.csv")["x"].values
    rna.append(r[:n].copy())
    atac.append(a[:n].copy())
rna, atac''',
   export='''for b in (1, 2, 3):
    mtb.io.export_dataset(rna[b - 1], "mydata/MYDIAG", atac=atac[b - 1],
                          atac_kind="gene_activity", labels="obs:celltype",
                          category="diagonal", batch_index=b, overwrite=True)''',
   notes=["The RNA batches and the ATAC batches are numbered separately. RNA batch 1 "
          "and ATAC batch 1 need not come from the same sample.",
          "For ATAC as a peak matrix, pass `atac_kind=\"peak\"`. {multi_peak} "
          "the peak matrix."],
 ),
 "mosaic_rna_adt": dict(
   name="MYMOSAIC",
   intro=("Your data needs raw counts, with one AnnData per batch and a cell type for "
          "each cell. `mtb.io.export_dataset` with `batch_index=` writes one batch per "
          "call. Number the batches as below: RNA, then RNA + ADT, then ADT."),
   standin="Here the first 1,500 cells of each `D38mini` batch take the place of your data:",
   data='''d = mtb.config.DEFAULT.data_path / "D38mini"
n = 1500
rna1 = mtb.io.read_canonical(d / "rna1.h5")[:n]
rna2 = mtb.io.read_canonical(d / "rna2.h5")[:n]
adt2 = mtb.io.read_canonical(d / "adt2.h5")[:n]
adt3 = mtb.io.read_canonical(d / "adt3.h5")[:n]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]''',
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
          "that accept each one. The command line writes one batch per call with "
          "`multibench convert ... --category mosaic --batch-index N`."],
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
 "mosaic_shared": dict(
   name="MYMOSAIC",
   intro=("Your data needs raw counts, with one AnnData per batch and a cell type for "
          "each cell. `mtb.io.export_dataset` with `batch_index=` writes one batch per "
          "call. Number the batches as below: RNA + ADT, then RNA + ATAC, then RNA. "
          "Give ATAC as peaks."),
   standin="Here the first 1,000 cells of each `D46mini` batch take the place of your data:",
   data='''d = mtb.config.DEFAULT.data_path / "D46mini"
n = 1000
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
 "mosaic_unshared": dict(
   name="MYMOSAIC",
   intro=("Your data needs raw counts, with one AnnData per batch and a cell type for "
          "each cell. `mtb.io.export_dataset` with `batch_index=` writes one batch per "
          "call. Number the batches as below: RNA + ADT, then RNA + ATAC, then ADT. "
          "Give ATAC as peaks, and the same proteins in both ADT batches."),
   standin="Here the first 1,000 cells of each `D49mini` batch take the place of your data:",
   data='''d = mtb.config.DEFAULT.data_path / "D49mini"
n = 1000
rna1 = mtb.io.read_canonical(d / "rna1.h5")[:n]
rna2 = mtb.io.read_canonical(d / "rna2.h5")[:n]
adt1 = mtb.io.read_canonical(d / "adt1.h5")[:n]
adt3 = mtb.io.read_canonical(d / "adt3.h5")[:n]
atac2 = mtb.io.read_canonical(d / "atac2.h5")[:n]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA + ADT
mtb.io.export_dataset(rna1, out, adt=adt1, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna2, out, atac=atac2, atac_kind="peak", labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ADT only
mtb.io.export_dataset(adt3, out, rna=None, adt="X", labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
   notes=["`mtb.describe_layout(\"mosaic\")` lists the batch patterns and the methods "
          "that accept each one. The command line writes one batch per call with "
          "`multibench convert ... --category mosaic --batch-index N`."],
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
 "cross_rna_atac": dict(
   name="MYCROSS",
   intro=("Your data needs raw RNA counts and an ATAC peak matrix of the same cells, "
          "with a cell type and a batch for each cell. `mtb.io.export_dataset` with "
          "`batch=` writes one set of files per batch, and `run_all` runs on that "
          "folder. Give the same peaks in every batch."),
   standin=("Here 60% of `D56mini`'s cells, as an RNA AnnData with a batch column and "
            "a peak AnnData, take the place of your data:"),
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D56mini"
rna, atac = [], []
for b in (1, 2, 3):
    r = mtb.io.read_canonical(d / f"rna{b}.h5")
    r.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    rna.append(r)
    atac.append(mtb.io.read_canonical(d / f"atac{b}.h5"))
rna = ad.concat(rna, label="batch", keys=["1", "2", "3"], index_unique="-")
atac = ad.concat(atac, keys=["1", "2", "3"], index_unique="-")
rna = sc.pp.subsample(rna, fraction=0.6, random_state=0, copy=True)
atac = atac[rna.obs_names].copy()
rna, atac''',
   export='''mtb.io.export_dataset(rna, "mydata/MYCROSS", atac=atac, atac_kind="peak",
                      labels="obs:celltype", batch="obs:batch", category="cross",
                      overwrite=True)''',
   notes=["For one AnnData per batch, call `export_dataset` once per batch with "
          "`batch_index=N` instead of `batch=`."],
 ),
 "cross_adt_atac": dict(
   name="MYCROSS",
   intro=("Your data needs raw ADT counts and an ATAC peak matrix of the same cells, "
          "with a cell type and a batch for each cell. `mtb.io.export_dataset` with "
          "`batch=` writes one set of files per batch, and `run_all` runs on that "
          "folder. `rna=None` says that the data has no RNA."),
   standin=("Here 60% of `D58mini`'s cells, as an ADT AnnData with a batch column and "
            "a peak AnnData, take the place of your data:"),
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D58mini"
adt, atac = [], []
for b in (1, 2):
    a = mtb.io.read_canonical(d / f"adt{b}.h5")
    a.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    adt.append(a)
    atac.append(mtb.io.read_canonical(d / f"atac{b}.h5"))
adt = ad.concat(adt, label="batch", keys=["1", "2"], index_unique="-")
atac = ad.concat(atac, keys=["1", "2"], index_unique="-")
adt = sc.pp.subsample(adt, fraction=0.6, random_state=0, copy=True)
atac = atac[adt.obs_names].copy()
adt, atac''',
   export='''mtb.io.export_dataset(adt, "mydata/MYCROSS", rna=None, adt="X", atac=atac,
                      atac_kind="peak", labels="obs:celltype", batch="obs:batch",
                      category="cross", overwrite=True)''',
   notes=["For one AnnData per batch, call `export_dataset` once per batch with "
          "`batch_index=N` instead of `batch=`."],
 ),
 "cross_rna_adt_atac": dict(
   name="MYCROSS",
   intro=("Your data needs raw RNA and ADT counts in one AnnData and an ATAC peak "
          "matrix of the same cells, with a cell type and a batch for each cell. "
          "`mtb.io.export_dataset` with `batch=` writes one set of files per batch, "
          "and `run_all` runs on that folder."),
   standin=("Here 60% of `D59mini`'s cells, as an AnnData with RNA, ADT and a batch "
            "column and a peak AnnData, take the place of your data:"),
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D59mini"
batches, atac = [], []
for b in (1, 2):
    a = mtb.io.read_canonical(d / f"rna{b}.h5")
    a.obsm["protein"] = mtb.io.read_canonical(d / f"adt{b}.h5").to_df()
    a.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    batches.append(a)
    atac.append(mtb.io.read_canonical(d / f"atac{b}.h5"))
adata = ad.concat(batches, label="batch", keys=["1", "2"], index_unique="-")
atac = ad.concat(atac, keys=["1", "2"], index_unique="-")
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
atac = atac[adata.obs_names].copy()''',
   export='''mtb.io.export_dataset(adata, "mydata/MYCROSS", rna="X", adt="obsm:protein",
                      atac=atac, atac_kind="peak", labels="obs:celltype",
                      batch="obs:batch", category="cross", overwrite=True)''',
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

    # ------------------------------------------------------------------ title
    title = _title(key) + "\n\n" + t["blurb"].format(methods=names)
    if key == "diagonal_rna_atac":
        title += "\n\n" + diagonal_atac_sentence()
    elif cat == "diagonal":
        title += "\n\n" + diagonal_multi_sentence(key)
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
                  out_dir="out/{ds}")
res.summary''')

    # ------------------------------------------------------------------- plot
    md("## 4. Plot\n\n" + PLOT_TEXT + "\n\n" + details(*PLOT_DETAILS))
    code("res.plot()")

    # --------------------------------------------------------------- own data
    peak_only = and_list(atac_forms("diagonal")[1])
    multi_peak = [m for m, v in task_variants(key) if "atac_peak1" in v.when["modalities"]]
    multi_peak = and_list(multi_peak) + (" reads" if len(multi_peak) == 1 else " read") \
        if multi_peak else ""
    notes = [n.replace("{peak_only}", peak_only).replace("{multi_peak}", multi_peak)
             for n in own["notes"]]
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
                   out_dir="out/{own['name']}")
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
        fewer, n_few, n_all = stored_on_fewer_batches(sds)
        if fewer:
            if fewer != ["UINMF"] or (n_few, n_all) != (2, 3) or len(labels_for_ds(key)) != 3:
                raise SystemExit(f"{key}: stored scores of {fewer} read {n_few} of {n_all} "
                                 f"batches: reword the stored-scores note")
            stored_notes.append(
                f"The stored scores of UINMF are from its two-batch variant, which read "
                f"batches 1 and 2 of `{sds}`. `run_all` on `{ds}` runs its three-batch "
                f"variant.")
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
    more = not_wrapped_sentence(key)
    more = "\n\n" + more if more else ""
    if has_all(key):
        md(f"""## {n_sec}. Every method of this task

{len(everyone)} method{'s have' if len(everyone) > 1 else ' has'} a variant for {t['cat']} {t['label']}: {and_list(everyone)}. Each runs on `{ds}`. To run them all, set `METHODS` to that list in section 2 and run the notebook again. That downloads {env_size_text(methods=everyone)}.{more}

[Every method on `{ds}`]({SITE}tutorials/{key}_all/) shows that run.""")
    else:
        md(f"""## {n_sec}. Every method of this task

{and_list(everyone)} {'are the only methods' if len(everyone) > 1 else 'is the only method'} with a variant for {t['cat']} {t['label']}, so this tutorial runs every method of the task.{more}""")

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
    return sorted({m for m, v in task_variants(key)
                   if "embedding" not in [v.output.kind] + [o.kind for o in v.extra_outputs]})


def build_all_methods(key):
    """The run of every method of a task on its dataset: install, run, plot."""
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
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
                  out_dir="out/{ds}_all")
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
