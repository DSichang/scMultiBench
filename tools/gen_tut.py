"""Generate the four integration tutorials (vertical / diagonal / mosaic / cross)
and the Colab quickstart.

Each tutorial is one straight path, run top to bottom on Linux or Colab:
install the package, download the data and the method environments, run two
methods with ``run_all``, plot, run the same calls on data in the reader's own
format, and draw the benchmark's stored scores. There are no flags, fallbacks
or helper functions. A step that cannot work on the reader's computer fails
with the package's own error: the environment install refuses off Linux.

Each tutorial runs two fast methods with small environments. Every size quoted
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


def check_methods(cat, s):
    """The tutorial's methods must read every batch of its dataset and share
    one ATAC form with the own-data demo; a registry change that breaks either
    stops the generator instead of shipping a tutorial that compares unequal
    runs or feeds a method the wrong file."""
    import multibench as mtb
    n = len(mtb.labels_for(s["ds"]))
    for m in s["methods"]:
        if len(mtb.labels_for(s["ds"], cat, m)) < n:
            raise SystemExit(f"{m} reads only some batches of {s['ds']}: pick another method")
        form = mtb.method_info(m)["atac"]
        if s.get("atac") and form not in (None, s["atac"]):
            raise SystemExit(f"{m} reads ATAC as {form}, the demo exports {s['atac']}")


# The one install cell every notebook shares; the module docstring says why
# numpy and pandas are pinned. IPython expands the {...} in a %pip line.
INSTALL_CELL = """import numpy, pandas
%pip install -q multibench-sc numpy=={numpy.__version__} pandas=={pandas.__version__}"""

COLAB_GPU_NOTE = ("On Colab, choose a GPU runtime before you run the notebook: Runtime -> "
                  "Change runtime type -> T4 GPU. On a CPU runtime the smaller CPU builds "
                  "of the environments are installed, and training methods run slower.")

SCEN = {
 "vertical": dict(
   ds="D11", methods=["Matilda", "sciPENN"], atac=None,
   blurb=("Vertical integration combines modalities measured in the same cells. "
          "Examples are RNA and surface protein from CITE-seq, or RNA and ATAC from "
          "10x Multiome. This tutorial runs {methods} on `D11`, a CITE-seq dataset of "
          "2,864 cells, and then on data in your own format."),
 ),
 "diagonal": dict(
   ds="D28", methods=["iNMF", "online_iNMF"], atac="gene_activity",
   blurb=("Diagonal integration combines RNA and ATAC measured in different cells, "
          "with no pairing between them. If your RNA and ATAC come from the same "
          "cells, as in 10x Multiome, use the vertical tutorial. This tutorial runs "
          "{methods} on `D28`, with 6,408 RNA cells and 4,606 ATAC cells, and then on "
          "data in your own format."),
 ),
 "mosaic": dict(
   ds="D46", methods=["StabMap", "scMoMaT"], atac="peak", stored_ds="D45",
   blurb=("Mosaic integration combines batches that share only some modalities. "
          "For example, a paired RNA + ATAC batch can link an RNA-only batch and an "
          "ATAC-only batch. Each method accepts one batch pattern, and every mosaic "
          "method reads ATAC as peaks. This tutorial runs {methods} on `D46`, with "
          "21,416 cells in three batches: RNA + ADT, RNA + ATAC, and RNA only."),
 ),
 "cross": dict(
   ds="D52", methods=["StabMap", "sciPENN"], atac=None,
   blurb=("Cross integration combines batches that all measure the same "
          "modalities. The task is to remove batch effects and keep the biological "
          "structure. Every cross method here reads RNA and ADT. For several 10x "
          "Multiome samples, use the vertical tutorial. This tutorial runs {methods} "
          "on `D52`, with 23,478 cells in three batches."),
 ),
}

# ---------------------------------------------------------------- own data
# Section 5 per category: the shape the reader's data usually has, built from
# a part of the tutorial's dataset (the "your data" cell), then the export
# call and the same run_all on the new folder. overwrite=True lets the
# notebook run twice; without it export_dataset refuses to replace a file.
OWN_INTRO = {
 "vertical": ("Your data needs raw counts for every modality and a cell type for each "
              "cell. `mtb.io.export_dataset` writes an AnnData as a dataset folder, and "
              "`run_all` runs on that folder. For RNA + ATAC, each method reads either "
              "peaks or gene activity: `mtb.method_info(m)[\"atac\"]` says which."),
 "diagonal": ("Your data needs raw RNA counts and ATAC as two AnnData objects, with a "
              "cell type for each cell. `mtb.io.export_dataset` writes them as a dataset "
              "folder, and `run_all` runs on that folder. iNMF and online_iNMF read ATAC "
              "as gene-activity scores."),
 "mosaic": ("Your data needs raw counts, with one AnnData per batch and a cell type for "
            "each cell. `mtb.io.export_dataset` with `batch_index=` writes one batch per "
            "call. Number the batches to match a pattern that "
            "`mtb.describe_layout(\"mosaic\")` lists, and give ATAC as peaks."),
 "cross": ("Your data needs raw RNA and ADT counts in one AnnData, with a cell type and "
           "a batch for each cell. `mtb.io.export_dataset` with `batch=` writes one set "
           "of files per batch, and `run_all` runs on that folder."),
}
OWN_STANDIN = {
 "vertical": "Here an AnnData made from 60% of `D11`'s cells takes the place of your data:",
 "diagonal": "Here 60% of `D28`'s RNA cells and 60% of its ATAC cells take the place of your data:",
 "mosaic": "Here the first 1,500 cells of each `D46` batch take the place of your data:",
 "cross": "Here one AnnData with 30% of `D52`'s cells and a batch column takes the place of your data:",
}
OWN_DATA = {
 "vertical": """import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D11"
adata = mtb.io.read_canonical(d / "rna.h5")
adata.obsm["protein"] = mtb.io.read_canonical(d / "adt.h5").to_df()
adata.obs["celltype"] = pd.read_csv(d / "cty.csv")["x"].values
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
adata""",
 "diagonal": """import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D28"
rna = mtb.io.read_canonical(d / "rna.h5")
rna.obs["celltype"] = pd.read_csv(d / "rna_cty.csv")["x"].values
atac = mtb.io.read_canonical(d / "atac_gas.h5")
atac.obs["celltype"] = pd.read_csv(d / "atac_cty.csv")["x"].values
rna = sc.pp.subsample(rna, fraction=0.6, random_state=0, copy=True)
atac = sc.pp.subsample(atac, fraction=0.6, random_state=0, copy=True)
rna, atac""",
 "mosaic": """d = mtb.config.DEFAULT.data_path / "D46"
n = 1500
rna = [mtb.io.read_canonical(d / f"rna{b}.h5")[:n] for b in (1, 2, 3)]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]
adt1 = mtb.io.read_canonical(d / "adt1.h5")[:n]
atac2 = mtb.io.read_canonical(d / "atac2.h5")[:n]""",
 "cross": """import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D52"
batches = []
for b in (1, 2, 3):
    a = mtb.io.read_canonical(d / f"rna{b}.h5")
    a.obsm["protein"] = mtb.io.read_canonical(d / f"adt{b}.h5").to_df()
    a.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    batches.append(a)
adata = ad.concat(batches, label="batch", keys=["1", "2", "3"], index_unique="-")
adata = sc.pp.subsample(adata, fraction=0.3, random_state=0, copy=True)
adata""",
}
OWN_EXPORT = {
 "vertical": """mtb.io.export_dataset(adata, "mydata/MYCITE", rna="X", adt="obsm:protein",
                      labels="obs:celltype", overwrite=True)""",
 "diagonal": """mtb.io.export_dataset(rna, "mydata/MYDIAG", atac=atac, atac_kind="gene_activity",
                      labels="obs:celltype", category="diagonal", overwrite=True)""",
 "mosaic": """out = "mydata/MYMOSAIC"
# batch 1: RNA + ADT
mtb.io.export_dataset(rna[0], out, adt=adt1, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna[1], out, atac=atac2, atac_kind="peak", labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: RNA only
mtb.io.export_dataset(rna[2], out, labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)""",
 "cross": """mtb.io.export_dataset(adata, "mydata/MYCROSS", rna="X", adt="obsm:protein",
                      labels="obs:celltype", batch="obs:batch", category="cross",
                      overwrite=True)""",
}
OWN_NAME = {"vertical": "MYCITE", "diagonal": "MYDIAG", "mosaic": "MYMOSAIC", "cross": "MYCROSS"}
OVERWRITE_NOTE = ("`overwrite=True` replaces the files of an earlier run of this cell. "
                  "Without it, `export_dataset` raises `FileExistsError` rather than "
                  "replace a file.")
OWN_DETAIL = {
 "vertical": [
     "A 10x Multiome MuData goes in with one call. Here the labels are in "
     "`mdata.obs`. For labels in `mdata[\"rna\"].obs`, write `labels=\"rna:celltype\"`.\n\n"
     "```python\n"
     "mtb.io.export_dataset(mdata, \"mydata/MYMULTIOME\", rna=\"rna\", atac=\"atac\",\n"
     "                      atac_kind=\"peak\", labels=\"obs:celltype\",\n"
     "                      category=\"vertical\")\n"
     "```",
     "A cellranger-arc AnnData read with `gex_only=False` holds genes and peaks in "
     "one `X`. A feature filter splits them: `rna=\"X[feature_types=Gene Expression]\"` "
     "and `atac=\"X[feature_types=Peaks]\"`.",
     "Keep several samples in one folder, without `batch=`. To score the batch "
     "mixing, pass the batch column to `run_all(batch=...)`.",
 ],
 "diagonal": [
     "The command line writes the same folder from two .h5ad files:\n\n"
     "```\n"
     "multibench convert rna.h5ad mydata/MYDIAG --rna X --atac-from atac.h5ad \\\n"
     "    --atac-kind gene_activity --labels obs:celltype --category diagonal\n"
     "```",
     # {peak_only}: filled in from find_methods by build_tutorial
     "For ATAC as a peak matrix, pass `atac_kind=\"peak\"`. {peak_only} read "
     "peaks. `mtb.describe_layout(\"diagonal\")` lists the files each method needs.",
 ],
 "mosaic": [
     "`mtb.describe_layout(\"mosaic\")` lists the batch patterns and the methods "
     "that accept each one. The command line writes one batch per call with "
     "`multibench convert ... --category mosaic --batch-index N`.",
 ],
 "cross": [
     "For one AnnData per batch, call `export_dataset` once per batch with "
     "`batch_index=N` instead of `batch=`.",
 ],
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


def build_tutorial(cat, s):
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    check_methods(cat, s)
    ds, methods = s["ds"], s["methods"]
    names = and_list(methods)
    own = OWN_NAME[cat]

    # ------------------------------------------------------------------ title
    title = f"# {cat.capitalize()} integration\n\n{s['blurb'].format(methods=names)}"
    if cat == "diagonal":
        title += "\n\n" + diagonal_atac_sentence()
    title += (f"\n\nMethods run on Linux. On macOS or Windows, "
              f"[open this notebook in Colab]({COLAB}tutorial_{cat}.ipynb).")
    md(title)

    # ---------------------------------------------------------------- install
    md("## 1. Install\n\n"
       "This cell installs `multibench-sc`. It keeps the numpy and pandas that are "
       "already installed, so Colab needs no restart.\n\n" + details(COLAB_GPU_NOTE))
    code(INSTALL_CELL)

    # ------------------------------------------------------ data and envs
    md(f"""## 2. Download the data and the environments

`mtb.data.fetch` downloads `{ds}` ({download_size([ds])}) once. Each method runs in its own environment. `mtb.env.install` downloads the environments for {names}, with no conda needed. {env_download_sentence(methods)}

""" + details(
        "`state` is `PACKED` for an environment downloaded now and `have` for one that "
        "was already there. Without `dry_run=False`, `mtb.env.install` downloads "
        "nothing and returns the plan with its sizes.",
        "Environments go to `mtb.config.DEFAULT.envs_dir`. To use another disk, set it "
        "before this cell.",
        f"From a terminal, `multibench env install --methods {','.join(methods)} --packed "
        f"--run` does the same. With `--category {cat}` instead of `--methods`, it "
        f"installs every {cat} environment: {env_size_text(category=cat)}."))
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
    md(f"""## 3. Run the methods

`run_all` runs each method on `{ds}` and scores its output with the scIB metrics. `res.summary` has one row per method with its status, run time and scores. Higher is better for every metric.

""" + details(*run_notes))
    code(f'''res = mtb.run_all("{ds}", "{cat}", methods=METHODS, out_dir="out/{ds}")
res.summary''')

    # ------------------------------------------------------------------- plot
    md("""## 4. Plot

`res.plot()` draws the scores as a bubble table. Circle size shows the rank within a column, and bigger is better. The fill compares a value with the other rows in the same column.

""" + details(
        "The lightest fill is the lowest value in this figure, not zero.",
        "Metrics are grouped by family: blue for dimension reduction and clustering, "
        "green for batch correction. Each family starts with an `Overall` bar. Its "
        "length and colour both show the family score.",
        "A column whose rows all hold the same value is drawn grey, and the note under "
        "the figure names it."))
    code("res.plot()")

    # --------------------------------------------------------------- own data
    notes = [n.format(peak_only=and_list(atac_forms("diagonal")[1])) if "{peak_only}" in n
             else n for n in OWN_DETAIL[cat]]
    notes += [
        f"The folder name, `{own}`, is the dataset name for `run_all`, and "
        f"`data_path` is the folder that holds it.",
        f"`mtb.scan(\"{own}\", \"{cat}\", data_path=\"mydata\")` checks the folder and "
        f"the environments without running anything. Its `reason` column says what is "
        f"missing.",
        OVERWRITE_NOTE,
    ]
    md(f"""## 5. Your own data

{OWN_INTRO[cat]}

""" + details(*notes, label="Details: export"))
    md(OWN_STANDIN[cat])
    code(OWN_DATA[cat])
    md("Write the folder, then run the same methods on it:")
    code(OWN_EXPORT[cat] + f'''

mine = mtb.run_all("{own}", "{cat}", methods=METHODS, data_path="mydata",
                   out_dir="out/{own}")
mine.summary''')
    code("mine.plot()")

    # ------------------------------------------------------------ stored scores
    sds = s.get("stored_ds", ds)
    n_pub, n_rerun = stored_method_counts(cat, sds)
    where = f"`{sds}`"
    stored_notes = []
    if sds != ds:
        stored_notes.append(f"There are no stored scores for `{ds}`. `{sds}` is a larger "
                            f"mosaic dataset with another batch pattern.")
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
    md(f"""## 6. Stored scores

The package ships stored scores for {n_rerun} methods on {where}. `load_results` reads them and `mtb.plot.bubble` draws them, without running anything.

""" + details(*stored_notes))
    code(f'''long = mtb.load_results("{cat}", dataset="{sds}", source="rerun")
mtb.plot.bubble(long)''')

    # -------------------------------------------------------- troubleshooting
    md("""## Troubleshooting

`res.failures` lists each method that failed or was skipped. Its `error` column ends with the method's error output.

""" + details(
        """| symptom | fix |
|---|---|
| `env.install` refuses on macOS or Windows | methods run only on Linux: use Colab or a Linux machine |
| a method needs an NVIDIA GPU | choose a GPU runtime on Colab, or a machine with a GPU |
| a warning that values are not whole numbers | export raw counts, for example with `rna="layer:counts"` |
| `... matrix/data as cells x features` | the matrix is transposed: export it again with `mtb.io.export_dataset` |
| a method times out | raise `timeout=` in `run_all` |
| low `label_order_confidence` | several label files fit the cell count: check `label_order_candidates` in `res.results` |
| batch metrics use the wrong batches | `res.rescore(batch=my_vector)` scores again without running the methods |"""))

    siblings = [c for c in SCEN if c != cat]
    md(f"""## Next steps

- `mtb.cite(METHODS)` returns the citations for the benchmark and the methods you ran.
- The other tutorials: {", ".join(f"[{c}]({SITE}tutorials/{c}/)" for c in siblings)}.
- The guides: [run]({SITE}tutorials/run/), [evaluate]({SITE}tutorials/evaluate/), [plot]({SITE}tutorials/plot/) and [discover methods]({SITE}tutorials/discover/).
- The [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/) has the full benchmark's rankings, with no install needed.""")
    return C


def build_colab_quickstart():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))
    md("""# scMultiBench API quickstart (Colab)

This notebook installs `multibench-sc`, looks up methods and draws the stored benchmark results. It runs no method, so it downloads nothing else. The category tutorials install the method environments and run the methods.""")
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
    md(f"""## Next steps

- The four integration tutorials ([vertical]({COLAB}tutorial_vertical.ipynb), [diagonal]({COLAB}tutorial_diagonal.ipynb), [mosaic]({COLAB}tutorial_mosaic.ipynb), [cross]({COLAB}tutorial_cross.ipynb)) install the environments, run methods and repeat the run on your own data.
- The [interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/) has the full benchmark's rankings.""")
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
