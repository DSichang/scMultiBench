# scMultiBench

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/colab_quickstart.ipynb)
[![Docs](https://img.shields.io/badge/docs-dsichang.github.io%2FscMultiBench-blue)](https://dsichang.github.io/scMultiBench/)
[![PyPI](https://img.shields.io/pypi/v/multibench-sc)](https://pypi.org/project/multibench-sc/)

Multitask benchmarking of single-cell multimodal omics integration methods,
with `multibench` - a typed Python API that runs the benchmark's 40 integration
methods across four scenarios (vertical, diagonal, mosaic, cross), scores them
with scIB metrics, and draws the paper's figures.

**Documentation and tutorials:** <https://dsichang.github.io/scMultiBench/>

## Quick start

```bash
pip install multibench-sc          # the API (import name: multibench) - a 1.3 MB wheel that ships the method
                                   # registry, the stored result tables, the env lockfiles and the references
```

That is the whole install; the tutorials and the stored benchmark tables need
no clone. Clone only to work on the package itself (edit the notebooks, run
the test suite): `git clone https://github.com/DSichang/scMultiBench.git &&
cd scMultiBench && pip install -e .`

```python
import multibench as mtb, pandas as pd

mtb.list_methods()                 # the 40-method registry
mtb.method_info("Matilda")         # everything known about one method (env, labels, reference, variants)
mtb.recommend("vertical", modalities=["rna", "adt"])   # ranked from the stored tables, with coverage
mtb.data.fetch("D11")              # reference CITE-seq dataset, 11 MB -> mtb.config.DEFAULT.data_path
mtb.scan("D11", "vertical")        # preflight per method variant: files_ok / env_ok with reasons, plus the exact
                                   # `command` run() would execute (= run_all(dry_run=True)); blocked rows stay
mtb.inputs_for("D11", "vertical", "Matilda")   # (dataset, category, method) -> {'rna': ..., 'adt': ..., 'cty': ...}
res = mtb.run_all("D11", "vertical", out_dir="out/")   # run + score
res.plot()                         # the paper-style bubble panel

# your own data: describe_layout says how the folder is laid out, one call writes it, then the same three calls
print(mtb.describe_layout("vertical"))   # role -> filename per category (`multibench layout vertical` on the command line)
mtb.io.export_dataset(adata, "data/MYCITE", rna="X", adt="obsm:protein", labels="obs:celltype")
#   selectors as above, or objects: adt=<DataFrame / AnnData / array> (adt_names=[...]), labels=<Series>
mtb.scan("MYCITE", "vertical", data_path="data")

# stored results, evaluation and figures need no conda environment
df = mtb.load_results("vertical", dataset="D11", source="rerun")        # or source="published"; metrics=["ARI", "NMI"] for a subset
m = mtb.evaluate(my_embedding, labels=mtb.labels_for("D11"))            # every applicable scIB metric; metrics="clustering" | "batch" | "all" | ["ARI", "NMI"]
mtb.plot.bubble(pd.concat([df, mtb.to_long(m, method="MyMethod", dataset="D11", category="vertical")]), save="d11.pdf")
```

Upgrading from 0.2.1? `scan` absorbed the run plan (its `command` column),
`method_info(m)["runtime"]` the runtime table, `inputs_for` / `labels_for`
take `(dataset, category, method)`, and `evaluate` / `load_results` /
`recommend` select metrics through the one `metrics=` knob - the old
spellings warn for one release; every old -> new pair is in the
[API reference](https://dsichang.github.io/scMultiBench/api/#deprecated-in-030).

Running methods needs their environments - prebuilt linux-64 conda-pack
archives that `env install --packed --run` unpacks into the envs dir
(`$MULTIBENCH_ENVS_DIR`, else conda's envs dir when conda is present, else
`~/.cache/multibench/envs`) and that `run`
activates directly, so **no conda binary is required**, on Colab included
(`env install --run` refuses on macOS / Windows unless `--force`); everything
else - the registry, the stored results, `scan`'s file gate, `evaluate`, the
figures - works on any machine. Install only the environments you need:

```bash
multibench env doctor                                      # what exists / is missing, with the install line
multibench env plan --category vertical                    # the envs a category needs, with download / disk sizes
multibench env install --methods Matilda --packed --run    # one method's env (prebuilt archive, lockfile fallback)
multibench env install --category vertical --packed --run  # one category
```

Everything is also available from the command line (`multibench --help`):

```bash
multibench layout vertical                             # how to lay out MY data
multibench convert my.h5ad data/MYCITE --rna X --adt obsm:protein --labels obs:celltype --category vertical
multibench scan D11 --category vertical               # preflight table: files_ok / env_ok / reason (--columns all: every column)
multibench find --category vertical --modalities rna,adt --needs-labels false
multibench params Matilda                              # the hyperparameters --param accepts, per variant
multibench run --method Matilda --category vertical --input rna=<data_path>/D11/rna.h5 --input adt=<data_path>/D11/adt.h5 --input cty=<data_path>/D11/cty.csv --out-dir out/Matilda --dry-run   # one method: prints the exact command line that would run; drop --dry-run to execute
multibench run-all D11 --category vertical --out-dir out/ --dry-run   # the scan table + the command per variant; nothing runs
multibench evaluate --output out/Matilda/embedding.h5 --labels <data_path>/D11/cty.csv --metrics ARI,NMI   # mtb.labels_for("D11") returns {'cty': <that path>}
multibench plot bubble --category vertical --dataset D11 --source rerun --out d11.pdf
multibench plot bubble --input mine.csv --category vertical --dataset D11 --source rerun --out d11.pdf   # your rows next to the stored table
multibench cite Matilda MOFA2                          # BibTeX for the benchmark + each method
```

`mtb.data.fetch` puts the reference datasets under `mtb.config.DEFAULT.data_path`
(`~/.cache/multibench/data` after a pip install, `<repo>/data` in a checkout) - see
[Get the data](https://dsichang.github.io/scMultiBench/installation/#get-the-data).

## Try it without installing anything

The [Colab quickstart](https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/colab_quickstart.ipynb)
installs the API (pinning `numpy` / `pandas` to what Colab already has, so
nothing is upgraded), explores the registry, and reproduces the benchmark
figures from the shipped result tables - entirely in the browser, in about a
minute; its last cell says where the time went. Each category tutorial opens
in Colab too: with its `INSTALL_ENVS` flag left `False` nothing is downloaded
and the run cells stand in the benchmark host's own `run_all` outputs (real
embeddings, so `evaluate` and the figures are real); set it `True` to
download the prebuilt environments (3-6 GB per tutorial, measured; no conda,
no kernel restart) and run the methods on the Colab runtime. The full
published rankings are browsable in the
[interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/).

## Citation

Liu C, Ding S, Kim HJ, Long S, Xiao D, Ghazanfar S, Yang P.
Multitask benchmarking of single-cell multimodal omics integration methods.
*Nature Methods* 22, 2449-2460 (2025). <https://doi.org/10.1038/s41592-025-02856-3>

Every method you run is third-party software with its own paper - please cite
it alongside the benchmark. `print(mtb.cite("Matilda", "MOFA2"))` - or
`print(mtb.cite(res.summary.method))` after a sweep - prints the benchmark's
reference followed by one line per method (`fmt="bibtex"` for the `.bib`
entries; `multibench cite Matilda MOFA2` prints BibTeX by default);
`mtb.method_info(name)` carries the same reference, repository and version.

This repository (`DSichang/scMultiBench`) is the API fork of
[PYangLab/scMultiBench](https://github.com/PYangLab/scMultiBench), which holds
the benchmark and the method scripts the package runs unmodified.
