# scMultiBench

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/colab_quickstart.ipynb)
[![Docs](https://img.shields.io/badge/docs-dsichang.github.io%2FscMultiBench-blue)](https://dsichang.github.io/scMultiBench/)
[![PyPI](https://img.shields.io/pypi/v/multibench-sc)](https://pypi.org/project/multibench-sc/)

Multitask benchmarking of single-cell multimodal omics integration methods,
with `multibench`: a Python API that runs the benchmark's 40 integration
methods across four categories (vertical, diagonal, mosaic, cross), scores
them with scIB metrics, and draws scIB-style bubble tables.

**Documentation and tutorials:** <https://dsichang.github.io/scMultiBench/>

## Install

```bash
pip install multibench-sc          # import name: multibench
```

Python 3.9 or newer, on Linux or macOS. Running a method also needs that
method's environment, a prebuilt Linux archive that installs without conda
([details](https://dsichang.github.io/scMultiBench/installation/#method-environments)):

```bash
multibench env install --methods Matilda --packed --run
```

## Quick start

```python
import multibench as mtb, pandas as pd

mtb.find_methods(category="vertical", modalities=["rna", "adt"])   # methods that fit your data
mtb.data.fetch("D11")                                 # a reference CITE-seq dataset, 11 MB
mtb.scan("D11", "vertical")                           # which methods can run here, and why not
res = mtb.run_all("D11", "vertical", out_dir="out/")  # run and score every runnable method
res.plot()                                            # bubble table

# stored results and evaluation need no method environment
df = mtb.load_results("vertical", dataset="D11", source="rerun")
m = mtb.evaluate(my_embedding, labels=mtb.labels_for("D11"))
mtb.plot.bubble(pd.concat([df, mtb.to_long(m, method="MyMethod", dataset="D11", category="vertical")]))

# your own data: the folder layout, written from an AnnData, then checked
print(mtb.describe_layout("vertical"))
mtb.io.export_dataset(adata, "data/MYCITE", rna="X", adt="obsm:protein", labels="obs:celltype")
mtb.scan("MYCITE", "vertical", data_path="data")
```

The same from the command line (`multibench --help`):

```bash
multibench find --category vertical --modalities rna,adt
multibench scan D11 --category vertical
multibench run --method Matilda --category vertical --input rna=<data_path>/D11/rna.h5 --input adt=<data_path>/D11/adt.h5 --input cty=<data_path>/D11/cty.csv --out-dir out/Matilda --dry-run
multibench plot bubble --category vertical --dataset D11 --source rerun --out d11.pdf
```

To try it in the browser, open the
[Colab quickstart](https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/colab_quickstart.ipynb).
The published rankings are also browsable in the
[interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/).

Upgrading from 0.2.1? Some old names still work in 0.3 with a
`DeprecationWarning`, others were removed; the old -> new table is in the
[API reference](https://dsichang.github.io/scMultiBench/api/#deprecated-in-030).

## Citation

Liu C, Ding S, Kim HJ, Long S, Xiao D, Ghazanfar S, Yang P.
Multitask benchmarking of single-cell multimodal omics integration methods.
*Nature Methods* 22, 2449-2460 (2025). <https://doi.org/10.1038/s41592-025-02856-3>

Each method you run has its own paper; please cite it alongside the
benchmark. `print(mtb.cite("Matilda", "MOFA2"))` prints the benchmark's
reference and one line per method (`multibench cite Matilda MOFA2` for BibTeX).

The benchmark and the method scripts live in
[PYangLab/scMultiBench](https://github.com/PYangLab/scMultiBench).
