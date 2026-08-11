# scMultiBench

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/colab_quickstart.ipynb)
[![Docs](https://img.shields.io/badge/docs-dsichang.github.io%2FscMultiBench-blue)](https://dsichang.github.io/scMultiBench/)

A systematic benchmark of single-cell multimodal integration, with `multibench` -
a typed Python API that runs 40 integration methods across four scenarios
(vertical, diagonal, mosaic, cross), scores them with scIB metrics, and draws
the paper's figures.

**Documentation and tutorials:** <https://dsichang.github.io/scMultiBench/>

## Quick start

```bash
git clone https://github.com/DSichang/scMultiBench.git
cd scMultiBench
pip install -e .
```

```python
import multibench as mtb

mtb.list_methods()                 # the 40-method registry
mtb.method_info("Matilda")         # everything known about one method
mtb.scan("D11", "vertical")        # what can run on a dataset, and why not
res = mtb.run_all("D11", "vertical", out_dir="out/")   # run + score
res.plot()                         # the paper-style bubble panel
```

Running methods needs their conda environments (Linux):

```bash
multibench env doctor                              # what exists / is missing
multibench env install --category vertical --run   # build one category
```

The benchmark datasets are downloaded separately - see
[Get the data](https://dsichang.github.io/scMultiBench/installation/#get-the-data).

## Try it without installing anything

The [Colab quickstart](https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/colab_quickstart.ipynb)
installs the API, explores the registry, and reproduces the benchmark figures
from the shipped result tables - entirely in the browser. The full published
rankings are browsable in the
[interactive explorer](https://shiny.maths.usyd.edu.au/scMultiBench/).

## Citation

Liu, Ding et al. Benchmarking single-cell multimodal data integrations.
*Nature Methods* 22, 2449-2460 (2025).
Every method you run is third-party software with its own paper - please cite
it alongside the benchmark; `mtb.method_info(name)` points to each method's
upstream repository and reference.
