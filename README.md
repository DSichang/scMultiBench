# scMultiBench

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSichang/scMultiBench/blob/main/notebooks/colab_quickstart.ipynb)
[![Docs](https://img.shields.io/badge/docs-dsichang.github.io%2FscMultiBench-blue)](https://dsichang.github.io/scMultiBench/)
[![PyPI](https://img.shields.io/pypi/v/multibench-sc)](https://pypi.org/project/multibench-sc/)

Multitask benchmarking of single-cell multimodal omics integration methods,
with `multibench`: a Python API that runs 36 integration methods across
four categories (vertical, diagonal, mosaic, cross), scores them with scIB
metrics, and draws scIB-style bubble tables.

**Documentation and tutorials:** <https://dsichang.github.io/scMultiBench/>




## Citation

Liu C, Ding S, Kim HJ, Long S, Xiao D, Ghazanfar S, Yang P.
Multitask benchmarking of single-cell multimodal omics integration methods.
*Nature Methods* 22, 2449-2460 (2025). <https://doi.org/10.1038/s41592-025-02856-3>

Each method you run has its own paper; please cite it alongside the
benchmark. `print(mtb.cite("Matilda", "MOFA2"))` prints the benchmark's
reference and one line per method (`multibench cite Matilda MOFA2` for BibTeX).

The benchmark and the method scripts live in
[PYangLab/scMultiBench](https://github.com/PYangLab/scMultiBench).
