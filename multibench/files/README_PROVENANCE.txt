Catalog tables read by multibench.catalog (multibench/data/catalog.py).

method.csv - one row per method of the registry (multibench/engine/methods.yaml).
  Columns: Methods = the registry id (MOFA2 is shown as MOFA+, Seurat_WNN as
  Seurat(WNN)), Programming Language = language, Deep Learning =
  deep_learning, Peak/Gene Activity = atac, Output = output, CellType
  Information Required = needs_labels, Integration Categories / Task
  Categories = categories / tasks joined with ';'.
  catalog.methods() overlays needs_labels, atac ('peak' | 'gene_activity' |
  None), categories and tasks from the registry at call time for every
  registered id, so those CSV columns are informational and may lag. The
  registry derives needs_labels from the variants' label roles and validates
  atac. language, deep_learning and output are read from the CSV and not
  overlaid; the registry has no deep-learning field.

dataset.csv - dataset ids from the benchmark's dataset tree (SD* = simulated).
  dataset      The id. catalog.datasets() also reads the older header
               'dataset name' and exposes a duplicate column of that name for
               one release.
  assay, tissue, n_cells, n_batches, source
               To be transcribed from the dataset table in the scMultiBench
               paper's supplement; nothing in this repository holds them, so
               they ship empty, and catalog.datasets() returns only the ones
               that hold a value. Do not fill them from the data tree (n_cells
               of a processed file is not the paper's n_cells). When
               transcribed, record the supplementary-table version here.
  Computed by catalog.datasets() at call time, not stored in the CSV:
  simulated    The id starts with 'SD'.
  category, has_results
               From multibench.available_datasets(category, source='both'):
               which category's published / re-run metric tables contain the
               dataset. Ids with stored results but no CSV row (D24, D11s,
               ...) are appended.

metric_full.csv - one row per scIB metric of the clustering and batch
  families, with a short description that states the range and that higher
  is better. The descriptions follow what multibench.evaluate computes
  (multibench/eval/scib.py): Leiden clusters from the optimal-resolution
  sweep for ARI/NMI/iF1, every cell type counted as an isolated label for
  iASW/iF1. PCR, which catalog.known_metrics() also lists, has no row.

An official scMultiBench file with the same columns can replace any of these.
