Catalog tables read by multibench.catalog (multibench/data/catalog.py).

method.csv - one row per method of the registry (multibench/engine/methods.yaml).
  Columns: Methods = the registry id (MOFA2 is shown as MOFA+, Seurat_WNN as
  Seurat(WNN)), Programming Language = language, Peak/Gene Activity = atac,
  CellType Information Required = needs_labels, Integration Categories /
  Task Categories = categories / tasks joined with ';'.
  catalog.methods() overlays needs_labels, atac ('peak' | 'gene_activity' |
  None), categories and tasks from the registry at call time for every
  registered id, so those CSV columns are informational and may lag. The
  registry derives needs_labels from the variants' label roles and validates
  atac. Deep Learning and Output exist only in the CSV.

dataset.csv - the dataset ids of the benchmark's dataset tree (SD* = simulated).
  dataset      The id. catalog.datasets() also reads the older header
               'dataset name' and exposes a duplicate column of that name for
               one release.
  assay, tissue, n_cells, n_batches, source
               To be transcribed from the dataset table in the scMultiBench
               paper's supplement; nothing in this repository holds them, so
               they ship empty. Do not fill them from the data tree (n_cells
               of a processed file is not the paper's n_cells). When
               transcribed, record the supplementary-table version here.
  Computed by catalog.datasets() at call time, not stored in the CSV:
  simulated    The id starts with 'SD'.
  category, has_results
               From multibench.available_datasets(category, source='both'):
               which category's published / re-run metric tables contain the
               dataset. Ids with stored results but no CSV row (D24, D11s,
               ...) are appended.

metric_full.csv - the canonical scIB metric codes with short descriptions.

An official scMultiBench file with the same columns can replace any of these.
