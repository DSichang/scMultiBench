Catalog tables read by multibench.catalog (multibench/data/catalog.py).

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
  Computed by catalog.datasets() at call time, not stored in the CSV (it
  returns the rows that are fetchable or have stored results):
  category, has_results
               From multibench.available_datasets(category, source='both'):
               which category's published / re-run metric tables contain the
               dataset. Ids with stored results but no CSV row (D24, D25)
               are appended.

metric_full.csv - one row per metric of the clustering, batch and
  classification families, with a short description that states the range
  and that higher is better. The descriptions follow what
  multibench.evaluate computes: for the scIB metrics
  (multibench/eval/scib.py), the article's protocol: Leiden clusters
  searched to the number of cell types for ARI/NMI/ASW/iASW/iF1 and to the
  number of batches for ARI_batch/NMI_batch, every label counted as
  isolated for iASW/iF1; for OCA, ACA, Spec, Sens and F1 (multibench/eval/classify.py),
  the mean over the reference batches.

An official scMultiBench file with the same columns can replace any of these.
