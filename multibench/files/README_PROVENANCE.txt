Catalog table read by multibench.catalog.metrics() (multibench/_data/catalog.py).

metric_full.csv - one row per metric of the clustering, batch and
  classification families, with a short description that states the range
  and that higher is better. The descriptions follow what
  multibench.evaluate computes: for the scIB metrics
  (multibench/eval/scib.py), the article's protocol: Leiden clusters
  searched to the number of cell types for ARI/NMI/ASW/iASW/iF1 and to the
  number of batches for ARI_batch/NMI_batch, every label counted as
  isolated for iASW/iF1; for OCA, ACA, Spec, Sens and F1 (multibench/eval/classify.py),
  the mean over the reference batches.
