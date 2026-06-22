These catalog tables (method.csv, dataset.csv, metric_full.csv) are DERIVED from the scmbench registry (scmbench/run/methods.yaml) and the real dataset tree (data/dataset_final/) by api_verify/_gen_catalog.py.
method.csv: one row per registry method (Matilda stub excluded) -> 40 rows; columns map id->Methods (MOFA2 shown as MOFA+, Seurat_WNN as Seurat(WNN)), language->Programming Language, atac->Peak/Gene Activity, needs_labels->CellType Information Required, categories/tasks joined with ';'.
dataset.csv: dataset ids present under data/dataset_final/ (SD* = simulated).
metric_full.csv: canonical scIB metrics with short descriptions.
If official scMultiBench files are located, they can replace these verbatim.
