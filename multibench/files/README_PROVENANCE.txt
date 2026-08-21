These catalog tables (method.csv, dataset.csv, metric_full.csv) are DERIVED from the scmbench registry (scmbench/run/methods.yaml) and the real dataset tree (data/dataset_final/) by api_verify/_gen_catalog.py.
method.csv: one row per registry method (Matilda stub excluded) -> 40 rows; columns map id->Methods (MOFA2 shown as MOFA+, Seurat_WNN as Seurat(WNN)), language->Programming Language, atac->Peak/Gene Activity, needs_labels->CellType Information Required, categories/tasks joined with ';'.
dataset.csv: dataset ids present under data/dataset_final/ (SD* = simulated). Column provenance, per column:
  - dataset            TREE-DERIVED (data/dataset_final/ listing). The id. (Header was 'dataset name' before 0.3; catalog.datasets() still exposes a 'dataset name' duplicate column for one release.)
  - simulated          TREE-DERIVED (catalog.datasets() computes it: id starts with 'SD'). Not stored in the CSV.
  - category           RESULTS-DERIVED AT CALL TIME (catalog.datasets() fills it from multibench.available_datasets(cat, source='both'), i.e. which category's published/re-run metric tables contain the dataset). Not stored in the CSV, so it cannot go stale.
  - has_results        RESULTS-DERIVED AT CALL TIME (same source). Not stored in the CSV.
  - assay, tissue, n_cells, n_batches, source
                       PAPER-DERIVED, NULLABLE. These must be transcribed from the scMultiBench paper's supplementary dataset table; nothing in this repository or in scMultiBench_ref holds them, so they ship EMPTY. Do not fill them from the data tree (n_cells of a processed file is not the paper's n_cells). When transcribed, record the supplementary-table version here.
metric_full.csv: canonical scIB metrics with short descriptions.
If official scMultiBench files are located, they can replace these verbatim.
