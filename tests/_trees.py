"""A saved ``run`` output tree, as ``BatchResult.save`` lays it out, for the
load_batch tests."""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

METHODS = ["Matilda", "MOFA2", "scMoMaT"]
DATASET = "D11"


def _write_tree(d: Path, dataset=DATASET, methods=METHODS) -> None:
    """A run output tree exactly as BatchResult.save + the method runs lay it out."""
    d.mkdir(parents=True)
    recs, rows = [], []
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        mdir = d / f"{m}_{dataset}"                     # the per-method folder
        mdir.mkdir()
        with h5py.File(mdir / "embedding.h5", "w") as f:
            f.create_dataset("data", data=rng.normal(size=(4, 50)))
        metrics = {"ARI": 0.5 + i / 10, "NMI": 0.6 + i / 10}
        recs.append({"method": m, "category": "vertical", "dataset": dataset,
                     "modalities": ["rna", "adt"], "output_kind": "embedding",
                     "env": "x", "n_tunable": 1, "status": "CHAIN_OK",
                     "run_sec": 1.0, "emb_shape": [50, 4], "metrics": metrics,
                     "out_dir": f"/benchmark/host/out/{m}_{dataset}"})
        for k, v in metrics.items():
            rows.append({"metric": k, "value": v, "method": m, "dataset": dataset,
                         "category": "vertical", "clustering": "default",
                         "source": "user"})
    pd.DataFrame(rows).to_csv(d / "long.csv", index=False)
    pd.DataFrame([{"method": r["method"], "status": r["status"]} for r in recs]
                 ).to_csv(d / "summary.csv", index=False)
    with open(d / "batch_result.json", "w") as fh:
        json.dump({"dataset": dataset, "category": "vertical", "records": recs}, fh)
