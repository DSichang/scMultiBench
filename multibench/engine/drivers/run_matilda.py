"""Package-side driver for Matilda's supported object API (`matilda-sc`).

Why a driver: the upstream pair `tools_scripts/Matilda/main_matilda_{train,task}.py`
only ran the TRAIN stage under our wiring, so Matilda never emitted an embedding
(it was the one method listed as "train-stage only"). Matilda's documented API
(https://pyanglab.github.io/Matilda/) is the object API — `matilda.train()` to fit
the shared model, then one verb per task over a combinable `matilda.task()`.

This driver trains once and then runs dimension-reduction + classification in a
SINGLE engine pass (the model loads once), writing:
  <save_path>/embedding.h5   dataset "data"  -> the integrated latent space (cells x z_dim)
  <save_path>/predict.csv                    -> per-cell predictions (when available)

Modality mode is auto-detected by Matilda from what we pass: rna+adt = CITE-seq,
rna+atac = SHARE-seq, rna+adt+atac = TEA-seq, rna alone = rna_only. No upstream
method script is modified.
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pandas as pd


def _load(path):
    """Read a Matilda-format .h5 into an AnnData (cells x features); None/NULL -> None."""
    if path is None or str(path).upper() in ("NULL", "NONE", ""):
        return None
    from matilda import io
    return io.read_matilda_h5(path)


def main() -> None:
    p = argparse.ArgumentParser("matilda-driver")
    # injected by the runner for driver variants; unused (we use the installed package)
    p.add_argument("--script_dir", default=None)
    p.add_argument("--rna", required=True)
    p.add_argument("--adt", default=None)
    p.add_argument("--atac", default=None)
    p.add_argument("--cty", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    import matilda

    rna, adt, atac = _load(a.rna), _load(a.adt), _load(a.atac)
    labels = pd.read_csv(a.cty).iloc[:, 0].astype(str).to_numpy()
    print(f"[matilda] rna={rna.shape}"
          f"{' adt=' + str(adt.shape) if adt is not None else ''}"
          f"{' atac=' + str(atac.shape) if atac is not None else ''}"
          f" labels={labels.shape} ({len(set(labels))} types)", flush=True)

    fit = matilda.train(rna, adt, atac, labels=labels,
                        seed=a.seed, epochs=a.epochs, device=a.device)
    print(f"[matilda] trained mode={getattr(fit, 'mode', '?')}", flush=True)

    res = matilda.task(rna, adt, atac, labels=labels, model=fit,
                       dim_reduce=True, classification=True, device=a.device)

    out = str(a.save_path)
    os.makedirs(out, exist_ok=True)

    latent = res.latent
    if hasattr(latent, "values"):          # DataFrame -> ndarray
        latent = latent.values
    latent = np.asarray(latent, dtype=np.float64)
    with h5py.File(os.path.join(out, "embedding.h5"), "w") as f:
        f.create_dataset("data", data=latent)
    print(f"[matilda] wrote embedding.h5 latent={latent.shape}", flush=True)

    preds = getattr(res, "predictions", None)
    if preds is not None:
        pd.DataFrame(preds).to_csv(os.path.join(out, "predict.csv"), index=False)
        print("[matilda] wrote predict.csv", flush=True)


if __name__ == "__main__":
    main()
