"""multibench driver for Matilda, using its object API (`matilda-sc`).

The upstream `tools_scripts/Matilda/main_matilda_train.py` runs only the train
stage, which writes no embedding. The driver trains once with `matilda.train()`
and runs `matilda.task()` on the trained model, writing:
  <save_path>/embedding.h5   dataset "data"  -> the latent space (cells x z_dim)
  <save_path>/predict.csv                    -> per-cell predictions (when available)

Matilda picks the mode from the modalities passed: rna+adt = CITE-seq,
rna+atac = SHARE-seq, rna+adt+atac = TEA-seq, rna alone = rna_only.
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
    # passed by the runner to every driver; unused (the installed package is imported)
    p.add_argument("--script_dir", default=None)
    p.add_argument("--rna", required=True)
    p.add_argument("--adt", default=None)
    p.add_argument("--atac", default=None)
    p.add_argument("--cty", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto")
    # the other knobs of main_matilda_train.py; unset leaves matilda's default
    shape_keys = ("batch_size", "z_dim", "hidden_rna", "hidden_adt", "hidden_atac")
    for k in shape_keys:
        p.add_argument(f"--{k}", type=int)
    p.add_argument("--lr", type=float)
    # on by default; the runner drops a Python False, so "False" turns it off
    p.add_argument("--augmentation", nargs="?", const="True", default="True")
    a = p.parse_args()
    shape = {k: getattr(a, k) for k in shape_keys if getattr(a, k) is not None}
    fit_kw = dict(shape, augmentation=a.augmentation != "False")
    if a.lr is not None:
        fit_kw["lr"] = a.lr

    import matilda

    rna, adt, atac = _load(a.rna), _load(a.adt), _load(a.atac)
    labels = pd.read_csv(a.cty).iloc[:, 0].astype(str).to_numpy()
    print(f"[matilda] rna={rna.shape}"
          f"{' adt=' + str(adt.shape) if adt is not None else ''}"
          f"{' atac=' + str(atac.shape) if atac is not None else ''}"
          f" labels={labels.shape} ({len(set(labels))} types)", flush=True)

    fit = matilda.train(rna, adt, atac, labels=labels,
                        seed=a.seed, epochs=a.epochs, device=a.device, **fit_kw)
    print(f"[matilda] trained mode={getattr(fit, 'mode', '?')}", flush=True)

    res = matilda.task(rna, adt, atac, labels=labels, model=fit,
                       dim_reduce=True, classification=True, seed=a.seed,
                       device=a.device, **shape)

    out = str(a.save_path)
    os.makedirs(out, exist_ok=True)

    latent = res.latent
    if hasattr(latent, "values"):          # a DataFrame
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
