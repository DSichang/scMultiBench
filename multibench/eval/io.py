"""Readers for evaluation inputs: embedding, labels, clustering."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_embedding(path: Path | str) -> np.ndarray:
    """Read embedding from h5 dataset 'data'; orient as (cells, dims).

    Note: orientation auto-detection assumes there are MORE cells than
    embedding dimensions. Square embeddings (cells == dims) or tall-thin
    embeddings cannot be auto-disambiguated and may be returned in the wrong
    orientation.
    """
    with h5py.File(path, "r") as f:
        X = np.asarray(f["data"])
    if X.shape[0] < X.shape[1]:
        X = X.T
    return X


def read_labels(path: Path | str) -> np.ndarray:
    """Read cell-type labels from a headerless csv; return integer codes.

    Matches the benchmark: drops the first row, takes column 0, factorizes.
    """
    raw = pd.read_csv(path, header=None, index_col=False)
    vals = raw.iloc[1:, 0]
    codes = pd.Categorical(vals).codes
    return np.asarray(codes).astype("int32")


def read_clustering(path: Path | str) -> np.ndarray:
    """Read precomputed clustering from h5 '/obs/cluster_leiden' (bytes -> int)."""
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["/obs/cluster_leiden"]).flatten()
    decoded = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x for x in raw]
    return np.asarray(decoded).astype(int)
