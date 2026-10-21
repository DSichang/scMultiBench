"""Round-5 student study, package work 'cfg': R5-04, R5-05 and R5-09.

- R5-04: while the method scripts are not fetched, the ``scan --strict``
  table shows the rows the scripts block as not runnable, with a reason, so
  it agrees with the error under it. ``mtb.scan`` and ``scan`` without
  ``--strict`` keep them runnable.
- R5-05: a failed download in ``mtb.data.fetch`` / ``fetch_outputs`` raises
  ``OSError`` naming the dataset, the URL and the manual route, and leaves
  nothing under ``data_path``. The progress line names the URL.
- R5-09: an ATAC matrix declared as peaks whose names are not
  ``chr:start-end`` gets the rename advice; gene activity is suggested only
  when the names support it.
"""
import importlib
import io
import re
import urllib.error
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry

F = importlib.import_module("multibench.data.fetch")
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


def test_raises_tables_list_the_download_error():
    import inspect
    for fn in (mtb.data.fetch,):
        raises = inspect.getdoc(fn).split("Raises\n------\n")[1].split("\n\n")[0]
        assert "OSError\n    The download failed; the message names the URL." in raises, \
            fn.__name__


# ================================================= R5-09: peak-name warnings
ad = pytest.importorskip("anndata")


def _adata(names, n=4):
    a = ad.AnnData(np.ones((n, len(names))))
    a.var_names = list(names)
    a.obs_names = [f"c{i}" for i in range(n)]
    return a


def _messages(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [str(x.message) for x in w if "ATAC feature names" in str(x.message)]


GENES = ["GAPDH", "ACTB", "CD3E", "MS4A1"]
PEAK_IDS = ["peak_0", "peak_1", "peak_2", "peak_3"]
COLON3 = ["chr4:171325400:171325903", "chr1:100:200", "chr2:5:9", "chrX:1:2"]


@pytest.mark.parametrize("names", [PEAK_IDS, COLON3], ids=["peak_0", "chr:start:end"])
def test_export_with_rna_gives_the_rename_advice(tmp_path, names):
    data = _adata(GENES)
    atac = _adata(names)
    msgs = _messages(lambda: ingest.export_dataset(
        data, tmp_path / "X", rna="X", atac=atac, atac_kind="peak"))
    assert msgs == ["Only 0% of the ATAC feature names look like peaks such as "
                    "chr1:100-200. If they are peaks, rename them to chr:start-end. "
                    "Some methods read the peak positions from the names, and "
                    "mtb.scan names these methods."]
    assert "did you mean" not in msgs[0] and "gene_activity" not in msgs[0]


def test_export_with_rna_gene_names_suggests_gene_activity(tmp_path):
    data = _adata(GENES)
    atac = _adata(GENES[:3] + ["peak_9"])
    msgs = _messages(lambda: ingest.export_dataset(
        data, tmp_path / "X", rna="X", atac=atac, atac_kind="peak"))
    assert msgs == ["Only 0% of the ATAC feature names look like peaks such as "
                    "chr1:100-200, and 75% are RNA gene names. If the matrix holds "
                    "gene activity, pass atac_kind='gene_activity'."]


def test_diagonal_export_compares_with_the_rna_side(tmp_path):
    rna = _adata(GENES)
    atac = _adata(GENES)
    atac.obs_names = [f"a{i}" for i in range(4)]
    msgs = _messages(lambda: ingest.export_dataset(
        rna, tmp_path / "X", rna="X", atac=atac, atac_kind="peak", category="diagonal"))
    assert len(msgs) == 1 and "100% are RNA gene names" in msgs[0]


def test_without_rna_names_both_options_rename_first(tmp_path):
    atac = _adata(PEAK_IDS)
    one = _messages(lambda: ingest._to_canonical(atac, tmp_path / 'a.h5', modality='atac_peak'))
    assert one == ["Only 0% of the ATAC feature names look like peaks such as "
                   "chr1:100-200. If they are peaks, rename them to chr:start-end. "
                   "Some methods read the peak positions from the names, and "
                   "mtb.scan names these methods. If the matrix holds gene activity, "
                   "pass modality='atac_gas'."]
    alone = _messages(lambda: ingest.export_dataset(atac, tmp_path / "Y", rna=None,
                                                    atac="X", atac_kind="peak"))
    assert alone[0].endswith("If the matrix holds gene activity, pass "
                             "atac_kind='gene_activity'.")
    assert alone[0].index("rename them") < alone[0].index("gene activity")


def test_gene_activity_with_peak_names_is_plain(tmp_path):
    peaks = _adata(["chr1_100_200", "chr1-300-400", "chr2:10-20", "chrX_5_9"])
    one = _messages(lambda: ingest._to_canonical(peaks, tmp_path / 'a.h5', modality='atac_gas'))
    assert one == ["100% of the ATAC feature names look like peaks such as "
                   "chr1:100-200, not genes. If the matrix holds peaks, pass "
                   "modality='atac_peak'."]
    two = _messages(lambda: ingest.export_dataset(peaks, tmp_path / "Y", rna=None,
                                                  atac="X", atac_kind="gene_activity"))
    assert two[0].endswith("pass atac_kind='peak'.")
    for m in one + two:
        assert "did you mean" not in m and " but " not in m


