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
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry

F = importlib.import_module("multibench.data.fetch")
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
MATILDA = ["scan", "D11", "--category", "vertical", "--methods", "Matilda",
           "--modalities", "rna,adt", "--format", "csv"]
# a sentence with its own subject, as every other reason of the column (R5-10)
REASON = "The method scripts are not fetched. Run multibench fetch --scripts."


# ================================================= R5-04: scan --strict table
@pytest.fixture
def no_scripts(tmp_path, monkeypatch):
    """Every env installed, a GPU, and an empty ``repo_path``."""
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    repo = tmp_path / "emptyrepo"
    repo.mkdir()
    monkeypatch.setattr(config.DEFAULT, "repo_path", repo)
    return repo


def _table(out: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(out), keep_default_na=False)


def test_strict_table_agrees_with_its_error(no_scripts, capsys):
    assert cli.main(MATILDA + ["--strict"]) == 1
    cap = capsys.readouterr()
    assert "Rows whose method scripts are not fetched: 1." in cap.err
    df = _table(cap.out)
    assert list(df["runnable"]) == [False]
    assert list(df["reason"]) == [REASON]
    # the caveat is unchanged
    assert "method scripts not found under" in df["caveat"].iloc[0]


def test_strict_table_every_row_without_methods(no_scripts, capsys):
    argv = ["scan", "D11", "--category", "vertical", "--modalities", "rna,adt",
            "--format", "csv"]
    assert cli.main(argv) == 0
    plain = _table(capsys.readouterr().out)
    assert plain["runnable"].any()
    assert cli.main(argv + ["--strict"]) == 1
    strict = _table(capsys.readouterr().out)
    assert list(strict["method"]) == list(plain["method"])
    assert not strict["runnable"].any()
    was = plain["runnable"].astype(bool)
    assert (strict.loc[was, "reason"] == REASON).all()
    # a row blocked by something else keeps its own reason
    assert list(strict.loc[~was, "reason"]) == list(plain.loc[~was, "reason"])


def test_strict_table_text_format(no_scripts, capsys):
    assert cli.main(MATILDA[:-2] + ["--strict"]) == 1
    lines = capsys.readouterr().out.splitlines()
    head = lines[0].split()
    row = next(line for line in lines if line.lstrip().startswith("Matilda"))
    assert row.split()[head.index("runnable")] == "False"
    assert "method scripts are not fetched" in row


def test_without_strict_the_table_and_python_scan_are_unchanged(no_scripts, capsys):
    assert cli.main(MATILDA) == 0
    df = _table(capsys.readouterr().out)
    assert list(df["runnable"]) == [True] and list(df["reason"]) == [""]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = mtb.scan("D11", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                       verbose=False).iloc[0]
    assert row["runnable"] and row["reason"] == ""


def test_strict_table_unchanged_once_the_scripts_are_present(no_scripts, monkeypatch,
                                                             capsys):
    monkeypatch.setattr(config, "scripts_present", lambda *a, **k: True)
    assert cli.main(MATILDA) == 0
    plain = capsys.readouterr().out
    assert cli.main(MATILDA + ["--strict"]) == 0
    assert capsys.readouterr().out == plain


# ================================================= R5-05: download errors
def _http_404(url):
    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)


def _refused(url):
    raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))


@pytest.mark.parametrize("dead, why", [(_http_404, "(HTTP 404)"),
                                       (_refused, "Connection refused")],
                         ids=["http-404", "url-error"])
def test_fetch_names_the_url_and_the_manual_route(tmp_path, monkeypatch, capsys,
                                                  dead, why):
    monkeypatch.setattr(F, "_download", dead)
    root = tmp_path / "data"
    url = f"{F.RELEASE_URL}/D46.tar.gz"
    with pytest.raises(OSError) as e:
        mtb.data.fetch("D46", data_path=root)
    msg = str(e.value)
    assert msg.startswith(f"Could not download D46 from {url} ")
    assert why in msg and "Get the data" in msg and str(root) in msg
    assert isinstance(e.value.__cause__, urllib.error.URLError)
    assert not root.exists()                            # nothing left under data_path
    assert capsys.readouterr().out == f"downloading D46 (97 MB) from {url} ...\n"


def test_fetch_leaves_an_existing_data_path_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "_download", _http_404)
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(OSError, match="Get the data"):
        mtb.data.fetch("D11", data_path=root, quiet=True)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("dead, why", [(_http_404, "(HTTP 404)"),
                                       (_refused, "Connection refused")],
                         ids=["http-404", "url-error"])
def test_fetch_outputs_names_the_url(tmp_path, monkeypatch, dead, why):
    monkeypatch.setattr(F, "_download", dead)
    root = tmp_path / "data"
    url = F._output_urls()["D11"]
    with pytest.raises(OSError) as e:
        mtb.data.fetch_outputs("D11", data_path=root, quiet=True)
    msg = str(e.value)
    assert msg.startswith(f"Could not download the stored outputs of D11 from {url} ")
    assert why in msg and "Get the data" in msg
    assert str(root / "outputs" / "D11") in msg
    assert not root.exists()


def test_cli_fetch_prints_the_sentence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(F, "_download", _http_404)
    monkeypatch.setattr(config.DEFAULT, "data_path", tmp_path / "data")
    assert cli.main(["fetch", "D46"]) == 1
    err = capsys.readouterr().err
    url = f"{F.RELEASE_URL}/D46.tar.gz"
    assert f"downloading D46 (97 MB) from {url} ...\n" in err
    assert f"error: Could not download D46 from {url} (HTTP 404). " in err
    assert "HTTP Error 404: Not Found" not in err


def test_raises_tables_list_the_download_error():
    import inspect
    for fn in (mtb.data.fetch, mtb.data.fetch_outputs):
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
    one = _messages(lambda: ingest.to_canonical(atac, tmp_path / "a.h5",
                                                modality="atac_peak"))
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
    one = _messages(lambda: ingest.to_canonical(peaks, tmp_path / "a.h5",
                                                modality="atac_gas"))
    assert one == ["100% of the ATAC feature names look like peaks such as "
                   "chr1:100-200, not genes. If the matrix holds peaks, pass "
                   "modality='atac_peak'."]
    two = _messages(lambda: ingest.export_dataset(peaks, tmp_path / "Y", rna=None,
                                                  atac="X", atac_kind="gene_activity"))
    assert two[0].endswith("pass atac_kind='peak'.")
    for m in one + two:
        assert "did you mean" not in m and " but " not in m


def test_cli_convert_spells_the_flags(tmp_path, capsys):
    src = tmp_path / "multi.h5ad"
    a = _adata(GENES)
    a.obsm["atac"] = np.ones((4, 4))
    a.uns["atac_names"] = PEAK_IDS
    a.write_h5ad(src)
    p = tmp_path / "atac.h5ad"
    _adata(PEAK_IDS).write_h5ad(p)
    assert cli.main(["convert", str(p), str(tmp_path / "o.h5"),
                     "--modality", "atac_peak"]) == 0
    err = capsys.readouterr().err
    assert re.search(r"warning: Only 0% of the ATAC feature names .* multibench scan "
                     r"names these methods\. If the matrix holds gene activity, pass "
                     r"--modality atac_gas\.", err), err
    assert cli.main(["convert", str(src), str(tmp_path / "D"), "--rna", "X",
                     "--atac", "obsm:atac", "--atac-kind", "peak"]) == 0
    err = capsys.readouterr().err
    assert "rename them to chr:start-end" in err and "--atac-kind" not in err
