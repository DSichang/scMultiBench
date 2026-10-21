"""Download demo datasets from the repository's release assets."""
from __future__ import annotations

import http.client
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from .. import config


def safe_extract(tar: tarfile.TarFile, dest) -> None:
    """``extractall`` that refuses an entry or link writing outside ``dest``.

    Every tarball the package opens goes through here. Where ``tarfile`` has
    the extraction filters (PEP 706) the ``'data'`` filter is passed too: the
    bare call warns on 3.12/3.13 and changes behaviour on 3.14.
    """
    import os
    dest = Path(dest).resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise RuntimeError(f"archive entry escapes target dir: {m.name!r}")
        if m.islnk() or m.issym():
            link = (target.parent / m.linkname).resolve() if not m.linkname.startswith("/") else Path(m.linkname)
            if not str(link).startswith(str(dest) + os.sep):
                raise RuntimeError(f"archive link escapes target dir: {m.name!r}")
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
    else:                                 # pragma: no cover - pre-filter Pythons
        tar.extractall(dest)


RELEASE_URL = "https://github.com/DSichang/scMultiBench/releases/download/data-v1"

#: datasets published as release assets, with approximate download sizes
AVAILABLE = {"D11": "11 MB", "D28": "137 MB", "D45": "290 MB",
             "D46": "97 MB", "D52": "179 MB",
             # one small dataset per integration task (tools/make_tutorial_data.py)
             "D27mini_vertical": "24 MB", "D22mini": "8 MB", "D27mini": "15 MB",
             "D45mini": "11 MB", "D46mini": "4 MB", "D38mini": "2 MB",
             "D52mini": "2 MB", "D37mini": "12 MB", "D49mini": "5 MB",
             "D56mini": "10 MB", "D58mini": "7 MB", "D59mini": "8 MB"}

#: the category of each fetchable dataset without stored results, as its tutorial runs it
DEMO_CATEGORIES = {"D46": "mosaic", "D27mini_vertical": "vertical",
                   "D22mini": "vertical", "D27mini": "diagonal", "D37mini": "diagonal",
                   "D38mini": "mosaic", "D45mini": "mosaic", "D46mini": "mosaic",
                   "D49mini": "mosaic", "D52mini": "cross", "D56mini": "cross",
                   "D58mini": "cross", "D59mini": "cross"}

def _download(url: str) -> Path:
    """Fetch ``url`` to a temporary file; the module's one network call."""
    tgz, _ = urllib.request.urlretrieve(url)
    return Path(tgz)


def _download_or_explain(url: str, what: str, by_hand: str) -> Path:
    """:func:`_download`, a failure raised as ``OSError`` naming ``what``,
    ``url`` and ``by_hand``, the sentence that says how to get it by hand."""
    try:
        return _download(url)
    except urllib.error.HTTPError as e:
        why = f"HTTP {e.code}"
        err = e
    except (OSError, http.client.HTTPException) as e:
        # URLError keeps the cause in .reason
        why = str(getattr(e, "reason", None) or e) or type(e).__name__
        err = e
    raise OSError(
        f"Could not download {what} from {url} ({why}). Check the network or "
        f"proxy, or {by_hand}") from err


def fetch(*datasets: str, verbose: bool = True) -> "Path | list[Path]":
    """Download demo datasets into the data root, skipping those present.

    Parameters
    ----------
    *datasets : str
        Dataset ids; ``mtb.catalog.datasets()`` lists them.
    verbose : bool
        Print one "downloading ..." line per dataset.

    Returns
    -------
    pathlib.Path or list of pathlib.Path
        The dataset folder, ``<data root>/<dataset>/``; a list of them for
        several datasets.

    Raises
    ------
    ValueError
        A dataset id that cannot be downloaded (the message lists the ids).
    OSError
        The download failed; the message names the URL.

    Examples
    --------
    >>> import multibench as mtb
    >>> folder = mtb.data.fetch("D11")          # about 11 MB
    >>> mtb.scan(folder, "vertical")

    Notes
    -----
    A non-empty ``<data root>/<dataset>/`` counts as present and is left
    alone. Each archive is unpacked into a temporary folder under the data
    root and then moved into place.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    root = Path(config.data_path)
    for ds in datasets:
        d = root / ds
        if d.is_dir() and any(d.iterdir()):
            continue
        if d.is_dir():                       # empty leftover of a failed run
            d.rmdir()
        if ds not in AVAILABLE:
            raise ValueError(
                f"{ds!r} is not in the release assets ({', '.join(sorted(AVAILABLE))}); "
                "see 'Get the data' in the installation guide for the full collection")
        url = f"{RELEASE_URL}/{ds}.tar.gz"
        if verbose:
            print(f"downloading {ds} ({AVAILABLE[ds]}) from {url} ...", flush=True)
        tgz = _download_or_explain(
            url, ds, f"download the file by hand and unpack it into {root}, as shown "
                     "under 'Get the data' in the installation guide.")
        root.mkdir(parents=True, exist_ok=True)
        # into place in one move: an interrupted extract never looks complete
        tmp = Path(_tempfile.mkdtemp(dir=root, prefix=f".{ds}-"))
        try:
            with tarfile.open(tgz) as t:
                safe_extract(t, tmp)
            if not (tmp / ds).is_dir():
                raise RuntimeError(f"downloaded archive did not contain {ds}/")
            (tmp / ds).rename(root / ds)
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)
    folders = [root / ds for ds in datasets]
    return folders[0] if len(folders) == 1 else folders
