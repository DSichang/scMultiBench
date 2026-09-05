"""Download reference datasets from the repository's release assets."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path


def safe_extract(tar: tarfile.TarFile, dest) -> None:
    """extractall with a path-traversal guard (absolute paths, .., links out).

    A crafted archive could otherwise write outside ``dest``; every tarball we
    open (datasets, packed envs) goes through here. On a Python whose
    ``tarfile`` has the extraction filters (3.8.17+/3.9.17+/3.12+) the
    ``'data'`` filter is passed as well - the archives are plain data, and
    the bare call raises ``DeprecationWarning`` on 3.12/3.13 and changes
    behaviour on 3.14; older interpreters keep the bare call (the guard
    above is what they have).
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

from .. import config

RELEASE_URL = "https://github.com/DSichang/scMultiBench/releases/download/data-v1"

# datasets published as release assets, with approximate download sizes
AVAILABLE = {"D11": "11 MB", "D28": "137 MB", "D45": "290 MB",
             "D46": "97 MB", "D52": "179 MB"}

#: ``{dataset: url}`` of the benchmark host's ``run_all`` output trees
#: (``mtb.data.fetch_outputs``); ships in the wheel next to ``packed_urls.json``
OUTPUT_MANIFEST = Path(__file__).resolve().parent.parent / "engine" / "output_urls.json"


def _download(url: str) -> Path:
    """Fetch ``url`` to a temporary file and return its path.

    The single network seam of this module: tests monkeypatch it to serve a
    local tarball, so the extraction / idempotence / filter logic runs
    without touching the network.
    """
    tgz, _ = urllib.request.urlretrieve(url)
    return Path(tgz)


def _output_urls() -> dict:
    """The ``{dataset: url}`` manifest, read from :data:`OUTPUT_MANIFEST`."""
    import json
    return dict(json.loads(OUTPUT_MANIFEST.read_text()))


def fetch(*datasets: str, data_path=None, quiet: bool = False) -> Path:
    """Ensure the named reference datasets exist locally, downloading if needed.

    Idempotent: datasets already present under the data root are left alone.
    Returns the data root. The full 65-dataset collection is linked from the
    scMultiBench README; this helper covers the tutorial reference sets.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    root = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    for ds in datasets:
        d = root / ds
        if d.is_dir() and any(d.iterdir()):
            continue
        if d.is_dir():                       # empty leftover from a failed run
            d.rmdir()
        if ds not in AVAILABLE:
            raise ValueError(
                f"{ds!r} is not in the release assets ({', '.join(sorted(AVAILABLE))}); "
                "see 'Get the data' in the installation guide for the full collection")
        root.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"downloading {ds} ({AVAILABLE[ds]}) ...", flush=True)
        tgz = _download(f"{RELEASE_URL}/{ds}.tar.gz")
        # extract to a scratch dir first and move into place atomically, so an
        # interrupted download/extract can never masquerade as a complete
        # dataset on the next run
        tmp = Path(_tempfile.mkdtemp(dir=root, prefix=f".{ds}-"))
        try:
            with tarfile.open(tgz) as t:
                safe_extract(t, tmp)
            if not (tmp / ds).is_dir():
                raise RuntimeError(f"downloaded archive did not contain {ds}/")
            (tmp / ds).rename(root / ds)
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)
    return root


def _methods_in_tree(folder: Path) -> list:
    """Method names recorded in ``<folder>/batch_result.json``, in run order."""
    import json
    with open(folder / "batch_result.json") as fh:
        blob = json.load(fh)
    return [r.get("method") for r in blob.get("records", [])]


def fetch_outputs(dataset: str, methods=None, *, data_path=None,
                  quiet: bool = False) -> Path:
    """Download the benchmark host's ``run_all`` output tree for a tutorial dataset.

    The tree is exactly what :func:`multibench.run_all` writes and
    :func:`multibench.load_batch` reads: ``batch_result.json``, ``long.csv``,
    ``summary.csv`` and one folder per method holding its ``embedding.h5``.
    It is the stand-in for running the methods on a host without their conda
    envs (Colab, a laptop): real embeddings, so ``evaluate`` / ``plot`` run on
    real results. URLs come from the shipped manifest
    ``multibench/engine/output_urls.json``.

    Parameters
    ----------
    dataset : str
        A dataset id in the manifest (``D11``, ``D28``, ``D46``, ``D52``).
    methods : list of str, optional
        Method names that must be in the tree. The whole tree is downloaded
        either way (one archive per dataset) and the dataset folder is
        returned either way; this only validates the names so a typo fails
        here rather than as an empty plot later. Restrict what ``load_batch``
        reads with its own ``methods=`` filter.
    data_path : path-like, keyword-only, optional
        Data root; default ``config.DEFAULT.data_path``. The tree lands in
        ``<data_path>/outputs/<dataset>/``.
    quiet : bool, keyword-only
        Suppress the one "downloading ..." line.

    Returns
    -------
    pathlib.Path
        ``<data_path>/outputs/<dataset>`` - pass it to ``load_batch``.

    Raises
    ------
    ValueError
        ``dataset`` is not in the manifest (the message lists the ids).
    KeyError
        A name in ``methods`` has no output in the tree (the message lists
        the methods the tree has).
    RuntimeError
        The archive did not contain a ``batch_result.json`` (or would write
        outside the target directory), or a foreign non-empty
        ``outputs/<dataset>/`` without one is in the way.

    Notes
    -----
    Idempotent: when ``<data_path>/outputs/<dataset>/batch_result.json``
    exists nothing is downloaded. Extraction goes to a scratch directory
    first and is moved into place atomically, so an interrupted download can
    never masquerade as a complete tree on the next call.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    urls = _output_urls()
    if dataset not in urls:
        raise ValueError(
            f"{dataset!r} has no shipped run_all outputs; available: "
            f"{', '.join(sorted(urls))}")
    root = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    outputs = root / "outputs"
    dest = outputs / dataset
    if not (dest / "batch_result.json").is_file():
        if dest.is_dir():
            if any(dest.iterdir()):
                raise RuntimeError(
                    f"{dest} exists without batch_result.json - remove it and "
                    f"the outputs will be fetched fresh")
            dest.rmdir()                     # empty leftover
        outputs.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"downloading run_all outputs for {dataset} from {urls[dataset]} ...",
                  flush=True)
        tgz = _download(urls[dataset])
        tmp = Path(_tempfile.mkdtemp(dir=outputs, prefix=f".{dataset}-"))
        try:
            with tarfile.open(tgz) as t:
                safe_extract(t, tmp)
            # the archive may be rooted at the tree itself or at one folder
            # (``outputs-D11/``, ``D11/``) - accept either, nothing deeper
            found = next((c for c in [tmp, *sorted(x for x in tmp.iterdir() if x.is_dir())]
                          if (c / "batch_result.json").is_file()), None)
            if found is None:
                raise RuntimeError(
                    f"downloaded archive for {dataset} did not contain batch_result.json")
            found.rename(dest)
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)
    if methods is not None:
        have = _methods_in_tree(dest)
        unknown = [m for m in methods if m not in have]
        if unknown:
            raise KeyError(
                f"no shipped output for {unknown} in {dataset}; methods in the "
                f"tree: {have}")
    return dest
