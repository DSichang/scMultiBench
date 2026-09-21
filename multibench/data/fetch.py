"""Download reference datasets from the repository's release assets."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path


def safe_extract(tar: tarfile.TarFile, dest) -> None:
    """extractall with a path-traversal guard (absolute paths, .., links out).

    A crafted archive could otherwise write outside ``dest``; every tarball
    the package opens (datasets, packed envs) goes through here. Where
    ``tarfile`` has the extraction filters (PEP 706: 3.12+ and the security
    releases of the older series) the ``'data'`` filter is passed as well:
    the archives are plain data, and the bare call raises
    ``DeprecationWarning`` on 3.12/3.13 and changes behaviour on 3.14.
    Interpreters without the filters keep the bare call behind the guard.
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
    """Download reference datasets into the data root, skipping those present.

    Parameters
    ----------
    *datasets : str
        Dataset ids to fetch; ``mtb.data.fetchable()`` lists them.
    data_path : Path | str | None, keyword-only
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``. Each dataset lands in
        ``<data_path>/<dataset>/``.
    quiet : bool, keyword-only
        Suppress the one "downloading ..." line per dataset.

    Returns
    -------
    pathlib.Path
        The data root (not the dataset folder) - pass it as ``data_path=``.

    Raises
    ------
    ValueError
        A dataset id is not a release asset (the message lists the ids).
    RuntimeError
        The downloaded archive lacks the ``<dataset>/`` folder.

    Examples
    --------
    >>> import multibench as mtb
    >>> root = mtb.data.fetch("D11")                     # <data_path>/D11/, about 11 MB
    >>> mtb.scan("D11", "vertical", data_path=root)
    >>> mtb.data.fetch("D11", "D28", data_path="data", quiet=True)

    Notes
    -----
    **What it covers.** The tutorial reference sets published as release
    assets (``mtb.data.fetchable()``); the progress line states each one's
    approximate download size. The full collection is linked from the
    scMultiBench README ("Get the data" in the installation guide).

    **Idempotence.** A non-empty ``<data_path>/<dataset>/`` counts as present
    and is left alone, whatever it holds (even for an id that is not a
    release asset); an empty leftover folder is removed and fetched again.

    **Atomic extraction.** Each archive is extracted into a scratch folder
    under the data root and moved into place, so an interrupted download or
    extraction never looks like a complete dataset on the next call. An
    archive entry that would write outside that folder raises
    ``RuntimeError``.

    See Also
    --------
    mtb.data.fetchable : the dataset ids this function can download.
    mtb.data.fetch_outputs : precomputed ``run_all`` outputs for a tutorial dataset.
    mtb.config.Config : ``data_path``, the default data root.
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
    """Download precomputed ``run_all`` outputs for a tutorial dataset.

    A stand-in for running the methods on a host without their conda envs
    (Colab, a laptop): ``evaluate`` and ``plot`` then run on real embeddings.

    Parameters
    ----------
    dataset : str
        Dataset id with shipped outputs: ``D11``, ``D28``, ``D46`` or ``D52``.
    methods : list of str, optional
        Method ids that must be in the tree; checked only, the whole tree
        is downloaded either way.
    data_path : Path | str | None, keyword-only
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``.
    quiet : bool, keyword-only
        Suppress the one "downloading ..." line.

    Returns
    -------
    pathlib.Path
        ``<data_path>/outputs/<dataset>`` - pass it to ``mtb.load_batch``.

    Raises
    ------
    ValueError
        ``dataset`` has no shipped outputs (the message lists the ids).
    KeyError
        A name in ``methods`` is not in the tree (the message lists its methods).
    RuntimeError
        The archive lacks ``batch_result.json``, or a non-empty ``outputs/<dataset>/`` lacks one.

    Examples
    --------
    >>> import multibench as mtb
    >>> out = mtb.data.fetch_outputs("D11")
    >>> res = mtb.load_batch(out)
    >>> res.summary[["method", "status", "ARI", "NMI"]]

    Notes
    -----
    **Tree contents.** Exactly what ``mtb.run_all`` writes and
    ``mtb.load_batch`` reads: ``batch_result.json``, ``long.csv``,
    ``summary.csv`` and one folder per method holding its ``embedding.h5``.
    The download URLs come from the shipped manifest
    ``multibench/engine/output_urls.json``.

    **Selecting methods.** ``methods`` only validates the names, so a typo
    fails here rather than as an empty plot later. To restrict what is
    loaded, use ``mtb.load_batch(out, methods=[...])``.

    **Idempotence.** When ``<data_path>/outputs/<dataset>/batch_result.json``
    exists nothing is downloaded. An empty leftover folder is replaced; a
    non-empty one without ``batch_result.json`` raises ``RuntimeError`` -
    remove it and call again.

    **Atomic extraction.** The archive is extracted into a scratch folder and
    moved into place, so an interrupted download never looks like a complete
    tree on the next call. The archive may be rooted at the tree itself or
    at one folder (``outputs-D11/``, ``D11/``); an entry that would write
    outside the scratch folder raises ``RuntimeError``.

    See Also
    --------
    mtb.load_batch : load the downloaded tree as a ``BatchResult``.
    mtb.run_all : produces the same tree from your own run.
    mtb.data.fetch : download the input datasets themselves.
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
