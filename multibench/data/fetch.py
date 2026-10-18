"""Download reference datasets from the repository's release assets."""
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

#: ``{dataset: url}`` of the benchmark host's ``run_all`` output trees (the
#: deprecated ``fetch_outputs``); ships in the wheel next to ``packed_urls.json``
OUTPUT_MANIFEST = Path(__file__).resolve().parent.parent / "engine" / "output_urls.json"


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


def _output_urls() -> dict:
    """The ``{dataset: url}`` manifest :data:`OUTPUT_MANIFEST`."""
    import json
    return dict(json.loads(OUTPUT_MANIFEST.read_text()))


def fetch(*datasets: str, data_path=None, quiet: bool = False) -> Path:
    """Download reference datasets into the data root, skipping those present.

    Parameters
    ----------
    *datasets : str
        Dataset ids to fetch; ``mtb.data.fetchable()`` lists them.
    data_path : Path | str | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``. Each dataset lands in
        ``<data_path>/<dataset>/``.
    quiet : bool
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
    OSError
        The download failed; the message names the URL.

    Examples
    --------
    >>> import multibench as mtb
    >>> # the data root; D11 lands in <data_path>/D11/ (about 11 MB)
    >>> root = mtb.data.fetch("D11")
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

    **Atomic extraction.** Each archive is unpacked into a temporary folder
    under the data root and then moved into place. An archive entry that
    would write outside that folder raises ``RuntimeError``.

    See Also
    --------
    mtb.data.fetchable : the dataset ids this function can download.
    mtb.config.Config : ``data_path``, the default data root.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    root = Path(data_path) if data_path is not None else config.DEFAULT.data_path
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
        if not quiet:
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
    return root


def _methods_in_tree(folder: Path) -> list:
    """Method names recorded in ``<folder>/batch_result.json``, in run order."""
    import json
    with open(folder / "batch_result.json") as fh:
        blob = json.load(fh)
    return [r.get("method") for r in blob.get("records", [])]


def fetch_outputs(dataset: str, methods=None, *, data_path=None,
                  quiet: bool = False) -> Path:
    """Deprecated: download the stored ``run_all`` outputs of ``dataset`` into
    ``<data_path>/outputs/<dataset>``; returns that folder."""
    import shutil as _shutil
    import tempfile as _tempfile

    urls = _output_urls()
    if dataset not in urls:
        raise ValueError(
            f"{dataset!r} has no shipped run_all outputs; available: "
            f"{', '.join(sorted(urls))}")
    if methods is not None:
        if isinstance(methods, str) and methods in urls:
            raise TypeError("fetch_outputs takes one dataset; call it once per dataset.")
        from ..engine import registry
        registry.check_id_list(methods, "methods")
    root = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    outputs = root / "outputs"
    dest = outputs / dataset
    url = urls[dataset]
    have_tree = (dest / "batch_result.json").is_file()
    # the tree records the URL it came from: a new manifest URL fetches it again
    source = dest / ".source"
    if not (have_tree and source.is_file() and source.read_text().strip() == url):
        old = None
        if have_tree:                        # the first free name: never delete a tree
            old = outputs / f"{dataset}.old"
            k = 2
            while old.exists():
                old, k = outputs / f"{dataset}.old-{k}", k + 1
        if dest.is_dir() and not have_tree:
            if any(dest.iterdir()):
                raise RuntimeError(
                    f"{dest} exists without batch_result.json - remove it and "
                    f"the outputs will be fetched fresh")
            dest.rmdir()                     # empty leftover
        if not quiet:
            print(f"downloading run_all outputs for {dataset} from {url} ..."
                  + (f" (the earlier tree moves to {old})" if old else ""), flush=True)
        tgz = _download_or_explain(
            url, f"the stored outputs of {dataset}",
            f"pass the tree you have, {dest}, to mtb.load_batch." if old else
            f"download the file by hand and unpack it into {dest}, so that "
            f"{dest / 'batch_result.json'} exists.")
        outputs.mkdir(parents=True, exist_ok=True)
        tmp = Path(_tempfile.mkdtemp(dir=outputs, prefix=f".{dataset}-"))
        try:
            with tarfile.open(tgz) as t:
                safe_extract(t, tmp)
            # rooted at the tree or at one folder (``outputs-D11/``), nothing deeper
            found = next((c for c in [tmp, *sorted(x for x in tmp.iterdir() if x.is_dir())]
                          if (c / "batch_result.json").is_file()), None)
            if found is None:
                raise RuntimeError(
                    f"downloaded archive for {dataset} did not contain batch_result.json")
            (found / ".source").write_text(url + "\n")
            if old is not None:
                dest.rename(old)
            found.rename(dest)
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)
    if methods is not None:
        have = _methods_in_tree(dest)

        def _id(m):                          # any letter case, or a display name
            if m in have:
                return m
            try:
                return registry.check_method(m)
            except KeyError:
                return m
        unknown = [m for m in methods if _id(m) not in have]
        if unknown:
            raise KeyError(
                f"no shipped output for {unknown} in {dataset}; methods in the "
                f"tree: {have}")
    return dest
