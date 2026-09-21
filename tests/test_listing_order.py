"""A method whose cell order comes from a directory listing gets a fixed order.

Concerto (cross) writes one shard per batch, ``tf_<k>.tfrecord`` under
``./tfrecord/RNA_tf/`` (``tf_<k>`` = the k-th ``--path1``/``--path2`` file),
and stacks its embedding rows in the order ``os.listdir`` returns those
shards (``main_Concerto.py:60``, ``concerto_function5_3.py:1961``). That
order belongs to the filesystem: for the same three names, created in
Concerto's order, APFS listed ``tf_1, tf_0, tf_2``, HFS+ and FAT
``tf_0, tf_1, tf_2``, and the benchmark host ``tf_2, tf_0, tf_1`` - the
stored D52 / D52s scores record ``cty3, cty1, cty2``, found there by
scoring every permutation. ``labels_for("D52", "cross", "Concerto")``
returned ``cty1, cty2, cty3`` - right on a sorted listing only - and
``evaluate`` took it without a word.

The runner now starts Concerto's cross variant through
``engine/drivers/run_concerto.py``, which runs the unmodified script with
the shards listed in batch order, so the rows follow the argument order on
every filesystem and ``labels_for`` is right as it stands.
"""
import os
import re
import runpy
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import multibench as mtb
from multibench.engine import registry, runner, schema

PKG = Path(mtb.__file__).resolve().parent
DRIVER = "engine/drivers/run_concerto.py"


def _driver():
    # run_path, not an import: no __pycache__ lands in the package's drivers/
    return runpy.run_path(str(PKG / DRIVER), run_name="run_concerto")


# ------------------------------------------------ the run, end to end
#: Stand-ins for the two upstream files: the same shard names, the same
#: creation order and the same listing expression, without TensorFlow.
_FAKE_HELPER = '''\
import os
import numpy as np


def concerto_make_tfrecord(counts, tf_path):
    # create_tfrecord: one shard per batch index k, then vocab_size.npz
    os.makedirs(tf_path, exist_ok=True)
    for k, n in enumerate(counts):
        with open(os.path.join(tf_path, 'tf_{}.tfrecord'.format(k)), 'w') as f:
            f.write('{} {}'.format(k, n))
    open(os.path.join(tf_path, 'vocab_size.npz'), 'w').close()


def concerto_test_multimodal(RNA_tf_path):
    # the upstream listing, verbatim; each row holds its batch index
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    rows = []
    for name in tf_list_1:
        k, n = map(int, open(os.path.join(RNA_tf_path, name)).read().split())
        rows += [k] * n
    return np.array(rows, dtype=float)[:, None]
'''

_FAKE_MAIN = '''\
import argparse
import h5py
from concerto_function5_3 import *

parser = argparse.ArgumentParser("Concerto")
parser.add_argument('--path1', nargs='+', default=[])
parser.add_argument('--path2', nargs='+', default=[])
parser.add_argument('--save_path', default='NULL')
args = parser.parse_args()
counts = []
for p in args.path1:
    with h5py.File(p, 'r') as f:
        counts.append(len(f['matrix/barcodes']))
concerto_make_tfrecord(counts, './tfrecord/RNA_tf/')
emb = concerto_test_multimodal('./tfrecord/RNA_tf/')
with h5py.File(args.save_path + '/embedding.h5', 'w') as f:
    f.create_dataset('data', data=emb)
'''

#: a filesystem whose listings come in descending name order
_DESCENDING_FS = '''\
import os
_listdir = os.listdir


def _descending(*args, **kwargs):
    return sorted(_listdir(*args, **kwargs), reverse=True)


os.listdir = _descending
'''


def test_concerto_rows_follow_labels_for_on_any_listing_order(root, tmp_path, monkeypatch):
    data = root / "data"
    repo = tmp_path / "repo"
    script_dir = repo / "tools_scripts" / "Concerto"
    script_dir.mkdir(parents=True)
    (script_dir / "main_Concerto.py").write_text(_FAKE_MAIN)
    (script_dir / "concerto_function5_3.py").write_text(_FAKE_HELPER)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_DESCENDING_FS)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(site))

    out = tmp_path / "out"
    mtb.run("Concerto", "cross", inputs=mtb.inputs_for("D52", "cross", "Concerto", data_path=data),
            out_dir=str(out), convert=False, cmd_template="{cmd}", repo_path=repo)
    with h5py.File(out / "embedding.h5", "r") as f:
        rows = f["data"][:, 0].astype(int)

    labels = mtb.labels_for("D52", "cross", "Concerto", data_path=data)
    expected = np.concatenate([np.full(len(pd.read_csv(path)), int(stem[3:]) - 1)
                               for stem, path in labels.items()])
    blocks = [int(k) for i, k in enumerate(rows) if i == 0 or rows[i - 1] != k]
    assert blocks == [0, 1, 2]
    np.testing.assert_array_equal(rows, expected)


def test_concerto_cross_command_runs_the_driver(root):
    inputs = mtb.inputs_for("D52", "cross", "Concerto", data_path=root / "data")
    cmd = mtb.run("Concerto", "cross", inputs=inputs, out_dir="out/Concerto_D52", dry_run=True)
    i = cmd.index(str(PKG / DRIVER))
    assert cmd[i - 1] == "python"
    assert cmd[i + 1] == "--script_dir"
    assert Path(cmd[i + 2]) == runner._repo_root_no_fetch() / "tools_scripts" / "Concerto"
    # the upstream arguments are unchanged: one flag per modality, batches in order
    assert cmd[i + 3] == "--path1"
    assert [Path(p).name for p in cmd[i + 4:i + 7]] == ["rna1.h5", "rna2.h5", "rna3.h5"]
    assert cmd[i + 7] == "--path2"
    assert [Path(p).name for p in cmd[i + 8:i + 11]] == ["adt1.h5", "adt2.h5", "adt3.h5"]


# ------------------------------------------------ the driver's ordering
def test_shards_are_listed_in_batch_order():
    order = _driver()["shards_in_batch_order"]
    listed = ["tf_2.tfrecord", "vocab_size.npz", "tf_10.tfrecord", "tf_0.tfrecord", "tf_1.tfrecord"]
    assert order(listed) == ["tf_0.tfrecord", "tf_1.tfrecord", "tf_2.tfrecord",
                             "tf_10.tfrecord", "vocab_size.npz"]


def test_a_listing_without_shards_is_unchanged():
    order = _driver()["shards_in_batch_order"]
    for listed in (["weight_encoder_epoch2.h5", "weight_encoder_epoch1.h5"], [],
                   [b"tf_1.tfrecord", b"tf_0.tfrecord"]):
        assert order(listed) == listed


def test_the_upstream_facts_the_driver_relies_on():
    concerto = runner._repo_root_no_fetch() / "tools_scripts" / "Concerto"
    main = (concerto / "main_Concerto.py").read_text()
    helper = (concerto / "concerto_function5_3.py").read_text()
    # batch k+1 is the k-th --path1/--path2 file, concatenated in that order
    assert 'batch_name = ["batch{}".format(i) for i in range(1, len(args.path1)+1)]' in main
    assert "for adt, rna, batch in zip(adt_files, rna_files, batch_name):" in main
    # its shard is tf_<k>.tfrecord, k = the batch's position in cell order
    assert "batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()" in helper
    assert "tfrecord_file = tf_path + '/tf.tfrecord'" in helper
    assert "place = batch_dict.index(batch)" in helper
    assert "file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch))" in helper
    # the rows follow the listing, made through the os module the driver wraps
    listing = "tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]"
    assert listing in main and listing in helper
    assert "from os import" not in main + helper


# ------------------------------------------------ every listing is accounted for
_LISTING = re.compile(r"os\.listdir|os\.scandir|glob\.glob|\bglob\(|\.iterdir\(|\.r?glob\(|"
                      r"os\.walk|list\.files|Sys\.glob|\bdir\(")

#: Every file of a registered method's upstream folder, and every package
#: driver, that lists a directory - and why the listing order cannot reach
#: the output rows unnoticed:
#: driver    - Concerto; engine/drivers/run_concerto.py lists the shards in batch order
#: staged    - reads a data_dir; run() stages it and records the order the
#:             script's glob returns there (runner.stage_slices)
#: by_name   - lists one folder and picks each file by its name
#: not_data  - lists code, not data
#: not_run   - a script in the folder that no variant runs
_ACCOUNTED = {
    "tools_scripts/Concerto/main_Concerto.py": "driver",
    "tools_scripts/Concerto/concerto_function5_3.py": "driver",
    "engine/drivers/run_concerto.py": "driver",
    "tools_scripts/GPSA/main_GPSA.py": "staged",
    "engine/drivers/run_gpsa.py": "staged",
    "tools_scripts/PASTE/main_PASTE_pairwise.py": "staged",
    "tools_scripts/PASTE/main_PASTE_center.py": "not_run",
    "tools_scripts/PASTE2/main_PASTE2.py": "staged",
    "tools_scripts/scMM/datasets.py": "by_name",
    "tools_scripts/UnitedNet/src/configs/__init__.py": "not_data",
    "tools_scripts/totalVI/main_totalVI_imputation.py": "not_run",
}


def _listing_files():
    repo = runner._repo_root_no_fetch()
    folders = {Path(v.entrypoint).parent for s in registry.load() for v in s.variants
               if v.entrypoint.startswith("tools_scripts/")}
    found = {}
    for rel_dir in folders:
        for f in (repo / rel_dir).rglob("*"):
            if f.suffix.lower() in (".py", ".r", ".rmd") and f.is_file():
                found[str(f.relative_to(repo))] = f
    for f in (PKG / "engine" / "drivers").iterdir():
        if f.suffix.lower() in (".py", ".r") and f.is_file():
            found[str(f.relative_to(PKG))] = f
    return {rel for rel, f in found.items()
            if any(_LISTING.search(line) and not line.lstrip().startswith("#")
                   for line in f.read_text(errors="ignore").splitlines())}


def test_every_directory_listing_is_accounted_for():
    assert _listing_files() == set(_ACCOUNTED)
    entrypoints = {v.entrypoint for s in registry.load() for v in s.variants}
    for rel, why in _ACCOUNTED.items():
        if why == "not_run":
            assert rel not in entrypoints, rel


def test_each_disposition_holds():
    for s in registry.load():
        for v in s.variants:
            rel = v.driver or v.entrypoint
            why = _ACCOUNTED.get(rel) or _ACCOUNTED.get(v.entrypoint)
            if why == "driver":
                batches = {schema._batch_of(r) for r in v.stacked_roles()} - {None}
                if len(batches) > 1:
                    assert v.driver == DRIVER, (s.id, v.when)
            elif why == "staged":
                assert v.roles() == ["data_dir"] and v.output.kind == "coords", (s.id, v.when)
    concerto = [v for v in registry.get("Concerto").variants if v.when["category"] == "cross"]
    assert concerto and all(v.driver == DRIVER for v in concerto)
