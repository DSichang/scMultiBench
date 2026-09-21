"""multibench driver for Concerto (cross: several RNA+ADT batches).

Concerto writes one shard per batch, ``tf_<k>.tfrecord`` under
``./tfrecord/RNA_tf/`` (``tf_<k>`` = the k-th ``--path1``/``--path2`` file),
and both trains on and embeds the shards in the order ``os.listdir`` returns
them (main_Concerto.py:60, concerto_function5_3.py:1961), so that listing
fixes the order of the embedding's rows. It is the filesystem's order: for
the same three shards APFS lists tf_1, tf_0, tf_2, HFS+ tf_0, tf_1, tf_2, and
the benchmark host listed tf_2, tf_0, tf_1 (cty3, cty1, cty2 on D52).

This driver runs the unmodified main_Concerto.py with ``os.listdir`` giving
the shards in batch order, so the rows follow the argument order on every
filesystem - the order ``mtb.labels_for`` returns. Nothing else changes: any
other listing comes back as the filesystem gave it.

Args: ``--script_dir <dir of main_Concerto.py>``, then main_Concerto.py's own.
Kept to Python 3.8 syntax (the scmb_concerto env).
"""
import os
import re
import runpy
import sys

# create_tfrecord's shard names: tf_<k>.tfrecord (the variant main_Concerto.py
# writes), tf_<k>_no_zero[_no_norm].tfrecord (its zero_filter variants)
_SHARD = re.compile(r"tf_(\d+)(?:_no_zero(?:_no_norm)?)?\.tfrecord")


def shards_in_batch_order(names):
    """``names`` with the Concerto shards first, by batch index, then the rest
    as listed; a listing without a shard (or of bytes names) is returned as is."""
    shards = [(int(m.group(1)), n) for n in names if isinstance(n, str)
              for m in [_SHARD.fullmatch(n)] if m]
    if not shards:
        return names
    rest = [n for n in names if not (isinstance(n, str) and _SHARD.fullmatch(n))]
    return [n for _, n in sorted(shards)] + rest


def main():
    argv = sys.argv[1:]
    if "--script_dir" not in argv:
        sys.exit("run_concerto.py: --script_dir <folder of main_Concerto.py> is required")
    i = argv.index("--script_dir")
    script_dir = argv[i + 1]
    del argv[i:i + 2]
    script = os.path.join(script_dir, "main_Concerto.py")

    listdir = os.listdir

    def listdir_in_batch_order(*args, **kwargs):
        return shards_in_batch_order(listdir(*args, **kwargs))

    # both files call os.listdir through the module, so this reaches every call
    os.listdir = listdir_in_batch_order
    # as `python main_Concerto.py`: its bgi/ and concerto_function5_3 import from there
    sys.path[0] = script_dir
    sys.argv = [script] + argv
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
