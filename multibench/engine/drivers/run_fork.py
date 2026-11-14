"""Run a method script with worker processes started by fork (macOS).

usage: python run_fork.py <script.py> [the script's arguments]

The method scripts were written on Linux, where a worker process is a fork of
the script's process. On macOS Python starts a worker by importing the script
again; a script without an ``if __name__ == "__main__":`` guard then runs from
its first line in every worker, the workers end, and the script waits for them
without end (seen with SCALEX and GLUE, whose data loaders use workers). This
sets the Linux behaviour and runs the script unchanged, as ``python script.py``
would: the same ``sys.argv``, its folder first on ``sys.path``.
"""
import multiprocessing
import os
import runpy
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path[0] = os.path.dirname(os.path.abspath(script))
    runpy.run_path(script, run_name="__main__")
