#!/usr/bin/env bash
# The matilda package is absent from this lockfile: the working env has it as an
# EDITABLE install from a local checkout, and freeze() skips `-e ` lines because
# a local path means nothing on another machine. A rebuild therefore produced an
# env that built cleanly and then failed with "No module named 'matilda'".
#
# Installed from PyPI, NOT from the checkout's git remote
# (github.com/DSichang/matilda-sc), which is private - `git ls-remote` there asks
# for credentials, so a new user could never clone it.
#
# NOTE a real version difference: the working env has matilda-sc 0.2.0 from that
# private checkout; PyPI publishes 0.2.1 and 0.1, not 0.2.0. A new user gets
# 0.2.1. It provides `import matilda` and runs the benchmark, but it is not
# byte-identical to what the recorded results were produced with.
set -euo pipefail
python -c "import matilda" 2>/dev/null && { echo "matilda already present"; exit 0; }
pip install --no-deps "matilda-sc==0.2.1"
python -c "import matilda; print('matilda OK')"
