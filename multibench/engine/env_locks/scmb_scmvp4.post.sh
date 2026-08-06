#!/usr/bin/env bash
# scMVP's own package is in no lockfile: the working env imports it from
# scmbench_tools/tools_scripts/scMVP/scMVP/scMVP/, a checkout OUTSIDE this
# repository, so a rebuild produced an env that built cleanly and then failed
# with "No module named 'scMVP.inference'".
#
# Installed from the public upstream at a pinned commit. NOTE this is upstream
# master, not necessarily byte-identical to the local checkout the recorded
# results were produced with - the local copy has no git remote to compare against.
set -euo pipefail
python -c "import scMVP.inference" 2>/dev/null && { echo "scMVP already present"; exit 0; }
pip install --no-deps "git+https://github.com/bm2-lab/scMVP.git@9db000194d6f4adff3f94cc9f4640473f09e1298"
python -c "import scMVP.inference; print('scMVP OK')"
