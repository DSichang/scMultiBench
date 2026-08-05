#!/usr/bin/env bash
# cobolt is not on PyPI - `pip freeze` records it as `cobolt==0.0.1`, which no
# index can satisfy. The working env installed it from git; the exact commit is
# recorded in that env's dist-info direct_url.json, so pin it here.
set -euo pipefail
python -c "import cobolt" 2>/dev/null && { echo "cobolt already present"; exit 0; }
pip install --no-deps \
  "git+https://github.com/epurdom/cobolt.git@cea9a5c6297326aca00a10aeaa198d21b07e4889"
python -c "import cobolt; print('cobolt OK')"
