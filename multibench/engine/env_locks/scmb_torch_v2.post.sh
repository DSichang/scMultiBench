#!/usr/bin/env bash
# cobolt and MultiMAP are not on PyPI. Worse, both names are taken there by
# UNRELATED projects (pypi multimap is 1.0.x; the benchmark's is MultiMAP 0.0.1),
# so `pip freeze`'s bare `name==version` pin cannot be satisfied correctly by any
# index. Commits below come from each package's dist-info direct_url.json in the
# working env, so this reproduces exactly what the benchmark ran.
set -euo pipefail

if ! python -c "import cobolt" 2>/dev/null; then
  pip install --no-deps \
    "git+https://github.com/epurdom/cobolt.git@cea9a5c6297326aca00a10aeaa198d21b07e4889"
fi
if ! python -c "import MultiMAP" 2>/dev/null; then
  pip install --no-deps \
    "git+https://github.com/Teichlab/MultiMAP.git@681e608c45cdb6b139dfb6700e40c7520bc6096d"
fi
python -c "import cobolt, MultiMAP; print('cobolt + MultiMAP OK')"
