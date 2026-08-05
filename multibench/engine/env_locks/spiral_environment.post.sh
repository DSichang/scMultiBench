#!/usr/bin/env bash
# SPIRAL is not on PyPI. The name `spiral` there is an UNRELATED project, so a
# `spiral==1.0` pin resolves to the wrong thing at best. The working env has it
# as a setup.py egg built from the copy shipped in this repo.
set -euo pipefail
python -c "import spiral" 2>/dev/null && { echo "spiral already present"; exit 0; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
src="$repo/multibench_codes/SPIRAL_latest"
if [ ! -d "$src" ]; then
  echo "SPIRAL source not found at $src" >&2
  echo "It ships in multibench_codes/, which is not tracked by git - obtain it" >&2
  echo "from the upstream SPIRAL repository and re-run this script." >&2
  exit 1
fi
pip install --no-deps "$src"
python -c "import spiral; print('spiral OK')"
