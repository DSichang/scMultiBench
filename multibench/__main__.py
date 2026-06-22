"""Enable ``python -m multibench`` (equivalent to the ``multibench`` console script)."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
