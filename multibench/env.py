"""The method environments: ``install`` downloads them (Linux only);
``install(dry_run=True)`` shows each one's state and size."""
from .engine.envs import install

__all__ = ["install"]
