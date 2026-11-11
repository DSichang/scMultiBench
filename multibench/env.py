"""The method environments: ``install`` downloads them (on Linux; on Windows
for the methods with a Windows environment); ``install(dry_run=True)`` shows
each one's state and size."""
from .engine.envs import install

__all__ = ["install"]
