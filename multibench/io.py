"""Your data in and out: ``export_dataset`` writes a dataset folder the methods
read, ``read`` reads one of its ``.h5`` files (or a run's
``embedding.h5``) back as an AnnData."""
from .engine.ingest import export_dataset, read

__all__ = ["export_dataset", "read"]
