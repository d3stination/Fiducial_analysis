"""ECG/ICG variability research pipeline."""

from .manifest import DatasetSpec, dataset_manifest
from .schema import Recording, ReferenceSeries

__all__ = ["DatasetSpec", "Recording", "ReferenceSeries", "dataset_manifest"]
__version__ = "0.1.0"

