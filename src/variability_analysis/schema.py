"""Common in-memory schema for native-rate ECG/ICG recordings.

The schema preserves original time axes and reference provenance. It performs
structural validation only; it does not claim physiological signal quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ReferenceSeries:
    """Reference values retained separately from future detected fiducials.

    Parameters
    ----------
    values:
        Original values without interpolation, imputation, or index conversion.
    unit:
        Unit stated by the source or ``unknown`` when not established.
    coordinate_system:
        Meaning of the values, such as ``time_s`` or
        ``matlab_sample_index_unresolved``.
    source:
        Stable provenance label identifying the modality/annotation stream.
    source_path:
        File from which the values were read.
    note:
        Scientific limitation or interpretation warning.
    """

    values: np.ndarray
    unit: str
    coordinate_system: str
    source: str
    source_path: Path
    note: str = ""

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values).copy()
        self.source_path = Path(self.source_path)


@dataclass
class Recording:
    """Dataset-independent ECG/ICG recording at its native sampling rate."""

    dataset_id: str
    dataset_name: str
    cohort: str
    condition: str
    paired_group: str
    subject_id: str
    source_subject_id: str
    recording_id: str
    source_path: Path
    time_s: np.ndarray
    ecg: np.ndarray
    icg: np.ndarray
    sampling_rate_hz: float
    sampling_rate_source: str
    timestamp_source: str
    icg_kind: str
    references: dict[str, ReferenceSeries] = field(default_factory=dict)
    extra_signals: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        self.time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
        self.ecg = np.asarray(self.ecg, dtype=float).reshape(-1)
        self.icg = np.asarray(self.icg, dtype=float).reshape(-1)
        self.extra_signals = {
            name: np.asarray(values, dtype=float).reshape(-1)
            for name, values in self.extra_signals.items()
        }
        self.warnings = tuple(dict.fromkeys(self.warnings))

    @property
    def sample_count(self) -> int:
        """Number of samples in the common ECG/ICG time base."""

        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        """Observed time span from the first through last timestamp."""

        return float(self.time_s[-1] - self.time_s[0]) if self.sample_count > 1 else 0.0

    @property
    def timestamp_sampling_rate_hz(self) -> float:
        """Median timestamp-derived sampling rate, or NaN when unavailable."""

        if self.sample_count < 2:
            return float("nan")
        differences = np.diff(self.time_s)
        return float(1.0 / np.median(differences))

    def validation_errors(self) -> list[str]:
        """Return structural errors without modifying or filtering the signals."""

        errors: list[str] = []
        n = self.sample_count
        if n < 2:
            errors.append("signal_has_fewer_than_two_samples")
        if self.ecg.size != n or self.icg.size != n:
            errors.append("ecg_icg_time_length_mismatch")
        if n and not np.all(np.isfinite(self.time_s)):
            errors.append("time_contains_nan_or_inf")
        if n > 1 and not np.all(np.diff(self.time_s) > 0):
            errors.append("time_not_strictly_increasing")
        if self.ecg.size and not np.all(np.isfinite(self.ecg)):
            errors.append("ecg_contains_nan_or_inf")
        if self.icg.size and not np.all(np.isfinite(self.icg)):
            errors.append("icg_contains_nan_or_inf")
        if not np.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0:
            errors.append("invalid_sampling_rate")
        for name, values in self.extra_signals.items():
            if values.size != n:
                errors.append(f"extra_signal_length_mismatch:{name}")
            elif not np.all(np.isfinite(values)):
                errors.append(f"extra_signal_contains_nan_or_inf:{name}")
        if n > 1 and np.isfinite(self.sampling_rate_hz) and self.sampling_rate_hz > 0:
            timestamp_rate = self.timestamp_sampling_rate_hz
            difference = abs(timestamp_rate - self.sampling_rate_hz) / self.sampling_rate_hz
            if difference > 0.01:
                errors.append("timestamp_sampling_rate_mismatch_over_1pct")
        return errors

