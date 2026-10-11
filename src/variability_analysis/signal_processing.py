"""Approved Phase 2 filtering, Pan-Tompkins R detection, and C detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, convolve, find_peaks, sosfiltfilt

from .phase2_config import Phase2Config
from .schema import Recording


@dataclass
class PanTompkinsResult:
    """Intermediate signals and final R indices from Pan-Tompkins."""

    filtered_ecg: np.ndarray
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray
    candidate_indices: np.ndarray
    r_indices: np.ndarray
    reason: str = ""


@dataclass
class ProcessedRecording:
    """Phase 2 signals at the recording's original sample positions."""

    recording: Recording
    filtered_ecg: np.ndarray
    integrated_ecg: np.ndarray
    filtered_icg: np.ndarray
    analysis_dzdt: np.ndarray
    r_indices: np.ndarray
    r_detection_reason: str


@dataclass(frozen=True)
class CDetection:
    """One legacy global-maximum C candidate and its QC status."""

    sample_index: int | None
    time_s: float | None
    rc_ms: float | None
    status: str
    reason: str


def zero_phase_bandpass(
    values: np.ndarray, sampling_rate_hz: float, low_hz: float,
    high_hz: float, order: int,
) -> np.ndarray:
    """Apply the approved SOS Butterworth zero-phase bandpass."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if not 0 < low_hz < high_hz < sampling_rate_hz / 2.0:
        raise ValueError(
            f"invalid bandpass {low_hz:g}-{high_hz:g} Hz at "
            f"fs={sampling_rate_hz:g} Hz"
        )
    sections = butter(
        order, [low_hz, high_hz], btype="bandpass",
        fs=sampling_rate_hz, output="sos",
    )
    return sosfiltfilt(sections, values)


def central_difference(values: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Differentiate without changing sample count or timestamps."""

    values = np.asarray(values, dtype=float).reshape(-1)
    result = np.empty_like(values)
    result[1:-1] = (values[2:] - values[:-2]) * sampling_rate_hz / 2.0
    result[0] = (values[1] - values[0]) * sampling_rate_hz
    result[-1] = (values[-1] - values[-2]) * sampling_rate_hz
    return result


def _initial_peak_levels(
    levels: np.ndarray, candidate_indices: np.ndarray, learning_samples: int,
) -> tuple[float, float]:
    learning = levels[candidate_indices[candidate_indices < learning_samples]]
    if learning.size == 0:
        learning = levels[candidate_indices]
    if learning.size == 0:
        return 0.0, 0.0
    signal_level = float(np.max(learning))
    remaining = learning[learning < signal_level]
    noise_level = float(np.mean(remaining)) if remaining.size else signal_level / 2.0
    return signal_level, noise_level


def _localized_filtered_peak(
    filtered_ecg: np.ndarray, candidate: int, radius: int,
) -> tuple[int, float]:
    start = max(0, candidate - radius)
    stop = min(filtered_ecg.size, candidate + radius + 1)
    local = np.abs(filtered_ecg[start:stop])
    if local.size == 0:
        return candidate, 0.0
    index = start + int(np.argmax(local))
    return index, float(abs(filtered_ecg[index]))


def pan_tompkins(
    ecg: np.ndarray, sampling_rate_hz: float,
    config: Phase2Config = Phase2Config(),
) -> PanTompkinsResult:
    """Detect R peaks using a native-rate offline Pan-Tompkins adaptation.

    The original 1985 processing sequence is retained: 5-15 Hz QRS
    bandpass, five-point derivative, squaring, 150 ms moving integration,
    adaptive signal/noise thresholds, a secondary half-threshold search-back,
    200 ms refractory period, and slope-based T-wave discrimination below
    360 ms. Filter coefficients are recomputed at the native sampling rate;
    no signal is resampled. Zero-phase filtering is an explicit offline
    adaptation and is recorded in the configuration.
    """

    ecg = np.asarray(ecg, dtype=float).reshape(-1)
    fs = float(sampling_rate_hz)
    filtered = zero_phase_bandpass(
        ecg, fs, config.r_bandpass_low_hz, config.r_bandpass_high_hz,
        config.r_bandpass_order,
    )
    derivative_kernel = np.array([1.0, 2.0, 0.0, -2.0, -1.0]) * fs / 8.0
    derivative = convolve(filtered, derivative_kernel, mode="same")
    squared = derivative**2
    integration_samples = max(1, int(round(config.r_integration_window_ms * fs / 1000.0)))
    integrated = convolve(
        squared, np.ones(integration_samples) / integration_samples, mode="same",
    )
    refractory = max(1, int(round(config.r_refractory_ms * fs / 1000.0)))
    candidates, _ = find_peaks(integrated, distance=refractory)
    if candidates.size == 0:
        return PanTompkinsResult(
            filtered, derivative, squared, integrated, candidates,
            np.array([], dtype=int), "no_integrated_candidates",
        )

    radius = max(1, int(round(config.r_localization_window_ms * fs / 1000.0)))
    localized = [_localized_filtered_peak(filtered, int(index), radius) for index in candidates]
    filtered_indices = np.array([item[0] for item in localized], dtype=int)
    filtered_levels = np.array([item[1] for item in localized], dtype=float)
    learning_samples = min(ecg.size, int(round(config.r_learning_window_s * fs)))
    signal_i, noise_i = _initial_peak_levels(integrated, candidates, learning_samples)
    signal_f, noise_f = _initial_peak_levels(filtered_levels, np.arange(candidates.size), candidates.searchsorted(learning_samples))
    alpha = config.r_adaptive_update_weight
    accepted: list[int] = []
    accepted_slopes: list[float] = []
    noise_candidates: list[tuple[int, int, float, float]] = []

    for position, (candidate, r_index, filtered_level) in enumerate(
        zip(candidates, filtered_indices, filtered_levels, strict=True)
    ):
        integrated_level = float(integrated[candidate])
        threshold_i = noise_i + config.r_threshold_fraction * (signal_i - noise_i)
        threshold_f = noise_f + config.r_threshold_fraction * (signal_f - noise_f)

        if len(accepted) >= 3:
            rr_mean = float(np.mean(np.diff(accepted[-9:])))
            if r_index - accepted[-1] > config.r_searchback_rr_factor * rr_mean:
                lower_i = config.r_secondary_threshold_fraction * threshold_i
                lower_f = config.r_secondary_threshold_fraction * threshold_f
                eligible = [
                    item for item in noise_candidates
                    if item[1] - accepted[-1] >= refractory
                    and item[2] >= lower_i and item[3] >= lower_f
                ]
                if eligible:
                    missed = max(eligible, key=lambda item: item[2])
                    accepted.append(missed[1])
                    accepted_slopes.append(
                        float(np.max(np.abs(derivative[max(0, missed[1] - radius):missed[1] + radius + 1])))
                    )
                    signal_i = 0.25 * missed[2] + 0.75 * signal_i
                    signal_f = 0.25 * missed[3] + 0.75 * signal_f

        is_signal = integrated_level >= threshold_i and filtered_level >= threshold_f
        if is_signal and accepted and r_index - accepted[-1] < refractory:
            is_signal = False
        slope_window = derivative[max(0, r_index - radius):min(ecg.size, r_index + radius + 1)]
        slope = float(np.max(np.abs(slope_window))) if slope_window.size else 0.0
        if (
            is_signal and accepted
            and (r_index - accepted[-1]) * 1000.0 / fs < config.r_t_wave_window_ms
            and accepted_slopes and slope < 0.5 * accepted_slopes[-1]
        ):
            is_signal = False

        if is_signal:
            accepted.append(int(r_index))
            accepted_slopes.append(slope)
            signal_i = alpha * integrated_level + (1.0 - alpha) * signal_i
            signal_f = alpha * filtered_level + (1.0 - alpha) * signal_f
        else:
            noise_i = alpha * integrated_level + (1.0 - alpha) * noise_i
            noise_f = alpha * filtered_level + (1.0 - alpha) * noise_f
            noise_candidates.append((position, int(r_index), integrated_level, filtered_level))

    r_indices = np.array(sorted(set(accepted)), dtype=int)
    reason = "" if r_indices.size >= 2 else "fewer_than_two_r_peaks"
    return PanTompkinsResult(
        filtered, derivative, squared, integrated, candidates, r_indices, reason,
    )


def preprocess_recording(
    recording: Recording, config: Phase2Config = Phase2Config(),
) -> ProcessedRecording:
    """Apply approved ECG/ICG preprocessing without resampling."""

    r_result = pan_tompkins(recording.ecg, recording.sampling_rate_hz, config)
    filtered_icg = zero_phase_bandpass(
        recording.icg, recording.sampling_rate_hz,
        config.icg_filter_low_hz, config.icg_filter_high_hz,
        config.icg_filter_order,
    )
    analysis_dzdt = (
        central_difference(filtered_icg, recording.sampling_rate_hz)
        if recording.icg_kind == "impedance" else filtered_icg.copy()
    )
    return ProcessedRecording(
        recording, r_result.filtered_ecg, r_result.integrated, filtered_icg,
        analysis_dzdt, r_result.r_indices, r_result.reason,
    )


def detect_c_point(
    processed: ProcessedRecording, r_index: int, next_r_index: int,
    config: Phase2Config = Phase2Config(),
) -> CDetection:
    """Apply the existing global-maximum C rule and retain invalid candidates."""

    fs = processed.recording.sampling_rate_hz
    start = int(r_index + round(config.c_search_start_ms * fs / 1000.0))
    stop = min(
        int(r_index + round(config.c_search_stop_ms * fs / 1000.0)),
        int(next_r_index) - 1,
        processed.analysis_dzdt.size - 1,
    )
    if stop < start:
        return CDetection(None, None, None, "INVALID", "empty_r_to_rplus300ms_window")
    window = processed.analysis_dzdt[start:stop + 1]
    if window.size == 0 or not np.any(np.isfinite(window)):
        return CDetection(None, None, None, "INVALID", "c_search_window_has_no_finite_values")
    c_index = start + int(np.nanargmax(window))
    time = processed.recording.time_s
    rc_ms = float((time[c_index] - time[r_index]) * 1000.0)
    status = "VALID" if config.c_valid_min_rc_ms <= rc_ms <= config.c_valid_max_rc_ms else "INVALID"
    reason = "" if status == "VALID" else "r_to_c_outside_60_200ms"
    return CDetection(c_index, float(time[c_index]), rc_ms, status, reason)

