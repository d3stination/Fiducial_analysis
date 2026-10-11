from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from .data_loading import load_config


def _legacy_api(config: dict):
    root = Path(config["sources"]["variability_root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import src as active_src
    legacy_src = str(root / "src")
    if legacy_src not in active_src.__path__:
        active_src.__path__.append(legacy_src)
    from src.variability_analysis.loaders import load_recording
    from src.variability_analysis.manifest import dataset_manifest
    from src.variability_analysis.signal_processing import central_difference, preprocess_recording
    spec = next(item for item in dataset_manifest(root) if item.dataset_id == config["project"]["dataset_id"])
    return root, spec, load_recording, preprocess_recording, central_difference


def _standardized_shape(signal: np.ndarray, points: int) -> np.ndarray:
    finite = np.isfinite(signal)
    if finite.sum() < 4:
        return np.full(points, np.nan)
    x = np.linspace(0, 1, signal.size)[finite]
    values = signal[finite]
    scale = values.std()
    if scale <= 0:
        return np.full(points, np.nan)
    values = (values - values.mean()) / scale
    return np.interp(np.linspace(0, 1, points), x, values)


def extract_morphology(phase3: pd.DataFrame, phase4: pd.DataFrame, config: dict | None = None, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extract only the six supervisor-specified morphology/quality factors."""
    config = config or load_config()
    root, spec, load_recording, preprocess_recording, central_difference = _legacy_api(config)
    points = int(config["analysis"]["waveform_resample_points"])
    valid = phase4["detection_status"].eq("DETECTED") & phase4["qc_status"].isin(["PASS", "FLAGGED"])
    candidates = phase4.loc[valid].copy()
    reference_index = None if reference is None else reference.set_index(["dataset_id", "subject_id", "recording_id", "beat_id"])
    rows = []
    for recording_id, beats in phase3.groupby("recording_id", sort=False):
        source = Path(beats["source_file"].iloc[0])
        recording = load_recording(spec, source, root)
        processed = preprocess_recording(recording)
        raw_dzdt = central_difference(recording.icg, recording.sampling_rate_hz)
        dzdt = processed.analysis_dzdt
        d1 = central_difference(dzdt, recording.sampling_rate_hz)
        d2 = central_difference(d1, recording.sampling_rate_hz)
        previous_shape = None
        rec_candidates = candidates.loc[candidates["recording_id"].eq(recording_id)]
        for beat in beats.sort_values("beat_number").itertuples(index=False):
            start, stop, c = int(beat.beat_start_r_sample_inclusive), int(beat.beat_end_next_r_sample_exclusive), int(beat.c_sample)
            segment = dzdt[start:stop]
            shape = _standardized_shape(segment, points)
            shape_nrmse = float(np.sqrt(np.nanmean((shape - previous_shape) ** 2))) if previous_shape is not None and np.isfinite(shape).all() and np.isfinite(previous_shape).all() else np.nan
            previous_shape = shape
            window = max(5, int(round(.051 * recording.sampling_rate_hz)) | 1)
            if window >= segment.size:
                window = max(5, (segment.size - 1) | 1)
            smooth = savgol_filter(segment, window, 3) if segment.size >= 7 and window < segment.size else segment.copy()
            residual = segment - smooth
            noise_rms = float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan
            signal_rms = float(np.sqrt(np.mean((smooth - smooth.mean()) ** 2))) if smooth.size else np.nan
            snr_db = float(20 * np.log10(signal_rms / noise_rms)) if noise_rms > 0 and signal_rms > 0 else np.nan
            row = {
                "dataset_id": beat.dataset_id, "subject_id": beat.subject_id, "recording_id": recording_id, "beat_id": beat.beat_id,
                "raw_dzdt_rms": float(np.sqrt(np.mean(raw_dzdt[start:stop] ** 2))),
                "signal_to_noise_ratio_db": snr_db,
                "beat_to_beat_shape_nrmse": shape_nrmse,
            }
            for kind, left, right in (("B", start, c + 1), ("X", c, stop)):
                region = dzdt[left:right]
                turning = len(find_peaks(region)[0]) + len(find_peaks(-region)[0]) if region.size >= 3 else 0
                inflections = int(np.count_nonzero(np.diff(np.signbit(d1[left:right])))) if region.size >= 2 else 0
                row[f"{kind.lower()}_competing_features_count"] = turning + inflections
            if reference_index is not None:
                ref = reference_index.loc[(beat.dataset_id, beat.subject_id, recording_id, beat.beat_id)]
                for kind, event, eligible, left, right in (
                    ("B", "echo_avo_time_s", "b_reference_eligibility", start, c + 1),
                    ("X", "echo_avc_time_s", "x_reference_eligibility", c, stop),
                ):
                    if bool(ref[eligible]) and np.isfinite(ref[event]):
                        index = int(np.argmin(np.abs(recording.time_s - float(ref[event]))))
                        baseline = float(np.median(dzdt[left:right]))
                        row[f"{kind.lower()}_reference_local_sharpness_abs_d2"] = abs(float(d2[index]))
                        row[f"{kind.lower()}_reference_local_slope_abs_d1"] = abs(float(d1[index]))
                        row[f"{kind.lower()}_reference_local_amplitude_from_region_median"] = abs(float(dzdt[index] - baseline))
            beat_candidates = rec_candidates.loc[rec_candidates["beat_id"].eq(beat.beat_id)]
            for candidate in beat_candidates.itertuples(index=False):
                index = int(candidate.candidate_sample_index)
                region_left, region_right = (start, c + 1) if candidate.candidate_type == "B" else (c, stop)
                baseline = float(np.median(dzdt[region_left:region_right]))
                prefix = candidate.candidate_name
                row[f"{prefix}__local_sharpness_abs_d2"] = abs(float(d2[index]))
                row[f"{prefix}__local_slope_abs_d1"] = abs(float(d1[index]))
                row[f"{prefix}__local_amplitude_from_region_median"] = abs(float(dzdt[index] - baseline))
            rows.append(row)
    return pd.DataFrame(rows)


def morphology_categories(master: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Predeclared within-participant median strata, computed without outcome access."""
    rows = []
    common_factors = ["signal_to_noise_ratio_db", "beat_to_beat_shape_nrmse"]
    for definition in registry.itertuples(index=False):
        kind = definition.candidate_type.lower()
        factors = {
            "sharpness": f"{kind}_reference_local_sharpness_abs_d2",
            "local_slope": f"{kind}_reference_local_slope_abs_d1",
            "amplitude": f"{kind}_reference_local_amplitude_from_region_median",
            "snr": common_factors[0], "beat_to_beat_variability": common_factors[1],
            "competing_features": f"{kind}_competing_features_count",
        }
        for factor, column in factors.items():
            data = master[["dataset_id", "subject_id", "recording_id", "beat_id", column]].copy()
            data = data.rename(columns={column: "factor_value"})
            medians = data.groupby("subject_id")["factor_value"].transform("median")
            low, high = ("low", "high") if factor != "beat_to_beat_variability" else ("consistent", "variable")
            data["category"] = np.where(data["factor_value"].le(medians), low, high)
            data.loc[data["factor_value"].isna(), "category"] = "unavailable"
            data["candidate_name"], data["candidate_type"], data["factor"] = definition.candidate_name, definition.candidate_type, factor
            rows.append(data)
    return pd.concat(rows, ignore_index=True)

