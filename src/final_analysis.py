from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["dataset_id", "subject_id", "recording_id", "beat_id"]


def _valid(rows: pd.DataFrame) -> pd.Series:
    return rows["detection_status"].eq("DETECTED") & rows["qc_status"].isin(["PASS", "FLAGGED"])


def _category(values: pd.Series, participants: pd.Series, low: str = "low", high: str = "high") -> pd.Series:
    medians = values.groupby(participants).transform("median")
    result = pd.Series(np.where(values.le(medians), low, high), index=values.index, dtype="object")
    result.loc[values.isna()] = "unavailable"
    return result


def selected_pair(kind: str) -> tuple[str, str]:
    return ("B4", "B5") if kind == "B" else ("X5", "X6")


def build_kind_tables(master: pd.DataFrame, phase4: pd.DataFrame, registry: pd.DataFrame,
                      kind: str) -> dict[str, pd.DataFrame]:
    methods = registry.loc[registry.candidate_type.eq(kind), ["algorithm_id", "candidate_name"]].copy()
    method_names = methods.candidate_name.tolist()
    id_map = methods.set_index("candidate_name").algorithm_id.to_dict()
    event = "echo_avo_time_s" if kind == "B" else "echo_avc_time_s"
    source_status = "echo_avo_source_status" if kind == "B" else "echo_avc_source_status"
    alignment_status = "echo_avo_alignment_status" if kind == "B" else "echo_avc_alignment_status"
    eligibility = "b_reference_eligibility" if kind == "B" else "x_reference_eligibility"
    reference_name = "ECHO AVO" if kind == "B" else "ECHO AVC"

    master_columns = KEYS + [
        "condition", "beat_number", "source_file", "sampling_frequency_hz",
        "sampling_frequency_source", "r_peak_sample_index", "next_r_peak_sample_index",
        "r_peak_time_s", "next_r_peak_time_s", "rr_interval_ms", "c_sample_index",
        "c_point_time_s", "c_relative_to_r_ms", "echo_r_time_s", "echo_r_match_status",
        "echo_r_minus_detected_r_ms", event, source_status, alignment_status, eligibility,
    ]
    kind_master = master[master_columns].copy()
    kind_master = kind_master.rename(columns={event: "reference_event_time_s", source_status: "reference_source_status",
                                              alignment_status: "reference_alignment_status", eligibility: "reference_eligible"})
    kind_master["reference_event"] = reference_name
    kind_master["time_origin"] = "recording timestamp; relative event times use detected ECG R"
    kind_master["signal_units"] = "source impedance: unverified device units; derived dZ/dt: device units/s"

    detection_columns = KEYS + [
        "candidate_name", "candidate_type", "candidate_sample_index", "candidate_timestamp_s",
        "detection_status", "qc_status", "qc_issue_codes", "associated_region",
        "search_start_sample", "search_stop_sample_exclusive", "method_variant",
        "dependency_candidate_name", "heart_rate_bpm", "formula_adjustment_ms",
    ]
    detection = phase4.loc[phase4.candidate_type.eq(kind), detection_columns].copy()
    detection.insert(4, "algorithm_id", detection.candidate_name.map(id_map))
    detection = detection.merge(
        kind_master[KEYS + ["r_peak_time_s", "reference_event_time_s", "reference_source_status",
                            "reference_alignment_status", "reference_eligible"]], on=KEYS, how="left", validate="many_to_one"
    )
    detection["attempted"] = ~detection.detection_status.eq("INELIGIBLE")
    detection["valid"] = _valid(detection)
    detection["missed"] = detection.attempted & ~detection.detection_status.eq("DETECTED")
    detection["invalid"] = detection.detection_status.eq("DETECTED") & detection.qc_status.eq("INVALID")
    detection["physiologically_implausible"] = detection.invalid & detection.qc_issue_codes.fillna("").str.contains(
        "not_strictly|outside|physiolog", case=False
    )
    detection["analysis_status"] = np.select(
        [~detection.attempted, detection.valid, detection.missed, detection.invalid],
        ["INELIGIBLE", "VALID", "MISSED", "INVALID"], default="UNCLASSIFIED",
    )
    if detection.analysis_status.eq("UNCLASSIFIED").any():
        raise RuntimeError(f"Unclassified {kind} detection rows")
    detection["candidate_time_from_r_ms"] = 1000 * (detection.candidate_timestamp_s - detection.r_peak_time_s)
    detection["reference_available"] = detection.reference_event_time_s.notna()
    detection["failure_reason"] = np.where(
        detection.analysis_status.eq("MISSED"), detection.detection_status,
        np.where(detection.analysis_status.eq("INVALID"), detection.qc_issue_codes, ""),
    )
    detection["time_unit"] = "ms"
    detection["time_origin"] = "detected ECG R"
    status_columns = KEYS + [
        "algorithm_id", "candidate_name", "candidate_type", "candidate_sample_index",
        "candidate_timestamp_s", "candidate_time_from_r_ms", "time_unit", "time_origin",
        "analysis_status", "detection_status", "qc_status", "failure_reason", "attempted",
        "valid", "missed", "invalid", "physiologically_implausible", "reference_available",
        "reference_eligible", "reference_source_status", "reference_alignment_status",
        "associated_region", "search_start_sample", "search_stop_sample_exclusive",
        "method_variant", "dependency_candidate_name", "heart_rate_bpm", "formula_adjustment_ms",
    ]
    detection = detection[status_columns].sort_values(KEYS + ["algorithm_id"]).reset_index(drop=True)
    valid_detections = detection.loc[detection.analysis_status.eq("VALID")].reset_index(drop=True)
    failed_detections = detection.loc[detection.analysis_status.isin(["MISSED", "INVALID"])].reset_index(drop=True)

    reference = kind_master[KEYS + [
        "condition", "r_peak_time_s", "next_r_peak_time_s", "echo_r_time_s", "echo_r_match_status",
        "echo_r_minus_detected_r_ms", "reference_event", "reference_event_time_s",
        "reference_source_status", "reference_alignment_status", "reference_eligible",
    ]].copy()
    reference["reference_time_from_detected_r_ms"] = 1000 * (reference.reference_event_time_s - reference.r_peak_time_s)
    reference["eligibility_definition"] = "matched source event strictly within detected R-to-next-R cycle"

    valid_wide = detection.pivot(index=KEYS, columns="algorithm_id", values="valid")
    complete_keys = valid_wide.index[valid_wide.all(axis=1)].to_frame(index=False)
    common_keys = complete_keys.merge(reference.loc[reference.reference_eligible, KEYS], on=KEYS, how="inner")
    common_keys["strict_common_reference"] = True
    errors = valid_detections.merge(common_keys, on=KEYS, how="inner", validate="many_to_one")
    errors = errors.merge(reference[KEYS + ["reference_event", "reference_event_time_s"]], on=KEYS, how="left", validate="many_to_one")
    errors["signed_error_ms"] = 1000 * (errors.candidate_timestamp_s - errors.reference_event_time_s)
    errors["absolute_error_ms"] = errors.signed_error_ms.abs()
    errors["comparison_set"] = "strict all-method common reference beats"
    error_columns = KEYS + [
        "algorithm_id", "candidate_name", "candidate_type", "candidate_timestamp_s",
        "candidate_time_from_r_ms", "reference_event", "reference_event_time_s",
        "signed_error_ms", "absolute_error_ms", "comparison_set",
    ]
    errors = errors[error_columns].sort_values(KEYS + ["algorithm_id"]).reset_index(drop=True)

    participant_detection = detection.loc[detection.attempted].groupby(
        ["subject_id", "algorithm_id", "candidate_name", "candidate_type"], observed=True
    ).agg(
        attempted_beats=("beat_id", "size"), valid_beats=("valid", "sum"),
        missed_beats=("missed", "sum"), invalid_beats=("invalid", "sum"),
        physiologically_implausible_beats=("physiologically_implausible", "sum"),
        recordings=("recording_id", "nunique"),
    ).reset_index()
    denominator = participant_detection.attempted_beats.replace(0, np.nan)
    participant_detection["success_rate_pct"] = 100 * participant_detection.valid_beats / denominator
    participant_detection["missed_rate_pct"] = 100 * participant_detection.missed_beats / denominator
    participant_detection["invalid_rate_pct"] = 100 * participant_detection.invalid_beats / denominator
    participant_detection["total_failure_rate_pct"] = 100 * (
        participant_detection.missed_beats + participant_detection.invalid_beats
    ) / denominator

    participant_accuracy = errors.groupby(
        ["subject_id", "algorithm_id", "candidate_name", "candidate_type"], observed=True
    ).agg(
        common_reference_beats=("beat_id", "size"),
        median_signed_error_ms=("signed_error_ms", "median"),
        median_absolute_error_ms=("absolute_error_ms", "median"),
        p90_absolute_error_ms=("absolute_error_ms", lambda x: x.quantile(.90)),
        p95_absolute_error_ms=("absolute_error_ms", lambda x: x.quantile(.95)),
    ).reset_index()
    participant = participant_detection.merge(
        participant_accuracy, on=["subject_id", "algorithm_id", "candidate_name", "candidate_type"], how="left", validate="one_to_one"
    )

    complete_detection = detection.merge(complete_keys, on=KEYS, how="inner")
    timing = complete_detection.pivot(index=KEYS, columns="algorithm_id", values="candidate_time_from_r_ms")
    ranges = pd.DataFrame({"within_beat_range_ms": timing.max(axis=1) - timing.min(axis=1)}).reset_index()
    range_participant = ranges.groupby("subject_id").within_beat_range_ms.agg(
        disagreement_beats="size", participant_median_range_ms="median",
        participant_p95_range_ms=lambda x: x.quantile(.95),
    ).reset_index()
    pair_a, pair_b = selected_pair(kind)
    pair = timing[[pair_a, pair_b]].dropna().copy()
    pair["paired_difference_ms"] = pair[pair_a] - pair[pair_b]
    pair_participant = pair.reset_index().groupby("subject_id").paired_difference_ms.agg(
        selected_pair_beats="size", participant_median_pair_difference_ms="median",
        participant_mean_pair_difference_ms="mean",
    ).reset_index()
    participant = participant.merge(range_participant, on="subject_id", how="left", validate="many_to_one")
    participant = participant.merge(pair_participant, on="subject_id", how="left", validate="many_to_one")
    participant["selected_pair_algorithm_id_a"] = pair_a
    participant["selected_pair_algorithm_id_b"] = pair_b
    participant["denominator_definition"] = "detection rates use attempted beats; accuracy uses strict all-method common reference beats"
    participant = participant.sort_values(["subject_id", "algorithm_id"]).reset_index(drop=True)

    features = detection.merge(
        master[KEYS + [
            "signal_to_noise_ratio_db", "beat_to_beat_shape_nrmse",
            f"{kind.lower()}_competing_features_count",
            f"{kind.lower()}_reference_local_sharpness_abs_d2",
            f"{kind.lower()}_reference_local_slope_abs_d1",
            f"{kind.lower()}_reference_local_amplitude_from_region_median",
        ]], on=KEYS, how="left", validate="many_to_one"
    )
    features = features.merge(reference[KEYS + ["reference_event", "reference_event_time_s"]], on=KEYS, how="left", validate="many_to_one")
    features = features.merge(common_keys, on=KEYS, how="left")
    features["strict_common_reference"] = features.strict_common_reference.fillna(False)
    features["signed_error_ms"] = np.where(
        features.strict_common_reference & features.valid,
        1000 * (features.candidate_timestamp_s - features.reference_event_time_s), np.nan,
    )
    features["absolute_error_ms"] = features.signed_error_ms.abs()
    features["candidate_minus_reference_ms"] = np.where(
        features.reference_eligible & features.valid,
        1000 * (features.candidate_timestamp_s - features.reference_event_time_s), np.nan,
    )
    rename = {
        f"{kind.lower()}_competing_features_count": "reference_region_competing_features_count",
        f"{kind.lower()}_reference_local_sharpness_abs_d2": "reference_local_sharpness_abs_d2_device_units_s3",
        f"{kind.lower()}_reference_local_slope_abs_d1": "reference_local_slope_abs_d1_device_units_s2",
        f"{kind.lower()}_reference_local_amplitude_from_region_median": "reference_local_amplitude_device_units_s",
    }
    features = features.rename(columns=rename)
    features["candidate_local_sharpness_abs_d2_device_units_s3"] = np.nan
    features["candidate_local_slope_abs_d1_device_units_s2"] = np.nan
    features["candidate_local_amplitude_device_units_s"] = np.nan
    for method in method_names:
        mask = features.candidate_name.eq(method)
        features.loc[mask, "candidate_local_sharpness_abs_d2_device_units_s3"] = master.set_index(KEYS).loc[
            pd.MultiIndex.from_frame(features.loc[mask, KEYS]), f"{method}__local_sharpness_abs_d2"
        ].to_numpy()
        features.loc[mask, "candidate_local_slope_abs_d1_device_units_s2"] = master.set_index(KEYS).loc[
            pd.MultiIndex.from_frame(features.loc[mask, KEYS]), f"{method}__local_slope_abs_d1"
        ].to_numpy()
        features.loc[mask, "candidate_local_amplitude_device_units_s"] = master.set_index(KEYS).loc[
            pd.MultiIndex.from_frame(features.loc[mask, KEYS]), f"{method}__local_amplitude_from_region_median"
        ].to_numpy()
    category_specs = [
        ("reference_local_sharpness_abs_d2_device_units_s3", "sharpness", "low", "high"),
        ("reference_local_slope_abs_d1_device_units_s2", "slope", "low", "high"),
        ("reference_local_amplitude_device_units_s", "amplitude", "low", "high"),
        ("signal_to_noise_ratio_db", "snr", "low", "high"),
        ("reference_region_competing_features_count", "competing_features", "low", "high"),
        ("beat_to_beat_shape_nrmse", "beat_to_beat_variability", "consistent", "variable"),
    ]
    for column, label, low, high in category_specs:
        features[f"{label}_category"] = _category(features[column], features.subject_id, low, high)
    morphology_columns = KEYS + [
        "algorithm_id", "candidate_name", "candidate_type", "analysis_status", "valid", "missed", "invalid",
        "reference_event", "reference_eligible", "strict_common_reference", "candidate_time_from_r_ms",
        "candidate_minus_reference_ms", "signed_error_ms", "absolute_error_ms",
        "reference_local_sharpness_abs_d2_device_units_s3", "reference_local_slope_abs_d1_device_units_s2",
        "reference_local_amplitude_device_units_s", "signal_to_noise_ratio_db",
        "reference_region_competing_features_count", "beat_to_beat_shape_nrmse",
        "candidate_local_sharpness_abs_d2_device_units_s3", "candidate_local_slope_abs_d1_device_units_s2",
        "candidate_local_amplitude_device_units_s", "sharpness_category", "slope_category",
        "amplitude_category", "snr_category", "competing_features_category", "beat_to_beat_variability_category",
    ]
    morphology = features[morphology_columns].sort_values(KEYS + ["algorithm_id"]).reset_index(drop=True)

    return {
        f"{kind}_master_beats.csv": kind_master,
        f"{kind}_detection_status.csv": detection,
        f"{kind}_valid_detections.csv": valid_detections,
        f"{kind}_failed_detections.csv": failed_detections,
        f"{kind}_reference_eligibility.csv": reference,
        f"{kind}_reference_errors.csv": errors,
        f"{kind}_participant_summary.csv": participant,
        f"{kind}_morphology_features.csv": morphology,
    }


def participant_balanced_limits(pair: pd.DataFrame, participant: str = "subject_id",
                                difference: str = "difference_ms") -> tuple[float, float, float]:
    groups = pair.groupby(participant)[difference]
    means = groups.mean()
    variances = groups.var(ddof=1).fillna(0)
    bias = float(means.mean())
    total_variance = float(((means - bias) ** 2 + variances).mean())
    spread = np.sqrt(total_variance)
    return bias, bias - 1.96 * spread, bias + 1.96 * spread
