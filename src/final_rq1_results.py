"""Build the audited numerical inputs for the final RQ1 figures and report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_loading import ROOT


KEYS = ["dataset_id", "subject_id", "recording_id", "beat_id"]
DATASETS = {
    "HeartCycle": {"folder": None, "conditions": ["overall"]},
    "IITB": {"folder": "IITB", "conditions": ["normal", "post_exercise"]},
    "ReBeat": {"folder": "ReBeat", "conditions": ["baseline", "cognitive_workload"]},
}
FEATURES = {
    "sharpness": "Peak/turning-point sharpness",
    "slope": "Local waveform slope",
    "amplitude": "Amplitude/prominence",
    "snr": "Signal quality/SNR",
    "competing": "Competing extrema/inflections",
    "variability": "Beat-to-beat morphology variability",
    "rr_interval_ms": "RR interval context",
}
PAIR = {"B": ("B4", "B5"), "X": ("X5", "X6")}
RNG = np.random.default_rng(20261009)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def methods(kind: str) -> list[str]:
    registry = read_csv(ROOT / "configs" / "canonical_method_registry.csv")
    return registry.loc[registry.candidate_type.eq(kind), "algorithm_id"].tolist()


def table_path(dataset: str, kind: str, name: str) -> Path:
    folder = DATASETS[dataset]["folder"]
    base = ROOT / "results" / "tables"
    if folder:
        base = base / folder
    return base / kind / f"{kind}_{name}.csv"


def load_status(dataset: str, kind: str) -> pd.DataFrame:
    status = read_csv(table_path(dataset, kind, "detection_status"))
    if dataset == "HeartCycle" or "condition" not in status:
        status["condition"] = "overall"
    for col in ["attempted", "valid", "missed", "invalid"]:
        status[col] = status[col].fillna(False).astype(bool)
    return status


def load_master(dataset: str, kind: str) -> pd.DataFrame:
    master = read_csv(table_path(dataset, kind, "master_beats"))
    if dataset == "HeartCycle" or "condition" not in master:
        master["condition"] = "overall"
    return master


def median_ci(values: pd.Series | np.ndarray, replicates: int = 4000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    estimate = float(np.median(values))
    if len(values) < 3:
        return estimate, np.nan, np.nan
    bootstrap = np.median(RNG.choice(values, size=(replicates, len(values)), replace=True), axis=1)
    return estimate, float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def build_rebeat_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = read_csv(ROOT / "work" / "final_rq1" / "rebeat_annotation_audit_rows.csv")
    audit = audit.loc[audit.one_based_match.astype(bool)].copy()
    audit["expert_b_timestamp_s"] = audit.b_sample_one_based_conversion / audit.sampling_frequency_hz
    audit["expert_x_timestamp_s"] = audit.x_sample_one_based_conversion / audit.sampling_frequency_hz
    mapping_columns = [
        "dataset_id", "condition", "recording_id", "annotation_file", "annotation_row",
        "sampling_frequency_hz", "beat_id_one_based_conversion", "b_sample_one_based_conversion",
        "x_sample_one_based_conversion", "expert_b_timestamp_s", "expert_x_timestamp_s",
        "b_less_than_x", "b_less_than_c_less_than_x",
    ]
    mapping = audit[mapping_columns].rename(columns={"beat_id_one_based_conversion": "beat_id"})
    mapping["subject_id"] = mapping.beat_id.str.split("|").str[1].map(
        lambda value: "rebeat:" + value.split("RawData_")[-1].split("_task_")[0]
        if value.startswith("RawData_") else value
    )
    # The beat identifier already carries the recording; authoritative subject IDs come from status joins.
    mapping = mapping.drop(columns="subject_id")

    all_errors, all_coverage = [], []
    for kind in ["B", "X"]:
        status = load_status("ReBeat", kind)
        ref_col = "expert_b_timestamp_s" if kind == "B" else "expert_x_timestamp_s"
        reference = mapping[["dataset_id", "condition", "recording_id", "beat_id", ref_col]].copy()
        joined = status.merge(reference, on=["dataset_id", "condition", "recording_id", "beat_id"], how="inner", validate="many_to_one")
        joined["reference_event"] = f"ReBeat expert {kind} annotation"
        joined["reference_event_time_s"] = joined[ref_col]
        joined["reference_matched"] = True
        common = joined.groupby("beat_id").valid.transform("all") & joined.groupby("beat_id").algorithm_id.transform("nunique").eq(len(methods(kind)))
        joined["strict_common_reference"] = common
        coverage = joined[[*KEYS, "condition", "algorithm_id", "candidate_name", "attempted", "valid", "missed", "invalid", "strict_common_reference"]].copy()
        coverage["candidate_type"] = kind
        coverage["reference_event"] = f"ReBeat expert {kind} annotation"
        all_coverage.append(coverage)

        error = joined.loc[joined.strict_common_reference & joined.valid].copy()
        error["signed_error_ms"] = 1000 * (error.candidate_timestamp_s - error.reference_event_time_s)
        error["absolute_error_ms"] = error.signed_error_ms.abs()
        error["candidate_time_from_r_ms"] = pd.to_numeric(error.candidate_time_from_r_ms, errors="coerce")
        error["comparison_set"] = "validated one-based expert annotation; strict all-method common beats"
        error["coordinate_sensitivity_signed_error_zero_based_ms"] = error.signed_error_ms - 4.0
        columns = [
            *KEYS, "condition", "algorithm_id", "candidate_name", "candidate_type",
            "candidate_timestamp_s", "candidate_time_from_r_ms", "reference_event",
            "reference_event_time_s", "signed_error_ms", "absolute_error_ms", "comparison_set",
            "coordinate_sensitivity_signed_error_zero_based_ms",
        ]
        error = error[columns].sort_values(["condition", "subject_id", "recording_id", "beat_id", "algorithm_id"])
        error.to_csv(table_path("ReBeat", kind, "expert_reference_errors"), index=False)
        all_errors.append(error)

    coverage = pd.concat(all_coverage, ignore_index=True)
    coverage.to_csv(ROOT / "results" / "tables" / "ReBeat" / "ReBeat_expert_reference_coverage.csv", index=False)
    errors = pd.concat(all_errors, ignore_index=True)
    participant = errors.groupby(["candidate_type", "condition", "subject_id", "algorithm_id", "candidate_name"], as_index=False).agg(
        common_reference_beats=("beat_id", "nunique"),
        median_signed_error_ms=("signed_error_ms", "median"),
        median_absolute_error_ms=("absolute_error_ms", "median"),
        p90_absolute_error_ms=("absolute_error_ms", lambda values: values.quantile(0.90)),
        p95_absolute_error_ms=("absolute_error_ms", lambda values: values.quantile(0.95)),
    )
    participant.to_csv(ROOT / "results" / "tables" / "ReBeat" / "ReBeat_expert_participant_accuracy.csv", index=False)
    mapping.to_csv(ROOT / "results" / "tables" / "ReBeat" / "ReBeat_expert_annotation_mapping.csv", index=False)
    return errors, participant, coverage


def heartcycle_errors(kind: str) -> pd.DataFrame:
    errors = read_csv(table_path("HeartCycle", kind, "reference_errors"))
    errors["condition"] = "overall"
    return errors


def definition_disagreement(dataset: str, kind: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status = load_status(dataset, kind)
    index = [*KEYS, "condition"]
    wide = status.loc[status.valid].pivot(index=index, columns="algorithm_id", values="candidate_time_from_r_ms")
    wide = wide.reindex(columns=methods(kind)).dropna()
    ranges = (wide.max(axis=1) - wide.min(axis=1)).rename("timing_range_ms").reset_index()
    frames = []
    for method in methods(kind):
        difference = (wide[method] - wide.drop(columns=method).median(axis=1)).abs()
        frame = difference.rename("absolute_difference_from_other_method_median_ms").reset_index()
        frame["algorithm_id"] = method
        frames.append(frame)
    peer = pd.concat(frames, ignore_index=True)
    peer = peer.merge(status[[*KEYS, "algorithm_id", "candidate_name"]], on=[*KEYS, "algorithm_id"], validate="one_to_one")
    participant = peer.groupby(["subject_id", "condition", "algorithm_id", "candidate_name"], as_index=False).agg(
        common_valid_beats=("beat_id", "nunique"),
        participant_median_peer_difference_ms=("absolute_difference_from_other_method_median_ms", "median"),
        participant_p95_peer_difference_ms=("absolute_difference_from_other_method_median_ms", lambda values: values.quantile(0.95)),
    )
    range_participant = ranges.groupby(["subject_id", "condition"], as_index=False).agg(
        common_valid_beats=("beat_id", "nunique"),
        participant_median_range_ms=("timing_range_ms", "median"),
        participant_p95_range_ms=("timing_range_ms", lambda values: values.quantile(0.95)),
    )
    peer.to_csv(table_path(dataset, kind, "definition_disagreement"), index=False)
    participant.to_csv(table_path(dataset, kind, "definition_participant_summary"), index=False)
    ranges.to_csv(table_path(dataset, kind, "common_beat_ranges"), index=False)
    range_participant.to_csv(table_path(dataset, kind, "range_participant_summary"), index=False)
    return peer, participant, range_participant


def paired_changes(dataset: str, kind: str, peer_participant: pd.DataFrame, range_participant: pd.DataFrame) -> pd.DataFrame:
    conditions = DATASETS[dataset]["conditions"]
    if len(conditions) != 2:
        return pd.DataFrame()
    status = load_status(dataset, kind)
    failure = status.loc[status.attempted].groupby(["subject_id", "condition", "algorithm_id"], as_index=False).agg(
        attempted=("beat_id", "size"), missed=("missed", "sum"), invalid=("invalid", "sum")
    )
    failure["value"] = 100 * (failure.missed + failure.invalid) / failure.attempted
    outputs = []
    for outcome, frame, value, extra, unit in [
        ("failure_rate", failure, "value", ["algorithm_id"], "percentage points"),
        ("within_beat_range", range_participant, "participant_median_range_ms", [], "ms"),
        ("definition_disagreement", peer_participant, "participant_median_peer_difference_ms", ["algorithm_id"], "ms"),
    ]:
        wide = frame.pivot(index=["subject_id", *extra], columns="condition", values=value).reindex(columns=conditions).dropna().reset_index()
        wide["normal_condition"] = conditions[0]
        wide["stress_condition"] = conditions[1]
        wide["normal_value"] = wide[conditions[0]]
        wide["stress_value"] = wide[conditions[1]]
        wide["stress_minus_normal"] = wide[conditions[1]] - wide[conditions[0]]
        wide["outcome"] = outcome
        wide["unit"] = unit
        wide["candidate_type"] = kind
        outputs.append(wide[["candidate_type", "outcome", "subject_id", *extra, "normal_condition", "stress_condition", "normal_value", "stress_value", "stress_minus_normal", "unit"]])
    return pd.concat(outputs, ignore_index=True)


def reference_inputs(rebeat_errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = []
    for kind in ["B", "X"]:
        heartcycle = heartcycle_errors(kind)
        heartcycle["dataset"] = "HeartCycle"
        heartcycle["reference_type"] = "ECHO AVO" if kind == "B" else "ECHO AVC"
        errors.append(heartcycle)
    rb = rebeat_errors.copy()
    rb["dataset"] = "ReBeat"
    rb["reference_type"] = rb.candidate_type.map({"B": "Expert ICG B annotation", "X": "Expert ICG X annotation"})
    errors.append(rb)
    combined = pd.concat(errors, ignore_index=True, sort=False)
    participant = combined.groupby(["dataset", "candidate_type", "condition", "reference_type", "subject_id", "algorithm_id", "candidate_name"], as_index=False).agg(
        common_reference_beats=("beat_id", "nunique"),
        median_signed_error_ms=("signed_error_ms", "median"),
        median_absolute_error_ms=("absolute_error_ms", "median"),
        p90_absolute_error_ms=("absolute_error_ms", lambda values: values.quantile(0.90)),
        p95_absolute_error_ms=("absolute_error_ms", lambda values: values.quantile(0.95)),
    )
    combined.to_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv", index=False)
    participant.to_csv(ROOT / "results" / "tables" / "RQ1_reference_participant_accuracy.csv", index=False)
    return combined, participant


def morphology_base(dataset: str, kind: str, reference_errors: pd.DataFrame, range_table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    status = load_status(dataset, kind)
    failure_rates = status.loc[status.attempted].groupby("algorithm_id").agg(n=("beat_id", "size"), failed=("valid", lambda values: (~values).sum()))
    failure_rates["rate"] = failure_rates.failed / failure_rates.n
    selected_method = failure_rates.sort_values(["rate"], ascending=False).index[0]
    selected_status = status.loc[status.algorithm_id.eq(selected_method), [*KEYS, "condition", "attempted", "valid", "missed", "invalid"]].copy()
    selected_status["failure"] = ((selected_status.missed | selected_status.invalid) & selected_status.attempted).astype(float)

    morphology = read_csv(table_path(dataset, kind, "morphology_features"))
    if "condition" not in morphology:
        morphology["condition"] = "overall"
    if dataset == "HeartCycle":
        columns = {
            "reference_local_sharpness_abs_d2_device_units_s3": "sharpness",
            "reference_local_slope_abs_d1_device_units_s2": "slope",
            "reference_local_amplitude_device_units_s": "amplitude",
            "signal_to_noise_ratio_db": "snr",
            "reference_region_competing_features_count": "competing",
            "beat_to_beat_shape_nrmse": "variability",
        }
        keep = [*KEYS, "condition", *columns]
        morphology = morphology[keep].drop_duplicates(KEYS).rename(columns=columns)
        master = load_master(dataset, kind)[[*KEYS, "rr_interval_ms"]]
        morphology = morphology.merge(master, on=KEYS, how="left", validate="one_to_one")
    else:
        keep = [*KEYS, "condition", "sharpness", "slope", "amplitude", "snr", "competing", "variability", "rr_interval_ms"]
        morphology = morphology[keep].drop_duplicates(KEYS)
    morphology = morphology.merge(range_table[[*KEYS, "condition", "timing_range_ms"]], on=[*KEYS, "condition"], how="left", validate="one_to_one")
    morphology = morphology.merge(selected_status, on=[*KEYS, "condition"], how="left", validate="one_to_one")
    errors = reference_errors.loc[(reference_errors.dataset.eq(dataset)) & reference_errors.candidate_type.eq(kind)]
    aggregate_error = errors.groupby([*KEYS, "condition"], as_index=False).absolute_error_ms.median().rename(columns={"absolute_error_ms": "median_algorithm_absolute_reference_error_ms"})
    morphology = morphology.merge(aggregate_error, on=[*KEYS, "condition"], how="left", validate="one_to_one")
    morphology["selected_failure_algorithm"] = selected_method
    return morphology, selected_method


def morphology_evidence(reference_errors: pd.DataFrame, ranges: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence, values = [], []
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            base, selected_method = morphology_base(dataset, kind, reference_errors, ranges[(dataset, kind)])
            base["dataset"] = dataset
            base["candidate_type"] = kind
            values.append(base)
            outcomes = {
                "reference_absolute_error": "median_algorithm_absolute_reference_error_ms",
                "selected_algorithm_failure": "failure",
                "across_method_disagreement": "timing_range_ms",
            }
            for condition, condition_rows in base.groupby("condition"):
                for feature, feature_label in FEATURES.items():
                    for outcome, outcome_column in outcomes.items():
                        rows = condition_rows[["subject_id", "beat_id", feature, outcome_column]].dropna()
                        participant_rho = []
                        for subject_id, subject in rows.groupby("subject_id"):
                            if len(subject) < 8 or subject[feature].nunique() < 2 or subject[outcome_column].nunique() < 2:
                                continue
                            rho = spearmanr(subject[feature], subject[outcome_column]).statistic
                            if np.isfinite(rho):
                                participant_rho.append((subject_id, float(rho), len(subject)))
                        estimate, low, high = median_ci([item[1] for item in participant_rho])
                        evidence.append({
                            "dataset": dataset, "candidate_type": kind, "condition": condition,
                            "feature": feature, "feature_label": feature_label, "outcome": outcome,
                            "selected_failure_algorithm": selected_method if outcome == "selected_algorithm_failure" else "",
                            "covered_participants": rows.subject_id.nunique(), "covered_beats": rows.beat_id.nunique(),
                            "failed_beats": int(rows[outcome_column].sum()) if outcome == "selected_algorithm_failure" else np.nan,
                            "feature_min": rows[feature].min() if len(rows) else np.nan,
                            "feature_median": rows[feature].median() if len(rows) else np.nan,
                            "feature_max": rows[feature].max() if len(rows) else np.nan,
                            "participant_effect_n": len(participant_rho),
                            "median_within_participant_spearman_rho": estimate,
                            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                            "interpretation_status": "supported_direction" if np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0) else "uncertain_or_unavailable",
                        })
    evidence = pd.DataFrame(evidence)
    values = pd.concat(values, ignore_index=True, sort=False)
    evidence.to_csv(ROOT / "results" / "tables" / "RQ1_morphology_evidence.csv", index=False)
    values.to_csv(ROOT / "results" / "tables" / "RQ1_morphology_plot_values.csv", index=False)
    return evidence, values


def bland_altman(dataset: str, kind: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b = PAIR[kind]
    status = load_status(dataset, kind)
    selected = status.loc[status.algorithm_id.isin([a, b]) & status.valid]
    wide = selected.pivot(index=[*KEYS, "condition"], columns="algorithm_id", values="candidate_time_from_r_ms").dropna().reset_index()
    wide["algorithm_a"] = a
    wide["algorithm_b"] = b
    wide["mean_time_from_r_ms"] = (wide[a] + wide[b]) / 2
    wide["signed_difference_a_minus_b_ms"] = wide[a] - wide[b]
    participant = wide.groupby(["subject_id", "condition"], as_index=False).agg(
        paired_beats=("beat_id", "nunique"),
        participant_mean_difference_ms=("signed_difference_a_minus_b_ms", "mean"),
        participant_median_difference_ms=("signed_difference_a_minus_b_ms", "median"),
        participant_sd_difference_ms=("signed_difference_a_minus_b_ms", "std"),
        participant_mean_square_difference=("signed_difference_a_minus_b_ms", lambda values: np.mean(np.square(values))),
    )
    summaries = []
    for condition, rows in participant.groupby("condition"):
        means = rows.participant_mean_difference_ms.to_numpy(float)
        seconds = rows.participant_mean_square_difference.to_numpy(float)
        bias = float(np.mean(means))
        sd = float(np.sqrt(max(0.0, np.mean(seconds) - bias**2)))
        low, high = bias - 1.96 * sd, bias + 1.96 * sd
        draw = RNG.integers(0, len(rows), size=(4000, len(rows)))
        boot_bias = means[draw].mean(axis=1)
        boot_second = seconds[draw].mean(axis=1)
        boot_sd = np.sqrt(np.maximum(0.0, boot_second - boot_bias**2))
        boot_low, boot_high = boot_bias - 1.96 * boot_sd, boot_bias + 1.96 * boot_sd
        summaries.append({
            "dataset": dataset, "candidate_type": kind, "condition": condition,
            "algorithm_a": a, "algorithm_b": b, "participants": len(rows),
            "paired_beats": int(rows.paired_beats.sum()), "participant_balanced_bias_ms": bias,
            "bias_ci_low": np.quantile(boot_bias, .025), "bias_ci_high": np.quantile(boot_bias, .975),
            "participant_balanced_loa_low_ms": low, "participant_balanced_loa_high_ms": high,
            "loa_low_ci_low": np.quantile(boot_low, .025), "loa_low_ci_high": np.quantile(boot_low, .975),
            "loa_high_ci_low": np.quantile(boot_high, .025), "loa_high_ci_high": np.quantile(boot_high, .975),
            "weighting": "equal participant weight; clustered participant bootstrap",
        })
    return wide, participant, pd.DataFrame(summaries)


def build_audit(rebeat_coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, settings in DATASETS.items():
        paired_n = 0
        if len(settings["conditions"]) == 2:
            subjects = []
            for kind in ["B"]:
                master = load_master(dataset, kind)
                subjects.append(set.intersection(*[set(master.loc[master.condition.eq(condition), "subject_id"]) for condition in settings["conditions"]]))
            paired_n = len(subjects[0])
        for kind in ["B", "X"]:
            master = load_master(dataset, kind)
            status = load_status(dataset, kind)
            for condition, master_rows in master.groupby("condition"):
                detection = status.loc[status.condition.eq(condition)]
                attempted_by_beat = detection.groupby("beat_id").attempted.any()
                eligible_beats = int(attempted_by_beat.sum())
                if dataset == "HeartCycle":
                    ref_type = "ECHO AVO" if kind == "B" else "ECHO AVC"
                    ref_eligible = int(master_rows.reference_eligible.fillna(False).astype(bool).sum())
                    common = heartcycle_errors(kind).beat_id.nunique()
                elif dataset == "ReBeat":
                    ref_type = f"Expert ICG {kind} annotation"
                    coverage = rebeat_coverage.loc[(rebeat_coverage.candidate_type.eq(kind)) & (rebeat_coverage.condition.eq(condition))]
                    ref_eligible = coverage.beat_id.nunique()
                    common = coverage.loc[coverage.strict_common_reference].beat_id.nunique()
                else:
                    ref_type, ref_eligible, common = "No independently interpretable B/X reference", 0, 0
                rows.append({
                    "dataset": dataset, "cohort": master_rows.cohort.iloc[0] if "cohort" in master_rows else "healthy",
                    "condition": condition, "candidate_type": kind,
                    "participant_n": master_rows.subject_id.nunique(), "recording_n": master_rows.recording_id.nunique(),
                    "total_beat_n": master_rows.beat_id.nunique(), "eligible_beat_n": eligible_beats,
                    "upstream_ineligible_beat_n": master_rows.beat_id.nunique() - eligible_beats,
                    "valid_detection_events": int(detection.valid.sum()), "missed_detection_events": int(detection.missed.sum()),
                    "invalid_detection_events": int(detection.invalid.sum()), "reference_type": ref_type,
                    "reference_eligible_or_matched_beat_n": ref_eligible, "strict_common_reference_beat_n": common,
                    "paired_participant_n": paired_n, "sampling_frequency_hz_min": master_rows.sampling_frequency_hz.min(),
                    "sampling_frequency_hz_max": master_rows.sampling_frequency_hz.max(),
                    "signal_units": master_rows.signal_units.iloc[0] if "signal_units" in master_rows else "unverified device units",
                    "reference_status": "validated matched subset" if ref_eligible else "not available",
                })
    audit = pd.DataFrame(rows)
    audit.to_csv(ROOT / "results" / "tables" / "RQ1_dataset_reference_audit.csv", index=False)
    return audit


def consolidated_results(reference_errors: pd.DataFrame, participant_accuracy: pd.DataFrame, peers: dict, audit: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    registry = read_csv(ROOT / "configs" / "canonical_method_registry.csv")
    rows = []
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            status = load_status(dataset, kind)
            peer_participant = peers[(dataset, kind)][1]
            for (condition, algorithm_id), detection in status.groupby(["condition", "algorithm_id"]):
                method = registry.loc[registry.algorithm_id.eq(algorithm_id)].iloc[0]
                source_audit = audit.loc[(audit.dataset.eq(dataset)) & audit.candidate_type.eq(kind) & audit.condition.eq(condition)].iloc[0]
                accuracy = participant_accuracy.loc[(participant_accuracy.dataset.eq(dataset)) & participant_accuracy.candidate_type.eq(kind) & participant_accuracy.condition.eq(condition) & participant_accuracy.algorithm_id.eq(algorithm_id)]
                error = reference_errors.loc[(reference_errors.dataset.eq(dataset)) & reference_errors.candidate_type.eq(kind) & reference_errors.condition.eq(condition) & reference_errors.algorithm_id.eq(algorithm_id)]
                peer = peer_participant.loc[(peer_participant.condition.eq(condition)) & peer_participant.algorithm_id.eq(algorithm_id)]
                morphology = evidence.loc[(evidence.dataset.eq(dataset)) & evidence.candidate_type.eq(kind) & evidence.condition.eq(condition) & evidence.participant_effect_n.ge(3)].copy()
                if len(morphology):
                    morphology["magnitude"] = morphology.median_within_participant_spearman_rho.abs()
                    top = morphology.sort_values("magnitude", ascending=False).iloc[0]
                    morphology_note = f"{top.feature_label} vs {top.outcome}: median within-participant rho {top.median_within_participant_spearman_rho:.2f} ({top.interpretation_status})."
                else:
                    morphology_note = "No sufficiently covered participant-level morphology association."
                attempted = int(detection.attempted.sum())
                valid, missed, invalid = int(detection.valid.sum()), int(detection.missed.sum()), int(detection.invalid.sum())
                reference_type = source_audit.reference_type
                if dataset == "IITB":
                    status_label = "completed; reference accuracy not available"
                else:
                    status_label = "completed on strict common reference subset"
                rows.append({
                    "dataset_cohort": f"{dataset} / {source_audit.cohort}", "condition": condition,
                    "participant_n": detection.subject_id.nunique(), "recording_n": detection.recording_id.nunique(),
                    "total_beat_n": detection.beat_id.nunique(), "eligible_attempted_beat_n": attempted,
                    "candidate_type": kind, "algorithm_id": algorithm_id, "algorithm_name": method.candidate_name,
                    "exact_algorithm_and_parameter_version": f"{method.implementation_version}; {method.parameters}",
                    "reference_type": reference_type, "reference_matched_beat_n": int(source_audit.reference_eligible_or_matched_beat_n),
                    "strict_common_reference_beat_n": int(error.beat_id.nunique()) if len(error) else 0,
                    "valid_detections": valid, "missed_detections": missed, "invalid_detections": invalid,
                    "success_pct": 100 * valid / attempted if attempted else np.nan,
                    "failure_pct": 100 * (missed + invalid) / attempted if attempted else np.nan,
                    "median_signed_reference_error_ms": float(error.signed_error_ms.median()) if len(error) else np.nan,
                    "participant_level_median_absolute_reference_error_ms": float(accuracy.median_absolute_error_ms.median()) if len(accuracy) else np.nan,
                    "p90_absolute_reference_error_ms": float(error.absolute_error_ms.quantile(.90)) if len(error) else np.nan,
                    "p95_absolute_reference_error_ms": float(error.absolute_error_ms.quantile(.95)) if len(error) else np.nan,
                    "participant_median_definition_disagreement_ms": float(peer.participant_median_peer_difference_ms.median()) if len(peer) else np.nan,
                    "common_beat_analysis_n": int(peer.common_valid_beats.sum()) if len(peer) else 0,
                    "main_morphology_or_failure_observation": morphology_note,
                    "analysis_status": status_label,
                    "one_sentence_finding": f"{algorithm_id} failure was {100 * (missed + invalid) / attempted:.2f}%" + (f" and participant-median reference MAE was {accuracy.median_absolute_error_ms.median():.2f} ms." if len(accuracy) else "; no independent timing accuracy estimate was available."),
                })
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "tables" / "RQ1_dataset_algorithm_results.csv", index=False)
    return table


def build_all() -> dict:
    rebeat_errors, _, rebeat_coverage = build_rebeat_reference()
    reference_errors, participant_accuracy = reference_inputs(rebeat_errors)
    audit = build_audit(rebeat_coverage)
    peers, ranges, changes, ba_values, ba_participants, ba_summaries = {}, {}, [], [], [], []
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            peer, peer_participant, range_participant = definition_disagreement(dataset, kind)
            peers[(dataset, kind)] = (peer, peer_participant, range_participant)
            ranges[(dataset, kind)] = read_csv(table_path(dataset, kind, "common_beat_ranges"))
            change = paired_changes(dataset, kind, peer_participant, range_participant)
            if len(change):
                change["dataset"] = dataset
                changes.append(change)
            values, participants, summary = bland_altman(dataset, kind)
            values["dataset"], values["candidate_type"] = dataset, kind
            participants["dataset"], participants["candidate_type"] = dataset, kind
            ba_values.append(values); ba_participants.append(participants); ba_summaries.append(summary)
    paired = pd.concat(changes, ignore_index=True)
    paired.to_csv(ROOT / "results" / "tables" / "RQ1_paired_condition_changes.csv", index=False)
    ba_values = pd.concat(ba_values, ignore_index=True, sort=False)
    ba_participants = pd.concat(ba_participants, ignore_index=True)
    ba_summaries = pd.concat(ba_summaries, ignore_index=True)
    ba_values.to_csv(ROOT / "results" / "tables" / "RQ1_bland_altman_values.csv", index=False)
    ba_participants.to_csv(ROOT / "results" / "tables" / "RQ1_bland_altman_participant_summary.csv", index=False)
    ba_summaries.to_csv(ROOT / "results" / "tables" / "RQ1_bland_altman_summary.csv", index=False)
    evidence, morphology_values = morphology_evidence(reference_errors, ranges)
    consolidated = consolidated_results(reference_errors, participant_accuracy, peers, audit, evidence)
    metadata = {
        "rebeat_reference_origin": "MATLAB one-based sample index converted to zero-based",
        "rebeat_matched_annotation_rows": int(rebeat_coverage.beat_id.nunique()),
        "frozen_B_methods": len(methods("B")), "frozen_X_methods": len(methods("X")),
        "selected_bland_altman_pairs": PAIR,
        "rows": {"audit": len(audit), "consolidated": len(consolidated), "morphology_evidence": len(evidence)},
    }
    (ROOT / "results" / "tables" / "RQ1_final_build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return {
        "audit": audit, "reference_errors": reference_errors, "participant_accuracy": participant_accuracy,
        "paired_changes": paired, "ba_values": ba_values, "ba_participants": ba_participants,
        "ba_summary": ba_summaries, "morphology_evidence": evidence,
        "morphology_values": morphology_values, "consolidated": consolidated,
    }


if __name__ == "__main__":
    build_all()
