"""Finalize Phase 4 without rerunning any fiducial detector."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import load_recording
from .manifest import dataset_manifest


EXCLUDED_CONSENSUS = {
    "B_morphology_consensus_40ms",
    "B_adaptive_multirepresentation_consensus_35ms",
    "X_morphology_consensus_60ms",
}
WEISSLER_B_DEPENDENCY = "B_55pct_RC"
WEISSLER_VARIANT = "provisional_average_coefficient"


def _codes(*values: str) -> str:
    result: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            parts = value
        elif pd.isna(value):
            parts = []
        else:
            parts = str(value).split(";")
        for code in parts:
            if code and code not in result:
                result.append(code)
    return ";".join(result)


def _nearest_indices(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.searchsorted(time_s, values, side="left")
    right = np.clip(right, 0, len(time_s) - 1)
    left = np.maximum(0, right - 1)
    choose_right = np.abs(time_s[right] - values) < np.abs(values - time_s[left])
    return np.where(choose_right, right, left).astype(int)


def _finalize_weissler(
    final: pd.DataFrame, phase3: pd.DataFrame, project_root: Path,
) -> pd.DataFrame:
    beat_key = ["dataset_id", "condition", "subject_id", "recording_id", "beat_id"]
    phase3_fields = beat_key + [
        "beat_number", "source_file", "beat_start_timestamp_s", "beat_end_timestamp_s",
        "beat_duration_ms", "c_timestamp_s", "x_fiducial_eligibility",
        "combined_issue_codes",
    ]
    beats = phase3[phase3_fields].set_index(beat_key)
    b55 = final[final["candidate_name"].eq(WEISSLER_B_DEPENDENCY)].set_index(beat_key)
    weissler_index = final.index[final["candidate_name"].eq("X_Weissler_HR")]

    final.loc[weissler_index, "method_variant"] = WEISSLER_VARIANT
    final.loc[weissler_index, "dependency_candidate_name"] = WEISSLER_B_DEPENDENCY
    specs = {spec.dataset_id: spec for spec in dataset_manifest(project_root)}

    for (dataset_id, recording_id), row_indices in final.loc[weissler_index].groupby(
        ["dataset_id", "recording_id"], sort=False
    ).groups.items():
        first = final.loc[next(iter(row_indices))]
        beat = beats.loc[(dataset_id, first["condition"], first["subject_id"], recording_id, first["beat_id"])]
        recording = load_recording(specs[dataset_id], Path(beat["source_file"]), project_root)

        calculated_indices: list[int] = []
        calculated_times: list[float] = []
        pending: list[tuple[int, tuple, pd.Series, pd.Series]] = []
        for row_index in row_indices:
            row = final.loc[row_index]
            key = tuple(row[column] for column in beat_key)
            beat = beats.loc[key]
            dependency = b55.loc[key]
            issues = beat["combined_issue_codes"]
            final.at[row_index, "dependency_b_sample_index"] = dependency["candidate_sample_index"]
            final.at[row_index, "dependency_b_timestamp_s"] = dependency["candidate_timestamp_s"]

            rr_ms = float(beat["beat_duration_ms"])
            hr = 60000.0 / rr_ms if np.isfinite(rr_ms) and rr_ms > 0 else np.nan
            final.at[row_index, "heart_rate_bpm"] = hr
            if not str(beat["x_fiducial_eligibility"]).startswith("ELIGIBLE"):
                final.at[row_index, "detection_status"] = "INELIGIBLE"
                final.at[row_index, "qc_status"] = "INVALID"
                final.at[row_index, "qc_issue_codes"] = _codes(issues, "INELIGIBLE_C_TO_NEXT_R_REGION")
                continue
            if dependency["detection_status"] != "DETECTED" or dependency["qc_status"] == "INVALID":
                final.at[row_index, "detection_status"] = "UNAVAILABLE_REQUIRED_B"
                final.at[row_index, "qc_status"] = "INVALID"
                final.at[row_index, "qc_issue_codes"] = _codes(
                    issues, "weissler_B_55pct_RC_missing_or_invalid"
                )
                continue
            if not np.isfinite(hr):
                final.at[row_index, "detection_status"] = "UNAVAILABLE_REQUIRED_HR"
                final.at[row_index, "qc_status"] = "INVALID"
                final.at[row_index, "qc_issue_codes"] = _codes(issues, "weissler_HR_missing_or_invalid")
                continue

            adjustment_ms = 415.5 - 1.65 * hr
            x_time = float(dependency["candidate_timestamp_s"]) + adjustment_ms / 1000.0
            final.at[row_index, "formula_adjustment_ms"] = adjustment_ms
            calculated_indices.append(row_index)
            calculated_times.append(x_time)
            pending.append((row_index, key, beat, dependency))

        if not pending:
            continue
        nearest = _nearest_indices(recording.time_s, np.asarray(calculated_times))
        for (row_index, key, beat, dependency), x_time, x_sample in zip(
            pending, calculated_times, nearest, strict=True
        ):
            c_sample = int(final.at[row_index, "c_sample"])
            next_r = int(final.at[row_index, "next_r_sample"])
            c_time = float(beat["c_timestamp_s"])
            next_r_time = float(beat["beat_end_timestamp_s"])
            issues = beat["combined_issue_codes"]
            invalid = []
            if not c_time < x_time < next_r_time:
                invalid.append("weissler_X_timestamp_not_strictly_between_C_and_next_R")
            if not c_sample < int(x_sample) < next_r:
                invalid.append("weissler_X_sample_not_strictly_between_C_and_next_R")
            tolerance = .51 / float(recording.sampling_rate_hz)
            if abs(float(recording.time_s[x_sample]) - x_time) > tolerance:
                invalid.append("weissler_X_timestamp_sample_mismatch")

            final.at[row_index, "candidate_sample_index"] = int(x_sample)
            final.at[row_index, "candidate_timestamp_s"] = x_time
            final.at[row_index, "detection_status"] = "DETECTED"
            final.at[row_index, "search_start_sample"] = c_sample + 1
            final.at[row_index, "search_stop_sample_exclusive"] = next_r
            final.at[row_index, "timestamp_representation"] = "formula_timestamp_nearest_native_sample_index"
            if invalid:
                final.at[row_index, "qc_status"] = "INVALID"
                final.at[row_index, "qc_issue_codes"] = _codes(issues, invalid)
            else:
                final.at[row_index, "qc_status"] = (
                    "FLAGGED" if dependency["qc_status"] == "FLAGGED" else "PASS"
                )
                final.at[row_index, "qc_issue_codes"] = _codes(issues)
    return final


def _candidate_summary(final: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_id", "cohort", "condition", "candidate_name", "candidate_type"]
    rows = []
    for key, group in final.groupby(keys, sort=False, dropna=False):
        eligible = ~group["detection_status"].eq("INELIGIBLE")
        detected = group["detection_status"].eq("DETECTED")
        rows.append(dict(zip(keys, key, strict=True)) | {
            "total_beat_rows": len(group),
            "eligible_beats": int(eligible.sum()),
            "detected_candidates": int(detected.sum()),
            "unavailable_or_missing": int((eligible & ~detected).sum()),
            "ineligible_beats": int((~eligible).sum()),
            "qc_passed_detections": int((detected & group["qc_status"].eq("PASS")).sum()),
            "qc_flagged_detections": int((detected & group["qc_status"].eq("FLAGGED")).sum()),
            "invalid_detections": int((detected & group["qc_status"].eq("INVALID")).sum()),
            "detection_coverage_among_eligible": detected.sum() / eligible.sum() if eligible.any() else np.nan,
            "issue_codes": _codes(*group["qc_issue_codes"].dropna().astype(str)),
        })
    return pd.DataFrame(rows)


def _recording_summary(final: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_id", "cohort", "condition", "subject_id", "recording_id"]
    rows = []
    for key, group in final.groupby(keys, sort=False, dropna=False):
        detected = group["detection_status"].eq("DETECTED")
        weissler = group["candidate_name"].eq("X_Weissler_HR")
        rows.append(dict(zip(keys, key, strict=True)) | {
            "beats": group["beat_id"].nunique(),
            "active_candidate_definitions": group["candidate_name"].nunique(),
            "candidate_rows": len(group),
            "detected_candidates": int(detected.sum()),
            "missing_or_unavailable_candidates": int((~detected & ~group["detection_status"].eq("INELIGIBLE")).sum()),
            "ineligible_candidates": int(group["detection_status"].eq("INELIGIBLE").sum()),
            "qc_passed_detections": int((detected & group["qc_status"].eq("PASS")).sum()),
            "qc_flagged_detections": int((detected & group["qc_status"].eq("FLAGGED")).sum()),
            "invalid_detections": int((detected & group["qc_status"].eq("INVALID")).sum()),
            "weissler_calculated": int((weissler & detected).sum()),
            "weissler_unavailable": int((weissler & ~detected & ~group["detection_status"].eq("INELIGIBLE")).sum()),
            "weissler_invalid": int((weissler & detected & group["qc_status"].eq("INVALID")).sum()),
            "weissler_flagged": int((weissler & detected & group["qc_status"].eq("FLAGGED")).sum()),
            "issue_codes": _codes(*group["qc_issue_codes"].dropna().astype(str)),
        })
    return pd.DataFrame(rows)


def run_phase4_finalization(project_root: Path, output_dir: Path) -> dict[str, int]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    canonical_path = output_dir / "phase4_bx_candidate_detection_all_beats.csv"
    source = pd.read_csv(canonical_path, low_memory=False)
    archive_path = output_dir / "phase4_bx_candidate_detection_all_beats_pre_finalization.csv"
    if not archive_path.exists():
        source.to_csv(archive_path, index=False, float_format="%.9f")
    phase3 = pd.read_csv(project_root / "outputs" / "phase3" / "phase3_beat_segmentation_all_beats.csv", low_memory=False)
    final = source.loc[~source["candidate_name"].isin(EXCLUDED_CONSENSUS)].copy()
    already_finalized = (
        "method_variant" in final.columns
        and final.loc[final["candidate_name"].eq("X_Weissler_HR"), "method_variant"]
        .eq(WEISSLER_VARIANT).all()
        and final.loc[final["candidate_name"].eq("X_Weissler_HR"), "candidate_timestamp_s"].notna().any()
    )
    for column in (
        "method_variant", "dependency_candidate_name", "dependency_b_sample_index",
        "dependency_b_timestamp_s", "heart_rate_bpm", "formula_adjustment_ms",
    ):
        if column not in final.columns:
            final[column] = "" if column in {"method_variant", "dependency_candidate_name"} else np.nan
    if not already_finalized:
        final = _finalize_weissler(final, phase3, project_root)
    final_path = output_dir / "phase4_bx_candidate_detection_all_beats_final.csv"
    final.to_csv(final_path, index=False, float_format="%.9f")
    final.to_csv(canonical_path, index=False, float_format="%.9f")

    summary = _candidate_summary(final)
    summary.to_csv(output_dir / "phase4_bx_candidate_qc_summary_final.csv", index=False, float_format="%.6f")
    summary.to_csv(output_dir / "phase4_bx_candidate_qc_summary.csv", index=False, float_format="%.6f")
    recording_summary = _recording_summary(final)
    recording_summary.to_csv(output_dir / "phase4_recording_qc_summary_final.csv", index=False)
    recording_summary.to_csv(output_dir / "phase4_recording_qc_summary.csv", index=False)

    definitions = pd.read_csv(output_dir / "phase4_candidate_definition_summary.csv")
    definitions = definitions.loc[~definitions["candidate_name"].isin(EXCLUDED_CONSENSUS)].copy()
    for column in (
        "exact_definition", "parameters", "implementation_status", "unresolved_ambiguity",
        "method_variant", "dependency_candidate_name", "original_sex_specific_equations",
    ):
        if column in definitions.columns:
            definitions[column] = definitions[column].fillna("").astype(object)
    definitions["active_in_phase4_final"] = True
    definitions["method_variant"] = "original_approved_definition"
    definitions["dependency_candidate_name"] = ""
    definitions["original_sex_specific_equations"] = ""
    mask = definitions["candidate_name"].eq("X_Weissler_HR")
    definitions.loc[mask, "exact_definition"] = "X = B_55pct_RC + (415.5 - 1.65 x HR) / 1000 seconds."
    definitions.loc[mask, "parameters"] = "HR in beats/min; adjustment in ms converted to seconds; no height input"
    definitions.loc[mask, "implementation_status"] = "IMPLEMENTED_PROVISIONAL"
    definitions.loc[mask, "unresolved_ambiguity"] = ""
    definitions.loc[mask, "method_variant"] = WEISSLER_VARIANT
    definitions.loc[mask, "dependency_candidate_name"] = WEISSLER_B_DEPENDENCY
    definitions.loc[mask, "original_sex_specific_equations"] = (
        "male: X=B+(-0.0017*HR+0.413)s; female: X=B+(-0.0016*HR+0.418)s"
    )
    definitions.to_csv(output_dir / "phase4_candidate_definition_summary_final.csv", index=False)
    definitions.to_csv(output_dir / "phase4_candidate_definition_summary.csv", index=False)

    parameters = pd.DataFrame([
        {"component": "active_catalogue", "parameter": "candidate_count", "value": 23, "unit": "definitions", "source": "researcher Phase 4 finalisation instruction"},
        {"component": "active_catalogue", "parameter": "excluded_consensus_candidates", "value": ";".join(sorted(EXCLUDED_CONSENSUS)), "unit": "names", "source": "researcher Phase 4 finalisation instruction"},
        {"component": "X_Weissler_HR", "parameter": "method_variant", "value": WEISSLER_VARIANT, "unit": "label", "source": "researcher Phase 4 finalisation instruction"},
        {"component": "X_Weissler_HR", "parameter": "formula", "value": "B_time + (415.5 - 1.65*HR)/1000", "unit": "seconds", "source": "researcher Phase 4 finalisation instruction"},
        {"component": "X_Weissler_HR", "parameter": "B_dependency", "value": WEISSLER_B_DEPENDENCY, "unit": "candidate", "source": "PLANS.md legacy dependency record"},
        {"component": "X_Weissler_HR", "parameter": "HR", "value": "60000 / RR_ms", "unit": "beats/min", "source": "Phase 3 consecutive detected R timestamps"},
        {"component": "X_Weissler_HR", "parameter": "height_input", "value": "not_used", "unit": "none", "source": "researcher instruction"},
    ])
    parameters.to_csv(output_dir / "phase4_processing_parameters_final.csv", index=False)
    parameters.to_csv(output_dir / "phase4_processing_parameters.csv", index=False)

    weissler = final[final["candidate_name"].eq("X_Weissler_HR")]
    report = [
        "# Phase 4 Finalisation Report", "",
        f"Generated: {datetime.now().astimezone().isoformat()}", "",
        "The original Phase 4 files were preserved. Final outputs exclude the three consensus definitions from the active catalogue and contain 23 individual definitions (10 B, 13 X). No detector, filter, R peak, C point, or beat boundary was rerun or changed.", "",
        "## Provisional Weissler method", "",
        "`X_Weissler_HR = B_55pct_RC + (415.5 - 1.65 x HR) / 1000` seconds, with `HR = 60000 / RR_ms`. The method is labelled `provisional_average_coefficient`; height is not used. The original male and female equations remain in the final definition table.", "",
        f"- Calculated: {(weissler.detection_status == 'DETECTED').sum()}",
        f"- Unavailable: {(~weissler.detection_status.isin(['DETECTED', 'INELIGIBLE'])).sum()}",
        f"- Ineligible: {(weissler.detection_status == 'INELIGIBLE').sum()}",
        f"- Invalid after calculation: {((weissler.detection_status == 'DETECTED') & (weissler.qc_status == 'INVALID')).sum()}",
        f"- QC-flagged but location-valid: {((weissler.detection_status == 'DETECTED') & (weissler.qc_status == 'FLAGGED')).sum()}", "",
        "## Limitations", "",
        "The average-coefficient equation is provisional and is not the original sex-specific Weissler formulation. Candidate coverage and interval validity do not establish physiological accuracy. Reference annotations remain separate.", "",
    ]
    report_text = "\n".join(report)
    (output_dir / "PHASE4_FINALISATION_REPORT.md").write_text(report_text, encoding="utf-8")
    (output_dir / "PHASE4_REPORT.md").write_text(report_text, encoding="utf-8")
    return {
        "beats": final["beat_id"].nunique(), "candidate_rows": len(final),
        "active_definitions": final["candidate_name"].nunique(),
        "weissler_calculated": int(weissler["detection_status"].eq("DETECTED").sum()),
        "weissler_invalid": int((weissler["detection_status"].eq("DETECTED") & weissler["qc_status"].eq("INVALID")).sum()),
    }
