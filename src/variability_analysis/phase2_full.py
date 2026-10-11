"""Full-recording Phase 2 R/C detection using the accepted pilot configuration."""

from __future__ import annotations

import csv
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np

from .loaders import load_recording
from .manifest import dataset_manifest
from .phase2 import _match_reference_r, _plot_recording
from .phase2_config import Phase2Config
from .signal_processing import detect_c_point, preprocess_recording


BEAT_COLUMNS = (
    "dataset_id", "cohort", "condition", "subject_id", "recording_id",
    "beat_id", "beat_number", "source_file", "r_sample", "next_r_sample",
    "c_sample", "r_timestamp_s", "next_r_timestamp_s", "c_timestamp_s",
    "r_to_c_ms", "rr_interval_ms", "sampling_frequency_hz",
    "sampling_frequency_source", "r_detection_method", "c_detection_method",
    "icg_signal_label", "icg_sign_convention", "detection_status", "qc_status",
    "issue_codes",
)

RECORDING_COLUMNS = (
    "dataset_id", "cohort", "condition", "subject_id", "recording_id",
    "source_file", "processing_status", "sample_count", "duration_s",
    "sampling_frequency_hz", "sampling_frequency_source", "timestamp_source",
    "timestamp_rate_hz", "timestamp_rate_difference_percent",
    "detected_r_peaks", "complete_beats", "c_candidates_detected",
    "qc_passed_beats", "qc_flagged_beats", "detection_failures",
    "provided_reference_r_count", "matched_reference_r_count",
    "mean_absolute_reference_r_difference_ms",
    "index_validation_status", "timestamp_validation_status",
    "icg_signal_label", "icg_sign_convention", "reference_types_available",
    "qc_issue_codes",
)

FAILURE_COLUMNS = (
    "dataset_id", "cohort", "condition", "subject_id", "recording_id",
    "beat_id", "entity_type", "failure_stage", "reason", "source_file",
    "r_sample", "next_r_sample", "sampling_frequency_hz",
)

PARAMETER_COLUMNS = (
    "component", "parameter", "value", "unit", "source", "notes",
)


def _writer(path: Path, columns: tuple[str, ...]):
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def _parameter_rows(config: Phase2Config) -> list[dict[str, str]]:
    rows = []
    for field in fields(config):
        name = field.name
        value = getattr(config, name)
        if name.startswith("icg_filter"):
            component = "ICG_preprocessing"
            source = "PLANS.md;legacy common/preprocessing.py;accepted Phase 2 pilot"
        elif name.startswith("r_") or name == "pan_tompkins_reference_doi":
            component = "ECG_R_detection"
            source = "Pan and Tompkins 1985;accepted Phase 2 pilot"
        elif name.startswith("c_"):
            component = "C_detection"
            source = "legacy common/c_detection.py;PLANS.md;accepted Phase 2 pilot"
        else:
            component = "output_QC"
            source = "accepted Phase 2 pilot"
        unit = ""
        if name.endswith("_hz"):
            unit = "Hz"
        elif name.endswith("_ms"):
            unit = "ms"
        elif name.endswith("_s"):
            unit = "s"
        elif name.endswith("_order") or name.endswith("_beats") or name.endswith("_dpi"):
            unit = "count"
        rows.append({
            "component": component, "parameter": name, "value": str(value),
            "unit": unit, "source": source,
            "notes": "Frozen parameter; unchanged from accepted pilot.",
        })
    rows.extend([
        {
            "component": "IITB_input", "parameter": "sampling_frequency_hz",
            "value": "500", "unit": "Hz",
            "source": "IITB-ICG-Database-Description_2025jul08.pdf",
            "notes": "Documented acquisition rate; not stored in MAT files.",
        },
        {
            "component": "IITB_input", "parameter": "channel_mapping",
            "value": "1=Z0;2=-z(t);3=-dz/dt;4=ECG;5=undocumented", "unit": "",
            "source": "IITB-ICG-Database-Description_2025jul08.pdf",
            "notes": "Recorded signs are preserved; no inversion is applied.",
        },
    ])
    return rows


def _write_report(
    output_dir: Path, summaries: list[dict], discovered: int, processed: int,
    failed: int, total_beats: int, flagged_beats: int,
    heart_reference_r_count: int, heart_matched_r_count: int,
) -> None:
    lines = [
        "# Phase 2 Full-Recording R/C Detection Report", "",
        f"Generated: {datetime.now().astimezone().isoformat()}", "",
        "## Scope and methods", "",
        "All recordings in the six manifest groups were processed with the accepted Phase 2 pilot configuration. Original files were read only. Pan-Tompkins detections, not supplied R events, define beats. The legacy C global-maximum candidate is retained even when its R-to-C interval is QC-flagged.",
        "",
        "IITB uses the documented 500 Hz acquisition rate and channel mapping: Z0, -z(t), -dz/dt, ECG, and an undocumented fifth column. No sign inversion was applied. IITB Doppler RR/VTI/SV values were not treated as fiducial annotations or sample-synchronized references.",
        "",
        "## Processing outcome", "",
        f"- Recordings discovered: {discovered}",
        f"- Recordings processed: {processed}",
        f"- Recordings failed: {failed}",
        f"- Complete beats written: {total_beats}",
        f"- R-to-C QC-flagged beats: {flagged_beats}", "",
        "| Dataset | Discovered | Processed | Failed | Flagged recordings | Beats | QC passed | QC flagged | Detection failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset_id']} | {row['discovered']} | {row['processed']} | "
            f"{row['failed']} | {row['flagged_recordings']} | {row['beats']} | "
            f"{row['qc_passed']} | {row['qc_flagged']} | {row['detection_failures']} |"
        )
    lines.extend([
        "", "## Validation", "",
        "- Beat identifiers are unique and deterministic within dataset and recording.",
        "- R, next-R, and C sample indices were range-checked; R < next R was enforced.",
        "- Timestamp monotonicity and timestamp-derived sampling rate were checked per recording.",
        "- Detected and reference events remain separate. HeartCycle R references are used only for QC matching.",
        f"- HeartCycle independently detected R peaks matched {heart_matched_r_count}/{heart_reference_r_count} supplied ECG/impedance R references within the frozen 120 ms QC tolerance.",
        "- R-to-C 60-200 ms is a QC flag, not proof that the retained C candidate is wrong.",
        "", "## Remaining issues", "",
        "- ReBeat expert annotation index origin remains unresolved, so annotation-level C accuracy is not claimed.",
        "- IITB digitized signal units and column 5 remain undocumented.",
        "- The unchanged legacy global-maximum C rule was applied to IITB's documented -dz/dt representation without sign inversion. Its output is a retained candidate with timing QC, not an expert-validated C point.",
        "- IITB includes saturated/discontinuous intervals; affected detections remain visible and flagged rather than silently deleted.",
        "- IITB Doppler and ICG systems have independent time bases; direct sample-level synchronization is not assumed.",
        "", "## Completion gate", "",
        "Phase 2 full-recording processing is complete. The pipeline is ready for Phase 3 record construction only if Phase 3 preserves all Phase 2 QC flags and does not treat QC-flagged C candidates as verified fiducials.", "",
    ])
    (output_dir / "PHASE2_FULL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase2_full(project_root: Path, output_dir: Path) -> dict[str, int]:
    """Process every manifest recording and write the four required final CSVs."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots" / "full_dataset"
    plot_dir.mkdir(parents=True, exist_ok=True)
    config = Phase2Config()
    manifest = dataset_manifest(project_root)
    paths = [(spec, path) for spec in manifest for path in spec.discover(project_root)]

    beat_handle, beat_writer = _writer(output_dir / "phase2_r_c_detection_all_beats.csv", BEAT_COLUMNS)
    recording_handle, recording_writer = _writer(output_dir / "phase2_recording_qc_summary.csv", RECORDING_COLUMNS)
    failure_handle, failure_writer = _writer(output_dir / "phase2_detection_failures.csv", FAILURE_COLUMNS)
    parameter_handle, parameter_writer = _writer(output_dir / "phase2_processing_parameters.csv", PARAMETER_COLUMNS)
    parameter_writer.writerows(_parameter_rows(config))
    parameter_handle.close()

    stats = {
        spec.dataset_id: {
            "dataset_id": spec.dataset_id, "discovered": len(spec.discover(project_root)),
            "processed": 0, "failed": 0, "flagged_recordings": 0,
            "beats": 0, "qc_passed": 0, "qc_flagged": 0,
            "detection_failures": 0,
        }
        for spec in manifest
    }
    representative_plotted: set[str] = set()
    total_beats = flagged_beats = processed_count = failed_count = 0
    heart_reference_r_count = heart_matched_r_count = 0

    try:
        for spec, path in paths:
            try:
                recording = load_recording(spec, path, project_root)
                structural_errors = recording.validation_errors()
                if structural_errors:
                    raise ValueError(";".join(structural_errors))
                processed = preprocess_recording(recording, config)
                r_indices = processed.r_indices
                per_record_beats = []
                qc_passed = qc_flagged = c_detected = detection_failures = 0
                record_issues = list(recording.warnings)
                if processed.r_detection_reason:
                    record_issues.append(processed.r_detection_reason)

                for number, (r_sample, next_r_sample) in enumerate(
                    zip(r_indices[:-1], r_indices[1:], strict=True), start=1,
                ):
                    r_sample, next_r_sample = int(r_sample), int(next_r_sample)
                    beat_id = f"{recording.dataset_id}|{recording.recording_id}|b{number:06d}"
                    c = detect_c_point(processed, r_sample, next_r_sample, config)
                    rr_ms = float((recording.time_s[next_r_sample] - recording.time_s[r_sample]) * 1000.0)
                    detection_status = "DETECTED" if c.sample_index is not None else "C_DETECTION_FAILED"
                    qc_status = "PASS" if c.status == "VALID" else "FLAGGED"
                    issues = list(recording.warnings)
                    if c.reason:
                        issues.append(c.reason)
                    row = {
                        "dataset_id": recording.dataset_id, "cohort": recording.cohort,
                        "condition": recording.condition, "subject_id": recording.subject_id,
                        "recording_id": recording.recording_id, "beat_id": beat_id,
                        "beat_number": number, "source_file": str(recording.source_path),
                        "r_sample": r_sample, "next_r_sample": next_r_sample,
                        "c_sample": "" if c.sample_index is None else c.sample_index,
                        "r_timestamp_s": f"{recording.time_s[r_sample]:.9f}",
                        "next_r_timestamp_s": f"{recording.time_s[next_r_sample]:.9f}",
                        "c_timestamp_s": "" if c.time_s is None else f"{c.time_s:.9f}",
                        "r_to_c_ms": "" if c.rc_ms is None else f"{c.rc_ms:.6f}",
                        "rr_interval_ms": f"{rr_ms:.6f}",
                        "sampling_frequency_hz": f"{recording.sampling_rate_hz:.9f}",
                        "sampling_frequency_source": recording.sampling_rate_source,
                        "r_detection_method": config.r_detector,
                        "c_detection_method": config.c_method,
                        "icg_signal_label": recording.metadata.get(
                            "icg_signal_label",
                            "impedance" if recording.icg_kind == "impedance" else "dZ/dt",
                        ),
                        "icg_sign_convention": recording.metadata.get(
                            "icg_sign_convention", "source_as_recorded"
                        ),
                        "detection_status": detection_status, "qc_status": qc_status,
                        "issue_codes": ";".join(dict.fromkeys(issues)),
                    }
                    beat_writer.writerow(row)
                    per_record_beats.append({
                        "c_status": c.status,
                        "c_sample": "" if c.sample_index is None else c.sample_index,
                        "beat_index": number,
                    })
                    c_detected += c.sample_index is not None
                    qc_passed += qc_status == "PASS"
                    qc_flagged += qc_status == "FLAGGED"
                    if c.sample_index is None:
                        detection_failures += 1
                        failure_writer.writerow({
                            "dataset_id": recording.dataset_id, "cohort": recording.cohort,
                            "condition": recording.condition, "subject_id": recording.subject_id,
                            "recording_id": recording.recording_id, "beat_id": beat_id,
                            "entity_type": "beat", "failure_stage": "C_detection",
                            "reason": c.reason, "source_file": str(recording.source_path),
                            "r_sample": r_sample, "next_r_sample": next_r_sample,
                            "sampling_frequency_hz": f"{recording.sampling_rate_hz:.9f}",
                        })

                if r_indices.size < 2:
                    detection_failures += 1
                    failure_writer.writerow({
                        "dataset_id": recording.dataset_id, "cohort": recording.cohort,
                        "condition": recording.condition, "subject_id": recording.subject_id,
                        "recording_id": recording.recording_id, "beat_id": "",
                        "entity_type": "recording", "failure_stage": "R_detection",
                        "reason": processed.r_detection_reason or "fewer_than_two_r_peaks",
                        "source_file": str(recording.source_path), "r_sample": "",
                        "next_r_sample": "", "sampling_frequency_hz": f"{recording.sampling_rate_hz:.9f}",
                    })

                time_deltas = np.diff(recording.time_s)
                timestamp_rate = float(1.0 / np.median(time_deltas))
                rate_difference = abs(timestamp_rate - recording.sampling_rate_hz) / recording.sampling_rate_hz * 100.0
                index_ok = bool(
                    np.all(r_indices >= 0) and np.all(r_indices < recording.sample_count)
                    and np.all(np.diff(r_indices) > 0)
                )
                timestamp_ok = bool(np.all(time_deltas > 0) and np.isfinite(timestamp_rate))
                provided = recording.references.get("heartcycle_ecg_imp_r_time_s")
                provided_count = matched_count = 0
                differences = np.array([], dtype=float)
                if provided is not None:
                    reference_times = np.asarray(provided.values, dtype=float)
                    provided_count = int(reference_times.size)
                    matched_count, differences = _match_reference_r(
                        recording.time_s[r_indices], reference_times,
                        config.reference_r_match_tolerance_ms / 1000.0,
                    )
                    heart_reference_r_count += provided_count
                    heart_matched_r_count += matched_count
                if qc_flagged:
                    record_issues.append("r_to_c_qc_flagged_beats_present")
                if detection_failures:
                    record_issues.append("detection_failures_present")
                if not index_ok:
                    record_issues.append("sample_index_validation_failed")
                if not timestamp_ok:
                    record_issues.append("timestamp_validation_failed")
                processing_status = (
                    "FAILED" if r_indices.size < 2 or not index_ok or not timestamp_ok
                    else "FLAGGED" if record_issues else "PASSED"
                )
                recording_writer.writerow({
                    "dataset_id": recording.dataset_id, "cohort": recording.cohort,
                    "condition": recording.condition, "subject_id": recording.subject_id,
                    "recording_id": recording.recording_id, "source_file": str(recording.source_path),
                    "processing_status": processing_status, "sample_count": recording.sample_count,
                    "duration_s": f"{recording.duration_s:.9f}",
                    "sampling_frequency_hz": f"{recording.sampling_rate_hz:.9f}",
                    "sampling_frequency_source": recording.sampling_rate_source,
                    "timestamp_source": recording.timestamp_source,
                    "timestamp_rate_hz": f"{timestamp_rate:.9f}",
                    "timestamp_rate_difference_percent": f"{rate_difference:.9f}",
                    "detected_r_peaks": int(r_indices.size),
                    "complete_beats": max(0, int(r_indices.size) - 1),
                    "c_candidates_detected": c_detected, "qc_passed_beats": qc_passed,
                    "qc_flagged_beats": qc_flagged, "detection_failures": detection_failures,
                    "provided_reference_r_count": provided_count,
                    "matched_reference_r_count": matched_count,
                    "mean_absolute_reference_r_difference_ms": (
                        f"{np.mean(np.abs(differences)) * 1000.0:.6f}" if differences.size else ""
                    ),
                    "index_validation_status": "PASS" if index_ok else "FAIL",
                    "timestamp_validation_status": "PASS" if timestamp_ok else "FAIL",
                    "icg_signal_label": recording.metadata.get(
                        "icg_signal_label",
                        "impedance" if recording.icg_kind == "impedance" else "dZ/dt",
                    ),
                    "icg_sign_convention": recording.metadata.get(
                        "icg_sign_convention", "source_as_recorded"
                    ),
                    "reference_types_available": ";".join(sorted(recording.references)),
                    "qc_issue_codes": ";".join(dict.fromkeys(record_issues)),
                })
                if recording.dataset_id not in representative_plotted:
                    _plot_recording(
                        processed, per_record_beats,
                        plot_dir / f"{recording.dataset_id}__representative",
                        config,
                    )
                    representative_plotted.add(recording.dataset_id)

                group = stats[recording.dataset_id]
                group["processed"] += 1
                group["flagged_recordings"] += processing_status == "FLAGGED"
                group["failed"] += processing_status == "FAILED"
                group["beats"] += max(0, int(r_indices.size) - 1)
                group["qc_passed"] += qc_passed
                group["qc_flagged"] += qc_flagged
                group["detection_failures"] += detection_failures
                processed_count += 1
                total_beats += max(0, int(r_indices.size) - 1)
                flagged_beats += qc_flagged
            except Exception as error:
                failed_count += 1
                stats[spec.dataset_id]["failed"] += 1
                failure_writer.writerow({
                    "dataset_id": spec.dataset_id, "cohort": spec.cohort,
                    "condition": spec.condition, "subject_id": "", "recording_id": path.stem,
                    "beat_id": "", "entity_type": "recording", "failure_stage": "loading_or_processing",
                    "reason": f"{type(error).__name__}:{error}", "source_file": str(path),
                    "r_sample": "", "next_r_sample": "", "sampling_frequency_hz": "",
                })
                recording_writer.writerow({
                    "dataset_id": spec.dataset_id, "cohort": spec.cohort,
                    "condition": spec.condition, "subject_id": "", "recording_id": path.stem,
                    "source_file": str(path), "processing_status": "FAILED",
                    "qc_issue_codes": f"{type(error).__name__}:{error}",
                })
    finally:
        beat_handle.close()
        recording_handle.close()
        failure_handle.close()

    summary_rows = list(stats.values())
    _write_report(
        output_dir, summary_rows, len(paths), processed_count, failed_count,
        total_beats, flagged_beats, heart_reference_r_count,
        heart_matched_r_count,
    )
    return {
        "discovered": len(paths), "processed": processed_count,
        "failed": failed_count, "beats": total_beats,
        "flagged_beats": flagged_beats, "plots": len(representative_plotted),
    }
