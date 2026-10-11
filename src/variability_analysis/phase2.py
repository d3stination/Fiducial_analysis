"""Run Phase 2 pilot preprocessing, R/C QC, plots, and report generation."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .loaders import load_recording
from .manifest import DatasetSpec, dataset_manifest
from .phase2_config import Phase2Config
from .schema import Recording
from .signal_processing import ProcessedRecording, detect_c_point, preprocess_recording


RECORDING_COLUMNS = (
    "dataset_id", "cohort", "condition", "selection_reason", "subject_id",
    "recording_id", "source_file", "sampling_rate_hz", "timestamp_source",
    "sample_count", "duration_s", "r_count", "complete_beat_count",
    "valid_c_count", "invalid_c_count", "c_coverage", "mean_rr_ms", "sd_rr_ms",
    "minimum_rr_ms", "maximum_rr_ms", "maximum_detected_hr_bpm",
    "provided_r_count", "matched_provided_r_count", "mean_abs_r_difference_ms",
    "maximum_abs_r_difference_ms", "plot_start_s", "plot_stop_s", "plot_files",
    "status", "reason_codes",
)

BEAT_COLUMNS = (
    "dataset_id", "cohort", "condition", "subject_id", "recording_id",
    "beat_index", "r_sample", "next_r_sample", "r_time_s", "next_r_time_s",
    "rr_ms", "heart_rate_bpm", "c_sample", "c_time_s", "r_to_c_ms",
    "c_status", "c_reason", "r_source", "c_method",
)

SUMMARY_COLUMNS = (
    "dataset_id", "cohort", "condition", "pilot_recordings", "passed_recordings",
    "warning_recordings", "failed_recordings", "detected_r_peaks", "complete_beats",
    "valid_c_beats", "invalid_c_beats", "c_coverage", "mean_heart_rate_bpm",
    "minimum_heart_rate_bpm", "maximum_heart_rate_bpm", "issue_codes",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_pilot_rows(inventory_path: Path) -> list[dict[str, str]]:
    """Select deterministic representative and edge-case pilot recordings."""

    rows = _read_csv(inventory_path)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset_id"], []).append(row)
    selected: list[dict[str, str]] = []

    def add(row: dict[str, str], reason: str) -> None:
        if any(existing["source_file"] == row["source_file"] for existing in selected):
            return
        copy = dict(row)
        copy["selection_reason"] = reason
        selected.append(copy)

    heart = grouped["heartcycle_healthy"]
    mean_duration = float(np.mean([float(row["duration_s"]) for row in heart]))
    add(min(heart, key=lambda row: abs(float(row["duration_s"]) - mean_duration)), "duration_nearest_group_mean")
    add(max(heart, key=lambda row: float(row["duration_s"])), "maximum_duration_for_multi_beat_review")
    rate_edge = next(
        (row for row in heart if "timestamp_rate_difference" in row["warning_codes"]),
        None,
    )
    if rate_edge:
        add(rate_edge, "nominal_vs_timestamp_rate_difference_over_1pct")

    for dataset_id, subjects in {
        "rebeat_baseline": ("rebeat:Subject_01", "rebeat:Subject_12"),
        "rebeat_cognitive": (
            "rebeat:Subject_01", "rebeat:Subject_05", "rebeat:Subject_12",
        ),
    }.items():
        reasons = {
            "rebeat:Subject_01": "typical_annotated_subject",
            "rebeat:Subject_05": "missing_cognitive_annotation_edge_case",
            "rebeat:Subject_12": "annotation_order_violation_edge_case",
        }
        for subject in subjects:
            add(next(row for row in grouped[dataset_id] if row["subject_id"] == subject), reasons[subject])

    for dataset_id in ("iitb_normal", "iitb_post_exercise", "iitb_cvd"):
        group = grouped[dataset_id]
        add(min(group, key=lambda row: float(row["duration_s"])), "minimum_duration")
        add(max(group, key=lambda row: float(row["duration_s"])), "maximum_duration")
    return selected


def _match_reference_r(
    detected_times: np.ndarray, reference_times: np.ndarray, tolerance_s: float,
) -> tuple[int, np.ndarray]:
    """Greedily match distinct R events for QC only, never for detection."""

    used: set[int] = set()
    differences: list[float] = []
    for reference in np.asarray(reference_times, dtype=float):
        distances = np.abs(detected_times - reference)
        candidates = [index for index in np.argsort(distances) if int(index) not in used]
        if candidates and distances[candidates[0]] <= tolerance_s:
            index = int(candidates[0])
            used.add(index)
            differences.append(float(detected_times[index] - reference))
    return len(differences), np.asarray(differences, dtype=float)


def _plot_window(processed: ProcessedRecording, config: Phase2Config) -> tuple[int, int]:
    r_indices = processed.r_indices
    if r_indices.size:
        first = max(0, int(r_indices[0] - round(0.25 * processed.recording.sampling_rate_hz)))
        final_r = int(r_indices[min(config.plot_max_beats - 1, r_indices.size - 1)])
        last = min(
            processed.recording.sample_count,
            final_r + int(round(0.50 * processed.recording.sampling_rate_hz)),
        )
        return first, max(first + 2, last)
    return 0, min(processed.recording.sample_count, int(round(10 * processed.recording.sampling_rate_hz)))


def _window_around_r_position(
    processed: ProcessedRecording, r_position: int, config: Phase2Config,
) -> tuple[int, int]:
    """Return a native-sample window around a selected detected beat."""

    r_indices = processed.r_indices
    left = max(0, r_position - 2)
    right = min(r_indices.size - 1, left + config.plot_max_beats - 1)
    left = max(0, right - config.plot_max_beats + 1)
    margin_before = int(round(0.25 * processed.recording.sampling_rate_hz))
    margin_after = int(round(0.50 * processed.recording.sampling_rate_hz))
    return (
        max(0, int(r_indices[left]) - margin_before),
        min(processed.recording.sample_count, int(r_indices[right]) + margin_after),
    )


def _plot_recording(
    processed: ProcessedRecording, beat_rows: list[dict], output_base: Path,
    config: Phase2Config, sample_window: tuple[int, int] | None = None,
) -> tuple[float, float]:
    recording = processed.recording
    start, stop = sample_window or _plot_window(processed, config)
    time = recording.time_s[start:stop]
    in_window_r = processed.r_indices[(processed.r_indices >= start) & (processed.r_indices < stop)]
    valid_c = np.array([
        int(row["c_sample"]) for row in beat_rows
        if row["c_status"] == "VALID" and row["c_sample"] != ""
        and start <= int(row["c_sample"]) < stop
    ], dtype=int)
    invalid_c = np.array([
        int(row["c_sample"]) for row in beat_rows
        if row["c_status"] == "INVALID" and row["c_sample"] != ""
        and start <= int(row["c_sample"]) < stop
    ], dtype=int)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {
        "raw": "#707070", "filtered": "#1f4e79", "analysis": "#2f6f62",
        "r": "#8c2d3e", "c": "#c47f28", "invalid": "#6f4e7c",
    }
    figure, axes = plt.subplots(4, 1, figsize=(7.2, 7.8), sharex=True, constrained_layout=True)
    axes[0].plot(time, recording.ecg[start:stop], color=colors["raw"], linewidth=0.7)
    axes[0].scatter(recording.time_s[in_window_r], recording.ecg[in_window_r], s=16, marker="o", facecolors="white", edgecolors=colors["r"], linewidths=0.9, label="Detected R")
    axes[0].set_ylabel("Raw ECG\n(source unit)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(time, processed.filtered_ecg[start:stop], color=colors["filtered"], linewidth=0.8)
    axes[1].scatter(recording.time_s[in_window_r], processed.filtered_ecg[in_window_r], s=16, marker="o", facecolors="white", edgecolors=colors["r"], linewidths=0.9)
    axes[1].set_ylabel("PT-filtered ECG\n(source unit)")
    axes[2].plot(time, recording.icg[start:stop], color=colors["raw"], linewidth=0.55, label="Raw")
    axes[2].plot(time, processed.filtered_icg[start:stop], color=colors["filtered"], linewidth=0.8, label="0.8–30 Hz")
    source_label = (
        "Impedance" if recording.icg_kind == "impedance"
        else str(recording.metadata.get("icg_signal_label", "dZ/dt"))
    )
    axes[2].set_ylabel(f"{source_label}\n(source unit)")
    axes[2].legend(frameon=False, loc="upper right", ncol=2)
    analysis_label = "Analysis dZ/dt" if recording.icg_kind == "impedance" else f"Analysis {source_label}"
    axes[3].plot(time, processed.analysis_dzdt[start:stop], color=colors["analysis"], linewidth=0.8, label=analysis_label)
    if valid_c.size:
        axes[3].scatter(recording.time_s[valid_c], processed.analysis_dzdt[valid_c], s=19, marker="s", facecolors="white", edgecolors=colors["c"], linewidths=1.0, label="Valid C")
    if invalid_c.size:
        axes[3].scatter(recording.time_s[invalid_c], processed.analysis_dzdt[invalid_c], s=19, marker="x", color=colors["invalid"], linewidths=1.0, label="Invalid C candidate")
    axes[3].set_ylabel(
        "Analysis dZ/dt\n(source unit/s)" if recording.icg_kind == "impedance"
        else f"Filtered {source_label}\n(source unit)"
    )
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(frameon=False, loc="upper right", ncol=2)
    for axis in axes:
        for r_index in in_window_r:
            axis.axvline(recording.time_s[r_index], color=colors["r"], alpha=0.18, linewidth=0.55)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        f"{recording.dataset_id} | {recording.subject_id} | {recording.recording_id}\n"
        f"Native fs={recording.sampling_rate_hz:.3f} Hz; R=Pan–Tompkins; C={config.c_method}",
        fontsize=9,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=config.plot_dpi, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return float(recording.time_s[start]), float(recording.time_s[stop - 1])


def _process_pilot_recording(
    spec: DatasetSpec, selection: dict[str, str], project_root: Path,
    plots_dir: Path, config: Phase2Config,
) -> tuple[dict, list[dict]]:
    recording = load_recording(spec, Path(selection["source_file"]), project_root)
    structural_errors = recording.validation_errors()
    if structural_errors:
        raise ValueError(";".join(structural_errors))
    processed = preprocess_recording(recording, config)
    beat_rows: list[dict] = []
    for beat_index in range(max(0, processed.r_indices.size - 1)):
        r_index = int(processed.r_indices[beat_index])
        next_r = int(processed.r_indices[beat_index + 1])
        detection = detect_c_point(processed, r_index, next_r, config)
        rr_ms = float((recording.time_s[next_r] - recording.time_s[r_index]) * 1000.0)
        beat_rows.append({
            "dataset_id": recording.dataset_id,
            "cohort": recording.cohort,
            "condition": recording.condition,
            "subject_id": recording.subject_id,
            "recording_id": recording.recording_id,
            "beat_index": beat_index + 1,
            "r_sample": r_index,
            "next_r_sample": next_r,
            "r_time_s": f"{recording.time_s[r_index]:.9f}",
            "next_r_time_s": f"{recording.time_s[next_r]:.9f}",
            "rr_ms": f"{rr_ms:.6f}",
            "heart_rate_bpm": f"{60000.0 / rr_ms:.6f}",
            "c_sample": detection.sample_index if detection.sample_index is not None else "",
            "c_time_s": f"{detection.time_s:.9f}" if detection.time_s is not None else "",
            "r_to_c_ms": f"{detection.rc_ms:.6f}" if detection.rc_ms is not None else "",
            "c_status": detection.status,
            "c_reason": detection.reason,
            "r_source": config.r_detector,
            "c_method": config.c_method,
        })

    safe_name = "__".join((recording.dataset_id, recording.subject_id.replace(":", "_"), recording.recording_id))
    windows: list[tuple[str, tuple[int, int]]] = [("overview", _plot_window(processed, config))]
    if processed.r_indices.size >= 2:
        longest_rr_position = int(np.argmax(np.diff(processed.r_indices)))
        windows.append(("longest_rr", _window_around_r_position(processed, longest_rr_position, config)))
    first_invalid = next((row for row in beat_rows if row["c_status"] == "INVALID"), None)
    if first_invalid is not None:
        windows.append((
            "first_invalid_c",
            _window_around_r_position(processed, int(first_invalid["beat_index"]) - 1, config),
        ))
    unique_windows: list[tuple[str, tuple[int, int]]] = []
    used_windows: set[tuple[int, int]] = set()
    for label, window in windows:
        if window not in used_windows:
            used_windows.add(window)
            unique_windows.append((label, window))
    plot_files = []
    plot_start = plot_stop = 0.0
    for position, (label, window) in enumerate(unique_windows):
        output_base = plots_dir / f"{safe_name}__{label}"
        start_time, stop_time = _plot_recording(
            processed, beat_rows, output_base, config, window,
        )
        plot_files.extend((str(output_base.with_suffix(".png")), str(output_base.with_suffix(".pdf"))))
        if position == 0:
            plot_start, plot_stop = start_time, stop_time
    rr = np.diff(recording.time_s[processed.r_indices]) * 1000.0
    heart_rate = 60000.0 / rr if rr.size else np.array([])
    valid_c = sum(row["c_status"] == "VALID" for row in beat_rows)
    invalid_c = len(beat_rows) - valid_c
    reasons = list(recording.warnings)
    if processed.r_detection_reason:
        reasons.append(processed.r_detection_reason)
    reasons.extend(sorted({row["c_reason"] for row in beat_rows if row["c_reason"]}))

    provided_r = recording.references.get("heartcycle_ecg_imp_r_time_s")
    provided_count = 0
    matched_count = 0
    r_differences = np.array([])
    if provided_r is not None:
        reference_times = np.asarray(provided_r.values, dtype=float)
        provided_count = int(reference_times.size)
        matched_count, r_differences = _match_reference_r(
            recording.time_s[processed.r_indices], reference_times,
            config.reference_r_match_tolerance_ms / 1000.0,
        )
    status = "FAILED" if processed.r_indices.size < 2 else "WARNING" if reasons or invalid_c else "PASSED"
    record_row = {
        "dataset_id": recording.dataset_id,
        "cohort": recording.cohort,
        "condition": recording.condition,
        "selection_reason": selection["selection_reason"],
        "subject_id": recording.subject_id,
        "recording_id": recording.recording_id,
        "source_file": str(recording.source_path),
        "sampling_rate_hz": f"{recording.sampling_rate_hz:.9f}",
        "timestamp_source": recording.timestamp_source,
        "sample_count": recording.sample_count,
        "duration_s": f"{recording.duration_s:.9f}",
        "r_count": processed.r_indices.size,
        "complete_beat_count": len(beat_rows),
        "valid_c_count": valid_c,
        "invalid_c_count": invalid_c,
        "c_coverage": f"{valid_c / len(beat_rows):.6f}" if beat_rows else "0.000000",
        "mean_rr_ms": f"{np.mean(rr):.6f}" if rr.size else "",
        "sd_rr_ms": f"{np.std(rr, ddof=1):.6f}" if rr.size > 1 else "",
        "minimum_rr_ms": f"{np.min(rr):.6f}" if rr.size else "",
        "maximum_rr_ms": f"{np.max(rr):.6f}" if rr.size else "",
        "maximum_detected_hr_bpm": f"{np.max(heart_rate):.6f}" if heart_rate.size else "",
        "provided_r_count": provided_count,
        "matched_provided_r_count": matched_count,
        "mean_abs_r_difference_ms": f"{np.mean(np.abs(r_differences)) * 1000.0:.6f}" if r_differences.size else "",
        "maximum_abs_r_difference_ms": f"{np.max(np.abs(r_differences)) * 1000.0:.6f}" if r_differences.size else "",
        "plot_start_s": f"{plot_start:.9f}",
        "plot_stop_s": f"{plot_stop:.9f}",
        "plot_files": ";".join(plot_files),
        "status": status,
        "reason_codes": ";".join(dict.fromkeys(reasons)),
    }
    return record_row, beat_rows


def _summaries(recording_rows: list[dict], beat_rows: list[dict]) -> list[dict]:
    summaries = []
    for dataset_id in dict.fromkeys(row["dataset_id"] for row in recording_rows):
        records = [row for row in recording_rows if row["dataset_id"] == dataset_id]
        beats = [row for row in beat_rows if row["dataset_id"] == dataset_id]
        valid = sum(row["c_status"] == "VALID" for row in beats)
        rates = np.array([float(row["heart_rate_bpm"]) for row in beats], dtype=float)
        issue_codes = sorted({
            code for row in records for code in row["reason_codes"].split(";") if code
        })
        summaries.append({
            "dataset_id": dataset_id,
            "cohort": records[0]["cohort"],
            "condition": records[0]["condition"],
            "pilot_recordings": len(records),
            "passed_recordings": sum(row["status"] == "PASSED" for row in records),
            "warning_recordings": sum(row["status"] == "WARNING" for row in records),
            "failed_recordings": sum(row["status"] == "FAILED" for row in records),
            "detected_r_peaks": sum(int(row["r_count"]) for row in records),
            "complete_beats": len(beats),
            "valid_c_beats": valid,
            "invalid_c_beats": len(beats) - valid,
            "c_coverage": f"{valid / len(beats):.6f}" if beats else "0.000000",
            "mean_heart_rate_bpm": f"{np.mean(rates):.6f}" if rates.size else "",
            "minimum_heart_rate_bpm": f"{np.min(rates):.6f}" if rates.size else "",
            "maximum_heart_rate_bpm": f"{np.max(rates):.6f}" if rates.size else "",
            "issue_codes": ";".join(issue_codes),
        })
    return summaries


def _write_report(
    output_dir: Path, summaries: list[dict], recording_rows: list[dict],
    config: Phase2Config,
) -> None:
    systematic_c_failures = [
        row for row in summaries
        if int(row["invalid_c_beats"]) > int(row["valid_c_beats"])
    ]
    failed_recordings = [row for row in recording_rows if row["status"] == "FAILED"]
    gate_met = not systematic_c_failures and not failed_recordings
    heart_rows = [row for row in recording_rows if row["dataset_id"] == "heartcycle_healthy"]
    heart_provided = sum(int(row["provided_r_count"]) for row in heart_rows)
    heart_matched = sum(int(row["matched_provided_r_count"]) for row in heart_rows)
    lines = [
        "# Phase 2 Pilot Preprocessing and R/C Validation Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        "## Frozen methods",
        "",
        f"- ICG/source filtering: {config.icg_filter_order}rd-order Butterworth "
        f"{config.icg_filter_low_hz:g}-{config.icg_filter_high_hz:g} Hz, zero phase.",
        "- R detection: native-rate offline Pan-Tompkins adaptation; provided HeartCycle R events are QC references only.",
        f"- C detection: `{config.c_method}`; global analysis-dZ/dt maximum in R to R+300 ms, valid only at 60-200 ms after R.",
        "- No B/X, biomarker, variability, spectral, feature, or modelling analysis was run.",
        "",
        "## Pilot results",
        "",
        "| Dataset/condition | Recordings | R peaks | Complete beats | Valid C | Invalid C | C coverage | HR range (bpm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset_id']} | {row['pilot_recordings']} | {row['detected_r_peaks']} | "
            f"{row['complete_beats']} | {row['valid_c_beats']} | {row['invalid_c_beats']} | "
            f"{float(row['c_coverage']):.1%} | {row['minimum_heart_rate_bpm']}–{row['maximum_heart_rate_bpm']} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- This is a deliberately small pilot and is not a dataset-level performance estimate.",
        "- Pan-Tompkins was adapted to each native sampling rate; zero-phase Butterworth filtering replaces the original 200 Hz causal integer filter and is explicitly recorded as an offline adaptation.",
        "- ReBeat reference sample origin remains unresolved, so expert C timing accuracy is not claimed.",
        "- IITB has no expert B/X/C annotations in the uploaded files; its C results are plausibility/QC results only.",
        "- HeartCycle supplied R events were not used to generate detections; their matching statistics are reference-QC only.",
        "",
        "## Visual and reference QC findings",
        "",
        f"- HeartCycle: {heart_matched}/{heart_provided} supplied reference R events matched an independently detected R within the frozen 120 ms QC tolerance. Individual timing errors remain in `pilot_recording_qc.csv`.",
        "- ReBeat baseline and cognitive-workload overview/longest-RR windows showed visually aligned R detections. Expert C accuracy is not claimed because the annotation sample-index origin remains unresolved.",
        "- IITB normal and CVD windows were generally plausible, but isolated invalid C candidates and long RR gaps remain visible and retained in QC.",
        "- IITB post-exercise early ECG windows showed visually aligned high-rate R detections. However, the frozen global-maximum C rule repeatedly selected an ICG peak less than 60 ms after R, causing 269/297 beats to fail C timing QC.",
        "- IITB post-exercise MN's longest-RR window contains severe saturation, discontinuities, and near-flat intervals in both recorded channels. Detections in that interval cannot be interpreted as physiological beats and were not silently removed.",
        "",
        "## Gate",
        "",
        f"**Status: {'MET' if gate_met else 'NOT MET'}.**",
        "",
        "Full-dataset processing is blocked because the frozen C rule fails systematically in the IITB post-exercise pilot. The IITB channel/sign mapping and handling of saturated intervals require confirmation. ReBeat annotation origin must also be resolved before annotation-based C accuracy can be reported.",
        "",
    ])
    (output_dir / "PHASE2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase2(project_root: Path, output_dir: Path) -> dict[str, int]:
    """Run only the approved Phase 2 pilot and write structured outputs."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    config = Phase2Config()
    inventory_path = project_root / "outputs" / "phase1" / "recording_inventory.csv"
    if not inventory_path.is_file():
        raise FileNotFoundError("Phase 1 recording inventory is required before Phase 2")
    selections = select_pilot_rows(inventory_path)
    specs = {spec.dataset_id: spec for spec in dataset_manifest(project_root)}
    recording_rows: list[dict] = []
    beat_rows: list[dict] = []
    plots_dir = output_dir / "plots"
    for selection in selections:
        record_row, rows = _process_pilot_recording(
            specs[selection["dataset_id"]], selection, project_root, plots_dir, config,
        )
        recording_rows.append(record_row)
        beat_rows.extend(rows)

    summaries = _summaries(recording_rows, beat_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase2_preprocessing_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8",
    )
    _write_csv(output_dir / "pilot_recording_qc.csv", RECORDING_COLUMNS, recording_rows)
    _write_csv(output_dir / "pilot_beat_qc.csv", BEAT_COLUMNS, beat_rows)
    _write_csv(output_dir / "dataset_condition_summary.csv", SUMMARY_COLUMNS, summaries)
    _write_report(output_dir, summaries, recording_rows, config)
    return {
        "pilot_recordings": len(recording_rows),
        "pilot_beats": len(beat_rows),
        "failed_recordings": sum(row["status"] == "FAILED" for row in recording_rows),
        "invalid_c_beats": sum(row["c_status"] == "INVALID" for row in beat_rows),
        "plots": sum(
            path.endswith(".png")
            for row in recording_rows for path in row["plot_files"].split(";")
        ),
    }

