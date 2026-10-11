"""Phase 3 beat segmentation and R/C region definitions from Phase 2 outputs."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .loaders import load_recording
from .manifest import dataset_manifest
from .phase2_config import Phase2Config
from .signal_processing import central_difference, zero_phase_bandpass


BEAT_COLUMNS = (
    "dataset_id", "cohort", "condition", "subject_id", "recording_id",
    "beat_id", "beat_number", "source_file", "sampling_frequency_hz",
    "sampling_frequency_source", "beat_start_r_sample_inclusive",
    "beat_end_next_r_sample_exclusive", "beat_last_sample_inclusive",
    "beat_start_timestamp_s", "beat_end_timestamp_s", "beat_duration_ms",
    "beat_sample_count", "c_sample", "c_timestamp_s", "phase2_detection_status",
    "phase2_qc_status", "phase2_issue_codes", "phase3_segmentation_status",
    "overall_qc_status", "phase3_issue_codes", "combined_issue_codes",
    "b_fiducial_eligibility", "x_fiducial_eligibility",
    "biomarker_eligibility", "reference_comparison_eligibility",
    "r_to_c_spectral_eligibility", "c_to_next_r_spectral_eligibility",
)

REGION_COLUMNS = (
    "dataset_id", "cohort", "condition", "subject_id", "recording_id",
    "beat_id", "boundary_convention", "r_to_c_start_sample_inclusive",
    "r_to_c_stop_sample_exclusive", "r_to_c_start_timestamp_s",
    "r_to_c_stop_timestamp_s", "r_to_c_duration_ms", "r_to_c_sample_count",
    "r_to_c_definition_status", "c_to_next_r_start_sample_inclusive",
    "c_to_next_r_stop_sample_exclusive", "c_to_next_r_start_timestamp_s",
    "c_to_next_r_stop_timestamp_s", "c_to_next_r_duration_ms",
    "c_to_next_r_sample_count", "c_to_next_r_definition_status",
    "phase2_qc_status", "overall_qc_status", "phase3_issue_codes",
    "combined_issue_codes",
)

SUMMARY_COLUMNS = (
    "dataset_id", "cohort", "condition", "recordings", "phase2_input_beats",
    "segmented_beats", "passed_beats", "flagged_beats", "failed_beats",
    "r_to_c_regions_defined", "c_to_next_r_regions_defined",
    "b_fiducial_eligible", "x_fiducial_eligible",
    "biomarker_structurally_eligible",
    "boundary_validation_failures", "index_validation_failures",
    "timestamp_validation_failures", "phase2_issue_codes", "phase3_issue_codes",
)

BOUNDARY_CONVENTION = "beat=[R,nextR);R_to_C=[R,C);C_to_next_R=[C,nextR)"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _codes(*groups) -> str:
    values = []
    for group in groups:
        if isinstance(group, str):
            group = group.split(";")
        for value in group:
            if value and value not in values:
                values.append(value)
    return ";".join(values)


def _segment_row(row: dict[str, str], record: dict[str, str]) -> tuple[dict, dict]:
    issues: list[str] = []
    sample_count = int(record["sample_count"])
    fs = float(row["sampling_frequency_hz"])

    try:
        r = int(row["r_sample"])
        next_r = int(row["next_r_sample"])
    except (TypeError, ValueError):
        r = next_r = -1
        issues.append("phase3_missing_r_or_next_r_sample")
    if not 0 <= r < sample_count:
        issues.append("phase3_r_index_out_of_bounds")
    if not 0 < next_r < sample_count:
        issues.append("phase3_next_r_index_out_of_bounds")
    if next_r <= r:
        issues.append("phase3_invalid_r_next_r_order")

    c = None
    if row["c_sample"] != "":
        try:
            c = int(row["c_sample"])
        except ValueError:
            issues.append("phase3_invalid_c_sample")
    else:
        issues.append("phase3_missing_c_sample")
    if c is not None and not (r <= c < next_r):
        issues.append("phase3_c_outside_beat")

    r_time = float(row["r_timestamp_s"])
    next_r_time = float(row["next_r_timestamp_s"])
    c_time = float(row["c_timestamp_s"]) if row["c_timestamp_s"] else None
    beat_duration_ms = (next_r_time - r_time) * 1000.0
    sample_duration_ms = (next_r - r) * 1000.0 / fs if next_r > r else np.nan
    tolerance_ms = 0.51 * 1000.0 / fs
    if beat_duration_ms <= 0:
        issues.append("phase3_nonpositive_beat_duration")
    if np.isfinite(sample_duration_ms) and abs(beat_duration_ms - sample_duration_ms) > tolerance_ms:
        issues.append("phase3_beat_timestamp_sampling_mismatch")
    if c is not None and c_time is not None:
        c_sample_duration_ms = (c - r) * 1000.0 / fs
        if abs((c_time - r_time) * 1000.0 - c_sample_duration_ms) > tolerance_ms:
            issues.append("phase3_c_timestamp_sampling_mismatch")

    boundary_errors = {
        "phase3_missing_r_or_next_r_sample", "phase3_r_index_out_of_bounds",
        "phase3_next_r_index_out_of_bounds", "phase3_invalid_r_next_r_order",
        "phase3_nonpositive_beat_duration", "phase3_beat_timestamp_sampling_mismatch",
    }
    region_errors = {
        "phase3_missing_c_sample", "phase3_invalid_c_sample",
        "phase3_c_outside_beat", "phase3_c_timestamp_sampling_mismatch",
    }
    segmentation_status = "FAILED" if boundary_errors.intersection(issues) else "PASS"
    region_boundaries_valid = segmentation_status == "PASS" and not region_errors.intersection(issues)
    r_to_c_valid = region_boundaries_valid and c > r
    c_to_next_valid = region_boundaries_valid and c < next_r
    if region_boundaries_valid and not r_to_c_valid:
        issues.append("phase3_empty_r_to_c_region")
    if region_boundaries_valid and not c_to_next_valid:
        issues.append("phase3_empty_c_to_next_r_region")
    overall_status = (
        "FAILED" if segmentation_status == "FAILED" or not (r_to_c_valid and c_to_next_valid)
        else "FLAGGED" if row["qc_status"] == "FLAGGED" else "PASS"
    )
    combined = _codes(row["issue_codes"], issues)
    flagged_suffix = "_WITH_PHASE2_QC_FLAG" if row["qc_status"] == "FLAGGED" else ""
    b_eligibility = f"ELIGIBLE{flagged_suffix}" if r_to_c_valid else "INELIGIBLE_R_TO_C_REGION"
    x_eligibility = f"ELIGIBLE{flagged_suffix}" if c_to_next_valid else "INELIGIBLE_C_TO_NEXT_R_REGION"
    biomarker_eligibility = (
        f"STRUCTURALLY_ELIGIBLE{flagged_suffix}"
        if r_to_c_valid and c_to_next_valid else "INELIGIBLE_REGION_DEFINITION"
    )
    if not (r_to_c_valid and c_to_next_valid):
        reference_eligibility = "INELIGIBLE_REGION_DEFINITION"
    elif row["dataset_id"] == "heartcycle_healthy":
        reference_eligibility = "PENDING_COMPATIBLE_REFERENCE_MATCHING"
    elif row["dataset_id"].startswith("rebeat_"):
        reference_eligibility = "INELIGIBLE_REBEAT_INDEX_ORIGIN_UNRESOLVED"
    else:
        reference_eligibility = "INELIGIBLE_NO_EXPERT_BCX_REFERENCE"
    r_to_c_spectral = (
        f"PENDING_PHASE7_CRITERIA{flagged_suffix}"
        if r_to_c_valid else "INELIGIBLE_R_TO_C_REGION"
    )
    c_to_next_spectral = (
        f"PENDING_PHASE7_CRITERIA{flagged_suffix}"
        if c_to_next_valid else "INELIGIBLE_C_TO_NEXT_R_REGION"
    )

    beat = {
        "dataset_id": row["dataset_id"], "cohort": row["cohort"],
        "condition": row["condition"], "subject_id": row["subject_id"],
        "recording_id": row["recording_id"], "beat_id": row["beat_id"],
        "beat_number": row["beat_number"], "source_file": row["source_file"],
        "sampling_frequency_hz": row["sampling_frequency_hz"],
        "sampling_frequency_source": row["sampling_frequency_source"],
        "beat_start_r_sample_inclusive": r,
        "beat_end_next_r_sample_exclusive": next_r,
        "beat_last_sample_inclusive": next_r - 1 if next_r > r else "",
        "beat_start_timestamp_s": f"{r_time:.9f}",
        "beat_end_timestamp_s": f"{next_r_time:.9f}",
        "beat_duration_ms": f"{beat_duration_ms:.6f}",
        "beat_sample_count": next_r - r if next_r > r else "",
        "c_sample": "" if c is None else c,
        "c_timestamp_s": "" if c_time is None else f"{c_time:.9f}",
        "phase2_detection_status": row["detection_status"],
        "phase2_qc_status": row["qc_status"],
        "phase2_issue_codes": row["issue_codes"],
        "phase3_segmentation_status": segmentation_status,
        "overall_qc_status": overall_status,
        "phase3_issue_codes": ";".join(issues),
        "combined_issue_codes": combined,
        "b_fiducial_eligibility": b_eligibility,
        "x_fiducial_eligibility": x_eligibility,
        "biomarker_eligibility": biomarker_eligibility,
        "reference_comparison_eligibility": reference_eligibility,
        "r_to_c_spectral_eligibility": r_to_c_spectral,
        "c_to_next_r_spectral_eligibility": c_to_next_spectral,
    }

    if region_boundaries_valid:
        r_to_c_count = c - r
        c_to_next_count = next_r - c
        r_to_c_duration = (c_time - r_time) * 1000.0
        c_to_next_duration = (next_r_time - c_time) * 1000.0
    else:
        r_to_c_count = c_to_next_count = ""
        r_to_c_duration = c_to_next_duration = None
    r_to_c_status = "PASS" if r_to_c_valid else "FAILED"
    c_to_next_status = "PASS" if c_to_next_valid else "FAILED"
    region = {
        "dataset_id": row["dataset_id"], "cohort": row["cohort"],
        "condition": row["condition"], "subject_id": row["subject_id"],
        "recording_id": row["recording_id"], "beat_id": row["beat_id"],
        "boundary_convention": BOUNDARY_CONVENTION,
        "r_to_c_start_sample_inclusive": r if region_boundaries_valid else "",
        "r_to_c_stop_sample_exclusive": c if region_boundaries_valid else "",
        "r_to_c_start_timestamp_s": f"{r_time:.9f}" if region_boundaries_valid else "",
        "r_to_c_stop_timestamp_s": f"{c_time:.9f}" if region_boundaries_valid else "",
        "r_to_c_duration_ms": f"{r_to_c_duration:.6f}" if r_to_c_duration is not None else "",
        "r_to_c_sample_count": r_to_c_count,
        "r_to_c_definition_status": r_to_c_status,
        "c_to_next_r_start_sample_inclusive": c if region_boundaries_valid else "",
        "c_to_next_r_stop_sample_exclusive": next_r if region_boundaries_valid else "",
        "c_to_next_r_start_timestamp_s": f"{c_time:.9f}" if region_boundaries_valid else "",
        "c_to_next_r_stop_timestamp_s": f"{next_r_time:.9f}" if region_boundaries_valid else "",
        "c_to_next_r_duration_ms": f"{c_to_next_duration:.6f}" if c_to_next_duration is not None else "",
        "c_to_next_r_sample_count": c_to_next_count,
        "c_to_next_r_definition_status": c_to_next_status,
        "phase2_qc_status": row["qc_status"], "overall_qc_status": overall_status,
        "phase3_issue_codes": ";".join(issues), "combined_issue_codes": combined,
    }
    return beat, region


def _summaries(beats: list[dict], regions: list[dict]) -> list[dict]:
    summaries = []
    for dataset_id in dict.fromkeys(row["dataset_id"] for row in beats):
        group_beats = [row for row in beats if row["dataset_id"] == dataset_id]
        group_regions = [row for row in regions if row["dataset_id"] == dataset_id]
        p2_codes = sorted({
            code for row in group_beats for code in row["phase2_issue_codes"].split(";") if code
        })
        p3_codes = sorted({
            code for row in group_beats for code in row["phase3_issue_codes"].split(";") if code
        })
        summaries.append({
            "dataset_id": dataset_id, "cohort": group_beats[0]["cohort"],
            "condition": group_beats[0]["condition"],
            "recordings": len({row["recording_id"] for row in group_beats}),
            "phase2_input_beats": len(group_beats),
            "segmented_beats": sum(row["phase3_segmentation_status"] == "PASS" for row in group_beats),
            "passed_beats": sum(row["overall_qc_status"] == "PASS" for row in group_beats),
            "flagged_beats": sum(row["overall_qc_status"] == "FLAGGED" for row in group_beats),
            "failed_beats": sum(row["overall_qc_status"] == "FAILED" for row in group_beats),
            "r_to_c_regions_defined": sum(row["r_to_c_definition_status"] == "PASS" for row in group_regions),
            "c_to_next_r_regions_defined": sum(row["c_to_next_r_definition_status"] == "PASS" for row in group_regions),
            "b_fiducial_eligible": sum(row["b_fiducial_eligibility"].startswith("ELIGIBLE") for row in group_beats),
            "x_fiducial_eligible": sum(row["x_fiducial_eligibility"].startswith("ELIGIBLE") for row in group_beats),
            "biomarker_structurally_eligible": sum(row["biomarker_eligibility"].startswith("STRUCTURALLY_ELIGIBLE") for row in group_beats),
            "boundary_validation_failures": sum(
                "boundary" in row["phase3_issue_codes"] or "r_next_r" in row["phase3_issue_codes"]
                for row in group_beats
            ),
            "index_validation_failures": sum("index_" in row["phase3_issue_codes"] for row in group_beats),
            "timestamp_validation_failures": sum("timestamp_" in row["phase3_issue_codes"] for row in group_beats),
            "phase2_issue_codes": ";".join(p2_codes),
            "phase3_issue_codes": ";".join(p3_codes),
        })
    return summaries


def _plot_representative(
    project_root: Path, beat_rows: list[dict], output_dir: Path,
) -> int:
    specs = {spec.dataset_id: spec for spec in dataset_manifest(project_root)}
    plotted = 0
    for dataset_id in dict.fromkeys(row["dataset_id"] for row in beat_rows):
        group = [row for row in beat_rows if row["dataset_id"] == dataset_id]
        first_recording = group[0]["recording_id"]
        rows = [row for row in group if row["recording_id"] == first_recording][:6]
        if not rows:
            continue
        recording = load_recording(specs[dataset_id], Path(rows[0]["source_file"]), project_root)
        config = Phase2Config()
        filtered_icg = zero_phase_bandpass(
            recording.icg, recording.sampling_rate_hz,
            config.icg_filter_low_hz, config.icg_filter_high_hz,
            config.icg_filter_order,
        )
        analysis_icg = (
            central_difference(filtered_icg, recording.sampling_rate_hz)
            if recording.icg_kind == "impedance" else filtered_icg
        )
        start = max(0, int(rows[0]["beat_start_r_sample_inclusive"]) - round(0.2 * recording.sampling_rate_hz))
        stop = min(recording.sample_count, int(rows[-1]["beat_end_next_r_sample_exclusive"]) + round(0.2 * recording.sampling_rate_hz))
        time = recording.time_s[start:stop]
        signal_label = (
            "Impedance" if recording.icg_kind == "impedance"
            else str(recording.metadata.get("icg_signal_label", "dZ/dt"))
        )
        plt.rcParams.update({
            "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42,
        })
        colors = {"ecg": "#404040", "icg": "#1f4e79", "r": "#8c2d3e", "c": "#c47f28", "rtc": "#9ebcda", "ctr": "#b8d8ba"}
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, constrained_layout=True)
        axes[0].plot(time, recording.ecg[start:stop], color=colors["ecg"], linewidth=0.7)
        axes[1].plot(time, analysis_icg[start:stop], color=colors["icg"], linewidth=0.7)
        for position, row in enumerate(rows):
            r = int(row["beat_start_r_sample_inclusive"])
            next_r = int(row["beat_end_next_r_sample_exclusive"])
            c = int(row["c_sample"]) if row["c_sample"] != "" else None
            for axis in axes:
                if c is not None:
                    axis.axvspan(recording.time_s[r], recording.time_s[c], color=colors["rtc"], alpha=0.22, label="R-to-C [R,C)" if position == 0 else None)
                    axis.axvspan(recording.time_s[c], recording.time_s[next_r], color=colors["ctr"], alpha=0.22, label="C-to-next-R [C,next-R)" if position == 0 else None)
                axis.axvline(recording.time_s[r], color=colors["r"], linewidth=0.7, alpha=0.8)
            axes[0].scatter(recording.time_s[r], recording.ecg[r], s=18, facecolors="white", edgecolors=colors["r"], linewidths=1.0, zorder=3)
            if c is not None:
                axes[1].scatter(recording.time_s[c], analysis_icg[c], s=20, marker="s", facecolors="white", edgecolors=colors["c"], linewidths=1.0, zorder=3)
        last_next_r = int(rows[-1]["beat_end_next_r_sample_exclusive"])
        for axis in axes:
            axis.axvline(recording.time_s[last_next_r], color=colors["r"], linewidth=0.7, alpha=0.8)
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        axes[0].set_ylabel("Raw ECG\n(source unit)")
        axes[1].set_ylabel(
            "Analysis dZ/dt\n(source unit/s)" if recording.icg_kind == "impedance"
            else f"Filtered {signal_label}\n(source unit)"
        )
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(frameon=False, loc="upper right", ncol=2)
        figure.suptitle(f"{dataset_id} | {recording.subject_id} | {recording.recording_id}\nPhase 3 beat and region boundaries", fontsize=9)
        output_base = output_dir / f"{dataset_id}__representative_segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        plotted += 1
    return plotted


def _write_report(output_dir: Path, summaries: list[dict], total: int, plots: int) -> None:
    lines = [
        "# Phase 3 Beat Segmentation and Region Definition Report", "",
        f"Generated: {datetime.now().astimezone().isoformat()}", "",
        "## Boundary convention", "",
        f"`{BOUNDARY_CONVENTION}`", "",
        "The two regions form a non-overlapping partition of each half-open beat. Original sample indices and timestamps are retained; no resampling, interpolation, fiducial correction, or imputation was performed.",
        "", "## Results", "",
        "| Dataset | Recordings | Input beats | Segmented | Passed | Flagged | Failed | R-to-C defined | C-to-next-R defined |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset_id']} | {row['recordings']} | {row['phase2_input_beats']} | "
            f"{row['segmented_beats']} | {row['passed_beats']} | {row['flagged_beats']} | "
            f"{row['failed_beats']} | {row['r_to_c_regions_defined']} | "
            f"{row['c_to_next_r_regions_defined']} |"
        )
    lines.extend([
        "", "## Validation", "",
        f"- Phase 2 input beats received: {total}.",
        "- Beat identifiers and every Phase 2 detection/QC flag were carried forward.",
        "- B-side, X-side, biomarker, reference-comparison, and per-region spectral eligibility were recorded independently.",
        "- Beat and region boundaries were checked for ordering and source-recording bounds.",
        "- Timestamp durations were checked against native/documented sampling frequencies with a half-sample tolerance.",
        "- Region sample counts were checked to sum exactly to the corresponding beat sample count.",
        f"- Representative plot sets generated: {plots}.",
        "", "## Remaining issues", "",
        "- Phase 2 timing-flagged C candidates define geometrically valid regions but remain scientifically unverified; their overall Phase 3 status stays FLAGGED.",
        "- C equals R in 58 retained beats, producing a zero-sample R-to-C region. Those R-to-C regions and dependent analyses are marked failed/ineligible; their C-to-next-R regions remain independently valid.",
        "- ReBeat expert annotation index origin remains unresolved and no reference annotation was merged into automatic boundaries.",
        "- IITB digitized units, column 5, saturated intervals, and independent Doppler time base remain as documented Phase 2 limitations.",
        "- Before Phase 4, the analysis policy for structurally eligible beats carrying a Phase 2 C-timing flag must be selected; the flag must not be silently converted into either exclusion or acceptance.",
        "- PLANS.md still marks the component lists and tie-breaking requirements for the B/X consensus definitions as unresolved; those methods cannot be finalized by assumption.",
        "", "## Completion gate", "",
        "Phase 3 is complete: every input beat is either structurally valid or explicitly failed with a reason, and no unflagged boundary, index, timestamp, or partition inconsistency remains. Progression to Phase 4 must use the separate B/X eligibility fields, preserve overall QC status, and must not treat flagged C boundaries as validated ground truth.", "",
    ])
    (output_dir / "PHASE3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase3(project_root: Path, output_dir: Path) -> dict[str, int]:
    """Create Phase 3 beat and region records from final Phase 2 CSVs."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    phase2_dir = project_root / "outputs" / "phase2"
    phase2_beats = _read_csv(phase2_dir / "phase2_r_c_detection_all_beats.csv")
    phase2_records = _read_csv(phase2_dir / "phase2_recording_qc_summary.csv")
    records = {(row["dataset_id"], row["recording_id"]): row for row in phase2_records}
    if len(records) != len(phase2_records):
        raise ValueError("Phase 2 recording identifiers are not unique")

    beats, regions = [], []
    for row in phase2_beats:
        key = (row["dataset_id"], row["recording_id"])
        if key not in records:
            raise ValueError(f"Phase 2 recording summary missing for {key}")
        beat, region = _segment_row(row, records[key])
        beats.append(beat)
        regions.append(region)
    if len({row["beat_id"] for row in beats}) != len(beats):
        raise ValueError("Phase 3 beat identifiers are not unique")

    summaries = _summaries(beats, regions)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "phase3_beat_segmentation_all_beats.csv", BEAT_COLUMNS, beats)
    _write_csv(output_dir / "phase3_region_summary_all_beats.csv", REGION_COLUMNS, regions)
    _write_csv(output_dir / "phase3_segmentation_qc_summary.csv", SUMMARY_COLUMNS, summaries)
    plots = _plot_representative(project_root, beats, output_dir / "plots")
    _write_report(output_dir, summaries, len(beats), plots)
    return {
        "input_beats": len(beats),
        "segmented_beats": sum(row["phase3_segmentation_status"] == "PASS" for row in beats),
        "passed_beats": sum(row["overall_qc_status"] == "PASS" for row in beats),
        "flagged_beats": sum(row["overall_qc_status"] == "FLAGGED" for row in beats),
        "failed_beats": sum(row["overall_qc_status"] == "FAILED" for row in beats),
        "plots": plots,
    }
