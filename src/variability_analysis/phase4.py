"""Phase 4 B/X candidate detection from validated Phase 3 beat records."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bx_candidates import BXSignals, Candidate, DEFINITIONS, DETECTORS
from .loaders import load_recording
from .manifest import dataset_manifest
from .phase2_config import Phase2Config
from .signal_processing import central_difference, zero_phase_bandpass


CANDIDATE_COLUMNS = (
    "dataset_id", "cohort", "condition", "subject_id", "recording_id",
    "beat_id", "candidate_name", "candidate_type", "candidate_sample_index",
    "candidate_timestamp_s", "detection_status", "qc_status", "associated_region",
    "r_sample", "c_sample", "next_r_sample", "search_start_sample",
    "search_stop_sample_exclusive", "timestamp_representation",
    "phase2_qc_status", "phase3_overall_qc_status", "qc_issue_codes",
)

SUMMARY_COLUMNS = (
    "dataset_id", "cohort", "condition", "candidate_name", "candidate_type",
    "total_beat_rows", "eligible_beats", "detected_candidates", "missing_candidates",
    "ineligible_beats", "not_implemented_ambiguous", "unavailable_required_metadata",
    "qc_passed_detections", "qc_flagged_detections", "invalid_locations",
    "detection_coverage_among_eligible", "issue_codes",
)

DEFINITION_COLUMNS = (
    "candidate_name", "candidate_type", "category", "signal_representation",
    "exact_definition", "search_region", "parameters", "interpolation_rule",
    "definition_source", "implementation_status", "unresolved_ambiguity",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _codes(*groups) -> str:
    result = []
    for group in groups:
        for code in group.split(";") if isinstance(group, str) else group:
            if code and code not in result:
                result.append(code)
    return ";".join(result)


def _signals(recording) -> BXSignals:
    config = Phase2Config()
    filtered = zero_phase_bandpass(
        recording.icg, recording.sampling_rate_hz,
        config.icg_filter_low_hz, config.icg_filter_high_hz,
        config.icg_filter_order,
    )
    dzdt = (
        central_difference(filtered, recording.sampling_rate_hz)
        if recording.icg_kind == "impedance" else filtered
    )
    d2 = central_difference(dzdt, recording.sampling_rate_hz)
    return BXSignals(
        recording.time_s, dzdt, d2,
        central_difference(d2, recording.sampling_rate_hz),
        recording.sampling_rate_hz,
    )


def _ineligible(reason: str) -> Candidate:
    return Candidate(None, None, "INELIGIBLE", reason, None, None, "none")


def _validate_candidate(
    candidate: Candidate, candidate_type: str, signals: BXSignals,
    r: int, c: int, next_r: int,
) -> tuple[str, str]:
    if candidate.status != "DETECTED":
        return "INVALID", candidate.reason
    index = candidate.sample_index
    timestamp = candidate.timestamp_s
    issues = []
    if index is None or not 0 <= index < signals.time_s.size:
        issues.append("candidate_sample_index_out_of_bounds")
    if timestamp is None or not np.isfinite(timestamp):
        issues.append("candidate_timestamp_missing_or_nonfinite")
    if index is not None and timestamp is not None:
        tolerance = .51 / signals.sampling_rate_hz
        if abs(float(signals.time_s[index]) - timestamp) > tolerance:
            issues.append("candidate_timestamp_sample_mismatch")
        if candidate.search_start_sample is not None and not (
            candidate.search_start_sample <= index < candidate.search_stop_sample_exclusive
        ):
            issues.append("candidate_outside_method_search_window")
        if candidate_type == "B" and not (r < index < c):
            issues.append("B_sample_not_strictly_between_R_and_C")
        if candidate_type == "X" and not (c < index < next_r):
            issues.append("X_sample_not_strictly_between_C_and_next_R")
        if candidate_type == "B" and not (signals.time_s[r] < timestamp < signals.time_s[c]):
            issues.append("B_not_strictly_between_R_and_C")
        if candidate_type == "X" and not (signals.time_s[c] < timestamp < signals.time_s[next_r]):
            issues.append("X_not_strictly_between_C_and_next_R")
    return ("INVALID", ";".join(issues)) if issues else ("PASS", "")


def _definition_rows() -> list[dict[str, str]]:
    rows = []
    for definition in DEFINITIONS:
        rows.append({
            "candidate_name": definition.name,
            "candidate_type": definition.candidate_type,
            "category": definition.category,
            "signal_representation": definition.signal,
            "exact_definition": definition.exact_definition,
            "search_region": definition.search_region,
            "parameters": definition.parameters,
            "interpolation_rule": (
                "Linear interpolation for threshold/zero crossings with nearest native sample index; native index for extrema; formula timestamp with nearest native sample index for fractional rules."
            ),
            "definition_source": "PLANS.md;inspected legacy common/candidate_methods.py and common/fiducials.py",
            "implementation_status": definition.implementation_status,
            "unresolved_ambiguity": definition.unresolved_ambiguity,
        })
    return rows


def _plot_examples(project_root: Path, examples: dict, output_dir: Path) -> int:
    specs = {spec.dataset_id: spec for spec in dataset_manifest(project_root)}
    colors = [
        "#2f6f62", "#8c2d3e", "#c47f28", "#4f6d8a", "#6f4e7c",
        "#8a6d3b", "#557a46", "#76504e", "#3d6b73", "#7a6f55",
        "#4f5d75", "#805d40",
    ]
    count = 0
    for (dataset_id, candidate_type), choices in examples.items():
        if "best" not in choices:
            continue
        panels = [choices["best"]]
        if "difficult" in choices and choices["difficult"]["beat"]["beat_id"] != panels[0]["beat"]["beat_id"]:
            panels.append(choices["difficult"])
        figure, axes = plt.subplots(len(panels), 1, figsize=(8.4, 3.8 * len(panels)), squeeze=False, constrained_layout=True)
        for panel_index, example in enumerate(panels):
            beat = example["beat"]
            recording = load_recording(specs[dataset_id], Path(beat["source_file"]), project_root)
            signals = _signals(recording)
            r = int(beat["beat_start_r_sample_inclusive"])
            c = int(beat["c_sample"])
            nr = int(beat["beat_end_next_r_sample_exclusive"])
            region_start, region_stop = (r, c + 1) if candidate_type == "B" else (c, nr)
            margin = max(1, round(.03 * recording.sampling_rate_hz))
            start, stop = max(0, region_start - margin), min(recording.sample_count, region_stop + margin)
            axis = axes[panel_index, 0]
            axis.plot(recording.time_s[start:stop], signals.dzdt[start:stop], color="#1f4e79", linewidth=.8)
            axis.axvspan(recording.time_s[region_start], recording.time_s[region_stop - 1], color="#d9e7ef", alpha=.45)
            axis.axvline(recording.time_s[r], color="#8c2d3e", linewidth=.8, label="R")
            axis.axvline(recording.time_s[c], color="#c47f28", linewidth=.8, label="C")
            axis.axvline(recording.time_s[nr], color="#8c2d3e", linewidth=.8, linestyle="--", label="next R")
            detected = [item for item in example["results"] if item[1].status == "DETECTED"]
            for number, (definition, candidate, qc_status) in enumerate(detected, start=1):
                index = candidate.sample_index
                axis.scatter(
                    candidate.timestamp_s, signals.dzdt[index], s=25, marker="o",
                    facecolors="white", edgecolors=colors[(number - 1) % len(colors)],
                    linewidths=1.1, zorder=4, label=f"{number}. {definition.name} ({qc_status})",
                )
            axis.set_ylabel(
                "Filtered dZ/dt\n(unverified impedance unit/s)"
                if recording.icg_kind == "impedance" else
                f"Filtered {recording.metadata.get('icg_signal_label', 'dZ/dt')}\n(unit unverified)"
            )
            axis.set_xlabel("Time (s)")
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(
                f"{example['label']}: {beat['subject_id']} | {beat['recording_id']} | {beat['beat_id']}\n"
                f"detected {example['detected_count']}/{example['implemented_count']}; "
                f"location-valid {example['valid_count']}/{example['implemented_count']} implemented {candidate_type} definitions",
                fontsize=8,
            )
            axis.legend(frameon=False, fontsize=5.8, loc="center left", bbox_to_anchor=(1.01, .5))
        figure.suptitle(f"{dataset_id} | Phase 4 {candidate_type}-candidate review", fontsize=10)
        output_dir.mkdir(parents=True, exist_ok=True)
        base = output_dir / f"{dataset_id}__{candidate_type.lower()}_candidate_review"
        figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        count += 1
    return count


def _write_report(output_dir: Path, summaries: list[dict], rows: int, plots: int) -> None:
    implemented_names = {
        definition.name for definition in DEFINITIONS
        if definition.implementation_status == "IMPLEMENTED"
    }
    dataset_totals = defaultdict(lambda: {
        "beats": 0, "eligible": 0, "detected": 0, "missing": 0,
        "passed": 0, "invalid": 0, "flagged": 0,
    })
    for row in summaries:
        if row["candidate_name"] not in implemented_names:
            continue
        totals = dataset_totals[row["dataset_id"]]
        totals["beats"] = max(totals["beats"], int(row["total_beat_rows"]))
        totals["eligible"] += int(row["eligible_beats"])
        totals["detected"] += int(row["detected_candidates"])
        totals["missing"] += int(row["missing_candidates"])
        totals["passed"] += int(row["qc_passed_detections"])
        totals["invalid"] += int(row["invalid_locations"])
        totals["flagged"] += int(row["qc_flagged_detections"])
    lines = [
        "# Phase 4 B/X Candidate Detection and Validation Report", "",
        f"Generated: {datetime.now().astimezone().isoformat()}", "",
        "## Scope", "",
        "Phase 4 used the final Phase 3 beat/R/C boundaries and repeated only the frozen 0.8-30 Hz ICG preprocessing needed to form dZ/dt and its derivatives. R and C were not redetected or changed. Every candidate definition has one output row per beat, including unavailable, ineligible, missing, invalid, and ambiguous results.",
        "", f"Candidate rows written: {rows}", "",
        "| Dataset / condition | Beats | Eligible implemented opportunities | Detected | Coverage | QC pass | Inherited-flag detections | Invalid locations | Missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, values in dataset_totals.items():
        coverage = values["detected"] / values["eligible"] if values["eligible"] else float("nan")
        lines.append(
            f"| {dataset_id} | {values['beats']} | {values['eligible']} | {values['detected']} | "
            f"{coverage:.2%} | {values['passed']} | {values['flagged']} | "
            f"{values['invalid']} | {values['missing']} |"
        )
    lines.extend([
        "", "## Deliberately unresolved definitions", "",
        "- `B_morphology_consensus_40ms`: component list, minimum candidate count, and tie-breaking are not approved.",
        "- `B_adaptive_multirepresentation_consensus_35ms`: signal representations, component list, and aggregation are not approved; no wavelets were introduced.",
        "- `X_morphology_consensus_60ms`: component list, minimum candidate count, and tie-breaking are not approved.",
        "- `X_Weissler_HR`: formula is registered, but verified biological sex is unavailable and the B dependency/source still requires confirmation; results are emitted as unavailable rather than guessed.",
        "", "## Validation", "",
        "- Candidate locations were checked against their method-specific windows and strict R<B<C or C<X<next-R ordering.",
        "- Native sample bounds and interpolated timestamp-to-nearest-sample consistency were checked.",
        "- Phase 2/3 QC and issue codes were carried into every candidate row.",
        "- Reference annotations were not merged with automatic candidates; no unsupported accuracy claim was made.",
        f"- Representative highest-coverage and difficult B/X review figure sets generated: {plots}.",
        "", "## Completion gate", "",
        "Implemented definitions are complete and validated at candidate-generation level. Phase 4 cannot be considered fully frozen until the three consensus definitions and Weissler metadata/dependency questions are resolved and the outputs are reviewed. Phase 5 has not started.", "",
    ])
    (output_dir / "PHASE4_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase4(project_root: Path, output_dir: Path) -> dict[str, int]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase3_rows = _read_csv(project_root / "outputs" / "phase3" / "phase3_beat_segmentation_all_beats.csv")
    grouped = defaultdict(list)
    for row in phase3_rows:
        grouped[(row["dataset_id"], row["recording_id"])].append(row)
    specs = {spec.dataset_id: spec for spec in dataset_manifest(project_root)}
    output_path = output_dir / "phase4_bx_candidate_detection_all_beats.csv"
    summary = defaultdict(lambda: defaultdict(int))
    summary_issues = defaultdict(set)
    examples: dict = defaultdict(dict)
    candidate_rows = 0

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        for (dataset_id, recording_id), beats in grouped.items():
            recording = load_recording(specs[dataset_id], Path(beats[0]["source_file"]), project_root)
            signals = _signals(recording)
            for beat in beats:
                r = int(beat["beat_start_r_sample_inclusive"])
                c = int(beat["c_sample"]) if beat["c_sample"] else None
                nr = int(beat["beat_end_next_r_sample_exclusive"])
                by_type = {"B": [], "X": []}
                for definition in DEFINITIONS:
                    eligibility = beat["b_fiducial_eligibility"] if definition.candidate_type == "B" else beat["x_fiducial_eligibility"]
                    eligible = eligibility.startswith("ELIGIBLE") and c is not None
                    if definition.implementation_status != "IMPLEMENTED":
                        candidate = DETECTORS[definition.name](signals, r, c if c is not None else r, nr)
                    else:
                        candidate = DETECTORS[definition.name](signals, r, c, nr) if eligible else _ineligible(eligibility)
                    validation_status, validation_reason = _validate_candidate(
                        candidate, definition.candidate_type, signals, r, c if c is not None else r, nr,
                    )
                    if candidate.status == "DETECTED":
                        qc_status = "INVALID" if validation_status == "INVALID" else (
                            "FLAGGED" if beat["overall_qc_status"] != "PASS" else "PASS"
                        )
                    else:
                        qc_status = "INVALID"
                    issues = _codes(
                        beat["phase2_issue_codes"], beat["phase3_issue_codes"],
                        candidate.reason, validation_reason,
                    )
                    writer.writerow({
                        "dataset_id": dataset_id, "cohort": beat["cohort"],
                        "condition": beat["condition"], "subject_id": beat["subject_id"],
                        "recording_id": recording_id, "beat_id": beat["beat_id"],
                        "candidate_name": definition.name,
                        "candidate_type": definition.candidate_type,
                        "candidate_sample_index": "" if candidate.sample_index is None else candidate.sample_index,
                        "candidate_timestamp_s": "" if candidate.timestamp_s is None else f"{candidate.timestamp_s:.9f}",
                        "detection_status": candidate.status, "qc_status": qc_status,
                        "associated_region": "R_to_C" if definition.candidate_type == "B" else "C_to_next_R",
                        "r_sample": r, "c_sample": "" if c is None else c,
                        "next_r_sample": nr,
                        "search_start_sample": "" if candidate.search_start_sample is None else candidate.search_start_sample,
                        "search_stop_sample_exclusive": "" if candidate.search_stop_sample_exclusive is None else candidate.search_stop_sample_exclusive,
                        "timestamp_representation": candidate.interpolation,
                        "phase2_qc_status": beat["phase2_qc_status"],
                        "phase3_overall_qc_status": beat["overall_qc_status"],
                        "qc_issue_codes": issues,
                    })
                    key = (dataset_id, beat["cohort"], beat["condition"], definition.name, definition.candidate_type)
                    counters = summary[key]
                    counters["total"] += 1
                    counters["eligible"] += eligible
                    counters["detected"] += candidate.status == "DETECTED"
                    counters["missing"] += candidate.status == "MISSING"
                    counters["ineligible"] += candidate.status == "INELIGIBLE"
                    counters["ambiguous"] += candidate.status == "NOT_IMPLEMENTED_AMBIGUOUS"
                    counters["metadata"] += candidate.status == "UNAVAILABLE_REQUIRED_METADATA"
                    counters["passed"] += candidate.status == "DETECTED" and qc_status == "PASS"
                    counters["flagged"] += candidate.status == "DETECTED" and qc_status == "FLAGGED"
                    counters["invalid"] += candidate.status == "DETECTED" and qc_status == "INVALID"
                    summary_issues[key].update(code for code in issues.split(";") if code)
                    by_type[definition.candidate_type].append((definition, candidate, qc_status))
                    candidate_rows += 1

                for candidate_type, results in by_type.items():
                    implemented = [item for item in results if item[0].implementation_status == "IMPLEMENTED"]
                    detected_count = sum(item[1].status == "DETECTED" for item in implemented)
                    if not implemented:
                        continue
                    valid_count = sum(
                        item[1].status == "DETECTED" and item[2] != "INVALID"
                        for item in implemented
                    )
                    choice = {
                        "beat": beat, "results": results, "detected_count": detected_count,
                        "valid_count": valid_count, "implemented_count": len(implemented),
                        "label": "highest coverage",
                    }
                    slot = examples[(dataset_id, candidate_type)]
                    score = (valid_count, detected_count)
                    if "best" not in slot or score > (
                        slot["best"]["valid_count"], slot["best"]["detected_count"]
                    ):
                        slot["best"] = choice
                    difficult = dict(choice)
                    difficult["label"] = "lowest coverage / difficult"
                    if "difficult" not in slot or score < (
                        slot["difficult"]["valid_count"], slot["difficult"]["detected_count"]
                    ):
                        slot["difficult"] = difficult

    summary_rows = []
    for key, counters in summary.items():
        dataset_id, cohort, condition, name, candidate_type = key
        eligible = counters["eligible"]
        summary_rows.append({
            "dataset_id": dataset_id, "cohort": cohort, "condition": condition,
            "candidate_name": name, "candidate_type": candidate_type,
            "total_beat_rows": counters["total"], "eligible_beats": eligible,
            "detected_candidates": counters["detected"], "missing_candidates": counters["missing"],
            "ineligible_beats": counters["ineligible"],
            "not_implemented_ambiguous": counters["ambiguous"],
            "unavailable_required_metadata": counters["metadata"],
            "qc_passed_detections": counters["passed"],
            "qc_flagged_detections": counters["flagged"],
            "invalid_locations": counters["invalid"],
            "detection_coverage_among_eligible": f"{counters['detected'] / eligible:.6f}" if eligible else "",
            "issue_codes": ";".join(sorted(summary_issues[key])),
        })
    with (output_dir / "phase4_bx_candidate_qc_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "phase4_candidate_definition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFINITION_COLUMNS)
        writer.writeheader()
        writer.writerows(_definition_rows())
    plots = _plot_examples(project_root, examples, output_dir / "plots")
    _write_report(output_dir, summary_rows, candidate_rows, plots)
    return {
        "beats": len(phase3_rows), "candidate_rows": candidate_rows,
        "definitions": len(DEFINITIONS), "plots": plots,
        "ambiguous_definitions": sum(d.implementation_status == "NOT_IMPLEMENTED_AMBIGUOUS" for d in DEFINITIONS),
    }
