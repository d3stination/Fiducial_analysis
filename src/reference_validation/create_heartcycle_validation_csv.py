"""Build the HeartCycle beatwise B/X and echo-reference validation table.

This script only joins existing Phase 3/4 outputs with the separate HeartCycle
ECHO event stream. It does not filter signals or redetect R, C, B, or X.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.variability_analysis.loaders import _heartcycle_channels  # noqa: E402


DATASET_ID = "heartcycle_healthy"
R_MATCH_TOLERANCE_S = 0.120
PHASE1_PATH = PROJECT_ROOT / "outputs" / "phase1" / "recording_inventory.csv"
PHASE3_PATH = PROJECT_ROOT / "outputs" / "phase3" / "phase3_beat_segmentation_all_beats.csv"
PHASE4_PATH = PROJECT_ROOT / "outputs" / "phase4" / "phase4_bx_candidate_detection_all_beats_final.csv"
RAW_DIR = PROJECT_ROOT / "Heart Cycle"
OUTPUT_DIR = PROJECT_ROOT / "HeartCycle_BX_Reference_Validation"
CSV_PATH = OUTPUT_DIR / "output" / "HeartCycle_BX_reference_validation_beatwise.csv"
REPORT_PATH = OUTPUT_DIR / "reports" / "validation_report.txt"

KEYS = ["dataset_id", "subject_id", "recording_id", "beat_id"]


def file_signature(path: Path, include_hash: bool = False) -> tuple[int, int, str]:
    stat = path.stat()
    digest = ""
    if include_hash:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    return stat.st_size, stat.st_mtime_ns, digest


def source_snapshot() -> dict[str, tuple[int, int, str]]:
    snapshot = {
        str(path.resolve()): file_signature(path)
        for path in sorted(RAW_DIR.glob("*.h5"))
    }
    for path in (PHASE1_PATH, PHASE3_PATH, PHASE4_PATH):
        snapshot[str(path.resolve())] = file_signature(path, include_hash=True)
    return snapshot


def channel_data(by_index: dict, index: int, expected_label: str) -> np.ndarray:
    channel = by_index.get(index)
    if channel is None or channel.label.upper() != expected_label:
        raise ValueError(
            f"channel {index:03d} is missing or labelled "
            f"{getattr(channel, 'label', None)!r}, expected {expected_label}"
        )
    return np.asarray(channel.data, dtype=float).reshape(-1)


def events_by_echo_beat(
    events: np.ndarray, echo_r: np.ndarray, event_type: str,
) -> dict[int, tuple[float, str]]:
    """Apply the inspected legacy HeartCycle echo-event assignment exactly."""

    assigned: dict[int, list[float]] = {}
    for event in events[np.isfinite(events)]:
        echo_index = int(np.searchsorted(echo_r, event, side="right") - 1)
        if echo_index < 0:
            continue
        end = echo_r[echo_index + 1] if echo_index + 1 < len(echo_r) else np.inf
        if not echo_r[echo_index] < event < end:
            continue
        if event_type == "AVO" and event - echo_r[echo_index] > 0.300:
            continue
        assigned.setdefault(echo_index, []).append(float(event))

    result: dict[int, tuple[float, str]] = {}
    for echo_index in range(len(echo_r)):
        values = assigned.get(echo_index, [])
        if len(values) == 1:
            result[echo_index] = (values[0], "MATCHED")
        elif len(values) > 1:
            result[echo_index] = (np.nan, "AMBIGUOUS")
        else:
            result[echo_index] = (np.nan, "MISSING")
    return result


def extract_echo_events(path: Path, detected_r_times: np.ndarray) -> list[dict]:
    """Match separate ECHO R/events to detected ECG R times without redetection."""

    with h5py.File(path, "r") as handle:
        channels = _heartcycle_channels(handle)
    by_index = {channel.index: channel for channel in channels}
    echo_channels = [channel for channel in channels if channel.label.upper() == "ECHO"]
    if len(echo_channels) != 1:
        raise ValueError(f"expected one ECHO modality block, found {len(echo_channels)}")

    anchor = echo_channels[0].index
    echo_r = np.sort(channel_data(by_index, anchor + 1, "RPEAKS"))
    avo = channel_data(by_index, anchor + 2, "AVO")
    avc = channel_data(by_index, anchor + 4, "AVC")
    echo_r = echo_r[np.isfinite(echo_r)]
    if echo_r.size > 1 and not np.all(np.diff(echo_r) > 0):
        raise ValueError("ECHO R events are not strictly increasing")

    opening = events_by_echo_beat(avo, echo_r, "AVO")
    closing = events_by_echo_beat(avc, echo_r, "AVC")
    matches: dict[int, list[int]] = {}
    for echo_index, echo_time in enumerate(echo_r):
        distances = np.abs(detected_r_times - echo_time)
        if distances.size:
            nearest = int(np.argmin(distances))
            if distances[nearest] <= R_MATCH_TOLERANCE_S:
                matches.setdefault(nearest, []).append(echo_index)

    rows = []
    for detected_index, detected_r in enumerate(detected_r_times):
        choices = matches.get(detected_index, [])
        r_status = "MATCHED" if len(choices) == 1 else "AMBIGUOUS" if choices else "MISSING"
        echo_index = choices[0] if r_status == "MATCHED" else None
        echo_time = float(echo_r[echo_index]) if echo_index is not None else np.nan
        avo_time, avo_status = opening.get(echo_index, (np.nan, r_status)) if echo_index is not None else (np.nan, r_status)
        avc_time, avc_status = closing.get(echo_index, (np.nan, r_status)) if echo_index is not None else (np.nan, r_status)
        rows.append({
            "echo_block_id": f"_{anchor:03d}",
            "echo_r_time_s": echo_time,
            "echo_r_match_status": r_status,
            "echo_r_minus_detected_r_ms": (
                (echo_time - detected_r) * 1000.0 if np.isfinite(echo_time) else np.nan
            ),
            "echo_avo_time_s": avo_time,
            "echo_avo_source_status": avo_status,
            "echo_avc_time_s": avc_time,
            "echo_avc_source_status": avc_status,
        })
    return rows


def load_base_beats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(PHASE1_PATH, low_memory=False)
    inventory = inventory.loc[inventory["dataset_id"].eq(DATASET_ID)].copy()

    phase3 = pd.read_csv(PHASE3_PATH, low_memory=False)
    phase3 = phase3.loc[phase3["dataset_id"].eq(DATASET_ID)].copy()
    phase4 = pd.read_csv(PHASE4_PATH, low_memory=False)
    phase4 = phase4.loc[phase4["dataset_id"].eq(DATASET_ID)].copy()
    if phase3.empty or phase4.empty:
        raise ValueError("HeartCycle rows are missing from Phase 3 or finalized Phase 4")

    base = phase3.rename(columns={
        "beat_start_r_sample_inclusive": "r_peak_sample_index",
        "beat_end_next_r_sample_exclusive": "next_r_peak_sample_index",
        "beat_start_timestamp_s": "r_peak_time_s",
        "beat_end_timestamp_s": "next_r_peak_time_s",
        "c_sample": "c_sample_index",
        "c_timestamp_s": "c_point_time_s",
    })
    columns = [
        "dataset_id", "condition", "subject_id", "recording_id", "beat_id",
        "beat_number", "source_file", "sampling_frequency_hz",
        "sampling_frequency_source", "r_peak_sample_index",
        "next_r_peak_sample_index", "r_peak_time_s", "next_r_peak_time_s",
        "c_sample_index", "c_point_time_s", "phase2_detection_status",
        "phase2_qc_status", "phase3_segmentation_status", "overall_qc_status",
        "combined_issue_codes",
    ]
    base = base[columns].copy()
    base["rr_interval_s"] = base["next_r_peak_time_s"] - base["r_peak_time_s"]
    base["rr_interval_ms"] = base["rr_interval_s"] * 1000.0
    base["c_relative_to_r_ms"] = (
        base["c_point_time_s"] - base["r_peak_time_s"]
    ) * 1000.0
    return inventory, phase4, base


def add_candidates(base: pd.DataFrame, phase4: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    duplicate_count = int(phase4.duplicated(KEYS + ["candidate_name"]).sum())
    if duplicate_count:
        raise ValueError(f"Phase 4 contains {duplicate_count} duplicate beat/candidate rows")

    b_names = phase4.loc[phase4["candidate_type"].eq("B"), "candidate_name"].drop_duplicates().tolist()
    x_names = phase4.loc[phase4["candidate_type"].eq("X"), "candidate_name"].drop_duplicates().tolist()
    indexed_base = pd.MultiIndex.from_frame(base[KEYS])
    candidate_fields = {
        "candidate_sample_index": "sample_index",
        "candidate_timestamp_s": "time_s",
        "detection_status": "detection_status",
        "qc_status": "qc_status",
        "qc_issue_codes": "issue_codes",
    }

    for name in b_names + x_names:
        rows = phase4.loc[phase4["candidate_name"].eq(name)].set_index(KEYS)
        rows = rows.reindex(indexed_base)
        for source, suffix in candidate_fields.items():
            base[f"{name}__{suffix}"] = rows[source].to_numpy()
        base[f"{name}__r_relative_ms"] = (
            base[f"{name}__time_s"] - base["r_peak_time_s"]
        ) * 1000.0

    return base, b_names, x_names


def add_echo_references(base: pd.DataFrame) -> pd.DataFrame:
    reference_rows = []
    for _, recording_beats in base.groupby("recording_id", sort=False):
        path = Path(recording_beats["source_file"].iloc[0])
        detected_r = recording_beats["r_peak_time_s"].to_numpy(dtype=float)
        extracted = extract_echo_events(path, detected_r)
        if len(extracted) != len(recording_beats):
            raise ValueError(f"ECHO matching row count differs for {path.name}")
        reference_rows.extend(extracted)

    echo = pd.DataFrame(reference_rows, index=base.index)
    base = pd.concat([base, echo], axis=1)
    base["echo_avo_relative_to_r_ms"] = (
        base["echo_avo_time_s"] - base["r_peak_time_s"]
    ) * 1000.0
    base["echo_avc_relative_to_r_ms"] = (
        base["echo_avc_time_s"] - base["r_peak_time_s"]
    ) * 1000.0

    def alignment_status(row: pd.Series, event: str) -> str:
        source_status = row[f"echo_{event}_source_status"]
        time = row[f"echo_{event}_time_s"]
        if source_status != "MATCHED" or not np.isfinite(time):
            return str(source_status)
        if row["r_peak_time_s"] < time < row["next_r_peak_time_s"]:
            return "MATCHED_WITHIN_DETECTED_RR"
        return "OUTSIDE_DETECTED_RR"

    base["echo_avo_alignment_status"] = base.apply(alignment_status, axis=1, event="avo")
    base["echo_avc_alignment_status"] = base.apply(alignment_status, axis=1, event="avc")
    valid_lvet = (
        base["echo_avo_alignment_status"].eq("MATCHED_WITHIN_DETECTED_RR")
        & base["echo_avc_alignment_status"].eq("MATCHED_WITHIN_DETECTED_RR")
        & base["echo_avc_time_s"].gt(base["echo_avo_time_s"])
    )
    base["echo_reference_lvet_ms"] = np.where(
        valid_lvet,
        (base["echo_avc_time_s"] - base["echo_avo_time_s"]) * 1000.0,
        np.nan,
    )
    base["echo_reference_lvet_status"] = np.select(
        [
            valid_lvet,
            ~base["echo_avo_alignment_status"].eq("MATCHED_WITHIN_DETECTED_RR"),
            ~base["echo_avc_alignment_status"].eq("MATCHED_WITHIN_DETECTED_RR"),
            base["echo_avc_time_s"].le(base["echo_avo_time_s"]),
        ],
        ["VALID", "AVO_UNAVAILABLE_OR_UNALIGNED", "AVC_UNAVAILABLE_OR_UNALIGNED", "AVC_NOT_AFTER_AVO"],
        default="UNAVAILABLE",
    )
    base["pep_ms"] = np.nan
    base["pep_status"] = "UNAVAILABLE_NO_VERIFIED_Q_ONSET_REFERENCE"
    return base


def validate(
    base: pd.DataFrame, inventory: pd.DataFrame, b_names: list[str], x_names: list[str],
    before_snapshot: dict[str, tuple[int, int, str]],
) -> dict[str, tuple[bool, str]]:
    key_unique = not base.duplicated(["subject_id", "recording_id", "beat_id"]).any()
    ordered = bool((base["next_r_peak_time_s"] > base["r_peak_time_s"]).all())
    source_counts = (inventory["subject_id"].nunique(), inventory["recording_id"].nunique())
    output_counts = (base["subject_id"].nunique(), base["recording_id"].nunique())

    formula_checks = {
        "RR": np.allclose(
            base["rr_interval_ms"],
            (base["next_r_peak_time_s"] - base["r_peak_time_s"]) * 1000.0,
            equal_nan=True,
        ),
        "C": np.allclose(
            base["c_relative_to_r_ms"],
            (base["c_point_time_s"] - base["r_peak_time_s"]) * 1000.0,
            equal_nan=True,
        ),
        "AVO": np.allclose(
            base["echo_avo_relative_to_r_ms"],
            (base["echo_avo_time_s"] - base["r_peak_time_s"]) * 1000.0,
            equal_nan=True,
        ),
        "AVC": np.allclose(
            base["echo_avc_relative_to_r_ms"],
            (base["echo_avc_time_s"] - base["r_peak_time_s"]) * 1000.0,
            equal_nan=True,
        ),
    }
    for name in b_names + x_names:
        formula_checks[name] = np.allclose(
            base[f"{name}__r_relative_ms"],
            (base[f"{name}__time_s"] - base["r_peak_time_s"]) * 1000.0,
            equal_nan=True,
        )
    lvet_rows = base["echo_reference_lvet_status"].eq("VALID")
    formula_checks["LVET"] = np.allclose(
        base.loc[lvet_rows, "echo_reference_lvet_ms"],
        (
            base.loc[lvet_rows, "echo_avc_time_s"]
            - base.loc[lvet_rows, "echo_avo_time_s"]
        ) * 1000.0,
        equal_nan=True,
    )

    after_snapshot = source_snapshot()
    return {
        "one row per R-to-R beat": (
            key_unique and len(base) == len(base.drop_duplicates(KEYS)),
            f"rows={len(base):,}; unique beat keys={len(base.drop_duplicates(KEYS)):,}",
        ),
        "R(i+1) later than R(i)": (ordered, f"violations={int((base['next_r_peak_time_s'] <= base['r_peak_time_s']).sum())}"),
        "beat IDs unique within subject/recording": (key_unique, f"duplicates={int(base.duplicated(['subject_id', 'recording_id', 'beat_id']).sum())}"),
        "all relative timing formulas": (all(formula_checks.values()), f"failed={','.join(k for k, v in formula_checks.items() if not v) or 'none'}"),
        "subject/recording counts match Phase 1 inventory": (source_counts == output_counts, f"source={source_counts}; output={output_counts}"),
        "all 10 B and 13 X definitions retained": (len(b_names) == 10 and len(x_names) == 13, f"B={len(b_names)}; X={len(x_names)}"),
        "source files unchanged during run": (before_snapshot == after_snapshot, f"tracked source files={len(before_snapshot)}"),
    }


def write_report(
    base: pd.DataFrame, inventory: pd.DataFrame, b_names: list[str], x_names: list[str],
    checks: dict[str, tuple[bool, str]],
) -> None:
    lines = [
        "HeartCycle B/X Reference Validation - table construction report",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Scope",
        "- This run builds a traceable beat-level table only. It does not validate, rank, select, or redefine any B/X candidate.",
        "- R-to-R beats and C points come unchanged from the completed Phase 3 output.",
        "- B/X detections come unchanged from the finalized Phase 4 output.",
        "- Separate ECHO R/AVO/AVC events are matched with the inspected legacy rule: nearest detected R within 120 ms, one-to-one at the detected-beat level, with ambiguous matches rejected.",
        "",
        "Source files",
        f"- Phase 1 inventory: {PHASE1_PATH}",
        f"- Phase 3 beat segmentation: {PHASE3_PATH}",
        f"- Finalized Phase 4 candidates: {PHASE4_PATH}",
        f"- HeartCycle raw directory: {RAW_DIR} ({len(list(RAW_DIR.glob('*.h5')))} HDF5 files)",
        "",
        "Dataset summary",
        f"- Subjects: {base['subject_id'].nunique()} (Phase 1 source: {inventory['subject_id'].nunique()})",
        f"- Recordings: {base['recording_id'].nunique()} (Phase 1 source: {inventory['recording_id'].nunique()})",
        f"- Total ECG R-to-R beats: {len(base)}",
        f"- Beats with C: {base['c_point_time_s'].notna().sum()}",
        f"- Beats with a uniquely matched ECHO R: {base['echo_r_match_status'].eq('MATCHED').sum()}",
        f"- Beats with AVO inside the matched detected R-to-R beat: {base['echo_avo_alignment_status'].eq('MATCHED_WITHIN_DETECTED_RR').sum()}",
        f"- Beats with AVC inside the matched detected R-to-R beat: {base['echo_avc_alignment_status'].eq('MATCHED_WITHIN_DETECTED_RR').sum()}",
        f"- Beats with reference LVET = AVC - AVO: {base['echo_reference_lvet_status'].eq('VALID').sum()}",
        f"- Beats with conventional PEP: {base['pep_ms'].notna().sum()}",
        "",
        "B-definition availability",
    ]
    for name in b_names:
        available = int(base[f"{name}__time_s"].notna().sum())
        lines.append(f"- {name}: available={available}; missing={len(base) - available}")
    lines.extend(["", "X-definition availability"])
    for name in x_names:
        available = int(base[f"{name}__time_s"].notna().sum())
        lines.append(f"- {name}: available={available}; missing={len(base) - available}")

    lines.extend([
        "",
        "Reference and biomarker limitations",
        "- No verified Q-onset/reference series exists in the inspected HeartCycle channels. Therefore pep_ms is intentionally missing for every beat; R-to-B is retained separately as each B definition's __r_relative_ms column.",
        "- Stored PEP/PPEjec labels were not promoted to conventional PEP: the plan records that they commonly behave as R-to-AVO, and no Q onset is available to verify Q-to-AVO timing.",
        "- Stored LVET labels were not used: their semantics are unresolved and can behave as R-to-AVC. Reference LVET is calculated only as matched ECHO AVC minus matched ECHO AVO within the same detected R-to-R beat.",
        "- The ECHO stream has its own time base. Absolute echo timestamps are retained with echo_r_minus_detected_r_ms and match statuses; relative AVO/AVC timing is calculated only after the existing 120 ms R-event match.",
        "- ECHO event channels provide event timestamps, not source waveform sample indices; no artificial echo sample index was created.",
        "",
        "Validation checks",
    ])
    for label, (passed, detail) in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} | {label} | {detail}")
    lines.extend([
        "",
        f"Output CSV: {CSV_PATH}",
        f"Rows: {len(base)}; columns: {len(base.columns)}",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly replace this new phase's CSV/report if they already exist.",
    )
    args = parser.parse_args()
    if (CSV_PATH.exists() or REPORT_PATH.exists()) and not args.overwrite:
        raise FileExistsError("output already exists; rerun with --overwrite only if replacement is intended")

    before_snapshot = source_snapshot()
    inventory, phase4, base = load_base_beats()
    base, b_names, x_names = add_candidates(base, phase4)
    base = add_echo_references(base)

    integer_columns = [
        "beat_number", "r_peak_sample_index", "next_r_peak_sample_index", "c_sample_index",
    ] + [f"{name}__sample_index" for name in b_names + x_names]
    for column in integer_columns:
        base[column] = pd.to_numeric(base[column], errors="coerce").astype("Int64")

    checks = validate(base, inventory, b_names, x_names, before_snapshot)
    failed = [label for label, (passed, _) in checks.items() if not passed]
    if failed:
        raise ValueError(f"validation failed before export: {', '.join(failed)}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(CSV_PATH, index=False, na_rep="", float_format="%.9f")
    write_report(base, inventory, b_names, x_names, checks)
    print(f"Created {CSV_PATH}")
    print(f"Created {REPORT_PATH}")
    print(f"Rows={len(base):,}; columns={len(base.columns)}; B={len(b_names)}; X={len(x_names)}")


if __name__ == "__main__":
    main()
