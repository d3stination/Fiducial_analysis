"""Replay HeartCycle preprocessing, R/C detection, B/X detection and ECHO alignment.

The raw-data directory remains external. A temporary junction exposes the
``Heart Cycle`` folder to the frozen pipeline, and all generated files are
written under ``validation/.heartcycle_replay``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation" / "heartcycle"
DATASET_ID = "heartcycle_healthy"
KEYS = ["dataset_id", "subject_id", "recording_id", "beat_id"]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation/.matplotlib"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filtered_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, low_memory=False, chunksize=200_000, usecols=columns):
        if "dataset_id" in chunk:
            chunk = chunk.loc[chunk.dataset_id.eq(DATASET_ID)]
        if len(chunk):
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns or [])


def compare_frames(name: str, new: pd.DataFrame, old: pd.DataFrame, key: list[str], exact: list[str], numeric: list[str], tolerance: float = 0.0) -> dict:
    left = new.sort_values(key).set_index(key)
    right = old.sort_values(key).set_index(key)
    index_match = left.index.equals(right.index)
    exact_mismatch = 0
    numeric_mismatch = 0
    if index_match:
        for column in exact:
            a = left[column].fillna("<NA>").astype(str)
            b = right[column].fillna("<NA>").astype(str)
            exact_mismatch += int(a.ne(b).sum())
        for column in numeric:
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
            numeric_mismatch += int((~np.isclose(a, b, atol=tolerance, rtol=0, equal_nan=True)).sum())
    return {
        "stage": name,
        "new_rows": len(new),
        "canonical_rows": len(old),
        "index_match": index_match,
        "exact_value_mismatch_n": exact_mismatch,
        "numeric_value_mismatch_n": numeric_mismatch,
        "passed": bool(index_match and len(new) == len(old) and exact_mismatch == 0 and numeric_mismatch == 0),
    }


def create_junction(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"could not create raw-data junction: {result.stdout} {result.stderr}")


def reference_metrics(reference: pd.DataFrame, point: str) -> pd.DataFrame:
    registry = pd.read_csv(ROOT / "configs/canonical_method_registry.csv")
    registry = registry.loc[registry.candidate_type.eq(point), ["algorithm_id", "candidate_name"]]
    event = "echo_avo_time_s" if point == "B" else "echo_avc_time_s"
    alignment = "echo_avo_alignment_status" if point == "B" else "echo_avc_alignment_status"
    valid = pd.DataFrame({
        row.algorithm_id: reference[f"{row.candidate_name}__detection_status"].eq("DETECTED")
        & reference[f"{row.candidate_name}__qc_status"].isin(["PASS", "FLAGGED"])
        for row in registry.itertuples(index=False)
    })
    strict = valid.all(axis=1) & reference[alignment].eq("MATCHED_WITHIN_DETECTED_RR")
    rows = []
    for row in registry.itertuples(index=False):
        method = row.algorithm_id
        time_column = f"{row.candidate_name}__time_s"
        selected = reference.loc[strict, ["subject_id", event, time_column]].copy()
        selected["signed"] = 1000 * (selected[time_column] - selected[event])
        selected["absolute"] = selected.signed.abs()
        participant_mae = selected.groupby("subject_id").absolute.median()
        rows.append({
            "point_type": point,
            "algorithm_id": method,
            "strict_common_reference_beat_n": int(strict.sum()),
            "median_signed_reference_error_ms": float(selected.signed.median()),
            "participant_level_median_absolute_reference_error_ms": float(participant_mae.median()),
            "p90_absolute_reference_error_ms": float(selected.absolute.quantile(0.90)),
            "p95_absolute_reference_error_ms": float(selected.absolute.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="External Variability Analysis root")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--reuse-work", action="store_true", help="Reuse a completed temporary replay after a comparison-only failure")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    raw = data_root / "Heart Cycle"
    canonical = data_root / "outputs"
    builder = data_root / "HeartCycle_BX_Reference_Validation/code/create_heartcycle_validation_csv.py"
    required = [raw, canonical / "phase2/phase2_r_c_detection_all_beats.csv", canonical / "phase3/phase3_beat_segmentation_all_beats.csv", canonical / "phase4/phase4_bx_candidate_detection_all_beats_final.csv", builder]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing HeartCycle replay inputs:\n" + "\n".join(missing))

    work = ROOT / "validation" / ".heartcycle_replay"
    reuse = args.reuse_work and (work / "outputs/phase4/phase4_bx_candidate_detection_all_beats_final.csv").exists()
    if work.exists() and not reuse:
        if work.resolve().parent != (ROOT / "validation").resolve():
            raise RuntimeError("unsafe replay work path")
        shutil.rmtree(work)
    if not reuse:
        (work / "outputs").mkdir(parents=True)
        create_junction(work / "Heart Cycle", raw)
        (work / "src").mkdir()
        shutil.copy2(ROOT / "src/__init__.py", work / "src/__init__.py")
        shutil.copytree(ROOT / "src/variability_analysis", work / "src/variability_analysis")

    sys.path.insert(0, str(ROOT))
    from src.variability_analysis.inventory import run_inventory
    from src.variability_analysis.phase2_full import run_phase2_full
    from src.variability_analysis.phase3 import run_phase3
    from src.variability_analysis.phase4 import run_phase4
    from src.variability_analysis.phase4_finalize import run_phase4_finalization

    if reuse:
        print("HeartCycle replay: reusing completed fresh outputs for comparison", flush=True)
        phase1_counts = phase2_counts = phase3_counts = phase4_counts = final_counts = {"reused": True}
    else:
        print("HeartCycle replay: Phase 1 source inventory", flush=True)
        phase1_counts = run_inventory(work, work / "outputs/phase1")
        print("HeartCycle replay: Phase 2 preprocessing and R/C detection", flush=True)
        phase2_counts = run_phase2_full(work, work / "outputs/phase2")
        print("HeartCycle replay: Phase 3 beat segmentation", flush=True)
        phase3_counts = run_phase3(work, work / "outputs/phase3")
        print("HeartCycle replay: Phase 4 frozen B/X definitions", flush=True)
        phase4_counts = run_phase4(work, work / "outputs/phase4")
        final_counts = run_phase4_finalization(work, work / "outputs/phase4")

        temp_builder = work / "HeartCycle_BX_Reference_Validation/code/create_heartcycle_validation_csv.py"
        temp_builder.parent.mkdir(parents=True)
        shutil.copy2(builder, temp_builder)
        print("HeartCycle replay: ECHO AVO/AVC reference alignment", flush=True)
        subprocess.run([sys.executable, str(temp_builder)], cwd=work, check=True)

    new2 = pd.read_csv(work / "outputs/phase2/phase2_r_c_detection_all_beats.csv", low_memory=False)
    old2 = filtered_csv(canonical / "phase2/phase2_r_c_detection_all_beats.csv")
    new3 = pd.read_csv(work / "outputs/phase3/phase3_beat_segmentation_all_beats.csv", low_memory=False)
    old3 = filtered_csv(canonical / "phase3/phase3_beat_segmentation_all_beats.csv")
    new4 = pd.read_csv(work / "outputs/phase4/phase4_bx_candidate_detection_all_beats_final.csv", low_memory=False)
    old4 = filtered_csv(canonical / "phase4/phase4_bx_candidate_detection_all_beats_final.csv")
    new_ref = pd.read_csv(work / "HeartCycle_BX_Reference_Validation/output/HeartCycle_BX_reference_validation_beatwise.csv", low_memory=False)
    old_ref = pd.read_csv(data_root / "HeartCycle_BX_Reference_Validation/output/HeartCycle_BX_reference_validation_beatwise.csv", low_memory=False)

    audits = [
        compare_frames("phase2_r_c", new2, old2, KEYS, ["detection_status", "qc_status"], ["r_sample", "next_r_sample", "c_sample", "r_timestamp_s", "next_r_timestamp_s", "c_timestamp_s"], 5e-9),
        compare_frames("phase3_segmentation", new3, old3, KEYS, ["phase2_detection_status", "phase2_qc_status", "phase3_segmentation_status", "overall_qc_status"], ["beat_start_r_sample_inclusive", "beat_end_next_r_sample_exclusive", "c_sample", "beat_start_timestamp_s", "beat_end_timestamp_s", "c_timestamp_s"], 5e-9),
        compare_frames("phase4_bx", new4, old4, KEYS + ["candidate_name"], ["candidate_type", "detection_status", "qc_status"], ["candidate_sample_index", "candidate_timestamp_s", "search_start_sample", "search_stop_sample_exclusive"], 5.1e-8),
        compare_frames(
            "echo_reference_alignment", new_ref, old_ref,
            ["subject_id", "recording_id", "beat_id"], ["beat_number"],
            [column for column in old_ref.columns if column not in {"subject_id", "recording_id", "beat_id", "beat_number"}],
            5e-7,
        ),
    ]

    metrics = pd.concat([reference_metrics(new_ref, "B"), reference_metrics(new_ref, "X")], ignore_index=True)
    expected = pd.read_csv(ROOT / "results/tables/dataset_algorithm_results.csv")
    expected = expected.loc[expected.dataset.eq("HeartCycle"), metrics.columns]
    metric_audit = compare_frames(
        "heartcycle_reference_metrics", metrics, expected,
        ["point_type", "algorithm_id"], [],
        ["strict_common_reference_beat_n", "median_signed_reference_error_ms", "participant_level_median_absolute_reference_error_ms", "p90_absolute_reference_error_ms", "p95_absolute_reference_error_ms"], 1e-6,
    )
    audits.append(metric_audit)

    release_hashes = {path.name: sha256(path) for path in (ROOT / "src/variability_analysis").glob("*.py")}
    external_hashes = {name: sha256(data_root / "src/variability_analysis" / name) for name in ["bx_candidates.py", "phase4.py", "signal_processing.py"]}
    hash_pass = all(release_hashes[name] == digest for name, digest in external_hashes.items())
    audits.append({"stage": "frozen_source_hashes", "new_rows": 3, "canonical_rows": 3, "index_match": True, "exact_value_mismatch_n": 0 if hash_pass else 1, "numeric_value_mismatch_n": 0, "passed": hash_pass})

    VALIDATION.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audits)
    audit.to_csv(VALIDATION / "heartcycle_replay_summary.csv", index=False)
    metrics.to_csv(VALIDATION / "heartcycle_replayed_reference_metrics.csv", index=False)
    payload = {
        "python": sys.version,
        "raw_h5_files": len(list(raw.glob("*.h5"))),
        "phase1_counts": phase1_counts,
        "phase2_counts": phase2_counts,
        "phase3_counts": phase3_counts,
        "phase4_counts": phase4_counts,
        "phase4_final_counts": final_counts,
        "checks_passed": int(audit.passed.sum()),
        "checks_total": len(audit),
        "all_passed": bool(audit.passed.all()),
    }
    (VALIDATION / "heartcycle_replay.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(audit.to_string(index=False), flush=True)

    if not args.keep_work:
        raw_link = work / "Heart Cycle"
        raw_link.rmdir()
        shutil.rmtree(work)
    if not audit.passed.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
