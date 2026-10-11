"""Replay IITB and ReBeat raw recordings and compare frozen detections."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation" / "iitb_rebeat"
DATASETS = {
    "iitb_normal": ("IITB", "normal"),
    "iitb_post_exercise": ("IITB", "post_exercise"),
    "rebeat_baseline": ("ReBeat", "baseline"),
    "rebeat_cognitive": ("ReBeat", "cognitive_workload"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    os.environ["P1_DATA_ROOT"] = str(data_root)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation/.matplotlib"))
    sys.path.insert(0, str(ROOT))

    from src.phase2_analysis import replay_sources

    master, status, _, inventory, signals, _, fingerprints = replay_sources()
    del signals, fingerprints
    status = status.loc[status.dataset_id.isin(DATASETS)].copy()
    status[["dataset", "release_condition"]] = status.dataset_id.map(DATASETS).apply(pd.Series)
    status["upstream_ineligible"] = status.detection_status.eq("INELIGIBLE")
    status["attempted"] = ~status.upstream_ineligible
    status["valid"] = status.detection_status.eq("DETECTED") & status.qc_status.isin(["PASS", "FLAGGED"])
    status["invalid"] = status.detection_status.eq("DETECTED") & status.qc_status.eq("INVALID")
    status["missed"] = status.attempted & ~status.detection_status.eq("DETECTED")
    group = ["dataset", "release_condition", "candidate_type", "algorithm_id"]
    replayed = status.groupby(group, sort=True).agg(
        total_beat_n=("beat_id", "size"),
        attempted_beat_n=("attempted", "sum"),
        valid_detection_n=("valid", "sum"),
        missed_detection_n=("missed", "sum"),
        invalid_detection_n=("invalid", "sum"),
        upstream_ineligible_n=("upstream_ineligible", "sum"),
    ).reset_index().rename(columns={"release_condition": "condition", "candidate_type": "point_type"})

    expected = pd.read_csv(ROOT / "results/tables/dataset_algorithm_results.csv")
    expected = expected.loc[expected.dataset.isin(["IITB", "ReBeat"]), replayed.columns]
    keys = ["dataset", "condition", "point_type", "algorithm_id"]
    left = replayed.sort_values(keys).set_index(keys)
    right = expected.sort_values(keys).set_index(keys)
    differences = left.compare(right) if left.index.equals(right.index) else pd.DataFrame()
    passed = left.index.equals(right.index) and differences.empty

    VALIDATION.mkdir(parents=True, exist_ok=True)
    replayed.to_csv(VALIDATION / "replayed_status_counts.csv", index=False)
    recording_summary = pd.DataFrame(inventory).groupby(["dataset_id", "condition"], sort=True).agg(
        recording_n=("recording_id", "nunique"),
        participant_n=("subject_id", "nunique"),
        beat_n=("beats", "sum"),
    ).reset_index()
    recording_summary.to_csv(VALIDATION / "recording_summary.csv", index=False)
    payload = {
        "python": sys.version,
        "recordings": len(inventory),
        "beats": len(master),
        "candidate_records": len(status),
        "strata": len(replayed),
        "canonical_strata": len(expected),
        "index_match": left.index.equals(right.index),
        "differing_cells": int(differences.size / 2),
        "passed": passed,
    }
    (VALIDATION / "iitb_rebeat_replay.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        differences.to_csv(VALIDATION / "status_count_differences.csv")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
