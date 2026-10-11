from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def disagreement_tables(master: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beat_output, pair_output = [], []
    for kind in ("B", "X"):
        methods = registry.loc[registry["candidate_type"].eq(kind), "candidate_name"].tolist()
        values = pd.DataFrame(index=master.index)
        valid = pd.DataFrame(index=master.index)
        for method in methods:
            values[method] = 1000 * (master[f"{method}__candidate_timestamp_s"] - master["r_peak_time_s"])
            valid[method] = master[f"{method}__detection_status"].eq("DETECTED") & master[f"{method}__qc_status"].isin(["PASS", "FLAGGED"])
        values = values.where(valid)
        complete = valid.all(axis=1)
        summary = master[["dataset_id", "subject_id", "recording_id", "beat_id", "condition"]].copy()
        summary["candidate_type"] = kind
        summary["valid_algorithm_count"] = valid.sum(axis=1)
        summary["all_algorithms_valid"] = complete
        summary["within_beat_range_ms"] = values.max(axis=1) - values.min(axis=1)
        summary.loc[~complete, "within_beat_range_ms"] = np.nan
        beat_output.append(summary)
        for first, second in combinations(methods, 2):
            rows = summary.loc[complete, ["dataset_id", "subject_id", "recording_id", "beat_id"]].copy()
            rows["candidate_type"], rows["method_i"], rows["method_j"] = kind, first, second
            rows["pairwise_difference_ms"] = values.loc[complete, first].to_numpy() - values.loc[complete, second].to_numpy()
            pair_output.append(rows)
    beat = pd.concat(beat_output, ignore_index=True)
    pairs = pd.concat(pair_output, ignore_index=True)
    participant = beat.dropna(subset=["within_beat_range_ms"]).groupby(["subject_id", "candidate_type"], observed=True).agg(
        beats=("within_beat_range_ms", "size"), median_range_ms=("within_beat_range_ms", "median"),
        p90_range_ms=("within_beat_range_ms", lambda x: x.quantile(.90)),
        p95_range_ms=("within_beat_range_ms", lambda x: x.quantile(.95)),
    ).reset_index()
    return beat, participant, pairs

