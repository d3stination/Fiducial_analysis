from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import KEYS, load_config


def valid_detection(rows: pd.DataFrame, config: dict | None = None) -> pd.Series:
    config = config or load_config()
    return rows["detection_status"].eq(config["analysis"]["valid_detection_status"]) & rows["qc_status"].isin(config["analysis"]["valid_qc_statuses"])


def build_detection_performance(phase4: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    rows = phase4.copy()
    rows["valid"] = valid_detection(rows, config)
    rows["ineligible"] = rows["detection_status"].eq("INELIGIBLE")
    rows["missed"] = ~rows["ineligible"] & ~rows["detection_status"].eq("DETECTED")
    rows["invalid"] = rows["detection_status"].eq("DETECTED") & rows["qc_status"].eq("INVALID")
    rows["physiologically_implausible"] = rows["invalid"] & rows["qc_issue_codes"].fillna("").str.contains("not_strictly|outside|physiolog", case=False)
    rows["attempted_denominator"] = ~rows["ineligible"]
    grouped = ["dataset_id", "cohort", "condition", "subject_id", "candidate_name", "candidate_type"]
    summary = rows.groupby(grouped, observed=True).agg(
        total_beats=("beat_id", "size"), eligible_beats=("attempted_denominator", "sum"),
        valid_beats=("valid", "sum"), missed_beats=("missed", "sum"),
        invalid_beats=("invalid", "sum"), physiologically_implausible_beats=("physiologically_implausible", "sum"),
        ineligible_beats=("ineligible", "sum"), recordings=("recording_id", "nunique"),
    ).reset_index()
    denominator = summary["eligible_beats"].replace(0, np.nan)
    for count, rate in (("valid_beats", "success_pct"), ("missed_beats", "missed_pct"), ("invalid_beats", "invalid_pct"), ("physiologically_implausible_beats", "physiologically_implausible_pct")):
        summary[rate] = 100 * summary[count] / denominator
    summary["missed_plus_invalid_pct"] = 100 * (summary["missed_beats"] + summary["invalid_beats"]) / denominator
    summary["denominator_definition"] = "eligible beats; ineligible excluded; missed and invalid retained"
    return summary


def wide_candidates(phase4: pd.DataFrame) -> pd.DataFrame:
    fields = ["candidate_timestamp_s", "detection_status", "qc_status", "qc_issue_codes"]
    wide = phase4.pivot(index=KEYS, columns="candidate_name", values=fields)
    wide.columns = [f"{method}__{field}" for field, method in wide.columns]
    return wide.reset_index()

