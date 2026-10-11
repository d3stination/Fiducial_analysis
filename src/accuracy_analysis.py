from __future__ import annotations

import numpy as np
import pandas as pd


def _valid_mask(master: pd.DataFrame, method: str) -> pd.Series:
    return master[f"{method}__detection_status"].eq("DETECTED") & master[f"{method}__qc_status"].isin(["PASS", "FLAGGED"])


def common_reference_beats(master: pd.DataFrame, methods: list[str], kind: str) -> pd.DataFrame:
    eligibility = "b_reference_eligibility" if kind == "B" else "x_reference_eligibility"
    mask = master[eligibility].astype(bool)
    for method in methods:
        mask &= _valid_mask(master, method)
    eligible = master.loc[mask].copy()
    participants = eligible.groupby("subject_id")["beat_id"].size()
    participants = participants[participants.gt(0)].index
    return eligible.loc[eligible["subject_id"].isin(participants)].copy()


def accuracy_tables(master: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beat_rows, participant_rows, metric_rows = [], [], []
    for kind, reference in (("B", "echo_avo_time_s"), ("X", "echo_avc_time_s")):
        methods = registry.loc[registry["candidate_type"].eq(kind), "candidate_name"].tolist()
        common = common_reference_beats(master, methods, kind)
        for method in methods:
            signed = 1000 * (common[f"{method}__candidate_timestamp_s"] - common[reference])
            part = pd.DataFrame({"subject_id": common["subject_id"], "signed_error_ms": signed})
            part["absolute_error_ms"] = part["signed_error_ms"].abs()
            part["candidate_name"], part["candidate_type"] = method, kind
            beat = common[["dataset_id", "subject_id", "recording_id", "beat_id"]].copy()
            beat["candidate_name"], beat["candidate_type"] = method, kind
            beat["signed_error_ms"], beat["absolute_error_ms"] = signed.to_numpy(), signed.abs().to_numpy()
            beat_rows.append(beat)
            participant_rows.append(part.groupby(["subject_id", "candidate_name", "candidate_type"], observed=True).agg(
                beats=("absolute_error_ms", "size"), median_absolute_error_ms=("absolute_error_ms", "median"),
                median_signed_error_ms=("signed_error_ms", "median"),
                p90_absolute_error_ms=("absolute_error_ms", lambda x: x.quantile(.90)),
                p95_absolute_error_ms=("absolute_error_ms", lambda x: x.quantile(.95)),
            ).reset_index())
            metric_rows.append({
                "candidate_name": method, "candidate_type": kind,
                "common_participants": common["subject_id"].nunique(), "common_beats": len(common),
                "median_signed_error_ms": float(signed.median()), "signed_error_iqr_ms": float(signed.quantile(.75) - signed.quantile(.25)),
                "median_absolute_error_ms": float(signed.abs().median()),
                "p90_absolute_error_ms": float(signed.abs().quantile(.90)), "p95_absolute_error_ms": float(signed.abs().quantile(.95)),
                "tolerance_ms": np.nan, "tolerance_exceedance_pct": np.nan,
                "tolerance_status": "NOT_SPECIFIED_BY_SUPERVISOR",
            })
    return pd.concat(beat_rows, ignore_index=True), pd.concat(participant_rows, ignore_index=True), pd.DataFrame(metric_rows)

