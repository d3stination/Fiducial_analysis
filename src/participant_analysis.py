from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def participant_mean_ci(values: pd.Series, confidence: float = .95) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(clean.mean())
    if clean.size == 1:
        return mean, np.nan, np.nan, 1
    half = float(t.ppf((1 + confidence) / 2, clean.size - 1) * clean.std(ddof=1) / np.sqrt(clean.size))
    return mean, max(0.0, mean - half), min(100.0, mean + half), int(clean.size)


def detection_group_summary(participant_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in participant_table.groupby(["candidate_name", "candidate_type"], observed=True):
        mean, low, high, n = participant_mean_ci(group["missed_plus_invalid_pct"])
        rows.append({"candidate_name": keys[0], "candidate_type": keys[1], "participants": n,
                     "mean_missed_plus_invalid_pct": mean, "ci95_low_pct": low, "ci95_high_pct": high})
    return pd.DataFrame(rows)

