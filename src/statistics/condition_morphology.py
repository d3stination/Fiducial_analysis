from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .morphology_associations import FEATURES, KEYS, load_features, load_status, paths


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "p1_completion"
CONDITIONS = {
    "IITB": ("normal", "post_exercise"),
    "ReBeat": ("baseline", "cognitive_workload"),
}
RNG = np.random.default_rng(20261009)


def bootstrap_rho(x: np.ndarray, y: np.ndarray, iterations: int = 4000) -> tuple[float, float, float]:
    """Pair-bootstrap Spearman's rho, re-ranking every resampled participant set."""
    rho = float(spearmanr(x, y).statistic)
    idx = RNG.integers(0, len(x), size=(iterations, len(x)))
    bx = rankdata(x[idx], axis=1, method="average")
    by = rankdata(y[idx], axis=1, method="average")
    bx = bx - bx.mean(axis=1, keepdims=True)
    by = by - by.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.square(bx).sum(axis=1) * np.square(by).sum(axis=1))
    draws = np.divide((bx * by).sum(axis=1), denominator, out=np.full(iterations, np.nan), where=denominator > 0)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 100:
        return rho, np.nan, np.nan
    return rho, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def bh(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["bh_q_value"] = np.nan
    for _, idx in frame.groupby(["dataset", "candidate_type", "outcome"]).groups.items():
        p = frame.loc[idx, "permutation_p"].dropna().sort_values()
        if p.empty:
            continue
        q = (p * len(p) / np.arange(1, len(p) + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
        frame.loc[q.index, "bh_q_value"] = q
    frame["exploratory_supported"] = frame.bh_q_value.lt(.05) & ((frame.bootstrap_95ci_low > 0) | (frame.bootstrap_95ci_high < 0))
    return frame


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, iterations: int = 10000) -> float:
    """Two-sided participant-label permutation p-value for Spearman's rho."""
    if not np.isfinite(observed):
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    rx = rx - rx.mean()
    unique_y, counts_y = np.unique(ry, return_counts=True)
    if len(unique_y) == 2 and counts_y.min() == 1:
        rare = unique_y[np.argmin(counts_y)]
        common = unique_y[np.argmax(counts_y)]
        correlations = []
        for position in range(len(ry)):
            candidate = np.full(len(ry), common, dtype=float)
            candidate[position] = rare
            candidate -= candidate.mean()
            denominator = np.sqrt(np.square(rx).sum() * np.square(candidate).sum())
            correlations.append(float(np.dot(rx, candidate) / denominator))
        threshold = abs(observed) - 1e-12 * max(1.0, abs(observed))
        return float(np.mean(np.abs(correlations) >= threshold))
    permutations = np.argsort(RNG.random((iterations, len(y))), axis=1)
    py = ry[permutations]
    py = py - py.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.square(rx).sum() * np.square(py).sum(axis=1))
    correlations = np.divide((py * rx).sum(axis=1), denominator, out=np.zeros(iterations), where=denominator > 0)
    # Algebraically equal tied-rank correlations can differ by about 1e-16
    # between scipy.stats.spearmanr and the vectorised Pearson calculation.
    threshold = abs(observed) - 1e-12 * max(1.0, abs(observed))
    return (1 + int((np.abs(correlations) >= threshold).sum())) / (iterations + 1)


def paired_medians(frame: pd.DataFrame, value: str, group: list[str], normal: str, stress: str) -> pd.DataFrame:
    summary = frame.groupby(group + ["condition"], as_index=False)[value].median()
    wide = summary.pivot(index=group, columns="condition", values=value).dropna(subset=[normal, stress]).reset_index()
    wide = wide.rename(columns={normal: "normal_value", stress: "stress_value"})
    wide["stress_minus_normal"] = wide.stress_value - wide.normal_value
    return wide


def main() -> None:
    global RNG
    RNG = np.random.default_rng(20261009)
    features = load_features()
    status = load_status()
    feature_rows, outcome_rows, association_rows = [], [], []
    feature_names = list(FEATURES) + ["rr_interval_ms"]

    # RR is taken from the canonical master beats and merged with candidate-independent features.
    enriched = []
    for dataset in ["IITB", "ReBeat"]:
        for kind in "BX":
            beat = features.loc[(features.dataset.eq(dataset)) & features.candidate_type.eq(kind)].copy()
            master = pd.read_csv(paths(dataset, kind) / f"{kind}_master_beats.csv", usecols=KEYS + ["rr_interval_ms"])
            enriched.append(beat.merge(master, on=KEYS, how="left", validate="one_to_one"))
    features = pd.concat(enriched, ignore_index=True)

    for dataset, (normal, stress_condition) in CONDITIONS.items():
        for kind in "BX":
            beat = features.loc[(features.dataset.eq(dataset)) & features.candidate_type.eq(kind)]
            matched = set(beat.loc[beat.condition.eq(normal), "subject_id"]) & set(beat.loc[beat.condition.eq(stress_condition), "subject_id"])
            beat = beat.loc[beat.subject_id.isin(matched)]
            for feature in feature_names:
                change = paired_medians(beat, feature, ["subject_id"], normal, stress_condition)
                change["dataset"] = dataset
                change["candidate_type"] = kind
                change["feature"] = feature
                change["normal_condition"] = normal
                change["stress_condition"] = stress_condition
                feature_rows.append(change)

            detection = status.loc[(status.dataset.eq(dataset)) & status.candidate_type.eq(kind) & status.subject_id.isin(matched)].copy()
            attempted = detection.loc[detection.attempted.astype(bool)].groupby(["subject_id", "condition", "algorithm_id", "candidate_name"], as_index=False).agg(
                attempted_n=("beat_id", "size"), failure_n=("analysis_status", lambda x: x.isin(["MISSED", "INVALID"]).sum())
            )
            attempted["failure_rate_pct"] = 100 * attempted.failure_n / attempted.attempted_n
            for keys, group in attempted.groupby(["algorithm_id", "candidate_name"]):
                change = paired_medians(group, "failure_rate_pct", ["subject_id"], normal, stress_condition)
                change["dataset"] = dataset; change["candidate_type"] = kind
                change["algorithm_id"], change["candidate_name"] = keys
                change["outcome"] = "failure_rate_pct"
                change["unit"] = "percentage points"
                outcome_rows.append(change)

            peer = pd.read_csv(paths(dataset, kind) / f"{kind}_definition_disagreement.csv")
            peer = peer.loc[peer.subject_id.isin(matched)]
            for keys, group in peer.groupby(["algorithm_id", "candidate_name"]):
                change = paired_medians(group, "absolute_difference_from_other_method_median_ms", ["subject_id"], normal, stress_condition)
                change["dataset"] = dataset; change["candidate_type"] = kind
                change["algorithm_id"], change["candidate_name"] = keys
                change["outcome"] = "peer_disagreement_ms"
                change["unit"] = "ms"
                outcome_rows.append(change)

            ranges = pd.read_csv(paths(dataset, kind) / f"{kind}_common_beat_ranges.csv")
            ranges = ranges.loc[ranges.subject_id.isin(matched)]
            change = paired_medians(ranges, "timing_range_ms", ["subject_id"], normal, stress_condition)
            change["dataset"] = dataset; change["candidate_type"] = kind
            change["algorithm_id"] = "ALL_METHOD_RANGE"; change["candidate_name"] = "all-method within-beat range"
            change["outcome"] = "within_beat_range_ms"; change["unit"] = "ms"
            outcome_rows.append(change)

    feature_changes = pd.concat(feature_rows, ignore_index=True)
    outcome_changes = pd.concat(outcome_rows, ignore_index=True)
    feature_changes.to_csv(OUT / "phase6_condition_feature_changes.csv", index=False)
    outcome_changes.to_csv(OUT / "phase6_condition_outcome_changes.csv", index=False)

    for keys, outcomes in outcome_changes.groupby(["dataset", "candidate_type", "algorithm_id", "candidate_name", "outcome", "unit"]):
        for feature in feature_names:
            changes = feature_changes.loc[
                feature_changes.dataset.eq(keys[0]) & feature_changes.candidate_type.eq(keys[1]) & feature_changes.feature.eq(feature),
                ["subject_id", "stress_minus_normal"],
            ].rename(columns={"stress_minus_normal": "feature_change"})
            joined = outcomes[["subject_id", "stress_minus_normal"]].rename(columns={"stress_minus_normal": "outcome_change"}).merge(changes, on="subject_id", validate="one_to_one").dropna()
            if len(joined) >= 8 and joined.feature_change.nunique() > 1 and joined.outcome_change.nunique() > 1:
                x, y = joined.feature_change.to_numpy(float), joined.outcome_change.to_numpy(float)
                rho, low, high = bootstrap_rho(x, y)
                p = permutation_p(x, y, rho)
            else:
                rho = low = high = p = np.nan
            association_rows.append({
                "dataset": keys[0], "candidate_type": keys[1], "algorithm_id": keys[2], "candidate_name": keys[3],
                "outcome": keys[4], "outcome_unit": keys[5], "feature": feature,
                "matched_participant_n": len(joined), "spearman_rho": rho,
                "bootstrap_95ci_low": low, "bootstrap_95ci_high": high, "permutation_p": p,
                "direction_definition": "positive means participants with a larger feature increase also had a larger reliability/disagreement increase",
            })
    associations = bh(pd.DataFrame(association_rows))
    associations["evidence_classification"] = np.where(
        associations.exploratory_supported, "supported morphology association",
        np.where(associations.spearman_rho.notna(), "descriptive/unsupported after multiplicity control", "insufficient variation or coverage"),
    )
    associations["causal_guardrail"] = "paired association only; condition, acquisition, RR and morphology changes are not isolated causal effects"
    associations.to_csv(OUT / "phase6_condition_associations.csv", index=False)

    records = pd.read_csv(OUT / "phase4_failure_reason_records.csv", low_memory=False)
    reason_rows = []
    for dataset, (normal, stress_condition) in CONDITIONS.items():
        dataset_records = records.loc[records.dataset.eq(dataset)]
        canonical = status.loc[status.dataset.eq(dataset)]
        for (kind, algorithm, candidate), all_rows in canonical.groupby(["candidate_type", "algorithm_id", "candidate_name"]):
            failed = dataset_records.loc[dataset_records.candidate_type.eq(kind) & dataset_records.algorithm_id.eq(algorithm)]
            for reason in sorted(records.primary_reason.unique()):
                participant_rates = []
                for (subject, condition), base in all_rows.groupby(["subject_id", "condition"]):
                    if reason == "upstream dependency/ineligibility":
                        denominator = len(base)
                    else:
                        denominator = int(base.attempted.sum())
                    numerator = len(failed.loc[failed.subject_id.eq(subject) & failed.condition.eq(condition) & failed.primary_reason.eq(reason)])
                    participant_rates.append({"subject_id": subject, "condition": condition, "numerator": numerator, "denominator": denominator, "rate_pct": 100 * numerator / denominator if denominator else np.nan})
                rates = pd.DataFrame(participant_rates)
                wide = rates.pivot(index="subject_id", columns="condition", values="rate_pct").dropna(subset=[normal, stress_condition])
                num = rates.pivot(index="subject_id", columns="condition", values="numerator").reindex(wide.index)
                den = rates.pivot(index="subject_id", columns="condition", values="denominator").reindex(wide.index)
                for subject in wide.index:
                    reason_rows.append({
                        "dataset": dataset, "candidate_type": kind, "algorithm_id": algorithm, "candidate_name": candidate,
                        "primary_reason": reason, "subject_id": subject, "normal_condition": normal, "stress_condition": stress_condition,
                        "normal_reason_n": int(num.loc[subject, normal]), "normal_denominator_n": int(den.loc[subject, normal]),
                        "normal_rate_pct": wide.loc[subject, normal], "stress_reason_n": int(num.loc[subject, stress_condition]),
                        "stress_denominator_n": int(den.loc[subject, stress_condition]), "stress_rate_pct": wide.loc[subject, stress_condition],
                        "stress_minus_normal_percentage_points": wide.loc[subject, stress_condition] - wide.loc[subject, normal],
                        "denominator_definition": "all method-beat records" if reason == "upstream dependency/ineligibility" else "eligible attempted method-beat records",
                    })
    reason_changes = pd.DataFrame(reason_rows)
    reason_changes.to_csv(OUT / "phase6_failure_reason_changes.csv", index=False)

    checks = pd.DataFrame([
        {"check": "matched_participants", "passed": feature_changes.groupby("dataset").subject_id.nunique().to_dict() == {"IITB": 18, "ReBeat": 24}, "observed": str(feature_changes.groupby("dataset").subject_id.nunique().to_dict()), "expected": "{'IITB': 18, 'ReBeat': 24}"},
        {"check": "seven_context_features", "passed": set(feature_changes.feature) == set(feature_names), "observed": feature_changes.feature.nunique(), "expected": 7},
        {"check": "all_methods_outcomes_present", "passed": outcome_changes.loc[outcome_changes.outcome.ne("within_beat_range_ms")].groupby("candidate_type").algorithm_id.nunique().to_dict() == {"B": 10, "X": 13}, "observed": str(outcome_changes.loc[outcome_changes.outcome.ne("within_beat_range_ms")].groupby("candidate_type").algorithm_id.nunique().to_dict()), "expected": "{'B': 10, 'X': 13}"},
        {"check": "reason_change_denominators_positive", "passed": bool((reason_changes.normal_denominator_n > 0).all() and (reason_changes.stress_denominator_n > 0).all()), "observed": int((reason_changes.normal_denominator_n <= 0).sum() + (reason_changes.stress_denominator_n <= 0).sum()), "expected": 0},
        {"check": "no_causal_classification", "passed": not associations.evidence_classification.str.contains("causal", case=False).any(), "observed": associations.evidence_classification.nunique(), "expected": "no causal label"},
    ])
    checks.to_csv(OUT / "phase6_validation.csv", index=False)
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed].to_string(index=False))

    supported = associations.loc[associations.exploratory_supported]
    report = [
        "# Phase 6 — matched condition, morphology and reliability investigation", "",
        "IITB comparisons use the same 18 participants at rest and post-exercise; ReBeat comparisons use the same 24 participants at baseline and cognitive workload. Beats are not paired across separately recorded conditions. Participant medians are computed within condition and then differenced as stress minus normal.", "",
        f"The analysis generated {len(feature_changes):,} participant-feature changes, {len(outcome_changes):,} participant-outcome changes, {len(associations):,} feature-change/outcome-change tests, and {len(reason_changes):,} participant-level verified-reason changes. {len(supported):,} associations met both the bootstrap-direction and within-family BH criteria.", "",
        "## Interpretation hierarchy", "",
        "- Frozen execution tokens establish confirmed algorithmic rejection mechanisms.",
        "- Multiplicity-controlled feature-change/outcome-change results are labelled supported morphology associations.",
        "- Other estimable relationships are descriptive or unsupported.",
        "- No relationship is labelled causal because condition also changes RR, acquisition context, motion exposure and other physiology.", "",
        "## Critical complete-case caution", "",
        "Disagreement changes use all-method common-valid beats, whereas failure changes use all eligible attempts. A lower retained-beat disagreement can result from excluding difficult beats and must be interpreted alongside common-valid coverage and reason-specific failures.", "",
        "## Outputs", "",
        "- `phase6_condition_feature_changes.csv`", "- `phase6_condition_outcome_changes.csv`",
        "- `phase6_condition_associations.csv`", "- `phase6_failure_reason_changes.csv`", "- `phase6_validation.csv`",
    ]
    if len(supported):
        report += ["", "## Strongest supported paired associations", ""]
        for row in supported.assign(abs_rho=supported.spearman_rho.abs()).sort_values(["bh_q_value", "abs_rho"], ascending=[True, False]).head(15).itertuples():
            report.append(f"- {row.dataset} {row.candidate_type} {row.algorithm_id}, {row.feature} change versus {row.outcome} change: rho={row.spearman_rho:.2f}, 95% bootstrap CI [{row.bootstrap_95ci_low:.2f}, {row.bootstrap_95ci_high:.2f}], BH q={row.bh_q_value:.3g}, n={row.matched_participant_n}.")
    (OUT / "phase6_condition_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
