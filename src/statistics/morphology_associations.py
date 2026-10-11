from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "p1_completion"
KEYS = ["dataset_id", "subject_id", "recording_id", "beat_id"]
FEATURES = {
    "sharpness": "Peak or turning-point sharpness",
    "slope": "Local waveform slope",
    "amplitude": "Signal amplitude",
    "snr": "Signal-to-noise ratio",
    "competing_features": "Competing peaks/minima/inflection points",
    "beat_to_beat_variability": "Beat-to-beat morphological variability",
}
UNITS = {
    "sharpness": "dataset-specific derivative units",
    "slope": "dataset-specific derivative units",
    "amplitude": "dataset-specific signal units",
    "snr": "dB",
    "competing_features": "count",
    "beat_to_beat_variability": "normalized error",
}
RNG = np.random.default_rng(20261009)


def paths(dataset: str, kind: str) -> Path:
    folder = {"HeartCycle": Path(), "IITB": Path("IITB"), "ReBeat": Path("ReBeat")}[dataset]
    return ROOT / "results" / "tables" / folder / kind


def load_features() -> pd.DataFrame:
    frames = []
    for dataset in ["HeartCycle", "IITB", "ReBeat"]:
        for kind in "BX":
            base = paths(dataset, kind)
            raw = pd.read_csv(base / f"{kind}_morphology_features.csv", low_memory=False)
            if dataset == "HeartCycle":
                rename = {
                    "reference_local_sharpness_abs_d2_device_units_s3": "sharpness",
                    "reference_local_slope_abs_d1_device_units_s2": "slope",
                    "reference_local_amplitude_device_units_s": "amplitude",
                    "signal_to_noise_ratio_db": "snr",
                    "reference_region_competing_features_count": "competing_features",
                    "beat_to_beat_shape_nrmse": "beat_to_beat_variability",
                }
                keep = KEYS + list(rename)
                beat = raw[keep].drop_duplicates(KEYS).rename(columns=rename)
                master = pd.read_csv(base / f"{kind}_master_beats.csv", usecols=KEYS + ["condition"])
                beat = beat.merge(master, on=KEYS, how="left", validate="one_to_one")
                beat["feature_region_basis"] = "independent ECHO-reference local region for sharpness/slope/amplitude/competing; beat-level signal metrics for SNR/variability"
            else:
                rename = {"variability": "beat_to_beat_variability", "competing": "competing_features"}
                beat = raw[KEYS + ["condition", "sharpness", "slope", "amplitude", "snr", "competing", "variability", "feature_status"]].rename(columns=rename)
                beat["feature_region_basis"] = "predefined candidate-independent B or X phase region"
            beat["dataset"] = dataset
            beat["candidate_type"] = kind
            frames.append(beat)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_status() -> pd.DataFrame:
    frames = []
    for dataset in ["HeartCycle", "IITB", "ReBeat"]:
        for kind in "BX":
            base = paths(dataset, kind)
            status = pd.read_csv(base / f"{kind}_detection_status.csv", low_memory=False)
            if "condition" not in status:
                master = pd.read_csv(base / f"{kind}_master_beats.csv", usecols=KEYS + ["condition"])
                status = status.merge(master, on=KEYS, how="left", validate="many_to_one")
            status["dataset"] = dataset
            frames.append(status)
    return pd.concat(frames, ignore_index=True, sort=False)


def participant_rhos(data: pd.DataFrame, feature: str, outcome: str) -> list[dict]:
    rows = []
    for participant, group in data[["subject_id", feature, outcome]].dropna().groupby("subject_id"):
        if len(group) < 8 or group[feature].nunique() < 2 or group[outcome].nunique() < 2:
            continue
        rho = float(spearmanr(group[feature], group[outcome]).statistic)
        if np.isfinite(rho):
            rows.append({"subject_id": participant, "participant_effect": rho, "participant_observation_n": len(group)})
    return rows


def participant_failure_contrasts(data: pd.DataFrame, feature: str) -> list[dict]:
    rows = []
    usable = data.loc[data.attempted.astype(bool), ["subject_id", feature, "failure"]].dropna()
    for participant, group in usable.groupby("subject_id"):
        failed = group.loc[group.failure, feature]
        valid = group.loc[~group.failure, feature]
        if len(failed) < 3 or len(valid) < 3:
            continue
        rows.append({
            "subject_id": participant,
            "participant_effect": float(failed.median() - valid.median()),
            "participant_observation_n": len(group),
            "participant_failed_n": len(failed),
            "participant_valid_n": len(valid),
        })
    return rows


def bootstrap_median(values: np.ndarray, iterations: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    draws = RNG.choice(values, size=(iterations, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, .025)), float(np.quantile(medians, .975))


def wilcoxon_p(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if len(nonzero) < 5:
        return np.nan
    return float(wilcoxon(nonzero, alternative="two-sided", method="approx").pvalue)


def summarize_effects(meta: dict, participant_rows: list[dict], available: pd.DataFrame, feature: str) -> tuple[dict, list[dict]]:
    effects = np.array([row["participant_effect"] for row in participant_rows], dtype=float)
    low, high = bootstrap_median(effects)
    summary = dict(meta)
    summary.update({
        "feature": feature,
        "feature_label": FEATURES[feature],
        "feature_units": "Spearman rho" if meta["effect_metric"] == "within-participant Spearman rho" else UNITS[feature],
        "available_observation_n": int(available[["subject_id", feature, meta["outcome_column"]]].dropna().shape[0]),
        "available_participant_n": int(available.loc[available[feature].notna() & available[meta["outcome_column"]].notna(), "subject_id"].nunique()),
        "participant_effect_n": len(effects),
        "participant_effect_observation_n": int(sum(row["participant_observation_n"] for row in participant_rows)),
        "median_participant_effect": float(np.median(effects)) if len(effects) else np.nan,
        "q25_participant_effect": float(np.quantile(effects, .25)) if len(effects) else np.nan,
        "q75_participant_effect": float(np.quantile(effects, .75)) if len(effects) else np.nan,
        "bootstrap_95ci_low": low,
        "bootstrap_95ci_high": high,
        "wilcoxon_raw_p": wilcoxon_p(effects),
        "direction": "positive" if len(effects) and np.median(effects) > 0 else ("negative" if len(effects) and np.median(effects) < 0 else "null_or_unavailable"),
    })
    detail = [dict(meta, feature=feature, feature_label=FEATURES[feature], **row) for row in participant_rows]
    return summary, detail


def bh_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["multiplicity_family"] = frame[["outcome", "dataset_id", "condition", "candidate_type"]].astype(str).agg("|".join, axis=1)
    frame["bh_q_value"] = np.nan
    for _, idx in frame.groupby("multiplicity_family").groups.items():
        p = frame.loc[idx, "wilcoxon_raw_p"].dropna().sort_values()
        if p.empty:
            continue
        adjusted = (p * len(p) / np.arange(1, len(p) + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
        frame.loc[adjusted.index, "bh_q_value"] = adjusted
    ci_excludes = (frame.bootstrap_95ci_low > 0) | (frame.bootstrap_95ci_high < 0)
    frame["exploratory_supported_after_bh"] = ci_excludes & frame.bh_q_value.lt(.05)
    frame["interpretation_guardrail"] = "exploratory participant-aware association; not a clinical threshold or causal estimate"
    return frame


def main() -> None:
    features = load_features()
    status = load_status()
    feature_cols = list(FEATURES)
    analysis_rows, participant_rows = [], []

    failure = status.merge(features[KEYS + ["candidate_type"] + feature_cols], on=KEYS + ["candidate_type"], how="left", validate="many_to_one")
    failure["failure"] = failure.analysis_status.isin(["MISSED", "INVALID"])
    fail_group = ["dataset", "dataset_id", "condition", "candidate_type", "algorithm_id", "candidate_name"]
    for keys, group in failure.groupby(fail_group, dropna=False):
        meta = dict(zip(fail_group, keys)) | {
            "outcome": "attempted detection failure",
            "outcome_column": "failure",
            "effect_metric": "within-participant median feature contrast: failed minus valid",
            "minimum_participant_data": "at least 3 failed and 3 valid attempted beats",
            "reference_type": "not applicable",
            "comparison_set": "eligible attempted beats with available feature values",
        }
        for feature in feature_cols:
            rows = participant_failure_contrasts(group, feature)
            summary, detail = summarize_effects(meta, rows, group.loc[group.attempted.astype(bool)], feature)
            summary["attempted_denominator_n"] = int(group.attempted.sum())
            summary["failure_n"] = int(group.failure.sum())
            analysis_rows.append(summary)
            participant_rows.extend(detail)

    reference = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv", low_memory=False)
    reference = reference.merge(features[KEYS + ["candidate_type"] + feature_cols], on=KEYS + ["candidate_type"], how="left", validate="many_to_one")
    ref_group = ["dataset", "dataset_id", "condition", "candidate_type", "algorithm_id", "candidate_name", "reference_type", "comparison_set"]
    for keys, group in reference.groupby(ref_group, dropna=False):
        labels = dict(zip(ref_group, keys))
        meta = {key: labels[key] for key in ["dataset", "dataset_id", "condition", "candidate_type", "algorithm_id", "candidate_name", "reference_type", "comparison_set"]} | {
            "outcome": "strict common-reference absolute timing error",
            "outcome_column": "absolute_error_ms",
            "effect_metric": "within-participant Spearman rho",
            "minimum_participant_data": "at least 8 paired observations with variation in feature and outcome",
        }
        for feature in feature_cols:
            rows = participant_rhos(group, feature, "absolute_error_ms")
            summary, detail = summarize_effects(meta, rows, group, feature)
            summary["attempted_denominator_n"] = np.nan
            summary["failure_n"] = np.nan
            analysis_rows.append(summary)
            participant_rows.extend(detail)

    peer_frames = []
    for dataset in ["HeartCycle", "IITB", "ReBeat"]:
        for kind in "BX":
            peer = pd.read_csv(paths(dataset, kind) / f"{kind}_definition_disagreement.csv")
            peer["dataset"] = dataset
            peer["candidate_type"] = kind
            peer_frames.append(peer)
    peer = pd.concat(peer_frames, ignore_index=True).merge(features[KEYS + ["candidate_type"] + feature_cols], on=KEYS + ["candidate_type"], how="left", validate="many_to_one")
    peer_group = ["dataset", "dataset_id", "condition", "candidate_type", "algorithm_id", "candidate_name"]
    for keys, group in peer.groupby(peer_group, dropna=False):
        meta = dict(zip(peer_group, keys)) | {
            "outcome": "method-specific peer disagreement",
            "outcome_column": "absolute_difference_from_other_method_median_ms",
            "effect_metric": "within-participant Spearman rho",
            "minimum_participant_data": "at least 8 paired observations with variation in feature and outcome",
            "reference_type": "not applicable",
            "comparison_set": "all-method common-valid beats",
        }
        for feature in feature_cols:
            rows = participant_rhos(group, feature, "absolute_difference_from_other_method_median_ms")
            summary, detail = summarize_effects(meta, rows, group, feature)
            summary["attempted_denominator_n"] = np.nan
            summary["failure_n"] = np.nan
            analysis_rows.append(summary)
            participant_rows.extend(detail)

    evidence = bh_adjust(pd.DataFrame(analysis_rows))
    evidence.to_csv(OUT / "phase5_all_method_morphology_evidence.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(OUT / "phase5_participant_effects.csv", index=False)

    range_frames = []
    for dataset in ["HeartCycle", "IITB", "ReBeat"]:
        for kind in "BX":
            ranges = pd.read_csv(paths(dataset, kind) / f"{kind}_common_beat_ranges.csv")
            ranges["dataset"] = dataset
            ranges["candidate_type"] = kind
            range_frames.append(ranges)
    ranges = pd.concat(range_frames, ignore_index=True)
    range_col = "timing_range_ms"
    ranges = ranges.merge(features[KEYS + ["candidate_type"] + feature_cols], on=KEYS + ["candidate_type"], how="left", validate="many_to_one")
    aggregate_rows, aggregate_participants = [], []
    group_cols = ["dataset", "dataset_id", "condition", "candidate_type"]
    for keys, group in ranges.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, keys)) | {
            "algorithm_id": "ALL_METHOD_RANGE", "candidate_name": "all-method within-beat range",
            "outcome": "aggregate all-method timing range", "outcome_column": range_col,
            "effect_metric": "within-participant Spearman rho",
            "minimum_participant_data": "at least 8 paired observations with variation in feature and outcome",
            "reference_type": "not applicable", "comparison_set": "all-method common-valid beats",
        }
        for feature in feature_cols:
            rows = participant_rhos(group, feature, range_col)
            summary, detail = summarize_effects(meta, rows, group, feature)
            aggregate_rows.append(summary)
            aggregate_participants.extend(detail)
    aggregate = bh_adjust(pd.DataFrame(aggregate_rows))
    aggregate.to_csv(OUT / "phase5_aggregate_range_morphology.csv", index=False)

    applicability_rows = []
    for keys, group in features.groupby(["dataset", "dataset_id", "condition", "candidate_type"], dropna=False):
        base = dict(zip(["dataset", "dataset_id", "condition", "candidate_type"], keys))
        for feature in feature_cols:
            if base["dataset"] == "HeartCycle" and feature in {"sharpness", "slope", "amplitude", "competing_features"}:
                basis = "independent ECHO-reference local region; availability-limited subset"
                limitation = "not available on every beat; failure associations condition on independent-reference feature availability"
            else:
                basis = "predefined candidate-independent phase/beat feature"
                limitation = "dataset-specific units and acquisition prevent pooling raw feature magnitudes across datasets"
            applicability_rows.append(base | {
                "feature": feature, "feature_label": FEATURES[feature], "units": UNITS[feature],
                "feature_basis": basis, "available_beat_n": int(group.loc[group[feature].notna(), "beat_id"].nunique()),
                "available_participant_n": int(group.loc[group[feature].notna(), "subject_id"].nunique()),
                "failure_outcome_applicable": True, "strict_reference_error_applicable": base["dataset"] in {"HeartCycle", "ReBeat"},
                "peer_disagreement_applicable": True, "limitation": limitation,
            })
    applicability = pd.DataFrame(applicability_rows)
    applicability.to_csv(OUT / "phase5_feature_definitions_applicability.csv", index=False)

    expected_evidence = (
        status[["dataset", "dataset_id", "condition", "candidate_type", "algorithm_id"]].drop_duplicates().shape[0] * 6
        + reference[["dataset", "dataset_id", "condition", "candidate_type", "algorithm_id"]].drop_duplicates().shape[0] * 6
        + peer[["dataset", "dataset_id", "condition", "candidate_type", "algorithm_id"]].drop_duplicates().shape[0] * 6
    )
    checks = pd.DataFrame([
        {"check": "all_expected_method_feature_outcome_rows_present", "passed": len(evidence) == expected_evidence, "observed": len(evidence), "expected": expected_evidence},
        {"check": "all_six_features_present", "passed": set(evidence.feature) == set(feature_cols), "observed": evidence.feature.nunique(), "expected": 6},
        {"check": "all_10_B_and_13_X_algorithms_present", "passed": evidence.loc[evidence.candidate_type.eq("B"), "algorithm_id"].nunique() == 10 and evidence.loc[evidence.candidate_type.eq("X"), "algorithm_id"].nunique() == 13, "observed": f"B={evidence.loc[evidence.candidate_type.eq('B'), 'algorithm_id'].nunique()};X={evidence.loc[evidence.candidate_type.eq('X'), 'algorithm_id'].nunique()}", "expected": "B=10;X=13"},
        {"check": "reference_rows_are_strict_common_samples", "passed": reference.comparison_set.astype(str).str.contains("strict all-method common|strict all-method", case=False).all(), "observed": reference.comparison_set.nunique(), "expected": "strict common-reference only"},
        {"check": "participant_effects_have_minimum_observations", "passed": pd.DataFrame(participant_rows).participant_observation_n.ge(6).all(), "observed": int(pd.DataFrame(participant_rows).participant_observation_n.min()), "expected": ">=6 for failure; >=8 for correlations"},
        {"check": "aggregate_range_has_all_expected_rows", "passed": len(aggregate) == ranges[["dataset", "dataset_id", "condition", "candidate_type"]].drop_duplicates().shape[0] * 6, "observed": len(aggregate), "expected": ranges[["dataset", "dataset_id", "condition", "candidate_type"]].drop_duplicates().shape[0] * 6},
    ])
    checks.to_csv(OUT / "phase5_validation.csv", index=False)
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed].to_string(index=False))

    supported = evidence.loc[evidence.exploratory_supported_after_bh].copy()
    outcome_counts = supported.groupby("outcome").size().to_dict()
    report = [
        "# Phase 5 — all-method morphology and quality analysis",
        "",
        "## Design",
        "",
        "All six prespecified morphology/quality features were evaluated for every applicable B and X definition. No feature was dichotomized for inference. Strict reference-error analyses use only common-valid all-method samples and keep ECHO AVO/ECHO AVC separate from ReBeat expert annotations. Peer disagreement uses each method's absolute deviation from the other-method median on common-valid beats. Failure analyses retain Phase 2 attempted-beat denominators and compare participant-level median feature values on failed versus valid beats.",
        "",
        "Participant correlations require at least eight paired observations with variation. Failure contrasts require at least three failed and three valid attempted beats. Across-participant summaries use the median effect and participant bootstrap 95% interval. Exploratory Wilcoxon tests are Benjamini-Hochberg adjusted across every algorithm-feature test within each dataset/condition/point-type/outcome family.",
        "",
        "## Coverage",
        "",
        f"The primary evidence table contains {len(evidence):,} method-feature-outcome rows and {len(participant_rows):,} participant-effect rows. The aggregate range sensitivity table contains {len(aggregate):,} feature-stratum rows. Supported exploratory associations after both bootstrap-direction and multiplicity checks: {len(supported):,}.",
        "",
        "## Supported-association counts",
        "",
    ]
    for outcome in ["attempted detection failure", "strict common-reference absolute timing error", "method-specific peer disagreement"]:
        report.append(f"- {outcome}: {int(outcome_counts.get(outcome, 0))}")
    report += [
        "",
        "## Scientific cautions",
        "",
        "- These are exploratory associations, not causal effects or clinical thresholds.",
        "- HeartCycle sharpness, slope, amplitude and competing-feature measures are defined around independent ECHO reference regions. Their failure analyses therefore apply only to beats with those features available and must not be generalized to all attempted beats.",
        "- Raw feature magnitudes are not pooled across datasets because units, acquisition and preprocessing differ.",
        "- Common-valid X analyses are conditioned on retained beats. Lower disagreement on that subset cannot be interpreted as better reliability without the corresponding common-valid coverage and excluded failures.",
        "- A non-supported result may reflect limited participant/event coverage rather than evidence of no association.",
        "",
        "## Files",
        "",
        "- `phase5_all_method_morphology_evidence.csv`: all methods, six features, three outcomes, coverage, participant effects, uncertainty and multiplicity control.",
        "- `phase5_participant_effects.csv`: participant-level correlations or failed-minus-valid contrasts.",
        "- `phase5_aggregate_range_morphology.csv`: aggregate all-method range sensitivity analysis.",
        "- `phase5_feature_definitions_applicability.csv`: definitions, availability and applicability limitations.",
        "- `phase5_validation.csv`: reproducibility checks.",
    ]
    if len(supported):
        strongest = supported.assign(abs_effect=supported.median_participant_effect.abs()).sort_values(["bh_q_value", "abs_effect"], ascending=[True, False]).head(12)
        report += ["", "## Strongest supported rows (for audit, not method ranking)", ""]
        for row in strongest.itertuples():
            report.append(
                f"- {row.dataset}/{row.condition} {row.candidate_type} {row.algorithm_id}, {row.outcome}, {row.feature}: "
                f"median participant effect {row.median_participant_effect:.3g}, 95% bootstrap CI "
                f"[{row.bootstrap_95ci_low:.3g}, {row.bootstrap_95ci_high:.3g}], BH q={row.bh_q_value:.3g}, n={row.participant_effect_n} participants."
            )
    (OUT / "phase5_morphology_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
