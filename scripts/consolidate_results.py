"""Build the public P1 result package from the verified working project.

This script does not rerun frozen detectors. It consolidates the accepted
scientific outputs into tables with one documented grain and copies only the
accepted 66-panel figure package. Raw recordings are never copied.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DERIVED = ROOT / "data" / "derived"
MANIFESTS = ROOT / "manifests"
SEED = 20261009


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_table(frame: pd.DataFrame, path: Path, key: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        if frame[key].isna().any(axis=None):
            raise ValueError(f"missing key values in {path.name}: {key}")
        duplicate_n = int(frame.duplicated(key).sum())
        if duplicate_n:
            raise ValueError(f"{path.name} has {duplicate_n} duplicate keys: {key}")
        frame = frame.sort_values(key, kind="stable")
    frame.replace([np.inf, -np.inf], np.nan).to_csv(path, index=False, na_rep="")


def median_bootstrap(values: pd.Series, replicates: int = 4000) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return np.nan, np.nan, np.nan
    estimate = float(np.median(array))
    if len(array) < 3:
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    draws = np.median(rng.choice(array, (replicates, len(array)), replace=True), axis=1)
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def dataset_summary(source: Path) -> pd.DataFrame:
    table = pd.read_csv(source / "results/tables/RQ1_dataset_algorithm_results_supervisor.csv", low_memory=False)
    table.insert(0, "dataset", table.dataset_cohort.str.split(" / ").str[0])
    table.insert(1, "cohort", table.dataset_cohort.str.split(" / ").str[1])
    table["upstream_ineligible_n"] = table.total_beat_n - table.eligible_attempted_beat_n
    table = table.rename(columns={
        "candidate_type": "point_type",
        "eligible_attempted_beat_n": "attempted_beat_n",
        "valid_detections": "valid_detection_n",
        "missed_detections": "missed_detection_n",
        "invalid_detections": "invalid_detection_n",
        "success_pct": "success_rate_pct",
        "failure_pct": "attempted_failure_rate_pct",
        "participant_median_definition_disagreement_ms": "participant_median_peer_disagreement_ms",
        "common_beat_analysis_n": "common_valid_beat_n",
    })
    columns = [
        "dataset", "cohort", "condition", "point_type", "algorithm_id", "algorithm_name",
        "exact_algorithm_and_parameter_version", "participant_n", "recording_n", "total_beat_n",
        "attempted_beat_n", "valid_detection_n", "missed_detection_n", "invalid_detection_n",
        "upstream_ineligible_n", "success_rate_pct", "attempted_failure_rate_pct", "reference_type",
        "reference_matched_beat_n", "strict_common_reference_beat_n", "median_signed_reference_error_ms",
        "participant_level_median_absolute_reference_error_ms", "p90_absolute_reference_error_ms",
        "p95_absolute_reference_error_ms", "participant_median_peer_disagreement_ms", "common_valid_beat_n",
        "analysis_status", "selected_for_detailed_case_study", "case_study_role", "selection_rationale",
        "verified_primary_failure_reasons", "attempted_failure_denominator_n", "failed_or_ineligible_n",
        "unresolved_proportion", "morphology_supported_association_n", "morphology_test_n",
        "morphology_participant_n", "matched_condition_changes", "selected_waveform_cases",
        "supplementary_status",
    ]
    return table[columns]


def reference_accuracy(source: Path) -> pd.DataFrame:
    table = pd.read_csv(source / "results/tables/RQ1_reference_participant_accuracy.csv")
    return table.rename(columns={"candidate_type": "point_type", "common_reference_beats": "strict_common_reference_beat_n"})


def failure_mechanisms(source: Path) -> pd.DataFrame:
    table = pd.read_csv(source / "results/tables/P1_verified_method_failure_reasons.csv")
    return table.rename(columns={"candidate_type": "point_type"})


def algorithm_disagreement(source: Path) -> pd.DataFrame:
    rows = []
    folders = {"HeartCycle": "", "IITB": "IITB", "ReBeat": "ReBeat"}
    for dataset, folder in folders.items():
        for point in "BX":
            base = source / "results/tables" / folder / point if folder else source / "results/tables" / point
            method = pd.read_csv(base / f"{point}_definition_participant_summary.csv")
            method.insert(0, "dataset", dataset)
            method.insert(2, "point_type", point)
            method["observation_type"] = "method_peer_difference"
            method = method.rename(columns={
                "common_valid_beats": "common_valid_beat_n",
                "participant_median_peer_difference_ms": "participant_median_disagreement_ms",
                "participant_p95_peer_difference_ms": "participant_p95_disagreement_ms",
            })
            rows.append(method)
            spread = pd.read_csv(base / f"{point}_range_participant_summary.csv")
            spread.insert(0, "dataset", dataset)
            spread.insert(2, "point_type", point)
            spread["algorithm_id"] = "ALL_METHOD_RANGE"
            spread["candidate_name"] = "all-method within-beat range"
            spread["observation_type"] = "all_method_range"
            spread = spread.rename(columns={
                "common_valid_beats": "common_valid_beat_n",
                "participant_median_range_ms": "participant_median_disagreement_ms",
                "participant_p95_range_ms": "participant_p95_disagreement_ms",
            })
            rows.append(spread)
    columns = ["dataset", "condition", "point_type", "subject_id", "algorithm_id", "candidate_name",
               "observation_type", "common_valid_beat_n", "participant_median_disagreement_ms",
               "participant_p95_disagreement_ms"]
    return pd.concat(rows, ignore_index=True, sort=False)[columns]


def condition_tables(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = pd.read_csv(source / "results/tables/RQ1_paired_condition_changes.csv")
    participant = participant.rename(columns={"candidate_type": "point_type"})
    participant["algorithm_id"] = participant.algorithm_id.fillna("ALL_METHOD_RANGE")
    summary_rows = []
    group = ["dataset", "point_type", "outcome", "algorithm_id", "normal_condition", "stress_condition", "unit"]
    for keys, values in participant.groupby(group, dropna=False, sort=False):
        estimate, low, high = median_bootstrap(values.stress_minus_normal)
        summary_rows.append(dict(zip(group, keys)) | {
            "matched_participant_n": int(values.subject_id.nunique()),
            "median_stress_minus_normal": estimate,
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
            "q25_stress_minus_normal": float(values.stress_minus_normal.quantile(0.25)),
            "q75_stress_minus_normal": float(values.stress_minus_normal.quantile(0.75)),
            "positive_change_n": int(values.stress_minus_normal.gt(0).sum()),
            "negative_change_n": int(values.stress_minus_normal.lt(0).sum()),
            "zero_change_n": int(values.stress_minus_normal.eq(0).sum()),
            "interpretation": "descriptive matched-participant change; not a causal condition effect",
        })
    return participant, pd.DataFrame(summary_rows)


def condition_morphology(source: Path) -> pd.DataFrame:
    table = pd.read_csv(source / "results/tables/P1_condition_morphology_associations.csv")
    target = (
        table.dataset.eq("IITB") & table.candidate_type.eq("X") & table.algorithm_id.eq("X3")
        & table.outcome.eq("failure_rate_pct") & table.feature.eq("beat_to_beat_variability")
    )
    if int(target.sum()) != 1:
        raise ValueError("IITB X3 corrected row was not found exactly once")
    table.loc[target, "permutation_p"] = 1 / 9
    table["permutation_method"] = "two-sided participant-label Monte Carlo permutation; 10,000 draws; seed 20261009"
    table.loc[target, "permutation_method"] = "two-sided exact permutation over 18 distinct placements of the sole non-zero failure change"
    table["bootstrap_interpretation"] = "participant bootstrap interval"
    table.loc[target, "bootstrap_interpretation"] = (
        "conditional on non-degenerate resamples; 17 of 18 failure changes were zero; the positive interval is not independent support"
    )
    table.loc[target, "exploratory_supported"] = False
    table.loc[target, "evidence_classification"] = "descriptive/unsupported after multiplicity control"
    return table


def clean_path_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for column in table.columns:
        if not (is_object_dtype(table[column].dtype) or is_string_dtype(table[column].dtype)):
            continue
        if "file" in column.lower() or "path" in column.lower() or "source" in column.lower():
            table[column] = table[column].map(
                lambda value: Path(str(value)).name if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value) else value
            )
    return table


def copy_derived(source: Path, data_root: Path | None) -> list[tuple[Path, Path, str]]:
    mapping = {
        source / "results/tables/RQ1_reference_errors.csv": DERIVED / "reference_errors.csv",
        source / "results/tables/RQ1_bland_altman_values.csv": DERIVED / "paired_method_difference_values.csv",
        source / "work/p1_completion/phase5_participant_effects.csv": DERIVED / "morphology_participant_effects.csv",
        source / "work/p1_completion/phase6_condition_feature_changes.csv": DERIVED / "condition_feature_changes.csv",
        source / "work/p1_completion/phase6_condition_outcome_changes.csv": DERIVED / "condition_outcome_changes.csv",
        source / "work/p1_completion/phase4_failure_reason_records.csv": DERIVED / "failure_reason_records.csv",
        source / "results/tables/P1_condition_morphology_statistical_audit.csv": DERIVED / "condition_morphology_statistical_audit.csv",
        source / "results/tables/P1_IITB_X3_participant_pairs.csv": DERIVED / "iitb_x3_participant_pairs.csv",
        source / "results/tables/P1_final_targeted_validation_summary.csv": DERIVED / "final_targeted_validation_summary.csv",
    }
    copied = []
    for old, new in mapping.items():
        table = clean_path_columns(pd.read_csv(old, low_memory=False))
        write_table(table, new)
        copied.append((old, new, "derived analytical input"))
    external = None if data_root is None else data_root / "HeartCycle_BX_Reference_Validation/output/HeartCycle_BX_reference_validation_beatwise.csv"
    if external is not None and external.exists():
        new = DERIVED / "heartcycle_reference_validation_beatwise.csv"
        write_table(clean_path_columns(pd.read_csv(external, low_memory=False)), new)
        copied.append((external, new, "HeartCycle reference-alignment evidence"))
    return copied


def figure_group(stem: str) -> str:
    return {
        "F01": "waveforms", "F02": "reference_accuracy", "F03": "error_distributions",
        "F04": "detection_failure", "F05": "disagreement", "F06": "morphology",
        "F07": "physiological_conditions", "F08": "cross_dataset", "F09": "bland_altman",
    }[stem[:3]]


def release_stem(stem: str) -> str:
    match = re.match(r"F(\d{2})([A-Z]?)_(.+)", stem)
    if not match:
        raise ValueError(f"unrecognized figure stem: {stem}")
    number, panel, rest = match.groups()
    return f"fig{number}{panel.lower()}_{rest.lower()}"


def copy_figures(source: Path) -> list[tuple[Path, Path, str]]:
    source_figures = source / "results/figures"
    source_data = source / "results/figure_data"
    source_reports = source / "results/reports"
    pngs = sorted(source_figures.rglob("*.png"))
    if len(pngs) != 66:
        raise ValueError(f"expected 66 accepted panels, found {len(pngs)}")
    copied = []
    for png in pngs:
        relative = png.relative_to(source_figures)
        old_stem = png.stem
        new_stem = release_stem(old_stem)
        target = FIGURES / figure_group(old_stem)
        target.mkdir(parents=True, exist_ok=True)
        for suffix, old_root in [(".png", source_figures), (".pdf", source_figures), (".csv", source_data)]:
            old = old_root / relative.with_suffix(suffix)
            new = target / f"{new_stem}{suffix}"
            if not old.exists():
                raise FileNotFoundError(old)
            shutil.copy2(old, new)
            copied.append((old, new, "accepted figure package"))
        old_report = source_reports / relative.with_suffix(".md")
        text = old_report.read_text(encoding="utf-8")
        text = re.sub(r"\]\([^)]*\.csv\)", f"]({new_stem}.csv)", text)
        if old_stem == "F06_P1_morphology_selected_effects":
            sample = (
                "\n**Dataset/sample:** HeartCycle B1 uses 17 participants and 986 strict common-reference method-beat pairs. "
                "IITB post-exercise X4 uses 16 estimable participants and 1,610 attempted method-beat records with feature values.\n"
                "\n**Result:** HeartCycle B1 median rho 0.216 (95% interval 0.117 to 0.305; q=0.0237). "
                "IITB X4 failed-minus-valid sharpness contrast -460 curvature units (95% interval -1,002 to -381; q=0.00525).\n"
                "\n**Limitation:** These participant-aware associations are exploratory and do not establish causation. "
                "The two panels use different datasets, outcomes, denominators and feature units, so their effect sizes are not directly comparable.\n"
            )
            heading, body = text.split("\n", 1)
            text = heading + sample + "\n" + body
        new_report = target / f"{new_stem}.md"
        new_report.write_text(text, encoding="utf-8")
        copied.append((old_report, new_report, "accepted figure interpretation"))
    return copied


def manifest_rows(source: Path, copied: list[tuple[Path, Path, str]]) -> pd.DataFrame:
    rows = []
    for old, new, role in copied:
        try:
            original = old.resolve().relative_to(source.resolve()).as_posix()
        except ValueError:
            original = "external_dependency/" + old.name
        rows.append({
            "canonical_original_path": original,
            "release_path": new.resolve().relative_to(ROOT.resolve()).as_posix(),
            "role": role,
            "original_sha256": sha256(old),
            "release_sha256": sha256(new),
            "byte_identical": sha256(old) == sha256(new),
        })
    return pd.DataFrame(rows)


def canonical_source_map(source: Path) -> dict[str, list[Path]]:
    disagreement = []
    for folder in [Path(), Path("IITB"), Path("ReBeat")]:
        for point in "BX":
            base = source / "results/tables" / folder / point
            disagreement.extend([
                base / f"{point}_definition_participant_summary.csv",
                base / f"{point}_range_participant_summary.csv",
            ])
    return {
        "dataset_algorithm_results.csv": [source / "results/tables/RQ1_dataset_algorithm_results_supervisor.csv"],
        "reference_accuracy.csv": [source / "results/tables/RQ1_reference_participant_accuracy.csv"],
        "detection_failure_mechanisms.csv": [source / "results/tables/P1_verified_method_failure_reasons.csv"],
        "algorithm_disagreement.csv": disagreement,
        "morphology_associations.csv": [source / "results/tables/P1_complete_six_feature_morphology_evidence.csv"],
        "condition_morphology_associations.csv": [source / "results/tables/P1_condition_morphology_associations.csv"],
        "paired_method_agreement.csv": [source / "results/tables/RQ1_bland_altman_summary.csv"],
        "waveform_case_evidence.csv": [source / "work/p1_completion/phase7_waveform_case_selection.csv"],
        "dataset_reference_audit.csv": [source / "results/tables/RQ1_dataset_reference_audit.csv"],
        "condition_comparisons.csv": [source / "results/tables/RQ1_paired_condition_changes.csv"],
        "condition_comparison_summary.csv": [source / "results/tables/RQ1_paired_condition_changes.csv"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Optional external data root for HeartCycle reference evidence")
    args = parser.parse_args()
    source = args.source_project.resolve()
    TABLES.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(source / "configs/canonical_method_registry.csv")
    registry["source_implementation"] = "src/variability_analysis/bx_candidates.py"
    if registry.algorithm_id.duplicated().any():
        raise ValueError("canonical method registry contains duplicate algorithm IDs")
    registry["_point_order"] = registry.algorithm_id.str[0].map({"B": 0, "X": 1})
    registry["_method_order"] = registry.algorithm_id.str[1:].astype(int)
    registry = registry.sort_values(["_point_order", "_method_order"]).drop(columns=["_point_order", "_method_order"])
    write_table(registry, ROOT / "configs/canonical_method_registry.csv")

    outputs: list[tuple[Path, Path, str]] = []
    tables = {
        "dataset_algorithm_results.csv": (dataset_summary(source), ["dataset", "condition", "point_type", "algorithm_id"]),
        "reference_accuracy.csv": (reference_accuracy(source), ["dataset", "condition", "point_type", "subject_id", "algorithm_id"]),
        "detection_failure_mechanisms.csv": (failure_mechanisms(source), ["dataset", "condition", "point_type", "algorithm_id", "primary_reason"]),
        "algorithm_disagreement.csv": (algorithm_disagreement(source), ["dataset", "condition", "point_type", "subject_id", "algorithm_id"]),
        "morphology_associations.csv": (pd.read_csv(source / "results/tables/P1_complete_six_feature_morphology_evidence.csv").rename(columns={"candidate_type": "point_type"}), ["dataset", "condition", "point_type", "algorithm_id", "outcome", "feature"]),
        "condition_morphology_associations.csv": (condition_morphology(source).rename(columns={"candidate_type": "point_type"}), ["dataset", "point_type", "algorithm_id", "outcome", "feature"]),
        "paired_method_agreement.csv": (pd.read_csv(source / "results/tables/RQ1_bland_altman_summary.csv").rename(columns={"candidate_type": "point_type"}), ["dataset", "condition", "point_type"]),
        "waveform_case_evidence.csv": (pd.read_csv(source / "work/p1_completion/phase7_waveform_case_selection.csv"), ["dataset", "panel"]),
        "dataset_reference_audit.csv": (pd.read_csv(source / "results/tables/RQ1_dataset_reference_audit.csv"), None),
    }
    participant, condition_summary = condition_tables(source)
    tables["condition_comparisons.csv"] = (participant, ["dataset", "point_type", "outcome", "subject_id", "algorithm_id"])
    tables["condition_comparison_summary.csv"] = (condition_summary, ["dataset", "point_type", "outcome", "algorithm_id"])
    sources = canonical_source_map(source)
    for name, (frame, key) in tables.items():
        path = TABLES / name
        write_table(frame, path, key)
        outputs.extend((old, path, "consolidated canonical result") for old in sources[name])

    data_root = args.data_root.resolve() if args.data_root else None
    outputs.extend(copy_derived(source, data_root))
    outputs.extend(copy_figures(source))
    figure_metadata = clean_path_columns(pd.read_csv(source / "results/tables/P1_figure_sample_metadata_audit.csv"))
    figure_metadata_path = MANIFESTS / "figure_sample_metadata_corrections.csv"
    write_table(figure_metadata, figure_metadata_path, ["relative_path"])
    outputs.append((source / "results/tables/P1_figure_sample_metadata_audit.csv", figure_metadata_path, "14-panel sample metadata correction audit"))
    outputs.append((source / "configs/canonical_method_registry.csv", ROOT / "configs/canonical_method_registry.csv", "canonical method registry"))
    report = ROOT / "results/reports/P1_Final_Supervisor_RQ1_Report.docx"
    if report.exists():
        outputs.append((source / "results/reports/P1_Final_Supervisor_RQ1_Report.docx", report, "metadata-scrubbed supervisor report"))
    if data_root:
        for name in ["bx_candidates.py", "phase4.py", "signal_processing.py"]:
            outputs.append((data_root / "src/variability_analysis" / name, ROOT / "src/variability_analysis" / name, "unchanged frozen implementation"))
    inventory = manifest_rows(source, [(old, new, role) for old, new, role in outputs if old.exists()])
    write_table(inventory, MANIFESTS / "source_to_release_inventory.csv", ["release_path", "canonical_original_path"])
    print(f"Canonical tables: {len(tables)}")
    print(f"Accepted panels: {len(list(FIGURES.rglob('*.png')))}")
    print(f"Derived inputs: {len(list(DERIVED.glob('*.csv')))}")


if __name__ == "__main__":
    main()
