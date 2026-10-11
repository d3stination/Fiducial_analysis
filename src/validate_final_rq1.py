"""Validation checks for the consolidated RQ1 numerical and visual package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data_loading import ROOT
from .final_rq1_results import DATASETS, KEYS, load_status, methods, table_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> dict:
    checks = {}
    registry = pd.read_csv(ROOT / "configs" / "canonical_method_registry.csv")
    checks["exact_method_counts"] = registry.candidate_type.value_counts().to_dict() == {"X": 13, "B": 10}
    checks["unique_registry_ids"] = not registry.algorithm_id.duplicated().any()

    # Original detector/preprocessing tables must remain byte-identical to the fresh replay baseline.
    phase2_hashes = json.loads((ROOT / "work" / "phase2" / "first_table_hashes.json").read_text())
    checks["phase2_source_tables_unchanged"] = all((ROOT / path).is_file() and sha256(ROOT / path) == digest for path, digest in phase2_hashes.items())
    heartcycle_hashes = json.loads((ROOT / "work" / "phase2" / "heartcycle_before.json").read_text())
    heartcycle_tables = {path: digest for path, digest in heartcycle_hashes.items() if path.startswith("results/tables/")}
    checks["heartcycle_source_tables_unchanged"] = all((ROOT / path).is_file() and sha256(ROOT / path) == digest for path, digest in heartcycle_tables.items())

    duplicate_rows = 0
    inconsistent_status_rows = 0
    paired_counts = {}
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            status = load_status(dataset, kind)
            duplicate_rows += int(status.duplicated(KEYS + ["algorithm_id"]).sum())
            attempted_class = status.valid.astype(int) + status.missed.astype(int) + status.invalid.astype(int)
            inconsistent_status_rows += int(((status.attempted & attempted_class.ne(1)) | (~status.attempted & attempted_class.ne(0))).sum())
        if len(DATASETS[dataset]["conditions"]) == 2:
            conditions = DATASETS[dataset]["conditions"]
            status = load_status(dataset, "B")
            paired = set.intersection(*[set(status.loc[status.condition.eq(condition), "subject_id"]) for condition in conditions])
            paired_counts[dataset] = len(paired)
    checks["duplicate_beat_algorithm_rows"] = duplicate_rows == 0
    checks["mutually_exclusive_status_totals"] = inconsistent_status_rows == 0
    checks["paired_participants"] = paired_counts == {"IITB": 18, "ReBeat": 24}

    errors = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv", low_memory=False)
    recalculated = 1000 * (errors.candidate_timestamp_s - errors.reference_event_time_s)
    checks["reference_error_formula_ms"] = bool(np.allclose(recalculated, errors.signed_error_ms, rtol=0, atol=1e-8))
    common_equal = True
    for _, rows in errors.groupby(["dataset", "candidate_type", "condition"]):
        counts = rows.groupby("algorithm_id").beat_id.nunique()
        common_equal &= counts.nunique() == 1 and len(counts) == len(methods(rows.candidate_type.iloc[0]))
    checks["same_common_reference_beats"] = bool(common_equal)
    checks["no_missing_detection_as_zero_error"] = not errors[["candidate_timestamp_s", "reference_event_time_s", "signed_error_ms"]].isna().any().any()

    mapping = pd.read_csv(ROOT / "results" / "tables" / "ReBeat" / "ReBeat_expert_annotation_mapping.csv")
    checks["rebeat_unique_annotation_beat_match"] = len(mapping) == 1842 and not mapping.duplicated(["dataset_id", "recording_id", "beat_id"]).any()
    checks["rebeat_b_before_x"] = bool(mapping.b_less_than_x.all())

    morphology = pd.read_csv(ROOT / "results" / "tables" / "RQ1_morphology_evidence.csv")
    checks["six_morphology_hypotheses_plus_rr"] = set(morphology.feature) == {"sharpness", "slope", "amplitude", "snr", "competing", "variability", "rr_interval_ms"}
    checks["morphology_participant_effects_bounded"] = bool(morphology.median_within_participant_spearman_rho.dropna().between(-1, 1).all())

    consolidated = pd.read_csv(ROOT / "results" / "tables" / "RQ1_dataset_algorithm_results.csv")
    checks["consolidated_expected_rows"] = len(consolidated) == 115
    checks["no_iitb_accuracy_mislabel"] = consolidated.loc[consolidated.dataset_cohort.str.startswith("IITB"), "median_signed_reference_error_ms"].isna().all()

    figures = sorted((ROOT / "results" / "figures").rglob("*.png"))
    pdfs = sorted((ROOT / "results" / "figures").rglob("*.pdf"))
    figure_data = sorted((ROOT / "results" / "figure_data").rglob("*.csv"))
    reports = {path.stem for path in (ROOT / "results" / "reports").rglob("*.md")}
    checks["every_figure_has_report"] = all(path.stem in reports for path in figures)
    checks["figure_count"] = len(figures) == 66
    checks["every_figure_has_vector_pdf"] = {path.stem for path in figures} == {path.stem for path in pdfs}
    checks["every_figure_has_plotting_data"] = {path.stem for path in figures} == {path.stem for path in figure_data}
    checks["figure_groups_1_to_9_present"] = {int(path.stem[1:3]) for path in figures} == set(range(1, 10))
    checks["bland_altman_migrated_to_figure9"] = all("bland_altman" not in path.stem for path in figures if path.stem.startswith("F08")) and any(path.stem.startswith("F09") for path in figures)
    checks["plotting_data_nonempty"] = all(path.stat().st_size > 40 for path in figure_data)
    checks["all_figures_300_dpi_and_nonblank"] = True
    for path in figures:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            extrema = image.convert("L").getextrema()
            if min(dpi) < 295 or extrema[0] == extrema[1] or image.width < 1200 or image.height < 800:
                checks["all_figures_300_dpi_and_nonblank"] = False
                break
    checks["archived_previous_outputs"] = all((ROOT / "archive" / name).is_file() for name in [
        "pre_final_rq1_results_20261009.zip", "pre_figure_repair_20261009.zip"
    ])

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    result = {"status": "PASS" if not failed else "FAIL", "checks": checks, "failed": failed}
    (ROOT / "results" / "tables" / "RQ1_validation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failed:
        raise AssertionError(f"Final RQ1 validation failed: {failed}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    validate()
