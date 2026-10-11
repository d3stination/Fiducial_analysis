"""Generate the canonical result-table dictionary from the released CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
OUTPUT = ROOT / "docs/results_dictionary.md"
GRAINS = {
    "dataset_algorithm_results.csv": "One dataset, condition, point type and frozen algorithm.",
    "reference_accuracy.csv": "One participant, reference type, condition and algorithm on the strict common-reference sample.",
    "detection_failure_mechanisms.csv": "One dataset, condition, algorithm and mutually exclusive primary failure reason.",
    "algorithm_disagreement.csv": "One participant and either one method-specific peer difference or one all-method range summary.",
    "condition_comparisons.csv": "One matched participant, outcome and method. ALL_METHOD_RANGE denotes the all-method spread outcome.",
    "condition_comparison_summary.csv": "One dataset, point type, paired outcome and method, summarized across matched participants.",
    "morphology_associations.csv": "One dataset, condition, point type, algorithm, outcome and prespecified waveform feature test.",
    "condition_morphology_associations.csv": "One dataset, point type, algorithm or all-method outcome, paired reliability outcome and feature-change test.",
    "paired_method_agreement.csv": "One dataset, condition, point type and predeclared algorithm pair.",
    "waveform_case_evidence.csv": "One reproducibly selected dataset and Figure 1 case category.",
    "dataset_reference_audit.csv": "One audited dataset and reference source record.",
}


def description(column: str) -> str:
    fixed = {
        "dataset": "Dataset label: HeartCycle, IITB or ReBeat.",
        "dataset_id": "Stable internal dataset identifier.",
        "cohort": "Recorded cohort label without causal interpretation.",
        "condition": "Acquisition condition within the dataset.",
        "point_type": "Fiducial family, B or X.",
        "candidate_type": "Fiducial family, B or X.",
        "algorithm_id": "Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range.",
        "algorithm_name": "Frozen implementation name.",
        "candidate_name": "Frozen implementation name.",
        "subject_id": "Pseudonymous participant identifier used for clustering and pairing.",
        "recording_id": "Stable recording identifier.",
        "beat_id": "Stable R-to-next-R beat identifier.",
        "reference_type": "Reference provenance. ECHO and expert ICG annotations remain distinct.",
        "primary_reason": "Mutually exclusive verified failure reason from the approved hierarchy.",
        "analysis_status": "Availability and comparability status for the result row.",
        "exploratory_supported": "True only when the prespecified bootstrap-direction and within-family BH criteria are met.",
        "bh_q_value": "Benjamini-Hochberg adjusted p value within the stated analysis family.",
        "permutation_p": "Two-sided participant-level permutation p value.",
        "spearman_rho": "Participant-level Spearman rank correlation.",
        "bootstrap_interpretation": "Required qualification for the reported bootstrap interval.",
        "permutation_method": "Permutation unit, tail and exact or Monte Carlo procedure.",
    }
    if column in fixed:
        return fixed[column]
    if column.endswith("_ms"):
        return column.replace("_", " ").removesuffix(" ms").capitalize() + ", in milliseconds."
    if column.endswith("_pct") or column.endswith("_rate_pct"):
        return column.replace("_", " ").capitalize() + ", in percent."
    if column.endswith("_n") or column.endswith("_count") or column.endswith("_beats"):
        return column.replace("_", " ").capitalize() + ", as an integer count."
    if "95ci_low" in column or "ci_low" in column:
        return "Lower bound of the stated 95% interval, in the outcome unit."
    if "95ci_high" in column or "ci_high" in column:
        return "Upper bound of the stated 95% interval, in the outcome unit."
    return column.replace("_", " ").capitalize() + "."


def main() -> None:
    lines = [
        "# Results dictionary", "",
        "Blank numeric cells mean unavailable or not applicable. They are not zero. Counts use the denominator named in the column. Stored scientific values retain full exported precision.", "",
    ]
    for path in sorted(TABLES.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        lines.extend([
            f"## {path.name}", "",
            GRAINS.get(path.name, "See the column definitions for the observation unit."), "",
            f"Rows: {len(frame):,}. Columns: {len(frame.columns):,}.", "",
            "| Column | Definition |",
            "|---|---|",
        ])
        for column in frame.columns:
            lines.append(f"| `{column}` | {description(column)} |")
        lines.append("")
    lines.extend([
        "## Denominator rules", "",
        "`total_beat_n` includes all beat-method records. `upstream_ineligible_n` is excluded from `attempted_beat_n`. "
        "Attempted records reconcile exactly as valid plus missed plus invalid. Reference metrics use strict all-method common-reference beats. "
        "Disagreement uses all-method common-valid beats and must be interpreted with the excluded-beat count.", "",
        "## Derived analytical inputs", "",
        "Files under `data/derived/` preserve event-level or participant-level evidence used to reproduce the canonical summaries. "
        "They are not additional competing result tables. Their original sources and hashes are listed in the manifests.", "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
